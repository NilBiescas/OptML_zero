import os
import yaml
import argparse
import torch
import time
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed, DataCollatorWithPadding, get_scheduler
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from optimizers.lozo import LOZOM, LOZO

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune classification model using config")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    dataset_config = config.get('dataset', {})
    model_config = config.get('model', {})
    train_config = config.get('training', {})
    opt_config = config.get('optimizer', {})
    hub_config = config.get('hub', {})
    
    push_to_hub = hub_config.get('push_to_hub', False)
    repo_id = hub_config.get('repo_id', None)
    
    seed = train_config.get('seed', 42)
    batch_size = train_config.get('batch_size', 16)
    epochs = train_config.get('epochs', 3)
    eval_epochs = train_config.get('eval_epochs', 1)
    max_tokens = train_config.get('max_tokens', None)
    
    # Initialize accelerator
    accelerator = Accelerator(log_with="wandb")
    
    # Try to set the WandB run name to match the RunAI job name
    run_name = os.environ.get("RUN_NAME", None)
    init_kwargs = {}
    if run_name:
        init_kwargs["init_kwargs"] = {"wandb": {"name": run_name}}
        
    accelerator.init_trackers(project_name="lozo-classification-training", config=config, **init_kwargs)
    
    # Crucial: set seed across all processes to ensure deterministic initializations
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    dataset_name = dataset_config.get('name', 'PolyAI/banking77')
    dataset_subset = dataset_config.get('subset', None)
    data_files = dataset_config.get('data_files', None)
    delimiter = dataset_config.get('delimiter', None)
    text_col = dataset_config.get('text_column', 'text')
    text_col2 = dataset_config.get('text_column2', None)
    label_col = dataset_config.get('label_column', 'label')
    
    load_kwargs = {}
    if data_files:
        load_kwargs['data_files'] = data_files
    if delimiter:
        load_kwargs['delimiter'] = delimiter
        if delimiter == '\t':
            import csv
            load_kwargs['quoting'] = csv.QUOTE_NONE

    accelerator.print(f"Loading dataset {dataset_name} (subset: {dataset_subset}) with kwargs {load_kwargs}...")
    try:
        if dataset_subset:
            dataset = load_dataset(dataset_name, dataset_subset, **load_kwargs)
        else:
            dataset = load_dataset(dataset_name, **load_kwargs)
    except Exception as e:
        accelerator.print(f"Failed to load dataset with HuggingFace datasets ({e}). Falling back to robust manual TSV loading...")
        import pandas as pd
        from datasets import Dataset, DatasetDict
        splits = {}
        for split, path in data_files.items():
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            header = lines[0].strip('\n').split('\t')
            records = []
            for line in lines[1:]:
                parts = line.strip('\n').split('\t')
                if len(parts) > len(header):
                    parts = parts[:len(header)]
                elif len(parts) < len(header):
                    parts = parts + [""] * (len(header) - len(parts))
                records.append({h: p for h, p in zip(header, parts)})
            splits[split] = Dataset.from_pandas(pd.DataFrame(records))
        dataset = DatasetDict(splits)
    
    few_shot_k = train_config.get('few_shot_k', None)
    if few_shot_k is not None:
        accelerator.print(f"Subsampling train set to {few_shot_k} shots per class (k={few_shot_k})...")
        df = dataset["train"].to_pandas()
        # Ensure we drop any '-1' labels from train set if they exist
        if label_col in df.columns:
            df = df[df[label_col] != -1]
            df = df.groupby(label_col, group_keys=False).apply(lambda x: x.sample(n=min(len(x), few_shot_k), random_state=seed)).reset_index(drop=True)
        from datasets import Dataset
        dataset["train"] = Dataset.from_pandas(df)

    
    model_name = model_config.get('name', 'Qwen/Qwen3.5-0.8B')
    
    accelerator.print(f"Dataset columns: {dataset['train'].column_names}")
    
    label_col = dataset_config.get("label_column", "label")
    if label_col not in dataset["train"].column_names:
        for p in ["label", "gold_label", "label1"]:
            if p in dataset["train"].column_names:
                accelerator.print(f"Warning: configured label column '{dataset_config.get('label_column', 'label')}' not found. Auto-detected '{p}'.")
                label_col = p
                break

    # Robustly map labels from the dataset to ensure they are [0, num_labels-1]
    if label_col in dataset["train"].column_names:
        raw_labels = dataset["train"][label_col]
        # Filter out '-1', '-', and '' regardless of if it's a string or int
        unique_labels = sorted(set(x for x in raw_labels if str(x).strip() not in ["-1", "-", ""]))
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: str(label) for label, idx in label2id.items()}
        num_labels = len(unique_labels)
    else:
        accelerator.print(f"Warning: label column '{label_col}' not found. Falling back to config.")
        num_labels = model_config.get('num_labels', 77)
        label2id = None

    accelerator.print(f"FINAL num_labels used for model: {num_labels}")
    
    accelerator.print(f"Loading tokenizer and classification model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # We do not pad to max_length here. We truncate to 128 and use dynamic padding.
    def tokenize_function(examples):
        if text_col2:
            tokenized = tokenizer(examples[text_col], examples[text_col2], truncation=True, max_length=128)
        else:
            tokenized = tokenizer(examples[text_col], truncation=True, max_length=128)
        
        if label2id is not None:
            # Look up label in mapping, ignore -1, -, and empty string
            tokenized["labels"] = [label2id[label] if str(label).strip() not in ["-1", "-", ""] else -1 for label in examples[label_col]]
        else:
            # Fallback if no mapping exists
            tokenized["labels"] = [int(label) for label in examples[label_col]]
        return tokenized
    
    accelerator.print("Tokenizing dataset...")
    with accelerator.main_process_first():
        tokenized_datasets = DatasetDict({
            split: dataset[split].map(
                tokenize_function, 
                batched=True, 
                remove_columns=dataset[split].column_names,
                load_from_cache_file=False
            )
            for split in dataset.keys()
        })
        
        # Verify labels are 0-indexed for the first few samples
        for i in range(min(5, len(tokenized_datasets['train']))):
            accelerator.print(f"Sample {i} mapped label = {tokenized_datasets['train'][i]['labels']}")

        tokenized_datasets.set_format("torch")
            
        # Handle MNLI splits explicitly
        if "validation" not in tokenized_datasets and "validation_matched" in tokenized_datasets:
            accelerator.print("Using 'validation_matched' as validation split.")
            tokenized_datasets["validation"] = tokenized_datasets["validation_matched"]
        if "test" not in tokenized_datasets and "validation_mismatched" in tokenized_datasets:
            accelerator.print("Using 'validation_mismatched' as test split.")
            tokenized_datasets["test"] = tokenized_datasets["validation_mismatched"]

        # Dynamically split 'train' to create a validation split if not present
        if "validation" not in tokenized_datasets:
            accelerator.print("Splitting training set to create a dynamic 'validation' split (10%)...")
            split_dataset = tokenized_datasets["train"].train_test_split(test_size=0.1, seed=seed)
            new_splits = {
                "train": split_dataset["train"],
                "validation": split_dataset["test"]
            }
            if "test" in tokenized_datasets:
                new_splits["test"] = tokenized_datasets["test"]
            tokenized_datasets = DatasetDict(new_splits)
            
    # Use DataCollatorWithPadding for dynamic padding (makes training up to 4x faster!)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    if "validation" in tokenized_datasets:
        eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size, collate_fn=data_collator)
    else:
        eval_dataloader = None
        
    if "test" in tokenized_datasets:
        test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size, collate_fn=data_collator)
    else:
        test_dataloader = None
        
    if accelerator.is_local_main_process:
        all_labels = []
        for split in tokenized_datasets.keys():
            labels = tokenized_datasets[split]["labels"]
            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.tolist())
            else:
                all_labels.extend([int(x) for x in labels])
        accelerator.print(f"GLOBAL MIN LABEL: {min(all_labels)}")
        accelerator.print(f"GLOBAL MAX LABEL: {max(all_labels)}")
        accelerator.print(f"NUM UNIQUE LABELS: {len(set(all_labels))}")

    # Print a few tokenized examples to verify what is going into the model
    if accelerator.is_local_main_process:
        accelerator.print("\n=== Sample Tokenized Inputs ===")
        for i in range(2):
            sample = tokenized_datasets["train"][i]
            input_ids = sample["input_ids"]
            label = sample["labels"]
            decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            
            accelerator.print(f"Example {i+1}:")
            accelerator.print(f"  Decoded Text: {decoded_text}")
            accelerator.print(f"  Input IDs (first 20): {input_ids[:20].tolist()}...")
            accelerator.print(f"  Label ID: {label.item() if hasattr(label, 'item') else label}\n")
        accelerator.print("===============================\n")
    
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    hf_config.num_labels = num_labels
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=hf_config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # EMERGENCY OVERRIDE: Ensure the model head matches num_labels
    head = getattr(model, "score", getattr(model, "classifier", None))
    
    # Safely get out_features and in_features depending on head type
    head_out = getattr(head, "out_features", getattr(getattr(head, "out_proj", None), "out_features", None))
    head_in = getattr(head, "in_features", getattr(getattr(head, "dense", None), "in_features", None))
    
    if head is None or model.config.num_labels != num_labels or (head_out is not None and head_out != num_labels):
        accelerator.print(f"CRITICAL WARNING: Model head mismatch detected. Re-initializing head to {num_labels} labels...")
        in_features = head_in if head_in is not None else getattr(model.config, "hidden_size", 1024)
        
        # If it's a RoBERTa model, we should ideally recreate the RobertaClassificationHead, but a Linear layer is a functional fallback
        new_head = torch.nn.Linear(in_features, num_labels)
        
        # CRITICAL FIX: Match the model's dtype and device to prevent mat1/mat2 dtype mismatch
        new_head.to(device=model.device, dtype=model.dtype)
        
        if hasattr(model, "score"):
            model.score = new_head
        elif hasattr(model, "classifier"):
            model.classifier = new_head
            
        model.config.num_labels = num_labels
        if hasattr(model, 'num_labels'):
            model.num_labels = num_labels
        model.config.id2label = {i: str(i) for i in range(num_labels)}
        model.config.label2id = {str(i): i for i in range(num_labels)}
        accelerator.print(f"Successfully forced classification head to {num_labels} labels with dtype {model.dtype}.")
    
    accelerator.print(f"Final model config num_labels: {model.config.num_labels}")
    if hasattr(model, 'num_labels'):
        accelerator.print(f"Final model num_labels attribute: {model.num_labels}")
    
    # Optional: freeze the model backbone to train only the classification head
    freeze_backbone = model_config.get('freeze_backbone', False)
    if freeze_backbone:
        accelerator.print("Freezing model backbone parameters. Only training classification head.")
        if hasattr(model, "base_model"):
            for param in model.base_model.parameters():
                param.requires_grad = False
        else:
            base_model = getattr(model, "base_model", getattr(model, "model", None))
            if base_model:
                for param in base_model.parameters():
                    param.requires_grad = False
            else:
                for name, param in model.named_parameters():
                    if "score" not in name and "classifier" not in name:
                        param.requires_grad = False
    else:
        accelerator.print("Model is fully trainable.")

    # STABLE PARAMETER ID INJECTION:
    # Inject a deterministic param_id into the parameter objects themselves
    # before passing them to the optimizer. This guarantees that `lozo.py` 
    # uses perfectly synchronized random seeds across all GPUs, regardless of 
    # how Accelerate/DDP wraps or reorders the parameters internally.
    for i, (name, p) in enumerate(model.named_parameters()):
        if p.requires_grad:
            p.param_id = i
    
    opt_name = opt_config.get('name', 'LOZO')
    opt_kwargs = opt_config.get('kwargs', {})
    
    is_zeroth_order = opt_name in ["LOZO", "LOZOM"]
    
    if is_zeroth_order:
        model.to(accelerator.device)
        if opt_name == "LOZOM":
            optimizer = LOZOM(model.parameters(), **opt_kwargs)
        else:
            optimizer = LOZO(model.parameters(), **opt_kwargs)
            
        optimizer, train_dataloader = accelerator.prepare(optimizer, train_dataloader)
        if test_dataloader:
            test_dataloader = accelerator.prepare(test_dataloader)
            
        # Scheduler setup (Optional)
        lr_scheduler = None
        if 'lr_scheduler' in train_config:
            scheduler_config = train_config.get('lr_scheduler', {})
            scheduler_type = scheduler_config.get('type', 'linear')
            warmup_ratio = scheduler_config.get('warmup_ratio', 0.0)
            warmup_steps = scheduler_config.get('warmup_steps', 0)
            start_lr = scheduler_config.get('start_lr', 0.0)
            
            if max_tokens is not None:
                # Estimate steps if max_tokens is used (approximation)
                avg_tokens_per_batch = 128 
                num_training_steps = max_tokens // (avg_tokens_per_batch * accelerator.num_processes)
            else:
                num_training_steps = len(train_dataloader) * epochs
            if warmup_steps == 0 and warmup_ratio > 0:
                num_warmup_steps = int(num_training_steps * warmup_ratio)
            else:
                num_warmup_steps = warmup_steps
    
            if start_lr > 0:
                peak_lr = opt_config.get('kwargs', {}).get('lr', 1e-6)
                import math # Ensure math is available for cosine
                def lr_lambda(current_step):
                    if current_step < num_warmup_steps:
                        return (start_lr + (peak_lr - start_lr) * float(current_step) / float(max(1, num_warmup_steps))) / peak_lr
                    if scheduler_type == "linear":
                        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
                    elif scheduler_type == "cosine":
                        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    return 1.0
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            else:
                lr_scheduler = get_scheduler(
                    name=scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                )
            lr_scheduler = accelerator.prepare(lr_scheduler)
    else:
        # Standard first-order optimizer
        if hasattr(torch.optim, opt_name):
            opt_class = getattr(torch.optim, opt_name)
            optimizer = opt_class(model.parameters(), **opt_kwargs)
        else:
            raise ValueError(f"Optimizer {opt_name} not found in torch.optim or custom definitions.")
            
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
        if eval_dataloader:
            eval_dataloader = accelerator.prepare(eval_dataloader)
        if test_dataloader:
            test_dataloader = accelerator.prepare(test_dataloader)
            
        # Scheduler setup (Optional)
        lr_scheduler = None
        if 'lr_scheduler' in train_config:
            scheduler_config = train_config.get('lr_scheduler', {})
            scheduler_type = scheduler_config.get('type', 'linear')
            warmup_ratio = scheduler_config.get('warmup_ratio', 0.0)
            warmup_steps = scheduler_config.get('warmup_steps', 0)
            
            if max_tokens is not None:
                # Estimate steps if max_tokens is used (approximation)
                avg_tokens_per_batch = 128 
                num_training_steps = max_tokens // (avg_tokens_per_batch * accelerator.num_processes)
            else:
                num_training_steps = len(train_dataloader) * epochs
            if warmup_steps == 0 and warmup_ratio > 0:
                num_warmup_steps = int(num_training_steps * warmup_ratio)
            else:
                num_warmup_steps = warmup_steps
                
            lr_scheduler = get_scheduler(
                name=scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
            lr_scheduler = accelerator.prepare(lr_scheduler)
            
    accelerator.print(f"Starting training for {epochs} epochs using {opt_name} optimizer")
    
    total_train_tokens = len(tokenized_datasets["train"]) * 128
    accelerator.print(f"Total tokens in the whole training set: {total_train_tokens}")
    accelerator.log({"dataset_total_tokens": total_train_tokens}, step=0)
    
    global_step = 0
    run_start_time = time.time()
    best_eval_accuracy = -1.0
    total_tokens_seen = 0
    
    epoch = 0
    while True:
        if max_tokens is None and epoch >= epochs:
            break
            
        if is_zeroth_order:
            model.eval() # Disable dropout / stochastic noise for stable zeroth-order updates
        else:
            model.train()
            
        total_loss = 0
        progress_bar = tqdm(train_dataloader, disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            step_start_time = time.time()
            if is_zeroth_order:
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                
                unwrapped_model = accelerator.unwrap_model(model)
                step_loss_container = []
                def closure():
                    if global_step == 0 and accelerator.is_local_main_process:
                        accelerator.print(f"DEBUG: labels max={batch['labels'].max().item()}, min={batch['labels'].min().item()}")
                    
                    outputs = unwrapped_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    loss = outputs.loss
                    step_loss_container.append(loss.detach())
                    return loss
                    
                optimizer.step(closure)
                
                # Perform distributed reduction OUTSIDE the closure to avoid deadlocks
                if len(step_loss_container) > 0:
                    loss_tensor = accelerator.reduce(step_loss_container[0], reduction="mean")
                    loss = loss_tensor.item()
                else:
                    loss = 0.0
                total_loss += loss
                progress_bar.set_description(f"Epoch {epoch+1} Loss: {loss:.4f}")
                train_loss_val = loss
            else:
                # First order standard training
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    
                total_loss += loss.item()
                progress_bar.set_description(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
                train_loss_val = loss.item()
                
            if lr_scheduler is not None:
                lr_scheduler.step()
            step_time = time.time() - step_start_time
            local_bsz = batch["labels"].size(0)
            
            # Real token count throughput
            step_tokens = batch["attention_mask"].sum().item()
            step_tokens *= accelerator.num_processes
            total_tokens_seen += step_tokens
            
            samples_per_second = (local_bsz * accelerator.num_processes) / step_time
            
            log_metrics = {
                "train_loss": train_loss_val,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "step_time_sec": step_time,
                "samples_per_second": samples_per_second,
                "total_tokens_seen": total_tokens_seen
            }
            if torch.cuda.is_available():
                log_metrics["gpu_memory_MB"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
                
            accelerator.log(log_metrics, step=global_step)
            global_step += 1
            
            if max_tokens is not None and total_tokens_seen >= max_tokens:
                accelerator.print(f"Reached max_tokens ({max_tokens}). Stopping training loop.")
                break
                
        avg_train_loss = total_loss / (len(train_dataloader) if len(train_dataloader) > 0 else 1)
        accelerator.print(f"Epoch {epoch+1} finished. Avg train loss: {avg_train_loss:.4f} | Total tokens seen: {total_tokens_seen}")
        
        # Evaluation
        if eval_dataloader and (epoch + 1) % eval_epochs == 0:
            accelerator.print(f"\n--- Starting Evaluation for Epoch {epoch+1} ---")
            model.eval()
            total_eval_loss = 0
            correct_preds = 0
            total_preds = 0
            eval_unwrapped_model = accelerator.unwrap_model(model)
            with torch.no_grad():
                for batch in eval_dataloader:
                    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                    outputs = eval_unwrapped_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    eval_loss = outputs.loss
                    avg_loss = accelerator.reduce(eval_loss.detach(), reduction="mean")
                    total_eval_loss += avg_loss.item()
                    
                    predictions = outputs.logits.argmax(dim=-1)
                    
                    # Ignore -1 labels for evaluation
                    valid_mask = batch["labels"] != -1
                    if valid_mask.sum() == 0:
                        continue
                    
                    local_correct = (predictions[valid_mask] == batch["labels"][valid_mask]).sum().to(accelerator.device)
                    local_total = valid_mask.sum().to(accelerator.device)
                    
                    batch_correct = accelerator.reduce(local_correct, reduction="sum")
                    batch_total = accelerator.reduce(local_total, reduction="sum")
                    
                    correct_preds += batch_correct.item()
                    total_preds += batch_total.item()
                        
            avg_eval_loss = total_eval_loss / len(eval_dataloader)
            accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
            elapsed_since_start = time.time() - run_start_time
            
            accelerator.print(f"Epoch {epoch+1} Eval Loss: {avg_eval_loss:.4f} | Accuracy: {accuracy:.4f} | Elapsed Time: {elapsed_since_start:.2f}s")
            accelerator.log({
                "eval_loss": avg_eval_loss,
                "eval_accuracy": accuracy,
                "total_elapsed_time_sec": elapsed_since_start,
                "epoch": epoch+1
            }, step=global_step)
            
            if accelerator.is_local_main_process:
                accelerator.print("Saving checkpoints on main process...")
                unwrapped_model = accelerator.unwrap_model(model)
                if accuracy > best_eval_accuracy:
                    best_eval_accuracy = accuracy
                    accelerator.print(f"New best accuracy ({accuracy:.4f})! Saving best_checkpoint_cls...")
                    unwrapped_model.save_pretrained("best_checkpoint_cls")
                    tokenizer.save_pretrained("best_checkpoint_cls")
                    
                accelerator.print("Saving last_checkpoint_cls...")
                unwrapped_model.save_pretrained("last_checkpoint_cls")
                tokenizer.save_pretrained("last_checkpoint_cls")
                accelerator.print("Checkpoints saved successfully.")
            
            # CRITICAL MULTI-GPU BARRIER: Wait for main process to finish disk I/O before continuing training!
            accelerator.wait_for_everyone()
            accelerator.print(f"--- Evaluation for Epoch {epoch+1} Complete ---\n")
                
        epoch += 1
            
    total_run_time = time.time() - run_start_time
    accelerator.print(f"Training completed in {total_run_time:.2f} seconds.")
    accelerator.log({"final_total_time_sec": total_run_time}, step=global_step)
    
    # Final evaluation on unseen test set
    if test_dataloader:
        accelerator.print("\n=== Running Final Evaluation on the Unseen Test Set ===")
        if os.path.exists("best_checkpoint_cls"):
            accelerator.print("Loading best classification checkpoint for final test evaluation...")
            model = AutoModelForSequenceClassification.from_pretrained("best_checkpoint_cls", trust_remote_code=True).to(accelerator.device)
            
        model.eval()
        total_test_loss = 0
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for batch in test_dataloader:
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                test_loss = outputs.loss
                avg_loss = accelerator.reduce(test_loss.detach(), reduction="mean")
                total_test_loss += avg_loss.item()
                
                predictions = outputs.logits.argmax(dim=-1)
                
                valid_mask = batch["labels"] != -1
                if valid_mask.sum() == 0:
                    continue
                
                local_correct = (predictions[valid_mask] == batch["labels"][valid_mask]).sum().to(accelerator.device)
                local_total = valid_mask.sum().to(accelerator.device)
                
                batch_correct = accelerator.reduce(local_correct, reduction="sum")
                batch_total = accelerator.reduce(local_total, reduction="sum")
                
                correct_preds += batch_correct.item()
                total_preds += batch_total.item()
                    
        avg_test_loss = total_test_loss / len(test_dataloader)
        test_accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
        
        accelerator.print(f"Final Test Loss: {avg_test_loss:.4f} | Final Test Accuracy: {test_accuracy:.4f}")
        accelerator.log({
            "test_loss": avg_test_loss,
            "test_accuracy": test_accuracy
        }, step=global_step)
        
    accelerator.wait_for_everyone()
    
    if push_to_hub and repo_id and accelerator.is_local_main_process:
        from huggingface_hub import HfApi
        api = HfApi()
        accelerator.print(f"Pushing classification checkpoints to Hugging Face Hub: {repo_id}")
        api.create_repo(repo_id=repo_id, exist_ok=True)
        
        if os.path.exists("best_checkpoint_cls"):
            api.upload_folder(
                folder_path="best_checkpoint_cls",
                repo_id=repo_id,
                path_in_repo="best_checkpoint_cls",
                commit_message="Upload best classification checkpoint"
            )
        if os.path.exists("last_checkpoint_cls"):
            api.upload_folder(
                folder_path="last_checkpoint_cls",
                repo_id=repo_id,
                path_in_repo="last_checkpoint_cls",
                commit_message="Upload last classification checkpoint"
            )
        accelerator.print("Push complete.")
        
    accelerator.end_training()

if __name__ == "__main__":
    main()
