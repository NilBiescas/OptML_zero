import os
import yaml
import argparse
import torch
import time
import math
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, DataCollatorForSeq2Seq, get_scheduler
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from optimizers.lozo import LOZOM, LOZO

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune generative model using config")
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
    max_tokens = train_config.get('max_tokens', None)
    
    # Initialize accelerator
    accelerator = Accelerator(log_with="wandb")
    
    # Try to set the WandB run name to match the RunAI job name
    run_name = os.environ.get("RUN_NAME", None)
    init_kwargs = {}
    if run_name:
        init_kwargs["init_kwargs"] = {"wandb": {"name": run_name}}
        
    accelerator.init_trackers(project_name="lozo-generative-training", config=config, **init_kwargs)
    
    # Crucial: set seed across all processes to ensure deterministic initializations
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    dataset_name = dataset_config.get('name', 'mteb/banking77')
    text_col = dataset_config.get('text_column', 'text')
    label_col = dataset_config.get('label_column', 'label')
    
    accelerator.print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, trust_remote_code=True)
    
    model_name = model_config.get('name', 'Qwen/Qwen3.5-0.8B')
    
    accelerator.print(f"Loading tokenizer and causal model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Robustly get/convert label to string
    has_label_text = "label_text" in dataset["train"].column_names
    label_names = None
    if not has_label_text:
        feature = dataset["train"].features[label_col]
        if hasattr(feature, "names"):
            label_names = feature.names
    # 1. Format the dataset into "User: <text>\nAssistant: <label_name>"
    def format_example(example):
        text = example[text_col]
        # Robust label selection
        if has_label_text:
            label = str(example["label_text"])
        elif label_names is not None:
            label = str(label_names[example[label_col]])
        else:
            label = str(example[label_col])

        prompt = f"User: {text}\nAssistant:"
        answer = f" {label}"

        full_text = prompt + answer

        # Tokenize the prompt to find where the assistant's answer starts
        prompt_tokenized = tokenizer(prompt, truncation=True, max_length=128)
        answer_start = len(prompt_tokenized["input_ids"])

        return {
            "formatted_text": full_text,
            "answer_start": answer_start
        }
        
    accelerator.print("Formatting dataset into conversational templates...")
    with accelerator.main_process_first():
        formatted_dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)
        
    # 2. Tokenize the formatted text
    def tokenize_function(examples):
        # We also need to keep answer_start to use it in add_labels
        tokenized = tokenizer(examples["formatted_text"], truncation=True, max_length=128)
        tokenized["answer_start"] = examples["answer_start"]
        return tokenized
        
    with accelerator.main_process_first():
        tokenized_datasets = formatted_dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=["formatted_text"]
        )
        
    # 3. Add labels and mask the prompt tokens with -100
    def add_labels(example):
        input_ids = example["input_ids"]
        labels = list(input_ids)
        answer_start = example["answer_start"]
        for i in range(min(answer_start, len(labels))):
            labels[i] = -100
        # Remove answer_start as it's no longer needed
        return {"labels": labels}
        
    with accelerator.main_process_first():
        tokenized_datasets = tokenized_datasets.map(add_labels, remove_columns=["answer_start"])
        tokenized_datasets.set_format("torch")
        
        # Dynamically split 'train' to create a validation split if not present (to leave 'test' untouched for final evaluation!)
        if "test" in tokenized_datasets and "validation" not in tokenized_datasets:
            accelerator.print("Splitting training set to create a dynamic 'validation' split (10%)...")
            split_dataset = tokenized_datasets["train"].train_test_split(test_size=0.1, seed=seed)
            tokenized_datasets = DatasetDict({
                "train": split_dataset["train"],
                "validation": split_dataset["test"],
                "test": tokenized_datasets["test"]
            })
            
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Use DataCollatorForSeq2Seq which handles dynamic padding of inputs and pads labels with -100
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    if "validation" in tokenized_datasets:
        eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size, collate_fn=data_collator)
    else:
        eval_dataloader = None
        
    if "test" in tokenized_datasets:
        test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size, collate_fn=data_collator)
    else:
        test_dataloader = None
        
    # Optional: freezing the model backbone is removed. Model must be fully trainable.
    accelerator.print("Model is fully trainable.")

    # STABLE PARAMETER ID INJECTION:
    # Inject a deterministic param_id into the parameter objects themselves
    # before passing them to the optimizer. This guarantees that `lozo.py` 
    # uses perfectly synchronized random seeds across all GPUs, regardless of 
    # how Accelerate/DDP wraps or reorders the parameters internally.
    for i, (name, p) in enumerate(model.named_parameters()):
        if p.requires_grad:
            p.param_id = i

    # Print a few examples to verify formatting and masking
    if accelerator.is_local_main_process:
        accelerator.print("\n=== Sample Tokenized Generative Inputs ===")
        for i in range(2):
            sample = tokenized_datasets["train"][i]
            input_ids = sample["input_ids"]
            labels = sample["labels"]
            
            # Reconstruct what the model is trained to predict vs what is masked
            decoded_input = tokenizer.decode(input_ids)
            decoded_target = tokenizer.decode([t for t in labels if t != -100])
            
            accelerator.print(f"Example {i+1}:")
            accelerator.print(f"  Decoded Input: {decoded_input}")
            accelerator.print(f"  Target Prediction (Unmasked): {decoded_target}")
            accelerator.print(f"  Masked Label IDs: {labels.tolist()[:30]}...\n")
        accelerator.print("==========================================\n")
        
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
    
    global_step = 0
    run_start_time = time.time()
    best_eval_loss = float('inf')
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
                
                debug_print = False
                if debug_print:
                    non_masked = (batch["labels"] != -100).sum().item()
                    total_elem = batch["labels"].numel()
                    accelerator.print(f"\n[DEBUG Step {global_step}] Batch input_ids: {batch['input_ids'].shape}, Non-masked labels: {non_masked}/{total_elem}")

                unwrapped_model = accelerator.unwrap_model(model)
                step_loss_container = []
                def closure():
                    outputs = unwrapped_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    loss = outputs.loss
                    if debug_print:
                        accelerator.print(f"[DEBUG Step {global_step}] Raw Model loss: {loss.item() if loss is not None else 'None'}")
                    # Distributed reduction for multi-GPU ZO gradient consistency
                    avg_loss = accelerator.reduce(loss.detach(), reduction="mean")
                    if debug_print:
                        accelerator.print(f"[DEBUG Step {global_step}] Reduced avg_loss: {avg_loss.item() if avg_loss is not None else 'None'}")
                    step_loss_container.append(avg_loss.item())
                    return avg_loss
                    
                optimizer.step(closure)
                loss = step_loss_container[0] if len(step_loss_container) > 0 else 0.0
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
        if eval_dataloader:
            accelerator.print(f"\n--- Starting Evaluation for Epoch {epoch+1} ---")
            model.eval()
            total_eval_loss = 0
            correct_tokens = 0
            total_tokens = 0
            eval_unwrapped_model = accelerator.unwrap_model(model)
            with torch.no_grad():
                for eval_step_idx, batch in enumerate(eval_dataloader):
                    # if eval_step_idx % 10 == 0 and accelerator.is_local_main_process:
                    #     accelerator.print(f"[Eval Epoch {epoch+1}] Processing batch {eval_step_idx}/{len(eval_dataloader)}")
                    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                    outputs = eval_unwrapped_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    eval_loss = outputs.loss
                    avg_loss = accelerator.reduce(eval_loss.detach(), reduction="mean")
                    total_eval_loss += avg_loss.item()
                    
                    # Compute token-level accuracy over prediction targets
                    # Shift logits and labels so that prediction L_i aligns with target y_{i+1}
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = batch["labels"][..., 1:].contiguous()
                    predictions = shift_logits.argmax(dim=-1)
                    
                    local_mask = (shift_labels != -100)
                    local_correct = (predictions[local_mask] == shift_labels[local_mask]).sum().to(accelerator.device)
                    local_total = local_mask.sum().to(accelerator.device)
                    
                    # Reduce scalars across all GPUs (100% robust against differing dynamic padding seq_len across GPUs)
                    batch_correct = accelerator.reduce(local_correct, reduction="sum")
                    batch_total = accelerator.reduce(local_total, reduction="sum")
                    
                    correct_tokens += batch_correct.item()
                    total_tokens += batch_total.item()
                    
                    # if eval_step_idx == 0 and accelerator.is_local_main_process and local_mask.sum() > 0:
                    #     sample_preds = predictions[local_mask][:20]
                    #     sample_targets = shift_labels[local_mask][:20]
                    #     pred_str = tokenizer.decode(sample_preds)
                    #     target_str = tokenizer.decode(sample_targets)
                    #     accelerator.print(f"\n[EVAL SANITY CHECK] Sample predictions decoded: {repr(pred_str)}")
                    #     accelerator.print(f"[EVAL SANITY CHECK] Sample targets decoded:     {repr(target_str)}\n")
                        
            avg_eval_loss = total_eval_loss / len(eval_dataloader)
            perplexity = math.exp(avg_eval_loss) if avg_eval_loss < 20 else float('inf')
            token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
            elapsed_since_start = time.time() - run_start_time
            
            accelerator.print(f"Epoch {epoch+1} Eval Loss: {avg_eval_loss:.4f} | Perplexity: {perplexity:.2f} | Token Accuracy: {token_acc:.4f} | Elapsed Time: {elapsed_since_start:.2f}s")
            accelerator.log({
                "eval_loss": avg_eval_loss,
                "perplexity": perplexity,
                "eval_token_accuracy": token_acc,
                "total_elapsed_time_sec": elapsed_since_start,
                "epoch": epoch+1
            }, step=global_step)
            
            if accelerator.is_local_main_process:
                accelerator.print("Saving checkpoints on main process...")
                unwrapped_model = accelerator.unwrap_model(model)
                if avg_eval_loss < best_eval_loss:
                    best_eval_loss = avg_eval_loss
                    accelerator.print(f"New best validation loss ({avg_eval_loss:.4f})! Saving best_checkpoint_causal...")
                    unwrapped_model.save_pretrained("best_checkpoint_causal")
                    tokenizer.save_pretrained("best_checkpoint_causal")
                    
                accelerator.print("Saving last_checkpoint_causal...")
                unwrapped_model.save_pretrained("last_checkpoint_causal")
                tokenizer.save_pretrained("last_checkpoint_causal")
                accelerator.print("Checkpoints saved successfully.")
            
            # CRITICAL MULTI-GPU BARRIER: Wait for main process to finish disk I/O before continuing training!
            accelerator.wait_for_everyone()
            accelerator.print(f"--- Evaluation for Epoch {epoch+1} Complete ---\n")
                
        epoch += 1
        # This condition is now redundant as max_tokens logic is handled per batch step
        # if max_tokens is not None and total_tokens_seen >= max_tokens:
        #     break
            
    total_run_time = time.time() - run_start_time
    accelerator.print(f"Training completed in {total_run_time:.2f} seconds.")
    accelerator.log({"final_total_time_sec": total_run_time}, step=global_step)
    
    # Final evaluation on unseen test set
    if test_dataloader:
        accelerator.print("\n=== Running Final Evaluation on the Unseen Test Set ===")
        if os.path.exists("best_checkpoint_causal"):
            accelerator.print("Loading best causal checkpoint for final test evaluation...")
            model = AutoModelForCausalLM.from_pretrained("best_checkpoint_causal", trust_remote_code=True).to(accelerator.device)
            
        model.eval()
        total_test_loss = 0
        correct_tokens = 0
        total_tokens = 0
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
                
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                predictions = shift_logits.argmax(dim=-1)
                
                local_mask = (shift_labels != -100)
                local_correct = (predictions[local_mask] == shift_labels[local_mask]).sum().to(accelerator.device)
                local_total = local_mask.sum().to(accelerator.device)
                
                batch_correct = accelerator.reduce(local_correct, reduction="sum")
                batch_total = accelerator.reduce(local_total, reduction="sum")
                
                correct_tokens += batch_correct.item()
                total_tokens += batch_total.item()
                    
        avg_test_loss = total_test_loss / len(test_dataloader)
        test_perplexity = math.exp(avg_test_loss) if avg_test_loss < 20 else float('inf')
        test_token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        
        accelerator.print(f"Final Test Loss: {avg_test_loss:.4f} | Final Test Perplexity: {test_perplexity:.2f} | Final Test Token Accuracy: {test_token_acc:.4f}")
        accelerator.log({
            "test_loss": avg_test_loss,
            "test_perplexity": test_perplexity,
            "test_token_accuracy": test_token_acc
        }, step=global_step)
        
    accelerator.wait_for_everyone()
    
    if push_to_hub and repo_id and accelerator.is_local_main_process:
        from huggingface_hub import HfApi
        api = HfApi()
        accelerator.print(f"Pushing causal checkpoints to Hugging Face Hub: {repo_id}")
        api.create_repo(repo_id=repo_id, exist_ok=True)
        
        if os.path.exists("best_checkpoint_causal"):
            api.upload_folder(
                folder_path="best_checkpoint_causal",
                repo_id=repo_id,
                path_in_repo="best_checkpoint_causal",
                commit_message="Upload best causal checkpoint"
            )
        if os.path.exists("last_checkpoint_causal"):
            api.upload_folder(
                folder_path="last_checkpoint_causal",
                repo_id=repo_id,
                path_in_repo="last_checkpoint_causal",
                commit_message="Upload last causal checkpoint"
            )
        accelerator.print("Push complete.")
        
    accelerator.end_training()

if __name__ == "__main__":
    main()
