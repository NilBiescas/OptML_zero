import os
import yaml
import argparse
import torch
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from optimizers.lozo import LOZOM, LOZO

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune model using config")
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
        
    accelerator.init_trackers(project_name="lozo-training", config=config, **init_kwargs)
    
    # Crucial: set seed across all processes to ensure identical weight initializations 
    # and identical U/V generations in the optimizer across all ranks.
    set_seed(seed)
    torch.manual_seed(seed)
    
    dataset_name = dataset_config.get('name', 'PolyAI/banking77')
    text_col = dataset_config.get('text_column', 'text')
    label_col = dataset_config.get('label_column', 'label')
    
    accelerator.print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name)
    
    model_name = model_config.get('name', 'Qwen/Qwen3.5-0.8B')
    num_labels = model_config.get('num_labels', 77)
    
    accelerator.print(f"Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(examples[text_col], padding="max_length", truncation=True, max_length=128)
    
    with accelerator.main_process_first():
        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=[text_col])
        tokenized_datasets.set_format("torch")
        
        # Ensure label column is named 'labels' if it's not already
        if label_col != 'label' and label_col in tokenized_datasets['train'].column_names:
            tokenized_datasets = tokenized_datasets.rename_column(label_col, 'label')
            
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size)
    if "test" in tokenized_datasets:
        eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size)
    elif "validation" in tokenized_datasets:
        eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size)
    else:
        eval_dataloader = None
        
    # Print a few tokenized examples to verify what is going into the model
    if accelerator.is_local_main_process:
        accelerator.print("\n=== Sample Tokenized Inputs ===")
        for i in range(3):
            sample = tokenized_datasets["train"][i]
            input_ids = sample["input_ids"]
            label = sample["label"]
            decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            
            accelerator.print(f"Example {i+1}:")
            accelerator.print(f"  Decoded Text: {decoded_text}")
            accelerator.print(f"  Input IDs (first 20): {input_ids[:20].tolist()}...")
            accelerator.print(f"  Label ID: {label.item() if hasattr(label, 'item') else label}\n")
        accelerator.print("===============================\n")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    opt_name = opt_config.get('name', 'LOZO')
    opt_kwargs = opt_config.get('kwargs', {})
    
    is_zeroth_order = opt_name in ["LOZO", "LOZOM"]
    
    if is_zeroth_order:
        # Move model to local device. We DO NOT pass it to accelerator.prepare to avoid DDP wrapper.
        # The ZO optimizer does not use backward(), so DDP's gradient syncing is unnecessary and could hang.
        model.to(accelerator.device)
        if opt_name == "LOZOM":
            optimizer = LOZOM(model.parameters(), **opt_kwargs)
        else:
            optimizer = LOZO(model.parameters(), **opt_kwargs)
            
        optimizer, train_dataloader = accelerator.prepare(optimizer, train_dataloader)
        if eval_dataloader:
            eval_dataloader = accelerator.prepare(eval_dataloader)
    else:
        # For standard first-order optimizers
        if hasattr(torch.optim, opt_name):
            opt_class = getattr(torch.optim, opt_name)
            optimizer = opt_class(model.parameters(), **opt_kwargs)
        else:
            raise ValueError(f"Optimizer {opt_name} not found in torch.optim or custom definitions.")
            
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
        if eval_dataloader:
            eval_dataloader = accelerator.prepare(eval_dataloader)
            
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
            
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            step_start_time = time.time()
            if is_zeroth_order:
                # Move batch to device
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                
                def closure():
                    outputs = model(**batch)
                    loss = outputs.loss
                    # Average loss across all processes
                    avg_loss = accelerator.reduce(loss.detach(), reduction="mean")
                    return avg_loss
                
                loss = optimizer.step(closure)
                total_loss += loss
                progress_bar.set_description(f"Epoch {epoch+1} Loss: {loss:.4f}")
                train_loss_val = loss
            else:
                # First order standard training
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    
                total_loss += loss.item()
                progress_bar.set_description(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
                train_loss_val = loss.item()
            
            step_time = time.time() - step_start_time
            # Calculate throughput based on the actual local batch size
            local_bsz = batch["label"].size(0)
            seq_len = batch["input_ids"].size(1)
            step_tokens = local_bsz * seq_len * accelerator.num_processes
            total_tokens_seen += step_tokens
            
            samples_per_second = (local_bsz * accelerator.num_processes) / step_time
            
            log_metrics = {
                "train_loss": train_loss_val,
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
            model.eval()
            correct = 0
            total = 0
            total_eval_loss = 0
            with torch.no_grad():
                for batch in eval_dataloader:
                    if is_zeroth_order:
                        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                        unwrapped_model = model
                    else:
                        unwrapped_model = accelerator.unwrap_model(model)
                        
                    outputs = unwrapped_model(**batch)
                    eval_loss = outputs.loss
                    predictions = outputs.logits.argmax(dim=-1)
                    
                    # Gather predictions across devices
                    predictions, labels = accelerator.gather_for_metrics((predictions, batch["label"]))
                    avg_loss = accelerator.reduce(eval_loss.detach(), reduction="mean")
                    
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                    total_eval_loss += avg_loss.item()
                    
            accuracy = correct / total
            avg_eval_loss = total_eval_loss / len(eval_dataloader)
            elapsed_since_start = time.time() - run_start_time
            
            accelerator.print(f"Epoch {epoch+1} Eval Accuracy: {accuracy:.4f} | Eval Loss: {avg_eval_loss:.4f} | Elapsed Time: {elapsed_since_start:.2f}s")
            accelerator.log({
                "eval_accuracy": accuracy, 
                "eval_loss": avg_eval_loss, 
                "total_elapsed_time_sec": elapsed_since_start,
                "epoch": epoch+1
            }, step=global_step)
            
            if accelerator.is_local_main_process:
                if accuracy > best_eval_accuracy:
                    best_eval_accuracy = accuracy
                    accelerator.print(f"New best accuracy ({accuracy:.4f})! Saving to local 'best_checkpoint' folder...")
                    unwrapped_model.save_pretrained("best_checkpoint")
                    tokenizer.save_pretrained("best_checkpoint")
                    
                # Always save the last checkpoint
                unwrapped_model.save_pretrained("last_checkpoint")
                tokenizer.save_pretrained("last_checkpoint")
                
        epoch += 1
        if max_tokens is not None and total_tokens_seen >= max_tokens:
            break

    total_run_time = time.time() - run_start_time
    accelerator.print(f"Training completed in {total_run_time:.2f} seconds.")
    # Log the final total time
    accelerator.log({"final_total_time_sec": total_run_time}, step=global_step)
    
    accelerator.end_training()
    accelerator.wait_for_everyone()
    
    if push_to_hub and repo_id and accelerator.is_local_main_process:
        from huggingface_hub import HfApi
        api = HfApi()
        accelerator.print(f"Pushing checkpoints to Hugging Face Hub: {repo_id}")
        api.create_repo(repo_id=repo_id, exist_ok=True)
        
        if os.path.exists("best_checkpoint"):
            accelerator.print("Uploading best_checkpoint...")
            api.upload_folder(
                folder_path="best_checkpoint",
                repo_id=repo_id,
                path_in_repo="best_checkpoint",
                commit_message="Upload best checkpoint"
            )
        if os.path.exists("last_checkpoint"):
            accelerator.print("Uploading last_checkpoint...")
            api.upload_folder(
                folder_path="last_checkpoint",
                repo_id=repo_id,
                path_in_repo="last_checkpoint",
                commit_message="Upload last checkpoint"
            )
        accelerator.print("Push complete.")

if __name__ == "__main__":
    main()
