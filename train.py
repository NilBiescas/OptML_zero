import os
import json
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
from huggingface_hub import HfApi

# Add subfolders to Python search path for dynamic helper loading
import sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_script_dir, "SubZero", "large_models"))
sys.path.append(os.path.join(_script_dir, "PseuZO"))
sys.path.append(os.path.join(_script_dir, "LOZO"))
from subzero_helper import SubZeroTrainerHelper
from pzo_helper import PZOTrainerHelper
from lozo_helper import LOZO, LOZOM, SparseMeZO, DiZO, MeZO, HiZOO, ConMeZO, FZOO, ZOMuon
from zo_helpers import LOZOTrainerHelper, PZOTrainerHelper, SubZeroTrainerHelper


@torch.no_grad()
def _compute_paper_acc(model, tokenizer, raw_examples, device):
    """Per-candidate log-likelihood eval that matches MeZO/ConMeZO/FZOO/
    ZO-Muon papers. Each example provides (prompt, all_choices, answer_idx);
    we score each candidate by summed token log-prob given the prompt and
    pick the argmax. Returns scalar accuracy in [0, 1].
    """
    correct = 0
    n = len(raw_examples)
    for ex in raw_examples:
        prompt   = ex["prompt"]
        choices  = ex["all_choices"]
        gold_idx = ex["answer_idx"]
        scores = []
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        plen = prompt_ids.size(1)
        for cand in choices:
            full_ids = tokenizer(prompt + cand,
                                 return_tensors="pt").input_ids.to(device)
            clen = full_ids.size(1) - plen
            if clen <= 0:
                scores.append(float("-inf")); continue
            out = model(full_ids)
            logp = torch.nn.functional.log_softmax(out.logits[0].float(), dim=-1)
            total = 0.0
            for k in range(clen):
                total += logp[plen - 1 + k, full_ids[0, plen + k]].item()
            scores.append(total)
        pred = int(max(range(len(scores)), key=lambda k: scores[k]))
        if pred == gold_idx:
            correct += 1
    return correct / n if n else 0.0

def _try_pull_checkpoint(repo_id: str) -> None:
    """Download last_checkpoint_causal/ from HF Hub into the working directory.

    Called at pod startup so a preempted job can resume from the last pushed
    checkpoint rather than starting from scratch. Silently does nothing if the
    repo or folder doesn't exist yet (first run).
    """
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=".",
            allow_patterns=["last_checkpoint_causal/**"],
            ignore_patterns=["*.lock"],
        )
    except Exception:
        pass  # repo doesn't exist yet or network error — start fresh


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
    # New: hard step cap (paper-faithful configs set this; the legacy paths
    # leave it None and let `epochs` drive termination).
    max_steps = train_config.get('max_steps', None)
    eval_steps = train_config.get('eval_steps', None)
    eval_epochs = train_config.get('eval_epochs', 1)

    # Initialize accelerator
    accelerator = Accelerator(log_with="wandb")
    
    # Try to set the WandB run name to match the RunAI job name.
    # Entity defaults to the wandb-logged-in user; override with WANDB_ENTITY.
    # Project defaults to WANDB_PROJECT env or the legacy project name.
    run_name    = os.environ.get("RUN_NAME", None)
    wandb_entity = os.environ.get("WANDB_ENTITY") or None
    wandb_kwargs = {}
    if wandb_entity:
        wandb_kwargs["entity"] = wandb_entity
    if run_name:
        wandb_kwargs["name"] = run_name
    init_kwargs = {"init_kwargs": {"wandb": wandb_kwargs}}

    project_name = os.environ.get("WANDB_PROJECT", "lozo-generative-training")
    accelerator.init_trackers(project_name=project_name, config=config, **init_kwargs)
    
    # --- Preemption resume: pull last checkpoint from HF Hub before anything else ---
    resume_state = None
    if push_to_hub and repo_id:
        with accelerator.main_process_first():
            _try_pull_checkpoint(repo_id)
        _state_path = "last_checkpoint_causal/training_state.json"
        if os.path.exists(_state_path):
            with open(_state_path) as _f:
                resume_state = json.load(_f)
            accelerator.print(
                f"[Resume] Found checkpoint — epoch {resume_state['epoch']}, "
                f"step {resume_state['global_step']}. Resuming."
            )
        else:
            accelerator.print("[Resume] No checkpoint on HF Hub — starting fresh.")

    # Crucial: set seed across all processes to ensure deterministic initializations
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Template / dataset dispatch.
    # - "chat" (default, backward-compatible) -> banking77-style
    #     "User: <text>\nAssistant: <label>" template loaded from a single
    #     HF dataset name. This is the original behaviour of the repo.
    # - "mezo" -> paper-faithful track. Use `data.mezo_tasks.load_mezo_task`
    #     with the MeZO/ZO-Muon templates, dataset ids, and 1000/500/1000
    #     sampling protocol. Required for reproducing ZO-Muon / ConMeZO / FZOO
    #     headline numbers.
    template_kind = dataset_config.get('template', 'chat')

    # Refuse classification-head configs in this entrypoint: they target the
    # separate classificationhead/train.py trainer that has its own head and
    # eval logic. Catch the misroute early so the user can re-run the right
    # script instead of training the wrong objective for hours.
    if model_config.get('classification_head'):
        raise ValueError(
            "model.classification_head=True is set, which targets "
            "classificationhead/train.py, not this script. "
            "Re-run with: python classificationhead/train.py --config <yaml>"
        )

    # Unified sequence length. Defaults: 128 for the legacy chat path (banking77
    # is short), 512 for MeZO tasks (SQuAD context can be long). Always overridable
    # via `training.max_seq_len`.
    default_max_seq_len = 512 if template_kind == "mezo" else 128
    max_seq_len = int(train_config.get('max_seq_len', default_max_seq_len))

    model_name = model_config.get('name', 'Qwen/Qwen3.5-0.8B')
    accelerator.print(f"Loading tokenizer and causal model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Raw (prompt, choices, gold_idx) tuples kept around so we can run the
    # paper-style per-candidate LL eval at every checkpoint. Only populated
    # for the mezo template path; chat path leaves this None.
    raw_eval_for_paper_acc = None

    if template_kind == "mezo":
        from data.mezo_tasks import load_mezo_task
        task_name = dataset_config.get('task')
        if not task_name:
            raise ValueError(
                "config.dataset.template == 'mezo' requires "
                "config.dataset.task (e.g. 'sst2', 'rte', 'boolq', ...)")
        accelerator.print(f"Loading MeZO-style task '{task_name}'...")
        num_train = dataset_config.get('num_train', 1000)
        num_dev   = dataset_config.get('num_dev', 500)
        num_eval  = dataset_config.get('num_eval', 1000)
        formatted_dataset = load_mezo_task(
            task_name, seed=seed,
            num_train=num_train, num_dev=num_dev, num_eval=num_eval,
        )

        # Capture raw eval examples BEFORE column-stripping, for the
        # per-candidate LL eval that matches paper accuracy. Use the held-out
        # "test" split (= the GLUE validation set, 1000 ex) — what the paper
        # reports on. The "validation" split here is just the dev carve from
        # the train pool (500 ex) and inflates accuracy by ~3pp.
        _paper_eval_split = "test" if "test" in formatted_dataset else "validation"
        raw_eval_for_paper_acc = [
            {"prompt":      ex["prompt"],
             "all_choices": list(ex["all_choices"]),
             "answer_idx":  int(ex["answer_idx"])}
            for ex in formatted_dataset[_paper_eval_split]
        ]
        accelerator.print(
            f"Captured {len(raw_eval_for_paper_acc)} raw eval examples "
            f"for paper-style per-candidate accuracy "
            f"(from '{_paper_eval_split}' split)"
        )

        # Compute answer_start per example for label-masking in tokenise step.
        # CRITICAL: must use the SAME max_length as tokenize_function below,
        # otherwise answer_start can index past the truncated input_ids and
        # the loss mask is silently wrong.
        def add_answer_start(example):
            prompt_tok = tokenizer(example["prompt"], truncation=True,
                                   max_length=max_seq_len)
            return {"answer_start": len(prompt_tok["input_ids"])}
        with accelerator.main_process_first():
            formatted_dataset = formatted_dataset.map(
                add_answer_start,
                remove_columns=[c for c in formatted_dataset["train"].column_names
                                if c not in ("formatted_text",)],
            )
    else:
        dataset_name = dataset_config.get('name', 'mteb/banking77')
        text_col = dataset_config.get('text_column', 'text')
        label_col = dataset_config.get('label_column', 'label')

        accelerator.print(f"Loading dataset {dataset_name}...")
        dataset = load_dataset(dataset_name, trust_remote_code=True)

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

            # Tokenize the prompt to find where the assistant's answer starts.
            # Use the unified max_seq_len so it matches tokenize_function.
            prompt_tokenized = tokenizer(prompt, truncation=True, max_length=max_seq_len)
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
        # Same max_length as the answer_start tokenisation so labels line up.
        tokenized = tokenizer(examples["formatted_text"], truncation=True,
                              max_length=max_seq_len)
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
            
    # Load model — from local checkpoint when resuming, otherwise from HF Hub.
    # Honour config.model.dtype ("float16" / "bfloat16" / "float32") if set, so
    # OPT-1.3B / OPT-13B paper-reproduction configs actually run in fp16 instead
    # of the default float32.
    _ckpt_dir = "last_checkpoint_causal"
    _model_src = _ckpt_dir if (resume_state and os.path.exists(_ckpt_dir)) else model_name
    accelerator.print(f"Loading model from: {_model_src}")
    _dtype_map = {"float16": torch.float16, "fp16": torch.float16,
                  "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                  "float32": torch.float32, "fp32": torch.float32}
    _from_pretrained_kwargs = {"trust_remote_code": True}
    # transformers 5.x + torch < 2.6 refuses .bin files (CVE-2025-32434);
    # force safetensors which doesn't go through torch.load.
    _from_pretrained_kwargs["use_safetensors"] = bool(
        model_config.get('use_safetensors', True))
    _dtype_str = model_config.get('dtype')
    if _dtype_str:
        if _dtype_str not in _dtype_map:
            raise ValueError(f"Unknown model.dtype '{_dtype_str}'. "
                             f"Known: {sorted(_dtype_map)}")
        _from_pretrained_kwargs["torch_dtype"] = _dtype_map[_dtype_str]
        accelerator.print(f"Loading model in {_dtype_str}")
    model = AutoModelForCausalLM.from_pretrained(_model_src, **_from_pretrained_kwargs)
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
            p.param_name = name

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
    
    is_zeroth_order = opt_name in [
        "LOZO", "LOZOM", "SparseMeZO", "DiZO", "MeZO", "HiZOO",
        "ConMeZO", "FZOO", "ZOMuon", "SubZero", "PseuZO",
    ]

    if is_zeroth_order:
        model.to(accelerator.device)
        if opt_name == "LOZOM":
            optimizer = LOZOM(model.parameters(), **opt_kwargs)
        elif opt_name == "SparseMeZO":
            optimizer = SparseMeZO(model.parameters(), **opt_kwargs)
        elif opt_name == "DiZO":
            optimizer = DiZO(model.parameters(), **opt_kwargs)
        elif opt_name == "MeZO":
            optimizer = MeZO(model.parameters(), **opt_kwargs)
        elif opt_name == "HiZOO":
            optimizer = HiZOO(model.parameters(), **opt_kwargs)
        elif opt_name == "ConMeZO":
            optimizer = ConMeZO(model.parameters(), **opt_kwargs)
        elif opt_name == "FZOO":
            optimizer = FZOO(model.parameters(), **opt_kwargs)
        elif opt_name == "ZOMuon":
            optimizer = ZOMuon(model.parameters(), **opt_kwargs)
        elif opt_name in ["SubZero", "PseuZO", "LOZO"]:
            # These methods use a standard first-order optimizer under the hood.
            # We instantiate standard SGD or Adam depending on configuration
            fo_name = opt_kwargs.get('first_order_name', 'SGD')
            
            # Clean up kwargs to only pass standard first-order arguments to SGD/Adam
            fo_kwargs = {
                'lr': float(opt_kwargs.get('lr', 1e-7)),
                'weight_decay': float(opt_kwargs.get('weight_decay', 0.0))
            }
            if fo_name == 'SGD' and 'momentum' in opt_kwargs:
                fo_kwargs['momentum'] = float(opt_kwargs['momentum'])
            
            opt_class = getattr(torch.optim, fo_name)
            optimizer = opt_class(model.parameters(), **fo_kwargs)
        else:
            optimizer = LOZO(model.parameters(), **opt_kwargs)

        _opt_state_path = "last_checkpoint_causal/optimizer_state.pt"
        if resume_state and os.path.exists(_opt_state_path):
            accelerator.print("Loading ZO optimizer state for resume...")
            try:
                optimizer.load_state_dict(
                    torch.load(_opt_state_path, map_location=accelerator.device, weights_only=False)
                )
            except Exception as e:
                accelerator.print(f"Warning: Could not load optimizer state ({e}).")
                accelerator.print(f"Syncing optimizer step count from global_step ({resume_state['global_step']}) instead.")
                for group in optimizer.param_groups:
                    for p in group['params']:
                        optimizer.state[p]['step'] = resume_state['global_step']

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

        # Initialize Exact Method Helpers if applicable
        subzero_helper = None
        pzo_helper = None
        lozo_helper = None
        if opt_name == "SubZero":
            class ArgsObj:
                def __init__(self, opt_cfg):
                    kwargs = opt_cfg.get('kwargs', {})
                    self.zo_eps = float(kwargs.get('zo_eps', 1e-3))
                    self.gauss_rank = int(kwargs.get('gauss_rank', 8))
                    self.update_interval = int(kwargs.get('update_interval', 2000))
                    self.mode = kwargs.get('mode', 'ft')
                    self.perturbation_mode = kwargs.get('perturbation_mode', 'two_side')
                    self.q = int(kwargs.get('q', 1))
                    self.gradient_accumulation_steps = 1
            subzero_helper = SubZeroTrainerHelper(ArgsObj(opt_config), optimizer, lr_scheduler)
        elif opt_name == "PseuZO":
            class ArgsObj:
                def __init__(self, opt_cfg):
                    kwargs = opt_cfg.get('kwargs', {})
                    self.zo_eps = float(kwargs.get('zo_eps', 1e-3))
                    self.weight_decay = float(kwargs.get('weight_decay', 0.0))
            pzo_helper = PZOTrainerHelper(ArgsObj(opt_config), optimizer, lr_scheduler)
            pzo_helper.reset_momentum_fb(float(opt_config.get('kwargs', {}).get('momentum_fb', 0.9)))
        elif opt_name == "LOZO":
            class ArgsObj:
                def __init__(self, opt_cfg):
                    kwargs = opt_cfg.get('kwargs', {})
                    self.zo_eps = float(kwargs.get('eps', 1e-3))
                    self.rank_r = int(kwargs.get('r', 4))
                    self.step_interval = int(kwargs.get('nu', 50))
                    self.weight_decay = float(kwargs.get('weight_decay', 0.0))
                    self.gradient_accumulation_steps = 1
            lozo_helper = LOZOTrainerHelper(ArgsObj(opt_config), optimizer, lr_scheduler)
    else:
        # Standard first-order optimizer
        if hasattr(torch.optim, opt_name):
            opt_class = getattr(torch.optim, opt_name)
            optimizer = opt_class(model.parameters(), **opt_kwargs)
        else:
            raise ValueError(f"Optimizer {opt_name} not found in torch.optim or custom definitions.")

        _opt_state_path = "last_checkpoint_causal/optimizer_state.pt"
        if resume_state and os.path.exists(_opt_state_path):
            accelerator.print("Loading FO optimizer state for resume...")
            try:
                optimizer.load_state_dict(
                    torch.load(_opt_state_path, map_location=accelerator.device, weights_only=False)
                )
            except Exception as e:
                accelerator.print(f"Warning: Could not load FO optimizer state ({e}).")

        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
        if eval_dataloader:
            eval_dataloader = accelerator.prepare(eval_dataloader)
        if test_dataloader:
            test_dataloader = accelerator.prepare(test_dataloader)
            
    # Optional LR scheduler (cosine with linear warmup). Configured via YAML.
    sched_cfg = config.get('scheduler', {}) or {}
    scheduler = None
    if sched_cfg.get('type') == 'cosine':
        from torch.optim.lr_scheduler import LambdaLR
        steps_per_epoch = len(train_dataloader)
        warmup_steps   = int(sched_cfg.get('warmup_steps', 0))
        t_max_epochs   = int(sched_cfg.get('t_max_epochs', epochs))
        t_max_steps    = max(1, t_max_epochs * steps_per_epoch)
        eta_min_ratio  = float(sched_cfg.get('eta_min_ratio', 0.0))

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, t_max_steps - warmup_steps))
            progress = min(1.0, progress)
            return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        accelerator.print(
            f"[Scheduler] cosine | warmup={warmup_steps} | t_max_steps={t_max_steps} "
            f"({t_max_epochs} epochs) | eta_min_ratio={eta_min_ratio}"
        )

        _sched_state_path = "last_checkpoint_causal/scheduler_state.pt"
        if resume_state and os.path.exists(_sched_state_path):
            accelerator.print("Loading scheduler state for resume...")
            scheduler.load_state_dict(torch.load(_sched_state_path, weights_only=False))

    accelerator.print(f"Starting training for {epochs} epochs using {opt_name} optimizer")
    
    global_step       = resume_state['global_step']       if resume_state else 0
    best_eval_loss    = resume_state['best_eval_loss']    if resume_state else float('inf')
    best_paper_acc    = (resume_state.get('best_paper_acc', -1.0)
                         if resume_state else -1.0)
    total_tokens_seen = resume_state['total_tokens_seen'] if resume_state else 0
    epoch             = resume_state['epoch']             if resume_state else 0
    run_start_time = time.time()

    def run_evaluation(curr_epoch, curr_global_step):
        nonlocal best_eval_loss, best_paper_acc
        if not eval_dataloader:
            return
        
        accelerator.print(f"\n--- Starting Evaluation (Epoch {curr_epoch+1}, Step {curr_global_step}) ---")
        model.eval()
        total_eval_loss = 0
        correct_tokens = 0
        total_tokens = 0
        eval_unwrapped_model = accelerator.unwrap_model(model)
        with torch.no_grad():
            for eval_step_idx, batch in enumerate(eval_dataloader):
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                outputs = eval_unwrapped_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                eval_loss = outputs.loss
                avg_loss = accelerator.reduce(eval_loss.detach(), reduction="mean")
                total_eval_loss += avg_loss.item()
                
                # Compute token-level accuracy
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
                    
        avg_eval_loss = total_eval_loss / len(eval_dataloader)
        perplexity = math.exp(avg_eval_loss) if avg_eval_loss < 20 else float('inf')
        token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        elapsed_since_start = time.time() - run_start_time

        # Paper-style per-candidate LL eval (MeZO/ConMeZO/FZOO/ZO-Muon
        # publish numbers under this metric, NOT per-token argmax). Cheap:
        # ~3s on a 1000-example eval set with OPT-1.3B on a 4090.
        paper_acc = None
        if raw_eval_for_paper_acc is not None and accelerator.is_local_main_process:
            paper_acc = _compute_paper_acc(
                eval_unwrapped_model, tokenizer,
                raw_eval_for_paper_acc, accelerator.device
            )

        log_dict = {
            "eval_loss": avg_eval_loss,
            "perplexity": perplexity,
            "eval_token_accuracy": token_acc,
            "total_elapsed_time_sec": elapsed_since_start,
            "epoch": curr_epoch+1
        }
        if paper_acc is not None:
            log_dict["paper_acc"] = paper_acc

        msg = (f"Eval: Epoch {curr_epoch+1} | Step {curr_global_step} | "
               f"Loss: {avg_eval_loss:.4f} | Perplexity: {perplexity:.2f} | "
               f"Token Accuracy: {token_acc:.4f}")
        if paper_acc is not None:
            msg += f" | Paper-Acc: {paper_acc*100:.2f}%"
        accelerator.print(msg)
        accelerator.log(log_dict, step=curr_global_step)

        if accelerator.is_local_main_process:
            accelerator.print("Saving checkpoints on main process...")
            unwrapped_model = accelerator.unwrap_model(model)
            # Save best by paper accuracy when available; fall back to lowest
            # eval loss for the legacy chat-template path.
            if paper_acc is not None:
                improved = paper_acc > best_paper_acc
                metric_name = "paper_acc"
                metric_value = paper_acc
                if improved:
                    best_paper_acc = paper_acc
            else:
                improved = avg_eval_loss < best_eval_loss
                metric_name = "loss"
                metric_value = avg_eval_loss
                if improved:
                    best_eval_loss = avg_eval_loss

            if improved:
                accelerator.print(
                    f"New best {metric_name} ({metric_value:.4f})! "
                    f"Saving best_checkpoint_causal..."
                )
                unwrapped_model.save_pretrained("best_checkpoint_causal")
                tokenizer.save_pretrained("best_checkpoint_causal")
                
                # Push BEST checkpoint to hub immediately
                if push_to_hub and repo_id:
                    try:
                        _api = HfApi()
                        _api.create_repo(repo_id=repo_id, exist_ok=True)
                        _api.upload_folder(
                            folder_path="best_checkpoint_causal",
                            repo_id=repo_id,
                            path_in_repo="best_checkpoint_causal",
                            commit_message=f"New best model: epoch {curr_epoch+1} step {curr_global_step} loss {avg_eval_loss:.4f}",
                        )
                        accelerator.print("Best checkpoint pushed to HF Hub.")
                    except Exception as _e:
                        accelerator.print(f"Warning: Best checkpoint push failed: {_e}")
                
            accelerator.print("Saving last_checkpoint_causal...")
            unwrapped_model.save_pretrained("last_checkpoint_causal")
            tokenizer.save_pretrained("last_checkpoint_causal")

            # Persist training state and optimizer state for preemption recovery
            with open("last_checkpoint_causal/training_state.json", "w") as _sf:
                json.dump({
                    "epoch":             curr_epoch + 1,
                    "global_step":       curr_global_step,
                    "best_eval_loss":    best_eval_loss,
                    "best_paper_acc":    best_paper_acc,
                    "total_tokens_seen": total_tokens_seen,
                }, _sf)
            torch.save(optimizer.state_dict(), "last_checkpoint_causal/optimizer_state.pt")
            if scheduler is not None:
                torch.save(scheduler.state_dict(), "last_checkpoint_causal/scheduler_state.pt")
            accelerator.print("Checkpoints saved successfully.")

            if push_to_hub and repo_id:
                try:
                    _api = HfApi()
                    _api.create_repo(repo_id=repo_id, exist_ok=True)
                    _api.upload_folder(
                        folder_path="last_checkpoint_causal",
                        repo_id=repo_id,
                        path_in_repo="last_checkpoint_causal",
                        commit_message=f"Resume checkpoint epoch {curr_epoch+1} step {curr_global_step}",
                    )
                    accelerator.print(f"Checkpoint pushed to HF Hub.")
                except Exception as _e:
                    accelerator.print(f"Warning: HF Hub push failed: {_e}")
        
        accelerator.wait_for_everyone()
        accelerator.print(f"--- Evaluation Complete ---\n")

    while True:
        # Outer-loop termination: max_steps takes precedence, then max_tokens,
        # then epoch count. The inner loop applies the same logic per step.
        if max_steps is not None and global_step >= max_steps:
            break
        if max_steps is None and max_tokens is None and epoch >= epochs:
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
                
                if opt_name == "SubZero":
                    subzero_helper.state.global_step = global_step
                    loss = subzero_helper.zo_subspace_step(unwrapped_model, batch)
                    subzero_helper.zo_subspace_update(unwrapped_model)
                    loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
                    total_loss += loss_val
                    progress_bar.set_description(f"Epoch {epoch+1} Loss: {loss_val:.4f}")
                    train_loss_val = loss_val
                elif opt_name == "PseuZO":
                    loss = pzo_helper.pzo_step(unwrapped_model, batch)
                    pzo_helper.pzo_update(unwrapped_model)
                    loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
                    total_loss += loss_val
                    progress_bar.set_description(f"Epoch {epoch+1} Loss: {loss_val:.4f}")
                    train_loss_val = loss_val
                elif opt_name == "LOZO":
                    loss = lozo_helper.lowrank_zo_step(unwrapped_model, batch)
                    lozo_helper.lowrank_zo_update()
                    loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
                    total_loss += loss_val
                    progress_bar.set_description(f"Epoch {epoch+1} Loss: {loss_val:.4f}")
                    train_loss_val = loss_val
                else:
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
            
            if scheduler is not None:
                scheduler.step()

            log_metrics = {
                "train_loss": train_loss_val,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "step_time_sec": step_time,
                "samples_per_second": samples_per_second,
                "total_tokens_seen": total_tokens_seen,
                "lr": optimizer.param_groups[0]['lr'],
            }
            if torch.cuda.is_available():
                log_metrics["gpu_memory_MB"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

            accelerator.log(log_metrics, step=global_step)
            global_step += 1

            # Step-based eval: trigger every eval_steps optimizer steps in
            # addition to the per-epoch trigger below.
            if eval_steps is not None and global_step > 0 and global_step % eval_steps == 0:
                run_evaluation(epoch, global_step)
                if is_zeroth_order:
                    model.eval()
                else:
                    model.train()

            if max_steps is not None and global_step >= max_steps:
                accelerator.print(f"Reached max_steps ({max_steps}). Stopping training loop.")
                break

            if max_tokens is not None and total_tokens_seen >= max_tokens:
                accelerator.print(f"Reached max_tokens ({max_tokens}). Stopping training loop.")
                break
                
        avg_train_loss = total_loss / (len(train_dataloader) if len(train_dataloader) > 0 else 1)
        accelerator.print(f"Epoch {epoch+1} finished. Avg train loss: {avg_train_loss:.4f} | Total tokens seen: {total_tokens_seen}")
        
        if (epoch + 1) % eval_epochs == 0:
            run_evaluation(epoch, global_step)

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
