# LOZO & LOZO-M: Low-Rank Zeroth-Order Fine-Tuning Pipeline

This repository implements a complete, memory-efficient distributed pipeline for fine-tuning Large Language Models (LLMs) using **Low-Rank Zeroth-Order (LOZO)** and **LOZO-M (LOZO with Momentum)** optimizers on a multi-GPU HPC cluster (managed via RunAI / Kubernetes).

The pipeline fine-tunes **Qwen3.5-0.8B** on the **PolyAI/banking77** dataset, bypassing standard backpropagation and gradient calculation to achieve near-inference memory usage.

---

## 🛠️ Repository Architecture

- `train.py`: Main configuration-driven training pipeline utilizing Hugging Face `accelerate` for distributed 4-GPU execution.
- `config.yaml`: Centralized configuration file for hyperparameters, dataset settings, and model settings.
- `optimizers/lozo.py`: Customized PyTorch implementation of the `LOZO` and `LOZO-M` optimizers (ICLR 2025).
- `.env`: Secure credentials file storing API keys for monitoring and remote checkpoints.
- `run_lozo.sh`: Bootstrapping entrypoint executed inside the cluster pod (handles safe directories, dependency installation, and `git pull`).
- `submit_lozo.sh`: Deployment script used to submit preemptible 4-GPU jobs to the RunAI cluster.

---

## 🚀 Step-by-Step Execution Guide

Follow these steps to configure, deploy, and monitor your training run.

### Step 1: Secure Credentials (`.env`)
Create a file named `.env` in the root of your project (this file is ignored by Git in `.gitignore` to prevent leaks). Add your API keys:

```env
# Get from https://wandb.ai/authorize
WANDB_API_KEY=your_wandb_api_key_here

# Get from https://huggingface.co/settings/tokens (Must have WRITE permissions)
HF_TOKEN=your_hugging_face_write_token_here
```

---

### Step 2: Configure Your Run (`config.yaml`)
Open `config.yaml` to adjust the settings. The key sections are:

1. **Training Boundaries:**
   - **Epoch-based:** Set the number of epochs to run:
     ```yaml
     training:
       epochs: 3
       batch_size: 16
       seed: 42
     ```
   - **Token-based:** If you want to train for an exact number of tokens instead, uncomment and set `max_tokens`:
     ```yaml
     training:
       epochs: 3
       batch_size: 16
       seed: 42
       max_tokens: 10000000 # Stops exactly after seeing 10M tokens (ignores epochs)
     ```

2. **Select Optimizer:**
   - For **LOZO**: Set `name` to `"LOZO"` and set your learning rate `lr`, perturbation scale `eps`, low-rank `r`, and lazy resampling interval `nu`:
     ```yaml
     optimizer:
       name: "LOZO"
       kwargs:
         lr: 1.0e-5
         eps: 1.0e-3
         r: 4
         nu: 50
     ```
   - For **LOZO-M** (with Momentum): Change the name to `"LOZOM"` and uncomment the momentum coefficient `beta`:
     ```yaml
     optimizer:
       name: "LOZOM"
       kwargs:
         lr: 1.0e-5
         eps: 1.0e-3
         r: 4
         nu: 50
         beta: 0.9
     ```
   - For **Standard Backprop**: You can also use standard PyTorch optimizers like `"AdamW"` or `"SGD"`. The pipeline automatically detects this and switches back to first-order backpropagation.

3. **Hugging Face Remote Repo:**
   Change the `repo_id` to your actual Hugging Face username and desired repository name:
   ```yaml
   hub:
     push_to_hub: true
     repo_id: "your-hf-username/lozo-qwen-banking77"
   ```

---

### Step 3: Push Your Code to GitHub
Because the cluster pod runs a `git pull` as soon as it boots to fetch your code, **you must push all changes to GitHub** before submitting the job:

```bash
git add train.py config.yaml optimizers/lozo.py run_lozo.sh submit_lozo.sh .gitignore
git commit -m "Configure pipeline metrics and credentials"
git push
```

---

### Step 4: Submit the RunAI Job
Log into your cluster's login node. Make sure your `.env` file is present in the directory, and submit the 4-GPU training job:

```bash
chmod +x submit_lozo.sh run_lozo.sh
./submit_lozo.sh
```

---

## 📊 Live Monitoring & Checkpoints

Since you cannot access the pods directly, the entire workflow is monitored and saved externally:

### 1. View Logs in Terminal
You can stream stdout/stderr logs directly from the login node using the CLI details printed when submitting:
```bash
runai logs -f <job-name> -p <project-name>
```

### 2. Live WandB Charts
All logs are streamed in real-time to Weights & Biases (under the project `lozo-training`). You can log in to your browser dashboard to visualize:
- **`train_loss`**: Smoothed loss descent per step.
- **`eval_accuracy` & `eval_loss`**: Validation metric updates evaluated per epoch.
- **`samples_per_second` & `step_time_sec`**: Throughput and latency benchmarking metrics.
- **`gpu_memory_MB`**: Peak active active VRAM allocations on the GPU.
- **`total_tokens_seen`**: Exact cumulative number of tokens processed.

### 3. Retrieve Checkpoints
As training proceeds:
- A local folder `best_checkpoint` will save the weights/tokenizers whenever a new `eval_accuracy` high score is achieved.
- A local folder `last_checkpoint` is rewritten at the end of every epoch.
- **At final termination**, the main process will create your specified Hugging Face repository and push both directories securely to your account. You can download your trained weights directly from Hugging Face!