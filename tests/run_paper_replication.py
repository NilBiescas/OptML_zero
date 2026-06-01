"""Paper-length replication runner for OPT-1.3B / SST-2.

For each of MeZO (baseline), ConMeZO, FZOO, ZOMuon: run train.py at the
paper-default step count, preserve the best checkpoint under
`runs/<run_name>/best_checkpoint/`, and push it to Hugging Face Hub as
`<hf_user>/<optimizer>-opt1.3b-sst2-replication`.

Total budget ~5 hours / ~$6 of GPU time on RTX 4090. Wandb runs land in
project `optml-zero-paper-replication`.

This is the "does our implementation actually replicate the paper?" check.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WANDB_PROJECT = "optml-zero-paper-replication"
# Vast.ai RTX 4090 48GB hourly price (observed in this session).
COST_PER_HOUR_USD = 1.4028

# (run_name, base_config, max_steps, eval_every, optional_kwargs_override)
# Paper-default LRs per official scripts/paper text. Only the THREE new
# methods (ConMeZO, FZOO, ZOMuon) — MeZO baseline is skipped because the
# original-MeZO paper numbers are already published and we are validating
# the new optimizer implementations against their own headline results.
RUNS = [
    ("ConMeZO-paper", "configs/conmezo/opt1.3b_sst2.yaml",         20000, 1000, {}),
    # FZOO budget = 40k forward passes / (Nq+1) = 4000 optimizer steps per
    # paper Figure 3/9 axes. Earlier 500-step override was for the Nq=2 draft.
    ("FZOO-paper",    "configs/fzoo/opt1.3b_sst2.yaml",             4000,  500, {}),
    ("ZOMuon-paper",  "configs/zomuon/opt1.3b_sst2.yaml",           8000,  500, {}),
]

# HF repo naming.
HF_USER         = os.environ.get("HF_USER", "chenghengli")
HF_REPO_PREFIX  = "optml-zero"


def render_config(base_path: Path, max_steps: int, eval_steps: int,
                  kw_override: dict, out_path: Path):
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["max_steps"]  = max_steps
    cfg["training"]["eval_steps"] = eval_steps
    cfg["training"]["epochs"]     = 1
    for k, v in kw_override.items():
        cfg["optimizer"]["kwargs"][k] = v
    # Disable in-train HF push; we do it manually post-run so we control naming.
    cfg.setdefault("hub", {})
    cfg["hub"]["push_to_hub"] = False
    cfg["hub"]["repo_id"]     = None
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def preserve_checkpoint(run_name: str) -> Path:
    """Copy best_checkpoint_causal to runs/<run_name>/best_checkpoint."""
    src = ROOT / "best_checkpoint_causal"
    if not src.exists():
        print(f"  WARN: no best_checkpoint_causal/ found for {run_name}", flush=True)
        return None
    dst_dir = ROOT / "runs" / run_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "best_checkpoint"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"  preserved checkpoint -> {dst}", flush=True)
    return dst


def push_to_hf(checkpoint_dir: Path, run_name: str):
    """Push the preserved checkpoint to HF as <HF_USER>/optml-zero-<run_name>."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("  huggingface_hub not installed, skipping push", flush=True)
        return None

    repo_id = f"{HF_USER}/{HF_REPO_PREFIX}-{run_name.lower()}"
    print(f"  pushing {checkpoint_dir} -> hf://{repo_id}", flush=True)
    try:
        create_repo(repo_id, exist_ok=True, private=False)
        api = HfApi()
        api.upload_folder(
            folder_path=str(checkpoint_dir),
            repo_id=repo_id,
            commit_message=f"Replication checkpoint: {run_name}",
        )
        print(f"  pushed -> https://huggingface.co/{repo_id}", flush=True)
        return repo_id
    except Exception as e:
        print(f"  FAIL push: {type(e).__name__}: {e}", flush=True)
        return None


def run_one(run_name: str, base: str, max_steps: int, eval_steps: int,
            kw_override: dict, tmpdir: Path):
    print("\n" + "=" * 70, flush=True)
    print(f"== {run_name}   ({max_steps} steps, eval every {eval_steps})", flush=True)
    print("=" * 70, flush=True)
    cfg_path = tmpdir / f"{run_name}.yaml"
    render_config(ROOT / base, max_steps, eval_steps, kw_override, cfg_path)
    env = os.environ.copy()
    env["RUN_NAME"]      = run_name
    env["WANDB_PROJECT"] = WANDB_PROJECT
    env.setdefault("HF_HOME", "/workspace/.hf_home")

    t0 = time.time()
    proc = subprocess.run(
        ["accelerate", "launch", "--num_processes", "1",
         str(ROOT / "train.py"), "--config", str(cfg_path)],
        cwd=str(ROOT), env=env,
    )
    dt = time.time() - t0
    cost_usd = dt / 3600.0 * COST_PER_HOUR_USD
    print(f"\n  wall_time = {dt:.0f}s ({dt/60:.1f} min)  "
          f"cost_on_4090 = ${cost_usd:.2f}  "
          f"sec_per_step = {dt/max_steps:.3f}  "
          f"exit={proc.returncode}", flush=True)
    if proc.returncode != 0:
        print(f"  RUN FAILED, skipping checkpoint preservation / HF push", flush=True)
        return {"run_name": run_name, "exit": proc.returncode, "wall_sec": dt}

    ckpt = preserve_checkpoint(run_name)
    hf_repo = push_to_hf(ckpt, run_name) if ckpt else None

    return {"run_name": run_name, "exit": 0, "wall_sec": dt,
            "max_steps": max_steps,
            "sec_per_step": dt / max(1, max_steps),
            "cost_usd": cost_usd,
            "checkpoint": str(ckpt) if ckpt else None, "hf_repo": hf_repo}


def main():
    tmpdir = Path(tempfile.mkdtemp(prefix="optml_replication_"))
    print(f"Working dir: {tmpdir}", flush=True)
    print(f"Wandb project: {WANDB_PROJECT}", flush=True)
    print(f"HF user: {HF_USER}", flush=True)

    results = []
    for run_name, base, steps, ev, ov in RUNS:
        results.append(run_one(run_name, base, steps, ev, ov, tmpdir))

    print("\n" + "=" * 100, flush=True)
    print(f"REPLICATION SUMMARY (RTX 4090 48GB @ ${COST_PER_HOUR_USD:.2f}/hr)", flush=True)
    print("=" * 100, flush=True)
    print(f"  {'status':10s} {'run':20s} {'wall':>10s} {'s/step':>8s} "
          f"{'cost':>8s}  hf_repo", flush=True)
    print("-" * 100, flush=True)
    total_wall = 0
    total_cost = 0
    for r in results:
        status  = "PASS" if r.get("exit") == 0 else f"FAIL(rc={r.get('exit')})"
        wall_m  = f"{r.get('wall_sec', 0)/60:.1f}m"
        sps     = f"{r.get('sec_per_step', 0):.2f}s"
        cost    = f"${r.get('cost_usd', 0):.2f}"
        hf      = r.get("hf_repo") or "-"
        print(f"  {status:10s} {r['run_name']:20s} {wall_m:>10s} {sps:>8s} "
              f"{cost:>8s}  {hf}", flush=True)
        total_wall += r.get('wall_sec', 0)
        total_cost += r.get('cost_usd', 0)
    print("-" * 100, flush=True)
    print(f"  TOTAL: {total_wall/60:.1f} min  ({total_wall/3600:.2f} h)  "
          f"${total_cost:.2f}", flush=True)

    shutil.rmtree(tmpdir, ignore_errors=True)
    return 0 if all(r.get("exit") == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
