"""Paper-faithful smoke sweep on OPT-1.3B / SST-2.

For each of ConMeZO, FZOO, ZOMuon we run 3 short LR variants (paper
default + one decade lower + one decade higher) through the real
train.py pipeline on facebook/opt-1.3b. Every run logs to wandb
project `optml-zero-paper-smoke`.

100 steps each. At ~1 s/step on RTX 4090, the whole sweep is ~15 min.
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
MAX_STEPS = 100
EVAL_STEPS = 50
WANDB_PROJECT = "optml-zero-paper-smoke"

# 3 LRs per optimizer: paper-default in the middle, +/- one decade.
SWEEPS = {
    "ConMeZO": {
        "base": "configs/conmezo/opt1.3b_sst2.yaml",
        "lrs":  [1.0e-8, 1.0e-7, 1.0e-6],
    },
    "FZOO": {
        "base": "configs/fzoo/opt1.3b_sst2.yaml",
        "lrs":  [5.0e-7, 5.0e-6, 5.0e-5],
    },
    "ZOMuon": {
        "base": "configs/zomuon/opt1.3b_sst2.yaml",
        "lrs":  [1.0e-3, 1.0e-2, 1.0e-1],
    },
}


def render_config(base_path: Path, lr: float, run_name: str, out_path: Path):
    """Load base config, override lr / max_steps / eval_steps / wandb name."""
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    cfg["optimizer"]["kwargs"]["lr"] = float(lr)
    cfg["training"]["max_steps"] = MAX_STEPS
    cfg["training"]["eval_steps"] = EVAL_STEPS
    cfg["training"]["epochs"] = 1
    cfg.setdefault("hub", {})
    cfg["hub"]["push_to_hub"] = False
    cfg["hub"]["repo_id"] = None
    # Tag wandb run via env var (train.py reads RUN_NAME).
    cfg["_run_name"] = run_name
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_one(opt: str, lr: float, base: str, tmpdir: Path):
    run_name = f"{opt}-lr{lr:.0e}-100steps"
    cfg_path = tmpdir / f"{run_name}.yaml"
    render_config(ROOT / base, lr, run_name, cfg_path)
    print(f"\n========== {run_name} ==========", flush=True)
    print(f"  config: {cfg_path}", flush=True)
    env = os.environ.copy()
    env["RUN_NAME"] = run_name
    env["WANDB_PROJECT"] = WANDB_PROJECT
    env.setdefault("HF_HOME", str(ROOT / "hf_cache"))
    t0 = time.time()
    proc = subprocess.run(
        ["accelerate", "launch", "--num_processes", "1",
         str(ROOT / "train.py"), "--config", str(cfg_path)],
        cwd=str(ROOT), env=env,
    )
    dt = time.time() - t0
    print(f"\n  -> exit={proc.returncode}, wall={dt:.1f}s", flush=True)
    return proc.returncode, dt


def main():
    tmpdir = Path(tempfile.mkdtemp(prefix="optml_sweep_"))
    print(f"Working dir: {tmpdir}", flush=True)
    print(f"Wandb project: {WANDB_PROJECT}", flush=True)
    print(f"Max steps per run: {MAX_STEPS}", flush=True)

    results = []
    for opt, spec in SWEEPS.items():
        for lr in spec["lrs"]:
            rc, dt = run_one(opt, lr, spec["base"], tmpdir)
            results.append((opt, lr, rc, dt))

    print("\n" + "=" * 60, flush=True)
    print("SWEEP SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for opt, lr, rc, dt in results:
        status = "PASS" if rc == 0 else f"FAIL(rc={rc})"
        print(f"  {status:10s} {opt:8s} lr={lr:.0e}  wall={dt:.1f}s", flush=True)

    shutil.rmtree(tmpdir, ignore_errors=True)
    return 0 if all(rc == 0 for _, _, rc, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
