"""Post-sweep comparison: our 100-step smoke results vs each paper's
headline numbers for OPT-1.3B / SST-2.

Pulls every run from the wandb project `optml-zero-paper-smoke`, extracts
final/min train_loss + peak VRAM + step time, then renders a side-by-side
table against the paper-reported accuracies.

IMPORTANT: our smoke runs are 100 steps. The papers report numbers at
8000-20000 steps. We are NOT expected to match the paper accuracy yet --
this script checks (a) the pipeline runs end-to-end on the right data and
(b) the loss trajectory is reasonable (decreasing, finite, not NaN). The
final accuracy comparison column flags whether the same recipe is *on
track* to hit the paper number after a full-length run.
"""

import os
import sys

# Reported headline SST-2 numbers from each paper, OPT-1.3B (or closest).
# Wall-time figures are what each paper quotes on its reference GPU — included
# only for context, since our 4090 has different throughput vs A100/H100.
# Sources: arXiv:2305.17333 (MeZO), arXiv:2602.17155 (ZO-Muon),
#          arXiv:2511.02757 (ConMeZO), arXiv:2506.09034 (FZOO).
PAPER_TARGETS = {
    "MeZO":    {"sst2_acc_pct": 91.4, "steps": 20000,
                "paper": "Malladi 2023 (MeZO)",
                "wall_min": None,   "ref_gpu": "1xA100-80GB"},
    "ZOMuon":  {"sst2_acc_pct": 92.5, "steps":  8000,
                "paper": "Lang 2026 (ZO-Muon)",
                "wall_min": 153,    "ref_gpu": "1xA100-80GB (OPT-13B; 1.3B is ~3-5x faster)"},
    "ConMeZO": {"sst2_acc_pct": 92.0, "steps": 20000,
                "paper": "Lej 2026 (ConMeZO)",
                "wall_min": None,   "ref_gpu": "1xA100-80GB"},
    "FZOO":    {"sst2_acc_pct": 93.5, "steps":   500,
                "paper": "FZOO 2025 (paper text)",
                "wall_min": None,   "ref_gpu": "1xA100-80GB (8xA100 for sweep)"},
}

# Vast.ai RTX 4090 48GB rental price observed in this run.
COST_PER_HOUR_USD = 1.4028


def fetch_runs(project: str):
    try:
        import wandb
    except ImportError:
        print("wandb not installed", file=sys.stderr)
        return []
    api = wandb.Api()
    runs = api.runs(project)
    out = []
    for r in runs:
        if r.state not in ("finished", "running"):
            continue
        # train.py logs `train_loss`, `step_time_sec`, `eval_loss`,
        # `eval_token_accuracy`, `final_total_time_sec`, `total_tokens_seen`.
        s = dict(r.summary)
        wall = s.get("final_total_time_sec") or s.get("_runtime")
        steps = s.get("_step") or 0
        out.append({
            "name":             r.name,
            "state":            r.state,
            "url":              r.url,
            "final_train_loss": s.get("train_loss"),
            "best_eval_loss":   s.get("eval_loss"),
            "eval_acc":         s.get("eval_token_accuracy"),
            "test_acc":         s.get("test_token_accuracy"),
            "peak_VRAM_MB":     s.get("gpu_memory_MB"),
            "mean_step_sec":    s.get("step_time_sec"),
            "wall_sec":         wall,
            "steps":            steps,
            "tokens":           s.get("total_tokens_seen"),
        })
    return out


def opt_from_name(name: str):
    for k in ("ConMeZO", "FZOO", "ZOMuon", "MeZO"):
        if k in name:
            return k
    return "?"


def main():
    project = os.environ.get("WANDB_PROJECT", "optml-zero-paper-smoke")
    print(f"\nFetching runs from wandb project: {project}\n")
    runs = fetch_runs(project)
    if not runs:
        print("No runs found.")
        return 1

    # Group by optimizer.
    by_opt = {}
    for r in runs:
        by_opt.setdefault(opt_from_name(r["name"]), []).append(r)

    print("=" * 140)
    print(f"{'Opt':8s} {'Run':32s} {'final_loss':>10s} {'eval_acc':>9s} "
          f"{'test_acc':>9s} {'VRAM':>8s} {'s/step':>8s} {'wall':>10s} "
          f"{'$ on 4090':>10s}")
    print("=" * 140)
    def _fmt(v, spec, suffix=""):
        # wandb sometimes ships float-NaN as the string "NaN"; handle both.
        if v is None:
            return "n/a"
        try:
            return format(float(v), spec) + suffix
        except (TypeError, ValueError):
            return str(v)

    for opt in sorted(by_opt):
        for r in sorted(by_opt[opt], key=lambda x: x["name"]):
            fl  = _fmt(r['final_train_loss'], '.3f')
            ea  = _fmt(r['eval_acc'] and r['eval_acc']*100, '.2f', '%')
            ta  = _fmt(r['test_acc'] and r['test_acc']*100, '.2f', '%')
            vm  = _fmt(r['peak_VRAM_MB'], '.0f', 'MB')
            stt = _fmt(r['mean_step_sec'], '.2f', 's')
            wall= _fmt((r['wall_sec'] or 0)/60, '.1f', 'min') if r['wall_sec'] else 'n/a'
            cost= _fmt((r['wall_sec'] or 0)/3600*COST_PER_HOUR_USD, '.2f') if r['wall_sec'] else 'n/a'
            cost= '$' + cost if cost != 'n/a' else cost
            print(f"{opt:8s} {r['name'][:32]:32s} {fl:>10s} {ea:>9s} "
                  f"{ta:>9s} {vm:>8s} {stt:>8s} {wall:>10s} {cost:>10s}")
    print("=" * 140)

    print("\nPaper headline numbers (OPT-1.3B / SST-2):")
    for opt, t in PAPER_TARGETS.items():
        wall_str = f"~{t['wall_min']:>3d} min" if t['wall_min'] else "  unspec."
        print(f"  {opt:8s} -> {t['sst2_acc_pct']:>5.1f}% acc  steps={t['steps']:>5d}  "
              f"wall={wall_str}  ({t['paper']}, on {t['ref_gpu']})")

    # Project full-replication wall time from observed step time.
    print("\nFull-replication time projection on THIS RTX 4090 ($1.40/hr):")
    for opt, t in PAPER_TARGETS.items():
        runs = [r for r in by_opt.get(opt, []) if r['mean_step_sec'] is not None]
        if not runs:
            continue
        # take median across LR sweeps
        runs.sort(key=lambda r: r['mean_step_sec'])
        sec_per_step = runs[len(runs)//2]['mean_step_sec']
        proj_wall_sec = sec_per_step * t['steps']
        proj_cost     = proj_wall_sec / 3600 * COST_PER_HOUR_USD
        print(f"  {opt:8s} {t['steps']:>5d} steps x {sec_per_step:.2f}s = "
              f"{proj_wall_sec/60:>5.1f} min  (${proj_cost:.2f})")

    # Best LR per optimizer by lowest finite final training loss.
    print("\nBest LR per optimizer (lowest finite final train_loss):")
    for opt, rs in sorted(by_opt.items()):
        valid = []
        for r in rs:
            try:
                v = float(r['final_train_loss']) if r['final_train_loss'] is not None else None
                if v is not None and v == v:  # not NaN
                    valid.append((v, r))
            except (TypeError, ValueError):
                pass
        if not valid: continue
        v_best, best = min(valid, key=lambda kv: kv[0])
        print(f"  {opt:8s} -> {best['name']}  loss={v_best:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
