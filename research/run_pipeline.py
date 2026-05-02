"""
run_pipeline.py
---------------
End-to-end orchestrator for the TokenSkip Compressed CoT Research Protocol.

Phases
──────
  0  data    — Split GSM8K (split_dataset.py)
  1  train   — Train HF models on TokenSkip CCoT (phase1/train.py)
  2  vector  — Extract truth vector (phase2/extract_vector.py)
  3  steer   — Run steering inference across 5 conditions (eval/evaluate_baselines.py)
  4  compare — Aggregate all results (eval/compare_all.py)

Usage
─────
    # Full pipeline on Phi-2
    python research/run_pipeline.py --model phi2

    # Skip Phase 1 (use existing v_truth)
    python research/run_pipeline.py --model phi2 --phases 0 2 3

    # Mini-500 dataset (fast smoke test)
    python research/run_pipeline.py --model phi2 --mini --phases 0 1 2 3
"""
from __future__ import annotations
import argparse, pathlib, subprocess, sys
from time import time

_RESEARCH_ROOT = pathlib.Path(__file__).resolve().parent
_REPO_ROOT     = _RESEARCH_ROOT.parent
sys.path.insert(0, str(_RESEARCH_ROOT))
from utils.io import load_config

_CFG = load_config()
_DS  = _CFG.get("dataset",{})
_P2  = _CFG.get("phase2",{})

def run(cmd: list, label: str, cwd: pathlib.Path | None = None) -> int:
    print(f"\n{'='*62}\n  PIPELINE  ›  {label}\n{'='*62}")
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        print(f"\n[pipeline] ✗ '{label}' failed with exit code {proc.returncode}")
    else:
        print(f"\n[pipeline] ✓ '{label}' completed successfully")
    return proc.returncode

def phase0_data(args) -> int:
    cmd = [sys.executable, str(_RESEARCH_ROOT/"data"/"split_dataset.py")]
    cmd += ["--mini"] if args.mini else ["--full"]
    if args.seed != 42: cmd += ["--seed", str(args.seed)]
    return run(cmd, "Phase 0: Split Dataset")

def phase1_train(args) -> int:
    from utils.model_registry import HF_IDS
    model_path = HF_IDS.get(args.model, args.model)
    cmd = [sys.executable, str(_RESEARCH_ROOT/"phase1"/"train.py"),
           "--model-type", args.model, "--model-path", model_path]
    return run(cmd, f"Phase 1: Train TokenSkip CCoT ({args.model})")

def phase2_vector(args) -> int:
    from utils.model_registry import HF_IDS
    model_path = HF_IDS.get(args.model, args.model)
    cmd = [sys.executable, str(_RESEARCH_ROOT/"phase2"/"extract_vector.py"),
           "--model-type", args.model, "--model-path", model_path]
    if args.steer_data: cmd += ["--steer-data", args.steer_data]
    return run(cmd, f"Phase 2: Extract Truth Vector ({args.model})")

def phase3_steer(args) -> int:
    from utils.model_registry import HF_IDS
    model_path = HF_IDS.get(args.model, args.model)
    cmd = [sys.executable, str(_RESEARCH_ROOT/"eval"/"evaluate_baselines.py"),
           "--models", args.model, "--model-paths", f"{args.model}={model_path}",
           "--alphas"] + [str(a) for a in args.alphas]
    if args.eval_data:  cmd += ["--eval-data",  args.eval_data]
    if args.vector_dir: cmd += ["--vector-root", args.vector_dir]
    return run(cmd, f"Phase 3: Evaluate 5 Conditions ({args.model})")

def phase4_compare(args) -> int:
    cmd = [sys.executable, str(_RESEARCH_ROOT/"eval"/"compare_all.py")]
    if args.csv:   cmd += ["--csv",   args.csv]
    if args.latex: cmd.append("--latex")
    return run(cmd, "Phase 4: Compare All Results")

PHASE_RUNNERS = {0:phase0_data, 1:phase1_train, 2:phase2_vector, 3:phase3_steer, 4:phase4_compare}
PHASE_NAMES   = {0:"data", 1:"train", 2:"vector", 3:"steer", 4:"compare"}

DEFAULT_ALPHAS = _P2.get("alpha_sweep", [0.0,0.1,0.5,1.0,2.0,5.0,10.0,20.0,50.0])

def main():
    p = argparse.ArgumentParser(description="End-to-end pipeline orchestrator for TokenSkip CCoT")
    p.add_argument("--phases",    nargs="+", type=int, default=[0,1,2,3,4], choices=[0,1,2,3,4])
    p.add_argument("--model",     default="phi2",
                   choices=["phi2","llama32_3b","qwen25_3b","qwen25_1_5b","qwen25_0_5b"])
    p.add_argument("--mini",      action="store_true", help="Use mini-500 dataset")
    p.add_argument("--alphas",    nargs="+", type=float, default=DEFAULT_ALPHAS)
    p.add_argument("--seed",      type=int,  default=42)
    p.add_argument("--steer-data",default=None)
    p.add_argument("--eval-data", default=None)
    p.add_argument("--vector-dir",default=None)
    p.add_argument("--csv",       default=None)
    p.add_argument("--latex",     action="store_true")
    args = p.parse_args()

    phases_to_run = sorted(set(args.phases))
    print("\n" + "="*62)
    print("  RESEARCH PIPELINE: TokenSkip Compressed CoT")
    print("="*62)
    print(f"  Phases  : {[PHASE_NAMES[ph] for ph in phases_to_run]}")
    print(f"  Model   : {args.model}")
    print(f"  Dataset : {'mini-500' if args.mini else 'full GSM8K'}")
    print(f"  Alphas  : {args.alphas}")
    print("="*62)

    failed = []
    t_start = time()
    for phase in phases_to_run:
        runner = PHASE_RUNNERS[phase]
        t0     = time()
        rc     = runner(args)
        elapsed= time()-t0
        status = "✓" if rc==0 else "✗"
        print(f"\n  {status} Phase {phase} ({PHASE_NAMES[phase]}) — {elapsed/60:.1f} min")
        if rc != 0:
            failed.append(phase)
            print(f"\n[pipeline] Stopping: phase {phase} failed.")
            break

    total = time()-t_start
    print(f"\n{'='*62}")
    print(f"  Pipeline complete in {total/60:.1f} min")
    if failed:
        print(f"  ✗ Failed phase(s): {failed}")
        sys.exit(1)
    else:
        print(f"  ✓ All phases succeeded")
        print(f"\n  Results → python research/eval/compare_all.py")
    print("="*62)

if __name__ == "__main__":
    main()
