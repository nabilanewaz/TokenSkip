"""
evaluate_baselines.py
---------------------
Orchestrator for the full evaluation grid:

  Conditions  × Models
  ─────────────────────────────────────────────────────────
  no_cot        microsoft/phi-2
  text_cot      meta-llama/Llama-3.2-3B
  ccot          Qwen/Qwen2.5-3B
  random_noise  (+ CODI-GPT2 via steer_inference.py)
  steered

For each model:
  • no_cot      → evaluation.py  with --no-cot flag
  • text_cot    → evaluation.py  normally (full chain-of-thought)
  • ccot        → hidden_steer.py  --condition ccot      (alpha=0, v_truth)
  • random_noise→ hidden_steer.py  --condition random_noise
  • steered     → hidden_steer.py  --condition steered   (alpha sweep)

Dataset: datasets/gsm8k_split/test.jsonl  (1,319 held-out examples)

Results layout
──────────────
  outputs/eval_grid/<model_tag>/<condition>/metrics.json

Usage
-----
  # Run all models × all conditions
  python evaluate_baselines.py --model-paths phi-2=/path/to/phi-2 ...

  # Run specific model and condition
  python evaluate_baselines.py --models phi2 --conditions no_cot text_cot

  # Print results table from existing outputs (no new runs)
  python evaluate_baselines.py --results-only

Model path configuration
────────────────────────
  Pass paths via --model-paths  MODEL_TAG=HF_ID_OR_LOCAL_PATH  pairs
  OR set environment variables:
      PHI2_PATH   LLAMA32_PATH   QWEN25_PATH
  OR edit MODEL_PATHS at the top of this file.
"""

import os, sys, json, subprocess, pathlib, argparse, re
from time import time

# ── Default model paths (edit or override via --model-paths / env vars) ───────
MODEL_PATHS = {
    "phi2":       os.environ.get("PHI2_PATH",    "microsoft/phi-2"),
    "llama32_3b": os.environ.get("LLAMA32_PATH", "meta-llama/Llama-3.2-3B"),
    "qwen25_3b":  os.environ.get("QWEN25_PATH",  "Qwen/Qwen2.5-3B"),
}

# ── Evaluation config ─────────────────────────────────────────────────────────
DEFAULT_EVAL_DATA    = "datasets/gsm8k_split/test.jsonl"
DEFAULT_STEER_DATA   = "datasets/gsm8k_split/steer_train.jsonl"
DEFAULT_VECTOR_ROOT  = "outputs/truth_vectors"
DEFAULT_OUT_ROOT     = "outputs/eval_grid"
DEFAULT_CONDITIONS   = ["no_cot", "text_cot", "ccot", "random_noise", "steered"]
ALL_MODELS           = ["phi2", "llama32_3b", "qwen25_3b"]

# Alpha sweep for steered condition (from protocol)
STEER_ALPHAS         = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

# Hidden layer to intervene on (as fraction of total layers — 0.75 = last quarter)
INTERVENTION_LAYER_FRAC = 0.75


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_cmd(cmd, cwd=None, env=None, label=""):
    """Stream a subprocess to stdout, return (exit_code, full_log)."""
    print(f"\n{'─'*60}\n  {label}\n{'─'*60}")
    lines = []
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True,
        encoding="utf-8", errors="replace",
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines)


def save_result(out_dir: pathlib.Path, data: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(data, indent=2))


# ── Condition runners ─────────────────────────────────────────────────────────

def run_no_cot(model_tag, model_path, eval_data, out_dir):
    """Run direct-answer (no chain-of-thought) evaluation."""
    cmd = [
        sys.executable, "evaluation.py",
        "--output-dir",    str(out_dir),
        "--model-path",    model_path,
        "--tokenizer-path", model_path,
        "--model-type",    model_tag,
        "--data-type",     "test",
        "--benchmark",     "gsm8k",
        "--max_new_tokens", "64",       # short — just the answer
        "--eval_batch_size", "8",
        "--temperature",   "0.0",
        "--seed",          "42",
        "--no-cot",                      # signal to use build_no_cot_prompt
        "--eval-data",     str(eval_data),
    ]
    rc, log = run_cmd(cmd, label=f"[no_cot] {model_tag}")
    return {"condition": "no_cot", "model": model_tag, "exit_code": rc,
            "out_dir": str(out_dir)}


def run_text_cot(model_tag, model_path, eval_data, out_dir):
    """Run standard text-based chain-of-thought evaluation."""
    cmd = [
        sys.executable, "evaluation.py",
        "--output-dir",    str(out_dir),
        "--model-path",    model_path,
        "--tokenizer-path", model_path,
        "--model-type",    model_tag,
        "--data-type",     "test",
        "--benchmark",     "gsm8k",
        "--max_new_tokens", "512",
        "--eval_batch_size", "4",
        "--temperature",   "0.0",
        "--seed",          "42",
        "--eval-data",     str(eval_data),
    ]
    rc, log = run_cmd(cmd, label=f"[text_cot] {model_tag}")
    return {"condition": "text_cot", "model": model_tag, "exit_code": rc,
            "out_dir": str(out_dir)}


def run_hidden_condition(model_tag, model_path, eval_data, steer_data,
                         vector_dir, out_dir, condition, alphas=None):
    """
    Run CCoT / random_noise / steered via hidden_steer.py.
    Uses hook-based latent intervention on the model's hidden states.
    """
    cmd = [
        sys.executable, "hidden_steer.py",
        "--model-path",   model_path,
        "--model-type",   model_tag,
        "--eval-data",    str(eval_data),
        "--steer-data",   str(steer_data),
        "--vector-dir",   str(vector_dir),
        "--out-dir",      str(out_dir),
        "--condition",    condition,
        "--layer-frac",   str(INTERVENTION_LAYER_FRAC),
        "--seed",         "42",
    ]
    if condition == "steered" and alphas:
        cmd += ["--alphas"] + [str(a) for a in alphas]
    rc, log = run_cmd(cmd, label=f"[{condition}] {model_tag}")
    return {"condition": condition, "model": model_tag, "exit_code": rc,
            "out_dir": str(out_dir)}


# ── Results loader ────────────────────────────────────────────────────────────

def collect_results(out_root: pathlib.Path):
    """Scan outputs/eval_grid/ and return a list of result dicts."""
    rows = []
    for metrics_path in sorted(out_root.glob("*/*/metrics.json")):
        try:
            data = json.loads(metrics_path.read_text())
            # condition dir structure: <model_tag>/<condition>/metrics.json
            parts = metrics_path.parts
            cond_idx = len(out_root.parts)
            data.setdefault("model",     parts[cond_idx]   if len(parts) > cond_idx   else "?")
            data.setdefault("condition", parts[cond_idx+1] if len(parts) > cond_idx+1 else "?")
            data["_path"] = str(metrics_path)
            rows.append(data)
        except Exception as e:
            print(f"[collect] Warning: could not load {metrics_path}: {e}")
    return rows


def print_results_table(rows):
    """Print a comparison table: model × condition → accuracy / flip_rate / cos_sim."""
    if not rows:
        print("[results] No results found.")
        return

    all_models     = sorted({r.get("model","?")     for r in rows})
    all_conditions = ["no_cot", "text_cot", "ccot", "random_noise", "steered"]

    # Metric extraction helper
    def get_metric(row, key, default="—"):
        v = row.get(key)
        if v is None:
            return default
        if isinstance(v, float):
            return f"{v:.2%}" if key in ("accuracy", "flip_rate") else f"{v:.4f}"
        return str(v)

    # ── Accuracy table ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  EVALUATION GRID — Accuracy (%)")
    print("=" * 80)
    cw = 14
    header = f"  {'Model':<16}" + "".join(f"{c[:cw]:>{cw}}" for c in all_conditions)
    print(header)
    print("  " + "─" * (16 + cw * len(all_conditions)))
    for model in all_models:
        row_str = f"  {model:<16}"
        for cond in all_conditions:
            match = next((r for r in rows if r.get("model")==model and r.get("condition")==cond), None)
            cell = get_metric(match, "accuracy") if match else "—"
            row_str += f"{cell:>{cw}}"
        print(row_str)
    print()

    # ── Flip rate table (where available) ──────────────────────────────────
    has_flip = any(r.get("flip_rate") is not None for r in rows)
    if has_flip:
        print("=" * 80)
        print("  EVALUATION GRID — Flip Rate (wrong→right, % of baseline errors)")
        print("=" * 80)
        print(header)
        print("  " + "─" * (16 + cw * len(all_conditions)))
        for model in all_models:
            row_str = f"  {model:<16}"
            for cond in all_conditions:
                match = next((r for r in rows if r.get("model")==model and r.get("condition")==cond), None)
                cell = get_metric(match, "flip_rate") if match else "—"
                row_str += f"{cell:>{cw}}"
            print(row_str)
        print()

    # ── Cosine sim table (where available) ─────────────────────────────────
    has_cos = any(r.get("mean_cos_sim") is not None for r in rows)
    if has_cos:
        print("=" * 80)
        print("  EVALUATION GRID — Mean Cosine Similarity (latent vs v_truth)")
        print("=" * 80)
        print(header)
        print("  " + "─" * (16 + cw * len(all_conditions)))
        for model in all_models:
            row_str = f"  {model:<16}"
            for cond in all_conditions:
                match = next((r for r in rows if r.get("model")==model and r.get("condition")==cond), None)
                cell = get_metric(match, "mean_cos_sim") if match else "—"
                row_str += f"{cell:>{cw}}"
            print(row_str)
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full evaluation grid: model × condition → accuracy/flip/cos_sim"
    )
    parser.add_argument("--eval-data",   default=DEFAULT_EVAL_DATA)
    parser.add_argument("--steer-data",  default=DEFAULT_STEER_DATA,
                        help="Data for truth vector extraction (Phase 2, steer_train split)")
    parser.add_argument("--out-root",    default=DEFAULT_OUT_ROOT)
    parser.add_argument("--vector-root", default=DEFAULT_VECTOR_ROOT,
                        help="Root dir for per-model truth vectors")
    parser.add_argument("--models",      nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS,
                        help="Which models to evaluate")
    parser.add_argument("--conditions",  nargs="+", default=DEFAULT_CONDITIONS,
                        choices=DEFAULT_CONDITIONS,
                        help="Which conditions to run")
    parser.add_argument("--model-paths", nargs="+", metavar="TAG=PATH",
                        help="Override model paths, e.g. phi2=/my/phi2  llama32_3b=/my/llama")
    parser.add_argument("--alphas",      nargs="+", type=float, default=STEER_ALPHAS,
                        help="Alpha values for the steered condition sweep")
    parser.add_argument("--results-only", action="store_true",
                        help="Only print existing results table, no new runs")
    args = parser.parse_args()

    # Override model paths
    if args.model_paths:
        for kv in args.model_paths:
            tag, _, path = kv.partition("=")
            if tag in MODEL_PATHS:
                MODEL_PATHS[tag] = path
            else:
                print(f"[warn] Unknown model tag '{tag}' in --model-paths, ignoring.")

    out_root    = pathlib.Path(args.out_root)
    eval_data   = pathlib.Path(args.eval_data)
    steer_data  = pathlib.Path(args.steer_data)
    vector_root = pathlib.Path(args.vector_root)

    # ── Results-only mode ───────────────────────────────────────────────────
    if args.results_only:
        rows = collect_results(out_root)
        print_results_table(rows)
        return

    # ── Validate data paths ─────────────────────────────────────────────────
    for p, name in [(eval_data, "eval-data"), (steer_data, "steer-data")]:
        if not p.exists():
            sys.exit(f"[eval_baselines] ✗ {name} not found: {p}\n"
                     f"  Run:  python split_dataset.py --full")

    # ── Run evaluation grid ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION GRID")
    print(f"  Models     : {args.models}")
    print(f"  Conditions : {args.conditions}")
    print(f"  Eval data  : {eval_data}  ({sum(1 for _ in open(eval_data))} examples)")
    print("=" * 70)

    all_results = []
    t_start = time()

    for model_tag in args.models:
        model_path = MODEL_PATHS.get(model_tag)
        if not model_path:
            print(f"[warn] No path for model '{model_tag}', skipping.")
            continue

        vector_dir = vector_root / model_tag

        for condition in args.conditions:
            out_dir = out_root / model_tag / condition
            t0 = time()

            try:
                if condition == "no_cot":
                    result = run_no_cot(model_tag, model_path, eval_data, out_dir)

                elif condition == "text_cot":
                    result = run_text_cot(model_tag, model_path, eval_data, out_dir)

                elif condition in ("ccot", "random_noise", "steered"):
                    result = run_hidden_condition(
                        model_tag, model_path, eval_data, steer_data,
                        vector_dir, out_dir, condition,
                        alphas=args.alphas if condition == "steered" else None,
                    )
                else:
                    print(f"[warn] Unknown condition '{condition}', skipping.")
                    continue

            except KeyboardInterrupt:
                print("\n[eval_baselines] Interrupted by user.")
                break
            except Exception as e:
                print(f"[eval_baselines] ERROR in {model_tag}/{condition}: {e}")
                result = {"condition": condition, "model": model_tag,
                          "error": str(e), "exit_code": -1}
                save_result(out_dir, result)

            elapsed = time() - t0
            result["elapsed_seconds"] = round(elapsed, 1)
            save_result(out_dir, result)
            all_results.append(result)
            status = "✓" if result.get("exit_code") == 0 else "✗"
            print(f"\n  {status} {model_tag}/{condition}  ({elapsed:.0f}s)")

    total = time() - t_start
    print(f"\n[eval_baselines] Finished in {total/60:.1f} min\n")

    # ── Final table ─────────────────────────────────────────────────────────
    rows = collect_results(out_root)
    print_results_table(rows)

    summary_path = out_root / "eval_grid_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"[eval_baselines] Full summary → {summary_path}")


if __name__ == "__main__":
    main()
