"""
run_full_experiment.py
======================
Master orchestrator for the complete research experiment:

    "Steering Continuous Reasoning via Latent Intervention"

Pipeline
--------
  Step 0 — Print experiment configuration (models, datasets, hyperparams)
  Step 1 — HPO: sweep alpha / layer_frac on VALIDATION set
  Step 2 — Full evaluation on TEST set using best hyperparameters
            ├── CODI-GPT2     : phase3_steer_inference.py  (+random-noise flag)
            └── HF models     : evaluate_baselines.py
                                 → evaluation.py     (no_cot, text_cot)
                                 → hidden_steer.py   (ccot, random_noise, steered)
  Step 3 — Aggregate results  : compare_all.py
  Step 4 — Write experiment report JSON

All stdout + stderr is simultaneously written to:
    outputs/logs/experiment_<TIMESTAMP>.log

Usage
-----
    # Run everything end-to-end
    python run_full_experiment.py

    # Skip HPO (use cached best_hyperparams.json or protocol defaults)
    python run_full_experiment.py --skip-hpo

    # Skip training/HPO, re-run evaluation only
    python run_full_experiment.py --skip-hpo --models phi2 llama32_3b

    # Run only specific conditions
    python run_full_experiment.py --conditions no_cot text_cot steered

    # Override model paths
    python run_full_experiment.py --model-paths phi2=/local/phi-2

    # Dry run — print plan without executing
    python run_full_experiment.py --dry-run
"""

import os
import sys
import json
import subprocess
import pathlib
import argparse
import datetime
from time import time


# ── Experiment config import ─────────────────────────────────────────────────
from experiment_config import (
    MODELS, MODEL_BY_TAG,
    DATASETS, get_dataset, get_split_path,
    CONDITIONS, CONDITION_ORDER,
    FIXED_SEED, FIXED_ALPHAS,
    BEST_HPARAMS_PATH, PATHS, METRICS,
    CODI_TRAIN_DEFAULTS,
    print_experiment_summary,
)


# ══════════════════════════════════════════════════════════════════════════════
#  TeeLogger — writes to stdout AND a log file simultaneously
# ══════════════════════════════════════════════════════════════════════════════

class TeeLogger:
    """Duplicate all writes to both the original stream and a file."""

    def __init__(self, stream, filepath: pathlib.Path):
        self._stream  = stream
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file    = open(filepath, "w", encoding="utf-8", buffering=1)
        self.filepath = filepath

    def write(self, msg):
        self._stream.write(msg)
        self._file.write(msg)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    # Make it usable as a context manager too
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Subprocess helper
# ══════════════════════════════════════════════════════════════════════════════

def run_cmd(cmd, label="", cwd=None, env=None, dry_run=False):
    """
    Stream subprocess output to stdout (which is already tee-ed to log).
    Returns (returncode, full_output_str).
    """
    bar = "-" * 60
    print(f"\n{bar}\n  {label}\n{bar}", flush=True)

    if dry_run:
        print(f"  [DRY-RUN] would execute:\n  {' '.join(str(c) for c in cmd)}\n")
        return 0, ""

    lines = []
    try:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=cwd,
            env=env or os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        proc.wait()
        return proc.returncode, "".join(lines)
    except Exception as e:
        print(f"  [ERROR] Could not launch process: {e}")
        return -1, str(e)


def read_metrics(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 — HPO
# ══════════════════════════════════════════════════════════════════════════════

def run_hpo(sweep_models, skip_hpo: bool, model_paths_override: dict,
            dry_run: bool) -> dict:
    """
    Run hpo_sweep.py --sweep alpha layer_frac for all non-CODI models.
    Returns the best_hyperparams dict (loaded or freshly computed).
    """
    if skip_hpo:
        print("\n[run_exp] --skip-hpo: loading cached hyperparameters …")
        if BEST_HPARAMS_PATH.exists():
            best = json.loads(BEST_HPARAMS_PATH.read_text())
            print(f"  Loaded from {BEST_HPARAMS_PATH}")
        else:
            print("  No cached best_hyperparams.json found -- using protocol defaults.")
            best = {}
        return best

    tags = [m["tag"] for m in sweep_models]
    cmd  = [
        sys.executable, "hpo_sweep.py",
        "--sweep", "alpha", "layer_frac",
        "--models", *tags,
    ]
    for tag, path in model_paths_override.items():
        cmd += ["--model-paths", f"{tag}={path}"]

    run_cmd(cmd, label="HPO sweep (alpha + layer_frac on validation set)",
            dry_run=dry_run)

    best = {}
    if BEST_HPARAMS_PATH.exists():
        best = json.loads(BEST_HPARAMS_PATH.read_text())
    return best


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2a — CODI-GPT2 full evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_codi_eval(test_data: pathlib.Path, best: dict,
                  out_root: pathlib.Path, conditions: list,
                  dry_run: bool) -> dict:
    """
    Run phase3_steer_inference.py (alpha sweep + random noise) on the test set.
    Returns a dict of condition → metrics.
    """
    tag        = "codi_gpt2"
    vector_dir = PATHS["phase2_vectors"] / tag
    codi_out   = out_root / tag
    codi_out.mkdir(parents=True, exist_ok=True)

    # Best alpha from HPO (fallback to full sweep)
    best_alpha_info = best.get(tag, {}).get("alpha", {})
    best_alpha = best_alpha_info.get("value") if isinstance(best_alpha_info, dict) else None
    alphas     = [best_alpha] if best_alpha is not None else FIXED_ALPHAS

    include_random = "random_noise" in conditions

    cmd = [
        sys.executable, "phase3_steer_inference.py",
        "--eval-data",  str(test_data),
        "--vector-dir", str(vector_dir),
        "--out-dir",    str(codi_out),
        "--alphas",     *[str(a) for a in alphas],
        "--seed",       str(FIXED_SEED),
    ]
    if include_random:
        cmd.append("--random-noise")

    run_cmd(cmd, label=f"CODI-GPT2 evaluation on test set  (α={alphas})",
            dry_run=dry_run)

    # Collect per-condition metrics
    results = {}
    summary_p = codi_out / "summary.json"
    if summary_p.exists():
        summary = json.loads(summary_p.read_text())
        for r in summary.get("results", []):
            alpha = r.get("alpha", 0.0)
            cond  = "ccot" if alpha == 0.0 else "steered"
            results[cond] = {
                "accuracy":     r.get("accuracy"),
                "mean_cos_sim": r.get("mean_cos_sim"),
            }

    flip_p = codi_out / "flip_analysis.json"
    if flip_p.exists():
        flips = json.loads(flip_p.read_text())
        if isinstance(flips, list):
            for f in flips:
                cond = f.get("condition", "steered")
                if cond in results:
                    results[cond]["flip_rate"] = f.get("flip_rate_pos")

    # Random-noise condition
    rn_metrics_p = codi_out / "random_noise" / "metrics.json"
    if rn_metrics_p.exists():
        results["random_noise"] = read_metrics(rn_metrics_p)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2b — HF model full evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_hf_eval(sweep_models, test_data: pathlib.Path, steer_data: pathlib.Path,
                best: dict, out_root: pathlib.Path,
                conditions: list, alphas_override=None,
                model_paths_override: dict = None,
                dry_run: bool = False) -> dict:
    """
    Run evaluate_baselines.py for all HF models × requested conditions.
    Returns {model_tag: {condition: metrics_dict}}.
    """
    if model_paths_override is None:
        model_paths_override = {}

    tags       = [m["tag"] for m in sweep_models if m["kind"] == "hf"]
    vector_root = PATHS["phase2_vectors"]

    # Build --alphas arg: per-model best alpha if available, else full sweep
    # We pass a comma-free list; evaluate_baselines.py accepts nargs
    def best_alphas_for(tag):
        info = best.get(tag, {}).get("alpha", {})
        if isinstance(info, dict) and info.get("value") is not None:
            return [info["value"]]
        return alphas_override or FIXED_ALPHAS

    # Build --model-paths override list
    path_args = []
    for tag in tags:
        override = model_paths_override.get(tag)
        if override:
            path_args += [f"{tag}={override}"]

    # Collect best layer_frac per model
    frac_args = []
    for tag in tags:
        info = best.get(tag, {}).get("layer_frac", {})
        frac = info.get("value", 0.75) if isinstance(info, dict) else 0.75
        frac_args += [f"{tag}={frac}"]

    cmd = [
        sys.executable, "evaluate_baselines.py",
        "--models",      *tags,
        "--conditions",  *conditions,
        "--eval-data",   str(test_data),
        "--steer-data",  str(steer_data),
        "--out-root",    str(out_root),
        "--vector-root", str(vector_root),
    ]
    if path_args:
        cmd += ["--model-paths", *path_args]

    run_cmd(cmd, label=f"HF models evaluation on test set  {tags}",
            dry_run=dry_run)

    # Collect results
    results = {}
    for tag in tags:
        results[tag] = {}
        for cond in conditions:
            mp = out_root / tag / cond / "metrics.json"
            if mp.exists():
                results[tag][cond] = read_metrics(mp)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 — compare_all.py
# ══════════════════════════════════════════════════════════════════════════════

def run_compare(report_path: pathlib.Path, dry_run: bool):
    cmd = [
        sys.executable, "compare_all.py",
        "--grid",  str(PATHS["eval_grid"]),
        "--codi",  str(PATHS["phase3_results"]),
        "--json",  str(report_path.with_suffix(".json")),
        "--csv",   str(report_path.with_suffix(".csv")),
        "--latex",
    ]
    run_cmd(cmd, label="Results aggregation (compare_all.py)", dry_run=dry_run)


# ══════════════════════════════════════════════════════════════════════════════
#  Report writer
# ══════════════════════════════════════════════════════════════════════════════

def write_report(report: dict, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(f"\n[run_exp] Experiment report → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run the full Steering Continuous Reasoning experiment"
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=[m["tag"] for m in MODELS] + ["all"],
        default=["all"],
        help="Models to evaluate (default: all)",
    )
    parser.add_argument(
        "--conditions", nargs="+",
        choices=CONDITION_ORDER + ["all"],
        default=["all"],
        help="Conditions to evaluate (default: all)",
    )
    parser.add_argument(
        "--dataset", default="gsm8k",
        help="Dataset to use (default: gsm8k)",
    )
    parser.add_argument(
        "--model-paths", nargs="*", default=[],
        metavar="TAG=PATH",
        help="Override HF paths, e.g.  phi2=/local/phi-2",
    )
    parser.add_argument(
        "--skip-hpo", action="store_true",
        help="Skip HPO sweep; use cached or protocol-default hyperparameters",
    )
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=None,
        help=f"Override alpha sweep values (default: {FIXED_ALPHAS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print all commands without executing them",
    )
    parser.add_argument(
        "--log-dir", default=str(PATHS["logs"]),
        help="Directory to write experiment log files",
    )
    args = parser.parse_args()

    # ── Timestamp ─────────────────────────────────────────────────────────────
    ts          = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir     = pathlib.Path(args.log_dir)
    log_path    = log_dir / f"experiment_{ts}.log"
    report_path = log_dir / f"report_{ts}"

    # ── Tee logging ───────────────────────────────────────────────────────────
    tee = TeeLogger(sys.stdout, log_path)
    sys.stdout = tee
    print(f"[run_exp] Log file: {log_path}")
    print(f"[run_exp] Started : {datetime.datetime.now().isoformat()}")

    try:
        _run(args, ts, report_path, log_path)
    except KeyboardInterrupt:
        print("\n[run_exp] Interrupted by user.")
    except Exception as e:
        import traceback
        print(f"\n[run_exp] FATAL ERROR: {e}")
        traceback.print_exc()
    finally:
        sys.stdout = tee._stream
        tee.close()
        print(f"[run_exp] Log written to: {log_path}")


def _run(args, ts: str, report_path: pathlib.Path, log_path: pathlib.Path):
    t_start = time()

    # ── Resolve model/condition lists ─────────────────────────────────────────
    if "all" in args.models:
        run_models = list(MODELS)
    else:
        run_models = [m for m in MODELS if m["tag"] in args.models]

    if "all" in args.conditions:
        run_conditions = list(CONDITION_ORDER)
    else:
        run_conditions = [c for c in CONDITION_ORDER if c in args.conditions]

    # Parse model path overrides
    path_overrides: dict[str, str] = {}
    for kv in args.model_paths or []:
        if "=" in kv:
            k, v = kv.split("=", 1)
            path_overrides[k] = v

    # ── Step 0 — Configuration summary ───────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  EXPERIMENT RUN  [{ts}]")
    print("=" * 70)
    print_experiment_summary()

    ds         = get_dataset(args.dataset)
    test_data  = get_split_path("test",        args.dataset)
    steer_data = get_split_path("steer_train", args.dataset)

    print(f"\n[run_exp] Models    : {[m['tag'] for m in run_models]}")
    print(f"[run_exp] Conditions: {run_conditions}")
    print(f"[run_exp] Test data : {test_data}")
    print(f"[run_exp] Dry run   : {args.dry_run}")

    # Verify dataset files exist
    for split_name in ("llm_train", "steer_train", "validation", "test"):
        p = get_split_path(split_name, args.dataset)
        status = "OK" if p.exists() else "MISSING"
        n = ds["split_sizes"].get(split_name, "?")
        print(f"  [{status}] {split_name:14s} {p}  ({n} examples)")
        if status == "MISSING" and not args.dry_run:
            print(f"\n[run_exp] ERROR: {p} not found. "
                  f"Run: python split_dataset.py --full")
            # Continue anyway — individual scripts will fail with clearer messages

    report = {
        "run_id":     ts,
        "started_at": datetime.datetime.now().isoformat(),
        "log_file":   str(log_path),
        "models":     [m["tag"] for m in run_models],
        "conditions": run_conditions,
        "dataset":    args.dataset,
        "split_sizes": ds["split_sizes"],
        "phases":     {},
    }

    # ── Step 1 — HPO ──────────────────────────────────────────────────────────
    phase_t = time()
    print(f"\n{'='*70}\n  STEP 1 -- Hyperparameter Optimisation\n{'='*70}")

    best = run_hpo(
        sweep_models=run_models,
        skip_hpo=args.skip_hpo,
        model_paths_override=path_overrides,
        dry_run=args.dry_run,
    )

    report["phases"]["hpo"] = {
        "elapsed_s": round(time() - phase_t, 1),
        "skipped":   args.skip_hpo,
        "best_hyperparams": best,
    }
    print(f"\n[run_exp] Best hyperparameters:\n{json.dumps(best, indent=2)}")

    # ── Step 2a — CODI evaluation ─────────────────────────────────────────────
    codi_models = [m for m in run_models if m["kind"] == "codi"]
    codi_results = {}

    if codi_models and (set(run_conditions) & {"ccot", "random_noise", "steered"}):
        phase_t = time()
        print(f"\n{'='*70}\n  STEP 2a -- CODI-GPT2 Evaluation\n{'='*70}")

        codi_results = run_codi_eval(
            test_data=test_data,
            best=best,
            out_root=PATHS["phase3_results"],
            conditions=run_conditions,
            dry_run=args.dry_run,
        )
        report["phases"]["codi_eval"] = {
            "elapsed_s": round(time() - phase_t, 1),
            "results":   codi_results,
        }

    # ── Step 2b — HF model evaluation ────────────────────────────────────────
    hf_models = [m for m in run_models if m["kind"] == "hf"]
    hf_results = {}

    if hf_models:
        phase_t = time()
        print(f"\n{'='*70}\n  STEP 2b -- HF Model Evaluation\n{'='*70}")

        hf_results = run_hf_eval(
            sweep_models=hf_models,
            test_data=test_data,
            steer_data=steer_data,
            best=best,
            out_root=PATHS["eval_grid"],
            conditions=run_conditions,
            alphas_override=args.alphas,
            model_paths_override=path_overrides,
            dry_run=args.dry_run,
        )
        report["phases"]["hf_eval"] = {
            "elapsed_s": round(time() - phase_t, 1),
            "results":   hf_results,
        }

    # ── Step 3 — Results aggregation ─────────────────────────────────────────
    phase_t = time()
    print(f"\n{'='*70}\n  STEP 3 -- Results Aggregation\n{'='*70}")

    run_compare(report_path=report_path, dry_run=args.dry_run)

    report["phases"]["compare"] = {
        "elapsed_s": round(time() - phase_t, 1),
        "csv":   str(report_path.with_suffix(".csv")),
        "json":  str(report_path.with_suffix(".json")),
    }

    # ── Step 4 — Final report ─────────────────────────────────────────────────
    elapsed = round(time() - t_start, 1)
    report["finished_at"]  = datetime.datetime.now().isoformat()
    report["total_elapsed_s"] = elapsed
    report["total_elapsed_min"] = round(elapsed / 60, 2)

    write_report(report, report_path.with_suffix(".experiment.json"))

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Total time : {elapsed/60:.1f} min")
    print(f"  Log        : {log_path}")
    print(f"  CSV        : {report_path.with_suffix('.csv')}")
    print(f"  JSON report: {report_path.with_suffix('.experiment.json')}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
