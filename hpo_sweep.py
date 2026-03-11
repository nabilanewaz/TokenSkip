"""
hpo_sweep.py
============
Hyperparameter optimisation — runs sweeps on the VALIDATION set (2 243 examples)
and writes the best hyperparameter values to:

    outputs/hpo/best_hyperparams.json

Best values are then consumed by run_full_experiment.py for the final test-set run.

Sweeps performed
----------------
  1. alpha         — steering strength for v_truth injection
                     values : [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
                     models : all
  2. layer_frac    — which transformer layer to intervene at (fraction)
                     values : [0.5, 0.625, 0.75]
                     models : HF models only (phi2, llama32_3b, qwen25_3b)
  3. seed          — reproducibility / variance check
                     values : [42, 123, 2025]
                     models : all

Note: learning-rate search for Phase-1 CODI training is expensive
(requires retraining). A separate flag --sweep-lr is provided but
disabled by default. The default LR of 2e-4 is used unless overridden.

Usage
-----
    # Sweep alpha for all models (fastest, most important)
    python hpo_sweep.py --sweep alpha

    # Sweep layer fraction for HF models
    python hpo_sweep.py --sweep layer_frac

    # Seed robustness check
    python hpo_sweep.py --sweep seed

    # Run all sweeps sequentially
    python hpo_sweep.py --sweep all

    # Limit to specific models
    python hpo_sweep.py --sweep alpha --models phi2 qwen25_3b

    # Print best hyperparameters (no new runs)
    python hpo_sweep.py --show-best

Outputs
-------
    outputs/hpo/<model>/<sweep_name>/<value>/metrics.json  — per-config results
    outputs/hpo/best_hyperparams.json                      — best values per model
    outputs/hpo/sweep_log.jsonl                            — full sweep history
"""

import os
import sys
import json
import subprocess
import pathlib
import argparse
from time import time

from experiment_config import (
    MODELS, MODEL_BY_TAG,
    get_dataset, get_split_path,
    HPO_GRIDS, FIXED_SEED, FIXED_ALPHAS,
    BEST_HPARAMS_PATH, PATHS,
    CODI_TRAIN_DEFAULTS,
)


# ── Paths ─────────────────────────────────────────────────────────────────────
HPO_ROOT        = PATHS["hpo_root"]
LOG_PATH        = HPO_ROOT / "sweep_log.jsonl"
VECTOR_ROOT     = PATHS["phase2_vectors"]
CODI_VECTOR_DIR = PATHS["codi_vector"]


# ── Subprocess helpers ────────────────────────────────────────────────────────

def run_cmd(cmd, label="", cwd=None, env=None, timeout=None):
    """Stream subprocess → stdout; return (returncode, full_log_str)."""
    print(f"\n{'-'*60}\n  {label}\n{'-'*60}", flush=True)
    lines = []
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env or os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True,
        encoding="utf-8", errors="replace",
    )
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("[hpo] TIMEOUT -- killing process")
    return proc.returncode, "".join(lines)


def read_metrics(metrics_path: pathlib.Path):
    """Load metrics.json and return dict (empty if missing or unreadable)."""
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text())
    except Exception:
        return {}


def append_log(record: dict):
    HPO_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── Load / save best hyperparameters ─────────────────────────────────────────

def load_best() -> dict:
    """Return {model_tag: {param: best_value, ...}, ...}"""
    if BEST_HPARAMS_PATH.exists():
        try:
            return json.loads(BEST_HPARAMS_PATH.read_text())
        except Exception:
            pass
    return {}


def save_best(best: dict):
    BEST_HPARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_HPARAMS_PATH.write_text(json.dumps(best, indent=2))
    print(f"[hpo] Best hyperparams → {BEST_HPARAMS_PATH}")


def update_best(best: dict, model_tag: str, param: str, value, metric_val: float):
    if model_tag not in best:
        best[model_tag] = {}
    prev = best[model_tag].get(param)
    if prev is None or metric_val > (prev.get("metric_val", -1) if isinstance(prev, dict) else -1):
        best[model_tag][param] = {
            "value":      value,
            "metric_val": metric_val,
        }


# ── Alpha sweep (all models) ──────────────────────────────────────────────────

def sweep_alpha_hf(model_tag: str, model_path: str, val_data: pathlib.Path,
                   steer_data: pathlib.Path, out_root: pathlib.Path,
                   alphas=None, layer_frac=0.75, seed=None):
    """Sweep alpha for one HF model on the validation set."""
    if alphas is None:
        alphas = FIXED_ALPHAS
    if seed is None:
        seed = FIXED_SEED

    results = []
    for alpha in alphas:
        out_dir = out_root / f"alpha_{alpha}"
        metrics_path = out_dir / "metrics.json"
        if metrics_path.exists():
            print(f"[hpo] Skipping alpha={alpha} (cached)")
            metrics = read_metrics(metrics_path)
        else:
            cmd = [
                sys.executable, "hidden_steer.py",
                "--model-path",  model_path,
                "--model-type",  model_tag,
                "--eval-data",   str(val_data),
                "--steer-data",  str(steer_data),
                "--vector-dir",  str(VECTOR_ROOT / model_tag),
                "--out-dir",     str(out_dir),
                "--condition",   "steered",
                "--alphas",      str(alpha),
                "--layer-frac",  str(layer_frac),
                "--seed",        str(seed),
            ]
            label = f"alpha_sweep/{model_tag}  α={alpha}"
            rc, _ = run_cmd(cmd, label=label)
            metrics = read_metrics(metrics_path)
            if rc != 0:
                print(f"[hpo] Warning: run exited with code {rc}")

        acc       = metrics.get("accuracy", 0.0)
        flip_rate = metrics.get("flip_rate")
        cos_sim   = metrics.get("mean_cos_sim")
        results.append({
            "alpha": alpha, "accuracy": acc,
            "flip_rate": flip_rate, "mean_cos_sim": cos_sim,
        })
        append_log({
            "sweep": "alpha", "model": model_tag, "alpha": alpha,
            "accuracy": acc, "flip_rate": flip_rate, "mean_cos_sim": cos_sim,
        })

    return results


def sweep_alpha_codi(val_data: pathlib.Path, vector_dir: pathlib.Path,
                     out_root: pathlib.Path, alphas=None, seed=None):
    """Sweep alpha for CODI-GPT2 on the validation set."""
    if alphas is None:
        alphas = FIXED_ALPHAS
    if seed is None:
        seed = FIXED_SEED

    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.json"
    if summary_path.exists():
        print("[hpo] CODI alpha sweep cached -- skipping")
        summary = json.loads(summary_path.read_text())
        return summary.get("results", [])

    cmd = [
        sys.executable, "phase3_steer_inference.py",
        "--eval-data",  str(val_data),
        "--vector-dir", str(vector_dir),
        "--out-dir",    str(out_root),
        "--alphas",     *[str(a) for a in alphas],
        "--seed",       str(seed),
    ]
    label = "alpha_sweep/codi_gpt2 (validation set)"
    rc, _ = run_cmd(cmd, label=label)
    if rc != 0:
        print(f"[hpo] Warning: CODI run exited with code {rc}")

    # Read summary
    results = []
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        results = summary.get("results", [])
        for r in results:
            append_log({
                "sweep": "alpha", "model": "codi_gpt2",
                "alpha": r.get("alpha"), "accuracy": r.get("accuracy"),
            })
    return results


# ── Layer-fraction sweep (HF models only) ────────────────────────────────────

def sweep_layer_frac(model_tag: str, model_path: str, val_data: pathlib.Path,
                     steer_data: pathlib.Path, out_root: pathlib.Path,
                     fracs=None, alpha=1.0, seed=None):
    """Sweep intervention layer fraction for one HF model."""
    if fracs is None:
        fracs = HPO_GRIDS["layer_frac"]["values"]
    if seed is None:
        seed = FIXED_SEED

    results = []
    for frac in fracs:
        key      = f"layer_{frac}"
        out_dir  = out_root / key
        mpath    = out_dir / "metrics.json"
        if mpath.exists():
            print(f"[hpo] Skipping layer_frac={frac} (cached)")
            metrics = read_metrics(mpath)
        else:
            cmd = [
                sys.executable, "hidden_steer.py",
                "--model-path",  model_path,
                "--model-type",  model_tag,
                "--eval-data",   str(val_data),
                "--steer-data",  str(steer_data),
                "--vector-dir",  str(VECTOR_ROOT / model_tag),
                "--out-dir",     str(out_dir),
                "--condition",   "steered",
                "--alphas",      str(alpha),
                "--layer-frac",  str(frac),
                "--seed",        str(seed),
            ]
            rc, _ = run_cmd(cmd, label=f"layer_frac_sweep/{model_tag}  frac={frac}")
            metrics = read_metrics(mpath)
            if rc != 0:
                print(f"[hpo] Warning: run exited with code {rc}")

        acc = metrics.get("accuracy", 0.0)
        results.append({"layer_frac": frac, "accuracy": acc})
        append_log({
            "sweep": "layer_frac", "model": model_tag,
            "layer_frac": frac, "accuracy": acc,
        })

    return results


# ── Seed robustness check ─────────────────────────────────────────────────────

def sweep_seed(model_tag: str, model_path: str, val_data: pathlib.Path,
               steer_data: pathlib.Path, out_root: pathlib.Path,
               seeds=None, alpha=1.0, layer_frac=0.75, kind="hf"):
    """Verify result stability across seeds."""
    if seeds is None:
        seeds = HPO_GRIDS["seed"]["values"]

    results = []
    for seed in seeds:
        out_dir  = out_root / f"seed_{seed}"
        mpath    = out_dir / "metrics.json"
        if mpath.exists():
            print(f"[hpo] Skipping seed={seed} (cached)")
            metrics = read_metrics(mpath)
        else:
            if kind == "codi":
                cmd = [
                    sys.executable, "phase3_steer_inference.py",
                    "--eval-data",  str(val_data),
                    "--vector-dir", str(CODI_VECTOR_DIR),
                    "--out-dir",    str(out_dir),
                    "--alphas",     str(alpha),
                    "--seed",       str(seed),
                ]
            else:
                cmd = [
                    sys.executable, "hidden_steer.py",
                    "--model-path",  model_path,
                    "--model-type",  model_tag,
                    "--eval-data",   str(val_data),
                    "--steer-data",  str(steer_data),
                    "--vector-dir",  str(VECTOR_ROOT / model_tag),
                    "--out-dir",     str(out_dir),
                    "--condition",   "steered",
                    "--alphas",      str(alpha),
                    "--layer-frac",  str(layer_frac),
                    "--seed",        str(seed),
                ]
            rc, _ = run_cmd(cmd, label=f"seed_sweep/{model_tag}  seed={seed}")
            metrics = read_metrics(mpath)
            if rc != 0:
                print(f"[hpo] Warning: run exited with code {rc}")

        acc = metrics.get("accuracy", 0.0)
        results.append({"seed": seed, "accuracy": acc})
        append_log({
            "sweep": "seed", "model": model_tag, "seed": seed, "accuracy": acc,
        })

    if results:
        accs = [r["accuracy"] for r in results if r["accuracy"] is not None]
        mean = sum(accs) / len(accs) if accs else 0
        var  = sum((a - mean) ** 2 for a in accs) / len(accs) if accs else 0
        std  = var ** 0.5
        print(f"[hpo] Seed robustness  {model_tag}: "
              f"mean={mean:.2%}  std={std:.4f}  n={len(accs)}")

    return results


# ── Best-value picker ─────────────────────────────────────────────────────────

def pick_best(results: list, key: str, metric: str = "accuracy") -> tuple:
    """Return (best_value, best_metric_val) from a list of result dicts."""
    if not results:
        return None, 0.0
    best = max(results, key=lambda r: r.get(metric) or 0.0)
    return best[key], best.get(metric, 0.0)


# ── Print best table ──────────────────────────────────────────────────────────

def print_best(best: dict):
    if not best:
        print("[hpo] No best hyperparameters saved yet.")
        return

    print("\n" + "=" * 60)
    print("  BEST HYPERPARAMETERS  (from validation sweep)")
    print("=" * 60)
    col_w = 14
    params = sorted({p for m in best.values() for p in m})
    hdr    = f"  {'Model':14s}" + "".join(f"{p[:col_w]:>{col_w}}" for p in params)
    print(hdr)
    print("  " + "-" * (14 + col_w * len(params)))
    for model_tag, hparams in sorted(best.items()):
        row = f"  {model_tag:14s}"
        for p in params:
            entry = hparams.get(p, {})
            v     = entry.get("value", "—") if isinstance(entry, dict) else "—"
            row  += f"{str(v):>{col_w}}"
        print(row)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep on the GSM8K validation set"
    )
    parser.add_argument(
        "--sweep", nargs="+",
        choices=["alpha", "layer_frac", "seed", "all"],
        default=["alpha"],
        help="Which parameter(s) to sweep",
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=[m["tag"] for m in MODELS],
        default=None,
        help="Subset of models to sweep (default: all)",
    )
    parser.add_argument(
        "--model-paths", nargs="*", default=[],
        metavar="TAG=PATH",
        help="Override HF paths, e.g.  phi2=/local/phi2",
    )
    parser.add_argument("--show-best", action="store_true",
                        help="Print best hyperparams and exit (no new runs)")
    args = parser.parse_args()

    # ── Handle --show-best ──────────────────────────────────────────────────
    if args.show_best:
        print_best(load_best())
        return

    # ── Override model paths ────────────────────────────────────────────────
    path_overrides = {}
    for kv in (args.model_paths or []):
        if "=" in kv:
            k, v = kv.split("=", 1)
            path_overrides[k] = v
    for m in MODELS:
        if m["tag"] in path_overrides:
            m["hf_id"] = path_overrides[m["tag"]]

    # ── Resolve model list ──────────────────────────────────────────────────
    sweep_tags = args.models or [m["tag"] for m in MODELS]
    sweep_models = [m for m in MODELS if m["tag"] in sweep_tags]

    # ── Dataset paths ───────────────────────────────────────────────────────
    val_data   = get_split_path("validation")
    steer_data = get_split_path("steer_train")

    # ── Expand "all" ────────────────────────────────────────────────────────
    sweeps = args.sweep
    if "all" in sweeps:
        sweeps = ["alpha", "layer_frac", "seed"]

    best = load_best()
    t0   = time()

    print(f"\n[hpo] Sweep(s)  : {sweeps}")
    print(f"[hpo] Models    : {[m['tag'] for m in sweep_models]}")
    print(f"[hpo] Val data  : {val_data}  ({2243} examples)")
    print(f"[hpo] Steer data: {steer_data}\n")

    for model in sweep_models:
        tag      = model["tag"]
        hf_id    = model["hf_id"]
        kind     = model["kind"]
        model_root = HPO_ROOT / tag

        # ── 1. Alpha sweep ──────────────────────────────────────────────────
        if "alpha" in sweeps:
            print(f"\n[hpo] -- Alpha sweep for {tag} " + "-"*26)
            alpha_out = model_root / "alpha_sweep"

            if kind == "codi":
                results = sweep_alpha_codi(
                    val_data=val_data,
                    vector_dir=CODI_VECTOR_DIR,
                    out_root=alpha_out,
                )
            else:
                results = sweep_alpha_hf(
                    model_tag=tag, model_path=hf_id,
                    val_data=val_data, steer_data=steer_data,
                    out_root=alpha_out,
                )

            best_a, best_acc = pick_best(results, "alpha")
            print(f"\n[hpo] Best alpha for {tag}: {best_a}  "
                  f"(val accuracy = {best_acc:.2%})")
            update_best(best, tag, "alpha", best_a, best_acc)

            # Save sweep results
            alpha_out.mkdir(parents=True, exist_ok=True)
            (alpha_out / "sweep_results.json").write_text(
                json.dumps({"model": tag, "sweep": "alpha",
                            "results": results, "best": best_a}, indent=2)
            )

        # ── 2. Layer-fraction sweep (HF only) ───────────────────────────────
        if "layer_frac" in sweeps and kind == "hf":
            print(f"\n[hpo] -- Layer-fraction sweep for {tag} " + "-"*17)
            # Use best alpha if available (fallback to 1.0)
            best_alpha = (best.get(tag, {}).get("alpha", {}) or {}).get("value", 1.0)
            frac_out   = model_root / "layer_frac_sweep"

            results = sweep_layer_frac(
                model_tag=tag, model_path=hf_id,
                val_data=val_data, steer_data=steer_data,
                out_root=frac_out, alpha=best_alpha,
            )

            best_f, best_acc = pick_best(results, "layer_frac")
            print(f"\n[hpo] Best layer_frac for {tag}: {best_f}  "
                  f"(val accuracy = {best_acc:.2%})")
            update_best(best, tag, "layer_frac", best_f, best_acc)

            frac_out.mkdir(parents=True, exist_ok=True)
            (frac_out / "sweep_results.json").write_text(
                json.dumps({"model": tag, "sweep": "layer_frac",
                            "results": results, "best": best_f}, indent=2)
            )

        # ── 3. Seed robustness ───────────────────────────────────────────────
        if "seed" in sweeps:
            print(f"\n[hpo] -- Seed robustness check for {tag} " + "-"*17)
            best_alpha = (best.get(tag, {}).get("alpha", {}) or {}).get("value", 1.0)
            best_frac  = (best.get(tag, {}).get("layer_frac", {}) or {}).get("value", 0.75)
            seed_out   = model_root / "seed_sweep"

            results = sweep_seed(
                model_tag=tag, model_path=hf_id,
                val_data=val_data, steer_data=steer_data,
                out_root=seed_out, alpha=best_alpha,
                layer_frac=best_frac, kind=kind,
            )

            seed_out.mkdir(parents=True, exist_ok=True)
            accs = [r["accuracy"] for r in results if r.get("accuracy") is not None]
            mean = sum(accs) / len(accs) if accs else 0
            std  = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5 if accs else 0
            (seed_out / "sweep_results.json").write_text(
                json.dumps({
                    "model": tag, "sweep": "seed",
                    "results": results,
                    "mean_accuracy": mean, "std_accuracy": std,
                }, indent=2)
            )

    # ── Save best hyperparameters ───────────────────────────────────────────
    save_best(best)
    print_best(best)

    elapsed = time() - t0
    print(f"\n[hpo] Total sweep time: {elapsed/60:.1f} min")
    print(f"[hpo] Log → {LOG_PATH}")
    print(f"[hpo] Best params → {BEST_HPARAMS_PATH}")


if __name__ == "__main__":
    main()
