"""
compare_all.py
--------------
Collects and displays the full evaluation grid:

  Models × Conditions → Accuracy / Flip Rate / Cosine Similarity

Reads results from two locations:

  1. outputs/eval_grid/<model>/<condition>/metrics.json
     (produced by evaluate_baselines.py → hidden_steer.py for phi2/llama/qwen)

  2. outputs/steering_results/summary.json  +
     outputs/steering_results/flip_analysis.json
     (produced by steer_inference.py for CODI-GPT2)

  3. outputs/*/gsm8k/*/samples/metrics.json
     (produced by evaluation.py for text_cot / no_cot TokenSkip runs)

Usage
-----
  python compare_all.py                        # default paths
  python compare_all.py --grid outputs/eval_grid  --codi outputs/steering_results
  python compare_all.py --csv results.csv      # also export to CSV
  python compare_all.py --latex                # print LaTeX table
"""

import json, sys, pathlib, argparse, csv
from typing import Optional


# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_GRID_ROOT  = "outputs/eval_grid"
DEFAULT_CODI_DIR   = "outputs/steering_results"

MODEL_DISPLAY = {
    "phi2":       "phi-2",
    "llama32_3b": "Llama-3.2-3B",
    "qwen25_3b":  "Qwen2.5-3B",
    "codi_gpt2":  "CODI-GPT2",
}
CONDITION_DISPLAY = {
    "no_cot":      "No CoT",
    "text_cot":    "Text CoT",
    "ccot":        "CCoT (unsteered)",
    "random_noise":"Random Noise",
    "steered":     "CCoT+v_truth",
}
CONDITION_ORDER = ["no_cot", "text_cot", "ccot", "random_noise", "steered"]
METRICS = ["accuracy", "flip_rate", "mean_cos_sim"]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_grid(grid_root: pathlib.Path) -> list[dict]:
    """Load all metrics.json from outputs/eval_grid/<model>/<condition>/"""
    rows = []
    for mp in sorted(grid_root.glob("*/*/metrics.json")):
        try:
            data = json.loads(mp.read_text())
            parts = mp.parts
            base  = len(grid_root.parts)
            model = parts[base]     if len(parts) > base     else "unknown"
            cond  = parts[base + 1] if len(parts) > base + 1 else "unknown"
            rows.append({
                "model":          model,
                "condition":      data.get("condition", cond),
                "accuracy":       data.get("accuracy"),
                "flip_rate":      data.get("flip_rate"),
                "mean_cos_sim":   data.get("mean_cos_sim"),
                "_path":          str(mp),
            })
        except Exception as e:
            print(f"[warn] Could not load {mp}: {e}")
    return rows


def load_codi_steering(codi_dir: pathlib.Path) -> list[dict]:
    """Load CODI steering sweep results from steer_inference.py output."""
    rows = []
    summary_path = codi_dir / "summary.json"
    flip_path    = codi_dir / "flip_analysis.json"

    if not summary_path.exists():
        return rows

    try:
        summary = json.loads(summary_path.read_text())
        flip    = json.loads(flip_path.read_text()) if flip_path.exists() else []
    except Exception as e:
        print(f"[warn] Could not load CODI results: {e}")
        return rows

    flip_by_cond = {f["condition"]: f for f in flip} if isinstance(flip, list) else {}

    for result in summary.get("results", []):
        alpha = result.get("alpha", 0.0)
        cond  = result.get("condition") or ("random_noise" if result.get("random_noise") else f"alpha_{alpha}")

        # Map alpha values to canonical condition names
        if not result.get("random_noise"):
            if alpha == 0.0:
                canonical = "ccot"
            else:
                canonical = "steered"
        else:
            canonical = "random_noise"

        fa = flip_by_cond.get(cond, {})
        rows.append({
            "model":        "codi_gpt2",
            "condition":    canonical,
            "alpha":        alpha,
            "accuracy":     result.get("accuracy"),
            "flip_rate":    fa.get("flip_rate_pos"),
            "mean_cos_sim": fa.get("mean_cos_sim"),
            "_path":        str(summary_path),
        })

    # If multiple steered entries, keep the best accuracy
    steered = [r for r in rows if r["condition"] == "steered" and r["model"] == "codi_gpt2"]
    other   = [r for r in rows if not (r["condition"] == "steered" and r["model"] == "codi_gpt2")]
    if steered:
        best = max(steered, key=lambda r: r.get("accuracy") or 0.0)
        rows = other + [best]

    return rows


def load_tokenskip_evals(outputs_root: pathlib.Path) -> list[dict]:
    """
    Load evaluation.py outputs (text_cot / no_cot) from outputs/
    These are stored in outputs/<ModelName>/<benchmark>/<size>/<condition>/samples/metrics.json
    """
    rows = []
    for mp in sorted(outputs_root.glob("*/*/*/samples/metrics.json")):
        try:
            data = json.loads(mp.read_text())
            # Infer model tag from directory name
            parts   = mp.parts
            model_d = parts[len(outputs_root.parts)]
            # Try to map to a known tag
            model_tag = _map_model_dirname(model_d)
            # Infer condition: if in "Original" path → text_cot; "no_cot" path → no_cot
            path_str = str(mp).lower()
            if "no_cot" in path_str:
                cond = "no_cot"
            else:
                cond = "text_cot"
            rows.append({
                "model":        model_tag,
                "condition":    cond,
                "accuracy":     data.get("accuracy"),
                "flip_rate":    None,
                "mean_cos_sim": None,
                "_path":        str(mp),
            })
        except Exception as e:
            print(f"[warn] Could not load TokenSkip eval {mp}: {e}")
    return rows


def _map_model_dirname(dirname: str) -> str:
    mapping = {
        "phi-2":         "phi2",
        "phi2":          "phi2",
        "llama-3.2-3b":  "llama32_3b",
        "llama3.2-3b":   "llama32_3b",
        "llama32_3b":    "llama32_3b",
        "qwen2.5-3b":    "qwen25_3b",
        "qwen25_3b":     "qwen25_3b",
        "codi":          "codi_gpt2",
        "codi-gpt2":     "codi_gpt2",
        "gpt2":          "codi_gpt2",
    }
    return mapping.get(dirname.lower(), dirname)


# ── Deduplication: when multiple entries for same (model, condition), keep best ─

def deduplicate(rows: list[dict]) -> list[dict]:
    seen = {}
    for r in rows:
        key = (r["model"], r["condition"])
        existing = seen.get(key)
        if existing is None:
            seen[key] = r
        else:
            # prefer highest accuracy
            if (r.get("accuracy") or 0.0) > (existing.get("accuracy") or 0.0):
                seen[key] = r
    return list(seen.values())


# ── Formatters ────────────────────────────────────────────────────────────────

def fmt(value, metric: str, default: str = "—") -> str:
    if value is None:
        return default
    try:
        v = float(value)
        if metric == "accuracy":
            return f"{v*100:.2f}%"
        elif metric == "flip_rate":
            return f"{v*100:.2f}%"
        else:
            return f"{v:.4f}"
    except (TypeError, ValueError):
        return str(value)


# ── Print tables ──────────────────────────────────────────────────────────────

def _print_table(rows: list[dict], metric: str, title: str):
    all_models     = sorted({r["model"] for r in rows},
                            key=lambda m: list(MODEL_DISPLAY.keys()).index(m)
                            if m in MODEL_DISPLAY else 99)
    all_conditions = [c for c in CONDITION_ORDER if any(r["condition"] == c for r in rows)]

    cw = 16
    print("\n" + "=" * (18 + cw * len(all_conditions)))
    print(f"  {title}")
    print("=" * (18 + cw * len(all_conditions)))
    hdr = f"  {'Model':<16}" + "".join(
        f"{CONDITION_DISPLAY.get(c, c)[:cw]:>{cw}}" for c in all_conditions
    )
    print(hdr)
    print("  " + "─" * (16 + cw * len(all_conditions)))
    for model in all_models:
        label = MODEL_DISPLAY.get(model, model)
        row_s = f"  {label:<16}"
        for cond in all_conditions:
            match = next((r for r in rows if r["model"] == model and r["condition"] == cond), None)
            cell  = fmt(match.get(metric) if match else None, metric)
            row_s += f"{cell:>{cw}}"
        print(row_s)
    print()


def print_all_tables(rows: list[dict]):
    _print_table(rows, "accuracy",     "ACCURACY  (% correct on GSM8K test)")
    has_flip = any(r.get("flip_rate") is not None for r in rows)
    if has_flip:
        _print_table(rows, "flip_rate",  "FLIP RATE  (wrong→right vs CCoT baseline, %)")
    has_cos = any(r.get("mean_cos_sim") is not None for r in rows)
    if has_cos:
        _print_table(rows, "mean_cos_sim", "COSINE SIMILARITY  (latent h vs v_truth)")


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(rows: list[dict], path: str):
    fields = ["model", "condition", "accuracy", "flip_rate", "mean_cos_sim", "_path"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = dict(r)
            for m in METRICS:
                v = row.get(m)
                if v is not None:
                    try:
                        row[m] = round(float(v), 6)
                    except (TypeError, ValueError):
                        pass
            w.writerow(row)
    print(f"[compare_all] CSV → {path}")


# ── LaTeX export ──────────────────────────────────────────────────────────────

def print_latex(rows: list[dict]):
    all_models = sorted({r["model"] for r in rows},
                        key=lambda m: list(MODEL_DISPLAY.keys()).index(m)
                        if m in MODEL_DISPLAY else 99)
    conditions = [c for c in CONDITION_ORDER if any(r["condition"] == c for r in rows)]
    n_cols = len(conditions)

    print("\n% ── LaTeX table ─────────────────────────────────────────────────")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Evaluation grid: Accuracy (\\%) on GSM8K test set}")
    print("\\label{tab:eval_grid}")
    print("\\begin{tabular}{l" + "r" * n_cols + "}")
    print("\\toprule")
    hdr_cells = " & ".join(f"\\textbf{{{CONDITION_DISPLAY.get(c,c)}}}" for c in conditions)
    print(f"\\textbf{{Model}} & {hdr_cells} \\\\")
    print("\\midrule")
    for model in all_models:
        label = MODEL_DISPLAY.get(model, model)
        cells = []
        for cond in conditions:
            match = next((r for r in rows if r["model"] == model and r["condition"] == cond), None)
            v     = match.get("accuracy") if match else None
            cells.append(f"{v*100:.1f}" if v is not None else "—")
        print(f"{label} & " + " & ".join(cells) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("% ────────────────────────────────────────────────────────────────\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Aggregate and display the full evaluation grid"
    )
    p.add_argument("--grid",   default=DEFAULT_GRID_ROOT,
                   help="Root of eval_grid outputs (from evaluate_baselines.py)")
    p.add_argument("--codi",   default=DEFAULT_CODI_DIR,
                   help="CODI steering results dir (from steer_inference.py)")
    p.add_argument("--outputs", default="outputs",
                   help="Root outputs dir (for TokenSkip evaluation.py results)")
    p.add_argument("--csv",    default=None,
                   help="Export results to this CSV file")
    p.add_argument("--latex",  action="store_true",
                   help="Also print a LaTeX table")
    p.add_argument("--json",   default=None,
                   help="Export combined results to this JSON file")
    args = p.parse_args()

    grid_root    = pathlib.Path(args.grid)
    codi_dir     = pathlib.Path(args.codi)
    outputs_root = pathlib.Path(args.outputs)

    rows = []

    # 1. Load eval_grid results (evaluate_baselines.py)
    if grid_root.exists():
        grid_rows = load_grid(grid_root)
        print(f"[compare_all] Loaded {len(grid_rows)} rows from {grid_root}")
        rows.extend(grid_rows)
    else:
        print(f"[compare_all] Grid root not found: {grid_root}")

    # 2. Load CODI steering results
    if codi_dir.exists():
        codi_rows = load_codi_steering(codi_dir)
        print(f"[compare_all] Loaded {len(codi_rows)} rows from CODI steering")
        rows.extend(codi_rows)
    else:
        print(f"[compare_all] CODI dir not found: {codi_dir}")

    # 3. Load TokenSkip evaluation.py results (if any)
    if outputs_root.exists():
        ts_rows = load_tokenskip_evals(outputs_root)
        if ts_rows:
            print(f"[compare_all] Loaded {len(ts_rows)} rows from TokenSkip eval outputs")
            rows.extend(ts_rows)

    if not rows:
        print("\n[compare_all] No results found.  Run evaluate_baselines.py first.")
        return

    rows = deduplicate(rows)
    print(f"[compare_all] After dedup: {len(rows)} rows\n")

    print_all_tables(rows)

    if args.latex:
        print_latex(rows)

    if args.csv:
        export_csv(rows, args.csv)

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"[compare_all] JSON → {args.json}")

    # Summary stats
    filled = {(r["model"], r["condition"]) for r in rows}
    total  = len(set(MODEL_DISPLAY)) * len(CONDITION_ORDER)
    print(f"[compare_all] Coverage: {len(filled)}/{total} model×condition cells filled")


if __name__ == "__main__":
    main()
