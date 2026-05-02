"""
eval/compare_all.py
-------------------
Aggregation Pipeline Point 3
Reads results from outputs/eval_grid and prints/exports final tables.

Conditions:
  no_cot       — Direct answer
  text_cot     — Standard CoT
  ccot         — Compressed CoT (TokenSkip)
  random_noise — Compressed CoT + random noise
  steered      — Compressed CoT + v_truth

Usage:
    python research/eval/compare_all.py
    python research/eval/compare_all.py --csv results.csv
    python research/eval/compare_all.py --latex
"""
from __future__ import annotations
import argparse, csv, json, pathlib, sys
from typing import Any

_RESEARCH_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RESEARCH_ROOT))

DEFAULT_GRID = "outputs/eval_grid"

MODEL_DISPLAY = {
    "phi2":       "Phi-2",
    "llama32_3b": "Llama-3.2-3B",
    "qwen25_3b":  "Qwen2.5-3B",
    "qwen25_1_5b":"Qwen2.5-1.5B",
    "qwen25_0_5b":"Qwen2.5-0.5B",
}
COND_DISPLAY = {
    "no_cot":       "No CoT",
    "text_cot":     "Text CoT",
    "ccot":         "CCoT (TokenSkip)",
    "random_noise": "Random Noise",
    "steered":      "CCoT + v_truth",
}
COND_ORDER = ["no_cot","text_cot","ccot","random_noise","steered"]
MODEL_ORDER = list(MODEL_DISPLAY.keys())

def load_grid(grid_root: pathlib.Path) -> list[dict]:
    rows = []
    # Load no_cot and text_cot (they are directly in model/condition/)
    for mp in sorted(grid_root.glob("*/*/metrics.json")):
        try:
            data = json.loads(mp.read_text(encoding="utf-8"))
            if data.get("condition") in ("no_cot", "text_cot"):
                model = mp.parts[len(grid_root.parts)]
                rows.append({"model": model, "condition": data["condition"],
                             "accuracy": data.get("accuracy"), "_path": str(mp)})
        except Exception:
            pass

    # Load ccot, random_noise, steered from their summary.json
    for sp in sorted(grid_root.glob("*/*/summary.json")):
        try:
            summary = json.loads(sp.read_text(encoding="utf-8"))
            model = summary.get("model")
            cond  = summary.get("condition")
            if cond not in ("ccot", "random_noise", "steered"): continue
            
            results = summary.get("results", [])
            if results:
                best = max(results, key=lambda r: r.get("accuracy") or 0.0)
                rows.append({"model": model, "condition": cond,
                             "accuracy": best.get("accuracy"),
                             "best_ratio": best.get("ratio"),
                             "actual_compression_ratio": best.get("actual_ratio"),
                             "alpha": best.get("alpha"),
                             "_path": str(sp)})
        except Exception:
            pass
    return rows

def deduplicate(rows: list[dict]) -> list[dict]:
    seen = {}
    for r in rows:
        key = (r["model"], r["condition"])
        if key not in seen or (r.get("accuracy") or 0) > (seen[key].get("accuracy") or 0):
            seen[key] = r
    return list(seen.values())

def fmt(v: Any, metric: str, default: str="—") -> str:
    if v is None: return default
    try:
        f = float(v)
        return f"{f*100:.2f}%" if metric == "accuracy" else f"{f:.4f}"
    except (TypeError, ValueError):
        return str(v)

def _print_table(rows, metric, title):
    models = [m for m in MODEL_ORDER if any(r["model"]==m for r in rows)]
    conds  = [c for c in COND_ORDER  if any(r["condition"]==c for r in rows)]
    cw     = 17
    print("\n"+"="*(18+cw*len(conds)))
    print(f"  {title}")
    print("="*(18+cw*len(conds)))
    print(f"  {'Model':<16}"+"".join(f"{COND_DISPLAY.get(c,c)[:cw]:>{cw}}" for c in conds))
    print("  "+"─"*(16+cw*len(conds)))
    for m in models:
        label = MODEL_DISPLAY.get(m, m)
        row   = f"  {label:<16}"
        for c in conds:
            match = next((r for r in rows if r["model"]==m and r["condition"]==c), None)
            row  += f"{fmt(match.get(metric) if match else None, metric):>{cw}}"
        print(row)
    print()

def export_csv(rows, path):
    fields = ["model","condition","accuracy","best_ratio","actual_compression_ratio","alpha","_path"]
    with open(path,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = dict(r)
            if row.get("accuracy") is not None:
                row["accuracy"] = round(float(row["accuracy"]), 6)
            w.writerow(row)
    print(f"[compare_all] CSV → {path}")

def print_latex(rows):
    models = [m for m in MODEL_ORDER if any(r["model"]==m for r in rows)]
    conds  = [c for c in COND_ORDER  if any(r["condition"]==c for r in rows)]
    print("\n% ── LaTeX table ─────────────────────────────────────────────")
    print("\\begin{table}[ht]\\centering")
    print("\\caption{Accuracy (\\%) on GSM8K test set}")
    print("\\label{tab:eval_grid}")
    print("\\begin{tabular}{l"+"r"*len(conds)+"}")
    print("\\toprule")
    print("\\textbf{Model} & "+" & ".join(f"\\textbf{{{COND_DISPLAY.get(c,c)}}}" for c in conds)+" \\\\")
    print("\\midrule")
    for m in models:
        label = MODEL_DISPLAY.get(m, m)
        cells = []
        for c in conds:
            match = next((r for r in rows if r["model"]==m and r["condition"]==c), None)
            v     = match.get("accuracy") if match else None
            cells.append(f"{v*100:.1f}" if v is not None else "—")
        print(f"{label} & "+" & ".join(cells)+" \\\\")
    print("\\bottomrule\\end{tabular}\\end{table}")
    print("% ────────────────────────────────────────────────────────────\n")

def main():
    p = argparse.ArgumentParser(description="Aggregation Pipeline Point 3")
    p.add_argument("--grid",   default=DEFAULT_GRID)
    p.add_argument("--csv",    default=None)
    p.add_argument("--latex",  action="store_true")
    args = p.parse_args()

    rows = []
    grid_root = pathlib.Path(args.grid)

    if grid_root.exists():
        g = load_grid(grid_root)
        print(f"[compare_all] {len(g)} rows from eval_grid")
        rows.extend(g)
    else:
        print(f"[compare_all] Grid not found: {grid_root}")
        return

    rows = deduplicate(rows)
    _print_table(rows, "accuracy", "ACCURACY (% correct on GSM8K D_test)")

    ts_rows = [r for r in rows if r.get("condition") in ("ccot","random_noise","steered") and r.get("best_ratio") is not None]
    if ts_rows:
        print("\n" + "="*60)
        print("  TOKENSKIP — Best Configuration per Model")
        print("="*60)
        print(f"  {'Model':<16} {'Condition':<18} {'Ratio':>6} {'Actual_r':>8} {'Alpha':>6} {'Accuracy':>8}")
        print("  " + "─"*68)
        for r in sorted(ts_rows, key=lambda x:(x["model"],x["condition"])):
            label = MODEL_DISPLAY.get(r["model"], r["model"])
            cdsp  = COND_DISPLAY.get(r["condition"], r["condition"])
            acc   = f"{r['accuracy']*100:.1f}%" if r.get("accuracy") is not None else "—"
            ar    = f"{r.get('actual_compression_ratio',0):.3f}"
            br    = f"{r.get('best_ratio',0):.1f}"
            al    = f"{r.get('alpha',0):.1f}" if r.get("alpha") is not None else "—"
            print(f"  {label:<16} {cdsp:<18} {br:>6} {ar:>8} {al:>6} {acc:>8}")
        print()

    if args.latex: print_latex(rows)
    if args.csv:   export_csv(rows, args.csv)

if __name__ == "__main__":
    main()
