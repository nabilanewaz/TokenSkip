"""eval/evaluate_baselines.py — Aggregation Pipeline Point 1."""
from __future__ import annotations
import argparse, json, os, pathlib, subprocess, sys
from time import time

_RESEARCH_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RESEARCH_ROOT))
from utils.io import load_config

_CFG  = load_config()
_DS   = _CFG.get("dataset", {})
_P2   = _CFG.get("phase2", {})
_EVAL = _CFG.get("eval", {})

MODEL_PATHS = {
    "phi2":       os.environ.get("PHI2_PATH",      "microsoft/phi-2"),
    "llama32_3b": os.environ.get("LLAMA32_PATH",   "meta-llama/Llama-3.2-3B"),
    "qwen25_3b":  os.environ.get("QWEN25_3B_PATH", "Qwen/Qwen2.5-3B"),
    "qwen25_1_5b":os.environ.get("QWEN25_15_PATH", "Qwen/Qwen2.5-1.5B"),
    "qwen25_0_5b":os.environ.get("QWEN25_05_PATH", "Qwen/Qwen2.5-0.5B"),
}
ALL_MODELS     = list(MODEL_PATHS.keys())
ALL_CONDITIONS = ["no_cot", "text_cot", "ccot", "random_noise", "steered"]
STEER_ALPHAS   = _P2.get("alpha_sweep", [0.0,0.1,0.5,1.0,2.0,5.0,10.0,20.0,50.0])
LAYER_FRAC     = _P2.get("intervention_layer_frac", 0.75)
_split_dir     = pathlib.Path(_DS.get("out_dir","datasets/gsm8k_split"))
DEFAULT_EVAL   = str(_split_dir / "test.jsonl")
DEFAULT_OUT    = "outputs/eval_grid"
DEFAULT_VEC    = "outputs/phase2_truth_vector"
STEER_SCRIPT   = str(_RESEARCH_ROOT / "phase3" / "steer.py")

def run_cmd(cmd, label=""):
    print(f"\n{'─'*60}\n  {label}\n{'─'*60}")
    lines = []
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         bufsize=1, universal_newlines=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        print(line, end="", flush=True); lines.append(line)
    p.wait()
    return p.returncode, "".join(lines)

def run_condition(model_tag, model_path, eval_data, vector_dir, out_dir, condition, alphas=None):
    cmd = [sys.executable, STEER_SCRIPT,
           "--model-path", model_path, "--model-type", model_tag,
           "--eval-data",  eval_data,
           "--vector-dir", str(vector_dir), "--out-dir", str(out_dir),
           "--condition",  condition,  "--layer-frac", str(LAYER_FRAC), "--seed","42"]
    if condition in ("steered", "random_noise") and alphas:
        cmd += ["--alphas"] + [str(a) for a in alphas]
    rc, _ = run_cmd(cmd, label=f"[{condition}] {model_tag}")
    return {"condition":condition,"model":model_tag,"exit_code":rc,"out_dir":str(out_dir)}

def collect_results(out_root):
    rows = []
    for mp in sorted(pathlib.Path(out_root).glob("*/*/metrics.json")):
        try:
            data  = json.loads(mp.read_text(encoding="utf-8"))
            parts = mp.parts; base = len(pathlib.Path(out_root).parts)
            data.setdefault("model",     parts[base]   if len(parts)>base   else "?")
            data.setdefault("condition", parts[base+1] if len(parts)>base+1 else "?")
            rows.append(data)
        except Exception as e:
            pass
            
    for sp in sorted(pathlib.Path(out_root).glob("*/*/summary.json")):
        try:
            summary = json.loads(sp.read_text(encoding="utf-8"))
            model   = summary.get("model")
            cond    = summary.get("condition")
            results = summary.get("results", [])
            if results:
                best = max(results, key=lambda r: r.get("accuracy") or 0.0)
                rows.append({"model":model,"condition":cond,
                             "accuracy":best.get("accuracy"),
                             "best_ratio":best.get("ratio"),
                             "actual_compression_ratio":best.get("actual_ratio")})
        except Exception as e:
            pass
    return rows

def deduplicate(rows):
    seen = {}
    for r in rows:
        key = (r["model"], r["condition"])
        if key not in seen or (r.get("accuracy") or 0) > (seen[key].get("accuracy") or 0):
            seen[key] = r
    return list(seen.values())

def print_table(rows):
    models = sorted({r.get("model","?") for r in rows}); cw = 14
    print("\n"+"="*80+"\n  EVALUATION GRID — Accuracy (%)\n"+"="*80)
    print(f"  {'Model':<16}"+"".join(f"{c[:cw]:>{cw}}" for c in ALL_CONDITIONS))
    print("  "+"─"*(16+cw*len(ALL_CONDITIONS)))
    for m in models:
        row = f"  {m:<16}"
        for c in ALL_CONDITIONS:
            match = next((r for r in rows if r.get("model")==m and r.get("condition")==c), None)
            acc   = match.get("accuracy") if match else None
            row  += f"{f'{acc*100:.2f}%' if isinstance(acc,float) else '—':>{cw}}"
        print(row)
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-data",    default=DEFAULT_EVAL)
    parser.add_argument("--out-root",     default=DEFAULT_OUT)
    parser.add_argument("--vector-root",  default=DEFAULT_VEC)
    parser.add_argument("--models",       nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--conditions",   nargs="+", default=ALL_CONDITIONS, choices=ALL_CONDITIONS)
    parser.add_argument("--model-paths",  nargs="+", metavar="TAG=PATH")
    parser.add_argument("--alphas",       nargs="+", type=float, default=STEER_ALPHAS)
    parser.add_argument("--results-only", action="store_true")
    args = parser.parse_args()

    if args.model_paths:
        for kv in args.model_paths:
            tag, _, path = kv.partition("=")
            if tag in MODEL_PATHS: MODEL_PATHS[tag] = path

    out_root   = pathlib.Path(args.out_root)
    vec_root   = pathlib.Path(args.vector_root)

    if args.results_only:
        print_table(deduplicate(collect_results(out_root))); return

    for p, name in [(pathlib.Path(args.eval_data),"eval-data")]:
        if not p.exists():
            sys.exit(f"[eval] ✗ {name} not found: {p}\n  Run: python research/data/split_dataset.py --full")

    print(f"\n{'='*70}\n  EVALUATION GRID\n  Models: {args.models}\n  Conditions: {args.conditions}")
    print(f"  Eval: {args.eval_data}\n{'='*70}")

    all_results = []; t_start = time()
    for model_tag in args.models:
        model_path = MODEL_PATHS.get(model_tag)
        vdir = vec_root
        for condition in args.conditions:
            out_dir = out_root
            try:
                result = run_condition(model_tag, model_path, args.eval_data, vdir, out_dir, condition, args.alphas)
            except KeyboardInterrupt:
                print("\n[eval] Interrupted."); break
            except Exception as e:
                result = {"condition":condition,"model":model_tag,"error":str(e),"exit_code":-1}
            all_results.append(result)

    print(f"\n[eval] Done in {(time()-t_start)/60:.1f} min\n")
    print_table(deduplicate(collect_results(out_root)))
    print(f"\n  Next: python research/eval/compare_all.py\n")

if __name__ == "__main__":
    main()
