# compare_metrics.py
import json, os, sys, glob
from pathlib import Path

def find_metrics(base):
    # matches your script’s layout
    matches = glob.glob(os.path.join(base, "*", "*", "test", "samples", "metrics.json"))
    if not matches:
        sys.exit(f"[ERR] metrics.json not found under: {base}")
    return Path(matches[0])

def load(mpath):
    j = json.loads(Path(mpath).read_text(encoding="utf-8"))
    return {
        "n": j.get("n_samples", 0),
        "acc": float(j.get("accuracy", 0.0)) * 100.0,
        "cot_len": float(j.get("avg_cot_length", 0.0)),
        "latency": j.get("sample_latency", None),
        "path": str(mpath)
    }

def pct(delta, base):
    return (100.0 * delta / base) if base else 0.0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_metrics.py <BASELINE_DIR> <COMPRESSED_DIR>")
        sys.exit(1)
    base_dir, short_dir = sys.argv[1], sys.argv[2]
    bm = load(find_metrics(base_dir))
    sm = load(find_metrics(short_dir))

    d_acc = sm["acc"] - bm["acc"]
    d_cot = sm["cot_len"] - bm["cot_len"]
    r_cot = pct(-d_cot, bm["cot_len"])  # reduction %
    d_lat = None if (bm["latency"] is None or sm["latency"] is None) else (sm["latency"] - bm["latency"])

    print("\n=== CoT Comparison (GSM8K) ===")
    print(f"Baseline metrics.json : {bm['path']}")
    print(f"Compressed metrics.json: {sm['path']}\n")
    print(f"{'Metric':<18}{'Baseline':>12}{'Compressed':>14}{'Δ (Comp - Base)':>18}{'Change %':>12}")
    print("-"*74)
    print(f"{'Accuracy (%)':<18}{bm['acc']:>12.2f}{sm['acc']:>14.2f}{d_acc:>18.2f}{'':>12}")
    print(f"{'Avg CoT length':<18}{bm['cot_len']:>12.1f}{sm['cot_len']:>14.1f}{d_cot:>18.1f}{r_cot:>11.1f}%")
    if bm["latency"] is not None and sm["latency"] is not None:
        chg_pct = pct(d_lat, bm["latency"])
        print(f"{'Sample latency(s)':<18}{bm['latency']:>12.3f}{sm['latency']:>14.3f}{d_lat:>18.3f}{chg_pct:>11.1f}%")
    print(f"{'N samples':<18}{bm['n']:>12}{sm['n']:>14}")
