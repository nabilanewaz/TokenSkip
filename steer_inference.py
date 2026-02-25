"""
steer_inference.py  —  Phase 3
-------------------------------
Inference-time intervention using the pre-computed truth vector.

Protocol equation:
    h_{t+1} = Model(h_t) + α · σ_l · v_truth / |v_truth|

Sweeps α ∈ {0.0, 0.1, 0.5, 1.0, 2.0, 5.0} (α=0.0 is the unsteered baseline).
Evaluates on datasets/gsm8k_split/test.jsonl (1,758 held-out examples).

Outputs
-------
    outputs/steering_results/alpha_{a}/metrics.json    — accuracy etc.
    outputs/steering_results/summary.json             — all α in one table
    outputs/steering_results/flip_analysis.json       — which examples flipped

Usage
-----
    # Full sweep on test split
    python steer_inference.py

    # Quick sanity check with validation split
    python steer_inference.py --eval-data datasets/gsm8k_split/validation.jsonl

    # Single alpha (useful for debugging)
    python steer_inference.py --alphas 0.0 1.0

    # Use fine-tuned CODI + fine-tuned truth vector
    python steer_inference.py --ckpt-dir outputs/codi_finetuned
                               --vector-dir outputs/truth_vector_finetuned
"""

import os, sys, json, re, pathlib, argparse, subprocess, shutil
from time import time
import torch


CODI_REPO  = "https://github.com/zhenyi4/CODI.git"
CODI_HF_ID = "zen-E/CODI-gpt2"
DEFAULT_EVAL_DATA  = "datasets/gsm8k_split/test.jsonl"
DEFAULT_VECTOR_DIR = "outputs/truth_vector"
DEFAULT_WORK_DIR   = "codi_workspace"
DEFAULT_OUT_DIR    = "outputs/steering_results"

# Protocol-specified alpha sweep
DEFAULT_ALPHAS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]


# ── Shared helpers ─────────────────────────────────────────────────────────────

def ensure_dependencies():
    pkgs = ["peft==0.15.2","datasets==3.6.0","huggingface_hub","transformers==4.52.4","accelerate==1.7.0"]
    subprocess.run([sys.executable,"-m","pip","install","--quiet","--force-reinstall","--no-deps"]+pkgs, check=True)
    subprocess.run([sys.executable,"-m","pip","install","--quiet"]+pkgs, check=True)

def clone_codi(work_dir):
    d = work_dir / "CODI"
    if not d.exists():
        subprocess.run(["git","clone",CODI_REPO,str(d)], check=True)
    return d

def get_checkpoint(work_dir, override=None):
    from huggingface_hub import snapshot_download
    if override:
        p = pathlib.Path(override); assert p.exists(); return p
    def ok(p): return (p/"model.safetensors").exists() or (p/"pytorch_model.bin").exists()
    f = work_dir/"ckpt_dir.txt"
    if f.exists():
        p = pathlib.Path(f.read_text().strip())
        if ok(p): return p
    p = pathlib.Path(snapshot_download(repo_id=CODI_HF_ID, force_download=True,
                                       ignore_patterns=["*.msgpack","*.h5","flax_model*"]))
    f.write_text(str(p)); return p

def apply_cuda_patch(code):
    if "DEVICE = " not in code:
        lines = code.split('\n'); ie = 0; d = 0
        for i,l in enumerate(lines):
            s = l.strip()
            if not s or s.startswith('#'): continue
            if d > 0:
                d += s.count('(') - s.count(')')
                if d == 0: ie = i+1
                continue
            if s.startswith('import ') or s.startswith('from '):
                d = 1 if ('(' in s and ')' not in s) else 0
                ie = i+1; continue
            break
        if ie > 0:
            pos = len('\n'.join(lines[:ie]))
            block = "\n# --- device patch ---\nimport torch as _torch\nDEVICE = 'cuda' if _torch.cuda.is_available() else 'cpu'\n# --------------------\n\n"
            code = code[:pos] + '\n' + block + code[pos:]
    code = re.sub(r"\.to\(['\"]cuda['\"]\)", ".to(DEVICE)", code)
    code = re.sub(r"\.cuda\(\)", ".to(DEVICE)", code)
    code = re.sub(r"device=['\"]cuda['\"]", "device=DEVICE", code)
    code = re.sub(r'^(\s*)device\s*=\s*["\']cuda["\']', r'\1device = DEVICE', code, flags=re.M)
    return code


# ── Build test_steered.py ──────────────────────────────────────────────────────

def build_steered_script(codi_dir, eval_data, vector_dir):
    """
    Patch test_fixed.py → test_steered.py.

    Injects the steering equation after each latent_embd assignment:
        latent_embd = latent_embd + alpha * sigma_t * v_norm_t

    alpha, v_truth_per_step, sigma_per_step loaded from env/files.
    Results per example collected and saved via env var STEER_OUT_PATH.
    """
    base_path = codi_dir / "test_fixed.py"
    if not base_path.exists():
        src = (codi_dir/"test.py").read_text(encoding="utf-8", errors="replace")
        pat = re.compile(r'^(?P<ind>\s*)pred_tokens\[b\]\.append\(next_token_ids\[b\]\.item\(\)\)\s*$', flags=re.M)
        if pat.search(src):
            src = pat.sub(lambda m: f"{m.group('ind')}next_token_ids = next_token_ids.view(-1)\n{m.group('ind')}pred_tokens[b].append(next_token_ids[b].item())", src, count=1)
        src = apply_cuda_patch(src)
        base_path.write_text(src, encoding="utf-8")

    code = base_path.read_text(encoding="utf-8", errors="replace")

    # 1. Inject globals
    PREAMBLE = f"""
# === STEERING GLOBALS (injected by steer_inference.py) ===
import os as _st_os, torch as _st_torch, json as _st_json
_ST_ALPHA        = float(_st_os.environ.get("ST_ALPHA", "0.0"))
_ST_OUT_PATH     = _st_os.environ.get("ST_OUT_PATH", "").strip()
_ST_VECTOR_DIR   = _st_os.environ.get("ST_VECTOR_DIR", "{vector_dir}").strip()
_ST_USE_PER_STEP = _st_os.environ.get("ST_USE_PER_STEP", "1") == "1"
_ST_RECORDS      = []
_ST_STEP_IDX     = 0

# Load v_truth and sigma at import time
def _load_steering_vectors():
    vd = pathlib.Path(_ST_VECTOR_DIR) if "pathlib" in dir() else __import__("pathlib").Path(_ST_VECTOR_DIR)
    v_step = _st_torch.load(str(vd / "v_truth_per_step.pt"))   # [L, D]
    sigma  = _st_torch.load(str(vd / "sigma_per_step.pt"))     # [L]
    v_global = _st_torch.load(str(vd / "v_truth.pt"))          # [D]
    # Normalise: v_hat_t = v_t / |v_t|
    v_hat_step = _st_torch.nn.functional.normalize(v_step, dim=-1)  # [L, D]
    v_hat_glob = _st_torch.nn.functional.normalize(v_global.unsqueeze(0), dim=-1).squeeze(0)  # [D]
    return v_hat_step, sigma, v_hat_glob

try:
    import pathlib as _st_pathlib
    _ST_V_HAT_STEP, _ST_SIGMA, _ST_V_HAT_GLOB = _load_steering_vectors()
    print(f"[steering] Loaded v_truth (alpha={{_ST_ALPHA}})")
except Exception as _e:
    print(f"[steering] WARNING: Could not load v_truth: {{_e}}")
    _ST_V_HAT_STEP = _ST_SIGMA = _ST_V_HAT_GLOB = None
# =========================================================
"""
    if "_ST_ALPHA" not in code:
        lines = code.split('\n'); ie = 0; d = 0
        for i,l in enumerate(lines):
            s = l.strip()
            if not s or s.startswith('#'): continue
            if d > 0:
                d += s.count('(') - s.count(')')
                if d == 0: ie = i+1
                continue
            if s.startswith('import ') or s.startswith('from '):
                d = 1 if ('(' in s and ')' not in s) else 0
                ie = i+1; continue
            break
        pos = len('\n'.join(lines[:ie]))
        code = code[:pos] + '\n' + PREAMBLE + '\n' + code[pos:]

    # 2. Inject steering after latent_embd assignments
    STEER_HOOK = """
# === STEERING INJECTION (h_t+1 = h_t + alpha * sigma_l * v/|v|) ===
if _ST_ALPHA != 0.0 and _ST_V_HAT_STEP is not None:
    _st_B, _st_L, _st_D = latent_embd.shape
    _st_dev = latent_embd.device
    if _ST_USE_PER_STEP:
        _st_v   = _ST_V_HAT_STEP.to(_st_dev)   # [L, D]
        _st_sig = _ST_SIGMA.to(_st_dev)         # [L]
        _st_delta = (_ST_ALPHA * _st_sig.unsqueeze(-1) * _st_v).unsqueeze(0)  # [1,L,D]
    else:
        _st_v     = _ST_V_HAT_GLOB.to(_st_dev)  # [D]
        _st_sig   = _ST_SIGMA.mean().to(_st_dev)
        _st_delta = (_ST_ALPHA * _st_sig * _st_v).view(1,1,-1)  # [1,1,D]
    latent_embd = latent_embd + _st_delta.expand(_st_B, -1, -1)
# =====================================================================
"""
    def inject_steer(m):
        ind = m.group("ind")
        hook = '\n'.join(ind+l for l in STEER_HOOK.strip().split('\n'))
        return m.group(0) + hook + '\n'
    lat_pat = re.compile(r'^(?P<ind>\s*)latent_embd\s*=\s*[^\n]+\n', flags=re.M)
    code, n_lat = lat_pat.subn(inject_steer, code, count=10)

    # 3. Record per-example result after decode
    RECORD_HOOK = """
# === STEER RESULT RECORD ===
if _ST_OUT_PATH:
    _st_preds  = locals().get("decoded", locals().get("pred_outputs", ""))
    _st_labels = locals().get("answers", locals().get("labels", ""))
    if isinstance(_st_preds,  str): _st_preds  = [_st_preds]
    if isinstance(_st_labels, str): _st_labels = [_st_labels]
    for _b in range(max(len(_st_preds), 1)):
        _ST_RECORDS.append({
            "pred":  str(_st_preds[_b]  if _b < len(_st_preds)  else ""),
            "gt":    str(_st_labels[_b] if _b < len(_st_labels) else ""),
        })
# ===========================
"""
    dec_pat = re.compile(
        r'^(?P<ind>\s*)\w+\s*=\s*tokenizer\.(?:decode|batch_decode)\([^\n]*skip_special_tokens\s*=\s*True[^\n]*\)\s*\n',
        flags=re.M)
    def inject_rec(m):
        ind = m.group("ind")
        hook = '\n'.join(ind+l for l in RECORD_HOOK.strip().split('\n'))
        return m.group(0) + hook + '\n'
    code, n_dec = dec_pat.subn(inject_rec, code, count=10)

    # 4. Save records at end
    SAVE_HOOK = """
# === STEER SAVE ===
if _ST_OUT_PATH and _ST_RECORDS:
    _st_json.dump(_ST_RECORDS, open(_ST_OUT_PATH, "w"))
    print(f"[steering] Saved {len(_ST_RECORDS)} records → {_ST_OUT_PATH}")
# ==================
"""
    if "_st_json.dump(_ST_RECORDS" not in code:
        accu_pat = re.compile(r'^accu\s*=\s*evaluation\s*\(', flags=re.M)
        if accu_pat.search(code):
            code = accu_pat.sub(lambda m: SAVE_HOOK + '\n' + m.group(0), code, count=1)
        else:
            code = code.rstrip() + '\n\n' + SAVE_HOOK

    # 5. Copy eval data as test set
    gsm8k_dst = codi_dir / "datasets" / "gsm8k"
    gsm8k_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(eval_data, gsm8k_dst / "test.jsonl")

    dst = codi_dir / "test_steered.py"
    dst.write_text(code, encoding="utf-8")
    print(f"[steer] Built test_steered.py (steer hooks:{n_lat}, record hooks:{n_dec})")
    return dst


# ── Run one alpha ──────────────────────────────────────────────────────────────

def extract_answer(text):
    for p in [r"####\s*([-+]?\d+\.?\d*)", r"answer is:?\s*([-+]?\d+\.?\d*)",
              r"\$\s*([-+]?\d+\.?\d*)", r"=\s*([-+]?\d+\.?\d*)\s*$"]:
        m = re.search(p, str(text), re.IGNORECASE)
        if m:
            try: return float(m.group(1))
            except: pass
    nums = re.findall(r"[-+]?\d+\.?\d*", str(text))
    try: return float(nums[-1]) if nums else None
    except: return None

def run_alpha(codi_dir, ckpt_dir, vector_dir, alpha, args, out_dir):
    records_path = out_dir / f"records_alpha_{alpha}.json"
    log_path     = out_dir / f"log_alpha_{alpha}.txt"

    env = os.environ.copy()
    env["ST_ALPHA"]        = str(alpha)
    env["ST_OUT_PATH"]     = str(records_path)
    env["ST_VECTOR_DIR"]   = str(vector_dir)
    env["ST_USE_PER_STEP"] = "1"

    cmd = [
        sys.executable, "test_steered.py",
        "--data_name","gsm8k","--model_name_or_path","gpt2",
        "--seed",str(args.seed),"--model_max_length","512",
        "--lora_r","128","--lora_alpha","32","--lora_init",
        "--batch_size",str(args.batch_size),
        "--greedy","True",
        "--num_latent","6","--use_prj","True","--prj_dim","768",
        "--prj_no_ln","False","--prj_dropout","0.0",
        "--inf_latent_iterations","6","--inf_num_iterations","1",
        "--remove_eos","True","--use_lora","True","--ckpt_dir",str(ckpt_dir),
    ]
    if args.bf16: cmd.append("--bf16")

    print(f"\n[steer] α={alpha} ─────────────────────────────────────────")
    lines = []
    proc = subprocess.Popen(cmd, cwd=str(codi_dir),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1, universal_newlines=True,
                            encoding="utf-8", errors="replace", env=env)
    for line in proc.stdout:
        print(line, end="", flush=True); lines.append(line)
    proc.wait()
    log = "".join(lines)
    log_path.write_text(log)

    # Parse accuracy from CODI stdout
    parsed_acc = None
    for pat in [r"accuracy[:\s]+([0-9]+\.[0-9]+)%?", r"accu[:\s]+([0-9]+\.[0-9]+)"]:
        m = re.search(pat, log, re.IGNORECASE|re.MULTILINE)
        if m:
            v = float(m.group(1))
            parsed_acc = v/100 if v > 1 else v
            break

    # Also compute from records if available
    records_acc = None
    if records_path.exists():
        try:
            recs = json.load(open(records_path))
            correct = sum(1 for r in recs
                          if (pa:=extract_answer(r.get("pred",""))) is not None
                          and (ga:=extract_answer(r.get("gt",""))) is not None
                          and abs(pa-ga) < 1e-4)
            records_acc = correct / len(recs) if recs else None
        except Exception as e:
            print(f"[steer] Warning: could not compute records accuracy: {e}")

    accuracy = records_acc if records_acc is not None else parsed_acc

    result = {
        "alpha": alpha,
        "accuracy": accuracy,
        "accuracy_from_stdout": parsed_acc,
        "accuracy_from_records": records_acc,
        "n_records": len(json.load(open(records_path))) if records_path.exists() else None,
        "exit_code": proc.returncode,
    }
    (out_dir / f"alpha_{alpha}" ).mkdir(parents=True, exist_ok=True)
    (out_dir / f"alpha_{alpha}" / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"[steer] α={alpha}  accuracy={accuracy:.2%}" if accuracy else f"[steer] α={alpha} accuracy=N/A")
    return result


# ── Flip analysis ─────────────────────────────────────────────────────────────

def compute_flip_analysis(results, out_dir):
    """Compare α=0 baseline against all other α to count flips."""
    baseline = next((r for r in results if r["alpha"] == 0.0), None)
    if baseline is None or not (out_dir/"records_alpha_0.0.json").exists():
        return

    try:
        base_recs = json.load(open(out_dir/"records_alpha_0.0.json"))
    except:
        return

    base_correct = []
    for r in base_recs:
        pa = extract_answer(r.get("pred",""))
        ga = extract_answer(r.get("gt",""))
        base_correct.append(pa is not None and ga is not None and abs(pa-ga)<1e-4)

    flip_summary = []
    for result in results:
        a = result["alpha"]
        if a == 0.0: continue
        rpath = out_dir / f"records_alpha_{a}.json"
        if not rpath.exists(): continue
        try: recs = json.load(open(rpath))
        except: continue
        if len(recs) != len(base_recs): continue

        n_pos_flip = 0  # wrong→right
        n_neg_flip = 0  # right→wrong
        for i, r in enumerate(recs):
            pa = extract_answer(r.get("pred",""))
            ga = extract_answer(r.get("gt",""))
            now_correct = pa is not None and ga is not None and abs(pa-ga)<1e-4
            if not base_correct[i] and now_correct:     n_pos_flip += 1
            elif base_correct[i] and not now_correct:   n_neg_flip += 1

        n_total_wrong = base_correct.count(False)
        flip_summary.append({
            "alpha": a,
            "flip_rate_pos": round(n_pos_flip / n_total_wrong, 4) if n_total_wrong else None,
            "n_wrong_to_right": n_pos_flip,
            "n_right_to_wrong": n_neg_flip,
            "net_gain": n_pos_flip - n_neg_flip,
        })

    (out_dir/"flip_analysis.json").write_text(json.dumps(flip_summary, indent=2))
    print("\n[steer] Flip Analysis:")
    print(f"  {'alpha':>6}  {'flip_rate':>10}  {'wrong→right':>12}  {'right→wrong':>12}  {'net':>6}")
    for f in flip_summary:
        fr = f"{f['flip_rate_pos']:.1%}" if f['flip_rate_pos'] is not None else "N/A"
        print(f"  {f['alpha']:>6.1f}  {fr:>10}  {f['n_wrong_to_right']:>12}  {f['n_right_to_wrong']:>12}  {f['net_gain']:>6}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Phase 3: Inference-time steering sweep over alpha")
    p.add_argument("--eval-data",   default=DEFAULT_EVAL_DATA)
    p.add_argument("--vector-dir",  default=DEFAULT_VECTOR_DIR)
    p.add_argument("--work-dir",    default=DEFAULT_WORK_DIR)
    p.add_argument("--out-dir",     default=DEFAULT_OUT_DIR)
    p.add_argument("--ckpt-dir",    default=None)
    p.add_argument("--alphas",      nargs="+", type=float, default=DEFAULT_ALPHAS,
                   help=f"Alpha values to sweep (default: {DEFAULT_ALPHAS})")
    p.add_argument("--batch-size",  type=int, default=4)
    p.add_argument("--seed",        type=int, default=11)
    p.add_argument("--bf16",        action="store_true")
    args = p.parse_args()

    work_dir   = pathlib.Path(args.work_dir).resolve()
    eval_data  = pathlib.Path(args.eval_data)
    vector_dir = pathlib.Path(args.vector_dir)
    out_dir    = pathlib.Path(args.out_dir)

    print("=" * 62)
    print("  Phase 3 — Steering Inference Sweep")
    print(f"  eval data  : {eval_data}")
    print(f"  vector dir : {vector_dir}")
    print(f"  alphas     : {args.alphas}")
    print(f"  out dir    : {out_dir}")
    print("=" * 62)

    # Validate inputs
    if not eval_data.exists():
        sys.exit(f"\n[steer] ✗ Eval data not found: {eval_data}\n  Run split_dataset.py first.\n")
    if not (vector_dir/"v_truth.pt").exists():
        sys.exit(f"\n[steer] ✗ v_truth.pt not found in {vector_dir}\n  Run extract_truth_vector.py first.\n")

    ensure_dependencies()
    codi_dir = clone_codi(work_dir)
    ckpt_dir = get_checkpoint(work_dir, args.ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_steered_script(codi_dir, eval_data, vector_dir)

    # Run sweep
    results = []
    t0 = time()
    for alpha in sorted(args.alphas):
        result = run_alpha(codi_dir, ckpt_dir, vector_dir, alpha, args, out_dir)
        results.append(result)

    elapsed = time() - t0

    # Summary table
    print("\n" + "=" * 62)
    print("  STEERING RESULTS SUMMARY")
    print("=" * 62)
    print(f"  {'alpha':>6}  {'accuracy':>10}  {'vs baseline':>12}")
    baseline_acc = next((r["accuracy"] for r in results if r["alpha"]==0.0), None)
    for r in results:
        acc  = r["accuracy"]
        diff = (acc - baseline_acc) if (acc and baseline_acc) else None
        diff_str = f"{diff:+.2%}" if diff is not None else "N/A"
        acc_str  = f"{acc:.2%}" if acc else "N/A"
        marker   = " ← baseline" if r["alpha"] == 0.0 else ""
        print(f"  {r['alpha']:>6.1f}  {acc_str:>10}  {diff_str:>12}{marker}")
    print("=" * 62)

    # Flip analysis
    compute_flip_analysis(results, out_dir)

    # Save summary
    summary = {
        "eval_data": str(eval_data),
        "vector_dir": str(vector_dir),
        "elapsed_seconds": elapsed,
        "results": results,
    }
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[steer] Summary → {out_dir}/summary.json")
    print(f"[steer] Flip analysis → {out_dir}/flip_analysis.json\n")

if __name__ == "__main__":
    main()