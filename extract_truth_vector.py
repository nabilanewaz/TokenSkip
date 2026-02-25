"""
extract_truth_vector.py  —  Phase 2
------------------------------------
Implements the "Truth Vector" extraction from the research protocol:

    v_truth = mean(H+) - mean(H-)

where H+ = latent trajectories that produced a CORRECT final answer,
      H- = latent trajectories that produced a WRONG final answer.

Uses datasets/gsm8k_split/steer_train.jsonl (704 examples = Dsteer).

Protocol note
-------------
The protocol specifies N runs per question at temperature T=1.0 to build a
diverse set of positive/negative traces. With CODI-GPT2 on CPU that would be
very slow. We offer two modes controlled by --n-samples:
    --n-samples 1   (default, fast)  : greedy, one pass — single trace per question.
    --n-samples N   (N>1, recommended): stochastic, N passes — richer H+/H-.

Outputs
-------
    outputs/truth_vector/v_truth.pt           global vector [D]
    outputs/truth_vector/v_truth_per_step.pt  per-step [k, D]
    outputs/truth_vector/sigma_per_step.pt    activation std [k]
    outputs/truth_vector/stats.json           metadata + balance

Usage
-----
    python extract_truth_vector.py
    python extract_truth_vector.py --n-samples 5
    python extract_truth_vector.py --ckpt-dir outputs/codi_finetuned
    python extract_truth_vector.py --skip-dump   # recompute vector from existing dump
"""

import os, sys, json, re, pathlib, argparse, subprocess, shutil
from time import time
import torch


CODI_REPO  = "https://github.com/zhenyi4/CODI.git"
CODI_HF_ID = "zen-E/CODI-gpt2"
DEFAULT_STEER_DATA = "datasets/gsm8k_split/steer_train.jsonl"
DEFAULT_WORK_DIR   = "codi_workspace"
DEFAULT_OUT_DIR    = "outputs/truth_vector"


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
        p = pathlib.Path(override).resolve()  # Convert to absolute path
        assert p.exists(), f"--ckpt-dir not found: {p}"
        return p
    def ok(p): return (p/"model.safetensors").exists() or (p/"pytorch_model.bin").exists()
    f = work_dir/"ckpt_dir.txt"
    if f.exists():
        p = pathlib.Path(f.read_text().strip()).resolve()
        if ok(p): return p
    p = pathlib.Path(snapshot_download(repo_id=CODI_HF_ID, force_download=True,
                                       ignore_patterns=["*.msgpack","*.h5","flax_model*"]))
    f.write_text(str(p)); return p

def apply_cuda_patch(code):
    if "DEVICE = " not in code:
        lines = code.split('\n'); import_end = 0; depth = 0
        for i,line in enumerate(lines):
            s = line.strip()
            if not s or s.startswith('#'): continue
            if depth > 0:
                depth += s.count('(') - s.count(')')
                if depth == 0: import_end = i+1
                continue
            if s.startswith('import ') or s.startswith('from '):
                depth = 1 if ('(' in s and ')' not in s) else 0
                import_end = i+1; continue
            break
        if import_end > 0:
            pos = len('\n'.join(lines[:import_end]))
            block = "\n# --- device patch ---\nimport torch as _torch\nDEVICE = 'cuda' if _torch.cuda.is_available() else 'cpu'\n# --------------------\n\n"
            code = code[:pos] + '\n' + block + code[pos:]
    code = re.sub(r"\.to\(['\"]cuda['\"]\)", ".to(DEVICE)", code)
    code = re.sub(r"\.cuda\(\)", ".to(DEVICE)", code)
    code = re.sub(r"device=['\"]cuda['\"]", "device=DEVICE", code)
    code = re.sub(r'^(\s*)device\s*=\s*["\']cuda["\']', r'\1device = DEVICE', code, flags=re.M)
    return code


# ── Build test_dump_tv.py ──────────────────────────────────────────────────────

def build_dump_script(codi_dir, steer_data):
    # Ensure test_fixed.py exists (apply patches if needed)
    base_path = codi_dir / "test_fixed.py"
    if not base_path.exists():
        src = (codi_dir/"test.py").read_text(encoding="utf-8", errors="replace")
        pat = re.compile(r'^(?P<ind>\s*)pred_tokens\[b\]\.append\(next_token_ids\[b\]\.item\(\)\)\s*$', flags=re.M)
        if pat.search(src):
            src = pat.sub(lambda m: f"{m.group('ind')}next_token_ids = next_token_ids.view(-1)\n{m.group('ind')}pred_tokens[b].append(next_token_ids[b].item())", src, count=1)
        src = apply_cuda_patch(src)
        base_path.write_text(src, encoding="utf-8")

    code = base_path.read_text(encoding="utf-8", errors="replace")

    # 1. Inject dump globals after imports
    PREAMBLE = """
# === TV DUMP GLOBALS ===
import os as _tv_os, torch as _tv_torch
_TV_DUMP_PATH = _tv_os.environ.get("TV_DUMP_PATH","").strip()
_TV_RECORDS   = []
_TV_LAST_LAT  = None
# =======================
"""
    if "_TV_RECORDS" not in code:
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

    # 2. Capture latent_embd
    CAPTURE = "_TV_LAST_LAT = latent_embd.detach().cpu()\n"
    def inject_cap(m):
        ind = m.group("ind")
        return m.group(0) + ind + CAPTURE
    lat_pat = re.compile(r'^(?P<ind>\s*)latent_embd\s*=\s*[^\n]+\n', flags=re.M)
    code, n_lat = lat_pat.subn(inject_cap, code, count=10)

    # 3. Record hook after tokenizer.decode
    RECORD_HOOK = """
# === TV RECORD ===
if _TV_DUMP_PATH and (_TV_LAST_LAT is not None):
    _tv_lat = _TV_LAST_LAT
    # Scan for the decoded output variable (just assigned above)
    _locals = locals()
    _preds = None
    for _var in ["decoded", "pred_outputs", "outputs", "predictions", "pred_output"]:
        if _var in _locals and _locals[_var]: _preds = _locals[_var]; break
    if _preds is None:  # Fallback: find the variable that was just assigned by tokenizer.decode
        _preds = list(_locals.values())[-1] if _locals else ""
    # Ground truth: check multiple possible names
    _gts = None
    for _var in ["answers", "labels", "gt_answers", "ground_truths", "answer"]:
        if _var in _locals and _locals[_var]: _gts = _locals[_var]; break
    if _gts is None: _gts = ""
    # Normalize to lists
    if isinstance(_preds, str): _preds = [_preds]
    if isinstance(_gts, str): _gts = [_gts]
    if _tv_torch.is_tensor(_tv_lat) and _tv_lat.dim() == 3:
        B = _tv_lat.size(0)
    else:
        B = 1; _tv_lat = _tv_lat.unsqueeze(0) if _tv_lat.dim()==2 else _tv_lat
    for _b in range(min(B, max(len(_preds), 1))):
        _TV_RECORDS.append({
            "latent":    _tv_lat[_b].float() if _tv_lat.dim()==3 else _tv_lat.float(),
            "pred_text": str(_preds[_b] if _b < len(_preds) else ""),
            "gt_text":   str(_gts[_b]   if _b < len(_gts)   else ""),
        })
# =================
"""
    dec_pat = re.compile(
        r'^(?P<ind>\s*)\w+\s*=\s*tokenizer\.(?:decode|batch_decode)\([^\n]*skip_special_tokens\s*=\s*True[^\n]*\)\s*\n',
        flags=re.M)
    def inject_rec(m):
        ind = m.group("ind")
        hook = '\n'.join(ind+l for l in RECORD_HOOK.strip().split('\n'))
        return m.group(0) + hook + '\n'
    code, n_dec = dec_pat.subn(inject_rec, code, count=10)

    # 4. Save hook at end
    SAVE_HOOK = """
# === TV SAVE ===
if _TV_DUMP_PATH and _TV_RECORDS:
    _tv_torch.save(_TV_RECORDS, _TV_DUMP_PATH)
    print(f"[tv_dump] Saved {len(_TV_RECORDS)} records → {_TV_DUMP_PATH}")
# ===============
"""
    if "_tv_torch.save(_TV_RECORDS" not in code:
        accu_pat = re.compile(r'^accu\s*=\s*evaluation\s*\(', flags=re.M)
        if accu_pat.search(code):
            code = accu_pat.sub(lambda m: SAVE_HOOK + '\n' + m.group(0), code, count=1)
        else:
            code = code.rstrip() + '\n\n' + SAVE_HOOK

    # 5. Copy steer_train.jsonl as the test set CODI will read
    gsm8k_dst = codi_dir / "datasets" / "gsm8k"
    gsm8k_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(steer_data, gsm8k_dst / "test.jsonl")

    dst = codi_dir / "test_dump_tv.py"
    dst.write_text(code, encoding="utf-8")
    print(f"[extract_tv] Built test_dump_tv.py (lat hooks:{n_lat}, dec hooks:{n_dec})")
    return dst


# ── Run dump ───────────────────────────────────────────────────────────────────

def run_dump(codi_dir, ckpt_dir, args, dump_path):
    env = os.environ.copy()
    env["TV_DUMP_PATH"] = str(dump_path.resolve())  # Use absolute path
    env["TV_N_SAMPLES"] = str(args.n_samples)

    use_greedy = "True" if args.n_samples == 1 else "False"
    cmd = [
        sys.executable, "test_dump_tv.py",
        "--data_name","gsm8k","--model_name_or_path","gpt2",
        "--seed",str(args.seed),"--model_max_length","512",
        "--lora_r","128","--lora_alpha","32","--lora_init",
        "--batch_size",str(args.batch_size),
        "--greedy",use_greedy,
        "--num_latent","6","--use_prj","True","--prj_dim","768",
        "--prj_no_ln","False","--prj_dropout","0.0",
        "--inf_latent_iterations","6","--inf_num_iterations","1",
        "--remove_eos","True","--use_lora","True","--ckpt_dir",str(ckpt_dir),
    ]
    if args.bf16: cmd.append("--bf16")

    print(f"\n[extract_tv] $ {' '.join(cmd)}\n")
    lines = []
    proc = subprocess.Popen(cmd, cwd=str(codi_dir),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1, universal_newlines=True,
                            encoding="utf-8", errors="replace", env=env)
    for line in proc.stdout:
        print(line, end="", flush=True); lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines)


# ── Compute v_truth ────────────────────────────────────────────────────────────

def extract_answer(text):
    """Extract numeric answer from model output. Handles multiple formats."""
    text = str(text).strip()
    if not text:
        return None
    
    # Try structured formats first
    for p in [r"####\s*([-+]?\d+\.?\d*)", r"answer is:?\s*([-+]?\d+\.?\d*)",
              r"\$\s*([-+]?\d+\.?\d*)", r"=\s*([-+]?\d+\.?\d*)\s*$",
              r"answer:?\s*([-+]?\d+\.?\d*)"]:  # Added "answer:" format
        m = re.search(p, text, re.IGNORECASE)
        if m:
            try: return float(m.group(1))
            except: pass
    
    # If text is a pure number (common in CODI short outputs)
    try:
        return float(text.strip())
    except:
        pass
    
    # Fallback: extract all numbers and take the last one
    nums = re.findall(r"[-+]?\d+\.?\d*", text)
    try: return float(nums[-1]) if nums else None
    except: return None

def compute_truth_vector(dump_path, out_dir):
    print(f"\n[extract_tv] Computing v_truth from {dump_path}...")
    records = torch.load(dump_path)
    print(f"[extract_tv] {len(records)} records")

    pos, neg = [], []
    no_pred = no_gt = 0
    sample_preds = []  # Collect samples for debugging

    for i, r in enumerate(records):
        lat = r.get("latent")
        if lat is None or not torch.is_tensor(lat): continue
        if lat.dim() == 3 and lat.size(0) == 1: lat = lat.squeeze(0)
        if lat.dim() != 2: continue

        pred_text = r.get("pred_text", "")
        gt_text = r.get("gt_text", "")
        
        # Collect first 5 samples for debugging
        if len(sample_preds) < 5:
            sample_preds.append({"pred": pred_text, "gt": gt_text})

        pa = extract_answer(pred_text)
        ga = extract_answer(gt_text)
        if pa is None: no_pred += 1; continue
        if ga is None: no_gt   += 1; continue

        (pos if abs(pa-ga)<1e-4 else neg).append(lat.float())

    # Show sample predictions for debugging
    if sample_preds:
        print("\n[extract_tv] Sample predictions (first 5):")
        for idx, s in enumerate(sample_preds):
            pred_ans = extract_answer(s["pred"])
            gt_ans = extract_answer(s["gt"])
            print(f"  [{idx+1}] pred='{s['pred'][:100]}' → {pred_ans}")
            print(f"      gt  ='{s['gt'][:100]}' → {gt_ans}")

    n_pos, n_neg = len(pos), len(neg)
    print(f"\n[extract_tv] H+:{n_pos}  H-:{n_neg}  no_pred:{no_pred}  no_gt:{no_gt}")

    if n_pos == 0 or n_neg == 0:
        print("[extract_tv] ✗ Need both positive and negative samples.")
        print("  If n_pos=0: the decode hook isn't capturing text — check test_dump_tv.py")
        print("  If n_neg=0: the model gets everything right (unlikely on steer set)")
        sys.exit(1)

    pos_stack = torch.stack(pos)  # [N+, L, D]
    neg_stack = torch.stack(neg)  # [N-, L, D]
    L, D = pos_stack.shape[1], pos_stack.shape[2]

    # Difference-of-means  (protocol eq.)
    v_per_step = pos_stack.mean(0) - neg_stack.mean(0)  # [L, D]
    v_global   = v_per_step.mean(0)                      # [D]

    # σ_l (activation std per step, for steering equation α·σ_l·v/|v|)
    all_lat    = torch.cat([pos_stack, neg_stack], 0)    # [N, L, D]
    sigma      = all_lat.std(0).mean(-1)                 # [L]

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(v_global,   out_dir/"v_truth.pt")
    torch.save(v_per_step, out_dir/"v_truth_per_step.pt")
    torch.save(sigma,      out_dir/"sigma_per_step.pt")

    stats = {
        "n_pos": n_pos, "n_neg": n_neg,
        "n_no_pred": no_pred, "n_no_gt": no_gt,
        "balance_ratio": round(n_pos/(n_pos+n_neg), 4),
        "L": L, "D": D,
        "v_truth_global_norm": float(v_global.norm()),
        "v_truth_per_step_norms": v_per_step.norm(dim=-1).tolist(),
        "sigma_per_step": sigma.tolist(),
    }
    (out_dir/"stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\n[extract_tv] ✓ v_truth computed")
    print(f"   global norm     : {stats['v_truth_global_norm']:.4f}")
    print(f"   per-step norms  : {[f'{v:.3f}' for v in stats['v_truth_per_step_norms']]}")
    print(f"   balance (H+/all): {stats['balance_ratio']:.1%}")
    print(f"\n   Saved to {out_dir}/")
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Phase 2: Extract truth vector from CODI latents")
    p.add_argument("--steer-data", default=DEFAULT_STEER_DATA)
    p.add_argument("--work-dir",   default=DEFAULT_WORK_DIR)
    p.add_argument("--out-dir",    default=DEFAULT_OUT_DIR)
    p.add_argument("--ckpt-dir",   default=None,
                   help="Use fine-tuned checkpoint; default: pretrained zen-E/CODI-gpt2")
    p.add_argument("--n-samples",  type=int, default=1,
                   help="Traces per question (1=greedy/fast, >1=stochastic/protocol-exact)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed",       type=int, default=11)
    p.add_argument("--bf16",       action="store_true")
    p.add_argument("--skip-dump",  action="store_true",
                   help="Skip CODI run; re-compute vector from existing dump file")
    args = p.parse_args()

    work_dir   = pathlib.Path(args.work_dir).resolve()
    steer_data = pathlib.Path(args.steer_data).resolve()  # Convert to absolute path
    out_dir    = pathlib.Path(args.out_dir).resolve()  # Convert to absolute path
    dump_path  = out_dir / "latent_dump.pt"

    print("=" * 62)
    print("  Phase 2 — Truth Vector Extraction")
    print(f"  steer data : {steer_data}")
    print(f"  n_samples  : {args.n_samples}  "
          f"({'greedy 1-pass' if args.n_samples==1 else 'stochastic multi-pass'})")
    print(f"  out dir    : {out_dir}")
    print("=" * 62)

    if not steer_data.exists():
        sys.exit(f"\n[extract_tv] ✗ Not found: {steer_data}\n  Run split_dataset.py first.\n")

    if not args.skip_dump:
        ensure_dependencies()
        codi_dir = clone_codi(work_dir)
        ckpt_dir = get_checkpoint(work_dir, args.ckpt_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        build_dump_script(codi_dir, steer_data)
        rc, log = run_dump(codi_dir, ckpt_dir, args, dump_path)
        (out_dir/"dump_run.log").write_text(log)
        if rc != 0:
            print(f"[extract_tv] CODI exited {rc} — checking if dump was saved anyway...")
        if not dump_path.exists():
            print(f"[extract_tv] ✗ Dump file missing: {dump_path}")
            print("  The decode hook may need adjustment — inspect codi_workspace/CODI/test_dump_tv.py")
            sys.exit(1)
    else:
        if not dump_path.exists():
            sys.exit(f"[extract_tv] --skip-dump set but {dump_path} not found.")
        print(f"[extract_tv] Using existing dump: {dump_path}")

    compute_truth_vector(dump_path, out_dir)
    print("\n[extract_tv] Next: python steer_inference.py\n")

if __name__ == "__main__":
    main()