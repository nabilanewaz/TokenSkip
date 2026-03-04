"""
phase2_extract_vector.py  —  Phase 2: Truth Vector Extraction
==============================================================
Computes the "Truth Direction" in the CODI latent space:

    v_truth^t = mean(H+^t) - mean(H-^t)

where:
    H+^t = latent states at reasoning step t from CORRECT traces
    H-^t = latent states at reasoning step t from WRONG traces

This implements the Difference-of-Means method from the protocol.
The computation is per-step (yielding v_truth_per_step [L, D]) and
globally averaged (v_truth [D]).

Also computes σ_l (activation std at layer l), used in Phase 3's
steering equation: h_{t+1} = Model(h_t) + α · σ_l · v̂

Dataset: datasets/gsm8k_split/steer_train.jsonl (Dsteer)

Critical rule: This set must NEVER be the same as validation/test.
              Only steer_train.jsonl is used here.

Usage
-----
    # Fast (greedy, 1 trace/question) — quick iteration
    python phase2_extract_vector.py

    # Richer (stochastic, N traces/question) — protocol-exact
    python phase2_extract_vector.py --n-samples 5

    # With fine-tuned Phase 1 checkpoint
    python phase2_extract_vector.py --ckpt-dir outputs/phase1_checkpoint

    # Recompute vector from a saved dump without re-running model
    python phase2_extract_vector.py --skip-inference

Outputs
-------
    outputs/phase2_truth_vector/v_truth.pt            global vector  [D]
    outputs/phase2_truth_vector/v_truth_per_step.pt   per-step       [L, D]
    outputs/phase2_truth_vector/sigma_per_step.pt     activation std [L]
    outputs/phase2_truth_vector/latent_dump.pt        raw dump (all traces)
    outputs/phase2_truth_vector/stats.json            balance + norms
"""

import os
import sys
import re
import json
import pathlib
import argparse
import subprocess
import shutil
from time import time

import torch


# ── Configuration ──────────────────────────────────────────────────────────────
CODI_HF_ID         = "zen-E/CODI-gpt2"
DEFAULT_STEER_DATA = "datasets/gsm8k_split/steer_train.jsonl"
DEFAULT_CKPT_DIR   = None                      # None → use pretrained zen-E/CODI-gpt2
DEFAULT_OUT_DIR    = "outputs/phase2_truth_vector"
DEFAULT_WORK_DIR   = "codi_workspace"


# ── Shared helpers ─────────────────────────────────────────────────────────────

def ensure_dependencies():
    pkgs = [
        "peft==0.15.2", "datasets==3.6.0", "huggingface_hub",
        "transformers==4.52.4", "accelerate==1.7.0", "safetensors",
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--force-reinstall", "--no-deps"] + pkgs, check=True
    )
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet"] + pkgs, check=True)


def get_checkpoint(work_dir: pathlib.Path, override=None) -> pathlib.Path:
    from huggingface_hub import snapshot_download
    if override:
        p = pathlib.Path(override)
        assert p.exists(), f"--ckpt-dir not found: {p}"
        return p
    ckpt_file = work_dir / "ckpt_dir.txt"
    if ckpt_file.exists():
        p = pathlib.Path(ckpt_file.read_text().strip())
        if (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists():
            return p
    p = pathlib.Path(snapshot_download(
        repo_id=CODI_HF_ID, force_download=True,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    ))
    ckpt_file.write_text(str(p))
    return p


def load_jsonl(path) -> list:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ── CODI model loading ─────────────────────────────────────────────────────────

def load_codi_model(ckpt_dir: pathlib.Path, bundle_dir: pathlib.Path, bf16: bool):
    """
    Load the CODI model directly from codi_bundle/src/model.py.
    Returns (model, tokenizer) on the appropriate device.
    """
    # Ensure codi_bundle is on sys.path
    bundle_str = str(bundle_dir.resolve())
    if bundle_str not in sys.path:
        sys.path.insert(0, bundle_str)

    from src.model import CODI, ModelArguments, TrainingArguments
    from peft import LoraConfig, TaskType
    import transformers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Phase 2] Loading CODI model on {device.upper()}...")

    # ModelArguments — match Phase 1 training config exactly
    model_args = ModelArguments(
        model_name_or_path="gpt2",
        lora_r=128,
        lora_alpha=32,
        full_precision=True,
        train=False,          # inference mode
        lora_init=True,
        ckpt_dir=str(ckpt_dir),
    )

    # TrainingArguments — minimal, just enough for model init
    training_args = transformers.TrainingArguments(
        output_dir="/tmp/codi_phase2",
        no_cuda=(device == "cpu"),
        bf16=(bf16 and device == "cuda"),
        use_cpu=(device == "cpu"),
    )
    # Inject CODI-specific fields that TrainingArguments doesn't have by default
    training_args.num_latent           = 6
    training_args.use_lora             = True
    training_args.use_prj              = True
    training_args.prj_dim              = 768
    training_args.prj_dropout          = 0.0
    training_args.prj_no_ln            = False
    training_args.distill_loss_div_std = False
    training_args.distill_loss_type    = "smooth_l1"
    training_args.distill_loss_factor  = 1.0
    training_args.ref_loss_factor      = 1.0
    training_args.inf_latent_iterations= 6
    training_args.inf_num_iterations   = 1
    training_args.remove_eos           = True
    training_args.print_ref_model_stats= False
    training_args.include_last_cot     = False
    training_args.fix_attn_mask        = False
    training_args.log_full             = False
    training_args.print_loss           = False
    training_args.max_token_num        = 1000
    training_args.restore_from        = ""
    training_args.expt_name           = "phase2"
    training_args.greedy              = True
    training_args.exp_mode            = False
    training_args.exp_data_num        = 10000

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=128,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"],
        init_lora_weights=True,
    )

    model = CODI(model_args, training_args, lora_config)
    if device == "cpu":
        model = model.float()
    model = model.to(device)
    model.eval()

    import transformers as _tf
    tokenizer = _tf.AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print(f"[Phase 2] ✓ Model loaded. Hidden dim D={model.dim}, num_latent={model.num_latent}")
    return model, tokenizer, device


# ── Latent trajectory collection ──────────────────────────────────────────────

def extract_answer_number(text: str):
    """Extract numeric answer from model output or ground truth."""
    text = str(text).strip().replace(",", "")
    if "####" in text:
        text = text.split("####")[-1].strip()
    for pat in [
        r"answer is:?\s*([-+]?\d+\.?\d*)",
        r"####\s*([-+]?\d+\.?\d*)",
        r"\$\\boxed\{([-+]?\d+\.?\d*)\}",
        r"=\s*([-+]?\d+\.?\d*)\s*$",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try: return float(m.group(1))
            except: pass
    # Fallback: last number in text
    nums = re.findall(r"[-+]?\d+\.?\d*", text)
    if nums:
        try: return float(nums[-1])
        except: pass
    # Direct parse (for clean ground truth like "42")
    try: return float(text.strip())
    except: return None


@torch.no_grad()
def collect_latent_trace(model, tokenizer, question: str, device: str):
    """
    Run one forward pass through the CODI model and collect:
        - The sequence of latent embeddings h_1 … h_k  (the reasoning trace)
        - The decoded final answer text

    Returns: (latent_stack [k, D], pred_text str)
    """
    # Build prompt in the CODI iCoT format
    prompt = question.strip()
    if not prompt.endswith("?"):
        prompt = prompt + "\nAnswer the above question. "
    else:
        prompt = prompt + "\nAnswer the above question. "

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    input_ids = enc["input_ids"].to(device)

    # Run the CODI encoder to get initial state
    outputs = model.codi(
        input_ids=input_ids,
        use_cache=True,
        output_hidden_states=True,
    )
    past_kv     = outputs.past_key_values
    latent_embd = outputs.hidden_states[-1][:, -1:, :]  # [1, 1, D]

    collected_latents = []
    target_dtype = latent_embd.dtype

    # Collect all k latent reasoning steps
    for step in range(model.num_latent):
        if model.use_prj:
            prj_dtype = next(model.prj.parameters()).dtype
            if latent_embd.dtype != prj_dtype:
                latent_embd = latent_embd.to(prj_dtype)
            latent_embd = model.prj(latent_embd)
            if latent_embd.dtype != target_dtype:
                latent_embd = latent_embd.to(target_dtype)

        # Store this step's latent (squeeze batch dim)
        collected_latents.append(latent_embd[0, 0].float().cpu())

        out = model.codi(
            inputs_embeds=latent_embd,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_kv,
        )
        past_kv     = out.past_key_values
        latent_embd = out.hidden_states[-1][:, -1:, :]

    # Decode the final answer from the last latent state
    answer_prompt = tokenizer(
        "The answer is:", return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)
    answer_embd = model.get_embd(model.codi, model.model_name)(answer_prompt)

    gen_out = model.codi.generate(
        inputs_embeds=answer_embd,
        past_key_values=past_kv,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    pred_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)

    latent_stack = torch.stack(collected_latents, dim=0)  # [k, D]
    return latent_stack, pred_text


def collect_all_traces(model, tokenizer, steer_data: list, device: str, n_samples: int):
    """
    For each question in steer_data, run n_samples forward passes.
    Returns lists of (latent_stack, pred_text, gt_text) tuples.
    """
    records = []
    n_total = len(steer_data)

    for i, item in enumerate(steer_data):
        question = item.get("question", "")
        gt_raw   = item.get("answer", "")
        gt_text  = gt_raw.split("####")[-1].strip() if "####" in gt_raw else gt_raw

        for sample_idx in range(n_samples):
            try:
                lat, pred = collect_latent_trace(model, tokenizer, question, device)
                records.append({
                    "latent":    lat,
                    "pred_text": pred,
                    "gt_text":   gt_text,
                    "question":  question,
                    "q_idx":     i,
                })
            except Exception as e:
                print(f"[Phase 2] ⚠  q={i} sample={sample_idx}: {e}")

        if (i + 1) % 50 == 0 or (i + 1) == n_total:
            n_pos = sum(1 for r in records
                        if (pa := extract_answer_number(r["pred_text"])) is not None
                        and (ga := extract_answer_number(r["gt_text"])) is not None
                        and abs(pa - ga) < 1e-4)
            print(f"[Phase 2]  {i+1}/{n_total} questions  |  "
                  f"{len(records)} traces  |  ~{n_pos} correct so far")

    return records


# ── Difference-of-Means ────────────────────────────────────────────────────────

def compute_truth_vector(records: list, out_dir: pathlib.Path) -> dict:
    """
    Protocol equation:
        v_truth^t = (1/|H+|) Σ_{h ∈ H+} h_t  -  (1/|H-|) Σ_{h ∈ H-} h_t

    Also computes σ_l (per-step activation std) for use in Phase 3.
    """
    print(f"\n[Phase 2] Computing Truth Vector from {len(records)} traces...")

    pos_lats, neg_lats = [], []
    n_no_pred = n_no_gt = 0

    for r in records:
        lat = r["latent"]
        if not torch.is_tensor(lat) or lat.dim() != 2:
            continue

        pa = extract_answer_number(r["pred_text"])
        ga = extract_answer_number(r["gt_text"])

        if pa is None: n_no_pred += 1; continue
        if ga is None: n_no_gt   += 1; continue

        if abs(pa - ga) < 1e-4:
            pos_lats.append(lat.float())
        else:
            neg_lats.append(lat.float())

    n_pos, n_neg = len(pos_lats), len(neg_lats)
    print(f"[Phase 2] H+: {n_pos}  H-: {n_neg}  "
          f"no_pred: {n_no_pred}  no_gt: {n_no_gt}")

    if n_pos == 0 or n_neg == 0:
        print("\n[Phase 2] ✗ Cannot compute v_truth — need BOTH positive AND negative samples.")
        if n_pos == 0:
            print("  All traces were WRONG. Consider:")
            print("  1. Using the pretrained zen-E/CODI-gpt2 (not a partially-trained checkpoint)")
            print("  2. Increasing --n-samples for more diversity")
        if n_neg == 0:
            print("  All traces were CORRECT (unlikely — check answer extraction)")
        sys.exit(1)

    pos_stack = torch.stack(pos_lats)  # [N+, L, D]
    neg_stack = torch.stack(neg_lats)  # [N-, L, D]
    L, D = pos_stack.shape[1], pos_stack.shape[2]

    # Per-step difference-of-means  →  v_truth_per_step [L, D]
    v_per_step = pos_stack.mean(dim=0) - neg_stack.mean(dim=0)

    # Global vector (average across all steps)  →  [D]
    v_global   = v_per_step.mean(dim=0)

    # Per-step activation std (for the steering α·σ_l term)
    all_lat = torch.cat([pos_stack, neg_stack], dim=0)    # [N, L, D]
    sigma   = all_lat.std(dim=0).mean(dim=-1)              # [L]

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(v_global,   out_dir / "v_truth.pt")
    torch.save(v_per_step, out_dir / "v_truth_per_step.pt")
    torch.save(sigma,      out_dir / "sigma_per_step.pt")

    per_step_norms = v_per_step.norm(dim=-1).tolist()
    stats = {
        "n_pos":                   n_pos,
        "n_neg":                   n_neg,
        "n_no_pred":               n_no_pred,
        "n_no_gt":                 n_no_gt,
        "balance_ratio":           round(n_pos / (n_pos + n_neg), 4),
        "L":                       L,
        "D":                       D,
        "v_truth_global_norm":     float(v_global.norm()),
        "v_truth_per_step_norms":  [round(v, 4) for v in per_step_norms],
        "sigma_per_step":          [round(s, 4) for s in sigma.tolist()],
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\n[Phase 2] ✓ Truth Vector computed")
    print(f"   Shape           : v_global [{D}],  v_per_step [{L}×{D}]")
    print(f"   Global norm     : {stats['v_truth_global_norm']:.4f}")
    print(f"   Per-step norms  : {[f'{v:.3f}' for v in per_step_norms]}")
    print(f"   Balance (H+/all): {stats['balance_ratio']:.1%}")
    print(f"   Sigma per step  : {[f'{s:.3f}' for s in sigma.tolist()]}")
    print(f"\n   Saved → {out_dir}/")

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Extract Truth Vector from CODI latent space"
    )
    parser.add_argument("--steer-data",  default=DEFAULT_STEER_DATA,
                        help=f"Dsteer split (default: {DEFAULT_STEER_DATA})")
    parser.add_argument("--ckpt-dir",   default=DEFAULT_CKPT_DIR,
                        help="Phase 1 checkpoint dir (default: pretrained zen-E/CODI-gpt2)")
    parser.add_argument("--bundle-dir", default="codi_bundle",
                        help="Path to codi_bundle/ directory")
    parser.add_argument("--work-dir",   default=DEFAULT_WORK_DIR)
    parser.add_argument("--out-dir",    default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-samples",  type=int, default=1,
                        help="Traces per question. 1=greedy (fast). >1=stochastic (richer H+/H-)")
    parser.add_argument("--bf16",       action="store_true",
                        help="Use bf16 on GPU")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Recompute v_truth from existing latent_dump.pt (skip model run)")
    args = parser.parse_args()

    steer_data_path = pathlib.Path(args.steer_data)
    out_dir         = pathlib.Path(args.out_dir)
    work_dir        = pathlib.Path(args.work_dir).resolve()
    bundle_dir      = pathlib.Path(args.bundle_dir).resolve()
    dump_path       = out_dir / "latent_dump.pt"

    print("\n" + "=" * 62)
    print("  Phase 2 — Truth Vector Extraction")
    print("  Method: Difference-of-Means in latent space")
    print("=" * 62)
    print(f"  Steer data : {steer_data_path}")
    print(f"  n_samples  : {args.n_samples} "
          f"({'greedy 1-pass' if args.n_samples == 1 else 'stochastic multi-pass'})")
    print(f"  Output     : {out_dir}")
    print()

    if not steer_data_path.exists():
        sys.exit(
            f"\n[Phase 2] ✗ Steer data not found: {steer_data_path}\n"
            f"  Run first: python split_dataset.py\n"
        )

    steer_data = load_jsonl(steer_data_path)
    print(f"[Phase 2] Loaded {len(steer_data)} steering examples from {steer_data_path.name}")

    if not args.skip_inference:
        ensure_dependencies()
        work_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = get_checkpoint(work_dir, args.ckpt_dir)

        model, tokenizer, device = load_codi_model(ckpt_dir, bundle_dir, args.bf16)

        print(f"\n[Phase 2] Collecting latent traces "
              f"({len(steer_data)} questions × {args.n_samples} sample(s))...")
        t0 = time()
        records = collect_all_traces(model, tokenizer, steer_data, device, args.n_samples)
        elapsed = time() - t0

        print(f"\n[Phase 2] Collection done in {elapsed:.1f}s — {len(records)} traces total")

        # Save dump for reuse / debugging
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(records, dump_path)
        print(f"[Phase 2] Dump saved → {dump_path}")

        # Show a few sample predictions
        print("\n[Phase 2] Sample predictions (first 5):")
        for rec in records[:5]:
            pa = extract_answer_number(rec["pred_text"])
            ga = extract_answer_number(rec["gt_text"])
            correct = "✓" if (pa is not None and ga is not None and abs(pa-ga) < 1e-4) else "✗"
            print(f"  {correct}  pred={pa}  gt={ga}  "
                  f"(text: '{rec['pred_text'][:60].strip()}')")

    else:
        if not dump_path.exists():
            sys.exit(f"[Phase 2] --skip-inference set but dump not found: {dump_path}")
        print(f"[Phase 2] Loading existing dump from {dump_path}...")
        records = torch.load(dump_path)
        print(f"[Phase 2] Loaded {len(records)} records from dump.")

    # Compute the truth vector
    stats = compute_truth_vector(records, out_dir)

    # Verify outputs
    required = ["v_truth.pt", "v_truth_per_step.pt", "sigma_per_step.pt"]
    for fname in required:
        fpath = out_dir / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f"  ✓ {fname}  ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {fname}  MISSING")

    print(f"\n[Phase 2] ✓ Done. Next step:")
    print(f"  python phase3_steer_inference.py --vector-dir {out_dir}\n")


if __name__ == "__main__":
    main()