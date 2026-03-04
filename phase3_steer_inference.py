"""
phase3_steer_inference.py  —  Phase 3: Inference-Time Intervention
===================================================================
Implements the steering equation from the protocol:

    h_{t+1} = Model(h_t)  +  α · σ_l · v̂_truth^t

where:
    α         = steering strength (hyperparameter to sweep)
    σ_l       = activation std at layer l (computed in Phase 2)
    v̂_truth^t = unit-normalised truth direction at step t

Sweep over α ∈ {0.0, 0.1, 0.5, 1.0, 2.0, 5.0}
α = 0.0 is the unsteered baseline / control group.

Evaluation metrics (Phase 4 in protocol):
    Accuracy         — % correct final answers
    Flip Rate        — % of baseline failures corrected by steering
    Cosine Similarity — alignment between steered h_t and v_truth

Dataset: datasets/gsm8k_split/test.jsonl (Deval — held-out, never touched before)

Usage
-----
    # Full sweep (all 6 alpha values) on test set
    python phase3_steer_inference.py

    # Quick sanity check on validation split
    python phase3_steer_inference.py \\
        --eval-data datasets/gsm8k_split/validation.jsonl

    # Single alpha (debugging)
    python phase3_steer_inference.py --alphas 0.0 1.0

    # With Phase 1 fine-tuned model + Phase 2 vector
    python phase3_steer_inference.py \\
        --ckpt-dir  outputs/phase1_checkpoint \\
        --vector-dir outputs/phase2_truth_vector

Outputs
-------
    outputs/phase3_results/alpha_<a>/metrics.json  — per-alpha results
    outputs/phase3_results/summary.json            — all alphas in one table
    outputs/phase3_results/flip_analysis.json      — which examples flipped
    outputs/phase3_results/trajectory_stats.json   — cosine sim with v_truth
"""

import os
import sys
import re
import json
import math
import pathlib
import argparse
import subprocess
from time import time
from copy import deepcopy

import torch
import torch.nn.functional as F


# ── Configuration ──────────────────────────────────────────────────────────────
CODI_HF_ID         = "zen-E/CODI-gpt2"
DEFAULT_EVAL_DATA  = "datasets/gsm8k_split/test.jsonl"
DEFAULT_VECTOR_DIR = "outputs/phase2_truth_vector"
DEFAULT_OUT_DIR    = "outputs/phase3_results"
DEFAULT_WORK_DIR   = "codi_workspace"

# Protocol-specified alpha values
DEFAULT_ALPHAS     = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]


# ── Dependency / checkpoint helpers ───────────────────────────────────────────

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
    """Load CODI model directly from codi_bundle."""
    bundle_str = str(bundle_dir.resolve())
    if bundle_str not in sys.path:
        sys.path.insert(0, bundle_str)

    from src.model import CODI, ModelArguments
    from peft import LoraConfig, TaskType
    import transformers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Phase 3] Loading CODI model on {device.upper()}...")

    model_args = ModelArguments(
        model_name_or_path="gpt2",
        lora_r=128,
        lora_alpha=32,
        full_precision=True,
        train=False,
        lora_init=True,
        ckpt_dir=str(ckpt_dir),
    )

    training_args = transformers.TrainingArguments(
        output_dir="/tmp/codi_phase3",
        no_cuda=(device == "cpu"),
        bf16=(bf16 and device == "cuda"),
        use_cpu=(device == "cpu"),
    )
    for k, v in {
        "num_latent": 6, "use_lora": True, "use_prj": True,
        "prj_dim": 768, "prj_dropout": 0.0, "prj_no_ln": False,
        "distill_loss_div_std": False, "distill_loss_type": "smooth_l1",
        "distill_loss_factor": 1.0, "ref_loss_factor": 1.0,
        "inf_latent_iterations": 6, "inf_num_iterations": 1,
        "remove_eos": True, "print_ref_model_stats": False,
        "include_last_cot": False, "fix_attn_mask": False,
        "log_full": False, "print_loss": False, "max_token_num": 1000,
        "restore_from": "", "expt_name": "phase3", "greedy": True,
        "exp_mode": False, "exp_data_num": 10000,
    }.items():
        setattr(training_args, k, v)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=True,
        r=128, lora_alpha=32, lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"], init_lora_weights=True,
    )

    model = CODI(model_args, training_args, lora_config)
    
    # Load fine-tuned checkpoint from Phase 1
    # Since train=False, init() wasn't called, so load manually
    from safetensors.torch import load_file
    
    # The train.py script creates nested directories, so search for the actual model file
    ckpt_candidates = [
        ckpt_dir / "model.safetensors",
        ckpt_dir / "pytorch_model.bin",
    ]
    
    # Also search in nested directories created by train.py
    for pattern in ["**/model.safetensors", "**/pytorch_model.bin"]:
        ckpt_candidates.extend(ckpt_dir.glob(pattern))
    
    # Find the first valid checkpoint
    ckpt_path = None
    for candidate in ckpt_candidates:
        if candidate.exists():
            ckpt_path = candidate
            break
    
    if ckpt_path:
        print(f"[Phase 3] Loading checkpoint: {ckpt_path}")
        if ckpt_path.suffix == ".safetensors":
            state_dict = load_file(str(ckpt_path))
        else:
            state_dict = torch.load(str(ckpt_path), map_location=device)
        
        # Convert to float32 on CPU
        if device == "cpu":
            state_dict = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}
        
        # Load the state dict
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Phase 3]   ⚠ Missing keys: {len(missing)} (this is normal for LoRA)")
        if unexpected:
            print(f"[Phase 3]   ⚠ Unexpected keys: {len(unexpected)}")
    else:
        print(f"[Phase 3] ⚠ No checkpoint found at {ckpt_dir}, using pretrained weights from HF")
    
    if device == "cpu":
        model = model.float()
    model = model.to(device)
    model.eval()

    import transformers as _tf
    tokenizer = _tf.AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print(f"[Phase 3] ✓ Model ready  (D={model.dim}, k={model.num_latent})")
    return model, tokenizer, device


# ── Answer extraction ──────────────────────────────────────────────────────────

def extract_answer_number(text: str):
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
    nums = re.findall(r"[-+]?\d+\.?\d*", text)
    if nums:
        try: return float(nums[-1])
        except: pass
    try: return float(text.strip())
    except: return None


# ── Steered inference ──────────────────────────────────────────────────────────

@torch.no_grad()
def run_steered_inference(
    model, tokenizer, question: str, device: str,
    alpha: float,
    v_hat_per_step: torch.Tensor,   # [L, D] — unit-normalised per-step truth direction
    sigma_per_step: torch.Tensor,   # [L]    — activation std per step
    collect_cosine: bool = True,
):
    """
    Run one forward pass applying the protocol steering equation at each latent step:

        h_{t+1} = Model(h_t)  +  α · σ_t · v̂_truth^t

    Returns:
        pred_text   str           — decoded final answer
        cosine_sims list[float]   — cos(h_t, v_truth^t) per step (diagnostic)
    """
    prompt = question.strip() + "\nAnswer the above question. "
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    out = model.codi(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        use_cache=True, 
        output_hidden_states=True
    )
    past_kv     = out.past_key_values
    latent_embd = out.hidden_states[-1][:, -1:, :]   # [1, 1, D]
    target_dtype = latent_embd.dtype

    cosine_sims = []

    for step in range(model.num_latent):
        # ── Projection ──────────────────────────────────────────────────────
        if model.use_prj:
            prj_dtype = next(model.prj.parameters()).dtype
            if latent_embd.dtype != prj_dtype:
                latent_embd = latent_embd.to(prj_dtype)
            latent_embd = model.prj(latent_embd)
            if latent_embd.dtype != target_dtype:
                latent_embd = latent_embd.to(target_dtype)

        # ── Protocol steering injection ──────────────────────────────────────
        if alpha != 0.0:
            v_t    = v_hat_per_step[step].to(device).to(target_dtype)    # [D]
            sig_t  = sigma_per_step[step].to(device).to(target_dtype)    # scalar
            delta  = (alpha * sig_t * v_t).view(1, 1, -1)               # [1, 1, D]
            latent_embd = latent_embd + delta

        # ── Cosine similarity diagnostic ─────────────────────────────────────
        if collect_cosine:
            h_flat  = latent_embd[0, 0].float()
            v_t_raw = v_hat_per_step[step].to(device).float()
            cos_val = F.cosine_similarity(h_flat.unsqueeze(0), v_t_raw.unsqueeze(0)).item()
            cosine_sims.append(cos_val)

        # Extend attention mask for new latent token
        attention_mask = torch.cat([
            attention_mask, 
            torch.ones((1, 1), dtype=attention_mask.dtype, device=device)
        ], dim=1)

        # ── Next latent step ─────────────────────────────────────────────────
        out = model.codi(
            inputs_embeds=latent_embd,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_kv,
        )
        past_kv     = out.past_key_values
        latent_embd = out.hidden_states[-1][:, -1:, :]

    # Decode final answer
    # NOTE: generate() does not support inputs_embeds + past_key_values well,
    #       so we use a manual greedy decoding loop instead.
    answer_prompt = tokenizer(
        "The answer is:", return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)
    get_embd = model.get_embd(model.codi, model.model_name)
    answer_embd = get_embd(answer_prompt)

    # Extend attention mask to cover past KV entries + new answer prompt tokens
    attention_mask = torch.cat([
        attention_mask,
        torch.ones((1, answer_embd.shape[1]), dtype=attention_mask.dtype, device=device)
    ], dim=1)

    # Feed "The answer is:" through the model with the accumulated KV cache
    ans_out = model.codi(
        inputs_embeds=answer_embd,
        attention_mask=attention_mask,
        use_cache=True,
        past_key_values=past_kv,
    )
    past_kv = ans_out.past_key_values
    next_logits = ans_out.logits[:, -1, :]  # [1, vocab]

    # Manual greedy decoding
    generated_ids = []
    max_new_tokens = 32
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        next_id = next_logits.argmax(dim=-1)  # [1]
        generated_ids.append(next_id.item())
        if next_id.item() == eos_id:
            break
        # Extend attention mask
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, 1), dtype=attention_mask.dtype, device=device)
        ], dim=1)
        next_embd = get_embd(next_id.unsqueeze(0))  # [1, 1, D]
        out = model.codi(
            inputs_embeds=next_embd,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_kv,
        )
        past_kv = out.past_key_values
        next_logits = out.logits[:, -1, :]

    pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return pred_text, cosine_sims


# ── Alpha sweep ────────────────────────────────────────────────────────────────

def run_alpha_sweep(
    model, tokenizer, eval_data: list, device: str,
    v_hat_per_step: torch.Tensor,
    sigma_per_step: torch.Tensor,
    alphas: list,
    out_dir: pathlib.Path,
) -> list:
    """
    Run the full alpha sweep. For each α, run inference on the eval set,
    compute accuracy, and return a list of result dicts.
    """
    all_results = []

    # Pre-extract ground truth answers
    gt_answers = []
    for item in eval_data:
        gt_raw = item.get("answer", "")
        gt_text = gt_raw.split("####")[-1].strip() if "####" in gt_raw else gt_raw
        gt_answers.append(gt_text)

    n_eval = len(eval_data)

    for alpha in sorted(alphas):
        alpha_dir = out_dir / f"alpha_{alpha}"
        alpha_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Phase 3] ── α = {alpha:.1f} ──────────────────────────────────")

        preds          = []
        correct_flags  = []
        cosine_by_step = [[] for _ in range(model.num_latent)]
        errors         = 0

        t0 = time()
        for i, (item, gt) in enumerate(zip(eval_data, gt_answers)):
            question = item.get("question", "")
            try:
                pred_text, cosines = run_steered_inference(
                    model, tokenizer, question, device,
                    alpha=alpha,
                    v_hat_per_step=v_hat_per_step,
                    sigma_per_step=sigma_per_step,
                )
                pa = extract_answer_number(pred_text)
                ga = extract_answer_number(gt)
                correct = (pa is not None and ga is not None and abs(pa - ga) < 1e-4)
            except Exception as e:
                print(f"  ⚠  q={i}: {e}")
                pred_text = ""
                cosines   = [0.0] * model.num_latent
                correct   = False
                errors    += 1

            preds.append(pred_text)
            correct_flags.append(correct)
            for step, c in enumerate(cosines):
                cosine_by_step[step].append(c)

            if (i + 1) % 100 == 0 or (i + 1) == n_eval:
                acc_so_far = sum(correct_flags) / len(correct_flags)
                print(f"  [{i+1}/{n_eval}]  acc={acc_so_far:.2%}")

        elapsed = time() - t0
        accuracy = sum(correct_flags) / n_eval if n_eval else 0.0

        # Mean cosine similarity per step
        mean_cosines = [
            round(sum(cs) / len(cs), 4) if cs else 0.0
            for cs in cosine_by_step
        ]

        # Save per-example predictions
        records = [
            {"question": q.get("question","")[:100], "pred": p, "gt": g, "correct": c}
            for q, p, g, c in zip(eval_data, preds, gt_answers, correct_flags)
        ]
        (alpha_dir / "predictions.json").write_text(
            json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8"
        )

        result = {
            "alpha":            alpha,
            "accuracy":         round(accuracy, 6),
            "n_correct":        sum(correct_flags),
            "n_total":          n_eval,
            "n_errors":         errors,
            "elapsed_s":        round(elapsed, 2),
            "mean_cosine_per_step": mean_cosines,
            "mean_cosine_global":   round(sum(mean_cosines) / len(mean_cosines), 4),
        }
        (alpha_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

        print(f"  α={alpha:.1f}  accuracy={accuracy:.2%}  "
              f"mean_cosine={result['mean_cosine_global']:.4f}  "
              f"time={elapsed:.0f}s")
        all_results.append(result)

    return all_results


# ── Flip analysis ──────────────────────────────────────────────────────────────

def compute_flip_analysis(all_results: list, out_dir: pathlib.Path) -> list:
    """
    For each steered α, compare per-example correct/wrong against α=0 baseline.

    Flip Rate = (wrong→right) / (total wrong at baseline)
    This is the primary metric for whether steering helps.
    """
    baseline = next((r for r in all_results if r["alpha"] == 0.0), None)
    if baseline is None:
        print("[Phase 3] ⚠  α=0.0 not found — cannot compute flip rate.")
        return []

    baseline_preds_path = out_dir / "alpha_0.0" / "predictions.json"
    if not baseline_preds_path.exists():
        print("[Phase 3] ⚠  Baseline predictions not found — skipping flip analysis.")
        return []

    base_records  = json.load(open(baseline_preds_path, encoding="utf-8"))
    base_correct  = [r["correct"] for r in base_records]
    n_base_wrong  = base_correct.count(False)

    flip_summary = []
    for result in sorted(all_results, key=lambda r: r["alpha"]):
        a = result["alpha"]
        if a == 0.0:
            continue
        preds_path = out_dir / f"alpha_{a}" / "predictions.json"
        if not preds_path.exists():
            continue
        records = json.load(open(preds_path, encoding="utf-8"))
        if len(records) != len(base_records):
            continue

        n_pos_flip = 0   # wrong → right (good)
        n_neg_flip = 0   # right → wrong (bad)
        for i, r in enumerate(records):
            now = r["correct"]
            was = base_correct[i]
            if not was and now:     n_pos_flip += 1
            elif was and not now:   n_neg_flip += 1

        flip_rate = round(n_pos_flip / n_base_wrong, 4) if n_base_wrong else None
        flip_summary.append({
            "alpha":              a,
            "accuracy":           result["accuracy"],
            "flip_rate":          flip_rate,
            "n_wrong_to_right":   n_pos_flip,
            "n_right_to_wrong":   n_neg_flip,
            "net_gain":           n_pos_flip - n_neg_flip,
            "delta_accuracy":     round(result["accuracy"] - baseline["accuracy"], 6),
        })

    (out_dir / "flip_analysis.json").write_text(
        json.dumps(flip_summary, indent=2), encoding="utf-8"
    )
    return flip_summary


# ── Trajectory faithfulness (geometric) ───────────────────────────────────────

def compute_trajectory_stats(all_results: list, out_dir: pathlib.Path):
    """
    Aggregate cosine similarity statistics across alphas and steps.
    Shows how strongly steering aligns latent trajectory with v_truth.
    """
    stats = []
    for result in sorted(all_results, key=lambda r: r["alpha"]):
        stats.append({
            "alpha":            result["alpha"],
            "accuracy":         result["accuracy"],
            "mean_cosine_global": result["mean_cosine_global"],
            "cosine_per_step":  result["mean_cosine_per_step"],
        })
    (out_dir / "trajectory_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    return stats


# ── Pretty summary table ───────────────────────────────────────────────────────

def print_summary_table(all_results: list, flip_summary: list):
    baseline_acc = next((r["accuracy"] for r in all_results if r["alpha"] == 0.0), None)
    flip_by_alpha = {f["alpha"]: f for f in flip_summary}

    print("\n" + "=" * 80)
    print("  PHASE 3 — STEERING RESULTS SUMMARY")
    print("=" * 80)
    print(f"  {'α':>6}  {'Accuracy':>10}  {'Δ vs base':>10}  "
          f"{'Flip Rate':>10}  {'net ±':>8}  {'Cos(h,v)':>9}")
    print("  " + "─" * 72)

    for r in sorted(all_results, key=lambda x: x["alpha"]):
        a         = r["alpha"]
        acc       = r["accuracy"]
        cos_g     = r["mean_cosine_global"]
        delta     = (acc - baseline_acc) if baseline_acc is not None else 0.0
        flip_info = flip_by_alpha.get(a, {})
        flip_str  = f"{flip_info.get('flip_rate', 0.0):.1%}" if a != 0.0 else "  baseline"
        net_str   = (f"+{flip_info['net_gain']}" if flip_info.get("net_gain", 0) > 0
                     else str(flip_info.get("net_gain", ""))) if a != 0.0 else ""
        marker    = "  ← control" if a == 0.0 else ""

        print(f"  {a:>6.1f}  {acc*100:>9.2f}%  {delta*100:>+9.2f}%  "
              f"{flip_str:>10}  {net_str:>8}  {cos_g:>9.4f}{marker}")

    print("=" * 80)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Inference-time steering over alpha sweep"
    )
    parser.add_argument("--eval-data",   default=DEFAULT_EVAL_DATA)
    parser.add_argument("--vector-dir",  default=DEFAULT_VECTOR_DIR,
                        help="Directory with v_truth*.pt and sigma*.pt from Phase 2")
    parser.add_argument("--ckpt-dir",    default=None,
                        help="Phase 1 checkpoint (default: pretrained zen-E/CODI-gpt2)")
    parser.add_argument("--bundle-dir",  default="codi_bundle")
    parser.add_argument("--work-dir",    default=DEFAULT_WORK_DIR)
    parser.add_argument("--out-dir",     default=DEFAULT_OUT_DIR)
    parser.add_argument("--alphas",      nargs="+", type=float,
                        default=DEFAULT_ALPHAS,
                        help=f"Alpha values to sweep (default: {DEFAULT_ALPHAS})")
    parser.add_argument("--use-global-vector", action="store_true",
                        help="Use global v_truth instead of per-step v_truth_per_step")
    parser.add_argument("--bf16",        action="store_true")
    args = parser.parse_args()

    eval_data_path = pathlib.Path(args.eval_data)
    vector_dir     = pathlib.Path(args.vector_dir)
    out_dir        = pathlib.Path(args.out_dir)
    work_dir       = pathlib.Path(args.work_dir).resolve()
    bundle_dir     = pathlib.Path(args.bundle_dir).resolve()

    print("\n" + "=" * 62)
    print("  Phase 3 — Inference-Time Steering Sweep")
    print("  Protocol: h_{t+1} = Model(h_t) + α·σ·v̂_truth")
    print("=" * 62)
    print(f"  Eval data  : {eval_data_path}")
    print(f"  Vector dir : {vector_dir}")
    print(f"  Alphas     : {sorted(args.alphas)}")
    print(f"  Output     : {out_dir}")
    print()

    # Validate prerequisites
    if not eval_data_path.exists():
        sys.exit(
            f"\n[Phase 3] ✗ Eval data not found: {eval_data_path}\n"
            f"  Run: python split_dataset.py\n"
        )
    for fname in ["v_truth.pt", "v_truth_per_step.pt", "sigma_per_step.pt"]:
        if not (vector_dir / fname).exists():
            sys.exit(
                f"\n[Phase 3] ✗ {fname} not found in {vector_dir}\n"
                f"  Run: python phase2_extract_vector.py\n"
            )

    # Load eval data
    eval_data = load_jsonl(eval_data_path)
    print(f"[Phase 3] Loaded {len(eval_data)} eval examples from {eval_data_path.name}")

    # Load truth vectors
    v_global       = torch.load(vector_dir / "v_truth.pt")          # [D]
    v_per_step_raw = torch.load(vector_dir / "v_truth_per_step.pt") # [L, D]
    sigma          = torch.load(vector_dir / "sigma_per_step.pt")   # [L]

    # Normalise to unit vectors
    v_hat_per_step = F.normalize(v_per_step_raw, dim=-1)            # [L, D]
    v_hat_global   = F.normalize(v_global.unsqueeze(0), dim=-1).squeeze(0)  # [D]

    # If using global vector, replicate it across all steps
    if args.use_global_vector:
        L = v_per_step_raw.shape[0]
        v_hat_per_step = v_hat_global.unsqueeze(0).expand(L, -1)
        print(f"[Phase 3] Using global v_truth (broadcast to all {L} steps)")
    else:
        print(f"[Phase 3] Using per-step v_truth ({v_hat_per_step.shape[0]} steps × {v_hat_per_step.shape[1]}D)")

    # Print vector stats from Phase 2
    stats_path = vector_dir / "stats.json"
    if stats_path.exists():
        stats = json.load(open(stats_path, encoding="utf-8"))
        print(f"[Phase 3] v_truth stats: "
              f"H+={stats['n_pos']}  H-={stats['n_neg']}  "
              f"balance={stats['balance_ratio']:.1%}  "
              f"global_norm={stats['v_truth_global_norm']:.4f}")

    # Load CODI model
    ensure_dependencies()
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = get_checkpoint(work_dir, args.ckpt_dir)
    model, tokenizer, device = load_codi_model(ckpt_dir, bundle_dir, args.bf16)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Run alpha sweep
    print(f"\n[Phase 3] Starting sweep over {len(args.alphas)} alpha values...\n")
    t_sweep = time()

    all_results = run_alpha_sweep(
        model, tokenizer, eval_data, device,
        v_hat_per_step=v_hat_per_step,
        sigma_per_step=sigma,
        alphas=args.alphas,
        out_dir=out_dir,
    )

    elapsed_total = time() - t_sweep
    print(f"\n[Phase 3] Sweep complete in {elapsed_total/60:.1f} min")

    # Flip analysis
    print("\n[Phase 3] Computing flip analysis...")
    flip_summary = compute_flip_analysis(all_results, out_dir)

    # Trajectory stats
    traj_stats = compute_trajectory_stats(all_results, out_dir)

    # Pretty summary
    print_summary_table(all_results, flip_summary)

    # Flip rate table
    if flip_summary:
        print("\n[Phase 3] FLIP RATE ANALYSIS")
        print(f"  {'α':>6}  {'wrong→right':>12}  {'right→wrong':>12}  "
              f"{'flip_rate':>10}  {'net_gain':>9}  {'Δacc':>8}")
        print("  " + "─" * 65)
        for f in sorted(flip_summary, key=lambda x: x["alpha"]):
            fr = f"{f['flip_rate']:.1%}" if f["flip_rate"] is not None else "N/A"
            print(f"  {f['alpha']:>6.1f}  {f['n_wrong_to_right']:>12}  "
                  f"{f['n_right_to_wrong']:>12}  {fr:>10}  "
                  f"{f['net_gain']:>9}  {f['delta_accuracy']*100:>+7.2f}%")

    # Cosine similarity summary
    print("\n[Phase 3] TRAJECTORY FAITHFULNESS (mean cosine sim h_t · v_truth^t)")
    print(f"  {'α':>6}  {'global':>8}  {'step0':>8}  {'step1':>8}  "
          f"{'step2':>8}  {'step3':>8}  {'step4':>8}  {'step5':>8}")
    print("  " + "─" * 68)
    for t in sorted(traj_stats, key=lambda x: x["alpha"]):
        cs = t["cosine_per_step"]
        cs_str = "  ".join(f"{c:>8.4f}" for c in cs)
        print(f"  {t['alpha']:>6.1f}  {t['mean_cosine_global']:>8.4f}  {cs_str}")

    # Save full summary
    summary = {
        "eval_data":       str(eval_data_path),
        "vector_dir":      str(vector_dir),
        "alphas_swept":    sorted(args.alphas),
        "n_eval":          len(eval_data),
        "elapsed_seconds": round(elapsed_total, 2),
        "results":         all_results,
        "flip_analysis":   flip_summary,
        "trajectory_stats":traj_stats,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Identify best alpha
    best = max(all_results, key=lambda r: r["accuracy"])
    baseline_acc = next((r["accuracy"] for r in all_results if r["alpha"] == 0.0), 0.0)
    print(f"\n[Phase 3] Best α = {best['alpha']}  "
          f"(accuracy: {best['accuracy']:.2%}  "
          f"Δ from baseline: {(best['accuracy']-baseline_acc)*100:+.2f}%)")

    print(f"\n[Phase 3] ✓ All results saved → {out_dir}/")
    print(f"  summary.json, flip_analysis.json, trajectory_stats.json")
    print(f"  alpha_*/metrics.json + predictions.json\n")


if __name__ == "__main__":
    main()