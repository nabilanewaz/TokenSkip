"""
phase2/extract_vector.py
------------------------
Extracts the truth vector (v_truth) from the residual stream of HuggingFace models.

Methodology:
1. Runs standard Text CoT generation on `steer_train.jsonl`
2. Classifies each trace as H+ (correct answer) or H- (incorrect answer)
3. Computes v_truth = mean(H+) - mean(H-) at the specified hook layer
4. Computes σ (activation std) for steering scaling

Usage:
    python research/phase2/extract_vector.py --model-type phi2 --model-path microsoft/phi-2
"""
from __future__ import annotations
import argparse, json, pathlib, sys, os
from time import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

_RESEARCH_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RESEARCH_ROOT))

from utils.io import load_jsonl, load_config
from utils.answer import answers_match, get_question, get_gt_answer, extract_answer_number
from utils.model_registry import get_config as get_model_cfg

_CFG = load_config()
_P1  = _CFG.get("phase1", {})
_DS  = _CFG.get("dataset", {})

DEFAULT_STEER    = str(pathlib.Path(_DS.get("out_dir", "datasets/gsm8k_split")) / "steer_train.jsonl")
DEFAULT_OUT_ROOT = "outputs/truth_vectors"
ALL_MODEL_TYPES  = ["phi2","llama32_3b","qwen25_3b","qwen25_1_5b","qwen25_0_5b"]

class ExtractionHook:
    def __init__(self):
        self.hidden_steps = []
        self._handle = None

    def register(self, layer):
        def _hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            self.hidden_steps.append(h[:, -1, :].detach().cpu())
            return output
        self._handle = layer.register_forward_hook(_hook)

    def reset(self):
        self.hidden_steps = []

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None

def get_hook_layer(model, layer_frac=0.75):
    for attr in ["model", "transformer"]:
        sub = getattr(model, attr, None)
        if sub is None: continue
        for ba in ["layers", "h", "blocks"]:
            bl = getattr(sub, ba, None)
            if bl is None: continue
            n   = len(bl)
            idx = max(0, int(n * layer_frac) - 1)
            return bl[idx], idx, n
    raise RuntimeError("Cannot locate transformer layers.")

def load_model(model_path, device, lora_dir=None):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True,
        device_map="auto" if device.type == "cuda" else None)
    
    if lora_dir and os.path.exists(lora_dir):
        print(f"[Phase 2] Loading LoRA adapter from {lora_dir}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)
        
    if device.type != "cuda": model = model.to(device)
    model.eval()
    return model, tok

def extract_truth_vector(model, tokenizer, model_type, steer_data_path, hook_layer, device, n_samples=5, temperature=1.0, use_per_step=True, max_new=512):
    mcfg = get_model_cfg(model_type)
    steer_data = load_jsonl(steer_data_path)

    hook = ExtractionHook()
    hook.register(hook_layer)

    pos_h, neg_h = [], []
    pos_step_sums, neg_step_sums = {}, {}
    pos_step_counts, neg_step_counts = {}, {}
    n_total = len(steer_data)

    print(f"\n[Phase 2] Extracting v_truth from {n_total} questions x {n_samples} samples...")
    t0 = time()

    for i, item in enumerate(steer_data):
        q  = get_question(item)
        gt = get_gt_answer(item)
        prompt = mcfg["build_prompt"](q, tokenizer, 1.0)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        for _ in range(n_samples):
            hook.reset()
            gen_kwargs = {
                "max_new_tokens": max_new,
                "pad_token_id": tokenizer.pad_token_id,
                "do_sample": (n_samples > 1),
            }
            if n_samples > 1:
                gen_kwargs["temperature"] = temperature
            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)
            if not hook.hidden_steps:
                continue

            traj = torch.cat(hook.hidden_steps, dim=0).float()  # [steps, hidden]
            pred_text = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            is_pos = answers_match(pred_text, gt)
            (pos_h if is_pos else neg_h).append(traj.mean(dim=0, keepdim=True))

            if use_per_step:
                for t in range(traj.shape[0]):
                    step_vec = traj[t]
                    if is_pos:
                        pos_step_sums[t] = pos_step_sums.get(t, torch.zeros_like(step_vec)) + step_vec
                        pos_step_counts[t] = pos_step_counts.get(t, 0) + 1
                    else:
                        neg_step_sums[t] = neg_step_sums.get(t, torch.zeros_like(step_vec)) + step_vec
                        neg_step_counts[t] = neg_step_counts.get(t, 0) + 1

        if (i+1) % 50 == 0 or (i+1) == n_total:
            print(f"  [{i+1}/{n_total}] H+: {len(pos_h)}  H-: {len(neg_h)}")

    hook.remove()
    print(f"[Phase 2] Generation done in {time()-t0:.1f}s")

    if len(pos_h) == 0 or len(neg_h) == 0:
        raise ValueError(f"Need BOTH H+ and H- to compute v_truth. Got H+: {len(pos_h)}, H-: {len(neg_h)}")

    pos_stack = torch.cat(pos_h, dim=0).float()
    neg_stack = torch.cat(neg_h, dim=0).float()
    v_truth = pos_stack.mean(dim=0) - neg_stack.mean(dim=0)
    v_norm  = F.normalize(v_truth.unsqueeze(0), dim=-1).squeeze(0)

    per_step = {}
    if use_per_step:
        common_steps = sorted(set(pos_step_sums.keys()) & set(neg_step_sums.keys()))
        for t in common_steps:
            if pos_step_counts.get(t, 0) == 0 or neg_step_counts.get(t, 0) == 0:
                continue
            p = pos_step_sums[t] / pos_step_counts[t]
            n = neg_step_sums[t] / neg_step_counts[t]
            per_step[str(t)] = F.normalize((p - n).unsqueeze(0), dim=-1).squeeze(0)

    all_stack = torch.cat([pos_stack, neg_stack], dim=0)
    sigma = float(all_stack.std(dim=0).mean())
    return v_norm, sigma, len(pos_h), len(neg_h), per_step

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-type", required=True, choices=ALL_MODEL_TYPES)
    p.add_argument("--steer-data", default=DEFAULT_STEER)
    p.add_argument("--out-root",   default=DEFAULT_OUT_ROOT)
    p.add_argument("--layer-frac", type=float, default=0.75)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_dir = str(pathlib.Path("outputs/phase1_checkpoint") / args.model_type)
    model, tokenizer = load_model(args.model_path, device, lora_dir)
    hook_layer, layer_idx, n_layers = get_hook_layer(model, args.layer_frac)
    
    print(f"\n[Phase 2] Truth Vector Extraction | model={args.model_type} | layer={layer_idx}/{n_layers-1}")
    
    v_norm, sigma, n_pos, n_neg, per_step = extract_truth_vector(
        model, tokenizer, args.model_type, args.steer_data, hook_layer, device,
        n_samples=_P1.get("n_samples", 5),
        temperature=_P1.get("temperature", 1.0),
        use_per_step=_P1.get("use_per_step", True),
    )
    
    out_dir = pathlib.Path(args.out_root) / args.model_type
    out_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(v_norm, out_dir / "v_truth.pt")
    torch.save(torch.tensor(sigma), out_dir / "sigma.pt")
    if per_step:
        torch.save({k: v.cpu() for k, v in per_step.items()}, out_dir / "v_truth_per_step.pt")
    
    stats = {
        "model": args.model_type,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "balance": n_pos / (n_pos + n_neg),
        "sigma": sigma,
        "v_norm_len": v_norm.shape[0],
        "n_samples": _P1.get("n_samples", 5),
        "temperature": _P1.get("temperature", 1.0),
        "use_per_step": _P1.get("use_per_step", True),
        "n_per_step_vectors": len(per_step)
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    
    print(f"\n[Phase 2] ✓ Extraction complete")
    print(f"   H+ : {n_pos} | H- : {n_neg}")
    print(f"   σ  : {sigma:.4f}")
    print(f"   v_truth saved to {out_dir}/")

if __name__ == "__main__":
    main()
