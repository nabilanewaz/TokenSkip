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
        self.hidden_state = None
        self._handle = None

    def register(self, layer):
        def _hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            # Capture the hidden state of the LAST token in the sequence
            self.hidden_state = h[:, -1, :].detach().cpu()
            return output
        self._handle = layer.register_forward_hook(_hook)

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

def extract_truth_vector(model, tokenizer, model_type, steer_data_path, hook_layer, device, max_new=512):
    mcfg = get_model_cfg(model_type)
    steer_data = load_jsonl(steer_data_path)
    
    hook = ExtractionHook()
    hook.register(hook_layer)
    
    pos_h, neg_h = [], []
    n_total = len(steer_data)
    
    print(f"\n[Phase 2] Extracting v_truth from {n_total} traces …")
    t0 = time()
    
    for i, item in enumerate(steer_data):
        q  = get_question(item)
        gt = get_gt_answer(item)
        prompt = mcfg["build_prompt"](q, tokenizer, 1.0)
        
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        
        # hook.hidden_state was captured during generate (first pass)
        h_t = hook.hidden_state
        
        pred_text = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if answers_match(pred_text, gt):
            pos_h.append(h_t)
        else:
            neg_h.append(h_t)
            
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
    
    all_stack = torch.cat([pos_stack, neg_stack], dim=0)
    sigma = float(all_stack.std(dim=0).mean())
    
    return v_norm, sigma, len(pos_h), len(neg_h)

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
    
    v_norm, sigma, n_pos, n_neg = extract_truth_vector(
        model, tokenizer, args.model_type, args.steer_data, hook_layer, device
    )
    
    out_dir = pathlib.Path(args.out_root) / args.model_type
    out_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(v_norm, out_dir / "v_truth.pt")
    torch.save(torch.tensor(sigma), out_dir / "sigma.pt")
    
    stats = {
        "model": args.model_type,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "balance": n_pos / (n_pos + n_neg),
        "sigma": sigma,
        "v_norm_len": v_norm.shape[0]
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    
    print(f"\n[Phase 2] ✓ Extraction complete")
    print(f"   H+ : {n_pos} | H- : {n_neg}")
    print(f"   σ  : {sigma:.4f}")
    print(f"   v_truth saved to {out_dir}/")

if __name__ == "__main__":
    main()
