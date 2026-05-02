"""
phase3/steer.py
---------------
Master inference script for evaluating the 5 CCoT conditions on HF models.

Conditions
──────────
  no_cot       — Direct answer, no reasoning
  text_cot     — Standard discrete chain-of-thought
  ccot         — Compressed CoT (TokenSkip), no intervention
  random_noise — Compressed CoT + random unit vector (control)
  steered      — Compressed CoT + v_truth steering

Usage
─────
    python research/phase2/steer.py \
        --model-path Qwen/Qwen2.5-3B --model-type qwen25_3b \
        --condition steered \
        --vector-dir outputs/truth_vectors/qwen25_3b

Outputs
───────
    <out-dir>/metrics.json                 [no_cot, text_cot]
    <out-dir>/ratio_<r>/metrics.json       [ccot]
    <out-dir>/ratio_<r>/alpha_<a>/metrics.json   [steered, random_noise]
    <out-dir>/summary.json
"""
from __future__ import annotations
import argparse, json, pathlib, sys, os
from time import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

_RESEARCH_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RESEARCH_ROOT))

from utils.io            import load_jsonl, load_config
from utils.answer        import answers_match, get_question, get_gt_answer
from utils.model_registry import get_config as get_model_cfg
from utils.tokenskip     import compress_cot_text

_CFG = load_config()
_P2  = _CFG.get("phase2", {})
_TS  = _CFG.get("tokenskip", {})
_DS  = _CFG.get("dataset", {})

_split_dir         = pathlib.Path(_DS.get("out_dir", "datasets/gsm8k_split"))
DEFAULT_EVAL       = str(_split_dir / "test.jsonl")
DEFAULT_VEC_ROOT   = "outputs/truth_vectors"
DEFAULT_OUT        = "outputs/phase2_results/default"
PROTOCOL_ALPHAS    = _P2.get("alpha_sweep", [0.0,0.1,0.5,1.0,2.0,5.0,10.0,20.0,50.0])
PROTOCOL_RATIOS    = _TS.get("ratios",      [0.5,0.6,0.7,0.8,0.9,1.0])
LLMLINGUA_MODEL    = _TS.get("llmlingua_model", "llmlingua-2-xlm-roberta-large-meetingbank")
ALL_MODEL_TYPES    = ["phi2","llama32_3b","qwen25_3b","qwen25_1_5b","qwen25_0_5b"]
CONDITIONS         = ["no_cot", "text_cot", "ccot", "random_noise", "steered"]

def load_model(model_path, device, lora_dir=None):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True,
        device_map="auto" if device.type == "cuda" else None)
    
    if lora_dir and os.path.exists(lora_dir):
        print(f"[Phase 3] Loading LoRA adapter from {lora_dir}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)
        
    if device.type != "cuda": model = model.to(device)
    model.eval()
    return model, tok

class BoundarySteeringHook:
    def __init__(self):
        self._handle = None
        self.delta   = None
        self.active  = False

    def register(self, layer, alpha: float, sigma: float, v_norm: torch.Tensor, device):
        if alpha == 0.0:
            self.active = False
            return
        self.delta  = (alpha * sigma * v_norm).to(device)
        self.active = True

        def _hook(module, inputs, output):
            if not self.active: return output
            hidden = output[0] if isinstance(output, tuple) else output
            # Inject at the very last token (the compression boundary)
            hidden[:, -1, :] = hidden[:, -1, :] + self.delta.to(hidden.device, hidden.dtype)
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

        self._handle = layer.register_forward_hook(_hook)

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None
        self.active = False

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

def build_answer_prompt(question: str, compressed_cot: str, model_type: str) -> str:
    templates = {
        "phi2":       (f"Instruct: {question}\nReasoning: {compressed_cot}\nOutput:"),
        "llama32_3b": (f"Question: {question}\nReasoning: {compressed_cot}\nAnswer:"),
        "qwen25_3b":  (f"Question: {question}\nSolution: {compressed_cot}\nAnswer:"),
        "qwen25_1_5b":(f"Question: {question}\nSolution: {compressed_cot}\nAnswer:"),
        "qwen25_0_5b":(f"Question: {question}\nSolution: {compressed_cot}\nAnswer:"),
    }
    return templates.get(model_type, f"Question: {question}\nReasoning: {compressed_cot}\nAnswer:")

def run_evaluation(
    model, tokenizer, model_type: str, condition: str, eval_data: list[dict], device,
    ratio: float = 1.0, alpha: float = 0.0, v_norm: torch.Tensor | None = None, sigma: float = 1.0,
    hook_layer=None, llmlingua_model_name: str = LLMLINGUA_MODEL
) -> dict:
    mcfg = get_model_cfg(model_type)
    records = []
    cot_before, cot_after, errors = [], [], 0
    hook = BoundarySteeringHook()

    if condition in ("steered", "random_noise") and alpha != 0.0 and v_norm is not None and hook_layer is not None:
        hook.register(hook_layer, alpha, sigma, v_norm, device)

    n = len(eval_data)
    for i, item in enumerate(eval_data):
        q  = get_question(item)
        gt = get_gt_answer(item)
        try:
            if condition == "no_cot":
                prompt = mcfg["build_no_cot_prompt"](q, tokenizer)
                enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=64, pad_token_id=tokenizer.pad_token_id)
                pred = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            
            elif condition == "text_cot":
                prompt = mcfg["build_prompt"](q, tokenizer, 1.0)
                enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id)
                pred = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                
            else: # ccot, random_noise, steered
                prompt = mcfg["build_prompt"](q, tokenizer, 1.0)
                enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id)
                full_cot = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                
                comp = compress_cot_text(full_cot, ratio, model_type, llmlingua_model_name)
                comp_cot = comp["compressed_cot"]
                cot_before.append(comp["original_tokens"])
                cot_after.append(comp["compressed_tokens"])
                
                ans_prompt = build_answer_prompt(q, comp_cot, model_type)
                ans_enc = tokenizer(ans_prompt, return_tensors="pt", truncation=True, max_length=768).to(device)
                with torch.no_grad():
                    # hook is active during this generate
                    ans_out = model.generate(**ans_enc, max_new_tokens=64, pad_token_id=tokenizer.pad_token_id)
                pred = tokenizer.decode(ans_out[0, ans_enc["input_ids"].shape[1]:], skip_special_tokens=True)

            correct = answers_match(pred, gt)
        except Exception as e:
            print(f"  ⚠ q={i}: {e}")
            pred, correct, errors = "", False, errors + 1

        records.append({"pred": pred, "gt": gt, "correct": correct})
        if (i+1) % 100 == 0 or (i+1) == n:
            acc = sum(r["correct"] for r in records) / len(records)
            print(f"  [{i+1}/{n}]  acc={acc:.1%}  cond={condition}")

    hook.remove()
    accuracy = sum(r["correct"] for r in records) / n if n else 0.0
    mean_orig = sum(cot_before) / n if n and cot_before else 0.0
    mean_comp = sum(cot_after) / n if n and cot_after else 0.0
    
    return {
        "condition": condition,
        "ratio": ratio if condition in ("ccot","random_noise","steered") else 1.0,
        "alpha": alpha if condition in ("random_noise","steered") else 0.0,
        "accuracy": round(accuracy, 6),
        "n_correct": sum(r["correct"] for r in records),
        "n_total": n,
        "n_errors": errors,
        "mean_cot_tokens_before": round(mean_orig, 1),
        "mean_cot_tokens_after": round(mean_comp, 1),
        "actual_compression_ratio": round(mean_comp / mean_orig, 4) if mean_orig > 0 else 1.0,
        "records": records,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path",  required=True)
    p.add_argument("--model-type",  required=True, choices=ALL_MODEL_TYPES)
    p.add_argument("--eval-data",   default=DEFAULT_EVAL)
    p.add_argument("--vector-dir",  default=DEFAULT_VEC_ROOT)
    p.add_argument("--out-dir",     default=DEFAULT_OUT)
    p.add_argument("--condition",   required=True, choices=CONDITIONS)
    p.add_argument("--ratios",  nargs="+", type=float, default=PROTOCOL_RATIOS)
    p.add_argument("--alphas",  nargs="+", type=float, default=PROTOCOL_ALPHAS)
    p.add_argument("--layer-frac",  type=float, default=0.75)
    p.add_argument("--llmlingua-model", default=LLMLINGUA_MODEL)
    p.add_argument("--seed",    type=int,   default=42)
    args = p.parse_args()

    import random, numpy as np
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = pathlib.Path(args.out_dir) / args.model_type / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Phase 3: steer.py | {args.model_type} | {args.condition}")
    print(f"{'='*62}")

    eval_data = load_jsonl(args.eval_data)
    lora_dir = str(pathlib.Path("outputs/phase1_checkpoint") / args.model_type)
    model, tokenizer = load_model(args.model_path, device, lora_dir)
    hook_layer, layer_idx, n_layers = get_hook_layer(model, args.layer_frac)

    v_norm, sigma = None, 1.0
    if args.condition in ("steered", "random_noise"):
        vdir = pathlib.Path(args.vector_dir) / args.model_type
        vp = vdir / "v_truth.pt"
        sp = vdir / "sigma.pt"
        if vp.exists():
            v_truth = torch.load(vp).to(device)
            v_norm  = F.normalize(v_truth.unsqueeze(0), dim=-1).squeeze(0)
            if sp.exists(): sigma = float(torch.load(sp))
            
            if args.condition == "random_noise":
                v_norm = torch.randn_like(v_norm)
                v_norm = F.normalize(v_norm.unsqueeze(0), dim=-1).squeeze(0)
                print(f"  v_truth swapped for RANDOM NOISE")
        else:
            sys.exit(f"⚠ v_truth not found at {vp} — run phase1/extract_vector.py first")

    t_total = time()
    all_rows = []
    
    if args.condition in ("no_cot", "text_cot"):
        res = run_evaluation(model, tokenizer, args.model_type, args.condition, eval_data, device)
        recs = res.pop("records")
        (out_dir/"metrics.json").write_text(json.dumps(res,indent=2),encoding="utf-8")
        all_rows.append(res)
        
    elif args.condition == "ccot":
        for ratio in sorted(args.ratios, reverse=True):
            rdir = out_dir / f"ratio_{ratio}"
            rdir.mkdir(parents=True, exist_ok=True)
            res = run_evaluation(model, tokenizer, args.model_type, args.condition, eval_data, device, ratio=ratio, llmlingua_model_name=args.llmlingua_model)
            recs = res.pop("records")
            (rdir/"metrics.json").write_text(json.dumps(res,indent=2),encoding="utf-8")
            all_rows.append(res)
            
    elif args.condition in ("steered", "random_noise"):
        for ratio in sorted(args.ratios, reverse=True):
            for alpha in sorted(args.alphas):
                adir = out_dir / f"ratio_{ratio}" / f"alpha_{alpha}"
                adir.mkdir(parents=True, exist_ok=True)
                res = run_evaluation(model, tokenizer, args.model_type, args.condition, eval_data, device, ratio=ratio, alpha=alpha, v_norm=v_norm, sigma=sigma, hook_layer=hook_layer, llmlingua_model_name=args.llmlingua_model)
                recs = res.pop("records")
                (adir/"metrics.json").write_text(json.dumps(res,indent=2),encoding="utf-8")
                all_rows.append(res)

    summary = {
        "model": args.model_type,
        "condition": args.condition,
        "results": all_rows,
        "elapsed": time()-t_total
    }
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"\n  Results → {out_dir}/")

if __name__ == "__main__":
    main()
