"""
hidden_steer.py
---------------
Hook-based latent-space steering for standard transformer LLMs
(microsoft/phi-2, meta-llama/Llama-3.2-3B, Qwen/Qwen2.5-3B).

Mirrors the CODI-based steer_inference.py for GPT-2, but operates
on the residual stream of any HuggingFace CausalLM via forward hooks.

Research protocol equation
───────────────────────────
  h_{l,t+1} = h_{l,t} + α · σ_l · v_truth_l / |v_truth_l|

where h_{l,t} is the hidden state at layer l, position t.

Conditions
──────────
  ccot          alpha=0, v_truth loaded  → unsteered CCoT (control)
  random_noise  alpha=1, random unit vec → controls for vector direction
  steered       alpha sweep [0.0 … 5.0] → the experimental condition

Phase 2 extraction (v_truth)
─────────────────────────────
  If --vector-dir does not contain v_truth.pt for this model, this script
  first runs extraction:
    1. Load frozen model
    2. Run steer_data examples at T=1.0, n_samples times
    3. Classify traces as H+ (correct) / H- (incorrect)
    4. Compute v_truth_l = mean(H+_l) - mean(H-_l) for each layer l

Outputs
────────
  <out_dir>/metrics.json       — accuracy, flip_rate, mean_cos_sim
  <out_dir>/records.json       — per-example predictions
  <out_dir>/condition_info.json— alpha, condition, model

Usage
─────
  python hidden_steer.py --model-path microsoft/phi-2 --model-type phi2 \\
      --condition steered --eval-data datasets/gsm8k_split/test.jsonl \\
      --steer-data datasets/gsm8k_split/steer_train.jsonl \\
      --out-dir outputs/eval_grid/phi2/steered
"""

import os, sys, json, re, pathlib, argparse, math
from time import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_EVAL_DATA   = "datasets/gsm8k_split/test.jsonl"
DEFAULT_STEER_DATA  = "datasets/gsm8k_split/steer_train.jsonl"
DEFAULT_VECTOR_DIR  = "outputs/truth_vectors"
DEFAULT_OUT_DIR     = "outputs/eval_grid/default/steered"

# Alpha sweep (protocol-specified)
DEFAULT_ALPHAS      = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
DEFAULT_N_SAMPLES   = 3       # stochastic passes per example for v_truth extraction
DEFAULT_LAYER_FRAC  = 0.75    # intervene at this fraction of total layers (last quarter)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_answer(text):
    """Extract a numeric answer from model output."""
    for pat in [
        r"\\boxed\{([^}]+)\}",
        r"####\s*([-+]?\d[\d,]*\.?\d*)",
        r"answer is:?\s*([-+]?\d[\d,]*\.?\d*)",
        r"=\s*([-+]?\d[\d,]*\.?\d*)\s*$",
    ]:
        m = re.search(pat, str(text), re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                pass
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", str(text))
    try:
        return float(nums[-1].replace(",", "")) if nums else None
    except ValueError:
        return None


def answers_match(pred_text, gt_text, tol=1e-4):
    pa = extract_answer(pred_text)
    ga = extract_answer(gt_text)
    if pa is None or ga is None:
        return False
    return abs(pa - ga) <= tol


def get_question(example):
    """Extract question string from either GSM8K or messages format."""
    if "messages" in example:
        for m in example["messages"]:
            if m.get("role") == "user":
                return m["content"]
    return example.get("question", "")


def get_answer(example):
    return example.get("answer", "")


# ── Model / tokenizer loader ──────────────────────────────────────────────────

def load_model_and_tokenizer(model_path, device):
    print(f"  [hidden_steer] Loading {model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    print(f"  [hidden_steer] Model loaded  ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")
    return model, tokenizer


def num_layers(model):
    """Return total transformer block count (model-agnostic)."""
    for attr in ["model", "transformer"]:
        sub = getattr(model, attr, None)
        if sub is not None:
            for block_attr in ["layers", "h", "blocks"]:
                blocks = getattr(sub, block_attr, None)
                if blocks is not None:
                    return len(blocks)
    return 12  # fallback


def get_hook_layer_idx(model, layer_frac):
    n = num_layers(model)
    return max(0, int(n * layer_frac) - 1)


# ── Hook-based hidden state capture & injection ───────────────────────────────

class HiddenStateCollector:
    """
    Registers a forward hook on a specific transformer layer to:
    1. Collect hidden states (last token) for v_truth extraction.
    2. Inject a steering delta during generation.
    """
    def __init__(self, model, layer_idx, device):
        self.layer_idx = layer_idx
        self.device    = device
        self.collected = []   # list of [D] tensors (last-token hidden states)
        self.delta     = None  # [D] tensor added at each forward pass when set
        self.cos_sims  = []   # recorded cosine similarities

        # Locate the correct block
        self._layer = self._find_layer(model, layer_idx)
        self._hook  = None

    def _find_layer(self, model, idx):
        for attr in ["model", "transformer"]:
            sub = getattr(model, attr, None)
            if sub is not None:
                for block_attr in ["layers", "h", "blocks"]:
                    blocks = getattr(sub, block_attr, None)
                    if blocks is not None:
                        return blocks[idx]
        raise RuntimeError("Cannot locate transformer layers in model architecture.")

    def register(self, collect=False, inject=False, v_norm=None, sigma=None):
        """Register the forward hook.
        collect=True : record last-token hidden state.
        inject=True  : add delta (alpha * sigma * v_norm) to hidden state.
        """
        self._collect = collect
        self._inject  = inject
        self._v_norm  = v_norm.to(self.device) if v_norm is not None else None
        self._sigma   = sigma

        def _hook(module, inputs, output):
            # output is often a tuple; hidden state is first element
            hidden = output[0] if isinstance(output, tuple) else output
            # hidden: [B, S, D]
            if self._collect:
                last = hidden[:, -1, :].detach().float()  # [B, D]
                self.collected.append(last)

            if self._inject and self._v_norm is not None and self.delta is not None:
                hidden = hidden + self.delta.to(hidden.device, hidden.dtype)
                if self._collect:
                    # Record cosine similarity between pre-injection last token and v_norm
                    pass
                # Record cosine sim after injection
                if self._v_norm is not None:
                    h_last = hidden[:, -1, :].detach().float()  # [B, D]
                    v = self._v_norm.float().unsqueeze(0).expand(h_last.shape[0], -1)
                    cos = F.cosine_similarity(h_last, v, dim=-1).mean().item()
                    self.cos_sims.append(cos)

                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden

            # Record cos sim even when not injecting
            if self._v_norm is not None and self._collect:
                h_last = hidden[:, -1, :].detach().float()
                v = self._v_norm.float().unsqueeze(0).expand(h_last.shape[0], -1)
                cos = F.cosine_similarity(h_last, v, dim=-1).mean().item()
                self.cos_sims.append(cos)

        self._hook = self._layer.register_forward_hook(_hook)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def set_delta(self, alpha, sigma, v_norm):
        """Compute and cache delta = alpha * sigma * v_norm."""
        if alpha == 0.0 or v_norm is None:
            self.delta = None
        else:
            self.delta = (alpha * sigma * v_norm).to(self.device)

    def clear(self):
        self.collected.clear()
        self.cos_sims.clear()


# ── v_truth extraction (Phase 2) ─────────────────────────────────────────────

def build_cot_prompt(model_type, question, tokenizer):
    """Build a text CoT prompt for v_truth extraction."""
    from model_registry import get_config
    cfg = get_config(model_type)
    return cfg["build_prompt"](question, tokenizer, compression_ratio=1.0)


def extract_truth_vector(model, tokenizer, model_type, steer_data,
                         collector, device, n_samples=3, seed=42,
                         max_new_tokens=256, batch_size=4):
    """
    Phase 2: collect H+ and H- hidden states from steer_data, compute v_truth.
    Returns (v_norm, sigma) where v_norm is the normalised truth direction [D]
    and sigma is the scalar activation std at the hook layer.
    """
    print("  [extract] Running truth vector extraction …")
    examples = load_jsonl(steer_data)
    rng = torch.Generator(); rng.manual_seed(seed)

    H_pos, H_neg = [], []  # lists of [D] float tensors

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        questions = [get_question(ex) for ex in batch]
        gts       = [get_answer(ex)   for ex in batch]
        prompts   = [build_cot_prompt(model_type, q, tokenizer) for q in questions]

        for _ in range(n_samples):
            encodings = tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=512
            ).to(device)

            collector.clear()
            collector.register(collect=True, inject=False)

            with torch.no_grad():
                out = model.generate(
                    **encodings,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            collector.remove()

            preds = tokenizer.batch_decode(
                out[:, encodings["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            # Classify traces and store hidden states
            for b_idx, (pred, gt) in enumerate(zip(preds, gts)):
                # Take the hidden state collected just before output tokens
                if collector.collected:
                    # Take last collected hidden state batch-indexed
                    hs = collector.collected[-1]  # [B, D] — last forward call
                    h = hs[min(b_idx, hs.shape[0]-1)].cpu()
                    if answers_match(pred, gt):
                        H_pos.append(h)
                    else:
                        H_neg.append(h)

        if (i // batch_size + 1) % 5 == 0:
            print(f"  [extract] {i + len(batch)}/{len(examples)}  "
                  f"H+={len(H_pos)}  H-={len(H_neg)}")

    if not H_pos or not H_neg:
        print("  [extract] ⚠ Insufficient H+/H- traces; using zero vector.")
        D = model.config.hidden_size
        return torch.zeros(D), torch.tensor(1.0)

    H_pos_t = torch.stack(H_pos)  # [N+, D]
    H_neg_t = torch.stack(H_neg)  # [N-, D]

    v_truth  = H_pos_t.mean(0) - H_neg_t.mean(0)          # [D]
    sigma    = torch.cat([H_pos_t, H_neg_t]).std(0).mean() # scalar
    v_norm   = F.normalize(v_truth.unsqueeze(0), dim=-1).squeeze(0)  # [D] unit vector

    print(f"  [extract] Done  H+={len(H_pos)}  H-={len(H_neg)}")
    print(f"  [extract] |v_truth|={v_truth.norm():.4f}  σ={sigma:.4f}")
    return v_norm, sigma


def load_or_extract_vector(model, tokenizer, model_type, steer_data,
                           collector, device, vector_dir,
                           n_samples, seed):
    """Load cached v_norm/sigma, or extract and save them."""
    vector_dir = pathlib.Path(vector_dir)
    v_path     = vector_dir / "v_truth.pt"
    s_path     = vector_dir / "sigma.pt"

    if v_path.exists() and s_path.exists():
        print(f"  [extract] Loading cached v_truth from {vector_dir}")
        v_norm = torch.load(v_path).to(device)
        sigma  = torch.load(s_path).to(device)
        return v_norm, sigma

    # Extract from scratch
    v_norm, sigma = extract_truth_vector(
        model, tokenizer, model_type,
        steer_data, collector, device,
        n_samples=n_samples, seed=seed,
    )
    vector_dir.mkdir(parents=True, exist_ok=True)
    torch.save(v_norm.cpu(), v_path)
    torch.save(sigma.cpu(),  s_path)
    print(f"  [extract] Saved v_truth → {v_path}")
    return v_norm.to(device), sigma.to(device)


# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_eval(model, tokenizer, model_type, eval_data,
             collector, device, alpha, v_norm, sigma,
             random_noise=False, seed=42, max_new_tokens=512, batch_size=4):
    """
    Evaluate on eval_data with a specific alpha value.
    If random_noise=True, replace v_norm with a random unit vector.
    Returns list of record dicts.
    """
    if random_noise:
        rng = torch.Generator(); rng.manual_seed(seed)
        v_noise = torch.randn_like(v_norm, generator=rng)
        v_applied = F.normalize(v_noise.unsqueeze(0), dim=-1).squeeze(0)
    else:
        v_applied = v_norm

    collector.set_delta(alpha, sigma, v_applied)
    collector.register(collect=True, inject=(alpha != 0.0), v_norm=v_applied, sigma=sigma)

    examples = load_jsonl(eval_data)
    records  = []

    for i in range(0, len(examples), batch_size):
        batch   = examples[i : i + batch_size]
        questions = [get_question(ex) for ex in batch]
        gts       = [get_answer(ex)   for ex in batch]

        from model_registry import get_config
        cfg     = get_config(model_type)
        prompts = [cfg["build_prompt"](q, tokenizer, 1.0) for q in questions]

        encodings = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to(device)

        collector.clear()
        collector._inject = (alpha != 0.0)

        with torch.no_grad():
            out = model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                do_sample=False,        # greedy for deterministic eval
                pad_token_id=tokenizer.pad_token_id,
            )

        preds = tokenizer.batch_decode(
            out[:, encodings["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        cos_mean = (sum(collector.cos_sims) / len(collector.cos_sims)
                    if collector.cos_sims else None)

        for pred, gt in zip(preds, gts):
            records.append({
                "pred":    pred,
                "gt":      gt,
                "correct": answers_match(pred, gt),
                "cos_sim": cos_mean,
            })

        if (i // batch_size + 1) % 10 == 0:
            done  = min(i + batch_size, len(examples))
            sofar = sum(1 for r in records if r["correct"])
            print(f"  [{i+1}/{len(examples)}]  acc so far: {sofar/len(records):.1%}")

    collector.remove()
    collector.clear()
    return records


# ── Flip rate computation ─────────────────────────────────────────────────────

def compute_flip_rate(baseline_records, test_records):
    """How many baseline-wrong examples became correct."""
    n_wrong = sum(1 for r in baseline_records if not r["correct"])
    if n_wrong == 0:
        return None
    n_flipped = sum(
        1 for b, t in zip(baseline_records, test_records)
        if not b["correct"] and t["correct"]
    )
    return n_flipped / n_wrong


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Hook-based latent steering for HF transformer models"
    )
    p.add_argument("--model-path",  required=True)
    p.add_argument("--model-type",  required=True,
                   choices=["phi2", "llama32_3b", "qwen25_3b"])
    p.add_argument("--eval-data",   default=DEFAULT_EVAL_DATA)
    p.add_argument("--steer-data",  default=DEFAULT_STEER_DATA)
    p.add_argument("--vector-dir",  default=DEFAULT_VECTOR_DIR)
    p.add_argument("--out-dir",     default=DEFAULT_OUT_DIR)
    p.add_argument("--condition",   default="steered",
                   choices=["ccot", "random_noise", "steered"],
                   help="Which experimental condition to run")
    p.add_argument("--alphas",      nargs="+", type=float, default=DEFAULT_ALPHAS)
    p.add_argument("--n-samples",   type=int,  default=DEFAULT_N_SAMPLES,
                   help="Stochastic passes per example for vector extraction")
    p.add_argument("--layer-frac",  type=float, default=DEFAULT_LAYER_FRAC,
                   help="Fraction of total layers to select for intervention (0..1)")
    p.add_argument("--batch-size",  type=int,  default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed",        type=int,  default=42)
    args = p.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir   = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 62)
    print("  hidden_steer.py")
    print(f"  model      : {args.model_path}  ({args.model_type})")
    print(f"  condition  : {args.condition}")
    print(f"  eval data  : {args.eval_data}")
    print(f"  out dir    : {out_dir}")
    print(f"  device     : {device}")
    print("=" * 62)

    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    layer_idx = get_hook_layer_idx(model, args.layer_frac)
    print(f"  [hook] Intervening at layer {layer_idx} / {num_layers(model)-1}")

    collector = HiddenStateCollector(model, layer_idx, device)

    # Load or compute truth vector
    v_norm, sigma = load_or_extract_vector(
        model, tokenizer, args.model_type, args.steer_data,
        collector, device,
        vector_dir=pathlib.Path(args.vector_dir) / args.model_type,
        n_samples=args.n_samples, seed=args.seed,
    )

    # ── Run the selected condition ───────────────────────────────────────────
    t0 = time()

    if args.condition == "ccot":
        # CCoT unsteered baseline (alpha=0)
        print("\n  [run] CCoT condition (alpha=0, no intervention) …")
        records = run_eval(
            model, tokenizer, args.model_type,
            args.eval_data, collector, device,
            alpha=0.0, v_norm=v_norm, sigma=sigma,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        accuracy = sum(1 for r in records if r["correct"]) / len(records)
        cos_mean = [r["cos_sim"] for r in records if r.get("cos_sim") is not None]
        cos_mean = sum(cos_mean) / len(cos_mean) if cos_mean else None
        result = {
            "condition":    "ccot",
            "alpha":        0.0,
            "accuracy":     round(accuracy, 6),
            "flip_rate":    None,
            "mean_cos_sim": round(cos_mean, 6) if cos_mean is not None else None,
            "n":            len(records),
        }
        (out_dir / "records.json").write_text(json.dumps(records, indent=2))

    elif args.condition == "random_noise":
        # CCoT with random-noise vector (control for vector direction)
        print("\n  [run] Random-noise condition (alpha=1.0, random unit vector) …")
        # First get ccot baseline for flip rate
        base_records = run_eval(
            model, tokenizer, args.model_type,
            args.eval_data, collector, device,
            alpha=0.0, v_norm=v_norm, sigma=sigma,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        # Then run with random noise
        rn_records = run_eval(
            model, tokenizer, args.model_type,
            args.eval_data, collector, device,
            alpha=1.0, v_norm=v_norm, sigma=sigma,
            random_noise=True,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        accuracy  = sum(1 for r in rn_records if r["correct"]) / len(rn_records)
        flip_rate = compute_flip_rate(base_records, rn_records)
        cos_mean  = [r["cos_sim"] for r in rn_records if r.get("cos_sim") is not None]
        cos_mean  = sum(cos_mean) / len(cos_mean) if cos_mean else None
        result = {
            "condition":    "random_noise",
            "alpha":        1.0,
            "accuracy":     round(accuracy, 6),
            "flip_rate":    round(flip_rate, 6) if flip_rate is not None else None,
            "mean_cos_sim": round(cos_mean, 6) if cos_mean is not None else None,
            "n":            len(rn_records),
        }
        (out_dir / "records.json").write_text(json.dumps(rn_records, indent=2))

    else:
        # Steered: full alpha sweep
        print(f"\n  [run] Steered condition  alphas={args.alphas} …")
        # Baseline (alpha=0) first
        base_records = run_eval(
            model, tokenizer, args.model_type,
            args.eval_data, collector, device,
            alpha=0.0, v_norm=v_norm, sigma=sigma,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        base_acc = sum(1 for r in base_records if r["correct"]) / len(base_records)

        # Always seed sweep_results with the baseline so best_alpha=0.0 is
        # always findable even when 0.0 is not in args.alphas.
        base_cos_vals = [r["cos_sim"] for r in base_records if r.get("cos_sim") is not None]
        base_cos_mean = sum(base_cos_vals) / len(base_cos_vals) if base_cos_vals else None
        sweep_results = [{
            "alpha":        0.0,
            "accuracy":     round(base_acc, 6),
            "flip_rate":    None,
            "mean_cos_sim": round(base_cos_mean, 6) if base_cos_mean is not None else None,
        }]
        best_alpha, best_acc = 0.0, base_acc

        for alpha in sorted(args.alphas):
            if alpha == 0.0:
                # Already in sweep_results; skip re-adding.
                continue
            records = run_eval(
                    model, tokenizer, args.model_type,
                    args.eval_data, collector, device,
                    alpha=alpha, v_norm=v_norm, sigma=sigma,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                )
            accuracy  = sum(1 for r in records if r["correct"]) / len(records)
            flip_rate = compute_flip_rate(base_records, records)

            cos_vals = [r["cos_sim"] for r in records if r.get("cos_sim") is not None]
            cos_mean = sum(cos_vals) / len(cos_vals) if cos_vals else None

            entry = {
                "alpha":        alpha,
                "accuracy":     round(accuracy, 6),
                "flip_rate":    round(flip_rate, 6) if flip_rate is not None else None,
                "mean_cos_sim": round(cos_mean, 6) if cos_mean is not None else None,
            }
            sweep_results.append(entry)

            # Save per-alpha outputs
            alpha_dir = out_dir / f"alpha_{alpha}"
            alpha_dir.mkdir(parents=True, exist_ok=True)
            (alpha_dir / "records.json").write_text(json.dumps(records, indent=2))
            (alpha_dir / "metrics.json").write_text(json.dumps(entry, indent=2))

            if accuracy > best_acc:
                best_acc, best_alpha = accuracy, alpha

            diff = accuracy - base_acc
            cos_s = f"{cos_mean:.4f}" if cos_mean is not None else "N/A"
            print(f"  α={alpha:>4.1f}  acc={accuracy:.2%}  ({diff:+.2%} vs base)  cos={cos_s}")

        # Summary: use best alpha result
        best = next(r for r in sweep_results if r["alpha"] == best_alpha)
        result = {
            "condition":       "steered",
            "best_alpha":      best_alpha,
            "baseline_acc":    round(base_acc, 6),
            "accuracy":        best["accuracy"],
            "flip_rate":       best["flip_rate"],
            "mean_cos_sim":    best["mean_cos_sim"],
            "sweep":           sweep_results,
            "n":               len(base_records),
        }

    elapsed = time() - t0
    result["elapsed_seconds"] = round(elapsed, 1)
    result["model"]     = args.model_type
    result["model_path"] = args.model_path

    # Print summary
    print("\n" + "─" * 50)
    print(f"  Condition   : {result['condition']}")
    print(f"  Accuracy    : {result.get('accuracy', 'N/A'):.2%}"
          if isinstance(result.get("accuracy"), float) else f"  Accuracy : N/A")
    if result.get("flip_rate") is not None:
        print(f"  Flip rate   : {result['flip_rate']:.2%}")
    if result.get("mean_cos_sim") is not None:
        print(f"  Cos sim     : {result['mean_cos_sim']:.4f}")
    print(f"  Time        : {elapsed:.0f}s")
    print("─" * 50)

    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    (out_dir / "condition_info.json").write_text(json.dumps({
        "model_type": args.model_type,
        "model_path": args.model_path,
        "condition":  args.condition,
        "layer_idx":  layer_idx,
        "n_layers":   num_layers(model),
        "layer_frac": args.layer_frac,
    }, indent=2))

    print(f"\n  [hidden_steer] Results → {out_dir}/metrics.json\n")


if __name__ == "__main__":
    main()
