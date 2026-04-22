# model_registry.py
"""
Central registry for all model-specific configuration.
To add a new model, add an entry to MODEL_CONFIGS below — nothing else needs changing.

Each entry defines:
    build_prompt(content, tokenizer, compression_ratio)  ->  str
        Builds the full prompt string for one user message.
    cot_field : str
        Key in the predictions JSONL that holds the raw CoT text (used by LLMLingua).
    stop_sequences(tokenizer) -> list[str]
        Strings that signal end-of-generation.
    cot_split : str
        String used to split off the CoT from the final-answer line when measuring length.
    lora_template : str
        Template name passed to LLaMA-Factory (--template).
"""

from typing import Callable


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _qwen_prompt(content, tokenizer, compression_ratio):
    if compression_ratio < 1.0:
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "Please reason step by step, and put your final answer within \\boxed{}.\n"
            f"{content}<|eot_id|><|eot_id|><|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        f"{content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _llama3_prompt(content, tokenizer, compression_ratio):
    bos = tokenizer.bos_token or ""
    eos = tokenizer.eos_token or ""
    if compression_ratio < 1.0:
        return (
            f"{bos}<|start_header_id|>user<|end_header_id|>\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}.\n"
            f"{content}\n{eos}{eos}{eos}"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return (
        f"{bos}<|start_header_id|>user<|end_header_id|>\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        f"{content}\n{eos}"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _gpt2_prompt(content, tokenizer, compression_ratio):
    """
    GPT-2 has no chat template and no instruction-following training.
    We use a few-shot style prefix so it has a pattern to continue,
    then a clear separator so generation stops at the next "Question:".

    The model has seen Q&A text during pre-training, so this format
    elicits more structured completions than a bare question.
    """
    ratio_hint = f" [compress:{compression_ratio}]" if compression_ratio < 1.0 else ""

    # One-shot example gives GPT-2 the answer format to mimic
    few_shot = (
        "Question: Janet's ducks lay 16 eggs per day. "
        "She eats 3 for breakfast and bakes 4 into muffins. "
        "She sells the rest for $2 each. How much does she make daily?\n"
        "Let's think step by step.\n"
        "Ducks lay 16 eggs. She uses 3+4=7. She sells 16-7=9 eggs. "
        "9*$2=$18. The final answer is: $\\boxed{18}$\n\n"
    )
    return (
        f"{few_shot}"
        f"Question: {content}{ratio_hint}\n"
        "Let's think step by step.\n"
    )


def _gpt2_stop(tokenizer):
    """
    Stop at EOS *or* at the start of a new question.
    Without the second stop string GPT-2 will chain into a new
    self-generated question and never terminate within the token budget.
    """
    stops = []
    if tokenizer.eos_token:
        stops.append(tokenizer.eos_token)
    # These appear naturally in GPT-2 pre-training text and signal a topic break
    stops.extend(["\n\nQuestion:", "\nQuestion:", "Question:"])
    return stops


def _phi2_no_cot_prompt(content, tokenizer, compression_ratio):
    """Phi-2 direct-answer (no reasoning) prompt."""
    return (
        f"Instruct: Answer the following math problem with only the final numeric answer "
        f"inside \\boxed{{}}. Do not show any working.\n"
        f"{content}\n"
        f"Output:"
    )


def _base_no_cot_prompt(content, tokenizer, compression_ratio):
    """Generic direct-answer prompt for base LMs (Llama-3.2-3B, Qwen2.5-3B)."""
    few_shot = (
        "Question: If Janet has 5 apples and eats 2, how many remain?\n"
        "Answer: $\\boxed{3}$\n\n"
    )
    return f"{few_shot}Question: {content}\nAnswer:"


def _phi2_prompt(content, tokenizer, compression_ratio):
    """
    microsoft/phi-2 uses 'Instruct: ... Output:' format.
    For no-CoT mode the system instruction omits step-by-step reasoning.
    """
    ratio_hint = f" [compress:{compression_ratio}]" if compression_ratio < 1.0 else ""
    return (
        f"Instruct: Solve the following math problem and put your final answer "
        f"within \\boxed{{}}. Show step-by-step reasoning.\n"
        f"{content}{ratio_hint}\n"
        f"Output:"
    )


def _phi2_stop(tokenizer):
    stops = []
    if tokenizer.eos_token:
        stops.append(tokenizer.eos_token)
    stops.extend(["Instruct:", "\n\nInstruct"])
    return stops


def _llama32_base_prompt(content, tokenizer, compression_ratio):
    """
    meta-llama/Llama-3.2-3B (base, not instruct).
    Uses a simple few-shot continuation format suited for a base LM.
    """
    ratio_hint = f" [compress:{compression_ratio}]" if compression_ratio < 1.0 else ""
    few_shot = (
        "Question: Janet's ducks lay 16 eggs per day. "
        "She eats 3 for breakfast and bakes 4 into muffins. "
        "She sells the rest for $2 each. How much does she make daily?\n"
        "Reasoning: She uses 3+4=7 eggs. She sells 16-7=9 eggs. 9*$2=$18.\n"
        "Answer: $\\boxed{18}$\n\n"
    )
    return (
        f"{few_shot}"
        f"Question: {content}{ratio_hint}\n"
        "Reasoning:"
    )


def _qwen25_base_prompt(content, tokenizer, compression_ratio):
    """
    Qwen/Qwen2.5-3B (base, not instruct).
    Uses a plain continuation format since there is no chat template.
    """
    ratio_hint = f" [compress:{compression_ratio}]" if compression_ratio < 1.0 else ""
    few_shot = (
        "Question: Janet's ducks lay 16 eggs per day. "
        "She eats 3 for breakfast and bakes 4 into muffins. "
        "She sells the rest for $2 each. How much does she make daily?\n"
        "Solution: She uses 3+4=7 eggs. She sells 16-7=9 eggs. 9×$2=$18.\n"
        "Answer: $\\boxed{18}$\n\n"
    )
    return (
        f"{few_shot}"
        f"Question: {content}{ratio_hint}\n"
        "Solution:"
    )


def _mistral_prompt(content, tokenizer, compression_ratio):
    """Mistral / Mixtral instruct format."""
    bos = tokenizer.bos_token or ""
    ratio_hint = f" [compress:{compression_ratio}]" if compression_ratio < 1.0 else ""
    return (
        f"{bos}[INST] Please reason step by step, and put your final answer within \\boxed{{}}.\n"
        f"{content}{ratio_hint} [/INST]"
    )


def _phi3_prompt(content, tokenizer, compression_ratio):
    """Phi-3 instruct format."""
    ratio_hint = f" [compress:{compression_ratio}]" if compression_ratio < 1.0 else ""
    return (
        "<|user|>\n"
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        f"{content}{ratio_hint}<|end|>\n"
        "<|assistant|>\n"
    )


def _default_stop(tokenizer):
    eos = tokenizer.eos_token
    stops = []
    if eos:
        stops.append(eos)
    return stops


def _llama3_stop(tokenizer):
    return [tokenizer.eos_token or "</s>", "<|eot_id|>"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    # ── Qwen 2.5 Instruct ──────────────────────────────────────────────────
    "qwen": {
        "build_prompt":     _qwen_prompt,
        "cot_field":        "model_output",
        "stop_sequences":   _default_stop,
        "cot_split":        "\n\nThe final answer is:",
        "lora_template":    "qwen",
    },

    # ── LLaMA-3 / LLaMA-3.1 Instruct ───────────────────────────────────────
    "llama3": {
        "build_prompt":     _llama3_prompt,
        "cot_field":        "cot",
        "stop_sequences":   _llama3_stop,
        "cot_split":        "\n\nThe final answer is:",
        "lora_template":    "llama3",
    },

    # ── GPT-2 (and GPT-2 Medium / Large / XL) ──────────────────────────────
    "gpt2": {
        "build_prompt":     _gpt2_prompt,
        "cot_field":        "model_output",
        "stop_sequences":   _gpt2_stop,
        "cot_split":        "\n\nThe final answer is:",
        "lora_template":    "default",
    },

    # ── CODI-GPT2 (zen-E/CODI-gpt2 checkpoint) ─────────────────────────────
    # NOTE: CODI has a completely different inference loop (latent tokens +
    # iterative refinement via LoRA r=128).  It CANNOT go through evaluation.py.
    # Use run_codi.py instead, which clones the CODI repo, downloads the
    # checkpoint, and writes metrics.json in the same format as evaluation.py.
    #
    #   python run_codi.py --run-name codi_gpt2_gsm8k
    #   python compare_metrics.py outputs/gpt2/gsm8k/ outputs/codi_gpt2_gsm8k/
    #
    # This entry exists so all_model_types() lists it and the README is complete.
    # build_prompt / stop_sequences are unused for CODI.
    "codi": {
        "build_prompt":     _gpt2_prompt,   # unused — CODI has its own data loader
        "cot_field":        "model_output",
        "stop_sequences":   _gpt2_stop,     # unused
        "cot_split":        "\n\nThe final answer is:",
        "lora_template":    "default",
        "_custom_runner":   "run_codi.py",  # signals evaluation.py to refuse gracefully
    },

    # ── Mistral / Mixtral Instruct ──────────────────────────────────────────
    "mistral": {
        "build_prompt":     _mistral_prompt,
        "cot_field":        "model_output",
        "stop_sequences":   _default_stop,
        "cot_split":        "\n\nThe final answer is:",
        "lora_template":    "mistral",
    },

    # ── Phi-3 Mini / Medium Instruct ───────────────────────────────────────
    "phi3": {
        "build_prompt":     _phi3_prompt,
        "cot_field":        "model_output",
        "stop_sequences":   _default_stop,
        "cot_split":        "\n\nThe final answer is:",
        "lora_template":    "phi",
    },
    # ── microsoft/phi-2 ───────────────────────────────────────────────────────
    # Use: --model-type phi2 --model-path microsoft/phi-2
    "phi2": {
        "build_prompt":        _phi2_prompt,
        "build_no_cot_prompt": _phi2_no_cot_prompt,
        "cot_field":           "model_output",
        "stop_sequences":      _phi2_stop,
        "cot_split":           "\nOutput:",
        "lora_template":       "phi",
    },

    # ── meta-llama/Llama-3.2-3B (base model) ──────────────────────────────────
    # Use: --model-type llama32_3b --model-path meta-llama/Llama-3.2-3B
    "llama32_3b": {
        "build_prompt":        _llama32_base_prompt,
        "build_no_cot_prompt": _base_no_cot_prompt,
        "cot_field":           "model_output",
        "stop_sequences":      _default_stop,
        "cot_split":           "\nAnswer:",
        "lora_template":       "llama3",
    },

    # ── Qwen/Qwen2.5-3B (base model) ──────────────────────────────────────────
    # Use: --model-type qwen25_3b --model-path Qwen/Qwen2.5-3B
    "qwen25_3b": {
        "build_prompt":        _qwen25_base_prompt,
        "build_no_cot_prompt": _base_no_cot_prompt,
        "cot_field":           "model_output",
        "stop_sequences":      _default_stop,
        "cot_split":           "\nAnswer:",
        "lora_template":       "qwen",
    },

    # ── Qwen/Qwen2.5-1.5B (base model) ────────────────────────────────────────
    # Use: --model-type qwen25_1_5b --model-path Qwen/Qwen2.5-1.5B
    "qwen25_1_5b": {
        "build_prompt":        _qwen25_base_prompt,
        "build_no_cot_prompt": _base_no_cot_prompt,
        "cot_field":           "model_output",
        "stop_sequences":      _default_stop,
        "cot_split":           "\nAnswer:",
        "lora_template":       "qwen",
    },

    # ── Qwen/Qwen2.5-0.5B (base model) ────────────────────────────────────────
    # Use: --model-type qwen25_0_5b --model-path Qwen/Qwen2.5-0.5B
    "qwen25_0_5b": {
        "build_prompt":        _qwen25_base_prompt,
        "build_no_cot_prompt": _base_no_cot_prompt,
        "cot_field":           "model_output",
        "stop_sequences":      _default_stop,
        "cot_split":           "\nAnswer:",
        "lora_template":       "qwen",
    },

    # ── Qwen/Qwen2.5-Math-1.5B (base model) ────────────────────────────────────
    # Use: --model-type qwen_math_1_5b --model-path Qwen/Qwen2.5-Math-1.5B
    "qwen_math_1_5b": {
        "build_prompt":        _qwen25_base_prompt,
        "build_no_cot_prompt": _base_no_cot_prompt,
        "cot_field":           "model_output",
        "stop_sequences":      _default_stop,
        "cot_split":           "\nAnswer:",
        "lora_template":       "qwen",
    },
}


def get_config(model_type: str) -> dict:
    """Return config dict for *model_type*, raising a clear error if unknown."""
    if model_type not in MODEL_CONFIGS:
        known = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Known types: {known}. "
            f"Add a new entry to MODEL_CONFIGS in model_registry.py to support it."
        )
    return MODEL_CONFIGS[model_type]


def all_model_types() -> list:
    """Return sorted list of all registered model type keys."""
    return sorted(MODEL_CONFIGS.keys())