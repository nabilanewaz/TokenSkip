"""
utils/model_registry.py
------------------------
Central registry for all model-specific configuration.

To add a new model: add one entry to MODEL_CONFIGS. Nothing else changes.

Each entry must provide:
    build_prompt(content, tokenizer, compression_ratio) -> str
        Build the full prompt string for one question.
    build_no_cot_prompt(content, tokenizer, compression_ratio) -> str  [optional]
        Prompt for the "No CoT" evaluation condition.
    stop_sequences(tokenizer) -> list[str]
        Strings that signal end-of-generation.
    cot_split : str
        Separator used to split CoT from the final-answer token.
    lora_template : str
        Template name for LLaMA-Factory fine-tuning (if used).
"""

from __future__ import annotations


# ── Prompt builders ───────────────────────────────────────────────────────────

def _phi2_prompt(content, tokenizer, compression_ratio=1.0):
    """microsoft/phi-2 uses 'Instruct: ... Output:' format."""
    ratio_hint = f" [compress:{compression_ratio}]" if compression_ratio < 1.0 else ""
    return (
        "Instruct: Solve the following math problem and put your final answer "
        "within \\boxed{}. Show step-by-step reasoning.\n"
        f"{content}{ratio_hint}\n"
        "Output:"
    )


def _phi2_no_cot_prompt(content, tokenizer, compression_ratio=1.0):
    """Phi-2 direct-answer (no reasoning) prompt."""
    return (
        "Instruct: Answer the following math problem with only the final "
        "numeric answer inside \\boxed{}. Do not show any working.\n"
        f"{content}\n"
        "Output:"
    )


def _phi2_stop(tokenizer):
    stops = []
    if tokenizer.eos_token:
        stops.append(tokenizer.eos_token)
    stops.extend(["Instruct:", "\n\nInstruct"])
    return stops


def _llama32_base_prompt(content, tokenizer, compression_ratio=1.0):
    """
    meta-llama/Llama-3.2-3B (base, not instruct).
    Uses a few-shot continuation format suited for a base LM.
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


def _base_no_cot_prompt(content, tokenizer, compression_ratio=1.0):
    """Generic direct-answer prompt for base LMs (Llama-3.2-3B, Qwen2.5-xB)."""
    few_shot = (
        "Question: If Janet has 5 apples and eats 2, how many remain?\n"
        "Answer: $\\boxed{3}$\n\n"
    )
    return f"{few_shot}Question: {content}\nAnswer:"


def _qwen25_base_prompt(content, tokenizer, compression_ratio=1.0):
    """
    Qwen/Qwen2.5-xB (base, not instruct).
    Plain continuation format — no chat template.
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


def _default_stop(tokenizer):
    return [tokenizer.eos_token] if tokenizer.eos_token else []


# ── Registry ──────────────────────────────────────────────────────────────────

MODEL_CONFIGS: dict[str, dict] = {

    # ── microsoft/phi-2 ───────────────────────────────────────────────────────
    "phi2": {
        "build_prompt":        _phi2_prompt,
        "build_no_cot_prompt": _phi2_no_cot_prompt,
        "cot_field":           "model_output",
        "stop_sequences":      _phi2_stop,
        "cot_split":           "\nOutput:",
        "lora_template":       "phi",
    },

    # ── meta-llama/Llama-3.2-3B (base model) ──────────────────────────────────
    "llama32_3b": {
        "build_prompt":        _llama32_base_prompt,
        "build_no_cot_prompt": _base_no_cot_prompt,
        "cot_field":           "model_output",
        "stop_sequences":      _default_stop,
        "cot_split":           "\nAnswer:",
        "lora_template":       "llama3",
    },

    # ── Qwen/Qwen2.5-3B (base model) ──────────────────────────────────────────
    "qwen25_3b": {
        "build_prompt":        _qwen25_base_prompt,
        "build_no_cot_prompt": _base_no_cot_prompt,
        "cot_field":           "model_output",
        "stop_sequences":      _default_stop,
        "cot_split":           "\nAnswer:",
        "lora_template":       "qwen",
    },

    # ── Qwen/Qwen2.5-1.5B (base model) ────────────────────────────────────────
    "qwen25_1_5b": {
        "build_prompt":        _qwen25_base_prompt,
        "build_no_cot_prompt": _base_no_cot_prompt,
        "cot_field":           "model_output",
        "stop_sequences":      _default_stop,
        "cot_split":           "\nAnswer:",
        "lora_template":       "qwen",
    },

    # ── Qwen/Qwen2.5-0.5B (base model) ────────────────────────────────────────
    "qwen25_0_5b": {
        "build_prompt":        _qwen25_base_prompt,
        "build_no_cot_prompt": _base_no_cot_prompt,
        "cot_field":           "model_output",
        "stop_sequences":      _default_stop,
        "cot_split":           "\nAnswer:",
        "lora_template":       "qwen",
    },
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_config(model_type: str) -> dict:
    """Return the config dict for *model_type*, raising a clear error if unknown."""
    if model_type not in MODEL_CONFIGS:
        known = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Known types: {known}. "
            f"Add a new entry to MODEL_CONFIGS in utils/model_registry.py."
        )
    return MODEL_CONFIGS[model_type]


def all_model_types() -> list[str]:
    """Return sorted list of all registered model type keys."""
    return sorted(MODEL_CONFIGS.keys())


# ── HF model ID look-up ───────────────────────────────────────────────────────

HF_IDS: dict[str, str] = {
    "phi2":       "microsoft/phi-2",
    "llama32_3b": "meta-llama/Llama-3.2-3B",
    "qwen25_3b":  "Qwen/Qwen2.5-3B",
    "qwen25_1_5b":"Qwen/Qwen2.5-1.5B",
    "qwen25_0_5b":"Qwen/Qwen2.5-0.5B",
}


def get_hf_id(model_type: str) -> str:
    """Return the HuggingFace model ID for *model_type*."""
    if model_type not in HF_IDS:
        raise ValueError(f"No HF ID registered for '{model_type}'.")
    return HF_IDS[model_type]
