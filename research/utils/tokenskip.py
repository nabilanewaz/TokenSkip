"""
utils/tokenskip.py
------------------
Core TokenSkip compression utilities shared across Phase 3 scripts.

Two compression surfaces:
  1. TEXT CoT  — LLMLingua-2 scores token importance, drops low-importance tokens
  2. LATENT    — Δ-norm scores each of k latent steps, skips redundant ones

The LLMLingua model is loaded lazily and cached in module scope (one-time
~1 GB download to HF cache; subsequent calls are instant).
"""
from __future__ import annotations
import pathlib
from typing import Optional

import torch
import torch.nn.functional as F

# ── Module-level LLMLingua cache ──────────────────────────────────────────────
_LLMLINGUA_MODEL: Optional[object] = None
_LLMLINGUA_NAME:  Optional[str]    = None


def _normalize_llmlingua_model_name(model_name: str) -> str:
    """Normalize legacy LLMLingua repo ids to HF canonical ids."""
    if model_name == "llmlingua-2-xlm-roberta-large-meetingbank":
        return "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
    return model_name


def get_llmlingua(model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"):
    """
    Lazy-load and cache a LLMLingua-2 PromptCompressor.
    Safe to call multiple times — returns the cached instance.
    """
    if model_name == "llmlingua-2-xlm-roberta-large-meetingbank":
        model_name = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
    global _LLMLINGUA_MODEL, _LLMLINGUA_NAME
    if _LLMLINGUA_MODEL is not None and _LLMLINGUA_NAME == model_name:
        return _LLMLINGUA_MODEL
    try:
        from llmlingua import PromptCompressor  # type: ignore
    except ImportError:
        raise ImportError(
            "LLMLingua is required for TokenSkip: pip install llmlingua"
        )
    print(f"  [tokenskip] Loading LLMLingua-2 compressor: {model_name} …")
    try:
        _LLMLINGUA_MODEL = PromptCompressor(model_name=model_name, use_llmlingua2=True)
        _LLMLINGUA_NAME = model_name
        print("  [tokenskip] ✓ Compressor ready")
        return _LLMLINGUA_MODEL
    except Exception as e:
        raise RuntimeError(f"Failed to load LLMLingua model '{model_name}': {e}") from e


# ── Text CoT compression ──────────────────────────────────────────────────────

def compress_cot_text(
    cot_text: str,
    ratio: float,
    model_type: str = "phi2",
    llmlingua_model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    force_reserve_digits: bool = True,
) -> dict:
    """
    Compress *cot_text* using LLMLingua-2 at target ratio.

    Returns a dict with:
        compressed_cot     : str   — compressed text
        original_tokens    : int
        compressed_tokens  : int
        actual_ratio       : float — tokens_after / tokens_before
    """
    if ratio >= 1.0 or not cot_text.strip():
        return {
            "compressed_cot":    cot_text,
            "original_tokens":   len(cot_text.split()),
            "compressed_tokens": len(cot_text.split()),
            "actual_ratio":      1.0,
        }

    try:
        compressor = get_llmlingua(llmlingua_model_name)
    except Exception as e:
        print(f"  [tokenskip] ⚠ Cannot load LLMLingua ({e}); returning original")
        return {
            "compressed_cot":    cot_text,
            "original_tokens":   len(cot_text.split()),
            "compressed_tokens": len(cot_text.split()),
            "actual_ratio":      1.0,
        }

    kwargs: dict = {"rate": ratio, "force_reserve_digit": force_reserve_digits}
    # Llama-3 benefits from keeping step markers
    if model_type in ("llama32_3b", "llama3"):
        kwargs["force_tokens"]       = ["Step", ":"]
        kwargs["drop_consecutive"]   = True

    try:
        result = compressor.compress_prompt(cot_text, **kwargs)
        return {
            "compressed_cot":    result["compressed_prompt"],
            "original_tokens":   result.get("origin_tokens",   len(cot_text.split())),
            "compressed_tokens": result.get("compressed_tokens",len(result["compressed_prompt"].split())),
            "actual_ratio":      result.get("rate", ratio),
        }
    except Exception as e:
        print(f"  [tokenskip] ⚠ LLMLingua failed ({e}), returning original")
        return {
            "compressed_cot":    cot_text,
            "original_tokens":   len(cot_text.split()),
            "compressed_tokens": len(cot_text.split()),
            "actual_ratio":      1.0,
        }


def batch_compress(
    cot_texts: list[str],
    ratio: float,
    model_type: str = "phi2",
    llmlingua_model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    device=None,
) -> list[dict]:
    """Compress a batch of CoT strings. Returns one result dict per string.

    `device` is accepted for backward compatibility and intentionally ignored
    because LLMLingua PromptCompressor manages device placement internally.
    """
    return [
        compress_cot_text(t, ratio, model_type, llmlingua_model_name)
        for t in cot_texts
    ]


# ── Latent step importance scoring ────────────────────────────────────────────

def score_latent_steps(latents: torch.Tensor) -> torch.Tensor:
    """
    Score each latent step by its Δ-norm (change from previous step).

    Args:
        latents : [k, D]  — k latent vectors from CODI reasoning loop

    Returns:
        scores  : [k]     — importance score per step (higher = more informative)
                            step 0 always gets the max score (no prior to compare to)
    """
    k = latents.shape[0]
    if k == 1:
        return torch.ones(1)

    deltas = torch.zeros(k)
    deltas[0] = float("inf")                           # first step always kept
    for t in range(1, k):
        deltas[t] = (latents[t] - latents[t-1]).norm()
    return deltas


def apply_latent_skip(
    latents: torch.Tensor,
    keep_ratio: float,
) -> tuple[torch.Tensor, list[int]]:
    """
    Select the top (keep_ratio × k) most informative latent steps.

    Args:
        latents    : [k, D]  — full latent sequence
        keep_ratio : float   — fraction of steps to keep  (1.0 = keep all)

    Returns:
        kept_latents : [k', D]  — sparse latent sequence (k' = ceil(ratio * k))
        kept_indices : [k']     — original step indices that were kept
    """
    k = latents.shape[0]
    if keep_ratio >= 1.0:
        return latents, list(range(k))

    n_keep = max(1, round(keep_ratio * k))
    scores = score_latent_steps(latents)
    _, top_idx = scores.topk(n_keep)
    kept_indices = sorted(top_idx.tolist())
    return latents[kept_indices], kept_indices


def latent_skip_stats(k: int, kept_indices: list[int]) -> dict:
    """Return a summary dict for the latent-skip operation."""
    return {
        "k_total":       k,
        "k_kept":        len(kept_indices),
        "kept_ratio":    round(len(kept_indices) / k, 4),
        "skipped_steps": [i for i in range(k) if i not in kept_indices],
    }