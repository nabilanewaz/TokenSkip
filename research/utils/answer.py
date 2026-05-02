"""
utils/answer.py
---------------
Numeric answer extraction from model outputs and GSM8K ground-truth strings.

GSM8K ground-truth format:
    "... some reasoning ... #### 42"

Model output formats handled (in priority order):
    1. #### <number>               (GSM8K-style)
    2. "answer is: <number>"
    3. \\boxed{<number>}           (LaTeX)
    4. = <number>  at line end
    5. last number in text         (fallback)
    6. direct float parse          (clean numeric string)
"""

from __future__ import annotations

import re


# ── Core extraction ───────────────────────────────────────────────────────────

_PATTERNS = [
    r"answer is:?\s*([-+]?\d+\.?\d*)",
    r"####\s*([-+]?\d+\.?\d*)",
    r"\$?\\boxed\{([-+]?\d+\.?\d*)\}",
    r"=\s*([-+]?\d+\.?\d*)\s*$",
]


def extract_answer_number(text: str) -> float | None:
    """
    Extract the numeric answer from *text*.

    Returns a float, or None if no number can be parsed.
    """
    text = str(text).strip().replace(",", "")

    # Fast path: GSM8K ground-truth "#### <num>"
    if "####" in text:
        text = text.split("####")[-1].strip()

    for pat in _PATTERNS:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    # Fallback: last number in text
    nums = re.findall(r"[-+]?\d+\.?\d*", text)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass

    # Final fallback: direct parse
    try:
        return float(text.strip())
    except ValueError:
        return None


def answers_match(pred_text: str, gt_text: str, tol: float = 1e-4) -> bool:
    """
    Return True iff the numeric answers in *pred_text* and *gt_text* agree
    within *tol*.
    """
    pa = extract_answer_number(pred_text)
    ga = extract_answer_number(gt_text)
    if pa is None or ga is None:
        return False
    return abs(pa - ga) <= tol


# ── GSM8K example field helpers ───────────────────────────────────────────────

def get_gt_answer(example: dict) -> str:
    """
    Return the ground-truth answer string from a GSM8K example dict.
    Strips the leading reasoning text, keeping only the "#### <num>" part
    (or the raw answer string if "####" is absent).
    """
    raw = example.get("answer", "")
    if "####" in raw:
        return raw.split("####")[-1].strip()
    return raw.strip()


def get_question(example: dict) -> str:
    """Return the question string from a GSM8K example dict."""
    # Support messages-format (some datasets wrap in chat turns)
    if "messages" in example:
        for msg in example["messages"]:
            if msg.get("role") == "user":
                return msg["content"]
    return example.get("question", "")
