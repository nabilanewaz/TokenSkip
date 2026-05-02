"""
data/split_dataset.py
---------------------
Splits GSM8K into the exact partitions defined in the research protocol:
"Steering Continuous Reasoning via Latent Intervention"

FULL MODE (default, --full flag or full_mode: true in protocol.yaml)
─────────────────────────────────────────────────────────────────────
GSM8K train pool (7,473 examples) split 60/10/30:

  llm_train     4,483   Base Training     — Train the CCoT model        (D_base)
  steer_train     747   Vector Extraction — Compute v_truth              (D_steer)
  validation    2,243   Validation        — Tune alpha, never Phase 2   (D_val)
  ──────────────────────────────────────────────────────────────────────────────
  TRAIN POOL    7,473
  test          1,319   Full GSM8K test.jsonl — held-out, NEVER touched  (D_test)

MINI-500 MODE (--mini flag, for fast local development)
────────────────────────────────────────────────────────
  llm_train      300
  steer_train     50
  validation     100
  test            50    Carved from the combined pool (no separate test.jsonl used)
  ──────────────────────────────────────────────────────────────────────────────
  TOTAL          500

CRITICAL RULE (protocol §Dataset Preparation):
  • D_test comes from the ORIGINAL GSM8K test.jsonl — never modified.
  • D_steer must NOT overlap with D_train or D_val.
  • All splits are deterministic given the same seed (default: 42).

Usage
─────
    # From the repo root (full protocol, recommended)
    python research/data/split_dataset.py --full

    # Mini-500 for quick smoke tests
    python research/data/split_dataset.py --mini

    # Print sizes without writing files
    python research/data/split_dataset.py --full --stats

    # Force download from HuggingFace
    python research/data/split_dataset.py --full --hf

Outputs
───────
    datasets/gsm8k_split/llm_train.jsonl
    datasets/gsm8k_split/steer_train.jsonl
    datasets/gsm8k_split/validation.jsonl
    datasets/gsm8k_split/test.jsonl
    datasets/gsm8k_split/split_config.json   ← metadata / audit trail
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

# ── Allow running from any working directory ──────────────────────────────────
_RESEARCH_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RESEARCH_ROOT))

from utils.io import load_jsonl, save_jsonl, load_config


# ── Defaults (overridden by protocol.yaml if present) ────────────────────────
_CFG = load_config()
_DS  = _CFG.get("dataset", {})

DEFAULT_SOURCE = _DS.get("source_dir", "datasets/gsm8k")
DEFAULT_OUT    = _DS.get("out_dir",    "datasets/gsm8k_split")
SEED           = _DS.get("seed",       42)

_FULL   = _DS.get("splits",       {})
_MINI   = _DS.get("mini_splits",  {})

N_LLM_TRAIN_FULL   = _FULL.get("llm_train",   4483)
N_STEER_TRAIN_FULL = _FULL.get("steer_train",  747)
N_VALIDATION_FULL  = _FULL.get("validation",  2243)
N_TEST_FULL        = _FULL.get("test",         1319)

N_LLM_TRAIN_MINI   = _MINI.get("llm_train",    300)
N_STEER_TRAIN_MINI = _MINI.get("steer_train",   50)
N_VALIDATION_MINI  = _MINI.get("validation",   100)
N_TEST_MINI        = _MINI.get("test",           50)

EXPECTED_TRAIN_POOL = N_LLM_TRAIN_FULL + N_STEER_TRAIN_FULL + N_VALIDATION_FULL  # 7473


# ── HuggingFace download ──────────────────────────────────────────────────────

def load_gsm8k_hf() -> tuple[list[dict], list[dict]]:
    """Download GSM8K from HuggingFace; returns (train_list, test_list)."""
    from datasets import load_dataset  # type: ignore
    ds    = load_dataset("gsm8k", "main")
    train = [
        {"question": r["question"], "answer": r["answer"], "split": "original_train"}
        for r in ds["train"]
    ]
    test  = [
        {"question": r["question"], "answer": r["answer"], "split": "original_test"}
        for r in ds["test"]
    ]
    print(f"  HuggingFace: {len(train)} train + {len(test)} test = {len(train)+len(test)} total")
    return train, test


def load_from_local(source_dir: pathlib.Path) -> tuple[list[dict], list[dict]]:
    """
    Load GSM8K from local JSONL files.  Falls back to HuggingFace if missing.
    Returns (train_list, test_list).
    """
    train_path = source_dir / "train.jsonl"
    test_path  = source_dir / "test.jsonl"

    if not train_path.exists():
        print(f"  Local files not found at {source_dir} — trying HuggingFace …")
        return load_gsm8k_hf()

    train = load_jsonl(train_path)
    for ex in train:
        ex.setdefault("split", "original_train")

    test: list[dict] = []
    if test_path.exists():
        test = load_jsonl(test_path)
        for ex in test:
            ex.setdefault("split", "original_test")

    print(f"  Local: {len(train)} train + {len(test)} test from {source_dir}")
    return train, test


# ── Splitting logic ───────────────────────────────────────────────────────────

def make_splits(
    train_pool: list[dict],
    seed: int,
    n_llm: int,
    n_steer: int,
    n_val: int,
    *,
    separate_test: list[dict] | None = None,
    n_test: int | None = None,
) -> dict[str, list[dict]]:
    """
    Shuffle *train_pool* deterministically and carve exact-count splits.

    If *separate_test* is provided (full-protocol mode), the test split is
    taken verbatim from that list and *n_test* is ignored.
    Otherwise *n_test* examples are taken from the tail of the pool (mini mode).
    """
    rng      = random.Random(seed)
    shuffled = train_pool.copy()
    rng.shuffle(shuffled)

    if separate_test is not None:
        cap = n_llm + n_steer + n_val
        if len(shuffled) < cap:
            sys.exit(
                f"[split] ✗ Train pool too small: need {cap}, got {len(shuffled)}"
            )
        shuffled = shuffled[:cap]
        test     = separate_test
    else:
        # Mini mode: carve test from the tail
        if n_test is None:
            n_test = len(shuffled) - n_llm - n_steer - n_val
        shuffled = shuffled[: n_llm + n_steer + n_val + n_test]
        test     = shuffled[n_llm + n_steer + n_val:]

    return {
        "llm_train":   shuffled[:n_llm],
        "steer_train": shuffled[n_llm : n_llm + n_steer],
        "validation":  shuffled[n_llm + n_steer : n_llm + n_steer + n_val],
        "test":        test,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_stats(splits: dict[str, list], mode_label: str) -> None:
    total = sum(len(v) for v in splits.values())
    rows  = [
        ("llm_train",   "Base Training (D_base)   — Phase 1 fine-tuning"),
        ("steer_train", "Vector Extraction (D_steer) — Phase 2 v_truth"),
        ("validation",  "Validation (D_val)        — alpha tuning"),
        ("test",        "Held-out Test (D_test)    — final evaluation"),
    ]
    print(f"\n  Mode : {mode_label}")
    print(f"  {'Split':<14} {'N':>6}  {'%':>7}  Purpose")
    print("  " + "─" * 70)
    for name, purpose in rows:
        n   = len(splits[name])
        pct = 100.0 * n / total if total else 0
        print(f"  {name:<14} {n:>6}  {pct:>6.1f}%  {purpose}")
    print("  " + "─" * 70)
    print(f"  {'TOTAL':<14} {total:>6}")
    print()
    print("  CRITICAL RULE: D_steer is the ONLY split used in Phase 2.")
    print("  D_test (test.jsonl) is NEVER touched in Phase 1 or Phase 2.\n")


def write_config(
    out_dir: pathlib.Path,
    splits: dict[str, list],
    seed: int,
    mode_label: str,
) -> None:
    total = sum(len(v) for v in splits.values())
    cfg   = {
        "protocol": "Steering Continuous Reasoning via Latent Intervention",
        "seed":     seed,
        "mode":     mode_label,
        "total":    total,
        "splits": {
            name: {
                "path":            str(out_dir / f"{name}.jsonl"),
                "n":               len(data),
                "pct":             round(100.0 * len(data) / total, 2),
                "protocol_label":  {
                    "llm_train":   "D_base  (Base Training)",
                    "steer_train": "D_steer (Vector Extraction — Phase 2 only)",
                    "validation":  "D_val   (Validation / alpha tuning)",
                    "test":        "D_test  (Held-out — NEVER touched in Phase 1/2)",
                }.get(name, name),
            }
            for name, data in splits.items()
        },
    }
    path = out_dir / "split_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"  wrote config → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split GSM8K per research protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE,
                        help=f"Directory with train.jsonl / test.jsonl (default: {DEFAULT_SOURCE})")
    parser.add_argument("--out-dir",    default=DEFAULT_OUT,
                        help=f"Output directory for split files (default: {DEFAULT_OUT})")
    parser.add_argument("--seed",       type=int, default=SEED)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true",
                      help="Full GSM8K protocol (7,473 train + 1,319 test) [default]")
    mode.add_argument("--mini", action="store_true",
                      help="Mini-500 subset (fast dev/smoke tests)")
    parser.add_argument("--stats", action="store_true",
                        help="Print split sizes and exit without writing files")
    parser.add_argument("--hf",   action="store_true",
                        help="Force download from HuggingFace (skip local files)")
    args = parser.parse_args()

    # Default to full mode if neither flag is given
    use_full = not args.mini

    if use_full:
        n_llm, n_steer, n_val = N_LLM_TRAIN_FULL, N_STEER_TRAIN_FULL, N_VALIDATION_FULL
        mode_label = (
            f"FULL  (train pool {EXPECTED_TRAIN_POOL}: "
            f"llm={n_llm} steer={n_steer} val={n_val}  +  test={N_TEST_FULL} separate)"
        )
    else:
        n_llm, n_steer, n_val = N_LLM_TRAIN_MINI, N_STEER_TRAIN_MINI, N_VALIDATION_MINI
        mode_label = f"MINI-500  (target {N_LLM_TRAIN_MINI+N_STEER_TRAIN_MINI+N_VALIDATION_MINI+N_TEST_MINI} examples)"

    source_dir = pathlib.Path(args.source_dir)
    out_dir    = pathlib.Path(args.out_dir)

    print(f"\nGSM8K Dataset Splitter")
    print(f"  mode   : {mode_label}")
    print(f"  source : {source_dir}")
    print(f"  output : {out_dir}")
    print(f"  seed   : {args.seed}\n")

    if args.hf:
        train_pool, test_set = load_gsm8k_hf()
    else:
        train_pool, test_set = load_from_local(source_dir)

    if use_full:
        if not test_set:
            # Try loading test.jsonl directly
            tp = source_dir / "test.jsonl"
            if tp.exists():
                test_set = load_jsonl(tp)
                for ex in test_set:
                    ex.setdefault("split", "original_test")
                print(f"  Loaded {len(test_set)} test examples from {tp}")
            else:
                sys.exit("[split] ✗ test.jsonl not found — run with --hf to download.")

        # For full mode: train_pool must be only the train examples
        train_only = [ex for ex in train_pool if ex.get("split") != "original_test"]
        splits = make_splits(
            train_only, args.seed, n_llm, n_steer, n_val,
            separate_test=test_set,
        )
    else:
        # Mini mode: combine everything, carve test from the shuffled pool
        combined = [ex for ex in (train_pool + test_set)]
        splits   = make_splits(
            combined, args.seed, n_llm, n_steer, n_val,
            n_test=N_TEST_MINI,
        )

    print_stats(splits, mode_label)

    if args.stats:
        print("--stats mode: no files written.")
        return

    print("Writing split files …")
    for name, subset in splits.items():
        save_jsonl(subset, out_dir / f"{name}.jsonl")

    write_config(out_dir, splits, args.seed, mode_label)

    print(f"\nDone.  All splits in: {out_dir}/")
    print("\nNext steps:")
    print(f"  Phase 1 (train) : python research/phase1/train.py")
    print(f"  Phase 2 (vector): python research/phase2/extract_vector.py")
    print(f"  Phase 3 (steer) : python research/phase3/steer_inference.py  [CODI-GPT2]")
    print(f"                    python research/phase3/hidden_steer.py       [HF models]")
    print(f"  Phase 4 (eval)  : python research/eval/compare_all.py")


if __name__ == "__main__":
    main()
