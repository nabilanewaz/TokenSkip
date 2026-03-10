"""
split_dataset.py
----------------
Splits GSM8K into the exact partition defined in the research protocol:
"Steering Continuous Reasoning via Latent Intervention"

MINI-500 MODE (default)
-----------------------
Fast experimentation subset (500 examples, proportional splits):

  llm_train       300   Base Training     — Train the base CCoT / CODI model
  steer_train      50   Vector Extraction — Compute v_truth (Dsteer)
  validation      100   Further Training  — Final hypothesis testing (Dfinal_train)
  test             50   Testing           — Held-out, never used for training
  ──────────────────
  TOTAL           500

FULL MODE (--full)
------------------
Full GSM8K protocol splits (7,473 train pool + 1,319 separate test set):

  llm_train     4,483   Base Training     — 60% of 7,473 train pool  (Dbase)
  steer_train     747   Vector Extraction — 10% of 7,473 train pool  (Dsteer)
  validation    2,243   Further Training  —  30% of 7,473 train pool (Dfinal_train)
  ────────────────────
  TRAIN POOL    7,473
  test          1,319   Full GSM8K test.jsonl — held-out, never touched during training

Critical rule (from protocol):
  validation (Dfinal_train) must NEVER be used to compute the truth vector.
  Only steer_train (Dsteer) is used for Phase 2.

All splits are deterministic given the same seed (default 42).

Usage
-----
    python split_dataset.py                  # mini-500 subset (default)
    python split_dataset.py --full           # full protocol (7473 train + 1319 test)
    python split_dataset.py --seed 123
    python split_dataset.py --stats          # print sizes only, no files written
    python split_dataset.py --hf             # force HuggingFace download
"""

import argparse, json, pathlib, random, sys


DEFAULT_SOURCE = "datasets/gsm8k"
DEFAULT_OUT    = "datasets/gsm8k_split"
SEED           = 42

# ── Mini-500 counts (fast experimentation) ────────────────────────────────────
N_LLM_TRAIN_MINI   = 300
N_STEER_TRAIN_MINI =  50
N_VALIDATION_MINI  = 100
N_TEST_MINI        =  50
TOTAL_MINI         = N_LLM_TRAIN_MINI + N_STEER_TRAIN_MINI + N_VALIDATION_MINI + N_TEST_MINI  # 500

# ── Full-dataset counts (thesis protocol: 60/10/30 of 7473 train pool) ──────────
N_LLM_TRAIN_FULL   = 4483   # 60% of 7473
N_STEER_TRAIN_FULL =  747   # 10% of 7473
N_VALIDATION_FULL  = 2243   # 30% of 7473
N_TEST_FULL        = 1319   # full GSM8K test.jsonl — kept separate (not from train pool)
EXPECTED_TRAIN_POOL = N_LLM_TRAIN_FULL + N_STEER_TRAIN_FULL + N_VALIDATION_FULL  # 7473
EXPECTED_TOTAL_FULL = EXPECTED_TRAIN_POOL + N_TEST_FULL  # 8792


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  wrote {len(data):>5} examples → {path}")

def load_gsm8k_hf():
    from datasets import load_dataset
    ds    = load_dataset("gsm8k", "main")
    train = [{"question": r["question"], "answer": r["answer"], "split": "original_train"} for r in ds["train"]]
    test  = [{"question": r["question"], "answer": r["answer"], "split": "original_test"}  for r in ds["test"]]
    combined = train + test
    print(f"  Loaded {len(train)} train + {len(test)} test = {len(combined)} total from HuggingFace")
    return combined

def load_from_source(source_dir, train_only=False):
    train_path = source_dir / "train.jsonl"
    test_path  = source_dir / "test.jsonl"
    if train_path.exists():
        train = load_jsonl(train_path)
        for ex in train: ex.setdefault("split", "original_train")
        if train_only:
            print(f"  Loaded {len(train)} train examples from {source_dir}")
            return train, []
        if test_path.exists():
            test = load_jsonl(test_path)
            for ex in test: ex.setdefault("split", "original_test")
            combined = train + test
            print(f"  Loaded {len(train)} train + {len(test)} test = {len(combined)} total from {source_dir}")
            return combined, test
    print(f"  Local files not found at {source_dir} — trying HuggingFace...")
    hf = load_gsm8k_hf()
    test = [ex for ex in hf if ex.get("split") == "original_test"]
    return hf, test


# ── Splitting logic ────────────────────────────────────────────────────────────

def split(data, seed, n_llm, n_steer, n_val, n_test=None, separate_test=None):
    """
    Shuffle data deterministically, then carve out exact counts.
    If separate_test is provided (list), it is used as-is for the test split
    and n_test is ignored.  Otherwise n_test examples are taken from data.
    """
    rng = random.Random(seed)
    shuffled = data.copy()
    rng.shuffle(shuffled)

    if separate_test is not None:
        # Full-protocol: train pool only — test comes from a separate source
        cap = n_llm + n_steer + n_val
        shuffled = shuffled[:cap]
        assert len(shuffled) == cap, (
            f"Train pool too small: need {cap}, got {len(shuffled)}")
        test = separate_test
    else:
        total = len(data)
        if n_test is None:
            n_test = total - n_llm - n_steer - n_val
        cap = n_llm + n_steer + n_val + n_test
        shuffled = shuffled[:cap]
        test = shuffled[n_llm + n_steer + n_val:]

    llm_train   = shuffled[:n_llm]
    steer_train = shuffled[n_llm : n_llm + n_steer]
    validation  = shuffled[n_llm + n_steer : n_llm + n_steer + n_val]

    return {
        "llm_train":   llm_train,
        "steer_train": steer_train,
        "validation":  validation,
        "test":        test,
    }


def print_stats(splits, mode_label):
    total = sum(len(v) for v in splits.values())
    rows = [
        ("llm_train",   "Base Training    — CODI/CCoT fine-tuning              (Dbase)"),
        ("steer_train", "Vector Extraction— v_truth computation                (Dsteer)"),
        ("validation",  "Further Training — hypothesis testing                 (Dfinal_train)"),
        ("test",        "Held-out Testing — never used during training"),
    ]
    print(f"\n  Mode: {mode_label}")
    print(f"  {'Split':<14} {'N':>6}  {'% of total':>10}  Purpose")
    print("  " + "─" * 75)
    for name, purpose in rows:
        n   = len(splits[name])
        pct = 100.0 * n / total
        print(f"  {name:<14} {n:>6}  {pct:>9.1f}%  {purpose}")
    print("  " + "─" * 75)
    print(f"  {'TOTAL':<14} {total:>6}  {'100.0%':>10}")
    print()
    print("  Critical rule: steer_train (Dsteer) is the ONLY split used to")
    print("  compute v_truth. validation (Dfinal_train) must never touch Phase 2.\n")


def write_config(out_dir, splits, seed, mode_label):
    total = sum(len(v) for v in splits.values())
    config = {
        "seed": seed,
        "mode": mode_label,
        "total": total,
        "protocol": "Steering Continuous Reasoning via Latent Intervention",
        "splits": {
            name: {
                "path": str(out_dir / f"{name}.jsonl"),
                "n":    len(data),
                "pct":  round(100.0 * len(data) / total, 2),
                "protocol_label": {
                    "llm_train":   "Dbase  (Base Training)",
                    "steer_train": "Dsteer (Vector Extraction)",
                    "validation":  "Dfinal_train (Further Training)",
                    "test":        "Deval  (Held-out Testing)",
                }.get(name, name),
            }
            for name, data in splits.items()
        },
        "exact_counts": {
            "llm_train":   len(splits["llm_train"]),
            "steer_train": len(splits["steer_train"]),
            "validation":  len(splits["validation"]),
            "test":        len(splits["test"]),
        },
        "critical_rule": (
            "steer_train (Dsteer) is the only split used to compute v_truth. "
            "validation (Dfinal_train) must never be used in Phase 2."
        ),
    }
    cfg_path = out_dir / "split_config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(config, indent=2))
    print(f"  wrote config      → {cfg_path}")
    return config


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Split GSM8K per research protocol — mini-500 by default"
    )
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir",    default=DEFAULT_OUT)
    parser.add_argument("--seed",       type=int, default=SEED)
    parser.add_argument("--full",       action="store_true",
                        help="Use full 8792-example dataset instead of the mini-500 subset")
    parser.add_argument("--stats",      action="store_true",
                        help="Print sizes only, no files written")
    parser.add_argument("--hf",         action="store_true",
                        help="Force HuggingFace download")
    args = parser.parse_args()

    # ── Select counts based on mode ──────────────────────────────────────────
    if args.full:
        n_llm, n_steer, n_val = N_LLM_TRAIN_FULL, N_STEER_TRAIN_FULL, N_VALIDATION_FULL
        n_test_target = N_TEST_FULL
        mode_label = (
            f"FULL  (train pool {EXPECTED_TRAIN_POOL}: "
            f"llm={n_llm} steer={n_steer} val={n_val}  +  test={n_test_target} separate)"
        )
    else:
        n_llm, n_steer, n_val = N_LLM_TRAIN_MINI, N_STEER_TRAIN_MINI, N_VALIDATION_MINI
        n_test_target = N_TEST_MINI
        mode_label = f"MINI-500  (target {TOTAL_MINI} examples)"

    source_dir = pathlib.Path(args.source_dir)
    out_dir    = pathlib.Path(args.out_dir)

    print(f"\nGSM8K Dataset Splitter")
    print(f"  mode   : {mode_label}")
    print(f"  source : {source_dir}")
    print(f"  output : {out_dir}")
    print(f"  seed   : {args.seed}")
    print(f"  target : llm={n_llm}  steer={n_steer}  val={n_val}  test={n_test_target}\n")

    if args.hf:
        combined_hf = load_gsm8k_hf()
        hf_test  = [ex for ex in combined_hf if ex.get("split") == "original_test"]
        hf_train = [ex for ex in combined_hf if ex.get("split") == "original_train"]
        all_data = hf_train if args.full else combined_hf  # mini uses combined pool
    else:
        all_data, hf_test = load_from_source(source_dir, train_only=args.full)

    if args.full:
        # Full mode: split only the training pool; test.jsonl used verbatim
        separate_test = hf_test if args.hf else hf_test
        if not separate_test:
            # fallback: load test.jsonl directly
            tp = source_dir / "test.jsonl"
            if tp.exists():
                separate_test = load_jsonl(tp)
                for ex in separate_test: ex.setdefault("split", "original_test")
                print(f"  Loaded {len(separate_test)} test examples from {tp}")
            else:
                sys.exit("[split] ✗ test.jsonl not found — cannot build separate test set.")
        data = all_data
        splits = split(data, args.seed, n_llm, n_steer, n_val, separate_test=separate_test)
    else:
        # Mini mode: carve test from the combined train+test pool
        # load_from_source(train_only=False) → (train+test combined, test_list)
        # we use the full combined list as the shufflable pool
        data = all_data   # already train+test when train_only=False
        splits = split(data, args.seed, n_llm, n_steer, n_val, n_test=n_test_target)

    print_stats(splits, mode_label)

    if args.stats:
        print("--stats mode: no files written.")
        return

    print("Writing split files...")
    for name, subset in splits.items():
        save_jsonl(subset, out_dir / f"{name}.jsonl")

    # CODI-compatible copy of test split
    codi_test = out_dir / "codi_test.jsonl"
    save_jsonl(splits["test"], codi_test)
    print(f"  CODI-compatible test → {codi_test}")

    write_config(out_dir, splits, args.seed, mode_label)

    print(f"\nDone. All splits in: {out_dir}/")
    print(f"\nPipeline:")
    print(f"  Phase 1 (train)  : python train_codi.py    --train-data {out_dir}/llm_train.jsonl")
    print(f"  Phase 2 (vector) : python extract_truth_vector.py  --steer-data {out_dir}/steer_train.jsonl")
    print(f"  Phase 3 (steer)  : python steer_inference.py        --eval-data  {out_dir}/test.jsonl")

if __name__ == "__main__":
    main()


#python split_dataset.py --full for restore