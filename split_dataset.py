"""
split_dataset.py
----------------
Splits GSM8K into the exact partition defined in the research protocol:
"Steering Continuous Reasoning via Latent Intervention"

Starting point: full GSM8K (7473 train + 1319 test = 8792 examples)

Exact protocol counts
---------------------
  llm_train     6,000   Base Training     — Train the base CCoT / CODI model
  steer_train     500   Vector Extraction — Compute v_truth (Dsteer)
  validation    1,500   Further Training  — Final hypothesis testing (Dfinal_train)
  test            792   Testing           — Held-out, never used for training
  ─────────────────────
  TOTAL         8,792

Critical rule (from protocol):
  validation (Dfinal_train) must NEVER be used to compute the truth vector.
  Only steer_train (Dsteer) is used for Phase 2.

All splits are deterministic given the same seed (default 42).

Usage
-----
    python split_dataset.py
    python split_dataset.py --seed 123
    python split_dataset.py --stats     # print sizes only, no files written
    python split_dataset.py --hf        # force HuggingFace download
"""

import argparse, json, pathlib, random


DEFAULT_SOURCE = "datasets/gsm8k"
DEFAULT_OUT    = "datasets/gsm8k_split"
SEED           = 42

# ── Exact counts from protocol ────────────────────────────────────────────────
N_LLM_TRAIN   = 6000   # Base Training      (Dbase)
N_STEER_TRAIN =  500   # Vector Extraction  (Dsteer)
N_VALIDATION  = 1500   # Further Training   (Dfinal_train)
N_TEST        =  792   # Held-out Testing
EXPECTED_TOTAL = N_LLM_TRAIN + N_STEER_TRAIN + N_VALIDATION + N_TEST  # 8792


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

def load_from_source(source_dir):
    train_path = source_dir / "train.jsonl"
    test_path  = source_dir / "test.jsonl"
    if train_path.exists() and test_path.exists():
        train = load_jsonl(train_path)
        test  = load_jsonl(test_path)
        for ex in train: ex.setdefault("split", "original_train")
        for ex in test:  ex.setdefault("split", "original_test")
        combined = train + test
        print(f"  Loaded {len(train)} train + {len(test)} test = {len(combined)} total from {source_dir}")
        return combined
    print(f"  Local files not found at {source_dir} — trying HuggingFace...")
    return load_gsm8k_hf()


# ── Splitting logic ────────────────────────────────────────────────────────────

def split(data, seed):
    total = len(data)
    if total != EXPECTED_TOTAL:
        print(f"  WARNING: expected {EXPECTED_TOTAL} examples, got {total}.")
        print(f"  Adjusting test size to absorb the difference ({total - N_LLM_TRAIN - N_STEER_TRAIN - N_VALIDATION} test examples).")

    rng = random.Random(seed)
    shuffled = data.copy()
    rng.shuffle(shuffled)

    # Protocol-exact counts; test absorbs any remainder
    n_llm   = N_LLM_TRAIN
    n_steer = N_STEER_TRAIN
    n_val   = N_VALIDATION
    n_test  = total - n_llm - n_steer - n_val  # = 792 when total = 8792

    if n_test < 0:
        raise ValueError(
            f"Dataset has only {total} examples but protocol requires "
            f"{n_llm + n_steer + n_val} for training alone. "
            f"Cannot create a test set."
        )

    llm_train   = shuffled[:n_llm]
    steer_train = shuffled[n_llm : n_llm + n_steer]
    validation  = shuffled[n_llm + n_steer : n_llm + n_steer + n_val]
    test        = shuffled[n_llm + n_steer + n_val :]

    assert len(llm_train) + len(steer_train) + len(validation) + len(test) == total

    return {
        "llm_train":   llm_train,
        "steer_train": steer_train,
        "validation":  validation,
        "test":        test,
    }


def print_stats(splits, total):
    rows = [
        ("llm_train",   "Base Training    — CODI/CCoT fine-tuning              (Dbase)"),
        ("steer_train", "Vector Extraction— v_truth computation                (Dsteer)"),
        ("validation",  "Further Training — hypothesis testing                 (Dfinal_train)"),
        ("test",        "Held-out Testing — never used during training"),
    ]
    print(f"\n  {'Split':<14} {'N':>6}  {'% of total':>10}  Purpose")
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


def write_config(out_dir, splits, seed):
    total = sum(len(v) for v in splits.values())
    config = {
        "seed": seed,
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
            "llm_train": N_LLM_TRAIN, "steer_train": N_STEER_TRAIN,
            "validation": N_VALIDATION, "test": N_TEST,
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
    parser = argparse.ArgumentParser(description="Split GSM8K per research protocol exact counts")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir",    default=DEFAULT_OUT)
    parser.add_argument("--seed",       type=int, default=SEED)
    parser.add_argument("--stats",      action="store_true", help="Print sizes only, no files written")
    parser.add_argument("--hf",         action="store_true", help="Force HuggingFace download")
    args = parser.parse_args()

    source_dir = pathlib.Path(args.source_dir)
    out_dir    = pathlib.Path(args.out_dir)

    print(f"\nGSM8K Dataset Splitter  (protocol-exact counts)")
    print(f"  source : {source_dir}")
    print(f"  output : {out_dir}")
    print(f"  seed   : {args.seed}")
    print(f"  target : llm={N_LLM_TRAIN}  steer={N_STEER_TRAIN}  val={N_VALIDATION}  test={N_TEST}\n")

    data   = load_gsm8k_hf() if args.hf else load_from_source(source_dir)
    total  = len(data)
    splits = split(data, args.seed)

    print_stats(splits, total)

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

    write_config(out_dir, splits, args.seed)

    print(f"\nDone. All splits in: {out_dir}/")
    print(f"\nPipeline:")
    print(f"  Phase 1 (train)  : python train_codi.py    --train-data {out_dir}/llm_train.jsonl")
    print(f"  Phase 2 (vector) : python extract_truth_vector.py  --steer-data {out_dir}/steer_train.jsonl")
    print(f"  Phase 3 (steer)  : python steer_inference.py        --eval-data  {out_dir}/test.jsonl")

if __name__ == "__main__":
    main()