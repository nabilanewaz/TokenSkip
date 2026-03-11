"""
experiment_config.py
====================
Single source of truth for the entire experiment.

Imported by:
    hpo_sweep.py          ← hyperparameter search
    run_full_experiment.py ← master orchestrator
    evaluate_baselines.py  ← HF-model condition runner
    compare_all.py         ← results aggregator

To add a new model, dataset, or hyperparameter range, edit the dicts below.
Nothing else needs changing — all scripts read from here.
"""

import os
import pathlib


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MODELS
# ══════════════════════════════════════════════════════════════════════════════
# Each entry:
#   hf_id   – HuggingFace model ID or local path (override via env vars below)
#   tag     – short key used in output directory names
#   kind    – "codi" → steer via phase3_steer_inference.py
#           – "hf"   → steer via hidden_steer.py
#   desc    – human-readable label for reports
MODELS = [
    {
        "tag":   "codi_gpt2",
        "hf_id": os.environ.get("CODI_PATH", "zen-E/CODI-gpt2"),
        "kind":  "codi",
        "desc":  "CODI-GPT2 (continuous CoT, 117 M params)",
    },
    {
        "tag":   "phi2",
        "hf_id": os.environ.get("PHI2_PATH", "microsoft/phi-2"),
        "kind":  "hf",
        "desc":  "microsoft/phi-2 (2.7 B params)",
    },
    {
        "tag":   "llama32_3b",
        "hf_id": os.environ.get("LLAMA32_PATH", "meta-llama/Llama-3.2-3B"),
        "kind":  "hf",
        "desc":  "meta-llama/Llama-3.2-3B (3 B params)",
    },
    {
        "tag":   "qwen25_3b",
        "hf_id": os.environ.get("QWEN25_PATH", "Qwen/Qwen2.5-3B"),
        "kind":  "hf",
        "desc":  "Qwen/Qwen2.5-3B (3 B params)",
    },
]

# Convenience lookup: tag → entry
MODEL_BY_TAG = {m["tag"]: m for m in MODELS}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATASETS
# ══════════════════════════════════════════════════════════════════════════════
# Split sizes come from split_dataset.py --full
# Source:   7 473 GSM8K train examples + 1 319 test examples
#
#   Phase 1 – Base Training      : 4 483  (60 %)
#   Phase 2 – Vector Extraction  :   747  (10 %)
#   Phase 3 – Validation          : 2 243  (30 %)
#   Phase 4 – Held-out Test Set  : 1 319  (separate GSM8K test.jsonl)
DATASETS = [
    {
        "name":     "gsm8k",
        "desc":     "Grade School Math 8K",
        "splits": {
            "llm_train":   "datasets/gsm8k_split/llm_train.jsonl",    # 4 483
            "steer_train": "datasets/gsm8k_split/steer_train.jsonl",  #   747
            "validation":  "datasets/gsm8k_split/validation.jsonl",   # 2 243
            "test":        "datasets/gsm8k_split/test.jsonl",         # 1 319
        },
        "split_sizes": {
            "llm_train":   4483,
            "steer_train": 747,
            "validation":  2243,
            "test":        1319,
        },
    },
]

# Default dataset used throughout the experiment
DEFAULT_DATASET = "gsm8k"

def get_dataset(name=DEFAULT_DATASET):
    for d in DATASETS:
        if d["name"] == name:
            return d
    raise KeyError(f"Unknown dataset '{name}'. Available: {[d['name'] for d in DATASETS]}")


def get_split_path(split_name, dataset_name=DEFAULT_DATASET):
    ds = get_dataset(dataset_name)
    if split_name not in ds["splits"]:
        raise KeyError(f"Unknown split '{split_name}' for dataset '{dataset_name}'.")
    return pathlib.Path(ds["splits"][split_name])


# ══════════════════════════════════════════════════════════════════════════════
# 3.  EVALUATION CONDITIONS
# ══════════════════════════════════════════════════════════════════════════════
CONDITIONS = [
    {
        "name": "no_cot",
        "desc": "Direct answer -- no chain-of-thought at all",
        "runner": "evaluation.py",   # + --no-cot flag
    },
    {
        "name": "text_cot",
        "desc": "Standard text-based chain-of-thought",
        "runner": "evaluation.py",
    },
    {
        "name": "ccot",
        "desc": "Continuous CoT, unsteered (alpha=0, control group)",
        "runner": "hidden_steer.py",  # or phase3_steer_inference.py for CODI
    },
    {
        "name": "random_noise",
        "desc": "Continuous CoT + random unit vector (direction control)",
        "runner": "hidden_steer.py",
    },
    {
        "name": "steered",
        "desc": "Continuous CoT + v_truth steering (the hypothesis)",
        "runner": "hidden_steer.py",
    },
]

CONDITION_ORDER = [c["name"] for c in CONDITIONS]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  HYPERPARAMETER GRIDS
# ══════════════════════════════════════════════════════════════════════════════

# ── 4a. Fixed / canonical values (used for final evaluation) ──────────────
FIXED_SEED   = 42          # used globally wherever a seed is required
FIXED_ALPHAS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]   # steering strength sweep

# CODI Phase-1 training defaults
CODI_TRAIN_DEFAULTS = {
    "seed":          FIXED_SEED,
    "num_epochs":    3,
    "learning_rate": 2e-4,      # best from sweep below
    "batch_size":    4,         # per-device; use 2 on CPU
    "lora_r":        128,
    "lora_alpha":    32,
    "num_latent":    6,
    "prj_dim":       768,
    "inf_latent_iterations": 6,
}

# ── 4b. Hyperparameter search grids (used by hpo_sweep.py) ───────────────
HPO_GRIDS = {
    # Alpha sweep — evaluated on validation set; best α per model used for test
    "alpha": {
        "param":   "alpha",
        "values":  FIXED_ALPHAS,
        "metric":  "accuracy",        # maximise
        "dataset": "validation",
    },

    # Learning-rate search for Phase-1 fine-tuning (CODI)
    "learning_rate": {
        "param":   "learning_rate",
        "values":  [1e-4, 2e-4, 5e-4],
        "metric":  "accuracy",
        "dataset": "validation",
        "model":   "codi_gpt2",       # only relevant for CODI
    },

    # Seed robustness check (run 3 seeds, report mean ± std)
    "seed": {
        "param":   "seed",
        "values":  [42, 123, 2025],
        "metric":  "accuracy",
        "dataset": "validation",
    },

    # Intervention layer fraction (how deep in the network to intervene)
    "layer_frac": {
        "param":   "layer_frac",
        "values":  [0.5, 0.625, 0.75],
        "metric":  "accuracy",
        "dataset": "validation",
        "kind":    "hf",              # only for HF models via hidden_steer.py
    },
}

# ── 4c. Best hyperparameters cache path ───────────────────────────────────
# hpo_sweep.py writes here; run_full_experiment.py reads from here
BEST_HPARAMS_PATH = pathlib.Path("outputs/hpo/best_hyperparams.json")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  OUTPUT PATHS
# ══════════════════════════════════════════════════════════════════════════════
PATHS = {
    "phase1_ckpt":      pathlib.Path("outputs/phase1_checkpoint"),
    "phase2_vectors":   pathlib.Path("outputs/truth_vectors"),
    "codi_vector":      pathlib.Path("outputs/phase2_truth_vector"),
    "phase3_results":   pathlib.Path("outputs/phase3_results"),
    "eval_grid":        pathlib.Path("outputs/eval_grid"),
    "hpo_root":         pathlib.Path("outputs/hpo"),
    "logs":             pathlib.Path("outputs/logs"),
    "codi_work_dir":    pathlib.Path("codi_workspace"),
    "codi_bundle":      pathlib.Path("codi_bundle"),
}

# ══════════════════════════════════════════════════════════════════════════════
# 6.  METRICS reported in every metrics.json
# ══════════════════════════════════════════════════════════════════════════════
METRICS = ["accuracy", "flip_rate", "mean_cos_sim"]


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Quick summary (printed at startup by run_full_experiment.py)
# ══════════════════════════════════════════════════════════════════════════════
def print_experiment_summary():
    ds = get_dataset()
    print("=" * 70)
    print("  EXPERIMENT CONFIGURATION SUMMARY")
    print("=" * 70)

    print("\n-- Models " + "-" * 60)
    for m in MODELS:
        print(f"  [{m['tag']:12s}]  {m['desc']}")
        print(f"               HF path : {m['hf_id']}")
        print(f"               Steering: {m['kind']}")

    print("\n-- Datasets " + "-" * 58)
    print(f"  {ds['name']} - {ds['desc']}")
    for split, path in ds["splits"].items():
        n = ds["split_sizes"].get(split, "?")
        print(f"    {split:14s}: {path}  ({n} examples)")

    print("\n-- Evaluation conditions " + "-" * 45)
    for c in CONDITIONS:
        print(f"  {c['name']:14s}: {c['desc']}")

    print("\n-- Hyperparameter grids " + "-" * 46)
    print(f"  alpha values   : {FIXED_ALPHAS}")
    for key, grid in HPO_GRIDS.items():
        print(f"  {key:16s}: {grid['values']}")

    print("\n-- Fixed hyperparameters " + "-" * 45)
    print(f"  seed           : {FIXED_SEED}")
    for k, v in CODI_TRAIN_DEFAULTS.items():
        print(f"  {k:16s}: {v}")

    print("=" * 70)


if __name__ == "__main__":
    print_experiment_summary()
