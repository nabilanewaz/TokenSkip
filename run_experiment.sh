#!/usr/bin/env bash
# run_experiment.sh — Full TokenSkip pipeline (phases 1-6)
# Usage: bash run_experiment.sh
# Run from the TokenSkip project root inside the tokenskip conda env.
set -euo pipefail

# ─────────────────────────────────────────────────────────────────
# 0. Environment setup  (skip if already done)
# ─────────────────────────────────────────────────────────────────
# git clone <your-repo> TokenSkip && cd TokenSkip
# conda create -n tokenskip python=3.10 -y
# conda activate tokenskip
# pip install -r requirements.txt

mkdir -p logs


# ─────────────────────────────────────────────────────────────────
# 1. Dataset preparation
# ─────────────────────────────────────────────────────────────────
python split_dataset.py --full

echo "--- Split sizes (expected 4483 / 747 / 2243 / 1319) ---"
wc -l datasets/gsm8k_split/*.jsonl


# ─────────────────────────────────────────────────────────────────
# 2. Phase 1 — Train CODI-GPT2
# ─────────────────────────────────────────────────────────────────
python phase1_train.py \
    --train-data datasets/gsm8k_split/llm_train.jsonl \
    --val-data   datasets/gsm8k_split/validation.jsonl \
    --output-dir outputs/phase1_checkpoint \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    --batch_size 4 \
    --seed 42 \
    --bf16 2>&1 | tee logs/phase1.log


# ─────────────────────────────────────────────────────────────────
# 3. Phase 2 — Extract truth vector
# ─────────────────────────────────────────────────────────────────
python phase2_extract_vector.py \
    --steer-data datasets/gsm8k_split/steer_train.jsonl \
    --ckpt-dir   outputs/phase1_checkpoint \
    --n-samples  5 \
    --seed       42 2>&1 | tee logs/phase2.log


# ─────────────────────────────────────────────────────────────────
# 4. HPO — Find best alpha on VALIDATION set
# ─────────────────────────────────────────────────────────────────

# Full sweep (alpha + layer_frac) — slower:
# python hpo_sweep.py --sweep alpha layer_frac \
#     --models codi_gpt2 phi2 llama32_3b qwen25_3b 2>&1 | tee logs/hpo.log

# Alpha-only sweep (faster):
python hpo_sweep.py --sweep alpha 2>&1 | tee logs/hpo.log

python hpo_sweep.py --show-best


# ─────────────────────────────────────────────────────────────────
# 5. Full evaluation on TEST set
# run_full_experiment.py writes its own logs to outputs/logs/
# ─────────────────────────────────────────────────────────────────
python run_full_experiment.py --skip-hpo

# Override models / conditions / local paths as needed, e.g.:
# python run_full_experiment.py --skip-hpo \
#     --models phi2 llama32_3b qwen25_3b codi_gpt2 \
#     --conditions no_cot text_cot ccot random_noise steered
#
# python run_full_experiment.py --skip-hpo \
#     --model-paths phi2=/models/phi-2 \
#                   llama32_3b=/models/Llama-3.2-3B \
#                   qwen25_3b=/models/Qwen2.5-3B


# ─────────────────────────────────────────────────────────────────
# 6. View results
# ─────────────────────────────────────────────────────────────────
python compare_all.py
python compare_all.py --csv results.csv --latex

echo "--- Per-run logs ---"
ls outputs/logs/
