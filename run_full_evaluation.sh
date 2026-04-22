#!/bin/bash
################################################################################
# run_full_evaluation.sh
# 
# Comprehensive TokenSkip Steering Evaluation Pipeline
# ====================================================
# Models: 6 HuggingFace models + CODI (7 total)
# Conditions: no_cot, text_cot, ccot, random_noise, steered (alpha sweep)
# Alpha sweep: [0, 0.5, 1, 2, 5, 10, 20, 50, -0.5, -1]
# Metrics: accuracy, fliprate, cosine_similarity, faithfulness, token_count
# Dataset: GSM8K (7473 train pool + 1319 test)
# 
# Usage:
#   bash run_full_evaluation.sh 2>&1 | tee evaluation_master.log
#   
# This script will:
#   1. Set up directories and logging
#   2. Split dataset (if needed)
#   3. Train CODI (Phase 1) — ~2-4h on GPU
#   4. Extract truth vector (Phase 2) — ~15min
#   5. Run CODI steering sweep (Phase 3) — ~1h
#   6. Evaluate 6 HF models on all conditions — ~2-3h
#   7. Aggregate all results
#   8. Compile token metrics
#
# STAGE CONTROL: Set to 1 to run, 0 to skip
# ============================================
STAGE_SPLIT=1
STAGE_PHASE1=1
STAGE_PHASE2=1
STAGE_PHASE3=1
STAGE_HF_BASELINES=1
STAGE_HF_STEERING=1
STAGE_AGGREGATE=1
STAGE_TOKEN_REPORT=1
################################################################################

set -e  # Exit on any error

# ── Configuration ─────────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1

# Timestamp for all logging
ts=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="logs"
mkdir -p "$LOG_ROOT" report outputs

# Alpha sweep values (10 total)
alphas=(0 0.5 1 2 5 10 20 50 -0.5 -1)
alphas_str="${alphas[@]}"

# HuggingFace models (6 models, Mistral excluded)
declare -A hf_models=(
    [phi2]="microsoft/phi-2"
    [llama32_3b]="meta-llama/Llama-3.2-3B"
    [qwen25_0_5b]="Qwen/Qwen2.5-0.5B"
    [qwen25_1_5b]="Qwen/Qwen2.5-1.5B"
    [qwen25_3b]="Qwen/Qwen2.5-3B"
    [qwen_math_1_5b]="Qwen/Qwen2.5-Math-1.5B"
)

# Model types for evaluation.py and hidden_steer.py
hf_model_types=(phi2 llama32_3b qwen25_0_5b qwen25_1_5b qwen25_3b qwen_math_1_5b)

# Supported steering models — NOW EXTENDED TO ALL 6 MODELS!
# (Updated in hidden_steer.py and model_registry.py)
steering_model_types=(phi2 llama32_3b qwen25_0_5b qwen25_1_5b qwen25_3b qwen_math_1_5b)

# Datasets
SPLIT_DIR="datasets/gsm8k_split"
TRAIN_DATA="$SPLIT_DIR/llm_train.jsonl"
STEER_DATA="$SPLIT_DIR/steer_train.jsonl"
VAL_DATA="$SPLIT_DIR/validation.jsonl"
TEST_DATA="$SPLIT_DIR/test.jsonl"

# Output directories
PHASE1_CKPT="outputs/phase1_checkpoint"
PHASE2_VECTOR="outputs/phase2_truth_vector"
PHASE3_RESULTS="outputs/phase3_results"
EVAL_GRID="outputs/eval_grid"

# ── Logging helpers ────────────────────────────────────────────────────────────

log_stage() {
    local stage=$1
    local msg=$2
    echo ""
    echo "=================================================================================="
    echo "[$ts] STAGE: $stage"
    echo "  $msg"
    echo "=================================================================================="
    echo ""
}

log_task() {
    local task=$1
    echo "[$(date +%H:%M:%S)] >>> $task"
}

log_done() {
    local task=$1
    echo "[$(date +%H:%M:%S)] ✓ $task"
}

# ── Stage 1: Data splitting ────────────────────────────────────────────────────

if [ $STAGE_SPLIT -eq 1 ]; then
    log_stage "SPLIT_DATASET" "Create GSM8K splits: 4483 train / 747 steer / 2243 val / 1319 test"
    
    if [ ! -f "$TEST_DATA" ]; then
        log_task "Splitting GSM8K..."
        python split_dataset.py --full --seed 42 2>&1 | tee "$LOG_ROOT/${ts}_split.log"
        log_done "Dataset split"
    else
        echo "  Dataset already split. Skipping."
    fi
fi

# ── Stage 2: Phase 1 — CODI Training ───────────────────────────────────────────

if [ $STAGE_PHASE1 -eq 1 ]; then
    log_stage "PHASE_1_TRAINING" "Train CODI-GPT2 on llm_train (4483 examples)"
    
    log_task "Phase 1: Base CODI training (curriculum: text CoT → latent CoT)..."
    python phase1_train.py \
        --train-data "$TRAIN_DATA" \
        --output-dir "$PHASE1_CKPT" \
        --num_epochs 3 \
        --batch_size 4 \
        --learning_rate 2e-4 \
        --bf16 \
        --seed 42 \
        2>&1 | tee "$LOG_ROOT/${ts}_phase1_train.log"
    
    log_done "Phase 1 training complete"
fi

# ── Stage 3: Phase 2 — Truth Vector Extraction ────────────────────────────────

if [ $STAGE_PHASE2 -eq 1 ]; then
    log_stage "PHASE_2_EXTRACTION" "Extract truth vector v_truth from steer_train (747 examples)"
    
    log_task "Phase 2: Extract truth direction (Difference-of-Means method)..."
    python phase2_extract_vector.py \
        --steer-data "$STEER_DATA" \
        --ckpt-dir "$PHASE1_CKPT" \
        --out-dir "$PHASE2_VECTOR" \
        --n-samples 5 \
        --bf16 \
        --seed 42 \
        2>&1 | tee "$LOG_ROOT/${ts}_phase2_extract.log"
    
    log_done "Phase 2 vector extraction complete"
fi

# ── Stage 4: Phase 3 — CODI Steering Sweep ────────────────────────────────────

if [ $STAGE_PHASE3 -eq 1 ]; then
    log_stage "PHASE_3_STEERING" "CODI steering sweep (alpha: $alphas_str)"
    
    log_task "Phase 3: Alpha sweep + random noise control on test set (1319 examples)..."
    python phase3_steer_inference.py \
        --eval-data "$TEST_DATA" \
        --vector-dir "$PHASE2_VECTOR" \
        --ckpt-dir "$PHASE1_CKPT" \
        --out-dir "$PHASE3_RESULTS" \
        --alphas $alphas_str \
        --random-noise \
        --bf16 \
        --seed 42 \
        2>&1 | tee "$LOG_ROOT/${ts}_phase3_steering.log"
    
    log_done "Phase 3 steering complete"
fi

# ── Stage 5: HuggingFace Model Baselines (no_cot, text_cot) ──────────────────

if [ $STAGE_HF_BASELINES -eq 1 ]; then
    log_stage "HF_BASELINES" "Evaluate 6 HF models: no_cot and text_cot conditions"
    
    for model_type in "${hf_model_types[@]}"; do
        model_name="${hf_models[$model_type]}"
        echo ""
        log_task "[$model_type] Baseline: no_cot"
        python evaluation.py \
            --model-type "$model_type" \
            --eval-data "$TEST_DATA" \
            --output-dir "$EVAL_GRID" \
            --no-cot \
            --eval_batch_size 8 \
            --temperature 0.0 \
            --seed 42 \
            2>&1 | tee "$LOG_ROOT/${ts}_${model_type}_no_cot.log"
        log_done "[$model_type] no_cot"
        
        log_task "[$model_type] Baseline: text_cot"
        python evaluation.py \
            --model-type "$model_type" \
            --eval-data "$TEST_DATA" \
            --output-dir "$EVAL_GRID" \
            --eval_batch_size 8 \
            --temperature 0.0 \
            --seed 42 \
            2>&1 | tee "$LOG_ROOT/${ts}_${model_type}_text_cot.log"
        log_done "[$model_type] text_cot"
    done
    
    log_done "All HF baselines complete"
fi

# ── Stage 6: HuggingFace Steering (ccot, random_noise, steered) ──────────────

if [ $STAGE_HF_STEERING -eq 1 ]; then
    log_stage "HF_STEERING" "Evaluate steering for all 6 models: ccot, random_noise, steered (alpha sweep)"
    
    # ✓ EXTENSION COMPLETE: Now supports all 6 models (phi2, llama32_3b, all 4 Qwen variants)
    # See model_registry.py and hidden_steer.py for implementation details
    
    echo "  FULL STEERING SUPPORT: ${steering_model_types[@]}"
    echo "  NOTE: qwen25_0_5b, qwen25_1_5b, qwen_math_1_5b require extension of hidden_steer.py"
    echo ""
    
    for model_type in "${steering_model_types[@]}"; do
        model_name="${hf_models[$model_type]}"
        
        # CCoT (unsteered) — equivalent to alpha=0 control
        log_task "[$model_type] Condition: ccot (unsteered baseline)"
        python hidden_steer.py \
            --model-type "$model_type" \
            --condition ccot \
            --eval-data "$TEST_DATA" \
            --steer-data "$STEER_DATA" \
            --output-dir "$EVAL_GRID" \
            --eval_batch_size 8 \
            --seed 42 \
            2>&1 | tee "$LOG_ROOT/${ts}_${model_type}_ccot.log"
        log_done "[$model_type] ccot"
        
        # Random noise control — proves v_truth matters
        log_task "[$model_type] Condition: random_noise (control)"
        python hidden_steer.py \
            --model-type "$model_type" \
            --condition random_noise \
            --eval-data "$TEST_DATA" \
            --steer-data "$STEER_DATA" \
            --output-dir "$EVAL_GRID" \
            --alphas 1.0 \
            --eval_batch_size 8 \
            --seed 42 \
            2>&1 | tee "$LOG_ROOT/${ts}_${model_type}_random_noise.log"
        log_done "[$model_type] random_noise"
        
        # Steered condition — full alpha sweep
        log_task "[$model_type] Condition: steered (alpha sweep: $alphas_str)"
        python hidden_steer.py \
            --model-type "$model_type" \
            --condition steered \
            --eval-data "$TEST_DATA" \
            --steer-data "$STEER_DATA" \
            --output-dir "$EVAL_GRID" \
            --alphas $alphas_str \
            --eval_batch_size 8 \
            --seed 42 \
            2>&1 | tee "$LOG_ROOT/${ts}_${model_type}_steered_sweep.log"
        log_done "[$model_type] steered (all alphas)"
    done
    
    log_done "All HF steering complete"
fi

# ── Stage 7: Aggregate Results ─────────────────────────────────────────────────

if [ $STAGE_AGGREGATE -eq 1 ]; then
    log_stage "AGGREGATE_RESULTS" "Compile all metrics into comparison tables"
    
    log_task "Aggregating all metrics.json files..."
    python compare_all.py \
        --eval-grid "$EVAL_GRID" \
        --output "$LOG_ROOT/${ts}_aggregated_results.csv" \
        2>&1 | tee "$LOG_ROOT/${ts}_aggregate.log"
    
    log_done "Results aggregated"
fi

# ── Stage 8: Token Counting & Compression Report ────────────────────────────

if [ $STAGE_TOKEN_REPORT -eq 1 ]; then
    log_stage "TOKEN_METRICS" "Extract token counts and compute compression ratios"
    
    log_task "Compiling token statistics..."
    python extract_token_metrics.py \
        --eval-grid "$EVAL_GRID" \
        --phase3-results "$PHASE3_RESULTS" \
        --output "$LOG_ROOT/${ts}_token_metrics.csv" \
        --output-json "$LOG_ROOT/${ts}_token_metrics.json" \
        2>&1 | tee "$LOG_ROOT/${ts}_token_extraction.log"
    
    log_done "Token metrics compiled"
fi

# ── Final Summary ──────────────────────────────────────────────────────────────

echo ""
echo "================================================================================"
echo "  EVALUATION PIPELINE COMPLETE"
echo "================================================================================"
echo ""
echo "  Master log   : ${LOG_ROOT}/evaluation_master.log"
echo "  All logs     : ${LOG_ROOT}/${ts}_*.log"
echo ""
echo "  Key outputs:"
echo "    • ${LOG_ROOT}/${ts}_aggregated_results.csv       (all metrics comparison)"
echo "    • ${LOG_ROOT}/${ts}_token_metrics.csv            (token counts & compression)"
echo "    • ${LOG_ROOT}/${ts}_token_metrics.json           (detailed token breakdown)"
echo "    • $PHASE3_RESULTS/summary.json               (CODI steering summary)"
echo "    • $EVAL_GRID/*/metrics.json                  (per-model/condition results)"
echo ""
echo "  To view aggregated results:"
echo "    cat ${LOG_ROOT}/${ts}_aggregated_results.csv"
echo "    cat ${LOG_ROOT}/${ts}_token_metrics.csv"
echo ""
echo "================================================================================"
echo ""
