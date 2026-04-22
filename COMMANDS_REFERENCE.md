# TokenSkip Evaluation — Quick Command Reference

## One-Line Quick Start

```bash
bash run_full_evaluation.sh 2>&1 | tee logs/evaluation_master.log
```

**Estimated time**: 6-8 hours on GPU (A100 recommended)

---

## Models

### Model Types (for evaluation.py and hidden_steer.py)

| Model | Type | Model ID |
|-------|------|----------|
| Phi-2 | `phi2` | microsoft/phi-2 |
| Llama-3.2-3B | `llama32_3b` | meta-llama/Llama-3.2-3B |
| Qwen 0.5B | `qwen25_0_5b` | Qwen/Qwen2.5-0.5B |
| Qwen 1.5B | `qwen25_1_5b` | Qwen/Qwen2.5-1.5B |
| Qwen 3B | `qwen25_3b` | Qwen/Qwen2.5-3B |
| Qwen Math 1.5B | `qwen_math_1_5b` | Qwen/Qwen2.5-Math-1.5B |

### Steering Support

**Supported**: `phi2`, `llama32_3b`, `qwen25_3b`
**Not yet**: `qwen25_0_5b`, `qwen25_1_5b`, `qwen_math_1_5b` (need model_registry.py extension)

---

## Datasets

```bash
# All paths:
TRAIN_DATA="datasets/gsm8k_split/llm_train.jsonl"      # 4,483 examples
STEER_DATA="datasets/gsm8k_split/steer_train.jsonl"    # 747 examples
VAL_DATA="datasets/gsm8k_split/validation.jsonl"       # 2,243 examples
TEST_DATA="datasets/gsm8k_split/test.jsonl"            # 1,319 examples
```

---

## Phase 1: Train CODI

```bash
python phase1_train.py \
    --train-data datasets/gsm8k_split/llm_train.jsonl \
    --output-dir outputs/phase1_checkpoint \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --bf16 \
    --seed 42 \
    2>&1 | tee logs/$(date +%Y%m%d_%H%M%S)_phase1_train.log
```

**Time**: ~2-4h on GPU | ~100h on CPU

---

## Phase 2: Extract Truth Vector

```bash
python phase2_extract_vector.py \
    --steer-data datasets/gsm8k_split/steer_train.jsonl \
    --ckpt-dir outputs/phase1_checkpoint \
    --out-dir outputs/phase2_truth_vector \
    --n-samples 5 \
    --bf16 \
    --seed 42 \
    2>&1 | tee logs/$(date +%Y%m%d_%H%M%S)_phase2_extract.log
```

**Time**: ~15 min on GPU

---

## Phase 3: CODI Steering Sweep

```bash
# Alpha values: [0, 0.5, 1, 2, 5, 10, 20, 50, -0.5, -1]

python phase3_steer_inference.py \
    --eval-data datasets/gsm8k_split/test.jsonl \
    --vector-dir outputs/phase2_truth_vector \
    --ckpt-dir outputs/phase1_checkpoint \
    --out-dir outputs/phase3_results \
    --alphas 0 0.5 1 2 5 10 20 50 -0.5 -1 \
    --random-noise \
    --bf16 \
    --seed 42 \
    2>&1 | tee logs/$(date +%Y%m%d_%H%M%S)_phase3_steering.log
```

**Time**: ~1h for 10 alphas on GPU

---

## HuggingFace Baselines

### No CoT (Direct Answer)

```bash
# Template
python evaluation.py \
    --model-type {MODEL_TYPE} \
    --eval-data datasets/gsm8k_split/test.jsonl \
    --output-dir outputs/eval_grid \
    --no-cot \
    --eval_batch_size 8 \
    --temperature 0.0 \
    --seed 42 \
    2>&1 | tee logs/$(date +%Y%m%d_%H%M%S)_{MODEL_TYPE}_no_cot.log

# Examples
python evaluation.py --model-type phi2 --no-cot --eval_batch_size 8
python evaluation.py --model-type llama32_3b --no-cot --eval_batch_size 4
python evaluation.py --model-type qwen25_3b --no-cot --eval_batch_size 8
python evaluation.py --model-type qwen25_1_5b --no-cot --eval_batch_size 8
python evaluation.py --model-type qwen25_0_5b --no-cot --eval_batch_size 16
python evaluation.py --model-type qwen_math_1_5b --no-cot --eval_batch_size 8
```

### Text CoT (Chain-of-Thought)

```bash
# Template (same as above but WITHOUT --no-cot flag)
python evaluation.py \
    --model-type {MODEL_TYPE} \
    --eval-data datasets/gsm8k_split/test.jsonl \
    --output-dir outputs/eval_grid \
    --eval_batch_size 8 \
    --temperature 0.0 \
    --seed 42 \
    2>&1 | tee logs/$(date +%Y%m%d_%H%M%S)_{MODEL_TYPE}_text_cot.log

# Examples
python evaluation.py --model-type phi2 --eval_batch_size 8
python evaluation.py --model-type llama32_3b --eval_batch_size 4
python evaluation.py --model-type qwen25_3b --eval_batch_size 8
python evaluation.py --model-type qwen25_1_5b --eval_batch_size 8
python evaluation.py --model-type qwen25_0_5b --eval_batch_size 16
python evaluation.py --model-type qwen_math_1_5b --eval_batch_size 8
```

---

## HuggingFace Steering

**NOTE**: Currently supported for `phi2`, `llama32_3b`, `qwen25_3b` only.

### CCoT (Unsteered Baseline)

```bash
# Template
python hidden_steer.py \
    --model-type {MODEL_TYPE} \
    --condition ccot \
    --eval-data datasets/gsm8k_split/test.jsonl \
    --steer-data datasets/gsm8k_split/steer_train.jsonl \
    --output-dir outputs/eval_grid \
    --eval_batch_size 8 \
    --seed 42 \
    2>&1 | tee logs/$(date +%Y%m%d_%H%M%S)_{MODEL_TYPE}_ccot.log

# Examples (supported models only)
python hidden_steer.py --model-type phi2 --condition ccot --eval_batch_size 8
python hidden_steer.py --model-type llama32_3b --condition ccot --eval_batch_size 4
python hidden_steer.py --model-type qwen25_3b --condition ccot --eval_batch_size 8
```

### Random Noise (Control)

```bash
# Template
python hidden_steer.py \
    --model-type {MODEL_TYPE} \
    --condition random_noise \
    --eval-data datasets/gsm8k_split/test.jsonl \
    --steer-data datasets/gsm8k_split/steer_train.jsonl \
    --output-dir outputs/eval_grid \
    --alphas 1.0 \
    --eval_batch_size 8 \
    --seed 42 \
    2>&1 | tee logs/$(date +%Y%m%d_%H%M%S)_{MODEL_TYPE}_random_noise.log

# Examples (supported models only)
python hidden_steer.py --model-type phi2 --condition random_noise --alphas 1.0 --eval_batch_size 8
python hidden_steer.py --model-type llama32_3b --condition random_noise --alphas 1.0 --eval_batch_size 4
python hidden_steer.py --model-type qwen25_3b --condition random_noise --alphas 1.0 --eval_batch_size 8
```

### Steered (Full Alpha Sweep)

```bash
# Template
python hidden_steer.py \
    --model-type {MODEL_TYPE} \
    --condition steered \
    --eval-data datasets/gsm8k_split/test.jsonl \
    --steer-data datasets/gsm8k_split/steer_train.jsonl \
    --output-dir outputs/eval_grid \
    --alphas 0 0.5 1 2 5 10 20 50 -0.5 -1 \
    --eval_batch_size 8 \
    --seed 42 \
    2>&1 | tee logs/$(date +%Y%m%d_%H%M%S)_{MODEL_TYPE}_steered_sweep.log

# Examples (supported models only)
python hidden_steer.py --model-type phi2 --condition steered \
    --alphas 0 0.5 1 2 5 10 20 50 -0.5 -1 --eval_batch_size 8

python hidden_steer.py --model-type llama32_3b --condition steered \
    --alphas 0 0.5 1 2 5 10 20 50 -0.5 -1 --eval_batch_size 4

python hidden_steer.py --model-type qwen25_3b --condition steered \
    --alphas 0 0.5 1 2 5 10 20 50 -0.5 -1 --eval_batch_size 8
```

**Time per model**: ~30-45 min (10 alphas)

---

## Aggregation & Reporting

### Aggregate All Metrics

```bash
python compare_all.py \
    --eval-grid outputs/eval_grid \
    --output logs/$(date +%Y%m%d_%H%M%S)_aggregated_results.csv \
    2>&1 | tee logs/aggregation.log
```

**Output**: CSV table with all models × conditions × alphas

### Extract Token Metrics

```bash
python extract_token_metrics.py \
    --eval-grid outputs/eval_grid \
    --phase3-results outputs/phase3_results \
    --output logs/$(date +%Y%m%d_%H%M%S)_token_metrics.csv \
    --output-json logs/$(date +%Y%m%d_%H%M%S)_token_metrics.json \
    2>&1 | tee logs/token_extraction.log
```

**Output**: 
- `*_token_metrics.csv` - Token counts and compression ratios
- `*_token_metrics.json` - Detailed breakdown

---

## Batch Execution (All Models, All Conditions)

### HF Baselines (All 6 Models)

```bash
ts=$(date +%Y%m%d_%H%M%S)
models=(phi2 llama32_3b qwen25_0_5b qwen25_1_5b qwen25_3b qwen_math_1_5b)

echo "Running baselines for all models..."
for model in "${models[@]}"; do
    echo "[$(date +%H:%M:%S)] $model — no_cot"
    python evaluation.py --model-type "$model" --no-cot --eval_batch_size 8 --seed 42 \
        2>&1 | tee "logs/${ts}_${model}_no_cot.log"
    
    echo "[$(date +%H:%M:%S)] $model — text_cot"
    python evaluation.py --model-type "$model" --eval_batch_size 8 --seed 42 \
        2>&1 | tee "logs/${ts}_${model}_text_cot.log"
done
```

### HF Steering (Supported 3 Models)

```bash
ts=$(date +%Y%m%d_%H%M%S)
steering_models=(phi2 llama32_3b qwen25_3b)
alphas="0 0.5 1 2 5 10 20 50 -0.5 -1"

echo "Running steering for supported models..."
for model in "${steering_models[@]}"; do
    echo "[$(date +%H:%M:%S)] $model — ccot"
    python hidden_steer.py --model-type "$model" --condition ccot --eval_batch_size 8 \
        2>&1 | tee "logs/${ts}_${model}_ccot.log"
    
    echo "[$(date +%H:%M:%S)] $model — random_noise"
    python hidden_steer.py --model-type "$model" --condition random_noise --alphas 1.0 --eval_batch_size 8 \
        2>&1 | tee "logs/${ts}_${model}_random_noise.log"
    
    echo "[$(date +%H:%M:%S)] $model — steered (10 alphas)"
    python hidden_steer.py --model-type "$model" --condition steered --alphas $alphas --eval_batch_size 8 \
        2>&1 | tee "logs/${ts}_${model}_steered_sweep.log"
done
```

---

## Alpha Values

Complete sweep: `0 0.5 1 2 5 10 20 50 -0.5 -1`

| Alpha | Category | Purpose |
|-------|----------|---------|
| 0.0 | Baseline | CCoT unsteered (control) |
| 0.5 | Weak steering | Gentle truth vector guidance |
| 1.0 | Standard steering | Default intervention strength |
| 2.0 | Strong steering | Double strength |
| 5.0 | Very strong | 5x steering magnitude |
| 10.0 | Extreme | Very aggressive intervention |
| 20.0 | Extreme+ | Test limits |
| 50.0 | Maximum | Strongest intervention |
| -0.5 | Negative weak | Opposite direction (sanity check) |
| -1.0 | Negative strong | Strong opposite direction |

---

## Metrics Captured

### Per-Run Metrics (metrics.json)

```json
{
  "accuracy": 0.82,
  "flip_rate": 0.15,
  "mean_cos_sim": 0.45,
  "avg_cot_length": 142,
  "num_examples": 1319
}
```

### Aggregated (CSV Report)

```csv
model,condition,alpha,accuracy,flip_rate,cosine_sim,tokens,compression_ratio
phi2,no_cot,—,0.68,—,—,0,—
phi2,text_cot,—,0.76,—,—,142,—
phi2,ccot,0.0,0.76,0.00,0.42,—,—
phi2,random_noise,1.0,0.75,-0.01,0.35,—,—
phi2,steered,0.5,0.78,0.08,0.48,—,—
phi2,steered,1.0,0.82,0.15,0.55,—,—
...
```

---

## Logging

All logs go to `logs/` directory with ISO timestamp:

```
logs/
├── evaluation_master.log           (main pipeline output)
├── 20260422_120000_split.log
├── 20260422_120015_phase1_train.log
├── 20260422_154020_phase2_extract.log
├── 20260422_160015_phase3_steering.log
├── 20260422_162030_phi2_no_cot.log
├── 20260422_162545_phi2_text_cot.log
├── 20260422_163100_phi2_ccot.log
├── 20260422_163600_phi2_random_noise.log
├── 20260422_164200_phi2_steered_sweep.log
├── ...
├── 20260422_180000_aggregated_results.csv
└── 20260422_180015_token_metrics.csv
```

**View live output**:
```bash
tail -f logs/evaluation_master.log
```

---

## Common Issues & Fixes

### OOM during Phase 1
```bash
python phase1_train.py --batch_size 2 --num_epochs 2
```

### OOM during HF evaluation
```bash
# Reduce batch size for specific model
python evaluation.py --model-type phi2 --eval_batch_size 4
```

### Models not found
```bash
# Login and download
huggingface-cli login
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-3B')"
```

### Permission denied on bash script
```bash
chmod +x run_full_evaluation.sh
```

### Wrong timestamp format
```bash
# Ensure this format exactly (no Unicode):
ts=$(date +%Y%m%d_%H%M%S)
echo "$ts"  # Should print: 20260422_162030
```

---

## Key Files & Outputs

| File | Purpose |
|------|---------|
| `run_full_evaluation.sh` | Main orchestration script |
| `EVALUATION_GUIDE.md` | This document |
| `extract_token_metrics.py` | Token extraction utility |
| `outputs/eval_grid/` | HF model evaluation results |
| `outputs/phase3_results/` | CODI steering results |
| `logs/evaluation_master.log` | Master log file |

---

## Troubleshooting Commands

```bash
# Check GPU memory
nvidia-smi

# Monitor training in real-time
watch -n 2 nvidia-smi

# Find all logs from today
ls -lh logs/*$(date +%Y%m%d)*.log

# Count examples in dataset
wc -l datasets/gsm8k_split/*.jsonl

# View last results
tail -20 logs/*aggregated*.csv

# Kill stuck Python processes
pkill -f "python.*evaluation"
```

---

**Quick Reference Complete** — see `EVALUATION_GUIDE.md` for full documentation
