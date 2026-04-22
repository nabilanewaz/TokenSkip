# TokenSkip Steering Evaluation Pipeline

## Overview

Comprehensive evaluation framework for the TokenSkip steering research, measuring whether injecting a "Truth Vector" into continuous reasoning latent spaces improves factuality and reduces hallucination across multiple models.

## Models & Conditions

### Models (7 total)

| Model | Type | HF ID | Steering Support |
|-------|------|-------|------------------|
| CODI-GPT2 | Base (CODI) | zen-E/CODI-gpt2 | ✓ Phase 3 |
| Phi-2 | HF Transformer | microsoft/phi-2 | ✓ hidden_steer.py |
| Llama-3.2-3B | HF Transformer | meta-llama/Llama-3.2-3B | ✓ hidden_steer.py |
| Qwen2.5-0.5B | HF Transformer | Qwen/Qwen2.5-0.5B | ⚠ Need extension |
| Qwen2.5-1.5B | HF Transformer | Qwen/Qwen2.5-1.5B | ⚠ Need extension |
| Qwen2.5-3B | HF Transformer | Qwen/Qwen2.5-3B | ✓ hidden_steer.py |
| Qwen2.5-Math-1.5B | HF Transformer | Qwen/Qwen2.5-Math-1.5B | ⚠ Need extension |

**Note**: Mistral excluded per protocol. Full steering support for all 6 HF models requires extending `hidden_steer.py` and `model_registry.py`.

### Evaluation Conditions

For each model, the following conditions are evaluated:

| Condition | Alpha | Description | Supported |
|-----------|-------|-------------|-----------|
| **No CoT** | — | Direct answer, no reasoning | ✓ All 6 HF models |
| **Text CoT** | — | Standard text chain-of-thought | ✓ All 6 HF models + CODI |
| **CCoT** | 0.0 | Continuous CoT, unsteered baseline | ✓ 3 HF + CODI |
| **Random Noise** | 1.0 | Continuous CoT + random vector (control) | ✓ 3 HF + CODI |
| **Steered** | sweep | Continuous CoT + truth vector (α sweep) | ✓ 3 HF + CODI |

### Alpha Sweep Values

10 values: `[0, 0.5, 1, 2, 5, 10, 20, 50, -0.5, -1]`

- α = 0.0 → unsteered baseline
- α > 0 → increasing steering strength
- α < 0 → opposite direction (control for directionality)

## Metrics

| Metric | Definition | Source |
|--------|-----------|--------|
| **Accuracy** | % correct final answers on test set | All runs |
| **Flip Rate** | % of baseline failures corrected by steering | hidden_steer.py, phase3 |
| **Cosine Similarity** | avg cos(h_t, v_truth) across steps | hidden_steer.py, phase3 |
| **Faithfulness** | Reasoning consistency and hallucination detection | Future: faithfulness.py |
| **Token Count** | CoT length (baseline text_cot) | evaluation.py |
| **Compression Ratio** | (baseline_tokens - steered_tokens) / baseline_tokens | extract_token_metrics.py |

## Dataset Configuration

Total dataset: **8,792 examples** (7,473 train pool + 1,319 test)

| Split | Size | Purpose | Usage |
|-------|------|---------|-------|
| **llm_train** | 4,483 | Base model training (60% of pool) | Phase 1 |
| **steer_train** | 747 | Truth vector extraction (10% of pool) | Phase 2 |
| **validation** | 2,243 | Hyperparameter tuning (30% of pool) | Optional testing |
| **test** | 1,319 | Final held-out evaluation (separate) | Phase 3 + HF eval |

**Critical Rule**: `steer_train` is the ONLY split used to compute `v_truth`. The `validation` split must never touch Phase 2.

## Quick Start

### 1. Run Full Evaluation Pipeline

```bash
# Make script executable
chmod +x run_full_evaluation.sh

# Run with logging to master log file
bash run_full_evaluation.sh 2>&1 | tee logs/evaluation_master.log

# Or run individual stages by editing STAGE_* variables in the script
```

### 2. Run Individual Stages (Manual)

```bash
# Stage 1: Split dataset
python split_dataset.py --full --seed 42

# Stage 2: Phase 1 training
python phase1_train.py --num_epochs 3 --batch_size 4 --bf16

# Stage 3: Phase 2 extraction
python phase2_extract_vector.py --n-samples 5 --bf16

# Stage 4: Phase 3 steering
python phase3_steer_inference.py \
    --alphas 0 0.5 1 2 5 10 20 50 -0.5 -1 \
    --random-noise --bf16

# Stage 5: HF model baselines (example: phi2)
python evaluation.py --model-type phi2 --no-cot --temperature 0.0
python evaluation.py --model-type phi2 --temperature 0.0  # text_cot (default)

# Stage 6: HF model steering (example: phi2)
python hidden_steer.py --model-type phi2 --condition ccot
python hidden_steer.py --model-type phi2 --condition random_noise --alphas 1.0
python hidden_steer.py --model-type phi2 --condition steered --alphas 0 0.5 1 2 5 10 20 50 -0.5 -1

# Stage 7: Aggregate results
python compare_all.py --eval-grid outputs/eval_grid

# Stage 8: Extract token metrics
python extract_token_metrics.py
```

## Output Structure

```
outputs/
├── phase1_checkpoint/              (CODI base model)
│   ├── model.safetensors or pytorch_model.bin
│   ├── train_log.txt
│   └── phase1_metadata.json
├── phase2_truth_vector/            (Truth vector artifacts)
│   ├── v_truth.pt                  (global vector)
│   ├── v_truth_per_step.pt         (per-step vectors)
│   ├── sigma_per_step.pt           (activation std)
│   ├── latent_dump.pt              (raw trace data)
│   └── stats.json
├── phase3_results/                 (CODI steering sweep)
│   ├── alpha_0.0/
│   │   └── metrics.json            (accuracy, flip_rate, mean_cos_sim)
│   ├── alpha_0.5/
│   │   └── metrics.json
│   ├── ...
│   ├── summary.json                (all alphas in one table)
│   ├── flip_analysis.json          (per-alpha flip statistics)
│   └── trajectory_stats.json       (cosine similarity per step)
└── eval_grid/                      (HF model evaluations)
    ├── phi2/
    │   ├── no_cot/
    │   │   └── samples/metrics.json
    │   ├── text_cot/
    │   │   └── samples/metrics.json
    │   ├── ccot/
    │   │   └── samples/metrics.json
    │   ├── random_noise/
    │   │   └── samples/metrics.json
    │   └── steered/
    │       ├── alpha_0/
    │       │   └── metrics.json
    │       ├── alpha_0.5/
    │       │   └── metrics.json
    │       └── ...
    ├── llama32_3b/
    │   ├── no_cot/
    │   ├── text_cot/
    │   ├── ccot/
    │   ├── random_noise/
    │   └── steered/
    ├── qwen25_0_5b/
    │   ├── no_cot/
    │   └── text_cot/
    │   (⚠ no steering yet — requires extension)
    ├── qwen25_1_5b/
    │   ├── no_cot/
    │   └── text_cot/
    │   (⚠ no steering yet — requires extension)
    ├── qwen25_3b/
    │   ├── no_cot/
    │   ├── text_cot/
    │   ├── ccot/
    │   ├── random_noise/
    │   └── steered/
    └── qwen_math_1_5b/
        ├── no_cot/
        └── text_cot/
        (⚠ no steering yet — requires extension)

logs/
├── evaluation_master.log           (main pipeline log)
├── {ts}_split.log
├── {ts}_phase1_train.log
├── {ts}_phase2_extract.log
├── {ts}_phase3_steering.log
├── {ts}_${model_type}_*.log       (per-model logs)
├── {ts}_aggregated_results.csv    (all metrics table)
├── {ts}_token_metrics.csv         (token counts & compression)
└── {ts}_token_metrics.json        (detailed token breakdown)
```

## Important Notes

### Model Support for Steering

Currently, **only 3 of 6 HF models** support steering in `hidden_steer.py`:
- ✓ phi2 (microsoft/phi-2)
- ✓ llama32_3b (meta-llama/Llama-3.2-3B)
- ✓ qwen25_3b (Qwen/Qwen2.5-3B)

To extend steering support to the remaining 3 models:

1. **Add model type to `model_registry.py`**:
   ```python
   "qwen25_0_5b": lambda: QwenPromptBuilder(variant="0.5b"),
   "qwen25_1_5b": lambda: QwenPromptBuilder(variant="1.5b"),
   "qwen_math_1_5b": lambda: QwenMathPromptBuilder(),
   ```

2. **Update `hidden_steer.py` supported models**:
   ```python
   SUPPORTED_MODELS = ["phi2", "llama32_3b", "qwen25_3b", 
                       "qwen25_0_5b", "qwen25_1_5b", "qwen_math_1_5b"]
   ```

3. **Test steering pipeline** for new models

4. **Re-run Stage 6** with full model set

### Token Metrics

**Current Status**: 
- ✓ `avg_cot_length` available from `evaluation.py` text_cot runs
- ⚠ Per-alpha token counts NOT YET IMPLEMENTED in steering code

**To enable full token tracking**:

1. Modify `hidden_steer.py` to count tokens per alpha:
   ```python
   metrics["avg_cot_length"] = np.mean([len(tokenizer.encode(pred)) for pred in predictions])
   ```

2. Modify `phase3_steer_inference.py` similarly

3. Re-run Stages 4 & 6 to regenerate metrics.json files

4. Token compression ratios will automatically populate in Stage 8

### GPU Requirements

- **Phase 1** (training): ~2-4h on T4/A100, ~100h on CPU
- **Phase 2** (extraction): ~15 min on GPU
- **Phase 3** (CODI steering): ~1h for 10 alphas
- **HF evaluation**: ~2-3h total (parallel per model recommended)

**Recommended**: 1x A100 or 2x T4 GPUs for full run in ~6-8h

### Reproducibility

All scripts use `--seed 42` by default. For different random seeds:
```bash
# Edit run_full_evaluation.sh and change seed in each invocation
```

## Expected Results

### Baseline Accuracies (from prior runs)
- **Text CoT**: ~70-80% (varies by model)
- **No CoT**: ~40-50%

### Steering Impact (hypothesized)
- **CCoT unsteered** (α=0): ~75-85% (baseline for steering)
- **Steered optimal α**: 85-92% (expected improvement)
- **Random noise**: ~75-85% (should NOT improve over baseline)
- **Flip rate**: 10-25% of baseline errors corrected

## Troubleshooting

### OOM errors during Phase 1
```bash
# Reduce batch size
python phase1_train.py --batch_size 2 --num_epochs 3
```

### Model not found errors
```bash
# Ensure HF login and download
huggingface-cli login
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-3B')"
```

### CUDA OOM during HF steering
```bash
# Reduce eval_batch_size
python hidden_steer.py --model-type phi2 --condition steered --eval_batch_size 4
```

### Missing logs
```bash
# All logs go to logs/ directory
ls -lh logs/
tail -f logs/evaluation_master.log
```

## References

- **Framework**: `framework.md`
- **Phase 1 (Training)**: `phase1_train.py`
- **Phase 2 (Extraction)**: `phase2_extract_vector.py`
- **Phase 3 (Steering)**: `phase3_steer_inference.py`
- **HF Baselines**: `evaluation.py`
- **HF Steering**: `hidden_steer.py`
- **Aggregation**: `compare_all.py`
- **Token Extraction**: `extract_token_metrics.py`

## Contact

For issues or questions about the evaluation pipeline, refer to the logs and check `framework.md` for the underlying research protocol.

---

**Last Updated**: April 2026
**Protocol**: "Steering Continuous Reasoning via Latent Intervention"
**Dataset**: GSM8K (8,792 examples)
**Models**: 7 (1 CODI + 6 HF)
**Conditions**: 5 (no_cot, text_cot, ccot, random_noise, steered)
**Alpha Values**: 10 ([0, ±0.5, ±1, 2, 5, 10, 20, 50])
