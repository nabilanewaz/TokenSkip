# TokenSkip Comprehensive Evaluation Guide

A complete framework for evaluating TokenSkip across multiple models, conditions, and steering strengths.

## Overview

This guide provides a structured approach to evaluate TokenSkip using:

- **7 Models**: Llama 3.2-3B, Phi-2, Qwen 2.5 (3B/1.5B/0.5B), Qwen Math 1.5B, Mistral 7B
- **5 Evaluation Conditions**: No CoT, Text CoT, Continuous CoT, Random Noise (control), Steered
- **10 Alpha Values**: 0, ±0.5, ±1, 2, 5, 10, 20, 50 (steering strength sweep)
- **6 Metrics**: Accuracy, Flip Rate, Cosine Similarity, Faithfulness, Token Counts, Compression Ratio

## Prerequisites

### System Requirements

```bash
# Python 3.12
python --version

# CUDA-capable GPU (recommended)
# For CPU-only: operations will be 10-50x slower

# Disk space: ~100-200GB for model weights + results
```

### Installation

```bash
# Create environment
conda create -n tokenskip python=3.12
conda activate tokenskip

# Install dependencies
cd D:\Thesis\TokenSkip
pip install -r requirements-full.txt
```

## Quick Start

### 1. View Available Commands

```bash
python eval_orchestrator.py --print-commands
```

### 2. Dry Run (See What Would Execute)

```bash
python eval_orchestrator.py --dry-run --models phi2 qwen_3b
```

### 3. Run Full Evaluation

```bash
# All models × all conditions × all alphas
python eval_orchestrator.py --dataset gsm8k

# Specific subset
python eval_orchestrator.py \
    --models qwen_3b qwen_1_5b \
    --conditions steered \
    --alphas 0 0.5 1 2 5
```

### 4. Generate Report

```bash
python eval_report.py --eval-dir outputs/eval_comprehensive
```

## Individual Model Evaluation Examples

### Example 1: Phi-2, No CoT

```bash
python evaluation.py \
    --model-path "microsoft/phi-2" \
    --model-type phi \
    --model-size "2.7B" \
    --benchmark gsm8k \
    --data-type test \
    --eval-data "datasets/gsm8k_split/test.jsonl" \
    --output-dir "outputs/eval_comprehensive/phi2/gsm8k/no_cot" \
    --max-new-tokens 512 \
    --eval-batch-size 32 \
    --temperature 0.0 \
    --seed 42 \
    --no-cot
```

### Example 2: Qwen 3B, Text CoT

```bash
python evaluation.py \
    --model-path "Qwen/Qwen2.5-3B" \
    --model-type qwen \
    --model-size "3B" \
    --benchmark gsm8k \
    --output-dir "outputs/eval_comprehensive/qwen_3b/gsm8k/text_cot" \
    --max-new-tokens 512 \
    --eval-batch-size 32 \
    --temperature 0.0 \
    --seed 42
```

### Example 3: Qwen 3B, Steered with Alpha=0.5

```bash
python evaluation.py \
    --model-path "Qwen/Qwen2.5-3B" \
    --model-type qwen \
    --model-size "3B" \
    --benchmark gsm8k \
    --output-dir "outputs/eval_comprehensive/qwen_3b/gsm8k/steered_a0.5" \
    --max-new-tokens 512 \
    --eval-batch-size 32 \
    --temperature 0.0 \
    --seed 42 \
    --ccot --condition steered --alpha 0.5
```

## Alpha Sweep (PowerShell)

```powershell
$alphas = @(0, 0.5, 1, 2, 5, 10, 20, 50, -0.5, -1)
foreach ($alpha in $alphas) {
    Write-Host "Running alpha=$alpha"
    python evaluation.py `
        --model-path "Qwen/Qwen2.5-3B" `
        --model-type qwen `
        --benchmark gsm8k `
        --output-dir "outputs/eval_comprehensive/qwen_3b/gsm8k/steered_a$alpha" `
        --ccot --condition steered --alpha $alpha `
        --seed 42
}
```

## Dataset Information

### GSM8K Splits

```
Total train pool: 7,473

Phase 1 (Base Training):      4,483 (60%) - llm_train.jsonl
Phase 2 (Vector Extraction):    747 (10%) - steer_train.jsonl
Phase 3 (Validation):         2,243 (30%) - validation.jsonl
Phase 4 (Test):               1,319      - test.jsonl
```

## Models

| Model | Path | Size |
|-------|------|------|
| Llama 3.2 | meta-llama/Llama-3.2-3B | 3B |
| Phi-2 | microsoft/phi-2 | 2.7B |
| Qwen 2.5-3B | Qwen/Qwen2.5-3B | 3B |
| Qwen 2.5-1.5B | Qwen/Qwen2.5-1.5B | 1.5B |
| Qwen 2.5-0.5B | Qwen/Qwen2.5-0.5B | 0.5B |
| Qwen Math 1.5B | Qwen/Qwen2.5-Math-1.5B | 1.5B |
| Mistral | mistralai/Mistral-7B-Instruct-v0.3 | 7B |

## Alpha Values

```
Negative:  -1.0, -0.5     (opposite steering direction)
Zero:       0.0           (no steering / baseline)
Positive:   0.5, 1, 2, 5, 10, 20, 50  (increasing intensity)
```

## Evaluation Conditions

| Condition | Description |
|-----------|-------------|
| no_cot | Direct answer, no reasoning |
| text_cot | Standard text chain-of-thought |
| ccot | Continuous CoT (unsteered, alpha=0) |
| random_noise | CCoT + random vector (control) |
| steered | CCoT + truth vector steering (alpha swept) |

## Metrics

| Metric | Range | Definition |
|--------|-------|-----------|
| Accuracy | [0, 1] | Fraction of correct predictions |
| Flip Rate | [0, 1] | % of wrong→correct with steering |
| Cosine Similarity | [-1, 1] | Alignment with truth direction |
| Faithfulness | [0, 1] | Quality + consistency score |
| Token Compression | [0, 1] | Compressed / Original tokens |

## Report Generation

```bash
# Generate CSV and summary
python eval_report.py \
    --eval-dir outputs/eval_comprehensive \
    --output-dir reports
```

Results:
- `reports/results.csv` - All metrics
- `reports/report_meta.json` - Best alphas per model

## View Results in Python

```python
import pandas as pd

# Load all results
df = pd.read_csv('reports/results.csv')

# Filter by model and condition
qwen_steered = df[(df['model_tag'] == 'qwen_3b') & (df['condition'] == 'steered')]

# Sort by accuracy
top_results = qwen_steered.sort_values('accuracy', ascending=False)
print(top_results[['alpha', 'accuracy', 'faithfulness', 'token_compression']])
```

## File Structure

```
outputs/eval_comprehensive/
├── llama_3b/gsm8k/{condition}/metrics.json
├── phi2/gsm8k/{condition}/metrics.json
├── qwen_3b/gsm8k/{condition}/metrics.json
├── qwen_1_5b/gsm8k/{condition}/metrics.json
├── qwen_0_5b/gsm8k/{condition}/metrics.json
├── qwen_math_1_5b/gsm8k/{condition}/metrics.json
└── mistral_7b/gsm8k/{condition}/metrics.json

logs/
├── eval_YYYYMMDD_HHMMSS.log
└── eval_YYYYMMDD_HHMMSS_manifest.json
```

## Token Analysis

```python
from eval.token_counter import TokenCounter
import json

counter = TokenCounter('Qwen/Qwen2.5-3B')

# Analyze response
response = "[RAT_START] Let me think... [RAT_END] Answer: 42"
stats = counter.analyze_response(response)
print(json.dumps(stats, indent=2))
```

## Faithfulness Metrics

```python
from eval.faithfulness import FaithfulnessMetrics

metrics = FaithfulnessMetrics()

# Check consistency
predictions = ["42", "42", "42"]
consistency = metrics.consistency_score(predictions)
print(f"Consistency: {consistency}")  # 1.0
```

## Expected Runtimes

- Per condition: 2-5 minutes
- Per model (all 5 conditions): 15-25 minutes
- Full matrix (7 models, 5 conditions, 10 alphas): 30-50 hours on GPU

## Troubleshooting

**Out of Memory:**
```bash
python evaluation.py ... --eval-batch-size 8
```

**Slow Evaluation:**
```bash
# Enable GPU
export CUDA_VISIBLE_DEVICES=0

# Use vLLM if supported
python evaluation.py ... --use-vllm
```

**Missing Model:**
```bash
# Pre-download
huggingface-cli download meta-llama/Llama-3.2-3B
```

## See Also

- [framework.md](framework.md) - Research protocol details
- [Readme.md](Readme.md) - TokenSkip overview
- [COLAB_SETUP.md](COLAB_SETUP.md) - Colab training
