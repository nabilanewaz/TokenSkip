# TokenSkip Evaluation Command Reference Card

## Installation

```bash
conda create -n tokenskip python=3.12
conda activate tokenskip
pip install -r requirements-full.txt
```

## Quick Commands

### View All Available Commands
```bash
python eval_orchestrator.py --print-commands
```

### Dry Run (Preview)
```bash
python eval_orchestrator.py --dry-run --models phi2 qwen_3b
```

### Run Full Evaluation
```bash
python eval_orchestrator.py --dataset gsm8k
```

### Generate Report
```bash
python eval_report.py --eval-dir outputs/eval_comprehensive --output-dir reports
```

---

## Model Codes & Commands

### Phi-2 No CoT
```bash
python evaluation.py --model-path "microsoft/phi-2" --model-type phi --model-size "2.7B" --benchmark gsm8k --output-dir "outputs/eval_comprehensive/phi2/gsm8k/no_cot" --no-cot --seed 42
```

### Llama 3.2-3B Text CoT
```bash
python evaluation.py --model-path "meta-llama/Llama-3.2-3B" --model-type llama --model-size "3B" --benchmark gsm8k --output-dir "outputs/eval_comprehensive/llama_3b/gsm8k/text_cot" --seed 42
```

### Qwen 2.5-3B CCoT (Unsteered)
```bash
python evaluation.py --model-path "Qwen/Qwen2.5-3B" --model-type qwen --model-size "3B" --benchmark gsm8k --output-dir "outputs/eval_comprehensive/qwen_3b/gsm8k/ccot" --ccot --alpha 0 --seed 42
```

### Qwen 2.5-3B Random Noise (Control)
```bash
python evaluation.py --model-path "Qwen/Qwen2.5-3B" --model-type qwen --model-size "3B" --benchmark gsm8k --output-dir "outputs/eval_comprehensive/qwen_3b/gsm8k/random_noise" --ccot --condition random_noise --seed 42
```

### Qwen 2.5-3B Steered (Alpha=0.5)
```bash
python evaluation.py --model-path "Qwen/Qwen2.5-3B" --model-type qwen --model-size "3B" --benchmark gsm8k --output-dir "outputs/eval_comprehensive/qwen_3b/gsm8k/steered_a0.5" --ccot --condition steered --alpha 0.5 --seed 42
```

### Qwen 2.5-1.5B Steered (Alpha=1)
```bash
python evaluation.py --model-path "Qwen/Qwen2.5-1.5B" --model-type qwen --model-size "1.5B" --benchmark gsm8k --output-dir "outputs/eval_comprehensive/qwen_1_5b/gsm8k/steered_a1" --ccot --condition steered --alpha 1 --seed 42
```

### Qwen 2.5-0.5B Steered (Alpha=2)
```bash
python evaluation.py --model-path "Qwen/Qwen2.5-0.5B" --model-type qwen --model-size "0.5B" --benchmark gsm8k --output-dir "outputs/eval_comprehensive/qwen_0_5b/gsm8k/steered_a2" --ccot --condition steered --alpha 2 --seed 42
```

### Qwen Math 1.5B Steered (Alpha=-1)
```bash
python evaluation.py --model-path "Qwen/Qwen2.5-Math-1.5B" --model-type qwen --model-size "1.5B (Math)" --benchmark gsm8k --output-dir "outputs/eval_comprehensive/qwen_math_1_5b/gsm8k/steered_a-1" --ccot --condition steered --alpha -1 --seed 42
```

### Mistral 7B Steered (Alpha=5)
```bash
python evaluation.py --model-path "mistralai/Mistral-7B-Instruct-v0.3" --model-type mistral --model-size "7B" --benchmark gsm8k --output-dir "outputs/eval_comprehensive/mistral_7b/gsm8k/steered_a5" --ccot --condition steered --alpha 5 --seed 42
```

---

## Alpha Sweep (PowerShell - All Alphas)

```powershell
$model = "Qwen/Qwen2.5-3B"
$model_tag = "qwen_3b"
$alphas = @(0, 0.5, 1, 2, 5, 10, 20, 50, -0.5, -1)

foreach ($alpha in $alphas) {
    Write-Host "Alpha = $alpha"
    python evaluation.py `
        --model-path $model `
        --model-type qwen `
        --model-size "3B" `
        --benchmark gsm8k `
        --output-dir "outputs/eval_comprehensive/$model_tag/gsm8k/steered_a$alpha" `
        --ccot --condition steered --alpha $alpha `
        --seed 42
}
```

---

## Alpha Sweep (Bash)

```bash
for alpha in 0 0.5 1 2 5 10 20 50 -0.5 -1; do
    echo "Alpha = $alpha"
    python evaluation.py \
        --model-path "Qwen/Qwen2.5-3B" \
        --model-type qwen \
        --model-size "3B" \
        --benchmark gsm8k \
        --output-dir "outputs/eval_comprehensive/qwen_3b/gsm8k/steered_a$alpha" \
        --ccot --condition steered --alpha $alpha \
        --seed 42
done
```

---

## Batch Commands (Multiple Models)

### Run All 7 Models - No CoT
```bash
python eval_orchestrator.py --models llama_3b phi2 qwen_3b qwen_1_5b qwen_0_5b qwen_math_1_5b mistral_7b --conditions no_cot --dataset gsm8k
```

### Run All Qwen Variants - Steered Only
```bash
python eval_orchestrator.py --models qwen_3b qwen_1_5b qwen_0_5b qwen_math_1_5b --conditions steered --alphas 0 0.5 1 2 5 --dataset gsm8k
```

### Run All Models - Text CoT
```bash
python eval_orchestrator.py --models llama_3b phi2 qwen_3b qwen_1_5b qwen_0_5b qwen_math_1_5b mistral_7b --conditions text_cot --dataset gsm8k
```

---

## Token Analysis

```python
from eval.token_counter import TokenCounter
import json

counter = TokenCounter('Qwen/Qwen2.5-3B')
response = "[RAT_START] thinking... [RAT_END] answer"
stats = counter.analyze_response(response)
print(json.dumps(stats, indent=2))
```

---

## Results Analysis

```python
import pandas as pd

# Load results
df = pd.read_csv('reports/results.csv')

# Best by accuracy
print(df.nlargest(5, 'accuracy')[['model_tag', 'condition', 'alpha', 'accuracy']])

# Filter by model
qwen = df[df['model_tag'] == 'qwen_3b']
print(qwen.sort_values('accuracy', ascending=False))

# Compare conditions for one model
qwen_no_cot = df[(df['model_tag'] == 'qwen_3b') & (df['condition'] == 'no_cot')]['accuracy'].values[0]
qwen_text = df[(df['model_tag'] == 'qwen_3b') & (df['condition'] == 'text_cot')]['accuracy'].values[0]
qwen_steered = df[(df['model_tag'] == 'qwen_3b') & (df['condition'] == 'steered')]['accuracy'].max()
print(f"no_cot: {qwen_no_cot:.4f}, text_cot: {qwen_text:.4f}, steered: {qwen_steered:.4f}")
```

---

## Condition Summary

| Code | Description | Flag |
|------|-------------|------|
| `no_cot` | Direct answer only | `--no-cot` |
| `text_cot` | Standard text reasoning | (default) |
| `ccot` | Continuous CoT (alpha=0) | `--ccot --alpha 0` |
| `random_noise` | CCoT + random vector | `--ccot --condition random_noise` |
| `steered` | CCoT + truth vector | `--ccot --condition steered --alpha <value>` |

---

## Alpha Values

```
Negative:  -1.0, -0.5      (opposite direction)
Zero:       0.0             (no steering baseline)
Positive:   0.5, 1, 2, 5, 10, 20, 50  (increasing intensity)
```

---

## Dataset Splits (GSM8K)

```
Phase 1 (Train):       4,483 (60%)  → llm_train.jsonl
Phase 2 (Vector):        747 (10%)  → steer_train.jsonl
Phase 3 (Validation):   2,243 (30%) → validation.jsonl
Phase 4 (Test):         1,319       → test.jsonl
─────────────────────────────────────────────────
Total:                  7,473
```

---

## Metrics Explained

- **Accuracy**: % correct answers [0, 1]
- **Flip Rate**: % of wrong→correct with steering [0, 1]
- **Cosine Similarity**: Alignment with truth vector [-1, 1]
- **Faithfulness**: Quality score (consistency - drift) [0, 1]
- **Token Compression**: Compressed/Original tokens [0, 1]

---

## Output Locations

```
outputs/eval_comprehensive/
  {model_tag}/gsm8k/{condition}/metrics.json

logs/
  eval_YYYYMMDD_HHMMSS.log
  eval_YYYYMMDD_HHMMSS_manifest.json

reports/
  results.csv
  report_meta.json
```

---

## Troubleshooting

**OOM Error:**
```bash
--eval-batch-size 8  # Reduce from 32
```

**Slow:**
```bash
--use-vllm  # Enable vLLM speedup
```

**Model not found:**
```bash
huggingface-cli download meta-llama/Llama-3.2-3B
```

**GPU not detected:**
```bash
export CUDA_VISIBLE_DEVICES=0
```

---

## Expected Runtimes

- Per eval: 2-5 minutes
- Per model (5 conditions): 15-25 minutes
- Full matrix (7 models, 5 conditions, 10 alphas): **30-50 hours on GPU**

---

## Common Workflows

### Workflow 1: Quick Test (1 Model, All Conditions)
```bash
# Test on Qwen 3B
python eval_orchestrator.py --models qwen_3b --conditions no_cot text_cot ccot random_noise steered --alphas 0 0.5 1 2 5 --dry-run

# Run it
python eval_orchestrator.py --models qwen_3b --conditions no_cot text_cot ccot random_noise steered --alphas 0 0.5 1 2 5
```

### Workflow 2: Alpha Sensitivity (1 Model, Steered Only)
```bash
# All alpha values for Qwen 3B
python eval_orchestrator.py --models qwen_3b --conditions steered --alphas 0 0.5 1 2 5 10 20 50 -0.5 -1
```

### Workflow 3: Model Comparison (All Sizes)
```bash
# Compare Qwen variants
python eval_orchestrator.py --models qwen_3b qwen_1_5b qwen_0_5b --conditions steered --alphas 0 1 5
```

### Workflow 4: Full Matrix (All Models, All Conditions, All Alphas)
```bash
python eval_orchestrator.py --dataset gsm8k
```

---

## Print This Card

File: `EVAL_COMMANDS.md`

Use for quick reference during evaluation runs.
