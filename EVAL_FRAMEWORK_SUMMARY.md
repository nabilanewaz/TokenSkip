# TokenSkip Evaluation Framework - What You Got

## Summary

I've created a **complete evaluation framework** for your TokenSkip project that enables comprehensive multi-model, multi-condition testing across all your specified parameters.

## Files Created

### 1. **requirements-full.txt** ✓
- Complete dependency list with all missing packages
- Includes: numpy, pandas, tqdm, regex, sympy, matplotlib, seaborn, safetensors, huggingface_hub, wandb
- Ready to install: `pip install -r requirements-full.txt`

### 2. **eval/token_counter.py** ✓
- Token counting and compression ratio analysis
- Extracts CoT portions from responses
- Aggregates statistics across runs
- Tracks: total tokens, CoT tokens, answer tokens, compression ratio

### 3. **eval/faithfulness.py** ✓
- Faithfulness metrics for steering evaluation
- Measures: consistency, steering drift, trajectory alignment, hallucination detection
- Computes aggregate faithfulness scores [0, 1]

### 4. **eval_orchestrator.py** ✓
**Main orchestration script** that handles:
- All 7 models (Llama, Phi, Qwen 2.5 variants, Qwen Math, Mistral)
- All 5 evaluation conditions (no_cot, text_cot, ccot, random_noise, steered)
- All 10 alpha values (0, ±0.5, ±1, 2, 5, 10, 20, 50)
- Comprehensive logging to `logs/` directory
- Manifest generation for reproducibility

**Usage:**
```bash
# Preview all commands
python eval_orchestrator.py --print-commands

# Dry run
python eval_orchestrator.py --dry-run

# Run full evaluation
python eval_orchestrator.py --dataset gsm8k

# Run specific subset
python eval_orchestrator.py --models phi2 qwen_3b --conditions steered --alphas 0 1 5
```

### 5. **eval_report.py** ✓
- Discovers all evaluation results
- Builds comprehensive comparison table
- Exports to CSV for analysis
- Generates summary table with best alphas
- Produces report metadata

**Usage:**
```bash
python eval_report.py --eval-dir outputs/eval_comprehensive --output-dir reports
```

### 6. **EVAL_QUICKSTART.py** ✓
- Printable quick-start guide
- Shows all key commands
- Explains each condition and metric
- Provides troubleshooting guide

**Usage:**
```bash
python EVAL_QUICKSTART.py
```

### 7. **EVAL_COMPREHENSIVE.md** ✓
- Complete evaluation guide (Markdown)
- Detailed examples for each model/condition combo
- Dataset information and splits
- Runtime expectations
- File structure documentation

### 8. **EVAL_COMMANDS.md** ✓
- Command reference card (printable)
- Quick lookup for all commands
- Model codes and alpha sweep examples
- PowerShell and Bash versions
- Common workflows

## What You Can Do Now

### 1. Basic Setup
```bash
pip install -r requirements-full.txt
```

### 2. Single Model Evaluation
- Phi-2 (no CoT)
- Llama 3.2-3B (text CoT)
- Qwen 2.5-3B (steered with alpha=0.5)
- etc.

### 3. Alpha Sweep
Test steering strength sensitivity:
```powershell
# All 10 alpha values for one model
for $alpha in @(0, 0.5, 1, 2, 5, 10, 20, 50, -0.5, -1) {
    python evaluation.py ... --alpha $alpha
}
```

### 4. Model Comparison
- Compare across model sizes (0.5B, 1.5B, 3B, 7B)
- Compare Qwen variants
- Compare different architectures

### 5. Condition Analysis
- Measure gap: no_cot → text_cot → ccot → steered
- Validate control (random_noise)
- Find optimal alpha per model

### 6. Metrics Tracking
- Accuracy per model/condition/alpha
- Flip rate (wrong→correct recovery)
- Cosine similarity with truth vector
- Faithfulness score (consistency + trajectory alignment)
- Token compression ratios

### 7. Results Reporting
- Export to CSV for further analysis
- Generate summary tables
- Identify best alpha values
- Compare across all dimensions

## File Organization

```
outputs/eval_comprehensive/
├── llama_3b/gsm8k/
│   ├── no_cot/metrics.json
│   ├── text_cot/metrics.json
│   ├── ccot/metrics.json
│   ├── random_noise/metrics.json
│   └── steered_aX/metrics.json (10 alpha variants)
├── phi2/gsm8k/
├── qwen_3b/gsm8k/
├── qwen_1_5b/gsm8k/
├── qwen_0_5b/gsm8k/
├── qwen_math_1_5b/gsm8k/
└── mistral_7b/gsm8k/

logs/
├── eval_20240422_120000.log
└── eval_20240422_120000_manifest.json

reports/
├── results.csv (all metrics in table format)
└── report_meta.json (best alphas + metadata)
```

## Key Metrics Captured

| Metric | Range | Definition |
|--------|-------|-----------|
| **Accuracy** | [0, 1] | Fraction of correct predictions |
| **Flip Rate** | [0, 1] | % of wrong examples that become correct |
| **Cosine Similarity** | [-1, 1] | Alignment with truth steering direction |
| **Faithfulness** | [0, 1] | Consistency + trajectory quality score |
| **Token Compression** | [0, 1] | Compressed tokens / Original tokens |
| **CoT Tokens** | N/A | Mean token count in reasoning |

## Models Supported

1. **Llama**: meta-llama/Llama-3.2-3B (3B)
2. **Phi**: microsoft/phi-2 (2.7B)
3. **Qwen 2.5 Series**:
   - Qwen/Qwen2.5-3B (3B)
   - Qwen/Qwen2.5-1.5B (1.5B)
   - Qwen/Qwen2.5-0.5B (0.5B)
4. **Qwen Math**: Qwen/Qwen2.5-Math-1.5B (1.5B)
5. **Mistral**: mistralai/Mistral-7B-Instruct-v0.3 (7B)

## Conditions Supported

1. **no_cot** - Direct answer (no reasoning)
2. **text_cot** - Standard text chain-of-thought
3. **ccot** - Continuous CoT (alpha=0, unsteered)
4. **random_noise** - CCoT + random vector (control)
5. **steered** - CCoT + truth vector steering (alpha swept)

## Alpha Values Supported

```
-1.0, -0.5  (negative steering)
 0.0        (no steering baseline)
 0.5, 1, 2, 5, 10, 20, 50  (positive steering)
```

## Dataset Information

**GSM8K** (Grade School Math 8K):
- Phase 1 (Training): 4,483 examples (60%)
- Phase 2 (Vector Extraction): 747 examples (10%)
- Phase 3 (Validation): 2,243 examples (30%)
- Phase 4 (Test): 1,319 examples
- **Total: 7,473 training + 1,319 test**

## Example Workflows

### Workflow 1: Quick Test
```bash
# Preview commands for Qwen 3B
python eval_orchestrator.py --models qwen_3b --dry-run

# Run full evaluation for one model
python eval_orchestrator.py --models qwen_3b
```

### Workflow 2: Alpha Sensitivity
```bash
# Test all alpha values for Qwen 3B (steered only)
python eval_orchestrator.py \
    --models qwen_3b \
    --conditions steered \
    --alphas 0 0.5 1 2 5 10 20 50 -0.5 -1
```

### Workflow 3: Model Comparison
```bash
# Compare all Qwen sizes on steered condition
python eval_orchestrator.py \
    --models qwen_3b qwen_1_5b qwen_0_5b \
    --conditions steered \
    --alphas 0 1 5
```

### Workflow 4: Full Matrix (Complete Evaluation)
```bash
# All models × All conditions × All alphas
python eval_orchestrator.py --dataset gsm8k
```

### Workflow 5: Post-Evaluation Analysis
```bash
# Generate reports
python eval_report.py --eval-dir outputs/eval_comprehensive --output-dir reports

# Analyze in Python
import pandas as pd
df = pd.read_csv('reports/results.csv')
best = df.nlargest(10, 'accuracy')[['model_tag', 'condition', 'alpha', 'accuracy']]
print(best)
```

## Expected Runtime

- **Per evaluation**: 2-5 minutes
- **Per model (all 5 conditions)**: 15-25 minutes
- **Full matrix** (7 models × 5 conditions × 10 alphas ≈ 350 runs): **30-50 hours on GPU**

## Next Steps

1. **Install dependencies:**
   ```bash
   pip install -r requirements-full.txt
   ```

2. **Test the setup:**
   ```bash
   python eval_orchestrator.py --dry-run --models phi2
   ```

3. **Run a quick evaluation:**
   ```bash
   python eval_orchestrator.py --models phi2 --conditions no_cot text_cot --dry-run
   ```

4. **Run full evaluation:**
   ```bash
   python eval_orchestrator.py --dataset gsm8k
   ```

5. **Generate results:**
   ```bash
   python eval_report.py --eval-dir outputs/eval_comprehensive --output-dir reports
   ```

6. **Analyze results:**
   ```python
   import pandas as pd
   df = pd.read_csv('reports/results.csv')
   print(df.sort_values('accuracy', ascending=False).head())
   ```

## Troubleshooting

- **Out of Memory**: Add `--eval-batch-size 8` to reduce batch size
- **Slow**: Enable GPU or use `--use-vllm` for speedup
- **Model not found**: Run `huggingface-cli download model-name`
- **Missing GPU**: Set `export CUDA_VISIBLE_DEVICES=0`

## Documentation Files

1. **EVAL_COMMANDS.md** - Quick reference card (print-friendly)
2. **EVAL_COMPREHENSIVE.md** - Complete guide with examples
3. **EVAL_QUICKSTART.py** - Interactive quick-start (run with `python`)
4. **This file** - Overview and summary

## Questions?

- Check `EVAL_COMPREHENSIVE.md` for detailed examples
- Check `EVAL_COMMANDS.md` for quick command lookup
- Run `python EVAL_QUICKSTART.py` for interactive guide
- Check logs in `logs/` directory for error messages

---

**Status**: Framework complete and ready to use! ✓

All scripts tested for syntax and integrated with existing codebase.
You now have everything needed to run comprehensive TokenSkip evaluation.
