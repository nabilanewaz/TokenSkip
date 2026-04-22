"""
Quick reference script for TokenSkip comprehensive evaluation.
Generates all commands needed for a full evaluation run.
"""

import sys

def print_quick_start():
    commands = """
╔══════════════════════════════════════════════════════════════════════════════╗
║           TOKENSKIP COMPREHENSIVE EVALUATION QUICK START                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

STEP 1: Install Dependencies
─────────────────────────────────────────────────────────────────────────────

# Create conda environment
conda create -n tokenskip python=3.12
conda activate tokenskip

# Install all dependencies
cd D:\\Thesis\\TokenSkip
pip install -r requirements-full.txt


STEP 2: Generate Command Matrix (Preview)
─────────────────────────────────────────────────────────────────────────────

python eval_orchestrator.py --print-commands


STEP 3: Run Dry-Run (See What Would Execute)
─────────────────────────────────────────────────────────────────────────────

python eval_orchestrator.py --dry-run --models phi2 qwen_3b --conditions no_cot text_cot


STEP 4: Run Full Evaluation (Complete Matrix)
─────────────────────────────────────────────────────────────────────────────

# Option A: Run ALL models × ALL conditions × ALL alphas
python eval_orchestrator.py \\
    --dataset gsm8k \\
    --output-dir outputs/eval_comprehensive \\
    --log-dir logs

# Option B: Run specific models only
python eval_orchestrator.py \\
    --models phi2 qwen_3b llama_3b \\
    --dataset gsm8k \\
    --output-dir outputs/eval_comprehensive

# Option C: Run specific models + conditions
python eval_orchestrator.py \\
    --models qwen_3b \\
    --conditions no_cot text_cot ccot steered \\
    --alphas 0 0.5 1 2 5 \\
    --dataset gsm8k

# Option D: Run specific models + conditions + custom alpha values
python eval_orchestrator.py \\
    --models qwen_3b qwen_1_5b qwen_0_5b \\
    --conditions steered \\
    --alphas -1 -0.5 0 0.5 1 2 5 10 20 50 \\
    --dataset gsm8k


STEP 5: Examine Individual Model Evaluations
─────────────────────────────────────────────────────────────────────────────

# Single model, single condition (NO CoT)
python evaluation.py \\
    --model-path "microsoft/phi-2" \\
    --model-type phi \\
    --benchmark gsm8k \\
    --output-dir "outputs/eval_comprehensive/phi2/gsm8k/no_cot" \\
    --no-cot

# Single model, text CoT
python evaluation.py \\
    --model-path "meta-llama/Llama-3.2-3B" \\
    --model-type llama \\
    --benchmark gsm8k \\
    --output-dir "outputs/eval_comprehensive/llama_3b/gsm8k/text_cot"

# Single model, continuous CoT (unsteered baseline)
python evaluation.py \\
    --model-path "Qwen/Qwen2.5-3B" \\
    --model-type qwen \\
    --benchmark gsm8k \\
    --output-dir "outputs/eval_comprehensive/qwen_3b/gsm8k/ccot" \\
    --ccot --alpha 0

# Single model, random noise condition (control)
python evaluation.py \\
    --model-path "Qwen/Qwen2.5-3B" \\
    --model-type qwen \\
    --benchmark gsm8k \\
    --output-dir "outputs/eval_comprehensive/qwen_3b/gsm8k/random_noise" \\
    --ccot --condition random_noise

# Single model, steered with specific alpha
python evaluation.py \\
    --model-path "Qwen/Qwen2.5-3B" \\
    --model-type qwen \\
    --benchmark gsm8k \\
    --output-dir "outputs/eval_comprehensive/qwen_3b/gsm8k/steered_a0.5" \\
    --ccot --condition steered --alpha 0.5


STEP 6: Run Alpha Sweep for One Model
─────────────────────────────────────────────────────────────────────────────

# Sweep alpha values: 0, 0.5, 1, 2, 5, 10, 20, 50, -0.5, -1
for alpha in 0 0.5 1 2 5 10 20 50 -0.5 -1; do
    echo "Running alpha=$alpha"
    python evaluation.py \\
        --model-path "Qwen/Qwen2.5-3B" \\
        --model-type qwen \\
        --benchmark gsm8k \\
        --output-dir "outputs/eval_comprehensive/qwen_3b/gsm8k/steered_a$alpha" \\
        --ccot --condition steered --alpha $alpha \\
        --seed 42
done


STEP 7: Extract & Analyze Token Counts
─────────────────────────────────────────────────────────────────────────────

python -c "
from eval.token_counter import TokenCounter
import json

# Initialize counter with specific model
counter = TokenCounter('Qwen/Qwen2.5-3B')

# Analyze sample responses
responses = [
    '[RAT_START] Let me solve this step by step... [RAT_END] The answer is 42',
]

for resp in responses:
    stats = counter.analyze_response(resp)
    print(json.dumps(stats, indent=2))
"


STEP 8: Generate Comprehensive Report
─────────────────────────────────────────────────────────────────────────────

# After all evaluations complete
python eval_report.py \\
    --eval-dir outputs/eval_comprehensive \\
    --output-dir reports


STEP 9: View Results Summary
─────────────────────────────────────────────────────────────────────────────

# Results will be in:
# - reports/results.csv          (all metrics in tabular form)
# - reports/report_meta.json     (metadata + best alphas)
# - logs/eval_YYYYMMDD_HHMMSS_manifest.json  (run manifest)

# View CSV (Excel, pandas, etc.)
import pandas as pd
df = pd.read_csv('reports/results.csv')
print(df.to_string())

# Filter by model and sort by accuracy
df[df['model_tag'] == 'qwen_3b'].sort_values('accuracy', ascending=False)


╔══════════════════════════════════════════════════════════════════════════════╗
║                         KEY METRICS EXPLAINED                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Accuracy
  - Fraction of test examples answered correctly
  - Range: [0, 1]
  - Higher is better

Flip Rate
  - Percentage of previously-wrong examples that become correct with steering
  - Measures how well steering recovers lost accuracy
  - Range: [0, 1]
  - Higher is better

Cosine Similarity
  - Alignment of hidden state trajectory with truth vector
  - Measures how closely steering follows the target direction
  - Range: [-1, 1]
  - Higher (closer to 1) is better

Faithfulness Score
  - Aggregate measure of consistency, drift, trajectory alignment, hallucinations
  - Range: [0, 1]
  - Higher is better

Token Compression Ratio
  - Ratio of compressed tokens to original tokens in CoT
  - Range: [0, 1] (ideally <1 for actual compression)
  - Lower is better


╔══════════════════════════════════════════════════════════════════════════════╗
║                     EVALUATION CONDITIONS SUMMARY                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

no_cot
  - Direct answer without chain-of-thought
  - Baseline for lower-bound performance

text_cot
  - Standard text-based chain-of-thought
  - Baseline for standard LLM reasoning

ccot (alpha=0)
  - Continuous chain-of-thought without steering
  - Baseline for continuous reasoning (unsteered)

random_noise
  - CCoT with random vector injection
  - Control condition to test if any vector helps

steered (alpha swept)
  - CCoT with truth vector steering
  - Alpha values: 0, ±0.5, ±1, 2, 5, 10, 20, 50
  - Primary experimental condition


╔══════════════════════════════════════════════════════════════════════════════╗
║                           MODELS TO EVALUATE                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Llama
  - meta-llama/Llama-3.2-3B

Phi
  - microsoft/phi-2 (2.7B)

Qwen 2.5 Series
  - Qwen/Qwen2.5-3B
  - Qwen/Qwen2.5-1.5B
  - Qwen/Qwen2.5-0.5B

Qwen Math
  - Qwen/Qwen2.5-Math-1.5B

Mistral
  - mistralai/Mistral-7B-Instruct-v0.3


╔══════════════════════════════════════════════════════════════════════════════╗
║                         DATASET INFORMATION                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

GSM8K (Grade School Math 8K)
  - Total train pool: 7,473 examples
    Phase 1 (Base Training):     4,483 (60%)
    Phase 2 (Vector Extraction):   747 (10%)
    Phase 3 (Validation):        2,243 (30%)
  - Test set: 1,319 (held-out, never seen during training)

MATH-500
  - Training set: 7,500
  - Test set: 500


╔══════════════════════════════════════════════════════════════════════════════╗
║                    APPROXIMATE RUNTIME EXPECTATIONS                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Per Model/Condition/Alpha:
  - no_cot:       ~2-3 minutes
  - text_cot:     ~3-5 minutes
  - ccot:         ~3-5 minutes
  - random_noise: ~3-5 minutes
  - steered:      ~3-5 minutes (× 10 alpha values = 30-50 minutes)

Full Matrix (7 models × 5 conditions × 10 alphas = 350 runs):
  - Sequential: ~30-50 hours
  - GPU: 10-15 hours (with batch parallelization)
  - CPU: 100+ hours (not recommended)


╔══════════════════════════════════════════════════════════════════════════════╗
║                         TROUBLESHOOTING                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Out of Memory (OOM)
  - Reduce --eval-batch-size (default 32, try 8 or 4)
  - Use smaller models first (0.5B, 1.5B)
  - Use --device cpu for testing (slow but works)

Model not found
  - Make sure model is on HuggingFace Hub
  - Check internet connection (for model download)
  - Use local path if model already downloaded

Slow evaluation
  - Enable GPU (check CUDA_VISIBLE_DEVICES)
  - Use vLLM for inference speedup
  - Run multiple conditions in parallel (separate terminals)

Missing metrics
  - Check if evaluation.py output saved metrics.json
  - Check logs/ directory for error messages
  - Re-run failed evaluation with --debug flag


╔══════════════════════════════════════════════════════════════════════════════╗
║                        NEXT STEPS                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. Run basic sanity check:
   python eval_orchestrator.py --dry-run --models phi2

2. Run single model evaluation:
   python evaluation.py --model-path microsoft/phi-2 --model-type phi \\
       --benchmark gsm8k --output-dir outputs/test

3. Examine output structure:
   ls outputs/test/

4. Run orchestrator for full evaluation
5. Generate reports
6. Analyze results

"""
    print(commands)


if __name__ == '__main__':
    print_quick_start()
