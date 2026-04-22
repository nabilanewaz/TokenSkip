#!/usr/bin/env python3
"""
TokenSkip Evaluation Framework - Visual Overview & Command Builder
"""

def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              TOKENSKIP COMPREHENSIVE EVALUATION FRAMEWORK                   ║
║                                                                              ║
║  Complete evaluation toolkit for multi-model, multi-alpha steering tests    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_structure():
    structure = """
FRAMEWORK STRUCTURE
═══════════════════════════════════════════════════════════════════════════════

EVALUATION GRID:
┌────────────────────────────────────────────────────────────────────────────┐
│  7 MODELS                                                                  │
│  ├─ Llama 3.2-3B                                                           │
│  ├─ Phi-2                                                                  │
│  ├─ Qwen 2.5-3B / 1.5B / 0.5B                                             │
│  ├─ Qwen Math 1.5B                                                         │
│  └─ Mistral 7B                                                             │
│                                                                             │
│  × 5 CONDITIONS (per model)                                               │
│    ├─ no_cot              (direct answer)                                  │
│    ├─ text_cot            (text reasoning)                                 │
│    ├─ ccot                (continuous reasoning, alpha=0)                 │
│    ├─ random_noise        (control - random vector)                        │
│    └─ steered             (truth vector steering, alpha swept)             │
│                                                                             │
│  × 10 ALPHA VALUES (for steered condition)                                │
│    ├─ -1.0, -0.5 (negative/opposite steering)                            │
│    ├─ 0.0        (no steering baseline)                                   │
│    └─ 0.5, 1, 2, 5, 10, 20, 50  (increasing intensity)                  │
│                                                                             │
│  = ~350 TOTAL EVALUATION RUNS                                              │
│                                                                             │
│  METRICS PER RUN: accuracy, flip_rate, cosine_sim, faithfulness, tokens  │
└────────────────────────────────────────────────────────────────────────────┘

COMMAND FLOW:
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  1. Install                                                                │
│     └─ pip install -r requirements-full.txt                              │
│                                                                             │
│  2. Preview Commands                                                      │
│     └─ python eval_orchestrator.py --print-commands                      │
│                                                                             │
│  3. Dry Run (No Execution)                                                │
│     └─ python eval_orchestrator.py --dry-run --models phi2 qwen_3b      │
│                                                                             │
│  4. Run Evaluation                                                        │
│     └─ python eval_orchestrator.py [--models] [--conditions] [--alphas]  │
│                                                                             │
│  5. Generate Report                                                       │
│     └─ python eval_report.py --eval-dir outputs/eval_comprehensive      │
│                                                                             │
│  6. Analyze Results                                                       │
│     └─ pandas read_csv('reports/results.csv')                            │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘

MODULES CREATED:
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  eval/token_counter.py      → Token counting & compression analysis       │
│  eval/faithfulness.py       → Faithfulness metrics                        │
│  eval_orchestrator.py       → Main orchestration (models, conditions)     │
│  eval_report.py             → Results aggregation & reporting             │
│  EVAL_QUICKSTART.py         → Interactive quick-start guide               │
│  EVAL_COMPREHENSIVE.md      → Detailed documentation                      │
│  EVAL_COMMANDS.md           → Command reference card                      │
│  requirements-full.txt      → Complete dependency list                    │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘

OUTPUT STRUCTURE:
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  outputs/eval_comprehensive/                                              │
│  ├─ llama_3b/gsm8k/                                                       │
│  │  ├─ no_cot/metrics.json                                                │
│  │  ├─ text_cot/metrics.json                                              │
│  │  ├─ ccot/metrics.json                                                  │
│  │  ├─ random_noise/metrics.json                                          │
│  │  └─ steered_a*.json  (10 alpha variants)                               │
│  ├─ phi2/gsm8k/ ...                                                       │
│  ├─ qwen_3b/gsm8k/ ...                                                    │
│  └─ [other models]                                                        │
│                                                                             │
│  logs/                                                                     │
│  ├─ eval_YYYYMMDD_HHMMSS.log         (detailed log)                       │
│  └─ eval_YYYYMMDD_HHMMSS_manifest.json  (run metadata)                    │
│                                                                             │
│  reports/                                                                  │
│  ├─ results.csv                    (all metrics in table)                  │
│  └─ report_meta.json               (best alphas + summary)                 │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
"""
    print(structure)

def print_quick_commands():
    quick = """
QUICK COMMAND REFERENCE
═══════════════════════════════════════════════════════════════════════════════

SETUP:
  pip install -r requirements-full.txt

VIEW AVAILABLE COMMANDS:
  python eval_orchestrator.py --print-commands

DRY RUN (Preview without executing):
  python eval_orchestrator.py --dry-run --models phi2

RUN SINGLE MODEL (All conditions):
  python eval_orchestrator.py --models qwen_3b

RUN SPECIFIC CONDITIONS:
  python eval_orchestrator.py --models qwen_3b --conditions steered --alphas 0 1 5

RUN FULL MATRIX (All models × conditions × alphas):
  python eval_orchestrator.py --dataset gsm8k

GENERATE RESULTS REPORT:
  python eval_report.py --eval-dir outputs/eval_comprehensive --output-dir reports

VIEW COMMAND REFERENCE:
  cat EVAL_COMMANDS.md

READ COMPREHENSIVE GUIDE:
  cat EVAL_COMPREHENSIVE.md

RUN INTERACTIVE QUICKSTART:
  python EVAL_QUICKSTART.py

═══════════════════════════════════════════════════════════════════════════════
"""
    print(quick)

def print_model_codes():
    models = """
MODEL CODES & PATHS
═══════════════════════════════════════════════════════════════════════════════

  llama_3b          → meta-llama/Llama-3.2-3B  (3B)
  phi2              → microsoft/phi-2  (2.7B)
  
  qwen_3b           → Qwen/Qwen2.5-3B  (3B)
  qwen_1_5b         → Qwen/Qwen2.5-1.5B  (1.5B)
  qwen_0_5b         → Qwen/Qwen2.5-0.5B  (0.5B)
  qwen_math_1_5b    → Qwen/Qwen2.5-Math-1.5B  (1.5B Math)
  
  mistral_7b        → mistralai/Mistral-7B-Instruct-v0.3  (7B)

═══════════════════════════════════════════════════════════════════════════════
"""
    print(models)

def print_conditions():
    conditions = """
EVALUATION CONDITIONS
═══════════════════════════════════════════════════════════════════════════════

  no_cot        Direct answer without reasoning (baseline lower bound)
  text_cot      Standard text-based chain-of-thought (baseline)
  ccot          Continuous CoT unsteered (alpha=0, baseline)
  random_noise  CCoT with random vector (negative control)
  steered       CCoT with truth vector steering (experimental, alpha swept)

═══════════════════════════════════════════════════════════════════════════════
"""
    print(conditions)

def print_alphas():
    alphas = """
ALPHA VALUES (STEERING STRENGTH)
═══════════════════════════════════════════════════════════════════════════════

  -1.0   Strongest opposite direction steering
  -0.5   Weak opposite direction
   0.0   ◄── NO STEERING (baseline)
   0.5   Weak positive steering
   1.0   Standard steering intensity
   2.0   2× standard intensity
   5.0   5× standard intensity
  10.0   10× standard intensity
  20.0   20× standard intensity
  50.0   50× standard intensity (very strong)

Recommendation: Start with 0, 0.5, 1, 2, 5 then sweep larger values if interesting

═══════════════════════════════════════════════════════════════════════════════
"""
    print(alphas)

def print_metrics():
    metrics = """
METRICS CAPTURED
═══════════════════════════════════════════════════════════════════════════════

Metric                 Range      Definition
─────────────────────────────────────────────────────────────────────────────
Accuracy              [0, 1]     Fraction of correct predictions
Flip Rate             [0, 1]     % of wrong→correct conversions with steering
Cosine Similarity     [-1, 1]    Alignment with truth steering vector
Faithfulness          [0, 1]     Consistency + trajectory quality - drift
Token Compression     [0, 1]     Compressed tokens / Original tokens
Mean CoT Tokens       N/A        Average token count in reasoning output

All saved to: outputs/eval_comprehensive/{model}/{condition}/metrics.json

═══════════════════════════════════════════════════════════════════════════════
"""
    print(metrics)

def print_example_commands():
    examples = """
EXAMPLE COMMANDS
═══════════════════════════════════════════════════════════════════════════════

Example 1: Phi-2, No CoT (direct answer)
  python evaluation.py \\
    --model-path "microsoft/phi-2" \\
    --model-type phi \\
    --benchmark gsm8k \\
    --output-dir "outputs/eval_comprehensive/phi2/gsm8k/no_cot" \\
    --no-cot

Example 2: Qwen 3B, Alpha sweep (all steering strengths)
  for alpha in 0 0.5 1 2 5 10 20 50 -0.5 -1; do
    python evaluation.py \\
      --model-path "Qwen/Qwen2.5-3B" \\
      --model-type qwen \\
      --benchmark gsm8k \\
      --output-dir "outputs/eval_comprehensive/qwen_3b/gsm8k/steered_a$alpha" \\
      --ccot --condition steered --alpha $alpha
  done

Example 3: Compare Qwen sizes on steered condition
  python eval_orchestrator.py \\
    --models qwen_3b qwen_1_5b qwen_0_5b \\
    --conditions steered \\
    --alphas 0 1 5

Example 4: Full evaluation (7 models, all conditions, all alphas)
  python eval_orchestrator.py --dataset gsm8k

═══════════════════════════════════════════════════════════════════════════════
"""
    print(examples)

def print_workflows():
    workflows = """
RECOMMENDED WORKFLOWS
═══════════════════════════════════════════════════════════════════════════════

WORKFLOW 1: Quick Sanity Check (15 minutes)
  1. python eval_orchestrator.py --dry-run --models phi2
  2. python evaluation.py [one command from output]
  3. ls outputs/eval_comprehensive/phi2/gsm8k/no_cot/

WORKFLOW 2: Single Model Full Evaluation (2-3 hours)
  1. python eval_orchestrator.py --models qwen_3b
  2. Wait for 5 conditions × 10 alphas to complete
  3. python eval_report.py --eval-dir outputs/eval_comprehensive

WORKFLOW 3: Model Comparison (5-10 hours)
  1. python eval_orchestrator.py --models qwen_3b qwen_1_5b phi2
  2. Filter by model in results.csv
  3. Compare performance metrics across sizes

WORKFLOW 4: Alpha Sensitivity (4-6 hours)
  1. python eval_orchestrator.py --models qwen_3b --conditions steered \\
       --alphas 0 0.5 1 2 5 10 20 50 -0.5 -1
  2. Plot accuracy vs alpha value
  3. Find optimal steering strength

WORKFLOW 5: Full Comprehensive Evaluation (30-50 hours on GPU)
  1. python eval_orchestrator.py --dataset gsm8k
  2. Monitor with: tail -f logs/eval_*.log
  3. After completion: python eval_report.py ...
  4. Analyze with pandas in Python

═══════════════════════════════════════════════════════════════════════════════
"""
    print(workflows)

def print_dataset_info():
    dataset = """
DATASET: GSM8K (Grade School Math 8K)
═══════════════════════════════════════════════════════════════════════════════

Total Training Examples: 7,473

  Phase 1 - Base Training:     4,483 (60%)  →  llm_train.jsonl
  Phase 2 - Vector Extraction:   747 (10%)  →  steer_train.jsonl
  Phase 3 - Validation:        2,243 (30%)  →  validation.jsonl
  ─────────────────────────────────────────────────────────────
  Subtotal (training pool):    7,473

  Phase 4 - Test:              1,319       →  test.jsonl
                               (from original GSM8K test set)
  ─────────────────────────────────────────────────────────────
  TOTAL:                       8,792

Key Points:
  • Test set is HELD OUT during all phases 1-3
  • No data leakage between phases
  • Standard GSM8K test set ensures reproducibility

═══════════════════════════════════════════════════════════════════════════════
"""
    print(dataset)

def print_runtimes():
    runtimes = """
EXPECTED RUNTIMES
═══════════════════════════════════════════════════════════════════════════════

Per Single Evaluation:
  no_cot               2-3 minutes
  text_cot             3-5 minutes
  ccot                 3-5 minutes
  random_noise         3-5 minutes
  steered (1 alpha)    3-5 minutes

Per Model (all 5 conditions, 10 alphas average):
  Sequential:          15-25 minutes per model

Full Matrix Runtimes:
  7 models × 5 conditions × 10 alphas ≈ 350 total runs

  GPU (recommended):       30-50 hours
  CPU (not recommended):   100+ hours
  4 GPUs parallel:         8-12 hours

Estimated Per Hour:
  Single GPU:   ~7-10 runs/hour
  CPU:          ~2-3 runs/hour

To speed up:
  • Use GPU with CUDA
  • Enable vLLM for inference speedup
  • Run multiple models in parallel (separate terminals)
  • Use smaller batch sizes if OOM

═══════════════════════════════════════════════════════════════════════════════
"""
    print(runtimes)

def print_troubleshooting():
    troubleshooting = """
TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════════

ISSUE: Out of Memory (OOM)
  Fix 1: Reduce batch size
    --eval-batch-size 8  (instead of 32)
  Fix 2: Use smaller model
    --model-path "Qwen/Qwen2.5-0.5B"
  Fix 3: Use CPU (slow)
    --device cpu

ISSUE: Model not found / Download error
  Fix 1: Pre-download model
    huggingface-cli download meta-llama/Llama-3.2-3B
  Fix 2: Check internet connection
  Fix 3: Set cache directory
    export HF_HOME=/path/to/cache

ISSUE: GPU not detected / Very slow inference
  Fix 1: Enable GPU
    export CUDA_VISIBLE_DEVICES=0
  Fix 2: Use vLLM
    --use-vllm
  Fix 3: Check CUDA installation
    python -c "import torch; print(torch.cuda.is_available())"

ISSUE: Evaluation hangs / Times out
  Fix 1: Set timeout
    Add timeout parameter to subprocess call
  Fix 2: Check GPU memory
    nvidia-smi (watch for increasing memory usage)
  Fix 3: Check logs
    tail -f logs/eval_*.log

ISSUE: Missing results / Incomplete runs
  Fix 1: Check output directory
    ls outputs/eval_comprehensive/MODEL/gsm8k/CONDITION/
  Fix 2: Check log files
    cat logs/eval_*.log | grep ERROR
  Fix 3: Re-run failed condition
    python evaluation.py [same arguments]

═══════════════════════════════════════════════════════════════════════════════
"""
    print(troubleshooting)

def main():
    import sys
    
    print_banner()
    
    if len(sys.argv) > 1:
        section = sys.argv[1].lower()
    else:
        section = 'all'
    
    if section in ['all', 'structure']:
        print_structure()
    
    if section in ['all', 'quick']:
        print_quick_commands()
    
    if section in ['all', 'models']:
        print_model_codes()
    
    if section in ['all', 'conditions']:
        print_conditions()
    
    if section in ['all', 'alphas']:
        print_alphas()
    
    if section in ['all', 'metrics']:
        print_metrics()
    
    if section in ['all', 'examples']:
        print_example_commands()
    
    if section in ['all', 'workflows']:
        print_workflows()
    
    if section in ['all', 'dataset']:
        print_dataset_info()
    
    if section in ['all', 'runtimes']:
        print_runtimes()
    
    if section in ['all', 'troubleshooting']:
        print_troubleshooting()
    
    footer = f"""
═══════════════════════════════════════════════════════════════════════════════
For more details:
  • EVAL_COMMANDS.md        → Command reference card
  • EVAL_COMPREHENSIVE.md   → Complete guide
  • EVAL_FRAMEWORK_SUMMARY.md → What was created
  • EVAL_QUICKSTART.py      → Interactive guide

Run "python EVAL_OVERVIEW.py [section]" for specific topics:
  structure, quick, models, conditions, alphas, metrics, examples,
  workflows, dataset, runtimes, troubleshooting

═══════════════════════════════════════════════════════════════════════════════
"""
    print(footer)

if __name__ == '__main__':
    main()
