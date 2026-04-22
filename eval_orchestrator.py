"""
Master evaluation orchestrator for TokenSkip.

Runs comprehensive evaluation across:
  - Multiple models (Llama, Phi, Qwen variants, Mistral)
  - Multiple conditions (no CoT, text CoT, CCoT, random noise, steered)
  - Multiple alpha values (steering strength)
  - Both GSM8K and MATH-500 datasets
  - Tracks: accuracy, flip rate, cosine similarity, faithfulness, token counts
"""

import os
import sys
import json
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Model configurations
MODELS = {
    # Llama
    'llama_3b': {
        'path': 'meta-llama/Llama-3.2-3B',
        'type': 'llama32_3b',
        'size': '3b',
        'supports_steering': True,
    },
    
    # Phi
    'phi2': {
        'path': 'microsoft/phi-2',
        'type': 'phi2',
        'size': '3b',
        'supports_steering': True,
    },
    
    # Qwen 2.5
    'qwen_3b': {
        'path': 'Qwen/Qwen2.5-3B',
        'type': 'qwen25_3b',
        'size': '3b',
        'supports_steering': True,
    },
    'qwen_1_5b': {
        'path': 'Qwen/Qwen2.5-1.5B',
        'type': 'qwen25_3b',
        'size': '1.5b',
        'supports_steering': True,
    },
    'qwen_0_5b': {
        'path': 'Qwen/Qwen2.5-0.5B',
        'type': 'qwen25_3b',
        'size': '0.5b',
        'supports_steering': True,
    },
    
    # Qwen Math
    'qwen_math_1_5b': {
        'path': 'Qwen/Qwen2.5-Math-1.5B',
        'type': 'qwen25_3b',
        'size': '1.5b',
        'supports_steering': True,
    },
    
    # Mistral
    'mistral_7b': {
        'path': 'mistralai/Mistral-7B-Instruct-v0.3',
        'type': 'mistral',
        'size': '7b',
        'supports_steering': False,
    },
}

# Alpha values for steering (includes negative)
ALPHA_VALUES = [0, 0.5, 1, 2, 5, 10, 20, 50, -0.5, -1]

# Evaluation conditions
CONDITIONS = {
    'no_cot': {
        'description': 'Direct answer without any chain-of-thought',
        'flags': ['--no-cot'],
    },
    'text_cot': {
        'description': 'Standard text-based chain-of-thought',
        'flags': [],  # Default behavior
    },
    'ccot': {
        'description': 'Continuous CoT without steering',
        'flags': [],
    },
    'random_noise': {
        'description': 'CCoT with random vector noise (control)',
        'flags': [],
    },
    'steered': {
        'description': 'CCoT with truth vector steering (parameterized by alpha)',
        'flags': [],
    },
}

# Datasets
DATASETS = {
    'gsm8k': {
        'train': 'datasets/gsm8k_split/llm_train.jsonl',  # Phase 1: 4483
        'steer': 'datasets/gsm8k_split/steer_train.jsonl',  # Phase 2: 747
        'val': 'datasets/gsm8k_split/validation.jsonl',  # Phase 3: 2243
        'test': 'datasets/gsm8k_split/test.jsonl',  # Phase 4: 1319
        'total_train': 7473,
    },
    'math': {
        'train': 'datasets/math-500/train.jsonl',
        'test': 'datasets/math-500/test.jsonl',
    },
}

DEFAULT_OUTPUT_DIR = "outputs/eval_comprehensive"
DEFAULT_LOG_DIR = "logs"

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str, run_name: str) -> logging.Logger:
    """Configure logging to file and console."""
    log_path = Path(log_dir) / f"{run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('TokenSkip')
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_eval_command(model_tag: str, dataset: str, condition: str, 
                       alpha: Optional[float] = None, 
                       output_dir: str = None) -> List[str]:
    """Build evaluation command for no_cot/text_cot via evaluation.py."""
    
    model = MODELS[model_tag]
    ds = DATASETS[dataset]
    cond = CONDITIONS[condition]
    
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    
    # Determine output subdirectory
    alpha_str = f"_a{alpha}" if alpha is not None else ""
    result_dir = f"{output_dir}/{model_tag}/{dataset}/{condition}{alpha_str}"
    
    cmd = [
        'python', 'evaluation.py',
        '--model-path', model['path'],
        '--tokenizer-path', model['path'],
        '--model-type', model['type'],
        '--model-size', model['size'],
        '--benchmark', dataset,
        '--data-type', 'test',
        '--eval-data', ds['test'],
        '--output-dir', result_dir,
        '--max_new_tokens', '512',
        '--eval_batch_size', '32',
        '--temperature', '0.0',
        '--seed', '42',
    ]
    
    # Add condition-specific flags
    cmd.extend(cond['flags'])
    
    return cmd


def build_hidden_steer_command(model_tag: str, dataset: str, condition: str,
                               alpha_values: Optional[List[float]] = None,
                               output_dir: str = None) -> List[str]:
    """Build hidden_steer.py command for ccot/random_noise/steered."""
    model = MODELS[model_tag]
    ds = DATASETS[dataset]
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    result_dir = f"{output_dir}/{model_tag}/{dataset}/{condition}"
    vector_dir = f"outputs/truth_vectors"

    cmd = [
        'python', 'hidden_steer.py',
        '--model-path', model['path'],
        '--model-type', model['type'],
        '--eval-data', ds['test'],
        '--steer-data', ds['steer'],
        '--vector-dir', vector_dir,
        '--out-dir', result_dir,
        '--condition', condition,
        '--layer-frac', '0.75',
        '--batch-size', '4',
        '--max-new-tokens', '512',
        '--seed', '42',
    ]
    if condition == 'steered' and alpha_values:
        cmd.extend(['--alphas'] + [str(a) for a in alpha_values])
    return cmd


def build_token_analysis_command(model_tag: str, dataset: str,
                                output_dir: str = None) -> List[str]:
    """Build command to analyze token counts."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    
    result_dir = f"{output_dir}/{model_tag}/{dataset}/token_analysis"
    
    cmd = [
        'python', '-c',
        f"""
from eval.token_counter import TokenCounter, log_token_analysis
import json

counter = TokenCounter('{MODELS[model_tag]['path']}')
results = []

# Load and analyze results
with open('datasets/{dataset}_split/test.jsonl') as f:
    for line in f:
        data = json.loads(line)
        if 'output' in data:
            result = counter.analyze_response(data['output'])
            results.append(result)

log_token_analysis('{result_dir}/token_stats.json', results)
"""
    ]
    
    return cmd


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationOrchestrator:
    """Coordinates multi-model, multi-condition evaluation runs."""
    
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR, 
                 log_dir: str = DEFAULT_LOG_DIR):
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        run_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = setup_logging(str(self.log_dir), run_name)
        self.run_name = run_name
        
        self.results = {}
    
    def run_model(self, model_tag: str, dataset: str = 'gsm8k',
                  conditions: List[str] = None, 
                  alpha_values: List[float] = None,
                  dry_run: bool = False) -> Dict:
        """Run evaluation for a specific model across conditions and alphas."""
        
        if conditions is None:
            conditions = ['no_cot', 'text_cot', 'ccot', 'random_noise', 'steered']
        
        if alpha_values is None:
            alpha_values = [0, 0.5, 1, 2, 5]
        
        model = MODELS[model_tag]
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Starting evaluation: {model_tag} ({model['size']})")
        self.logger.info(f"{'='*70}")
        
        model_results = {
            'model': model_tag,
            'model_path': model['path'],
            'model_type': model['type'],
            'model_size': model['size'],
            'dataset': dataset,
            'timestamp': datetime.now().isoformat(),
            'conditions': {}
        }
        
        for condition in conditions:
            self.logger.info(f"\n  Condition: {condition}")
            model_results['conditions'][condition] = {}

            if condition in ['ccot', 'random_noise', 'steered'] and not model['supports_steering']:
                self.logger.warning(
                    f"    Skipping {condition}: model_type '{model['type']}' is not supported by hidden_steer.py"
                )
                model_results['conditions'][condition]['skipped'] = True
                continue
            
            if condition in ['ccot', 'random_noise']:
                cmd = build_hidden_steer_command(
                    model_tag, dataset, condition,
                    None,
                    str(self.output_dir)
                )
                self._run_command(cmd, model_tag, condition, None, dry_run)
                model_results['conditions'][condition]['completed'] = True

            elif condition == 'steered':
                cmd = build_hidden_steer_command(
                    model_tag, dataset, condition,
                    alpha_values,
                    str(self.output_dir)
                )
                self._run_command(cmd, model_tag, condition, None, dry_run)
                model_results['conditions'][condition]['alphas'] = alpha_values
                model_results['conditions'][condition]['completed'] = True
            else:
                # Single run for non-steered conditions
                cmd = build_eval_command(
                    model_tag, dataset, condition, None,
                    str(self.output_dir)
                )
                self._run_command(cmd, model_tag, condition, None, dry_run)
                model_results['conditions'][condition]['completed'] = True
        
        self.results[model_tag] = model_results
        self.logger.info(f"✓ Completed {model_tag}")
        
        return model_results
    
    def run_all_models(self, dataset: str = 'gsm8k', 
                      dry_run: bool = False) -> Dict:
        """Run evaluation for all models."""
        self.logger.info(f"Starting comprehensive evaluation run")
        self.logger.info(f"Dataset: {dataset}")
        self.logger.info(f"Models: {list(MODELS.keys())}")
        self.logger.info(f"Alpha values: {ALPHA_VALUES}")
        
        for model_tag in MODELS.keys():
            self.run_model(model_tag, dataset, alpha_values=ALPHA_VALUES, 
                          dry_run=dry_run)
        
        # Save results manifest
        self._save_results_manifest()
        return self.results
    
    def _run_command(self, cmd: List[str], model_tag: str, 
                    condition: str, alpha: Optional[float],
                    dry_run: bool = False):
        """Execute a single evaluation command."""
        cmd_str = ' '.join(str(x) for x in cmd)
        
        if dry_run:
            self.logger.info(f"    [DRY RUN] {cmd_str[:100]}...")
            return
        
        self.logger.info(f"    Running: {condition}" + 
                        (f" (alpha={alpha})" if alpha is not None else ""))
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                self.logger.info(f"    ✓ Completed successfully")
            else:
                self.logger.error(f"    ✗ Failed with code {result.returncode}")
                self.logger.error(f"    stderr: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            self.logger.error(f"    ✗ Timeout after 1 hour")
        except Exception as e:
            self.logger.error(f"    ✗ Exception: {e}")
    
    def _save_results_manifest(self):
        """Save results manifest to JSON."""
        manifest_path = self.log_dir / f"{self.run_name}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"\n✓ Results manifest saved: {manifest_path}")


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND GENERATION FOR USERS
# ─────────────────────────────────────────────────────────────────────────────

def print_command_matrix():
    """Print evaluation command matrix for manual execution."""
    
    print("\n" + "="*100)
    print("TOKENSKIP COMPREHENSIVE EVALUATION COMMAND MATRIX")
    print("="*100)
    print(f"\nDataset: GSM8K")
    print(f"Total Training Data: {DATASETS['gsm8k']['total_train']} examples")
    print(f"  - Phase 1 (Base Training): 4483 (60%)")
    print(f"  - Phase 2 (Vector Extraction): 747 (10%)")
    print(f"  - Phase 3 (Validation): 2243 (30%)")
    print(f"  - Phase 4 (Test): 1319")
    
    print(f"\nModels ({len(MODELS)}):")
    for tag, cfg in MODELS.items():
        print(f"  - {tag}: {cfg['path']} ({cfg['size']})")
    
    print(f"\nConditions ({len(CONDITIONS)}):")
    for cond, cfg in CONDITIONS.items():
        print(f"  - {cond}: {cfg['description']}")
    
    print(f"\nAlpha Values (Steering Strength): {ALPHA_VALUES}")
    print(f"  Total command combinations: {len(MODELS)} × {len(CONDITIONS)} × ~{len(ALPHA_VALUES)} = ~{len(MODELS) * len(CONDITIONS) * len(ALPHA_VALUES)}")
    
    print("\n" + "-"*100)
    print("SAMPLE COMMANDS")
    print("-"*100)
    
    # Show a few example commands
    examples = [
        ('phi2', 'gsm8k', 'no_cot', None),
        ('phi2', 'gsm8k', 'text_cot', None),
        ('qwen_3b', 'gsm8k', 'ccot', None),
        ('qwen_3b', 'gsm8k', 'random_noise', None),
        ('qwen_3b', 'gsm8k', 'steered', None),
    ]
    
    for model_tag, dataset, condition, alpha in examples:
        if condition in ['ccot', 'random_noise', 'steered']:
            cmd = build_hidden_steer_command(
                model_tag, dataset, condition,
                ALPHA_VALUES if condition == 'steered' else None
            )
        else:
            cmd = build_eval_command(model_tag, dataset, condition, alpha)
        print(f"\n# Model: {model_tag}, Condition: {condition}" + 
              (f", Alpha: {alpha}" if alpha is not None else ""))
        print(" ".join(cmd))
    
    print("\n" + "="*100)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='TokenSkip comprehensive evaluation orchestrator'
    )
    parser.add_argument(
        '--models', nargs='+', default=list(MODELS.keys()),
        help='Models to evaluate (default: all)'
    )
    parser.add_argument(
        '--conditions', nargs='+', 
        default=['no_cot', 'text_cot', 'ccot', 'random_noise', 'steered'],
        help='Evaluation conditions'
    )
    parser.add_argument(
        '--alphas', nargs='+', type=float, default=ALPHA_VALUES,
        help='Alpha values for steering'
    )
    parser.add_argument(
        '--dataset', default='gsm8k', choices=['gsm8k', 'math'],
        help='Dataset to evaluate on'
    )
    parser.add_argument(
        '--output-dir', default=DEFAULT_OUTPUT_DIR,
        help='Output directory for results'
    )
    parser.add_argument(
        '--log-dir', default=DEFAULT_LOG_DIR,
        help='Directory for logs'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print commands without executing'
    )
    parser.add_argument(
        '--print-commands', action='store_true',
        help='Print command matrix and exit'
    )
    
    args = parser.parse_args()
    
    if args.print_commands:
        print_command_matrix()
        return
    
    orchestrator = EvaluationOrchestrator(
        output_dir=args.output_dir,
        log_dir=args.log_dir
    )
    
    for model_tag in args.models:
        if model_tag not in MODELS:
            print(f"Unknown model: {model_tag}")
            continue
        
        orchestrator.run_model(
            model_tag,
            dataset=args.dataset,
            conditions=args.conditions,
            alpha_values=args.alphas,
            dry_run=args.dry_run
        )


if __name__ == '__main__':
    main()
