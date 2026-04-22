#!/usr/bin/env python3
"""
extract_token_metrics.py
========================
Extract token counts and compute compression ratios across all evaluation runs.

This script aggregates:
  • avg_cot_length from evaluation.py (text_cot baseline token counts)
  • Token stats from hidden_steer.py per-alpha outputs
  • Token stats from phase3_steer_inference.py alpha sweep
  • Compression ratio: (baseline_tokens - steered_tokens) / baseline_tokens

Output formats: CSV and JSON for easy analysis

Usage:
    python extract_token_metrics.py \\
        --eval-grid outputs/eval_grid \\
        --phase3-results outputs/phase3_results \\
        --output report/token_metrics.csv \\
        --output-json report/token_metrics.json
"""

import argparse
import json
import pathlib
import csv
from collections import defaultdict


def extract_text_cot_tokens(eval_grid_dir: pathlib.Path) -> dict:
    """
    Extract avg_cot_length (token count) from text_cot baseline runs.
    
    Returns: {model_type: avg_tokens}
    """
    text_cot_tokens = {}
    eval_grid = pathlib.Path(eval_grid_dir)
    
    # Look for text_cot/samples/metrics.json
    for metrics_path in eval_grid.glob("**/text_cot/samples/metrics.json"):
        model_dir = metrics_path.parent.parent.parent  # back to model dir
        model_type = model_dir.name
        
        try:
            data = json.loads(metrics_path.read_text())
            avg_cot = data.get("avg_cot_length", None)
            if avg_cot is not None:
                text_cot_tokens[model_type] = avg_cot
                print(f"  ✓ {model_type:20s} text_cot baseline: {avg_cot:.0f} tokens")
        except Exception as e:
            print(f"  ✗ {model_type:20s} text_cot error: {e}")
    
    return text_cot_tokens


def extract_steering_tokens(eval_grid_dir: pathlib.Path) -> dict:
    """
    Extract token stats from hidden_steer.py steered condition runs.
    For each (model, alpha) pair, record the token count if available.
    
    Returns: {model_type: {alpha: token_count}}
    """
    steering_tokens = defaultdict(dict)
    eval_grid = pathlib.Path(eval_grid_dir)
    
    # Look for steered/alpha_X/metrics.json
    for metrics_path in eval_grid.glob("**/steered/alpha_*/metrics.json"):
        parts = metrics_path.parent.parent.parent.name  # model name
        model_type = parts
        alpha_dir = metrics_path.parent.name  # "alpha_1.0"
        alpha_str = alpha_dir.replace("alpha_", "")
        
        try:
            alpha = float(alpha_str)
            data = json.loads(metrics_path.read_text())
            # hidden_steer.py currently doesn't emit token counts
            # This is a placeholder for if token tracking is added
            if "avg_cot_length" in data:
                steering_tokens[model_type][alpha] = data["avg_cot_length"]
        except Exception as e:
            pass
    
    return steering_tokens


def extract_phase3_tokens(phase3_results_dir: pathlib.Path) -> dict:
    """
    Extract token stats from phase3_steer_inference.py CODI alpha sweep.
    
    Returns: {alpha: token_stats}
    """
    phase3_tokens = {}
    phase3 = pathlib.Path(phase3_results_dir)
    
    if not phase3.exists():
        return phase3_tokens
    
    # Look for alpha_X/metrics.json
    for metrics_path in phase3.glob("alpha_*/metrics.json"):
        alpha_dir = metrics_path.parent.name  # "alpha_0.0"
        alpha_str = alpha_dir.replace("alpha_", "")
        
        try:
            alpha = float(alpha_str)
            data = json.loads(metrics_path.read_text())
            # phase3_steer_inference.py doesn't currently emit token counts
            # This is placeholder for future integration
            if "mean_tokens" in data:
                phase3_tokens[alpha] = data["mean_tokens"]
        except Exception as e:
            pass
    
    return phase3_tokens


def compute_compression_ratios(text_cot_baseline: dict, steered: dict) -> dict:
    """
    Compute compression ratio for each model's steered conditions vs text_cot baseline.
    
    compression_ratio = (baseline_tokens - steered_tokens) / baseline_tokens
    Positive = compression (fewer tokens needed)
    Negative = expansion (more tokens needed)
    """
    ratios = {}
    
    for model_type in steered:
        if model_type not in text_cot_baseline:
            continue
        
        base_tokens = text_cot_baseline[model_type]
        ratios[model_type] = {}
        
        for alpha, steered_tokens in steered[model_type].items():
            if steered_tokens and base_tokens:
                ratio = (base_tokens - steered_tokens) / base_tokens
                ratios[model_type][alpha] = ratio
    
    return ratios


def create_csv_report(
    text_cot_tokens: dict,
    compression_ratios: dict,
    output_path: pathlib.Path,
):
    """
    Create CSV report with token metrics.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = [
        ["Model Type", "Text CoT Baseline Tokens", "Note"],
    ]
    
    for model_type in sorted(text_cot_tokens.keys()):
        tokens = text_cot_tokens[model_type]
        note = ""
        
        if model_type in compression_ratios:
            alphas = sorted(compression_ratios[model_type].keys())
            avg_compression = sum(compression_ratios[model_type].values()) / len(alphas)
            note = f"Avg compression: {avg_compression:.1%} (alphas: {alphas})"
        
        rows.append([model_type, f"{tokens:.0f}", note])
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"\n  ✓ CSV report: {output_path}")


def create_json_report(
    text_cot_tokens: dict,
    steering_tokens: dict,
    phase3_tokens: dict,
    compression_ratios: dict,
    output_path: pathlib.Path,
):
    """
    Create detailed JSON report with all token metrics.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": pathlib.Path.cwd().name,
        "text_cot_baseline": {
            model: {
                "avg_tokens": tokens,
                "description": "Average CoT tokens from text_cot evaluation"
            }
            for model, tokens in text_cot_tokens.items()
        },
        "steering_per_alpha": {
            model: {
                str(alpha): tokens
                for alpha, tokens in alphas.items()
            }
            for model, alphas in steering_tokens.items()
        },
        "phase3_codi": {
            str(alpha): tokens
            for alpha, tokens in phase3_tokens.items()
        },
        "compression_ratios": {
            model: {
                str(alpha): ratio
                for alpha, ratio in alphas.items()
            }
            for model, alphas in compression_ratios.items()
        },
        "notes": {
            "text_cot_baseline": (
                "Average number of tokens in the chain-of-thought from text_cot baseline runs. "
                "This is the standard CoT performance without any steering."
            ),
            "steering_per_alpha": (
                "Token counts per alpha value for each model (currently placeholder; "
                "hidden_steer.py does not yet emit token stats)."
            ),
            "phase3_codi": (
                "Token counts from CODI phase3_steer_inference.py alpha sweep "
                "(currently placeholder; phase3 does not yet emit token stats)."
            ),
            "compression_ratios": (
                "Positive = fewer tokens (compression). "
                "Negative = more tokens (expansion). "
                "Calculated as (baseline - steered) / baseline."
            ),
            "future": (
                "To enable per-alpha token tracking: "
                "1. Modify hidden_steer.py to count tokens per alpha "
                "2. Modify phase3_steer_inference.py to emit token counts "
                "3. Add avg_cot_length or token_count to each alpha's metrics.json"
            ),
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print(f"  ✓ JSON report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract token metrics from all evaluation runs"
    )
    parser.add_argument("--eval-grid", default="outputs/eval_grid",
                        help="Directory with HF model eval results")
    parser.add_argument("--phase3-results", default="outputs/phase3_results",
                        help="Directory with Phase 3 CODI steering results")
    parser.add_argument("--output", default="report/token_metrics.csv",
                        help="Output CSV file")
    parser.add_argument("--output-json", default="report/token_metrics.json",
                        help="Output JSON file")
    args = parser.parse_args()
    
    eval_grid = pathlib.Path(args.eval_grid)
    phase3 = pathlib.Path(args.phase3_results)
    output_csv = pathlib.Path(args.output)
    output_json = pathlib.Path(args.output_json)
    
    print("\n" + "=" * 70)
    print("  Extract Token Metrics & Compression Ratios")
    print("=" * 70 + "\n")
    
    # Extract from all sources
    print("[1] Extracting text_cot baseline tokens from evaluation.py...")
    text_cot_tokens = extract_text_cot_tokens(eval_grid)
    print(f"    Found: {len(text_cot_tokens)} models\n")
    
    print("[2] Extracting steered condition tokens from hidden_steer.py...")
    steering_tokens = extract_steering_tokens(eval_grid)
    print(f"    Found: {len(steering_tokens)} models (with steering data)\n")
    
    print("[3] Extracting Phase 3 CODI tokens from phase3_steer_inference.py...")
    phase3_tokens = extract_phase3_tokens(phase3)
    print(f"    Found: {len(phase3_tokens)} alpha values\n")
    
    print("[4] Computing compression ratios...")
    compression_ratios = compute_compression_ratios(text_cot_tokens, steering_tokens)
    print(f"    Computed ratios for {len(compression_ratios)} models\n")
    
    print("[5] Writing reports...")
    create_csv_report(text_cot_tokens, compression_ratios, output_csv)
    create_json_report(text_cot_tokens, steering_tokens, phase3_tokens,
                      compression_ratios, output_json)
    
    print("\n" + "=" * 70)
    print("  Token Metrics Extraction Complete")
    print("=" * 70)
    print(f"\n  CSV output  : {output_csv}")
    print(f"  JSON output : {output_json}")
    print(f"\n  NOTE: Per-alpha token tracking is a FUTURE feature.")
    print(f"  Currently, avg_cot_length is only available from text_cot baselines.")
    print(f"  To enable per-alpha compression tracking:")
    print(f"    1. Extend hidden_steer.py to emit token counts")
    print(f"    2. Extend phase3_steer_inference.py to emit token counts")
    print(f"    3. Re-run steering stages\n")


if __name__ == "__main__":
    main()
