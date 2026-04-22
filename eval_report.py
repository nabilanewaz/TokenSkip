"""
Results aggregation and reporting for TokenSkip evaluation.

Combines metrics from all evaluation runs and generates:
- Comprehensive results table
- CSV exports for analysis
- Visualizations
- Comparison plots
"""

import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime


class ResultsAggregator:
    """Aggregates and analyzes TokenSkip evaluation results."""
    
    def __init__(self, eval_output_dir: str):
        self.eval_dir = Path(eval_output_dir)
        self.results = {}
        self.metrics_data = []
    
    def discover_results(self) -> Dict:
        """Discover and load all metrics.json files from eval results."""
        self.results = {}
        
        for model_dir in self.eval_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_tag = model_dir.name
            self.results[model_tag] = {'conditions': {}}
            
            for dataset_dir in model_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                
                dataset = dataset_dir.name
                if dataset not in self.results[model_tag]:
                    self.results[model_tag][dataset] = {}
                
                for condition_dir in dataset_dir.iterdir():
                    if not condition_dir.is_dir():
                        continue
                    
                    condition = condition_dir.name
                    metrics_file = condition_dir / 'metrics.json'
                    
                    if metrics_file.exists():
                        with open(metrics_file) as f:
                            metrics = json.load(f)
                            self.results[model_tag]['conditions'][condition] = metrics
        
        return self.results
    
    def build_comparison_table(self, dataset: str = 'gsm8k') -> List[Dict]:
        """
        Build comparison table with all metrics.
        
        Returns:
            List of dicts with columns:
            - model_tag
            - model_size
            - condition
            - alpha
            - accuracy
            - fliprate
            - cosine_similarity
            - faithfulness
            - token_ratio
        """
        rows = []
        
        for model_tag, model_results in self.results.items():
            if 'conditions' not in model_results:
                continue
            
            # Extract model size if available
            model_info = model_results.get('model_info', {})
            model_size = model_info.get('size', 'unknown')
            
            for condition, metrics in model_results['conditions'].items():
                # Check if this is a steered alpha variant
                alpha_val = None
                if condition.startswith('alpha_'):
                    alpha_val = float(condition.replace('alpha_', ''))
                    condition_name = 'steered'
                elif '_a' in condition:
                    parts = condition.rsplit('_a', 1)
                    if len(parts) == 2 and parts[1].replace('.', '', 1).lstrip('-').isdigit():
                        condition_name = parts[0]
                        alpha_val = float(parts[1])
                    else:
                        condition_name = condition
                else:
                    condition_name = condition
                
                row = {
                    'model_tag': model_tag,
                    'model_size': model_size,
                    'condition': condition_name,
                    'alpha': alpha_val,
                    'accuracy': metrics.get('accuracy', None),
                    'fliprate': metrics.get('fliprate', None),
                    'cosine_similarity': metrics.get('cosine_similarity', None),
                    'faithfulness': metrics.get('faithfulness_score', None),
                    'token_compression': metrics.get('compression_ratio', None),
                    'mean_cot_tokens': metrics.get('mean_cot_tokens', None),
                    'mean_answer_tokens': metrics.get('mean_answer_tokens', None),
                }
                rows.append(row)
        
        return rows
    
    def export_to_csv(self, output_path: str, table: List[Dict]):
        """Export comparison table to CSV."""
        if not table:
            print("No results to export")
            return
        
        keys = table[0].keys()
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(table)
        
        print(f"✓ Results exported to {output_path}")
    
    def print_summary_table(self, table: List[Dict]):
        """Print human-readable summary table."""
        if not table:
            print("No results to display")
            return
        
        print("\n" + "="*150)
        print("TOKENSKIP COMPREHENSIVE EVALUATION RESULTS")
        print("="*150)
        
        # Group by model
        by_model = {}
        for row in table:
            model = row['model_tag']
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(row)
        
        for model_tag in sorted(by_model.keys()):
            rows = by_model[model_tag]
            model_size = rows[0]['model_size']
            
            print(f"\n{'─'*150}")
            print(f"Model: {model_tag} ({model_size})")
            print(f"{'─'*150}")
            print(f"{'Condition':<15} {'Alpha':<8} {'Accuracy':<10} {'Flip Rate':<10} "
                  f"{'Cosine Sim':<12} {'Faithfulness':<12} {'Token Ratio':<12}")
            print("-"*150)
            
            for row in sorted(rows, key=lambda x: (x['condition'], x.get('alpha') or 0)):
                condition = row['condition']
                alpha = row['alpha'] if row['alpha'] is not None else '-'
                accuracy = f"{row['accuracy']:.4f}" if row['accuracy'] is not None else 'N/A'
                fliprate = f"{row['fliprate']:.4f}" if row['fliprate'] is not None else 'N/A'
                cosine = f"{row['cosine_similarity']:.4f}" if row['cosine_similarity'] is not None else 'N/A'
                faith = f"{row['faithfulness']:.4f}" if row['faithfulness'] is not None else 'N/A'
                token = f"{row['token_compression']:.4f}" if row['token_compression'] is not None else 'N/A'
                
                print(f"{condition:<15} {str(alpha):<8} {accuracy:<10} {fliprate:<10} "
                      f"{cosine:<12} {faith:<12} {token:<12}")
    
    def compute_best_alpha(self, table: List[Dict]) -> Dict:
        """Find best alpha value for each model/condition combo."""
        best = {}
        
        steered_rows = [r for r in table if r['condition'] == 'steered' and r['alpha'] is not None]
        
        for row in steered_rows:
            key = (row['model_tag'], row['condition'])
            
            if key not in best:
                best[key] = row
            else:
                # Compare by accuracy (primary), then by faithfulness
                if (row['accuracy'] or 0) > (best[key]['accuracy'] or 0):
                    best[key] = row
                elif ((row['accuracy'] or 0) == (best[key]['accuracy'] or 0) and
                      (row['faithfulness'] or 0) > (best[key]['faithfulness'] or 0)):
                    best[key] = row
        
        return best
    
    def generate_report(self, output_dir: str = 'reports'):
        """Generate comprehensive HTML report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Discover and build table
        self.discover_results()
        table = self.build_comparison_table()
        best_alphas = self.compute_best_alpha(table)
        
        # Export CSV
        csv_path = output_path / 'results.csv'
        self.export_to_csv(str(csv_path), table)
        
        # Print summary
        self.print_summary_table(table)
        
        # Print best alphas
        print(f"\n{'─'*100}")
        print("BEST ALPHA VALUES (by accuracy)")
        print(f"{'─'*100}")
        for (model, condition), row in best_alphas.items():
            print(f"{model:20} | Alpha: {row['alpha']:6.1f} | "
                  f"Accuracy: {row['accuracy']:.4f} | "
                  f"Faithfulness: {row['faithfulness']:.4f}")
        
        # Save report metadata
        report_meta = {
            'generated': datetime.now().isoformat(),
            'eval_dir': str(self.eval_dir),
            'num_models': len(set(r['model_tag'] for r in table)),
            'num_conditions': len(set(r['condition'] for r in table)),
            'num_results': len(table),
            'best_alphas': {str(k): v['alpha'] for k, v in best_alphas.items()},
        }
        
        with open(output_path / 'report_meta.json', 'w') as f:
            json.dump(report_meta, f, indent=2)
        
        print(f"\n✓ Report generated in {output_path}")
        print(f"  - results.csv")
        print(f"  - report_meta.json")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate and report TokenSkip evaluation results'
    )
    parser.add_argument(
        '--eval-dir', default='outputs/eval_comprehensive',
        help='Directory containing evaluation results'
    )
    parser.add_argument(
        '--output-dir', default='reports',
        help='Output directory for reports'
    )
    
    args = parser.parse_args()
    
    aggregator = ResultsAggregator(args.eval_dir)
    aggregator.generate_report(args.output_dir)


if __name__ == '__main__':
    main()
