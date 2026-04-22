"""
Faithfulness metrics for evaluating steering fidelity.

Faithfulness measures how well the model stays grounded when steering is applied,
including consistency checks and reasoning quality assessments.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class FaithfulnessMetrics:
    """Compute faithfulness scores for steered reasoning outputs."""
    
    @staticmethod
    def consistency_score(predictions: List[str], texts: List[str] = None) -> float:
        """
        Measure consistency of predictions across multiple runs.
        Higher score = more consistent predictions.
        """
        if len(predictions) < 2:
            return 1.0
        
        unique_preds = len(set(predictions))
        consistency = 1.0 - (unique_preds - 1) / len(predictions)
        return max(0.0, min(1.0, consistency))
    
    @staticmethod
    def steering_drift(original_hidden_states: np.ndarray, 
                      steered_hidden_states: np.ndarray,
                      max_drift_threshold: float = 5.0) -> Dict:
        """
        Measure how much steering has moved the model's internal representations.
        
        Args:
            original_hidden_states: Hidden states without steering (N, D)
            steered_hidden_states: Hidden states with steering (N, D)
            max_drift_threshold: Euclidean distance threshold
        
        Returns:
            Dict with drift metrics
        """
        if original_hidden_states.shape != steered_hidden_states.shape:
            return {'error': 'Shape mismatch'}
        
        # Euclidean distance
        diffs = steered_hidden_states - original_hidden_states
        distances = np.linalg.norm(diffs, axis=-1)  # Per-token distances
        
        return {
            'mean_drift': float(np.mean(distances)),
            'median_drift': float(np.median(distances)),
            'max_drift': float(np.max(distances)),
            'std_drift': float(np.std(distances)),
            'excessive_drift_ratio': float(np.mean(distances > max_drift_threshold))
        }
    
    @staticmethod
    def answer_consistency_with_reasoning(reasoning: str, answer: str,
                                         required_terms: List[str] = None) -> float:
        """
        Check if final answer is logically supported by the reasoning.
        Higher score = more faithful reasoning leading to answer.
        """
        score = 1.0
        
        # Basic check: does reasoning mention the answer?
        if answer and answer not in reasoning:
            score -= 0.3
        
        # Check for required intermediate terms
        if required_terms:
            terms_found = sum(1 for term in required_terms if term.lower() in reasoning.lower())
            coverage = terms_found / len(required_terms) if required_terms else 1.0
            score *= coverage
        
        return max(0.0, min(1.0, score))
    
    @staticmethod
    def trajectory_alignment(hidden_states: np.ndarray, 
                           truth_vector: np.ndarray) -> float:
        """
        Measure alignment of hidden state trajectory with truth vector.
        
        Args:
            hidden_states: Sequence of hidden states (T, D)
            truth_vector: Target direction vector (D,)
        
        Returns:
            Mean cosine similarity across trajectory
        """
        if hidden_states.shape[-1] != truth_vector.shape[0]:
            return 0.0
        
        # Normalize
        truth_norm = truth_vector / (np.linalg.norm(truth_vector) + 1e-8)
        
        cosines = []
        for h in hidden_states:
            h_norm = h / (np.linalg.norm(h) + 1e-8)
            cos_sim = np.dot(h_norm, truth_norm)
            cosines.append(cos_sim)
        
        return float(np.mean(cosines)) if cosines else 0.0
    
    @staticmethod
    def hallucination_detection(text: str, ground_truth_entities: List[str]) -> Dict:
        """
        Detect potential hallucinations by checking for entities
        not in ground truth.
        """
        text_lower = text.lower()
        
        # Extract numeric values as potential hallucinations
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        gt_numbers = re.findall(r'\d+(?:\.\d+)?', ' '.join(ground_truth_entities))
        
        spurious_numbers = [n for n in numbers if n not in gt_numbers]
        
        return {
            'has_spurious_numbers': len(spurious_numbers) > 0,
            'spurious_number_count': len(spurious_numbers),
            'hallucination_ratio': len(spurious_numbers) / max(1, len(numbers)) if numbers else 0.0
        }
    
    @staticmethod
    def compute_faithfulness_score(metrics: Dict) -> float:
        """
        Aggregate multiple faithfulness components into single score.
        
        Weights:
        - consistency: 0.3
        - minimal drift: 0.2
        - trajectory alignment: 0.3
        - no hallucinations: 0.2
        """
        score = 0.0
        
        # Consistency component
        if 'consistency' in metrics:
            score += 0.3 * metrics['consistency']
        
        # Drift component (lower is better)
        if 'mean_drift' in metrics:
            # Normalize to [0, 1] with good region around 0.5-1.0
            drift_score = 1.0 / (1.0 + metrics['mean_drift'])
            score += 0.2 * drift_score
        
        # Trajectory alignment component
        if 'trajectory_alignment' in metrics:
            score += 0.3 * max(0.0, metrics['trajectory_alignment'])
        
        # Hallucination component
        if 'hallucination_ratio' in metrics:
            halluc_score = 1.0 - metrics['hallucination_ratio']
            score += 0.2 * halluc_score
        
        return max(0.0, min(1.0, score))


def compute_eval_faithfulness(eval_result: Dict, 
                             original_result: Dict = None) -> Dict:
    """
    Compute faithfulness metrics for a single evaluation result.
    """
    metrics = FaithfulnessMetrics()
    faith_metrics = {}
    
    # Consistency across samples
    if 'predictions' in eval_result:
        faith_metrics['consistency'] = metrics.consistency_score(eval_result['predictions'])
    
    # Check for hallucinations
    if 'text' in eval_result and 'ground_truth' in eval_result:
        haluc = metrics.hallucination_detection(
            eval_result['text'],
            eval_result['ground_truth']
        )
        faith_metrics.update(haluc)
    
    # Compute aggregate faithfulness score
    faith_metrics['faithfulness_score'] = metrics.compute_faithfulness_score(faith_metrics)
    
    return faith_metrics


def aggregate_faithfulness(results: List[Dict]) -> Dict:
    """Aggregate faithfulness metrics across results."""
    import numpy as np
    
    agg = {}
    metrics_to_agg = [
        'consistency', 'hallucination_ratio', 'faithfulness_score'
    ]
    
    for metric in metrics_to_agg:
        values = [r.get(metric) for r in results if metric in r]
        if values:
            agg[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
    
    return agg
