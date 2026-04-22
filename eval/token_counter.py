"""
Token counting and compression analysis utilities.
Tracks original vs compressed CoT token counts and compression ratios.
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer


class TokenCounter:
    """Utility for counting and tracking tokens in CoT outputs."""
    
    def __init__(self, tokenizer_path: str = "gpt2"):
        """Initialize tokenizer for token counting."""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def extract_cot(self, text: str, start_marker: str = "[RAT_START]", 
                   end_marker: str = "[RAT_END]") -> Optional[str]:
        """Extract CoT portion from full response."""
        pattern = re.escape(start_marker) + r"(.*?)" + re.escape(end_marker)
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1) if match else None
    
    def analyze_response(self, full_response: str, original_cot: Optional[str] = None,
                        compressed_cot: Optional[str] = None) -> Dict:
        """
        Analyze token counts in a response.
        
        Returns:
            Dict with keys:
            - total_tokens: tokens in full response
            - cot_tokens: tokens in extracted CoT
            - answer_tokens: tokens after CoT
            - original_cot_tokens: if original_cot provided
            - compressed_cot_tokens: if compressed_cot provided
            - compression_ratio: compressed / original
        """
        result = {
            'total_tokens': self.count_tokens(full_response),
        }
        
        cot = self.extract_cot(full_response)
        if cot:
            result['cot_tokens'] = self.count_tokens(cot)
            # Rough estimate: answer is after CoT marker
            answer_part = full_response.split("[RAT_END]")[-1] if "[RAT_END]" in full_response else ""
            result['answer_tokens'] = self.count_tokens(answer_part)
        
        if original_cot:
            result['original_cot_tokens'] = self.count_tokens(original_cot)
        
        if compressed_cot:
            result['compressed_cot_tokens'] = self.count_tokens(compressed_cot)
        
        if original_cot and compressed_cot:
            orig_count = self.count_tokens(original_cot)
            comp_count = self.count_tokens(compressed_cot)
            if orig_count > 0:
                result['compression_ratio'] = comp_count / orig_count
                result['compression_percentage'] = round((1 - result['compression_ratio']) * 100, 2)
        
        return result


def aggregate_token_stats(results: List[Dict]) -> Dict:
    """
    Aggregate token statistics across multiple results.
    
    Args:
        results: List of result dicts with token counts
    
    Returns:
        Dict with mean, median, min, max for each token type
    """
    import numpy as np
    
    stats = {}
    token_fields = [
        'total_tokens', 'cot_tokens', 'answer_tokens',
        'original_cot_tokens', 'compressed_cot_tokens'
    ]
    
    for field in token_fields:
        values = [r.get(field) for r in results if field in r]
        if values:
            stats[field] = {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'min': int(np.min(values)),
                'max': int(np.max(values)),
                'std': float(np.std(values)),
            }
    
    # Compression ratio stats
    comp_ratios = [r.get('compression_ratio') for r in results 
                   if 'compression_ratio' in r]
    if comp_ratios:
        stats['compression_ratio'] = {
            'mean': float(np.mean(comp_ratios)),
            'median': float(np.median(comp_ratios)),
            'min': float(np.min(comp_ratios)),
            'max': float(np.max(comp_ratios)),
        }
    
    return stats


def log_token_analysis(output_path: str, results: List[Dict]):
    """Save token analysis to JSON file."""
    stats = aggregate_token_stats(results)
    
    with open(output_path, 'w') as f:
        json.dump({
            'num_examples': len(results),
            'statistics': stats,
            'detailed_results': results
        }, f, indent=2)
    
    print(f"✓ Token analysis saved to {output_path}")
