import os
import json
from tqdm import tqdm
from llmlingua import PromptCompressor

from model_registry import get_config   # ← replaces all if/elif model_type blocks


def load_jsonl(file, encoding='utf-8'):
    data = []
    with open(file, 'r', encoding=encoding) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'a+', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def filter_correct_outputs(input_path, output_path):
    """Keep only examples where the model's answer was correct."""
    data = load_jsonl(input_path)
    correct = [d for d in data if d['accuracy']]
    print(f"Correct filter: {len(data)} → {len(correct)} ({len(correct)/len(data)*100:.1f}%)")
    save_jsonl(correct, output_path)


def filter_formatted_outputs(input_path, output_path, model_type):
    """
    Remove very long CoTs (>500 tokens) and, for llama3, extract the CoT
    portion before the final-answer line.
    All logic is now driven by model_registry — no if/elif needed here.
    """
    cfg  = get_config(model_type)
    data = load_jsonl(input_path)

    formatted = []
    for item in data:
        if item['cot_length'] > 500:
            continue
        # llama3 stores raw output that needs splitting; others already have cot
        if model_type == "llama3":
            parts = item["output"].split('\n\nThe final answer is:')
            if len(parts) == 2:
                item["cot"] = parts[0]
                formatted.append(item)
        else:
            # For qwen, gpt2, mistral, phi3 etc. the cot_field is ready as-is
            formatted.append(item)

    print(f"Format filter : {len(data)} → {len(formatted)}")
    save_jsonl(formatted, output_path)


def LLMLingua(data, compression_ratio=0.5, model_type="qwen",
              llmlingua_path="llmlingua-2-xlm-roberta-large-meetingbank"):
    """Compress CoT outputs with LLMLingua-2 at the given ratio."""
    cfg       = get_config(model_type)
    cot_field = cfg["cot_field"]          # e.g. "model_output" or "cot"

    llm_lingua = PromptCompressor(
        model_name=llmlingua_path,
        use_llmlingua2=True,
    )

    compressed = []
    for item in tqdm(data):
        cot_text = item[cot_field]

        # llama3 benefits from preserving step markers
        if model_type == "llama3":
            result = llm_lingua.compress_prompt(
                cot_text, rate=compression_ratio,
                force_tokens=['Step', ':'],
                force_reserve_digit=True,
                drop_consecutive=True,
            )
        else:
            result = llm_lingua.compress_prompt(cot_text, rate=compression_ratio)

        compressed.append({
            'question':              item['messages'][0]['content'],
            'input':                 item['prompt'],
            'output':                item['model_output'],
            'answer':                item['answer'],
            'model_answer':          item['prediction'],
            'is_correct':            item['accuracy'],
            'cot':                   cot_text,
            'compressed_cot':        result['compressed_prompt'],
            'original_cot_tokens':   result['origin_tokens'],
            'compressed_cot_tokens': result['compressed_tokens'],
            'compression_rate':      result['rate'],
        })
    return compressed


def get_average_compress_rate(data):
    rate = sum(d['compressed_cot_tokens'] / d['original_cot_tokens'] for d in data) / len(data)
    print(f"Average compression rate: {rate:.3f}")


def compress_cot_outputs(input_path, output_dir, model_type, llmlingua_path):
    """Run LLMLingua at every standard ratio and save results."""
    data       = load_jsonl(input_path)
    ratio_list = [0.9, 0.8, 0.7, 0.6, 0.5]
    for ratio in ratio_list:
        out_path       = os.path.join(output_dir, f"train_outputs_compressed_ratio_{ratio}.jsonl")
        compressed     = LLMLingua(data, compression_ratio=ratio,
                                   model_type=model_type, llmlingua_path=llmlingua_path)
        save_jsonl(compressed, out_path)
        get_average_compress_rate(compressed)


def data_processing(input_dir, model_type,
                    llmlingua_path="llmlingua-2-xlm-roberta-large-meetingbank"):
    """
    Full pipeline: correct → formatted → compressed.
    Works for any model_type registered in model_registry.
    """
    input_path     = os.path.join(input_dir, "Original/train/samples/predictions.jsonl")
    correct_path   = os.path.join(input_dir, "Original/train/samples/predictions_correct.jsonl")
    formatted_path = os.path.join(input_dir, "Original/train/samples/predictions_formatted.jsonl")
    compressed_dir = os.path.join(input_dir, "Compression")

    filter_correct_outputs(input_path=input_path, output_path=correct_path)
    filter_formatted_outputs(input_path=correct_path, output_path=formatted_path, model_type=model_type)
    compress_cot_outputs(input_path=formatted_path, output_dir=compressed_dir,
                         model_type=model_type, llmlingua_path=llmlingua_path)


# Convenience aliases kept for backward compatibility
def data_processing_gsm8k(input_dir="outputs/Qwen2.5-7B-Instruct/gsm8k/7b/",
                           model_type="qwen",
                           llmlingua_path="/your_model_path/llmlingua-2-xlm-roberta-large-meetingbank"):
    data_processing(input_dir=input_dir, model_type=model_type, llmlingua_path=llmlingua_path)


if __name__ == '__main__':
    data_processing_gsm8k()