import os
import json
import random
import numpy as np

from model_registry import get_config   # ← drives cot_field per model


def load_jsonl(file, encoding='utf-8'):
    data = []
    with open(file, 'r', encoding=encoding) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def load_all_data(input_dir):
    original  = load_jsonl(os.path.join(input_dir, "Original/train/samples/predictions_formatted.jsonl"))
    ratios    = [0.9, 0.8, 0.7, 0.6, 0.5]
    compressed = [
        load_jsonl(os.path.join(input_dir, f"Compression/train_outputs_compressed_ratio_{r}.jsonl"))
        for r in ratios
    ]
    return original, compressed, [1.0] + ratios


def get_llamafactory_input(input_dir, output_path, model_type):
    """
    Merge original + compressed CoT data into a LLaMA-Factory JSON file.
    The compression ratio is embedded in the instruction field so the model
    can condition on it at inference time.

    Works for any model_type in model_registry.
    """
    cfg       = get_config(model_type)
    cot_field = cfg["cot_field"]   # "model_output" or "cot" depending on model

    original, compressed_list, ratio_list = load_all_data(input_dir)
    all_pools = [original] + compressed_list

    datalines = []
    for i in range(len(original)):
        pool_idx = random.choice(range(len(all_pools)))

        if pool_idx == 0:
            # Uncompressed — use raw model output
            item         = original[i]
            question     = item['messages'][0]['content']
            answer       = item['prediction']
            cot          = item[cot_field]
            instruction  = "Please reason step by step, and put your final answer within \\boxed{}."
            input_field  = question
        else:
            # Compressed — embed ratio in input so model learns to hit the target
            ratio        = ratio_list[pool_idx]
            item         = compressed_list[pool_idx - 1][i]
            question     = item['question']
            answer       = item['model_answer']
            cot          = item['compressed_cot']
            instruction  = "Please reason step by step, and put your final answer within \\boxed{}."
            input_field  = f"{question}<|eot_id|>{ratio}<|eot_id|>"

        output_text = f"{cot}\n\nThe final answer is: $\\boxed{{{answer}}}$"

        datalines.append({
            "instruction": instruction,
            "input":       input_field,
            "output":      output_text,
        })

    random.shuffle(datalines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_json(datalines, output_path)
    print(f"Wrote {len(datalines)} examples → {output_path}")


if __name__ == '__main__':
    seed_everything(42)

    # ── Edit these three lines to switch models ──────────────────────────────
    MODEL_TYPE = "qwen"
    INPUT_DIR  = "outputs/Qwen2.5-7B-Instruct/gsm8k/7b/"
    OUTPUT     = "outputs/mydataset_compressed_gsm8k_llmlingua2_qwen_7B.json"
    # ─────────────────────────────────────────────────────────────────────────

    get_llamafactory_input(
        input_dir=INPUT_DIR,
        output_path=OUTPUT,
        model_type=MODEL_TYPE,
    )