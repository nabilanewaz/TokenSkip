import os
import sys
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from time import time
from copy import deepcopy
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from eval.utils import generate_completions
from data_processing.process_utils import *
from data_processing.answer_extraction import *
from eval.eval_script import *
from model_registry import get_config, all_model_types   # ← single import for all model logic


def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_data(path):
    if path.endswith("json"):
        data = json.load(open(path, "r"))
    elif path.endswith("jsonl"):
        data = []
        with open(path, "r") as file:
            for line in file:
                data.append(json.loads(line))
    else:
        raise NotImplementedError()
    return data


def build_prompts(args, test_data, tokenizer):
    """
    Build prompt strings for every example using the model-specific template
    from model_registry.  Returns the list of prompt strings and mutates
    each example dict with example['prompt'].
    """
    cfg = get_config(args.model_type)
    build_fn = cfg["build_prompt"]

    prompts = []
    for example in test_data:
        prompt = ""
        for mess in example['messages']:
            if mess['role'] == 'user':
                prompt += build_fn(mess['content'], tokenizer, args.compression_ratio)
            elif mess['role'] == 'assistant':
                prompt += mess['content'].rstrip()
        prompt = prompt.lstrip()
        example['prompt'] = prompt
        prompts.append(prompt)
    return prompts


def infer(args, test_data, answer_extraction_fn):
    cfg = get_config(args.model_type)

    # CODI has its own inference pipeline — direct the user to run_codi.py
    if cfg.get("_custom_runner"):
        sys.exit(
            f"\n[ERROR] model_type='{args.model_type}' uses a custom inference pipeline.\n"
            f"  Run it with:  python {cfg['_custom_runner']} --run-name codi_gpt2_gsm8k\n"
            f"  Then compare: python compare_metrics.py <baseline_dir> outputs/codi_gpt2_gsm8k/\n"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    # GPT-2 (and similar models) have no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = build_prompts(args, test_data, tokenizer)

    print("Loading model and tokenizer...")

    if args.use_vllm:
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
        except ImportError as e:
            raise RuntimeError(
                "vLLM is not installed. Run without --use_vllm for CPU-only."
            ) from e

        tp = (
            len([d for d in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if d.strip()])
            if os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
            else (torch.cuda.device_count() if torch.cuda.is_available() else 1)
        )

        if args.use_adapter:
            model = LLM(model=args.model_path, tokenizer=args.tokenizer_path,
                        trust_remote_code=True, enable_lora=True,
                        tensor_parallel_size=tp, max_model_len=16000)
        else:
            model = LLM(model=args.model_path, tokenizer=args.tokenizer_path,
                        trust_remote_code=True, tensor_parallel_size=tp)

        stop_words = cfg["stop_sequences"](tokenizer)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time()

        if args.use_adapter:
            outputs = model.generate(
                prompts,
                SamplingParams(temperature=args.temperature, top_p=1.0,
                               max_tokens=args.max_new_tokens, n=1, stop=stop_words),
                lora_request=LoRARequest("adapter", 1, args.adapter_path)
            )
        else:
            outputs = model.generate(
                prompts,
                SamplingParams(temperature=args.temperature, top_p=1.0,
                               max_tokens=args.max_new_tokens, n=1, stop=stop_words)
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time = time() - start_time
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        outputs = [o.outputs[0].text for o in outputs]

    else:
        # ── HuggingFace / CPU path ─────────────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map="cpu",
        )

        if args.use_adapter:
            model = PeftModel.from_pretrained(model, args.adapter_path, device_map="cpu")
            model = model.merge_and_unload()

        # Greedy decoding — silence HF warnings about unused sampling params.
        # Some models (e.g. GPT-2) have no saved generation_config.json and
        # return a bare default object; others (Qwen) embed sampling defaults
        # that conflict with do_sample=False.  We patch whichever exists.
        try:
            gc = model.generation_config
            gc.do_sample    = False
            gc.temperature  = None   # must be None, not 1.0, when do_sample=False
            gc.top_k        = None
            gc.top_p        = None
        except Exception:
            pass  # model has no generation_config at all — safe to ignore

        tokenizer.padding_side = "left"

        # Build stop token ID sequences from all stop strings in the registry.
        # EOS is always included; model-specific strings (e.g. "\nQuestion:") are
        # encoded and appended so KeyWordsCriteria can catch them.
        stop_strings = cfg["stop_sequences"](tokenizer)
        stop_id_sequences = []
        for s in stop_strings:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if ids:
                stop_id_sequences.append(ids)
        # Always include bare EOS as a fallback
        if tokenizer.eos_token_id is not None:
            eos_seq = [tokenizer.eos_token_id]
            if eos_seq not in stop_id_sequences:
                stop_id_sequences.append(eos_seq)

        do_sample = args.temperature != 0.0

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time()
        outputs, _ = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature,
            top_p=1.0,
            batch_size=args.eval_batch_size,
            stop_id_sequences=stop_id_sequences or None,
            end_of_generation_id_sequence=(
                [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None
            ),
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time = time() - start_time

    # ── Measure CoT lengths ────────────────────────────────────────────────
    cot_split = cfg["cot_split"]
    cot_lengths = []
    for completion in outputs:
        cot = completion.split(cot_split)[0]
        cot_lengths.append(tokenizer(cot, return_tensors="pt")['input_ids'].shape[1])

    # ── Extract answers ────────────────────────────────────────────────────
    predictions = [
        eval(answer_extraction_fn)(item['messages'][-2]['content'], output, task='cot')
        for item, output in tqdm(zip(test_data, outputs), desc="Extracting answers", total=len(outputs))
    ]
    assert len(outputs) > 0

    results = []
    for example, output, pred, cot_len in zip(test_data, outputs, predictions, cot_lengths):
        item = deepcopy(example)
        item.update({'model_output': output, 'prediction': pred, 'cot_length': cot_len})
        results.append(item)

    return results, total_time


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",        type=str,   default="outputs/")
    parser.add_argument("--model-path",        type=str,   required=True)
    parser.add_argument("--tokenizer-path",    type=str,   default=None)
    parser.add_argument("--adapter-path",      type=str,   default=None)
    parser.add_argument("--model-size",        type=str,
                        choices=['0.5b','1b','1.5b','3b','7b','8b','13b','14b','33b','34b','70b','117m','345m','762m','1.5b'],
                        default="7b")
    parser.add_argument("--model-type",        type=str,
                        choices=all_model_types(),          # ← auto-populated from registry
                        required=True)
    parser.add_argument("--use_adapter",       action='store_true', default=False)
    parser.add_argument("--compression_ratio", type=float,  default=1.0)
    parser.add_argument("--benchmark",         type=str,   choices=['gsm8k', 'math'], default="gsm8k")
    parser.add_argument("--data-type",         type=str,   choices=['train', 'test'], default="test")
    parser.add_argument("--use_vllm",          action='store_true', default=False)
    parser.add_argument("--max_num_examples",  type=int,   default=10**15)
    parser.add_argument("--max_new_tokens",    type=int,   default=512)
    parser.add_argument("--eval_batch_size",   type=int,   default=2,
                        help="Keep at 1-4 for CPU. 16+ only safe with a GPU.")
    parser.add_argument("--temperature",       type=float,  default=0.0)
    parser.add_argument("--debug_samples",    type=int,   default=0,
                        help="Print this many raw model outputs after inference. Use 3-5 to diagnose prompt/extraction issues.")
    parser.add_argument("--seed",              type=int,   default=42)
    args, _ = parser.parse_known_args()

    # Default tokenizer path to model path if not given
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    # Enforce shorter budget for compressed no-LoRA runs
    if args.compression_ratio < 1.0 and not args.use_adapter:
        args.max_new_tokens = max(32, int(args.max_new_tokens * args.compression_ratio))

    if args.benchmark == 'math' and args.use_adapter:
        args.max_new_tokens = int(args.max_new_tokens * args.compression_ratio)

    print(f"\nModel      : {args.model_path}")
    print(f"Model type : {args.model_type}")
    print(f"Benchmark  : {args.benchmark} / {args.data_type}")
    print(f"Max tokens : {args.max_new_tokens}  |  batch: {args.eval_batch_size}  |  T={args.temperature}")
    if args.use_adapter:
        print(f"Adapter    : {args.adapter_path}  ratio={args.compression_ratio}")
    print()

    if args.use_adapter:
        args.output_dir = os.path.join(
            args.output_dir, f"{args.model_size}/TokenSkip/{args.compression_ratio}/"
        )
    else:
        args.output_dir = os.path.join(
            args.output_dir, f"{args.model_size}/Original/{args.data_type}/"
        )

    test_conf = read_data(f"configs/{args.benchmark}_{args.data_type}.json")

    for src, info in test_conf.items():
        fname      = os.path.join(args.output_dir, "test_data", "test.jsonl")
        input_dir  = os.path.dirname(fname)
        output_dir = os.path.join(args.output_dir, "samples")
        os.makedirs(input_dir,  exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        metric_path = os.path.join(output_dir, "metrics.json")
        if os.path.exists(metric_path) and read_data(metric_path).get('n_samples', 0) > 0:
            print(f"Skipping {src} — results already exist at {metric_path}")
            continue

        # Write processed test data
        with open(fname, "w") as f:
            for i, sample in enumerate(tqdm(read_data(info['test_path']), desc=f'Processing {src}')):
                fn = eval(info['process_fn'])
                sample['id'] = sample.get('id', f"{src}-{i}")
                for j, item in enumerate(fn(sample)):
                    item['dataset'] = src
                    item['id'] = f"{src}-test-{i}-{j}"
                    assert 'answer' in item
                    print(json.dumps(item), file=f, flush=True)

        set_random_seed(args.seed)

        # Load and prepare test data
        test_data = []
        with open(os.path.join(input_dir, "test.jsonl")) as fin:
            for line in fin:
                example = json.loads(line)
                messages = example['messages']
                assert messages[-1]['role'] == 'assistant'
                example['reference'] = example.get('reference', '') or [
                    m['content'] for m in messages if m['role'] == 'assistant'
                ]
                for m in messages:
                    if m['role'] == 'assistant':
                        m['content'] = ''
                example['messages'] = messages
                test_data.append(example)

        if args.max_num_examples and len(test_data) > args.max_num_examples:
            test_data = random.sample(test_data, args.max_num_examples)

        results, total_time = infer(args, test_data, info['answer_extraction_fn'])

        # ── Debug: print raw outputs so you can see what the model generates ──
        if args.debug_samples > 0:
            print(f"\n{'='*70}")
            print(f"DEBUG: first {args.debug_samples} raw outputs")
            print('='*70)
            for r in results[:args.debug_samples]:
                print(f"\n--- Question ---\n{r['messages'][0]['content'][:200]}")
                print(f"\n--- Raw output ---\n{r['model_output']}")
                print(f"\n--- Extracted prediction: {r['prediction']!r}  |  Answer: {r['answer']!r} ---")
                print('-'*70)
            print()

        os.environ['TOKENIZERS_PARALLELISM'] = "false"

        # Score
        labels = []
        invalid = []
        for item in results:
            if len(item['prediction']) == 0:
                invalid.append(item)
                labels.append(False)
            else:
                labels.append(eval_math(item))

        for item, label in zip(results, labels):
            item['accuracy'] = label

        acc = sum(l for l in labels) / len(labels) if labels else 0.0
        avg_cot = sum(r['cot_length'] for r in results) / len(results) if results else 0.0

        print(f"Accuracy        : {acc*100:.2f}%")
        print(f"Avg CoT length  : {avg_cot:.1f} tokens")
        print(f"Invalid outputs : {len(invalid)}")

        # Save predictions
        pred_path = os.path.join(output_dir, "predictions.jsonl")
        if os.path.exists(pred_path):
            os.remove(pred_path)
        with open(pred_path, 'a', encoding='utf-8') as fout:
            for item in results:
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')

        # Save metrics
        with open(metric_path, "w") as fout:
            json.dump({
                "n_samples":      len(results),
                "accuracy":       acc,
                "avg_cot_length": avg_cot,
                "sample_latency": (total_time / len(test_data) if test_data else None),
            }, fout, indent=4)