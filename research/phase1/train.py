"""
phase1/train.py
---------------
Phase 1: TokenSkip Compressed CoT Fine-Tuning

This script fine-tunes a standard HuggingFace model (e.g., Phi-2, Qwen) 
to natively generate Compressed Chain-of-Thought (CCoT) via LoRA.

Method:
1. Loads the `llm_train.jsonl` dataset.
2. Compresses the ground truth text reasoning using TokenSkip (LLMLingua-2) at a target ratio.
3. Formats the data using the prompt templates from `utils/model_registry.py`.
4. Trains the model using standard Causal LM next-token prediction via `peft` LoRA.
5. Saves the LoRA adapter to `outputs/phase1_checkpoint/<model>`.

Usage:
    python research/phase1/train.py --model-type phi2 --model-path microsoft/phi-2 --ratio 0.8
"""
from __future__ import annotations
import argparse, json, pathlib, sys, os
from time import time
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

_RESEARCH_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RESEARCH_ROOT))

from utils.io import load_jsonl, load_config
from utils.answer import get_question, get_gt_answer
from utils.model_registry import get_config as get_model_cfg
from utils.tokenskip import batch_compress

_CFG = load_config()
_DS  = _CFG.get("dataset", {})
_P1  = _CFG.get("phase1", {})

DEFAULT_TRAIN = str(pathlib.Path(_DS.get("out_dir", "datasets/gsm8k_split")) / "llm_train.jsonl")
DEFAULT_OUT   = "outputs/phase1_checkpoint"
ALL_MODEL_TYPES = ["phi2","llama32_3b","qwen25_3b","qwen25_1_5b","qwen25_0_5b"]

def prepare_dataset(data: list[dict], tokenizer, model_type: str, ratio: float, device: torch.device):
    """Pre-compresses the CoT and formats the text strings for Causal LM training."""
    mcfg = get_model_cfg(model_type)
    
    # Extract questions, reasoning, and answers
    questions, reasons, answers = [], [], []
    for item in data:
        q = get_question(item)
        gt_raw = item.get("answer", "")
        # Split GSM8K original format (Reasoning #### Answer)
        if "####" in gt_raw:
            r_str, a_str = gt_raw.split("####")
            r_str = r_str.strip()
            a_str = a_str.strip()
        else:
            r_str = "Let's think."
            a_str = gt_raw.strip()
            
        questions.append(q)
        reasons.append(r_str)
        answers.append(a_str)
        
    print(f"[Phase 1] Compressing {len(reasons)} reasoning chains at ratio {ratio} via TokenSkip...")
    t0 = time()
    compressed = batch_compress(reasons, ratio, model_type, device=device)
    print(f"[Phase 1] Compression done in {time()-t0:.1f}s")
    
    texts = []
    for q, comp, ans in zip(questions, compressed, answers):
        comp_cot = comp["compressed_cot"]
        # Format: Question -> Compressed CoT -> Answer
        # We use build_prompt for the prefix, and append the CoT and Answer.
        prompt = mcfg["build_prompt"](q, tokenizer, ratio)
        split_token = mcfg["cot_split"]
        
        # Assemble the full text trajectory to train on
        full_text = f"{prompt}{comp_cot}{split_token} {ans}{tokenizer.eos_token}"
        texts.append(full_text)
        
    dataset = Dataset.from_dict({"text": texts})
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)
        
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset

def main():
    p = argparse.ArgumentParser(description="Phase 1: TokenSkip CCoT LoRA Fine-Tuning")
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-type", required=True, choices=ALL_MODEL_TYPES)
    p.add_argument("--train-data", default=DEFAULT_TRAIN)
    p.add_argument("--out-root",   default=DEFAULT_OUT)
    p.add_argument("--ratio",      type=float, default=0.8, help="Target compression ratio for training")
    p.add_argument("--epochs",     type=int, default=_P1.get("num_epochs", 3))
    p.add_argument("--batch-size", type=int, default=_P1.get("batch_size", 4))
    p.add_argument("--lr",         type=float, default=_P1.get("learning_rate", 2e-4))
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    import random, numpy as np
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = pathlib.Path(args.out_root) / args.model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Phase 1: TokenSkip Fine-Tuning | model={args.model_type}")
    print(f"  ratio={args.ratio} | epochs={args.epochs} | bs={args.batch_size}")
    print(f"{'='*62}")

    raw_data = load_jsonl(args.train_data)
    print(f"[Phase 1] Loaded {len(raw_data)} training examples from {args.train_data}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    train_dataset = prepare_dataset(raw_data, tokenizer, args.model_type, args.ratio, device)

    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device.type == "cuda" else torch.float32
    
    print(f"[Phase 1] Loading base model '{args.model_path}' ({dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True,
        device_map="auto" if device.type == "cuda" else None
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=_P1.get("lora_r", 128),
        lora_alpha=_P1.get("lora_alpha", 32),
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] if "qwen" in args.model_type or "llama" in args.model_type else ["Wqkv", "out_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        seed=args.seed,
        report_to="none"
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    print(f"\n[Phase 1] Starting training...")
    t0 = time()
    trainer.train()
    
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"\n[Phase 1] ✓ Training complete in {(time()-t0)/60:.1f} min")
    print(f"  LoRA adapter saved to {out_dir}/")

if __name__ == "__main__":
    main()
