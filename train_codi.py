"""
train_codi.py
-------------
Fine-tunes CODI-GPT2 on a custom dataset (llm_train.jsonl from split_dataset.py).

What this does
--------------
1. Ensures CODI repo is cloned and checkpoint is downloaded (reuses run_codi.py logic)
2. Converts llm_train.jsonl → CODI's expected training format
3. Inspects CODI's train.py to understand its CLI interface
4. Runs CODI training on llm_train.jsonl (4,220 examples)
5. Saves the fine-tuned checkpoint for use in run_codi.py

Usage
-----
    # Train on the 48% llm_train split
    python train_codi.py

    # Custom paths
    python train_codi.py --train-data datasets/gsm8k_split/llm_train.jsonl
                         --val-data   datasets/gsm8k_split/validation.jsonl
                         --output-dir outputs/codi_finetuned

    # After training, evaluate the fine-tuned model
    python run_codi.py --run-name codi_finetuned_eval
                       --ckpt-dir outputs/codi_finetuned
"""

import os
import sys
import json
import re
import subprocess
import argparse
import shutil
import pathlib
from time import time


# ── Reuse constants from run_codi.py ──────────────────────────────────────────
CODI_REPO  = "https://github.com/zhenyi4/CODI.git"
CODI_HF_ID = "zen-E/CODI-gpt2"

DEFAULT_TRAIN_DATA = "datasets/gsm8k_split/llm_train.jsonl"
DEFAULT_VAL_DATA   = "datasets/gsm8k_split/validation.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/codi_finetuned"
DEFAULT_WORK_DIR   = "codi_workspace"


# ── Dependency + repo helpers (mirrors run_codi.py) ───────────────────────────

def ensure_dependencies():
    packages = [
        "peft==0.15.2", "datasets==3.6.0", "huggingface_hub",
        "transformers==4.52.4", "accelerate==1.7.0",
    ]
    print("[train_codi] Installing/verifying CODI dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--force-reinstall", "--no-deps"] + packages, check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet"] + packages, check=True)
    print("[train_codi] ✓ Dependencies ready.")


def clone_codi(work_dir: pathlib.Path) -> pathlib.Path:
    codi_dir = work_dir / "CODI"
    if not codi_dir.exists():
        print("[train_codi] Cloning CODI repo...")
        subprocess.run(["git", "clone", CODI_REPO, str(codi_dir)], check=True)
    else:
        print("[train_codi] CODI repo already present.")
    return codi_dir


def get_checkpoint(work_dir: pathlib.Path) -> pathlib.Path:
    from huggingface_hub import snapshot_download

    def weights_present(p):
        return (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()

    ckpt_file = work_dir / "ckpt_dir.txt"
    if ckpt_file.exists():
        ckpt_dir = pathlib.Path(ckpt_file.read_text().strip())
        if weights_present(ckpt_dir):
            print(f"[train_codi] Using cached checkpoint: {ckpt_dir}")
            return ckpt_dir

    print(f"[train_codi] Downloading {CODI_HF_ID}...")
    ckpt_dir = pathlib.Path(snapshot_download(
        repo_id=CODI_HF_ID, force_download=True,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    ))
    ckpt_file.write_text(str(ckpt_dir), encoding="utf-8")
    print(f"[train_codi] ✓ Checkpoint: {ckpt_dir}")
    return ckpt_dir


# ── CODI train.py inspection ──────────────────────────────────────────────────

def inspect_train_script(codi_dir: pathlib.Path) -> dict:
    """
    Read CODI/train.py and extract:
    - Which arguments it accepts
    - Whether it takes --data_path or --data_name
    - Whether it has --output_dir or --save_dir
    Returns a dict of findings.
    """
    train_py = codi_dir / "train.py"
    if not train_py.exists():
        return {"found": False}

    src = train_py.read_text(encoding="utf-8", errors="replace")

    findings = {"found": True, "args": set(), "src_snippet": src[:500]}

    # Extract all argparse arguments
    for m in re.finditer(r'add_argument\s*\(\s*["\']--([^"\']+)["\']', src):
        findings["args"].add(m.group(1))

    # Key flags we care about
    findings["has_data_path"]     = "data_path"     in findings["args"]
    findings["has_data_name"]     = "data_name"      in findings["args"]
    findings["has_output_dir"]    = "output_dir"     in findings["args"]
    findings["has_save_dir"]      = "save_dir"       in findings["args"]
    findings["has_train_file"]    = "train_file"     in findings["args"]
    findings["has_num_epochs"]    = "num_train_epochs" in findings["args"] or \
                                    "num_epochs"     in findings["args"]
    findings["has_learning_rate"] = "learning_rate"  in findings["args"]

    print(f"[train_codi] train.py args found: {sorted(findings['args'])}")
    return findings


# ── Data format conversion ────────────────────────────────────────────────────

def load_jsonl(path: pathlib.Path) -> list:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_to_codi_format(src_path: pathlib.Path, dst_path: pathlib.Path):
    """
    Convert split_dataset.py output to CODI's expected training format.

    CODI's train.py expects the same format as GSM8K:
        {"question": "...", "cot": "...", "answer": "<number>"}

    split_dataset.py preserves the original GSM8K fields.
    """
    raw = load_jsonl(src_path)
    converted = []
    skipped = 0

    for item in raw:
        # GSM8K format with question, cot, and answer
        if "question" in item and "answer" in item:
            entry = {
                "question": item["question"],
                "answer":   item["answer"],
            }
            # Include cot if present (required for CODI training)
            if "cot" in item:
                entry["cot"] = item["cot"]
            converted.append(entry)
        # TokenSkip internal format (has messages list)
        elif "messages" in item:
            q = next((m["content"] for m in item["messages"] if m["role"] == "user"), None)
            a = item.get("answer", "")
            cot = item.get("cot", "")
            if q and a:
                entry = {"question": q, "answer": a}
                if cot:
                    entry["cot"] = cot
                converted.append(entry)
            else:
                skipped += 1
        else:
            skipped += 1

    # Debug: check first item
    has_cot = "cot" in converted[0] if converted else False
    print(f"[train_codi] Converted {len(converted)} examples "
          f"({skipped} skipped) from {src_path.name}. First item has cot: {has_cot}")
    save_jsonl(converted, dst_path)
    return converted


# ── CUDA patch (same as run_codi.py) ─────────────────────────────────────────

def patch_train_py(codi_dir: pathlib.Path) -> pathlib.Path:
    """Patch train.py → train_fixed.py: replace hardcoded .to('cuda') and add custom dataset loading."""
    src_path = codi_dir / "train.py"
    dst_path = codi_dir / "train_fixed.py"

    if not src_path.exists():
        print("[train_codi] ⚠ train.py not found in CODI repo.")
        return None

    code = src_path.read_text(encoding="utf-8", errors="replace")

    # Insert DEVICE constant after imports
    if "DEVICE = " not in code:
        lines      = code.split('\n')
        import_end = 0
        paren_depth = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == '' or stripped.startswith('#'):
                continue
            if paren_depth > 0:
                paren_depth += stripped.count('(') - stripped.count(')')
                if paren_depth == 0:
                    import_end = i + 1
                continue
            if stripped.startswith('import ') or stripped.startswith('from '):
                if '(' in stripped and ')' not in stripped:
                    paren_depth = 1
                else:
                    import_end = i + 1
                continue
            break

        if import_end > 0:
            pos = len('\n'.join(lines[:import_end]))
            device_block = (
                "\n# --- CPU/GPU device patch ---\n"
                "import torch as _torch\n"
                "DEVICE = 'cuda' if _torch.cuda.is_available() else 'cpu'\n"
                "# ----------------------------\n\n"
            )
            code = code[:pos] + '\n' + device_block + code[pos:]

    # Replace CUDA references
    code = re.sub(r"\.to\(['\"]cuda['\"]\)", ".to(DEVICE)", code)
    code = re.sub(r"\.cuda\(\)",             ".to(DEVICE)", code)
    code = re.sub(r"device=['\"]cuda['\"]",  "device=DEVICE", code)
    code = re.sub(r'^(\s*)device\s*=\s*["\']cuda["\']',
                  r'\1device = DEVICE', code, flags=re.M)
    
    # Fix dtype issue on CPU: use float32 instead of float16
    # Replace torch_dtype conditional to use float32 on CPU
    if 'CPU_DTYPE_FIX' not in code:
        dtype_fix = '''
# CPU dtype fix
import torch as _torch_dtype
if not _torch_dtype.cuda.is_available():
    _USE_FP32_ON_CPU = True
else:
    _USE_FP32_ON_CPU = False
CPU_DTYPE_FIX = True
'''
        # Insert after the DEVICE block
        code = code.replace('# ----------------------------\n\n', 
                           '# ----------------------------\n' + dtype_fix + '\n')
        
        # Replace torch_dtype assignments
        code = re.sub(
            r'torch_dtype=\(\s*torch\.float16 if training_args\.bf16 is False else torch\.bfloat16\s*\)',
            'torch_dtype=(torch.float32 if _USE_FP32_ON_CPU else (torch.float16 if training_args.bf16 is False else torch.bfloat16))',
            code
        )

    # Add custom dataset loading support before the else clause in make_supervised_data_module
    if 'custom_local:' not in code:
        # Find and replace the final else clause with our custom loader + the else clause
        old_else = '        else:\n            raise NotImplementedError(f"Dataset {data_args.data_name} is not supported.")'
        new_code = '''        elif data_args.data_name.startswith("custom_local:"):
            # Custom local dataset loading added by train_codi.py
            import pathlib
            data_path = pathlib.Path(data_args.data_name.replace("custom_local:", ""))
            dataset = load_dataset("json", data_files=str(data_path), split="train")
            # Use "icot" format (GSM8K-Aug) instead of "icot_full" to avoid empty CoT issues
            train_dataset = SupervisedDataset(data_name="icot", raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        else:
            raise NotImplementedError(f"Dataset {data_args.data_name} is not supported.")'''
        code = code.replace(old_else, new_code)
    
    # Fix dtype mismatch on CPU by converting model to float32 after CODI initialization
    if 'CPU_DTYPE_CONVERSION' not in code:
        # Find where CODI model is instantiated and add dtype conversion
        old_model_init = '    model = CODI(model_args, training_args, lora_config)'
        new_model_init = '''    model = CODI(model_args, training_args, lora_config)
    # CPU dtype conversion to avoid Half/Float mismatch - CPU_DTYPE_CONVERSION
    if not torch.cuda.is_available():
        model = model.float()  # Convert all parameters to float32 on CPU'''
        if old_model_init in code:
            code = code.replace(old_model_init, new_model_init)

    dst_path.write_text(code, encoding="utf-8")
    print(f"[train_codi] ✓ Patched train.py → train_fixed.py")
    return dst_path


# ── Training command builder ──────────────────────────────────────────────────

def build_train_cmd(findings: dict, args, ckpt_dir: pathlib.Path,
                    train_data_codi: pathlib.Path,
                    val_data_codi: pathlib.Path,
                    output_dir: pathlib.Path) -> list:
    """
    Build the training command based on what train.py actually accepts.
    Falls back gracefully for every optional argument.
    """
    script = "train_fixed.py"

    cmd = [
        sys.executable, script,
        "--model_name_or_path", "gpt2",
        "--seed",               str(args.seed),
        "--model_max_length",   str(args.model_max_length),
        "--lora_r",             str(args.lora_r),
        "--lora_alpha",         str(args.lora_alpha),
        "--lora_init",
        "--num_latent",         str(args.num_latent),
        "--use_prj",            "True",
        "--prj_dim",            str(args.prj_dim),
        "--prj_no_ln",          "False",
        "--prj_dropout",        "0.0",
        "--inf_latent_iterations", str(args.inf_latent_iterations),
        "--remove_eos",         "True",
        "--use_lora",           "True",
        "--ckpt_dir",           str(ckpt_dir),
        "--batch_size",         str(args.batch_size),
    ]

    # Data path — use --data_name which CODI expects (dataclass field in DataArguments)
    # We pass a special marker that our patched code will recognize
    cmd += ["--data_name", f"custom_local:{train_data_codi}"]

    # Output directory — always use --output_dir (standard Transformers argument)
    cmd += ["--output_dir", str(output_dir)]

    # Training hyperparameters — these are standard Transformers arguments
    cmd += ["--num_train_epochs", str(args.num_epochs)]
    cmd += ["--learning_rate", str(args.learning_rate)]

    if args.bf16:
        cmd.append("--bf16")

    return cmd


# ── Streaming subprocess ───────────────────────────────────────────────────────

def run_streaming(cmd: list, cwd: str, log_path: pathlib.Path) -> tuple[int, str]:
    """Run cmd, stream to stdout, save to log_path. Returns (returncode, log)."""
    print(f"\n[train_codi] $ {' '.join(cmd)}\n")
    lines = []
    process = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True,
        encoding="utf-8", errors="replace",
    )
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    process.wait()
    log = "".join(lines)
    log_path.write_text(log, encoding="utf-8")
    return process.returncode, log


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train CODI-GPT2 on llm_train.jsonl from split_dataset.py"
    )
    parser.add_argument("--train-data",  default=DEFAULT_TRAIN_DATA,
                        help=f"Training split (default: {DEFAULT_TRAIN_DATA})")
    parser.add_argument("--val-data",    default=DEFAULT_VAL_DATA,
                        help=f"Validation split (default: {DEFAULT_VAL_DATA})")
    parser.add_argument("--output-dir",  default=DEFAULT_OUTPUT_DIR,
                        help=f"Where to save the fine-tuned checkpoint")
    parser.add_argument("--work-dir",    default=DEFAULT_WORK_DIR)

    # Match CODI's training hyperparameters
    parser.add_argument("--seed",                  type=int,   default=11)
    parser.add_argument("--model_max_length",      type=int,   default=512)
    parser.add_argument("--lora_r",                type=int,   default=128)
    parser.add_argument("--lora_alpha",            type=int,   default=32)
    parser.add_argument("--num_latent",            type=int,   default=6)
    parser.add_argument("--prj_dim",               type=int,   default=768)
    parser.add_argument("--inf_latent_iterations", type=int,   default=6)
    parser.add_argument("--batch_size",            type=int,   default=2,
                        help="Keep low on CPU (default 2). Use 4-8 with GPU.")
    parser.add_argument("--num_epochs",            type=int,   default=3)
    parser.add_argument("--learning_rate",         type=float, default=2e-4)
    parser.add_argument("--bf16",                  action="store_true", default=False)
    args = parser.parse_args()

    work_dir    = pathlib.Path(args.work_dir).resolve()
    train_data  = pathlib.Path(args.train_data)
    val_data    = pathlib.Path(args.val_data)
    output_dir  = pathlib.Path(args.output_dir).resolve()

    print("=" * 62)
    print("  CODI-GPT2 Training")
    print(f"  train data  : {train_data}  ({sum(1 for _ in open(train_data, encoding='utf-8'))} examples)")
    print(f"  val data    : {val_data}")
    print(f"  output dir  : {output_dir}")
    print(f"  epochs      : {args.num_epochs}  |  lr: {args.learning_rate}")
    print(f"  bf16        : {args.bf16}")
    print("=" * 62)

    # Validate inputs
    if not train_data.exists():
        sys.exit(
            f"\n[train_codi] ✗ Training data not found: {train_data}\n"
            f"  Run 'python split_dataset.py' first to generate the splits.\n"
        )

    # 1) Dependencies + repo
    ensure_dependencies()
    codi_dir = clone_codi(work_dir)
    ckpt_dir = get_checkpoint(work_dir)

    # 2) Inspect train.py
    findings = inspect_train_script(codi_dir)
    if not findings["found"]:
        sys.exit(
            "\n[train_codi] ✗ CODI/train.py not found in the repo.\n"
            "  The CODI repo may not include a training script publicly.\n"
            "  Check https://github.com/zhenyi4/CODI for a train.py.\n"
            "  If it's in a different branch: cd codi_workspace/CODI && git branch -a\n"
        )

    # 3) Patch train.py for CPU compatibility
    patched = patch_train_py(codi_dir)
    if patched is None:
        sys.exit("[train_codi] Cannot continue without train.py.")

    # 4) Convert data to CODI format and place in CODI's datasets folder
    codi_data_dir = codi_dir / "datasets" / "gsm8k_custom"
    codi_data_dir.mkdir(parents=True, exist_ok=True)

    train_codi_path = codi_data_dir / "train.jsonl"
    val_codi_path   = codi_data_dir / "val.jsonl"

    convert_to_codi_format(train_data, train_codi_path)
    if val_data.exists():
        convert_to_codi_format(val_data, val_codi_path)
    else:
        print(f"[train_codi] ⚠ Val data not found at {val_data}, skipping.")
        val_codi_path = None

    # Also copy into the standard gsm8k location so --data_name gsm8k still works
    standard_gsm8k = codi_dir / "datasets" / "gsm8k"
    standard_gsm8k.mkdir(parents=True, exist_ok=True)
    shutil.copy(train_codi_path, standard_gsm8k / "train.jsonl")
    print(f"[train_codi] Copied train split → {standard_gsm8k / 'train.jsonl'}")

    # 5) Build command
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_train_cmd(
        findings, args, ckpt_dir,
        train_codi_path,
        val_codi_path or train_codi_path,
        output_dir,
    )

    # 6) Run training
    log_path = output_dir / "train_log.txt"
    print(f"\n[train_codi] Starting training — log → {log_path}\n")
    t0 = time()
    rc, log = run_streaming(cmd, cwd=str(codi_dir), log_path=log_path)
    elapsed = time() - t0

    print(f"\n[train_codi] Training finished in {elapsed/3600:.2f}h  (exit code {rc})")

    if rc != 0:
        print(
            f"\n[train_codi] ✗ Training failed. Common causes:\n"
            f"  - train.py has different argument names than expected\n"
            f"    → check {log_path} for 'unrecognized arguments'\n"
            f"    → then re-run with adjusted flags\n"
            f"  - OOM on CPU: reduce --batch_size to 1\n"
        )
        # Print last 30 lines of log to help diagnose
        last_lines = log.strip().split('\n')[-30:]
        print('\n'.join(last_lines))
        sys.exit(1)

    # 7) Verify checkpoint was saved
    saved_weights = list(output_dir.glob("*.bin")) + list(output_dir.glob("*.safetensors"))
    if saved_weights:
        print(f"\n[train_codi] ✓ Checkpoint saved:")
        for f in saved_weights:
            print(f"    {f}")
    else:
        print(f"\n[train_codi] ⚠ No weights found in {output_dir} — check train_log.txt")

    print(f"\n[train_codi] Next: evaluate the fine-tuned model with:")
    print(f"  python run_codi.py --run-name codi_finetuned_eval --work-dir {work_dir}")
    print(f"  (then edit run_codi.py's ckpt_dir to point to {output_dir})\n")


if __name__ == "__main__":
    main()