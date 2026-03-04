"""
phase1_train.py  —  Phase 1: Base Model Training
==================================================
Trains CODI-GPT2 on llm_train.jsonl (the base training split) using the
curriculum defined in the research protocol:

    Stage A: Standard text CoT (discrete tokens) — grounds reasoning
    Stage B: Continuous latent CoT (Coconut/CODI style) — end-to-end

Architecture
------------
  Input: text tokens x
  Reasoning: k continuous hidden states h1…hk ∈ R^d (not decoded)
  Output: final answer tokens y
  Loss: Cross-Entropy on final answer only + distillation loss

Usage
-----
    # Default — trains on datasets/gsm8k_split/llm_train.jsonl
    python phase1_train.py

    # GPU (Colab T4 / A100) — much faster
    python phase1_train.py --bf16 --batch_size 4

    # Custom paths
    python phase1_train.py \\
        --train-data datasets/gsm8k_split/llm_train.jsonl \\
        --output-dir outputs/phase1_checkpoint \\
        --num_epochs 3

Outputs
-------
    outputs/phase1_checkpoint/     fine-tuned CODI checkpoint
    outputs/phase1_checkpoint/train_log.txt
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

# ── Configuration ──────────────────────────────────────────────────────────────
CODI_HF_ID         = "zen-E/CODI-gpt2"
DEFAULT_TRAIN_DATA = "datasets/gsm8k_split/llm_train.jsonl"
DEFAULT_VAL_DATA   = "datasets/gsm8k_split/validation.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/phase1_checkpoint"
DEFAULT_WORK_DIR   = "codi_workspace"

# CODI hyperparameters (match the published paper config)
LORA_R        = 128
LORA_ALPHA    = 32
NUM_LATENT    = 6
PRJ_DIM       = 768
INF_LATENT    = 6
MODEL_MAX_LEN = 512


# ── Dependency management ──────────────────────────────────────────────────────

def ensure_dependencies():
    """Pin CODI-compatible package versions."""
    packages = [
        "peft==0.15.2",
        "datasets==3.6.0",
        "huggingface_hub",
        "transformers==4.52.4",
        "accelerate==1.7.0",
        "safetensors",
    ]
    print("[Phase 1] Checking dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--force-reinstall", "--no-deps"] + packages,
        check=True
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet"] + packages,
        check=True
    )
    print("[Phase 1] ✓ Dependencies ready.")


def get_checkpoint(work_dir: pathlib.Path) -> pathlib.Path:
    """Download or retrieve cached zen-E/CODI-gpt2 pretrained weights."""
    from huggingface_hub import snapshot_download

    def weights_present(p: pathlib.Path) -> bool:
        return (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()

    ckpt_file = work_dir / "ckpt_dir.txt"
    if ckpt_file.exists():
        ckpt_dir = pathlib.Path(ckpt_file.read_text().strip())
        if weights_present(ckpt_dir):
            print(f"[Phase 1] Using cached checkpoint: {ckpt_dir}")
            return ckpt_dir

    print(f"[Phase 1] Downloading pretrained {CODI_HF_ID}...")
    ckpt_dir = pathlib.Path(snapshot_download(
        repo_id=CODI_HF_ID,
        force_download=True,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    ))

    if not weights_present(ckpt_dir):
        files = list(ckpt_dir.iterdir())
        sys.exit(
            f"[Phase 1] ✗ Weights missing after download. Files: {[f.name for f in files]}"
        )

    ckpt_file.write_text(str(ckpt_dir), encoding="utf-8")
    print(f"[Phase 1] ✓ Checkpoint: {ckpt_dir}")
    return ckpt_dir


# ── Data conversion ────────────────────────────────────────────────────────────

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


def convert_to_icot_format(src_path: pathlib.Path, dst_path: pathlib.Path):
    """
    Convert GSM8K split → CODI's iCoT (implicit CoT) format.

    Expected output schema per line:
        {"question": "...", "cot": "...", "answer": "42"}

    The original GSM8K answer field has the format:
        "Janet's ducks lay 16 eggs ... #### 16"
    We split on "####" to get the CoT and numeric answer separately.
    """
    raw = load_jsonl(src_path)
    converted = []
    skipped = 0

    for item in raw:
        question = item.get("question", "")
        raw_answer = item.get("answer", "")

        if "####" in raw_answer:
            cot_text, num_answer = raw_answer.split("####", 1)
            cot_text   = cot_text.strip()
            num_answer = num_answer.strip().replace(",", "")
        else:
            # Already clean (may have been pre-split)
            cot_text   = item.get("cot", raw_answer)
            num_answer = item.get("answer", raw_answer).strip().replace(",", "")

        if not question or not num_answer:
            skipped += 1
            continue

        converted.append({
            "question": question,
            "cot":      cot_text,
            "answer":   num_answer,
        })

    print(f"[Phase 1] Converted {len(converted)} examples ({skipped} skipped) — {src_path.name}")
    save_jsonl(converted, dst_path)
    return converted


# ── Training script preparation ────────────────────────────────────────────────

def prepare_codi_bundle(bundle_dir: pathlib.Path) -> bool:
    """
    Verify that codi_bundle/train.py and codi_bundle/src/model.py exist.
    These are the pre-patched CPU/GPU-compatible files from the repository.
    """
    train_py  = bundle_dir / "train.py"
    model_py  = bundle_dir / "src" / "model.py"
    init_py   = bundle_dir / "src" / "__init__.py"

    missing = [p for p in [train_py, model_py] if not p.exists()]
    if missing:
        print(f"[Phase 1] ✗ Missing bundle files: {missing}")
        print("  Ensure codi_bundle/ is present. Run 'powershell .\\bundle_codi.ps1' to rebuild.")
        return False

    if not init_py.exists():
        init_py.touch()

    print(f"[Phase 1] ✓ codi_bundle ready: {bundle_dir}")
    return True


def build_training_command(
    ckpt_dir:      pathlib.Path,
    train_data:    pathlib.Path,
    output_dir:    pathlib.Path,
    bundle_dir:    pathlib.Path,
    args,
) -> list:
    """
    Build the training CLI command for codi_bundle/train.py.
    """
    cmd = [
        sys.executable, "train.py",
        # Model
        "--model_name_or_path", "gpt2",
        "--seed",               str(args.seed),
        "--model_max_length",   str(MODEL_MAX_LEN),
        # LoRA
        "--lora_r",             str(LORA_R),
        "--lora_alpha",         str(LORA_ALPHA),
        "--lora_init",
        "--use_lora",           "True",
        # CODI latent reasoning
        "--num_latent",         str(NUM_LATENT),
        "--use_prj",            "True",
        "--prj_dim",            str(PRJ_DIM),
        "--prj_no_ln",          "False",
        "--prj_dropout",        "0.0",
        "--inf_latent_iterations", str(INF_LATENT),
        "--remove_eos",         "True",
        # Pretrained checkpoint to fine-tune from
        "--ckpt_dir",           str(ckpt_dir),
        # Data — uses custom_local: prefix supported by patched train.py
        "--data_name",          f"custom_local:{train_data.resolve()}",
        # Training hyperparameters
        "--output_dir",         str(output_dir),
        "--num_train_epochs",   str(args.num_epochs),
        "--learning_rate",      str(args.learning_rate),
        "--per_device_train_batch_size", str(args.batch_size),
        "--logging_steps",      "50",
        "--save_steps",         "500",
        "--expt_name",          "phase1",
    ]

    # Enable wandb if user is logged in
    try:
        import wandb
        if wandb.api.api_key:
            cmd.extend(["--report_to", "wandb"])
    except:
        pass  # wandb not available or not logged in, will use default "none"

    if args.bf16:
        cmd.append("--bf16")

    return cmd


# ── Streaming subprocess ───────────────────────────────────────────────────────

def run_with_logging(cmd: list, cwd: str, log_path: pathlib.Path) -> tuple:
    """Run cmd, stream output live, save to log. Returns (returncode, log_str)."""
    print(f"\n[Phase 1] $ {' '.join(str(c) for c in cmd)}\n")
    lines = []
    proc = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True,
        encoding="utf-8", errors="replace",
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    log = "".join(lines)
    log_path.write_text(log, encoding="utf-8")
    return proc.returncode, log


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Train base CODI-GPT2 model for continuous reasoning"
    )
    parser.add_argument("--train-data",    default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--val-data",      default=DEFAULT_VAL_DATA)
    parser.add_argument("--output-dir",    default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--work-dir",      default=DEFAULT_WORK_DIR,
                        help="Where pretrained checkpoint is cached")
    parser.add_argument("--bundle-dir",    default="codi_bundle",
                        help="Path to codi_bundle/ directory")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--num_epochs",    type=int,   default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size",    type=int,   default=2,
                        help="Per-device batch size. Use 2 on CPU, 4-8 on GPU.")
    parser.add_argument("--bf16",          action="store_true",
                        help="Enable bfloat16 (GPU only, speeds up training ~2x)")
    args = parser.parse_args()

    train_data  = pathlib.Path(args.train_data)
    output_dir  = pathlib.Path(args.output_dir).resolve()
    work_dir    = pathlib.Path(args.work_dir).resolve()
    bundle_dir  = pathlib.Path(args.bundle_dir).resolve()

    print("\n" + "=" * 62)
    print("  Phase 1 — Base CODI-GPT2 Training")
    print("  Protocol: Coconut/CCoT curriculum on GSM8K")
    print("=" * 62)

    # Validate inputs
    if not train_data.exists():
        sys.exit(
            f"\n[Phase 1] ✗ Training data not found: {train_data}\n"
            f"  Run first:  python split_dataset.py\n"
        )

    n_train = sum(1 for _ in open(train_data, encoding="utf-8"))
    print(f"\n  Training examples : {n_train}")
    print(f"  Output directory  : {output_dir}")
    print(f"  Epochs            : {args.num_epochs}  |  LR: {args.learning_rate}")
    print(f"  Batch size        : {args.batch_size}  |  bf16: {args.bf16}")
    print()

    # 1) Install / verify dependencies
    ensure_dependencies()

    # 2) Verify codi_bundle
    if not prepare_codi_bundle(bundle_dir):
        sys.exit(1)

    # 3) Download pretrained checkpoint
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = get_checkpoint(work_dir)

    # 4) Convert data → CODI iCoT format
    codi_data_dir = bundle_dir / "data" / "phase1"
    codi_data_dir.mkdir(parents=True, exist_ok=True)
    train_codi = codi_data_dir / "train.jsonl"
    convert_to_icot_format(train_data, train_codi)

    # 5) Build and run training command
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_training_command(ckpt_dir, train_codi, output_dir, bundle_dir, args)

    log_path = output_dir / "train_log.txt"
    print(f"\n[Phase 1] Starting curriculum training — log: {log_path}")
    print(f"[Phase 1] Estimated time: ~2h on T4 GPU | ~100h on CPU\n")

    t0 = time()
    rc, log = run_with_logging(cmd, cwd=str(bundle_dir), log_path=log_path)
    elapsed = time() - t0

    print(f"\n[Phase 1] Finished in {elapsed/3600:.2f}h  (exit code: {rc})")

    if rc != 0:
        last_lines = log.strip().split("\n")[-30:]
        print("\n[Phase 1] ✗ Training failed. Last log lines:")
        print("\n".join(last_lines))
        print(f"\nFull log: {log_path}")
        sys.exit(1)

    # 6) Verify checkpoint was saved
    saved = (
        list(output_dir.glob("**/*.bin")) +
        list(output_dir.glob("**/*.safetensors"))
    )
    if saved:
        print(f"\n[Phase 1] ✓ Checkpoint saved:")
        for f in saved[:5]:
            print(f"    {f}")
    else:
        print(f"\n[Phase 1] ⚠  No weights found in {output_dir}")
        print("  The model may still have been saved under an epoch subdirectory.")
        print(f"  Check: {output_dir}")

    # 7) Save metadata
    meta = {
        "phase":        1,
        "model":        "CODI-GPT2 (zen-E/CODI-gpt2)",
        "pretrained":   str(ckpt_dir),
        "train_data":   str(train_data),
        "n_train":      n_train,
        "epochs":       args.num_epochs,
        "learning_rate":args.learning_rate,
        "batch_size":   args.batch_size,
        "elapsed_hours":round(elapsed / 3600, 3),
        "seed":         args.seed,
        "output_dir":   str(output_dir),
    }
    (output_dir / "phase1_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"\n[Phase 1] ✓ Done. Next step:")
    print(f"  python phase2_extract_vector.py --ckpt-dir {output_dir}\n")


if __name__ == "__main__":
    main()