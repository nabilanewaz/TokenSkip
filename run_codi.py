"""
run_codi.py
-----------
Runs CODI-GPT2 on GSM8K and writes outputs in the same format as evaluation.py:
    outputs/<run_name>/samples/metrics.json
    outputs/<run_name>/samples/predictions.jsonl

This means compare_metrics.py works unchanged.

Usage (CPU):
    python run_codi.py --run-name codi_gpt2_gsm8k

Usage (GPU, faster):
    python run_codi.py --run-name codi_gpt2_gsm8k --bf16

Then compare with a baseline:
    python compare_metrics.py outputs/gpt2/gsm8k/ outputs/codi_gpt2_gsm8k/

Requirements:
    pip install peft datasets huggingface_hub
    git must be available on PATH (for cloning CODI repo)
"""

import os
import sys
import json
import subprocess
import argparse
import re
import shutil
import pathlib
from time import time


# ── Constants ─────────────────────────────────────────────────────────────────
CODI_REPO   = "https://github.com/zhenyi4/CODI.git"
CODI_HF_ID  = "zen-E/CODI-gpt2"


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_dependencies():
    """Install/reinstall packages CODI needs, forcing correct versions."""
    # These are pinned to what the CODI notebook used — don't change without testing
    packages = [
        "peft==0.15.2",
        "datasets==3.6.0",
        "huggingface_hub",
        "transformers==4.52.4",
        "accelerate==1.7.0",
    ]
    print(f"[run_codi] Installing CODI dependencies (force-reinstall to avoid conflicts)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--force-reinstall", "--no-deps"] + packages,
        check=True
    )
    # Install deps of deps normally (no force, just fill gaps)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet"] + packages,
        check=True
    )
    print("[run_codi] ✓ CODI dependencies ready.")


def clone_codi(work_dir: pathlib.Path) -> pathlib.Path:
    codi_dir = work_dir / "CODI"
    if not codi_dir.exists():
        print("[run_codi] Cloning CODI repo...")
        subprocess.run(
            ["git", "clone", CODI_REPO, str(codi_dir)],
            check=True
        )
    else:
        print("[run_codi] CODI repo already present, skipping clone.")
    return codi_dir


def download_checkpoint(work_dir: pathlib.Path) -> pathlib.Path:
    """
    Download zen-E/CODI-gpt2 via huggingface_hub.
    Verifies weights are present — if the previous download timed out and left
    an incomplete snapshot, forces a fresh download.
    """
    from huggingface_hub import snapshot_download

    def weights_present(path: pathlib.Path) -> bool:
        return (path / "model.safetensors").exists() or \
               (path / "pytorch_model.bin").exists()

    ckpt_file = work_dir / "ckpt_dir.txt"

    # Check cached path first
    if ckpt_file.exists():
        ckpt_dir = pathlib.Path(ckpt_file.read_text().strip())
        if weights_present(ckpt_dir):
            print(f"[run_codi] Using cached checkpoint: {ckpt_dir}")
            return ckpt_dir
        else:
            print(f"[run_codi] Cached checkpoint at {ckpt_dir} is incomplete (weights missing).")
            print(f"[run_codi] Forcing fresh download...")
            ckpt_file.unlink()  # remove stale cache pointer

    print(f"[run_codi] Downloading {CODI_HF_ID} checkpoint (this may take a few minutes)...")
    ckpt_dir = pathlib.Path(snapshot_download(
        repo_id=CODI_HF_ID,
        force_download=True,    # bypass the broken incomplete snapshot
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],  # skip non-PyTorch formats
    ))

    if not weights_present(ckpt_dir):
        # List what actually downloaded to help diagnose
        files = list(ckpt_dir.iterdir())
        print(f"[run_codi] ✗ Weights still missing after download. Files present:")
        for f in files:
            print(f"           {f.name}")
        sys.exit("[run_codi] Cannot continue without model weights.")

    ckpt_file.write_text(str(ckpt_dir))
    print(f"[run_codi] ✓ Checkpoint ready: {ckpt_dir}")
    return ckpt_dir


def patch_test_py(codi_dir: pathlib.Path) -> pathlib.Path:
    """
    Patch CODI/test.py → test_fixed.py with two fixes:
    1. batch=1 scalar bug: next_token_ids[b] crashes when tensor is 0-dim
    2. Hardcoded .to('cuda') calls — replaced with .to(DEVICE) so CPU-only torch works
    """
    dst = codi_dir / "test_fixed.py"

    src  = codi_dir / "test.py"
    code = src.read_text()

    changed = False

    # ── Fix 1: scalar next_token_ids bug ──────────────────────────────────
    pat1 = re.compile(
        r'^(?P<ind>\s*)pred_tokens\[b\]\.append\(next_token_ids\[b\]\.item\(\)\)\s*$',
        flags=re.M
    )
    if pat1.search(code):
        code = pat1.sub(
            lambda m: (
                f"{m.group('ind')}next_token_ids = next_token_ids.view(-1)\n"
                f"{m.group('ind')}pred_tokens[b].append(next_token_ids[b].item())"
            ),
            code, count=1
        )
        print("[run_codi] ✓ Patched next_token_ids scalar bug")
        changed = True
    else:
        print("[run_codi] ⚠ Scalar bug pattern not found (may already be patched)")

    # ── Fix 2: hardcoded CUDA — insert DEVICE constant + replace all .to('cuda') ──
    if "DEVICE = " not in code:
        # Insert device detection right after the import block (handle multi-line imports)
        lines = code.split('\n')
        import_end = 0
        in_import = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            # Start of an import
            if stripped.startswith('import ') or stripped.startswith('from '):
                in_import = True
                # Check if it's a multi-line import (ends with '(' or '\')
                if '(' in line and ')' not in line:
                    # Multi-line import starting
                    continue
                elif line.rstrip().endswith('\\'):
                    # Backslash continuation
                    continue
                else:
                    # Single-line import
                    import_end = i + 1
                    in_import = False
                    continue
            # Inside a multi-line import
            if in_import:
                if ')' in line:
                    # End of multi-line import
                    import_end = i + 1
                    in_import = False
                continue
            # Non-import, non-blank line - we're past imports
            if not in_import:
                break
        
        # Insert device block after the last import
        device_block = (
            "\n# --- CPU/GPU device patch (injected by run_codi.py) ---\n"
            "import torch as _torch\n"
            "DEVICE = 'cuda' if _torch.cuda.is_available() else 'cpu'\n"
            "# -------------------------------------------------------\n\n"
        )
        if import_end > 0:
            pos = len('\n'.join(lines[:import_end]))
            code = code[:pos] + '\n' + device_block + code[pos:]
        else:
            # Fallback: insert at top
            code = device_block + code
        print("[run_codi] ✓ Inserted DEVICE constant")
        changed = True

    # Replace every .to('cuda') and .to("cuda") with .to(DEVICE)
    before = code
    code = re.sub(r"\.to\(['\"]cuda['\"]\)", ".to(DEVICE)", code)
    # Also catch model.cuda() calls
    code = re.sub(r"\.cuda\(\)", ".to(DEVICE)", code)
    # And tensor(..., device='cuda')
    code = re.sub(r"device=['\"]cuda['\"]", "device=DEVICE", code)
    if code != before:
        print("[run_codi] ✓ Replaced hardcoded CUDA references with DEVICE")
        changed = True

    if changed or not dst.exists():
        dst.write_text(code)
        print(f"[run_codi] ✓ Wrote test_fixed.py")
    else:
        print("[run_codi] test_fixed.py already up to date.")

    return dst


def parse_codi_accuracy(log: str) -> float:
    """Extract the final accuracy number from CODI's stdout."""
    # CODI prints something like:  "GSM8K test accuracy: 0.4238"  or  "accuracy: 42.38"
    patterns = [
        r"accuracy[:\s]+([0-9]+\.[0-9]+)%?",
        r"accu[:\s]+([0-9]+\.[0-9]+)",
        r"([0-9]+\.[0-9]+)\s*$",
    ]
    for p in patterns:
        m = re.search(p, log, re.IGNORECASE | re.MULTILINE)
        if m:
            v = float(m.group(1))
            return v / 100.0 if v > 1.0 else v
    return 0.0


def run_codi_inference(codi_dir: pathlib.Path, ckpt_dir: pathlib.Path,
                       args) -> tuple[str, float]:
    """Run test_fixed.py, stream output live, save to log file, return (log, elapsed)."""
    log_path = codi_dir / "codi_run.txt"

    bf16_flag = "--bf16" if args.bf16 else ""

    cmd = [
        sys.executable, "test_fixed.py",
        "--data_name",              "gsm8k",
        "--model_name_or_path",     "gpt2",
        "--seed",                   str(args.seed),
        "--model_max_length",       str(args.model_max_length),
        "--lora_r",                 str(args.lora_r),
        "--lora_alpha",             str(args.lora_alpha),
        "--lora_init",
        "--batch_size",             str(args.batch_size),
        "--greedy",                 "True",
        "--num_latent",             str(args.num_latent),
        "--use_prj",                "True",
        "--prj_dim",                str(args.prj_dim),
        "--prj_no_ln",              "False",
        "--prj_dropout",            "0.0",
        "--inf_latent_iterations",  str(args.inf_latent_iterations),
        "--inf_num_iterations",     "1",
        "--remove_eos",             "True",
        "--use_lora",               "True",
        "--ckpt_dir",               str(ckpt_dir),
    ]
    if bf16_flag:
        cmd.append("--bf16")

    print(f"\n[run_codi] $ {' '.join(cmd)}\n")

    lines = []
    t0 = time()

    # Stream stdout+stderr live to console AND accumulate for log file
    process = subprocess.Popen(
        cmd,
        cwd=str(codi_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr into stdout
        bufsize=1,
        universal_newlines=True,
        encoding="utf-8",
        errors="replace",
    )

    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)

    process.wait()
    elapsed = time() - t0

    log = "".join(lines)
    log_path.write_text(log, encoding="utf-8")

    if process.returncode != 0:
        sys.exit(f"\n[run_codi] CODI inference failed (exit {process.returncode}). "
                 f"See log: {log_path}")

    return log, elapsed


def write_outputs(output_dir: pathlib.Path, accuracy: float,
                  elapsed: float, n_samples: int, log: str):
    """Write metrics.json (and a stub predictions.jsonl) so compare_metrics.py works."""
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json — same schema as evaluation.py
    metrics = {
        "n_samples":      n_samples,
        "accuracy":       accuracy,
        "avg_cot_length": None,       # CODI uses latents, not token-length CoT
        "sample_latency": elapsed / n_samples if n_samples else None,
        "source":         "CODI-GPT2 (zen-E/CODI-gpt2)",
    }
    (samples_dir / "metrics.json").write_text(json.dumps(metrics, indent=4))
    print(f"\n[run_codi] ✓ metrics.json → {samples_dir / 'metrics.json'}")

    # Stub predictions.jsonl — CODI doesn't expose per-example predictions easily
    # but we write one line so downstream tools don't crash
    stub = {"note": "CODI per-example predictions not extracted", "accuracy": accuracy}
    with open(samples_dir / "predictions.jsonl", "w") as f:
        f.write(json.dumps(stub) + "\n")

    # Also save the full CODI log for inspection
    (samples_dir / "codi_run.txt").write_text(log)
    print(f"[run_codi] ✓ Full CODI log  → {samples_dir / 'codi_run.txt'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run CODI-GPT2 on GSM8K and write metrics.json in TokenSkip format"
    )
    parser.add_argument("--run-name",             default="codi_gpt2_gsm8k",
                        help="Output folder name under outputs/")
    parser.add_argument("--output-base",          default="outputs",
                        help="Root outputs directory")
    parser.add_argument("--work-dir",             default="codi_workspace",
                        help="Directory where CODI repo and checkpoint are stored")

    # CODI inference settings (match the notebook exactly)
    parser.add_argument("--seed",                 type=int,   default=11)
    parser.add_argument("--model_max_length",     type=int,   default=512)
    parser.add_argument("--lora_r",               type=int,   default=128)
    parser.add_argument("--lora_alpha",           type=int,   default=32)
    parser.add_argument("--num_latent",           type=int,   default=6)
    parser.add_argument("--prj_dim",              type=int,   default=768)
    parser.add_argument("--inf_latent_iterations",type=int,   default=6)
    parser.add_argument("--batch_size",           type=int,   default=4)
    parser.add_argument("--ckpt-dir",            default=None,
                        help="Override checkpoint directory (default: auto-download zen-E/CODI-gpt2). "
                             "Use this to evaluate a fine-tuned model from train_codi.py.")
    parser.add_argument("--test-data",           default=None,
                        help="Path to custom test split (default: CODI uses its built-in GSM8K). "
                             "Pass datasets/gsm8k_split/test.jsonl to use the 20%% held-out split.")

    # Hardware
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Use bf16 (requires GPU). Omit for CPU — slower but works.")

    args = parser.parse_args()

    work_dir   = pathlib.Path(args.work_dir).resolve()
    output_dir = pathlib.Path(args.output_base) / args.run_name
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  CODI-GPT2 on GSM8K")
    print(f"  work dir   : {work_dir}")
    print(f"  output dir : {output_dir}")
    print(f"  bf16       : {args.bf16}")
    print("=" * 60)

    # 1) Dependencies
    ensure_dependencies()

    # 2) Setup
    codi_dir = clone_codi(work_dir)

    # Use provided checkpoint dir or download the pretrained one
    if args.ckpt_dir:
        ckpt_dir = pathlib.Path(args.ckpt_dir).resolve()
        print(f"[run_codi] Using custom checkpoint: {ckpt_dir}")
    else:
        ckpt_dir = download_checkpoint(work_dir)

    # Always re-patch — stale test_fixed.py may be missing the CUDA fix
    stale = codi_dir / "test_fixed.py"
    if stale.exists():
        stale.unlink()
        print("[run_codi] Removed stale test_fixed.py — will re-patch.")
    patch_test_py(codi_dir)

    # 3) Copy GSM8K datasets into CODI's expected location if needed
    gsm8k_src = pathlib.Path("datasets/gsm8k")
    gsm8k_dst = codi_dir / "datasets" / "gsm8k"
    if gsm8k_src.exists() and not gsm8k_dst.exists():
        print(f"[run_codi] Copying GSM8K dataset into CODI workspace...")
        shutil.copytree(gsm8k_src, gsm8k_dst)

    # Override test split if --test-data provided
    if args.test_data:
        test_src = pathlib.Path(args.test_data)
        if not test_src.exists():
            sys.exit(f"[run_codi] --test-data not found: {test_src}")
        gsm8k_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(test_src, gsm8k_dst / "test.jsonl")
        n_test = sum(1 for _ in open(test_src, encoding="utf-8"))
        args.n_samples = n_test
        print(f"[run_codi] Using custom test data: {test_src} ({n_test} examples)")

    # 4) Run inference
    print("\n[run_codi] Starting CODI inference...\n")
    log, elapsed = run_codi_inference(codi_dir, ckpt_dir, args)

    # 5) Parse accuracy
    accuracy = parse_codi_accuracy(log)
    print(f"\n[run_codi] Parsed accuracy: {accuracy*100:.2f}%  (elapsed: {elapsed:.1f}s)")

    # 6) Write outputs
    write_outputs(output_dir, accuracy, elapsed, args.n_samples, log)

    print(f"\n[run_codi] Done. Compare with another run:")
    print(f"  python compare_metrics.py outputs/<other_run>/ outputs/{args.run_name}/\n")


if __name__ == "__main__":
    main()