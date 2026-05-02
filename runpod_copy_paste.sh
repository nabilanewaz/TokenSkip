#!/bin/bash
# runpod_copy_paste.sh
# 
# Copy-paste this ENTIRE SCRIPT into RunPod SSH terminal
# It will do everything automatically!
#
# Usage:
#   1. SSH into pod: ssh root@your-pod-id.runpod.io -p port
#   2. Copy-paste this entire script
#   3. Wait ~15 minutes
#   4. Run: bash run_full_evaluation.sh

echo "🚀 RunPod TokenSkip Auto-Setup"
echo "================================"
echo "This will take ~15 minutes..."
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: System & Python (3 min)
# ─────────────────────────────────────────────────────────────────────────────

echo "[1/5] Updating system..."
apt-get update > /dev/null 2>&1
apt-get install -y git curl wget screen > /dev/null 2>&1

echo "[1/5] Creating Python environment..."
cd /root
python -m venv tokenskip_env
source tokenskip_env/bin/activate

echo "[1/5] Installing PyTorch..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Clone Your Code
# ─────────────────────────────────────────────────────────────────────────────

echo "[2/5] Getting TokenSkip code..."

# OPTION A: From GitHub (uncomment and replace with your repo)
# git clone https://github.com/your-org/TokenSkip.git

# OPTION B: If code is already uploaded, it's here:
# /root/TokenSkip

# OPTION C: Create empty structure (for manual upload later)
if [ ! -d "TokenSkip" ]; then
    mkdir -p TokenSkip
    cd TokenSkip
    mkdir -p logs outputs datasets report
    cd /root
fi

cd TokenSkip

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: Dependencies (2 min)
# ─────────────────────────────────────────────────────────────────────────────

echo "[3/5] Installing dependencies..."
pip install -q transformers peft datasets huggingface-hub accelerate safetensors wandb

# ─────────────────────────────────────────────────────────────────────────────
# PART 4: Download Data (5 min)
# ─────────────────────────────────────────────────────────────────────────────

echo "[4/5] Downloading GSM8K dataset..."
mkdir -p datasets/gsm8k_split

if [ -f "split_dataset.py" ]; then
    python split_dataset.py --full --seed 42
else
    # If split_dataset.py not present, download manually
    python -c "
from datasets import load_dataset
import json
import pathlib

print('Downloading GSM8K from HuggingFace...')
ds = load_dataset('gsm8k', 'main')

# Save raw
train_path = pathlib.Path('datasets/gsm8k/train.jsonl')
test_path = pathlib.Path('datasets/gsm8k/test.jsonl')
train_path.parent.mkdir(parents=True, exist_ok=True)

with open(train_path, 'w') as f:
    for ex in ds['train']:
        json.dump({'question': ex['question'], 'answer': ex['answer']}, f)
        f.write('\n')

with open(test_path, 'w') as f:
    for ex in ds['test']:
        json.dump({'question': ex['question'], 'answer': ex['answer']}, f)
        f.write('\n')

print(f'✓ Downloaded {len(ds[\"train\"])} train + {len(ds[\"test\"])} test')
"
fi

# ─────────────────────────────────────────────────────────────────────────────
# PART 5: Verify GPU (1 min)
# ─────────────────────────────────────────────────────────────────────────────

echo "[5/5] Verifying GPU setup..."
python -c "
import torch
print(f'✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NOT FOUND\"}')
print(f'✓ CUDA: {torch.version.cuda}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'✓ Memory: {props.total_memory / 1e9:.1f} GB')
"

# ─────────────────────────────────────────────────────────────────────────────
# DONE!
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✓ READY TO RUN!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Next:"
echo "  cd /root/TokenSkip"
echo "  bash run_full_evaluation.sh 2>&1 | tee logs/eval.log"
echo ""
echo "Or run in background:"
echo "  screen -S eval"
echo "  bash run_full_evaluation.sh"
echo "  (Ctrl+A, then D to detach)"
echo ""
echo "Monitor progress:"
echo "  tail -f /root/TokenSkip/logs/evaluation_master.log"
echo ""
echo "GPU usage:"
echo "  nvidia-smi -l 1"
echo ""
echo "Stop pod when done (saves money):"
echo "  Click Stop in RunPod web UI"
echo ""
