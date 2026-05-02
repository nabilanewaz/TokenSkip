#!/bin/bash
# runpod_quick_setup.sh
# Copy-paste this script in RunPod terminal for instant setup
# Usage: bash runpod_quick_setup.sh

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  RunPod TokenSkip Evaluation Setup${NC}"
echo -e "${BLUE}║  Estimated time: 15-20 minutes${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: System packages
echo -e "${BLUE}[1/7]${NC} Installing system packages..."
apt-get update > /dev/null 2>&1
apt-get install -y git curl wget screen htop > /dev/null 2>&1
echo -e "${GREEN}✓${NC} System packages ready"

# Step 2: Python environment
echo -e "${BLUE}[2/7]${NC} Setting up Python environment..."
cd /root
if [ ! -d "tokenskip_env" ]; then
    python -m venv tokenskip_env
fi
source tokenskip_env/bin/activate
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Python environment ready"

# Step 3: PyTorch (CUDA 11.8)
echo -e "${BLUE}[3/7]${NC} Installing PyTorch (CUDA 11.8)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1
echo -e "${GREEN}✓${NC} PyTorch installed"

# Step 4: Python dependencies
echo -e "${BLUE}[4/7]${NC} Installing Python dependencies..."
pip install transformers peft datasets huggingface-hub accelerate safetensors \
    wandb tensorboard > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Dependencies installed"

# Step 5: Clone/setup TokenSkip
echo -e "${BLUE}[5/7]${NC} Setting up TokenSkip repository..."
if [ ! -d "TokenSkip" ]; then
    echo "Cloning repository (or paste your repo URL):"
    echo "  git clone <your-repo-url>"
    echo ""
    echo "For now, creating empty TokenSkip directory..."
    mkdir -p TokenSkip
fi
cd TokenSkip
mkdir -p logs outputs datasets report
echo -e "${GREEN}✓${NC} Repository ready"

# Step 6: Download GSM8K
echo -e "${BLUE}[6/7]${NC} Downloading GSM8K dataset..."
if [ ! -f "datasets/gsm8k_split/test.jsonl" ]; then
    mkdir -p datasets/gsm8k_split
    python -c "
from datasets import load_dataset
print('  Downloading GSM8K...')
ds = load_dataset('gsm8k', 'main', cache_dir='/root/.cache/huggingface')
print(f'  ✓ Loaded {len(ds[\"train\"])} train + {len(ds[\"test\"])} test')
" 2>/dev/null
else
    echo "  Dataset already exists, skipping..."
fi
echo -e "${GREEN}✓${NC} Dataset ready"

# Step 7: Verify GPU
echo -e "${BLUE}[7/7]${NC} Verifying GPU setup..."
python -c "
import torch
print(f'  GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU device: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
" 2>/dev/null

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✓ Setup Complete!${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate environment:"
echo -e "   ${BLUE}source /root/tokenskip_env/bin/activate${NC}"
echo ""
echo "2. Copy your code to TokenSkip/ (or use git clone):"
echo -e "   ${BLUE}cd /root/TokenSkip${NC}"
echo ""
echo "3. Run evaluation:"
echo -e "   ${BLUE}bash run_full_evaluation.sh${NC}"
echo ""
echo "4. Monitor with screen (runs in background):"
echo -e "   ${BLUE}screen -S eval${NC}"
echo -e "   ${BLUE}bash run_full_evaluation.sh${NC}"
echo -e "   (Press Ctrl+A, then D to detach)"
echo ""
echo "5. Check logs:"
echo -e "   ${BLUE}tail -f logs/evaluation_master.log${NC}"
echo ""
echo "6. Monitor GPU:"
echo -e "   ${BLUE}nvidia-smi -l 1${NC}"
echo ""
