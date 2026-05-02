# TokenSkip Evaluation on RunPod

Complete guide to run the full evaluation pipeline on RunPod GPU infrastructure.

## What is RunPod?

RunPod is a serverless GPU cloud platform with:
- Hourly billing (no long-term contracts)
- Fast setup (1-2 minutes)
- Support for NVIDIA A100, H100, RTX 4090, etc.
- Pre-configured PyTorch/CUDA environments

**Cost estimate** (RunPod pricing as of 2024):
- **A100 (40GB)**: $0.44/hr → ~$3.50 for full pipeline (8 hours)
- **H100 (80GB)**: $0.79/hr → ~$6.30 for full pipeline
- **RTX 4090 (24GB)**: $0.28/hr → ~$2.24 for full pipeline

---

## Step 1: Create RunPod Account & Pod

### 1.1 Sign up
1. Go to https://www.runpod.io
2. Create account and add payment method
3. Navigate to "Pods" → "GPU Pods"

### 1.2 Select GPU & Template
1. Click **"Deploy"**
2. **GPU Selection** (recommended):
   - **A100 (40GB)** ← Best balance of speed & cost
   - Or RTX 4090 if you want to save ~$1-2
   - Avoid H100/A6000 unless budget is unlimited

3. **Container Image**:
   - Select: **PyTorch 2.x** (Official NVIDIA)
   - OR: **runpod/pytorch** (popular base)

4. **Container Disk**: **20 GB** (minimum, for datasets + models)

5. Click **"Deploy"** and wait 1-2 minutes

---

## Step 2: Connect to Your Pod

### 2.1 Via Jupyter Notebook (Easiest for beginners)
1. Pod dashboard → click pod name
2. Click **"Jupyter Notebook"** button
3. Jupyter opens in new tab

### 2.2 Via SSH Terminal (Recommended for scripting)
1. Pod dashboard → copy **SSH command**
2. In your local terminal:
   ```bash
   ssh root@your-pod-id.runpod.io -p 12345
   ```
   (Replace with your actual pod ID and port)

### 2.3 Via RunPod CLI (Advanced)
```bash
# Install RunPod CLI
pip install runpod

# List your pods
runpod pod list

# Connect
runpod ssh pod-id
```

**We'll use SSH for this guide** (fastest for batch jobs)

---

## Step 3: Setup Environment in Pod

Once connected via SSH:

```bash
# Update system packages
apt-get update && apt-get install -y git wget curl

# Navigate to home
cd /root

# Clone TokenSkip repository (or upload your code)
git clone https://github.com/your-org/TokenSkip.git
cd TokenSkip

# Create virtual environment (optional but recommended)
python -m venv tokenskip_env
source tokenskip_env/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft datasets huggingface-hub accelerate safetensors

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

**Expected output**:
```
GPU: NVIDIA A100-SXM4-40GB
CUDA: True
```

---

## Step 4: Download Data & Models

### 4.1 Download GSM8K Dataset
```bash
# Create datasets directory
mkdir -p datasets/gsm8k_split

# Download GSM8K (pre-split)
cd datasets
python -c "
from datasets import load_dataset
ds = load_dataset('gsm8k', 'main')
ds['train'].to_json('gsm8k/train.jsonl')
ds['test'].to_json('gsm8k/test.jsonl')
print(f'Downloaded: {len(ds[\"train\"])} train + {len(ds[\"test\"])} test')
"

# Split into train/steer/validation/test
cd ..
python split_dataset.py --full --seed 42
```

**Estimated time**: 5-10 minutes

### 4.2 Pre-cache Models (Optional but Recommended)
```bash
# Pre-download models to cache (avoid re-downloading during runs)
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
models = [
    'microsoft/phi-2',
    'meta-llama/Llama-3.2-3B',
    'Qwen/Qwen2.5-3B',
    'Qwen/Qwen2.5-1.5B',
    'Qwen/Qwen2.5-0.5B',
    'Qwen/Qwen2.5-Math-1.5B',
]
for model_id in models:
    print(f'Downloading {model_id}...')
    AutoTokenizer.from_pretrained(model_id)
    # Don't load full model to save RAM (just tokenizer)
print('Cached!')
"
```

**Estimated time**: 15-30 minutes (first time only)

---

## Step 5: Run Evaluation Pipeline

### Option A: Run Full Pipeline (Recommended)

```bash
# Make script executable
chmod +x run_full_evaluation.sh

# Run with logging
nohup bash run_full_evaluation.sh > logs/runpod_master.log 2>&1 &

# Monitor progress in real-time
tail -f logs/runpod_master.log

# Or check GPU usage
watch -n 2 nvidia-smi
```

**Expected duration**: 6-8 hours on A100

### Option B: Run Individual Stages (For Debugging)

```bash
# Stage 1: Split (fast)
python split_dataset.py --full --seed 42

# Stage 2: Phase 1 Training (slow ~2-4h)
python phase1_train.py \
    --train-data datasets/gsm8k_split/llm_train.jsonl \
    --output-dir outputs/phase1_checkpoint \
    --num_epochs 3 \
    --batch_size 4 \
    --bf16 \
    --seed 42 \
    2>&1 | tee logs/phase1_train.log

# Stage 3: Phase 2 Extraction (fast ~15min)
python phase2_extract_vector.py \
    --steer-data datasets/gsm8k_split/steer_train.jsonl \
    --ckpt-dir outputs/phase1_checkpoint \
    --out-dir outputs/phase2_truth_vector \
    --n-samples 5 \
    --bf16 \
    --seed 42 \
    2>&1 | tee logs/phase2_extract.log

# Stage 4: Phase 3 Steering (slow ~1h)
python phase3_steer_inference.py \
    --eval-data datasets/gsm8k_split/test.jsonl \
    --vector-dir outputs/phase2_truth_vector \
    --ckpt-dir outputs/phase1_checkpoint \
    --out-dir outputs/phase3_results \
    --alphas 0 0.5 1 2 5 10 20 50 -0.5 -1 \
    --random-noise \
    --bf16 \
    --seed 42 \
    2>&1 | tee logs/phase3_steering.log

# Stage 5-6: HF Model Baselines & Steering (slow ~2-3h)
# See COMMANDS_REFERENCE.md for exact commands

# Stage 7: Aggregate Results (fast)
python compare_all.py --eval-grid outputs/eval_grid

# Stage 8: Extract Token Metrics (fast)
python extract_token_metrics.py
```

---

## Step 6: Monitor Job Progress

### Option 1: Real-time Log Monitoring (SSH)
```bash
# Watch main log
tail -f logs/evaluation_master.log

# Check GPU utilization
nvidia-smi -l 1  # Update every 1 second

# Check disk space
df -h
```

### Option 2: Save Results & Download

While running, in another SSH terminal:

```bash
# Compress results for download
tar -czf results_$(date +%Y%m%d_%H%M%S).tar.gz \
    logs/ \
    outputs/ \
    report/

# Show file size
ls -lh results_*.tar.gz

# (You can download via RunPod web UI: Pod → Files)
```

---

## Step 7: Download Results

### From RunPod Web UI (Easiest)
1. Go to Pod dashboard
2. Click **"Files"**
3. Navigate to `/root/TokenSkip/logs` and `outputs/`
4. Download `.csv`, `.json`, and `.log` files

### From Local Machine (Via SCP)
```bash
# Download all logs and results
scp -r root@your-pod-id.runpod.io:/root/TokenSkip/logs ~/TokenSkip_results/
scp -r root@your-pod-id.runpod.io:/root/TokenSkip/outputs ~/TokenSkip_results/

# Or specific file
scp root@your-pod-id.runpod.io:/root/TokenSkip/logs/*.csv ~/Downloads/
```

---

## Complete RunPod Setup Script

Save as `runpod_setup.sh` and run once:

```bash
#!/bin/bash
# Complete RunPod setup script

set -e

echo "======================================"
echo "RunPod TokenSkip Setup"
echo "======================================"

# 1. System setup
echo "[1/6] Updating system..."
apt-get update && apt-get install -y git wget curl screen

# 2. Setup Python environment
echo "[2/6] Setting up Python environment..."
cd /root
python -m venv tokenskip_env
source tokenskip_env/bin/activate
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch
echo "[3/6] Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
echo "[4/6] Installing Python dependencies..."
pip install transformers peft datasets huggingface-hub accelerate safetensors

# 5. Clone repository
echo "[5/6] Cloning TokenSkip repository..."
git clone https://github.com/your-org/TokenSkip.git
cd TokenSkip

# 6. Download data
echo "[6/6] Downloading GSM8K dataset..."
python split_dataset.py --full --seed 42

echo ""
echo "======================================"
echo "✓ Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source /root/tokenskip_env/bin/activate"
echo "  2. Run pipeline: bash run_full_evaluation.sh"
echo "  3. Monitor: tail -f logs/evaluation_master.log"
echo ""
```

**Run it**:
```bash
chmod +x runpod_setup.sh
bash runpod_setup.sh
```

---

## Recommended RunPod Workflow

```
┌─────────────────────────────────┐
│  1. Create Pod (A100, 20GB disk) │
└─────────┬───────────────────────┘
          │
          ↓
┌─────────────────────────────────┐
│  2. SSH into Pod                 │
└─────────┬───────────────────────┘
          │
          ↓
┌─────────────────────────────────┐
│  3. Run setup script             │
│     bash runpod_setup.sh         │
└─────────┬───────────────────────┘
          │ (5-10 min)
          ↓
┌─────────────────────────────────┐
│  4. Start pipeline in screen     │
│     screen -S eval              │
│     bash run_full_evaluation.sh  │
│     Ctrl+A, D to detach         │
└─────────┬───────────────────────┘
          │ (6-8 hours on A100)
          ↓
┌─────────────────────────────────┐
│  5. Monitor progress             │
│     screen -r eval              │
│     tail -f logs/eval_master.log│
└─────────┬───────────────────────┘
          │
          ↓
┌─────────────────────────────────┐
│  6. Download results via web UI  │
│     or scp from local machine    │
└─────────┬───────────────────────┘
          │
          ↓
┌─────────────────────────────────┐
│  7. Stop Pod (to save money!)    │
└─────────────────────────────────┘
```

---

## Cost Optimization Tips

### Reduce Costs
1. **Use RTX 4090** instead of A100 (saves ~$1.50/hr)
2. **Reduce batch sizes** (trades speed for lower RAM requirements)
3. **Run only necessary stages** (skip CODI Phase 1 if only testing HF models)
4. **Use spot pricing** (RunPod Spot instances - 50% cheaper but can pause)

### Speed vs Cost Trade-offs
```
Full evaluation (all 7 models):
  A100 (40GB)    — 8 hours  @ $0.44/hr = $3.50  ← RECOMMENDED
  H100 (80GB)    — 6 hours  @ $0.79/hr = $4.74
  RTX 4090 (24GB)— 10 hours @ $0.28/hr = $2.80  ← CHEAPEST

HF models only (skip CODI):
  Any GPU        — 2 hours  → ~$0.56-2.00 total
```

---

## Troubleshooting on RunPod

### Problem: Out of Memory (OOM)
```bash
# Reduce batch sizes
python hidden_steer.py --model-type phi2 --eval_batch_size 4  # instead of 8
python phase1_train.py --batch_size 2  # instead of 4
```

### Problem: Pod Disconnects
```bash
# Use screen or tmux to run in background
screen -S evaluation
bash run_full_evaluation.sh
# Detach: Ctrl+A, then D
# Reattach: screen -r evaluation
```

### Problem: Slow Model Downloads
```bash
# Pre-cache models while running other stages
# Or increase timeout
pip install --default-timeout=1000 transformers

# Check internet speed
speedtest-cli
```

### Problem: Disk Full
```bash
# Check disk usage
df -h
du -sh *

# Clean cache
rm -rf ~/.cache/huggingface/hub/*
```

---

## Advanced: Run Multiple Pods in Parallel

For even faster results, split work across multiple pods:

**Pod 1: CODI Phase 1-3**
```bash
python phase1_train.py --num_epochs 3 --batch_size 4 --bf16
python phase2_extract_vector.py --n-samples 5 --bf16
python phase3_steer_inference.py --alphas 0 0.5 1 2 5 10 20 50 -0.5 -1 --bf16
```

**Pod 2: Phi-2 + Llama-3.2-3B (baselines + steering)**
```bash
python evaluation.py --model-type phi2 --no-cot
python evaluation.py --model-type phi2
python hidden_steer.py --model-type phi2 --condition ccot
python hidden_steer.py --model-type phi2 --condition random_noise --alphas 1.0
python hidden_steer.py --model-type phi2 --condition steered --alphas 0 0.5 1 2 5 10 20 50 -0.5 -1

# (repeat for llama32_3b)
```

**Pod 3: Qwen models**
```bash
# All 4 Qwen variants (0.5B, 1.5B, 3B, Math 1.5B)
# Each model takes ~20-30min for full sweep
```

**Cost**: ~$2-3 per pod × 3 pods = ~$6-9 total (faster than sequential)

---

## Post-Evaluation: Analyze Results

After downloading results:

```bash
# View aggregated metrics
cat logs/*aggregated_results.csv

# View token metrics
cat logs/*token_metrics.csv

# Parse JSON results
python3 -c "
import json
with open('logs/*token_metrics.json') as f:
    data = json.load(f)
    print(json.dumps(data, indent=2))
"

# Generate plots (if you want)
# pip install matplotlib pandas
# python plot_results.py
```

---

## Stopping Your Pod

**Important**: Stop the pod when done to avoid unnecessary charges!

### From RunPod Web UI
1. Go to Pods dashboard
2. Find your pod
3. Click **"Stop"** button

### From SSH
```bash
# Stop pod (graceful)
runpod pod stop <pod-id>

# Or just disconnect and stop from UI
exit
```

**Cost**: Running pod = $0.44/hr (A100). Stopped pod = free!

---

## RunPod vs Local GPU Comparison

| Factor | RunPod A100 | Local GPU (RTX 4090) |
|--------|------------|---------------------|
| **Setup time** | 2 min | 5 min |
| **Cost (8h run)** | $3.50 | $0 (already own) |
| **Speed** | Fast (~8h) | Slower (~10h) |
| **Maintenance** | None | Power, cooling, updates |
| **Can stop anytime** | ✓ Yes | ✗ No |
| **Share across team** | ✓ Easy | ✗ Hard |
| **Scale (multiple runs)** | ✓ Easy | ✗ Limited |

---

## Quick Command Cheat Sheet (RunPod)

```bash
# SSH into pod
ssh root@<pod-id>.runpod.io -p <port>

# Download data
python split_dataset.py --full

# Run full evaluation
bash run_full_evaluation.sh 2>&1 | tee logs/runpod.log

# Monitor in background
screen -S eval
bash run_full_evaluation.sh
# Ctrl+A, D to detach

# Check status
screen -r eval
tail -f logs/evaluation_master.log

# Download results (from your local machine)
scp root@<pod-id>.runpod.io:/root/TokenSkip/logs/*.csv ~/Downloads/

# Stop pod
runpod pod stop <pod-id>
```

---

**Questions?** See `EVALUATION_GUIDE.md` and `COMMANDS_REFERENCE.md` for more details.
