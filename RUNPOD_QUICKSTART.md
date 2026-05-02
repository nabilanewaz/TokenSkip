# RunPod Quick Start — TL;DR

## 5-Minute Setup

### Step 1: Create Pod (2 min)
1. Go to **runpod.io** → Sign up
2. **Pods** → **GPU Pods** → **Deploy**
3. Select: **A100 (40GB)** or **RTX 4090**
4. Container: **PyTorch 2.x** 
5. Disk: **20 GB**
6. Click **Deploy**

### Step 2: Connect via SSH (1 min)
```bash
# Copy SSH command from Pod dashboard and run:
ssh root@your-pod-id.runpod.io -p 12345
```

### Step 3: One-Line Setup (10 min)
```bash
# Copy this entire block and paste in SSH terminal:

cd /root && apt-get update -qq && apt-get install -y git screen >/dev/null 2>&1 && \
python -m venv tokenskip_env && source tokenskip_env/bin/activate && \
pip install -q torch transformers peft datasets huggingface-hub accelerate safetensors && \
git clone https://github.com/your-org/TokenSkip.git && \
cd TokenSkip && \
mkdir -p logs outputs && \
python split_dataset.py --full --seed 42 && \
echo "✓ Ready to run!"
```

### Step 4: Run Pipeline (8 hours)
```bash
# Option A: Direct (attached to terminal)
bash run_full_evaluation.sh 2>&1 | tee logs/eval.log

# Option B: Background with screen (recommended)
screen -S eval
bash run_full_evaluation.sh
# Press Ctrl+A, then D to detach
# Reconnect: screen -r eval
```

### Step 5: Download Results
**From Pod web UI**:
1. Pod Dashboard → **Files**
2. Download `logs/*.csv` and `outputs/eval_grid`

**Or from local machine**:
```bash
scp -r root@your-pod-id.runpod.io:/root/TokenSkip/logs ~/results/
```

### Step 6: Stop Pod (Save Money!)
Click **Stop** in Pod dashboard
- Running cost: **$0.44/hr** (A100)
- Stopped cost: **$0**

---

## Cost Breakdown

| GPU | Speed | Cost/hr | 8hr Run |
|-----|-------|---------|---------|
| A100 (40GB) | ~8h | $0.44 | **$3.50** ← BEST |
| H100 (80GB) | ~6h | $0.79 | $4.74 |
| RTX 4090 | ~10h | $0.28 | **$2.80** ← CHEAPEST |

---

## Common Issues

| Issue | Fix |
|-------|-----|
| Pod not running | Check: `nvidia-smi` |
| Out of Memory | Use `--batch_size 2` or `--eval_batch_size 4` |
| Got disconnected | Use `screen` (see Step 4) |
| Models downloading slowly | They cache after first run |
| Disk full | Check: `df -h` and `du -sh *` |

---

## Monitor Progress (Live)

```bash
# In another SSH terminal to same pod:

# Watch main log
tail -f logs/evaluation_master.log

# GPU utilization
watch -n 2 nvidia-smi

# Disk usage
watch -n 5 'du -sh * && echo "---" && df -h /'
```

---

## Upload Your Code

If code is on GitHub:
```bash
git clone https://github.com/your-org/TokenSkip.git
```

Or upload via SCP from local machine:
```bash
scp -r ~/TokenSkip root@your-pod-id.runpod.io:/root/
```

Or use RunPod web UI → **Files** → Upload zip

---

## Full Timeline

```
T+0:00   Pod created
T+0:02   SSH connected
T+0:12   Setup complete
T+0:20   Data downloaded
T+0:30   First training starts (Phase 1)
T+2:30   Vector extraction (Phase 2)
T+2:45   CODI steering (Phase 3)
T+3:45   HF model evaluation (Baselines)
T+5:45   HF model steering
T+7:45   Aggregation & token metrics
T+8:00   Complete! Download results
T+8:01   Stop pod (save money)
```

---

## Advanced: Parallel Pods

For faster results, use **3 pods** in parallel:

**Pod 1**: CODI (Phase 1-3) only
**Pod 2**: Phi-2 + Llama-3.2 (both conditions/alphas)
**Pod 3**: Qwen models (all 4 variants)

Then download and merge results.

Cost: **~$9 total** but **~3x faster** (2.5 hours instead of 8)

---

## Helpful Commands

```bash
# Check everything is working
nvidia-smi
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Run specific stage
python phase1_train.py --num_epochs 1 --batch_size 2  # quick test
python evaluation.py --model-type phi2 --no-cot --eval_batch_size 4

# View results as they run
tail -f logs/*steered*.log
tail -f logs/*aggregated*.log

# Compress for download
tar -czf results.tar.gz logs/ outputs/
ls -lh results.tar.gz
```

---

## Support

**Full documentation**: See `RUNPOD_SETUP.md`
**Command reference**: See `COMMANDS_REFERENCE.md`
**Evaluation guide**: See `EVALUATION_GUIDE.md`

---

**Total cost for full evaluation**: ~$3-4 (A100)  
**Total time**: ~8-10 hours  
**Effort**: ~10 minutes (mostly waiting)
