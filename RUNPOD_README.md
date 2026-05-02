# How to Run TokenSkip Evaluation on RunPod — Complete Guide

## 📚 Documentation Files Created

| File | Purpose | Read Time |
|------|---------|-----------|
| **RUNPOD_QUICKSTART.md** | 5-minute quick start (TL;DR) | 3 min ⭐ START HERE |
| **RUNPOD_SETUP.md** | Complete detailed setup guide | 15 min |
| **runpod_copy_paste.sh** | Auto-setup script (copy-paste) | 15 min runtime |
| **runpod_quick_setup.sh** | Interactive setup script | 10-15 min |

---

## 🚀 Ultra-Quick Start (90 seconds)

### Step 1: Create Pod on RunPod.io
- Go to **runpod.io/gpu-cloud**
- Click **Deploy** → Select **A100 (40GB)** or **RTX 4090**
- Container: **PyTorch 2.x**
- Disk: **20 GB**
- Wait 1-2 minutes

### Step 2: SSH Connect
```bash
ssh root@YOUR-POD-ID.runpod.io -p PORT
```
(Copy command from RunPod dashboard)

### Step 3: Run Setup Script
```bash
cd /root && bash runpod_copy_paste.sh
```

### Step 4: Run Evaluation
```bash
cd /root/TokenSkip
bash run_full_evaluation.sh
```

### Step 5: Download Results
Pod Dashboard → **Files** → Download logs and outputs

### Step 6: Stop Pod
Click **Stop** (free up costs)

**Total time**: ~8-10 hours | **Total cost**: ~$3-5 ✅

---

## 📋 Three Ways to Use RunPod

### Option 1: Minimal Setup (Fastest)
```bash
# Copy-paste this one command:
cd /root && python -m venv env && source env/bin/activate && \
pip install -q torch transformers && git clone YOUR-REPO && \
cd TokenSkip && python split_dataset.py --full && \
bash run_full_evaluation.sh
```

### Option 2: Guided Setup (Recommended)
```bash
# Copy-paste the full runpod_copy_paste.sh script
```

### Option 3: Manual Setup (For Understanding)
Follow detailed steps in **RUNPOD_SETUP.md**

---

## 💰 Cost Breakdown

**Full 8-hour pipeline:**
- **A100 (40GB)** ← BEST SPEED/COST: $0.44/hr = **$3.50**
- **RTX 4090 (24GB)** ← CHEAPEST: $0.28/hr = **$2.80** (slower, ~10h)
- **H100 (80GB)** ← FASTEST: $0.79/hr = **$6.30** (overkill)

**Stopped pod cost**: **$0/hr** (stop when done!)

---

## 🎯 Timeline

```
T+0:00   Pod created
         ↓
T+0:02   SSH connected
         ↓
T+0:05   Setup script finishes (environments, deps, data)
         ↓
T+0:30   Phase 1 training starts (CODI base model) — 2-4 hours
         ↓
T+2:30   Phase 2 extraction (truth vector) — 15 minutes
         ↓
T+2:45   Phase 3 CODI steering sweep (10 alphas) — 1 hour
         ↓
T+3:45   HF model baselines (no_cot, text_cot) — 1 hour
         ↓
T+4:45   HF model steering (ccot, random, steered) — 2-3 hours
         ↓
T+7:45   Aggregation & token metrics — 5 minutes
         ↓
T+8:00   ✓ COMPLETE — Download results
         ↓
T+8:01   Stop pod (save money)
```

---

## 🔧 Running Stages Separately (If You Want)

```bash
# Test just CODI (fastest, ~2 hours):
python phase1_train.py --num_epochs 1 --batch_size 4
python phase2_extract_vector.py --n-samples 5
python phase3_steer_inference.py --alphas 0 0.5 1.0

# Test just HF baselines (fastest, ~30 min):
python evaluation.py --model-type phi2 --no-cot --eval_batch_size 8
python evaluation.py --model-type phi2 --eval_batch_size 8

# Full evaluation:
bash run_full_evaluation.sh
```

---

## 📊 GPU Comparison for RunPod

| GPU | VRAM | Cost/hr | Speed | Best For |
|-----|------|---------|-------|----------|
| **RTX 4090** | 24GB | $0.28 | Slow (10h) | Budget-conscious |
| **A100** | 40GB | $0.44 | Medium (8h) | **RECOMMENDED** |
| **H100** | 80GB | $0.79 | Fast (6h) | Time-critical (expensive) |

---

## 🎮 Running in Background (Recommended)

When SSH session might disconnect:

```bash
# Method 1: Use screen
screen -S eval
bash run_full_evaluation.sh
# Press Ctrl+A, then D to detach
# Later: screen -r eval

# Method 2: Use nohup
nohup bash run_full_evaluation.sh > eval.log 2>&1 &

# Monitor in new terminal
tail -f eval.log
```

---

## 📥 Downloading Results

### Method 1: RunPod Web UI (Easiest)
1. Pod Dashboard → **Files**
2. Navigate to `/root/TokenSkip/logs` and `/root/TokenSkip/outputs`
3. Download CSV/JSON files

### Method 2: SCP from Local Machine
```bash
# Download logs
scp -r root@YOUR-POD-ID.runpod.io:/root/TokenSkip/logs ~/results/

# Download specific files
scp root@YOUR-POD-ID.runpod.io:/root/TokenSkip/logs/*aggregated*.csv ~/Downloads/
scp root@YOUR-POD-ID.runpod.io:/root/TokenSkip/logs/*token_metrics.* ~/Downloads/
```

### Method 3: Compress and Download
```bash
# In pod:
tar -czf results.tar.gz logs/ outputs/ && ls -lh results.tar.gz

# Then download via Web UI or SCP
```

---

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| **GPU not detected** | `nvidia-smi` in SSH → Check container has CUDA |
| **OOM errors** | Reduce batch: `--batch_size 2` or `--eval_batch_size 4` |
| **Models downloading slowly** | Normal first time (~30 min), cached after |
| **Pod disconnects** | Use `screen` or `nohup` (see above) |
| **Disk full** | Check: `df -h` and `du -sh *` |
| **Out of quota** | Increase disk in pod settings or reduce dataset |

---

## 🎓 What Gets Generated

After evaluation completes, you'll have:

```
logs/
├── evaluation_master.log              (main log)
├── YYYYMMDD_HHMMSS_*.log             (stage logs)
├── YYYYMMDD_HHMMSS_aggregated_results.csv
└── YYYYMMDD_HHMMSS_token_metrics.csv

outputs/
├── phase1_checkpoint/                (CODI weights)
├── phase2_truth_vector/              (v_truth vectors)
├── phase3_results/                   (CODI steering)
└── eval_grid/
    ├── phi2/
    ├── llama32_3b/
    ├── qwen25_0_5b/
    ├── qwen25_1_5b/
    ├── qwen25_3b/
    └── qwen_math_1_5b/
```

---

## 🎬 Example: Step-by-Step Session

```bash
# 1. Create Pod at runpod.io (A100 40GB, 20GB disk)
# 2. SSH into pod

$ ssh root@abc123.runpod.io -p 12345

# 3. Run setup
$ cd /root && bash runpod_copy_paste.sh

# (Wait 15 minutes for setup...)

# 4. Start evaluation
$ cd TokenSkip
$ bash run_full_evaluation.sh
    [Stage 1] Splitting dataset...
    [Stage 2] Phase 1 training (CODI)... ████████░░ 60%
    [Stage 3] Phase 2 extraction... ✓
    [Stage 4] Phase 3 steering... ████████████ 100%
    [Stage 5-6] HF baselines & steering... ████████░░ 60%
    [Stage 7] Aggregation... ✓
    [Stage 8] Token metrics... ✓
    
    ✓ Complete! Results in logs/ and outputs/

# 5. Download results
$ (In RunPod web UI) Download logs/ and outputs/

# 6. Stop pod
$ (Click Stop in RunPod dashboard)

# Cost: $3.50 ✓
```

---

## 🚀 Pro Tips

1. **Pre-cache models while training**: In a separate SSH session, run `python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')"` before Phase 1 finishes

2. **Parallelize with multiple pods**: 
   - Pod 1: CODI phases
   - Pod 2: Phi-2 + Llama
   - Pod 3: Qwen models
   - Run simultaneously, then merge results

3. **Save money**: Use **Spot GPU** (50% cheaper but can pause) if evaluation is not time-critical

4. **Monitor from anywhere**: `ssh` into pod and run `tail -f logs/eval.log`

---

## 📖 Full Documentation

- **Quick overview**: `RUNPOD_QUICKSTART.md`
- **Detailed guide**: `RUNPOD_SETUP.md`
- **All commands**: `COMMANDS_REFERENCE.md`
- **Evaluation protocol**: `EVALUATION_GUIDE.md`
- **Framework details**: `framework.md`

---

## ✅ Checklist

Before running:
- [ ] RunPod account created
- [ ] GPU pod deployed (A100 recommended)
- [ ] SSH connected to pod
- [ ] Python environment created
- [ ] Dependencies installed
- [ ] Dataset downloaded
- [ ] Code cloned/uploaded

During run:
- [ ] Monitoring logs with `tail -f`
- [ ] Checking GPU with `nvidia-smi`
- [ ] Monitoring disk space with `df -h`

After run:
- [ ] Results downloaded from pod
- [ ] Pod stopped (to save money)
- [ ] Results analyzed locally

---

**Total effort**: ~10-15 minutes (mostly waiting)  
**Total cost**: ~$3-5 for full evaluation  
**Total time**: ~8-10 hours  
**Result quality**: Production-ready metrics across 7 models ✅

**Questions?** Refer to the documentation files or RunPod support: https://docs.runpod.io
