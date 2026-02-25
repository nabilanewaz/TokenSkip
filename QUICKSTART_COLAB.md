# Quick Start: Running on Google Colab GPU

## Step 1: Push Your Code to GitHub (1 minute)

```powershell
# Add the new files
git add train_codi.py colab_training.ipynb patch_codi_model_cpu.py
git add datasets/gsm8k_split/*.jsonl
git add COLAB_SETUP.md prepare_colab_package.py

# Commit and push
git commit -m "Add Colab training setup for Phase 1"
git push origin main
```

## Step 2: Open Google Colab (2 minutes)

1. Go to: https://colab.research.google.com/
2. File → Open notebook → GitHub
3. Enter: `nabilanewaz/TokenSkip`
4. Select: `colab_training.ipynb`

**OR** just click this direct link:
👉 https://colab.research.google.com/github/nabilanewaz/TokenSkip/blob/main/colab_training.ipynb

## Step 3: Enable GPU (30 seconds)

In Colab:
- Runtime → Change runtime type → **T4 GPU** → Save

## Step 4: Run Training (2 hours)

Just click **Runtime → Run all** and wait!

The notebook will:
1. ✅ Verify GPU (should show T4 with ~15GB memory)
2. ✅ Clone your repo
3. ✅ Install dependencies  
4. ✅ Load your 6,000 training examples
5. ✅ Train CODI for 3 epochs (~2 hours)
6. ✅ Download the checkpoint

## Monitoring Progress

While training, you can:
- Watch the progress bar update
- Run the "Monitor training" cell repeatedly
- See loss decreasing (indicates learning)

## After Training

The last cell will download `codi_checkpoint.zip` automatically. Then locally:

```powershell
# Extract the checkpoint
Expand-Archive codi_checkpoint.zip -DestinationPath G:\Thesis\TokenSkip\

# Verify it worked
ls outputs/codi_finetuned/

# Ready for Phase 2!
python extract_truth_vector.py --steer-data datasets/gsm8k_split/steer_train.jsonl
```

## Troubleshooting

**"No accelerator found"?**
→ Go back to Runtime → Change runtime type → Select T4 GPU

**"Repository not found"?**  
→ Make sure you pushed: `git push origin main`

**Session disconnected?**
→ Colab free tier has 12h runtime limit. Your training finishes in 2h, so you're safe.

**Out of memory?**
→ The notebook uses batch_size=4. If OOM, change to batch_size=2 in the training cell.

---

## Timeline

- **CPU (local)**: 100+ hours ⏰
- **GPU (Colab T4)**: ~2 hours ⚡

You're 50x faster on the free tier GPU!
