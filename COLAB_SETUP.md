# Running CODI Training on Google Colab (Free T4 GPU)

## Step 1: Upload Your Code to GitHub

```powershell
# Initialize git repo (if not already done)
cd G:\Thesis\TokenSkip
git init
git add train_codi.py datasets/gsm8k_split/*.jsonl patch_codi_model_cpu.py
git commit -m "CODI training setup"

# Create repo on GitHub and push
gh repo create TokenSkip --private --source=. --remote=origin --push
```

## Step 2: Open Google Colab

1. Go to https://colab.research.google.com/
2. Select **Runtime → Change runtime type → T4 GPU** (free tier)
3. Create a new notebook

## Step 3: Setup Environment (Run in Colab cells)

```python
# Cell 1: Clone your repository
!git clone https://github.com/YOUR_USERNAME/TokenSkip.git
%cd TokenSkip

# Cell 2: Install dependencies
!pip install peft==0.15.2 datasets==3.6.0 huggingface_hub transformers==4.52.4 accelerate==1.7.0 -q

# Cell 3: Apply CPU patches (not needed on GPU, but harmless)
!python patch_codi_model_cpu.py

# Cell 4: Verify GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Step 4: Run Training

```python
# Cell 5: Start training (will take ~2-3 hours on T4)
!python train_codi.py --num_epochs 3 --batch_size 4 --bf16
```

**Note**: Increase `--batch_size` to 4 or 8 on GPU for faster training. Remove `--bf16` flag if you get errors (T4 supports it though).

## Step 5: Monitor Progress

```python
# Cell 6: Check training progress (run this repeatedly)
!tail -n 30 outputs/codi_finetuned/train_log.txt
```

## Step 6: Download Results

```python
# Cell 7: After training completes, zip and download the checkpoint
!zip -r codi_checkpoint.zip outputs/codi_finetuned/
from google.colab import files
files.download('codi_checkpoint.zip')
```

## Alternative: Google Drive Integration

If you have large data or want to avoid losing progress:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy data from Drive
!cp -r /content/drive/MyDrive/TokenSkip/datasets .

# Save checkpoints to Drive
!python train_codi.py --output-dir /content/drive/MyDrive/TokenSkip/codi_output
```

## Troubleshooting

**Out of Memory?**
- Reduce batch size: `--batch_size 2`
- Shorter sequences work better on free tier

**Session Timeout?**
- Free Colab disconnects after 12h idle or 24h active
- T4 training should finish in 2-3 hours, so you're safe
- Consider saving checkpoints periodically if you modify the code

**Can't upload large data?**
- Use Google Drive mount (shown above)
- Or upload just the split files (they're small JSONL)

## Expected Timeline on T4 GPU

- Data loading: ~30 seconds
- Training (3 epochs, 6000 examples): **~2 hours**
- Total: **~2 hours** vs 100+ hours on CPU

## After Training

Download the checkpoint and place it in your local workspace:
```powershell
# On your local machine
Expand-Archive codi_checkpoint.zip -DestinationPath G:\Thesis\TokenSkip\
```

Then proceed to Phase 2:
```powershell
python extract_truth_vector.py --steer-data datasets/gsm8k_split/steer_train.jsonl
```
