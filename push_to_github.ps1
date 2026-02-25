#!/usr/bin/env pwsh
# Push Colab training setup to GitHub

Write-Host "🚀 Pushing Colab training setup to GitHub..." -ForegroundColor Cyan

# Check if we're in the right directory
if (!(Test-Path "train_codi.py")) {
    Write-Host "❌ Error: train_codi.py not found. Are you in the TokenSkip directory?" -ForegroundColor Red
    exit 1
}

# Check git status
Write-Host "`n📋 Current status:" -ForegroundColor Yellow
git status --short

# Add files
Write-Host "`n➕ Adding Colab training files..." -ForegroundColor Yellow
git add train_codi.py
git add colab_training.ipynb
git add patch_codi_model_cpu.py
git add prepare_colab_package.py
git add COLAB_SETUP.md
git add QUICKSTART_COLAB.md
git add datasets/gsm8k_split/*.jsonl

# Show what will be committed
Write-Host "`n📝 Files to commit:" -ForegroundColor Yellow
git status --short

# Commit
Write-Host "`n💾 Committing changes..." -ForegroundColor Yellow
git commit -m "Add Colab training setup: train_codi.py + notebook + data splits" -m "- CODI training script with CPU/GPU compatibility
- Colab notebook for T4 GPU training (~2h vs 100h CPU)  
- GSM8K splits: 6000 train + 500 steer + 1500 val + 792 test
- Setup guides and utilities"

# Push
Write-Host "`n⬆️  Pushing to GitHub..." -ForegroundColor Yellow
git push origin main

Write-Host "`n✅ Done! Your repo is updated:" -ForegroundColor Green
Write-Host "   https://github.com/nabilanewaz/TokenSkip" -ForegroundColor Cyan

Write-Host "`n🎯 Next steps:" -ForegroundColor Yellow
Write-Host "   1. Open: https://colab.research.google.com/github/nabilanewaz/TokenSkip/blob/main/colab_training.ipynb"
Write-Host "   2. Set Runtime → T4 GPU"
Write-Host "   3. Run all cells!"
Write-Host "   4. Wait ~2 hours"
Write-Host "   5. Download checkpoint and continue to Phase 2" -ForegroundColor Green
