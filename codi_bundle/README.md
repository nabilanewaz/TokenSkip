# CODI Training - Standalone Version for Colab

This version doesn't require cloning the CODI repo. All necessary files are bundled.

## Files Included

- `codi_bundle/train.py` - Patched training script (GPU/CPU compatible)  
- `codi_bundle/src/model.py` - CODI model implementation
- Works standalone with just these files + your data

## Usage in Colab

The updated `colab_training.ipynb` uses this bundle automatically - no CODI repo cloning needed!

## What Changed

**Before**: Clone CODI → patch train.py → hope it has all files
**After**: Use pre-patched bundle → just works ✅

## Local Testing

```powershell
# Navigate to bundle
cd codi_bundle

# Train (same arguments as before)
python train.py --model_name_or_path gpt2 --data_name icot ...
```
