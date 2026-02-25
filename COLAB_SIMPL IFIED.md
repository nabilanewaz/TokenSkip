# Simplified Colab Training (No CODI Repo Needed!)

Everything you need is bundled in `codi_bundle/`. Just use it directly!

## Quick Start

```python
# 1. Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 2. Navigate to bundle
%cd codi_bundle

# 3. Train!
!python train.py \
  --model_name_or_path gpt2 \
  --data_name "icot" \
  --batch_size 4 \
  --num_train_epochs 3 \
  --learning_rate 0.0002 \
  --output_dir ../outputs/codi_trained
```

That's it! The bundle includes:
- ✅ train.py (patched for GPU/CPU)
- ✅ src/model.py (CODI implementation)  
- ✅ All dependencies handled automatically

No cloning, no patching, just works!
