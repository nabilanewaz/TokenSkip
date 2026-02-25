"""
Prepare minimal package for Google Colab training
Copies only essential files to reduce upload size
"""
import shutil
import pathlib

print("Preparing Colab training package...")

# Create temporary directory
colab_dir = pathlib.Path("colab_package")
colab_dir.mkdir(exist_ok=True)

# Essential files
files_to_copy = [
    "train_codi.py",
    "patch_codi_model_cpu.py",
    "datasets/gsm8k_split/llm_train.jsonl",
    "datasets/gsm8k_split/validation.jsonl",
    "datasets/gsm8k_split/split_config.json",
    "colab_training.ipynb",
]

for file_path in files_to_copy:
    src = pathlib.Path(file_path)
    if src.exists():
        dst = colab_dir / file_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        size = dst.stat().st_size / 1024
        print(f"  ✓ {file_path} ({size:.1f} KB)")
    else:
        print(f"  ⚠ {file_path} not found, skipping")

# Create README
readme = colab_dir / "README.md"
readme.write_text("""# TokenSkip - Colab Training Package

## Upload to Google Drive

1. Zip this folder: `Compress-Archive -Path colab_package -DestinationPath colab_package.zip`
2. Upload `colab_package.zip` to Google Drive
3. Open `colab_training.ipynb` in Google Colab
4. Follow the notebook instructions

## Files Included

- `train_codi.py` - Main training script
- `datasets/gsm8k_split/*.jsonl` - Training/validation data (~3 MB total)
- `colab_training.ipynb` - Ready-to-run notebook

## What's NOT included (will be downloaded by Colab)

- CODI repository (cloned from GitHub)
- Pretrained GPT-2 checkpoint (downloaded from HuggingFace)
- Python dependencies (installed via pip)

Total package size: ~3-4 MB
""", encoding="utf-8")

print(f"\n✓ Package ready in: {colab_dir.absolute()}")
print(f"\nNext steps:")
print(f"  1. Compress: Compress-Archive -Path {colab_dir} -DestinationPath colab_package.zip")
print(f"  2. Upload colab_package.zip to Google Drive")
print(f"  3. Open colab_training.ipynb in Colab")
print(f"  4. Run all cells!")
