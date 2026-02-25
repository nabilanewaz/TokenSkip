"""
Patch CODI src/model.py to use float32 on CPU
"""
import pathlib
import re

codi_model_py = pathlib.Path("codi_workspace/CODI/src/model.py")

if not codi_model_py.exists():
    print(f"File not found: {codi_model_py}")
    exit(1)

code = codi_model_py.read_text(encoding="utf-8")

# Replace torch_dtype logic to use float32 on CPU
pattern = r'torch_dtype=\(\s*torch\.float16 if training_args\.bf16 is False else torch\.bfloat16\s*\)'
replacement = '''torch_dtype=(
                        torch.float32 if not torch.cuda.is_available() else
                        (torch.float16 if training_args.bf16 is False else torch.bfloat16)
                    )'''

code = re.sub(pattern, replacement, code)

# Also ensure checkpoint loading respects dtype  
# Find the load_state_dict call and add map_location parameter
if 'map_location=' not in code:
    old_load = 'state_dict = load_file(self.training_args.restore_from)'
    new_load = '''state_dict = load_file(self.training_args.restore_from)
            # Convert to float32 on CPU
            if not torch.cuda.is_available():
                state_dict = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}'''
    code = code.replace(old_load, new_load)

codi_model_py.write_text(code, encoding="utf-8")
print(f"✓ Patched {codi_model_py} for CPU float32 dtype")
