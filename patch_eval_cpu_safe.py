# patch_eval_cpu_safe.py
from pathlib import Path
import re

p = Path("evaluation.py")
assert p.exists(), "evaluation.py not found next to this script."

src = p.read_text(encoding="utf-8")
bk = Path("evaluation.py.bak_cpu")
if not bk.exists():
    bk.write_text(src, encoding="utf-8")

s = src
changed = False

def sub(pattern, repl, flags=re.M):
    global s, changed
    s2 = re.sub(pattern, repl, s, flags=flags)
    if s2 != s:
        s = s2
        changed = True

# --------------------------------------------------------------------
# 1) Remove any top-level vLLM imports (import vllm, LLM, SamplingParams, LoRARequest)
# --------------------------------------------------------------------
sub(r'^[ \t]*(?:from\s+vllm(?:\.[\w\.]+)?\s+import\s+.*|import\s+vllm(?:\.[\w\.]*)*)\r?\n', "")

# --------------------------------------------------------------------
# 2) Remove stray "\1" artifacts from earlier bad patches
# --------------------------------------------------------------------
if "\\1" in s:
    s = s.replace("\\1", "")
    changed = True

# --------------------------------------------------------------------
# 3) Guard all torch.cuda.synchronize() calls
# --------------------------------------------------------------------
s = s.replace("torch.cuda.synchronize()", "torch.cuda.synchronize() if torch.cuda.is_available() else None")

# --------------------------------------------------------------------
# 4) Force CPU path for common HF codepaths:
#    - float16 -> float32
#    - device_map=\"auto\" -> device_map=\"cpu\"
# --------------------------------------------------------------------
s = s.replace("torch_dtype=torch.float16", "torch_dtype=torch.float32")
s = s.replace('device_map="auto"', 'device_map="cpu"')

# --------------------------------------------------------------------
# 5) Inject lazy vLLM import inside the vLLM branch (once)
#    We look for a line that is exactly "if args.use_vllm:" (with optional spaces)
#    and insert a try/except lazy import block below it if not present.
# --------------------------------------------------------------------
if "Run with --use_vllm False for CPU-only." not in s:
    m = re.search(r'^\s*if\s+args\.use_vllm\s*:\s*$', s, flags=re.M)
    if m:
        insert_pos = s.find("\n", m.end()) + 1
        lazy_block = (
            "        try:\n"
            "            from vllm import LLM, SamplingParams\n"
            "            from vllm.lora.request import LoRARequest\n"
            "        except ImportError as e:\n"
            '            raise RuntimeError("vLLM is not installed. Run with --use_vllm False for CPU-only.") from e\n'
        )
        s = s[:insert_pos] + lazy_block + s[insert_pos:]
        changed = True

# --------------------------------------------------------------------
# 6) Make tensor_parallel_size robust if code directly uses CUDA_VISIBLE_DEVICES
#    Replace common pattern: tensor_parallel_size=len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))
#    with a safe tp_size block.
# --------------------------------------------------------------------
tp_pattern = r"tensor_parallel_size\s*=\s*len\(\s*os\.environ\[['\"]CUDA_VISIBLE_DEVICES['\"]\]\.split\(\s*['\"]\s*,\s*['\"]\s*\)\s*\)"
if re.search(tp_pattern, s):
    s = re.sub(
        tp_pattern,
        "tensor_parallel_size=(lambda: (len([d for d in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if d.strip()])\n"
        "    if os.environ.get('CUDA_VISIBLE_DEVICES','').strip() else (torch.cuda.device_count() if torch.cuda.is_available() else 1)))()",
        s
    )
    changed = True

# --------------------------------------------------------------------
# 7) Prevent division by zero for sample_latency (total_time / len(test_data))
# --------------------------------------------------------------------
sub(r"sample_latency\"\s*:\s*total_time\s*/\s*len\(test_data\)",
    'sample_latency": (total_time / len(test_data) if len(test_data) else None)')

# --------------------------------------------------------------------
# 8) Optional: if code computes accuracy/avg_cot_length without guarding n==0
# --------------------------------------------------------------------
sub(r'"accuracy"\s*:\s*sum\(item\[\'accuracy\'\]\s*for\s*item\s*in\s*results\)\s*/\s*len\(results\)',
    '"accuracy": (sum(item[\'accuracy\'] for item in results) / len(results) if len(results) else 0.0)')
sub(r'"avg_cot_length"\s*:\s*avg_cot_length',
    '"avg_cot_length": (avg_cot_length if len(results) else 0.0)')

if changed:
    p.write_text(s, encoding="utf-8")
    print("✅ evaluation.py patched for CPU-only/vLLM-optional runs.")
    print("Backup at evaluation.py.bak_cpu")
else:
    print("ℹ️ evaluation.py already looks CPU-safe; no changes made.")
