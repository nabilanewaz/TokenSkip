# patch_enforce_compression_and_fix_indent.py
"""
Merges the behavior of:
- patch_enforce_compression.py  (enforce shorter generations when compression_ratio < 1.0 and !use_adapter)
- patch_fix_indent.py           (safely insert greedy-decoding cleanup block with correct indentation)

Usage:
    python patch_enforce_compression_and_fix_indent.py

Notes:
- Creates a backup evaluation.py.bak_cot_all if not already present.
- Idempotent: safe to run multiple times.
"""
from pathlib import Path
import re

p = Path("evaluation.py")
assert p.exists(), "evaluation.py not found next to this script."

src = p.read_text(encoding="utf-8")
bk = Path("evaluation.py.bak_cot_all")
if not bk.exists():
    bk.write_text(src, encoding="utf-8")

s = src
changed = False

def insert_after(pattern, block, flags=re.M):
    global s, changed
    m = re.search(pattern, s, flags)
    if not m:
        return False
    pos = s.find("\n", m.end()) + 1
    s2 = s[:pos] + block + s[pos:]
    if s2 != s:
        s = s2
        changed = True
        return True
    return False

# --------------------------------------------------------------------
# 1) Enforce shorter generations when compression_ratio < 1.0 and not using LoRA
#    Insert right after args parsing line.
# --------------------------------------------------------------------
enforce_block = (
    "\n"
    "    # --- enforce shorter generations for compressed runs (no-LoRA path) ---\n"
    "    if args.compression_ratio < 1.0 and not args.use_adapter:\n"
    "        args.max_new_tokens = max(32, int(args.max_new_tokens * args.compression_ratio))\n"
    "\n"
)
insert_after(r"args,\s*unparsed_args\s*=\s*parser\.parse_known_args\(\)", enforce_block)

# --------------------------------------------------------------------
# 2) Remove any previously injected greedy-decoding block to avoid duplicates
# --------------------------------------------------------------------
greedy_pat = re.compile(
    r"\n\s*# ---- Greedy decoding cleanup: remove sampling-only params so warnings disappear ----\n"
    r"\s*gc = model\.generation_config\n"
    r"\s*gc\.do_sample = False\n"
    r"\s*gc\.temperature = None\n"
    r"\s*gc\.top_k = None\n"
    r"\s*gc\.top_p = None\n",
    re.M,
)
s2 = greedy_pat.sub("\n", s)
if s2 != s:
    s = s2
    changed = True

# --------------------------------------------------------------------
# 3) Insert greedy-decoding cleanup after a safe point in the HF branch
#    Prefer: after adapter merge; else after AutoModelForCausalLM.from_pretrained(...)
# --------------------------------------------------------------------
greedy_block = (
    "\n"
    "        # ---- Greedy decoding cleanup: remove sampling-only params so warnings disappear ----\n"
    "        gc = model.generation_config\n"
    "        gc.do_sample = False\n"
    "        gc.temperature = None\n"
    "        gc.top_k = None\n"
    "        gc.top_p = None\n"
    "\n"
)

# Try after PEFT adapter merge
insert_after_pat = re.compile(
    r"(model\s*=\s*PeftModel\.from_pretrained\(.*?\)\s*\n\s*model\s*=\s*model\.merge_and_unload\(\)\s*\n)",
    re.S,
)

def insert_after_adapter(text):
    m = insert_after_pat.search(text)
    if m:
        i = text.find("\n", m.end()) + 1
        return text[:i] + greedy_block + text[i:], True
    # Fallback: after the HF model load
    load_pat = re.compile(r"(model\s*=\s*AutoModelForCausalLM\.from_pretrained\([^)]*\)\s*\n)", re.S)
    m2 = load_pat.search(text)
    if m2:
        i = text.find("\n", m2.end()) + 1
        return text[:i] + greedy_block + text[i:], True
    return text, False

s3, ok = insert_after_adapter(s)
if ok:
    s = s3
    changed = True

# --------------------------------------------------------------------
# 4) Write back if anything changed
# --------------------------------------------------------------------
if changed:
    p.write_text(s, encoding="utf-8")
    print("✅ evaluation.py patched: enforced compressed-CoT token budget and fixed greedy-decoding insertion.")
    print("Backup at evaluation.py.bak_cot_all")
else:
    print("ℹ️ No changes made; patterns not found or already patched.")
