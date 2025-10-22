# patch_enforce_compression.py
"""
Adds:
- Enforced shorter generations when compression_ratio < 1.0 (no-LoRA path)
- Greedy-decoding cleanup in the HF branch
- Removes compression_ratio tokens from prompt templates (for fair CoT compression)

Usage:
    python patch_enforce_compression_and_fix_indent.py
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
# 1) Enforce shorter generations (only scaling max_new_tokens)
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
# 2) Remove compression_ratio tokens from any prompt template strings
# --------------------------------------------------------------------
ratio_injections = [
    r"\{args\.compression_ratio\}",
    r"<\|eot_id\|>\s*\{args\.compression_ratio\}\s*<\|eot_id\|>",
    r"\{tokenizer\.eos_token\}\s*\{args\.compression_ratio\}\s*\{tokenizer\.eos_token\}",
]
for pat in ratio_injections:
    new_s = re.sub(pat, "", s)
    if new_s != s:
        s = new_s
        changed = True

# --------------------------------------------------------------------
# 3) Remove any existing greedy-decoding block to avoid duplicates
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
# 4) Insert greedy-decoding cleanup after adapter merge or HF model load
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

def insert_after_adapter(text):
    insert_after_pat = re.compile(
        r"(model\s*=\s*PeftModel\.from_pretrained\(.*?\)\s*\n\s*model\s*=\s*model\.merge_and_unload\(\)\s*\n)",
        re.S,
    )
    m = insert_after_pat.search(text)
    if m:
        i = text.find("\n", m.end()) + 1
        return text[:i] + greedy_block + text[i:], True
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
# 5) Write back
# --------------------------------------------------------------------
if changed:
    p.write_text(s, encoding="utf-8")
    print("✅ evaluation.py patched: enforced compression budget, fixed greedy-decoding, and removed ratio tokens from prompts.")
    print("Backup at evaluation.py.bak_cot_all")
else:
    print("ℹ️ No changes made; patterns not found or already patched.")
