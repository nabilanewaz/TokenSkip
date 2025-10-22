# validate_side_by_side.py
import os
import json
import random
import textwrap

# Adjust these paths exactly to match your folders
BASE_DIR     = r"outputs\Qwen2.5-0.5B-Instruct\full_cot\7b\Original\test\samples"
COMPRESS_DIR = r"outputs\Qwen2.5-0.5B-Instruct\compressed_cot\7b\Original\test\samples"
NUM_EXAMPLES = 10
WRAP_WIDTH   = 80

def load_runs(run_dir):
    data = {}
    if not os.path.isdir(run_dir):
        raise RuntimeError(f"Directory not found: {run_dir}")
    for fname in os.listdir(run_dir):
        if fname.endswith(".jsonl") or fname.endswith(".json"):
            path = os.path.join(run_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                # If .jsonl file: each line is a JSON object
                if fname.endswith(".jsonl"):
                    for line in f:
                        rec = json.loads(line)
                        eid = rec.get("id", rec.get("example_id", None))
                        if eid is None:
                            continue
                        data[eid] = rec
                else:
                    rec = json.load(f)
                    eid = rec.get("id", rec.get("example_id", None))
                    if eid is None:
                        eid = fname
                    data[eid] = rec
    return data

def print_side_by_side(eid, base_out, comp_out):
    print(f"\n=== Example ID: {eid} ===")
    print("- Baseline (full CoT) -------------------------------")
    print(textwrap.fill(base_out, WRAP_WIDTH))
    print("\n- Compressed CoT ------------------------------------")
    print(textwrap.fill(comp_out, WRAP_WIDTH))
    print("-----------------------------------------------------\n")

def main():
    base_data = load_runs(BASE_DIR)
    comp_data = load_runs(COMPRESS_DIR)

    common_ids = list(set(base_data.keys()) & set(comp_data.keys()))
    if len(common_ids) < NUM_EXAMPLES:
        raise RuntimeError(f"Not enough common examples: found {len(common_ids)}")

    sample_ids = random.sample(common_ids, NUM_EXAMPLES)
    sample_ids.sort()

    for eid in sample_ids:
        base_rec = base_data[eid]
        comp_rec = comp_data[eid]
        base_out = base_rec.get("model_output", base_rec.get("generated_text", "")) 
        comp_out = comp_rec.get("model_output", comp_rec.get("generated_text", ""))
        print_side_by_side(eid, base_out, comp_out)

if __name__ == "__main__":
    main()
