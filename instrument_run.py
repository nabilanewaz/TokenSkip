# instrument_run.py
import os
import json
import argparse
import pandas as pd

def process_predictions_jsonl(predictions_path):
    recs = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # Expect keys: "id", "model_output", "prediction", "accuracy", maybe "cot_length"
            eid = rec.get("id", rec.get("example_id"))
            output = rec.get("model_output", "")
            pred = rec.get("prediction", None)
            correct = bool(rec.get("accuracy", False))
            cot_len = rec.get("cot_length", None)
            # If cot_length missing, approximate by token count of output
            if cot_len is None:
                cot_len = len(output.split())
            truncated = False
            # heuristic: if output ends with "…" or answer missing, mark truncated
            if not output.strip().endswith("}") and not ("\\boxed" in output):
                truncated = True
            recs.append({
                "example_id": eid,
                "cot_length": cot_len,
                "correct": correct,
                "prediction": pred,
                "truncated": truncated
            })
    return pd.DataFrame(recs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to samples directory of a run")
    parser.add_argument("--out-csv", required=True, help="Path to output CSV")
    args = parser.parse_args()

    # Find predictions.jsonl (or .json)
    preds_file = os.path.join(args.run_dir, "predictions.jsonl")
    if not os.path.isfile(preds_file):
        preds_file = os.path.join(args.run_dir, "predictions.json")
    if not os.path.isfile(preds_file):
        raise FileNotFoundError(f"No predictions file found in {args.run_dir}")

    df = process_predictions_jsonl(preds_file)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote instrumentation CSV to {args.out_csv}")

if __name__ == "__main__":
    main()
