# compare_runs_report.py
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_tag(csv_path, tag):
    df = pd.read_csv(csv_path)
    df["run"] = tag
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-csv", required=True, help="CSV for full-CoT run")
    parser.add_argument("--comp-csv", required=True, help="CSV for compressed-CoT run")
    parser.add_argument("--output-prefix", required=True, help="Prefix for output files (can include path)")
    args = parser.parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_prefix + "_summary.csv")
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df_full = load_and_tag(args.full_csv, "full")
    df_comp = load_and_tag(args.comp_csv, "compressed")
    df = pd.concat([df_full, df_comp], ignore_index=True)

    # Summary table
    summary = df.groupby("run").agg(
        n_examples = ("example_id", "count"),
        accuracy   = ("correct", "mean"),
        avg_cot_length = ("cot_length", "mean"),
        pct_truncated  = ("truncated", "mean")
    ).reset_index()
    print("Summary:")
    print(summary)

    summary.to_csv(f"{args.output_prefix}_summary.csv", index=False)

    # Histogram of CoT lengths by run
    plt.figure(figsize=(10,6))
    sns.histplot(data=df, x="cot_length", hue="run",
                 bins=30, kde=False, element="step", stat="count")
    plt.title("Distribution of CoT Lengths: Full vs Compressed")
    plt.xlabel("CoT length (tokens)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_cot_length_hist.png")
    plt.close()

    # Boxplot: CoT length vs correctness
    plt.figure(figsize=(8,6))
    sns.boxplot(data=df, x="run", y="cot_length", hue="correct")
    plt.title("CoT Length vs Correctness by Run")
    plt.xlabel("Run type")
    plt.ylabel("CoT length (tokens)")
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_cot_length_vs_correctness.png")
    plt.close()

    print(f"📄 Plots and summary saved with prefix: {args.output_prefix}")

if __name__ == "__main__":
    main()
