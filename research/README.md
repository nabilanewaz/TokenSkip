# Steering Compressed Chain-of-Thought (TokenSkip) via Latent Intervention

**Thesis Research Framework**

> **Hypothesis**: By compressing the discrete Chain-of-Thought using TokenSkip (LLMLingua-2), we can establish a tighter "Compressed CoT" (CCoT) latent space. Truthfulness and logical validity are encoded as linear directions in this latent space. Injecting a "Truth Vector" ($\mathbf{v}_{truth}$) at the compression boundary reduces hallucination and improves final answer accuracy while saving tokens.

---

## Repository Structure

```
research/
├── README.md                    ← Full command reference
├── requirements.txt             ← Pinned for RunPod CUDA 12.x
├── configs/protocol.yaml        ← Single source of truth for ALL hyperparameters
├── utils/
│   ├── io.py                    ← load_jsonl, save_jsonl, load_config
│   ├── answer.py                ← extract_answer_number, answers_match
│   ├── model_registry.py        ← Model prompt builders + HF IDs
│   └── tokenskip.py             ← LLMLingua-2 compression
├── data/split_dataset.py        ← Full 60/10/30 + mini-500 mode
├── phase1/train.py              ← Train HF models on TokenSkip CCoT
├── phase2/extract_vector.py     ← Difference-of-Means (v_truth, σ)
├── phase3/steer.py              ← Master inference script (TokenSkip + v_truth steering)
├── eval/
│   ├── evaluate_baselines.py    ← Aggregation Point 1 (runs 5 conditions across models)
│   └── compare_all.py           ← Aggregation Point 3 (tables, CSV, LaTeX)
└── run_pipeline.py              ← One-command orchestrator
```

---

## Dataset Splits (GSM8K)

| Split | Variable | N | Purpose |
|---|---|---|---|
| `steer_train.jsonl` | $\mathcal{D}_{steer}$ | 747 | Phase 1 vector extraction **only** |
| `validation.jsonl` | $\mathcal{D}_{val}$ | 2,243 | Alpha tuning |
| `test.jsonl` | $\mathcal{D}_{test}$ | 1,319 | Final held-out evaluation |

**Critical rule**: `test.jsonl` is from `gsm8k/test.jsonl` and is **never** used in Phase 1.

---

## Models

| Tag | HF ID | Base or Instruct |
|---|---|---|
| `phi2` | `microsoft/phi-2` | Instruct |
| `llama32_3b` | `meta-llama/Llama-3.2-3B` | Base |
| `qwen25_3b` | `Qwen/Qwen2.5-3B` | Base |
| `qwen25_1_5b` | `Qwen/Qwen2.5-1.5B` | Base |
| `qwen25_0_5b` | `Qwen/Qwen2.5-0.5B` | Base |

---

## Pipeline Execution

We use a modular, 4-phase architecture orchestrable via `run_pipeline.py`.

### The 1-Click Run
Run the entire pipeline (data $\to$ vector $\to$ steer $\to$ compare) on Phi-2:
```bash
python research/run_pipeline.py --model phi2
```

### Smoke Testing
Run the full 4 phases locally on the `mini-500` dataset (fast execution):
```bash
python research/run_pipeline.py --model phi2 --mini
```

---

## Phase Definitions

### Phase 0: Data
Shuffles and splits GSM8K deterministically.
```bash
python research/data/split_dataset.py --full
```

### Phase 1: Training
Fine-tunes the base HuggingFace models on TokenSkip compressed chain-of-thought using LoRA.
```bash
python research/phase1/train.py --model-type phi2 --model-path microsoft/phi-2 --ratio 0.8
```

### Phase 2: Vector Extraction
Generates full Text CoT traces for $\mathcal{D}_{steer}$ using the fine-tuned LoRA model.
Computes difference of means: $\mathbf{v}_{truth} = \text{mean}(H^+) - \text{mean}(H^-)$ at the last token of the prompt.
```bash
python research/phase2/extract_vector.py --model-type phi2 --model-path microsoft/phi-2
```

### Phase 3: Inference & Steering
Evaluates all 5 conditions across target ratio and alpha sweeps. Loads the Phase 1 LoRA model automatically if found.
```bash
python research/eval/evaluate_baselines.py --models phi2
```

### Phase 4: Aggregation & Comparison
Consolidates metrics into final thesis tables (Accuracy, Compression Ratio, Alpha).
```bash
python research/eval/compare_all.py --csv results.csv --latex
```

---

## Evaluation Conditions

| Condition | Description | α | Vector |
|---|---|---|---|
| **No CoT** | Direct answer, no reasoning | N/A | N/A |
| **Text CoT** | Standard discrete chain-of-thought | N/A | N/A |
| **CCoT (TokenSkip)** | Compressed CoT, no intervention | 0.0 | — |
| **Random Noise** | CCoT + random unit vector (control) | 1.0 | $\mathcal{N}(0,1)$ |
| **CCoT + v_truth** | Compressed CoT + truth-vector steering | swept | $\mathbf{v}_{truth}$ |

**Alpha sweep**: `{0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0}` (9 values)
**TokenSkip compression ratio sweep**: `{0.5, 0.6, 0.7, 0.8, 0.9, 1.0}` (6 values)

**Steering equation**:
At the compression boundary token, before decoding the final answer:
$$h_t \leftarrow h_t + \alpha \cdot \sigma \cdot \mathbf{v}_{truth}$$

---

## Configuration
All hyperparameters are in `research/configs/protocol.yaml`. Edit that file to change anything — every script reads from it at startup.
