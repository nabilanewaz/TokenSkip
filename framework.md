
# Research Protocol: Steering Continuous Reasoning via Latent Intervention

## Abstract & Hypothesis
**Objective**: To investigate whether reasoning in a continuous latent space (as seen in Tokenskip architecture) can be "steered" toward factuality using a linear intervention vector derived from the model's own activations.

**Hypothesis**: Truthfulness and logical validity are encoded as linear directions in the high-dimensional latent space of the model.
Injecting this "Truth Vector" ($\mathbf{v}_{truth}$) during the continuous reasoning phase will reduce "reasoning drift" (hallucination) and improve final answer accuracy.

## Dataset Preparation (GSM8K)
Strict data isolation is required to prevent data leakage.  The GSM8K dataset provides a **fixed test set of 1,319 examples** (never touched during training or vector extraction) and a **train pool of 7,473 examples** that is split 60/10/30 into three subsets:

### The Splits
| Subset Name | Variable | Size | Purpose |
| --- | --- | --- | --- |
| **Base Training** | $\mathcal{D}_{train}$ | 4,483 (60 %) | Train the base CCoT / LLM to reason. |
| **Vector Extraction** | $\mathcal{D}_{steer}$ | 747 (10 %) | Compute the truth vector $\mathbf{v}_{truth}$. |
| **Validation** | $\mathcal{D}_{val}$ | 2,243 (30 %) | Tune steering strength $\alpha$ without touching test. |
| **Test** | $\mathcal{D}_{test}$ | 1,319 (separate) | Final held-out evaluation. |

**Total train pool**: 4,483 + 747 + 2,243 = 7,473 (GSM8K `train.jsonl`).

Critical Rule: $\mathcal{D}_{test}$ comes from the original GSM8K `test.jsonl` and is **never** used in Phase 1 or 2.  $\mathcal{D}_{steer}$ must not overlap with $\mathcal{D}_{train}$.

## Models Under Investigation

| Tag | Model | Backbone | Steering Path |
| --- | --- | --- | --- |
| `codi_gpt2` | CODI-GPT2 ([zen-E/CODI-gpt2](https://huggingface.co/zen-E/CODI-gpt2)) | GPT-2 | Patched `test.py` (steer_inference.py) |
| `phi2` | microsoft/phi-2 | Phi-2 (2.7B) | Hook-based residual injection (hidden_steer.py) |
| `llama32_3b` | meta-llama/Llama-3.2-3B | Llama 3.2 (3B) | Hook-based residual injection |
| `qwen25_3b` | Qwen/Qwen2.5-3B | Qwen 2.5 (3B) | Hook-based residual injection |
| `qwen25_0.5b` | Qwen/Qwen2.5-0.5B | Qwen 2.5 (0.5B) | Hook-based residual injection |
| `qwen25_1.5b` | Qwen/Qwen2.5-1.5B | Qwen 2.5 (1.5B) | Hook-based residual injection |

## Model Architecture
We utilize a Continuous Chain-of-Thought (CCoT) architecture
- **Input**: Text tokens $x$.
- **Reasoning Phase**: A sequence of $k$ continuous hidden states $h_1, h_2, \dots, h_k$ where $h_t \in \mathbb{R}^d$. These are not decoded into text during training.
- **Output**: Final answer tokens $y$.
- **Mechanism**: The model operates in "Latent Mode," feeding the hidden state $h_t$ back into the input for step $t+1$ without passing through a discrete softmax layer.

## Phase 1: Base Model Training
- Goal: Create a model capable of continuous reasoning, even if imperfect.
- Format: Convert GSM8K examples into the format: Question -> [RAT_START] -> (Reasoning Steps) -> [RAT_END] -> Answer.
- Curriculum Learning:
    - Stage A: Train on standard text CoT (discrete tokens) to ground the reasoning.
    - Stage B: Gradually replace discrete reasoning tokens with continuous vectors (as per Coconut paper).
- Loss Function: Standard Next-Token Prediction (Cross-Entropy) on the final answer only. The latent steps are optimized end-to-end via backpropagation.
- Stopping Criterion: Stop when accuracy on a held-out validation set plateaus. Save Checkpoint.

## Phase 2: Extracting the Truth Vector ($\mathbf{v}_{truth}$)
- Goal: Identify the direction of "correct reasoning" in the latent space using $\mathcal{D}_{steer}$.
- Data Collection:
    - For each question $x_i$ in $\mathcal{D}_{steer}$: Run the frozen model $N$ times with temperature $T=1.0$.
    - Classify the traces:
        - Positive Set ($H^+$): Latent trajectories where the final answer $y$ matches the ground truth.
        - Negative Set ($H^-$): Latent trajectories where the final answer $y$ is incorrect.
    - Store the continuous vectors $h_{i,t}$ for each step $t$.
- Vector Calculation (Difference-of-Means): Compute the "Truth Direction" by subtracting the mean "Wrong" state from the mean "Right" state:
$$\mathbf{v}_{truth} = \frac{1}{|H^+|} \sum_{h \in H^+} h - \frac{1}{|H^-|} \sum_{h \in H^-} h$$

Note: Perform this calculation separately for each reasoning depth $t$ (yielding $\mathbf{v}_{truth}^t$) or average across all steps for a single global vector.

## Phase 3: Inference-Time Intervention (The Experiment)
Goal: Test if injecting $\mathbf{v}_{truth}$ improves performance on $\mathcal{D}_{test}$.

### Evaluation Conditions

| Condition | Description | $\alpha$ | Vector |
| --- | --- | --- | --- |
| **No CoT** | Model answers directly — no reasoning at all. | N/A | N/A |
| **Text CoT** | Standard text chain-of-thought (discrete tokens). | N/A | N/A |
| **CCoT (unsteered)** | Continuous CoT, no intervention. | 0.0 | — |
| **Random Noise** | Continuous CoT + random vector injection (control). | 1.0 | $\mathcal{N}(0,1)$ |
| **CCoT + $\mathbf{v}_{truth}$** | Continuous CoT with truth-vector steering. | swept | $\mathbf{v}_{truth}$ |

### The Steering Equation
During the continuous reasoning loop on the Evaluation Set:
$$h_{t+1} = \text{Model}(h_t) + \alpha \cdot \sigma_{l} \cdot \frac{\mathbf{v}_{truth}}{|\mathbf{v}_{truth}|}$$

$\alpha$ (Steering Strength): A scalar hyperparameter controlling the intervention intensity. $\sigma_l$ is the standard deviation of the activations at layer $l$ of the model.

### Hyperparameter Sweep
Run the evaluation loop for different values of $\alpha$:
- $\alpha \in \{0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0\}$
- $\alpha = 0.0$ is the Baseline (Control Group).

## Phase 4: Evaluation Metrics
We assess performance using three distinct lenses.

| Metric | Definition |
| --- | --- |
| **Accuracy** | Fraction of $\mathcal{D}_{test}$ answered correctly. |
| **Flip Rate** | % of CCoT-wrong examples that become correct after steering ($\Delta = $ steered − unsteered). |
| **Cosine Similarity** | $\cos(h_t, \mathbf{v}_{truth})$ averaged across all latent steps and examples — measures trajectory alignment with the truth direction. |
| **Token Count** | Average number of reasoning tokens used during CCoT. |
| **Trajectory Faithfulness** | Average cosine similarity between consecutive hidden states — measures the coherence of the reasoning trajectory. |
| **Latency** | Average inference time per example — measures the overhead of the continuous reasoning process. |

### Aggregation Pipeline
1. `evaluate_baselines.py` — orchestrates the full model × condition grid for HF models.
2. `steer_inference.py --random-noise` — runs the CODI-GPT2 pipeline including the random-noise control.
3. `compare_all.py` — reads all `metrics.json` outputs and prints/exports the final tables.

