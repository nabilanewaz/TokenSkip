import torch
import tqdm
from transformers import StoppingCriteria, GenerationConfig


class KeyWordsCriteria(StoppingCriteria):
    """
    Stops generation when any stop sequence appears at the end of the
    decoded output for every item in the batch.

    CPU fix: instead of re-decoding a sliding window on every token
    (O(n²)), we decode only the last `max_stop_len` newly-generated
    tokens — O(1) per step.
    """

    def __init__(self, stop_id_sequences, tokenizer, prompt_length):
        assert isinstance(stop_id_sequences[0], list), \
            "stop_id_sequences should be a list of list of ids"
        self.tokenizer        = tokenizer
        self.stop_id_sequences = stop_id_sequences
        self.stop_sequences   = [tokenizer.decode(s) for s in stop_id_sequences]
        self.prompt_length    = prompt_length
        # Longest stop sequence — we only need to inspect this many tokens
        self.max_stop_len     = max(len(s) for s in stop_id_sequences) + 4

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor, **kwargs) -> bool:
        all_stopped = []
        for i in range(input_ids.shape[0]):
            generated = input_ids[i][self.prompt_length:]
            # Only decode the tail — avoids O(n²) re-decoding on CPU
            tail_ids  = generated[-self.max_stop_len:].tolist()
            tail_text = self.tokenizer.decode(tail_ids)
            stopped   = any(tail_text.endswith(s) for s in self.stop_sequences)
            all_stopped.append(stopped)
        return all(all_stopped)


@torch.no_grad()
def generate_completions(model, tokenizer, prompts, batch_size=1,
                          stop_id_sequences=None,
                          end_of_generation_id_sequence=None,
                          disable_tqdm=False, **generation_kwargs):
    generations       = []
    finish_completion = []

    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    stop_sequences = (
        [tokenizer.decode(s) for s in stop_id_sequences]
        if stop_id_sequences else []
    )
    end_of_generation_sequence = (
        tokenizer.decode(end_of_generation_id_sequence)
        if end_of_generation_id_sequence else None
    )

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    generation_kwargs['use_cache'] = True

    for i in range(0, len(prompts), batch_size):
        batch_prompts      = prompts[i:i + batch_size]
        tokenized          = tokenizer(batch_prompts, padding="longest",
                                       return_tensors="pt", add_special_tokens=True)
        batch_input_ids    = tokenized.input_ids
        attention_mask     = tokenized.attention_mask

        # Move to model device (works for both CPU and CUDA)
        device = next(model.parameters()).device
        batch_input_ids = batch_input_ids.to(device)
        attention_mask  = attention_mask.to(device)

        batch_finish = [False] * len(batch_prompts) * num_return_sequences

        try:
            # Strip sampling-only kwargs when do_sample=False to silence HF warnings
            gen_kwargs = dict(generation_kwargs)
            if not gen_kwargs.get("do_sample", False):
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)
                gen_kwargs.pop("top_k", None)

            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                stopping_criteria=(
                    [KeyWordsCriteria(stop_id_sequences, tokenizer,
                                      batch_input_ids.size(1))]
                    if stop_id_sequences else None
                ),
                pad_token_id=tokenizer.eos_token_id,
                **gen_kwargs,
            )

            # Remove stop tokens that bled through (batch-level stopping)
            if stop_id_sequences:
                for out_idx in range(batch_outputs.shape[0]):
                    for tok_idx in range(batch_input_ids.shape[1],
                                        batch_outputs.shape[1]):
                        tail = tokenizer.decode(
                            batch_outputs[out_idx,
                                          tok_idx:tok_idx + max(len(s) for s in stop_id_sequences) + 4]
                        )
                        if any(tail.startswith(s) for s in stop_sequences):
                            if (end_of_generation_sequence and
                                    tail.startswith(end_of_generation_sequence)):
                                batch_finish[out_idx] = True
                            batch_outputs[out_idx, tok_idx:] = tokenizer.pad_token_id
                            break

            batch_outputs  = tokenizer.batch_decode(batch_outputs,  skip_special_tokens=True)
            batch_prompts  = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            batch_prompts  = [p for p in batch_prompts for _ in range(num_return_sequences)]
            batch_gen      = [o[len(p):] for p, o in zip(batch_prompts, batch_outputs)]

        except Exception as e:
            print(f"[WARN] Generation failed for batch {i//batch_size}: {e}")
            batch_gen   = [""] * len(batch_prompts) * num_return_sequences
            batch_finish = [False] * len(batch_gen)

        generations       += batch_gen
        finish_completion += batch_finish

        if not disable_tqdm:
            progress.update(len(batch_prompts) // num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences
    return generations, finish_completion


# ── Kept for completeness but not used by the main eval pipeline ──────────────

@torch.no_grad()
def get_next_word_predictions(model, tokenizer, prompts, candidate_token_ids=None,
                               batch_size=1, return_token_predictions=False,
                               disable_tqdm=False):
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    device = next(model.parameters()).device

    for i in range(0, len(prompts), batch_size):
        batch_prompts  = prompts[i:i + batch_size]
        tokenized      = tokenizer(batch_prompts, padding="longest",
                                    return_tensors="pt", add_special_tokens=False)
        batch_input_ids = tokenized.input_ids.to(device)
        attention_mask  = tokenized.attention_mask.to(device)

        batch_logits = model(input_ids=batch_input_ids,
                              attention_mask=attention_mask).logits[:, -1, :]
        if candidate_token_ids is not None:
            batch_logits = batch_logits[:, candidate_token_ids]

        batch_probs   = torch.softmax(batch_logits, dim=-1)
        batch_indices = torch.argmax(batch_probs, dim=-1)

        if return_token_predictions:
            if candidate_token_ids is not None:
                tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_preds = [tokens[idx] for idx in batch_indices]
            else:
                batch_preds = tokenizer.convert_ids_to_tokens(batch_indices)
            predictions += batch_preds
        else:
            predictions += batch_indices.tolist()
        probs += batch_probs.tolist()

        if not disable_tqdm:
            progress.update(len(batch_prompts))

    return predictions, probs