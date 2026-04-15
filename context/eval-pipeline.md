# Eval Pipeline — Prompt → Decoder Output

Add an inference-time pipeline so a raw user prompt can be run end-to-end through Subject → Encoder → Decoder and produce decoded text. The training path (`train_pretraining.py`) already wires the three components together on pre-split `prefix/middle/suffix` token batches; we need the analogous **deployment** path described in `subject_model.get_middle_activations` where only a "user" span exists.

## Motivation

`DecoderModel` today exposes only `forward_train`, which returns a loss. There is no way to:
1. Feed soft tokens into the decoder and autoregressively decode text.
2. Run the full pipeline from a raw string prompt.
3. Load a saved checkpoint back into encoder + decoder for evaluation.

This plan closes those gaps with minimal surface area and no changes to training.

---

## Changes

### 1. `src/architecture/decoder_model.py` — add `generate`

Add a new inference method that mirrors the embed-concat logic in `forward_train` but omits the target half and delegates autoregressive decoding to HuggingFace.

```python
@torch.no_grad()
def generate(
    self,
    soft_token_acts: Float[Tensor, "batch n_soft d_model"],
    soft_token_mask: Bool[Tensor, "batch n_soft"] | None = None,
    context_ids: Int[Tensor, "batch n_context"] | None = None,
    max_new_tokens: int = 256,
    do_sample: bool = False,
) -> list[str]:
```

**Implementation outline**
1. Build `embed_parts = [soft_token_acts]` and `mask_parts = [soft_token_mask or ones]`.
2. If `context_ids` is provided, embed them via `self.model.get_input_embeddings()`, append to `embed_parts`, and append `(context_ids != pad_token_id).long()` to `mask_parts`.
3. `inputs_embeds = torch.cat(embed_parts, dim=1)`; `attention_mask = torch.cat(mask_parts, dim=1)`.
4. Call:
   ```python
   output_ids = self.model.generate(
       inputs_embeds=inputs_embeds,
       attention_mask=attention_mask,
       max_new_tokens=max_new_tokens,
       do_sample=do_sample,
       pad_token_id=self.tokenizer.pad_token_id,
       eos_token_id=self.tokenizer.eos_token_id,
   )
   ```
5. Return `self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)`.

**Notes**
- When `generate` is called with `inputs_embeds`, HF returns only the newly generated token IDs (not the prefix), so `batch_decode` yields just the completion.
- KV cache is handled automatically by `self.model.generate`, so no manual greedy loop is needed.
- Works with the PEFT-wrapped model — `get_peft_model` returns a `PeftModel` that forwards `generate` to the base model with adapters applied.

### 2. Fix existing annotation bug in `forward_train`

`soft_token_mask` is annotated as `Bool[Tensor, "batch n_soft d_model"]` but masks are per-token, not per-feature. Correct to `Bool[Tensor, "batch n_soft"]`. Update both the parameter annotation and the docstring visuals. Pure annotation change — no runtime effect.

### 3. Checkpoint loading

Training already saves via `save_checkpoint` in [src/training/utils.py](src/training/utils.py):
- `encoder.pt` — raw `state_dict`
- `decoder_lora/` — `PeftModel.save_pretrained` output

Add a `load_checkpoint` helper alongside `save_checkpoint` (same file) with shape:

```python
def load_checkpoint(
    encoder: SparseEncoder,
    decoder: DecoderModel,
    checkpoint_dir: str,
    device: str,
) -> None:
    enc_state = torch.load(os.path.join(checkpoint_dir, "encoder.pt"), map_location=device)
    encoder.load_state_dict(enc_state)
    decoder.model.load_adapter(os.path.join(checkpoint_dir, "decoder_lora"), adapter_name="default")
```

Rationale for living in `training/utils.py`: it's the symmetric counterpart to `save_checkpoint` and reuses the same directory layout convention. Eval code imports it; no duplication.

### 4. `src/architecture/pcd_inference.py` — new wrapper

A thin orchestration class that owns a `SubjectModel`, `SparseEncoder`, and `DecoderModel` and exposes a single `generate(prompt) -> str`.

**Why a plain class, not `nn.Module`:** `SubjectModel` wraps a `HookedTransformer` and is not an `nn.Module`; it has no `.to()` / `.parameters()`. Mixing it into an `nn.Module` hierarchy would be awkward and buy nothing for inference. The wrapper just holds references.

**Why inference-only (no `.forward` that matches training):** `train_pretraining.py` operates on pre-split `prefix_ids`/`middle_ids`/`suffix_ids` tensors and extracts activations at fixed positions. A raw-text `.forward` would not naturally slot in there. Unifying train/inference interfaces is a larger refactor worth doing separately; keep this wrapper scoped to inference to avoid scope creep.

```python
class PCDInferenceModel:
    def __init__(self, config: PCDConfig):
        self.config = config
        self.subject = SubjectModel(config)
        self.encoder = SparseEncoder(config).to(config.device).eval()
        self.decoder = DecoderModel(config).to(config.device).eval()

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        load_checkpoint(self.encoder, self.decoder, checkpoint_dir, self.config.device)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        # 1. Chat-template and tokenize the prompt via subject tokenizer.
        templated = self.subject.apply_chat_template(prompt)
        enc = self.subject.tokenize(templated)
        tokens = enc["input_ids"]                 # [1, seq]
        attn   = enc["attention_mask"]            # [1, seq]

        # 2. Extract residual-stream activations at l_read over the full user span.
        #    Deployment case in subject_model.py — no fixed prefix/middle/suffix split.
        seq_len = tokens.shape[1]
        activations = self.subject.get_middle_activations(
            tokens=tokens,
            attention_mask=attn,
            start_extract=0,
            end_extract=seq_len,
        )  # [1, seq, d_model]

        # 3. Encode → sparse re-embedding (soft tokens for the decoder).
        with torch.autocast(device_type=self.config.device, dtype=self.config.dtype):
            sparse_embedding, _info = self.encoder(activations)

        # 4. Generate from decoder.
        outputs = self.decoder.generate(
            soft_token_acts=sparse_embedding,
            max_new_tokens=max_new_tokens,
        )
        return outputs[0]
```

**Extraction span choice.** We extract activations over the entire chat-templated sequence (`start=0`, `end=seq_len`) rather than trying to isolate only the user-content tokens. Rationale:
- During pretraining, activations are drawn from the `middle` slice of FineWeb text — there is no system/user distinction.
- At deployment the whole templated sequence is what the subject model actually saw; slicing out just the user content risks misaligning with what the encoder learned to interpret.
- If we later finetune on SynthSys (where a system prompt + user message structure is explicit), we can revisit and use a soft-token mask to restrict to the user span.

### 5. `src/eval/evaluate_prompt.py` — new CLI entrypoint

```python
"""
Run a prompt through the full PCD pipeline (Subject → Encoder → Decoder).

Usage:
    python -m src.eval.evaluate_prompt \
        --prompt "the user is a woman, what should I wear at a wedding?" \
        --checkpoint out/checkpoints/<run-tag>/step_<N>
"""
import argparse

from src.architecture.pcd_inference import PCDInferenceModel
from src.pcd_config import PCDConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a prompt through the PCD pipeline")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a step_<N> directory containing encoder.pt and decoder_lora/")
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = PCDConfig()
    pipeline = PCDInferenceModel(config)
    pipeline.load_checkpoint(args.checkpoint)

    output = pipeline.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    print(output)


if __name__ == "__main__":
    main()
```

Also create `src/eval/__init__.py` (empty) so the module is importable via `python -m`.

---

## Explicit non-goals

- **No `forward_inference` on `DecoderModel`.** HF's `model.generate(inputs_embeds=...)` gives KV-cached autoregressive decoding for free; a manual greedy loop over a raw-logits method would be O(n²) and duplicate well-tested functionality. If we later need raw next-token logits for analysis, add it then.
- **No training-path changes.** `train_pretraining.py` continues to work directly with encoder/decoder/subject components — the wrapper is inference-only.
- **No batched CLI.** `evaluate_prompt.py` takes a single `--prompt`. Batched evaluation can be a separate script if/when needed.
- **No sampling knobs beyond `max_new_tokens`.** Greedy decode (`do_sample=False`) is the default and matches the training objective (next-token CE).

## File summary

| File | Change |
|---|---|
| `src/architecture/decoder_model.py` | Add `generate`; fix `soft_token_mask` shape annotation |
| `src/training/utils.py` | Add `load_checkpoint` helper |
| `src/architecture/pcd_inference.py` | **New** — `PCDInferenceModel` wrapper |
| `src/eval/__init__.py` | **New** — empty, package marker |
| `src/eval/evaluate_prompt.py` | **New** — CLI entrypoint |

## Smoke test

After implementation, run on any saved checkpoint:

```bash
python -m src.eval.evaluate_prompt \
    --prompt "the user is a woman, what should I wear at a wedding?" \
    --checkpoint out/checkpoints/<run-tag>/step_<N>
```

Expected: a decoded string completion. It will likely be low-quality early in training (the decoder is learning to interpret concept-encoded soft tokens); the point of the smoke test is that the pipeline runs end-to-end without shape or dtype errors.
