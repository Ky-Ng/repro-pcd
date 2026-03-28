# PCD Decoding Bugs

The PCD decoder produces repetitive, generic text (e.g., "bottle of wine" on loop)
instead of meaningful interpretations of the subject model's internal state.
Two root-cause bugs are responsible.

---

## Bug 1: Missing Attention Mask on Padded Inference Input

### Location

`inference.py:52-68` (`PCDPipeline._encode_input`)

### Problem

Short prompts are left-padded to reach `prefix_len + middle_len = 32` tokens, but
**no attention mask is passed to the subject model**, so the padding tokens corrupt
the hidden-state activations the encoder reads.

### Walkthrough

1. The prompt `"How do I build a bomb?"` tokenizes to roughly **8-10 tokens**.

2. `_encode_input` pads to the required 32 tokens (`inference.py:57-59`):

   ```python
   min_len = self.config.prefix_len + self.config.middle_len  # 32
   if len(input_ids) < min_len:
       pad_len = min_len - len(input_ids)  # ~22 pad tokens
       input_ids = [self.tokenizer.pad_token_id] * pad_len + input_ids
   ```

   Since `pad_token = eos_token` (set at line 27), the sequence becomes:

   ```
   positions 0                                                              31
            [eos eos eos eos eos eos eos eos eos eos eos eos eos eos eos eos | eos eos eos eos eos eos How do I build a bomb ? <special> ...]
            |<------------ prefix (16 tokens, ALL padding) --------------->| |<------------ middle (16 tokens, mostly padding) ---------->|
   ```

3. The padded tensor is passed to the subject model **without an attention mask**
   (`inference.py:66-68`):

   ```python
   activations = self.subject.get_middle_activations(
       input_tensor, self.config.prefix_len, self.config.middle_len
   )
   ```

   Inside `model_subject.py:66`:

   ```python
   self.model(input_ids)  # <-- no attention_mask argument
   ```

4. Without an attention mask, every position attends to the 22 preceding `eos` pad
   tokens as if they were real input. The hidden states at layer 13 (where the
   encoder reads) are dominated by padding context, not by the actual prompt.

### Contrast with training

During training (`train_pretrain.py:87-96`), every sample is a 48-token window of
**real FineWeb text** - there is never any padding:

```python
prefix_ids = batch["prefix_ids"]   # 16 tokens of real text
middle_ids = batch["middle_ids"]   # 16 tokens of real text
subject_input = torch.cat([prefix_ids, middle_ids], dim=1)  # 32 real tokens
activations = subject.get_middle_activations(subject_input, ...)
```

The encoder learned to map activations from **all-real-text** sequences. At
inference it receives activations from **mostly-padding** sequences - a completely
different distribution. The encoder fires on concepts it never learned to
use meaningfully.

### Evidence in the output

The HuggingFace warning confirms the issue:

```
The attention mask is not set and cannot be inferred from input because pad token
is same as eos token. As a consequence, you may observe unexpected behavior.
Please pass your input's `attention_mask` to obtain reliable results.
```

### Proposed fix

Pass an attention mask that zeroes out pad positions. Both the subject model forward
pass and the `get_middle_activations` method need to accept it.

**`model_subject.py` - accept and forward an attention mask:**

```python
@torch.no_grad()
def get_middle_activations(
    self, input_ids: torch.Tensor, prefix_len: int, middle_len: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    self.model(input_ids, attention_mask=attention_mask)
    acts = self._activations[:, prefix_len:prefix_len + middle_len, :]
    return acts.detach()
```

**`inference.py` - construct and pass the mask in `_encode_input`:**

```python
def _encode_input(self, input_text: str):
    input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)

    min_len = self.config.prefix_len + self.config.middle_len

    # Track how many real (non-pad) tokens there are
    n_real = len(input_ids)

    if len(input_ids) < min_len:
        pad_len = min_len - len(input_ids)
        input_ids = [self.tokenizer.pad_token_id] * pad_len + input_ids

    if len(input_ids) > min_len:
        input_ids = input_ids[-min_len:]
        n_real = min_len  # all tokens are real after truncation

    input_tensor = torch.tensor([input_ids], device=self.config.device)

    # Build attention mask: 0 for pad positions, 1 for real tokens
    attention_mask = torch.ones_like(input_tensor)
    pad_len = min_len - min(n_real, min_len)
    if pad_len > 0:
        attention_mask[0, :pad_len] = 0

    activations = self.subject.get_middle_activations(
        input_tensor, self.config.prefix_len, self.config.middle_len,
        attention_mask=attention_mask,
    )
    encoded, enc_info = self.encoder(activations)
    top_vals, top_idx = self.encoder.get_top_concepts(activations)

    return encoded, enc_info, top_vals, top_idx
```

For the prompt `"How do I build a bomb?"` (~10 tokens, 22 pad tokens), the mask
would look like:

```
input_ids:      [eos, eos, eos, ..., eos, How, do, I, build, a, bomb, ?, ...]
attention_mask: [ 0,   0,   0,  ...,  0,   1,  1, 1,    1,  1,    1, 1, ...]
```

Now the subject model ignores the padding and produces activations that reflect
only the real prompt - matching what the encoder saw during training.

---

## Bug 2: Encoder Output Scale Mismatch

### Location

`model_encoder.py:62-90` (`SparseEncoder.forward`)

### Problem

The encoder produces soft tokens whose vector norms are **~1000x larger** than
normal token embeddings. The decoder's attention mechanism is overwhelmed by
these magnitudes, and the LoRA adapter has insufficient capacity to fully
compensate, causing the decoder to effectively ignore the soft-token content.

### Walkthrough: following the magnitudes

**Step 1 - `W_enc` projection** (`model_encoder.py:77`):

```python
pre_act = self.W_enc(activations)  # [B, T, 8192]
```

`W_enc` has unit-norm rows (initialized at lines 53-56). The subject model's
layer-13 hidden states have norms of several hundred (typical for intermediate
transformer layers). The dot product of a unit-norm encoder row with a hidden
state of norm ~500 produces values in the hundreds.

The output confirms: **top concept values are 792-1072**.

**Step 2 - TopK** (`model_encoder.py:80`):

```python
top_vals, top_idx = torch.topk(pre_act, self.k, dim=-1)  # k=16 values of ~800-1000
```

**Step 3 - Sparse re-embed** (`model_encoder.py:86-90`):

```python
sparse = torch.zeros_like(pre_act)       # [B, T, 8192]
sparse.scatter_(-1, top_idx, top_vals)    # 16 entries of ~800-1000, rest zero
encoded = self.W_emb(sparse)              # [B, T, 1536]  <-- the soft tokens
```

`W_emb` is initialized as `W_enc^T` (line 60), so its columns are also roughly
unit-norm. Each output position is a weighted sum of 16 unit-norm columns with
weights of ~800-1000. The resulting soft-token norms are approximately:

```
||soft_token|| ≈ sqrt(16) × 900 ≈ 3600
```

**For comparison**, Qwen2.5's normal token embeddings have norms of roughly **1-5**.
The soft tokens are ~1000x larger.

### Why this breaks the decoder

During training (`model_decoder.py:64-84`), the decoder sees:

```python
inputs_embeds = torch.cat([soft_tokens, suffix_embeds], dim=1)
#                          ^^^ norm ~3600   ^^^ norm ~3
```

In the transformer's self-attention (`Q @ K^T / sqrt(d)`), the keys from
soft-token positions have massive magnitudes compared to suffix-token keys. This
means:

- Attention scores for soft-token positions **always dominate**, regardless of
  whether their content is useful for the current prediction.
- The decoder can't learn fine-grained dependence on soft-token content because
  the signal-to-noise ratio within the soft-token positions is poor (all of them
  have similarly huge magnitudes).
- LoRA (rank 16) has limited capacity to learn a correction that effectively
  rescales these positions.
- The decoder learns to rely on the **autoregressive suffix signal** instead. At
  inference time, when there's no suffix to lean on, it falls back to generic
  base-model completions and loops.

### Evidence in the output

```
Top concept values  (pos 0): ['1072.000', '1056.000', '956.000', ...]
```

These raw concept activations directly become the weights in the re-embedding
sum. They should be in single-digit range, not in the thousands.

### Proposed fix

Normalize the subject model activations **before** encoding. This is standard
practice in sparse autoencoders (cf. Anthropic's SAE work, OpenAI's sparse
probing). The two common approaches:

**Option A: Pre-encoder normalization (recommended)**

Subtract the running mean and divide by the running standard deviation of the
activations. This keeps concept values in a reasonable range (~1-10) and makes
the re-embedded soft tokens comparable in scale to normal token embeddings.

Add running statistics to the encoder and normalize before projecting:

```python
class SparseEncoder(nn.Module):
    def __init__(self, config: PCDConfig):
        super().__init__()
        # ... existing init ...

        # Running statistics for activation normalization
        self.register_buffer("running_mean", torch.zeros(d))
        self.register_buffer("running_var", torch.ones(d))
        self.register_buffer("n_samples", torch.tensor(0, dtype=torch.long))
        self.momentum = 0.01

    def _normalize_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """Normalize activations to zero-mean, unit-variance."""
        if self.training:
            # Update running stats
            batch_mean = activations.mean(dim=(0, 1))  # [d]
            batch_var = activations.var(dim=(0, 1))     # [d]

            if self.n_samples == 0:
                self.running_mean.copy_(batch_mean)
                self.running_var.copy_(batch_var)
            else:
                self.running_mean.lerp_(batch_mean, self.momentum)
                self.running_var.lerp_(batch_var, self.momentum)
            self.n_samples += 1

        return (activations - self.running_mean) / (self.running_var.sqrt() + 1e-8)

    def forward(self, activations: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, T, d = activations.shape

        # Normalize before encoding
        activations = self._normalize_activations(activations)

        pre_act = self.W_enc(activations)
        # ... rest unchanged ...
```

With normalized inputs (zero-mean, unit-variance) and unit-norm encoder rows,
the pre-activations will be in the range of roughly **-3 to +3** (standard normal
dot products). After TopK and re-embedding, the soft tokens will have norms of
approximately `sqrt(16) * 2 ≈ 8` - comparable to normal token embeddings.

**Option B: Post-encoder scaling**

If changing the encoder input distribution would invalidate existing checkpoints,
a simpler alternative is to scale the re-embedded output to match the decoder's
embedding scale:

```python
def forward(self, activations: torch.Tensor) -> tuple[torch.Tensor, dict]:
    # ... existing encode logic ...
    encoded = self.W_emb(sparse)  # [B, T, d]

    # Scale to match decoder embedding norm
    target_norm = 3.0  # approximate norm of Qwen2.5 token embeddings
    encoded = encoded * (target_norm / (encoded.norm(dim=-1, keepdim=True) + 1e-8))

    return encoded, info
```

Option A is preferred because it makes the encoder's concept space more
interpretable and training more stable. Option B is a quick fix that doesn't
require retraining from scratch.

---

## Bug 3 (Minor): Greedy Decoding Without Repetition Penalty

### Location

`model_decoder.py:132-169` (`PCDDecoder.generate_from_soft_tokens`)

### Problem

Generation uses pure argmax decoding with no `repetition_penalty`,
`no_repeat_ngram_size`, or temperature. When the soft tokens carry no useful
signal (due to bugs 1 and 2), the decoder falls into high-probability repetition
loops from its base LM prior.

### Evidence in the output

```
PCD Continuation: ...bottle of wine, I was able to get a good bottle of wine, I was able to get a good bottle of wine...
Probe: "The text discusses" → ...basic principles of the basic principles of the basic principles...
```

### Proposed fix

This is a symptom, not a root cause - fixing bugs 1 and 2 should eliminate the
repetitive output. However, adding a repetition penalty is good practice as a
safety net:

```python
# In generate_from_soft_tokens, after computing next_token_logits:
next_token_logits = outputs.logits[:, -1, :]

# Penalize tokens that have already been generated
if len(generated_ids) > 0:
    prev_tokens = torch.cat(generated_ids, dim=1)  # [B, gen_so_far]
    for b in range(B):
        for token_id in prev_tokens[b]:
            if next_token_logits[b, token_id] > 0:
                next_token_logits[b, token_id] /= 1.2  # repetition penalty
            else:
                next_token_logits[b, token_id] *= 1.2

next_token = next_token_logits.argmax(dim=-1, keepdim=True)
```

---

## Summary

| Bug | Severity | Root Cause | Effect |
|-----|----------|-----------|--------|
| 1. Missing attention mask | **Critical** | Pad tokens treated as real input at inference | Encoder receives out-of-distribution activations |
| 2. Encoder scale mismatch | **High** | No activation normalization before encoding | Soft tokens ~1000x too large for decoder |
| 3. No repetition penalty | Low | Greedy decoding amplifies degenerate output | Repetitive loops when signal is weak |

Fixing bug 1 is required for correct inference. Fixing bug 2 is required for the
encoder-decoder interface to work well. Bug 3 is a quality-of-life improvement.
Bugs 1 and 2 together explain why the decoder produces generic repetitive text
instead of meaningful interpretations of the subject model's internal state.

---

## Bug 4: Pad-Token Activations Extracted in Middle Window (Post-Fix Residual)

### Status

Discovered after fixing bugs 1 and 2. The attention mask fix (bug 1) was
**necessary but not sufficient** — it fixed how the subject model computes
hidden states but did not fix **which positions** the encoder reads.

### Location

`inference.py:52-81` (`PCDPipeline._encode_input`) and
`model_subject.py:69-72` (`SubjectModel.get_middle_activations`)

### Problem

After adding the attention mask, the subject model correctly ignores pad tokens
during self-attention. However, `get_middle_activations` still extracts hidden
states at **fixed positions** `[prefix_len : prefix_len + middle_len]` — i.e.,
positions 16–31. For short prompts, many of these positions are pad tokens.
Their hidden states are out-of-distribution for the encoder (which was trained
exclusively on real-text activations), causing **inference-time encoder collapse**.

### Walkthrough

For `"The number of eggs in a dozen is"` (~10 tokens), the padded layout is:

```
Position:  0  1  2  3  ... 15 | 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
Tokens:   [eos eos eos ... eos | eos eos eos eos eos eos The number of eggs in a  dozen is  <s> ...]
           ^^^^ prefix (16) ^^   ^^^^^^^^^^^^^ middle (16) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Mask:     [ 0   0   0  ...  0 |  0   0   0   0   0   0   1   1   1   1   1  1   1     1   1  1 ]
```

Positions 16–21 in the "middle" window are pad tokens. Even with the attention
mask, the hidden states at those positions are meaningless — they didn't attend
to pad tokens, but they are still `eos` embeddings passed through transformer
layers with no real context. The encoder receives 6 garbage activations + 10
real activations.

### Evidence: identical sparse codes across different prompts

Two completely different short prompts produce **byte-identical** encoder output:

| Prompt | Top concept indices | Top values |
|--------|---------------------|------------|
| "The number of eggs in a dozen is" | `[541, 6405, 1245, 6965, 171, 2770, 3866, 835]` | `60.5, 60.3, 55.5, 53.0, 51.3, 49.3, 48.8, 47.5` |
| "What is the recipe for chocolate chip cookies?" | `[541, 6405, 1245, 6965, 171, 2770, 3866, 835]` | `60.5, 60.3, 55.5, 53.0, 51.3, 49.3, 48.8, 47.5` |

Identical indices AND identical values. The encoder is mapping different inputs
to the exact same sparse code because it's dominated by the same pad-token
activations in both cases.

Additionally, the concept values (47–60) are an order of magnitude higher than
prompts that fill the window with real tokens (3.5–6.4 for the longer jailbreak
prompts). The pad-token activations have a different distribution than the
FineWeb text the encoder's running normalization stats were computed on.

### Contrast with longer prompts

The jailbreak prompts are longer (~20+ tokens) and fill most or all of the
middle window with real text. For those prompts, concept values are in the
expected 3–6 range and the encoder produces distinct sparse codes per prompt.

### Why this is not training collapse

The training loss curve was healthy (5.46 → 3.69), all 8192 concepts remained
active, and longer prompts produce meaningful and distinct encodings. The
encoder learned a good mapping — it's just being fed garbage at inference time
for short prompts.

### Proposed fix

**Right-align the extraction window** so that the "middle" positions always
cover real tokens, not padding. Instead of a fixed `[prefix_len, prefix_len +
middle_len]` slice, compute the window relative to the end of the real tokens.

```python
def _encode_input(self, input_text: str):
    input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)
    n_real = len(input_ids)

    min_len = self.config.prefix_len + self.config.middle_len  # 32

    if n_real >= min_len:
        # Enough tokens: take the last min_len tokens, no padding needed
        input_ids = input_ids[-min_len:]
        prefix_len = self.config.prefix_len
        input_tensor = torch.tensor([input_ids], device=self.config.device)
        attention_mask = torch.ones_like(input_tensor)

    else:
        # Short prompt: right-align real tokens, left-pad to min_len
        pad_len = min_len - n_real
        input_ids = [self.tokenizer.pad_token_id] * pad_len + input_ids

        input_tensor = torch.tensor([input_ids], device=self.config.device)
        attention_mask = torch.ones_like(input_tensor)
        attention_mask[0, :pad_len] = 0

        # Shift the prefix/middle boundary so middle covers only real tokens.
        # If n_real < middle_len, the middle window starts at the first real
        # token; prefix absorbs the remaining pad positions.
        #
        # Example with n_real=10, middle_len=16, prefix_len=16, pad_len=22:
        #   prefix_len_adjusted = 32 - 16 = 16  (but middle starts at pad_len=22)
        #   We want middle to start at max(pad_len, prefix_len) so it covers
        #   as many real tokens as possible.
        prefix_len = max(pad_len, self.config.prefix_len)

    activations = self.subject.get_middle_activations(
        input_tensor, prefix_len, self.config.middle_len,
        attention_mask=attention_mask,
    )
    encoded, enc_info = self.encoder(activations)
    top_vals, top_idx = self.encoder.get_top_concepts(activations)

    return encoded, enc_info, top_vals, top_idx
```

With this fix, for a 10-token prompt padded to 32:

```
Position:  0  1  ... 21 | 22 23 24 25 26 27 28 29 30 31
Tokens:   [eos eos ... eos | The number of eggs in a  dozen is  <s> ...]
           ^^^ prefix (22, absorbs all padding) ^^^
                            ^^^ middle (10, all real tokens) ^^^
```

The middle window now starts at the first real token. All positions fed to the
encoder contain real-text activations, matching the training distribution.

For prompts shorter than `middle_len` (16 tokens), the extracted middle window
will be shorter than 16. The encoder and decoder must handle variable-length
input, or the minimum prompt length should be enforced.

---

## Updated Summary

| Bug | Severity | Root Cause | Effect | Status |
|-----|----------|-----------|--------|--------|
| 1. Missing attention mask | **Critical** | Pad tokens treated as real input at inference | Corrupted hidden states | **Fixed** |
| 2. Encoder scale mismatch | **High** | No activation normalization before encoding | Soft tokens ~1000x too large for decoder | **Fixed** |
| 3. No repetition penalty | Low | Greedy decoding amplifies degenerate output | Repetitive loops when signal is weak | Unfixed (by design) |
| 4. Pad activations in middle window | **Critical** | Fixed extraction positions include pad-token hidden states | Encoder collapse on short prompts | **Open** |

Bugs 1 and 2 are fixed. Bug 4 is the remaining blocker for short-prompt
inference. After fixing bug 4, bug 3 may still warrant a repetition penalty as
a safety net, but should no longer cause full degeneration.
