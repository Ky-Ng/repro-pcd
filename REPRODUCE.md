# Reproducing Predictive Concept Decoders (PCDs) on a Student Budget

**Paper:** [Predictive Concept Decoders: Training Scalable End-to-End Interpretability Assistants](https://arxiv.org/abs/2512.15712) (Huang et al., 2025)

**Goal:** Build a minimal working PCD and use it to reveal what an LLM "thinks" about a jailbreak prompt.

---

## Table of Contents

1. [Background & Intuition](#1-background--intuition)
2. [Architecture Overview](#2-architecture-overview)
3. [Student Budget Adaptations](#3-student-budget-adaptations)
4. [Setup & Dependencies](#4-setup--dependencies)
5. [Phase 1: Configuration (`config.py`)](#5-phase-1-configuration)
6. [Phase 2: Subject Model (`model_subject.py`)](#6-phase-2-subject-model)
7. [Phase 3: Sparse Encoder (`model_encoder.py`)](#7-phase-3-sparse-encoder)
8. [Phase 4: Decoder with LoRA (`model_decoder.py`)](#8-phase-4-decoder-with-lora)
9. [Phase 5: Data Pipeline (`data.py`)](#9-phase-5-data-pipeline)
10. [Phase 6: Pretraining (`train_pretrain.py`)](#10-phase-6-pretraining)
11. [Phase 7: Inference & Demo (`inference.py`, `demo_jailbreak.py`)](#11-phase-7-inference--demo)
12. [Expected Results](#12-expected-results)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Background & Intuition

### What are PCDs?

Predictive Concept Decoders are **interpretability assistants** — they let you ask a language model "what are you thinking?" by reading its internal activations and translating them into natural language.

The key insight: instead of reconstructing activations (like Sparse Autoencoders), PCDs predict **future tokens** from internal representations. This forces the encoder to learn concepts that are *useful for understanding model behavior*, not just faithful reconstructions.

### The Three Components

```
                    ┌─────────────────┐
   Input Text ───►  │  Subject Model  │  (frozen Qwen2.5-1.5B-Instruct)
                    │   Layer 13      │
                    └────────┬────────┘
                             │ activations [B, 16, 1536]
                             ▼
                    ┌─────────────────┐
                    │    Encoder      │  (learned: W_enc, b_enc, W_emb)
                    │  TopK(k=16)     │
                    └────────┬────────┘
                             │ sparse concepts → re-embedded [B, 16, 1536]
                             ▼
                    ┌─────────────────┐
   Question ──────► │    Decoder      │  (Qwen2.5-1.5B + LoRA)
                    │  (soft tokens   │
                    │   + question)   │
                    └────────┬────────┘
                             │
                             ▼
                       Answer text
```

**Communication bottleneck:** The encoder never sees the question. The decoder never sees the raw activations. All information must pass through a sparse set of 16 concept activations per token position. This forces the encoder to learn *general-purpose* representations.

### Why it works for jailbreaks

When a model processes a jailbreak prompt like "how to build a bomb," its internal activations encode *what it understands about the text* — even if it ultimately refuses. The PCD encoder captures these internal concepts, and the decoder can articulate them in natural language.

---

## 2. Architecture Overview

### Encoder Math

Given activation vector `a(i) ∈ R^d` from layer `l_read` of the subject model:

```
pre_act = W_enc @ a(i) + b_enc        # R^d → R^m  (project to concept space)
sparse  = TopK(pre_act, k=16)          # zero out all but top-16 values
a'(i)   = W_emb @ sparse              # R^m → R^d  (re-embed to hidden space)
```

- `W_enc ∈ R^(m × d)`: Each row is a "concept template" — a direction in activation space
- `b_enc ∈ R^m`: Bias (initialized to zero)
- `W_emb ∈ R^(d × m)`: Re-embedding matrix (initialized as `W_enc^T`)
- `TopK`: Keeps only the k=16 largest values, zeros the rest

### Decoder Setup

The decoder is the **same base model** (Qwen2.5-1.5B-Instruct) with a **LoRA adapter** for parameter efficiency. The encoder's output is injected as "soft tokens" at the start of the decoder's input:

```
Decoder input = [soft_token_1, ..., soft_token_16, question_token_1, ..., question_token_Q]
```

This is implemented via `inputs_embeds` — we bypass the decoder's embedding layer and directly provide the concatenated tensor.

### Training Objective

**Pretraining loss** (on FineWeb):
```
L = -Σ log p_decoder(suffix_token_t | suffix_tokens_1..t-1, encoder(middle_activations))
```

The subject model processes `prefix + middle` tokens. We read activations from the middle positions at layer 13. The encoder compresses them. The decoder predicts the suffix tokens.

**Auxiliary loss** (dead concept revival):
```
L_aux = -ε * mean(pre_activations_of_dead_concepts)
```

This pushes dead (never-selected) concepts toward being activated.

---

## 3. Student Budget Adaptations

| Aspect | Paper (Full) | Our Reproduction |
|--------|-------------|-----------------|
| Subject/Decoder model | Llama-3.1-8B-Instruct | **Qwen2.5-1.5B-Instruct** |
| Hidden dimension (d) | 4096 | **1536** |
| Number of concepts (m) | 32,768 | **8,192** |
| TopK (k) | 16 | **16** (unchanged) |
| Read layer (l_read) | 15 (of 32) | **13** (of 28, ~47% depth) |
| Prefix/Middle/Suffix | 16/16/16 | **16/16/16** (unchanged) |
| Pretraining data | 72M tokens | **~5M tokens** |
| Training steps | Not specified | **~5,000** |
| LoRA rank | Not specified | **16** |
| GPU requirement | A100 80GB | **Consumer 16-24GB GPU** |

**Why Qwen2.5-1.5B-Instruct?**
- Instruct-tuned with safety training (essential for jailbreak demo — the model needs to *refuse* harmful requests)
- 1.5B parameters → two copies fit in ~6GB in bf16
- No gated access (unlike Llama models)
- Well-supported by PEFT/LoRA

---

## 4. Setup & Dependencies

### Prerequisites
- Python 3.10+
- CUDA-capable GPU with ≥16GB VRAM
- ~10GB disk space for models + data

### Install

```bash
pip install torch transformers peft datasets accelerate tqdm
```

Or use the provided requirements file:
```bash
pip install -r requirements.txt
```

### Verify GPU Access

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

---

## 5. Phase 1: Configuration

**File:** `config.py`

All hyperparameters are centralized in a single `PCDConfig` dataclass. This is the single source of truth — no magic numbers elsewhere.

Key parameters:
```python
@dataclass
class PCDConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hidden_dim: int = 1536          # d
    num_concepts: int = 8192        # m
    topk: int = 16                  # k
    l_read: int = 13                # layer to read activations
    l_write: int = 0                # layer to inject soft tokens
    prefix_len: int = 16
    middle_len: int = 16
    suffix_len: int = 16
    lora_r: int = 16
    lora_alpha: int = 32
    lr_encoder: float = 3e-4
    lr_decoder: float = 1e-4
    batch_size: int = 16
    grad_accum_steps: int = 4       # effective batch = 64
    max_train_steps: int = 5000
```

---

## 6. Phase 2: Subject Model

**File:** `model_subject.py`

The subject model is the LLM whose "thoughts" we want to decode. It is loaded fully frozen.

### How activation extraction works

We use a **PyTorch forward hook** on layer 13 of the transformer. During the forward pass, the hook captures the hidden states at that layer:

```python
def _capture_hook(self, module, input, output):
    if isinstance(output, tuple):
        self._activations = output[0]  # hidden states
    else:
        self._activations = output
```

Then we extract only the middle token positions (positions 16-31 from a 32-token input):

```python
def get_middle_activations(self, input_ids, prefix_len, middle_len):
    with torch.no_grad():
        self.model(input_ids)  # triggers hook
    return self._activations[:, prefix_len:prefix_len + middle_len, :]
```

### Why middle tokens only?

This is the **communication bottleneck** design. The paper deliberately excludes prefix token activations because:
- If prefix activations were included, the encoder could simply pass through token embeddings as a shortcut
- By only seeing middle-token activations, the encoder must learn *conceptual* representations — what the model has understood from the context, not just what tokens appeared

### Verification

```python
from config import PCDConfig
from model_subject import SubjectModel

config = PCDConfig()
subject = SubjectModel(config)

# Input: 32 tokens (prefix=16, middle=16)
input_ids = torch.randint(0, 1000, (1, 32), device='cuda')
acts = subject.get_middle_activations(input_ids, 16, 16)
assert acts.shape == (1, 16, 1536)  # [batch, middle_len, hidden_dim]
```

---

## 7. Phase 3: Sparse Encoder

**File:** `model_encoder.py`

This is the core novel component of PCDs.

### Architecture

```
activations [B, 16, 1536]
       │
       ▼ W_enc (linear: 1536 → 8192)
pre_activations [B, 16, 8192]
       │
       ▼ TopK(k=16)
sparse [B, 16, 8192]  (only 16 non-zero per position)
       │
       ▼ W_emb (linear: 8192 → 1536)
encoded [B, 16, 1536]
```

### Initialization

Following the paper:
- `W_enc` rows are initialized with **unit norm** (random directions in activation space)
- `b_enc` is initialized to **zero**
- `W_emb` is initialized as **`W_enc^T`** (the transpose of the encoder weights)

This means initially, each concept direction can both detect (via `W_enc`) and reconstruct (via `W_emb`) the same direction in activation space.

### TopK and gradient flow

`torch.topk` returns values that are part of the computation graph — gradients flow through the **selected** top-k values. The *selection* of which concepts are in the top-k is non-differentiable (hard gating), but the values themselves get gradients. This acts as a sparse gating mechanism.

### Dead concept revival

With 8192 concepts and TopK=16, most concepts go unused in any given batch. Over time, some concepts may never activate ("dead concepts"). The auxiliary loss pushes their pre-activations upward:

```python
L_aux = -ε * mean(pre_activations[dead_concepts])
```

where dead concepts are those not selected in any top-k for the last `dead_concept_window` batches.

### Verification

```python
from model_encoder import SparseEncoder
encoder = SparseEncoder(config).to('cuda').to(torch.bfloat16)

acts = torch.randn(2, 16, 1536, device='cuda', dtype=torch.bfloat16)
encoded, info = encoder(acts)
assert encoded.shape == (2, 16, 1536)
print(f"Active concepts: {info['n_active_concepts']}")
```

---

## 8. Phase 4: Decoder with LoRA

**File:** `model_decoder.py`

### LoRA Setup

The decoder uses the same Qwen2.5-1.5B-Instruct base model but adds LoRA adapters to all attention and MLP projections:

```python
LoraConfig(
    r=16,               # rank
    lora_alpha=32,       # scaling
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)
```

This adds ~18.5M trainable parameters (1.18% of the 1.5B total).

### Soft-token patching via `inputs_embeds`

The key implementation detail: HuggingFace transformers support passing `inputs_embeds` instead of `input_ids` to the model's forward method. This bypasses the embedding layer entirely.

We construct the decoder's input by concatenating:
1. **Soft tokens** from the encoder (already in hidden-dim space)
2. **Text token embeddings** (looked up from the decoder's embedding table)

```python
def forward_train(self, soft_tokens, suffix_ids):
    # soft_tokens: [B, 16, 1536] from encoder
    # suffix_ids: [B, 16] target suffix tokens

    suffix_embeds = self.get_token_embeddings(suffix_ids[:, :-1])  # [B, 15, 1536]
    inputs_embeds = torch.cat([soft_tokens, suffix_embeds], dim=1)  # [B, 31, 1536]

    outputs = self.model(inputs_embeds=inputs_embeds, ...)

    # Loss on suffix positions only
    suffix_logits = outputs.logits[:, 15:, :]  # positions that predict suffix tokens
    loss = cross_entropy(suffix_logits, suffix_ids)
    return loss
```

### Why `inputs_embeds` instead of hooks?

The paper describes patching at "layer 0" of the decoder. Using `inputs_embeds` achieves this cleanly — the soft tokens enter the transformer at the very first layer, just like regular token embeddings would. No hooks needed.

### Verification

```python
decoder = PCDDecoder(config)
decoder.model.to('cuda')

soft = torch.randn(2, 16, 1536, device='cuda', dtype=torch.bfloat16)
suffix = torch.randint(0, 1000, (2, 16), device='cuda')
loss = decoder.forward_train(soft, suffix)
loss.backward()  # gradients flow back to soft tokens (and thus to encoder)
```

---

## 9. Phase 5: Data Pipeline

**File:** `data.py`

### FineWeb Streaming

We stream from HuggingFace's FineWeb dataset (`sample-10BT` subset) to avoid downloading the full dataset:

```python
ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
```

### Tokenization and Chunking

Each document is tokenized and split into non-overlapping 48-token windows:

```
Document tokens: [t1, t2, t3, ..., t96, t97, ...]
                  ├──── window 1 ────┤├──── window 2 ────┤
                  │prefix│middle│suffix│prefix│middle│suffix│
                  │  16  │  16  │  16  │  16  │  16  │  16  │
```

Documents shorter than 48 tokens are skipped.

### Caching

The tokenized windows are cached to disk (`data_cache/fineweb_windows.pt`) so subsequent runs don't need to re-download and tokenize.

### Running

```python
from data import prepare_data, get_dataloader
from config import PCDConfig

config = PCDConfig()
dataset = prepare_data(config, num_examples=100_000)  # ~4.8M tokens
loader = get_dataloader(dataset, config)

batch = next(iter(loader))
# batch["prefix_ids"]: [16, 16]
# batch["middle_ids"]: [16, 16]
# batch["suffix_ids"]: [16, 16]
```

---

## 10. Phase 6: Pretraining

**File:** `train_pretrain.py`

### What gets trained

| Component | Trainable? | Parameters |
|-----------|-----------|------------|
| Subject model | **No** (frozen) | 0 |
| Encoder (W_enc, b_enc, W_emb) | **Yes** | ~25M |
| Decoder base weights | **No** (frozen) | 0 |
| Decoder LoRA adapters | **Yes** | ~18.5M |

**Total trainable: ~43.5M parameters** (out of ~3.1B total loaded)

### Forward Pass (per batch)

```
1. subject_input = cat(prefix_ids, middle_ids)     → [B, 32]
2. activations = subject.layer_13_hook(input)       → [B, 16, 1536]
3. encoded, info = encoder(activations)             → [B, 16, 1536]
4. decoder_input = cat(encoded, embed(suffix[:-1])) → [B, 31, 1536]
5. logits = decoder(inputs_embeds=decoder_input)    → [B, 31, vocab_size]
6. loss = cross_entropy(logits[16:], suffix_ids)    → scalar
7. total_loss = loss + aux_dead_concept_loss
8. total_loss.backward()
```

### Gradient Flow

```
Decoder loss
    ↓ backward through decoder LoRA
    ↓ backward through soft token positions
    ↓ backward through W_emb (re-embedding)
    ↓ backward through TopK selected values
    ↓ backward through W_enc (concept projection)
    ✗ stops at subject model (frozen, no grad)
```

### Optimizer & Schedule

- **AdamW** with two parameter groups:
  - Encoder: lr=3e-4
  - Decoder LoRA: lr=1e-4
- **Cosine schedule** with 500-step linear warmup
- Gradient clipping at norm 1.0
- Gradient accumulation: 4 micro-batches → effective batch size 64

### Running

```bash
python train_pretrain.py
```

Expected output:
```
[train] step 50   | loss=5.04  | active_concepts=8192 | dead_concepts=0
[train] step 100  | loss=4.42  | active_concepts=8192 | dead_concepts=0
...
[train] step 5000 | loss=3.80  | active_concepts=8100 | dead_concepts=92
```

Key things to watch:
- **Loss should decrease** from ~6 to ~3.5-4.0 over 5000 steps
- **Active concepts** should stay above 90% of m
- **Checkpoints** saved every 1000 steps to `checkpoints/step_N/`

---

## 11. Phase 7: Inference & Demo

**File:** `inference.py`, `demo_jailbreak.py`

### Important: Pretrained vs Finetuned Inference

The PCD decoder was only pretrained on next-token prediction (predicting suffix tokens from encoded middle activations). It was **not** finetuned to answer questions about the subject model. This means:

- **Don't ask questions** — the decoder doesn't know how to answer questions
- **Instead, use probe phrases** — short text prefixes like "The text discusses" or "The main topic is" that the decoder can naturally continue
- **Or use pure continuation** — just give the soft tokens and let the decoder generate freely

The paper's QA capability comes from the finetuning phase on SynthSys, which we skip.

### Running the Demo

```bash
python demo_jailbreak.py --checkpoint checkpoints/step_5000
```

This runs three prompts and for each shows:
1. **Subject model response** — what the model actually says (refusal for jailbreak, recipe for cookies)
2. **PCD continuation** — free generation from encoded activations (reveals captured concepts)
3. **Probe outputs** — guided generation from "The text discusses...", "The main topic is..." etc.

### Custom Prompts

```bash
python demo_jailbreak.py --prompt "Your custom prompt here"
```

---

## 12. Expected Results

### Actual Training Metrics

From our 5000-step run on ~4.8M tokens:

| Step | Loss | Active Concepts | Dead Concepts |
|------|------|----------------|---------------|
| 50 | 5.95 | 8192 (100%) | 0 |
| 500 | 4.58 | 8192 (100%) | 0 |
| 1000 | 3.99 | 7672 (94%) | 520 |
| 2000 | 3.73 | 7987 (97%) | 205 |
| 3000 | 3.65 | 6976 (85%) | 1216 |
| 4000 | 3.55 | 6162 (75%) | 2030 |
| 5000 | 3.57 | 5757 (70%) | 2435 |

### Actual Demo Results

The PCD successfully captures **topic-level signal** from the subject model's activations:

| Input | Probe: "The main topic is..." | Domain Match? |
|-------|-------------------------------|---------------|
| Jailbreak (bomb) | "the development of a new type of gas weapon" | **Yes** (weapons) |
| Chemistry (explosives) | "the development of the new drugs" | **Yes** (chemistry) |
| Cookies (benign) | food/restaurant references | **Yes** (food) |

**Key observation:** The top concept indices are **completely different** across the three prompts, confirming that the encoder learned distinct sparse representations for different topics.

**What this demonstrates:**
1. The encoder learned to compress subject model activations into meaningful sparse concepts
2. Different topics activate different concept combinations
3. The decoder can reconstruct topic-relevant text from the sparse codes
4. The communication bottleneck (encoder doesn't see question, decoder doesn't see raw activations) forces general-purpose representations

**What we don't achieve at this scale:**
- The paper's 75% accuracy on structured QA tasks (requires SynthSys finetuning)
- Explicit articulation of "the model is thinking about weapons" (requires QA finetuning)
- Highly interpretable individual concepts (requires more pretraining data and auto-interp)

---

## 13. Troubleshooting

### CUDA Out of Memory

If you run out of VRAM:
1. Reduce `batch_size` to 8 (increase `grad_accum_steps` to 8)
2. Enable gradient checkpointing: add `decoder.model.gradient_checkpointing_enable()` before training
3. Use float16 instead of bfloat16 if your GPU doesn't support bf16

### Loss not decreasing

- Check that subject model activations have non-trivial values (not all zeros)
- Verify gradients flow to encoder: `encoder.W_enc.weight.grad` should be non-None after backward
- Try increasing learning rates (5e-4 for encoder, 3e-4 for decoder)

### All concepts dying

- Increase `aux_loss_coeff` (try 1e-3)
- Decrease `dead_concept_window` (try 500)
- Check that `TopK` is working correctly (exactly k=16 non-zero values per position)

### Decoder outputs are garbage

- Make sure the LoRA adapter is applied correctly (check `decoder.model.print_trainable_parameters()`)
- Verify soft tokens are in the right dtype (match the decoder's embedding dtype)
- Try generating from the decoder with just the question (no soft tokens) to verify LoRA didn't break it

### Slow training

- Ensure you're using bf16/fp16 mixed precision (check `torch.autocast` is active)
- Use `pin_memory=True` in DataLoader (already set)
- Reduce `num_workers` if CPU is bottleneck

---

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | ~50 | Central hyperparameter configuration |
| `model_subject.py` | ~70 | Frozen subject model + activation hooks |
| `model_encoder.py` | ~150 | Sparse encoder: W_enc → TopK → W_emb |
| `model_decoder.py` | ~150 | LoRA decoder with soft-token patching |
| `data.py` | ~100 | FineWeb streaming, tokenization, caching |
| `train_pretrain.py` | ~130 | Pretraining loop (encoder + decoder LoRA) |
| `inference.py` | ~100 | End-to-end PCD inference pipeline |
| `demo_jailbreak.py` | ~80 | Jailbreak demonstration script |
| `utils.py` | ~50 | Checkpointing and logging utilities |

---

## Key Differences from the Paper

1. **Model scale:** 1.5B vs 8B — smaller representations, fewer concepts
2. **Data scale:** ~5M vs 72M tokens — less concept diversity
3. **No finetuning:** We skip the SynthSys QA finetuning phase (could be added as Phase 8)
4. **No auto-interp evaluation:** We don't run the concept auto-labeling pipeline
5. **Same model for subject & decoder:** Both are Qwen2.5-1.5B-Instruct (paper also uses the same model for both, so this matches)

Despite these simplifications, the core PCD mechanism — sparse bottleneck between subject model activations and a decoder — is faithfully reproduced.
