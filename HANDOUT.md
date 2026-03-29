# Reproducing Predictive Concept Decoders (PCDs): A Complete Tutorial

> Based on *Predictive Concept Decoders: Training Scalable End-to-End Interpretability Assistants* (Huang et al., 2025, arXiv:2512.15712).
> This handout walks you through building a minimal, budget-friendly PCD from scratch using **Qwen2.5-1.5B-Instruct** as both subject and decoder model.

---

## Table of Contents

1. [What Are PCDs and Why Do They Matter](#1-what-are-pcds-and-why-do-they-matter)
2. [Architecture Overview](#2-architecture-overview)
3. [Environment Setup](#3-environment-setup)
4. [Project Structure](#4-project-structure)
5. [Step 1: Configuration (`config.py`)](#5-step-1-configuration)
6. [Step 2: Subject Model (`model_subject.py`)](#6-step-2-subject-model)
7. [Step 3: Sparse Encoder (`model_encoder.py`)](#7-step-3-sparse-encoder)
8. [Step 4: LoRA Decoder (`model_decoder.py`)](#8-step-4-lora-decoder)
9. [Step 5: Pretraining Data Pipeline (`data.py`)](#9-step-5-pretraining-data-pipeline)
10. [Step 6: Pretraining Loop (`train_pretrain.py`)](#10-step-6-pretraining-loop)
11. [Step 7: Fine-tuning Data Pipeline (`data_finetune.py`)](#11-step-7-fine-tuning-data-pipeline)
12. [Step 8: Fine-tuning Loop (`train_finetune.py`)](#12-step-8-fine-tuning-loop)
13. [Step 9: Inference Pipeline (`inference.py`)](#13-step-9-inference-pipeline)
14. [Step 10: Utilities (`utils.py`)](#14-step-10-utilities)
15. [Running the Pipeline End to End](#15-running-the-pipeline-end-to-end)
16. [Expected Results and Diagnostics](#16-expected-results-and-diagnostics)
17. [Key Bugs We Fixed (and You Must Too)](#17-key-bugs-we-fixed-and-you-must-too)
18. [Paper vs. This Reproduction](#18-paper-vs-this-reproduction)
19. [Glossary](#19-glossary)

---

## 1. What Are PCDs and Why Do They Matter

Traditional mechanistic interpretability tools like Sparse Autoencoders (SAEs) reconstruct a model's internal activations -- they learn to compress and decompress hidden states. PCDs take a fundamentally different approach: instead of reconstructing activations, they **predict future tokens** from those activations, forcing the encoder to learn *actionable*, *meaningful* concepts rather than just statistically faithful reconstructions.

The core idea:

```
"Don't just learn what the model's internal state looks like.
 Learn what it's going to DO with that state."
```

A PCD has three parts:
1. A **frozen subject model** (the LLM you want to interpret)
2. A **sparse encoder** that reads the subject model's layer-13 activations and compresses them into a small set of active "concepts"
3. A **decoder** (LoRA-adapted version of the same LLM) that receives those compressed concepts as soft tokens and must predict what comes next

The communication bottleneck -- the encoder can only pass k=16 sparse concept activations -- forces it to learn a genuine vocabulary of interpretable features.

---

## 2. Architecture Overview

```
                        INPUT TEXT
                           |
                    [Tokenize: 48 tokens]
                    [16 prefix | 16 middle | 16 suffix]
                           |
                    +------+------+
                    |             |
                    v             v
            SUBJECT MODEL     (suffix_ids saved
            (frozen Qwen2.5-1.5B)    as targets)
                    |
                    v
            Layer 13 activations
            [B, 16, 1536]
                    |
                    v
            SPARSE ENCODER
            +---------------------------+
            | 1. Center (subtract mean) |
            | 2. L2-normalize per token |
            | 3. W_enc: 1536 -> 8192    |
            | 4. TopK(k=16)             |
            | 5. W_emb: 8192 -> 1536    |
            +---------------------------+
                    |
                    v
            Soft tokens [B, 16, 1536]
                    |
                    v
            DECODER (Qwen2.5-1.5B + LoRA)
            [soft_tokens] + [suffix_embeds]
                    |
                    v
            Cross-entropy loss on suffix prediction
```

**Key design principles:**

- The encoder never sees the question/suffix. The decoder never sees raw activations. This is the **communication bottleneck** that forces learning.
- Only the encoder weights and decoder LoRA adapters are trained. The subject model and decoder base weights are frozen.
- TopK sparsity means exactly 16 out of 8192 concepts fire per token position -- the rest are hard zeros.

---

## 3. Environment Setup

### Hardware Requirements

- **GPU**: 16GB+ VRAM (e.g., RTX 4090, A100, or any GPU that can fit Qwen2.5-1.5B in bf16 twice -- once for subject, once for decoder)
- **RAM**: 32GB+ recommended (for data caching)
- **Disk**: ~20GB (model weights + cached data + checkpoints)

### Software Requirements

Create a file called `requirements.txt`:

```
torch>=2.1.0
transformers>=4.36.0
peft>=0.7.0
datasets>=2.16.0
accelerate>=0.25.0
tqdm
```

For fine-tuning data generation only (optional, requires API key):
```
anthropic
```

### Installation

```bash
# Create environment
python -m venv pcd-env
source pcd-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional, for fine-tuning data generation with LLM judge)
pip install anthropic
export ANTHROPIC_API_KEY="your-key-here"
```

### Directory Structure to Create

```bash
mkdir -p data_cache checkpoints checkpoints_finetune feature_annotations
```

---

## 4. Project Structure

When you are done, your project should have this structure:

```
repro-pcd/
  config.py                  # Central configuration dataclass
  model_subject.py           # Subject model + activation hooks
  model_encoder.py           # Sparse encoder (TopK bottleneck)
  model_decoder.py           # Decoder with LoRA + soft-token patching
  data.py                    # FineWeb streaming + tokenization
  data_finetune.py           # SynthSys QA data generation
  train_pretrain.py          # Stage 1 training loop
  train_finetune.py          # Stage 2 training loop
  inference.py               # End-to-end inference pipeline
  utils.py                   # Checkpointing + logging helpers
  requirements.txt
  data_cache/                # Cached tokenized data (auto-generated)
  checkpoints/               # Pretrain checkpoints (auto-generated)
  checkpoints_finetune/      # Finetune checkpoints (auto-generated)
```

Build each file in the order listed below. Each section gives you the exact code and explains every design decision.

---

## 5. Step 1: Configuration

**File: `config.py`**

This is your single source of truth for every hyperparameter. All other files import from here.

```python
"""Central configuration for the PCD reproduction."""

from dataclasses import dataclass, field
import torch


@dataclass
class PCDConfig:
    # ---------- Model identity ----------
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hidden_dim: int = 1536          # d -- Qwen2.5-1.5B hidden size
    num_layers: int = 28            # total transformer layers in the model

    # ---------- Encoder ----------
    num_concepts: int = 8192        # m -- concept dictionary size (paper uses 32768)
    topk: int = 16                  # k -- concepts active per token position
    l_read: int = 13                # layer to tap activations (~47% depth)

    # ---------- Decoder (LoRA) ----------
    l_write: int = 0                # layer to inject soft tokens (input embeddings)
    lora_r: int = 16                # LoRA rank
    lora_alpha: int = 32            # LoRA alpha (scaling = alpha/r = 2.0)
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",       # MLP
    ])

    # ---------- Data geometry ----------
    prefix_len: int = 16            # context the subject sees before the "interesting" part
    middle_len: int = 16            # the tokens whose activations we encode
    suffix_len: int = 16            # the tokens the decoder must predict
    total_window: int = 48          # prefix + middle + suffix

    # ---------- Pretraining (Stage 1) ----------
    lr_encoder: float = 3e-4        # encoder learns faster (new params)
    lr_decoder: float = 1e-4        # decoder learns slower (LoRA on pretrained weights)
    weight_decay: float = 0.01
    batch_size: int = 16
    grad_accum_steps: int = 4       # effective batch = 16 * 4 = 64
    max_train_steps: int = 5000
    warmup_steps: int = 500
    aux_loss_coeff: float = 1e-4    # dead-concept revival loss weight
    dead_concept_window: int = 1000 # steps without firing before a concept is "dead"

    # ---------- Fine-tuning (Stage 2) ----------
    finetune_steps: int = 4000
    finetune_warmup_steps: int = 400
    finetune_mix_ratio: float = 0.5     # 50% FineWeb batches (anti-forgetting)
    finetune_checkpoint_dir: str = "checkpoints_finetune"
    pretrain_checkpoint: str = "checkpoints/step_5000"

    # ---------- SynthSys QA data ----------
    synthsys_cache_path: str = "data_cache/synthsys_qa.json"
    synthsys_num_examples: int = 360  # 18 attrs × 4 values × 5 questions
    synthsys_max_response_tokens: int = 64
    synthsys_user_middle_len: int = 32

    # ---------- Inference ----------
    max_new_tokens: int = 128

    # ---------- System ----------
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    seed: int = 42

    # ---------- Paths / logging ----------
    data_cache_dir: str = "data_cache"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 50          # print metrics every N steps
    save_interval: int = 1000       # save checkpoint every N steps
```

### Key decisions explained

| Parameter | Value | Why |
|-----------|-------|-----|
| `l_read = 13` | 13 of 28 layers (~47%) | Paper reads at ~47% depth. Middle layers carry rich semantic representations but haven't yet committed to specific output tokens. |
| `num_concepts = 8192` | 8192 | Paper uses 32768, but 8192 is enough to demonstrate the mechanism at lower compute. |
| `topk = 16` | 16 | Same as the paper. Forces extreme sparsity: only 0.2% of concepts active per position. |
| `lr_encoder > lr_decoder` | 3e-4 vs 1e-4 | Encoder is randomly initialized and needs to learn fast. Decoder is already pretrained, just needs gentle adaptation via LoRA. |
| `grad_accum_steps = 4` | 4 | Simulates effective batch size of 64 while keeping memory usage at batch_size=16. |
| `prefix_len = middle_len = suffix_len = 16` | 16 each | Same as paper. The 48-token window is the atomic training unit. |

---

## 6. Step 2: Subject Model

**File: `model_subject.py`**

The subject model is the LLM you want to interpret. It is **completely frozen** -- you never update its weights. You run text through it and tap the hidden states at layer 13 using a PyTorch forward hook.

```python
"""Subject model wrapper with activation extraction via forward hooks."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import PCDConfig


class SubjectModel:
    """Wraps the subject LLM and extracts intermediate activations.

    The subject model is fully frozen. A forward hook on layer `l_read`
    captures hidden states so the encoder can compress them.
    """

    def __init__(self, config: PCDConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.dtype,
            trust_remote_code=True,
        ).to(self.device).eval()

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # Register hook to capture activations at l_read
        self._activations = None
        self._hook_handle = self.model.model.layers[config.l_read].register_forward_hook(
            self._capture_hook
        )

    def _capture_hook(self, module, input, output):
        """Forward hook that stores hidden states from the target layer."""
        # For Qwen2, layer output is (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            self._activations = output[0]
        else:
            self._activations = output

    @torch.no_grad()
    def get_middle_activations(
        self, input_ids: torch.Tensor, prefix_len: int, middle_len: int,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the subject model and return activations for middle token positions.

        Args:
            input_ids: [batch, prefix_len + middle_len] token IDs
            prefix_len: number of prefix tokens
            middle_len: number of middle tokens
            attention_mask: [batch, prefix_len + middle_len] optional mask
                (1 for real tokens, 0 for padding)

        Returns:
            Tensor of shape [batch, middle_len, hidden_dim]
        """
        self.model(input_ids, attention_mask=attention_mask)
        # Extract only the middle token positions
        acts = self._activations[:, prefix_len:prefix_len + middle_len, :]
        return acts.detach()

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 128) -> torch.Tensor:
        """Generate text from the subject model (for comparison in demos)."""
        return self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
```

### How the forward hook works

When you call `self.model(input_ids)`, PyTorch executes every layer sequentially. The hook we registered on `layers[13]` intercepts the output of that layer and stores it in `self._activations`. We then slice out just the "middle" positions (positions `prefix_len` through `prefix_len + middle_len`).

**Why only middle positions?** The prefix provides context to the subject model but we don't encode it. This creates a richer information bottleneck: the encoder must learn to compress what the model "knows" about the middle tokens given the prefix context.

**Critical: attention masks.** If your input has padding tokens, you MUST pass an attention mask. Without it, pad tokens corrupt the activations at real token positions via attention. During training we use fixed 48-token windows with no padding, so no mask is needed. During inference with variable-length inputs, we avoid padding entirely (see Section 13).

---

## 7. Step 3: Sparse Encoder

**File: `model_encoder.py`**

This is the core novelty of PCDs. The encoder maps high-dimensional activations (1536-d) into a sparse concept space (8192-d, only 16 active), then re-embeds back to 1536-d as "soft tokens" for the decoder.

```python
"""Sparse linear encoder with TopK bottleneck.

Implements the PCD encoder: a'(i) = W_emb(TopK(W_enc @ a(i) + b_enc))
"""

import torch
import torch.nn as nn

from config import PCDConfig


class SparseEncoder(nn.Module):
    """Sparse concept encoder.

    Maps activations from the subject model's hidden space (d) into a sparse
    concept space (m) via TopK, then re-embeds back to hidden space (d).
    """

    def __init__(self, config: PCDConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        m = config.num_concepts
        k = config.topk

        self.k = k
        self.m = m

        # W_enc: project from hidden dim to concept space
        # Shape: (m, d) as nn.Linear(d, m)
        self.W_enc = nn.Linear(d, m)

        # W_emb: re-embed sparse concepts back to hidden dim
        # Shape: (d, m) as nn.Linear(m, d, bias=False)
        self.W_emb = nn.Linear(m, d, bias=False)

        # Initialize: W_enc rows unit-normalized, W_emb = W_enc^T
        self._initialize_weights()

        # Running statistics for pre-encoder activation normalization
        self.register_buffer("running_mean", torch.zeros(d))
        self.register_buffer("running_var", torch.ones(d))
        self.register_buffer("n_samples", torch.tensor(0, dtype=torch.long))
        self.norm_momentum = 0.01

        # Dead concept tracking
        self.register_buffer(
            "concept_usage", torch.zeros(m, dtype=torch.long)
        )
        self.register_buffer(
            "steps_since_active", torch.zeros(m, dtype=torch.long)
        )
        self.total_steps = 0

    def _initialize_weights(self):
        """Initialize encoder weights with unit-norm rows, embed as transpose."""
        with torch.no_grad():
            # W_enc: random unit-norm rows
            nn.init.kaiming_uniform_(self.W_enc.weight)
            self.W_enc.weight.div_(
                self.W_enc.weight.norm(dim=1, keepdim=True) + 1e-8
            )
            # Bias initialized to zero (default)
            nn.init.zeros_(self.W_enc.bias)
            # W_emb initialized as W_enc^T
            self.W_emb.weight.copy_(self.W_enc.weight.t())

    def _normalize_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """Normalize activations via centering + per-token L2 normalization.

        Subtracts the running mean (learned during training) to center, then
        normalizes each token's activation vector to unit norm. This makes the
        encoder invariant to the global magnitude of activations, which varies
        drastically between short and long sequences.
        """
        if self.training:
            batch_mean = activations.mean(dim=(0, 1))  # [d]
            batch_var = activations.var(dim=(0, 1))     # [d]

            if self.n_samples == 0:
                self.running_mean.copy_(batch_mean)
                self.running_var.copy_(batch_var)
            else:
                self.running_mean.lerp_(batch_mean, self.norm_momentum)
                self.running_var.lerp_(batch_var, self.norm_momentum)
            self.n_samples += 1

        # Center using running mean, then L2-normalize each token vector
        centered = activations - self.running_mean
        return centered / (centered.norm(dim=-1, keepdim=True) + 1e-8)

    def forward(
        self, activations: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Encode activations through sparse bottleneck.

        Args:
            activations: [batch, seq_len, d] from subject model

        Returns:
            encoded: [batch, seq_len, d] re-embedded sparse concepts
            info: dict with auxiliary loss and diagnostics
        """
        B, T, d = activations.shape

        # Normalize before encoding so concept values stay in a reasonable range
        activations = self._normalize_activations(activations)

        # Project to concept space: [B, T, m]
        pre_act = self.W_enc(activations)

        # Keep only top-k activations per position
        top_vals, top_idx = torch.topk(pre_act, self.k, dim=-1)  # [B, T, k]

        # Create sparse representation and re-embed
        sparse = torch.zeros_like(pre_act)  # [B, T, m]
        sparse.scatter_(-1, top_idx, top_vals)

        # Re-embed: [B, T, d]
        encoded = self.W_emb(sparse)

        # Compute auxiliary loss for dead concept revival
        aux_loss = self._compute_aux_loss(pre_act, top_idx)

        # Track concept usage
        if self.training:
            self._update_concept_usage(top_idx)

        # Diagnostics
        n_active = (self.steps_since_active < self.config.dead_concept_window).sum().item()
        info = {
            "aux_loss": aux_loss,
            "n_active_concepts": n_active,
            "n_dead_concepts": self.m - n_active,
            "mean_top_val": top_vals.mean().item(),
        }

        return encoded, info

    def _compute_aux_loss(
        self, pre_act: torch.Tensor, top_idx: torch.Tensor
    ) -> torch.Tensor:
        """Auxiliary loss to revive dead concepts.

        Pushes pre-activations of dead concepts upward so they have a chance
        of entering the top-k.
        """
        dead_mask = self.steps_since_active >= self.config.dead_concept_window
        n_dead = dead_mask.sum().item()

        if n_dead == 0:
            return torch.tensor(0.0, device=pre_act.device, dtype=pre_act.dtype)

        # Get pre-activations of dead concepts
        dead_pre_act = pre_act[:, :, dead_mask]  # [B, T, n_dead]

        # Loss: negative mean (pushes values up)
        aux_loss = -dead_pre_act.mean() * self.config.aux_loss_coeff

        return aux_loss

    def _update_concept_usage(self, top_idx: torch.Tensor):
        """Track which concepts are being used."""
        self.total_steps += 1

        # Mark all concepts as one step older
        self.steps_since_active += 1

        # Reset counter for concepts that fired in this batch
        fired = top_idx.unique()
        self.steps_since_active[fired] = 0
        self.concept_usage[fired] += 1

    def get_top_concepts(
        self, activations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the top-k concept indices and values (for inspection).

        Args:
            activations: [batch, seq_len, d]

        Returns:
            top_vals: [batch, seq_len, k]
            top_idx: [batch, seq_len, k]
        """
        activations = self._normalize_activations(activations)
        pre_act = self.W_enc(activations)
        return torch.topk(pre_act, self.k, dim=-1)
```

### Encoder internals explained

**Weight initialization:** `W_enc` rows are unit-normalized random vectors -- each row represents a "concept direction" in activation space. `W_emb` starts as the transpose of `W_enc`, creating a natural encoder/decoder pair. This is the same initialization strategy used in SAEs.

**Normalization (CRITICAL):** Raw layer-13 activations vary in magnitude depending on sequence length and position. Without normalization, the encoder's soft token outputs can be 100-1000x larger than normal token embeddings, completely overwhelming the decoder. We center by subtracting the running mean, then L2-normalize each token vector to unit norm. The running mean uses exponential moving average (momentum=0.01) during training and is frozen during eval.

**TopK sparsity:** After projecting to 8192 dimensions, we keep only the top 16 values and zero out everything else. This is a hard gate, not a soft approximation. The gradient flows through the top-k values but not through the zeroed-out positions.

**Dead concept revival:** Some concepts may never enter the top-k and become permanently unused ("dead"). The auxiliary loss pushes their pre-activation values upward, giving them a chance to become active again. A concept is considered dead if it hasn't fired in `dead_concept_window` (1000) consecutive batches.

---

## 8. Step 4: LoRA Decoder

**File: `model_decoder.py`**

The decoder is a second copy of Qwen2.5-1.5B with LoRA adapters. It receives the encoder's soft tokens as input embeddings and must predict the next tokens.

```python
"""Decoder model with LoRA adapter and soft-token patching."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from config import PCDConfig


class PCDDecoder(nn.Module):
    """Decoder that reads sparse-encoded activations via soft tokens.

    The encoder's output is injected as soft tokens at the start of the
    decoder's input, followed by regular text tokens (suffix during training,
    question during inference). The decoder uses LoRA for parameter efficiency.
    """

    def __init__(self, config: PCDConfig):
        super().__init__()
        self.config = config

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.dtype,
            trust_remote_code=True,
        )

        # Apply LoRA
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(base_model, lora_config)

        # Tokenizer (shared with subject)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_token_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get the decoder's token embeddings for regular text tokens."""
        return self.model.get_input_embeddings()(token_ids)

    def forward_train(
        self,
        soft_tokens: torch.Tensor,
        suffix_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Training forward pass: predict suffix tokens given soft tokens.

        Layout:
            input:  [soft_0, soft_1, ..., soft_15, suf_0, suf_1, ..., suf_14]
            target: [                              suf_0, suf_1, ..., suf_15]

        The last soft token predicts suf_0; suf_14 predicts suf_15.
        We exclude suf_15 from input because nothing follows it to predict.

        Args:
            soft_tokens: [B, middle_len, d] encoded activations from encoder
            suffix_ids: [B, suffix_len] target suffix token IDs

        Returns:
            loss: scalar cross-entropy loss on suffix prediction
        """
        B, n_soft, d = soft_tokens.shape

        # Get embeddings for suffix tokens (teacher-forced input: all but last)
        suffix_embeds = self.get_token_embeddings(suffix_ids[:, :-1])

        # Concatenate: [soft_tokens] + [suffix_embeds]
        inputs_embeds = torch.cat([soft_tokens, suffix_embeds], dim=1)

        total_len = inputs_embeds.shape[1]
        attention_mask = torch.ones(B, total_len, device=inputs_embeds.device, dtype=torch.long)

        # Forward pass
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # [B, total_len, vocab_size]

        # Loss on suffix positions only
        # Position (n_soft - 1) predicts suf_0, position (n_soft) predicts suf_1, etc.
        suffix_logits = logits[:, n_soft - 1:, :]  # [B, suffix_len, vocab_size]
        targets = suffix_ids  # [B, suffix_len]

        loss = nn.functional.cross_entropy(
            suffix_logits.reshape(-1, suffix_logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )

        return loss

    def forward_train_qa(
        self,
        soft_tokens: torch.Tensor,
        question_ids: torch.Tensor,
        answer_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Fine-tuning forward pass: predict answer given soft tokens + question.

        Layout:
            input:  [soft_tokens] + [question_embeds] + [answer_embeds[:-1]]
            target: answer_ids (loss computed only on answer positions)

        Args:
            soft_tokens: [B, middle_len, d] from frozen encoder
            question_ids: [B, q_len] question token IDs
            answer_ids: [B, a_len] answer token IDs (targets)

        Returns:
            loss: scalar cross-entropy on answer prediction
        """
        B, n_soft, d = soft_tokens.shape
        q_len = question_ids.shape[1]
        a_len = answer_ids.shape[1]

        q_embeds = self.get_token_embeddings(question_ids)
        a_embeds = self.get_token_embeddings(answer_ids[:, :-1])

        inputs_embeds = torch.cat([soft_tokens, q_embeds, a_embeds], dim=1)

        total_len = inputs_embeds.shape[1]
        attention_mask = torch.ones(B, total_len, device=inputs_embeds.device, dtype=torch.long)

        # Mask padding in question
        q_pad_mask = (question_ids != self.tokenizer.pad_token_id).long()
        attention_mask[:, n_soft:n_soft + q_len] = q_pad_mask

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        # Loss only on answer positions
        answer_start = n_soft + q_len - 1
        answer_logits = logits[:, answer_start:answer_start + a_len, :]

        loss = nn.functional.cross_entropy(
            answer_logits.reshape(-1, answer_logits.size(-1)),
            answer_ids.reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )

        return loss

    @torch.no_grad()
    def generate_from_soft_tokens(
        self,
        soft_tokens: torch.Tensor,
        prompt_ids: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        repetition_penalty: float = 1.3,
    ) -> list[str]:
        """Generate text given soft tokens and an optional text prompt.

        Uses KV-cache for efficient autoregressive generation.
        Repetition penalty (1.3) prevents degenerate repetitive output.

        Args:
            soft_tokens: [B, n_soft, d] encoded activations
            prompt_ids: [B, q_len] optional text prompt (None = free generation)
            max_new_tokens: maximum tokens to generate
            repetition_penalty: >1.0 discourages repetition

        Returns:
            List of generated strings (one per batch element)
        """
        B = soft_tokens.shape[0]

        if prompt_ids is not None:
            q_embeds = self.get_token_embeddings(prompt_ids)
            inputs_embeds = torch.cat([soft_tokens, q_embeds], dim=1)
        else:
            inputs_embeds = soft_tokens

        total_len = inputs_embeds.shape[1]
        attention_mask = torch.ones(B, total_len, device=inputs_embeds.device, dtype=torch.long)

        # First forward: process all prefix (soft + optional question)
        generated_ids = []
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token)

        # Autoregressive loop with KV cache
        for _ in range(max_new_tokens - 1):
            curr_attention_mask = torch.ones(
                B, total_len + len(generated_ids),
                device=soft_tokens.device, dtype=torch.long,
            )
            outputs = self.model(
                input_ids=next_token,
                attention_mask=curr_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0 and len(generated_ids) > 0:
                prev_tokens = torch.cat(generated_ids, dim=1)
                for b in range(B):
                    for token_id in prev_tokens[b].unique():
                        if next_token_logits[b, token_id] > 0:
                            next_token_logits[b, token_id] /= repetition_penalty
                        else:
                            next_token_logits[b, token_id] *= repetition_penalty

            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token)

            if (next_token == self.tokenizer.eos_token_id).all():
                break

        generated = torch.cat(generated_ids, dim=1)
        texts = [
            self.tokenizer.decode(generated[i], skip_special_tokens=True)
            for i in range(B)
        ]
        return texts
```

### Decoder design decisions

**Why LoRA, not full fine-tuning?** The decoder needs to understand the soft tokens while retaining its language modeling ability. Full fine-tuning on 1.5B params would be expensive and risks catastrophic forgetting. LoRA adds only ~18.5M trainable parameters (~1.2% of total) while keeping the base model frozen.

**LoRA targets all projections:** We adapt `q_proj, k_proj, v_proj, o_proj` (attention) and `gate_proj, up_proj, down_proj` (MLP) at every layer. This gives the decoder maximum flexibility to reinterpret soft tokens.

**Soft-token patching:** The encoder's output is directly used as `inputs_embeds` at the decoder's input layer. This is "layer 0 patching" -- the simplest and most effective approach. The soft tokens occupy the first 16 positions, followed by regular token embeddings.

**Repetition penalty:** Without it, the decoder often falls into degenerate loops during generation. A penalty of 1.3 divides positive logits by 1.3 for already-generated tokens, gently discouraging repetition.

---

## 9. Step 5: Pretraining Data Pipeline

**File: `data.py`**

We stream the FineWeb dataset (a large, clean web text corpus) and chunk it into fixed-size 48-token windows.

```python
"""Data loading and preprocessing for PCD pretraining.

Streams FineWeb, tokenizes, and chunks into prefix/middle/suffix windows.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from config import PCDConfig


class FineWebDataset(Dataset):
    """Pre-tokenized FineWeb dataset chunked into 48-token windows."""

    def __init__(self, token_windows: list[torch.Tensor], config: PCDConfig):
        self.windows = token_windows
        self.config = config

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]  # [48]
        prefix = window[: self.config.prefix_len]
        middle = window[self.config.prefix_len : self.config.prefix_len + self.config.middle_len]
        suffix = window[self.config.prefix_len + self.config.middle_len :]
        return {
            "prefix_ids": prefix,
            "middle_ids": middle,
            "suffix_ids": suffix,
        }


def prepare_data(
    config: PCDConfig,
    num_examples: int = 100_000,
    cache_path: str | None = None,
) -> FineWebDataset:
    """Stream FineWeb, tokenize, and chunk into training windows.

    The first run streams from HuggingFace and caches tokenized windows
    to disk. Subsequent runs load from cache instantly.

    Args:
        config: PCD configuration
        num_examples: approximate number of 48-token windows to create
        cache_path: path to cache the tokenized windows

    Returns:
        FineWebDataset ready for DataLoader
    """
    if cache_path is None:
        cache_path = os.path.join(config.data_cache_dir, "fineweb_windows.pt")

    # Check cache
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        windows = torch.load(cache_path, weights_only=True)
        print(f"Loaded {len(windows)} windows ({len(windows) * config.total_window / 1e6:.1f}M tokens)")
        return FineWebDataset(windows, config)

    print("Preparing FineWeb data (streaming)...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    # Stream FineWeb
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    windows = []
    total_tokens = 0
    window_size = config.total_window

    for example in tqdm(ds, desc="Tokenizing FineWeb", total=num_examples):
        if len(windows) >= num_examples:
            break

        text = example["text"]
        if len(text.strip()) < 50:
            continue

        # Tokenize without special tokens (we want raw text tokens)
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        if len(token_ids) < window_size:
            continue

        # Chunk into non-overlapping 48-token windows
        for start in range(0, len(token_ids) - window_size + 1, window_size):
            window = torch.tensor(token_ids[start : start + window_size], dtype=torch.long)
            windows.append(window)
            total_tokens += window_size

            if len(windows) >= num_examples:
                break

    print(f"Created {len(windows)} windows ({total_tokens / 1e6:.1f}M tokens)")

    # Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(windows, cache_path)
    print(f"Cached to {cache_path}")

    return FineWebDataset(windows, config)


def get_dataloader(dataset: FineWebDataset, config: PCDConfig, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader for the FineWeb dataset."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
```

### Data pipeline details

**Source:** HuggingFace FineWeb `sample-10BT` -- a 10-billion-token sample of high-quality web text. We stream it so we don't need to download the whole thing.

**Chunking:** Each document is tokenized and split into non-overlapping 48-token windows. Documents shorter than 48 tokens are skipped. We use `add_special_tokens=False` because we want raw text continuations, not chat-formatted inputs.

**Window anatomy:**
```
[token_0 ... token_15 | token_16 ... token_31 | token_32 ... token_47]
       prefix (16)          middle (16)             suffix (16)
```
- **Prefix:** Context fed to the subject model alongside middle tokens. Never encoded.
- **Middle:** The tokens whose layer-13 activations we extract and encode.
- **Suffix:** The decoder's prediction target. Never seen by subject or encoder.

**Caching:** The first run takes ~30-60 minutes to stream and tokenize 100k windows (~4.8M tokens). The result is cached to `data_cache/fineweb_windows.pt` and subsequent runs load instantly.

**Scale comparison:** The paper uses ~72M tokens. We use ~4.8M (15x less). This is sufficient to demonstrate the mechanism but produces a less capable encoder.

---

## 10. Step 6: Pretraining Loop

**File: `train_pretrain.py`**

This is Stage 1: jointly train the encoder and decoder LoRA on next-token prediction.

```python
"""Pretraining loop for PCD.

Jointly trains the encoder and decoder (LoRA) on FineWeb next-token prediction.
The subject model is frozen.
"""

import os
import math
import torch
from tqdm import tqdm

from config import PCDConfig
from model_subject import SubjectModel
from model_encoder import SparseEncoder
from model_decoder import PCDDecoder
from data import prepare_data, get_dataloader
from utils import save_checkpoint, log_metrics


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(config: PCDConfig | None = None):
    if config is None:
        config = PCDConfig()

    torch.manual_seed(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ---- Load models ----
    print("Loading subject model...")
    subject = SubjectModel(config)

    print("Loading encoder...")
    encoder = SparseEncoder(config).to(config.device).to(config.dtype)

    print("Loading decoder with LoRA...")
    decoder = PCDDecoder(config)
    decoder.model.to(config.device)

    # Print parameter counts
    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.model.parameters() if p.requires_grad)
    print(f"Encoder params: {enc_params:,}")
    print(f"Decoder trainable (LoRA) params: {dec_params:,}")

    # ---- Data ----
    print("Preparing data...")
    dataset = prepare_data(config)
    dataloader = get_dataloader(dataset, config)

    # ---- Optimizer ----
    # Two parameter groups with different learning rates
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": config.lr_encoder},
            {"params": [p for p in decoder.model.parameters() if p.requires_grad],
             "lr": config.lr_decoder},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_steps, config.max_train_steps
    )

    # ---- Training loop ----
    print(f"\nStarting pretraining for {config.max_train_steps} steps...")
    print(f"Effective batch size: {config.batch_size * config.grad_accum_steps}")

    encoder.train()
    decoder.model.train()
    global_step = 0
    accum_loss = 0.0
    accum_aux_loss = 0.0

    while global_step < config.max_train_steps:
        for batch in dataloader:
            if global_step >= config.max_train_steps:
                break

            prefix_ids = batch["prefix_ids"].to(config.device)
            middle_ids = batch["middle_ids"].to(config.device)
            suffix_ids = batch["suffix_ids"].to(config.device)

            # 1. Subject model: get layer-13 activations (frozen, no grad)
            subject_input = torch.cat([prefix_ids, middle_ids], dim=1)
            with torch.no_grad():
                activations = subject.get_middle_activations(
                    subject_input, config.prefix_len, config.middle_len
                )

            # 2. Encoder + decoder forward under mixed precision
            with torch.autocast(device_type="cuda", dtype=config.dtype):
                encoded, enc_info = encoder(activations)
                loss = decoder.forward_train(encoded, suffix_ids)
                aux_loss = enc_info["aux_loss"]
                total_loss = (loss + aux_loss) / config.grad_accum_steps

            # 3. Backward
            total_loss.backward()

            accum_loss += loss.item()
            accum_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss

            # 4. Optimizer step every grad_accum_steps
            if (global_step + 1) % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + [p for p in decoder.model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            # Logging
            if global_step % config.log_interval == 0:
                avg_loss = accum_loss / config.log_interval
                avg_aux = accum_aux_loss / config.log_interval
                log_metrics(global_step, {
                    "loss": avg_loss,
                    "aux_loss": avg_aux,
                    "active_concepts": enc_info["n_active_concepts"],
                    "dead_concepts": enc_info["n_dead_concepts"],
                    "lr_enc": scheduler.get_last_lr()[0],
                    "lr_dec": scheduler.get_last_lr()[1],
                })
                accum_loss = 0.0
                accum_aux_loss = 0.0

            # Checkpointing
            if global_step % config.save_interval == 0:
                save_checkpoint(encoder, decoder, optimizer, global_step, loss.item(), config)

    # Final save
    save_checkpoint(encoder, decoder, optimizer, global_step, loss.item(), config)
    print(f"\nPretraining complete! {global_step} steps.")


if __name__ == "__main__":
    train()
```

### Training loop walkthrough

Each training step does the following:

```
1. Load batch: {prefix_ids, middle_ids, suffix_ids} each [16, 16]
2. Concatenate prefix + middle -> [16, 32] and feed to subject model
3. Extract layer-13 activations at middle positions -> [16, 16, 1536]
4. Encoder: normalize -> W_enc -> TopK(16) -> W_emb -> soft tokens [16, 16, 1536]
5. Decoder: [soft_tokens | suffix_embeds[:-1]] -> logits -> CE loss on suffix
6. Add aux_loss (dead concept revival)
7. Backward, accumulate gradients
8. Every 4 micro-steps: clip gradients (norm=1.0), optimizer step, LR schedule step
```

**Gradient flow:** Gradients flow from the decoder loss through `W_emb`, through the TopK gate (for the 16 active concepts only), through `W_enc`, but NOT through the subject model (which is frozen under `torch.no_grad()`).

**Mixed precision:** We use `torch.autocast` with bfloat16 for the encoder and decoder forward passes. This halves memory for activations and speeds up matmuls.

**Learning rate schedule:** Linear warmup for 500 steps (helps avoid early instability), then cosine decay to 0 over the remaining 4500 steps.

---

## 11. Step 7: Fine-tuning Data Pipeline

**File: `data_finetune.py`**

This generates SynthSys-style QA pairs. The idea: prime the subject model with a system prompt like "The user is Vegan", ask a user question like "What should I make for dinner?", and check whether the subject model's response is consistent with the attribute. If yes, we build a multiple-choice question for the decoder.

> **Note:** This file requires an `ANTHROPIC_API_KEY` environment variable to run the consistency judge (Claude Haiku). If you don't have API access, you can skip data generation and use a pre-cached `data_cache/synthsys_qa.json` file.

```python
"""SynthSys-style QA data for PCD fine-tuning (Stage 2).

Generates user-attribute QA pairs following the SynthSys(8B) approach:
  1. Define 18 user attributes with 4 values each (A/B/C/D MCQ)
  2. For each (attribute, value, question) triple:
     - Format: <SYSTEM> The user is {value} <USER> {question}
     - Generate subject model response
     - LLM-judge consistency check
     - If consistent: build MCQ and add to dataset
  3. The decoder learns to decode the subject model's latent user model
"""

import os
import json
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import anthropic

from config import PCDConfig


# 18 training attributes, each with 4 values for A/B/C/D multiple choice
TRAIN_ATTRIBUTES = {
    "ethnicity": ["White", "Black", "Asian", "Hispanic"],
    "profession": ["Engineer", "Teacher", "Doctor", "Artist"],
    "dietary restriction": ["Vegan", "Vegetarian", "Omnivore", "Pescatarian"],
    "religion": ["Christian", "Muslim", "Atheist", "Buddhist"],
    "education level": ["High school", "Bachelor's degree", "Master's degree", "PhD"],
    "income level": ["Low income", "Middle income", "High income", "Very high income"],
    "marital status": ["Single", "Married", "Divorced", "Widowed"],
    "political leaning": ["Liberal", "Conservative", "Moderate", "Libertarian"],
    "primary hobby": ["Sports", "Reading", "Gaming", "Cooking"],
    "personality type": ["Introverted", "Extroverted", "Ambivert", "Reserved"],
    "fitness level": ["Sedentary", "Moderately active", "Very active", "Athletic"],
    "living situation": ["Urban apartment", "Suburban house", "Rural area", "College dorm"],
    "pet ownership": ["Dog owner", "Cat owner", "No pets", "Multiple pets"],
    "music preference": ["Pop", "Rock", "Classical", "Hip-hop"],
    "travel experience": ["Rarely travels", "Domestic traveler", "International traveler", "Digital nomad"],
    "tech savviness": ["Beginner", "Intermediate", "Advanced", "Expert"],
    "parental status": ["No children", "Young children", "Teenage children", "Adult children"],
    "sleep schedule": ["Early bird", "Night owl", "Irregular schedule", "Standard schedule"],
}

# 2 held-out attributes for evaluation
EVAL_ATTRIBUTES = {
    "gender": ["Female", "Male", "Non-binary", "Prefer not to say"],
    "age": ["18-25", "26-40", "41-60", "Over 60"],
}

# 5 questions per attribute designed to elicit attribute-revealing responses
USER_QUESTIONS = {
    "ethnicity": [
        "What are some traditional dishes I should learn to cook?",
        "I'm looking for cultural events to attend this weekend, any ideas?",
        "What holidays should I prepare for this year?",
        "Can you recommend some music from my cultural background?",
        "What's a meaningful gift I could give to my family?",
    ],
    "profession": [
        "I'm stressed about work, any advice?",
        "What skills should I develop for career growth?",
        "How should I handle a difficult colleague?",
        "What's the best way to organize my workday?",
        "I need to give a presentation tomorrow, any tips?",
    ],
    # ... (5 questions per attribute for all 18 attributes)
    # See the full source file for complete question lists.
    # Each question is designed to naturally elicit responses that
    # reflect the user's attribute when the system prompt primes the model.
}
```

The data generation pipeline has these helper functions:

### `_format_chat_input(tokenizer, system_prompt, user_message)`
Tokenizes the system + user prompt using Qwen's chat template and returns separate token ID lists so we can identify which positions correspond to the user's message (the part we want to encode).

### `_generate_response(subject_model, tokenizer, system_ids, user_ids, ...)`
Feeds the chat-formatted input to the frozen subject model and generates a response.

### `_check_consistency(client, attribute_name, attribute_value, user_question, response)`
Sends the response to Claude Haiku and asks: "Is this response consistent with the assumption that the user is {value}?" Returns True/False. This filters out cases where the subject model ignores the system prompt.

### `_build_mcq(attribute_name, values, correct_value)`
Builds a multiple-choice question like:
```
What does the model think the dietary restriction of the user is?
(A) Vegan (B) Vegetarian (C) Omnivore (D) Pescatarian
Answer:
```
With the correct answer being e.g., " A".

### `SynthSysDataset` and collation

The dataset returns `{system_ids, user_ids, question_ids, answer_ids}` per example. System IDs vary slightly in length across attributes, so the custom collate function right-pads them to the batch maximum.

---

## 12. Step 8: Fine-tuning Loop

**File: `train_finetune.py`**

Stage 2: freeze the encoder, continue training decoder LoRA on QA pairs, mixing in 50% FineWeb to prevent forgetting.

```python
"""Fine-tuning loop for PCD (Stage 2).

Freezes the encoder and fine-tunes the decoder (LoRA) on QA pairs,
mixing in 50% FineWeb next-token prediction to prevent forgetting.
"""

import os
import math
import random
import torch
from tqdm import tqdm
from peft import PeftModel

from config import PCDConfig
from model_subject import SubjectModel
from model_encoder import SparseEncoder
from model_decoder import PCDDecoder
from data import prepare_data, get_dataloader
from data_finetune import prepare_qa_data, get_qa_dataloader
from utils import log_metrics


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_finetune_checkpoint(decoder, optimizer, step, loss, config):
    """Save decoder LoRA and optimizer state for fine-tuning."""
    ckpt_dir = os.path.join(config.finetune_checkpoint_dir, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    decoder.model.save_pretrained(os.path.join(ckpt_dir, "decoder_lora"))
    torch.save(
        {"optimizer": optimizer.state_dict(), "step": step, "loss": loss},
        os.path.join(ckpt_dir, "optimizer.pt"),
    )
    print(f"Fine-tune checkpoint saved at step {step} to {ckpt_dir}")


def train(config: PCDConfig | None = None):
    if config is None:
        config = PCDConfig()

    torch.manual_seed(config.seed)
    os.makedirs(config.finetune_checkpoint_dir, exist_ok=True)

    # ---- Load models ----
    print("Loading subject model...")
    subject = SubjectModel(config)

    # Load FROZEN encoder from pretrain checkpoint
    print("Loading encoder from pretrain checkpoint...")
    encoder = SparseEncoder(config).to(config.device).to(config.dtype)
    encoder_path = os.path.join(config.pretrain_checkpoint, "encoder.pt")
    encoder.load_state_dict(
        torch.load(encoder_path, map_location=config.device, weights_only=True)
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Load decoder with PRETRAINED LoRA (continue training)
    print("Loading decoder with pretrained LoRA...")
    decoder = PCDDecoder(config)
    decoder_lora_path = os.path.join(config.pretrain_checkpoint, "decoder_lora")
    decoder.model = PeftModel.from_pretrained(
        decoder.model.get_base_model(),
        decoder_lora_path,
        torch_dtype=config.dtype,
        is_trainable=True,  # Keep LoRA trainable
    ).to(config.device)

    # ---- Data ----
    print("Preparing SynthSys QA fine-tuning data...")
    qa_dataset = prepare_qa_data(config, subject)
    qa_dataloader = get_qa_dataloader(qa_dataset, config)

    print("Preparing FineWeb data for mixing...")
    fineweb_dataset = prepare_data(config)
    fineweb_dataloader = get_dataloader(fineweb_dataset, config)

    # ---- Optimizer (decoder LoRA only) ----
    optimizer = torch.optim.AdamW(
        [p for p in decoder.model.parameters() if p.requires_grad],
        lr=config.lr_decoder,
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.finetune_warmup_steps, config.finetune_steps
    )

    # ---- Training loop ----
    decoder.model.train()
    global_step = 0
    qa_iter = iter(qa_dataloader)
    fineweb_iter = iter(fineweb_dataloader)

    while global_step < config.finetune_steps:
        # Coin flip: 50% FineWeb, 50% QA
        is_fineweb = random.random() < config.finetune_mix_ratio

        if is_fineweb:
            # Standard pretraining objective (prevents catastrophic forgetting)
            try:
                batch = next(fineweb_iter)
            except StopIteration:
                fineweb_iter = iter(fineweb_dataloader)
                batch = next(fineweb_iter)

            prefix_ids = batch["prefix_ids"].to(config.device)
            middle_ids = batch["middle_ids"].to(config.device)
            suffix_ids = batch["suffix_ids"].to(config.device)

            subject_input = torch.cat([prefix_ids, middle_ids], dim=1)
            with torch.no_grad():
                activations = subject.get_middle_activations(
                    subject_input, config.prefix_len, config.middle_len
                )
                encoded, _ = encoder(activations)

            with torch.autocast(device_type="cuda", dtype=config.dtype):
                loss = decoder.forward_train(encoded, suffix_ids)
                loss = loss / config.grad_accum_steps
            loss.backward()

        else:
            # QA objective (the new capability we're teaching)
            try:
                batch = next(qa_iter)
            except StopIteration:
                qa_iter = iter(qa_dataloader)
                batch = next(qa_iter)

            system_ids = batch["system_ids"].to(config.device)
            user_ids = batch["user_ids"].to(config.device)
            question_ids = batch["question_ids"].to(config.device)
            answer_ids = batch["answer_ids"].to(config.device)

            # Subject sees [system + user], encoder gets user-position activations
            subject_input = torch.cat([system_ids, user_ids], dim=1)
            sys_len = system_ids.shape[1]
            usr_len = user_ids.shape[1]

            with torch.no_grad():
                activations = subject.get_middle_activations(
                    subject_input, sys_len, usr_len
                )
                encoded, _ = encoder(activations)

            with torch.autocast(device_type="cuda", dtype=config.dtype):
                loss = decoder.forward_train_qa(encoded, question_ids, answer_ids)
                loss = loss / config.grad_accum_steps
            loss.backward()

        # Gradient step
        if (global_step + 1) % config.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in decoder.model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        # Logging and checkpointing (same pattern as pretraining)
        if global_step % config.save_interval == 0:
            save_finetune_checkpoint(decoder, optimizer, global_step, loss.item(), config)

    save_finetune_checkpoint(decoder, optimizer, global_step, loss.item(), config)
    print(f"\nFine-tuning complete! {global_step} steps.")


if __name__ == "__main__":
    train()
```

### Fine-tuning key differences from pretraining

| Aspect | Pretraining | Fine-tuning |
|--------|-------------|-------------|
| Encoder | Training | **Frozen** |
| Decoder LoRA | Training (from random init) | Training (from pretrained checkpoint) |
| Data | 100% FineWeb | **50% FineWeb + 50% QA** |
| Objective | Next-token prediction | **Next-token + QA answer prediction** |
| Optimizer | Encoder + Decoder params | **Decoder params only** |
| Steps | 5000 | 4000 |

**Why mix FineWeb during fine-tuning?** Without it, the decoder rapidly forgets its general language modeling ability and becomes useless for free-form generation. The 50/50 mix maintains the pretraining capability while adding the QA skill.

**Why freeze the encoder?** The encoder's concept vocabulary was learned during pretraining. If we continued training it on QA data, it would overfit to the 18 user attributes and lose its general-purpose concept structure.

---

## 13. Step 9: Inference Pipeline

**File: `inference.py`**

The inference pipeline ties everything together for running PCD on arbitrary text.

```python
"""End-to-end PCD inference pipeline."""

import json
import os
import torch
from transformers import AutoTokenizer
from peft import PeftModel

from config import PCDConfig
from model_subject import SubjectModel
from model_encoder import SparseEncoder
from model_decoder import PCDDecoder


class PCDPipeline:
    """Run the full PCD pipeline: subject -> encoder -> decoder."""

    def __init__(
        self,
        config: PCDConfig,
        encoder_path: str,
        decoder_lora_path: str,
        feature_labels_path: str | None = "feature_annotations/concept_labels.json",
    ):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load all three components
        self.subject = SubjectModel(config)

        self.encoder = SparseEncoder(config).to(config.device).to(config.dtype)
        self.encoder.load_state_dict(
            torch.load(encoder_path, map_location=config.device, weights_only=True)
        )
        self.encoder.eval()

        self.decoder = PCDDecoder(config)
        self.decoder.model = PeftModel.from_pretrained(
            self.decoder.model.get_base_model(),
            decoder_lora_path,
            torch_dtype=config.dtype,
        ).to(config.device)
        self.decoder.model.eval()

        # Optional: load concept labels for interpretability
        self.feature_labels = {}
        if feature_labels_path and os.path.exists(feature_labels_path):
            with open(feature_labels_path) as f:
                raw = json.load(f)
            for k, v in raw.items():
                self.feature_labels[int(k)] = v.get("label", "unlabelled")

    def _encode_input(self, input_text: str):
        """Tokenize, run subject model, encode -- NO PADDING.

        Critical: we feed raw tokens without any padding. This avoids
        the train/inference distribution mismatch that caused Bug #4.
        """
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)
        n_tokens = len(input_ids)
        input_tensor = torch.tensor([input_ids], device=self.config.device)

        # Use the last middle_len tokens as "middle", rest is prefix
        middle_len = min(n_tokens, self.config.middle_len)
        prefix_len = n_tokens - middle_len

        activations = self.subject.get_middle_activations(
            input_tensor, prefix_len, middle_len,
        )
        encoded, enc_info = self.encoder(activations)
        top_vals, top_idx = self.encoder.get_top_concepts(activations)

        return encoded, enc_info, top_vals, top_idx

    @torch.no_grad()
    def __call__(self, input_text: str, max_new_tokens: int | None = None) -> dict:
        """Run PCD: returns continuation, probe outputs, and concept info."""
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        encoded, enc_info, top_vals, top_idx = self._encode_input(input_text)

        # Free generation from soft tokens
        pcd_continuation = self.decoder.generate_from_soft_tokens(
            encoded, prompt_ids=None, max_new_tokens=max_new_tokens,
        )

        # Probed generation with short prefixes
        probes = ["The text discusses", "This passage is about", "The main topic is"]
        probe_outputs = {}
        for probe in probes:
            probe_ids = self.tokenizer.encode(probe, add_special_tokens=False)
            probe_tensor = torch.tensor([probe_ids], device=self.config.device)
            output = self.decoder.generate_from_soft_tokens(
                encoded, probe_tensor, max_new_tokens=64,
            )
            probe_outputs[probe] = output[0]

        # Subject model's direct response (for comparison)
        full_input = self.tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
        if full_input.ndim == 1:
            full_input = full_input.unsqueeze(0)
        full_input = full_input.to(self.config.device)
        subject_gen = self.subject.generate(full_input, max_new_tokens=max_new_tokens)
        subject_response = self.tokenizer.decode(
            subject_gen[0][full_input.shape[1]:], skip_special_tokens=True
        )

        return {
            "pcd_continuation": pcd_continuation[0],
            "probe_outputs": probe_outputs,
            "subject_response": subject_response,
            "top_concept_indices": top_idx[0].cpu().tolist(),
            "top_concept_values": top_vals[0].cpu().tolist(),
            "n_active_concepts": enc_info["n_active_concepts"],
        }

    @torch.no_grad()
    def ask(self, input_text: str, question: str, max_new_tokens: int = 32) -> dict:
        """Ask a question about encoded activations (post fine-tuning)."""
        encoded, enc_info, top_vals, top_idx = self._encode_input(input_text)

        q_ids = self.tokenizer.encode(question, add_special_tokens=False)
        q_tensor = torch.tensor([q_ids], device=self.config.device)
        answer = self.decoder.generate_from_soft_tokens(
            encoded, q_tensor, max_new_tokens=max_new_tokens,
        )

        return {
            "question": question,
            "answer": answer[0],
            "top_concept_indices": top_idx[0].cpu().tolist(),
            "top_concept_values": top_vals[0].cpu().tolist(),
        }
```

### Inference modes

**Mode 1 -- Free continuation (pretrained model):** Give the decoder only soft tokens and let it generate. The output reveals what semantic information the encoder captured from the subject model's activations.

**Mode 2 -- Probed continuation (pretrained model):** Give the decoder soft tokens + a short text probe like "The main topic is" and let it complete. This steers the generation toward specific types of information.

**Mode 3 -- QA (fine-tuned model):** Give the decoder soft tokens + a structured question and let it answer. Used for attribute probing after Stage 2 fine-tuning.

---

## 14. Step 10: Utilities

**File: `utils.py`**

```python
"""Shared utilities for logging and checkpointing."""

import os
import torch
from config import PCDConfig


def save_checkpoint(encoder, decoder, optimizer, step: int, loss: float, config: PCDConfig):
    """Save encoder, decoder LoRA, and optimizer state."""
    ckpt_dir = os.path.join(config.checkpoint_dir, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(encoder.state_dict(), os.path.join(ckpt_dir, "encoder.pt"))
    decoder.model.save_pretrained(os.path.join(ckpt_dir, "decoder_lora"))
    torch.save(
        {"optimizer": optimizer.state_dict(), "step": step, "loss": loss},
        os.path.join(ckpt_dir, "optimizer.pt"),
    )
    print(f"Checkpoint saved at step {step} to {ckpt_dir}")


def load_checkpoint(encoder, decoder_model, optimizer, ckpt_dir: str):
    """Load encoder and optimizer from checkpoint. Decoder LoRA loaded separately."""
    encoder.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "encoder.pt"), weights_only=True)
    )
    opt_state = torch.load(os.path.join(ckpt_dir, "optimizer.pt"), weights_only=True)
    optimizer.load_state_dict(opt_state["optimizer"])
    return opt_state["step"], opt_state["loss"]


def log_metrics(step: int, metrics: dict, prefix: str = "train"):
    """Print formatted training metrics."""
    parts = [f"[{prefix}] step {step}"]
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    print(" | ".join(parts))
```

### Checkpoint structure

Each checkpoint directory contains:
```
checkpoints/step_5000/
  encoder.pt           # Full encoder state_dict (W_enc, W_emb, running stats, usage counters)
  decoder_lora/        # PEFT adapter (adapter_config.json + adapter_model.safetensors)
  optimizer.pt         # AdamW state + step count + loss at checkpoint
```

The encoder is saved as a full state dict (~100MB). The decoder LoRA is saved via PEFT's `.save_pretrained()` which stores only the adapter weights (~75MB), not the full 1.5B base model.

---

## 15. Running the Pipeline End to End

### Stage 1: Pretraining

```bash
# This will:
# 1. Download Qwen2.5-1.5B-Instruct (~3GB, cached after first run)
# 2. Stream and tokenize FineWeb data (~30-60 min first time, then cached)
# 3. Train for 5000 steps (~2-4 hours on a single GPU)
python train_pretrain.py
```

**What to watch for during pretraining:**
- Loss should drop from ~6.0 to ~3.5 over 5000 steps
- Active concepts should gradually decrease from 8192 to ~5700 (some concepts will die)
- If loss is stuck above 5.0 after 1000 steps, something is wrong

### Stage 2: Fine-tuning

```bash
# Prerequisites:
# - Completed pretraining (checkpoints/step_5000/ must exist)
# - ANTHROPIC_API_KEY set (for QA data generation, first time only)
#
# This will:
# 1. Generate 360 QA pairs (~10-15 min, then cached)
# 2. Fine-tune decoder LoRA for 4000 steps (~1-2 hours)
python train_finetune.py
```

### Inference

```python
from config import PCDConfig
from inference import PCDPipeline

config = PCDConfig()

# After pretraining only:
pipeline = PCDPipeline(
    config,
    encoder_path="checkpoints/step_5000/encoder.pt",
    decoder_lora_path="checkpoints/step_5000/decoder_lora",
)
result = pipeline("The discovery of penicillin revolutionized modern medicine.")
print(result["pcd_continuation"])
print(result["probe_outputs"])

# After fine-tuning:
pipeline_ft = PCDPipeline(
    config,
    encoder_path="checkpoints/step_5000/encoder.pt",  # same encoder
    decoder_lora_path="checkpoints_finetune/step_4000/decoder_lora",
)
answer = pipeline_ft.ask(
    "I'm planning a big dinner party this weekend.",
    "What does the model think the dietary restriction of the user is?\n(A) Vegan (B) Vegetarian (C) Omnivore (D) Pescatarian\nAnswer:"
)
print(answer["answer"])
```

---

## 16. Expected Results and Diagnostics

### Pretraining metrics

| Step | Loss | Active Concepts | Dead Concepts | Notes |
|------|------|-----------------|---------------|-------|
| 50 | ~5.95 | 8192 (100%) | 0 | Random performance, all concepts active |
| 500 | ~4.50 | ~8100 (99%) | ~92 | Warmup ending, loss dropping fast |
| 1000 | ~3.99 | ~7672 (94%) | ~520 | First concepts dying off |
| 2000 | ~3.75 | ~6800 (83%) | ~1392 | Concept vocabulary stabilizing |
| 3000 | ~3.65 | ~6200 (76%) | ~1992 | Diminishing returns |
| 5000 | ~3.57 | ~5757 (70%) | ~2435 | Convergence |

**Interpreting these numbers:**
- **Loss ~3.57** means the decoder's perplexity on suffix prediction is e^3.57 ~ 35, which is reasonable for 16 tokens of web text given only 16 sparse concept codes as input.
- **70% active concepts** means 5757 of 8192 concepts are meaningfully used. The other 30% are "dead" -- they never win the top-k competition. This is normal and expected with TopK sparsity.

### Sanity checks

1. **Early loss too high (>7)?** Check that the encoder normalization is working. Print `encoded.norm()` -- it should be in the range 1-10, not 100-1000.
2. **Loss not decreasing?** Check that gradients are flowing through both encoder and decoder. Print `encoder.W_enc.weight.grad.norm()` after a backward pass.
3. **All concepts dead?** Your TopK k value might be too small, or learning rate too high. Check that `top_vals` are not all near-zero.
4. **CUDA OOM?** Reduce `batch_size` to 8 and increase `grad_accum_steps` to 8 (keeps effective batch at 64).

---

## 17. Key Bugs We Fixed (and You Must Too)

These are real bugs we encountered during reproduction. If you skip these fixes, your PCD will not work.

### Bug 1: Missing attention mask on padded input

**Symptom:** Garbage activations, loss doesn't converge.
**Cause:** When input sequences have padding, attention computes over pad tokens, corrupting real-token activations.
**Fix:** Always pass `attention_mask` to the subject model when inputs contain padding. During training we use fixed-length windows (no padding needed). During inference we avoid padding entirely by using raw tokens.

### Bug 2: Encoder output scale mismatch

**Symptom:** Loss stays high (~6+), decoder outputs gibberish.
**Cause:** Raw layer-13 activations have norms of 50-500. After `W_enc -> TopK -> W_emb`, the soft tokens have norms of 1000+, while normal token embeddings have norms of ~1-5. The decoder can't handle inputs that are 100-1000x larger than its training distribution.
**Fix:** Normalize activations before encoding: center by subtracting running mean, then L2-normalize each token to unit norm. This keeps soft token norms comparable to real embeddings.

### Bug 3: Repetitive generation at inference

**Symptom:** Decoder generates "the the the the" or similar repetitive loops.
**Cause:** Greedy decoding without any repetition suppression.
**Fix:** Apply repetition penalty (1.3) during generation. Already-generated tokens have their logits divided by 1.3, making the model prefer novel tokens.

### Bug 4: Pad-token activations in the middle window

**Symptom:** Short prompts at inference produce nonsensical PCD output.
**Cause:** If the input is shorter than `prefix_len + middle_len` (32 tokens), padding fills the gap. Pad-token activations at layer 13 are meaningless but get encoded and fed to the decoder.
**Fix:** At inference, don't pad. Use the raw input length: set `middle_len = min(n_tokens, config.middle_len)` and `prefix_len = n_tokens - middle_len`. The per-token L2 normalization handles variable sequence lengths gracefully.

---

## 18. Paper vs. This Reproduction

| Aspect | Paper (Huang et al. 2025) | This Reproduction |
|--------|---------------------------|-------------------|
| Subject model | Llama 3.1-8B-Instruct | Qwen2.5-1.5B-Instruct |
| Decoder model | Llama 3.1-8B-Instruct | Qwen2.5-1.5B-Instruct |
| Hidden dim (d) | 4096 | 1536 |
| Concepts (m) | 32,768 | 8,192 |
| TopK (k) | 16 | 16 |
| Read layer | Layer 15 of 32 (~47%) | Layer 13 of 28 (~47%) |
| Window size | 16/16/16 | 16/16/16 |
| Training data | ~72M tokens | ~4.8M tokens |
| Pretraining steps | Not specified | 5,000 |
| LoRA rank | Not specified | 16 |
| Fine-tuning data | 78,964 verified pairs (SynthSys 8B) | 360 self-generated pairs (18 attrs x 4 values x 5 questions) |
| Fine-tuning steps | Not specified | 4,000 |
| GPU requirement | Multi-GPU (8B model) | Single 16GB GPU |
| Feature labeling | Yes (automated) | Not included in this tutorial |

**What this reproduction demonstrates:**
- The core PCD architecture works at small scale
- Sparse encoding with TopK preserves meaningful semantic information
- The communication bottleneck forces genuine concept learning
- QA fine-tuning can teach the decoder to answer questions about encoded activations

**What this reproduction does NOT demonstrate:**
- Full-scale performance (would need 8B model + 72M tokens)
- Automated feature labeling (requires a separate large-scale pipeline)
- Comprehensive evaluation on held-out attributes

---

## 19. Glossary

| Term | Definition |
|------|------------|
| **PCD** | Predictive Concept Decoder -- an interpretability tool that predicts future tokens from sparse-encoded internal activations |
| **Subject model** | The frozen LLM being interpreted |
| **Encoder** | The sparse linear encoder that maps activations to concept space |
| **Decoder** | The LoRA-adapted LLM that reads soft tokens and generates text |
| **Soft tokens** | The encoder's output vectors, injected as input embeddings to the decoder |
| **TopK** | Hard sparsity gate: keep only the k largest values, zero out the rest |
| **Concept** | One of the m dimensions in the encoder's sparse space. Each concept direction corresponds to a learned feature of the subject model's internal representations |
| **Dead concept** | A concept that never fires (never enters top-k) across many training batches |
| **Auxiliary loss** | A small loss term that pushes dead concepts' pre-activations upward to revive them |
| **LoRA** | Low-Rank Adaptation -- adds small trainable matrices to frozen model weights |
| **Communication bottleneck** | The design constraint that information must pass through only k sparse concept activations between subject and decoder |
| **SynthSys** | Synthetic System-prompt data -- QA pairs generated by priming the subject model with user attributes |
| **FineWeb** | A large open-source web text dataset used for pretraining data |
| **l_read** | The transformer layer from which we extract activations (layer 13) |
| **Running mean** | Exponential moving average of activation means, used for centering normalization |
