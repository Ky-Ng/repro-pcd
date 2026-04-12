Batched Decoder Forward for Inference
- Kind of non-trivial, feel like this is out of schope

```py
@torch.no_grad()
def generate(
    self,
    soft_token_acts: Float[Tensor, "batch n_soft d_model"],
    context_ids: Int[Tensor, "batch n_context"] | None = None,
    soft_token_mask: Bool[Tensor, "batch n_soft"] | None = None,
    n_tokens: int | None = 64,
) -> Int[Tensor, "batch n_generated"]:
    """
    Greedy autoregressive generation from soft-token prefix (+ optional context).

    Inputs follow the training convention (right-padded). Internally re-packs
    each row into a left-padded prefix so every row's real content ends at
    position seq_len-1 — giving clean RoPE geometry and a trivial generation
    boundary.

    Args:
        soft_token_acts: Encoder activations patched in as the initial soft tokens.
        context_ids: Optional query/context token IDs, right-padded with pad_token_id.
        soft_token_mask: Optional mask over soft tokens (True = real, False = pad).
            Assumed right-padded if provided.
        n_tokens: Max number of new tokens to generate.

    Returns:
        Generated token IDs, shape [batch, n_generated]. Rows that hit EOS early
        are filled with pad_token_id for remaining positions.
    """
    B, n_soft, d_model = soft_token_acts.shape
    device = soft_token_acts.device
    embed_layer = self.model.get_input_embeddings()
    pad_id = self.tokenizer.pad_token_id
    eos_id = self.tokenizer.eos_token_id

    # 1) Build right-padded prefix (matches training layout).
    if soft_token_mask is None:
        soft_mask = torch.ones(B, n_soft, device=device, dtype=torch.long)
    else:
        soft_mask = soft_token_mask.long()

    if context_ids is not None:
        embeds_rp = torch.cat([soft_token_acts, embed_layer(context_ids)], dim=1)
        mask_rp = torch.cat([soft_mask, (context_ids != pad_id).long()], dim=1)
    else:
        embeds_rp = soft_token_acts
        mask_rp = soft_mask

    prefix_len = embeds_rp.shape[1]

    # 2) Re-pack to left-padded: real tokens right-aligned, pads on the left.
    # Stable sort pulls real tokens (mask=1) to the front while preserving order.
    _, sorted_src = mask_rp.sort(dim=1, descending=True, stable=True)
    real_len = mask_rp.sum(dim=1)                            # [B]
    pad_amount = prefix_len - real_len                       # [B]
    target_pos = torch.arange(prefix_len, device=device).unsqueeze(0).expand(B, -1)
    k = target_pos - pad_amount.unsqueeze(1)                 # rank within real tokens
    source_idx = sorted_src.gather(1, k.clamp(min=0))        # [B, prefix_len]

    input_embeds = embeds_rp.gather(
        1, source_idx.unsqueeze(-1).expand(-1, -1, d_model)
    )
    attention_mask = (k >= 0).long()                         # 1 on real, 0 on left-pad
    # Zero embeds at pad slots (cosmetic; attention ignores them anyway).
    input_embeds = input_embeds * attention_mask.unsqueeze(-1).to(input_embeds.dtype)

    # 3) Explicit position_ids so RoPE starts at 0 on each row's first real token.
    #    pads get position 1 as a harmless placeholder (masked out by attention).
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)

    # 4) Autoregressive loop. Every row's "next position" is always seq_len-1.
    generated = torch.empty((B, 0), dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    was_training = self.model.training
    self.model.eval()
    try:
        for _ in range(n_tokens):
            out = self.model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            next_logits = out.logits[:, -1, :]               # left-pad => shared boundary
            next_tokens = next_logits.argmax(dim=-1)
            next_tokens = torch.where(
                finished, torch.full_like(next_tokens, pad_id), next_tokens
            )
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

            if eos_id is not None:
                finished = finished | (next_tokens == eos_id)
            if finished.all():
                break

            next_embeds = embed_layer(next_tokens).unsqueeze(1)
            input_embeds = torch.cat([input_embeds, next_embeds], dim=1)
            attention_mask = torch.cat(
                [attention_mask, (~finished).long().unsqueeze(1)], dim=1
            )
            position_ids = torch.cat(
                [position_ids, position_ids[:, -1:] + 1], dim=1
            )
    finally:
        if was_training:
            self.model.train()

    return generated
```

UnBatched
```py
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from jaxtyping import Bool, Float, Int

@torch.no_grad()
def generate(
    self,
    soft_token_acts: Float[Tensor, "batch n_soft d_model"],
    context_ids: Int[Tensor, "batch n_context"] | None = None,
    soft_token_mask: Bool[Tensor, "batch n_soft"] | None = None,
    n_tokens: int | None = 64,
) -> Int[Tensor, "batch n_generated"]:
    """
    Greedy autoregressive generation from soft-token prefix (+ optional context).

    Inputs follow the training convention (right-padded). Internally re-packs
    each row into a left-padded prefix so every row's real content ends at
    position seq_len-1 — giving clean RoPE geometry and a trivial generation
    boundary.

    Args:
        soft_token_acts: Encoder activations patched in as the initial soft tokens.
        context_ids: Optional query/context token IDs, right-padded with pad_token_id.
        soft_token_mask: Optional mask over soft tokens (True = real, False = pad).
            Assumed right-padded if provided.
        n_tokens: Max number of new tokens to generate.

    Returns:
        Generated token IDs, shape [batch, n_generated]. Rows that hit EOS early
        are filled with pad_token_id for remaining positions.
    """
    B, n_soft, d_model = soft_token_acts.shape
    device = soft_token_acts.device
    embed_layer = self.model.get_input_embeddings()
    pad_id = self.tokenizer.pad_token_id
    eos_id = self.tokenizer.eos_token_id

    # 1) Build right-padded prefix (matches training layout).
    if soft_token_mask is None:
        soft_mask = torch.ones(B, n_soft, device=device, dtype=torch.long)
    else:
        soft_mask = soft_token_mask.long()

    if context_ids is not None:
        embeds_rp = torch.cat([soft_token_acts, embed_layer(context_ids)], dim=1)
        mask_rp = torch.cat([soft_mask, (context_ids != pad_id).long()], dim=1)
    else:
        embeds_rp = soft_token_acts
        mask_rp = soft_mask

    prefix_len = embeds_rp.shape[1]

    # 2) Re-pack to left-padded: real tokens right-aligned, pads on the left.
    # Stable sort pulls real tokens (mask=1) to the front while preserving order.
    _, sorted_src = mask_rp.sort(dim=1, descending=True, stable=True)
    real_len = mask_rp.sum(dim=1)                            # [B]
    pad_amount = prefix_len - real_len                       # [B]
    target_pos = torch.arange(prefix_len, device=device).unsqueeze(0).expand(B, -1)
    k = target_pos - pad_amount.unsqueeze(1)                 # rank within real tokens
    source_idx = sorted_src.gather(1, k.clamp(min=0))        # [B, prefix_len]

    input_embeds = embeds_rp.gather(
        1, source_idx.unsqueeze(-1).expand(-1, -1, d_model)
    )
    attention_mask = (k >= 0).long()                         # 1 on real, 0 on left-pad
    # Zero embeds at pad slots (cosmetic; attention ignores them anyway).
    input_embeds = input_embeds * attention_mask.unsqueeze(-1).to(input_embeds.dtype)

    # 3) Explicit position_ids so RoPE starts at 0 on each row's first real token.
    #    pads get position 1 as a harmless placeholder (masked out by attention).
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)

    # 4) Autoregressive loop. Every row's "next position" is always seq_len-1.
    generated = torch.empty((B, 0), dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    was_training = self.model.training
    self.model.eval()
    try:
        for _ in range(n_tokens):
            out = self.model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            next_logits = out.logits[:, -1, :]               # left-pad => shared boundary
            next_tokens = next_logits.argmax(dim=-1)
            next_tokens = torch.where(
                finished, torch.full_like(next_tokens, pad_id), next_tokens
            )
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

            if eos_id is not None:
                finished = finished | (next_tokens == eos_id)
            if finished.all():
                break

            next_embeds = embed_layer(next_tokens).unsqueeze(1)
            input_embeds = torch.cat([input_embeds, next_embeds], dim=1)
            attention_mask = torch.cat(
                [attention_mask, (~finished).long().unsqueeze(1)], dim=1
            )
            position_ids = torch.cat(
                [position_ids, position_ids[:, -1:] + 1], dim=1
            )
    finally:
        if was_training:
            self.model.train()

    return generated


@torch.no_grad()
def generate_unbatched(
    self,
    soft_token_acts: Float[Tensor, "batch n_soft d_model"],
    context_ids: Int[Tensor, "batch n_context"] | None = None,
    soft_token_mask: Bool[Tensor, "batch n_soft"] | None = None,
    n_tokens: int | None = 64,
) -> Int[Tensor, "batch n_generated"]:
    """
    Same signature as `generate`, but runs each row independently.

    No padding, no position_ids, no re-packing — each row is trimmed to its
    real content and generated on its own. Simpler and easier to reason about,
    but loses batch parallelism on the GPU.

    Returns:
        Right-padded tensor of generated IDs, shape [batch, max_generated_len].
    """
    B = soft_token_acts.shape[0]
    device = soft_token_acts.device
    embed_layer = self.model.get_input_embeddings()
    pad_id = self.tokenizer.pad_token_id
    eos_id = self.tokenizer.eos_token_id

    was_training = self.model.training
    self.model.eval()
    per_row_outputs: list[Tensor] = []
    try:
        for b in range(B):
            # Trim soft tokens to their real positions.
            if soft_token_mask is not None:
                keep = soft_token_mask[b].bool()
                soft_row = soft_token_acts[b, keep]            # [n_soft_real, d]
            else:
                soft_row = soft_token_acts[b]                  # [n_soft, d]

            parts = [soft_row]
            if context_ids is not None:
                ctx_row = context_ids[b]
                ctx_row = ctx_row[ctx_row != pad_id]            # strip right-pad
                if ctx_row.numel() > 0:
                    parts.append(embed_layer(ctx_row))

            input_embeds = torch.cat(parts, dim=0).unsqueeze(0)  # [1, prefix_len, d]

            # HF generate with inputs_embeds returns only the newly generated ids.
            out_ids = self.model.generate(
                inputs_embeds=input_embeds,
                max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
            per_row_outputs.append(out_ids[0].to(device=device, dtype=torch.long))
    finally:
        if was_training:
            self.model.train()

    return pad_sequence(per_row_outputs, batch_first=True, padding_value=pad_id)

```

For decoding tokens later

```py
self.tokenizer.batch_decode(ids, skip_special_tokens=True)
```