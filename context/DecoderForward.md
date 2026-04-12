# Decoder Forward

Detailing the issues with the Decoder Forward:

1. Pretraining has n_prefix=n_middle=n_suffix 

    Decoder Input:[n_middle soft tokens from encoder] [n_suffix ground-truth tokens]. The decoder receives encoded activations as soft tokens patched at layer 0, then the suffix tokens follow as regular hard tokens. Loss is computed only on the suffix token positions. The prefix is never seen by the decoder at all; it only influences the subject model's activations on the middle tokens.

2. Finetuning has two training types

    a. Continued FineWeb pretraining objective, loss computed only on the n_suffix tokens
    b. SynthSys [n_middle soft tokens from encoder] [MCQ question tokens] [answer token] where loss is calcualted only on the answer token (multiple choice letter)

3. Inference

    a. [subject model user tokens] patched into the decoder model + [query embed from user]

The sample solution is below

```
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

        Args:
            soft_tokens: [B, middle_len, d] encoded activations from encoder
            suffix_ids: [B, suffix_len] target suffix token IDs

        Returns:
            loss: scalar cross-entropy loss on suffix prediction
        """
        B, n_soft, d = soft_tokens.shape
        suffix_len = suffix_ids.shape[1]

        # Get embeddings for suffix tokens (teacher-forced input: all but last)
        suffix_embeds = self.get_token_embeddings(suffix_ids[:, :-1])  # [B, suffix_len-1, d]

        # Concatenate: [soft_tokens] + [suffix_embeds]
        inputs_embeds = torch.cat([soft_tokens, suffix_embeds], dim=1)  # [B, n_soft + suffix_len - 1, d]

        total_len = inputs_embeds.shape[1]

        # Attention mask: all ones (causal mask is applied internally)
        attention_mask = torch.ones(B, total_len, device=inputs_embeds.device, dtype=torch.long)

        # Forward pass
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # [B, total_len, vocab_size]

        # Compute loss only on suffix positions
        # The positions that should predict suffix tokens are [n_soft-1 ... n_soft+suffix_len-2]
        # because position i predicts token at position i+1
        suffix_logits = logits[:, n_soft - 1:, :]  # [B, suffix_len, vocab_size]

        # Target: all suffix tokens
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

        The decoder sees [soft_tokens] + [question_embeds] + [answer_embeds]
        and is trained to predict the answer tokens (loss only on answer positions).

        This mirrors the paper's QA fine-tuning where the decoder learns to
        answer questions about the subject model's encoded activations.

        Args:
            soft_tokens: [B, middle_len, d] encoded activations from frozen encoder
            question_ids: [B, q_len] question token IDs
            answer_ids: [B, a_len] answer token IDs (targets)

        Returns:
            loss: scalar cross-entropy loss on answer prediction
        """
        B, n_soft, d = soft_tokens.shape
        q_len = question_ids.shape[1]
        a_len = answer_ids.shape[1]

        # Get embeddings for question and answer tokens
        q_embeds = self.get_token_embeddings(question_ids)  # [B, q_len, d]
        a_embeds = self.get_token_embeddings(answer_ids[:, :-1])  # [B, a_len-1, d]

        # Concatenate: [soft_tokens] + [question_embeds] + [answer_embeds]
        inputs_embeds = torch.cat([soft_tokens, q_embeds, a_embeds], dim=1)

        total_len = inputs_embeds.shape[1]
        attention_mask = torch.ones(B, total_len, device=inputs_embeds.device, dtype=torch.long)

        # Mask out padding in question_ids from attention
        # (pad tokens should not attend or be attended to)
        q_pad_mask = (question_ids != self.tokenizer.pad_token_id).long()  # [B, q_len]
        attention_mask[:, n_soft:n_soft + q_len] = q_pad_mask

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # [B, total_len, vocab_size]

        # Loss only on answer positions
        # The last position before answers (n_soft + q_len - 1) predicts first answer token
        # Positions [n_soft + q_len - 1 ... n_soft + q_len + a_len - 2] predict answer tokens
        answer_start = n_soft + q_len - 1
        answer_logits = logits[:, answer_start:answer_start + a_len, :]  # [B, a_len, vocab]

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

        Args:
            soft_tokens: [B, n_soft, d] encoded activations
            prompt_ids: [B, q_len] optional prompt token IDs (None = generate from soft tokens only)
            max_new_tokens: maximum tokens to generate
            repetition_penalty: multiplicative penalty for previously generated tokens
                (>1.0 discourages repetition, 1.0 = no penalty)

        Returns:
            List of generated strings
        """
        B = soft_tokens.shape[0]

        if prompt_ids is not None:
            # Concatenate soft tokens + prompt embeddings
            q_embeds = self.get_token_embeddings(prompt_ids)
            inputs_embeds = torch.cat([soft_tokens, q_embeds], dim=1)
        else:
            # Just use soft tokens
            inputs_embeds = soft_tokens

        total_len = inputs_embeds.shape[1]
        attention_mask = torch.ones(B, total_len, device=inputs_embeds.device, dtype=torch.long)

        # Generate autoregressively
        # We need to use a custom generation approach since HF generate
        # doesn't natively support starting from inputs_embeds + continuing
        generated_ids = []
        past_key_values = None

        # First forward: process all prefix (soft + question)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # [B, 1]
        generated_ids.append(next_token)

        # Subsequent tokens: feed one token at a time with KV cache
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

            # Apply repetition penalty to previously generated tokens
            if repetition_penalty != 1.0 and len(generated_ids) > 0:
                prev_tokens = torch.cat(generated_ids, dim=1)  # [B, gen_so_far]
                for b in range(B):
                    for token_id in prev_tokens[b].unique():
                        if next_token_logits[b, token_id] > 0:
                            next_token_logits[b, token_id] /= repetition_penalty
                        else:
                            next_token_logits[b, token_id] *= repetition_penalty

            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated_ids.append(next_token)

            # Stop if all sequences have generated EOS
            if (next_token == self.tokenizer.eos_token_id).all():
                break

        # Decode
        generated = torch.cat(generated_ids, dim=1)  # [B, gen_len]
        texts = [
            self.tokenizer.decode(generated[i], skip_special_tokens=True)
            for i in range(B)
        ]
        return texts
```