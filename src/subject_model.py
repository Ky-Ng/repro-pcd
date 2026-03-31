from transformers import BatchEncoding
from jaxtyping import Float, Int

from src.pcd_config import PCDConfig
from transformer_lens import HookedTransformer

import torch
from torch import Tensor


class SubjectModel:
    """
    Frozen Subject Model where we extract activations from l_read with a forward hook
    """
    def __init__(self, config: PCDConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype

        self.model = HookedTransformer.from_pretrained(
            config.model_name,
            device=self.device,
            dtype=self.dtype,
        )
        self.tokenizer = self.model.tokenizer
        self.padding_side = config.padding_side

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.n_prefix = config.n_prefix
        self.n_middle = config.n_middle
        self.n_suffix = config.n_suffix

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def get_middle_activations(self, tokens: Int[Tensor, "batch seq"], attention_mask: Int[Tensor, "batch seq"] = None) -> torch.Tensor:
        """
        Extract residual stream activations at l_read for the middle tokens

        tokens: [Batch, Seq]
        prefix_len: number of prefix tokens to skip
        middle_len: number of middle tokens to extract
        attention_mask: [Batch, Seq] — needed for padded batches

        Returns: [Batch, middle_len, d_model]
        """
        hook_name = f"blocks.{self.config.l_read}.hook_resid_pre"
        _, cache = self.model.run_with_cache(tokens, names_filter=hook_name, attention_mask=attention_mask)
        resid = cache[hook_name]  # [Batch, Seq, d_model]
        return resid[:, self.n_prefix:self.n_prefix + self.n_middle, :]

    def apply_chat_template(self, s: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": s}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def tokenize(self, s) -> BatchEncoding:
        """
        s: str or list[str]
        Returns: (tokens [Batch, Seq], attention_mask [Batch, Seq])
        """
        inputs = self.tokenizer(
            s,
            return_tensors="pt",
            padding=True,
            padding_side=self.padding_side
        ).to(self.device)

        return inputs

    # `decode` and `generate` are for visibility/sanity checking
    def decode(self, tokens: Int[Tensor, "batch seq"]) -> str:
        return self.model.to_string(tokens)

    def generate(self, s: Int[Tensor, "batch seq"], max_new_tokens: int = 128) -> str:
        """
        s: Int[Tensor, "batch seq"] — pass tokens, note TransformerLens handles padding internally (left-padding)
        """
        return self.model.generate(s, max_new_tokens=max_new_tokens, temperature=0, stop_at_eos=True, padding_side="left")    
