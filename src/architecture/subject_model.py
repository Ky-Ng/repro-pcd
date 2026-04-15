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
        self.padding_side = config.padding_side_subject_model

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.n_prefix = config.n_prefix
        self.n_middle = config.n_middle
        self.n_suffix = config.n_suffix

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def get_middle_activations(self, tokens: Int[Tensor, "batch seq"], attention_mask: Int[Tensor, "batch seq"], start_extract: int, end_extract: int) -> torch.Tensor:
        """
        Extract residual stream activations at l_read for the middle tokens

        FineWeb (pretraining + finetuning)  : "prefix" + "middle" with fixed lengths from PCDConfig
        SynthSys (finetuning)               : "system" + "user" with variable lengths
        Deployment (inference)              : "user" 

        Args:
            tokens (Int[Tensor, "batch seq"): Inputs to model
            
            attention_mask (Int[Tensor, "batch seq"]): Which tokens to attend to
            
            start_extract (int): Positions to start extracting from

            end_extract (int): Position to stop extracting from

        Returns: 
            residuals at the extraction positions of shape [batch, end_extract-start_extract, d_model]
        """
        hook_name = f"blocks.{self.config.l_read}.hook_resid_pre"
        _, cache = self.model.run_with_cache(tokens, names_filter=hook_name, attention_mask=attention_mask)
        resid = cache[hook_name]  # [Batch, Seq, d_model]
        return resid[:, start_extract:end_extract, :]

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
