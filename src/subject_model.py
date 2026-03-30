from src.pcd_config import PCDConfig
from transformer_lens import HookedTransformer
import torch

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

        for p in self.model.parameters():
            p.requires_grad = False

    def apply_chat_template(self, s: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": s}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def tokenize(self, s: str) -> torch.Tensor:
        return self.model.to_tokens(s) # [Batch, Seq]

    def decode(self, tokens: torch.Tensor) -> str:
        return self.model.to_string(tokens)

    def generate(self, tokens: torch.Tensor, max_new_tokens: int = 128) -> str:
        return self.model.generate(tokens, max_new_tokens=max_new_tokens, temperature=0)    
