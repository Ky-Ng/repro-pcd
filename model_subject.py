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
        self, input_ids: torch.Tensor, prefix_len: int, middle_len: int
    ) -> torch.Tensor:
        """Run the subject model and return activations for middle token positions.

        Args:
            input_ids: [batch, prefix_len + middle_len] token IDs
            prefix_len: number of prefix tokens
            middle_len: number of middle tokens

        Returns:
            Tensor of shape [batch, middle_len, hidden_dim]
        """
        self.model(input_ids)
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
