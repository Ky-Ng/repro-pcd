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

        # Project to concept space: [B, T, m]
        pre_act = self.W_enc(activations)

        # TopK: keep only top-k activations per position
        top_vals, top_idx = torch.topk(pre_act, self.k, dim=-1)  # [B, T, k]

        # Create sparse representation and re-embed
        # Efficient: gather W_emb columns for top-k concepts and weight them
        # W_emb.weight is [d, m], we need columns at top_idx positions
        # Reshape for gathering: top_idx is [B, T, k]
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
        pre_act = self.W_enc(activations)
        return torch.topk(pre_act, self.k, dim=-1)
