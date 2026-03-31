0from torch import nn
import torch
from jaxtyping import Float, Int
from torch import Tensor

from src.pcd_config import PCDConfig

class SparseEncoder(nn.Module):
    """
    Sparse Encoder which extracts features from the Subject Model to be interpreted by the Decoder model.
    Also provides the "communication bottleneck"
    """

    def __init__(self, config: PCDConfig):        
        super().__init__()
        self.W_enc = nn.Linear(config.d_model, config.n_concepts)
        self.W_emb = nn.Linear(config.n_concepts, config.d_model, bias=False)

        self.k = config.topk
        self.k_aux = config.k_aux
        self.n_concepts = config.n_concepts
        
        self.dead_concept_steps = config.dead_concept_steps
        self.aux_loss_coeff = config.aux_loss_coeff

        # Batch Statistics for pre-up projection normalization
        self.register_buffer("running_mean", torch.zeros(config.d_model, dtype=torch.float32))
        self.register_buffer("running_var", torch.ones(config.d_model, dtype=torch.float32))
        self.register_buffer("n_samples", torch.tensor(0, dtype=torch.long))
        self.norm_momentum = 0.01 # How quickly to update the mean/variance

        # Auxiliary Loss + Dead Concept Tracking
        
        # Used to track concept death
        self.register_buffer(
            "steps_since_active", torch.zeros(config.n_concepts, dtype=torch.long)
        )

        # Used for training metadata
        self.register_buffer(
            "concept_usage", torch.zeros(config.n_concepts, dtype=torch.long)
        )
        self.total_steps = 0


    def _initialize_weights(self) -> None:
        """
        Used to stabilize training
            - W_enc has (small) unit-norm rows
            - W_emb is the transpose of W_enc to act as an idenity-like matrix
        """
        with torch.no_grad():
            # Set W_enc to random unit-norm rows
            nn.init.kaiming_uniform(self.W_enc.weight) 
            self.W_enc.weight.div_(
                self.W_enc.weight.norm(dim=1, keepdim=True) + 1e-8
            )

            # Zero initialize the bias
            nn.init.zeros_(self.W_enc.bias)

            # Initialize W_emb to w_enc^T
            self.W_emb.weight.copy_(self.W_enc.weight.t())
        
    def _normalize_activations(self, activations: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
        """
        Normalizes activations via centering + per-token L2 Norm

        Mean center: Calculate running mean learned during training
        Normalize: Use L2 Norm of each token vector

        Note: lerp -> Linear Interpolation
        ```
        # For a lerp with momentum 0.1
        running_mean = 0.9 * running_mean + 0.1 * batch_mean
        ```
        """

        if self.training:
            batch_mean = activations.mean(dim=[0,1]) # [batch, seq, d_model] -> [d_model]
            batch_var = activations.var(dim=[0,1]) # [batch, seq, d_model] -> [d_model]

            if self.n_samples == 0:
                self.running_mean.copy_(batch_mean)
                self.running_var.copy_(batch_var)
            else:
                self.running_mean.lerp_(batch_mean, self.norm_momentum)
                self.running_var.lerp_(batch_var, self.norm_momentum)
            self.n_samples += 1 
        
        # Mean center and normalize by L2
        centered = activations - self.running_mean
        return centered / (centered.norm(dim=-1, keepdim=True) + 1e-8)


    def forward(self, activations: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
        """
        Args:
            activations (Float[Tensor, "batch seq d_concepts"]): Subject Model residual stream
        
        Returns:
            encoded (Float[Tensor, "batch seq d_concepts"]): re-embedded sparse concepts
            info: dict with aux loss and metadata
        """
        # Normalize activations and let the projections do the scaling
        activations = self._normalize_activations(activations)

        pre_act = self.W_enc(activations) # [batch, seq, d_concepts]

        top_vals, top_idx = torch.topk(pre_act, k=self.k, dim=-1)

        # Create a [batch, seq, d_concepts] vector with k top_vals at top_idx and zeros elswhere
        # E.g. [0., 0., topk_2, 0., topk_1, ...]
        sparse_concepts = torch.zeros_like(pre_act)
        sparse_concepts.scatter_(dim=-1, index=top_idx, src=top_vals) # Choose the topk concepts, leave zeros elsewhere
        
        encoded = self.W_emb(sparse_concepts)

        if self.training:
            self._update_concept_usage(top_idx)

        # Additional Metadata
        aux_loss = self._compute_aux_loss(pre_act)

        n_active = (self.steps_since_active < self.config.dead_concept_window).sum().item()
        
        info = {
            "aux_loss": aux_loss,
            "n_active_concepts": n_active,
            "n_dead_concepts": self.n_concepts - n_active,
            "mean_top_val": top_vals.mean().item(),
        }

        return encoded, info
                
    def _compute_aux_loss(
            self, pre_act: Float[Tensor, "batch seq n_concepts"]
    ) -> Float[Tensor, "aux_loss"]:
        """
        Compute auxiliary loss for dead concept revival
        """

        dead_mask = self.steps_since_active >= self.dead_concept_steps
        n_dead = dead_mask.sum().item()

        if n_dead == 0:
            return torch.tensor(0.0, device=pre_act.device, dtype=pre_act.dtype)

        # Boost loss by negative mean of top k_aux dead pre-activation concepts
        all_dead_pre_act = pre_act[:, :, dead_mask] # [batch, seq, n_dead]
        k = min(self.k_aux, n_dead)
        top_vals, _ = torch.topk(all_dead_pre_act, k=k, dim=-1) # [batch, seq, k]

        aux_loss = -self.aux_loss_coeff * top_vals.mean()

        return aux_loss

    def _update_concept_usage(self, top_idx: Int[Tensor, "batch seq n_concepts"]) -> None:
        """
        Update concept "age" how close it is to death and metadata on concept usage
        """
        self.total_steps += 1

        # Increase age of each concept
        self.steps_since_active += 1

        # Reset age for active concepts in this batch

        fired = top_idx.unique() # [k_concepts]
        self.steps_since_active[fired] = 0
        self.concept_usage[fired] += 1
    
    def get_top_concepts(
            self, activations: Float[Tensor, "batch seq d_model"]
    ) -> tuple[Float[Tensor, "batch seq"], Int[Tensor, "batch seq"]]:
        """
        Get Top-k concepts and values from activations for post-hoc inspection

        Args:
            activations (Float[Tensor, "batch seq d_model"]): Subject Model activations
        
        Returns:
            top_vals (Float[Tensor, "batch seq"]): top k activating concept strength
            top_idx (Int[Tensor, "batch seq"]): concept index
        """
        activations = self._normalize_activations(activations)
        pre_act = self.W_enc(activations)
        return torch.topk(pre_act, k=self.k, dim=-1)

