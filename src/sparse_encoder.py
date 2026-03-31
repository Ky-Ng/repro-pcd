from torch import nn
import torch
from jaxtyping import Float
from torch import Tensor

from src.pcd_config import PCDConfig

class SparseEncoder(nn.Module):
    """
    Sparse Encoder which extracts features from the Subject Model to be interpreted by the Decoder model.
    Also provides the "communication bottleneck"
    """

    def __init__(self, config: PCDConfig):        
        super().__init__()
        self.W_enc = nn.Linear(config.d_model, config.d_concepts)
        self.W_emb = nn.Linear(config.d_concepts, config.d_model, bias=False)

        self.k = config.topk

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
        
    # def _nomralize_activations(activations: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
    #     """
    #     Normalizes activations via centering + per-token L2 Norm

    #     Calculated over 
    #     """

    def forward(self, activations: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
        """
        Args:
            activations (Float[Tensor, "batch seq d_concepts"]): Subject Model residual stream
        
        Returns:
            encoded (Float[Tensor, "batch seq d_concepts"]): re-embedded sparse concepts
            info: dict with aux loss and metadata
        """

        pre_act = self.W_enc(activations) # [batch, seq, d_concepts]

        top_vals, top_idx = torch.topk(pre_act, k=self.k, dim=-1)

        # Create a [batch, seq, d_concepts] vector with k top_vals at top_idx and zeros elswhere
        # E.g. [0., 0., topk_2, 0., topk_1, ...]
        sparse_concepts = torch.zeros_like(pre_act)
        sparse_concepts.scatter_(dim=-1, index=top_idx, src=top_vals) # Choose the topk concepts, leave zeros elsewhere
        
        down = self.W_emb(sparse_concepts)

        info = {

        }

        return down, info
                
