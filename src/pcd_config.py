from dataclasses import dataclass
import torch

@dataclass
class PCDConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct" # https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
    padding_side: str = "left" # Use left padding, training unaffected since seq len same per batch, only for bulk inference
    d_model: int = 1536 # Specific to Qwen2.5-1.5B
    
    # Subject Model
    l_read: int = 14  # Read activations 1/2 way through the model (28 layers)

    # Sparse Encoder
    d_concepts: int = d_model * 8 # = 12,288, 8x espansion used in paper
    topk: int = 16 # same as paper

    # Follow paper's prompt splitting
    n_prefix: int = 16 
    n_middle: int = 16
    n_suffix: int = 16
    
    # GPU Setup
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16
