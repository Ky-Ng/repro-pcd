from dataclasses import dataclass
import torch

@dataclass
class PCDConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct" # https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
    
    # GPU Setup
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
