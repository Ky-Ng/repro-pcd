import os

import torch
from torch.optim import Optimizer

from src.architecture.decoder_model import DecoderModel
from src.architecture.sparse_encoder import SparseEncoder

def log_metrics(step: int, metrics: dict, prefix: str = "train"):
    """Print formatted training metrics."""
    parts = [f"[{prefix}] step {step}"]
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    print(" | ".join(parts))

def save_checkpoint(
    encoder: SparseEncoder,
    decoder: DecoderModel,
    optimizer: Optimizer,
    step: int,
    loss: float,
    checkpoint_dir: str
):
    """Save encoder, decoder LoRA, and optimizer state."""
    ckpt_dir = os.path.join(checkpoint_dir, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Encoder
    torch.save(encoder.state_dict(), os.path.join(ckpt_dir, "encoder.pt"))

    # Decoder LoRA adapter
    decoder.model.save_pretrained(os.path.join(ckpt_dir, "decoder_lora"))

    # Optimizer
    torch.save(
        {"optimizer": optimizer.state_dict(), "step": step, "loss": loss},
        os.path.join(ckpt_dir, "optimizer.pt"),
    )
    print(f"Checkpoint saved at step {step} to {ckpt_dir}")