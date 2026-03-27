"""Shared utilities for logging and checkpointing."""

import os
import torch
from config import PCDConfig


def save_checkpoint(
    encoder,
    decoder,
    optimizer,
    step: int,
    loss: float,
    config: PCDConfig,
):
    """Save encoder, decoder LoRA, and optimizer state."""
    ckpt_dir = os.path.join(config.checkpoint_dir, f"step_{step}")
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


def load_checkpoint(
    encoder,
    decoder_model,
    optimizer,
    ckpt_dir: str,
):
    """Load encoder and optimizer from checkpoint. Decoder LoRA loaded separately."""
    encoder.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "encoder.pt"), weights_only=True)
    )

    opt_state = torch.load(os.path.join(ckpt_dir, "optimizer.pt"), weights_only=True)
    optimizer.load_state_dict(opt_state["optimizer"])

    return opt_state["step"], opt_state["loss"]


def log_metrics(step: int, metrics: dict, prefix: str = "train"):
    """Print formatted training metrics."""
    parts = [f"[{prefix}] step {step}"]
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    print(" | ".join(parts))
