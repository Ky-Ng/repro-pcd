import os

import torch
from torch.optim import Optimizer
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file

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

def load_checkpoint(
    encoder: SparseEncoder,
    decoder: DecoderModel,
    checkpoint_dir: str,
    device: str,
) -> None:
    """Load encoder weights and decoder LoRA weights from a `step_<N>` checkpoint directory.

    Mutates `encoder` and `decoder` in place. Mirrors the layout produced by
    `save_checkpoint`:

        <checkpoint_dir>/
            encoder.pt
            decoder_lora/adapter_model.safetensors
    """
    # Encoder: in-place copy into existing parameter tensors
    enc_state = torch.load(
        os.path.join(checkpoint_dir, "encoder.pt"), map_location=device
    )
    encoder.load_state_dict(enc_state)

    # Decoder LoRA: in-place copy into the existing "default" adapter
    lora_sd = load_file(
        os.path.join(checkpoint_dir, "decoder_lora", "adapter_model.safetensors"),
        device=device,
    )
    set_peft_model_state_dict(decoder.model, lora_sd)