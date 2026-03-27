"""Pretraining loop for PCD.

Jointly trains the encoder and decoder (LoRA) on FineWeb next-token prediction.
The subject model is frozen.
"""

import os
import math
import torch
from tqdm import tqdm

from config import PCDConfig
from model_subject import SubjectModel
from model_encoder import SparseEncoder
from model_decoder import PCDDecoder
from data import prepare_data, get_dataloader
from utils import save_checkpoint, log_metrics


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(config: PCDConfig | None = None):
    if config is None:
        config = PCDConfig()

    torch.manual_seed(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ---- Load models ----
    print("Loading subject model...")
    subject = SubjectModel(config)

    print("Loading encoder...")
    encoder = SparseEncoder(config).to(config.device).to(config.dtype)

    print("Loading decoder with LoRA...")
    decoder = PCDDecoder(config)
    decoder.model.to(config.device)

    # Print parameter counts
    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.model.parameters() if p.requires_grad)
    print(f"Encoder params: {enc_params:,}")
    print(f"Decoder trainable (LoRA) params: {dec_params:,}")

    # ---- Data ----
    print("Preparing data...")
    dataset = prepare_data(config)
    dataloader = get_dataloader(dataset, config)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": config.lr_encoder},
            {"params": [p for p in decoder.model.parameters() if p.requires_grad],
             "lr": config.lr_decoder},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_steps, config.max_train_steps
    )

    # ---- Training loop ----
    print(f"\nStarting pretraining for {config.max_train_steps} steps...")
    print(f"Effective batch size: {config.batch_size * config.grad_accum_steps}")

    encoder.train()
    decoder.model.train()
    global_step = 0
    accum_loss = 0.0
    accum_aux_loss = 0.0

    while global_step < config.max_train_steps:
        for batch in dataloader:
            if global_step >= config.max_train_steps:
                break

            prefix_ids = batch["prefix_ids"].to(config.device)
            middle_ids = batch["middle_ids"].to(config.device)
            suffix_ids = batch["suffix_ids"].to(config.device)

            # 1. Get subject model activations (frozen, no grad)
            subject_input = torch.cat([prefix_ids, middle_ids], dim=1)
            with torch.no_grad():
                activations = subject.get_middle_activations(
                    subject_input, config.prefix_len, config.middle_len
                )

            # 2. Encode through sparse bottleneck
            with torch.autocast(device_type="cuda", dtype=config.dtype):
                encoded, enc_info = encoder(activations)

                # 3. Decoder forward: predict suffix from soft tokens
                loss = decoder.forward_train(encoded, suffix_ids)

                # 4. Add auxiliary dead concept loss
                aux_loss = enc_info["aux_loss"]
                total_loss = (loss + aux_loss) / config.grad_accum_steps

            # 5. Backward
            total_loss.backward()

            accum_loss += loss.item()
            accum_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss

            # Gradient accumulation step
            if (global_step + 1) % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + [p for p in decoder.model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            # Logging
            if global_step % config.log_interval == 0:
                avg_loss = accum_loss / config.log_interval
                avg_aux = accum_aux_loss / config.log_interval
                log_metrics(global_step, {
                    "loss": avg_loss,
                    "aux_loss": avg_aux,
                    "active_concepts": enc_info["n_active_concepts"],
                    "dead_concepts": enc_info["n_dead_concepts"],
                    "lr_enc": scheduler.get_last_lr()[0],
                    "lr_dec": scheduler.get_last_lr()[1],
                })
                accum_loss = 0.0
                accum_aux_loss = 0.0

            # Checkpointing
            if global_step % config.save_interval == 0:
                save_checkpoint(encoder, decoder, optimizer, global_step, loss.item(), config)

    # Final save
    save_checkpoint(encoder, decoder, optimizer, global_step, loss.item(), config)
    print(f"\nPretraining complete! {global_step} steps.")


if __name__ == "__main__":
    train()
