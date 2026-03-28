"""Fine-tuning loop for PCD (Stage 2).

Freezes the encoder and fine-tunes the decoder (LoRA) on QA pairs,
mixing in 50% FineWeb next-token prediction to prevent forgetting.

Per the paper:
  - Encoder is frozen (concepts already learned during pretraining)
  - Decoder LoRA continues training on QA data
  - 50% of batches are FineWeb (pretraining objective) to reduce forgetting
  - 4000 training steps, same hyperparameters as pretraining
"""

import os
import math
import random
import torch
from tqdm import tqdm
from peft import PeftModel

from config import PCDConfig
from model_subject import SubjectModel
from model_encoder import SparseEncoder
from model_decoder import PCDDecoder
from data import prepare_data, get_dataloader
from data_finetune import prepare_qa_data, get_qa_dataloader
from utils import log_metrics


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_finetune_checkpoint(decoder, optimizer, step, loss, config):
    """Save decoder LoRA and optimizer state for fine-tuning."""
    ckpt_dir = os.path.join(config.finetune_checkpoint_dir, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    decoder.model.save_pretrained(os.path.join(ckpt_dir, "decoder_lora"))
    torch.save(
        {"optimizer": optimizer.state_dict(), "step": step, "loss": loss},
        os.path.join(ckpt_dir, "optimizer.pt"),
    )
    print(f"Fine-tune checkpoint saved at step {step} to {ckpt_dir}")


def train(config: PCDConfig | None = None):
    if config is None:
        config = PCDConfig()

    torch.manual_seed(config.seed)
    os.makedirs(config.finetune_checkpoint_dir, exist_ok=True)

    # ---- Load models ----
    print("Loading subject model...")
    subject = SubjectModel(config)

    print("Loading encoder from pretrain checkpoint...")
    encoder = SparseEncoder(config).to(config.device).to(config.dtype)
    encoder_path = os.path.join(config.pretrain_checkpoint, "encoder.pt")
    encoder.load_state_dict(
        torch.load(encoder_path, map_location=config.device, weights_only=True)
    )
    encoder.eval()  # Frozen during fine-tuning
    for p in encoder.parameters():
        p.requires_grad = False

    print("Loading decoder with pretrained LoRA...")
    decoder = PCDDecoder(config)
    # Load pretrained LoRA weights
    decoder_lora_path = os.path.join(config.pretrain_checkpoint, "decoder_lora")
    decoder.model = PeftModel.from_pretrained(
        decoder.model.get_base_model(),
        decoder_lora_path,
        torch_dtype=config.dtype,
        is_trainable=True,  # Keep LoRA trainable for fine-tuning
    ).to(config.device)

    # Print parameter counts
    dec_params = sum(p.numel() for p in decoder.model.parameters() if p.requires_grad)
    print(f"Decoder trainable (LoRA) params: {dec_params:,}")
    print(f"Encoder: frozen ({sum(p.numel() for p in encoder.parameters()):,} params)")

    # ---- Data ----
    # QA data
    print("Preparing QA fine-tuning data...")
    qa_dataset = prepare_qa_data(config, subject, num_examples=10_000)
    qa_dataloader = get_qa_dataloader(qa_dataset, config)

    # FineWeb data (for mixing to prevent forgetting)
    print("Preparing FineWeb data for mixing...")
    fineweb_dataset = prepare_data(config)
    fineweb_dataloader = get_dataloader(fineweb_dataset, config)

    # ---- Optimizer (decoder LoRA only) ----
    optimizer = torch.optim.AdamW(
        [p for p in decoder.model.parameters() if p.requires_grad],
        lr=config.lr_decoder,
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.finetune_warmup_steps, config.finetune_steps
    )

    # ---- Training loop ----
    print(f"\nStarting fine-tuning for {config.finetune_steps} steps...")
    print(f"Batch size: {config.batch_size}, Grad accum: {config.grad_accum_steps}")
    print(f"FineWeb mix ratio: {config.finetune_mix_ratio}")

    decoder.model.train()
    global_step = 0
    accum_qa_loss = 0.0
    accum_fineweb_loss = 0.0
    qa_steps = 0
    fineweb_steps = 0

    qa_iter = iter(qa_dataloader)
    fineweb_iter = iter(fineweb_dataloader)

    while global_step < config.finetune_steps:
        # Decide whether this micro-batch is QA or FineWeb
        is_fineweb = random.random() < config.finetune_mix_ratio

        if is_fineweb:
            # --- FineWeb pretraining objective (prevent forgetting) ---
            try:
                batch = next(fineweb_iter)
            except StopIteration:
                fineweb_iter = iter(fineweb_dataloader)
                batch = next(fineweb_iter)

            prefix_ids = batch["prefix_ids"].to(config.device)
            middle_ids = batch["middle_ids"].to(config.device)
            suffix_ids = batch["suffix_ids"].to(config.device)

            # Subject activations (no grad, frozen)
            subject_input = torch.cat([prefix_ids, middle_ids], dim=1)
            with torch.no_grad():
                activations = subject.get_middle_activations(
                    subject_input, config.prefix_len, config.middle_len
                )
                encoded, _ = encoder(activations)

            # Decoder forward: next-token prediction on suffix
            with torch.autocast(device_type="cuda", dtype=config.dtype):
                loss = decoder.forward_train(encoded, suffix_ids)
                loss = loss / config.grad_accum_steps

            loss.backward()
            accum_fineweb_loss += loss.item() * config.grad_accum_steps
            fineweb_steps += 1

        else:
            # --- QA fine-tuning objective ---
            try:
                batch = next(qa_iter)
            except StopIteration:
                qa_iter = iter(qa_dataloader)
                batch = next(qa_iter)

            prefix_ids = batch["prefix_ids"].to(config.device)
            middle_ids = batch["middle_ids"].to(config.device)
            question_ids = batch["question_ids"].to(config.device)
            answer_ids = batch["answer_ids"].to(config.device)

            # Subject activations (no grad, frozen)
            subject_input = torch.cat([prefix_ids, middle_ids], dim=1)
            with torch.no_grad():
                activations = subject.get_middle_activations(
                    subject_input, config.prefix_len, config.middle_len
                )
                encoded, _ = encoder(activations)

            # Decoder forward: predict answer given soft tokens + question
            with torch.autocast(device_type="cuda", dtype=config.dtype):
                loss = decoder.forward_train_qa(encoded, question_ids, answer_ids)
                loss = loss / config.grad_accum_steps

            loss.backward()
            accum_qa_loss += loss.item() * config.grad_accum_steps
            qa_steps += 1

        # Gradient accumulation step
        if (global_step + 1) % config.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in decoder.model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        # Logging
        if global_step % config.log_interval == 0:
            avg_qa = accum_qa_loss / max(qa_steps, 1)
            avg_fw = accum_fineweb_loss / max(fineweb_steps, 1)
            log_metrics(global_step, {
                "qa_loss": avg_qa,
                "fineweb_loss": avg_fw,
                "qa_batches": qa_steps,
                "fineweb_batches": fineweb_steps,
                "lr": scheduler.get_last_lr()[0],
            }, prefix="finetune")
            accum_qa_loss = 0.0
            accum_fineweb_loss = 0.0
            qa_steps = 0
            fineweb_steps = 0

        # Checkpointing
        if global_step % config.save_interval == 0:
            save_finetune_checkpoint(
                decoder, optimizer, global_step, loss.item(), config
            )

    # Final save
    save_finetune_checkpoint(
        decoder, optimizer, global_step, loss.item(), config
    )
    print(f"\nFine-tuning complete! {global_step} steps.")


if __name__ == "__main__":
    train()
