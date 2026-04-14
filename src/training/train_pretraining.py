"""
Train PCD Encoder and Decoder Model jointly on FineWeb.
Note: Subject Model is Frozen
"""

import os

import torch
import wandb

from src.architecture.decoder_model import DecoderModel
from src.architecture.sparse_encoder import SparseEncoder
from src.architecture.subject_model import SubjectModel
from src.data.fine_web_dataset import get_dataloader, get_fineweb_dataset
from src.pcd_config import PCDConfig
from src.training.utils import log_metrics, save_checkpoint


def train(config: PCDConfig, wandb_run_name: str = "Pretraining_Run") -> None:
    torch.manual_seed(config.seed)

    with wandb.init(
        project=config.wandb_project,
        name=wandb_run_name,
        dir=config.wandb_dir,
        # Add all non private config key-value pairs to the wandb run
        config={k: v for k, v in vars(config).items() if not k.startswith("_")}
    ) as run:

        run_tag = f"{run.name}-{run.id}"
        output_dir = os.path.join(config.checkpoints_dir, run_tag)
        os.makedirs(output_dir, exist_ok=True)

        # Load Models
        print("Loading Subject Model")

        # No .to(device) because SubjectModel is a plain class, not nn.Module which contains a HookedTransformer
        subject = SubjectModel(config)

        print("Loading Encoder Model")
        encoder = SparseEncoder(config).to(config.device)

        print("Loading Decoder")
        decoder = DecoderModel(config).to(config.device)

        # Print parameter counts for sanity checking
        enc_params = sum(p.numel() for p in encoder.parameters())
        dec_params_total = sum(p.numel() for p in decoder.parameters())
        dec_params_trainable = sum(p.numel()
                                   for p in decoder.parameters() if p.requires_grad)

        print(f"Encoder params: {enc_params:,}")
        print(
            f"Decoder trainable (LoRA) params: {dec_params_trainable:,} / {dec_params_total:,} = {dec_params_trainable/dec_params_total * 100:.2f}%")

        # Get Data
        print("Preparing data...")
        dataset = get_fineweb_dataset(
            config,
            cache_path=os.path.join(
                config.data_cache_dir, "fineweb_windows.pt"),
            num_examples=100_000,
            ds_name="HuggingFaceFW/fineweb"
        )
        dataloader = get_dataloader(dataset, config)

        # Setup Optimizer/scheduler
        # https://yassin01.medium.com/adam-vs-adamw-understanding-weight-decay-and-its-impact-on-model-performance-b7414f0af8a1
        # https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        # https://arxiv.org/abs/1711.05101
        # https://arxiv.org/abs/1412.6980

        optimizer = torch.optim.AdamW(
            [
                {"params": list(encoder.parameters())},
                {"params": [p for p in decoder.parameters() if p.requires_grad]},
            ],
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.max_train_steps
        )

        # Training Loop
        print(f"\nStarting pretraining for {config.max_train_steps} steps...")
        print(
            f"Effective batch size: {config.batch_size * config.grad_accum_steps}")

        # batch --> collect gradient and loss --> step when ready, log if needed, save checkpoint
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

                # 1. Extract Subject Model activations
                # [batch, seq=n_prefix+n_middle]
                subject_inputs = torch.cat([prefix_ids, middle_ids], dim=1)
                with torch.no_grad():
                    activations = subject.get_middle_activations(
                        tokens=subject_inputs,
                        attention_mask=torch.ones_like(subject_inputs),
                        start_extract=config.n_prefix,
                        end_extract=config.n_prefix + config.n_middle
                    )

                # 2. Pass activations --> Encoder --> Decoder
                # Encoder: FP32 master weights, model intermediate computations in config.dtype (via autocast)
                # Decoder: Base weights in config.dtype (frozen), LoRA adapter weights in FP32,
                #          all forward compute runs in config.dtype under autocast

                with torch.autocast(device_type=config.device, dtype=config.dtype):
                    sparse_embedding, encoder_info = encoder(activations)

                    # 3. Get loss from decoder forward (predict suffix from middle activations)
                    loss = decoder.forward_train(
                        soft_token_acts=sparse_embedding,
                        target_ids=suffix_ids
                    )

                    # 4. Add aux loss to prevent dead concepts from dying
                    aux_loss = encoder_info["aux_loss"]
                    effective_batch_loss = (
                        loss + aux_loss) / config.grad_accum_steps

                # 5. Compute and store gradient
                effective_batch_loss.backward()

                accum_loss += loss.item()
                accum_aux_loss += aux_loss.item()  # used only for logging

                if (global_step + 1) % config.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) +
                        [p for p in decoder.model.parameters() if p.requires_grad],
                        max_norm=1.0
                    )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                global_step += 1

                # Logging
                if global_step % config.log_interval == 0:
                    avg_loss = accum_loss / config.log_interval
                    avg_aux = accum_aux_loss / config.log_interval
                    metrics = {
                        "loss": avg_loss,
                        "aux_loss": avg_aux,
                        "active_concepts": encoder_info["n_alive"],
                        "dead_concepts": encoder_info["n_dead"],
                        "mean_top_act": encoder_info["mean_top_act"],
                        "percent_concepts_alive": encoder_info["percent_concepts_alive"],
                        "lr": scheduler.get_last_lr()[0],
                    }
                    log_metrics(global_step, metrics)
                    wandb.log(metrics, step=global_step)
                    accum_loss = 0.0
                    accum_aux_loss = 0.0

                # Save Checkpoint
                if global_step % config.save_interval == 0:
                    save_checkpoint(
                        encoder=encoder,
                        decoder=decoder,
                        optimizer=optimizer,
                        step=global_step,
                        loss=loss.item(),
                        checkpoint_dir=output_dir
                    )

        # Final Save and Report
        save_checkpoint(
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            step=global_step,
            loss=loss.item(),
            checkpoint_dir=output_dir
        )
        print(f"\nPretraining complete! {global_step} steps.")
