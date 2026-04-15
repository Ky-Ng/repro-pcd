from dataclasses import dataclass, field
import os
import torch

OUTPUT_DIR = "out"

@dataclass
class PCDConfig:
    # GPU Setup
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16
    seed: int = 42

    # Model
    model_name: str = (
        "Qwen/Qwen2.5-1.5B-Instruct"  # https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
    )
    padding_side: str = (
        "left"  # Use left padding, training unaffected since seq len same per batch, only for bulk inference
    )
    d_model: int = 1536  # Specific to Qwen2.5-1.5B, https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html
    n_vocab: int = 151936

    # Subject Model
    l_read: int = 14  # Read activations 1/2 way through the model (28 layers)

    # Sparse Encoder
    n_concepts: int = d_model * 8  # = 12,288, 8x espansion used in paper
    topk: int = 16  # same as paper

    # Follow paper's prompt splitting
    n_prefix: int = 16
    n_middle: int = 16
    n_suffix: int = 16
    tokens_per_window: int = 48

    # Auxiliary Loss / Dead Concept revival
    dead_concept_tokens_thresh: int = (
        1_000_000  # If a concept has not been used in 1M tokens, it will be marked as dead
    )
    k_aux: int = 500  # max 500 dead concepts to revive, same as paper
    aux_loss_coeff: float = 1 / 32  # Gao et al. (OpenAI TopK SAE) default; note: paper uses 1e-4
    norm_momentum: float = 0.01  # Used to update mean/variance of encoder during training

    # Decoder Model
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Training Configurations
    max_train_steps: int = 20000

    lr: float = 1e-4
    weight_decay: float = 0.01

    # Effective Batch Size = batch_size * grad_accum_steps = 64
    batch_size: int = 16
    grad_accum_steps: int = 4

    # Checkpoints for saving
    checkpoints_dir: str = os.path.join(OUTPUT_DIR, "checkpoints")
    wandb_project: str = "PCD_reproduction"
    wandb_dir: str = OUTPUT_DIR
    log_interval: int = 50
    save_interval: int = 2500

    # Datasets
    data_cache_dir: str = os.path.join(OUTPUT_DIR, "data_cache")
