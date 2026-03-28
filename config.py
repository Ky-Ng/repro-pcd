"""Central configuration for the PCD reproduction."""

from dataclasses import dataclass, field
import torch


@dataclass
class PCDConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hidden_dim: int = 1536  # d, from Qwen2.5-1.5B
    num_layers: int = 28

    # Encoder
    num_concepts: int = 8192  # m (paper uses 32768)
    topk: int = 16  # k for TopK sparsity
    l_read: int = 13  # layer to read activations (~47% depth)

    # Decoder
    l_write: int = 0  # layer to patch soft tokens into
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Data
    prefix_len: int = 16
    middle_len: int = 16
    suffix_len: int = 16
    total_window: int = 48  # prefix + middle + suffix

    # Training
    lr_encoder: float = 3e-4
    lr_decoder: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 16
    grad_accum_steps: int = 4  # effective batch = 64
    max_train_steps: int = 5000
    warmup_steps: int = 500
    aux_loss_coeff: float = 1e-4
    dead_concept_window: int = 1000  # batches before concept considered dead

    # Fine-tuning (Stage 2)
    finetune_steps: int = 4000
    finetune_warmup_steps: int = 400
    finetune_mix_ratio: float = 0.5  # fraction of batches that are FineWeb (vs QA)
    finetune_checkpoint_dir: str = "checkpoints_finetune"
    pretrain_checkpoint: str = "checkpoints/step_5000"  # load from this pretrain ckpt

    # Inference
    max_new_tokens: int = 128

    # System
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    seed: int = 42

    # Paths
    data_cache_dir: str = "data_cache"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 50
    save_interval: int = 1000
