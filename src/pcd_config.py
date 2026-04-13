from dataclasses import dataclass, field
import torch


@dataclass
class PCDConfig:
    # GPU Setup
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16

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

    # Auxiliary Loss / Dead Concept revival
    dead_concept_steps: int = (
        1000  # If a concept has not been used in 1000 steps, it will be marked as dead
    )
    k_aux: int = 500  # max 500 dead concepts to revive, same as paper
    aux_loss_coeff: float = 1e-4

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
    max_train_steps = 4000
    warmup_steps = 0
    lr = 1e-4

