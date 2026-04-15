import argparse
import os

import torch
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file

from src.architecture.decoder_model import DecoderModel
from src.pcd_config import PCDConfig


def load_decoder_lora(decoder: DecoderModel, checkpoint_dir: str, device: str) -> None:
    lora_sd = load_file(
        os.path.join(checkpoint_dir, "decoder_lora", "adapter_model.safetensors"),
        device=device,
    )
    set_peft_model_state_dict(decoder.model, lora_sd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate the decoder's chat-template ability with zero-masked dummy "
            "soft tokens. Use to check whether instruction-following survives FineWeb pretraining."
        )
    )
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional step_<N> checkpoint dir. If omitted, evaluates the untrained baseline.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = PCDConfig()

    decoder = DecoderModel(config).to(config.device).eval()

    if args.checkpoint:
        load_decoder_lora(decoder, args.checkpoint, config.device)
        print(f"Loaded decoder LoRA from {args.checkpoint}")
    else:
        print("No checkpoint provided -- evaluating untrained baseline.")

    templated = decoder.apply_chat_template(args.prompt)
    inputs = decoder.tokenize(templated)
    context_ids = inputs["input_ids"]

    # Dummy soft tokens zero-masked so attention ignores them; positions still consumed.
    dummy_soft_token_acts = torch.rand(
        [1, config.n_middle, config.d_model],
        dtype=config.dtype, device=config.device,
    )
    soft_token_mask = torch.zeros(
        [1, config.n_middle], dtype=torch.long, device=config.device,
    )

    output = decoder.generate(
        soft_token_acts=dummy_soft_token_acts,
        soft_token_mask=soft_token_mask,
        context_ids=context_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    print()
    print("=== Prompt ===")
    print(args.prompt)
    print()
    print("=== Output ===")
    print(output[0])


if __name__ == "__main__":
    main()
