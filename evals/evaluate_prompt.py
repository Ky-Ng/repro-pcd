import argparse
from src.architecture.pcd_inference_model import PCDInferenceModel
from src.pcd_config import PCDConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a prompt through the PCD pipeline")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a step_<N> directory containing encoder.pt and decoder_lora/")
    p.add_argument("--decoder_question", type=str, required=False,
                   help="Question asked directly to the Decoder model")
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = PCDConfig()
    pipeline = PCDInferenceModel(config)
    pipeline.load_checkpoint(args.checkpoint)

    output = pipeline.generate(
        args.prompt, 
        decoder_question=args.decoder_question,
        max_new_tokens=args.max_new_tokens
    )
    print(output)


if __name__ == "__main__":
    main()