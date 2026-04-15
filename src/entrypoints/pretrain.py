"""
CLI entrypoint for FineWeb pretraining.

Usage:
    python -m src.entrypoints.pretrain --run-name "bigger_aux_v1"
"""
import argparse

from src.pcd_config import PCDConfig
from src.training.train_pretraining import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain PCD on FineWeb")
    parser.add_argument(
        "--run-name",
        type=str,
        default="Pretraining_Run",
        help="wandb run name",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PCDConfig()
    train(config, wandb_run_name=args.run_name)


if __name__ == "__main__":
    main()
