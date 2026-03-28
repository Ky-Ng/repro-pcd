"""Upload PCD checkpoints to Hugging Face Hub.

Usage:
    # Upload pretrain checkpoints
    python upload_checkpoints.py --checkpoint_dir checkpoints --repo_id user/repo-name

    # Upload finetune checkpoints
    python upload_checkpoints.py --checkpoint_dir checkpoints_finetune --repo_id user/repo-name

    # Upload a single step only
    python upload_checkpoints.py --checkpoint_dir checkpoints --repo_id user/repo-name --steps 5000

    # Dry run (list files without uploading)
    python upload_checkpoints.py --checkpoint_dir checkpoints --repo_id user/repo-name --dry_run
"""

import argparse
import os
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def collect_checkpoint_files(checkpoint_dir: str, steps: list[int] | None = None) -> list[tuple[str, str]]:
    """Collect all files to upload from checkpoint directory.

    Args:
        checkpoint_dir: root checkpoint directory (e.g. 'checkpoints/')
        steps: optional list of specific steps to upload. If None, uploads all.

    Returns:
        List of (local_path, repo_path) tuples
    """
    files = []
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    step_dirs = sorted(checkpoint_dir.iterdir())
    for step_dir in step_dirs:
        if not step_dir.is_dir() or not step_dir.name.startswith("step_"):
            continue

        step_num = int(step_dir.name.split("_")[1])
        if steps is not None and step_num not in steps:
            continue

        for root, _, filenames in os.walk(step_dir):
            for filename in filenames:
                local_path = os.path.join(root, filename)
                # Repo path preserves structure: step_XXXX/...
                repo_path = os.path.relpath(local_path, checkpoint_dir)
                files.append((local_path, repo_path))

    return files


def upload_checkpoints(
    checkpoint_dir: str,
    repo_id: str,
    steps: list[int] | None = None,
    repo_type: str = "model",
    private: bool = False,
    commit_message: str | None = None,
    dry_run: bool = False,
):
    """Upload checkpoint files to a Hugging Face Hub repository.

    Args:
        checkpoint_dir: local directory containing step_XXXX/ subdirectories
        repo_id: HF repo ID (e.g. 'user/model-name')
        steps: optional list of specific steps to upload
        repo_type: 'model', 'dataset', or 'space'
        private: whether to create a private repo
        commit_message: custom commit message
        dry_run: if True, only list files without uploading
    """
    api = HfApi()

    files = collect_checkpoint_files(checkpoint_dir, steps)
    if not files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return

    print(f"Found {len(files)} files to upload from {checkpoint_dir}:")
    for local_path, repo_path in files:
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  {repo_path} ({size_mb:.1f} MB)")

    total_mb = sum(os.path.getsize(f[0]) for f in files) / (1024 * 1024)
    print(f"\nTotal: {total_mb:.1f} MB")

    if dry_run:
        print("\n[DRY RUN] Skipping upload.")
        return

    # Create repo if it doesn't exist
    print(f"\nCreating/verifying repo: {repo_id}")
    create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)

    # Upload all files in a single commit
    if commit_message is None:
        step_names = sorted(set(Path(rp).parts[0] for _, rp in files))
        commit_message = f"Upload checkpoints: {', '.join(step_names)}"

    print(f"Uploading to {repo_id} ...")

    operations = []
    from huggingface_hub import CommitOperationAdd
    for local_path, repo_path in files:
        operations.append(
            CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local_path)
        )

    api.create_commit(
        repo_id=repo_id,
        repo_type=repo_type,
        operations=operations,
        commit_message=commit_message,
    )

    print(f"Done! Uploaded to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload PCD checkpoints to Hugging Face Hub")
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Local checkpoint directory (e.g. 'checkpoints' or 'checkpoints_finetune')",
    )
    parser.add_argument(
        "--repo_id", type=str, required=True,
        help="HF repo ID (e.g. 'user/model-name')",
    )
    parser.add_argument(
        "--steps", type=int, nargs="*", default=None,
        help="Specific steps to upload (default: all)",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create a private repo",
    )
    parser.add_argument(
        "--commit_message", type=str, default=None,
        help="Custom commit message",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="List files without uploading",
    )
    args = parser.parse_args()

    upload_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        repo_id=args.repo_id,
        steps=args.steps,
        private=args.private,
        commit_message=args.commit_message,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
