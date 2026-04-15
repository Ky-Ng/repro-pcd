"""
Upload a local directory to a Hugging Face Hub repo.

Examples:
    # Upload a training checkpoint directory to a model repo
    python -m src.data.upload_to_hf \
        --local-dir out/checkpoints/my_experiment \
        --repo-id my-user/my-repo \
        --path-in-repo checkpoints/my_experiment

    # Upload a data cache to a dataset repo
    python -m src.data.upload_to_hf \
        --local-dir out/src/data_cache \
        --repo-id my-user/my-repo \
        --path-in-repo data_cache \
        --repo-type dataset

Requires HF_TOKEN env var or --token.
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def upload_directory(
    local_dir: str,
    repo_id: str,
    path_in_repo: str | None = None,
    repo_type: str = "model",
    private: bool = False,
    token: str | None = None,
    commit_message: str | None = None,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> str:
    local_path = Path(local_dir).expanduser().resolve()
    if not local_path.is_dir():
        raise ValueError(f"local_dir does not exist or is not a directory: {local_path}")

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if path_in_repo is None:
        path_in_repo = local_path.name

    api = HfApi(token=token)
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        token=token,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message or f"Upload {local_path.name}",
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )

    prefix = "datasets/" if repo_type == "dataset" else ("spaces/" if repo_type == "space" else "")
    url = f"https://huggingface.co/{prefix}{repo_id}/tree/main/{path_in_repo}"
    print(f"Uploaded {local_path} -> {url}")
    return url


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a directory to a Hugging Face Hub repo.")
    parser.add_argument("--local-dir", required=True, help="Local directory to upload.")
    parser.add_argument("--repo-id", required=True, help="Target repo id, e.g. 'user/name'.")
    parser.add_argument("--path-in-repo", default=None,
                        help="Destination path inside the repo (defaults to local dir name).")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--private", action="store_true", help="Create repo as private if new.")
    parser.add_argument("--token", default=None, help="HF token (else uses $HF_TOKEN).")
    parser.add_argument("--commit-message", default=None)
    parser.add_argument("--allow-patterns", nargs="*", default=None,
                        help="Glob patterns; only matching files are uploaded.")
    parser.add_argument("--ignore-patterns", nargs="*", default=None,
                        help="Glob patterns; matching files are skipped.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    upload_directory(
        local_dir=args.local_dir,
        repo_id=args.repo_id,
        path_in_repo=args.path_in_repo,
        repo_type=args.repo_type,
        private=args.private,
        token=args.token,
        commit_message=args.commit_message,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
    )
