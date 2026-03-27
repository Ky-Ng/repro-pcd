"""Data loading and preprocessing for PCD pretraining.

Streams FineWeb, tokenizes, and chunks into prefix/middle/suffix windows.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from config import PCDConfig


class FineWebDataset(Dataset):
    """Pre-tokenized FineWeb dataset chunked into 48-token windows."""

    def __init__(self, token_windows: list[torch.Tensor], config: PCDConfig):
        self.windows = token_windows
        self.config = config

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]  # [48]
        prefix = window[: self.config.prefix_len]
        middle = window[self.config.prefix_len : self.config.prefix_len + self.config.middle_len]
        suffix = window[self.config.prefix_len + self.config.middle_len :]
        return {
            "prefix_ids": prefix,
            "middle_ids": middle,
            "suffix_ids": suffix,
        }


def prepare_data(
    config: PCDConfig,
    num_examples: int = 100_000,
    cache_path: str | None = None,
) -> FineWebDataset:
    """Stream FineWeb, tokenize, and chunk into training windows.

    Args:
        config: PCD configuration
        num_examples: approximate number of 48-token windows to create
        cache_path: path to cache the tokenized windows

    Returns:
        FineWebDataset ready for DataLoader
    """
    if cache_path is None:
        cache_path = os.path.join(config.data_cache_dir, "fineweb_windows.pt")

    # Check cache
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        windows = torch.load(cache_path, weights_only=True)
        print(f"Loaded {len(windows)} windows ({len(windows) * config.total_window / 1e6:.1f}M tokens)")
        return FineWebDataset(windows, config)

    print("Preparing FineWeb data (streaming)...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    # Stream FineWeb
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    windows = []
    total_tokens = 0
    window_size = config.total_window

    for example in tqdm(ds, desc="Tokenizing FineWeb", total=num_examples):
        if len(windows) >= num_examples:
            break

        text = example["text"]
        if len(text.strip()) < 50:
            continue

        # Tokenize
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        if len(token_ids) < window_size:
            continue

        # Chunk into non-overlapping windows
        for start in range(0, len(token_ids) - window_size + 1, window_size):
            window = torch.tensor(token_ids[start : start + window_size], dtype=torch.long)
            windows.append(window)
            total_tokens += window_size

            if len(windows) >= num_examples:
                break

    print(f"Created {len(windows)} windows ({total_tokens / 1e6:.1f}M tokens)")

    # Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(windows, cache_path)
    print(f"Cached to {cache_path}")

    return FineWebDataset(windows, config)


def get_dataloader(dataset: FineWebDataset, config: PCDConfig, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader for the FineWeb dataset."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
