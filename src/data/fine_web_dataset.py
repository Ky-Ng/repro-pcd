import os

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from jaxtyping import Int
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer

from src.pcd_config import PCDConfig


class FineWebDataset(Dataset):
    """
    Wrapper to split token windows of length window_size where
        window_size = config.n_prefix + config.n_middle + config.n_suffix
    to be used by the DataLoader

    Note that this Dataset operates on token ids
    """

    def __init__(self, token_windows: list[Int[Tensor, "window_size"]], config: PCDConfig):
        """
        Args:
            token_windows (list[Int[Tensor, "window_size"]]):
                tokens of length window_size
            config (PCDConfig)
        """
        self.windows = token_windows
        self.config = config

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        window = self.windows[idx]
        prefix = window[:self.config.n_prefix]
        middle = window[self.config.n_prefix: self.config.n_prefix +
                        self.config.n_middle]
        suffix = window[self.config.n_prefix + self.config.n_middle:]

        return {
            "prefix_ids": prefix,
            "middle_ids": middle,
            "suffix_ids": suffix
        }


def get_fineweb_dataset(
    config: PCDConfig,
    cache_path: str | None = None,
    num_examples: int = 100_000,
    ds_name:   str = "HuggingFaceFW/fineweb"
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

    # Check if Cache exists
    if os.path.exists(cache_path):
        print(f"Loading FineWeb from cache")
        windows = torch.load(cache_path, weights_only=True)
        print(
            f"Loaded in {len(windows)} and {len(windows) * config.tokens_per_window / 1e6:.1f}M tokens)")
        return FineWebDataset(windows, config)

    # No cache: stream in FineWeb + tokenize + chunk --> return and cache
    print("No cached dataset, streaming in fineweb")
    return _create_fineweb_dataset(config, num_examples, cache_path, ds_name)


def _create_fineweb_dataset(
    config: PCDConfig,
    num_examples: int,
    cache_path: str,
    ds_name:   str = "HuggingFaceFW/fineweb"
) -> FineWebDataset:

    # Load in Dataset
    print(f"Tokenizing in {ds_name}")
    ds = load_dataset(
        path=ds_name,
        name="sample-10BT",  # 10 Billion Token subset of FineWeb which we only use a fraction of
        split="train",
        streaming=True
    )

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Iterate through streamed dataset and
    windows = []
    total_tokens = 0
    tokens_per_window = config.tokens_per_window

    for ex in tqdm(ds, desc="Tokenizing FineWeb", total=num_examples):
        if len(windows) >= num_examples:
            break

        # View example here: https://huggingface.co/datasets/HuggingFaceFW/fineweb/viewer/CC-MAIN-2013-20
        text = ex["text"]

        # strip out short text (measured in chars)
        if len(text.strip()) < 50:
            continue

        token_ids = tokenizer.encode(text, add_special_tokens=False)

        # Remove examples that are smaller than our window size
        if len(token_ids) < tokens_per_window:
            continue

        # Iterate and chunk
        for start in range(0, len(token_ids)-tokens_per_window+1, tokens_per_window):
            window = torch.tensor(
                token_ids[start:start+tokens_per_window], dtype=torch.long)
            windows.append(window)
            total_tokens += tokens_per_window

            if len(windows) >= num_examples:
                break

    print(f"Created {len(windows)} token windows totalling {total_tokens / 1e-6:.1f}M tokens")

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
