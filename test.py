import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import torch

from loader import (
    build_split_datalist,
    load_medical_loader_config,
    make_medical_dataloader,
    make_medical_dataset,
)


def _to_python_number(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def _summarize_tensor(name: str, tensor: torch.Tensor) -> str:
    shape = tuple(tensor.shape)
    dtype = str(tensor.dtype)
    min_value = float(tensor.min().item())
    max_value = float(tensor.max().item())
    return f"{name}: shape={shape}, dtype={dtype}, min={min_value:.4f}, max={max_value:.4f}"


def _summarize_split_items(items: list[Dict[str, Any]]) -> str:
    dataset_counter = Counter(item["dataset_name"] for item in items)
    return ", ".join(f"{name}={count}" for name, count in sorted(dataset_counter.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DataLoader2 dataset discovery and loading.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="Path to DataLoader2 config.yaml",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size used for the loading smoke test.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers used for the loading smoke test.",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="none",
        choices=["none", "cache", "persistent"],
        help="Cache mode override for the test run.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = load_medical_loader_config(args.config)
    cfg.data.loader.batch_size = int(args.batch_size)
    cfg.data.loader.num_workers = int(args.num_workers)
    cfg.data.cache.mode = str(args.cache_mode)

    datalist = build_split_datalist(cfg)

    print(f"Config: {Path(args.config).resolve()}")
    print("Split sizes:")
    for split in ("training", "validation", "test"):
        items = datalist[split]
        print(f"  {split}: {len(items)}")
        print(f"    datasets: {_summarize_split_items(items)}")

    print("\nSample and batch validation:")
    for split in ("training", "validation", "test"):
        dataset = make_medical_dataset(cfg, split=split, is_train=False)
        if len(dataset) == 0:
            print(f"  [{split}] dataset is empty, skipping load test.")
            continue

        sample = dataset[0]
        print(f"  [{split}] first sample:")
        print(f"    sample_id: {sample.get('sample_id', 'N/A')}")
        print(f"    dataset_name: {sample.get('dataset_name', 'N/A')}")
        print(f"    {_summarize_tensor('image', sample['image'])}")
        print(f"    {_summarize_tensor('seg_label', sample['seg_label'].float())}")
        print(f"    T_label: {_to_python_number(sample['T_label'])}")

        loader = make_medical_dataloader(cfg, split=split, is_train=False)
        batch = next(iter(loader))
        print(f"  [{split}] first batch:")
        print(f"    image shape: {tuple(batch['image'].shape)}")
        print(f"    seg_label shape: {tuple(batch['seg_label'].shape)}")
        print(f"    T_label shape: {tuple(batch['T_label'].shape)}")
        print(f"    T_label values: {_to_python_number(batch['T_label'])}")


if __name__ == "__main__":
    main()
