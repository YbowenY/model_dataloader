import argparse
import importlib.util
import shutil
from collections import Counter
from pathlib import Path

import torch
import yaml
from monai.utils import set_determinism


def import_loader_module(loader_path: str):
    loader_path = Path(loader_path).resolve()
    spec = importlib.util.spec_from_file_location("medical_loader", str(loader_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_yaml(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_cpu_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.as_tensor(x).detach().cpu()


def get_channel_dim(x: torch.Tensor):
    """
    Dataset item:
        image:     [C, H, W, D]
        seg_label: [C, H, W, D]

    DataLoader batch:
        image:     [B, C, H, W, D]
        seg_label: [B, C, H, W, D]
    """
    if x.ndim == 4:
        return 0
    if x.ndim == 5:
        return 1
    raise AssertionError(f"Expected tensor ndim 4 or 5, got shape={tuple(x.shape)}")


def get_spatial_shape(x: torch.Tensor):
    return tuple(int(v) for v in x.shape[-3:])


def get_channel_count(x: torch.Tensor):
    return int(x.shape[get_channel_dim(x)])


def print_tensor_summary(name: str, x):
    x = to_cpu_tensor(x)
    print(f"\n[{name}]")
    print(f"  shape: {tuple(x.shape)}")
    print(f"  dtype: {x.dtype}")
    if x.numel() > 0 and x.dtype.is_floating_point:
        print(f"  min/max: {x.min().item():.6f} / {x.max().item():.6f}")
        print(f"  mean/std: {x.mean().item():.6f} / {x.std().item():.6f}")
    elif x.numel() > 0:
        print(f"  min/max: {x.min().item()} / {x.max().item()}")
    return x


def get_enabled_dataset_channel_counts(cfg_yaml):
    channel_definitions = cfg_yaml["data"]["channel_definitions"]
    enabled = [d for d in cfg_yaml["data"]["datasets"] if d.get("enabled", True)]

    counts = []
    dataset_info = []

    for ds in enabled:
        channel_type = ds["channel_type"]
        image_keys = channel_definitions[channel_type]["image_keys"]
        counts.append(len(image_keys))
        dataset_info.append((ds["name"], channel_type, image_keys))

    return counts, dataset_info


def clear_cache_if_requested(cfg_yaml, clear_cache: bool):
    if not clear_cache:
        return

    cache_dir = cfg_yaml["data"]["cache"].get("cache_dir")
    if not cache_dir:
        print("[clear_cache] No cache_dir configured. Skip.")
        return

    cache_path = Path(cache_dir)
    if cache_path.exists():
        print(f"[clear_cache] Removing cache dir: {cache_path}")
        shutil.rmtree(cache_path)
    else:
        print(f"[clear_cache] Cache dir does not exist: {cache_path}")


def inspect_config(cfg_yaml):
    print("=" * 100)
    print("[Config summary]")

    print(f"modality: {cfg_yaml['data']['modality']}")
    print(f"default_t_label: {cfg_yaml['data']['default_t_label']}")
    print(f"batch_size: {cfg_yaml['data']['loader']['batch_size']}")
    print(f"num_workers: {cfg_yaml['data']['loader']['num_workers']}")
    print(f"spatial_size: {cfg_yaml['data']['preprocessing']['spatial_size']}")
    print(f"patch_size: {cfg_yaml['data']['preprocessing'].get('patch_size')}")
    print(f"convert_brats_classes: {cfg_yaml['data']['preprocessing']['convert_brats_classes']}")
    print(f"crop_foreground: {cfg_yaml['data']['preprocessing']['crop_foreground']}")
    print(f"inference.use_sliding_window: {cfg_yaml.get('inference', {}).get('use_sliding_window', False)}")
    print(f"inference.roi_size: {cfg_yaml.get('inference', {}).get('roi_size')}")

    counts, dataset_info = get_enabled_dataset_channel_counts(cfg_yaml)

    print("\n[Enabled datasets]")
    for name, channel_type, image_keys in dataset_info:
        print(f"  - {name}: channel_type={channel_type}, image_keys={image_keys}")

    unique_counts = sorted(set(counts))
    print(f"\nEnabled dataset channel counts: {unique_counts}")

    assert len(unique_counts) == 1, (
        "Enabled datasets have different channel counts. "
        "你的 config 注释里也提醒：不同通道数的数据集不要混在同一次 run 中。"
    )


def inspect_raw_splits(loader_module, config_path: str, cfg_yaml, max_raw_items: int):
    print("\n" + "=" * 100)
    print("[Raw datalist check]")

    split_datalist = loader_module.build_split_datalist(config_path)

    for split, items in split_datalist.items():
        counter = Counter(item["dataset_name"] for item in items)
        print(f"\nSplit: {split}")
        print(f"  total samples: {len(items)}")
        print(f"  by dataset: {dict(counter)}")

    source_seg_key = cfg_yaml["data"]["source_keys"]["seg_label"]

    for split, items in split_datalist.items():
        print("\n" + "-" * 100)
        print(f"[Raw path check] split={split}")

        if len(items) == 0:
            print("  WARNING: this split is empty.")
            continue

        for idx, item in enumerate(items[:max_raw_items]):
            print(f"\n  sample[{idx}]")
            print(f"    dataset_name: {item.get('dataset_name')}")
            print(f"    sample_id: {item.get('sample_id')}")
            print(f"    split_group_id: {item.get('split_group_id')}")

            image_keys = item["_image_source_keys"]
            print(f"    image_source_keys: {image_keys}")

            for key in image_keys:
                path = Path(item[key])
                print(f"    {key}: {path} | exists={path.exists()}")
                assert path.exists(), f"Missing image file: key={key}, path={path}"

            label_path = Path(item[source_seg_key])
            print(f"    {source_seg_key}: {label_path} | exists={label_path.exists()}")
            assert label_path.exists(), f"Missing label file: {label_path}"

    return split_datalist


def check_transformed_sample(
    sample,
    *,
    split: str,
    cfg_yaml,
    expected_image_channels: int,
    expected_fixed_spatial: bool,
    name: str,
):
    assert isinstance(sample, dict), f"{name} should be dict, got {type(sample)}"

    required_keys = ["image", "seg_label", "T_label"]
    for key in required_keys:
        assert key in sample, f"{name} missing key: {key}, existing keys={list(sample.keys())}"

    image = print_tensor_summary(f"{name}.image", sample["image"])
    label = print_tensor_summary(f"{name}.seg_label", sample["seg_label"])
    t_label = print_tensor_summary(f"{name}.T_label", sample["T_label"])

    assert image.ndim in (4, 5), f"image should be [C,H,W,D] or [B,C,H,W,D], got {tuple(image.shape)}"
    assert label.ndim in (4, 5), f"seg_label should be [C,H,W,D] or [B,C,H,W,D], got {tuple(label.shape)}"

    image_spatial = get_spatial_shape(image)
    label_spatial = get_spatial_shape(label)

    assert image_spatial == label_spatial, (
        f"Spatial mismatch: image={image_spatial}, seg_label={label_spatial}"
    )

    image_channels = get_channel_count(image)
    assert image_channels == expected_image_channels, (
        f"Unexpected image channel count: got {image_channels}, expected {expected_image_channels}"
    )

    assert image.dtype.is_floating_point, f"image should be float, got {image.dtype}"
    assert label.dtype in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.long,
    ), f"seg_label should be integer type after Lambdad, got {label.dtype}"

    convert_brats = bool(cfg_yaml["data"]["preprocessing"]["convert_brats_classes"])
    if convert_brats:
        label_channels = get_channel_count(label)
        assert label_channels == 3, (
            f"convert_brats_classes=true 时，期望 seg_label 是 3 通道 BraTS 标签，"
            f"但现在 label_channels={label_channels}, shape={tuple(label.shape)}"
        )

        unique_vals = torch.unique(label)
        unique_list = unique_vals.tolist()
        print(f"\n[{name}.seg_label unique]")
        print(f"  values: {unique_list[:50]}")
        print(f"  num_unique: {len(unique_list)}")

        allowed = {0, 1}
        assert set(int(v) for v in unique_list).issubset(allowed), (
            f"convert_brats_classes=true 后，seg_label 应该是多通道二值 mask，"
            f"但发现 unique values={unique_list}"
        )
    else:
        unique_vals = torch.unique(label)
        print(f"\n[{name}.seg_label unique]")
        print(f"  values: {unique_vals[:50].tolist()}")
        print(f"  num_unique: {unique_vals.numel()}")

    spatial_size = tuple(cfg_yaml["data"]["preprocessing"]["spatial_size"])

    if expected_fixed_spatial:
        assert image_spatial == spatial_size, (
            f"{split} expected fixed spatial_size={spatial_size}, but got image_spatial={image_spatial}"
        )
    else:
        # validation/test + sliding window: 不强制 128^3，只要求已经能正常读入且 image/label 对齐。
        if image_spatial == spatial_size:
            print(
                f"\n[NOTE] {name} spatial shape equals spatial_size={spatial_size}. "
                "这可能只是该样本前景裁剪后刚好接近 128，也可能说明 val/test 仍然做了空间裁剪；"
                "请结合多个样本一起判断。"
            )

    foreground_voxels = int((label > 0).sum().item())
    total_voxels = int(label.numel())
    foreground_ratio = foreground_voxels / max(total_voxels, 1)

    print(f"\n[{name}.foreground]")
    print(f"  foreground_voxels: {foreground_voxels}")
    print(f"  total_voxels: {total_voxels}")
    print(f"  foreground_ratio: {foreground_ratio:.8f}")

    if foreground_voxels == 0:
        print(
            f"  WARNING: {name} 的 label 前景为 0。"
            "如果这是 training 且使用 RandSpatialCropd，随机 crop 到背景 patch 是可能的；"
            "如果 validation/test 也为 0，请检查 seg_label 路径、标签值编码和 ConvertToMultiChannelBasedOnBratsClassesd。"
        )

    print(f"\n[{name}.T_label]")
    print(f"  shape: {tuple(t_label.shape)}")
    print(f"  values: {t_label.flatten()[:20].tolist()}")

    print(f"\n[PASS] {name} basic checks passed.")


def inspect_dataset_items(
    loader_module,
    config_path: str,
    cfg_yaml,
    split: str,
    num_items: int,
    expected_image_channels: int,
):
    print("\n" + "=" * 100)
    print(f"[Dataset transform check] split={split}")

    is_train = split == "training"
    use_sliding_window = bool(cfg_yaml.get("inference", {}).get("use_sliding_window", False))

    # 训练集应该做 RandSpatialCropd -> 固定 128^3
    # 验证/测试集如果 use_sliding_window=true，应该不做空间裁剪，只保留 foreground crop 后的完整 volume
    expected_fixed_spatial = is_train or (not use_sliding_window)

    dataset = loader_module.make_medical_dataset(
        config=config_path,
        split=split,
        is_train=is_train,
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"is_train: {is_train}")
    print(f"use_sliding_window: {use_sliding_window}")
    print(f"expected_fixed_spatial: {expected_fixed_spatial}")

    assert len(dataset) > 0, f"{split} dataset is empty."

    spatial_shapes = []

    for idx in range(min(num_items, len(dataset))):
        print("\n" + "-" * 100)
        print(f"Checking transformed dataset[{idx}]")

        sample = dataset[idx]

        # 兼容某些 crop transform 返回 list 的情况。
        if isinstance(sample, list):
            print(f"dataset[{idx}] returned list with {len(sample)} samples.")
            for j, sub_sample in enumerate(sample):
                check_transformed_sample(
                    sub_sample,
                    split=split,
                    cfg_yaml=cfg_yaml,
                    expected_image_channels=expected_image_channels,
                    expected_fixed_spatial=expected_fixed_spatial,
                    name=f"{split}.dataset[{idx}][{j}]",
                )
                spatial_shapes.append(get_spatial_shape(to_cpu_tensor(sub_sample["image"])))
        else:
            check_transformed_sample(
                sample,
                split=split,
                cfg_yaml=cfg_yaml,
                expected_image_channels=expected_image_channels,
                expected_fixed_spatial=expected_fixed_spatial,
                name=f"{split}.dataset[{idx}]",
            )
            spatial_shapes.append(get_spatial_shape(to_cpu_tensor(sample["image"])))

    print("\n" + "-" * 100)
    print(f"[{split}] checked spatial shapes:")
    for s in spatial_shapes:
        print(f"  {s}")

    if (split in {"validation", "test"}) and use_sliding_window:
        spatial_size = tuple(cfg_yaml["data"]["preprocessing"]["spatial_size"])
        all_equal_spatial_size = all(s == spatial_size for s in spatial_shapes)

        if all_equal_spatial_size:
            print(
                f"\n[WARNING] {split} 的前 {len(spatial_shapes)} 个样本全部是 {spatial_size}。"
                "如果你已经按 use_sliding_window=true 修改了 loader，val/test 理论上不应该强制 RandSpatialCropd。"
                "当然，也可能只是这些样本前景裁剪后刚好就是 128^3。建议多测几个样本。"
            )


def inspect_dataloader_batch(
    loader_module,
    config_path: str,
    cfg_yaml,
    split: str,
    expected_image_channels: int,
):
    print("\n" + "=" * 100)
    print(f"[DataLoader batch check] split={split}")

    is_train = split == "training"
    use_sliding_window = bool(cfg_yaml.get("inference", {}).get("use_sliding_window", False))
    batch_size = int(cfg_yaml["data"]["loader"]["batch_size"])

    expected_fixed_spatial = is_train or (not use_sliding_window)

    try:
        loader = loader_module.make_medical_dataloader(
            config=config_path,
            split=split,
            is_train=is_train,
        )

        batch = next(iter(loader))

        check_transformed_sample(
            batch,
            split=split,
            cfg_yaml=cfg_yaml,
            expected_image_channels=expected_image_channels,
            expected_fixed_spatial=expected_fixed_spatial,
            name=f"{split}.loader_batch",
        )

    except RuntimeError as e:
        if (split in {"validation", "test"}) and use_sliding_window and batch_size > 1:
            print("\n[EXPECTED COLLATE WARNING]")
            print(
                f"{split} split 开启 use_sliding_window=true 且 batch_size={batch_size}。"
                "如果 val/test 不做空间裁剪，不同病例 foreground crop 后 shape 可能不同，"
                "PyTorch 默认 collate 可能无法 stack 成 batch。"
            )
            print("\n原始错误：")
            print(e)
            print(
                "\n解决方式：验证/测试 dataloader 使用 batch_size=1，"
                "或者自定义 collate_fn，或者只在 inference 阶段逐 case 做 sliding_window_inference。"
            )
        else:
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loader_path",
        type=str,
        required=True,
        help="你的 dataloader .py 文件路径，例如 ./medical_loader.py",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="你的 config.yaml 路径",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["training", "validation", "test"],
        choices=["training", "validation", "test"],
        help="要测试的 split",
    )
    parser.add_argument(
        "--num_items",
        type=int,
        default=2,
        help="每个 split 检查多少个 transformed dataset item",
    )
    parser.add_argument(
        "--max_raw_items",
        type=int,
        default=2,
        help="每个 split 检查多少个 raw datalist item 的路径",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        help="如果你刚修改过 transform，建议打开，避免 PersistentDataset 旧缓存影响测试。",
    )
    args = parser.parse_args()

    set_determinism(seed=args.seed)

    cfg_yaml = load_yaml(args.config)
    clear_cache_if_requested(cfg_yaml, args.clear_cache)

    loader_module = import_loader_module(args.loader_path)

    inspect_config(cfg_yaml)

    channel_counts, _ = get_enabled_dataset_channel_counts(cfg_yaml)
    expected_image_channels = sorted(set(channel_counts))[0]

    inspect_raw_splits(
        loader_module=loader_module,
        config_path=args.config,
        cfg_yaml=cfg_yaml,
        max_raw_items=args.max_raw_items,
    )

    for split in args.splits:
        inspect_dataset_items(
            loader_module=loader_module,
            config_path=args.config,
            cfg_yaml=cfg_yaml,
            split=split,
            num_items=args.num_items,
            expected_image_channels=expected_image_channels,
        )

        inspect_dataloader_batch(
            loader_module=loader_module,
            config_path=args.config,
            cfg_yaml=cfg_yaml,
            split=split,
            expected_image_channels=expected_image_channels,
        )

    print("\n" + "=" * 100)
    print("[ALL DONE]")
    print("如果所有 [PASS] 都通过，说明 datalist 扫描、image/label 读取、通道数、label 转换和空间尺寸基本正确。")


if __name__ == "__main__":
    main()