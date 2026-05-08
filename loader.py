import csv
import json
import logging
import math
import random
import re
from collections import defaultdict
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import yaml
from monai.data import CacheDataset, Dataset, PersistentDataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    ConvertToMultiChannelBasedOnBratsClassesd,
    RandSpatialCropd
)
from monai.transforms.transform import MapTransform
from torch.utils.data import DataLoader

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:  # pragma: no cover - environment dependent
    DictConfig = dict  # type: ignore[misc,assignment]
    OmegaConf = None


logger = logging.getLogger(__name__)


class ConfigNode(dict):
    """Minimal dict wrapper with attribute-style access."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_config_node(value: Any) -> Any:
    if isinstance(value, Mapping):
        return ConfigNode({key: _to_config_node(sub_value) for key, sub_value in value.items()})
    if isinstance(value, list):
        return [_to_config_node(item) for item in value]
    return value


class StandardizeMedicalBatchd(MapTransform):
    """
    Normalize sample dictionaries to a stable contract:
    batch["image"], batch["seg_label"], batch["T_label"].

    Each sample may provide its own image source keys through
    `_image_source_keys`, which makes it possible to keep channel layouts
    configurable at the dataset level.
    """

    def __init__(
        self,
        image_source_keys: Optional[Sequence[str]] = None,
        seg_source_key: str = "seg_label",
        t_source_key: str = "T_label",
        has_t_label: bool = False,
        default_t_label: int = -1,
        image_source_keys_field: str = "_image_source_keys",
        has_t_label_field: str = "_has_t_label",
    ) -> None:
        base_keys = list(image_source_keys) if image_source_keys else []
        super().__init__(keys=base_keys + [seg_source_key, t_source_key])
        self.image_source_keys = list(image_source_keys) if image_source_keys else []
        self.seg_source_key = seg_source_key
        self.t_source_key = t_source_key
        self.has_t_label = has_t_label
        self.default_t_label = default_t_label
        self.image_source_keys_field = image_source_keys_field
        self.has_t_label_field = has_t_label_field

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        d = dict(data)

        image_source_keys = list(d.get(self.image_source_keys_field, self.image_source_keys))
        if not image_source_keys:
            raise ValueError("No image source keys found for the current sample.")

        missing_image_keys = [key for key in image_source_keys if key not in d]
        if missing_image_keys:
            raise KeyError(f"Missing image keys in sample: {missing_image_keys}")

        image_values = [d[key] for key in image_source_keys]
        d["image"] = image_values[0] if len(image_values) == 1 else image_values

        if self.seg_source_key not in d:
            raise KeyError(f'Missing segmentation key "{self.seg_source_key}" in sample.')
        d["seg_label"] = d[self.seg_source_key]

        has_t_label = bool(d.get(self.has_t_label_field, self.has_t_label))
        if has_t_label and self.t_source_key in d and d[self.t_source_key] is not None:
            d["T_label"] = d[self.t_source_key]
        else:
            d["T_label"] = self.default_t_label

        # 删除不同数据集各自的原始模态 key，避免 DataLoader collate 报 KeyError
        for key in image_source_keys:
            if key not in {"image", "seg_label", "T_label"}:
                d.pop(key, None)

        if self.seg_source_key not in {"image", "seg_label", "T_label"}:
            d.pop(self.seg_source_key, None)

        if self.t_source_key not in {"image", "seg_label", "T_label"}:
            d.pop(self.t_source_key, None)

        return d


def load_medical_loader_config(config: Union[str, Path, DictConfig, Mapping[str, Any]]) -> DictConfig:
    if OmegaConf is not None and isinstance(config, DictConfig):
        cfg = config
    elif isinstance(config, (str, Path)):
        if OmegaConf is not None:
            cfg = OmegaConf.load(str(config))
        else:
            with open(config, "r", encoding="utf-8") as f:
                cfg = _to_config_node(yaml.safe_load(f))
    else:
        if OmegaConf is not None:
            cfg = OmegaConf.create(config)
        else:
            cfg = _to_config_node(config)

    _validate_medical_loader_config(cfg)
    return cfg


def _validate_medical_loader_config(cfg: DictConfig) -> None:
    modality = str(cfg.data.modality).lower()
    if modality not in {"ct", "mri"}:
        raise ValueError(f'Unsupported modality "{cfg.data.modality}". Expected "ct" or "mri".')

    cache_mode = str(cfg.data.cache.mode).lower()
    if cache_mode not in {"persistent", "cache", "none"}:
        raise ValueError(f'Unsupported cache mode "{cfg.data.cache.mode}".')

    ratio_keys = {"training", "validation", "test"}
    configured_ratio_keys = set(cfg.data.split.ratios.keys())
    if configured_ratio_keys != ratio_keys:
        raise ValueError(
            "Split ratios must define exactly these keys: "
            '"training", "validation", "test".'
        )

    ratio_sum = sum(float(cfg.data.split.ratios[key]) for key in ratio_keys)
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    enabled_datasets = [ds for ds in cfg.data.datasets if bool(ds.get("enabled", True))]
    if not enabled_datasets:
        raise ValueError("At least one dataset must be enabled in data.datasets.")

    channel_sizes = set()
    for dataset_cfg in enabled_datasets:
        channel_type = str(dataset_cfg.channel_type)
        if channel_type not in cfg.data.channel_definitions:
            raise ValueError(
                f'Unknown channel_type "{channel_type}" '
                f'for dataset "{dataset_cfg.name}".'
            )

        layout = str(dataset_cfg.get("layout", "directory_cases"))
        if layout == "directory_cases":
            if "case_glob" not in dataset_cfg or not str(dataset_cfg.case_glob).strip():
                raise ValueError(f'Dataset "{dataset_cfg.name}" is missing `case_glob`.')
        elif layout == "flat_suffix_grouped":
            if "grouped_file_glob" not in dataset_cfg or not str(dataset_cfg.grouped_file_glob).strip():
                raise ValueError(f'Dataset "{dataset_cfg.name}" is missing `grouped_file_glob`.')
            if "suffixes" not in dataset_cfg:
                raise ValueError(f'Dataset "{dataset_cfg.name}" is missing `suffixes`.')
        else:
            raise ValueError(
                f'Dataset "{dataset_cfg.name}" has unsupported layout "{layout}".'
            )

        image_keys = _get_dataset_image_source_keys(cfg, dataset_cfg)
        channel_sizes.add(len(image_keys))

        seg_key = str(cfg.data.source_keys.seg_label)
        if layout == "directory_cases":
            patterns = dataset_cfg.patterns
            for image_key in image_keys:
                if image_key not in patterns:
                    raise ValueError(
                        f'Dataset "{dataset_cfg.name}" is missing a file pattern for image key "{image_key}".'
                    )
            if seg_key not in patterns:
                raise ValueError(
                    f'Dataset "{dataset_cfg.name}" is missing a file pattern for "{seg_key}".'
                )
        else:
            suffixes = dataset_cfg.suffixes
            for image_key in image_keys:
                if image_key not in suffixes:
                    raise ValueError(
                        f'Dataset "{dataset_cfg.name}" is missing a suffix for image key "{image_key}".'
                    )
            if seg_key not in suffixes:
                raise ValueError(
                    f'Dataset "{dataset_cfg.name}" is missing a suffix for "{seg_key}".'
                )

        if bool(dataset_cfg.get("has_t_label", False)):
            has_fixed = "fixed_t_label" in dataset_cfg and dataset_cfg.fixed_t_label is not None
            has_map = "t_label_map_path" in dataset_cfg and dataset_cfg.t_label_map_path is not None
            if not has_fixed and not has_map:
                raise ValueError(
                    f'Dataset "{dataset_cfg.name}" declares has_t_label=true, '
                    "but neither fixed_t_label nor t_label_map_path is configured."
                )

    if len(channel_sizes) > 1:
        logger.warning(
            "Active datasets use different channel counts. "
            "This is allowed during discovery, but batching incompatible channel "
            "types together will fail unless they are loaded separately."
        )


def _get_dataset_image_source_keys(cfg: DictConfig, dataset_cfg: DictConfig) -> List[str]:
    channel_type = str(dataset_cfg.channel_type)
    channel_cfg = cfg.data.channel_definitions[channel_type]
    image_source_keys = list(channel_cfg.image_keys)
    if not image_source_keys:
        raise ValueError(f'No image_keys configured for channel_type "{channel_type}".')
    return image_source_keys


def _maybe_build_reader(file_format: str, explicit_reader: str):
    explicit_reader = str(explicit_reader)
    if explicit_reader != "auto":
        if explicit_reader == "NibabelReader":
            from monai.data import NibabelReader

            return NibabelReader()
        if explicit_reader == "PydicomReader":
            from monai.data import PydicomReader

            return PydicomReader()
        if explicit_reader == "ITKReader":
            from monai.data import ITKReader

            return ITKReader()
        raise ValueError(f'Unsupported reader "{explicit_reader}".')

    normalized_format = str(file_format).lower()
    if normalized_format in {"auto", ""}:
        return None
    if normalized_format in {"nii", "nii.gz", "nifti"}:
        from monai.data import NibabelReader

        return NibabelReader()
    if normalized_format in {"dcm", "dicom"}:
        from monai.data import PydicomReader

        return PydicomReader()
    raise ValueError(f'Unsupported file format "{file_format}".')


def _build_intensity_transform(cfg: DictConfig):
    modality = str(cfg.data.modality).lower()
    if modality == "ct":
        intensity_cfg = cfg.data.preprocessing.intensity.ct
        return ScaleIntensityRanged(
            keys=["image"],
            a_min=float(intensity_cfg.a_min),
            a_max=float(intensity_cfg.a_max),
            b_min=float(intensity_cfg.b_min),
            b_max=float(intensity_cfg.b_max),
            clip=bool(intensity_cfg.clip),
        )

    intensity_cfg = cfg.data.preprocessing.intensity.mri
    return ScaleIntensityRangePercentilesd(
        keys=["image"],
        lower=float(intensity_cfg.lower),
        upper=float(intensity_cfg.upper),
        b_min=float(intensity_cfg.b_min),
        b_max=float(intensity_cfg.b_max),
        clip=bool(intensity_cfg.clip),
        channel_wise=True,
    )


def _build_resize_transform(cfg: DictConfig):
    spatial_size = cfg.data.preprocessing.spatial_size
    if spatial_size is None:
        return None

    spatial_size = tuple(int(v) for v in spatial_size)
    return Resized(
        keys=["image", "seg_label"],
        spatial_size=spatial_size,
        mode=("trilinear", "nearest"),
    )

def _build_pad_crop_transform(cfg: DictConfig):
    spatial_size = cfg.data.preprocessing.spatial_size
    if spatial_size is None:
        return None

    spatial_size = tuple(int(v) for v in spatial_size)

    return ResizeWithPadOrCropd(
        keys=["image", "seg_label"],
        spatial_size=spatial_size,
        mode="constant",
    )

def _build_spatial_crop_transform(cfg: DictConfig):
    spatial_size = cfg.data.preprocessing.spatial_size
    if spatial_size is None:
        return None

    spatial_size = tuple(int(v) for v in spatial_size)

    return Compose(
        [
            SpatialPadd(
                keys=["image", "seg_label"],
                spatial_size=spatial_size,
                mode="constant",
            ),
            RandSpatialCropd(
                keys=["image", "seg_label"],
                roi_size=spatial_size,
                random_size=False,
            ),
        ]
    )

def _build_spacing_transform(cfg: DictConfig):
    spacing = cfg.data.preprocessing.spacing
    if spacing is None:
        return None

    spacing = tuple(float(v) for v in spacing)
    return Spacingd(
        keys=["image", "seg_label"],
        pixdim=spacing,
        mode=("bilinear", "nearest"),
    )


def make_medical_transform(
    config: Union[str, Path, DictConfig, Mapping[str, Any]],
    is_train: bool,
) -> Compose:
    cfg = load_medical_loader_config(config)

    source_keys = cfg.data.source_keys
    preprocessing = cfg.data.preprocessing
    train_aug = cfg.data.augmentation.train

    image_reader = _maybe_build_reader(cfg.data.io.image_file_format, cfg.data.io.image_reader)
    seg_reader = _maybe_build_reader(cfg.data.io.seg_label_file_format, cfg.data.io.seg_label_reader)

    transforms = [
        # 统一字段名
        StandardizeMedicalBatchd(
            image_source_keys=None,
            seg_source_key=str(source_keys.seg_label),
            t_source_key=str(source_keys.t_label),
            has_t_label=False,
            default_t_label=int(cfg.data.default_t_label),
        ),
        # 读取文件
        LoadImaged(keys=["image"], reader=image_reader, ensure_channel_first=True),
        LoadImaged(keys=["seg_label"], reader=seg_reader, ensure_channel_first=True),
    ]

    # 统一方向
    if preprocessing.orientation:
        transforms.append(Orientationd(keys=["image", "seg_label"], axcodes=str(preprocessing.orientation)))
    
    # 统一体素间距
    spacing_transform = _build_spacing_transform(cfg)
    if spacing_transform is not None:
        transforms.append(spacing_transform)

    if preprocessing.convert_brats_classes:
        transforms.append(ConvertToMultiChannelBasedOnBratsClassesd(keys="seg_label"))

    # 裁剪背景
    if preprocessing.crop_foreground.enabled:
        transforms.append(
            CropForegroundd(
                keys=["image", "seg_label"],
                source_key="image",
                select_fn=lambda x: x > float(preprocessing.crop_foreground.threshold),
            )
        )

    use_sliding_window = bool(cfg.get("inference", {}).get("use_sliding_window", False))
    apply_spatial_crop = is_train or not use_sliding_window
    # 空间裁剪
    if apply_spatial_crop:
        spatial_crop_transform = _build_spatial_crop_transform(cfg)
        if spatial_crop_transform is not None:
            transforms.append(spatial_crop_transform)

    # 强度归一化
    transforms.append(_build_intensity_transform(cfg))
    
    # 训练集增强
    if is_train:
        flip_prob = float(train_aug.rand_flip_prob)
        for axis in range(3):
            transforms.append(RandFlipd(keys=["image", "seg_label"], prob=flip_prob, spatial_axis=axis))
        transforms.extend(
            [
                RandScaleIntensityd(
                    keys=["image"],
                    factors=float(train_aug.rand_scale_intensity_factor),
                    prob=float(train_aug.rand_scale_intensity_prob),
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=float(train_aug.rand_shift_intensity_offset),
                    prob=float(train_aug.rand_shift_intensity_prob),
                ),
            ]
        )

    transforms.extend(
        [
            EnsureTyped(keys=["image", "seg_label", "T_label"]),
            Lambdad(keys=["image"], func=lambda x: x.float()),
            Lambdad(keys=["seg_label"], func=lambda x: x.long()),
            Lambdad(keys=["T_label"], func=lambda x: torch.as_tensor(x, dtype=torch.long)),
        ]
    )
    return Compose(transforms)


def _resolve_identifier(case_dir: Path, root_dir: Path, mode: str) -> str:
    mode = str(mode)
    if mode == "dirname":
        return case_dir.name
    if mode == "relative_path":
        return case_dir.relative_to(root_dir).as_posix()
    if mode == "parent_dirname":
        return case_dir.parent.name
    if mode == "grandparent_dirname":
        return case_dir.parent.parent.name
    raise ValueError(f'Unsupported identifier mode "{mode}".')


def _resolve_single_pattern_match(case_dir: Path, pattern: str, key_name: str) -> str:
    matches = sorted(case_dir.glob(str(pattern)))
    if not matches:
        raise FileNotFoundError(
            f'No match for key "{key_name}" using pattern "{pattern}" in "{case_dir}".'
        )
    if len(matches) > 1:
        raise ValueError(
            f'Multiple matches for key "{key_name}" using pattern "{pattern}" in "{case_dir}": '
            f"{[str(match) for match in matches]}"
        )
    return str(matches[0].resolve())


def _resolve_path_match(case_dir: Path, pattern_or_patterns: Any, key_name: str) -> str:
    if (
        not isinstance(pattern_or_patterns, (str, bytes, Mapping))
        and hasattr(pattern_or_patterns, "__iter__")
    ):
        pattern_list = [str(pattern) for pattern in pattern_or_patterns]
        errors = []
        for pattern in pattern_list:
            try:
                return _resolve_single_pattern_match(case_dir, pattern, key_name)
            except FileNotFoundError as exc:
                errors.append(str(exc))

        raise FileNotFoundError(
            f'No match for key "{key_name}" using any configured pattern in "{case_dir}". '
            f"Attempts: {errors}"
        )

    return _resolve_single_pattern_match(case_dir, str(pattern_or_patterns), key_name)


def _load_t_label_lookup(dataset_cfg: DictConfig, root_dir: Path) -> Dict[str, int]:
    label_path = dataset_cfg.get("t_label_map_path")
    if label_path is None:
        return {}

    path = Path(str(label_path))
    if not path.is_absolute():
        path = root_dir / path

    if not path.exists():
        raise FileNotFoundError(f'T-label map file not found: "{path}"')

    suffixes = "".join(path.suffixes).lower()
    key_column = str(dataset_cfg.get("t_label_key_column", "case_id"))
    value_column = str(dataset_cfg.get("t_label_value_column", "T_label"))

    if suffixes.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, Mapping):
            return {str(key): int(value) for key, value in data.items()}

        if isinstance(data, list):
            lookup = {}
            for row in data:
                lookup[str(row[key_column])] = int(row[value_column])
            return lookup

        raise ValueError(f'Unsupported JSON structure in T-label map file "{path}".')

    delimiter = "\t" if suffixes.endswith(".tsv") else ","
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return {str(row[key_column]): int(row[value_column]) for row in reader}


def _resolve_t_label(
    dataset_cfg: DictConfig,
    sample_id: str,
    case_dir: Path,
    lookup: Mapping[str, int],
) -> int:
    if "fixed_t_label" in dataset_cfg and dataset_cfg.fixed_t_label is not None:
        return int(dataset_cfg.fixed_t_label)

    lookup_key_mode = str(dataset_cfg.get("t_label_lookup_key_mode", "sample_id"))
    if lookup_key_mode == "sample_id":
        lookup_key = sample_id
    elif lookup_key_mode == "dirname":
        lookup_key = case_dir.name
    elif lookup_key_mode == "parent_dirname":
        lookup_key = case_dir.parent.name
    else:
        raise ValueError(f'Unsupported t_label_lookup_key_mode "{lookup_key_mode}".')

    if lookup_key not in lookup:
        raise KeyError(
            f'No T-label found for lookup key "{lookup_key}" '
            f'in dataset "{dataset_cfg.name}".'
        )
    return int(lookup[lookup_key])


def _discover_dataset_samples(
    cfg: DictConfig,
    dataset_cfg: DictConfig,
) -> List[Dict[str, Any]]:
    layout = str(dataset_cfg.get("layout", "directory_cases"))
    if layout == "flat_suffix_grouped":
        return _discover_flat_suffix_grouped_samples(cfg, dataset_cfg)
    if layout != "directory_cases":
        raise ValueError(f'Unsupported layout "{layout}" for dataset "{dataset_cfg.name}".')

    root_dir = Path(str(dataset_cfg.root_dir)).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f'Dataset root does not exist: "{root_dir}"')

    case_glob = str(dataset_cfg.case_glob)
    case_dirs = sorted(path for path in root_dir.glob(case_glob) if path.is_dir())
    if not case_dirs:
        raise ValueError(
            f'No case directories found for dataset "{dataset_cfg.name}" '
            f'with case_glob "{case_glob}".'
        )

    image_source_keys = _get_dataset_image_source_keys(cfg, dataset_cfg)
    patterns = dataset_cfg.patterns
    seg_source_key = str(cfg.data.source_keys.seg_label)
    t_source_key = str(cfg.data.source_keys.t_label)
    has_t_label = bool(dataset_cfg.get("has_t_label", False))
    t_label_lookup = _load_t_label_lookup(dataset_cfg, root_dir) if has_t_label else {}
    skip_incomplete_cases = bool(dataset_cfg.get("skip_incomplete_cases", True))

    sample_id_mode = str(dataset_cfg.get("sample_id_mode", "relative_path"))
    split_group_mode = str(dataset_cfg.get("split_group_mode", sample_id_mode))

    samples: List[Dict[str, Any]] = []
    for case_dir in case_dirs:
        try:
            sample_id = _resolve_identifier(case_dir, root_dir, sample_id_mode)
            split_group_id = _resolve_identifier(case_dir, root_dir, split_group_mode)

            sample = {
                "dataset_name": str(dataset_cfg.name),
                "sample_id": sample_id,
                "split_group_id": f"{dataset_cfg.name}:{split_group_id}",
                "_image_source_keys": image_source_keys,
                "_has_t_label": has_t_label,
            }

            for image_key in image_source_keys:
                sample[image_key] = _resolve_path_match(case_dir, patterns[image_key], image_key)

            sample[seg_source_key] = _resolve_path_match(
                case_dir,
                patterns[seg_source_key],
                seg_source_key,
            )

            if has_t_label:
                sample[t_source_key] = _resolve_t_label(dataset_cfg, sample_id, case_dir, t_label_lookup)

            samples.append(sample)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            if not skip_incomplete_cases:
                raise
            logger.warning(
                'Skipping incomplete case "%s" in dataset "%s": %s',
                case_dir,
                dataset_cfg.name,
                exc,
            )

    return samples


def _extract_split_group_id_from_sample_id(
    dataset_cfg: DictConfig,
    sample_id: str,
) -> str:
    split_group_regex = dataset_cfg.get("split_group_regex")
    if split_group_regex is None:
        return sample_id

    match = re.search(str(split_group_regex), sample_id)
    if match is None:
        raise ValueError(
            f'Dataset "{dataset_cfg.name}" could not extract split group from sample_id "{sample_id}" '
            f'with regex "{split_group_regex}".'
        )

    if match.groups():
        return str(match.group(1))
    return str(match.group(0))


def _discover_flat_suffix_grouped_samples(
    cfg: DictConfig,
    dataset_cfg: DictConfig,
) -> List[Dict[str, Any]]:
    root_dir = Path(str(dataset_cfg.root_dir)).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f'Dataset root does not exist: "{root_dir}"')

    image_source_keys = _get_dataset_image_source_keys(cfg, dataset_cfg)
    seg_source_key = str(cfg.data.source_keys.seg_label)
    t_source_key = str(cfg.data.source_keys.t_label)
    has_t_label = bool(dataset_cfg.get("has_t_label", False))
    t_label_lookup = _load_t_label_lookup(dataset_cfg, root_dir) if has_t_label else {}
    skip_incomplete_cases = bool(dataset_cfg.get("skip_incomplete_cases", True))
    suffixes = dataset_cfg.suffixes
    grouped_file_glob = str(dataset_cfg.grouped_file_glob)

    key_suffix_pairs = [(image_key, str(suffixes[image_key])) for image_key in image_source_keys]
    key_suffix_pairs.append((seg_source_key, str(suffixes[seg_source_key])))

    suffix_to_key = {suffix: key for key, suffix in key_suffix_pairs}
    grouped_candidates: Dict[str, Dict[str, Any]] = {}

    for file_path in sorted(root_dir.glob(grouped_file_glob)):
        if not file_path.is_file():
            continue

        file_name = file_path.name
        matched_key = None
        matched_suffix = None
        for suffix, key in suffix_to_key.items():
            suffix_token = f"_{suffix}"
            if file_name.endswith(f"{suffix_token}.nii.gz"):
                matched_key = key
                matched_suffix = suffix_token
                break
            if file_name.endswith(f"{suffix_token}.nii"):
                matched_key = key
                matched_suffix = suffix_token
                break

        if matched_key is None or matched_suffix is None:
            continue

        base_name = file_name.split(matched_suffix)[0]
        sample = grouped_candidates.setdefault(base_name, {})
        sample[matched_key] = str(file_path.resolve())

    samples: List[Dict[str, Any]] = []
    for base_name, grouped_sample in sorted(grouped_candidates.items()):
        try:
            split_group_value = _extract_split_group_id_from_sample_id(dataset_cfg, base_name)
            sample = {
                "dataset_name": str(dataset_cfg.name),
                "sample_id": base_name,
                "split_group_id": f"{dataset_cfg.name}:{split_group_value}",
                "_image_source_keys": image_source_keys,
                "_has_t_label": has_t_label,
            }

            for image_key in image_source_keys:
                if image_key not in grouped_sample:
                    raise FileNotFoundError(
                        f'Missing grouped file for image key "{image_key}" in sample "{base_name}".'
                    )
                sample[image_key] = grouped_sample[image_key]

            if seg_source_key not in grouped_sample:
                raise FileNotFoundError(
                    f'Missing grouped file for key "{seg_source_key}" in sample "{base_name}".'
                )
            sample[seg_source_key] = grouped_sample[seg_source_key]

            if has_t_label:
                sample[t_source_key] = _resolve_t_label(dataset_cfg, base_name, root_dir, t_label_lookup)

            samples.append(sample)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            if not skip_incomplete_cases:
                raise
            logger.warning(
                'Skipping incomplete grouped sample "%s" in dataset "%s": %s',
                base_name,
                dataset_cfg.name,
                exc,
            )

    if not samples:
        raise ValueError(f'No grouped samples found for dataset "{dataset_cfg.name}".')

    return samples


def _compute_split_counts(ratios: Mapping[str, Any], total_groups: int) -> Dict[str, int]:
    normalized_ratios = {
        split: float(ratios[split]) / sum(float(ratios[key]) for key in ("training", "validation", "test"))
        for split in ("training", "validation", "test")
    }

    raw_counts = {
        split: normalized_ratios[split] * total_groups
        for split in ("training", "validation", "test")
    }
    split_counts = {split: math.floor(count) for split, count in raw_counts.items()}

    remainder = total_groups - sum(split_counts.values())
    if remainder > 0:
        remainder_order = sorted(
            ("training", "validation", "test"),
            key=lambda split: raw_counts[split] - split_counts[split],
            reverse=True,
        )
        for idx in range(remainder):
            split_counts[remainder_order[idx]] += 1

    return split_counts


def _split_dataset_samples(
    dataset_cfg: DictConfig,
    dataset_samples: List[Dict[str, Any]],
    global_split_cfg: DictConfig,
    dataset_index: int,
) -> Dict[str, List[Dict[str, Any]]]:
    grouped_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in dataset_samples:
        grouped_samples[str(sample["split_group_id"])].append(sample)

    group_ids = sorted(grouped_samples.keys())
    seed = int(dataset_cfg.get("split_seed", global_split_cfg.seed)) + int(dataset_index)
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    ratios = dataset_cfg.get("split_ratios", global_split_cfg.ratios)
    split_counts = _compute_split_counts(ratios, len(group_ids))

    training_end = split_counts["training"]
    validation_end = training_end + split_counts["validation"]

    split_group_ids = {
        "training": group_ids[:training_end],
        "validation": group_ids[training_end:validation_end],
        "test": group_ids[validation_end:],
    }

    split_samples = {"training": [], "validation": [], "test": []}
    for split, split_ids in split_group_ids.items():
        for group_id in split_ids:
            split_samples[split].extend(grouped_samples[group_id])

        split_samples[split] = sorted(
            split_samples[split],
            key=lambda sample: (str(sample["dataset_name"]), str(sample["sample_id"])),
        )

    return split_samples


def build_split_datalist(
    config: Union[str, Path, DictConfig, Mapping[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    cfg = load_medical_loader_config(config)

    split_datalist: Dict[str, List[Dict[str, Any]]] = {
        "training": [],
        "validation": [],
        "test": [],
    }

    enabled_datasets = [ds for ds in cfg.data.datasets if bool(ds.get("enabled", True))]
    for dataset_index, dataset_cfg in enumerate(enabled_datasets):
        dataset_samples = _discover_dataset_samples(cfg, dataset_cfg)
        dataset_splits = _split_dataset_samples(
            dataset_cfg=dataset_cfg,
            dataset_samples=dataset_samples,
            global_split_cfg=cfg.data.split,
            dataset_index=dataset_index,
        )

        for split_name in split_datalist:
            split_datalist[split_name].extend(dataset_splits[split_name])

    return split_datalist


def save_split_datalist(
    config: Union[str, Path, DictConfig, Mapping[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    cfg = load_medical_loader_config(config)
    datalist = build_split_datalist(cfg)

    if output_path is None:
        output_path = cfg.data.get("generated_datalist_path")
    if output_path is None:
        raise ValueError("Please provide output_path or set data.generated_datalist_path in the config.")

    output_path = Path(str(output_path)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(datalist, f, indent=2, ensure_ascii=False)

    logger.info('Saved discovered datalist to "%s".', output_path)
    return output_path


def make_medical_dataset(
    config: Union[str, Path, DictConfig, Mapping[str, Any]],
    split: str = "training",
    is_train: Optional[bool] = None,
):
    cfg = load_medical_loader_config(config)
    datalist = build_split_datalist(cfg)
    if split not in datalist:
        raise KeyError(f'Unknown split "{split}". Available splits: {list(datalist.keys())}')

    split_entries = datalist[split]
    transform = make_medical_transform(cfg, is_train=(split == "training") if is_train is None else is_train)

    cache_mode = str(cfg.data.cache.mode).lower()
    cache_dir = str(cfg.data.cache.cache_dir)

    logger.info("Creating medical dataset from discovered folder structure (%s).", split)
    logger.info("# of %s samples: %d", split, len(split_entries))

    if cache_mode == "persistent":
        return PersistentDataset(split_entries, transform=transform, cache_dir=cache_dir)
    if cache_mode == "cache":
        return CacheDataset(
            split_entries,
            transform=transform,
            cache_rate=float(cfg.data.cache.cache_rate),
            num_workers=int(cfg.data.cache.num_workers),
        )
    return Dataset(split_entries, transform=transform)


def make_medical_dataloader(
    config: Union[str, Path, DictConfig, Mapping[str, Any]],
    split: str = "training",
    is_train: Optional[bool] = None,
) -> DataLoader:
    cfg = load_medical_loader_config(config)
    dataset = make_medical_dataset(cfg, split=split, is_train=is_train)

    if split == "training":
        shuffle = True if is_train is None else bool(is_train)
        drop_last = bool(cfg.data.loader.drop_last)
        batch_size = int(cfg.data.loader.batch_size)
    else:
        shuffle = False
        drop_last = False
        use_sliding_window = bool(cfg.get("inference", {}).get("use_sliding_window", False))
        batch_size = 1 if use_sliding_window else int(cfg.data.loader.batch_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(cfg.data.loader.num_workers),
        pin_memory=bool(cfg.data.loader.pin_memory),
        drop_last=drop_last,
    )


def make_medical_dataloaders(
    config: Union[str, Path, DictConfig, Mapping[str, Any]],
    splits: Sequence[str] = ("training", "validation", "test"),
) -> Dict[str, DataLoader]:
    cfg = load_medical_loader_config(config)
    discovered_splits = build_split_datalist(cfg)

    loaders = {}
    for split in splits:
        if split not in discovered_splits:
            continue
        loaders[split] = make_medical_dataloader(cfg, split=split)
    return loaders
