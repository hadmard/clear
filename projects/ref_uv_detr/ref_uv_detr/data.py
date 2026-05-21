# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# mypy: disable-error-code=misc

"""Paired UV/white COCO data loading for Ref-UV DETR.

The datamodule mirrors RF-DETR's Lightning data path but swaps in a paired
dataset that keeps UV and white-light images under the same geometric transform.
Only the UV image is returned as the detector input; the white-guided RNFR prior
is stored in the target dict as ``ref_uv_prior`` and consumed by the model
wrapper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import albumentations as alb
import numpy as np
import torch
import torchvision
from PIL import Image

from ref_uv_detr.priors import REFERENCE_PRIOR_CHANNELS, build_reference_prior
from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.datasets.aug_config import AUG_CONFIG
from rfdetr.datasets.coco import ConvertCoco, _build_train_resize_config, compute_multi_scale_scales
from rfdetr.datasets.transforms import _build_albu_transform, _is_geometric_transform
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.utilities.box_ops import box_xyxy_to_cxcywh
from rfdetr.utilities.logger import get_logger

logger = get_logger()

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _iter_config_entries(config: Dict[str, Any] | Sequence[Dict[str, Any]]) -> Iterable[tuple[str, Dict[str, Any]]]:
    """Yield single Albumentations config entries.

    Args:
        config: RF-DETR augmentation config, either as a mapping or a list of
            single-key mappings.

    Yields:
        ``(name, params)`` pairs.

    Raises:
        ValueError: If a list entry is not a single-key mapping.
    """
    if isinstance(config, dict):
        for name, params in config.items():
            yield name, params
        return

    for entry in config:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"Augmentation entries must be single-key dictionaries, got {entry!r}.")
        name, params = next(iter(entry.items()))
        yield name, params


def _build_transforms_from_config(
    config: Dict[str, Any] | Sequence[Dict[str, Any]],
    *,
    geometric_only: bool,
) -> list[alb.BasicTransform]:
    """Build Albumentations transforms from RF-DETR-style config.

    Args:
        config: RF-DETR augmentation config.
        geometric_only: If ``True``, drop photometric transforms so white-light
            reference intensities remain a stable structural prior.

    Returns:
        List of instantiated Albumentations transforms.
    """
    transforms: list[alb.BasicTransform] = []
    for name, params in _iter_config_entries(config):
        transform = _build_albu_transform(name, params)
        if geometric_only and not _is_geometric_transform(transform):
            logger.debug("Skipping non-geometric paired transform %s for white-reference stability.", name)
            continue
        transforms.append(transform)
    return transforms


def _normalize_uv_tensor(uv_rgb: np.ndarray) -> torch.Tensor:
    """Convert transformed UV RGB pixels into RF-DETR's normalized tensor.

    Args:
        uv_rgb: ``H x W x 3`` RGB array after augmentation.

    Returns:
        Float tensor with shape ``3 x H x W``.
    """
    uv = uv_rgb.astype(np.float32, copy=False)
    if uv_rgb.dtype == np.uint8:
        uv = uv / 255.0
    uv = np.clip(uv, 0.0, 1.0)
    tensor = torch.from_numpy(np.moveaxis(uv, -1, 0)).float()
    mean = tensor.new_tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(_IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def _filter_target_fields(target: Dict[str, Any], num_boxes: int, kept_idxs: Sequence[int]) -> Dict[str, Any]:
    """Filter per-instance target fields after Albumentations box pruning.

    Args:
        target: Original target dictionary.
        num_boxes: Number of boxes before augmentation.
        kept_idxs: Original box indices retained by Albumentations.

    Returns:
        Shallow target copy with per-instance tensors filtered.
    """
    global_fields = {"boxes", "labels", "orig_size", "size", "image_id"}
    kept = torch.as_tensor(list(kept_idxs), dtype=torch.long)
    out = target.copy()
    for key, value in target.items():
        if key in global_fields:
            continue
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == num_boxes:
            out[key] = value[kept]
    return out


class PairedReferenceTransform:
    """Apply RF-DETR-compatible paired UV/white transforms and build priors.

    Args:
        image_set: Dataset split name (``"train"``, ``"val"``, or ``"test"``).
        resolution: Target RF-DETR resolution.
        multi_scale: Whether to use RF-DETR's multi-scale training scales.
        expanded_scales: Whether to broaden the multi-scale range.
        skip_random_resize: Whether to use only the largest multi-scale value.
        square_resize_div_64: Whether to use square RF-DETR resizing.
        patch_size: Model patch size.
        num_windows: Model attention-window count.
        aug_config: RF-DETR augmentation config. ``None`` uses the project
            default.
        geometric_only: If ``True``, only paired geometric augmentations are
            applied to both UV and white images.
    """

    def __init__(
        self,
        image_set: str,
        resolution: int,
        *,
        multi_scale: bool,
        expanded_scales: bool,
        skip_random_resize: bool,
        square_resize_div_64: bool,
        patch_size: int,
        num_windows: int,
        aug_config: Optional[Dict[str, Any]],
        geometric_only: bool = True,
    ) -> None:
        self.image_set = image_set
        self.resolution = resolution

        if image_set == "train":
            scales = [resolution]
            if multi_scale:
                scales = compute_multi_scale_scales(resolution, expanded_scales, patch_size, num_windows)
                if skip_random_resize:
                    scales = [scales[-1]]
            resize_config = _build_train_resize_config(scales, square=square_resize_div_64, max_size=1333)
            transforms = _build_transforms_from_config(resize_config, geometric_only=False)
            resolved_aug_config = aug_config if aug_config is not None else AUG_CONFIG
            transforms.extend(_build_transforms_from_config(resolved_aug_config, geometric_only=geometric_only))
        elif image_set in {"val", "test", "val_speed"}:
            if square_resize_div_64:
                resize_config = [{"Resize": {"height": resolution, "width": resolution}}]
            else:
                resize_config = [
                    {"SmallestMaxSize": {"max_size": resolution}},
                    {"LongestMaxSize": {"max_size": 1333}},
                ]
            transforms = _build_transforms_from_config(resize_config, geometric_only=False)
        else:
            raise ValueError(f"Unknown image_set={image_set!r}.")

        self.transform = alb.Compose(
            transforms,
            bbox_params=alb.BboxParams(
                format="pascal_voc",
                label_fields=["category_ids", "idxs"],
                min_visibility=0.0,
                clip=True,
            ),
            additional_targets={"white": "image"},
        )

    def __call__(
        self,
        uv_image: Image.Image,
        white_image: Image.Image,
        target: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Transform a paired sample and attach ``ref_uv_prior`` to the target.

        Args:
            uv_image: UV PIL RGB image.
            white_image: White-light PIL RGB image.
            target: RF-DETR target dict after COCO conversion, with absolute
                ``xyxy`` boxes.

        Returns:
            Tuple of normalized UV image tensor and target dict.
        """
        uv_np = np.array(uv_image.convert("RGB"))
        white_np = np.array(white_image.convert("RGB"))
        boxes = target["boxes"]
        labels = target["labels"]
        num_boxes = int(boxes.shape[0])
        idxs = list(range(num_boxes))
        augmented = self.transform(
            image=uv_np,
            white=white_np,
            bboxes=boxes.cpu().numpy() if num_boxes else np.zeros((0, 4), dtype=np.float32),
            category_ids=labels.cpu().tolist(),
            idxs=idxs,
        )

        uv_aug = augmented["image"]
        white_aug = augmented["white"]
        height, width = uv_aug.shape[:2]
        target_out = _filter_target_fields(target, num_boxes, augmented.get("idxs", idxs))

        bboxes_aug = np.asarray(augmented["bboxes"], dtype=np.float32).reshape(-1, 4)
        labels_aug = torch.as_tensor(augmented["category_ids"], dtype=torch.long)
        boxes_xyxy = torch.as_tensor(bboxes_aug, dtype=torch.float32)
        if boxes_xyxy.numel() > 0:
            target_out["boxes"] = box_xyxy_to_cxcywh(boxes_xyxy) / torch.tensor(
                [width, height, width, height],
                dtype=torch.float32,
            )
            target_out["area"] = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        else:
            target_out["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target_out["area"] = torch.zeros((0,), dtype=torch.float32)
        target_out["labels"] = labels_aug
        target_out["size"] = torch.as_tensor([height, width], dtype=torch.int64)

        prior = build_reference_prior(uv_aug, white_aug)
        if prior.shape[0] != REFERENCE_PRIOR_CHANNELS:
            raise RuntimeError(f"Expected {REFERENCE_PRIOR_CHANNELS} prior channels, got {prior.shape[0]}.")
        target_out["ref_uv_prior"] = torch.from_numpy(prior).float()
        target_out["ref_uv_prior_valid"] = torch.ones((), dtype=torch.bool)
        return _normalize_uv_tensor(uv_aug), target_out


class PairedCocoDetection(torchvision.datasets.CocoDetection):
    """COCO detection dataset with paired white-light reference images.

    Args:
        img_folder: UV image folder.
        ann_file: COCO annotation file for UV images.
        white_root: Root directory containing white-light images.
        transforms: Paired transform that receives UV image, white image, and
            target.
        remap_category_ids: Whether to remap sparse category IDs to contiguous
            detector labels.
        uv_token: Optional substring in UV file names to replace when looking
            for white images.
        white_token: Replacement substring for ``uv_token``.
        strict_pairs: If ``True``, missing white images raise
            ``FileNotFoundError``.
    """

    def __init__(
        self,
        img_folder: str | Path,
        ann_file: str | Path,
        *,
        white_root: str | Path,
        transforms: PairedReferenceTransform,
        remap_category_ids: bool = True,
        uv_token: str = "uv",
        white_token: str = "white",
        strict_pairs: bool = True,
    ) -> None:
        super().__init__(str(img_folder), str(ann_file))
        self._transforms = transforms
        self.white_root = Path(white_root)
        folder = Path(img_folder)
        self.split_name = folder.name
        if folder.name.lower() in {"uv", "images"} and folder.parent.name.lower() in {
            "train",
            "valid",
            "val",
            "test",
        }:
            self.split_name = folder.parent.name
        self.uv_token = uv_token
        self.white_token = white_token
        self.strict_pairs = strict_pairs

        self.cat2label: dict[Any, int] | None
        self.label2cat: dict[int, Any] | None
        if remap_category_ids:
            self.cat2label = {cat_id: i for i, cat_id in enumerate(sorted(self.coco.cats.keys()))}
            self.label2cat = {label: cat_id for cat_id, label in self.cat2label.items()}
            setattr(self.coco, "label2cat", self.label2cat)
        else:
            self.cat2label = None
            self.label2cat = None
        self.prepare = ConvertCoco(include_masks=False, cat2label=self.cat2label)

    def _white_candidates(self, file_name: str, uv_path: Path) -> list[Path]:
        """Return candidate white-light paths for one UV file name.

        Args:
            file_name: COCO ``file_name`` entry.
            uv_path: Resolved UV path.

        Returns:
            Ordered candidate paths.
        """
        names = [file_name]
        lowered = file_name.lower()
        if self.uv_token and self.uv_token.lower() in lowered:
            # Preserve the original path casing except for the explicit token.
            white_name = file_name.replace(self.uv_token, self.white_token)
            white_name = white_name.replace(self.uv_token.upper(), self.white_token.upper())
            names.append(white_name)

        candidates: list[Path] = []
        split_aliases = [self.split_name]
        if self.split_name == "valid":
            split_aliases.append("val")
        elif self.split_name == "val":
            split_aliases.append("valid")
        for name in names:
            for split in split_aliases:
                candidates.append(self.white_root / split / "white" / name)
                candidates.append(self.white_root / split / name)
            candidates.append(self.white_root / name)
            candidates.append(uv_path.with_name(Path(name).name))
        return candidates

    def _load_white_image(self, file_name: str, uv_path: Path) -> Image.Image:
        """Load the white-light image paired with a UV image.

        Args:
            file_name: COCO ``file_name`` entry.
            uv_path: Resolved UV image path.

        Returns:
            PIL RGB white-light image.

        Raises:
            FileNotFoundError: If no candidate exists and ``strict_pairs`` is
                enabled.
        """
        for candidate in self._white_candidates(file_name, uv_path):
            if candidate.exists():
                return Image.open(candidate).convert("RGB")
        if self.strict_pairs:
            tried = "\n".join(str(path) for path in self._white_candidates(file_name, uv_path)[:6])
            raise FileNotFoundError(f"No white-light pair found for {uv_path}. Tried:\n{tried}")
        logger.warning("Missing white-light pair for %s; using UV image as a neutral fallback.", uv_path)
        return Image.open(uv_path).convert("RGB")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Return one paired training sample.

        Args:
            idx: Dataset index.

        Returns:
            Tuple of normalized UV tensor and RF-DETR target dict.
        """
        uv_image, annotations = super().__getitem__(idx)
        image_id = self.ids[idx]
        info = self.coco.loadImgs(image_id)[0]
        file_name = info["file_name"]
        uv_path = Path(self.root) / file_name
        white_image = self._load_white_image(file_name, uv_path)

        target = {"image_id": image_id, "annotations": annotations}
        uv_image, target = self.prepare(uv_image, target)
        return self._transforms(uv_image, white_image, target)


def _split_paths(dataset_dir: str | Path, image_set: str) -> tuple[Path, Path]:
    """Resolve RF-DETR Roboflow-COCO split paths.

    Args:
        dataset_dir: Dataset root.
        image_set: RF-DETR split name.

    Returns:
        Tuple of ``(image_folder, annotation_file)``.

    Raises:
        FileNotFoundError: If the expected split annotation file is missing.
    """
    root = Path(dataset_dir)
    split_key = image_set.split("_")[0]
    split_aliases = {
        "train": ("train",),
        "val": ("valid", "val"),
        "test": ("test",),
    }.get(split_key, (split_key,))
    candidates: list[tuple[Path, Path]] = []
    for split in split_aliases:
        candidates.extend(
            [
                (root / split, root / split / "_annotations.coco.json"),
                (root / split / "uv", root / split / "uv" / "_annotations.coco.json"),
            ]
        )

    for img_folder, ann_file in candidates:
        if ann_file.exists():
            return img_folder, ann_file

    tried = "\n".join(str(ann_file) for _, ann_file in candidates)
    raise FileNotFoundError(
        "Ref-UV DETR expects either Roboflow COCO split folders or "
        f"dual folders like train/uv and val/uv. Tried:\n{tried}"
    )


def build_paired_roboflow_coco(
    image_set: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    white_root: str | Path,
    uv_token: str,
    white_token: str,
    strict_pairs: bool,
) -> PairedCocoDetection:
    """Build the paired Roboflow-COCO dataset for a split.

    Args:
        image_set: Split name.
        model_config: RF-DETR model configuration.
        train_config: RF-DETR training configuration.
        white_root: White-light image root.
        uv_token: Filename token identifying UV images.
        white_token: Filename token identifying white images.
        strict_pairs: Whether missing pairs are fatal.

    Returns:
        Paired COCO dataset.
    """
    img_folder, ann_file = _split_paths(train_config.dataset_dir, image_set)
    transform = PairedReferenceTransform(
        image_set=image_set,
        resolution=model_config.resolution,
        multi_scale=train_config.multi_scale,
        expanded_scales=train_config.expanded_scales,
        skip_random_resize=not train_config.do_random_resize_via_padding,
        square_resize_div_64=train_config.square_resize_div_64,
        patch_size=model_config.patch_size,
        num_windows=model_config.num_windows,
        aug_config=train_config.aug_config,
    )
    return PairedCocoDetection(
        img_folder,
        ann_file,
        white_root=white_root,
        transforms=transform,
        remap_category_ids=True,
        uv_token=uv_token,
        white_token=white_token,
        strict_pairs=strict_pairs,
    )


class PairedRFDETRDataModule(RFDETRDataModule):
    """RF-DETR datamodule using paired UV/white COCO samples.

    Args:
        model_config: RF-DETR model configuration.
        train_config: RF-DETR training configuration.
        white_root: White-light image root.
        uv_token: Filename token identifying UV images.
        white_token: Filename token identifying white images.
        strict_pairs: Whether missing white-light pairs should raise.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
        *,
        white_root: str | Path,
        uv_token: str = "uv",
        white_token: str = "white",
        strict_pairs: bool = True,
    ) -> None:
        super().__init__(model_config, train_config)
        self.white_root = Path(white_root)
        self.uv_token = uv_token
        self.white_token = white_token
        self.strict_pairs = strict_pairs

    def setup(self, stage: str) -> None:
        """Build paired datasets for the requested Lightning stage.

        Args:
            stage: Lightning stage name.
        """
        if stage == "fit":
            if self._dataset_train is None:
                self._dataset_train = build_paired_roboflow_coco(
                    "train",
                    self.model_config,
                    self.train_config,
                    white_root=self.white_root,
                    uv_token=self.uv_token,
                    white_token=self.white_token,
                    strict_pairs=self.strict_pairs,
                )
            if self._dataset_val is None:
                self._dataset_val = build_paired_roboflow_coco(
                    "val",
                    self.model_config,
                    self.train_config,
                    white_root=self.white_root,
                    uv_token=self.uv_token,
                    white_token=self.white_token,
                    strict_pairs=self.strict_pairs,
                )
        elif stage == "validate":
            if self._dataset_val is None:
                self._dataset_val = build_paired_roboflow_coco(
                    "val",
                    self.model_config,
                    self.train_config,
                    white_root=self.white_root,
                    uv_token=self.uv_token,
                    white_token=self.white_token,
                    strict_pairs=self.strict_pairs,
                )
        elif stage == "test":
            if self._dataset_test is None:
                self._dataset_test = build_paired_roboflow_coco(
                    "test",
                    self.model_config,
                    self.train_config,
                    white_root=self.white_root,
                    uv_token=self.uv_token,
                    white_token=self.white_token,
                    strict_pairs=self.strict_pairs,
                )
        elif stage == "predict":
            if self._dataset_val is None:
                self._dataset_val = build_paired_roboflow_coco(
                    "val",
                    self.model_config,
                    self.train_config,
                    white_root=self.white_root,
                    uv_token=self.uv_token,
                    white_token=self.white_token,
                    strict_pairs=self.strict_pairs,
                )
