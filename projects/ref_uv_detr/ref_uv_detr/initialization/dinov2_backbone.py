# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""DINOv2-only backbone initialization helpers.

RF-DETR has two different pretrained-weight paths:

* ``pretrain_weights=<rf-detr checkpoint>`` loads a whole detector checkpoint,
  including the DETR transformer, query embeddings, and detection heads.
* ``pretrain_weights=None`` skips the RF-DETR detector checkpoint. The model
  builder then asks the DINOv2 backbone to load its original visual pretraining
  weights when the backbone shape is compatible with DINOv2.

The project experiments in this folder need the second behavior: DINOv2 starts
from visual pretraining, while RF-DETR's projector/decoder/query/head modules
start from their normal random initialization and all parameters remain
trainable.  This helper keeps that setup explicit in the training scripts.
"""

from __future__ import annotations

from typing import Any

DINO_V2_PATCH_SIZE = 14
"""Patch size used by the original DINOv2 checkpoints."""

DINO_V2_IMAGE_SIZE = 518
"""Training image size baked into the original DINOv2 absolute position table."""

DINO_V2_POSITIONAL_ENCODING_SIZE = DINO_V2_IMAGE_SIZE // DINO_V2_PATCH_SIZE
"""DINOv2's square position table side length in patch tokens."""

DINO_V2_COMPATIBLE_RESOLUTION = 560
"""Safe default RF-DETR resolution for DINOv2-compatible windowed backbones.

``560`` is divisible by both common Ref-UV window block sizes:

* base-style ``patch_size=14, num_windows=4`` -> block size 56
* small/nano/medium/large-style ``patch_size=14, num_windows=2`` -> block size 28

The DINOv2 position table remains at 518 px (37 patches) and is interpolated by
the backbone at runtime when training at 560 px.
"""


def apply_dinov2_backbone_only_init(model_kwargs: dict[str, Any], *, resolution: int | None) -> dict[str, Any]:
    """Mutate model-construction kwargs for DINOv2-backbone-only initialization.

    Args:
        model_kwargs: Keyword dictionary passed to an RF-DETR variant/config
            constructor. It is mutated in place and returned for convenience.
        resolution: Optional user-requested training resolution. When omitted,
            a DINOv2/window-attention compatible default is used.

    Returns:
        The same ``model_kwargs`` dictionary after applying initialization
        settings.
    """
    # This is the key switch: no RF-DETR detector checkpoint is loaded, so the
    # transformer decoder, query embeddings, detection heads, and projector keep
    # their constructor random initialization.
    model_kwargs["pretrain_weights"] = None

    # RF-DETR's small/nano/medium/large variants default to patch_size=16. The
    # original DINOv2 checkpoint is patch_size=14, and RF-DETR's backbone code
    # intentionally refuses to load DINOv2 when those shapes disagree. In this
    # experiment mode we choose the DINOv2-native patch table so the backbone
    # really does load visual pretraining rather than silently becoming scratch.
    model_kwargs["patch_size"] = DINO_V2_PATCH_SIZE
    model_kwargs["positional_encoding_size"] = DINO_V2_POSITIONAL_ENCODING_SIZE

    # Keep a user override if present, otherwise pick a resolution known to be
    # valid for both RF-DETR base-style and small-style window sizes.
    if resolution is not None:
        model_kwargs["resolution"] = resolution
    else:
        model_kwargs["resolution"] = DINO_V2_COMPATIBLE_RESOLUTION

    return model_kwargs


def dinov2_backbone_only_notes() -> dict[str, Any]:
    """Return JSON-serializable provenance notes for training artifacts.

    Returns:
        Notes describing the intended initialization contract.
    """
    return {
        "initialization": "dinov2_backbone_only",
        "rf_detr_checkpoint": None,
        "dinov2_patch_size": DINO_V2_PATCH_SIZE,
        "dinov2_position_table_image_size": DINO_V2_IMAGE_SIZE,
        "random_init_modules": [
            "backbone.0.projector",
            "transformer",
            "query_feat",
            "refpoint_embed",
            "class_embed",
            "bbox_embed",
            "Ref-UV reference branch when present",
        ],
        "trainability": "all parameters remain trainable unless another config explicitly freezes them",
    }
