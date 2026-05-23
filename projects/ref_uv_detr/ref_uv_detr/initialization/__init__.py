# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Initialization helpers for Ref-UV experiments."""

from ref_uv_detr.initialization.dinov2_backbone import (
    DINO_V2_COMPATIBLE_RESOLUTION,
    DINO_V2_IMAGE_SIZE,
    DINO_V2_PATCH_SIZE,
    DINO_V2_POSITIONAL_ENCODING_SIZE,
    apply_dinov2_backbone_only_init,
    dinov2_backbone_only_notes,
)
from ref_uv_detr.initialization.rfdetr_backbone import (
    BackboneOnlyLoadReport,
    apply_rfdetr_backbone_only_init,
    default_rfdetr_weights_for_config,
    load_rfdetr_backbone_weights,
    rfdetr_backbone_only_notes,
)

__all__ = [
    "BackboneOnlyLoadReport",
    "DINO_V2_COMPATIBLE_RESOLUTION",
    "DINO_V2_IMAGE_SIZE",
    "DINO_V2_PATCH_SIZE",
    "DINO_V2_POSITIONAL_ENCODING_SIZE",
    "apply_dinov2_backbone_only_init",
    "apply_rfdetr_backbone_only_init",
    "default_rfdetr_weights_for_config",
    "dinov2_backbone_only_notes",
    "load_rfdetr_backbone_weights",
    "rfdetr_backbone_only_notes",
]
