# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for Ref-UV initialization-mode wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ref_uv_detr.initialization import (  # noqa: E402
    DINO_V2_COMPATIBLE_RESOLUTION,
    DINO_V2_PATCH_SIZE,
    DINO_V2_POSITIONAL_ENCODING_SIZE,
    apply_dinov2_backbone_only_init,
    apply_rfdetr_backbone_only_init,
    load_rfdetr_backbone_weights,
    rfdetr_backbone_only_notes,
)

from rfdetr.config import RFDETRSmallConfig  # noqa: E402


class TinyBackboneTarget(nn.Module):
    """Tiny model exposing RF-DETR-like encoder and projector state keys."""

    def __init__(self) -> None:
        super().__init__()
        backbone_stage = nn.Module()
        backbone_stage.encoder = nn.Linear(2, 2, bias=False)
        backbone_stage.projector = nn.Linear(2, 2, bias=False)
        nn.init.zeros_(backbone_stage.encoder.weight)
        nn.init.zeros_(backbone_stage.projector.weight)
        self.backbone = nn.ModuleList([backbone_stage])


def _patch_backbone_loader_io(monkeypatch: pytest.MonkeyPatch, checkpoint: dict[str, object]) -> None:
    """Replace checkpoint I/O with an in-memory checkpoint for loader tests."""
    monkeypatch.setattr("ref_uv_detr.initialization.rfdetr_backbone.download_pretrain_weights", lambda *a, **kw: None)
    monkeypatch.setattr("ref_uv_detr.initialization.rfdetr_backbone.validate_pretrain_weights", lambda *a, **kw: None)
    monkeypatch.setattr("ref_uv_detr.initialization.rfdetr_backbone.torch.load", lambda *a, **kw: checkpoint)


def test_apply_dinov2_backbone_only_init_disables_rfdetr_checkpoint() -> None:
    """Backbone-only mode must not request an RF-DETR detector checkpoint."""
    kwargs: dict[str, object] = {}

    apply_dinov2_backbone_only_init(kwargs, resolution=None)

    assert kwargs["pretrain_weights"] is None
    assert kwargs["patch_size"] == DINO_V2_PATCH_SIZE
    assert kwargs["positional_encoding_size"] == DINO_V2_POSITIONAL_ENCODING_SIZE
    assert kwargs["resolution"] == DINO_V2_COMPATIBLE_RESOLUTION


def test_apply_dinov2_backbone_only_init_preserves_resolution_override() -> None:
    """A user-provided resolution should be explicit, not hidden by the helper."""
    kwargs: dict[str, object] = {}

    apply_dinov2_backbone_only_init(kwargs, resolution=672)

    assert kwargs["resolution"] == 672


def test_apply_rfdetr_backbone_only_init_disables_full_checkpoint() -> None:
    """RF-DETR-backbone mode must avoid the normal full-checkpoint loader."""
    kwargs: dict[str, object] = {}

    source = apply_rfdetr_backbone_only_init(
        kwargs,
        config_cls=RFDETRSmallConfig,
        source_weights=None,
        resolution=512,
    )

    assert kwargs["pretrain_weights"] is None
    assert kwargs["resolution"] == 512
    assert source.endswith("rf-detr-small.pth")


def test_rfdetr_backbone_only_notes_records_random_detector_modules() -> None:
    """Run notes should make the filtered initialization explicit."""
    notes = rfdetr_backbone_only_notes("/tmp/rf-detr-small.pth")

    assert notes["initialization"] == "rfdetr_backbone_only"
    assert notes["rfdetr_backbone_source"] == "/tmp/rf-detr-small.pth"
    assert notes["loaded_modules"] == ["backbone.0.encoder"]
    assert "backbone.0.projector" in notes["random_init_modules"]
    assert "transformer" in notes["random_init_modules"]


def test_load_rfdetr_backbone_weights_loads_encoder_not_projector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Filtered RF-DETR load should update only ``backbone.0.encoder`` keys."""
    target = TinyBackboneTarget()
    checkpoint = {
        "model": {
            "backbone.0.encoder.weight": torch.ones((2, 2)),
            "backbone.0.projector.weight": torch.ones((2, 2)),
        }
    }
    _patch_backbone_loader_io(monkeypatch, checkpoint)
    config = RFDETRSmallConfig(pretrain_weights=None, device="cpu")

    report = load_rfdetr_backbone_weights(target, config, "/tmp/rf-detr-small.pth")

    assert report.loaded_keys == 1
    assert report.missing_backbone_keys == ()
    assert report.unexpected_backbone_keys == ()
    assert torch.equal(target.backbone[0].encoder.weight, torch.ones((2, 2)))
    assert torch.equal(target.backbone[0].projector.weight, torch.zeros((2, 2)))


def test_load_rfdetr_backbone_weights_rejects_encoder_shape_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compatibility check should fail clearly before ``load_state_dict`` shape errors."""
    target = TinyBackboneTarget()
    checkpoint = {"model": {"backbone.0.encoder.weight": torch.ones((3, 2))}}
    _patch_backbone_loader_io(monkeypatch, checkpoint)
    config = RFDETRSmallConfig(pretrain_weights=None, device="cpu")

    with pytest.raises(ValueError, match="shape mismatch"):
        load_rfdetr_backbone_weights(target, config, "/tmp/rf-detr-base.pth")


def test_uv_teacher_defaults_to_rfdetr_backbone_only(monkeypatch) -> None:
    """The UV teacher script should default to the stronger backbone-only experiment."""
    import train_uv_teacher

    monkeypatch.setattr(sys, "argv", ["train_uv_teacher.py"])

    args = train_uv_teacher.parse_args()

    assert args.backbone_only_rfdetr is True


def test_ref_uv_build_configs_defaults_to_rfdetr_backbone_only(monkeypatch) -> None:
    """Ref-UV defaults should initialize RF-DETR backbone only and keep detector parts random."""
    import train_ref_uv

    monkeypatch.setattr(sys, "argv", ["train_ref_uv.py"])
    monkeypatch.setattr(train_ref_uv.RFDETR, "_detect_num_classes_for_training", staticmethod(lambda _: 2))

    args = train_ref_uv.parse_args()
    model_config, train_config = train_ref_uv.build_configs(args)

    assert model_config.pretrain_weights is None
    assert model_config.patch_size == RFDETRSmallConfig.model_fields["patch_size"].default
    assert model_config.resolution == RFDETRSmallConfig.model_fields["resolution"].default
    assert train_config.notes["initialization"] == "rfdetr_backbone_only"
    assert train_config.notes["rfdetr_backbone_source"].endswith("rf-detr-small.pth")


def test_ref_uv_build_configs_keeps_dinov2_backbone_ablation(monkeypatch) -> None:
    """The previous DINOv2-only baseline should remain available."""
    import train_ref_uv

    monkeypatch.setattr(sys, "argv", ["train_ref_uv.py", "--no-backbone-only-rfdetr"])
    monkeypatch.setattr(train_ref_uv.RFDETR, "_detect_num_classes_for_training", staticmethod(lambda _: 2))

    args = train_ref_uv.parse_args()
    model_config, train_config = train_ref_uv.build_configs(args)

    assert model_config.pretrain_weights is None
    assert model_config.patch_size == DINO_V2_PATCH_SIZE
    assert model_config.positional_encoding_size == DINO_V2_POSITIONAL_ENCODING_SIZE
    assert model_config.resolution == DINO_V2_COMPATIBLE_RESOLUTION
    assert train_config.notes["initialization"] == "dinov2_backbone_only"
