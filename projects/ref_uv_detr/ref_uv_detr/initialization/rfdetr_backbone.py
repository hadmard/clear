# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""RF-DETR-checkpoint backbone-only initialization helpers.

This mode is different from ``dinov2_backbone_only``:

* the source checkpoint is an official RF-DETR detector checkpoint;
* only keys under ``backbone.0.encoder.`` are loaded into the new model;
* projector, decoder, query embeddings, box/class heads, and Ref-UV modules
  keep their normal constructor initialization for the current task.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from rfdetr.assets.model_weights import download_pretrain_weights, validate_pretrain_weights
from rfdetr.config import ModelConfig
from rfdetr.models.weights import interpolate_position_embeddings
from rfdetr.utilities.logger import get_logger

logger = get_logger()

ENCODER_STATE_PREFIX = "backbone.0.encoder."
"""State-dict prefix used by RF-DETR's DINOv2 encoder inside the detector backbone."""


@dataclass(frozen=True)
class BackboneOnlyLoadReport:
    """Summary of a backbone-only checkpoint load.

    Args:
        source_weights: Source RF-DETR checkpoint path.
        loaded_keys: Number of encoder keys loaded into the target model.
        missing_backbone_keys: Encoder keys expected by the target model but
            absent from the filtered checkpoint.
        unexpected_backbone_keys: Filtered encoder keys not consumed by the
            target model.
    """

    source_weights: str
    loaded_keys: int
    missing_backbone_keys: tuple[str, ...]
    unexpected_backbone_keys: tuple[str, ...]


def default_rfdetr_weights_for_config(config_cls: type[ModelConfig]) -> str:
    """Return the official pretrained checkpoint path for a model config class.

    Args:
        config_cls: RF-DETR model config class.

    Returns:
        Expanded path to the config class's default official checkpoint.

    Raises:
        ValueError: If the config class has no default ``pretrain_weights``.
    """
    default_weights = config_cls.model_fields["pretrain_weights"].default
    if default_weights is None:
        raise ValueError(f"{config_cls.__name__} does not define default pretrain_weights.")
    return config_cls.expand_path(default_weights)


def apply_rfdetr_backbone_only_init(
    model_kwargs: dict[str, Any],
    *,
    config_cls: type[ModelConfig],
    source_weights: str | None,
    resolution: int | None,
) -> str:
    """Mutate model kwargs for official RF-DETR-backbone-only initialization.

    Args:
        model_kwargs: Keyword dictionary passed to an RF-DETR config
            constructor. It is mutated in place and returned for convenience.
        config_cls: RF-DETR config class used to resolve the default official
            checkpoint when ``source_weights`` is omitted.
        source_weights: Optional explicit RF-DETR checkpoint to use as the
            backbone source.
        resolution: Optional user-requested training resolution.

    Returns:
        Expanded source checkpoint path used by the later backbone-only load.
    """
    # Disable RF-DETR's normal full-checkpoint loader. The filtered backbone
    # load happens after the target model is constructed.
    model_kwargs["pretrain_weights"] = None
    if resolution is not None:
        model_kwargs["resolution"] = resolution

    if source_weights is None:
        return default_rfdetr_weights_for_config(config_cls)
    return config_cls.expand_path(source_weights)


def _normalise_checkpoint_model_state(checkpoint: dict[str, Any], source_weights: str) -> dict[str, torch.Tensor]:
    """Extract a bare RF-DETR model state dict from supported checkpoint formats.

    Args:
        checkpoint: Loaded checkpoint object.
        source_weights: Checkpoint path used for error messages.

    Returns:
        Bare model state dict with no Lightning ``"model."`` prefix.

    Raises:
        ValueError: If no supported model state can be found.
    """
    if "model" in checkpoint:
        return checkpoint["model"]

    if "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {source_weights!r} has neither 'model' nor 'state_dict'.")

    model_state = {}
    for key, value in checkpoint["state_dict"].items():
        if not key.startswith("model."):
            continue
        stripped = key[len("model.") :]
        if stripped.startswith("_orig_mod."):
            stripped = stripped[len("_orig_mod.") :]
        model_state[stripped] = value
    if not model_state:
        raise ValueError(f"Checkpoint {source_weights!r} state_dict has no keys under 'model.'.")
    return model_state


def _format_key_sample(keys: list[str], *, limit: int = 5) -> str:
    """Return a compact key sample for compatibility error messages.

    Args:
        keys: Sorted key list.
        limit: Maximum number of keys to include.

    Returns:
        Comma-separated key sample, with an ellipsis when truncated.
    """
    sample = ", ".join(keys[:limit])
    if len(keys) > limit:
        sample += ", ..."
    return sample


def _validate_encoder_state_compatibility(
    nn_model: torch.nn.Module,
    encoder_state: dict[str, torch.Tensor],
    source_weights: str,
) -> None:
    """Validate that filtered encoder weights exactly match the target encoder.

    Args:
        nn_model: Target RF-DETR/LWDETR model.
        encoder_state: Filtered checkpoint state under ``ENCODER_STATE_PREFIX``.
        source_weights: Checkpoint path used for error messages.

    Raises:
        ValueError: If encoder keys are missing, unexpected, or shape-incompatible.
    """
    target_encoder_state = {
        key: value for key, value in nn_model.state_dict().items() if key.startswith(ENCODER_STATE_PREFIX)
    }
    if not target_encoder_state:
        raise ValueError(f"Target model has no '{ENCODER_STATE_PREFIX}' keys.")

    checkpoint_keys = set(encoder_state)
    target_keys = set(target_encoder_state)
    missing = sorted(target_keys.difference(checkpoint_keys))
    unexpected = sorted(checkpoint_keys.difference(target_keys))
    mismatched = []
    for key in sorted(checkpoint_keys.intersection(target_keys)):
        checkpoint_shape = tuple(encoder_state[key].shape)
        target_shape = tuple(target_encoder_state[key].shape)
        if checkpoint_shape != target_shape:
            mismatched.append(f"{key}: checkpoint {checkpoint_shape} != target {target_shape}")

    if not missing and not unexpected and not mismatched:
        return

    parts = [f"RF-DETR encoder weights at {source_weights!r} are not compatible with the target model."]
    if missing:
        parts.append(f"missing target encoder key(s): [{_format_key_sample(missing)}]")
    if unexpected:
        parts.append(f"unexpected checkpoint encoder key(s): [{_format_key_sample(unexpected)}]")
    if mismatched:
        parts.append(f"shape mismatch(es): [{_format_key_sample(mismatched)}]")
    parts.append(
        "Use the official checkpoint for the same RF-DETR variant/encoder settings, "
        "or switch to --no-backbone-only-rfdetr for the DINOv2-only/scratch ablation."
    )
    raise ValueError(" ".join(parts))


def load_rfdetr_backbone_weights(
    nn_model: torch.nn.Module,
    model_config: ModelConfig,
    source_weights: str,
) -> BackboneOnlyLoadReport:
    """Load only official RF-DETR encoder weights into ``nn_model``.

    Args:
        nn_model: Target RF-DETR/LWDETR model.
        model_config: Target model config. Its ``positional_encoding_size`` is
            used to interpolate backbone absolute position embeddings if needed.
        source_weights: Official RF-DETR checkpoint path or model filename.

    Returns:
        Backbone-only load report.

    Raises:
        ValueError: If the checkpoint contains no encoder keys.
    """
    source_weights = model_config.expand_path(source_weights)
    download_pretrain_weights(source_weights)
    validate_pretrain_weights(source_weights, strict=False)

    checkpoint = torch.load(source_weights, map_location="cpu", weights_only=False)
    checkpoint_state = _normalise_checkpoint_model_state(checkpoint, source_weights)
    encoder_state = {
        key: value for key, value in checkpoint_state.items() if key.startswith(ENCODER_STATE_PREFIX)
    }
    if not encoder_state:
        raise ValueError(f"Checkpoint {source_weights!r} contains no '{ENCODER_STATE_PREFIX}' keys.")

    # Keep the official checkpoint architecture, but still support a target
    # config whose PE grid was changed intentionally.
    interpolate_position_embeddings(encoder_state, model_config.positional_encoding_size)
    _validate_encoder_state_compatibility(nn_model, encoder_state, source_weights)
    incompatible = nn_model.load_state_dict(encoder_state, strict=False)

    target_encoder_keys = {key for key in nn_model.state_dict() if key.startswith(ENCODER_STATE_PREFIX)}
    loaded_keys = sorted(set(encoder_state).intersection(target_encoder_keys))
    missing_backbone_keys = tuple(sorted(target_encoder_keys.difference(encoder_state)))
    unexpected_backbone_keys = tuple(
        sorted(key for key in getattr(incompatible, "unexpected_keys", ()) if key.startswith(ENCODER_STATE_PREFIX))
    )

    logger.info(
        "Loaded %d RF-DETR encoder key(s) from %s; %d encoder key(s) missing, %d unexpected.",
        len(loaded_keys),
        os.path.basename(source_weights),
        len(missing_backbone_keys),
        len(unexpected_backbone_keys),
    )
    return BackboneOnlyLoadReport(
        source_weights=source_weights,
        loaded_keys=len(loaded_keys),
        missing_backbone_keys=missing_backbone_keys,
        unexpected_backbone_keys=unexpected_backbone_keys,
    )


def rfdetr_backbone_only_notes(source_weights: str) -> dict[str, Any]:
    """Return JSON-serializable notes for RF-DETR-backbone-only runs.

    Args:
        source_weights: Source official RF-DETR checkpoint path.

    Returns:
        Notes describing the initialization contract.
    """
    return {
        "initialization": "rfdetr_backbone_only",
        "rfdetr_backbone_source": source_weights,
        "loaded_modules": ["backbone.0.encoder"],
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
