# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# mypy: disable-error-code="misc"

"""Ref-UV DETR model wrapper.

This module keeps RF-DETR's UV backbone, detection heads, and training API, then
adds a narrow reference path inside the decoder cross-attention.  White/RNFR
features therefore influence object queries before the original RF-DETR heads,
not through a separate head-fusion shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F  # noqa: N812 -- project-conventional alias
from torch import Tensor, nn

from ref_uv_detr.decoder_fusion import (
    DecoderReferenceFusionConfig,
    clear_decoder_reference_context,
    collect_decoder_reference_aux,
    enable_decoder_reference_fusion,
    set_decoder_reference_context,
)
from ref_uv_detr.priors import REFERENCE_PRIOR_CHANNELS


def _group_norm(channels: int) -> nn.GroupNorm:
    """Return a small, channel-safe GroupNorm layer.

    Args:
        channels: Number of channels to normalize.

    Returns:
        GroupNorm layer.
    """
    groups = 8 if channels % 8 == 0 else 1
    return nn.GroupNorm(groups, channels)


@dataclass
class RefUVConfig:
    """Configuration for the decoder-level reference fusion path.

    Args:
        prior_channels: Number of RNFR/ExB/edge/raw-white prior channels.
        max_alignment_offset_px: Maximum learned alignment offset per feature
            level, in feature-map pixels.
        max_cls_beta: Retained for older configs. Decoder-only fusion does not
            use a separate classification head-fusion beta.
        max_box_beta: Absolute cap for decoder reference residual strength.
        initial_ref_beta: Initial decoder reference residual strength. A small
            positive value gives the reference branch gradient from the first
            epoch; ``0`` keeps a strict UV-only start.
        gate_bias: Initial reliability-gate bias. Negative values make the
            reference path conservative at the beginning of fine-tuning.
        decoder_fusion_start_layer: First decoder layer allowed to use reference
            cross-attention. The default keeps layer 0 UV-only.
    """

    prior_channels: int = REFERENCE_PRIOR_CHANNELS
    max_alignment_offset_px: float = 2.0
    max_cls_beta: float = 0.10
    max_box_beta: float = 0.30
    initial_ref_beta: float = 0.0
    gate_bias: float = -2.0
    decoder_fusion_start_layer: int = 1


class WhiteReferencedResidualTokenizer(nn.Module):
    """Encode RNFR and white-structure maps into multi-scale prior features.

    Args:
        prior_channels: Number of input prior channels.
        hidden_dim: Detector hidden dimension.
        num_levels: Number of RF-DETR feature levels.
    """

    def __init__(self, prior_channels: int, hidden_dim: int, num_levels: int) -> None:
        super().__init__()
        stem_dim = max(32, hidden_dim // 4)
        self.stem = nn.Sequential(
            nn.Conv2d(prior_channels, stem_dim, kernel_size=3, padding=1),
            _group_norm(stem_dim),
            nn.GELU(),
            nn.Conv2d(stem_dim, stem_dim, kernel_size=3, padding=1),
            _group_norm(stem_dim),
            nn.GELU(),
        )
        self.level_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(stem_dim, hidden_dim, kernel_size=1),
                    _group_norm(hidden_dim),
                    nn.GELU(),
                )
                for _ in range(num_levels)
            ]
        )
        self.lesion_head = nn.Conv2d(stem_dim, 1, kernel_size=1)

    def forward(self, prior: Tensor, level_shapes: list[tuple[int, int]]) -> tuple[list[Tensor], Tensor]:
        """Encode prior maps at the detector's feature resolutions.

        Args:
            prior: Batched prior tensor with shape ``B x C x H x W``.
            level_shapes: Target ``(height, width)`` for each detector feature
                level.

        Returns:
            Tuple of multi-scale prior features and dense lesionness logits.
        """
        base = self.stem(prior)
        lesion_logits = self.lesion_head(base)
        levels = []
        for shape, proj in zip(level_shapes, self.level_projs):
            resized = F.interpolate(base, size=shape, mode="bilinear", align_corners=False)
            levels.append(proj(resized))
        return levels, lesion_logits


class MisalignmentAwareReferenceAlignment(nn.Module):
    """Align white-reference prior features to UV detector features.

    Args:
        hidden_dim: Detector hidden dimension.
        num_levels: Number of feature levels.
        max_offset_px: Maximum offset in feature-map pixels.
    """

    def __init__(self, hidden_dim: int, num_levels: int, max_offset_px: float) -> None:
        super().__init__()
        self.max_offset_px = max_offset_px
        self.offset_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
                    _group_norm(hidden_dim),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1),
                )
                for _ in range(num_levels)
            ]
        )
        for head in self.offset_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _base_grid(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Build a normalized sampling grid.

        Args:
            batch_size: Batch size.
            height: Feature-map height.
            width: Feature-map width.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Grid tensor with shape ``B x H x W x 2``.
        """
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        grid = torch.stack((x, y), dim=-1)
        return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def forward(self, uv_features: list[Tensor], prior_features: list[Tensor]) -> list[Tensor]:
        """Align prior features to UV features.

        Args:
            uv_features: RF-DETR UV feature maps.
            prior_features: Reference prior feature maps.

        Returns:
            Aligned prior feature maps.
        """
        aligned = []
        for uv, prior, head in zip(uv_features, prior_features, self.offset_heads):
            batch_size, _, height, width = prior.shape
            offsets = torch.tanh(head(torch.cat([uv, prior], dim=1))) * self.max_offset_px
            scale_x = 2.0 / max(width - 1, 1)
            scale_y = 2.0 / max(height - 1, 1)
            offset_grid = torch.stack((offsets[:, 0] * scale_x, offsets[:, 1] * scale_y), dim=-1)
            grid = self._base_grid(batch_size, height, width, prior.device, prior.dtype) + offset_grid
            aligned.append(F.grid_sample(prior, grid, mode="bilinear", padding_mode="border", align_corners=True))
        return aligned


class RefUVLWDETR(nn.Module):
    """RF-DETR detector with decoder-level reference fusion.

    Args:
        base_model: Fully constructed RF-DETR ``LWDETR`` model.
        config: Ref-UV fusion configuration.
    """

    def __init__(self, base_model: nn.Module, config: RefUVConfig) -> None:
        super().__init__()
        if getattr(base_model, "segmentation_head", None) is not None:
            raise ValueError("RefUVLWDETR currently supports detection models only, not segmentation heads.")

        self.num_queries = base_model.num_queries
        self.transformer = base_model.transformer
        self.class_embed = base_model.class_embed
        self.bbox_embed = base_model.bbox_embed
        self.segmentation_head = base_model.segmentation_head
        self.refpoint_embed = base_model.refpoint_embed
        self.query_feat = base_model.query_feat
        self.backbone = base_model.backbone
        self.aux_loss = base_model.aux_loss
        self.group_detr = base_model.group_detr
        self.lite_refpoint_refine = base_model.lite_refpoint_refine
        self.bbox_reparam = base_model.bbox_reparam
        self.two_stage = base_model.two_stage
        self._export = False

        hidden_dim = self.transformer.d_model
        num_levels = self.transformer.num_feature_levels
        self.reference_tokenizer = WhiteReferencedResidualTokenizer(config.prior_channels, hidden_dim, num_levels)
        self.reference_alignment = MisalignmentAwareReferenceAlignment(
            hidden_dim,
            num_levels,
            config.max_alignment_offset_px,
        )
        enable_decoder_reference_fusion(
            self.transformer,
            DecoderReferenceFusionConfig(
                max_ref_beta=config.max_box_beta,
                initial_ref_beta=config.initial_ref_beta,
                gate_bias=config.gate_bias,
                start_layer=config.decoder_fusion_start_layer,
            ),
        )
        self.prior_channels = config.prior_channels
        self.ref_aux: dict[str, Tensor] = {}

    @property
    def beta_cls(self) -> Tensor:
        """Return a compatibility scalar for older training logs."""
        return next(self.parameters()).new_zeros(())

    @property
    def beta_box(self) -> Tensor:
        """Return the mean decoder reference beta for training logs."""
        betas = []
        for layer in self.transformer.decoder.layers:
            if hasattr(layer, "ref_beta"):
                betas.append(layer.ref_beta.reshape(1))
        if not betas:
            return next(self.parameters()).new_zeros(())
        return torch.cat(betas).mean()

    def _build_prior_batch(self, samples: Any, targets: Optional[list[dict[str, Tensor]]]) -> Tensor:
        """Pad per-image prior tensors to the RF-DETR batch shape.

        Args:
            samples: UV image batch.
            targets: Target dictionaries containing ``ref_uv_prior``.

        Returns:
            Batched prior tensor.
        """
        batch_size, _, height, width = samples.tensors.shape
        prior_batch = samples.tensors.new_zeros((batch_size, self.prior_channels, height, width))
        if targets is None:
            return prior_batch
        for index, target in enumerate(targets):
            prior = target.get("ref_uv_prior")
            if prior is None:
                continue
            channels = min(self.prior_channels, prior.shape[0])
            prior_height = min(height, prior.shape[-2])
            prior_width = min(width, prior.shape[-1])
            prior_batch[index, :channels, :prior_height, :prior_width] = prior[
                :channels,
                :prior_height,
                :prior_width,
            ].to(device=samples.tensors.device, dtype=samples.tensors.dtype)
        return prior_batch

    @staticmethod
    def _flatten_reference_features(reference_features: list[Tensor]) -> Tensor:
        """Flatten multi-scale reference feature maps for deformable attention.

        Args:
            reference_features: Reference feature maps in RF-DETR level order.

        Returns:
            Flattened memory with shape ``B x sum(H_l W_l) x C``.
        """
        return torch.cat([feature.flatten(2).transpose(1, 2) for feature in reference_features], dim=1)

    def _compute_reference_context(
        self,
        srcs: list[Tensor],
        prior_batch: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute decoder reference memory and lesion logits.

        Args:
            srcs: UV feature maps.
            prior_batch: Padded RNFR prior tensor.

        Returns:
            Tuple of ``(reference_memory, lesion_logits)``.
        """
        level_shapes = [(src.shape[-2], src.shape[-1]) for src in srcs]
        prior_features, lesion_logits = self.reference_tokenizer(prior_batch, level_shapes)
        aligned_prior = self.reference_alignment(srcs, prior_features)
        return self._flatten_reference_features(aligned_prior), lesion_logits

    def forward(self, samples: Any, targets: Optional[list[dict[str, Tensor]]] = None) -> dict[str, Any]:
        """Run Ref-UV DETR forward pass.

        Args:
            samples: RF-DETR UV image batch.
            targets: Optional target dictionaries carrying ``ref_uv_prior``.

        Returns:
            RF-DETR output dictionary with fused final predictions.
        """
        if isinstance(samples, (list, torch.Tensor)):
            from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for feat in features:
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight = self.query_feat.weight
        else:
            refpoint_embed_weight = self.refpoint_embed.weight[: self.num_queries]
            query_feat_weight = self.query_feat.weight[: self.num_queries]

        prior_batch = self._build_prior_batch(samples, targets)
        reference_memory, lesion_logits = self._compute_reference_context(srcs, prior_batch)
        set_decoder_reference_context(self.transformer, reference_memory, lesion_logits)
        try:
            hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
                srcs,
                masks,
                poss,
                refpoint_embed_weight,
                query_feat_weight,
            )
        finally:
            clear_decoder_reference_context(self.transformer)

        if hs is None:
            raise RuntimeError("RefUVLWDETR requires decoder layers; got hs=None.")

        if self.bbox_reparam:
            outputs_coord_delta = self.bbox_embed(hs)
            outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
            outputs_coord = torch.concat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
        else:
            outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

        outputs_class = self.class_embed(hs)

        self.ref_aux = collect_decoder_reference_aux(self.transformer)
        self.ref_aux["lesion_logits"] = lesion_logits
        self.ref_aux["beta_cls"] = self.beta_cls.detach().reshape(1)
        self.ref_aux["beta_box"] = self.beta_box.detach().reshape(1)

        out: dict[str, Any] = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            group_detr = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc = []
            for group_index in range(group_detr):
                cls_enc.append(self.transformer.enc_out_class_embed[group_index](hs_enc_list[group_index]))
            out["enc_outputs"] = {"pred_logits": torch.cat(cls_enc, dim=1), "pred_boxes": ref_enc}

        return out

    @torch.jit.unused  # type: ignore[untyped-decorator]
    def _set_aux_loss(self, outputs_class: Tensor, outputs_coord: Tensor) -> list[dict[str, Tensor]]:
        """Return auxiliary decoder losses in RF-DETR format.

        Args:
            outputs_class: Per-layer class logits.
            outputs_coord: Per-layer box predictions.

        Returns:
            List of auxiliary output dictionaries.
        """
        return [{"pred_logits": cls, "pred_boxes": box} for cls, box in zip(outputs_class[:-1], outputs_coord[:-1])]

    def reinitialize_detection_head(self, num_classes: int) -> None:
        """Resize the classification head.

        Args:
            num_classes: Target number of output features, matching RF-DETR's
                underlying ``LWDETR.reinitialize_detection_head`` convention.
        """
        from rfdetr.models.lwdetr import _resize_linear

        self.class_embed = _resize_linear(self.class_embed, num_classes)
        if self.two_stage:
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [_resize_linear(module, num_classes) for module in self.transformer.enc_out_class_embed]
            )

    def update_drop_path(self, drop_path_rate: float, vit_encoder_num_layers: int) -> None:
        """Delegate RF-DETR drop-path updates to the wrapped backbone.

        Args:
            drop_path_rate: Maximum drop-path rate.
            vit_encoder_num_layers: Number of ViT layers to update.
        """
        layers = self._get_backbone_encoder_layers()
        if layers is None:
            return
        layer_count = min(vit_encoder_num_layers, len(layers))
        drop_rates = [value.item() for value in torch.linspace(0, drop_path_rate, layer_count)]
        for index in range(layer_count):
            drop_path = getattr(layers[index], "drop_path", None)
            if drop_path is not None and hasattr(drop_path, "drop_prob"):
                drop_path.drop_prob = drop_rates[index]

    def _get_backbone_encoder_layers(self) -> Optional[nn.ModuleList]:
        """Resolve ViT encoder layers from the RF-DETR backbone.

        Returns:
            ModuleList of encoder layers, or ``None`` if the backbone layout is
            unknown.
        """
        encoder = self.backbone[0].encoder
        if hasattr(encoder, "blocks"):
            return encoder.blocks
        if hasattr(encoder, "trunk") and hasattr(encoder.trunk, "blocks"):
            return encoder.trunk.blocks
        if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "encoder"):
            nested_encoder = encoder.encoder.encoder
            if hasattr(nested_encoder, "layer"):
                return nested_encoder.layer
        return None

    def update_dropout(self, drop_rate: float) -> None:
        """Update decoder dropout probability.

        Args:
            drop_rate: New dropout probability.
        """
        for module in self.transformer.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate
