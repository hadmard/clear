# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# mypy: disable-error-code="misc"

"""Ref-UV DETR model wrapper.

This module keeps RF-DETR's UV backbone and decoder intact, then adds a narrow
reference path that can only affect decoder queries through query-level tokens.
That is the main guardrail against the raw white image polluting UV fluorescence
semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F  # noqa: N812 -- project-conventional alias
from torch import Tensor, nn

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
    """Configuration for the reference-guided fusion path.

    Args:
        prior_channels: Number of RNFR/white-structure prior channels.
        max_cls_beta: Absolute cap for reference influence on classification.
        max_box_beta: Absolute cap for reference influence on localization.
        gate_bias: Initial reliability-gate bias. Negative values make the
            reference path conservative at the beginning of fine-tuning.
    """

    prior_channels: int = REFERENCE_PRIOR_CHANNELS
    max_cls_beta: float = 0.10
    max_box_beta: float = 0.30
    gate_bias: float = -2.0


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


class QueryReferenceSampler(nn.Module):
    """Sample prior features at decoder query locations.

    Args:
        hidden_dim: Detector hidden dimension.
        num_levels: Number of feature levels.
    """

    def __init__(self, hidden_dim: int, num_levels: int) -> None:
        super().__init__()
        self.level_weights = nn.Parameter(torch.zeros(num_levels))
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _query_grid(boxes: Tensor) -> Tensor:
        """Return a grid_sample-compatible grid from normalized boxes.

        Args:
            boxes: Query boxes in normalized ``cx, cy, w, h`` format.

        Returns:
            Grid tensor with shape ``B x Q x 1 x 2``.
        """
        centers = boxes[..., :2].detach().clamp(0.0, 1.0)
        return (centers * 2.0 - 1.0).unsqueeze(2)

    def forward(self, prior_features: list[Tensor], boxes: Tensor) -> Tensor:
        """Sample and aggregate query reference tokens.

        Args:
            prior_features: Reference prior feature maps at detector feature resolutions.
            boxes: Query boxes in normalized ``cx, cy, w, h`` format.

        Returns:
            Query reference tokens with shape ``B x Q x C``.
        """
        grid = self._query_grid(boxes)
        weights = torch.softmax(self.level_weights, dim=0)
        samples = []
        for feature, weight in zip(prior_features, weights):
            sampled = F.grid_sample(feature, grid, mode="bilinear", padding_mode="border", align_corners=True)
            sampled = sampled.squeeze(-1).transpose(1, 2)
            samples.append(sampled * weight)
        return self.proj(torch.stack(samples, dim=0).sum(dim=0))


class LesionAwareReferenceGate(nn.Module):
    """Predict a query-level reference reliability score.

    Args:
        hidden_dim: Detector hidden dimension.
        gate_bias: Initial bias for the final gate layer.
    """

    def __init__(self, hidden_dim: int, gate_bias: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.constant_(self.net[-1].bias, gate_bias)

    def forward(
        self,
        query: Tensor,
        reference: Tensor,
        class_confidence: Tensor,
        boxes: Tensor,
        lesionness: Tensor,
    ) -> Tensor:
        """Compute query-level reference reliability.

        Args:
            query: UV decoder query tokens.
            reference: Sampled reference tokens.
            class_confidence: UV-only class confidence per query.
            boxes: Query boxes in normalized ``cx, cy, w, h`` format.
            lesionness: RNFR lesionness sampled at query centers.

        Returns:
            Gate values with shape ``B x Q x 1``.
        """
        features = torch.cat([query, reference, class_confidence.detach(), boxes.detach(), lesionness], dim=-1)
        return torch.sigmoid(self.net(features))


class ReferenceQueryUpdate(nn.Module):
    """Produce a reference-conditioned query update.

    Args:
        hidden_dim: Detector hidden dimension.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, query: Tensor, reference: Tensor) -> Tensor:
        """Return an additive query update.

        Args:
            query: UV decoder query tokens.
            reference: Query reference tokens.

        Returns:
            Update tensor with shape ``B x Q x C``.
        """
        return self.net(torch.cat([query, reference], dim=-1))


class RefUVLWDETR(nn.Module):
    """RF-DETR detector with reference-guided UV query fusion.

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
        self.reference_sampler = QueryReferenceSampler(hidden_dim, num_levels)
        self.reference_gate = LesionAwareReferenceGate(hidden_dim, config.gate_bias)
        self.reference_update = ReferenceQueryUpdate(hidden_dim)
        self.beta_cls_raw = nn.Parameter(torch.zeros(()))
        self.beta_box_raw = nn.Parameter(torch.zeros(()))
        self.max_cls_beta = config.max_cls_beta
        self.max_box_beta = config.max_box_beta
        self.prior_channels = config.prior_channels
        self.ref_aux: dict[str, Tensor] = {}

    @property
    def beta_cls(self) -> Tensor:
        """Return capped classification fusion strength."""
        return self.max_cls_beta * torch.tanh(self.beta_cls_raw)

    @property
    def beta_box(self) -> Tensor:
        """Return capped localization fusion strength."""
        return self.max_box_beta * torch.tanh(self.beta_box_raw)

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
    def _sample_scalar_map(map_logits: Tensor, boxes: Tensor) -> Tensor:
        """Sample a scalar map at query centers.

        Args:
            map_logits: Dense scalar logits with shape ``B x 1 x H x W``.
            boxes: Query boxes in normalized ``cx, cy, w, h`` format.

        Returns:
            Sampled scalar values with shape ``B x Q x 1``.
        """
        grid = QueryReferenceSampler._query_grid(boxes)
        sampled = F.grid_sample(
            torch.sigmoid(map_logits),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).transpose(1, 2)

    def _compute_reference_update(
        self,
        srcs: list[Tensor],
        prior_batch: Tensor,
        query: Tensor,
        boxes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute reference update, gate, and lesion logits for final queries.

        The white-reference prior is sampled directly at the UV query centers.
        There is intentionally no learned spatial alignment block in this
        ablation, so the reference branch is simpler and easier to compare
        against the earlier alignment-enabled run.

        Args:
            srcs: UV feature maps.
            prior_batch: Padded RNFR prior tensor.
            query: Final UV decoder query tokens.
            boxes: UV-only query boxes.

        Returns:
            Tuple of ``(update, gate, lesion_logits)``.
        """
        level_shapes = [(src.shape[-2], src.shape[-1]) for src in srcs]
        prior_features, lesion_logits = self.reference_tokenizer(prior_batch, level_shapes)
        reference = self.reference_sampler(prior_features, boxes)
        class_confidence = torch.sigmoid(self.class_embed(query)).amax(dim=-1, keepdim=True)
        lesionness = self._sample_scalar_map(lesion_logits, boxes)
        gate = self.reference_gate(query, reference, class_confidence, boxes, lesionness)
        update = self.reference_update(query, reference) * gate
        return update, gate, lesion_logits

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

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs,
            masks,
            poss,
            refpoint_embed_weight,
            query_feat_weight,
        )

        if hs is None:
            raise RuntimeError("RefUVLWDETR requires decoder layers; got hs=None.")

        if self.bbox_reparam:
            outputs_coord_delta = self.bbox_embed(hs)
            outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
            outputs_coord = torch.concat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
        else:
            outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

        prior_batch = self._build_prior_batch(samples, targets)
        ref_update, gate, lesion_logits = self._compute_reference_update(srcs, prior_batch, hs[-1], outputs_coord[-1])
        q_cls = hs[-1] + self.beta_cls * ref_update
        q_box = hs[-1] + self.beta_box * ref_update

        outputs_class_base = self.class_embed(hs)
        outputs_class_final = self.class_embed(q_cls)
        outputs_class = torch.cat([outputs_class_base[:-1], outputs_class_final.unsqueeze(0)], dim=0)

        if self.bbox_reparam:
            final_delta = self.bbox_embed(q_box)
            final_cxcy = final_delta[..., :2] * ref_unsigmoid[-1, ..., 2:] + ref_unsigmoid[-1, ..., :2]
            final_wh = final_delta[..., 2:].exp() * ref_unsigmoid[-1, ..., 2:]
            final_boxes = torch.concat([final_cxcy, final_wh], dim=-1)
        else:
            final_boxes = (self.bbox_embed(q_box) + ref_unsigmoid[-1]).sigmoid()
        outputs_coord = torch.cat([outputs_coord[:-1], final_boxes.unsqueeze(0)], dim=0)

        self.ref_aux = {
            "gate": gate,
            "lesion_logits": lesion_logits,
            "beta_cls": self.beta_cls.detach().reshape(1),
            "beta_box": self.beta_box.detach().reshape(1),
        }

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
