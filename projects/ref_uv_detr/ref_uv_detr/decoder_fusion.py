# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Decoder-level reference fusion for paired UV/white RF-DETR.

This module keeps RF-DETR's original UV decoder path intact and adds a narrow
reference cross-attention branch inside each decoder layer.  The reference branch
receives compact RNFR/white features and uses its own deformable offsets, so mild
UV/white misalignment can be handled at object-query level instead of by head
fusion after decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812 -- project-conventional alias
from torch import Tensor, nn

from rfdetr.models.ops.modules import MSDeformAttn


@dataclass
class DecoderReferenceFusionConfig:
    """Configuration for decoder-layer reference fusion.

    Args:
        max_ref_beta: Maximum absolute residual strength for the final decoder
            layer. Earlier layers receive a smaller scheduled cap.
        initial_ref_beta: Initial absolute residual strength for enabled decoder
            layers. ``0`` keeps the historical UV-equivalent start.
        gate_bias: Initial bias for the query-level reference gate.
        start_layer: First decoder layer allowed to use reference features. A
            value of ``1`` keeps the first layer UV-only and lets later layers
            refine object queries with white/RNFR evidence.
    """

    max_ref_beta: float = 0.30
    initial_ref_beta: float = 0.0
    gate_bias: float = -2.0
    start_layer: int = 1


class ReferenceFusionDecoderLayer(nn.Module):
    """RF-DETR decoder layer with an additional reference cross-attention path.

    Args:
        base_layer: Existing RF-DETR decoder layer whose UV attention, self
            attention, FFN, norms, and dropout modules are reused.
        layer_index: Index of this layer in the decoder stack.
        num_layers: Total number of decoder layers.
        config: Fusion configuration.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        layer_index: int,
        num_layers: int,
        config: DecoderReferenceFusionConfig,
    ) -> None:
        super().__init__()
        self.self_attn = base_layer.self_attn
        self.dropout1 = base_layer.dropout1
        self.norm1 = base_layer.norm1

        self.cross_attn = base_layer.cross_attn
        self.ref_cross_attn = MSDeformAttn(
            d_model=self.cross_attn.d_model,
            n_levels=self.cross_attn.n_levels,
            n_heads=self.cross_attn.n_heads,
            n_points=self.cross_attn.n_points,
        )
        self.ref_proj = nn.Linear(self.cross_attn.d_model, self.cross_attn.d_model)

        self.linear1 = base_layer.linear1
        self.dropout = base_layer.dropout
        self.linear2 = base_layer.linear2
        self.norm2 = base_layer.norm2
        self.norm3 = base_layer.norm3
        self.dropout2 = base_layer.dropout2
        self.dropout3 = base_layer.dropout3
        self.activation = base_layer.activation
        self.normalize_before = base_layer.normalize_before
        self.group_detr = base_layer.group_detr
        self.nhead = base_layer.nhead

        self.layer_index = layer_index
        self.reference_enabled = layer_index >= config.start_layer
        self.max_ref_beta = self._scheduled_beta(config.max_ref_beta, layer_index, num_layers, config.start_layer)
        self.ref_beta_raw = nn.Parameter(
            torch.tensor(self._initial_beta_raw(config.initial_ref_beta), dtype=torch.float32)
        )
        hidden_dim = self.cross_attn.d_model
        self.ref_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.constant_(self.ref_gate[-1].bias, config.gate_bias)

        self._reference_memory: Optional[Tensor] = None
        self._lesion_logits: Optional[Tensor] = None
        self.last_gate: Optional[Tensor] = None

    @staticmethod
    def _scheduled_beta(max_ref_beta: float, layer_index: int, num_layers: int, start_layer: int) -> float:
        """Return a conservative per-layer beta cap.

        Args:
            max_ref_beta: Final-layer cap.
            layer_index: Decoder layer index.
            num_layers: Number of decoder layers.
            start_layer: First reference-enabled layer.

        Returns:
            Scheduled beta cap for this layer.
        """
        if layer_index < start_layer:
            return 0.0
        active_layers = max(num_layers - start_layer, 1)
        active_index = layer_index - start_layer + 1
        return float(max_ref_beta) * float(active_index) / float(active_layers)

    def _initial_beta_raw(self, initial_ref_beta: float) -> float:
        """Return the raw parameter value needed for the requested beta start.

        Args:
            initial_ref_beta: Requested absolute beta value for this active
                decoder layer.

        Returns:
            Raw scalar before ``tanh``.
        """
        if not self.reference_enabled or self.max_ref_beta <= 0.0 or initial_ref_beta <= 0.0:
            return 0.0
        ratio = min(float(initial_ref_beta) / float(self.max_ref_beta), 0.99)
        return math.atanh(ratio)

    @property
    def ref_beta(self) -> Tensor:
        """Return the current capped reference residual strength."""
        return self.ref_beta_raw.new_tensor(self.max_ref_beta) * torch.tanh(self.ref_beta_raw)

    def set_reference_context(self, reference_memory: Optional[Tensor], lesion_logits: Optional[Tensor]) -> None:
        """Set flattened reference memory for the next decoder forward pass.

        Args:
            reference_memory: Flattened reference features with shape
                ``B x sum(H_l W_l) x C``.
            lesion_logits: Optional dense lesionness logits from the reference
                encoder with shape ``B x 1 x H x W``.
        """
        self._reference_memory = reference_memory
        self._lesion_logits = lesion_logits

    def clear_reference_context(self) -> None:
        """Drop cached reference tensors after a decoder forward pass."""
        self._reference_memory = None
        self._lesion_logits = None

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        """Add positional embedding when provided."""
        return tensor if pos is None else tensor + pos

    @staticmethod
    def _sample_scalar_map(map_logits: Tensor, reference_points: Tensor) -> Tensor:
        """Sample scalar logits at query reference centers.

        Args:
            map_logits: Dense scalar map with shape ``B x 1 x H x W``.
            reference_points: Decoder reference points with shape
                ``B x Q x L x 4`` or ``B x Q x L x 2``.

        Returns:
            Sampled values with shape ``B x Q x 1``.
        """
        centers = reference_points[:, :, 0, :2].detach().clamp(0.0, 1.0)
        grid = (centers * 2.0 - 1.0).unsqueeze(2)
        sampled = F.grid_sample(
            torch.sigmoid(map_logits),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).transpose(1, 2)

    def _reference_update(
        self,
        tgt: Tensor,
        uv_update: Tensor,
        reference_points: Tensor,
        memory_key_padding_mask: Optional[Tensor],
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        spatial_shapes_hw: list[tuple[int, int]] | None,
    ) -> Tensor:
        """Compute gated reference residual for one decoder layer.

        Args:
            tgt: Query tensor after self-attention.
            uv_update: Original RF-DETR UV cross-attention update.
            reference_points: Decoder reference points.
            memory_key_padding_mask: Padding mask shared with UV memory.
            spatial_shapes: Feature spatial shapes.
            level_start_index: Flattened level start offsets.
            spatial_shapes_hw: Python ``(H, W)`` pairs for export-safe views.

        Returns:
            Reference residual with shape ``B x Q x C``.
        """
        if not self.reference_enabled or self._reference_memory is None:
            self.last_gate = None
            return torch.zeros_like(uv_update)

        ref_update = self.ref_cross_attn(
            tgt,
            reference_points,
            self._reference_memory,
            spatial_shapes,
            level_start_index,
            memory_key_padding_mask,
            input_spatial_shapes_hw=spatial_shapes_hw,
        )
        if self._lesion_logits is None:
            lesionness = ref_update.new_zeros((*ref_update.shape[:2], 1))
        else:
            lesionness = self._sample_scalar_map(self._lesion_logits, reference_points).to(dtype=ref_update.dtype)

        gate_input = torch.cat([tgt, uv_update, ref_update, lesionness], dim=-1)
        gate = torch.sigmoid(self.ref_gate(gate_input))
        self.last_gate = gate
        return self.ref_beta * gate * self.ref_proj(ref_update)

    def forward_post(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        query_sine_embed: Optional[Tensor] = None,
        is_first: bool = False,
        reference_points: Optional[Tensor] = None,
        spatial_shapes: Optional[Tensor] = None,
        level_start_index: Optional[Tensor] = None,
        spatial_shapes_hw: list[tuple[int, int]] | None = None,
    ) -> Tensor:
        """Run decoder layer with UV and reference deformable cross-attention."""
        del memory_mask, pos, query_sine_embed, is_first
        if reference_points is None or spatial_shapes is None or level_start_index is None:
            raise ValueError("reference_points, spatial_shapes, and level_start_index are required.")

        batch_size, num_queries, _ = tgt.shape

        q = k = tgt + query_pos
        v = tgt
        if self.training:
            q = torch.cat(q.split(num_queries // self.group_detr, dim=1), dim=0)
            k = torch.cat(k.split(num_queries // self.group_detr, dim=1), dim=0)
            v = torch.cat(v.split(num_queries // self.group_detr, dim=1), dim=0)

        tgt2 = self.self_attn(q, k, v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False)[0]

        if self.training:
            tgt2 = torch.cat(tgt2.split(batch_size, dim=0), dim=1)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        cross_query = self.with_pos_embed(tgt, query_pos)
        uv_update = self.cross_attn(
            cross_query,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            memory_key_padding_mask,
            input_spatial_shapes_hw=spatial_shapes_hw,
        )
        ref_residual = self._reference_update(
            cross_query,
            uv_update,
            reference_points,
            memory_key_padding_mask,
            spatial_shapes,
            level_start_index,
            spatial_shapes_hw,
        )

        tgt = tgt + self.dropout2(uv_update + ref_residual)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        query_sine_embed: Optional[Tensor] = None,
        is_first: bool = False,
        reference_points: Optional[Tensor] = None,
        spatial_shapes: Optional[Tensor] = None,
        level_start_index: Optional[Tensor] = None,
        spatial_shapes_hw: list[tuple[int, int]] | None = None,
    ) -> Tensor:
        """Forward pass matching RF-DETR's decoder-layer signature."""
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
            query_sine_embed,
            is_first,
            reference_points,
            spatial_shapes,
            level_start_index,
            spatial_shapes_hw=spatial_shapes_hw,
        )


def enable_decoder_reference_fusion(transformer: nn.Module, config: DecoderReferenceFusionConfig) -> None:
    """Replace RF-DETR decoder layers with reference-aware wrappers in place.

    Args:
        transformer: RF-DETR transformer module.
        config: Fusion configuration.
    """
    layers = transformer.decoder.layers
    num_layers = len(layers)
    transformer.decoder.layers = nn.ModuleList(
        [
            ReferenceFusionDecoderLayer(layer, layer_index, num_layers, config)
            for layer_index, layer in enumerate(layers)
        ]
    )


def set_decoder_reference_context(
    transformer: nn.Module,
    reference_memory: Optional[Tensor],
    lesion_logits: Optional[Tensor],
) -> None:
    """Attach reference context to all reference-aware decoder layers."""
    for layer in transformer.decoder.layers:
        if hasattr(layer, "set_reference_context"):
            layer.set_reference_context(reference_memory, lesion_logits)


def clear_decoder_reference_context(transformer: nn.Module) -> None:
    """Clear reference context from all reference-aware decoder layers."""
    for layer in transformer.decoder.layers:
        if hasattr(layer, "clear_reference_context"):
            layer.clear_reference_context()


def collect_decoder_reference_aux(transformer: nn.Module) -> dict[str, Tensor]:
    """Collect gate and beta tensors from decoder fusion layers.

    Args:
        transformer: Reference-aware RF-DETR transformer.

    Returns:
        Auxiliary tensors useful for logging and regularization.
    """
    gates = [layer.last_gate for layer in transformer.decoder.layers if getattr(layer, "last_gate", None) is not None]
    betas = [layer.ref_beta.reshape(1) for layer in transformer.decoder.layers if hasattr(layer, "ref_beta")]
    aux: dict[str, Tensor] = {}
    if gates:
        aux["gate"] = torch.stack([gate.mean() for gate in gates]).mean().reshape(1)
    if betas:
        aux["decoder_betas"] = torch.cat(betas)
    return aux
