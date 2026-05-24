# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Shape tests for Ref-UV model components."""

from __future__ import annotations

import torch
from ref_uv_detr.decoder_fusion import DecoderReferenceFusionConfig, ReferenceFusionDecoderLayer
from ref_uv_detr.modeling import (
    MisalignmentAwareReferenceAlignment,
    WhiteReferencedResidualTokenizer,
)
from ref_uv_detr.priors import REFERENCE_PRIOR_CHANNELS

from rfdetr.models.transformer import TransformerDecoderLayer


def test_reference_tokenizer_alignment_and_sampler_shapes() -> None:
    """Reference modules preserve RF-DETR feature/query shapes with alignment."""
    batch_size = 2
    hidden_dim = 32
    prior = torch.randn(batch_size, REFERENCE_PRIOR_CHANNELS, 64, 64)
    uv_features = [
        torch.randn(batch_size, hidden_dim, 16, 16),
        torch.randn(batch_size, hidden_dim, 8, 8),
    ]
    tokenizer = WhiteReferencedResidualTokenizer(REFERENCE_PRIOR_CHANNELS, hidden_dim, num_levels=2)
    prior_features, lesion_logits = tokenizer(prior, [(16, 16), (8, 8)])
    alignment = MisalignmentAwareReferenceAlignment(hidden_dim, num_levels=2, max_offset_px=2.0)
    aligned_features = alignment(uv_features, prior_features)

    assert lesion_logits.shape == (batch_size, 1, 64, 64)
    assert [feature.shape for feature in prior_features] == [
        (batch_size, hidden_dim, 16, 16),
        (batch_size, hidden_dim, 8, 8),
    ]
    assert [feature.shape for feature in aligned_features] == [
        (batch_size, hidden_dim, 16, 16),
        (batch_size, hidden_dim, 8, 8),
    ]


def test_decoder_reference_fusion_zero_beta_matches_uv_layer() -> None:
    """Decoder fusion starts as the original UV-only layer when beta is zero."""
    torch.manual_seed(7)
    batch_size = 2
    num_queries = 5
    hidden_dim = 32
    base_layer = TransformerDecoderLayer(
        hidden_dim,
        sa_nhead=4,
        ca_nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        group_detr=1,
        num_feature_levels=2,
        dec_n_points=2,
    )
    fused_layer = ReferenceFusionDecoderLayer(
        base_layer,
        layer_index=1,
        num_layers=3,
        config=DecoderReferenceFusionConfig(max_ref_beta=0.30, start_layer=1),
    )

    tgt = torch.randn(batch_size, num_queries, hidden_dim)
    query_pos = torch.randn(batch_size, num_queries, hidden_dim)
    memory = torch.randn(batch_size, 20, hidden_dim)
    reference_memory = torch.randn(batch_size, 20, hidden_dim)
    spatial_shapes = torch.tensor([[4, 4], [2, 2]], dtype=torch.long)
    level_start_index = torch.tensor([0, 16], dtype=torch.long)
    reference_points = torch.rand(batch_size, num_queries, 2, 4)

    fused_layer.set_reference_context(reference_memory, lesion_logits=None)
    out_fused = fused_layer(
        tgt,
        memory,
        query_pos=query_pos,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        spatial_shapes_hw=[(4, 4), (2, 2)],
    )
    fused_layer.clear_reference_context()

    out_uv = base_layer(
        tgt,
        memory,
        query_pos=query_pos,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        spatial_shapes_hw=[(4, 4), (2, 2)],
    )

    assert torch.allclose(out_fused, out_uv, atol=1e-6)


def test_decoder_reference_fusion_initial_beta_opens_active_layers() -> None:
    """Initial beta can give enabled reference layers a small non-zero signal."""
    hidden_dim = 32
    config = DecoderReferenceFusionConfig(max_ref_beta=0.30, initial_ref_beta=0.03, start_layer=1)

    disabled_base = TransformerDecoderLayer(
        hidden_dim,
        sa_nhead=4,
        ca_nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        group_detr=1,
        num_feature_levels=2,
        dec_n_points=2,
    )
    active_base = TransformerDecoderLayer(
        hidden_dim,
        sa_nhead=4,
        ca_nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        group_detr=1,
        num_feature_levels=2,
        dec_n_points=2,
    )

    disabled_layer = ReferenceFusionDecoderLayer(disabled_base, layer_index=0, num_layers=3, config=config)
    active_layer = ReferenceFusionDecoderLayer(active_base, layer_index=2, num_layers=3, config=config)

    assert disabled_layer.ref_beta.item() == 0.0
    assert torch.isclose(active_layer.ref_beta, torch.tensor(0.03), atol=1e-6)
