# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Shape tests for Ref-UV model components."""

from __future__ import annotations

import torch
from ref_uv_detr.modeling import QueryReferenceSampler, WhiteReferencedResidualTokenizer
from ref_uv_detr.priors import REFERENCE_PRIOR_CHANNELS


def test_reference_tokenizer_and_sampler_shapes_without_alignment() -> None:
    """Reference modules preserve RF-DETR feature/query shapes without alignment."""
    batch_size = 2
    hidden_dim = 32
    prior = torch.randn(batch_size, REFERENCE_PRIOR_CHANNELS, 64, 64)
    boxes = torch.tensor(
        [
            [[0.5, 0.5, 0.2, 0.2], [0.25, 0.75, 0.1, 0.1]],
            [[0.2, 0.3, 0.3, 0.2], [0.8, 0.6, 0.2, 0.2]],
        ],
        dtype=torch.float32,
    )

    tokenizer = WhiteReferencedResidualTokenizer(REFERENCE_PRIOR_CHANNELS, hidden_dim, num_levels=2)
    prior_features, lesion_logits = tokenizer(prior, [(16, 16), (8, 8)])
    sampler = QueryReferenceSampler(hidden_dim, num_levels=2)
    reference = sampler(prior_features, boxes)

    assert lesion_logits.shape == (batch_size, 1, 64, 64)
    assert [feature.shape for feature in prior_features] == [
        (batch_size, hidden_dim, 16, 16),
        (batch_size, hidden_dim, 8, 8),
    ]
    assert reference.shape == (batch_size, 2, hidden_dim)
