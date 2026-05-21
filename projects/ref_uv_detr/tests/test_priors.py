# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for Ref-UV prior construction."""

from __future__ import annotations

import numpy as np
from ref_uv_detr.priors import REFERENCE_PRIOR_CHANNELS, build_reference_prior


def test_reference_prior_shape_and_finiteness() -> None:
    """RNFR prior construction returns finite channels with the expected shape."""
    white = np.zeros((32, 40, 3), dtype=np.uint8)
    white[4:28, 5:35] = np.array([70, 130, 55], dtype=np.uint8)
    uv = white.copy()
    uv[12:20, 15:25, 2] = 220

    prior = build_reference_prior(uv, white)

    assert prior.shape == (REFERENCE_PRIOR_CHANNELS, 32, 40)
    assert np.isfinite(prior).all()
    assert prior[8].max() > 0.0
    assert prior[9:].max() == 0.0


def test_reference_prior_is_centered_for_identical_images() -> None:
    """RNFR channels stay near zero when UV and white images match."""
    image = np.full((24, 24, 3), 120, dtype=np.uint8)
    image[..., 1] = 160

    prior = build_reference_prior(image, image)

    rnfr = prior[:3]
    assert float(np.abs(rnfr).max()) < 1e-4


def test_reference_prior_suppresses_background_structure() -> None:
    """White-derived prior channels stay quiet outside the leaf support."""
    white = np.zeros((64, 64, 3), dtype=np.uint8)
    white[8:48, 8:48] = np.array([70, 145, 55], dtype=np.uint8)
    white[52:56, :] = 255
    uv = white.copy()
    uv[20:30, 20:30, 2] = 230
    uv[52:56, :] = 0

    prior = build_reference_prior(uv, white)

    background = prior[:, 58:, 4:60]
    assert float(np.abs(background).max()) < 1e-4
