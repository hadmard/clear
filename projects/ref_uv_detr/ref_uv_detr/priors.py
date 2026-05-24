# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Reference-normalized UV/white prior construction.

The functions in this file are deliberately model-agnostic.  They transform a
registered UV image and its white-light reference into compact prior maps for
the decoder-level reference branch.  The channel set is intentionally small:
RNFR carries UV-vs-white fluorescence residuals, ExB emphasizes blue response,
white edges provide structure, and raw white RGB gives the reference encoder
the leaf texture needed for deformable query sampling.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from scipy import ndimage

REFERENCE_PRIOR_NAMES: Final[tuple[str, ...]] = (
    "rnfr_r",
    "rnfr_g",
    "rnfr_b",
    "exb_residual",
    "white_edge",
    "white_raw_r",
    "white_raw_g",
    "white_raw_b",
)
REFERENCE_PRIOR_CHANNELS: Final[int] = len(REFERENCE_PRIOR_NAMES)


def _as_float_rgb(image: np.ndarray) -> np.ndarray:
    """Return an ``H x W x 3`` RGB array in ``[0, 1]``.

    Args:
        image: RGB-like array. ``uint8`` arrays are scaled by 255; floating point
            arrays are clipped to ``[0, 1]``.

    Returns:
        Float32 RGB image.

    Raises:
        ValueError: If the input is not an RGB image.
    """
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected an RGB image with shape HxWx3, got {image.shape!r}.")
    rgb = image[..., :3].astype(np.float32, copy=False)
    if image.dtype == np.uint8:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _robust_leaf_mask(white_rgb: np.ndarray) -> np.ndarray:
    """Estimate a leaf support mask from the white-light image.

    The mask is intentionally conservative about background: it starts from
    green/saturated foreground pixels, keeps the dominant connected leaf
    components, and then closes small holes.  If the heuristic becomes too
    sparse, it falls back to a broader non-dark foreground so prior generation
    never fails mid-training.

    Args:
        white_rgb: White-light RGB image in ``[0, 1]``.

    Returns:
        Boolean ``H x W`` mask.
    """
    gray = white_rgb.mean(axis=-1)
    max_c = white_rgb.max(axis=-1)
    min_c = white_rgb.min(axis=-1)
    saturation = max_c - min_c
    red = white_rgb[..., 0]
    green = white_rgb[..., 1]
    blue = white_rgb[..., 2]

    dark_cutoff = max(0.035, float(np.percentile(gray, 5)))
    non_dark = gray > dark_cutoff
    green_dominant = (green > red * 0.82) & (green > blue * 0.82)
    green_excess = (2.0 * green - red - blue) > 0.035
    colorful_leaf = saturation > 0.055
    mask = non_dark & ((green_dominant & colorful_leaf) | green_excess)

    # Keep large connected regions instead of grid lines, labels, and small
    # reflections.  Multiple components are allowed because one frame can
    # contain several separated leaves.
    labeled, component_count = ndimage.label(mask)
    if component_count:
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0
        min_component_area = max(64, int(mask.size * 0.002))
        keep_labels = np.flatnonzero(component_sizes >= min_component_area)
        if keep_labels.size:
            mask = np.isin(labeled, keep_labels)

    # Grow through darker veins and tiny gaps, then smooth boundaries without
    # spilling into the low-saturation grid/background.
    likely_leaf = non_dark & ((saturation > 0.035) | green_dominant)
    mask = ndimage.binary_dilation(mask, structure=np.ones((5, 5), dtype=bool), iterations=1) & likely_leaf
    mask = ndimage.binary_closing(mask, structure=np.ones((7, 7), dtype=bool))
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndimage.binary_fill_holes(mask)

    if float(mask.mean()) < 0.03:
        mask = non_dark & ((saturation > 0.045) | green_dominant)
    if float(mask.mean()) < 0.03:
        mask = np.ones_like(gray, dtype=bool)

    return mask.astype(bool, copy=False)


def _center_over_leaf(values: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    """Subtract the per-channel median over the leaf region.

    Args:
        values: ``H x W x C`` residual image.
        leaf_mask: Boolean leaf support mask.

    Returns:
        Centered residual image with the same shape as ``values``.
    """
    flat = values[leaf_mask]
    if flat.size == 0:
        center = np.zeros((values.shape[-1],), dtype=np.float32)
    else:
        center = np.median(flat, axis=0).astype(np.float32)
    return values - center.reshape((1, 1, -1))


def _leaf_confidence(leaf_mask: np.ndarray) -> np.ndarray:
    """Return a soft confidence map for where white-derived priors are trusted.

    Args:
        leaf_mask: Boolean leaf support mask.

    Returns:
        Float32 ``H x W`` map in ``[0, 1]``.
    """
    confidence = leaf_mask.astype(np.float32, copy=False)
    confidence = ndimage.gaussian_filter(confidence, sigma=2.0)
    max_value = float(confidence.max())
    if max_value <= 1e-6:
        return np.zeros_like(confidence, dtype=np.float32)
    return np.clip(confidence / max_value, 0.0, 1.0).astype(np.float32, copy=False)


def _robust_unit_scale(values: np.ndarray, limit: float = 3.0) -> np.ndarray:
    """Clip residual maps and scale them into roughly ``[-1, 1]``.

    Args:
        values: Residual map.
        limit: Absolute clipping limit before division.

    Returns:
        Float32 residual map.
    """
    return (np.clip(values, -limit, limit) / limit).astype(np.float32, copy=False)


def _white_edge_map(white_rgb: np.ndarray) -> np.ndarray:
    """Compute a normalized Sobel edge map from white-light structure.

    Args:
        white_rgb: White-light RGB image in ``[0, 1]``.

    Returns:
        Float32 ``H x W`` edge strength map in ``[0, 1]``.
    """
    gray = white_rgb.mean(axis=-1)
    grad_x = ndimage.sobel(gray, axis=1, mode="nearest")
    grad_y = ndimage.sobel(gray, axis=0, mode="nearest")
    edge = np.hypot(grad_x, grad_y)
    scale = float(np.percentile(edge, 99))
    if scale <= 1e-6:
        return np.zeros_like(gray, dtype=np.float32)
    return np.clip(edge / scale, 0.0, 1.0).astype(np.float32)


def build_reference_prior(
    uv_image: np.ndarray,
    white_image: np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    """Build compact reference prior channels.

    Channel layout is given by :data:`REFERENCE_PRIOR_NAMES`.  The first three
    channels are RNFR:

    ``log((UV_c + eps) / (White_c + eps)) - median_leaf(...)``.

    The remaining channels are the existing ExB residual, white Sobel edge, and
    raw white RGB.  All white-derived channels are softly limited to the leaf
    support so the decoder reference branch sees structure without learning the
    calibration grid/background as a shortcut.

    Args:
        uv_image: UV RGB image after the same geometric transform as the white
            image.
        white_image: White-light RGB image after the same geometric transform as
            the UV image.
        eps: Numerical stabilizer for log ratios.

    Returns:
        Float32 prior tensor with shape ``C x H x W``.

    Raises:
        ValueError: If the UV and white images have different spatial shapes.
    """
    uv = _as_float_rgb(uv_image)
    white = _as_float_rgb(white_image)
    if uv.shape[:2] != white.shape[:2]:
        raise ValueError(f"UV and white images must have the same H,W, got {uv.shape} and {white.shape}.")

    leaf_mask = _robust_leaf_mask(white)
    leaf_confidence = _leaf_confidence(leaf_mask)

    rnfr = np.log((uv + eps) / (white + eps))
    rnfr = _robust_unit_scale(_center_over_leaf(rnfr, leaf_mask), limit=3.0)
    rnfr = rnfr * leaf_confidence[..., None]

    exb_uv = 2.0 * uv[..., 2] - uv[..., 1] - uv[..., 0]
    exb_white = 2.0 * white[..., 2] - white[..., 1] - white[..., 0]
    exb = _center_over_leaf((exb_uv - exb_white)[..., None], leaf_mask)[..., 0]
    exb = _robust_unit_scale(exb, limit=2.0)
    exb = exb * leaf_confidence

    edge = _white_edge_map(white) * leaf_confidence
    white_raw = white * leaf_confidence[..., None]

    prior_hwc = np.concatenate(
        [
            rnfr,
            exb[..., None],
            edge[..., None],
            white_raw,
        ],
        axis=-1,
    )
    return np.moveaxis(prior_hwc.astype(np.float32, copy=False), -1, 0)
