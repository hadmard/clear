# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Reference-normalized UV/white prior construction.

The functions in this file are deliberately model-agnostic.  They transform a
registered UV image and its white-light reference into compact prior maps that
the detector can use as structural evidence without giving raw white RGB a free
path into the classifier.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from scipy import ndimage

REFERENCE_PRIOR_NAMES: Final[tuple[str, ...]] = (
    "rnfr_r",
    "rnfr_g",
    "rnfr_b",
    "uv_minus_white_r",
    "uv_minus_white_g",
    "uv_minus_white_b",
    "log_bg_residual",
    "exb_residual",
    "white_edge",
    "white_leaf_mask",
    "white_distance_to_boundary",
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

    The mask is intentionally conservative: it removes black borders and very
    flat background while keeping low-saturation leaf pixels.  If the heuristic
    becomes too sparse, it falls back to all non-dark pixels so prior generation
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
    green_support = white_rgb[..., 1] >= (0.75 * white_rgb[..., 0])

    dark_cutoff = max(0.03, float(np.percentile(gray, 3)))
    mask = (gray > dark_cutoff) & ((saturation > 0.035) | green_support)
    if float(mask.mean()) < 0.05:
        mask = gray > dark_cutoff
    if float(mask.mean()) < 0.05:
        mask = np.ones_like(gray, dtype=bool)

    # Small holes make the leaf-level median noisy near veins and specular spots.
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5), dtype=bool))
    mask = ndimage.binary_fill_holes(mask)
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


def _distance_to_leaf_boundary(leaf_mask: np.ndarray) -> np.ndarray:
    """Compute a normalized inside-leaf distance-to-boundary map.

    Args:
        leaf_mask: Boolean leaf support mask.

    Returns:
        Float32 ``H x W`` distance map in ``[0, 1]``.
    """
    distance = ndimage.distance_transform_edt(leaf_mask).astype(np.float32)
    max_distance = float(distance.max())
    if max_distance <= 1e-6:
        return np.zeros_like(distance, dtype=np.float32)
    return np.clip(distance / max_distance, 0.0, 1.0).astype(np.float32)


def build_reference_prior(
    uv_image: np.ndarray,
    white_image: np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    """Build Reference-Normalized Fluorescence Residual prior channels.

    Channel layout is given by :data:`REFERENCE_PRIOR_NAMES`.  The first three
    channels are RNFR:

    ``log((UV_c + eps) / (White_c + eps)) - median_leaf(...)``.

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

    rnfr = np.log((uv + eps) / (white + eps))
    rnfr = _robust_unit_scale(_center_over_leaf(rnfr, leaf_mask), limit=3.0)

    uv_minus_white = (uv - white).astype(np.float32, copy=False)

    uv_log_bg = np.log((uv[..., 2] + eps) / (uv[..., 1] + eps))
    white_log_bg = np.log((white[..., 2] + eps) / (white[..., 1] + eps))
    log_bg = _center_over_leaf((uv_log_bg - white_log_bg)[..., None], leaf_mask)[..., 0]
    log_bg = _robust_unit_scale(log_bg, limit=3.0)

    exb_uv = 2.0 * uv[..., 2] - uv[..., 1] - uv[..., 0]
    exb_white = 2.0 * white[..., 2] - white[..., 1] - white[..., 0]
    exb = _center_over_leaf((exb_uv - exb_white)[..., None], leaf_mask)[..., 0]
    exb = _robust_unit_scale(exb, limit=2.0)

    edge = _white_edge_map(white)
    distance = _distance_to_leaf_boundary(leaf_mask)

    prior_hwc = np.concatenate(
        [
            rnfr,
            uv_minus_white,
            log_bg[..., None],
            exb[..., None],
            edge[..., None],
            leaf_mask.astype(np.float32)[..., None],
            distance[..., None],
        ],
        axis=-1,
    )
    return np.moveaxis(prior_hwc.astype(np.float32, copy=False), -1, 0)
