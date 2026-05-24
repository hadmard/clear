# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Confidence-threshold sweep for precision/recall/F1 computation."""

from typing import Any

import numpy as np


def sweep_confidence_thresholds(
    per_class_data: list[dict[str, Any]],
    conf_thresholds: Any,
    classes_with_gt: list[int],
) -> list[dict[str, Any]]:
    """Sweep confidence thresholds and compute precision/recall/F1 at each.

    The global PR-curve values are computed by micro-aggregating TP/FP/FN over
    all classes at IoU=0.5 matching data prepared by ``build_matching_data()``.
    This makes the selected threshold correspond to the maximum global F1 point
    on the PR curve, while per-class arrays remain available for reporting at
    that same threshold.

    Args:
        per_class_data: Per-class matching data list indexed by class id.
            Each entry is a dict with keys ``"scores"``, ``"matches"``,
            ``"ignore"``, and ``"total_gt"``.
        conf_thresholds: Iterable of float confidence thresholds to evaluate.
        classes_with_gt: List of class indices that have at least one GT
            instance — used for macro-averaging.

    Returns:
        List of result dicts, one per threshold, each containing:
            - ``"confidence_threshold"``: float
            - ``"micro_f1"``: float
            - ``"micro_precision"``: float
            - ``"micro_recall"``: float
            - ``"macro_f1"``: float
            - ``"macro_precision"``: float
            - ``"macro_recall"``: float
            - ``"per_class_prec"``: float ndarray
            - ``"per_class_rec"``: float ndarray
            - ``"per_class_f1"``: float ndarray
    """
    num_classes = len(per_class_data)
    results = []

    for conf_thresh in conf_thresholds:
        per_class_precisions = []
        per_class_recalls = []
        per_class_f1s = []
        total_tp = 0
        total_fp = 0
        total_gt_count = 0

        for k in range(num_classes):
            data = per_class_data[k]
            scores = data["scores"]
            matches = data["matches"]
            ignore = data["ignore"]
            class_total_gt = data["total_gt"]

            above_thresh = scores >= conf_thresh
            valid = above_thresh & ~ignore

            valid_matches = matches[valid]

            tp = np.sum(valid_matches != 0)
            fp = np.sum(valid_matches == 0)
            fn = class_total_gt - tp
            total_tp += tp
            total_fp += fp
            total_gt_count += class_total_gt

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class_precisions.append(precision)
            per_class_recalls.append(recall)
            per_class_f1s.append(f1)

        if len(classes_with_gt) > 0:
            macro_precision = np.mean([per_class_precisions[k] for k in classes_with_gt])
            macro_recall = np.mean([per_class_recalls[k] for k in classes_with_gt])
            macro_f1 = np.mean([per_class_f1s[k] for k in classes_with_gt])
        else:
            macro_precision = 0.0
            macro_recall = 0.0
            macro_f1 = 0.0

        micro_fn = total_gt_count - total_tp
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + micro_fn) if (total_tp + micro_fn) > 0 else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )

        results.append(
            {
                "confidence_threshold": conf_thresh,
                "micro_f1": micro_f1,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "macro_f1": macro_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "per_class_prec": np.array(per_class_precisions),
                "per_class_rec": np.array(per_class_recalls),
                "per_class_f1": np.array(per_class_f1s),
            }
        )

    return results
