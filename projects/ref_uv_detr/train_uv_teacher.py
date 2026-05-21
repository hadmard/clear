#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Train the pure-UV RF-DETR teacher with the official ``model.train`` API."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from rfdetr.variants import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall


def _timestamped_output_dir(prefix: str) -> str:
    """Return an output directory ending with the current wall-clock timestamp.

    Args:
        prefix: Base experiment directory.

    Returns:
        Timestamped output directory.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(prefix) / stamp)


TRAINING_DEFAULTS: dict[str, object] = {
    # Pure-UV teacher still expects a standard Roboflow COCO UV dataset root.
    # For the paired data loader, use train_ref_uv.py.
    "dataset_dir": "data",
    "output_dir": _timestamped_output_dir("output/uv_teacher_2x4090"),
    "variant": "small",
    "resolution": None,
    # Train long enough for small leaf datasets, but let validation mAP stop the
    # run once the UV-only teacher has clearly plateaued.
    "epochs": 120,
    # Two RTX 4090 cards: this is per-GPU batch size. Effective global batch is
    # batch_size * grad_accum_steps * devices = 8 * 1 * 2 = 16, matching the
    # RF-DETR documentation's recommended effective batch target.
    "batch_size": 8,
    "grad_accum_steps": 1,
    # Stay close to RF-DETR's official fine-tuning defaults. The backbone's
    # actual LR is further reduced by RF-DETR's layer/component decay.
    "lr": 1e-4,
    "lr_encoder": 1.5e-4,
    "warmup_epochs": 1.0,
    "lr_scheduler": "cosine",
    "lr_min_factor": 0.05,
    "accelerator": "gpu",
    "devices": 2,
    # RF-DETR detection variants can leave architecture parameters unused on a
    # given step, so use the DDP mode Lightning recommends for that case.
    "strategy": "ddp_find_unused_parameters_true",
    "num_workers": 8,
    "checkpoint_interval": 5,
    "skip_best_epochs": 3,
    "early_stopping": True,
    "early_stopping_patience": 18,
    "early_stopping_min_delta": 0.001,
    "early_stopping_use_ema": True,
    "seed": 42,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 4,
}

_VARIANTS = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "large": RFDETRLarge,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", default=TRAINING_DEFAULTS["dataset_dir"], help="Roboflow COCO UV dataset root."
    )
    parser.add_argument("--output-dir", default=TRAINING_DEFAULTS["output_dir"], help="Teacher output directory.")
    parser.add_argument(
        "--variant", choices=sorted(_VARIANTS), default=TRAINING_DEFAULTS["variant"], help="RF-DETR variant."
    )
    parser.add_argument(
        "--resolution", type=int, default=TRAINING_DEFAULTS["resolution"], help="Optional model resolution override."
    )
    parser.add_argument("--epochs", type=int, default=TRAINING_DEFAULTS["epochs"], help="Training epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=TRAINING_DEFAULTS["batch_size"], help="Per-device batch size."
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=TRAINING_DEFAULTS["grad_accum_steps"],
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--lr", type=float, default=TRAINING_DEFAULTS["lr"], help="Learning rate.")
    parser.add_argument(
        "--lr-encoder", type=float, default=TRAINING_DEFAULTS["lr_encoder"], help="Backbone learning rate."
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=TRAINING_DEFAULTS["warmup_epochs"],
        help="Linear warmup length in epochs.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("step", "cosine"),
        default=TRAINING_DEFAULTS["lr_scheduler"],
        help="RF-DETR learning-rate scheduler.",
    )
    parser.add_argument(
        "--lr-min-factor",
        type=float,
        default=TRAINING_DEFAULTS["lr_min_factor"],
        help="Minimum LR factor for cosine annealing.",
    )
    parser.add_argument(
        "--accelerator", default=TRAINING_DEFAULTS["accelerator"], help='Lightning accelerator, e.g. "gpu".'
    )
    parser.add_argument("--devices", default=TRAINING_DEFAULTS["devices"], help="Lightning devices, e.g. 2.")
    parser.add_argument("--strategy", default=TRAINING_DEFAULTS["strategy"], help='Lightning strategy, e.g. "ddp".')
    parser.add_argument("--num-workers", type=int, default=TRAINING_DEFAULTS["num_workers"], help="DataLoader workers.")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=TRAINING_DEFAULTS["checkpoint_interval"],
        help="Save archive checkpoints every N epochs.",
    )
    parser.add_argument(
        "--skip-best-epochs",
        type=int,
        default=TRAINING_DEFAULTS["skip_best_epochs"],
        help="Ignore the first N epochs for best-checkpoint and early-stopping decisions.",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable validation-mAP early stopping.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=TRAINING_DEFAULTS["early_stopping_patience"],
        help="Epochs without enough mAP improvement before stopping.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=TRAINING_DEFAULTS["early_stopping_min_delta"],
        help="Minimum mAP improvement counted by early stopping.",
    )
    parser.add_argument(
        "--no-early-stopping-ema",
        action="store_true",
        help="Track raw validation mAP instead of EMA mAP for early stopping.",
    )
    parser.add_argument("--seed", type=int, default=TRAINING_DEFAULTS["seed"], help="Optional training seed.")
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable DataLoader pinned memory.",
    )
    parser.add_argument(
        "--no-persistent-workers",
        action="store_true",
        help="Disable persistent DataLoader workers.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=TRAINING_DEFAULTS["prefetch_factor"],
        help="DataLoader prefetch factor when num_workers > 0.",
    )
    return parser.parse_args()


def main() -> None:
    """Train the UV-only teacher."""
    args = parse_args()
    model_kwargs: dict[str, Any] = {}
    if args.resolution is not None:
        model_kwargs["resolution"] = args.resolution
    model = _VARIANTS[args.variant](**model_kwargs)  # type: ignore[no-untyped-call]
    train_kwargs: dict[str, Any] = {
        "dataset_dir": args.dataset_dir,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "warmup_epochs": args.warmup_epochs,
        "lr_scheduler": args.lr_scheduler,
        "lr_min_factor": args.lr_min_factor,
        "num_workers": args.num_workers,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "strategy": args.strategy,
        "checkpoint_interval": args.checkpoint_interval,
        "skip_best_epochs": args.skip_best_epochs,
        "early_stopping": TRAINING_DEFAULTS["early_stopping"] and not args.no_early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "early_stopping_use_ema": TRAINING_DEFAULTS["early_stopping_use_ema"] and not args.no_early_stopping_ema,
        "seed": args.seed,
        "pin_memory": TRAINING_DEFAULTS["pin_memory"] and not args.no_pin_memory,
        "persistent_workers": (
            TRAINING_DEFAULTS["persistent_workers"] and not args.no_persistent_workers and args.num_workers > 0
        ),
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "notes": {
            "method": "UV-only teacher for Ref-UV DETR",
            "recommended_next_step": "Use checkpoint_best_total.pth as --teacher-checkpoint for train_ref_uv.py.",
        },
    }
    model.train(**train_kwargs)  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    main()
