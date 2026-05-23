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

from ref_uv_detr.initialization import (
    apply_dinov2_backbone_only_init,
    apply_rfdetr_backbone_only_init,
    dinov2_backbone_only_notes,
    load_rfdetr_backbone_weights,
    rfdetr_backbone_only_notes,
)

from rfdetr.config import RFDETRLargeConfig, RFDETRMediumConfig, RFDETRNanoConfig, RFDETRSmallConfig, TrainConfig
from rfdetr.detr import RFDETR
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
    "output_dir": _timestamped_output_dir("output/uv_rfdetr_backbone_b40_perf"),
    "variant": "small",
    # Keep this fixed for apples-to-apples comparison with earlier UV-only runs.
    # DINOv2's own positional table stays at 37 patches; RF-DETR interpolates it
    # at runtime for this 560px training/evaluation resolution.
    "resolution": 560,
    # Use the official RF-DETR checkpoint as a stronger feature initializer, but
    # keep projector/detector modules random by loading only encoder weights in
    # the custom Lightning path below. Disable this for DINOv2-only ablations.
    "backbone_only_rfdetr": True,
    "backbone_pretrain_weights": None,
    # Keep the earlier strict experiment available: original DINOv2 visual
    # pretraining only, with RF-DETR's projector/decoder/query/head random.
    "backbone_only_dinov2": True,
    # Train long enough for small leaf datasets, but let validation mAP stop the
    # run once the UV-only teacher has clearly plateaued.
    "epochs": 180,
    # Two RTX 4090 cards: effective global batch is 10 * 2 * 2 = 40.
    # Keep real per-device batch high for throughput, but avoid an overly large
    # effective batch on the 791-image UV training split.
    "batch_size": 10,
    "grad_accum_steps": 2,
    # Because only DINOv2 is pretrained, the DETR decoder/query/head stack is
    # effectively training from scratch. The first run plateaued after the cosine
    # LR decayed too low, so keep the random detector path more energetic while
    # still using a gentler encoder LR.
    # More aggressive than the batch-16/32 runs, but not linearly scaled to 80.
    "lr": 3.5e-4,
    "lr_encoder": 1.75e-4,
    # RF-DETR defaults use 0.8 layer decay, which made the earliest DINOv2
    # layers train at about 1e-6 before scheduling. Since this experiment relies
    # on the backbone adapting to UV imagery, use a gentler decay while keeping
    # all RF-DETR detector modules randomly initialized.
    "lr_vit_layer_decay": 0.9,
    "lr_component_decay": 0.7,
    "warmup_epochs": 10.0,
    "lr_scheduler": "cosine",
    # Keep more LR late in the run; the previous best still appeared while the
    # cosine schedule had substantial LR left, and PM kept improving late.
    "lr_min_factor": 0.35,
    "accelerator": "gpu",
    "devices": 2,
    # RF-DETR detection variants can leave architecture parameters unused on a
    # given step, so use the DDP mode Lightning recommends for that case.
    "strategy": "ddp_find_unused_parameters_true",
    # With two DDP ranks this creates 32 workers total on the 64-thread EPYC host.
    "num_workers": 16,
    # Best checkpoints are still saved immediately; archive checkpoints can be
    # less frequent because each RF-DETR small Lightning checkpoint is large.
    "checkpoint_interval": 10,
    "skip_best_epochs": 10,
    "early_stopping": True,
    "early_stopping_patience": 90,
    "early_stopping_min_delta": 0.001,
    "early_stopping_use_ema": True,
    # Keep the augmentation/resize path comparable to the previous UV-only run.
    "multi_scale": True,
    "resume": None,
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

_MODEL_CONFIGS = {
    "nano": RFDETRNanoConfig,
    "small": RFDETRSmallConfig,
    "medium": RFDETRMediumConfig,
    "large": RFDETRLargeConfig,
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
    parser.add_argument(
        "--backbone-only-rfdetr",
        action=argparse.BooleanOptionalAction,
        default=TRAINING_DEFAULTS["backbone_only_rfdetr"],
        help=(
            "Initialize only backbone encoder weights from an official RF-DETR checkpoint. "
            "This uses the official Lightning stack instead of model.train() so the checkpoint "
            "can be filtered before training."
        ),
    )
    parser.add_argument(
        "--backbone-pretrain-weights",
        default=TRAINING_DEFAULTS["backbone_pretrain_weights"],
        help="Optional official RF-DETR checkpoint used only as a backbone source.",
    )
    parser.add_argument(
        "--backbone-only-dinov2",
        action=argparse.BooleanOptionalAction,
        default=TRAINING_DEFAULTS["backbone_only_dinov2"],
        help=(
            "Initialize only the DINOv2 backbone from its original pretrained weights. "
            "Ignored when --backbone-only-rfdetr is enabled. Use --no-backbone-only-rfdetr "
            "--no-backbone-only-dinov2 to restore the RF-DETR full-checkpoint fine-tuning path."
        ),
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
        "--lr-vit-layer-decay",
        type=float,
        default=TRAINING_DEFAULTS["lr_vit_layer_decay"],
        help="Layer-wise LR decay for the DINOv2 ViT encoder.",
    )
    parser.add_argument(
        "--lr-component-decay",
        type=float,
        default=TRAINING_DEFAULTS["lr_component_decay"],
        help="RF-DETR component LR decay applied to encoder/decoder groups.",
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
        "--multi-scale",
        action=argparse.BooleanOptionalAction,
        default=TRAINING_DEFAULTS["multi_scale"],
        help="Enable or disable RF-DETR multi-scale resize.",
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
        "--resume",
        default=TRAINING_DEFAULTS["resume"],
        help="Optional Lightning checkpoint for continuing an interrupted UV-only run.",
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
    rfdetr_backbone_source: str | None = None
    if args.backbone_only_rfdetr:
        rfdetr_backbone_source = apply_rfdetr_backbone_only_init(
            model_kwargs,
            config_cls=_MODEL_CONFIGS[args.variant],
            source_weights=args.backbone_pretrain_weights,
            resolution=args.resolution,
        )
    elif args.backbone_only_dinov2:
        apply_dinov2_backbone_only_init(model_kwargs, resolution=args.resolution)
    elif args.resolution is not None:
        model_kwargs["resolution"] = args.resolution

    notes: dict[str, Any] = {
        "method": "UV-only teacher for Ref-UV DETR",
        "recommended_next_step": "Use checkpoint_best_total.pth as --teacher-checkpoint for train_ref_uv.py.",
    }
    if args.backbone_only_rfdetr and rfdetr_backbone_source is not None:
        notes.update(rfdetr_backbone_only_notes(rfdetr_backbone_source))
    elif args.backbone_only_dinov2:
        notes.update(dinov2_backbone_only_notes())
    else:
        notes["initialization"] = "rfdetr_full_detector_checkpoint"

    train_kwargs: dict[str, Any] = {
        "dataset_dir": args.dataset_dir,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "lr_vit_layer_decay": args.lr_vit_layer_decay,
        "lr_component_decay": args.lr_component_decay,
        "warmup_epochs": args.warmup_epochs,
        "lr_scheduler": args.lr_scheduler,
        "lr_min_factor": args.lr_min_factor,
        "multi_scale": args.multi_scale,
        "num_workers": args.num_workers,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "strategy": args.strategy,
        "resume": args.resume,
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
        "notes": notes,
    }

    if args.backbone_only_rfdetr and rfdetr_backbone_source is not None:
        from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

        model_config = _MODEL_CONFIGS[args.variant](**model_kwargs)
        model_config.num_classes = RFDETR._detect_num_classes_for_training(args.dataset_dir)
        model_config.model_name = f"UVTeacher-{args.variant}"
        train_config = TrainConfig(**train_kwargs)

        module = RFDETRModelModule(model_config, train_config)
        load_rfdetr_backbone_weights(module.model, model_config, rfdetr_backbone_source)
        datamodule = RFDETRDataModule(model_config, train_config)
        trainer = build_trainer(train_config, model_config)
        trainer.fit(module, datamodule, ckpt_path=train_config.resume or None)
        return

    model = _VARIANTS[args.variant](**model_kwargs)  # type: ignore[no-untyped-call]
    model.train(**train_kwargs)  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    main()
