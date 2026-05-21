#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Train Ref-UV DETR with RF-DETR's official Lightning trainer."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ref_uv_detr.data import PairedRFDETRDataModule
from ref_uv_detr.modeling import RefUVConfig
from ref_uv_detr.training import RefUVModelModule

from rfdetr.assets.model_weights import download_pretrain_weights, get_model_cache_dir
from rfdetr.config import (
    ModelConfig,
    RFDETRLargeConfig,
    RFDETRMediumConfig,
    RFDETRNanoConfig,
    RFDETRSmallConfig,
    TrainConfig,
)
from rfdetr.detr import RFDETR
from rfdetr.training.trainer import build_trainer
from rfdetr.utilities.logger import get_logger

logger = get_logger()


def _timestamped_output_dir(prefix: str) -> str:
    """Return an output directory ending with the current wall-clock timestamp.

    Args:
        prefix: Base experiment directory.

    Returns:
        Timestamped output directory.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(prefix) / stamp)


TRAINING_DEFAULTS: dict[str, Any] = {
    # Edit this block for routine training. CLI arguments still override these
    # values when you want a one-off run.
    "dataset_dir": "data",
    "white_dir": "data",
    "output_dir": _timestamped_output_dir("output/ref_uv_small_2x4090"),
    "variant": "small",
    "resolution": None,
    "pretrain_weights": None,
    "teacher_checkpoint": None,
    # The Ref-UV branch loads an extra reference path and, when enabled, a
    # frozen UV teacher. Give it a longer ceiling than the pure teacher, and
    # rely on validation mAP early stopping to avoid over-training.
    "epochs": 120,
    # Two RTX 4090 cards: this is per-GPU batch size. Effective global batch is
    # batch_size * grad_accum_steps * devices = 8 * 1 * 2 = 16. This keeps the
    # global batch aligned with RF-DETR's recommended target while using more of
    # the available VRAM and avoiding unnecessary gradient accumulation.
    "batch_size": 8,
    "grad_accum_steps": 1,
    # Keep the fine-tuning rate close to RF-DETR's official defaults. Newly
    # added Ref-UV modules use lr; RF-DETR's backbone groups use lr_encoder plus
    # layer/component decay, so the UV representation changes conservatively.
    "lr": 1e-4,
    "lr_encoder": 1.5e-4,
    "warmup_epochs": 1.0,
    "lr_scheduler": "cosine",
    "lr_min_factor": 0.05,
    "num_workers": 8,
    "device": "gpu",
    "devices": 2,
    # beta_cls/beta_box intentionally start at zero, so early DDP steps can
    # leave parts of the reference branch without gradients.
    "strategy": "ddp_find_unused_parameters_true",
    "uv_token": "uv",
    "white_token": "white",
    "allow_missing_white": False,
    "lambda_teacher_cls": 0.2,
    "lambda_teacher_box": 0.5,
    "lambda_prior": 0.05,
    "lambda_gate": 0.001,
    "max_cls_beta": 0.10,
    "max_box_beta": 0.30,
    "multi_scale": True,
    "use_ema": True,
    "checkpoint_interval": 5,
    "skip_best_epochs": 5,
    "early_stopping": True,
    "early_stopping_patience": 20,
    "early_stopping_min_delta": 0.001,
    "early_stopping_use_ema": True,
    "eval_interval": 1,
    "seed": 42,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 4,
    "run_test": False,
    "no_pretrain": False,
}

_MODEL_CONFIGS: dict[str, type[ModelConfig]] = {
    "nano": RFDETRNanoConfig,
    "small": RFDETRSmallConfig,
    "medium": RFDETRMediumConfig,
    "large": RFDETRLargeConfig,
}


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI value.

    Args:
        value: Raw CLI value.

    Returns:
        Parsed integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not positive.
    """
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}.")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=TRAINING_DEFAULTS["dataset_dir"], help="UV dataset root.")
    parser.add_argument("--white-dir", default=TRAINING_DEFAULTS["white_dir"], help="Paired white-light root.")
    parser.add_argument("--output-dir", default=TRAINING_DEFAULTS["output_dir"], help="Training output directory.")
    parser.add_argument(
        "--variant",
        choices=sorted(_MODEL_CONFIGS),
        default=TRAINING_DEFAULTS["variant"],
        help="RF-DETR variant.",
    )
    parser.add_argument(
        "--resolution",
        type=_positive_int,
        default=TRAINING_DEFAULTS["resolution"],
        help="Optional input resolution override.",
    )
    parser.add_argument(
        "--pretrain-weights",
        default=TRAINING_DEFAULTS["pretrain_weights"],
        help="Optional initial RF-DETR checkpoint.",
    )
    parser.add_argument(
        "--no-pretrain",
        action="store_true",
        default=TRAINING_DEFAULTS["no_pretrain"],
        help="Initialize RF-DETR from scratch.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default=TRAINING_DEFAULTS["teacher_checkpoint"],
        help="Pure-UV teacher checkpoint for preservation loss.",
    )
    parser.add_argument("--epochs", type=_positive_int, default=TRAINING_DEFAULTS["epochs"], help="Training epochs.")
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=TRAINING_DEFAULTS["batch_size"],
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=_positive_int,
        default=TRAINING_DEFAULTS["grad_accum_steps"],
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--lr", type=float, default=TRAINING_DEFAULTS["lr"], help="Decoder/reference learning rate.")
    parser.add_argument(
        "--lr-encoder",
        type=float,
        default=TRAINING_DEFAULTS["lr_encoder"],
        help="Backbone learning rate.",
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
    parser.add_argument("--num-workers", type=int, default=TRAINING_DEFAULTS["num_workers"], help="DataLoader workers.")
    parser.add_argument(
        "--device",
        default=TRAINING_DEFAULTS["device"],
        help='Trainer accelerator: "auto", "cpu", "gpu", or "cuda".',
    )
    parser.add_argument("--devices", default=TRAINING_DEFAULTS["devices"], help='Lightning devices, e.g. 2 or "0,1".')
    parser.add_argument("--strategy", default=TRAINING_DEFAULTS["strategy"], help='Lightning strategy, e.g. "ddp".')
    parser.add_argument("--resume", default=None, help="Lightning/RF-DETR checkpoint to resume.")
    parser.add_argument(
        "--checkpoint-interval",
        type=_positive_int,
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
        "--uv-token",
        default=TRAINING_DEFAULTS["uv_token"],
        help="Filename token replaced when searching white images.",
    )
    parser.add_argument(
        "--white-token",
        default=TRAINING_DEFAULTS["white_token"],
        help="Replacement token for white image names.",
    )
    parser.add_argument(
        "--allow-missing-white",
        action="store_true",
        default=TRAINING_DEFAULTS["allow_missing_white"],
        help="Use UV image as fallback if white is missing.",
    )
    parser.add_argument(
        "--lambda-teacher-cls",
        type=float,
        default=TRAINING_DEFAULTS["lambda_teacher_cls"],
        help="Teacher class preservation weight.",
    )
    parser.add_argument(
        "--lambda-teacher-box",
        type=float,
        default=TRAINING_DEFAULTS["lambda_teacher_box"],
        help="Teacher box preservation weight.",
    )
    parser.add_argument(
        "--lambda-prior",
        type=float,
        default=TRAINING_DEFAULTS["lambda_prior"],
        help="Lesion prior supervision weight.",
    )
    parser.add_argument(
        "--lambda-gate",
        type=float,
        default=TRAINING_DEFAULTS["lambda_gate"],
        help="Reference gate sparsity weight.",
    )
    parser.add_argument(
        "--max-cls-beta",
        type=float,
        default=TRAINING_DEFAULTS["max_cls_beta"],
        help="Classification reference cap.",
    )
    parser.add_argument(
        "--max-box-beta",
        type=float,
        default=TRAINING_DEFAULTS["max_box_beta"],
        help="Box reference cap.",
    )
    parser.add_argument("--no-multi-scale", action="store_true", help="Disable RF-DETR multi-scale resize.")
    parser.add_argument("--no-ema", action="store_true", help="Disable RF-DETR EMA callback.")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable validation-mAP early stopping.")
    parser.add_argument(
        "--early-stopping-patience",
        type=_positive_int,
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
    parser.add_argument(
        "--eval-interval",
        type=_positive_int,
        default=TRAINING_DEFAULTS["eval_interval"],
        help="Run COCO validation every N epochs.",
    )
    parser.add_argument("--seed", type=int, default=TRAINING_DEFAULTS["seed"], help="Optional training seed.")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable DataLoader pinned memory.")
    parser.add_argument(
        "--no-persistent-workers",
        action="store_true",
        help="Disable persistent DataLoader workers.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=_positive_int,
        default=TRAINING_DEFAULTS["prefetch_factor"],
        help="DataLoader prefetch factor when num_workers > 0.",
    )
    parser.add_argument("--run-test", action="store_true", help="Run test split after training if available.")
    return parser.parse_args()


def _expand_and_download_pretrain(model_config: ModelConfig) -> None:
    """Resolve and download model weights like ``RFDETR.train`` does.

    Args:
        model_config: Model config to mutate in place.
    """
    weights = model_config.pretrain_weights
    if weights is None:
        return
    if not os.path.dirname(weights):
        os.makedirs(get_model_cache_dir(), exist_ok=True)
        weights = os.path.join(get_model_cache_dir(), weights)
    else:
        weights = os.path.realpath(os.path.expanduser(weights))
        os.makedirs(os.path.dirname(weights), exist_ok=True)
    model_config.pretrain_weights = weights
    download_pretrain_weights(weights)


def build_configs(args: argparse.Namespace) -> tuple[ModelConfig, TrainConfig]:
    """Build RF-DETR model/train configs from CLI arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Tuple of model and train configs.
    """
    model_kwargs: dict[str, Any] = {}
    if args.resolution is not None:
        model_kwargs["resolution"] = args.resolution
    if args.no_pretrain:
        model_kwargs["pretrain_weights"] = None
    if args.pretrain_weights is not None:
        model_kwargs["pretrain_weights"] = args.pretrain_weights
    model_config = _MODEL_CONFIGS[args.variant](**model_kwargs)
    _expand_and_download_pretrain(model_config)

    num_classes = RFDETR._detect_num_classes_for_training(args.dataset_dir)
    model_config.num_classes = num_classes
    model_config.model_name = f"RefUV-{args.variant}"

    train_config = TrainConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        dataset_file="roboflow",
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        warmup_epochs=args.warmup_epochs,
        lr_scheduler=args.lr_scheduler,
        lr_min_factor=args.lr_min_factor,
        num_workers=args.num_workers,
        accelerator=args.device,
        devices=args.devices,
        strategy=args.strategy,
        resume=args.resume,
        multi_scale=TRAINING_DEFAULTS["multi_scale"] and not args.no_multi_scale,
        use_ema=TRAINING_DEFAULTS["use_ema"] and not args.no_ema,
        checkpoint_interval=args.checkpoint_interval,
        skip_best_epochs=args.skip_best_epochs,
        early_stopping=TRAINING_DEFAULTS["early_stopping"] and not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_use_ema=(
            TRAINING_DEFAULTS["early_stopping_use_ema"] and not args.no_early_stopping_ema and not args.no_ema
        ),
        eval_interval=args.eval_interval,
        seed=args.seed,
        pin_memory=TRAINING_DEFAULTS["pin_memory"] and not args.no_pin_memory,
        persistent_workers=(
            TRAINING_DEFAULTS["persistent_workers"] and not args.no_persistent_workers and args.num_workers > 0
        ),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        run_test=TRAINING_DEFAULTS["run_test"] or args.run_test,
        augmentation_backend="cpu",
        notes={
            "method": "Ref-UV DETR",
            "white_reference_dir": str(Path(args.white_dir).expanduser()),
            "teacher_checkpoint": args.teacher_checkpoint,
        },
    )
    return model_config, train_config


def save_run_config(
    output_dir: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    class_names: Optional[list[str]],
) -> None:
    """Save Ref-UV run configuration for reproducibility.

    Args:
        output_dir: Output directory.
        model_config: RF-DETR model configuration.
        train_config: RF-DETR train configuration.
        class_names: Optional dataset class names.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "Ref-UV DETR",
        "model_config": model_config.model_dump(),
        "train_config": train_config.model_dump(),
        "class_names": class_names,
    }
    with open(Path(output_dir) / "ref_uv_training_config.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def main() -> None:
    """Run Ref-UV DETR training."""
    args = parse_args()
    model_config, train_config = build_configs(args)
    ref_uv_config = RefUVConfig(max_cls_beta=args.max_cls_beta, max_box_beta=args.max_box_beta)

    module = RefUVModelModule(
        model_config,
        train_config,
        ref_uv_config=ref_uv_config,
        teacher_checkpoint=args.teacher_checkpoint,
        lambda_teacher_cls=args.lambda_teacher_cls,
        lambda_teacher_box=args.lambda_teacher_box,
        lambda_prior=args.lambda_prior,
        lambda_gate=args.lambda_gate,
    )
    datamodule = PairedRFDETRDataModule(
        model_config,
        train_config,
        white_root=args.white_dir,
        uv_token=args.uv_token,
        white_token=args.white_token,
        strict_pairs=not args.allow_missing_white,
    )
    trainer = build_trainer(train_config, model_config)
    trainer.fit(module, datamodule, ckpt_path=train_config.resume or None)

    save_run_config(train_config.output_dir, model_config, train_config, datamodule.class_names)
    logger.info("Ref-UV DETR training finished. Artifacts are in %s", train_config.output_dir)


if __name__ == "__main__":
    main()
