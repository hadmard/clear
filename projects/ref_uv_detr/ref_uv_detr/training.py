# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Lightning module for Ref-UV DETR training."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F  # noqa: N812 -- project-conventional alias

from ref_uv_detr.initialization import load_rfdetr_backbone_weights
from ref_uv_detr.modeling import RefUVConfig, RefUVLWDETR
from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.models.lwdetr import build_model_from_config
from rfdetr.models.weights import load_pretrain_weights
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy


def _aligned_giou_loss(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute elementwise GIoU loss for aligned ``cxcywh`` boxes.

    Args:
        boxes_a: Predicted boxes with shape ``... x 4``.
        boxes_b: Teacher boxes with shape ``... x 4``.

    Returns:
        Scalar ``1 - GIoU`` loss.
    """
    a = box_cxcywh_to_xyxy(boxes_a).clamp(0.0, 1.0)
    b = box_cxcywh_to_xyxy(boxes_b).clamp(0.0, 1.0)

    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]

    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = area_a + area_b - intersection
    iou = intersection / union.clamp(min=1e-6)

    enc_lt = torch.minimum(a[..., :2], b[..., :2])
    enc_rb = torch.maximum(a[..., 2:], b[..., 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    enc_area = (enc_wh[..., 0] * enc_wh[..., 1]).clamp(min=1e-6)
    giou = iou - (enc_area - union) / enc_area
    return (1.0 - giou).mean()


def _rasterize_box_targets(
    targets: tuple[dict[str, torch.Tensor], ...] | list[dict[str, torch.Tensor]],
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Rasterize normalized target boxes into coarse lesion prior targets.

    Args:
        targets: RF-DETR target dictionaries.
        height: Output map height.
        width: Output map width.
        device: Output device.
        dtype: Output dtype.

    Returns:
        Tensor with shape ``B x 1 x H x W``.
    """
    maps = torch.zeros((len(targets), 1, height, width), device=device, dtype=dtype)
    for batch_index, target in enumerate(targets):
        boxes = target["boxes"]
        if boxes.numel() == 0:
            continue
        xyxy = box_cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)
        xyxy_px = xyxy * xyxy.new_tensor([width, height, width, height])
        for box in xyxy_px:
            x0 = int(torch.floor(box[0]).clamp(0, width - 1).item())
            y0 = int(torch.floor(box[1]).clamp(0, height - 1).item())
            x1 = int(torch.ceil(box[2]).clamp(0, width).item())
            y1 = int(torch.ceil(box[3]).clamp(0, height).item())
            if x1 > x0 and y1 > y0:
                maps[batch_index, 0, y0:y1, x0:x1] = 1.0
    return maps


class RefUVModelModule(RFDETRModelModule):
    """RF-DETR LightningModule with Ref-UV fusion and UV-teacher constraints.

    Args:
        model_config: RF-DETR model configuration.
        train_config: RF-DETR training configuration.
        ref_uv_config: Ref-UV fusion configuration.
        teacher_checkpoint: Optional pure-UV RF-DETR checkpoint path.
        lambda_teacher_cls: Weight for teacher classification preservation.
        lambda_teacher_box: Weight for teacher box preservation.
        lambda_prior: Weight for coarse lesion-prior supervision.
        lambda_gate: Weight for gate sparsity.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
        *,
        ref_uv_config: Optional[RefUVConfig] = None,
        teacher_checkpoint: Optional[str | Path] = None,
        rfdetr_backbone_weights: Optional[str | Path] = None,
        lambda_teacher_cls: float = 0.2,
        lambda_teacher_box: float = 0.5,
        lambda_prior: float = 0.05,
        lambda_gate: float = 0.001,
    ) -> None:
        super().__init__(model_config, train_config)
        if rfdetr_backbone_weights is not None:
            # Load only the official detector encoder before wrapping the base
            # LWDETR model. Projector/decoder/query/head parameters stay task-random.
            load_rfdetr_backbone_weights(self.model, self.model_config, str(rfdetr_backbone_weights))
        self.model = RefUVLWDETR(self.model, ref_uv_config or RefUVConfig())
        self.lambda_teacher_cls = lambda_teacher_cls
        self.lambda_teacher_box = lambda_teacher_box
        self.lambda_prior = lambda_prior
        self.lambda_gate = lambda_gate
        self.teacher_model = self._build_teacher(teacher_checkpoint) if teacher_checkpoint else None

    def _build_teacher(self, teacher_checkpoint: str | Path) -> torch.nn.Module:
        """Build and freeze a pure-UV RF-DETR teacher.

        Args:
            teacher_checkpoint: Checkpoint path.

        Returns:
            Frozen RF-DETR model.
        """
        teacher_config = copy.deepcopy(self.model_config)
        teacher_config.pretrain_weights = str(Path(teacher_checkpoint).expanduser().resolve())
        teacher = build_model_from_config(teacher_config, self.train_config)
        load_pretrain_weights(teacher, teacher_config)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        return teacher

    def _teacher_losses(self, outputs: Dict[str, Any], samples: Any) -> dict[str, torch.Tensor]:
        """Compute UV-preserving teacher losses.

        Args:
            outputs: Student Ref-UV outputs.
            samples: UV input batch.

        Returns:
            Dictionary of teacher loss terms.
        """
        if self.teacher_model is None:
            return {}
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(samples)
        if teacher_outputs["pred_logits"].shape != outputs["pred_logits"].shape:
            return {}

        teacher_probs = torch.sigmoid(teacher_outputs["pred_logits"])
        loss_cls = F.binary_cross_entropy_with_logits(outputs["pred_logits"], teacher_probs)
        loss_l1 = F.l1_loss(outputs["pred_boxes"], teacher_outputs["pred_boxes"])
        loss_giou = _aligned_giou_loss(outputs["pred_boxes"], teacher_outputs["pred_boxes"])
        return {
            "loss_teacher_cls": loss_cls,
            "loss_teacher_box_l1": loss_l1,
            "loss_teacher_box_giou": loss_giou,
        }

    def _reference_aux_losses(
        self,
        targets: tuple[dict[str, torch.Tensor], ...] | list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Compute prior and gate regularization losses.

        Args:
            targets: RF-DETR target dictionaries.

        Returns:
            Dictionary of auxiliary reference losses.
        """
        aux = getattr(self.model, "ref_aux", {})
        losses: dict[str, torch.Tensor] = {}
        gate = aux.get("gate")
        if gate is not None:
            losses["loss_gate_sparse"] = gate.mean()

        lesion_logits = aux.get("lesion_logits")
        if lesion_logits is not None:
            box_map = _rasterize_box_targets(
                targets,
                lesion_logits.shape[-2],
                lesion_logits.shape[-1],
                device=lesion_logits.device,
                dtype=lesion_logits.dtype,
            )
            losses["loss_lesion_prior"] = F.binary_cross_entropy_with_logits(lesion_logits, box_map)
        return losses

    def training_step(self, batch: tuple[Any, ...], batch_idx: int) -> torch.Tensor:
        """Compute Ref-UV training loss for one batch.

        Args:
            batch: Tuple of ``(NestedTensor samples, targets)``.
            batch_idx: Batch index.

        Returns:
            Scaled loss tensor for Lightning.
        """
        samples, targets = batch
        targets = list(targets)
        batch_size = len(targets)
        outputs = self.model(samples, targets)

        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        det_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

        teacher_losses = self._teacher_losses(outputs, samples)
        aux_losses = self._reference_aux_losses(targets)
        extra_loss = samples.tensors.new_zeros(())
        if "loss_teacher_cls" in teacher_losses:
            extra_loss = extra_loss + self.lambda_teacher_cls * teacher_losses["loss_teacher_cls"]
        if "loss_teacher_box_l1" in teacher_losses:
            extra_loss = extra_loss + self.lambda_teacher_box * (
                teacher_losses["loss_teacher_box_l1"] + teacher_losses["loss_teacher_box_giou"]
            )
        if "loss_lesion_prior" in aux_losses:
            extra_loss = extra_loss + self.lambda_prior * aux_losses["loss_lesion_prior"]
        if "loss_gate_sparse" in aux_losses:
            extra_loss = extra_loss + self.lambda_gate * aux_losses["loss_gate_sparse"]

        loss = det_loss + extra_loss
        loss_scaled = loss / self.trainer.accumulate_grad_batches

        train_log_sync_dist = bool(self.train_config.train_log_sync_dist)
        train_log_on_step = bool(self.train_config.train_log_on_step)
        log_items = {f"train/{k}": v for k, v in loss_dict.items()}
        log_items.update({f"train/{k}": v for k, v in teacher_losses.items()})
        log_items.update({f"train/{k}": v for k, v in aux_losses.items()})
        self.log_dict(
            log_items,
            on_step=train_log_on_step,
            on_epoch=True,
            sync_dist=train_log_sync_dist,
            batch_size=batch_size,
        )
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=train_log_on_step,
            on_epoch=True,
            sync_dist=train_log_sync_dist,
            batch_size=batch_size,
        )
        self.log("train/ref_beta_cls", self.model.beta_cls.detach(), on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/ref_beta_box", self.model.beta_box.detach(), on_step=True, on_epoch=True, batch_size=batch_size)

        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        group_lrs = [pg["lr"] for pg in optimizer.param_groups if "lr" in pg]
        if group_lrs:
            self.log("train/lr", group_lrs[0], prog_bar=True, on_step=True, on_epoch=False)
            self.log("train/lr_min", min(group_lrs), prog_bar=True, on_step=True, on_epoch=False)
            self.log("train/lr_max", max(group_lrs), prog_bar=True, on_step=True, on_epoch=False)
        return loss_scaled

    def validation_step(self, batch: tuple[Any, ...], batch_idx: int) -> Dict[str, Any]:
        """Run validation with paired reference priors.

        Args:
            batch: Tuple of ``(NestedTensor samples, targets)``.
            batch_idx: Batch index.

        Returns:
            Results and targets for COCO evaluation callbacks.
        """
        samples, targets = batch
        targets = list(targets)
        outputs = self.model(samples, targets)
        if self.train_config.compute_val_loss:
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
            self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=len(targets))

        orig_sizes = torch.stack([target["orig_size"] for target in targets])
        results = self.postprocess(outputs, orig_sizes)
        return {"results": results, "targets": targets}

    def test_step(self, batch: tuple[Any, ...], batch_idx: int) -> Dict[str, Any]:
        """Run test with paired reference priors.

        Args:
            batch: Tuple of ``(NestedTensor samples, targets)``.
            batch_idx: Batch index.

        Returns:
            Results and targets for COCO evaluation callbacks.
        """
        samples, targets = batch
        targets = list(targets)
        outputs = self.model(samples, targets)
        if self.train_config.compute_test_loss:
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
            self.log("test/loss", loss, sync_dist=True, batch_size=len(targets))

        orig_sizes = torch.stack([target["orig_size"] for target in targets])
        results = self.postprocess(outputs, orig_sizes)
        return {"results": results, "targets": targets}

    def predict_step(self, batch: tuple[Any, ...], batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Run prediction with paired reference priors.

        Args:
            batch: Tuple of ``(NestedTensor samples, targets)``.
            batch_idx: Batch index.
            dataloader_idx: Dataloader index.

        Returns:
            Postprocessed detections.
        """
        samples, targets = batch
        targets = list(targets)
        with torch.no_grad():
            outputs = self.model(samples, targets)
        orig_sizes = torch.stack([target["orig_size"] for target in targets])
        return self.postprocess(outputs, orig_sizes)
