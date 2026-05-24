# Ref-UV DETR

Reference-Guided UV Fluorescence Detection Transformer for paired UV/white leaf imagery.

The implementation is intentionally isolated from the RF-DETR source tree. It reuses RF-DETR's model builders, loss, postprocess, callbacks, optimizer grouping, and Lightning trainer, while adding a narrow reference branch around the decoder-query stage.

## Main Idea

White light is treated as a same-leaf structural reference, not as a second disease-recognition modality.

Ref-UV DETR therefore keeps the UV backbone and RF-DETR heads as the semantic path and only lets white-derived evidence influence object queries inside the decoder:

1. **Reference-Normalized Fluorescence Residual (RNFR)**
    `log((UV + eps) / (White + eps))`, centered by the leaf-level median.

2. **Compact reference prior**
    The reference encoder consumes RNFR RGB, the existing ExB residual, white edge, and raw white RGB.

3. **Decoder-level deformable fusion**
    Decoder layers keep the UV deformable cross-attention and add an independent reference deformable cross-attention branch.

4. **Query-level reference gate**
    Each query predicts whether to use reference evidence. Decoder beta starts at zero, so training begins equivalent to UV-only.

5. **UV teacher preservation**
    An optional pure-UV checkpoint supplies classification and box distillation losses so fusion cannot casually forget the UV-only model.

## Expected Dataset Layout

UV annotations should follow Roboflow COCO:

```text
dataset_uv/
  train/
    _annotations.coco.json
    image_001_uv.jpg
  valid/
    _annotations.coco.json
    image_101_uv.jpg
  test/
    _annotations.coco.json
    image_201_uv.jpg
```

White images can be in a parallel root with the same split folders and filenames, or with `uv` replaced by `white`:

```text
dataset_white/
  train/
    image_001_white.jpg
  valid/
    image_101_white.jpg
  test/
    image_201_white.jpg
```

The loader also supports this dual-modality layout:

```text
data/
  train/uv/
  train/white/
  val/uv/
  val/white/
```

## Train

Yes: train the UV-only teacher first, then train Ref-UV DETR with that teacher
checkpoint. The teacher should normally use the same RF-DETR variant as the
Ref-UV run so the preservation loss compares like-for-like predictions.

### 1. Train The Pure-UV Teacher

The teacher uses the official high-level RF-DETR `model.train(...)` interface.
For routine two-card RTX 4090 training, edit the `TRAINING_DEFAULTS` block in
[train_uv_teacher.py](train_uv_teacher.py), then run:

```bash
conda run -n rfdetr env PYTHONPATH=src \
    python projects/ref_uv_detr/train_uv_teacher.py
```

The current defaults are conservative fine-tuning values:

```text
output_dir="output/uv_dinov2_only_tuned/YYYYMMDD_HHMMSS"
variant="small"
initialization="dinov2_backbone_only"
pretrain_weights=None
patch_size=14
positional_encoding_size=37
resolution=560
epochs=120
batch_size=4
grad_accum_steps=2
lr=2e-4
lr_encoder=5e-5
warmup_epochs=5.0
lr_scheduler="cosine"
lr_min_factor=0.20
device/accelerator="gpu"
devices=2
strategy="ddp_find_unused_parameters_true"
early_stopping=True
early_stopping_patience=50
early_stopping_use_ema=True
checkpoint_interval=5
```

This gives a global effective batch size of `8 * 1 * 2 = 16`, matching
RF-DETR's documented multi-GPU recommendation. The initialization intentionally
loads only the original DINOv2 visual backbone weights; RF-DETR's projector,
decoder, query embeddings, and detection heads start random and are trained
from the first optimizer step. Use `--no-backbone-only-dinov2` only when you
want RF-DETR's full detector checkpoint fine-tuning path instead.

After training, use the best
teacher artifact, usually:

```text
output/uv_dinov2_only_tuned/<timestamp>/checkpoint_best_total.pth
```

### 2. Train Ref-UV DETR

Use the requested conda environment:

```bash
conda run -n rfdetr env PYTHONPATH=projects/ref_uv_detr:src \
    python projects/ref_uv_detr/train_ref_uv.py \
    --dataset-dir /path/to/dataset_uv \
    --white-dir /path/to/dataset_white \
    --variant small \
    --teacher-checkpoint output/uv_dinov2_only_tuned/<timestamp>/checkpoint_best_total.pth \
    --epochs 120 \
    --batch-size 4 \
    --grad-accum-steps 2 \
    --lr 1e-4 \
    --lr-encoder 1.5e-4 \
    --warmup-epochs 1 \
    --lr-scheduler cosine \
    --lr-min-factor 0.05 \
    --device gpu \
    --devices 2 \
    --strategy ddp_find_unused_parameters_true
```

For this workstation, [train_ref_uv.py](train_ref_uv.py) has an editable
`TRAINING_DEFAULTS` block near the top. It is set for two RTX 4090 cards with
extra VRAM headroom for the reference branch and frozen teacher:

```text
output_dir="output/ref_uv_small_2x4090/YYYYMMDD_HHMMSS"
initialization="dinov2_backbone_only"
pretrain_weights=None
patch_size=14
positional_encoding_size=37
resolution=560
device="gpu"
devices=2
strategy="ddp_find_unused_parameters_true"
batch_size=8
grad_accum_steps=1
lr=1e-4
lr_encoder=1.5e-4
warmup_epochs=1.0
lr_scheduler="cosine"
lr_min_factor=0.05
num_workers=8
early_stopping=True
early_stopping_patience=20
early_stopping_use_ema=True
checkpoint_interval=5
```

The output folder is generated at process start, for example
`output/ref_uv_small_2x4090/20260520_153012`. Passing `--output-dir` manually
overrides this timestamped default.

Routine training can therefore be launched by editing that block and running:

```bash
conda run -n rfdetr env PYTHONPATH=projects/ref_uv_detr:src \
    python projects/ref_uv_detr/train_ref_uv.py
```

If filenames do not use `uv` and `white`, set:

```bash
--uv-token UV --white-token White
```

## Recommended Ablations

Run these as separate output directories:

```text
UV-only RF-DETR baseline
RNFR only, no teacher
RNFR + alignment
RNFR + alignment + query gate
RNFR + alignment + query gate + decoupled beta
Full Ref-UV DETR + UV teacher preservation
```

The paper table should report `mAP@50`, `mAP@50:95`, per-class AP, and small/medium/large lesion AP. If white is doing what we want, the clearest gain should appear in `mAP@50:95` and false-positive suppression near leaf boundaries/specular artifacts.
