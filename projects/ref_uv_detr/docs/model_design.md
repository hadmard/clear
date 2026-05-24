# Ref-UV DETR 模型设计说明

本文档对应当前代码实现，核心文件在：

- `projects/ref_uv_detr/ref_uv_detr/priors.py`
- `projects/ref_uv_detr/ref_uv_detr/data.py`
- `projects/ref_uv_detr/ref_uv_detr/modeling.py`
- `projects/ref_uv_detr/ref_uv_detr/training.py`
- `projects/ref_uv_detr/train_ref_uv.py`

## 1. 核心判断

当前模型的基本判断是：

> White 不是第二个病害识别模态，而是 UV 病斑检测的同叶片结构参考系。

也就是说，模型不应该把白光图当作另一张可以直接分类病害的 RGB 图。白光图更适合提供叶片边界、叶脉结构、反光伪影、背景位置、叶片内外关系等结构性线索。UV 图仍然负责病害语义，因为病害荧光响应主要来自 UV。

因此当前实现刻意避免：

- 不做完整 UV encoder + white encoder 的双主干。
- 不把 raw white RGB 直接拼进 RF-DETR 主干。
- 不让 white 特征直接主导分类。
- 不在未对齐的情况下做全图硬融合。

模型只允许 white 派生出的参考 prior 进入 decoder cross-attention，以 query-level gate 的方式影响 object query。这个限制是为了防止 white 的纹理、光照、背景信息绕过 decoder 直接污染分类/定位 head。

## 2. 总体结构

当前结构可以概括为：

```text
UV image
  -> RF-DETR backbone
  -> UV multi-scale memory

UV image + White image
  -> RNFR / ExB / white edge / raw white RGB
  -> lightweight reference encoder
  -> misalignment-aware alignment
  -> reference multi-scale memory

RF-DETR decoder layer
  -> query self-attention
  -> UV deformable cross-attention
  -> reference deformable cross-attention with independent offsets
  -> query-level gate
  -> shared FFN
  -> original RF-DETR class / box heads
```

更具体地说，代码里的主模型是 `RefUVLWDETR`。它包装一个已经由 RF-DETR 官方构建好的 `LWDETR` 模型，然后复用原始模型的 backbone、class head、box head、query embedding、reference point embedding、aux loss 和 two-stage 逻辑。新增 reference 只进入 decoder cross-attention，不再做 head 前 `q_cls/q_box` 融合。

新增的模块只有参考分支：

- `WhiteReferencedResidualTokenizer`
- `MisalignmentAwareReferenceAlignment`
- `ReferenceFusionDecoderLayer`

这使得当前项目仍然尽量复用 RF-DETR 官方训练接口、优化器分组、criterion、postprocess、callback 和 Lightning trainer。

## 3. 数据流

数据入口是 `PairedRFDETRDataModule`。它继承 RF-DETR 的 datamodule，但把普通 COCO dataset 换成了 `PairedCocoDetection`。

一个训练样本包含：

```text
UV image
white-light paired image
COCO target boxes and labels
```

但是 dataloader 最终返回给 detector 的图像仍然只有 UV：

```text
samples = normalized UV tensor
targets["ref_uv_prior"] = white-guided reference prior
```

也就是说，white 不作为 detector image 输入，而是被提前转换成一个参考 prior，放进 target dict 里，由模型 wrapper 在 forward 时读取。

### 3.1 支持的数据布局

当前代码支持两类布局。

Roboflow COCO split 形式：

```text
dataset_uv/
  train/
    _annotations.coco.json
    image_001_uv.jpg
  valid/
    _annotations.coco.json
    image_101_uv.jpg

dataset_white/
  train/
    image_001_white.jpg
  valid/
    image_101_white.jpg
```

双模态目录形式：

```text
data/
  train/uv/
  train/white/
  val/uv/
  val/white/
```

white 配对查找逻辑会尝试：

- 同 split 下的 `white/` 子目录。
- 同 split 下的同名文件。
- 用 `uv_token` 替换成 `white_token` 后的文件名。
- 如果允许 `--allow-missing-white`，缺失时用 UV 自身作为 fallback。

默认 `strict_pairs=True`，也就是 white 缺失会报错，避免训练时悄悄退化。

### 3.2 成对增强

`PairedReferenceTransform` 对 UV 和 white 使用同一个 Albumentations 几何变换，保证框、UV、white 三者空间一致。

训练时：

- resize/multi-scale 仍按 RF-DETR 逻辑走。
- 对 white reference 默认只保留几何增强。
- photometric 增强不会随便作用到 white reference prior。

这个设计是为了让 white 保持稳定的结构参考，而不是被颜色扰动破坏。

## 4. Reference-Normalized Fluorescence Residual

当前 prior 构造在 `build_reference_prior()` 中完成，一共 8 个通道：

```text
rnfr_r
rnfr_g
rnfr_b
exb_residual
white_edge
white_raw_r
white_raw_g
white_raw_b
```

### 4.1 RNFR

最重要的是前三个 RNFR 通道：

```text
R_c(x) = log((UV_c(x) + eps) / (White_c(x) + eps))
RNFR_c(x) = R_c(x) - median(R_c over leaf region)
```

然后会做 robust clipping，把数值压到大约 `[-1, 1]`。

这个设计的含义是：同一片叶子在 white 光下的局部亮度和结构，被当作 UV 响应的参考基线。模型看的不是绝对 UV 强度，而是“相对于白光结构参考之后，UV 是否异常”。

### 4.2 辅助残差通道

除 RNFR 外，只保留当前代码中已有、最贴近 UV 荧光异常的 `ExB_UV - ExB_White`，其中 `ExB = 2B - G - R`。这个通道偏向描述蓝通道响应异常，帮助 reference encoder 获取病斑候选区域。

### 4.3 white 结构通道

white 还提供一个结构图和三通道原始参考图：

- `white_edge`：白光灰度 Sobel 边缘。
- `white_raw_r/g/b`：经过同样几何增强后的白光 RGB，作为 reference encoder 的结构纹理输入。

它们的作用不是分类病害，而是帮助 decoder 在 cross-attention 里理解边界、叶片外背景、叶缘伪影和结构定位。

## 5. 参考分支模块

### 5.1 WhiteReferencedResidualTokenizer

输入：

```text
B x 8 x H x W reference prior
```

输出：

```text
multi-scale prior features
dense lesionness logits
```

实现上，它先用一个小 CNN stem 编码 8 通道 prior，再根据 RF-DETR 当前特征层的分辨率，把 prior feature resize 到每个 feature level，并通过 `1x1 Conv + GroupNorm + GELU` 投影到 RF-DETR hidden dimension。

额外的 `lesion_head` 会输出一个 dense lesionness logit map，用于：

- query gate 的输入特征。
- 训练时的粗粒度 lesion prior 辅助损失。

### 5.2 MisalignmentAwareReferenceAlignment

UV 和 white 即使是配对拍摄，也可能存在轻微错位。当前模块在每个 feature level 上预测一个小 offset：

```text
offset_l = OffsetHead(concat(F_uv_l, P_l))
P_l_aligned = grid_sample(P_l, base_grid + offset_l)
```

offset 被 `tanh` 限制，并乘以 `max_alignment_offset_px`。默认最大偏移是 2 个 feature-map pixel。

最后一个 offset conv 初始化为 0，所以训练一开始等价于不做位移。这样模型不会在初期因为随机 offset 把 reference prior 采乱。

### 5.3 ReferenceFusionDecoderLayer

当前版本不再做最后一层 head 前 query 采样融合，而是把 reference memory 送入 decoder layer。每层 decoder 先保持 RF-DETR 原始 self-attention 和 UV deformable cross-attention，再额外执行一条 reference deformable cross-attention：

```text
uv_update = MSDeformAttn(query, UV memory)
ref_update = MSDeformAttn(query, reference memory)
gate = sigmoid(MLP(query, uv_update, ref_update, lesionness))
query = query + uv_update + beta_l * gate * ref_update
query = FFN(query)
```

reference cross-attention 有独立的 sampling offsets 和 attention weights，因此可以在 object-query 层面处理 UV/white 的轻微错位。默认从第 2 个 decoder layer 开始启用 reference，layer 0 保持 UV-only。

### 5.4 Query-level gate

gate 是 query-level 的可靠性判断。输入包括 decoder query、UV cross-attention update、reference cross-attention update，以及从 reference encoder 的 dense lesionness map 采样到的候选强度。

默认最后一层 bias 是 `-2.0`，所以初始 gate 大约是 `sigmoid(-2) = 0.119`。同时 decoder reference beta 从 0 开始，保证训练第一步等价于 UV-only 路径。

## 6. Decoder-only 融合

当前版本已经去掉 head 前融合：

```text
不再使用:
q_cls = q_uv + beta_cls * ref_update
q_box = q_uv + beta_box * ref_update
```

最终预测重新回到 RF-DETR 原始形式：

```text
outputs_class = class_embed(hs)
outputs_coord = bbox_embed(hs)
```

区别是 `hs` 已经在 decoder layer 内通过 UV memory 和 reference memory 融合过。这样 white/RNFR 只能通过 deformable cross-attention 改变 object query，不能绕过 decoder 直接进入分类/定位 head。

## 7. Forward 过程

`RefUVLWDETR.forward(samples, targets)` 的关键步骤如下：

```text
1. samples 进入 RF-DETR backbone，得到 UV multi-scale features。
2. 从 targets 中取出 ref_uv_prior，并 pad 到 batch tensor 尺寸。
3. reference tokenizer 把 prior 编码成 multi-scale prior features 和 lesion logits。
4. alignment 模块把 prior features 对齐到 UV features。
5. aligned prior features 被 flatten 成 reference memory，并挂到 decoder fusion layers。
6. RF-DETR transformer decoder 在每层 cross-attention 中融合 UV memory 和 reference memory。
7. decoder 输出融合后的 hs 和 ref_unsigmoid。
8. 原始 class head 和 box head 输出最终 logits / boxes。
```

如果推理或某些调用没有传入 `targets`，模型会构造全 0 prior。这样不会崩，但也意味着 reference 分支没有实际信息。因此当前 paired validation/test/predict 都在 `RefUVModelModule` 中显式调用：

```python
outputs = self.model(samples, targets)
```

## 8. 训练目标

基础检测损失仍然使用 RF-DETR 原始 criterion：

```text
L_det = RF-DETR matching loss + cls/box/GIoU/aux losses
```

当前额外加了四类约束：

```text
L = L_det
  + lambda_teacher_cls * L_teacher_cls
  + lambda_teacher_box * (L_teacher_box_l1 + L_teacher_box_giou)
  + lambda_prior * L_lesion_prior
  + lambda_gate * L_gate_sparse
```

默认权重：

```text
lambda_teacher_cls = 0.2
lambda_teacher_box = 0.5
lambda_prior = 0.05
lambda_gate = 0.001
```

### 8.1 UV teacher preservation

如果传入 `--teacher-checkpoint`，会构建一个冻结的 pure-UV RF-DETR teacher。

teacher loss 当前实现为：

- student logits 对 teacher sigmoid 概率做 BCE。
- student boxes 对 teacher boxes 做 L1。
- student boxes 对 teacher boxes 做 aligned GIoU。

目标是让 Ref-UV 模型在融合 reference 的同时，不轻易忘掉 UV-only 模型已经学到的检测能力。

### 8.2 Lesion prior auxiliary loss

`WhiteReferencedResidualTokenizer` 会输出 dense lesionness logits。训练时，代码把 GT boxes rasterize 成粗二值图，然后做 BCE：

```text
L_lesion_prior = BCEWithLogits(lesion_logits, rasterized_boxes)
```

这个 loss 不要求 dense mask 精准，只是让 RNFR prior 分支知道“病斑大概在哪里”。

### 8.3 Gate sparsity

gate sparse loss 是：

```text
L_gate_sparse = mean(gate)
```

它鼓励模型不要对所有 query 都打开 reference 分支。这样 white reference 更像一个按需使用的校正器，而不是默认参与所有预测的第二模态。

## 9. 当前训练参数

`train_ref_uv.py` 顶部有 `TRAINING_DEFAULTS`，可以直接编辑。当前针对两张 RTX 4090 的默认值是：

```text
variant = "small"
initialization = "dinov2_backbone_only"
pretrain_weights = None
patch_size = 14
positional_encoding_size = 37
resolution = 560
epochs = 120
batch_size = 4
grad_accum_steps = 2
device = "gpu"
devices = 2
strategy = "ddp_find_unused_parameters_true"
num_workers = 8
multi_scale = True
use_ema = True
early_stopping = True
early_stopping_patience = 50
checkpoint_interval = 5
```

初始化含义：

```text
DINOv2 backbone = original DINOv2 pretrained weights
RF-DETR projector / decoder / query / heads = random init
Ref-UV reference branch = random init
all parameters = trainable
```

这个模式通过 `--backbone-only-dinov2` 控制。关闭它并传入
`--pretrain-weights` 时，才走完整 RF-DETR detector checkpoint fine-tuning。

全局有效 batch：

```text
8 per GPU * 1 grad accumulation * 2 GPUs = 16
```

学习率策略：

```text
lr = 2e-4
lr_encoder = 5e-5
warmup_epochs = 5.0
lr_scheduler = "cosine"
lr_min_factor = 0.20
```

RF-DETR 当前 scheduler 是 step-level `LambdaLR`：

```text
warmup 阶段:
  lr_factor = current_step / warmup_steps

cosine 阶段:
  lr_factor = lr_min_factor
            + (1 - lr_min_factor) * 0.5 * (1 + cos(pi * progress))
```

因此 warmup 后主学习率从 `2e-4` 平滑下降到：

```text
2e-4 * 0.20 = 4e-5
```

输出目录默认按启动时间生成：

```text
output/ref_uv_small_2x4090/YYYYMMDD_HHMMSS
```

建议先用 `train_uv_teacher.py` 训练纯 UV teacher，再把
`checkpoint_best_total.pth` 传给 `train_ref_uv.py` 的 `--teacher-checkpoint`。
teacher 脚本保持官方 `model.train(...)` 接口，默认使用 `batch_size=4`、
`grad_accum_steps=2`、`devices=2`，同样保持全局有效 batch 为 16。

## 10. 为什么这个结构比简单融合更合理

简单融合通常是：

```text
UV feature + white feature -> detector
```

这样的问题是 white 特征太自由。它可能带来叶脉、阴影、背景、反光、拍摄差异等无关信息，最后造成负迁移。尤其是在病害语义主要来自 UV 的情况下，raw white 直接进入分类路径很容易拖累 UV-only。

当前 Ref-UV DETR 的约束更强：

```text
white 不直接进主干
white 不直接进 head
white 先变成 RNFR/ExB/edge/raw-white reference prior
reference prior 先对齐再使用
每个 decoder query 自己决定是否使用 reference cross-attention
reference beta 从 0 起步，按 decoder 层保守放开
teacher loss 约束模型不忘掉 UV-only
```

这使它更贴近当前任务的真实矛盾：不是“多一个模态就一定更好”，而是“white 只能在它擅长的位置提供帮助”。

## 11. 当前实现的边界

这个版本已经可以作为论文主模型原型，但还需要清楚它目前的边界：

1. 参考融合发生在 decoder cross-attention 内，默认第 1 层保持 UV-only，后续层逐步引入 reference。
    这比 head 前融合更贴近 DAMSDet/MS-DETR 的 object-level fusion，也能保留 RF-DETR 原始 head。

2. reference 分支同时有轻量 `grid_sample` alignment 和独立 deformable offsets。
    前者给 reference memory 一个温和对齐起点，后者在 query 级别处理剩余错位。

3. lesion prior 是 box rasterization，不是真实 lesion mask。
    它是粗监督，不能被解释为像素级病斑分割。

4. 推理时必须能拿到 paired white 图并生成 prior。
    如果不传 reference prior，模型会退化到全 0 prior 路径。

5. decoder reference beta 初始为 0。
    这对稳定性很好，但 DDP 前期可能存在未使用参数，所以训练默认使用 `ddp_find_unused_parameters_true`。

## 12. 推荐实验表述

论文里可以把当前方法描述为：

> We propose Ref-UV DETR, a reference-guided detection transformer that preserves UV fluorescence semantics while using white-light imaging as an aligned structural reference for lesion localization and false-positive suppression.

推荐消融实验：

```text
UV-only RF-DETR
RNFR prior only
RNFR + alignment
RNFR + alignment + query gate
RNFR + alignment + query gate + decoupled beta
Full Ref-UV DETR + UV teacher preservation
```

如果这个设计有效，最应该提升的是：

- `mAP@50:95`
- 边界更准的框
- 叶缘/反光/背景附近 false positive 更少
- UV-only 已经很强时，Ref-UV 不发生明显负迁移
