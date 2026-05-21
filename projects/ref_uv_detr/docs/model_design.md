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

模型只允许 white 派生出的参考 prior 在最后 decoder query 阶段，以 query-level gate 的方式小幅修正分类和定位。这个限制是为了防止 white 的纹理、光照、背景信息污染 UV-only 检测能力。

## 2. 总体结构

当前结构可以概括为：

```text
UV image
  -> RF-DETR backbone
  -> RF-DETR transformer decoder
  -> UV decoder queries
  -> UV-only class logits and boxes

UV image + White image
  -> Reference-Normalized Fluorescence Residual / white structure maps
  -> reference tokenizer
  -> misalignment-aware alignment
  -> query-centered reference sampling
  -> query-level reference gate

final query
  -> classification-localization decoupled fusion
  -> final logits and boxes
```

更具体地说，代码里的主模型是 `RefUVLWDETR`。它包装一个已经由 RF-DETR 官方构建好的 `LWDETR` 模型，然后复用原始模型的 backbone、transformer、class head、box head、query embedding、reference point embedding、aux loss 和 two-stage 逻辑。

新增的模块只有参考分支：

- `WhiteReferencedResidualTokenizer`
- `MisalignmentAwareReferenceAlignment`
- `QueryReferenceSampler`
- `LesionAwareReferenceGate`
- `ReferenceQueryUpdate`

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

当前 prior 构造在 `build_reference_prior()` 中完成，一共 11 个通道：

```text
rnfr_r
rnfr_g
rnfr_b
uv_minus_white_r
uv_minus_white_g
uv_minus_white_b
log_bg_residual
exb_residual
white_edge
white_leaf_mask
white_distance_to_boundary
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

除 RNFR 外，还加入了：

- `UV - White` RGB 残差。
- `log(UV_B / UV_G) - log(White_B / White_G)`。
- `ExB_UV - ExB_White`，其中 `ExB = 2B - G - R`。

这些通道偏向描述蓝通道/荧光响应异常，帮助 prior encoder 获取病斑候选区域。

### 4.3 white 结构通道

white 还提供三个结构图：

- `white_edge`：白光灰度 Sobel 边缘。
- `white_leaf_mask`：白光估计出的叶片区域。
- `white_distance_to_boundary`：叶片内部到边界的归一化距离。

它们的作用不是分类病害，而是帮助模型理解边界、叶片外背景、叶缘伪影和结构定位。

## 5. 参考分支模块

### 5.1 WhiteReferencedResidualTokenizer

输入：

```text
B x 11 x H x W reference prior
```

输出：

```text
multi-scale prior features
dense lesionness logits
```

实现上，它先用一个小 CNN stem 编码 11 通道 prior，再根据 RF-DETR 当前特征层的分辨率，把 prior feature resize 到每个 feature level，并通过 `1x1 Conv + GroupNorm + GELU` 投影到 RF-DETR hidden dimension。

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

### 5.3 QueryReferenceSampler

这个模块不做全图融合，而是在每个 query 的预测框中心采样 reference token。

输入：

```text
aligned prior features
UV-only query boxes in cx, cy, w, h
```

过程：

```text
query center -> grid_sample from every feature level
level softmax weight -> weighted sum
MLP projection -> reference token
```

输出：

```text
r_q: B x Q x C
```

也就是说，每个候选病斑 query 自己拿到一个局部 reference token，而不是整张图被 white 全局污染。

### 5.4 LesionAwareReferenceGate

gate 是 query-level 的可靠性判断。

输入包括：

```text
UV decoder query
reference token
UV class confidence
UV query box
sampled lesionness
```

输出：

```text
s_q = sigmoid(MLP(...))
```

默认最后一层 bias 是 `-2.0`，所以初始 gate 大约是 `sigmoid(-2) = 0.119`。这让 reference 分支在训练初期偏保守。

### 5.5 ReferenceQueryUpdate

参考 token 不直接替换 query，而是产生一个 additive update：

```text
u_q = MLP([q_uv, r_q]) * s_q
```

然后再分别送到分类和定位分支。

## 6. 分类-定位解耦融合

这是当前模型最关键的保护机制之一。

最终 query 被拆成：

```text
q_cls = q_uv + beta_cls * u_q
q_box = q_uv + beta_box * u_q
```

其中：

```text
beta_cls = max_cls_beta * tanh(beta_cls_raw)
beta_box = max_box_beta * tanh(beta_box_raw)
```

默认：

```text
max_cls_beta = 0.10
max_box_beta = 0.30
beta_cls_raw = 0
beta_box_raw = 0
```

所以模型初始时完全等价于 UV-only 的最后 query 输出；训练过程中才逐步学会是否利用 reference。并且定位分支的上限更大，分类分支的上限更小。

这对应论文叙事：

```text
UV for disease semantics.
White reference for structural localization and false-positive suppression.
```

当前实现只融合最后一层 decoder output：

- 中间 aux decoder outputs 仍保持 RF-DETR 原始 UV 路径。
- 最终 `pred_logits` 来自 `q_cls`。
- 最终 `pred_boxes` 来自 `q_box`。

这样做的好处是改动小、稳定、容易 smoke test，也更符合“white 只做受限参考”的假设。

## 7. Forward 过程

`RefUVLWDETR.forward(samples, targets)` 的关键步骤如下：

```text
1. samples 进入 RF-DETR backbone，得到 UV multi-scale features。
2. features 进入 RF-DETR transformer decoder，得到 hs 和 ref_unsigmoid。
3. 使用原始 bbox head 得到 UV-only boxes。
4. 从 targets 中取出 ref_uv_prior，并 pad 到 batch tensor 尺寸。
5. reference tokenizer 把 prior 编码成 multi-scale prior features。
6. alignment 模块把 prior features 对齐到 UV features。
7. sampler 根据最终层 UV-only boxes 的中心点采样 query reference token。
8. gate 判断每个 query 使用 reference 的可靠程度。
9. reference update 分别以 beta_cls、beta_box 注入分类和定位 query。
10. class head 输出最终 logits，box head 输出最终 boxes。
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
early_stopping_patience = 20
checkpoint_interval = 5
```

全局有效 batch：

```text
4 per GPU * 2 grad accumulation * 2 GPUs = 16
```

学习率策略：

```text
lr = 1e-4
lr_encoder = 1.5e-4
warmup_epochs = 1.0
lr_scheduler = "cosine"
lr_min_factor = 0.05
```

RF-DETR 当前 scheduler 是 step-level `LambdaLR`：

```text
warmup 阶段:
  lr_factor = current_step / warmup_steps

cosine 阶段:
  lr_factor = lr_min_factor
            + (1 - lr_min_factor) * 0.5 * (1 + cos(pi * progress))
```

因此 warmup 后主学习率从 `1e-4` 平滑下降到：

```text
1e-4 * 0.05 = 5e-6
```

输出目录默认按启动时间生成：

```text
output/ref_uv_small_2x4090/YYYYMMDD_HHMMSS
```

建议先用 `train_uv_teacher.py` 训练纯 UV teacher，再把
`checkpoint_best_total.pth` 传给 `train_ref_uv.py` 的 `--teacher-checkpoint`。
teacher 脚本保持官方 `model.train(...)` 接口，默认使用 `batch_size=8`、
`grad_accum_steps=1`、`devices=2`，同样保持全局有效 batch 为 16。

## 10. 为什么这个结构比简单融合更合理

简单融合通常是：

```text
UV feature + white feature -> detector
```

这样的问题是 white 特征太自由。它可能带来叶脉、阴影、背景、反光、拍摄差异等无关信息，最后造成负迁移。尤其是在病害语义主要来自 UV 的情况下，raw white 直接进入分类路径很容易拖累 UV-only。

当前 Ref-UV DETR 的约束更强：

```text
white 不直接进主干
white 不直接改写所有 query
white 先变成 RNFR/结构 prior
reference prior 先对齐再使用
每个 query 自己决定是否使用 reference
分类和定位分开控制 reference 强度
teacher loss 约束模型不忘掉 UV-only
```

这使它更贴近当前任务的真实矛盾：不是“多一个模态就一定更好”，而是“white 只能在它擅长的位置提供帮助”。

## 11. 当前实现的边界

这个版本已经可以作为论文主模型原型，但还需要清楚它目前的边界：

1. 参考融合只发生在最后一层 decoder query 上。
    这更稳定，也更容易超过 UV-only；后续如果数据量足够，可以尝试逐层 decoder fusion。

2. alignment 是轻量 `grid_sample` offset，不是完整 deformable attention。
    目前足够表达弱错位，且计算开销小。

3. lesion prior 是 box rasterization，不是真实 lesion mask。
    它是粗监督，不能被解释为像素级病斑分割。

4. 推理时必须能拿到 paired white 图并生成 prior。
    如果不传 reference prior，模型会退化到全 0 prior 路径。

5. `beta_cls` 和 `beta_box` 初始为 0。
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
