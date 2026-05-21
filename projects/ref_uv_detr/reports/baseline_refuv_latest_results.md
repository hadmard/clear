# Baseline 与 Ref-UV 模块最新结果

记录时间：2026-05-21

本文只总结当前已经完成的两组实验：纯 UV baseline teacher，以及在其基础上加入 white reference 的 Ref-UV DETR。指标来自本地训练输出目录中的 `metrics.csv` 和训练日志。

## 1. 当前 Baseline 模型

Baseline 是纯 UV 输入的 RF-DETR Small，用作 Ref-UV 的 teacher，也作为当前效果对照。

| 项目           | 配置                                                |
| -------------- | --------------------------------------------------- |
| 模型           | RF-DETR Small                                       |
| Backbone       | `dinov2_windowed_small`                             |
| 输入           | UV 图像                                             |
| 类别           | `NPML`, `PML`, `PM`                                 |
| 类别数         | 3                                                   |
| 分辨率         | 512                                                 |
| 查询数         | 300                                                 |
| Decoder 层数   | 3                                                   |
| 训练设备       | 2 x GPU                                             |
| Batch 设置     | `batch_size=8`, `grad_accum_steps=1`, 有效 batch 16 |
| 训练轮数上限   | 120                                                 |
| Early stopping | 开启，patience 18                                   |
| EMA            | 开启                                                |

输出目录：

```text
output/uv_teacher_2x4090/20260521_024032
```

推荐 teacher checkpoint：

```text
output/uv_teacher_2x4090/20260521_024032/checkpoint_best_total.pth
```

### Baseline 最佳结果

| 指标                  | epoch | step |   数值 |
| --------------------- | ----: | ---: | -----: |
| best regular mAP50:95 |    33 | 1699 | 0.5888 |
| best EMA mAP50:95     |    30 | 1549 | 0.5961 |
| best F1               |    36 | 1849 | 0.7703 |
| best PM AP50:95       |    30 | 1549 | 0.2981 |

Baseline 的 `checkpoint_best_total.pth` 来自 EMA 分支，日志记录的最佳 EMA mAP50:95 为 `0.5961`。

按 best EMA mAP50:95 对应行统计：

| 指标         |   数值 |
| ------------ | -----: |
| mAP50:95     | 0.5869 |
| EMA mAP50:95 | 0.5961 |
| mAP50        | 0.7816 |
| mAP75        | 0.6124 |
| mAR          | 0.7023 |
| F1           | 0.7652 |
| Precision    | 0.7983 |
| Recall       | 0.7380 |
| AP NPML      | 0.7416 |
| AP PML       | 0.7209 |
| AP PM        | 0.2981 |
| Val loss     | 4.5665 |

## 2. Ref-UV 模块当前设计

Ref-UV DETR 的主干仍然是 RF-DETR Small。UV 图像仍然是主语义路径，white 图像只作为同叶片结构参考，不直接替代 UV 特征。

当前 Ref 模块包含以下部分：

1. Reference-Normalized Fluorescence Residual，简称 RNFR。

    - 计算形式为 `log((UV + eps) / (White + eps))`。
    - 先做 leaf-level median centering，减少整片叶子亮度差异。

2. White structure prior。

    - 从 white 图生成局部结构、边缘和强度参考。
    - 与 RNFR 一起组成 reference prior。

3. Misalignment-aware alignment。

    - 用轻量 offset head 做对齐。
    - 通过 `grid_sample` 将 reference prior 对齐到 RF-DETR 的多尺度特征层。

4. Query-level reference sampler。

    - 不做整图融合。
    - 每个 DETR query 根据自己的预测框中心采样局部 reference token。

5. Lesion-aware gated fusion。

    - 每个 query 预测自己的 gate。
    - gate 控制 reference token 对当前 query 的影响强弱。

6. 分类和定位解耦。

    - `beta_cls` 和 `beta_box` 分别控制分类与 box 更新。
    - 当前设计让 white reference 更偏向帮助定位，避免污染 UV 分类语义。

7. UV teacher preservation。

    - 使用纯 UV teacher checkpoint 约束 Ref-UV。
    - 包含 classification preservation 和 box preservation。

8. Lesion prior auxiliary loss。

    - 用 box rasterization 生成粗 lesion prior target。
    - 辅助 reference path 学到病斑相关区域，但它不是像素级 mask。

## 3. Ref-UV 最新训练设置

| 项目           | 配置                                                |
| -------------- | --------------------------------------------------- |
| 模型           | RefUV-small                                         |
| 基础模型       | RF-DETR Small                                       |
| Backbone       | `dinov2_windowed_small`                             |
| 输入           | UV 图像 + paired white reference                    |
| Teacher        | UV-only baseline `checkpoint_best_total.pth`        |
| 类别           | `NPML`, `PML`, `PM`                                 |
| 分辨率         | 512                                                 |
| 训练设备       | 2 x GPU                                             |
| Batch 设置     | `batch_size=4`, `grad_accum_steps=2`, 有效 batch 16 |
| 训练轮数上限   | 120                                                 |
| Early stopping | 开启，patience 20                                   |
| EMA            | 开启                                                |

输出目录：

```text
output/ref_uv_small_2x4090/20260521_045314
```

当前推荐 Ref-UV checkpoint：

```text
output/ref_uv_small_2x4090/20260521_045314/checkpoint_best_total.pth
```

训练日志记录：

```text
Best total checkpoint saved from EMA (regular=0.5957, ema=0.6060)
```

## 4. Ref-UV 最新结果

| 指标                  | epoch | step |   数值 |
| --------------------- | ----: | ---: | -----: |
| best regular mAP50:95 |    24 | 1249 | 0.5957 |
| best EMA mAP50:95     |    25 | 1299 | 0.6060 |
| best F1               |    41 | 2099 | 0.7723 |
| best PM AP50:95       |    35 | 1799 | 0.3024 |

按 best EMA mAP50:95 对应行统计：

| 指标         |   数值 |
| ------------ | -----: |
| mAP50:95     | 0.5932 |
| EMA mAP50:95 | 0.6060 |
| mAP50        | 0.7869 |
| mAP75        | 0.6114 |
| mAR          | 0.7026 |
| F1           | 0.7604 |
| Precision    | 0.7814 |
| Recall       | 0.7470 |
| AP NPML      | 0.7615 |
| AP PML       | 0.7261 |
| AP PM        | 0.2920 |
| Val loss     | 4.3528 |

按 best regular mAP50:95 对应行统计：

| 指标         |   数值 |
| ------------ | -----: |
| mAP50:95     | 0.5957 |
| EMA mAP50:95 | 0.6037 |
| mAP50        | 0.7958 |
| mAP75        | 0.6249 |
| mAR          | 0.7033 |
| F1           | 0.7622 |
| Precision    | 0.7976 |
| Recall       | 0.7379 |
| AP NPML      | 0.7591 |
| AP PML       | 0.7370 |
| AP PM        | 0.2910 |
| Val loss     | 4.3808 |

## 5. 与 Baseline 的对比

| 对比项                | Baseline | Ref-UV |    提升 |
| --------------------- | -------: | -----: | ------: |
| best regular mAP50:95 |   0.5888 | 0.5957 | +0.0069 |
| best EMA mAP50:95     |   0.5961 | 0.6060 | +0.0099 |
| best F1               |   0.7703 | 0.7723 | +0.0020 |
| best PM AP50:95       |   0.2981 | 0.3024 | +0.0042 |
| best val loss         |   4.5331 | 4.3528 | -0.1803 |

按各自 best regular mAP50:95 行看类别 AP：

| 类别 | Baseline AP | Ref-UV AP |    变化 |
| ---- | ----------: | --------: | ------: |
| NPML |      0.7461 |    0.7591 | +0.0130 |
| PML  |      0.7241 |    0.7370 | +0.0129 |
| PM   |      0.2963 |    0.2910 | -0.0053 |

按各自 best EMA mAP50:95 行看类别 AP：

| 类别 | Baseline AP | Ref-UV AP |    变化 |
| ---- | ----------: | --------: | ------: |
| NPML |      0.7416 |    0.7615 | +0.0198 |
| PML  |      0.7209 |    0.7261 | +0.0052 |
| PM   |      0.2981 |    0.2920 | -0.0061 |

## 6. 当前结论

Ref-UV 模块目前是有效的。相比纯 UV baseline，最佳 EMA mAP50:95 从 `0.5961` 提升到 `0.6060`，提升 `+0.0099`；best regular mAP50:95 从 `0.5888` 提升到 `0.5957`，提升 `+0.0069`。

收益主要来自 NPML 和 PML。Ref-UV 在 best regular 行上分别提升 `+0.0130` 和 `+0.0129`，说明 white reference 对主要类别的结构定位和判别有帮助。

PM 仍然是当前瓶颈。按 best PM AP 看，Ref-UV 可以达到 `0.3024`，比 baseline 的 `0.2981` 略高；但在整体 best mAP 行上，PM AP 没有稳定提升。这说明 Ref 模块对 PM 的收益还不稳定，后续优先看 PM 类的数据量、标注一致性、采样策略和类别权重。

当前推荐使用 Ref-UV 的 EMA best total checkpoint 作为最新模型：

```text
output/ref_uv_small_2x4090/20260521_045314/checkpoint_best_total.pth
```

## 7. 后续建议

1. 先保留当前 Ref-UV 作为主实验版本。
2. 下一轮优先围绕 PM 类做数据和 loss 调整，而不是继续堆更复杂的 reference 模块。
3. 做一个最小消融表：无 teacher、无 gate、无 lesion prior、无 RNFR，只用 white structure。
4. 如果 PM 类仍然不稳，优先检查 PM 的样本数、框尺度分布、难例和类别混淆，而不是直接换大模型。
