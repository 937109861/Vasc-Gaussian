# X-Gaussian 项目方法与运行流程说明

本文档用于梳理当前项目的完整技术路线，重点解释输入输出、训练流程、二维新视角渲染、三维 CT 重建、PriorNet 结构先验、评估方式和当前血管瘤实验中的问题。适合用于向老师汇报前快速建立整体理解。

## 1. 项目目标

本项目不是单纯做一个三维 CT 重建网络，而是一个基于 3D Gaussian 表示的稀疏视角 X-ray/CT 重建框架。它同时有两个输出目标：

1. 二维新视角 X-ray 投影渲染。
2. 三维 CT volume 导出与重建评估。

项目的核心思想是：用一组三维 Gaussian 参数表示人体组织或血管结构，通过可微 X-ray 渲染器从不同角度生成二维投影图；训练完成后，再把这些 Gaussian 显式 splat 到三维体素网格中，导出 CT volume。

当前新增的 PriorNet 结构先验模块，是为了解决稀疏投影条件下三维结构不稳定的问题，例如细小血管断裂、假血管、毛刺、背景伪影和二维投影拟合好但三维结构不正确。

## 2. 核心数据

项目主要使用 `.pickle` 数据文件，例如：

```text
data/chest_50.pickle
data/aneurism_50.pickle
data/head_50.pickle
data/abdomen_50.pickle
```

这些 pickle 文件通常包含以下内容：

```text
image
```

GT CT volume，也就是完整的真实三维 CT 体数据。在实验中它用于监督、评估、生成结构标签和 volume loss。

```text
train.projections
val.projections
```

二维 X-ray 投影图。每一张投影图对应一个扫描角度。

```text
train.angles
val.angles
```

每张投影对应的角度。

```text
DSD / DSO
nDetector / dDetector
nVoxel / dVoxel
offOrigin / offDetector
```

cone-beam CT 几何参数，用来描述射线源、探测器、体素网格和物理空间。

在代码中，X-ray 数据读取主要发生在：

```text
scene/dataset_readers.py
```

其中 `Xray_readCamerasFromTransforms()` 读取投影和角度，构建训练/测试相机；`Xray_readNerfSyntheticInfo()` 读取 CT volume、几何参数和初始化点云。

## 3. 稀疏投影、GT CT 和 FDK 的区别

稀疏投影不是最终 CT 图像，而是 CT 扫描过程中从少量角度采集到的二维 X-ray 投影图。

```text
稀疏投影 = 少量角度下的 X-ray projection images
GT CT = 完整三维 CT volume，即实验中的答案
FDK = 由稀疏投影和几何参数重建出来的粗 CT volume
```

真实 sparse-view CT 场景中，输入通常是：

```text
少量 X-ray 投影 + 扫描几何参数
```

目标是恢复：

```text
高质量三维 CT volume
```

在实验数据里我们有 GT CT，是为了监督和评估。但如果一个方法在训练/测试阶段直接使用 GT CT 来生成结构先验，那么它相当于使用了答案，这种先验是 oracle prior，不能代表真实可用流程。

## 4. 原始 X-Gaussian 主流程

原始主流程如下：

```text
pickle 数据
  -> 读取 train/val projections 和 camera geometry
  -> 初始化 3D Gaussian
  -> 可微 X-ray 渲染
  -> 与 GT projection 计算二维 loss
  -> 可选三维 volume loss / regularization
  -> densify / prune
  -> 保存 point_cloud.ply
  -> DGR splatting 导出 CT volume
  -> 评估二维和三维结果
```

主训练入口：

```text
train.py
```

核心 Gaussian 表示：

```text
scene/gaussian_model.py
```

可微 X-ray 渲染器：

```text
gaussian_renderer/__init__.py
```

三维体数据导出：

```text
export_ct_volume.py
```

二维评估：

```text
render.py
metrics.py
evaluation/eval_nvs.py
```

三维评估：

```text
evaluation/eval_volume.py
```

## 5. GaussianModel_Xray 是什么

本项目主模型不是传统 CNN 或 MLP，而是一组可优化的 3D Gaussian 参数。

每个 Gaussian 主要包含：

```text
xyz
```

三维位置。

```text
scaling
```

Gaussian 在三个方向上的尺度。

```text
rotation
```

Gaussian 的三维方向。

```text
opacity
```

不透明度。

```text
radiodensity
```

X-ray/CT 密度分支。当前项目用它更直接地表达 CT 衰减密度。

```text
features_dc / features_rest
```

SH 特征，主要继承自 Gaussian Splatting 的颜色/特征表示。

这些参数在 `scene/gaussian_model.py` 中被定义为 `nn.Parameter`，由 Adam 优化器更新。

训练时随机选择一个相机视角：

```text
Gaussian 参数 + camera
  -> renderer
  -> 预测 X-ray projection
  -> 与真实 projection 计算 loss
  -> backward
  -> optimizer.step()
```

## 6. 二维新视角渲染

二维任务的目标是：训练好 Gaussian 后，在未见过角度下渲染新的 X-ray 投影图。

流程：

```text
训练视角 projections
  -> 优化 3D Gaussian
  -> 输入 test camera
  -> render.py 渲染新视角 X-ray
  -> metrics.py 计算 PSNR / SSIM / LPIPS
```

这部分评价模型是否能正确拟合和泛化二维投影。

需要注意：二维指标高，不代表三维结构一定正确。例如当前血管瘤实验中曾出现：

```text
2D PSNR 很高
2D SSIM 很高
但 3D foreground density 很低
```

这说明二维投影存在多解性。很多三维结构都可能投影出类似的二维图，因此需要三维约束和结构先验。

## 7. 三维 CT volume 重建

三维输出不是网络直接吐出一个 dense volume，而是训练完成后从 Gaussian 参数显式导出。

导出流程：

```text
训练好的 point_cloud.ply
  -> 读取 Gaussian xyz / scaling / rotation / opacity / radiodensity
  -> DGR-style splatting
  -> recon_volume.npy
  -> 可选 median / smooth 后处理
  -> .mhd / .raw
  -> 3D Slicer 查看
```

导出脚本：

```text
export_ct_volume.py
```

常见导出模式：

```text
--mode dgr
--density_mapping_mode radiodensity
```

或 v14 effective density 配置中：

```text
--density_mapping_mode effective_density
```

导出后的三维评估使用：

```text
evaluation/eval_volume.py
```

输出包括：

```text
volume_psnr
volume_mse
volume_mae
slice_metrics
foreground metrics
soft_tissue / hard_tissue metrics
overlap dice / precision / recall
```

对于血管瘤这类稀疏结构，不能只看 overall PSNR，因为背景体素太多。更应该看：

```text
foreground pred_mean
hard_tissue pred_mean
overlap dice
recall
3D Slicer 视觉形态
```

## 8. 之前的结构先验是什么

之前项目中的结构先验主要是 GT-derived prior：

```text
GT CT volume data["image"]
  -> normalize
  -> threshold
  -> occupancy_mask
  -> dilation
  -> roi_mask
```

相关代码：

```text
priors/volume_prior.py
scene/dataset_readers.py
refinement/prior_refinement.py
losses/occupancy_loss.py
```

这种先验很强，因为它直接来自 GT CT。它可以作为 oracle upper bound，用来证明结构约束对 Gaussian 优化有帮助。但它在真实 sparse-view 重建中不可直接使用，因为真实测试时没有这个病人的高质量 GT CT。

## 9. 为什么引入 PriorNet

PriorNet 的目的不是替代 X-Gaussian，而是替代原来直接从 GT CT 得到的 oracle prior。

原来：

```text
GT CT -> threshold -> roi_mask / occupancy_mask
```

现在：

```text
稀疏投影 + 几何参数
  -> FDK 粗重建 V_fdk
  -> PriorNet
  -> soft structure prior S_phi
```

PriorNet 希望从真实可获得的粗重建 `V_fdk` 中预测结构概率图，作为 Gaussian 训练的软引导。

对于血管瘤，它希望解决：

```text
细血管断裂
假血管
毛刺
背景伪影
二维投影正确但三维结构错误
```

对于其他部位，它也可以作为 anatomical structure prior 使用，不局限于血管。

## 10. FDK 粗重建

PriorNet 的输入是 `V_fdk`。它由 sparse projections 和几何参数通过 FDK 得到。

新增脚本：

```text
scripts/reconstruct_fdk.py
```

示例：

```bash
python scripts/reconstruct_fdk.py \
  -s data/aneurism_50.pickle \
  -o output/fdk/aneurism_fdk.npy \
  --split train \
  --preview
```

输出：

```text
output/fdk/aneurism_fdk.npy
output/fdk/aneurism_fdk.json
output/fdk/slices/
```

FDK 依赖 TIGRE。由于 TIGRE 新版本要求 Python >= 3.10，而原项目环境是 Python 3.7，因此推荐单独建一个 `tigre_fdk` 环境生成 FDK，再切回 `x_gaussian` 环境训练。

## 11. PriorNet 网络结构

新增网络文件：

```text
models/prior_net.py
```

当前实现是一个轻量 3D U-Net。

输入：

```text
V_fdk: [B, 1, D, H, W]
```

输出：

```text
logits: [B, 1, D, H, W]
S_phi = sigmoid(logits)
```

`S_phi` 表示 soft structure prior，每个体素的值在 0 到 1 之间：

```text
越接近 1 -> 越可能是真实结构区域
越接近 0 -> 越可能是背景或伪影区域
```

网络结构：

```text
V_fdk
  -> Conv3D blocks
  -> Downsample encoder
  -> Bottleneck
  -> Upsample decoder
  -> skip connections
  -> 1x1x1 Conv
  -> logits
  -> sigmoid
  -> S_phi
```

## 12. PriorNet 训练

新增训练脚本：

```text
scripts/train_prior_net.py
```

血管瘤配对文件：

```text
configs/priornet_aneurism_pairs.json
```

内容：

```json
[
  {
    "fdk": "output/fdk/aneurism_fdk.npy",
    "gt": "data/aneurism_50.pickle"
  }
]
```

训练逻辑：

```text
输入: V_fdk
标签: O* = threshold(V_gt)
输出: S_phi = PriorNet(V_fdk)
loss: BCE + Dice + TV3D
```

loss 文件：

```text
losses/prior_net_loss.py
```

loss 作用：

```text
BCE
  逐体素分类。

Dice
  处理血管等前景稀疏问题。

TV3D
  平滑结构概率图，减少孤立噪声。
```

训练命令示例：

```bash
python scripts/train_prior_net.py \
  --pairs_json configs/priornet_aneurism_pairs.json \
  --out_dir output/prior_net/aneurism \
  --target_size 128 \
  --patch_size 96 \
  --epochs 100 \
  --steps_per_epoch 200 \
  --base_channels 16 \
  --depth 3 \
  --threshold_mode quantile \
  --threshold_value 0.975 \
  --auto_pos_weight \
  --amp
```

输出：

```text
output/prior_net/aneurism/best.pth
output/prior_net/aneurism/last.pth
training_config.json
training_summary.json
```

## 13. PriorNet 预测先验

训练完成后，用 `best.pth` 对 FDK volume 做推理：

```text
V_fdk -> PriorNet -> S_phi
```

脚本：

```text
scripts/predict_prior_net.py
```

示例：

```bash
python scripts/predict_prior_net.py \
  --checkpoint output/prior_net/aneurism/best.pth \
  --fdk output/fdk/aneurism_fdk.npy \
  --out_dir output/priors/aneurism_priornet \
  --geometry_pickle data/aneurism_50.pickle \
  --threshold 0.5 \
  --roi_dilation 2 \
  --max_points 50000
```

输出：

```text
output/priors/aneurism_priornet/prior_data.npz
output/priors/aneurism_priornet/prior_volume.npy
output/priors/aneurism_priornet/occupancy_mask.npy
output/priors/aneurism_priornet/roi_mask.npy
output/priors/aneurism_priornet/point_cloud.npy
output/priors/aneurism_priornet/point_cloud.ply
output/priors/aneurism_priornet/prior_stats.json
output/priors/aneurism_priornet/slices/
```

其中：

```text
prior_volume / soft_prior
```

是 soft structure prior。

```text
occupancy_mask
```

由 `S_phi > threshold` 得到。

```text
roi_mask
```

由 occupancy mask 膨胀得到。

```text
point_cloud
```

从 prior 区域采样出的点云，可用于 hybrid 初始化。

## 14. PriorNet 如何接入 X-Gaussian

PriorNet 输出的 `prior_data.npz` 可以从三个位置影响 Gaussian 训练：

### 14.1 初始化

如果配置中使用：

```yaml
init_mode: hybrid_init
prior_path: output/priors/aneurism_priornet/prior_data.npz
hybrid_prior_points: ...
hybrid_acui_points: ...
```

则初始化点云由两部分组成：

```text
ACUI 均匀采样点
PriorNet 结构区域采样点
```

如果 prior 点太多，容易把网络一开始就限制在块状 ROI 中。血管瘤中应避免过多硬 prior points。

### 14.2 Occupancy loss

配置：

```yaml
use_occ_loss: true
occ_source: soft
prior_path: output/priors/aneurism_priornet/prior_data.npz
```

作用：

```text
Gaussian 当前 occupancy grid
  vs
PriorNet soft prior S_phi
```

计算 L1/BCE/Dice 等约束，让 Gaussian 分布更接近结构区域。

### 14.3 ROI densify / prune

配置：

```yaml
use_prior_refinement: true
roi_densify_enabled: true
empty_prune_enabled: false/true
```

ROI densify 会让落在 ROI 区域的 Gaussian 更容易 densify。

empty prune 会剪掉 ROI 外低密度 Gaussian。血管瘤中要谨慎开启，因为细血管和不确定区域容易被误删。

## 15. 当前推荐的实验分组

为了讲清楚 PriorNet 的作用，推荐至少保留以下分组：

```text
A. 无结构先验 baseline
B. GT oracle prior
C. PriorNet predicted prior
```

更具体到血管瘤 v14：

```text
GT oracle:
configs/dgr_volume_aneurism_30k_v14_strong_prior_ultralowmem_vol32_nopsrefine.yaml

PriorNet 对照:
configs/dgr_volume_aneurism_30k_v14_priornet_ultralowmem_vol32_nopsrefine.yaml

PriorNet soft guide 去块状:
configs/dgr_volume_aneurism_30k_v14_priornet_softguide_vol64_nopsrefine.yaml
```

三者意义：

```text
GT oracle prior
  使用 GT threshold 生成结构先验，代表结构先验上限。

PriorNet predicted prior
  使用 FDK -> PriorNet 得到的先验，更符合真实 sparse-view 场景。

Soft guide version
  降低硬 prior 的影响，减少块状伪影，追求更自然的血管形态。
```

## 16. 当前血管瘤实验观察

早期 v8 + PriorNet 结果：

```text
2D 指标很好
SSIM 约 0.994
PSNR 约 46.4
LPIPS 约 0.018
```

但三维结果前景密度不足：

```text
foreground pred_mean 很低
hard_tissue pred_mean 很低
高阈值 overlap 几乎为 0
```

说明二维投影拟合好，但三维结构没有被真正重建出来。

后来 v14 + PriorNet 结果：

```text
volume_psnr 提升到约 27.18
foreground pred_mean 提升
hard_tissue pred_mean 提升
threshold 0.247 dice 提升到约 0.47
```

这说明 v14 的 effective_density、foreground balance 和 PriorNet 共同增强了前景密度。

但是可视化 `860.png` 显示块状结构明显，说明先验太硬、低分辨率 volume loss 和 prior 初始化可能导致块状伪影。为此新增了 soft guide 版本，弱化硬 ROI 控制。

## 17. 为什么会出现块状伪影

块状伪影可能来自：

1. PriorNet 输出的 mask 本身是低分辨率或边界粗糙。
2. `hybrid_prior_points` 太多，初始化点集中在硬 ROI 区域。
3. `empty_prune_enabled: true` 过早剪掉 ROI 外结构，使形态变成块状。
4. `volume_grid_size: 32` 的训练期 volume loss 太粗。
5. `volume_use_prior_mask: true` 把 volume loss 限制在硬 ROI 中。
6. 导出时 DGR splatting 半径和后处理让团块更明显。

因此 soft guide 配置做了这些调整：

```text
减少 hybrid_prior_points
增加 ACUI 初始化比例
关闭 empty prune
降低 roi_densify_weight
降低 lambda_occ
occ_grid_size 从 16 提到 32
volume_grid_size 从 32 提到 64
关闭 volume_use_prior_mask
```

## 18. 常用训练命令

### 18.1 FDK

```bash
python scripts/reconstruct_fdk.py \
  -s data/aneurism_50.pickle \
  -o output/fdk/aneurism_fdk.npy \
  --split train \
  --preview
```

### 18.2 PriorNet 训练

```bash
python scripts/train_prior_net.py \
  --pairs_json configs/priornet_aneurism_pairs.json \
  --out_dir output/prior_net/aneurism \
  --target_size 128 \
  --patch_size 96 \
  --epochs 100 \
  --steps_per_epoch 200 \
  --base_channels 16 \
  --depth 3 \
  --threshold_mode quantile \
  --threshold_value 0.975 \
  --auto_pos_weight \
  --amp
```

### 18.3 PriorNet 预测 prior

```bash
python scripts/predict_prior_net.py \
  --checkpoint output/prior_net/aneurism/best.pth \
  --fdk output/fdk/aneurism_fdk.npy \
  --out_dir output/priors/aneurism_priornet \
  --geometry_pickle data/aneurism_50.pickle \
  --threshold 0.5 \
  --roi_dilation 2 \
  --max_points 50000
```

### 18.4 训练 PriorNet soft guide 版本

```bash
python train.py --config configs/dgr_volume_aneurism_30k_v14_priornet_softguide_vol64_nopsrefine.yaml
```

### 18.5 导出 3D CT

```bash
python export_ct_volume.py \
  -m output/volume_main/aneurism_dgr_volume_30k_v14_priornet_softguide_vol64_nopsrefine \
  -s data/aneurism_50.pickle \
  --iteration -1 \
  --mode dgr \
  --density_mapping_mode effective_density \
  --dgr_splat_radius 3 \
  --dgr_min_sigma_voxels 0.40 \
  --dgr_max_sigma_voxels 3.5 \
  --dgr_sigma_scale 1.10 \
  --dgr_reference_grid_size 128 \
  --post_median_size 3 \
  --post_smooth_sigma 0.15 \
  --export_mode baseline \
  --slicer_export \
  --out_dir output/volume_main/aneurism_dgr_volume_30k_v14_priornet_softguide_vol64_nopsrefine/ct_volume_dgr_eff_ref128_median3_smooth015
```

### 18.6 评估 3D CT

```bash
python evaluation/eval_volume.py \
  --pred output/volume_main/aneurism_dgr_volume_30k_v14_priornet_softguide_vol64_nopsrefine/ct_volume_dgr_eff_ref128_median3_smooth015/recon_volume.npy \
  --gt_pickle data/aneurism_50.pickle \
  --out_dir output/volume_main/aneurism_dgr_volume_30k_v14_priornet_softguide_vol64_nopsrefine/ct_volume_dgr_eff_ref128_median3_smooth015/eval \
  --label aneurism_30k_v14_priornet_softguide_vol64_nopsrefine
```

### 18.7 2D 渲染和评估

```bash
python render.py \
  -m output/volume_main/aneurism_dgr_volume_30k_v14_priornet_softguide_vol64_nopsrefine \
  --iteration -1 \
  --skip_train
```

```bash
python metrics.py \
  -m output/volume_main/aneurism_dgr_volume_30k_v14_priornet_softguide_vol64_nopsrefine
```

## 19. 多部位扩展

PriorNet 不是血管专用网络。当前实现是通用的 3D U-Net。

对于不同部位，可以共用同一个网络结构：

```text
models/prior_net.py
```

区别在于训练数据和 checkpoint：

```text
单部位 PriorNet
  只用某一个部位训练，容易过拟合但验证快。

多部位 general PriorNet
  用 chest/head/abdomen/aneurism 等多个 FDK-GT 配对一起训练，泛化性更好。

general + finetune
  先训练通用模型，再对血管瘤等困难部位微调。
```

多部位配对文件可以写成：

```json
[
  {"fdk": "output/fdk/chest_fdk.npy", "gt": "data/chest_50.pickle"},
  {"fdk": "output/fdk/head_fdk.npy", "gt": "data/head_50.pickle"},
  {"fdk": "output/fdk/abdomen_fdk.npy", "gt": "data/abdomen_50.pickle"},
  {"fdk": "output/fdk/aneurism_fdk.npy", "gt": "data/aneurism_50.pickle"}
]
```

汇报时应强调：PriorNet 架构是通用的，血管瘤只是难例和重点验证对象。

## 20. 向老师汇报时的核心表述

可以这样总结：

```text
本项目基于 3D Gaussian 表示实现稀疏视角 X-ray/CT 重建，
同时支持二维新视角 X-ray 渲染和三维 CT volume 导出。
原始 X-Gaussian 主要依赖二维投影监督，容易出现二维指标较好但三维结构不稳定的问题。
为此，本项目引入 PriorNet，从 FDK 粗重建中预测 soft anatomical structure prior，
并将该先验注入 Gaussian 初始化、occupancy loss 和 ROI densify 过程，
以提升三维结构合理性，同时保持二维新视角渲染能力。
```

对于 GT prior 和 PriorNet prior 的区别，可以这样说：

```text
GT threshold prior 是 oracle upper bound，能够证明结构先验有效，
但真实 sparse-view CT 重建中无法直接获得目标病人的 GT CT。
PriorNet prior 通过 FDK 粗重建预测结构概率图，
更符合真实可用流程。
```

对于当前实验结果，可以这样说：

```text
当前 v14 + PriorNet 已经显著提高前景密度和 overlap 指标，
但强 prior 配置出现块状伪影。
因此后续采用 soft guide 策略，降低硬 ROI 初始化、剪枝和低分辨率 mask 对 Gaussian 的直接支配，
目标是在保持前景恢复能力的同时改善血管形态连续性和细节自然性。
```

## 21. 当前需要重点观察的结果

后续实验不要只看单一 PSNR。建议同时看：

```text
2D:
PSNR / SSIM / LPIPS

3D:
volume_psnr / volume_mae
foreground pred_mean
hard_tissue pred_mean
overlap dice / precision / recall

可视化:
central slices
3D Slicer volume rendering
血管主体是否连通
细血管是否保留
假血管和毛刺是否减少
块状伪影是否缓解
```

对于血管瘤，视觉形态和 overlap recall 往往比整体 volume PSNR 更关键。

