# X-Gaussian Structure Prior

本项目是一个面向 sparse-view X-ray/CT 的三维重建实验框架。核心思想是：不用传统 CNN 直接输出 CT volume，而是用一组三维 Gaussian 表示人体组织或血管结构；训练时让这些 Gaussian 从不同角度渲染出二维 X-ray 投影，并与真实投影对齐；训练完成后，再把 Gaussian 显式 splat 到三维体素网格中，导出可在 3D Slicer 中查看的 CT volume。

当前重点实验对象是血管瘤/血管类稀疏结构。项目希望解决的问题是：二维投影指标可能很好，但三维血管结构仍可能断裂、变块、出现假血管、毛刺和背景伪影。因此本项目在 X-Gaussian 基础上加入了 radiodensity 密度分支、低分辨率 volume loss、PriorNet 结构先验、occupancy 约束、ROI densify 和 Gaussian 正则。

## 一句话概括

```text
少量 X-ray 投影 + cone-beam CT 几何参数
    -> 优化一组三维 Gaussian
    -> 渲染二维 X-ray 监督训练
    -> 用 PriorNet / volume loss / 正则约束增强三维结构
    -> DGR-style splatting 导出三维 CT volume
```

真实可用流程的输入应该是“稀疏 X-ray 投影 + 扫描几何参数”。实验数据中的 GT CT volume 主要用于监督、PriorNet 标签和评估，不应被理解为真实部署时可直接获得的输入。

## 当前血管瘤主线

当前血管瘤 soft-guide 版本配置：

```text
configs/dgr_volume_aneurism_30k_v14_priornet_softguide_vol64_nopsrefine.yaml
```

该配置的目标是使用 PriorNet 作为软引导，而不是让 prior mask 硬性支配重建结果，从而减少块状伪影并尽量保持血管连续性。

关键设置：

```yaml
source_path: data/aneurism_50.pickle
prior_path: output/priors/aneurism_priornet/prior_data.npz
iterations: 30000
density_render_mode: effective_density
init_mode: hybrid_init
hybrid_acui_points: 7000
hybrid_prior_points: 3000
use_occ_loss: true
use_volume_loss: true
volume_grid_size: 64
use_gaussian_reg: true
empty_prune_enabled: false
```

## 输入数据

项目主要读取 `.pickle` 格式的 X-ray/CT 数据，例如：

```text
data/aneurism_50.pickle
data/chest_50.pickle
data/head_50.pickle
data/abdomen_50.pickle
```

以当前 `data/aneurism_50.pickle` 为例，实际内容如下：

```text
image:              (256, 256, 256), float32, GT CT volume
train.projections:  (50, 512, 512),   float32, 训练 X-ray 投影
train.angles:       (50,),            每张训练投影对应的角度
val.projections:    (50, 512, 512),   float32, 验证 X-ray 投影
val.angles:         (50,),            每张验证投影对应的角度

DSD: 1500.0
DSO: 1000.0
nDetector: [512, 512]
dDetector: [1.0, 1.0]
nVoxel: [256, 256, 256]
dVoxel: [1.0, 1.0, 1.0]
offOrigin: [0, 0, 0]
offDetector: [0, 0]
mode: cone
accuracy: 0.5
```

字段含义：

```text
train.projections / val.projections
```

二维 X-ray 投影图。每张图是一个扫描角度下的 projection。

```text
train.angles / val.angles
```

每张投影对应的扫描角度，代码使用这些角度构造 X-ray camera pose。

```text
image
```

完整 GT CT volume。它在实验中用于 volume loss、PriorNet 训练标签和评估。真实 sparse-view CT 场景中，这个 GT volume 不应该作为重建输入。

```text
DSD / DSO / nDetector / dDetector / nVoxel / dVoxel
```

cone-beam CT 几何参数，用来描述射线源、探测器和体素网格。

数据读取入口：

```text
scene/dataset_readers.py
```

关键函数：

```text
ConeGeometry()
get_voxels()
angle2pose()
Xray_readCamerasFromTransforms()
Xray_readNerfSyntheticInfo()
```

## 从角度到相机

每个投影角度会被转换成一个 X-ray camera：

```text
angle
    -> angle2pose(DSO, angle)
    -> camera-to-world matrix
    -> world-to-camera matrix
    -> CameraInfo_Xray
```

相机中保存：

```text
R, T
FoVx, FoVy
image
angle
width, height
```

其中相机位置近似为：

```text
T = [DSO * cos(angle), DSO * sin(angle), 0]
```

## Gaussian 表示

项目的主模型是 `GaussianModel_Xray`，定义在：

```text
scene/gaussian_model.py
```

每个 Gaussian 包含：

```text
xyz
```

三维中心位置。

```text
scaling
```

Gaussian 在 x/y/z 三个方向上的尺度。

```text
rotation
```

Gaussian 的三维方向，使用 quaternion 表示。

```text
opacity
```

不透明度，控制这个 Gaussian 对渲染的贡献。

```text
radiodensity
```

CT/X-ray 衰减密度分支，是本项目重建 CT 密度的重要参数。

```text
features_dc / features_rest
```

继承自 3D Gaussian Splatting 的 SH 特征。当前血管实验中主要依赖 density，不依赖复杂颜色。

当前血管瘤配置使用：

```text
effective_density = opacity * radiodensity
```

这比单独使用 opacity 更适合 CT/X-ray，因为 CT 重建关注的是衰减密度。

## 初始化规则

初始化方式由 `init_mode` 控制。当前血管瘤主线使用：

```yaml
init_mode: hybrid_init
hybrid_acui_points: 7000
hybrid_prior_points: 3000
acui_jitter: 0.10
init_anisotropy_ratio: 3.0
```

含义：

```text
ACUI 均匀空间采样点 7000 个
PriorNet 结构区域采样点 3000 个
组合成初始 Gaussian 点云
```

`init_anisotropy_ratio: 3.0` 会让初始 Gaussian 具有一定细长性，更符合血管/组织这类非点状结构。

## PriorNet 结构先验

PriorNet 的作用是从 FDK 粗重建中预测 soft structure prior，用来引导 X-Gaussian 的三维结构。

完整流程：

```text
稀疏 X-ray projections + 几何参数
    -> FDK 粗重建 V_fdk
    -> PriorNet 3D U-Net
    -> soft structure prior S_phi
    -> prior_data.npz
    -> 注入 X-Gaussian 训练
```

PriorNet 不是最终重建器，它只是给 Gaussian 优化提供结构提示。

### 1. FDK 粗重建

脚本：

```text
scripts/reconstruct_fdk.py
```

命令：

```bash
python scripts/reconstruct_fdk.py \
  -s data/aneurism_50.pickle \
  -o output/fdk/aneurism_fdk.npy \
  --split train \
  --preview
```

输入：

```text
train.projections
train.angles
DSD / DSO / detector geometry / voxel geometry
```

输出：

```text
output/fdk/aneurism_fdk.npy
output/fdk/aneurism_fdk.json
output/fdk/slices/
```

FDK 依赖 TIGRE。

### 2. PriorNet 训练

模型定义：

```text
models/prior_net.py
```

结构是轻量 3D U-Net：

```text
V_fdk [B, 1, D, H, W]
    -> Conv3D encoder
    -> bottleneck
    -> ConvTranspose3D decoder
    -> skip connections
    -> logits [B, 1, D, H, W]
    -> sigmoid
    -> S_phi
```

训练配对文件：

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

训练命令：

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

PriorNet 的标签来自 GT CT：

```text
GT CT
    -> normalize
    -> quantile threshold
    -> binary structure target
```

PriorNet loss：

```text
loss = BCE + Dice + TV3D
```

含义：

```text
BCE
```

逐体素二分类，判断该 voxel 是否属于结构区域。

```text
Dice
```

处理血管前景稀疏问题，避免大量背景主导训练。

```text
TV3D
```

三维平滑约束，减少 prior 概率图的孤立噪声。

### 3. PriorNet 推理

命令：

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
prior_volume.npy
occupancy_mask.npy
roi_mask.npy
point_cloud.npy
point_cloud.ply
prior_data.npz
prior_stats.json
slices/
```

其中：

```text
prior_volume / soft_prior
```

每个 voxel 的结构概率，范围 0 到 1。

```text
occupancy_mask
```

`soft_prior >= threshold` 得到的结构 mask。

```text
roi_mask
```

对 occupancy mask 膨胀得到的 ROI。

```text
points / point_cloud
```

从 ROI 或 occupancy 区域采样出来的点，用于 hybrid 初始化。

## X-Gaussian 训练

训练入口：

```text
train.py
```

血管瘤主线命令：

```bash
python train.py --config configs/dgr_volume_aneurism_30k_v14_priornet_softguide_vol64_nopsrefine.yaml
```

训练流程：

```text
1. 读取 pickle
2. 构造 train / val cameras
3. 初始化 3D Gaussians
4. 每轮随机选择一个训练视角
5. differentiable renderer 渲染预测 X-ray
6. 与真实 projection 计算 loss
7. backward 更新 xyz / scale / rotation / opacity / radiodensity
8. 定期 densify / prune
9. 定期保存 point_cloud.ply 和评估快照
```

训练输出：

```text
output/volume_main/aneurism_dgr_volume_30k_v14_priornet_softguide_vol64_nopsrefine/
    cfg_args
    cameras.json
    input.ply
    loss_history.jsonl
    metrics_history.jsonl
    training_summary.json
    point_cloud/iteration_30000/point_cloud.ply
    train/...
    test/...
    volume_debug/...
    occupancy_debug/...
```

## Differentiable X-ray Renderer

渲染器位置：

```text
gaussian_renderer/__init__.py
```

输入：

```text
camera
Gaussian xyz / scale / rotation / density
background
```

输出：

```text
predicted X-ray projection
```

当前血管瘤配置：

```yaml
density_render_mode: effective_density
density_render_constant_color: true
```

所以渲染时每个 Gaussian 的贡献主要由：

```text
opacity * radiodensity
```

决定。

## Loss 和约束

这里的“正则”指 regularization 约束项，不是文本正则表达式 regex。本项目核心算法中没有依赖 regex 做重建。

总 loss 可以概括为：

```text
total_loss =
  lambda_render * render_loss
+ lambda_projection_foreground * projection_foreground_loss
+ lambda_projection_background * projection_background_loss
+ edge_loss_weight * edge_loss
+ lambda_occ * occupancy_loss
+ lambda_volume * volume_loss
+ lambda_scale_floor * scale_floor_loss
+ lambda_scale_aniso * scale_aniso_loss
+ lambda_density_entropy * density_entropy_loss
+ lambda_density_sparsity * density_sparsity_loss
```

当前血管瘤 soft-guide 主要启用：

```yaml
lambda_render: 0.20
lambda_dssim: 0.2

use_projection_support_loss: true
lambda_projection_foreground: 0.06
lambda_projection_background: 0.30

use_occ_loss: true
lambda_occ: 0.0003

use_volume_loss: true
lambda_volume: 0.02

use_gaussian_reg: true
lambda_scale_floor: 0.0005
lambda_scale_aniso: 0.001
lambda_density_sparsity: 0.0015
```

### 1. 二维投影损失

```text
render_loss = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)
```

当前：

```text
render_loss = 0.8 * L1 + 0.2 * (1 - SSIM)
```

作用：让预测 X-ray projection 和真实 projection 尽量一致。

### 2. Projection Support Loss

位置：

```text
losses/structure_losses.py
```

配置：

```yaml
projection_foreground_threshold: 0.02
projection_mask_dilation: 2
projection_background_margin: 0.01
```

规则：

```text
foreground_mask = target >= 0.02
protected_mask = dilation(foreground_mask, radius=2)
background_mask = outside protected_mask
```

前景损失：

```text
mean(|prediction - target| over foreground_mask)
```

背景假阳性损失：

```text
mean(ReLU(prediction - target - 0.01) over background_mask)
```

作用：增强稀疏血管前景，抑制背景中凭空出现的亮伪影。

### 3. Occupancy Loss

位置：

```text
losses/occupancy_loss.py
```

配置：

```yaml
occ_grid_size: 32
occ_loss_type: l1
occ_source: soft
occ_warmup_from_iter: 1000
```

流程：

```text
Gaussian xyz + opacity
    -> 32^3 occupancy grid
PriorNet soft_prior
    -> downsample 到 32^3
二者计算 L1
```

作用：让 Gaussian 空间分布更接近 PriorNet 预测的结构概率。

### 4. Volume Loss

位置：

```text
losses/volume_loss.py
```

配置：

```yaml
volume_grid_size: 64
volume_loss_type: l1
volume_density_mode: effective_density
volume_splat_mode: trilinear
volume_loss_interval: 100
volume_warmup_from_iter: 3000
```

流程：

```text
Gaussian effective_density
    -> splat 到 64^3 volume
GT CT image
    -> normalize
    -> downsample 到 64^3
二者计算 weighted L1
```

组织/前景加权：

```yaml
volume_sparse_foreground_balance: true
volume_foreground_weight: 8.0
volume_soft_tissue_weight: 16.0
volume_hard_tissue_weight: 32.0
```

背景抑制：

```yaml
volume_background_suppression_weight: 3.0
volume_background_suppression_threshold: 0.0015
```

平滑约束：

```yaml
volume_tv_weight: 0.00010
```

作用：弥补二维投影多解性，让 Gaussian 训练更关注真实三维密度。

### 5. Gaussian 正则

位置：

```text
losses/gaussian_regularization.py
```

尺度下限：

```text
min_scale = min(scale_x, scale_y, scale_z)
scale_floor_loss = ReLU(scale_floor - min_scale)
```

当前：

```yaml
scale_floor: 0.15
lambda_scale_floor: 0.0005
```

各向异性约束：

```text
ratio = max_scale / min_scale
scale_aniso_loss = ReLU(log(ratio) - log(max_ratio))
```

当前：

```yaml
scale_aniso_max_ratio: 16.0
scale_aniso_loss_mode: log
lambda_scale_aniso: 0.001
```

密度稀疏约束：

```text
density_sparsity_loss = mean(effective_density)
```

当前：

```yaml
density_sparsity_mode: effective_density
lambda_density_sparsity: 0.0015
density_sparsity_from_iter: 3000
```

作用：避免 Gaussian 塌缩、过度拉长或全空间铺满密度。

## Densify / Prune 规则

位置：

```text
scene/gaussian_model.py
```

当前配置：

```yaml
densify_from_iter: 500
densify_until_iter: 6000
densification_interval: 400
densify_grad_threshold: 0.00008
densify_min_opacity: 0.005
prune_density_mode: effective_density
prune_min_effective_density: 0.002
prune_density_from_iter: 3000
```

Densify：

```text
如果某个 Gaussian 的 view-space 梯度较大
说明该区域拟合不好
则 clone 或 split 生成更多 Gaussian
```

Prune：

```text
如果 effective_density < 0.002
则删除该 Gaussian
```

Prior refinement：

```yaml
roi_densify_enabled: true
roi_densify_weight: 1.1
empty_prune_enabled: false
```

含义：

```text
落在 PriorNet ROI 内的 Gaussian 更容易 densify
但不启用硬 empty prune，避免误删细血管
```

## 为什么使用 soft guide

早期强 prior 版本容易出现块状结构。原因可能包括：

```text
1. PriorNet mask 本身分辨率较低或边界粗糙
2. hybrid_prior_points 太多，初始化过度集中在硬 ROI
3. empty prune 过早删除 ROI 外不确定结构
4. volume_grid_size 太小，三维监督过粗
5. volume loss 只在 prior mask 内计算
6. DGR 导出半径和后处理放大块状感
```

当前 soft-guide 版本的改动：

```text
减少 prior 初始化点
增加 ACUI 均匀点比例
关闭 empty prune
降低 roi_densify_weight
降低 lambda_occ
occ_grid_size 提到 32
volume_grid_size 提到 64
关闭 volume_use_prior_mask
```

目标是在保留 PriorNet 结构提示的同时，减少块状伪影，让血管形态更自然。

## 三维 CT 导出

导出脚本：

```text
export_ct_volume.py
```

推荐命令：

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

导出过程：

```text
point_cloud.ply
    -> 读取 xyz / scaling / rotation / opacity / radiodensity
    -> density = opacity * radiodensity
    -> DGR-style anisotropic splatting
    -> median filter
    -> Gaussian smoothing
    -> normalize
    -> 保存 CT volume
```

输出：

```text
recon_volume.npy
recon_volume.mhd
recon_volume.raw
recon_volume_slicer.npy
recon_volume_slicer.mhd
recon_volume_slicer.raw
preview_slices/
export_stats.json
volume_density_histogram.json
```

`.mhd/.raw` 可导入 3D Slicer。

## 评估

三维评估：

```bash
python evaluation/eval_volume.py \
  --pred output/volume_main/aneurism_dgr_volume_30k_v14_priornet_softguide_vol64_nopsrefine/ct_volume_dgr_eff_ref128_median3_smooth015/recon_volume.npy \
  --gt_pickle data/aneurism_50.pickle \
  --out_dir output/volume_main/aneurism_dgr_volume_30k_v14_priornet_softguide_vol64_nopsrefine/ct_volume_dgr_eff_ref128_median3_smooth015/eval \
  --label aneurism_30k_v14_priornet_softguide_vol64_nopsrefine
```

二维渲染评估：

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

三维评估指标：

```text
volume_psnr
volume_mse
volume_mae
central slice PSNR / SSIM / MAE
foreground pred_mean
soft_tissue metrics
hard_tissue metrics
overlap dice / precision / recall
```

对血管瘤这类稀疏结构，不建议只看 overall PSNR，因为背景体素太多。更重要的是：

```text
血管主体是否连续
细血管是否保留
假血管和毛刺是否减少
foreground / hard_tissue pred_mean 是否提升
overlap Dice 和 recall 是否提升
3D Slicer 中形态是否自然
```

## 目录说明

```text
arguments/              参数定义
config/                 原始基础配置
configs/                当前实验配置
cuda_rasterizer/        CUDA 光栅化模块
evaluation/             二维/三维评估脚本
gaussian_renderer/      X-ray Gaussian 渲染器
losses/                 projection / occupancy / volume / regularization loss
models/                 PriorNet 3D U-Net
priors/                 prior 构建与体数据工具
refinement/             ROI densify、projection support、topology prior
scene/                  数据读取、相机、Gaussian 模型
scripts/                FDK、PriorNet、结果打包等脚本
submodules/             diff-gaussian-rasterization 和 simple-knn
utils/                  voxelization、导出、图像、日志工具
output/                 训练输出，通常不提交
data/                   数据文件，通常不提交
```

## 环境

推荐：

```bash
conda env create -f environment.yml
conda activate x_gaussian
```

如果 CUDA 扩展没有自动编译：

```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

FDK 依赖 TIGRE。由于 TIGRE 环境可能和原 X-Gaussian 环境冲突，建议单独建一个 `tigre_fdk` 环境生成 FDK，再切回 `x_gaussian` 环境训练。

## 常用命令总览

```bash
# 1. FDK 粗重建
python scripts/reconstruct_fdk.py \
  -s data/aneurism_50.pickle \
  -o output/fdk/aneurism_fdk.npy \
  --split train \
  --preview

# 2. 训练 PriorNet
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

# 3. 预测 PriorNet prior
python scripts/predict_prior_net.py \
  --checkpoint output/prior_net/aneurism/best.pth \
  --fdk output/fdk/aneurism_fdk.npy \
  --out_dir output/priors/aneurism_priornet \
  --geometry_pickle data/aneurism_50.pickle \
  --threshold 0.5 \
  --roi_dilation 2 \
  --max_points 50000

# 4. 训练 X-Gaussian
python train.py --config configs/dgr_volume_aneurism_30k_v14_priornet_softguide_vol64_nopsrefine.yaml

# 5. 导出 CT volume
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
  --slicer_export
```

## 汇报时的核心表述

```text
本项目基于 3D Gaussian 表示实现 sparse-view X-ray/CT 重建。输入为少量二维 X-ray 投影及 cone-beam 几何参数，模型通过可微 X-ray 渲染监督优化 Gaussian 的位置、尺度、方向、不透明度和 radiodensity 密度。为改善血管瘤这类稀疏结构在三维上的断裂、伪影和块状问题，项目引入 FDK + PriorNet 生成 soft structure prior，并将其用于 hybrid 初始化、occupancy loss 和 ROI densify。同时加入低分辨率 volume loss、投影前景/背景约束、Gaussian 尺度与稀疏正则。训练完成后，通过 DGR-style splatting 将 Gaussian 导出为三维 CT volume，并用 PSNR、MAE、Dice、recall 和 3D Slicer 形态进行评估。
```
