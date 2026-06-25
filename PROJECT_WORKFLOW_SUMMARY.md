# X-Gaussian 项目完整流程总结

本文档用于快速理解和讲述当前 X-Gaussian 项目的完整技术路线，重点覆盖从输入数据、二维渲染、二维评估，到三维 CT 重建、三维评估和最终输出的全过程。

## 1. 项目一句话概括

本项目是在原始 X-Gaussian 的基础上，把“二维 X-ray 新视角投影渲染”扩展成“可导出的三维 CT 体重建”。

也就是说，项目输入少量 X-ray 投影图和扫描几何参数，训练一组三维 Gaussian 来表示人体内部结构。训练完成后，这组三维 Gaussian 既可以用于渲染新的二维 X-ray 投影，也可以进一步 splat 到三维体素网格中，导出可在 3D Slicer 中观察和评估的 CT volume。

## 2. 整体流程

```text
pickle 数据
  -> 读取 CT 几何、训练投影、验证投影、GT CT volume
  -> 初始化 3D Gaussian 点云
  -> 可微 X-ray 渲染
  -> 二维投影 loss + 三维 volume loss + 结构/密度正则
  -> densify / prune 优化 Gaussian 分布
  -> 保存 point_cloud.ply
  -> render.py 做二维新视角渲染
  -> metrics.py 做二维 PSNR / SSIM / LPIPS
  -> export_ct_volume.py 导出三维 CT volume
  -> evaluation/eval_volume.py 做三维 PSNR / MAE / Dice / 切片评估
  -> 输出 PNG、PLY、NPY、MHD/RAW、JSON 指标
```

可以把整个项目理解成两条输出路线：

```text
训练好的 3D Gaussian
  -> 二维路线：渲染 X-ray projection，评估新视角合成质量
  -> 三维路线：导出 CT volume，评估三维结构重建质量
```

## 3. 输入数据

项目主要输入是 `data/*.pickle`，例如：

```text
data/chest_50.pickle
data/aneurism_50.pickle
data/head_50.pickle
data/abdomen_50.pickle
```

数据读取主要发生在：

```text
scene/dataset_readers.py
```

pickle 文件通常包含以下内容：

```text
image
```

GT 三维 CT volume。实验中用于三维监督、volume loss 和最终三维评估。

```text
train.projections
train.angles
```

训练用二维 X-ray 投影图和对应扫描角度。

```text
val.projections
val.angles
```

验证或测试用二维 X-ray 投影图和对应角度。

```text
DSD / DSO
nDetector / dDetector
nVoxel / dVoxel
offOrigin / offDetector
```

cone-beam CT 几何参数，用来描述 X-ray 源、探测器、体素网格和物理空间。

代码会根据这些参数构建每个投影视角对应的虚拟 camera。每个 camera 都有自己的角度、投影图、视场角和外参。

## 4. 核心表示：3D Gaussian

项目主体不是传统 CNN 直接输出 CT volume，而是一组三维 Gaussian 参数。

核心模型定义在：

```text
scene/gaussian_model.py
```

每个 Gaussian 主要包含：

```text
xyz
```

三维空间位置。

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

不透明度或可见性强度。

```text
radiodensity
```

X-ray/CT 密度分支，是本项目的重要扩展。它比单纯使用 opacity 更接近 CT 衰减密度表达。

```text
features_dc / features_rest
```

继承自 Gaussian Splatting 的特征参数。

训练过程中，这些参数都作为 `nn.Parameter` 被优化器更新。

## 5. 训练流程

训练入口是：

```text
train.py
```

典型命令如下：

```bash
python train.py --config configs/dgr_volume_chest_60k_v8_128_reg.yaml
```

训练过程可以概括为：

```text
1. 读取配置文件
2. 读取 pickle 数据
3. 构建 train/test cameras
4. 初始化 3D Gaussian 点云
5. 随机选择一个训练视角
6. 调用 gaussian_renderer.render() 渲染二维 X-ray 投影
7. 与真实 projection 计算二维 loss
8. 可选计算 occupancy loss / volume loss / projection support loss
9. 加入 Gaussian 尺度、各向异性、密度稀疏等正则
10. backward + optimizer.step()
11. 按间隔执行 densify / prune
12. 保存 point_cloud.ply、训练日志、渲染快照和分析统计
```

训练中最核心的一步是：

```text
Gaussian 参数 + camera
  -> 可微 X-ray renderer
  -> 预测 projection
  -> 与 GT projection 计算损失
  -> 反向传播更新 Gaussian
```

## 6. 二维 X-ray 渲染

二维渲染器位于：

```text
gaussian_renderer/__init__.py
```

它会把三维 Gaussian 投影到当前 camera 下，通过 CUDA rasterizer 得到二维图像。

渲染时可以选择不同密度模式：

```text
legacy
```

使用 `opacity` 作为渲染密度。

```text
radiodensity
```

使用 `radiodensity` 作为渲染密度。

```text
effective_density
```

使用 `opacity * radiodensity` 作为有效密度。

训练完成后，可以单独运行：

```bash
python render.py \
  -m output/volume_main/chest_dgr_volume_60k_v8_128_reg \
  --iteration -1 \
  --skip_train
```

`render.py` 会加载训练好的：

```text
point_cloud/iteration_xxxxx/point_cloud.ply
```

然后对 train 或 test camera 渲染图像。

典型输出路径是：

```text
test/ours_60000/renders/
test/ours_60000/gt/
test/ours_60000/gt_original/
test/ours_60000/render_stats.json
```

其中：

```text
renders/
```

预测的二维 X-ray 渲染结果。

```text
gt/
```

归一化后的 GT 投影图，用于二维指标计算。

```text
gt_original/
```

原始 GT 投影图。

```text
render_stats.json
```

记录渲染耗时、FPS、checkpoint hash 等信息。

当前 chest 主实验 60000 iteration 的训练视角渲染统计为：

```text
num_views = 50
render_time_sec = 1.2154
fps = 41.14
```

## 7. 二维评估

二维评估脚本是：

```text
metrics.py
```

运行方式：

```bash
python metrics.py \
  -m output/volume_main/chest_dgr_volume_60k_v8_128_reg
```

它会读取：

```text
renders/
gt/
```

并计算：

```text
PSNR
```

像素级重建质量，越高越好。

```text
SSIM
```

结构相似度，越高越好。

```text
LPIPS
```

感知距离，越低越好。

输出文件包括：

```text
results.json
per_view.json
```

需要注意的是，二维指标高并不代表三维结构一定正确。稀疏视角 CT 是一个多解问题，不同三维结构可能投影出相似的二维 X-ray 图。因此本项目除了二维投影 loss，还引入了三维 volume loss、occupancy loss、PriorNet 先验和 Gaussian 正则，目的是让三维结构也更合理。

## 8. 三维 CT 重建与导出

三维 CT volume 不是网络直接输出的 dense volume，而是训练完成后，从 Gaussian 参数显式导出。

导出入口是：

```text
export_ct_volume.py
```

推荐 chest 主线导出命令：

```bash
python export_ct_volume.py \
  -m output/volume_main/chest_dgr_volume_60k_v8_128_reg \
  -s data/chest_50.pickle \
  --iteration -1 \
  --mode dgr \
  --density_mapping_mode radiodensity \
  --dgr_splat_radius 3 \
  --dgr_min_sigma_voxels 0.75 \
  --dgr_max_sigma_voxels 7.5 \
  --dgr_sigma_scale 1.7 \
  --dgr_reference_grid_size 128 \
  --post_median_size 3 \
  --post_smooth_sigma 0.15 \
  --export_mode baseline \
  --slicer_export
```

导出流程为：

```text
读取 point_cloud.ply
  -> 取出 xyz / scaling / rotation / opacity / radiodensity
  -> 根据 density_mapping_mode 得到每个 Gaussian 的密度权重
  -> 使用 DGR-style splatting 写入 3D voxel grid
  -> 得到 recon_volume
  -> median filter 抑制孤立噪声和孔洞
  -> Gaussian smoothing 轻微平滑块状伪影
  -> 保存 npy / mhd / raw / slicer 显示体
```

常用导出模式：

```text
--mode center
```

只把 Gaussian 中心写入体素，简单但粗糙。

```text
--mode gaussian
```

按 Gaussian 尺度扩散写入体素。

```text
--mode dgr
```

DGR-style splatting，是当前推荐路线。

常用密度映射：

```text
--density_mapping_mode radiodensity
```

使用 CT 密度分支导出。

```text
--density_mapping_mode effective_density
```

使用 `opacity * radiodensity` 导出，aneurism v14 系列中较常用。

当前 chest 主实验导出统计：

```text
volume_shape = 128 x 128 x 128
num_gaussians = 46906
num_used_gaussians = 44213
dgr_valid_gaussian_ratio = 0.9426
post_median_size = 3
post_smooth_sigma = 0.15
```

## 9. 三维评估

三维评估脚本是：

```text
evaluation/eval_volume.py
```

运行方式：

```bash
python evaluation/eval_volume.py \
  --pred output/volume_main/chest_dgr_volume_60k_v8_128_reg/ct_volume_dgr_ref128_median3_smooth015/recon_volume.npy \
  --gt_pickle data/chest_50.pickle \
  --out_dir output/volume_main/chest_dgr_volume_60k_v8_128_reg/ct_volume_dgr_ref128_median3_smooth015/eval \
  --label chest_dgr_volume_60k_v8_128_reg
```

它会比较：

```text
pred = recon_volume.npy
gt = pickle 中的 image
```

输出：

```text
volume_metrics.json
slice_metrics.csv
slices/axial_z_gt.png
slices/axial_z_pred.png
slices/axial_z_absdiff.png
slices/coronal_y_gt.png
slices/coronal_y_pred.png
slices/coronal_y_absdiff.png
slices/sagittal_x_gt.png
slices/sagittal_x_pred.png
slices/sagittal_x_absdiff.png
```

三维指标包括：

```text
volume_psnr
volume_mse
volume_mae
```

整体三维体重建误差。

```text
slice_metrics
```

中心 axial、coronal、sagittal 三个切片的 PSNR、SSIM、MSE、MAE。

```text
foreground / soft_tissue / hard_tissue
```

不同组织区域内的重建质量。

```text
overlap dice / precision / recall
```

阈值区域的重叠程度，尤其适合看血管、骨等前景结构是否被恢复。

对于血管瘤这种稀疏结构，不能只看整体 volume PSNR。整体 PSNR 很容易被大量背景体素主导，更应关注：

```text
foreground pred_mean
hard_tissue pred_mean
overlap dice
recall
3D Slicer 可视化形态
```

## 10. PriorNet 结构先验支线

PriorNet 是为了缓解“二维投影拟合好，但三维结构不稳定”的问题。

原始 GT prior 的流程是：

```text
GT CT volume
  -> threshold
  -> occupancy_mask
  -> roi_mask
  -> prior_data.npz
```

这种方式可以作为 oracle upper bound，但真实 sparse-view CT 场景中，测试时不能直接拿到目标病人的 GT CT。

因此 PriorNet 的目标是把先验来源改成真实可获得的粗重建：

```text
稀疏 X-ray projections + 几何参数
  -> FDK 粗重建 V_fdk
  -> PriorNet
  -> soft structure prior S_phi
  -> prior_data.npz
  -> 接入 X-Gaussian 训练
```

相关脚本：

```text
scripts/reconstruct_fdk.py
scripts/train_prior_net.py
scripts/predict_prior_net.py
scripts/build_prior.py
```

### 10.1 FDK 粗重建

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

### 10.2 PriorNet 训练

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

训练目标：

```text
输入：FDK 粗重建 V_fdk
标签：由 GT CT threshold 得到的结构 mask
输出：soft structure prior S_phi
loss：BCE + Dice + TV3D
```

输出：

```text
best.pth
last.pth
training_config.json
training_summary.json
```

### 10.3 PriorNet 预测 prior

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
prior_data.npz
prior_volume.npy
occupancy_mask.npy
roi_mask.npy
point_cloud.npy
point_cloud.ply
prior_stats.json
slices/
```

其中 `prior_data.npz` 可以被 X-Gaussian 训练配置使用。

## 11. Prior 如何接入 X-Gaussian

PriorNet 输出的 `prior_data.npz` 可以从三个位置影响 Gaussian 训练。

### 11.1 初始化

配置中使用：

```yaml
init_mode: hybrid_init
prior_path: output/priors/aneurism_priornet/prior_data.npz
hybrid_prior_points: 3000
hybrid_acui_points: 7000
```

含义是初始点云由两部分组成：

```text
ACUI 均匀采样点
PriorNet 结构区域采样点
```

### 11.2 Occupancy Loss

配置中使用：

```yaml
use_occ_loss: true
occ_source: soft
prior_path: output/priors/aneurism_priornet/prior_data.npz
```

含义是把当前 Gaussian 分布映射到低分辨率 occupancy grid，然后和 PriorNet 的 soft prior 做一致性约束。

### 11.3 ROI Densify / Prune

配置中使用：

```yaml
use_prior_refinement: true
roi_densify_enabled: true
empty_prune_enabled: false
```

含义是让 ROI 区域内的 Gaussian 更容易 densify。当前 soft guide 版本关闭了 hard empty prune，避免过早删除不确定区域，减少块状伪影。

## 12. 当前推荐配置

### 12.1 Chest 主线

```text
configs/dgr_volume_chest_60k_v8_128_reg.yaml
```

特点：

```text
训练 60000 iteration
ACUI 初始化
radiodensity 密度分支
128³ volume loss
DGR-style volume splatting
median3 + smooth0.15 后处理
```

推荐导出路线：

```text
v8_128_reg training
  -> DGR radiodensity export
  -> median size 3
  -> smooth sigma 0.15
  -> 3D Slicer display adjustment
```

### 12.2 Aneurism PriorNet soft guide

```text
configs/dgr_volume_aneurism_30k_v14_priornet_softguide_vol64_nopsrefine.yaml
```

特点：

```text
训练 30000 iteration
effective_density 渲染
PriorNet soft prior
hybrid_init，但 ACUI 占主导
使用 soft occupancy loss
开启 ROI densify
关闭 hard empty prune
64³ volume loss
```

这个配置的目的，是减少强 prior 带来的块状伪影，同时保留前景结构恢复能力。

## 13. 当前实验结果摘录

### 13.1 Chest 主实验导出统计

```text
iteration = 60000
volume_shape = 128 x 128 x 128
num_gaussians = 46906
num_used_gaussians = 44213
dgr_valid_gaussian_ratio = 0.9426
post_median_size = 3
post_smooth_sigma = 0.15
```

输出文件：

```text
recon_volume.npy
recon_volume.mhd
recon_volume.raw
recon_volume_slicer.npy
recon_volume_slicer.mhd
recon_volume_slicer.raw
recon_volume_slicer.tiff
export_stats.json
volume_density_histogram.json
```

### 13.2 Aneurism PriorNet soft guide 三维评估

当前 `_diagnostics_unpack` 中保存的结果：

```text
volume_psnr = 27.9905
volume_mse = 0.001588
volume_mae = 0.003128
```

区域指标：

```text
foreground pred_mean = 0.1287
soft_tissue pred_mean = 0.0594
hard_tissue pred_mean = 0.3584
```

overlap 指标：

```text
threshold 0.0117 dice = 0.6470
threshold 0.2471 dice = 0.5918
threshold 1.0000 dice = 0.0001
```

相比 PriorNet ultralowmem vol32 版本：

```text
volume_psnr: 27.1844 -> 27.9905
volume_mae: 0.003399 -> 0.003128
threshold 0.2471 dice: 0.4718 -> 0.5918
```

说明 soft guide 版本在三维整体误差和结构重叠指标上都有提升。

## 14. 最终输出总结

完整项目运行后，会产生以下几类结果。

```text
point_cloud.ply
```

训练好的 3D Gaussian 参数，是后续二维渲染和三维导出的核心。

```text
renders/*.png
```

二维新视角 X-ray 渲染结果。

```text
gt/*.png
gt_original/*.png
```

二维评估用真值投影。

```text
results.json
per_view.json
```

二维 PSNR、SSIM、LPIPS 指标。

```text
recon_volume.npy
```

导出的三维 CT volume，主要用于定量评估。

```text
recon_volume.mhd
recon_volume.raw
```

标准 MetaImage 体数据，可被医学软件读取。

```text
recon_volume_slicer.mhd
recon_volume_slicer.raw
recon_volume_slicer.tiff
```

针对 3D Slicer 显示优化后的体数据。

```text
volume_metrics.json
slice_metrics.csv
```

三维整体、区域和切片评估结果。

```text
preview_slices/
slices/
```

导出体数据和评估结果的可视化切片。

## 15. 汇报时可以这样讲

本项目基于 3D Gaussian 表示实现稀疏视角 X-ray/CT 重建，同时支持二维新视角 X-ray 渲染和三维 CT volume 导出。输入是少量 X-ray 投影、对应扫描角度和 cone-beam CT 几何参数。训练阶段，模型用一组三维 Gaussian 表示人体组织结构，每次随机选择一个投影视角，通过可微 X-ray 渲染器生成二维投影，并与真实投影计算损失。

在原始 X-Gaussian 只关注二维投影拟合的基础上，本项目进一步加入 radiodensity 密度分支、低分辨率 volume loss、occupancy loss、Gaussian regularization 和 PriorNet soft structure prior，使 Gaussian 不仅能投影出正确的二维图像，也能在三维空间中形成更合理的结构分布。

训练完成后，模型保存为 `point_cloud.ply`。之后一方面可以用 `render.py` 渲染新视角二维 X-ray，并通过 `metrics.py` 计算 PSNR、SSIM 和 LPIPS；另一方面可以用 `export_ct_volume.py` 将 Gaussian 通过 DGR-style splatting 导出为 CT volume，再用 `evaluation/eval_volume.py` 计算三维 PSNR、MAE、组织区域指标和 Dice，并导出 `.mhd/.raw` 文件用于 3D Slicer 可视化。

项目的核心价值在于：它不是只追求二维投影指标，而是尝试解决稀疏视角 CT 中“二维拟合好但三维结构不可靠”的问题。通过三维 volume 约束和 PriorNet 结构先验，模型能够在保持二维渲染质量的同时，提高三维结构恢复的合理性。

## 16. 汇报时需要强调的注意点

1. 二维 PSNR / SSIM 高，不等于三维 CT 一定正确。稀疏投影存在多解性。

2. GT prior 是 oracle upper bound，只能证明结构先验有效；真实可用流程应强调 FDK -> PriorNet -> soft prior。

3. 对血管瘤等稀疏结构，不能只看整体 volume PSNR，还要看 foreground、hard tissue、overlap Dice、recall 和 3D Slicer 形态。

4. 当前 soft guide 版本的意义是弱化 hard prior，减少块状伪影，同时保留结构引导能力。

5. 最终结论应表述为“提升三维结构合理性和可视化效果”，不要夸大为“已经实现完整高质量临床 CT 重建”。
