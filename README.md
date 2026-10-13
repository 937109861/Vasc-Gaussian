# Vasc-Gaussian / X-Gaussian Structure Prior

本仓库是一个面向 **sparse-view cerebrovascular and aneurysm reconstruction** 的实验项目，基于 X-Gaussian 框架进行扩展。项目的核心目标不是单纯提高二维投影 PSNR，而是解决稀疏视角 X-ray / CT 血管重建中更关键的问题：

```text
二维投影拟合得好，并不代表三维结构重建一定正确。
```

在 sparse-view 条件下，模型可能在有限角度的二维 X-ray 投影上取得较好的指标，但三维空间中仍然出现结构断裂、虚假连接、局部膨胀、球状伪影和表面毛刺。这个问题在血管、细骨、边界复杂组织等细小结构中尤其明显。

因此，本项目围绕 **Vessel-Aware Radiative Gaussian Splatting** 展开：用一组三维 Gaussian 显式表示血管 / CT 结构，并通过二维 X-ray 投影监督训练；同时引入血管结构先验、体积约束、管状结构增强和 Gaussian 尺度约束，使最终三维脑血管与动脉瘤结构更加可靠。

当前论文题目为：

```text
Vasc-Gaussian: Vessel-Aware Radiative Gaussian Splatting for Sparse-View Cerebrovascular and Aneurysm Reconstruction
```

## Agnes Harness 竞赛接入（VascPilot）

仓库现包含一个面向 AI+科学与工程实践的 Agnes Harness 接入层。它把现有训练、渲染、体数据导出和拓扑评估封装为受信任工具，形成“预检 -> 审批 -> 执行 -> 反馈 -> 验证 -> 对比”的闭环，同时禁止智能体提交任意终端命令。

- 部署与操作：[`docs/HACKATHON_AGH_GUIDE.md`](docs/HACKATHON_AGH_GUIDE.md)
- 参赛说明草稿：[`docs/COMPETITION_SUBMISSION.md`](docs/COMPETITION_SUBMISSION.md)
- AGH 插件和安全桥：[`agh/`](agh/)
- AGH Skill：[`.agh/skills/vasc-reconstruction/SKILL.md`](.agh/skills/vasc-reconstruction/SKILL.md)
- 第三方与赛前工作披露：[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)

`quick` 模式仅用于验证端到端执行链路，不代表最终科学结果；正式指标必须来自 `full` 运行。模型 Key、原始私有数据和运行追踪不得提交到 Git。

## 1. 项目解决的问题

传统 sparse-view CT / X-ray 血管重建通常面临几个问题：

```text
1. 投影视角少，三维重建存在多解性；
2. 只优化二维投影一致性时，三维结构可能不真实；
3. 细小结构容易断裂或消失；
4. Gaussian 表示容易产生局部膨胀、球状伪影和毛刺；
5. 血管这类拓扑复杂结构容易出现假连接或断裂。
```

本项目希望解决的是：

```text
如何在 sparse-view X-ray 投影监督下，通过血管结构先验引导 3D Gaussian 优化，使三维脑血管和动脉瘤结构更加连续、细节更稳定、伪影更少。
```

项目当前重点验证对象是 `aneurism_50.pickle` 颅内血管 / 动脉瘤数据。仓库中也保留 chest、head、foot、abdomen、pancreas 等通用 CT 数据，可用于后续扩展或泛化分析，但当前论文主线以脑血管和动脉瘤重建为核心。

## 2. 整体流程

项目整体流程如下：

```text
Sparse-view X-ray projections + cone-beam geometry
        |
        v
Read dataset and construct X-ray cameras
        |
        v
Initialize 3D Gaussians
        |
        v
Render 2D X-ray projections from Gaussians
        |
        v
Optimize with projection loss
        |
        +--> PriorNet / vesselness prior
        +--> occupancy loss
        +--> low-resolution volume loss
        +--> Gaussian scale regularization
        |
        v
Export Gaussians to 256^3 CT volume
        |
        v
Vessel-aware post-processing / Slicer export
        |
        v
2D and 3D evaluation
```

简单来说：

```text
项目不是直接用 CNN 输出 CT volume，
而是优化一组三维 Gaussian，
让它们从不同角度渲染出的二维 X-ray 投影接近真实投影，
同时用结构先验约束三维空间中的形状。
```

## 3. 数据格式

项目主要读取 `.pickle` 数据，例如：

```text
data/aneurism_50.pickle
data/chest_50.pickle
data/head_50.pickle
data/foot_50.pickle
data/abdomen_50.pickle
data/pancreas_50.pickle
```

以 `aneurism_50.pickle` 为例，主要字段包括：

```text
image:              (256, 256, 256), GT CT volume
train.projections:  (50, 512, 512), training X-ray projections
train.angles:       (50,), projection angles
val.projections:    (50, 512, 512), validation projections
val.angles:         (50,), validation angles

DSD:                source-detector distance
DSO:                source-origin distance
nDetector:          detector resolution
dDetector:          detector spacing
nVoxel:             CT volume resolution
dVoxel:             voxel spacing
mode:               cone-beam geometry
```

需要注意：

```text
GT CT volume 在实验中用于 PriorNet 训练、volume loss 和评价。
真实 sparse-view CT 部署时，输入应只有投影图和几何参数。
```

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

## 4. X-ray Camera 与投影渲染

每个 X-ray 投影角度会被转换成一个虚拟相机：

```text
angle
  -> angle2pose(DSO, angle)
  -> camera-to-world matrix
  -> world-to-camera matrix
  -> CameraInfo_Xray
```

训练时，模型从这些相机角度渲染二维 X-ray projection，并与真实投影图计算 loss。

二维渲染和评估相关文件：

```text
render.py
gaussian_renderer/
evaluation/eval_nvs.py
```

## 5. 3D Gaussian 表示

项目核心模型是 `GaussianModel_Xray`：

```text
scene/gaussian_model.py
```

每个 Gaussian 主要包含：

```text
xyz             三维中心位置
scaling         x/y/z 方向尺度
rotation        三维方向，quaternion 表示
opacity         不透明度
radiodensity    CT/X-ray 衰减密度分支
features_dc     继承自 3DGS 的颜色/特征项
```

当前 CT / 血管重建更关注密度而不是颜色，因此核心密度通常使用：

```text
effective_density = opacity * radiodensity
```

这个设计比单独使用 opacity 更适合 X-ray / CT 场景，因为 CT 重建关注的是体内结构的衰减密度。

## 6. Baseline X-Gaussian 流程

最基础的流程是：

```text
initialize Gaussians
  -> render X-ray projections
  -> compare with ground-truth projections
  -> optimize Gaussian position / opacity / scale / density
  -> export volume
```

这个 baseline 能够拟合二维投影，但在三维结构上容易出现：

```text
细结构不连续
局部团块
球状噪声
血管假连接
表面毛刺
```

这也是后续加入结构先验的原因。

## 7. PriorNet 结构先验

PriorNet 的作用是从 coarse volume 中预测 soft structure prior，用于引导 Gaussian 优化。

典型流程：

```text
Sparse-view projections
    -> FDK coarse reconstruction
    -> PriorNet 3D network
    -> soft prior volume
    -> prior_data.npz
    -> X-Gaussian initialization and losses
```

相关脚本：

```text
scripts/reconstruct_fdk.py
scripts/train_prior_net.py
scripts/predict_prior_net.py
scripts/build_prior.py
```

原始 PriorNet prior 可以用于：

```text
Gaussian initialization
occupancy loss
volume loss
ROI guidance
post-processing masks
```

但对血管任务来说，原始 PriorNet prior 仍存在一个问题：它主要是结构概率或强度响应，不一定充分表达“管状结构”。

## 8. Vesselness-Fused Soft Prior

为增强细血管结构，本项目新增了 `Vesselness-Fused Soft Prior`。

实现文件：

```text
scripts/build_vesselness_prior.py
```

这个模块受 Frangi vesselness filter 启发，不是简单阈值化 prior，也不是直接用 Frangi 结果做最终分割，而是：

```text
PriorNet soft prior
    + multiscale Frangi vesselness response
    -> vesselness-fused soft prior
```

Frangi vesselness 使用多尺度 Hessian 特征值。设：

```text
|lambda_1| <= |lambda_2| <= |lambda_3|
```

代码中计算：

```text
RA = |lambda_2| / |lambda_3|
RB = |lambda_1| / sqrt(|lambda_2 * lambda_3|)
S  = sqrt(lambda_1^2 + lambda_2^2 + lambda_3^2)
```

这些量分别用于区分：

```text
plate-like structures
blob-like structures
background noise
tube-like vessel structures
```

然后将 vesselness 与 PriorNet prior 融合：

```text
P_fused(x) = P_prior(x) + w * P_vesselness(x) * (1 - P_prior(x))
```

生成命令示例：

```bash
python scripts/build_vesselness_prior.py \
  --prior output/priors/aneurism_priornet/prior_data.npz \
  --vessel_source output/priors/aneurism_priornet/prior_data.npz \
  --vessel_source_key soft_prior \
  --out_dir output/priors/aneurism_vesselness_fused \
  --geometry_pickle data/aneurism_50.pickle \
  --scales 1.0 1.5 2.0 2.5 \
  --fusion_mode boost \
  --vessel_weight 0.35 \
  --threshold 0.5 \
  --roi_dilation 2 \
  --max_points 50000
```

输出：

```text
output/priors/aneurism_vesselness_fused/prior_data.npz
output/priors/aneurism_vesselness_fused/vesselness.npy
output/priors/aneurism_vesselness_fused/prior_volume.npy
output/priors/aneurism_vesselness_fused/point_cloud.ply
```

该 prior 后续参与：

```text
初始化采样
occupancy loss
volume guidance
vessel ROI guidance
```

## 9. Occupancy Loss 与 Volume Loss

为了避免只依赖二维投影监督，项目加入低分辨率三维辅助 loss。

当前血管主线使用：

```yaml
occ_grid_size: 64
volume_grid_size: 64
```

这两个 64 并不是最终输出分辨率，而是训练中临时构建的低分辨率辅助监督网格。最终导出的 CT volume 仍然是：

```text
256 x 256 x 256
```

相关文件：

```text
utils/differentiable_volume.py
train.py
```

为什么不直接把 loss 提高到 128 或 256？

```text
高分辨率 volume loss 显存更高；
此前实验中高分辨率配置容易导致 Gaussian 被 prune 到 0；
低分辨率 loss 更稳定，适合先约束整体结构。
```

## 10. Scale-Aware Gaussian Regularization

血管重建中的毛刺、球状伪影和假连接，部分来自 Gaussian 椭球过度膨胀。

原始正则主要包含：

```text
scale_floor: 防止 Gaussian 过小
scale_aniso: 限制长短轴比例
```

但它缺少最大尺度约束。因此本项目新增：

```text
L_scale_max = mean(relu(max_scale - scale_max_axis)^2)
```

涉及文件：

```text
losses/gaussian_regularization.py
arguments/__init__.py
train.py
```

当前最终版本使用：

```yaml
scale_floor: 0.12
scale_max_axis: 1.35
scale_aniso_max_ratio: 16.0
lambda_scale_floor: 0.0003
lambda_scale_max: 0.00035
lambda_scale_aniso: 0.0007
```

实验结论：

```text
v3: scale 约束较强，毛刺减少，但血管变瘦、强核心下降；
v4: 放松 scale 约束，在保留细血管和抑制毛刺之间更平衡。
```

## 11. 当前 Aneurism 最佳配置

当前最稳定的血管训练配置：

```text
configs/aneurism_lowres_v4_scalemax_loose.yaml
```

训练：

```bash
python train.py --config configs/aneurism_lowres_v4_scalemax_loose.yaml
```

该配置基于：

```text
vesselness-fused prior
64^3 occupancy loss
64^3 volume loss
scale-aware Gaussian regularization
effective_density rendering
```

相关消融配置：

```text
configs/aneurism_lowres_v1.yaml
    Original PriorNet prior

configs/aneurism_lowres_v2.yaml
    Vesselness-fused prior

configs/aneurism_lowres_v3_scalemax.yaml
    Strong scale-max regularization

configs/aneurism_lowres_v4_scalemax_loose.yaml
    Final loose scale-aware version
```

## 12. Volume Export

训练完成后，将 Gaussian 显式 splat 到 256^3 CT volume：

```text
export_ct_volume.py
```

当前视觉效果较好的导出版本是 `fine_loose`：

```bash
python export_ct_volume.py \
  --model_path output/volume_main/aneurism_lowres_v4_scalemax_loose \
  --source_path data/aneurism_50.pickle \
  --iteration 30000 \
  --mode dgr \
  --density_mapping_mode effective_density \
  --dgr_reference_grid_size 128 \
  --dgr_splat_radius 3 \
  --dgr_min_sigma_voxels 0.30 \
  --dgr_max_sigma_voxels 2.65 \
  --dgr_sigma_scale 0.86 \
  --post_median_size 1 \
  --post_smooth_sigma 0.0 \
  --post_contrast_gamma 1.0 \
  --slicer_export \
  --out_dir output/volume_main/aneurism_lowres_v4_scalemax_loose/ct_volume_dgr_eff_ref128_fine_loose
```

检查导出是否非空：

```bash
python - <<'PY'
import numpy as np
p='output/volume_main/aneurism_lowres_v4_scalemax_loose/ct_volume_dgr_eff_ref128_fine_loose/recon_volume.npy'
v=np.load(p)
print(v.shape, v.min(), v.max(), v.mean(), np.count_nonzero(v))
PY
```

## 13. Vessel Post-Processing

血管可视化还需要后处理：

```text
scripts/export_vessel_slicer_hysteresis.py
```

它包含：

```text
hysteresis threshold
component filtering
topology ROI filtering
weak vessel bridge
floating component removal
weak tail suppression
Slicer volume export
```

当前最佳后处理：

```bash
python scripts/export_vessel_slicer_hysteresis.py \
  --input output/volume_main/aneurism_lowres_v4_scalemax_loose/ct_volume_dgr_eff_ref128_fine_loose/recon_volume.npy \
  --source_path data/aneurism_50.pickle \
  --out_dir output/volume_main/aneurism_lowres_v4_scalemax_loose/ct_volume_vessel_topology_bridge_v12_fine_loose_clean \
  --basename recon_volume_vessel_slicer \
  --low_threshold 0.024 \
  --high_threshold 0.060 \
  --min_component_voxels 280 \
  --min_seed_voxels 24 \
  --keep_largest 0 \
  --closing 0 \
  --opening 0 \
  --smooth_sigma 0.25 \
  --percentile_low 0.5 \
  --percentile_high 99.2 \
  --gamma 0.50 \
  --topology_prior_path output/priors/aneurism_priornet_practical_topology/topology_prior.npz \
  --topology_roi_source support_roi \
  --topology_roi_dilation 1 \
  --topology_keep_strong_threshold 0.060 \
  --topology_component_min_prior_overlap 0.0 \
  --topology_bridge_weak_vessels \
  --topology_bridge_low_threshold 0.020 \
  --topology_bridge_max_distance 1.2 \
  --topology_bridge_closing 0 \
  --topology_inside_weak_min_raw 0.020 \
  --topology_inside_weak_max_raw 0.13 \
  --topology_inside_weak_boost 1.22 \
  --topology_inside_weak_floor 0.09 \
  --topology_outside_weak_max_raw 0.075 \
  --topology_outside_weak_scale 0.60 \
  --remove_floating_components \
  --floating_component_keep_min_voxels 3500 \
  --floating_component_keep_strong_threshold 0.28 \
  --floating_component_keep_distance 0 \
  --suppress_weak_tails \
  --weak_tail_max_raw 0.075 \
  --weak_tail_min_distance 1.3 \
  --weak_tail_opening 0
```

查看统计：

```bash
cat output/volume_main/aneurism_lowres_v4_scalemax_loose/ct_volume_vessel_topology_bridge_v12_fine_loose_clean/recon_volume_vessel_slicer_stats.json
```

## 14. Evaluation

三维 volume 评估：

```bash
python evaluation/eval_volume.py \
  --pred output/volume_main/aneurism_lowres_v4_scalemax_loose/ct_volume_dgr_eff_ref128_fine_loose/recon_volume.npy \
  --gt_pickle data/aneurism_50.pickle \
  --out_dir output/volume_main/aneurism_lowres_v4_scalemax_loose/eval_volume_fine_loose \
  --label aneurism_lowres_v4_scalemax_loose \
  --prior_path output/priors/aneurism_vesselness_fused/prior_data.npz
```

二维渲染：

```bash
python render.py \
  -m output/volume_main/aneurism_lowres_v4_scalemax_loose \
  -s data/aneurism_50.pickle \
  --iteration 30000 \
  --skip_train
```

二维评估：

```bash
python evaluation/eval_nvs.py \
  --model_path output/volume_main/aneurism_lowres_v4_scalemax_loose \
  --iteration 30000 \
  --split test
```

建议不要只看 PSNR / SSIM。血管重建更应该关注：

```text
Dice
Precision
Recall
foreground metrics
occupancy region metrics
visible_voxels
largest_component_voxels
num_components
topology_bridge_added_voxels
```

## 15. Repository Structure

```text
arguments/                    Command-line and YAML config parameters
configs/                      Experiment configs, ignored by Git by default
cuda_rasterizer/              CUDA rasterization module
data/                         Local datasets, ignored by Git
evaluation/                   2D and 3D evaluation scripts
gaussian_renderer/            X-ray Gaussian renderer
losses/                       Projection, scale, density and volume losses
priors/                       Prior volume utilities
scene/                        Dataset reader, cameras, Gaussian model
scripts/                      Prior building, FDK, post-processing scripts
utils/                        Volume, image, geometry utilities
output/                       Training and export outputs, ignored by Git
```

Key files:

```text
train.py
render.py
export_ct_volume.py
scripts/build_vesselness_prior.py
scripts/export_vessel_slicer_hysteresis.py
losses/gaussian_regularization.py
utils/differentiable_volume.py
scene/gaussian_model.py
```

## 16. Paper Writing Direction

论文不建议写成单纯“血管分割”或“传统 CT 重建”，更适合定位为：

```text
Vessel-aware radiative Gaussian splatting for sparse-view cerebrovascular and aneurysm reconstruction.
```

推荐贡献：

```text
1. 提出 Vasc-Gaussian，一个面向稀疏视角脑血管与动脉瘤重建的 vessel-aware radiative Gaussian splatting 框架。

2. 提出 Vesselness-Fused Soft Prior，将 PriorNet soft prior 与 Frangi vesselness 响应融合，增强细小血管结构引导。

3. 提出 Scale-aware Gaussian Regularization，通过最大轴尺度和各向异性约束抑制 Gaussian 膨胀、球状伪影和表面毛刺。
```

建议实验：

```text
Baseline X-Gaussian
Original PriorNet prior
Vesselness-Fused Soft Prior
Vesselness + Scale-aware Regularization
Final export / post-processing
```

如果只针对血管写论文，重点补：

```text
1. Baseline / PriorNet / Vesselness / Final 主对比；
2. scale_max v3 vs v4 消融；
3. with / without topology bridge；
4. sharp / fine / fine_loose export 消融；
5. sparse-view robustness, e.g. 20 / 30 / 50 views if time permits.
```

## 17. GitHub Sync Notes

`data/` 和 `output/` 不应提交到 GitHub。

当前 `.gitignore` 中也忽略了 `configs/`，因此如果要同步实验配置，需要强制添加：

```bash
git add -f configs/aneurism_lowres_v1.yaml \
  configs/aneurism_lowres_v2.yaml \
  configs/aneurism_lowres_v3_scalemax.yaml \
  configs/aneurism_lowres_v4_scalemax_loose.yaml
```

推荐同步的核心文件：

```bash
git add README.md \
  arguments/__init__.py \
  losses/gaussian_regularization.py \
  train.py \
  scripts/build_vesselness_prior.py
```

然后：

```bash
git status
git commit -m "Add Vasc-Gaussian vesselness prior and scale-aware regularization"
git push origin main
```

不要直接使用 `git add .`，否则可能把数据、输出、旧文档或临时总结一起提交。
