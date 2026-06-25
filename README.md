# X-Gaussian-Structure-Prior

基于 X-Gaussian 的稀疏 X 射线 CT 三维重建实验项目。项目目标是在原始 X-Gaussian 的二维新视角投影重建能力基础上，引入可微三维体约束、DGR 风格体素化导出和结构先验分析，使模型不仅能生成二维新视角图像，也能导出可在 3D Slicer 中观察的三维 CT 体数据。

当前推荐主线不是 v9。最终采用的是 `dgr_volume_chest_60k_v8_128_reg.yaml` 训练配置，并配合 DGR 导出、`median3 + smooth0.15` 后处理获得较稳定的三维重建结果。

## 项目定位

原始 X-Gaussian 主要产出是二维新视角 X-ray 图像。本项目在此基础上扩展到三维体重建：

1. 使用 3D Gaussian 表示 CT 空间结构。
2. 通过二维投影渲染损失保持新视角成像能力。
3. 引入三维体约束，使 Gaussian 分布更接近真实 CT 体密度。
4. 使用 DGR 风格 Gaussian splatting 将训练后的 Gaussian 显式导出为 CT volume。
5. 输出 `.npy`、`.mhd/.raw` 和 Slicer 可视化体数据，支持三维切片与体绘制观察。

## 方法概览

主流程：

```text
CT pickle 数据
    -> ConeGeometry / 投影视角构建
    -> 3D Gaussian 初始化
    -> 2D X-ray differentiable rendering
    -> 2D 投影损失 + 3D volume loss + Gaussian regularization
    -> 保存 point_cloud.ply
    -> DGR volume export
    -> median / Gaussian smoothing
    -> 3D Slicer 可视化与定量评估
```

核心改动包括：

- `radiodensity` 密度分支：为 CT 体密度重建提供更直接的密度表达。
- `volume_loss`：在训练过程中对 Gaussian splatting 后的体数据施加三维监督。
- DGR 导出：将 Gaussian 按空间尺度和密度贡献 splat 到 128³ CT volume。
- Gaussian 正则：约束 Gaussian 尺度下限、各向异性和密度熵，降低块状伪影。
- 结构先验分析：保留 occupancy / ROI 相关配置，用于副实验或消融对比。

## 目录说明

```text
arguments/              参数定义
config/                 原项目基础配置
configs/                论文复现用精选实验配置
cuda_rasterizer/        CUDA 光栅化相关代码
evaluation/             体重建和新视角结果评估脚本
gaussian_renderer/      Gaussian X-ray 渲染模块
losses/                 2D/3D 损失函数与正则项
priors/                 结构先验构建与读取
refinement/             结构先验 refinement 相关模块
scene/                  数据集、相机、Gaussian 模型定义
scripts/                辅助实验脚本
submodules/             diff-gaussian-rasterization 与 simple-knn
utils/                  体素化、图像、系统工具函数
fig/                    论文或 README 展示图
```

未上传内容：

- `data/`：数据集体积文件较大，不随仓库发布。
- `output/`：训练输出、checkpoint、导出体数据不随仓库发布。
- 根目录截图、PPT、临时文件不随仓库发布。

## 环境配置

推荐环境与项目实验环境保持一致：

```bash
conda env create -f environment.yml
conda activate x_gaussian
```

如果 CUDA 扩展没有自动编译，可以手动安装：

```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

实验主要在 NVIDIA RTX 4090 24GB 上运行。不同 CUDA / PyTorch / 显卡架构可能需要重新编译 `submodules/` 中的 CUDA 扩展。

## 数据准备

默认实验数据路径：

```text
data/chest_50.pickle
```

该 pickle 需要包含 CT 体数据和投影视角相关信息。由于数据体积和版权原因，`data/` 不上传到 GitHub。

## 推荐配置

当前仓库只保留论文复现和关键消融配置：

```text
configs/baseline_chest_30k.yaml
configs/dgr_volume_chest_5k_v2.yaml
configs/dgr_volume_chest_30k_v3_128.yaml
configs/dgr_volume_chest_60k_v4_128_strong.yaml
configs/dgr_volume_chest_60k_v5_128_dense_init.yaml
configs/dgr_volume_chest_60k_v6_128_antiblock.yaml
configs/dgr_volume_chest_60k_v7_128_supersample.yaml
configs/dgr_volume_chest_60k_v8_128_reg.yaml
configs/acui_refinement_occ_chest_30k.yaml
configs/acui_refinement_occ_vol_v3_chest_30k.yaml
```

主实验使用：

```text
configs/dgr_volume_chest_60k_v8_128_reg.yaml
```

快速验证使用：

```text
configs/dgr_volume_chest_5k_v2.yaml
```

## 训练

快速 5k 验证：

```bash
python train.py --config configs/dgr_volume_chest_5k_v2.yaml
```

最终 60k 主实验：

```bash
python train.py --config configs/dgr_volume_chest_60k_v8_128_reg.yaml
```

基础对照实验：

```bash
python train.py --config configs/baseline_chest_30k.yaml
```

训练输出默认保存到 `output/`，该目录不会上传到 GitHub。

## 二维新视角图像

二维新视角图像是原始 X-Gaussian 的主要产出，本项目仍然保留。训练过程中会在配置指定的 `render_iterations` 上生成渲染结果；也可以在训练结束后单独运行：

```bash
python render.py \
  -m output/volume_main/chest_dgr_volume_60k_v8_128_reg \
  --iteration -1 \
  --skip_train
```

典型输出路径：

```text
output/volume_main/chest_dgr_volume_60k_v8_128_reg/test/ours_60000/renders/
output/volume_main/chest_dgr_volume_60k_v8_128_reg/test/ours_60000/gt/
```

## 三维 CT 体导出

最终推荐导出命令：

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
  --slicer_export \
  --out_dir output/volume_main/chest_dgr_volume_60k_v8_128_reg/ct_volume_dgr_ref128_median3_smooth015_final
```

说明：

- `--density_mapping_mode radiodensity` 使用 CT 密度分支导出体数据。
- `--mode dgr` 使用 DGR 风格 splatting。
- `--post_median_size 3` 用于抑制孤立噪声和孔洞。
- `--post_smooth_sigma 0.15` 是当前较稳定的轻量平滑参数。
- 不建议默认使用更强 gamma 或更强平滑，否则可能压掉细结构。

## 体重建评估

导出后可使用：

```bash
python evaluation/eval_volume.py \
  --pred output/volume_main/chest_dgr_volume_60k_v8_128_reg/ct_volume_dgr_ref128_median3_smooth015_final/recon_volume.npy \
  --gt_pickle data/chest_50.pickle \
  --out_dir output/volume_main/chest_dgr_volume_60k_v8_128_reg/ct_volume_dgr_ref128_median3_smooth015_final/eval \
  --label v8_128_reg_smooth015
```

评估输出包括：

- 整体 volume PSNR / MSE / MAE。
- 中央 axial / coronal / sagittal 切片 PSNR、SSIM。
- soft tissue / hard tissue 区域指标。
- GT、prediction、difference 的切片对比图。

## 3D Slicer 查看

使用导出目录中的 `.mhd/.raw` 或 Slicer display volume 文件导入 3D Slicer。建议优先在 Slicer 中调节窗宽窗位、阈值和体绘制 transfer function，而不是在导出阶段过度增强 gamma 或平滑。

## 实验结论

当前最稳定路线是：

```text
v8_128_reg training
    -> DGR radiodensity export
    -> median size 3
    -> smooth sigma 0.15
    -> Slicer display adjustment
```

v9 edge-aware low-memory 配置在实验中没有取得更好的三维重建质量，已不作为推荐路线保留。

## 致谢

本项目基于 X-Gaussian 和 3D Gaussian Splatting 相关实现进行扩展，重点面向稀疏 X 射线 CT 的三维结构重建、体数据导出和医学可视化分析。
