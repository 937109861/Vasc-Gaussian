# Development Log

## 2026-05-11: Phase 0 Planning and Baseline Platform Scaffold

### Changed

- Added project planning documentation.
- Added experiment tracking documentation.
- Added a cloud-friendly baseline runner.
- Added a lightweight NVS evaluation script.
- Added a baseline config copy under `configs/` for the new research workflow.

### Why

The project needs a stable baseline before prior-guided initialization, refinement, structure-aware loss, or downstream reconstruction are added. The current repository already contains some analysis helpers and previous phase-2 experiments, so Phase 0 focuses on making the baseline command, output paths, and metrics reproducible without changing the training algorithm.

### How To Run

Dry run on any machine:

```bash
python scripts/run_baseline.py --dry_run
```

Run baseline training on a CUDA cloud server:

```bash
python scripts/run_baseline.py --config configs/baseline_chest.yaml --gpu_id 0 --eval
```

Evaluate an existing render directory:

```bash
python evaluation/eval_nvs.py --model_path output/baseline/chest_acui --iteration 20000
```

### Current Best Baseline

The current reference baseline is ACUI initialization with the chest sparse-view dataset:

```bash
python train.py --config configs/baseline_chest.yaml --eval --gpu_id 0
```

### Failed Attempts

None recorded yet.

## 2026-05-11: Phase 1 Independent Prior Builder

### Changed

- Added `priors/volume_prior.py` for volume normalization, thresholding, occupancy extraction, ROI dilation, point-cloud sampling, and output serialization.
- Added `scripts/build_prior.py` as a cloud-friendly command line entry.
- Kept `train.py`, `scene/__init__.py`, and Gaussian training logic unchanged.

### Why

The next research step needs reconstruction priors, but the baseline must remain independent. Building priors as a separate preprocessing step gives reproducible intermediate artifacts that can later be used by initialization, refinement, and downstream reconstruction modules.

### How To Run

Build a GT-volume prior from the existing chest pickle:

```bash
python scripts/build_prior.py \
  --input data/chest_50.pickle \
  --source_type gt \
  --out_dir output/priors/chest_gt_q98 \
  --threshold_mode quantile \
  --threshold_value 0.98 \
  --sample_stride 1 \
  --max_points 50000 \
  --roi_dilation 2
```

For a standalone FDK or NAF `.npy` volume, provide the original pickle for geometry:

```bash
python scripts/build_prior.py \
  --input path/to/fdk_volume.npy \
  --source_type fdk \
  --geometry_pickle data/chest_50.pickle \
  --out_dir output/priors/chest_fdk_q98
```

### Output Format

```text
occupancy_mask.npy
roi_mask.npy
point_cloud.npy
point_cloud.ply
prior_data.npz
prior_stats.json
slices/*_volume.png
slices/*_occupancy.png
slices/*_roi.png
```

### Verification

Check that `prior_stats.json` reports nonzero `occupancy_voxels`, `roi_voxels`, and `num_points`, then inspect the slice PNGs under `slices/`.

## 2026-05-11: Phase 2 Initialization Modes

### Changed

- Added `prior_path` to model parameters.
- Updated X-ray scene loading to pass `prior_path`.
- Updated X-ray initialization to support:
  - `acui`
  - `prior_init`
  - `hybrid_init`
- Kept backward-compatible aliases:
  - `point_only` -> `prior_init`
  - `hybrid` -> `hybrid_init`
- Added three 5k short-training configs for early-signal comparison.

### Why

Prior construction is now independent, so initialization should consume the saved prior artifact instead of rebuilding prior points inside the data loader. This makes experiments reproducible and allows the same prior to be reused across ablations.

### How To Run

Build prior first:

```bash
python scripts/build_prior.py \
  --input data/chest_50.pickle \
  --source_type gt \
  --out_dir output/priors/chest_gt_q98 \
  --threshold_mode quantile \
  --threshold_value 0.98 \
  --sample_stride 1 \
  --max_points 50000 \
  --roi_dilation 2
```

Run 5k initialization comparison:

```bash
python train.py --config configs/acui_chest_5k.yaml --eval --gpu_id 0
python train.py --config configs/prior_init_chest_5k.yaml --eval --gpu_id 0
python train.py --config configs/hybrid_init_chest_5k.yaml --eval --gpu_id 0
```

Render and evaluate each run:

```bash
python render.py -m output/stage2/chest_acui_5k -s data/chest_50.pickle --iteration -1 --skip_train
python evaluation/eval_nvs.py --model_path output/stage2/chest_acui_5k --iteration -1

python render.py -m output/stage2/chest_prior_init_5k -s data/chest_50.pickle --iteration -1 --skip_train
python evaluation/eval_nvs.py --model_path output/stage2/chest_prior_init_5k --iteration -1

python render.py -m output/stage2/chest_hybrid_init_5k -s data/chest_50.pickle --iteration -1 --skip_train
python evaluation/eval_nvs.py --model_path output/stage2/chest_hybrid_init_5k --iteration -1
```

### Verification

Check each run's `init/init_stats.json`:

- ACUI should report `num_prior_points = 0`.
- Prior init should report only prior points.
- Hybrid init should report both ACUI and prior points.

## 2026-05-11: Phase 3 Prior-Guided Training Refinement

### Changed

- Added `refinement/prior_refinement.py`.
- Added optional `grad_multipliers` and `extra_prune_mask` arguments to `GaussianModel_Xray.densify_and_prune`.
- Added training-loop integration for prior-guided refinement.
- Added refinement config flags:
  - `use_prior_refinement`
  - `refinement_prior_path`
  - `refinement_from_iter`
  - `roi_densify_enabled`
  - `roi_densify_weight`
  - `empty_prune_enabled`
  - `empty_prune_opacity`
- Added stage-3 configs:
  - `configs/acui_refinement_chest_5k.yaml`
  - `configs/hybrid_refinement_chest_5k.yaml`

### Why

Stage 2 showed that init-only prior usage does not provide meaningful early NVS gains. The prior should influence training dynamics, especially densification and pruning, while keeping CUDA and baseline behavior unchanged.

### How To Run

Use the existing Stage 2 comparison as baseline and init-only:

```bash
python train.py --config configs/acui_chest_5k.yaml --eval --gpu_id 0
python train.py --config configs/hybrid_init_chest_5k.yaml --eval --gpu_id 0
```

Run refinement-only and init+refinement:

```bash
python train.py --config configs/acui_refinement_chest_5k.yaml --eval --gpu_id 0
python train.py --config configs/hybrid_refinement_chest_5k.yaml --eval --gpu_id 0
```

Render and evaluate:

```bash
python render.py -m output/stage3/chest_acui_refinement_5k -s data/chest_50.pickle --iteration -1 --skip_train
python evaluation/eval_nvs.py --model_path output/stage3/chest_acui_refinement_5k --iteration -1

python render.py -m output/stage3/chest_hybrid_refinement_5k -s data/chest_50.pickle --iteration -1 --skip_train
python evaluation/eval_nvs.py --model_path output/stage3/chest_hybrid_refinement_5k --iteration -1
```

### Verification

Check:

```text
refinement/prior_refinement.json
refinement/refinement_history.jsonl
metrics_history.jsonl
analysis/iteration_5000/gaussian_stats.json
test/ours_5000/nvs_metrics.json
```

If refinement harms metrics or prunes too aggressively, set `empty_prune_enabled: false` first, then tune `roi_densify_weight`.

## 2026-05-11: Phase 3 Refinement Ablation Setup

### Changed

- Added `configs/acui_roi_densify_chest_5k.yaml`.
- Added `configs/acui_empty_prune_chest_5k.yaml`.
- Added `evaluation/summarize_stage_results.py`.

### Why

The best current result is ACUI initialization plus prior refinement. The next question is whether the gain comes from ROI densification, empty-region pruning, or their combination. These configs isolate each component while keeping the same ACUI initialization.

### How To Run

```bash
python train.py --config configs/acui_roi_densify_chest_5k.yaml --eval --gpu_id 0
python render.py -m output/stage3/chest_acui_roi_densify_5k -s data/chest_50.pickle --iteration -1 --skip_train
python evaluation/eval_nvs.py --model_path output/stage3/chest_acui_roi_densify_5k --iteration -1

python train.py --config configs/acui_empty_prune_chest_5k.yaml --eval --gpu_id 0
python render.py -m output/stage3/chest_acui_empty_prune_5k -s data/chest_50.pickle --iteration -1 --skip_train
python evaluation/eval_nvs.py --model_path output/stage3/chest_acui_empty_prune_5k --iteration -1
```

Summarize all relevant Stage 3 runs:

```bash
python evaluation/summarize_stage_results.py \
  --model_paths \
  output/stage2/chest_acui_5k \
  output/stage3/chest_acui_roi_densify_5k \
  output/stage3/chest_acui_empty_prune_5k \
  output/stage3/chest_acui_refinement_5k \
  --out_csv output/stage3/refinement_ablation_summary.csv \
  --out_json output/stage3/refinement_ablation_summary.json
```

## 2026-05-12: Phase 4 Sobel Structure Loss

### Changed

- Added `losses/structure_losses.py`.
- Added `use_edge_loss` and `edge_loss_weight` to optimization parameters.
- Added optional Sobel edge loss to `train.py`.
- Added `loss_history.jsonl` logging for L1, SSIM term, base loss, edge loss, and total loss.
- Added Stage 4 configs:
  - `configs/acui_edge_chest_5k.yaml`
  - `configs/acui_refinement_edge_chest_5k.yaml`

### Why

The project goal includes structure-aware optimization. Sobel edge loss is the smallest useful prototype because X-ray reconstruction quality depends heavily on edge and structure consistency, and the loss can be added without changing renderer or CUDA kernels.

### How To Run

```bash
python train.py --config configs/acui_edge_chest_5k.yaml --eval --gpu_id 0
python render.py -m output/stage4/chest_acui_edge_5k -s data/chest_50.pickle --iteration -1 --skip_train
python evaluation/eval_nvs.py --model_path output/stage4/chest_acui_edge_5k --iteration -1

python train.py --config configs/acui_refinement_edge_chest_5k.yaml --eval --gpu_id 0
python render.py -m output/stage4/chest_acui_refinement_edge_5k -s data/chest_50.pickle --iteration -1 --skip_train
python evaluation/eval_nvs.py --model_path output/stage4/chest_acui_refinement_edge_5k --iteration -1
```

### Verification

Check:

```text
loss_history.jsonl
test/ours_5000/nvs_metrics.json
analysis/iteration_5000/gaussian_stats.json
```

If edge loss hurts PSNR/SSIM or makes images too sharp/noisy, reduce `edge_loss_weight` before combining it with downstream reconstruction.

## 2026-05-12: Phase 4-B Masked Structure Loss

### Changed

- Added masked Sobel edge loss using strong target-edge regions.
- Added edge mask config fields:
  - `edge_loss_mask_mode`
  - `edge_mask_quantile`
  - `edge_mask_dilation`
- Added masked edge configs:
  - `configs/acui_refinement_masked_edge001_chest_5k.yaml`
  - `configs/acui_refinement_masked_edge005_chest_5k.yaml`

### Why

Full-image Sobel loss hurt PSNR and did not improve visual structure. Masking the loss to strong target-edge regions avoids pushing background and low-value projection regions, making the structure-aware term more consistent with the reconstruction-guided refinement direction.

### How To Run

```bash
python train.py --config configs/acui_refinement_masked_edge001_chest_5k.yaml --eval --gpu_id 0
python render.py -m output/stage4/chest_acui_refinement_masked_edge001_5k -s data/chest_50.pickle --iteration -1 --skip_train
python evaluation/eval_nvs.py --model_path output/stage4/chest_acui_refinement_masked_edge001_5k --iteration -1

python train.py --config configs/acui_refinement_masked_edge005_chest_5k.yaml --eval --gpu_id 0
python render.py -m output/stage4/chest_acui_refinement_masked_edge005_5k -s data/chest_50.pickle --iteration -1 --skip_train
python evaluation/eval_nvs.py --model_path output/stage4/chest_acui_refinement_masked_edge005_5k --iteration -1
```

## 2026-05-12: Phase 5 Volume Evaluation Scaffold

### Changed

- Added `evaluation/eval_volume.py`.
- Added `scripts/run_stage5_volume_eval.py`.

### Why

The final project goal is not only 2D NVS, but improved downstream 3D volume reconstruction. The minimum viable evaluation is to export Gaussian-derived volumes for baseline and the best prior-refinement model, then compare them with the GT CT volume using volume PSNR, MAE, MSE, central-slice SSIM, and slice visualizations.

### How To Run

```bash
python scripts/run_stage5_volume_eval.py \
  --source_path data/chest_50.pickle \
  --baseline_model output/stage2/chest_acui_5k \
  --best_model output/stage3/chest_acui_refinement_5k \
  --iteration -1 \
  --mode gaussian \
  --density opacity \
  --cutoff 3.0 \
  --max_gaussians 0
```

Outputs are written under:

```text
output/stage2/chest_acui_5k/ct_volume/
output/stage3/chest_acui_refinement_5k/ct_volume/
```

Each evaluated volume has:

```text
recon_volume.npy
export_stats.json
eval/volume_metrics.json
eval/slice_metrics.csv
eval/slices/*_gt.png
eval/slices/*_pred.png
eval/slices/*_absdiff.png
```
