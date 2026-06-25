# Experiments

## Baseline: Chest ACUI

### Purpose

Establish a reproducible X-Gaussian baseline for later comparison against prior initialization, refinement, structure-aware loss, and downstream reconstruction.

### Command

```bash
python scripts/run_baseline.py --config configs/baseline_chest.yaml --gpu_id 0 --eval
```

### Expected Output

Root:

```text
output/baseline/chest_acui/
```

Important files:

```text
cfg_args
cameras.json
dataset_split.json
training_summary.json
metrics_history.jsonl
analysis/gaussian_stats_history.jsonl
progress_renders/test/iteration_*/
point_cloud/iteration_*/point_cloud.ply
```

### NVS Evaluation

If renders already exist under `test/ours_<iteration>/renders` and `test/ours_<iteration>/gt`:

```bash
python evaluation/eval_nvs.py --model_path output/baseline/chest_acui --iteration 20000
```

If renders do not exist yet, render first:

```bash
python render.py -m output/baseline/chest_acui -s data/chest_50.pickle --iteration 20000 --skip_train
python evaluation/eval_nvs.py --model_path output/baseline/chest_acui --iteration 20000
```

### Result Table

| Experiment | Iteration | PSNR | SSIM | Train Time | Inference FPS | Gaussian Count | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| chest_acui | 20000 | TBD | TBD | TBD | TBD | TBD | Run on cloud server |

## Prior Build: Chest GT Q98

### Purpose

Extract a structure prior from the GT CT volume in `data/chest_50.pickle` without modifying baseline training.

### Command

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

### Expected Output

```text
output/priors/chest_gt_q98/
  occupancy_mask.npy
  roi_mask.npy
  point_cloud.npy
  point_cloud.ply
  prior_data.npz
  prior_stats.json
  slices/
```

### Result Table

| Prior | Source | Threshold | Occupancy Ratio | ROI Ratio | Points | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| chest_gt_q98 | GT volume | 0.98 quantile | TBD | TBD | TBD | Run on cloud server |
| _local_smoke_chest_gt_q98 | GT volume | 0.98 quantile | 0.0200 | 0.0505 | 1000 | Local smoke test with `sample_stride=2`, `roi_dilation=1`, `max_points=1000` |

## Stage 2: Initialization Early Signal

### Purpose

Compare whether using the reconstruction prior only at Gaussian initialization improves early sparse-view X-ray NVS signal.

### Commands

```bash
python train.py --config configs/acui_chest_5k.yaml --eval --gpu_id 0
python train.py --config configs/prior_init_chest_5k.yaml --eval --gpu_id 0
python train.py --config configs/hybrid_init_chest_5k.yaml --eval --gpu_id 0
```

### Expected Outputs

```text
output/stage2/chest_acui_5k/
output/stage2/chest_prior_init_5k/
output/stage2/chest_hybrid_init_5k/
```

Important files per run:

```text
init/init_stats.json
metrics_history.jsonl
analysis/gaussian_stats_history.jsonl
progress_renders/test/iteration_5000/
test/ours_5000/nvs_metrics.json
```

### Result Table

| Experiment | Init Mode | Prior Points | ACUI Points | Iteration | PSNR | SSIM | Gaussian Count | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| chest_acui_5k | acui | 0 | 4096 | 5000 | 8.5194 | 0.4548 | 158916 | Baseline short run |
| chest_prior_init_5k | prior_init | 41992 | 0 | 5000 | 8.5113 | 0.4545 | 393157 | More Gaussians and slower, no early gain |
| chest_hybrid_init_5k | hybrid_init | 25000 | 4096 | 5000 | 8.5203 | 0.4547 | 274859 | Similar to ACUI, negligible PSNR gain |

### Stage 2 Analysis

At 5k iterations, prior-only initialization does not improve NVS metrics and increases Gaussian count from 158916 to 393157. Hybrid initialization is numerically closest to ACUI, with a negligible PSNR difference of about +0.001 dB and slightly lower SSIM.

The initial prior is valid as an initialization source, but init-only is not enough to produce a meaningful early-signal improvement. The next phase should use the prior during training refinement, especially for densification and pruning decisions.

## Stage 3: Prior-Guided Refinement

### Purpose

Test whether using the reconstruction prior during training improves early NVS signal compared with init-only prior usage.

### Comparison Groups

```text
baseline:          configs/acui_chest_5k.yaml
init-only:         configs/hybrid_init_chest_5k.yaml
refinement-only:   configs/acui_refinement_chest_5k.yaml
init+refinement:   configs/hybrid_refinement_chest_5k.yaml
```

### Commands

```bash
python train.py --config configs/acui_refinement_chest_5k.yaml --eval --gpu_id 0
python train.py --config configs/hybrid_refinement_chest_5k.yaml --eval --gpu_id 0
```

### Evaluation

```bash
python render.py -m output/stage3/chest_acui_refinement_5k -s data/chest_50.pickle --iteration -1 --skip_train
python evaluation/eval_nvs.py --model_path output/stage3/chest_acui_refinement_5k --iteration -1

python render.py -m output/stage3/chest_hybrid_refinement_5k -s data/chest_50.pickle --iteration -1 --skip_train
python evaluation/eval_nvs.py --model_path output/stage3/chest_hybrid_refinement_5k --iteration -1
```

### Result Table

| Experiment | Init Mode | Refinement | PSNR | SSIM | Gaussian Count | ROI Gaussians | Extra Prune Candidates | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| chest_acui_5k | acui | off | 8.5194 | 0.4548 | 158916 | N/A | N/A | Stage 2 baseline |
| chest_hybrid_init_5k | hybrid_init | off | 8.5203 | 0.4547 | 274859 | N/A | N/A | Stage 2 init-only |
| chest_acui_roi_densify_5k | acui | ROI densify only | TBD | TBD | TBD | TBD | TBD | New ablation |
| chest_acui_empty_prune_5k | acui | empty prune only | TBD | TBD | TBD | TBD | TBD | New ablation |
| chest_acui_refinement_5k | acui | on | 8.5223 | 0.4548 | 127415 | 39991 | 14210 | Best 5k tradeoff: slightly better metrics, fewer Gaussians |
| chest_hybrid_refinement_5k | hybrid_init | on | 8.5175 | 0.4547 | 256585 | 111892 | 18704 | Hybrid init plus refinement appears too strong |

### Current Refinement Settings

```text
roi_densify_weight = 2.0
empty_prune_opacity = 0.01
refinement_from_iter = 500
densification_interval = 200
```

### Stage 3 Analysis

The refinement-only setting is the best current direction. Compared with ACUI, it improves PSNR from 8.5194 to 8.5223, keeps SSIM essentially unchanged but slightly higher, and reduces Gaussian count from 158916 to 127415. This suggests the ROI-guided pruning/densification prior is useful mainly as a training-time allocation bias.

The init+refinement setting performs worse than both ACUI and refinement-only. Since hybrid initialization already adds 25000 prior points, applying ROI densification and empty pruning on top likely over-constrains the early Gaussian distribution. For the next ablation, keep ACUI initialization and tune refinement flags separately.

### Stage 3 Next Ablation

Run:

```bash
python train.py --config configs/acui_roi_densify_chest_5k.yaml --eval --gpu_id 0
python train.py --config configs/acui_empty_prune_chest_5k.yaml --eval --gpu_id 0
```

Decision rule:

- If ROI densify only improves PSNR/SSIM but increases Gaussian count, keep it and tune `roi_densify_weight`.
- If empty prune only keeps PSNR/SSIM and reduces Gaussian count, keep it as the efficiency component.
- If combined is best, keep `configs/acui_refinement_chest_5k.yaml` as the current full refinement prototype.
- If combined is worse than one component, disable the harmful component before moving to structure-aware loss.

## Stage 4: Sobel Edge Loss

### Purpose

Test whether structure-aware Sobel edge supervision improves sparse-view X-ray NVS, both without and with prior-guided refinement.

### Comparison Groups

```text
baseline:            configs/acui_chest_5k.yaml
refinement:          configs/acui_refinement_chest_5k.yaml
edge-only:           configs/acui_edge_chest_5k.yaml
refinement + edge:   configs/acui_refinement_edge_chest_5k.yaml
```

### Commands

```bash
python train.py --config configs/acui_edge_chest_5k.yaml --eval --gpu_id 0
python render.py -m output/stage4/chest_acui_edge_5k -s data/chest_50.pickle --iteration -1 --skip_train
python evaluation/eval_nvs.py --model_path output/stage4/chest_acui_edge_5k --iteration -1

python train.py --config configs/acui_refinement_edge_chest_5k.yaml --eval --gpu_id 0
python render.py -m output/stage4/chest_acui_refinement_edge_5k -s data/chest_50.pickle --iteration -1 --skip_train
python evaluation/eval_nvs.py --model_path output/stage4/chest_acui_refinement_edge_5k --iteration -1
```

### Result Table

| Experiment | Prior Refinement | Edge Loss | Edge Weight | PSNR | SSIM | Gaussian Count | Edge Loss Final | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| chest_acui_5k | off | off | 0.00 | 8.5194 | 0.4548 | 158916 | N/A | Baseline |
| chest_acui_refinement_5k | on | off | 0.00 | 8.5223 | 0.4548 | 127415 | N/A | Best Stage 3 |
| chest_acui_edge_5k | off | on | 0.05 | 8.5148 | 0.4548 | 166019 | 0.0113 | Edge-only hurts PSNR |
| chest_acui_refinement_edge_5k | on | on | 0.05 | 8.5179 | 0.4548 | 130565 | 0.0112 | Lower than refinement-only |
| chest_acui_refinement_edge001_5k | on | on | 0.01 | 8.5199 | 0.4548 | 129151 | 0.0117 | Better than 0.05 but still below refinement-only |

### Decision Rule

- If edge-only improves structure metrics or visual edges without hurting PSNR/SSIM, keep Sobel as a standalone structure term.
- If refinement + edge is best, it becomes the current full method.
- If edge loss hurts, reduce `edge_loss_weight` to `0.01` before discarding it.

### Stage 4 Analysis

With `edge_loss_weight = 0.05`, Sobel edge loss does not improve the current prototype. Edge-only reduces PSNR from 8.5194 to 8.5148 and increases Gaussian count. Refinement + edge reduces PSNR from 8.5223 to 8.5179 compared with refinement-only, although SSIM is marginally higher. The visual example does not show a clear structural improvement.

Reducing the edge weight to `0.01` improves over `0.05`, but it still does not outperform refinement-only. For the current prototype, Sobel edge loss should not be part of the best model. Keep it as an explored negative/neutral ablation and move to downstream reconstruction evaluation with `configs/acui_refinement_chest_5k.yaml`.

## Stage 4-B: Masked Sobel Edge Loss

### Purpose

Retest structure-aware edge supervision with a mask over strong target-edge regions instead of applying Sobel loss to the whole X-ray projection.

### Commands

```bash
python train.py --config configs/acui_refinement_masked_edge001_chest_5k.yaml --eval --gpu_id 0
python render.py -m output/stage4/chest_acui_refinement_masked_edge001_5k -s data/chest_50.pickle --iteration -1 --skip_train
python evaluation/eval_nvs.py --model_path output/stage4/chest_acui_refinement_masked_edge001_5k --iteration -1

python train.py --config configs/acui_refinement_masked_edge005_chest_5k.yaml --eval --gpu_id 0
python render.py -m output/stage4/chest_acui_refinement_masked_edge005_5k -s data/chest_50.pickle --iteration -1 --skip_train
python evaluation/eval_nvs.py --model_path output/stage4/chest_acui_refinement_masked_edge005_5k --iteration -1
```

### Result Table

| Experiment | Edge Mask | Edge Weight | PSNR | SSIM | Gaussian Count | Edge Loss Final | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| chest_acui_refinement_5k | none | 0.00 | 8.5223 | 0.4548 | 127415 | N/A | Best Stage 3 |
| chest_acui_refinement_edge001_5k | none | 0.01 | 8.5199 | 0.4548 | 129151 | 0.0117 | Full-image Sobel |
| chest_acui_refinement_masked_edge001_5k | target_edge | 0.01 | 8.5193 | 0.4548 | 126732 | 0.0119 | Lower than refinement-only |
| chest_acui_refinement_masked_edge005_5k | target_edge | 0.05 | 8.5137 | 0.4548 | 132041 | 0.0119 | High weight hurts |

### Decision Rule

- If masked edge improves over refinement-only, keep it as the structure-aware loss.
- If masked edge only improves SSIM but hurts PSNR slightly, keep it as optional and validate with downstream reconstruction.
- If masked edge still hurts both quality and efficiency, Stage 4 remains a negative ablation and the full prototype stays prior-refinement-only.

### Stage 4-B Analysis

Masked Sobel edge loss is more targeted than full-image Sobel, but it still does not outperform the prior-refinement-only model. The low-weight masked version (`0.01`) gives PSNR 8.5193 compared with 8.5223 for refinement-only. The higher-weight masked version (`0.05`) hurts more. The visual example does not show a clear structure improvement.

Stage 4 should be recorded as a negative/neutral ablation: naive Sobel and target-edge masked Sobel are not reliable improvements for the current X-ray NVS setup. The current best prototype remains `configs/acui_refinement_chest_5k.yaml`.

## Stage 5: Downstream Volume Evaluation

### Purpose

Evaluate whether the best prior-guided refinement model produces a better 3D Gaussian-derived CT volume than the ACUI baseline.

### Comparison Groups

```text
baseline: output/stage2/chest_acui_5k
best:     output/stage3/chest_acui_refinement_5k
```

### Command

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

### Outputs

```text
output/stage2/chest_acui_5k/ct_volume/*/eval/volume_metrics.json
output/stage3/chest_acui_refinement_5k/ct_volume/*/eval/volume_metrics.json
```

### Result Table

| Experiment | Volume PSNR | Volume MAE | Axial SSIM | Coronal SSIM | Sagittal SSIM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| chest_acui_5k | 10.3803 | 0.2252 | 0.0003 | 0.0009 | 0.0039 | Baseline Gaussian-derived volume |
| chest_acui_refinement_5k | 10.3945 | 0.2248 | 0.0004 | 0.0010 | 0.0042 | Best prior-refinement method |

### Decision Rule

- If the best method improves volume PSNR or central-slice SSIM, it supports the downstream reconstruction claim.
- If NVS improves but volume metrics do not, the method improves 2D rendering efficiency but does not yet prove downstream reconstruction benefit.
- Slice comparisons should be inspected because volume PSNR can be sensitive to intensity normalization and sparse Gaussian export artifacts.

### Stage 5 Analysis

The prior-refinement model slightly improves all measured Gaussian-derived volume metrics while using fewer Gaussians. Volume PSNR improves from 10.3803 to 10.3945 and volume MAE decreases from 0.2252 to 0.2248. Central slice SSIM also improves on axial, coronal, and sagittal slices.

The absolute slice SSIM values are very low, and visual slices show that direct Gaussian-to-volume export is still a crude reconstruction proxy. This result supports a weak downstream benefit claim, but a stronger final evaluation should later use rendered novel views with a standard reconstruction method such as FDK/SART if available.
