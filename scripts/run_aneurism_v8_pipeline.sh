#!/usr/bin/env bash
set -euo pipefail

# Aneurism v8 pipeline:
# 1. train 2D X-ray Gaussian model with DGR volume loss
# 2. render held-out validation views
# 3. compute 2D render metrics
# 4. export 3D CT volume with DGR splatting
# 5. evaluate exported volume against the pickle GT

PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG="configs/dgr_volume_aneurism_60k_v8_128_reg.yaml"
SOURCE_PATH="data/aneurism_50.pickle"
MODEL_PATH="output/volume_main/aneurism_dgr_volume_60k_v8_128_reg"
EXPORT_DIR="${MODEL_PATH}/ct_volume_dgr_ref128_median3_smooth015_final"

echo "[1/5] Training aneurism v8"
"${PYTHON_BIN}" train.py --config "${CONFIG}"

echo "[2/5] Rendering test views"
"${PYTHON_BIN}" render.py \
  -m "${MODEL_PATH}" \
  --iteration -1 \
  --skip_train

echo "[3/5] Evaluating 2D render metrics"
"${PYTHON_BIN}" metrics.py \
  -m "${MODEL_PATH}"

echo "[4/5] Exporting 3D CT volume"
"${PYTHON_BIN}" export_ct_volume.py \
  -m "${MODEL_PATH}" \
  -s "${SOURCE_PATH}" \
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
  --out_dir "${EXPORT_DIR}"

echo "[5/5] Evaluating 3D CT volume"
"${PYTHON_BIN}" evaluation/eval_volume.py \
  --pred "${EXPORT_DIR}/recon_volume.npy" \
  --gt_pickle "${SOURCE_PATH}" \
  --out_dir "${EXPORT_DIR}/eval" \
  --label aneurism_v8_128_reg_smooth015

echo "Done."
echo "Model: ${MODEL_PATH}"
echo "2D renders: ${MODEL_PATH}/test/ours_60000/renders"
echo "3D volume: ${EXPORT_DIR}"
