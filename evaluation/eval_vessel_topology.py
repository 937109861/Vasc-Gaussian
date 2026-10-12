import argparse
import csv
import json
import os
import pickle
from pathlib import Path

import numpy as np
from scipy import ndimage


def normalize_volume(volume):
    volume = np.asarray(volume, dtype=np.float32)
    vmin = float(volume.min())
    vmax = float(volume.max())
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.float32), vmin, vmax
    return (volume - vmin) / (vmax - vmin), vmin, vmax


def load_gt_volume(source_path):
    with open(source_path, "rb") as handle:
        data = pickle.load(handle)
    if "image" not in data:
        raise KeyError(f"GT pickle does not contain image volume: {source_path}")
    spacing = np.asarray(data.get("dVoxel", [1.0, 1.0, 1.0]), dtype=np.float32)
    return np.asarray(data["image"], dtype=np.float32), spacing


def load_pred_volume(pred_path):
    payload = np.load(pred_path, allow_pickle=True)
    if isinstance(payload, np.lib.npyio.NpzFile):
        for key in ("recon_volume", "volume", "prediction", "pred", "data", "soft_prior"):
            if key in payload:
                return np.asarray(payload[key], dtype=np.float32)
        raise KeyError(f"No volume-like key found in npz file: {pred_path}")
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        item = payload.item() if payload.shape == () else payload
        if isinstance(item, dict):
            for key in ("recon_volume", "volume", "prediction", "pred", "data"):
                if key in item:
                    return np.asarray(item[key], dtype=np.float32)
            raise KeyError(f"No volume-like key found in object npy file: {pred_path}")
        return np.asarray(item, dtype=np.float32)
    return np.asarray(payload, dtype=np.float32)


def resolve_threshold(volume, explicit_threshold, gt_positive_quantile, epsilon=1e-6):
    if explicit_threshold is not None:
        return float(explicit_threshold)
    positives = volume[volume > float(epsilon)]
    if positives.size == 0:
        return float("inf")
    return float(np.quantile(positives, float(gt_positive_quantile)))


def make_mask(volume, threshold):
    if not np.isfinite(threshold):
        return np.zeros_like(volume, dtype=bool)
    return np.asarray(volume >= float(threshold), dtype=bool)


def binary_dice(pred_mask, gt_mask):
    pred_count = int(pred_mask.sum())
    gt_count = int(gt_mask.sum())
    tp = int(np.logical_and(pred_mask, gt_mask).sum())
    denom = pred_count + gt_count
    return float(2.0 * tp / denom) if denom else 1.0


def binary_precision_recall(pred_mask, gt_mask):
    tp = int(np.logical_and(pred_mask, gt_mask).sum())
    fp = int(np.logical_and(pred_mask, ~gt_mask).sum())
    fn = int(np.logical_and(~pred_mask, gt_mask).sum())
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    return precision, recall


def skeletonize_3d(mask):
    try:
        from skimage.morphology import skeletonize
    except ImportError as error:
        raise ImportError(
            "scikit-image is required for clDice/CL-Recall. Install it with: "
            "pip install scikit-image"
        ) from error

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    try:
        skeleton = skeletonize(mask, method="lee")
    except TypeError:
        skeleton = skeletonize(mask)
    return np.asarray(skeleton, dtype=bool)


def centerline_metrics(pred_mask, gt_mask):
    pred_skel = skeletonize_3d(pred_mask)
    gt_skel = skeletonize_3d(gt_mask)

    pred_skel_count = int(pred_skel.sum())
    gt_skel_count = int(gt_skel.sum())

    # clDice follows the topology-preserving clDice definition:
    # Tprec = |S_pred intersect GT| / |S_pred|
    # Tsens = |S_gt intersect Pred| / |S_gt|
    tprec = (
        float(np.logical_and(pred_skel, gt_mask).sum() / pred_skel_count)
        if pred_skel_count
        else 0.0
    )
    tsens = (
        float(np.logical_and(gt_skel, pred_mask).sum() / gt_skel_count)
        if gt_skel_count
        else 0.0
    )
    cldice = float(2.0 * tprec * tsens / (tprec + tsens)) if tprec + tsens else 0.0
    return {
        "cldice": cldice,
        "centerline_precision": tprec,
        "centerline_recall": tsens,
        "pred_centerline_voxels": pred_skel_count,
        "gt_centerline_voxels": gt_skel_count,
    }


def surface_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask.copy()
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    surface = np.logical_and(mask, ~eroded)
    return surface if surface.any() else mask.copy()


def hd95(pred_mask, gt_mask, spacing):
    pred_surface = surface_mask(pred_mask)
    gt_surface = surface_mask(gt_mask)
    if not pred_surface.any() and not gt_surface.any():
        return 0.0
    if not pred_surface.any() or not gt_surface.any():
        return float("inf")

    spacing = tuple(float(x) for x in np.asarray(spacing, dtype=np.float32).reshape(-1)[:3])
    dt_to_gt = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)
    pred_to_gt = dt_to_gt[pred_surface]
    del dt_to_gt

    dt_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    gt_to_pred = dt_to_pred[gt_surface]
    del dt_to_pred

    distances = np.concatenate([pred_to_gt.astype(np.float32), gt_to_pred.astype(np.float32)])
    if distances.size == 0:
        return float("inf")
    return float(np.percentile(distances, 95.0))


def evaluate(pred_path, gt_pickle, out_dir, label, pred_threshold, gt_threshold, gt_quantile, save_masks):
    pred_raw = load_pred_volume(pred_path)
    gt_raw, spacing = load_gt_volume(gt_pickle)
    if pred_raw.shape != gt_raw.shape:
        raise ValueError(f"Volume shape mismatch: pred {pred_raw.shape}, gt {gt_raw.shape}")

    pred, pred_min, pred_max = normalize_volume(pred_raw)
    gt, gt_min, gt_max = normalize_volume(gt_raw)

    gt_thr = resolve_threshold(gt, gt_threshold, gt_quantile)
    pred_thr = float(pred_threshold) if pred_threshold is not None else gt_thr
    pred_mask = make_mask(pred, pred_thr)
    gt_mask = make_mask(gt, gt_thr)

    precision, recall = binary_precision_recall(pred_mask, gt_mask)
    centerline = centerline_metrics(pred_mask, gt_mask)
    metrics = {
        "label": label,
        "prediction": str(pred_path),
        "gt": str(gt_pickle),
        "shape": list(gt.shape),
        "prediction_raw_min": pred_min,
        "prediction_raw_max": pred_max,
        "gt_raw_min": gt_min,
        "gt_raw_max": gt_max,
        "pred_threshold": pred_thr,
        "gt_threshold": gt_thr,
        "gt_positive_quantile": float(gt_quantile),
        "pred_voxels": int(pred_mask.sum()),
        "gt_voxels": int(gt_mask.sum()),
        "dice": binary_dice(pred_mask, gt_mask),
        "precision": precision,
        "recall": recall,
        "cldice": centerline["cldice"],
        "centerline_precision": centerline["centerline_precision"],
        "centerline_recall": centerline["centerline_recall"],
        "pred_centerline_voxels": centerline["pred_centerline_voxels"],
        "gt_centerline_voxels": centerline["gt_centerline_voxels"],
        "hd95": hd95(pred_mask, gt_mask, spacing),
        "spacing": [float(x) for x in spacing.reshape(-1)[:3]],
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_masks:
        np.save(out_dir / "pred_mask.npy", pred_mask.astype(np.uint8))
        np.save(out_dir / "gt_mask.npy", gt_mask.astype(np.uint8))

    with open(out_dir / "vessel_topology_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    with open(out_dir / "vessel_topology_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        fields = [
            "label",
            "dice",
            "cldice",
            "centerline_recall",
            "hd95",
            "precision",
            "recall",
            "pred_voxels",
            "gt_voxels",
            "pred_threshold",
            "gt_threshold",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: metrics[field] for field in fields})

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate vessel topology metrics: Dice, clDice, centerline recall, and HD95."
    )
    parser.add_argument("--pred", required=True, help="Predicted .npy/.npz volume or vessel volume.")
    parser.add_argument("--gt_pickle", required=True, help="X-Gaussian pickle containing GT image volume.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--label", default="vessel")
    parser.add_argument(
        "--pred_threshold",
        type=float,
        default=None,
        help="Threshold on normalized prediction. Defaults to the resolved GT threshold.",
    )
    parser.add_argument(
        "--gt_threshold",
        type=float,
        default=None,
        help="Threshold on normalized GT. Overrides --gt_positive_quantile.",
    )
    parser.add_argument(
        "--gt_positive_quantile",
        type=float,
        default=0.05,
        help="GT positive-value quantile used as threshold; default matches eval_volume.py low-threshold Dice.",
    )
    parser.add_argument("--save_masks", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = evaluate(
        pred_path=args.pred,
        gt_pickle=args.gt_pickle,
        out_dir=args.out_dir,
        label=args.label,
        pred_threshold=args.pred_threshold,
        gt_threshold=args.gt_threshold,
        gt_quantile=args.gt_positive_quantile,
        save_masks=args.save_masks,
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {Path(args.out_dir) / 'vessel_topology_metrics.json'}")
    print(f"Saved: {Path(args.out_dir) / 'vessel_topology_metrics.csv'}")


if __name__ == "__main__":
    main()
