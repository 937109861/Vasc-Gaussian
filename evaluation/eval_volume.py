import argparse
import csv
import json
import os
import pickle
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


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
    return np.asarray(data["image"], dtype=np.float32)


def load_pred_volume(pred_path):
    try:
        payload = np.load(pred_path)
    except ValueError as error:
        if "allow_pickle=False" not in str(error):
            raise
        payload = np.load(pred_path, allow_pickle=True)

    if isinstance(payload, np.lib.npyio.NpzFile):
        for key in ("recon_volume", "volume", "prediction", "pred", "data"):
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


def psnr(prediction, target, max_value=1.0):
    mse = float(np.mean((prediction - target) ** 2))
    if mse <= 1e-12:
        return float("inf"), mse
    return float(20.0 * np.log10(max_value) - 10.0 * np.log10(mse)), mse


def masked_volume_metrics(prediction, target, mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != prediction.shape or not mask.any():
        return None
    pred = prediction[mask]
    gt = target[mask]
    region_psnr, region_mse = psnr(pred, gt)
    return {
        "count": int(mask.sum()),
        "psnr": region_psnr,
        "mse": region_mse,
        "mae": float(np.mean(np.abs(pred - gt))),
        "pred_mean": float(pred.mean()),
        "gt_mean": float(gt.mean()),
    }


def binary_overlap_metrics(prediction, target, threshold):
    pred_mask = prediction >= float(threshold)
    target_mask = target >= float(threshold)
    true_positive = int(np.logical_and(pred_mask, target_mask).sum())
    false_positive = int(np.logical_and(pred_mask, ~target_mask).sum())
    false_negative = int(np.logical_and(~pred_mask, target_mask).sum())
    pred_count = int(pred_mask.sum())
    target_count = int(target_mask.sum())
    dice_denom = 2 * true_positive + false_positive + false_negative
    return {
        "threshold": float(threshold),
        "prediction_count": pred_count,
        "target_count": target_count,
        "dice": float(2 * true_positive / dice_denom) if dice_denom else 1.0,
        "precision": float(true_positive / (true_positive + false_positive))
        if true_positive + false_positive
        else 0.0,
        "recall": float(true_positive / (true_positive + false_negative))
        if true_positive + false_negative
        else 0.0,
    }


def tissue_region_metrics(
    prediction,
    target,
    soft_range=(0.05, 0.75),
    hard_min=0.90,
    foreground_epsilon=1e-6,
):
    foreground_mask = target > float(foreground_epsilon)
    values = target[foreground_mask]
    if values.size == 0:
        return {
            "foreground": None,
            "soft_tissue": None,
            "hard_tissue": None,
            "thresholds": {
                "foreground_epsilon": float(foreground_epsilon),
                "soft_low": None,
                "soft_high": None,
                "hard_threshold": None,
            },
        }

    soft_low = float(np.quantile(values, soft_range[0]))
    soft_high = float(np.quantile(values, soft_range[1]))
    hard_threshold = float(np.quantile(values, hard_min))
    soft_mask = foreground_mask & (target >= soft_low) & (target <= soft_high)
    hard_mask = foreground_mask & (target >= hard_threshold)
    return {
        "foreground": masked_volume_metrics(prediction, target, foreground_mask),
        "soft_tissue": masked_volume_metrics(prediction, target, soft_mask),
        "hard_tissue": masked_volume_metrics(prediction, target, hard_mask),
        "thresholds": {
            "foreground_epsilon": float(foreground_epsilon),
            "soft_low": soft_low,
            "soft_high": soft_high,
            "hard_threshold": hard_threshold,
        },
    }


def load_prior_masks(prior_path):
    payload = np.load(prior_path)
    roi_mask = payload["roi_mask"].astype(bool) if "roi_mask" in payload else None
    occupancy_mask = payload["occupancy_mask"].astype(bool) if "occupancy_mask" in payload else None
    return roi_mask, occupancy_mask


def ssim_2d(image_a, image_b):
    image_a = image_a.astype(np.float32, copy=False)
    image_b = image_b.astype(np.float32, copy=False)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_a = float(image_a.mean())
    mu_b = float(image_b.mean())
    var_a = float(image_a.var())
    var_b = float(image_b.var())
    cov = float(((image_a - mu_a) * (image_b - mu_b)).mean())
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)))


def central_slice_metrics(prediction, target):
    mids = [size // 2 for size in target.shape]
    slices = {
        "axial_z": (slice(None), slice(None), mids[2]),
        "coronal_y": (slice(None), mids[1], slice(None)),
        "sagittal_x": (mids[0], slice(None), slice(None)),
    }
    rows = {}
    for name, index in slices.items():
        pred_slice = prediction[index]
        gt_slice = target[index]
        slice_psnr, slice_mse = psnr(pred_slice, gt_slice)
        rows[name] = {
            "psnr": slice_psnr,
            "mse": slice_mse,
            "mae": float(np.mean(np.abs(pred_slice - gt_slice))),
            "ssim": ssim_2d(pred_slice, gt_slice),
        }
    return rows, slices


def save_image(image, path):
    image = np.asarray(image, dtype=np.float32)
    image = np.clip(image, 0.0, 1.0)
    imageio.imwrite(path, (image * 255).astype(np.uint8))


def save_slice_comparisons(prediction, target, slices, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, index in slices.items():
        pred_slice = prediction[index]
        gt_slice = target[index]
        diff = np.abs(pred_slice - gt_slice)
        if diff.max() > diff.min():
            diff = (diff - diff.min()) / (diff.max() - diff.min())
        save_image(gt_slice, out_dir / f"{name}_gt.png")
        save_image(pred_slice, out_dir / f"{name}_pred.png")
        save_image(diff, out_dir / f"{name}_absdiff.png")


def evaluate_volume(pred_path, gt_path, out_dir, label, prior_path=None):
    prediction_raw = load_pred_volume(pred_path)
    target_raw = load_gt_volume(gt_path)
    if prediction_raw.shape != target_raw.shape:
        raise ValueError(f"Volume shape mismatch: pred {prediction_raw.shape}, gt {target_raw.shape}")

    prediction, pred_min, pred_max = normalize_volume(prediction_raw)
    target, gt_min, gt_max = normalize_volume(target_raw)
    volume_psnr, volume_mse = psnr(prediction, target)
    slice_rows, slices = central_slice_metrics(prediction, target)
    save_slice_comparisons(prediction, target, slices, Path(out_dir) / "slices")
    region_metrics = {}
    region_metrics["tissue"] = tissue_region_metrics(prediction, target)
    positive_values = target[target > 1e-6]
    if positive_values.size > 0:
        overlap_thresholds = sorted(
            {
                float(np.quantile(positive_values, quantile))
                for quantile in (0.05, 0.50, 0.90)
            }
        )
        region_metrics["overlap"] = [
            binary_overlap_metrics(prediction, target, threshold)
            for threshold in overlap_thresholds
        ]
    if prior_path:
        roi_mask, occupancy_mask = load_prior_masks(prior_path)
        if roi_mask is not None:
            region_metrics["roi"] = masked_volume_metrics(prediction, target, roi_mask)
            region_metrics["outside_roi"] = masked_volume_metrics(prediction, target, ~roi_mask)
        if occupancy_mask is not None:
            region_metrics["occupancy"] = masked_volume_metrics(prediction, target, occupancy_mask)
            region_metrics["empty_occupancy"] = masked_volume_metrics(prediction, target, ~occupancy_mask)

    summary = {
        "label": label,
        "prediction": str(pred_path),
        "gt": str(gt_path),
        "shape": list(target.shape),
        "prediction_raw_min": pred_min,
        "prediction_raw_max": pred_max,
        "gt_raw_min": gt_min,
        "gt_raw_max": gt_max,
        "volume_psnr": volume_psnr,
        "volume_mse": volume_mse,
        "volume_mae": float(np.mean(np.abs(prediction - target))),
        "slice_metrics": slice_rows,
        "prior_path": str(prior_path) if prior_path else None,
        "region_metrics": region_metrics,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate exported X-Gaussian CT volume against GT volume.")
    parser.add_argument("--pred", required=True, help="Predicted .npy volume from export_ct_volume.py.")
    parser.add_argument("--gt_pickle", required=True, help="Original X-Gaussian data pickle containing GT image volume.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--label", default="volume")
    parser.add_argument("--prior_path", default="")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary = evaluate_volume(args.pred, args.gt_pickle, args.out_dir, args.label, prior_path=args.prior_path or None)

    json_path = Path(args.out_dir) / "volume_metrics.json"
    csv_path = Path(args.out_dir) / "slice_metrics.csv"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slice", "psnr", "ssim", "mse", "mae"])
        writer.writeheader()
        for slice_name, metrics in summary["slice_metrics"].items():
            writer.writerow({"slice": slice_name, **metrics})

    print(json.dumps(summary, indent=2))
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
