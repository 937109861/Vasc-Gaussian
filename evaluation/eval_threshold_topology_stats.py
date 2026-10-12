import argparse
import csv
import json
import os
import pickle
from pathlib import Path

import numpy as np
from scipy import ndimage

from eval_vessel_topology import (
    binary_dice,
    binary_precision_recall,
    centerline_metrics,
    load_pred_volume,
    make_mask,
    normalize_volume,
    resolve_threshold,
)


def load_gt_volume(source_path):
    with open(source_path, "rb") as handle:
        data = pickle.load(handle)
    if "image" not in data:
        raise KeyError(f"GT pickle does not contain image volume: {source_path}")
    return np.asarray(data["image"], dtype=np.float32)


def component_stats(mask, small_component_max_voxels):
    mask = np.asarray(mask, dtype=bool)
    pred_voxels = int(mask.sum())
    if pred_voxels == 0:
        return {
            "num_components": 0,
            "small_cc": 0,
            "largest_component_voxels": 0,
            "lcc_ratio": 0.0,
            "non_lcc_components": 0,
        }

    structure = ndimage.generate_binary_structure(3, 2)
    labels, num_components = ndimage.label(mask, structure=structure)
    counts = np.bincount(labels.reshape(-1))
    component_sizes = counts[1:]
    if component_sizes.size == 0:
        return {
            "num_components": 0,
            "small_cc": 0,
            "largest_component_voxels": 0,
            "lcc_ratio": 0.0,
            "non_lcc_components": 0,
        }

    largest = int(component_sizes.max())
    largest_index = int(component_sizes.argmax())
    small_limit = int(small_component_max_voxels)
    small_cc = int(
        sum(
            1
            for idx, size in enumerate(component_sizes)
            if idx != largest_index and int(size) < small_limit
        )
    )
    return {
        "num_components": int(num_components),
        "small_cc": small_cc,
        "largest_component_voxels": largest,
        "lcc_ratio": float(largest / pred_voxels) if pred_voxels else 0.0,
        "non_lcc_components": int(max(0, num_components - 1)),
    }


def evaluate(
    pred_path,
    gt_pickle,
    out_dir,
    label,
    threshold,
    gt_threshold,
    gt_quantile,
    small_component_max_voxels,
):
    pred_raw = load_pred_volume(pred_path)
    gt_raw = load_gt_volume(gt_pickle)
    if pred_raw.shape != gt_raw.shape:
        raise ValueError(f"Volume shape mismatch: pred {pred_raw.shape}, gt {gt_raw.shape}")

    pred, pred_min, pred_max = normalize_volume(pred_raw)
    gt, gt_min, gt_max = normalize_volume(gt_raw)

    gt_thr = resolve_threshold(gt, gt_threshold, gt_quantile)
    pred_thr = float(threshold) if threshold is not None else gt_thr
    pred_mask = make_mask(pred, pred_thr)
    gt_mask = make_mask(gt, gt_thr)

    precision, recall = binary_precision_recall(pred_mask, gt_mask)
    centerline = centerline_metrics(pred_mask, gt_mask)
    components = component_stats(pred_mask, small_component_max_voxels)
    metrics = {
        "label": label,
        "prediction": str(pred_path),
        "gt": str(gt_pickle),
        "shape": list(gt.shape),
        "prediction_raw_min": pred_min,
        "prediction_raw_max": pred_max,
        "gt_raw_min": gt_min,
        "gt_raw_max": gt_max,
        "threshold": pred_thr,
        "gt_threshold": gt_thr,
        "gt_positive_quantile": float(gt_quantile),
        "small_component_max_voxels": int(small_component_max_voxels),
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
        **components,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "threshold_topology_stats.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    fields = [
        "label",
        "threshold",
        "dice",
        "cldice",
        "centerline_recall",
        "small_cc",
        "lcc_ratio",
        "num_components",
        "largest_component_voxels",
        "pred_voxels",
        "precision",
        "recall",
    ]
    with open(out_dir / "threshold_topology_stats.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: metrics[field] for field in fields})

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate threshold sensitivity and connected-component topology statistics."
    )
    parser.add_argument("--pred", required=True, help="Predicted .npy/.npz volume.")
    parser.add_argument("--gt_pickle", required=True, help="X-Gaussian pickle containing GT image volume.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--label", default="threshold_topology")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold on normalized prediction. Defaults to resolved GT threshold tau.",
    )
    parser.add_argument(
        "--gt_threshold",
        type=float,
        default=None,
        help="Threshold on normalized GT. Overrides --gt_positive_quantile.",
    )
    parser.add_argument("--gt_positive_quantile", type=float, default=0.05)
    parser.add_argument(
        "--small_component_max_voxels",
        type=int,
        default=280,
        help="Non-largest components below this size are counted as Small CC.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = evaluate(
        pred_path=args.pred,
        gt_pickle=args.gt_pickle,
        out_dir=args.out_dir,
        label=args.label,
        threshold=args.threshold,
        gt_threshold=args.gt_threshold,
        gt_quantile=args.gt_positive_quantile,
        small_component_max_voxels=args.small_component_max_voxels,
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {Path(args.out_dir) / 'threshold_topology_stats.json'}")
    print(f"Saved: {Path(args.out_dir) / 'threshold_topology_stats.csv'}")


if __name__ == "__main__":
    raise SystemExit(main())
