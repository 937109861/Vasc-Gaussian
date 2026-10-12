import argparse
import json
from pathlib import Path

import numpy as np


def load_volume(path, key=None):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    payload = np.load(path)
    if key:
        return payload[key].astype(np.float32)
    for candidate in ("prior_volume", "soft_prior", "vesselness", "volume", "image"):
        if candidate in payload:
            return payload[candidate].astype(np.float32)
    return payload[payload.files[0]].astype(np.float32)


def load_mask(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(bool)
    payload = np.load(path)
    for key in ("eval_mask", "mask", "label"):
        if key in payload:
            return payload[key].astype(bool)
    return payload[payload.files[0]].astype(bool)


def metrics_at_threshold(volume, mask, threshold):
    pred = volume >= float(threshold)
    tp = int(np.logical_and(pred, mask).sum())
    pred_count = int(pred.sum())
    gt_count = int(mask.sum())
    return {
        "threshold": float(threshold),
        "pred_voxels": pred_count,
        "gt_voxels": gt_count,
        "tp": tp,
        "dice": float((2.0 * tp) / max(pred_count + gt_count, 1)),
        "precision": float(tp / max(pred_count, 1)),
        "recall": float(tp / max(gt_count, 1)),
    }


def threshold_for_topk(volume, k):
    flat = volume.reshape(-1)
    k = max(1, min(int(k), flat.size))
    kth = flat.size - k
    return float(np.partition(flat, kth)[kth])


def main():
    parser = argparse.ArgumentParser(description="Top-k and percentile sweep for ImageCAS eval_mask overlap.")
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prediction_key", default=None)
    parser.add_argument("--roi_mask", default="", help="Optional .npy/.npz ROI mask. Ranking is restricted to this ROI.")
    parser.add_argument("--roi_key", default=None)
    parser.add_argument("--topk_factors", nargs="+", type=float, default=[1, 2, 5, 10, 20, 50, 100])
    parser.add_argument("--percentiles", nargs="+", type=float, default=[99, 99.5, 99.9, 99.95, 99.99])
    args = parser.parse_args()

    volume = load_volume(args.prediction, key=args.prediction_key)
    mask = load_mask(args.eval)
    if volume.shape != mask.shape:
        raise ValueError(f"Prediction shape {volume.shape} != eval mask shape {mask.shape}")
    roi_mask = None
    if args.roi_mask:
        roi_mask = load_volume(args.roi_mask, key=args.roi_key).astype(bool)
        if roi_mask.shape != volume.shape:
            raise ValueError(f"ROI shape {roi_mask.shape} != prediction shape {volume.shape}")
        volume_for_ranking = np.where(roi_mask, volume, -np.inf).astype(np.float32)
    else:
        volume_for_ranking = volume

    gt_count = int(mask.sum())
    topk = []
    for factor in args.topk_factors:
        threshold = threshold_for_topk(volume_for_ranking, max(1, int(round(gt_count * float(factor)))))
        item = metrics_at_threshold(volume_for_ranking, mask, threshold)
        item["factor"] = float(factor)
        topk.append(item)

    percentile_metrics = []
    for percentile in args.percentiles:
        if roi_mask is not None:
            ranking_values = volume[roi_mask]
            threshold = float(np.percentile(ranking_values, float(percentile))) if ranking_values.size else float("inf")
        else:
            threshold = float(np.percentile(volume, float(percentile)))
        item = metrics_at_threshold(volume_for_ranking, mask, threshold)
        item["percentile"] = float(percentile)
        percentile_metrics.append(item)

    summary = {
        "prediction": args.prediction,
        "eval": args.eval,
        "shape": list(volume.shape),
        "mask_voxels": gt_count,
        "roi_mask": args.roi_mask or None,
        "roi_voxels": int(roi_mask.sum()) if roi_mask is not None else None,
        "mask_voxels_inside_roi": int(np.logical_and(mask, roi_mask).sum()) if roi_mask is not None else None,
        "prediction_min": float(volume.min()),
        "prediction_max": float(volume.max()),
        "prediction_percentiles": {
            str(p): float(np.percentile(volume, p)) for p in [0, 1, 5, 50, 95, 99, 99.5, 99.9, 100]
        },
        "topk": topk,
        "percentiles": percentile_metrics,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "topk_sweep_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
