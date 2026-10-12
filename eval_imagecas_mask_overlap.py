import argparse
import json
from pathlib import Path

import numpy as np


def load_prediction(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    payload = np.load(path)
    for key in ("prior_volume", "soft_prior", "vesselness", "volume", "image"):
        if key in payload:
            return payload[key].astype(np.float32)
    return payload[payload.files[0]].astype(np.float32)


def load_eval_mask(path):
    payload = np.load(path)
    for key in ("eval_mask", "mask", "label"):
        if key in payload:
            return payload[key].astype(bool)
    return payload[payload.files[0]].astype(bool)


def compute_metrics(prediction, mask, threshold):
    pred = prediction >= float(threshold)
    tp = int(np.logical_and(pred, mask).sum())
    fp = int(np.logical_and(pred, ~mask).sum())
    fn = int(np.logical_and(~pred, mask).sum())
    pred_voxels = int(pred.sum())
    gt_voxels = int(mask.sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    dice = (2.0 * tp) / max(pred_voxels + gt_voxels, 1)
    return {
        "pred_voxels": pred_voxels,
        "gt_voxels": gt_voxels,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "pred_min": float(prediction.min()),
        "pred_max": float(prediction.max()),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ImageCAS prediction/prior overlap with eval_mask.")
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    prediction = load_prediction(args.prediction)
    mask = load_eval_mask(args.eval)
    if prediction.shape != mask.shape:
        raise ValueError(f"Prediction shape {prediction.shape} != eval mask shape {mask.shape}")

    metrics = {
        "prediction": args.prediction,
        "eval": args.eval,
        "threshold": float(args.threshold),
        **compute_metrics(prediction, mask, float(args.threshold)),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "mask_overlap_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
