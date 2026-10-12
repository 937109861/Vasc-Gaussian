import argparse
import json
import pickle
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from priors.volume_prior import (
    ConeGeometry,
    dilate_mask,
    sample_points,
    voxel_positions_for_volume,
    write_ascii_ply,
)
from scripts.build_vesselness_prior import frangi_vesselness_3d, read_volume


def normalize_source(volume, mode, pmin, pmax):
    volume = np.asarray(volume, dtype=np.float32)
    raw_min = float(volume.min())
    raw_max = float(volume.max())
    if mode == "identity":
        normalized = np.clip(volume, 0.0, 1.0)
        lo, hi = 0.0, 1.0
    elif mode == "minmax":
        lo, hi = raw_min, raw_max
        if hi <= lo:
            normalized = np.zeros_like(volume, dtype=np.float32)
        else:
            normalized = np.clip((volume - lo) / (hi - lo), 0.0, 1.0)
    elif mode == "percentile":
        lo = float(np.percentile(volume, pmin))
        hi = float(np.percentile(volume, pmax))
        if hi <= lo:
            normalized = np.zeros_like(volume, dtype=np.float32)
        else:
            normalized = np.clip((volume - lo) / (hi - lo), 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")
    return normalized.astype(np.float32), {
        "raw_min": raw_min,
        "raw_max": raw_max,
        "normalization_mode": mode,
        "normalization_low": float(lo),
        "normalization_high": float(hi),
    }


def soft_bandpass(volume, low, high, margin):
    volume = np.asarray(volume, dtype=np.float32)
    low = float(low)
    high = float(high)
    margin = max(float(margin), 1e-6)
    if high <= low:
        raise ValueError("--intensity_high must be greater than --intensity_low")

    lower = np.clip((volume - (low - margin)) / margin, 0.0, 1.0)
    upper = np.clip(((high + margin) - volume) / margin, 0.0, 1.0)
    band = np.minimum(lower, upper)
    band[(volume >= low) & (volume <= high)] = 1.0
    return np.clip(band, 0.0, 1.0).astype(np.float32)


def normalize_unit(volume, eps=1e-6):
    volume = np.asarray(volume, dtype=np.float32)
    vmin = float(volume.min())
    vmax = float(volume.max())
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - vmin) / max(vmax - vmin, eps), 0.0, 1.0).astype(np.float32)


def threshold_candidates(fused, args):
    if args.threshold_mode == "absolute":
        threshold = float(args.threshold_value)
    elif args.threshold_mode == "quantile":
        values = fused[fused > 0.0] if bool(args.threshold_positive_only) else fused.reshape(-1)
        if values.size == 0:
            threshold = float("inf")
        else:
            threshold = float(np.quantile(values, float(args.threshold_value)))
    elif args.threshold_mode == "top_voxels":
        target = max(1, int(args.target_voxels))
        flat = fused[fused > 0.0] if bool(args.threshold_positive_only) else fused.reshape(-1)
        if flat.size == 0:
            return np.zeros_like(fused, dtype=bool), float("inf")
        target = min(target, flat.size)
        kth = max(0, flat.size - target)
        threshold = float(np.partition(flat, kth)[kth])
    else:
        raise ValueError(f"Unsupported threshold mode: {args.threshold_mode}")
    if args.min_threshold is not None:
        threshold = max(threshold, float(args.min_threshold))
    mask = fused >= threshold
    return mask, threshold


def spatial_roi_mask(shape, args):
    if args.spatial_roi == "none":
        return np.ones(shape, dtype=bool), {
            "spatial_roi": "none",
            "spatial_roi_voxels": int(np.prod(shape)),
            "spatial_roi_ratio": 1.0,
        }

    z_size, y_size, x_size = [int(v) for v in shape]
    z, y, x = np.indices(shape, dtype=np.float32)
    center = np.asarray(args.roi_center_zyx, dtype=np.float32)
    radius = np.asarray(args.roi_radius_zyx, dtype=np.float32)
    if np.all(center <= 1.0):
        center = center * np.asarray([z_size - 1, y_size - 1, x_size - 1], dtype=np.float32)
    if np.all(radius <= 1.0):
        radius = radius * np.asarray([z_size, y_size, x_size], dtype=np.float32)
    radius = np.maximum(radius, 1.0)

    box = (
        (np.abs(z - center[0]) <= radius[0])
        & (np.abs(y - center[1]) <= radius[1])
        & (np.abs(x - center[2]) <= radius[2])
    )
    ellipsoid = (
        ((z - center[0]) / radius[0]) ** 2
        + ((y - center[1]) / radius[1]) ** 2
        + ((x - center[2]) / radius[2]) ** 2
    ) <= 1.0

    if args.spatial_roi == "box":
        mask = box
    elif args.spatial_roi == "ellipsoid":
        mask = ellipsoid
    elif args.spatial_roi == "box_ellipsoid":
        mask = box & ellipsoid
    else:
        raise ValueError(f"Unsupported spatial ROI mode: {args.spatial_roi}")

    return mask.astype(bool), {
        "spatial_roi": args.spatial_roi,
        "roi_center_zyx": [float(v) for v in center],
        "roi_radius_zyx": [float(v) for v in radius],
        "spatial_roi_voxels": int(mask.sum()),
        "spatial_roi_ratio": float(mask.mean()),
    }


def filter_components(mask, min_voxels=0, max_components=0):
    if int(min_voxels) <= 0 and int(max_components) <= 0:
        return mask.astype(bool), {"num_components": None, "kept_components": None, "removed_voxels": 0}

    from scipy import ndimage

    labels, num = ndimage.label(mask)
    if num == 0:
        return mask.astype(bool), {"num_components": 0, "kept_components": 0, "removed_voxels": 0}

    counts = np.bincount(labels.reshape(-1))
    component_ids = np.arange(1, num + 1)
    keep = component_ids[counts[1:] >= int(min_voxels)]
    if int(max_components) > 0 and keep.size > int(max_components):
        order = np.argsort(counts[keep])[::-1]
        keep = keep[order[: int(max_components)]]
    kept_mask = np.isin(labels, keep)
    return kept_mask, {
        "num_components": int(num),
        "kept_components": int(keep.size),
        "removed_voxels": int(mask.sum() - kept_mask.sum()),
    }


def load_geometry(path):
    if not path:
        return None
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return ConeGeometry(payload)


def save_slice_png(volume, path):
    volume = np.asarray(volume, dtype=np.float32)
    if volume.max() > volume.min():
        volume = (volume - volume.min()) / (volume.max() - volume.min())
    else:
        volume = np.zeros_like(volume, dtype=np.float32)
    imageio.imwrite(path, (np.clip(volume, 0.0, 1.0) * 255).astype(np.uint8))


def save_previews(out_dir, volumes):
    out_dir = Path(out_dir)
    slices_dir = out_dir / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)
    shape = next(iter(volumes.values())).shape
    mids = [int(size) // 2 for size in shape]
    views = {
        "axial_z": (slice(None), slice(None), mids[2]),
        "coronal_y": (slice(None), mids[1], slice(None)),
        "sagittal_x": (mids[0], slice(None), slice(None)),
    }
    for view_name, index in views.items():
        for name, volume in volumes.items():
            save_slice_png(volume[index], slices_dir / f"{view_name}_{name}.png")
    for name, volume in volumes.items():
        save_slice_png(np.max(volume, axis=2), out_dir / f"{name}_mip_z.png")


def eval_against_mask(volumes, mask_path, threshold):
    if not mask_path:
        return None
    payload = np.load(mask_path)
    if "eval_mask" in payload:
        mask = payload["eval_mask"].astype(bool)
    elif "mask" in payload:
        mask = payload["mask"].astype(bool)
    else:
        mask = payload[payload.files[0]].astype(bool)
    metrics = {}
    gt_count = int(mask.sum())
    for name, volume in volumes.items():
        pred = np.asarray(volume) >= float(threshold)
        tp = int(np.logical_and(pred, mask).sum())
        pred_count = int(pred.sum())
        precision = tp / max(pred_count, 1)
        recall = tp / max(gt_count, 1)
        dice = (2.0 * tp) / max(pred_count + gt_count, 1)
        metrics[name] = {
            "threshold": float(threshold),
            "pred_voxels": pred_count,
            "gt_voxels": gt_count,
            "tp": tp,
            "dice": float(dice),
            "precision": float(precision),
            "recall": float(recall),
        }
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Build an ImageCAS-specific coronary soft prior without using GT labels for prior construction."
    )
    parser.add_argument("--input", required=True, help="No-GT source volume, usually FDK recon_volume.npy.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--geometry_pickle", required=True)
    parser.add_argument("--input_key", default=None)
    parser.add_argument("--normalization", choices=["identity", "minmax", "percentile"], default="identity")
    parser.add_argument("--norm_percentiles", nargs=2, type=float, default=[0.5, 99.5])
    parser.add_argument("--intensity_low", type=float, default=0.44)
    parser.add_argument("--intensity_high", type=float, default=0.72)
    parser.add_argument("--intensity_margin", type=float, default=0.06)
    parser.add_argument("--intensity_power", type=float, default=1.0)
    parser.add_argument("--spatial_roi", choices=["none", "box", "ellipsoid", "box_ellipsoid"], default="none")
    parser.add_argument(
        "--roi_center_zyx",
        nargs=3,
        type=float,
        default=[0.44, 0.41, 0.52],
        help="ROI center in z y x. Values <=1 are treated as fractions of volume size.",
    )
    parser.add_argument(
        "--roi_radius_zyx",
        nargs=3,
        type=float,
        default=[0.28, 0.30, 0.33],
        help="ROI radius in z y x. Values <=1 are treated as fractions of volume size.",
    )
    parser.add_argument("--scales", nargs="+", type=float, default=[0.7, 1.0, 1.4, 2.0])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--chunk_size", type=int, default=500000)
    parser.add_argument("--vessel_weight", type=float, default=0.65)
    parser.add_argument("--intensity_weight", type=float, default=0.35)
    parser.add_argument("--threshold_mode", choices=["absolute", "quantile", "top_voxels"], default="quantile")
    parser.add_argument("--threshold_value", type=float, default=0.995)
    parser.add_argument("--target_voxels", type=int, default=100000)
    parser.add_argument("--threshold_positive_only", action="store_true")
    parser.add_argument("--min_threshold", type=float, default=None)
    parser.add_argument("--min_component_voxels", type=int, default=0)
    parser.add_argument("--max_components", type=int, default=0)
    parser.add_argument("--roi_dilation", type=int, default=3)
    parser.add_argument("--points_source", choices=["occupancy", "roi"], default="roi")
    parser.add_argument("--max_points", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", default="", help="Optional eval npz for reporting only; never used to build the prior.")
    parser.add_argument("--eval_threshold", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source, source_key = read_volume(args.input, key=args.input_key)
    source_norm, norm_stats = normalize_source(
        source,
        args.normalization,
        float(args.norm_percentiles[0]),
        float(args.norm_percentiles[1]),
    )

    intensity_prior = soft_bandpass(
        source_norm,
        low=float(args.intensity_low),
        high=float(args.intensity_high),
        margin=float(args.intensity_margin),
    )
    if float(args.intensity_power) != 1.0:
        intensity_prior = np.power(intensity_prior, float(args.intensity_power)).astype(np.float32)

    spatial_roi, spatial_stats = spatial_roi_mask(source_norm.shape, args)
    intensity_prior = (intensity_prior * spatial_roi.astype(np.float32)).astype(np.float32)
    vessel_source = source_norm * intensity_prior
    vesselness = frangi_vesselness_3d(
        vessel_source,
        scales=args.scales,
        alpha=float(args.alpha),
        beta=float(args.beta),
        gamma=float(args.gamma),
        bright=True,
        chunk_size=int(args.chunk_size),
    )
    fused = (
        float(args.intensity_weight) * intensity_prior
        + float(args.vessel_weight) * vesselness * intensity_prior
    )
    fused = normalize_unit(fused)

    occupancy_mask, threshold = threshold_candidates(fused, args)
    occupancy_mask, cc_stats = filter_components(
        occupancy_mask,
        min_voxels=int(args.min_component_voxels),
        max_components=int(args.max_components),
    )
    roi_mask = dilate_mask(occupancy_mask, int(args.roi_dilation))

    geometry = load_geometry(args.geometry_pickle)
    positions = voxel_positions_for_volume(fused.shape, geometry)
    if positions.shape[:3] != fused.shape:
        axes = [np.linspace(-1.0, 1.0, int(size), dtype=np.float32) for size in fused.shape]
        grid = np.meshgrid(*axes, indexing="ij")
        positions = np.asarray(grid, dtype=np.float32).transpose([1, 2, 3, 0])
    points_mask = roi_mask if args.points_source == "roi" else occupancy_mask
    points = positions[points_mask].reshape(-1, 3).astype(np.float32)
    points = sample_points(points, int(args.max_points), int(args.seed))

    np.save(out_dir / "source_norm.npy", source_norm.astype(np.float32))
    np.save(out_dir / "intensity_prior.npy", intensity_prior.astype(np.float32))
    np.save(out_dir / "spatial_roi.npy", spatial_roi.astype(np.uint8))
    np.save(out_dir / "vesselness.npy", vesselness.astype(np.float32))
    np.save(out_dir / "prior_volume.npy", fused.astype(np.float32))
    np.save(out_dir / "occupancy_mask.npy", occupancy_mask.astype(np.uint8))
    np.save(out_dir / "roi_mask.npy", roi_mask.astype(np.uint8))
    np.save(out_dir / "point_cloud.npy", points.astype(np.float32))
    np.savez_compressed(
        out_dir / "prior_data.npz",
        occupancy_mask=occupancy_mask.astype(np.uint8),
        roi_mask=roi_mask.astype(np.uint8),
        points=points.astype(np.float32),
        prior_volume=fused.astype(np.float32),
        soft_prior=fused.astype(np.float32),
        vesselness=vesselness.astype(np.float32),
        intensity_prior=intensity_prior.astype(np.float32),
        spatial_roi=spatial_roi.astype(np.uint8),
        source_prior=intensity_prior.astype(np.float32),
        volume_shape=np.asarray(fused.shape, dtype=np.int32),
    )
    if points.shape[0] > 0:
        write_ascii_ply(out_dir / "point_cloud.ply", points)

    save_previews(
        out_dir,
        {
            "source_norm": source_norm,
            "intensity_prior": intensity_prior,
            "spatial_roi": spatial_roi.astype(np.float32),
            "vesselness": vesselness,
            "fused_prior": fused,
            "occupancy": occupancy_mask.astype(np.float32),
            "roi": roi_mask.astype(np.float32),
        },
    )

    eval_metrics = eval_against_mask(
        {"intensity_prior": intensity_prior, "vesselness": vesselness, "fused_prior": fused},
        args.eval,
        float(args.eval_threshold),
    )
    stats = {
        "input": str(args.input),
        "input_key": source_key,
        "geometry_pickle": str(args.geometry_pickle),
        "shape": list(fused.shape),
        **norm_stats,
        "intensity_low": float(args.intensity_low),
        "intensity_high": float(args.intensity_high),
        "intensity_margin": float(args.intensity_margin),
        "intensity_power": float(args.intensity_power),
        **spatial_stats,
        "scales": [float(v) for v in args.scales],
        "vessel_weight": float(args.vessel_weight),
        "intensity_weight": float(args.intensity_weight),
        "threshold_mode": args.threshold_mode,
        "threshold_value": float(args.threshold_value),
        "threshold_applied": float(threshold),
        "target_voxels": int(args.target_voxels),
        "occupancy_voxels": int(occupancy_mask.sum()),
        "occupancy_ratio": float(occupancy_mask.mean()),
        "roi_voxels": int(roi_mask.sum()),
        "roi_ratio": float(roi_mask.mean()),
        "num_points": int(points.shape[0]),
        "component_filter": cc_stats,
        "note": "GT labels are not used to build this prior. Optional eval metrics are reporting only.",
        "eval_metrics": eval_metrics,
    }
    with open(out_dir / "imagecas_coronary_prior_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(json.dumps({"outputs": str(out_dir / "prior_data.npz"), "stats": stats}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
