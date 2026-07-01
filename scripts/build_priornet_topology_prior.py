import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from priors.volume_prior import dilate_mask


def read_prior_volume(path, key=None):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path).astype(np.float32), "npy"
    if suffix == ".npz":
        payload = np.load(path)
        keys = [key] if key else [
            "soft_prior",
            "prior_volume",
            "vessel_prob",
            "volume",
            "recon_volume",
            "occupancy_mask",
        ]
        for candidate in keys:
            if candidate and candidate in payload:
                return payload[candidate].astype(np.float32), candidate
        return payload[payload.files[0]].astype(np.float32), payload.files[0]
    raise ValueError(f"Unsupported prior input: {path}")


def normalize_volume(volume, eps=1e-6):
    volume = np.asarray(volume, dtype=np.float32)
    vmin = float(volume.min())
    vmax = float(volume.max())
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - vmin) / max(vmax - vmin, eps)).astype(np.float32)


def threshold_volume(volume, mode, value):
    if mode == "absolute":
        threshold = float(value)
    elif mode == "quantile":
        threshold = float(np.quantile(volume, float(value)))
    else:
        raise ValueError(f"Unsupported threshold mode: {mode}")
    return volume >= threshold, threshold


def connected_component_cleanup(mask, min_voxels=0, keep_largest=False):
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError(
            "scipy is required for topology prior connected-component cleanup. "
            "Install scipy or add it to the environment."
        ) from exc

    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return mask, {"num_components": 0, "kept_components": 0, "largest_component_voxels": 0}

    structure = ndimage.generate_binary_structure(3, 2)
    labels, num_labels = ndimage.label(mask, structure=structure)
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0

    if keep_largest:
        largest = int(np.argmax(sizes))
        cleaned = labels == largest
        kept = 1 if sizes[largest] > 0 else 0
    elif int(min_voxels) > 0:
        keep_labels = np.flatnonzero(sizes >= int(min_voxels))
        cleaned = np.isin(labels, keep_labels)
        kept = int(len(keep_labels))
    else:
        cleaned = mask
        kept = int(num_labels)

    stats = {
        "num_components": int(num_labels),
        "kept_components": kept,
        "largest_component_voxels": int(sizes.max()) if sizes.size else 0,
    }
    return cleaned.astype(bool), stats


def morphological_cleanup(mask, opening=0, closing=0):
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError(
            "scipy is required for topology prior morphological cleanup."
        ) from exc

    cleaned = np.asarray(mask, dtype=bool)
    structure = ndimage.generate_binary_structure(3, 2)
    if int(opening) > 0:
        cleaned = ndimage.binary_opening(cleaned, structure=structure, iterations=int(opening))
    if int(closing) > 0:
        cleaned = ndimage.binary_closing(cleaned, structure=structure, iterations=int(closing))
    return cleaned.astype(bool)


def seeded_component_cleanup(candidate_mask, seed_mask, min_voxels=0, min_seed_voxels=1, keep_largest=False):
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError(
            "scipy is required for topology prior connected-component cleanup. "
            "Install scipy or add it to the environment."
        ) from exc

    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    seed_mask = np.asarray(seed_mask, dtype=bool) & candidate_mask
    if not np.any(candidate_mask):
        return candidate_mask, {"num_components": 0, "kept_components": 0, "largest_component_voxels": 0}

    structure = ndimage.generate_binary_structure(3, 2)
    labels, num_labels = ndimage.label(candidate_mask, structure=structure)
    sizes = np.bincount(labels.reshape(-1), minlength=num_labels + 1)
    seed_counts = np.bincount(labels[seed_mask].reshape(-1), minlength=num_labels + 1)
    sizes[0] = 0
    seed_counts[0] = 0
    component_ids = np.arange(1, num_labels + 1)
    keep_ids = component_ids[
        (sizes[component_ids] >= int(min_voxels))
        & (seed_counts[component_ids] >= int(min_seed_voxels))
    ]

    if keep_largest and keep_ids.size > 0:
        keep_ids = keep_ids[np.argsort(sizes[keep_ids])[::-1]][:1]

    keep = np.zeros(num_labels + 1, dtype=bool)
    keep[keep_ids] = True
    cleaned = keep[labels]
    stats = {
        "num_components": int(num_labels),
        "kept_components": int(keep_ids.size),
        "largest_component_voxels": int(sizes.max()) if sizes.size else 0,
        "seed_voxels": int(seed_mask.sum()),
        "min_seed_voxels": int(min_seed_voxels),
    }
    return cleaned.astype(bool), stats


def limit_centerline_ratio(centerline, score, max_ratio):
    max_ratio = float(max_ratio)
    if max_ratio <= 0.0:
        return centerline
    centerline = np.asarray(centerline, dtype=bool)
    count = int(centerline.sum())
    max_count = max(1, int(centerline.size * max_ratio))
    if count <= max_count:
        return centerline

    values = np.asarray(score, dtype=np.float32)[centerline]
    if values.size == 0:
        return centerline
    cutoff_index = max(0, values.size - max_count)
    cutoff = float(np.partition(values, cutoff_index)[cutoff_index])
    limited = centerline & (score >= cutoff)
    if int(limited.sum()) > max_count:
        coords = np.flatnonzero(limited.reshape(-1))
        limited_flat = np.zeros(limited.size, dtype=bool)
        limited_flat[coords[:max_count]] = True
        limited = limited_flat.reshape(limited.shape)
    return limited


def build_support_roi(
    soft_prior,
    vessel_mask,
    support_low_threshold=None,
    support_min_component_voxels=0,
    support_dilation=1,
):
    if support_low_threshold is None:
        support = np.asarray(vessel_mask, dtype=bool)
    else:
        support = np.asarray(soft_prior >= float(support_low_threshold), dtype=bool)
        support, _ = connected_component_cleanup(
            support,
            min_voxels=int(support_min_component_voxels),
            keep_largest=False,
        )
        support = np.logical_or(support, vessel_mask)
    if int(support_dilation) > 0:
        support = dilate_mask(support, int(support_dilation)).astype(bool)
    return support.astype(bool)


def build_centerline(mask, method="auto", ridge_quantile=0.75):
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError("scipy is required for topology prior distance maps.") from exc

    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return mask.copy(), np.zeros_like(mask, dtype=np.float32), "empty"

    inside_distance = ndimage.distance_transform_edt(mask).astype(np.float32)

    if method in ("auto", "skimage"):
        try:
            from skimage.morphology import skeletonize

            skeleton = skeletonize(mask).astype(bool)
            if np.any(skeleton):
                return skeleton, inside_distance, "skimage"
        except Exception:
            if method == "skimage":
                raise

    if method not in ("auto", "ridge", "skimage"):
        raise ValueError(f"Unsupported centerline method: {method}")

    local_max = inside_distance == ndimage.maximum_filter(inside_distance, size=3)
    vessel_values = inside_distance[mask]
    min_distance = float(np.quantile(vessel_values, float(ridge_quantile))) if vessel_values.size else 0.0
    skeleton = mask & local_max & (inside_distance >= max(min_distance, 1.0))
    if not np.any(skeleton):
        fallback_threshold = float(np.quantile(vessel_values, 0.90)) if vessel_values.size else 0.0
        skeleton = mask & (inside_distance >= max(fallback_threshold, 1.0))
    if not np.any(skeleton):
        skeleton = mask.copy()
    return skeleton.astype(bool), inside_distance, "ridge"


def _centerline_neighbor_count(centerline):
    from scipy import ndimage

    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    return ndimage.convolve(centerline.astype(np.uint8), kernel, mode="constant", cval=0)


def prune_short_centerline_branches(centerline, min_branch_voxels=0):
    min_branch_voxels = int(min_branch_voxels)
    centerline = np.asarray(centerline, dtype=bool).copy()
    if min_branch_voxels <= 1 or not np.any(centerline):
        return centerline, {"short_branches_removed": 0, "short_branch_voxels_removed": 0}

    removed_paths = []
    neighbor_offsets = [
        np.asarray([dx, dy, dz], dtype=np.int64)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    neighbor_count = _centerline_neighbor_count(centerline)
    endpoint_coords = np.argwhere(centerline & (neighbor_count <= 1))
    shape = np.asarray(centerline.shape, dtype=np.int64)

    for endpoint in endpoint_coords:
        point = tuple(endpoint.tolist())
        if not centerline[point]:
            continue
        path = [point]
        previous = None
        current = endpoint.astype(np.int64)

        while len(path) <= min_branch_voxels:
            neighbors = []
            for offset in neighbor_offsets:
                candidate = current + offset
                if np.any(candidate < 0) or np.any(candidate >= shape):
                    continue
                candidate_tuple = tuple(candidate.tolist())
                if not centerline[candidate_tuple] or candidate_tuple == previous:
                    continue
                neighbors.append(candidate_tuple)

            if not neighbors:
                break
            if len(neighbors) > 1:
                if len(path) < min_branch_voxels:
                    removed_paths.append(path)
                break

            previous = tuple(current.tolist())
            current = np.asarray(neighbors[0], dtype=np.int64)
            path.append(tuple(current.tolist()))

            degree = int(neighbor_count[tuple(current.tolist())])
            if degree >= 3:
                if len(path) < min_branch_voxels:
                    removed_paths.append(path[:-1])
                break

    removed_voxels = set()
    for path in removed_paths:
        for point in path:
            removed_voxels.add(point)
    for point in removed_voxels:
        centerline[point] = False

    return centerline, {
        "short_branches_removed": int(len(removed_paths)),
        "short_branch_voxels_removed": int(len(removed_voxels)),
    }


def build_distance_map(centerline, roi, max_distance_percentile=95.0):
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError("scipy is required for topology prior distance maps.") from exc

    centerline = np.asarray(centerline, dtype=bool)
    roi = np.asarray(roi, dtype=bool)
    if not np.any(centerline):
        distance = (~roi).astype(np.float32)
    else:
        distance = ndimage.distance_transform_edt(~centerline).astype(np.float32)
    if np.any(roi):
        scale_values = distance[roi]
    else:
        scale_values = distance.reshape(-1)
    max_distance = float(np.percentile(scale_values, float(max_distance_percentile))) if scale_values.size else 1.0
    max_distance = max(max_distance, 1e-6)
    return np.clip(distance / max_distance, 0.0, 1.0).astype(np.float32)


def build_candidate_distance_map(candidate_mask, max_distance_percentile=95.0):
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError("scipy is required for topology prior distance maps.") from exc

    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    distance = ndimage.distance_transform_edt(~candidate_mask).astype(np.float32)
    outside_values = distance[~candidate_mask]
    if outside_values.size:
        max_distance = float(np.percentile(outside_values, float(max_distance_percentile)))
    else:
        max_distance = 1.0
    max_distance = max(max_distance, 1e-6)
    return np.clip(distance / max_distance, 0.0, 1.0).astype(np.float32)


def save_preview_slices(arrays, out_dir):
    out_dir = Path(out_dir)
    preview_dir = out_dir / "slices"
    preview_dir.mkdir(parents=True, exist_ok=True)
    shape = next(iter(arrays.values())).shape
    mids = [size // 2 for size in shape]
    views = {
        "axial_z": (slice(None), slice(None), mids[2]),
        "coronal_y": (slice(None), mids[1], slice(None)),
        "sagittal_x": (mids[0], slice(None), slice(None)),
    }
    for view, index in views.items():
        for name, array in arrays.items():
            image = np.asarray(array[index], dtype=np.float32)
            if image.max() > image.min():
                image = (image - image.min()) / max(float(image.max() - image.min()), 1e-6)
            else:
                image = np.zeros_like(image, dtype=np.float32)
            imageio.imwrite(preview_dir / f"{view}_{name}.png", (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(
        description="Build a vessel topology prior from a PriorNet soft prior."
    )
    parser.add_argument("--input", required=True, help="PriorNet prior_data.npz, prior_volume.npy, or soft_prior npz.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--input_key", default=None, help="Optional npz key to read.")
    parser.add_argument("--threshold_mode", default="absolute", choices=["absolute", "quantile"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--low_threshold", type=float, default=None, help="Optional hysteresis low threshold.")
    parser.add_argument("--high_threshold", type=float, default=None, help="Optional hysteresis seed threshold.")
    parser.add_argument("--min_component_voxels", type=int, default=0)
    parser.add_argument("--min_seed_voxels", type=int, default=1)
    parser.add_argument("--keep_largest_component", action="store_true")
    parser.add_argument("--mask_opening", type=int, default=0)
    parser.add_argument("--mask_closing", type=int, default=0)
    parser.add_argument("--roi_dilation", type=int, default=2)
    parser.add_argument(
        "--support_low_threshold",
        type=float,
        default=None,
        help="Optional lower threshold for a wider support ROI that protects weak vessels.",
    )
    parser.add_argument("--support_min_component_voxels", type=int, default=80)
    parser.add_argument("--support_dilation", type=int, default=1)
    parser.add_argument("--centerline_method", default="auto", choices=["auto", "skimage", "ridge"])
    parser.add_argument("--ridge_quantile", type=float, default=0.90)
    parser.add_argument("--max_centerline_ratio", type=float, default=0.003)
    parser.add_argument("--min_centerline_branch_voxels", type=int, default=0)
    parser.add_argument("--max_distance_percentile", type=float, default=95.0)
    parser.add_argument("--no_preview", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    soft_prior, source_key = read_prior_volume(args.input, key=args.input_key)
    soft_prior = normalize_volume(soft_prior)
    if args.low_threshold is not None and args.high_threshold is not None:
        low_threshold = float(args.low_threshold)
        high_threshold = float(args.high_threshold)
        if high_threshold < low_threshold:
            raise ValueError("--high_threshold must be >= --low_threshold")
        candidate_mask = soft_prior >= low_threshold
        seed_mask = soft_prior >= high_threshold
        vessel_mask, component_stats = seeded_component_cleanup(
            candidate_mask,
            seed_mask,
            min_voxels=args.min_component_voxels,
            min_seed_voxels=args.min_seed_voxels,
            keep_largest=args.keep_largest_component,
        )
        threshold = low_threshold
        threshold_mode = "hysteresis"
    else:
        vessel_mask, threshold = threshold_volume(soft_prior, args.threshold_mode, args.threshold)
        vessel_mask, component_stats = connected_component_cleanup(
            vessel_mask,
            min_voxels=args.min_component_voxels,
            keep_largest=args.keep_largest_component,
        )
        high_threshold = None
        threshold_mode = args.threshold_mode
    vessel_mask = morphological_cleanup(
        vessel_mask,
        opening=args.mask_opening,
        closing=args.mask_closing,
    )
    vessel_mask, post_morph_component_stats = connected_component_cleanup(
        vessel_mask,
        min_voxels=args.min_component_voxels,
        keep_largest=args.keep_largest_component,
    )
    roi_mask = dilate_mask(vessel_mask, int(args.roi_dilation)).astype(bool)
    support_roi_mask = build_support_roi(
        soft_prior,
        vessel_mask,
        support_low_threshold=args.support_low_threshold,
        support_min_component_voxels=args.support_min_component_voxels,
        support_dilation=args.support_dilation,
    )
    centerline_mask, inside_distance, centerline_source = build_centerline(
        vessel_mask,
        method=args.centerline_method,
        ridge_quantile=args.ridge_quantile,
    )
    centerline_mask = limit_centerline_ratio(
        centerline_mask,
        inside_distance,
        max_ratio=args.max_centerline_ratio,
    )
    centerline_mask, branch_stats = prune_short_centerline_branches(
        centerline_mask,
        min_branch_voxels=args.min_centerline_branch_voxels,
    )
    distance_map = build_distance_map(
        centerline_mask,
        roi_mask,
        max_distance_percentile=args.max_distance_percentile,
    )
    candidate_distance_map = build_candidate_distance_map(
        support_roi_mask,
        max_distance_percentile=args.max_distance_percentile,
    )

    out_path = out_dir / "topology_prior.npz"
    np.savez_compressed(
        out_path,
        soft_prior=soft_prior.astype(np.float32),
        M_high=vessel_mask.astype(np.uint8),
        M_candidate=support_roi_mask.astype(np.uint8),
        vessel_mask=vessel_mask.astype(np.uint8),
        roi_mask=roi_mask.astype(np.uint8),
        support_roi_mask=support_roi_mask.astype(np.uint8),
        candidate_mask=support_roi_mask.astype(np.uint8),
        centerline_mask=centerline_mask.astype(np.uint8),
        distance_map=distance_map.astype(np.float32),
        centerline_distance_map=distance_map.astype(np.float32),
        candidate_distance_map=candidate_distance_map.astype(np.float32),
        inside_distance=inside_distance.astype(np.float32),
        volume_shape=np.asarray(soft_prior.shape, dtype=np.int32),
    )

    if not args.no_preview:
        save_preview_slices(
            {
                "soft_prior": soft_prior,
                "vessel_mask": vessel_mask.astype(np.float32),
                "roi_mask": roi_mask.astype(np.float32),
                "support_roi_mask": support_roi_mask.astype(np.float32),
                "centerline_mask": centerline_mask.astype(np.float32),
                "distance_map": distance_map,
                "candidate_distance_map": candidate_distance_map,
            },
            out_dir,
        )

    stats = {
        "input": str(args.input),
        "input_key": source_key,
        "output": str(out_path),
        "shape": list(soft_prior.shape),
        "threshold_mode": threshold_mode,
        "threshold_requested": float(args.threshold),
        "threshold_applied": float(threshold),
        "low_threshold": float(args.low_threshold) if args.low_threshold is not None else None,
        "high_threshold": float(high_threshold) if high_threshold is not None else None,
        "vessel_voxels": int(vessel_mask.sum()),
        "vessel_ratio": float(vessel_mask.mean()) if vessel_mask.size else 0.0,
        "roi_voxels": int(roi_mask.sum()),
        "roi_ratio": float(roi_mask.mean()) if roi_mask.size else 0.0,
        "support_roi_voxels": int(support_roi_mask.sum()),
        "support_roi_ratio": float(support_roi_mask.mean()) if support_roi_mask.size else 0.0,
        "centerline_voxels": int(centerline_mask.sum()),
        "centerline_ratio": float(centerline_mask.mean()) if centerline_mask.size else 0.0,
        "centerline_source": centerline_source,
        "max_centerline_ratio": float(args.max_centerline_ratio),
        "roi_dilation": int(args.roi_dilation),
        "support_low_threshold": float(args.support_low_threshold) if args.support_low_threshold is not None else None,
        "support_min_component_voxels": int(args.support_min_component_voxels),
        "support_dilation": int(args.support_dilation),
        "mask_opening": int(args.mask_opening),
        "mask_closing": int(args.mask_closing),
        "ridge_quantile": float(args.ridge_quantile),
        "max_distance_percentile": float(args.max_distance_percentile),
        "min_centerline_branch_voxels": int(args.min_centerline_branch_voxels),
        **component_stats,
        **{f"post_morph_{key}": value for key, value in post_morph_component_stats.items()},
        **branch_stats,
    }
    with open(out_dir / "topology_prior_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
