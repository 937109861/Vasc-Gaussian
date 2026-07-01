import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.export_volume import make_slicer_display_volume, save_mhd_uint16, save_tiff_stack_uint16


class VolumeGeometry:
    def __init__(self, payload):
        self.nVoxel = np.asarray(payload["nVoxel"])
        self.dVoxel = np.asarray(payload["dVoxel"], dtype=np.float32)


def load_volume(path):
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.lib.npyio.NpzFile):
        for key in ("recon_volume", "volume", "prediction", "pred", "data"):
            if key in payload:
                return np.asarray(payload[key], dtype=np.float32)
        raise KeyError(f"No volume-like key found in npz file: {path}")
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        item = payload.item() if payload.shape == () else payload
        if isinstance(item, dict):
            for key in ("recon_volume", "volume", "prediction", "pred", "data"):
                if key in item:
                    return np.asarray(item[key], dtype=np.float32)
            raise KeyError(f"No volume-like key found in object npy file: {path}")
        return np.asarray(item, dtype=np.float32)
    return np.asarray(payload, dtype=np.float32)


def normalize_volume(volume, eps=1e-6):
    volume = np.asarray(volume, dtype=np.float32)
    vmin = float(volume.min())
    vmax = float(volume.max())
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.float32), vmin, vmax
    return ((volume - vmin) / max(vmax - vmin, eps)).astype(np.float32), vmin, vmax


def volume_distribution_stats(volume):
    volume = np.asarray(volume, dtype=np.float32)
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return {
            "finite_voxels": 0,
            "nonzero_voxels": 0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "p999": 0.0,
            "max": 0.0,
        }
    return {
        "finite_voxels": int(finite.size),
        "nonzero_voxels": int(np.count_nonzero(finite)),
        "p50": float(np.percentile(finite, 50.0)),
        "p90": float(np.percentile(finite, 90.0)),
        "p95": float(np.percentile(finite, 95.0)),
        "p99": float(np.percentile(finite, 99.0)),
        "p999": float(np.percentile(finite, 99.9)),
        "max": float(finite.max()),
    }


def save_preview_slices(volume, out_dir, basename, slab_width=15):
    preview_dir = Path(out_dir) / "preview_slices"
    preview_dir.mkdir(parents=True, exist_ok=True)
    mids = [size // 2 for size in volume.shape]
    center_slices = {
        "axial_z": volume[:, :, mids[2]],
        "coronal_y": volume[:, mids[1], :],
        "sagittal_x": volume[mids[0], :, :],
    }
    for name, image in center_slices.items():
        imageio.imwrite(
            preview_dir / f"{basename}_{name}_center.png",
            (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8),
        )

    axis_specs = {"sagittal_x": 0, "coronal_y": 1, "axial_z": 2}
    for name, axis in axis_specs.items():
        projection = np.max(volume, axis=axis)
        imageio.imwrite(
            preview_dir / f"{basename}_{name}_mip.png",
            (np.clip(projection, 0.0, 1.0) * 255).astype(np.uint8),
        )
        reduce_axes = tuple(index for index in range(3) if index != axis)
        best_index = int(np.argmax(np.sum(volume, axis=reduce_axes)))
        best_slice = np.take(volume, best_index, axis=axis)
        imageio.imwrite(
            preview_dir / f"{basename}_{name}_best_{best_index:03d}.png",
            (np.clip(best_slice, 0.0, 1.0) * 255).astype(np.uint8),
        )
        slab_width = max(1, int(slab_width))
        slab_start = max(0, best_index - slab_width // 2)
        slab_end = min(volume.shape[axis], slab_start + slab_width)
        slab = np.take(volume, range(slab_start, slab_end), axis=axis)
        slab_mip = np.max(slab, axis=axis)
        imageio.imwrite(
            preview_dir / f"{basename}_{name}_slab_mip_{slab_start:03d}_{slab_end - 1:03d}.png",
            (np.clip(slab_mip, 0.0, 1.0) * 255).astype(np.uint8),
        )


def seeded_component_mask(candidate_mask, seed_mask, min_component_voxels, min_seed_voxels, keep_largest):
    structure = ndimage.generate_binary_structure(3, 2)
    labels, num_labels = ndimage.label(candidate_mask, structure=structure)
    if num_labels == 0:
        return candidate_mask, {
            "num_components": 0,
            "kept_components": 0,
            "removed_voxels": int(candidate_mask.sum()),
            "largest_component_voxels": 0,
        }

    flat_labels = labels.reshape(-1)
    sizes = np.bincount(flat_labels, minlength=num_labels + 1)
    seed_counts = np.bincount(flat_labels[seed_mask.reshape(-1)], minlength=num_labels + 1)
    component_ids = np.arange(1, num_labels + 1)

    keep = np.zeros(num_labels + 1, dtype=bool)
    keep_ids = component_ids[
        (sizes[component_ids] >= int(min_component_voxels))
        & (seed_counts[component_ids] >= int(min_seed_voxels))
    ]

    if keep_largest > 0 and keep_ids.size > int(keep_largest):
        order = keep_ids[np.argsort(sizes[keep_ids])[::-1]]
        keep_ids = order[: int(keep_largest)]
    keep[keep_ids] = True
    cleaned = keep[labels]
    return cleaned, {
        "num_components": int(num_labels),
        "kept_components": int(keep_ids.size),
        "removed_voxels": int(candidate_mask.sum() - cleaned.sum()),
        "largest_component_voxels": int(sizes[component_ids].max()) if component_ids.size else 0,
        "min_component_voxels": int(min_component_voxels),
        "min_seed_voxels": int(min_seed_voxels),
    }


def _pick_npz_mask(payload, keys, prior_path):
    for key in keys:
        if key in payload:
            return payload[key].astype(bool), key
    raise KeyError(f"None of {keys} found in topology prior: {prior_path}")


def load_topology_prior_masks(prior_path, shape, roi_source="support_roi", roi_dilation=0):
    payload = np.load(prior_path)
    roi_key_map = {
        "support_roi": ("support_roi_mask", "M_candidate", "candidate_mask", "roi_mask"),
        "candidate": ("M_candidate", "candidate_mask", "support_roi_mask", "roi_mask"),
        "roi": ("roi_mask", "support_roi_mask", "M_candidate", "candidate_mask"),
        "vessel": ("M_high", "vessel_mask", "roi_mask"),
    }
    roi_keys = roi_key_map.get(str(roi_source), roi_key_map["support_roi"])
    roi_mask, roi_key = _pick_npz_mask(payload, roi_keys, prior_path)
    vessel_mask, vessel_key = _pick_npz_mask(payload, ("M_high", "vessel_mask", "roi_mask"), prior_path)

    if tuple(roi_mask.shape) != tuple(shape):
        raise ValueError(f"Topology ROI shape {roi_mask.shape} does not match volume shape {shape}")
    if tuple(vessel_mask.shape) != tuple(shape):
        raise ValueError(f"Topology vessel shape {vessel_mask.shape} does not match volume shape {shape}")

    if int(roi_dilation) > 0:
        structure = ndimage.generate_binary_structure(3, 2)
        roi_mask = ndimage.binary_dilation(roi_mask, structure=structure, iterations=int(roi_dilation))

    return {
        "roi_mask": roi_mask.astype(bool),
        "vessel_mask": vessel_mask.astype(bool),
        "roi_key": roi_key,
        "vessel_key": vessel_key,
    }


def apply_topology_prior_to_masks(candidate_mask, seed_mask, volume_for_mask, args):
    if not args.topology_prior_path:
        return candidate_mask, seed_mask, {}

    prior = load_topology_prior_masks(
        args.topology_prior_path,
        candidate_mask.shape,
        roi_source=args.topology_roi_source,
        roi_dilation=args.topology_roi_dilation,
    )
    roi_mask = prior["roi_mask"]
    vessel_mask = prior["vessel_mask"]
    keep_strong_threshold = float(args.topology_keep_strong_threshold)
    strong_mask = volume_for_mask >= keep_strong_threshold

    candidate_before = int(candidate_mask.sum())
    seed_before = int(seed_mask.sum())
    outside_roi_candidate = candidate_mask & ~roi_mask
    suppressed_candidate = outside_roi_candidate & ~strong_mask
    suppressed_seed = seed_mask & ~roi_mask & ~strong_mask

    filtered_candidate = candidate_mask & (roi_mask | strong_mask)
    filtered_seed = seed_mask & (roi_mask | strong_mask)

    stats = {
        "topology_prior_path": args.topology_prior_path,
        "topology_roi_source": args.topology_roi_source,
        "topology_roi_key": prior["roi_key"],
        "topology_vessel_key": prior["vessel_key"],
        "topology_roi_dilation": int(args.topology_roi_dilation),
        "topology_roi_voxels": int(roi_mask.sum()),
        "topology_vessel_voxels": int(vessel_mask.sum()),
        "topology_keep_strong_threshold": keep_strong_threshold,
        "topology_candidate_before": candidate_before,
        "topology_seed_before": seed_before,
        "topology_candidate_outside_roi_before": int(outside_roi_candidate.sum()),
        "topology_candidate_suppressed_outside_roi": int(suppressed_candidate.sum()),
        "topology_seed_suppressed_outside_roi": int(suppressed_seed.sum()),
        "topology_candidate_after": int(filtered_candidate.sum()),
        "topology_seed_after": int(filtered_seed.sum()),
        "topology_strong_outside_roi_kept": int((outside_roi_candidate & strong_mask).sum()),
    }
    return filtered_candidate, filtered_seed, stats


def apply_weak_vessel_bridge(candidate_mask, seed_mask, volume_for_mask, args):
    if not bool(args.topology_bridge_weak_vessels):
        return candidate_mask, {}

    if args.topology_prior_path:
        prior = load_topology_prior_masks(
            args.topology_prior_path,
            candidate_mask.shape,
            roi_source=args.topology_roi_source,
            roi_dilation=args.topology_roi_dilation,
        )
        roi_mask = prior["roi_mask"]
        roi_key = prior["roi_key"]
    else:
        roi_mask = np.ones_like(candidate_mask, dtype=bool)
        roi_key = "all"

    distance_to_candidate = ndimage.distance_transform_edt(~candidate_mask).astype(np.float32)
    weak_bridge = (
        roi_mask
        & ~candidate_mask
        & (volume_for_mask >= float(args.topology_bridge_low_threshold))
        & (volume_for_mask < float(args.low_threshold))
        & (distance_to_candidate <= float(args.topology_bridge_max_distance))
    )

    if int(args.topology_bridge_closing) > 0 and np.any(weak_bridge):
        structure = ndimage.generate_binary_structure(3, 2)
        weak_bridge = ndimage.binary_closing(
            weak_bridge,
            structure=structure,
            iterations=int(args.topology_bridge_closing),
        ) & roi_mask & ~candidate_mask

    bridged_candidate = candidate_mask | weak_bridge
    stats = {
        "topology_bridge_weak_vessels": True,
        "topology_bridge_roi_key": roi_key,
        "topology_bridge_low_threshold": float(args.topology_bridge_low_threshold),
        "topology_bridge_max_distance": float(args.topology_bridge_max_distance),
        "topology_bridge_closing": int(args.topology_bridge_closing),
        "topology_bridge_added_voxels": int(weak_bridge.sum()),
        "topology_bridge_candidate_before": int(candidate_mask.sum()),
        "topology_bridge_candidate_after": int(bridged_candidate.sum()),
    }
    return bridged_candidate, stats


def topology_component_filter(cleaned_mask, seed_mask, volume_for_mask, args):
    if not args.topology_prior_path or float(args.topology_component_min_prior_overlap) <= 0.0:
        return cleaned_mask, {}

    prior = load_topology_prior_masks(
        args.topology_prior_path,
        cleaned_mask.shape,
        roi_source=args.topology_roi_source,
        roi_dilation=args.topology_roi_dilation,
    )
    roi_mask = prior["roi_mask"]
    structure = ndimage.generate_binary_structure(3, 2)
    labels, num_labels = ndimage.label(cleaned_mask, structure=structure)
    if num_labels == 0:
        return cleaned_mask, {
            "topology_component_filter_enabled": True,
            "topology_component_filter_components": 0,
            "topology_component_filter_kept": 0,
            "topology_component_filter_removed": 0,
        }

    flat_labels = labels.reshape(-1)
    sizes = np.bincount(flat_labels, minlength=num_labels + 1)
    prior_counts = np.bincount(flat_labels[roi_mask.reshape(-1)], minlength=num_labels + 1)
    strong_component = np.zeros(num_labels + 1, dtype=bool)
    strong_threshold = float(args.topology_component_keep_strong_threshold)
    strong_labels = np.unique(labels[(cleaned_mask & (volume_for_mask >= strong_threshold))])
    strong_component[strong_labels] = True

    component_ids = np.arange(1, num_labels + 1)
    overlap = np.zeros(num_labels + 1, dtype=np.float32)
    valid = sizes > 0
    overlap[valid] = prior_counts[valid].astype(np.float32) / sizes[valid].astype(np.float32)
    keep_ids = component_ids[
        (overlap[component_ids] >= float(args.topology_component_min_prior_overlap))
        | (prior_counts[component_ids] >= int(args.topology_component_min_prior_voxels))
        | strong_component[component_ids]
    ]

    keep = np.zeros(num_labels + 1, dtype=bool)
    keep[keep_ids] = True
    filtered = keep[labels]
    removed = cleaned_mask & ~filtered
    stats = {
        "topology_component_filter_enabled": True,
        "topology_component_filter_components": int(num_labels),
        "topology_component_filter_kept": int(keep_ids.size),
        "topology_component_filter_removed": int(num_labels - keep_ids.size),
        "topology_component_filter_removed_voxels": int(removed.sum()),
        "topology_component_min_prior_overlap": float(args.topology_component_min_prior_overlap),
        "topology_component_min_prior_voxels": int(args.topology_component_min_prior_voxels),
        "topology_component_keep_strong_threshold": strong_threshold,
        "topology_component_filter_seed_voxels_removed": int((removed & seed_mask).sum()),
        "topology_component_filter_largest_removed_voxels": int(
            sizes[np.setdiff1d(component_ids, keep_ids)].max()
        ) if keep_ids.size < component_ids.size else 0,
    }
    return filtered, stats


def filter_floating_components(cleaned_mask, volume_for_mask, args):
    if not bool(args.remove_floating_components):
        return cleaned_mask, {}

    structure = ndimage.generate_binary_structure(3, 2)
    labels, num_labels = ndimage.label(cleaned_mask, structure=structure)
    if num_labels <= 1:
        return cleaned_mask, {
            "floating_component_filter_enabled": True,
            "floating_component_components": int(num_labels),
            "floating_component_kept": int(num_labels),
            "floating_component_removed": 0,
            "floating_component_removed_voxels": 0,
        }

    sizes = np.bincount(labels.reshape(-1), minlength=num_labels + 1)
    sizes[0] = 0
    component_ids = np.arange(1, num_labels + 1)
    main_id = int(np.argmax(sizes))
    main_mask = labels == main_id

    keep = np.zeros(num_labels + 1, dtype=bool)
    keep[main_id] = True

    distance_to_main = None
    if float(args.floating_component_keep_distance) > 0.0:
        distance_to_main = ndimage.distance_transform_edt(~main_mask).astype(np.float32)

    kept_by_size = []
    kept_by_strength = []
    kept_by_distance = []
    removed_ids = []
    for component_id in component_ids:
        if component_id == main_id:
            continue
        component_mask = labels == component_id
        component_size = int(sizes[component_id])
        component_max = float(volume_for_mask[component_mask].max()) if component_size > 0 else 0.0
        component_min_distance = (
            float(distance_to_main[component_mask].min())
            if distance_to_main is not None and component_size > 0
            else None
        )

        keep_component = False
        if component_size >= int(args.floating_component_keep_min_voxels):
            keep_component = True
            kept_by_size.append(int(component_id))
        if component_max >= float(args.floating_component_keep_strong_threshold):
            keep_component = True
            kept_by_strength.append(int(component_id))
        if (
            component_min_distance is not None
            and component_min_distance <= float(args.floating_component_keep_distance)
        ):
            keep_component = True
            kept_by_distance.append(int(component_id))

        keep[component_id] = keep_component
        if not keep_component:
            removed_ids.append(int(component_id))

    filtered = keep[labels]
    removed = cleaned_mask & ~filtered
    removed_sizes = sizes[removed_ids] if removed_ids else np.asarray([], dtype=np.int64)
    stats = {
        "floating_component_filter_enabled": True,
        "floating_component_components": int(num_labels),
        "floating_component_main_voxels": int(sizes[main_id]),
        "floating_component_kept": int(keep[component_ids].sum()),
        "floating_component_removed": int(len(removed_ids)),
        "floating_component_removed_voxels": int(removed.sum()),
        "floating_component_keep_min_voxels": int(args.floating_component_keep_min_voxels),
        "floating_component_keep_strong_threshold": float(args.floating_component_keep_strong_threshold),
        "floating_component_keep_distance": float(args.floating_component_keep_distance),
        "floating_component_kept_by_size": len(kept_by_size),
        "floating_component_kept_by_strength": len(kept_by_strength),
        "floating_component_kept_by_distance": len(kept_by_distance),
        "floating_component_largest_removed_voxels": int(removed_sizes.max()) if removed_sizes.size else 0,
    }
    return filtered, stats


def filter_weak_tail_voxels(cleaned_mask, seed_mask, volume_for_mask, args):
    if not bool(args.suppress_weak_tails):
        return cleaned_mask, {}

    protected_mask = seed_mask.astype(bool)
    prior_stats = {}
    if args.topology_prior_path:
        prior = load_topology_prior_masks(
            args.topology_prior_path,
            cleaned_mask.shape,
            roi_source=args.topology_roi_source,
            roi_dilation=args.topology_roi_dilation,
        )
        vessel_mask = prior["vessel_mask"]
        protected_mask = protected_mask | vessel_mask
        prior_stats = {
            "weak_tail_topology_vessel_key": prior["vessel_key"],
            "weak_tail_protected_prior_voxels": int(vessel_mask.sum()),
        }

    if not np.any(protected_mask):
        return cleaned_mask, {
            "weak_tail_filter_enabled": True,
            "weak_tail_removed_voxels": 0,
            "weak_tail_reason": "empty_protected_mask",
            **prior_stats,
        }

    distance_to_protected = ndimage.distance_transform_edt(~protected_mask).astype(np.float32)
    weak_tail_mask = (
        cleaned_mask
        & (volume_for_mask <= float(args.weak_tail_max_raw))
        & (distance_to_protected >= float(args.weak_tail_min_distance))
    )

    if int(args.weak_tail_opening) > 0 and np.any(weak_tail_mask):
        structure = ndimage.generate_binary_structure(3, 2)
        weak_tail_mask = ndimage.binary_opening(
            weak_tail_mask,
            structure=structure,
            iterations=int(args.weak_tail_opening),
        )

    filtered = cleaned_mask & ~weak_tail_mask
    stats = {
        "weak_tail_filter_enabled": True,
        "weak_tail_removed_voxels": int(weak_tail_mask.sum()),
        "weak_tail_max_raw": float(args.weak_tail_max_raw),
        "weak_tail_min_distance": float(args.weak_tail_min_distance),
        "weak_tail_opening": int(args.weak_tail_opening),
        "weak_tail_candidate_voxels_before_opening": int(
            (
                cleaned_mask
                & (volume_for_mask <= float(args.weak_tail_max_raw))
                & (distance_to_protected >= float(args.weak_tail_min_distance))
            ).sum()
        ),
        **prior_stats,
    }
    return filtered, stats


def apply_topology_display_adjustment(display_volume, cleaned_mask, volume_for_mask, args):
    if not args.topology_prior_path:
        return display_volume, {}
    if (
        float(args.topology_inside_weak_boost) == 1.0
        and float(args.topology_inside_weak_floor) <= 0.0
        and float(args.topology_outside_weak_scale) == 1.0
    ):
        return display_volume, {}

    prior = load_topology_prior_masks(
        args.topology_prior_path,
        cleaned_mask.shape,
        roi_source=args.topology_roi_source,
        roi_dilation=args.topology_roi_dilation,
    )
    roi_mask = prior["roi_mask"]
    vessel_mask = prior["vessel_mask"]

    adjusted = display_volume.astype(np.float32, copy=True)
    inside_weak = (
        cleaned_mask
        & roi_mask
        & (volume_for_mask >= float(args.topology_inside_weak_min_raw))
        & (volume_for_mask <= float(args.topology_inside_weak_max_raw))
    )
    outside_weak = (
        cleaned_mask
        & ~roi_mask
        & ~vessel_mask
        & (volume_for_mask <= float(args.topology_outside_weak_max_raw))
    )

    if np.any(inside_weak):
        values = adjusted[inside_weak] * float(args.topology_inside_weak_boost)
        if float(args.topology_inside_weak_floor) > 0.0:
            values = np.maximum(values, float(args.topology_inside_weak_floor))
        adjusted[inside_weak] = np.clip(values, 0.0, 1.0)

    if np.any(outside_weak):
        adjusted[outside_weak] = np.clip(
            adjusted[outside_weak] * float(args.topology_outside_weak_scale),
            0.0,
            1.0,
        )

    stats = {
        "topology_display_adjustment_enabled": True,
        "topology_inside_weak_voxels": int(inside_weak.sum()),
        "topology_outside_weak_voxels": int(outside_weak.sum()),
        "topology_inside_weak_min_raw": float(args.topology_inside_weak_min_raw),
        "topology_inside_weak_max_raw": float(args.topology_inside_weak_max_raw),
        "topology_inside_weak_boost": float(args.topology_inside_weak_boost),
        "topology_inside_weak_floor": float(args.topology_inside_weak_floor),
        "topology_outside_weak_max_raw": float(args.topology_outside_weak_max_raw),
        "topology_outside_weak_scale": float(args.topology_outside_weak_scale),
        "topology_display_mean_before": float(display_volume[cleaned_mask].mean()) if np.any(cleaned_mask) else 0.0,
        "topology_display_mean_after": float(adjusted[cleaned_mask].mean()) if np.any(cleaned_mask) else 0.0,
    }
    return adjusted, stats


def main():
    parser = argparse.ArgumentParser(
        description="Export a vessel Slicer volume with hysteresis thresholding for weak connected vessels."
    )
    parser.add_argument("--input", required=True, help="Input recon_volume.npy.")
    parser.add_argument("--source_path", required=True, help="Dataset pickle containing CT geometry.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--basename", default="recon_volume_vessel_slicer")
    parser.add_argument("--low_threshold", type=float, default=0.016)
    parser.add_argument("--high_threshold", type=float, default=0.040)
    parser.add_argument("--min_component_voxels", type=int, default=160)
    parser.add_argument("--min_seed_voxels", type=int, default=12)
    parser.add_argument("--keep_largest", type=int, default=0)
    parser.add_argument("--closing", type=int, default=1)
    parser.add_argument("--opening", type=int, default=0)
    parser.add_argument("--smooth_sigma", type=float, default=0.10)
    parser.add_argument("--percentile_low", type=float, default=0.5)
    parser.add_argument("--percentile_high", type=float, default=99.0)
    parser.add_argument("--gamma", type=float, default=0.58)
    parser.add_argument(
        "--vessel_floor",
        type=float,
        default=0.0,
        help="Minimum display intensity inside the final vessel mask. This strengthens weak vessels without adding new voxels.",
    )
    parser.add_argument(
        "--vessel_boost",
        type=float,
        default=1.0,
        help="Multiplicative display boost inside the final vessel mask after percentile/gamma mapping.",
    )
    parser.add_argument("--preview_slab_width", type=int, default=15)
    parser.add_argument("--no_tiff", action="store_true")
    parser.add_argument(
        "--topology_prior_path",
        default="",
        help=(
            "Optional PriorNet-derived topology_prior.npz. When set, only weak candidate voxels outside "
            "the prior ROI are suppressed; strong existing vessels are kept."
        ),
    )
    parser.add_argument(
        "--topology_roi_source",
        default="support_roi",
        choices=("support_roi", "candidate", "roi", "vessel"),
        help="Mask source used as the permissive vessel candidate region inside topology_prior.npz.",
    )
    parser.add_argument("--topology_roi_dilation", type=int, default=0)
    parser.add_argument(
        "--topology_keep_strong_threshold",
        type=float,
        default=0.12,
        help="Voxels stronger than this are kept even outside the topology ROI.",
    )
    parser.add_argument(
        "--topology_component_min_prior_overlap",
        type=float,
        default=0.0,
        help="If >0, remove connected components with too little overlap with the topology ROI.",
    )
    parser.add_argument("--topology_component_min_prior_voxels", type=int, default=8)
    parser.add_argument(
        "--topology_component_keep_strong_threshold",
        type=float,
        default=0.16,
        help="Connected components with at least one voxel above this value are kept despite weak prior overlap.",
    )
    parser.add_argument(
        "--topology_bridge_weak_vessels",
        action="store_true",
        help="Add low-density voxels near existing candidates inside the topology ROI to bridge broken weak vessels.",
    )
    parser.add_argument(
        "--topology_bridge_low_threshold",
        type=float,
        default=0.016,
        help="Raw lower threshold for optional weak-vessel bridge voxels.",
    )
    parser.add_argument(
        "--topology_bridge_max_distance",
        type=float,
        default=2.0,
        help="Maximum voxel distance from existing candidate mask for optional weak-vessel bridge voxels.",
    )
    parser.add_argument(
        "--topology_bridge_closing",
        type=int,
        default=0,
        help="Optional binary closing iterations for weak-vessel bridge voxels before adding them.",
    )
    parser.add_argument(
        "--topology_inside_weak_min_raw",
        type=float,
        default=0.024,
        help="Lower raw-density bound for topology-guided weak-vessel display boost.",
    )
    parser.add_argument(
        "--topology_inside_weak_max_raw",
        type=float,
        default=0.11,
        help="Upper raw-density bound for topology-guided weak-vessel display boost.",
    )
    parser.add_argument(
        "--topology_inside_weak_boost",
        type=float,
        default=1.0,
        help="Display-only boost for weak voxels inside the topology ROI. Default keeps old behavior.",
    )
    parser.add_argument(
        "--topology_inside_weak_floor",
        type=float,
        default=0.0,
        help="Display-only minimum intensity for boosted weak voxels inside the topology ROI.",
    )
    parser.add_argument(
        "--topology_outside_weak_max_raw",
        type=float,
        default=0.09,
        help="Raw-density upper bound for topology-guided outside-ROI weak suppression.",
    )
    parser.add_argument(
        "--topology_outside_weak_scale",
        type=float,
        default=1.0,
        help="Display-only scale for weak voxels outside the topology ROI. Default keeps old behavior.",
    )
    parser.add_argument(
        "--remove_floating_components",
        action="store_true",
        help="After hysteresis cleanup, remove small disconnected components away from the largest vessel tree.",
    )
    parser.add_argument(
        "--floating_component_keep_min_voxels",
        type=int,
        default=3000,
        help="Disconnected components at least this large are kept.",
    )
    parser.add_argument(
        "--floating_component_keep_strong_threshold",
        type=float,
        default=0.18,
        help="Disconnected components with max raw density above this are kept.",
    )
    parser.add_argument(
        "--floating_component_keep_distance",
        type=float,
        default=0.0,
        help="If >0, disconnected components within this voxel distance of the largest component are kept.",
    )
    parser.add_argument(
        "--suppress_weak_tails",
        action="store_true",
        help="Remove low-density tail voxels far from high-confidence seeds/topology vessel mask.",
    )
    parser.add_argument(
        "--weak_tail_max_raw",
        type=float,
        default=0.055,
        help="Low-density tail voxels at or below this raw value are candidates for removal.",
    )
    parser.add_argument(
        "--weak_tail_min_distance",
        type=float,
        default=3.0,
        help="Candidate tail voxels at least this voxel distance from protected seeds are removed.",
    )
    parser.add_argument(
        "--weak_tail_opening",
        type=int,
        default=0,
        help="Optional morphology opening iterations on weak-tail candidates before removal.",
    )
    args = parser.parse_args()

    raw = load_volume(args.input)
    volume, raw_min, raw_max = normalize_volume(raw)
    if args.smooth_sigma > 0.0:
        volume_for_mask = ndimage.gaussian_filter(volume, sigma=float(args.smooth_sigma))
    else:
        volume_for_mask = volume

    candidate_mask = volume_for_mask >= float(args.low_threshold)
    seed_mask = volume_for_mask >= float(args.high_threshold)
    structure = ndimage.generate_binary_structure(3, 2)
    if args.closing > 0:
        candidate_mask = ndimage.binary_closing(candidate_mask, structure=structure, iterations=int(args.closing))
    if args.opening > 0:
        candidate_mask = ndimage.binary_opening(candidate_mask, structure=structure, iterations=int(args.opening))

    candidate_mask, seed_mask, topology_mask_stats = apply_topology_prior_to_masks(
        candidate_mask,
        seed_mask,
        volume_for_mask,
        args,
    )
    candidate_mask, topology_bridge_stats = apply_weak_vessel_bridge(
        candidate_mask,
        seed_mask,
        volume_for_mask,
        args,
    )

    cleaned_mask, component_stats = seeded_component_mask(
        candidate_mask,
        seed_mask,
        min_component_voxels=args.min_component_voxels,
        min_seed_voxels=args.min_seed_voxels,
        keep_largest=args.keep_largest,
    )
    cleaned_mask, topology_component_stats = topology_component_filter(
        cleaned_mask,
        seed_mask,
        volume_for_mask,
        args,
    )
    cleaned_mask, floating_component_stats = filter_floating_components(
        cleaned_mask,
        volume_for_mask,
        args,
    )
    cleaned_mask, weak_tail_stats = filter_weak_tail_voxels(
        cleaned_mask,
        seed_mask,
        volume_for_mask,
        args,
    )
    if not cleaned_mask.any():
        os.makedirs(args.out_dir, exist_ok=True)
        failure_stats = {
            "input": args.input,
            "source_path": args.source_path,
            "shape": list(volume.shape),
            "raw_min": raw_min,
            "raw_max": raw_max,
            "low_threshold": float(args.low_threshold),
            "high_threshold": float(args.high_threshold),
            "candidate_voxels": int(candidate_mask.sum()),
            "seed_voxels": int(seed_mask.sum()),
            "closing": int(args.closing),
            "opening": int(args.opening),
            "smooth_sigma": float(args.smooth_sigma),
            "raw_distribution": volume_distribution_stats(raw),
            "normalized_distribution": volume_distribution_stats(volume),
            **topology_mask_stats,
            **topology_bridge_stats,
            **component_stats,
            **topology_component_stats,
            **floating_component_stats,
            **weak_tail_stats,
        }
        stats_path = os.path.join(args.out_dir, args.basename + "_empty_stats.json")
        with open(stats_path, "w", encoding="utf-8") as handle:
            json.dump(failure_stats, handle, indent=2)
        print(f"Saved empty-mask diagnostics: {stats_path}")
        raise ValueError("No voxels remain; lower thresholds or component filters.")

    display_volume, display_stats = make_slicer_display_volume(
        volume,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        gamma=args.gamma,
        sample_mask=cleaned_mask,
    )
    display_volume[~cleaned_mask] = 0.0
    if args.vessel_boost != 1.0 or args.vessel_floor > 0.0:
        vessel_values = display_volume[cleaned_mask] * float(args.vessel_boost)
        if args.vessel_floor > 0.0:
            vessel_values = np.maximum(vessel_values, float(args.vessel_floor))
        display_volume[cleaned_mask] = np.clip(vessel_values, 0.0, 1.0)
    display_volume, topology_display_stats = apply_topology_display_adjustment(
        display_volume,
        cleaned_mask,
        volume_for_mask,
        args,
    )

    with open(args.source_path, "rb") as handle:
        geometry = VolumeGeometry(pickle.load(handle))

    os.makedirs(args.out_dir, exist_ok=True)
    npy_path = os.path.join(args.out_dir, args.basename + ".npy")
    np.save(npy_path, display_volume.astype(np.float32))
    mhd_path, raw_path = save_mhd_uint16(display_volume, geometry, args.out_dir, args.basename)
    save_preview_slices(display_volume, args.out_dir, args.basename, slab_width=args.preview_slab_width)

    tiff_path = None
    if not args.no_tiff:
        tiff_path = save_tiff_stack_uint16(display_volume, args.out_dir, args.basename)

    stats = {
        "input": args.input,
        "source_path": args.source_path,
        "shape": list(volume.shape),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "low_threshold": float(args.low_threshold),
        "high_threshold": float(args.high_threshold),
        "candidate_voxels": int(candidate_mask.sum()),
        "seed_voxels": int(seed_mask.sum()),
        "visible_voxels": int(cleaned_mask.sum()),
        "visible_ratio": float(cleaned_mask.mean()),
        "closing": int(args.closing),
        "opening": int(args.opening),
        "smooth_sigma": float(args.smooth_sigma),
        "vessel_floor": float(args.vessel_floor),
        "vessel_boost": float(args.vessel_boost),
        **topology_mask_stats,
        **topology_bridge_stats,
        **component_stats,
        **topology_component_stats,
        **floating_component_stats,
        **weak_tail_stats,
        **display_stats,
        **topology_display_stats,
        "npy": npy_path,
        "mhd": mhd_path,
        "raw": raw_path,
        "tiff": tiff_path,
    }
    with open(os.path.join(args.out_dir, args.basename + "_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(f"Saved Slicer MetaImage: {mhd_path}")
    print(f"Saved Slicer RAW: {raw_path}")
    print(f"Saved previews: {os.path.join(args.out_dir, 'preview_slices')}")
    print(f"Visible voxel ratio: {stats['visible_ratio']:.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
