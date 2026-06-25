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

    cleaned_mask, component_stats = seeded_component_mask(
        candidate_mask,
        seed_mask,
        min_component_voxels=args.min_component_voxels,
        min_seed_voxels=args.min_seed_voxels,
        keep_largest=args.keep_largest,
    )
    if not cleaned_mask.any():
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
        **component_stats,
        **display_stats,
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
