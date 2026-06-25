import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.export_volume import (
    make_slicer_display_volume,
    save_mhd_uint16,
    save_tiff_stack_uint16,
)


class VolumeGeometry:
    def __init__(self, payload):
        self.nVoxel = np.asarray(payload["nVoxel"])
        self.dVoxel = np.asarray(payload["dVoxel"], dtype=np.float32)


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

    axis_specs = {
        "sagittal_x": 0,
        "coronal_y": 1,
        "axial_z": 2,
    }
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
            preview_dir
            / f"{basename}_{name}_slab_mip_{slab_start:03d}_{slab_end - 1:03d}.png",
            (np.clip(slab_mip, 0.0, 1.0) * 255).astype(np.uint8),
        )


def remove_small_components(mask, min_component_voxels):
    min_component_voxels = int(min_component_voxels)
    mask = np.asarray(mask, dtype=bool)
    if min_component_voxels <= 1:
        return mask, 0, 0

    shape = mask.shape
    visited = np.zeros(shape, dtype=bool)
    filtered = np.zeros(shape, dtype=bool)
    component_count = 0
    removed_count = 0
    foreground = np.argwhere(mask)
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if dx != 0 or dy != 0 or dz != 0
    ]

    for seed_array in foreground:
        seed = tuple(int(value) for value in seed_array)
        if visited[seed]:
            continue

        component_count += 1
        visited[seed] = True
        stack = [seed]
        component = []
        while stack:
            x, y, z = stack.pop()
            component.append((x, y, z))
            for dx, dy, dz in offsets:
                nx, ny, nz = x + dx, y + dy, z + dz
                if (
                    0 <= nx < shape[0]
                    and 0 <= ny < shape[1]
                    and 0 <= nz < shape[2]
                    and mask[nx, ny, nz]
                    and not visited[nx, ny, nz]
                ):
                    visited[nx, ny, nz] = True
                    stack.append((nx, ny, nz))

        if len(component) >= min_component_voxels:
            component_array = np.asarray(component, dtype=np.intp)
            filtered[
                component_array[:, 0],
                component_array[:, 1],
                component_array[:, 2],
            ] = True
        else:
            removed_count += 1

    return filtered, component_count, removed_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert an exported sparse vessel volume into a foreground-aware 3D Slicer volume."
    )
    parser.add_argument("--input", required=True, help="Existing recon_volume.npy.")
    parser.add_argument("--source_path", required=True, help="Dataset pickle containing CT geometry.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--basename", default="recon_volume_vessel_slicer")
    parser.add_argument(
        "--min_value",
        type=float,
        default=0.01,
        help="Minimum displayed vessel density.",
    )
    parser.add_argument("--percentile_low", type=float, default=1.0)
    parser.add_argument("--percentile_high", type=float, default=99.5)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument(
        "--preserve_below_threshold",
        action="store_true",
        help="Keep values below min_value. Normally disabled because weak background Gaussians become visible noise.",
    )
    parser.add_argument(
        "--min_component_voxels",
        type=int,
        default=32,
        help="Remove thresholded 26-connected components smaller than this many voxels. 0 disables filtering.",
    )
    parser.add_argument("--preview_slab_width", type=int, default=15)
    parser.add_argument("--no_tiff", action="store_true")
    args = parser.parse_args()

    volume = np.load(args.input).astype(np.float32)
    with open(args.source_path, "rb") as handle:
        geometry = VolumeGeometry(pickle.load(handle))

    threshold_mask = volume > float(args.min_value)
    sample_mask, component_count, removed_component_count = remove_small_components(
        threshold_mask,
        min_component_voxels=args.min_component_voxels,
    )
    if not sample_mask.any():
        raise ValueError(
            "No voxels remain after threshold/component filtering; lower "
            "--min_value or --min_component_voxels."
        )

    display_volume, display_stats = make_slicer_display_volume(
        volume,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        gamma=args.gamma,
        sample_mask=sample_mask,
    )
    if not args.preserve_below_threshold:
        display_volume[~sample_mask] = 0.0

    os.makedirs(args.out_dir, exist_ok=True)
    npy_path = os.path.join(args.out_dir, args.basename + ".npy")
    np.save(npy_path, display_volume.astype(np.float32))
    mhd_path, raw_path = save_mhd_uint16(
        display_volume,
        geometry,
        args.out_dir,
        args.basename,
    )
    save_preview_slices(
        display_volume,
        args.out_dir,
        args.basename,
        slab_width=args.preview_slab_width,
    )

    tiff_path = None
    if not args.no_tiff:
        tiff_path = save_tiff_stack_uint16(
            display_volume,
            args.out_dir,
            args.basename,
        )

    stats = {
        "input": args.input,
        "source_path": args.source_path,
        "shape": list(volume.shape),
        "min_value": float(args.min_value),
        "threshold_voxels": int(threshold_mask.sum()),
        "visible_voxels": int(sample_mask.sum()),
        "visible_ratio": float(sample_mask.mean()),
        "preserve_below_threshold": bool(args.preserve_below_threshold),
        "min_component_voxels": int(args.min_component_voxels),
        "component_count": int(component_count),
        "removed_component_count": int(removed_component_count),
        "preview_slab_width": int(args.preview_slab_width),
        **display_stats,
        "npy": npy_path,
        "mhd": mhd_path,
        "raw": raw_path,
        "tiff": tiff_path,
    }
    stats_path = os.path.join(args.out_dir, args.basename + "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(f"Saved Slicer MetaImage: {mhd_path}")
    print(f"Saved Slicer RAW: {raw_path}")
    print(f"Saved previews: {os.path.join(args.out_dir, 'preview_slices')}")
    print(f"Visible voxel ratio: {stats['visible_ratio']:.6f}")


if __name__ == "__main__":
    main()
