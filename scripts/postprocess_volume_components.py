import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy import ndimage


def normalize_volume(volume, eps=1e-6):
    volume = np.asarray(volume, dtype=np.float32)
    vmin = float(volume.min())
    vmax = float(volume.max())
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.float32), vmin, vmax
    return ((volume - vmin) / max(vmax - vmin, eps)).astype(np.float32), vmin, vmax


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


def keep_components(mask, min_size=0, keep_largest=0):
    structure = ndimage.generate_binary_structure(3, 2)
    labels, num_labels = ndimage.label(mask, structure=structure)
    if num_labels == 0:
        return mask, {"num_components": 0, "kept_components": 0, "removed_voxels": int(mask.sum())}

    counts = np.bincount(labels.reshape(-1))
    component_ids = np.arange(1, num_labels + 1)
    component_sizes = counts[component_ids]

    keep = np.ones(num_labels + 1, dtype=bool)
    keep[0] = False
    if min_size > 0:
        keep[component_ids[component_sizes < int(min_size)]] = False
    if keep_largest > 0:
        order = component_ids[np.argsort(component_sizes)[::-1]]
        allowed = set(int(idx) for idx in order[: int(keep_largest)])
        for idx in component_ids:
            if int(idx) not in allowed:
                keep[idx] = False

    cleaned = keep[labels]
    stats = {
        "num_components": int(num_labels),
        "kept_components": int(np.sum(keep[1:])),
        "removed_voxels": int(mask.sum() - cleaned.sum()),
        "largest_component_voxels": int(component_sizes.max()) if component_sizes.size else 0,
    }
    return cleaned, stats


def save_preview_slices(volume, out_dir):
    out_dir = Path(out_dir)
    preview_dir = out_dir / "preview_slices"
    preview_dir.mkdir(parents=True, exist_ok=True)
    mids = [size // 2 for size in volume.shape]
    views = {
        "axial_z.png": volume[:, :, mids[2]],
        "coronal_y.png": volume[:, mids[1], :],
        "sagittal_x.png": volume[mids[0], :, :],
    }
    for name, image in views.items():
        imageio.imwrite(preview_dir / name, (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="Clean small blocky components from an exported CT volume.")
    parser.add_argument("--input", required=True, help="Input recon_volume.npy.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.18, help="Foreground threshold after normalization.")
    parser.add_argument("--min_component_size", type=int, default=200, help="Remove connected components below this voxel count.")
    parser.add_argument("--keep_largest", type=int, default=0, help="If >0, keep only the largest K components.")
    parser.add_argument("--mask_dilation", type=int, default=0)
    parser.add_argument("--mask_closing", type=int, default=1)
    parser.add_argument("--smooth_sigma", type=float, default=0.35)
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_volume(args.input)
    volume, raw_min, raw_max = normalize_volume(raw)
    mask = volume >= float(args.threshold)

    structure = ndimage.generate_binary_structure(3, 2)
    if args.mask_closing > 0:
        mask = ndimage.binary_closing(mask, structure=structure, iterations=int(args.mask_closing))
    if args.mask_dilation > 0:
        mask = ndimage.binary_dilation(mask, structure=structure, iterations=int(args.mask_dilation))

    cleaned_mask, component_stats = keep_components(
        mask,
        min_size=args.min_component_size,
        keep_largest=args.keep_largest,
    )

    cleaned = volume * cleaned_mask.astype(np.float32)
    if args.smooth_sigma > 0.0:
        smoothed = ndimage.gaussian_filter(cleaned, sigma=float(args.smooth_sigma))
        mask_float = ndimage.gaussian_filter(cleaned_mask.astype(np.float32), sigma=float(args.smooth_sigma))
        cleaned = np.where(mask_float > 1e-6, smoothed, 0.0)
    cleaned, clean_min, clean_max = normalize_volume(cleaned)
    if args.gamma != 1.0:
        cleaned = np.power(np.clip(cleaned, 0.0, 1.0), float(args.gamma)).astype(np.float32)

    out_path = out_dir / "recon_volume_component_clean.npy"
    np.save(out_path, cleaned.astype(np.float32))
    save_preview_slices(cleaned, out_dir)

    stats = {
        "input": str(args.input),
        "output": str(out_path),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "clean_min_before_final_norm": clean_min,
        "clean_max_before_final_norm": clean_max,
        "threshold": float(args.threshold),
        "min_component_size": int(args.min_component_size),
        "keep_largest": int(args.keep_largest),
        "mask_dilation": int(args.mask_dilation),
        "mask_closing": int(args.mask_closing),
        "smooth_sigma": float(args.smooth_sigma),
        "gamma": float(args.gamma),
        "foreground_voxels_before": int(mask.sum()),
        "foreground_voxels_after": int(cleaned_mask.sum()),
        **component_stats,
    }
    with open(out_dir / "component_clean_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
