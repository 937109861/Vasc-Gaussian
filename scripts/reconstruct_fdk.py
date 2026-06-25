import argparse
import json
import pickle
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def normalize_volume(volume):
    volume = np.asarray(volume, dtype=np.float32)
    vmin = float(volume.min())
    vmax = float(volume.max())
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.float32), vmin, vmax
    return ((volume - vmin) / (vmax - vmin)).astype(np.float32), vmin, vmax


def save_preview_slices(volume, out_dir):
    out_dir = Path(out_dir)
    slices_dir = out_dir / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)
    volume_norm, _, _ = normalize_volume(volume)
    mids = [size // 2 for size in volume_norm.shape]
    views = {
        "axial_z.png": volume_norm[:, :, mids[2]],
        "coronal_y.png": volume_norm[:, mids[1], :],
        "sagittal_x.png": volume_norm[mids[0], :, :],
    }
    for name, image in views.items():
        imageio.imwrite(slices_dir / name, (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))


def build_tigre_geometry(data):
    try:
        import tigre
    except ImportError as exc:
        raise ImportError(
            "TIGRE is required for FDK reconstruction. Install it on the server, "
            "then rerun this script."
        ) from exc

    geo = tigre.geometry(mode="cone", default=False)
    geo.DSD = float(data["DSD"])
    geo.DSO = float(data["DSO"])
    geo.nDetector = np.asarray(data["nDetector"], dtype=np.int32)
    geo.dDetector = np.asarray(data["dDetector"], dtype=np.float32)
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.nVoxel = np.asarray(data["nVoxel"], dtype=np.int32)
    geo.dVoxel = np.asarray(data["dVoxel"], dtype=np.float32)
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    geo.offOrigin = np.asarray(data.get("offOrigin", [0, 0, 0]), dtype=np.float32)
    geo.offDetector = np.asarray(data.get("offDetector", [0, 0]), dtype=np.float32)
    geo.accuracy = float(data.get("accuracy", 0.5) or 0.5)
    geo.mode = data.get("mode", "cone") or "cone"
    geo.filter = data.get("filter", "ram_lak") or "ram_lak"
    return geo


def load_projection_block(data, split):
    if split not in data:
        raise KeyError(f"Pickle does not contain split '{split}'")
    block = data[split]
    projections = np.asarray(block["projections"], dtype=np.float32)
    angles = np.asarray(block["angles"], dtype=np.float32)
    if projections.shape[0] != angles.shape[0]:
        raise ValueError(
            f"Projection count {projections.shape[0]} does not match angle count {angles.shape[0]}"
        )
    return projections, angles


def main():
    parser = argparse.ArgumentParser(description="Reconstruct a coarse FDK volume from an X-Gaussian pickle.")
    parser.add_argument("-s", "--source", required=True, help="Input X-Gaussian .pickle file.")
    parser.add_argument("-o", "--out", required=True, help="Output .npy path for the FDK volume.")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Projection split used for FDK.")
    parser.add_argument("--transpose_detector", action="store_true", help="Swap detector H/W before calling TIGRE.")
    parser.add_argument("--preview", action="store_true", help="Save middle slice PNG previews next to the output.")
    args = parser.parse_args()

    try:
        import tigre.algorithms as algs
    except ImportError as exc:
        raise ImportError(
            "TIGRE is required for FDK reconstruction. Install it on the server, "
            "then rerun this script."
        ) from exc

    source = Path(args.source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(source, "rb") as handle:
        data = pickle.load(handle)

    projections, angles = load_projection_block(data, args.split)
    if args.transpose_detector:
        projections = np.transpose(projections, (0, 2, 1)).copy()

    geo = build_tigre_geometry(data)
    volume = algs.fdk(projections, geo, angles).astype(np.float32)
    np.save(out, volume)

    stats = {
        "source": str(source),
        "out": str(out),
        "split": args.split,
        "num_projections": int(projections.shape[0]),
        "projection_shape": list(projections.shape[1:]),
        "angles_min": float(angles.min()),
        "angles_max": float(angles.max()),
        "volume_shape": list(volume.shape),
        "volume_min": float(volume.min()),
        "volume_max": float(volume.max()),
        "transpose_detector": bool(args.transpose_detector),
    }
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    if args.preview:
        save_preview_slices(volume, out.parent)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
