import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from priors.volume_prior import get_voxels, load_xray_pickle


def load_gaussian_xyz(path):
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)


def grid_bounds(mask_shape, geometry_pickle=None):
    if geometry_pickle:
        _, geometry = load_xray_pickle(geometry_pickle)
        positions = get_voxels(geometry)
        return positions[0, 0, 0].astype(np.float32), positions[-1, -1, -1].astype(np.float32)
    return np.full(3, -1.0, dtype=np.float32), np.full(3, 1.0, dtype=np.float32)


def xyz_to_indices(xyz, mask_shape, grid_min, grid_max):
    shape = np.asarray(mask_shape, dtype=np.int64)
    denom = np.maximum(grid_max - grid_min, 1e-8)
    normalized = (xyz - grid_min[None, :]) / denom[None, :]
    indices = np.rint(normalized * (shape[None, :] - 1)).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < shape[None, :]), axis=1)
    indices = np.clip(indices, 0, shape[None, :] - 1)
    return indices, valid


def classify_gaussians(xyz, roi_mask, occupancy_mask, grid_min, grid_max):
    indices, valid = xyz_to_indices(xyz, roi_mask.shape, grid_min, grid_max)
    roi = np.zeros(xyz.shape[0], dtype=bool)
    occupancy = np.zeros(xyz.shape[0], dtype=bool)
    roi[valid] = roi_mask[indices[valid, 0], indices[valid, 1], indices[valid, 2]]
    occupancy[valid] = occupancy_mask[indices[valid, 0], indices[valid, 1], indices[valid, 2]]
    return indices, valid, roi, occupancy


def projection(mask, axis):
    return mask.max(axis=axis).astype(np.float32)


def scatter_projection(ax, indices, valid, region_mask, dims, shape, title):
    ax.imshow(np.zeros((shape[dims[1]], shape[dims[0]])), cmap="gray", origin="lower")
    selected = valid & region_mask
    if selected.any():
        ax.scatter(indices[selected, dims[0]], indices[selected, dims[1]], s=1, c="#d62728", alpha=0.35)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def save_prior_panels(roi_mask, occupancy_mask, out_dir):
    views = [
        ("axial", 2, (0, 1)),
        ("coronal", 1, (0, 2)),
        ("sagittal", 0, (1, 2)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for col, (name, axis, _) in enumerate(views):
        axes[0, col].imshow(projection(occupancy_mask, axis), cmap="gray", origin="lower")
        axes[0, col].set_title(f"{name} occupancy")
        axes[1, col].imshow(projection(roi_mask, axis), cmap="gray", origin="lower")
        axes[1, col].set_title(f"{name} ROI")
        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "prior_roi_occupancy_projections.png", dpi=200)
    plt.close(fig)


def save_gaussian_panels(indices, valid, roi, occupancy, shape, out_dir):
    views = [
        ("axial", (0, 1)),
        ("coronal", (0, 2)),
        ("sagittal", (1, 2)),
    ]
    regions = [
        ("ROI Gaussians", roi),
        ("Occupancy Gaussians", occupancy),
        ("Outside ROI Gaussians", valid & ~roi),
    ]
    fig, axes = plt.subplots(len(regions), len(views), figsize=(12, 10))
    for row, (region_name, region_mask) in enumerate(regions):
        for col, (view_name, dims) in enumerate(views):
            scatter_projection(
                axes[row, col],
                indices,
                valid,
                region_mask,
                dims,
                shape,
                f"{view_name} {region_name}",
            )
    fig.tight_layout()
    fig.savefig(out_dir / "gaussian_allocation_projections.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize prior masks and Gaussian allocation.")
    parser.add_argument("--prior_path", required=True, help="prior_data.npz containing roi_mask and occupancy_mask.")
    parser.add_argument("--gaussian_ply", required=True, help="Saved Gaussian point_cloud.ply.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--geometry_pickle", default=None, help="Original X-Gaussian pickle for physical grid bounds.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prior = np.load(args.prior_path)
    roi_mask = prior["roi_mask"].astype(bool)
    occupancy_mask = prior["occupancy_mask"].astype(bool)
    xyz = load_gaussian_xyz(args.gaussian_ply)

    grid_min, grid_max = grid_bounds(roi_mask.shape, args.geometry_pickle)
    indices, valid, roi, occupancy = classify_gaussians(xyz, roi_mask, occupancy_mask, grid_min, grid_max)

    summary = {
        "prior_path": args.prior_path,
        "gaussian_ply": args.gaussian_ply,
        "num_gaussians": int(xyz.shape[0]),
        "num_inside_grid": int(valid.sum()),
        "num_roi_gaussians": int(roi.sum()),
        "roi_gaussian_ratio": float(roi.mean()) if xyz.shape[0] else 0.0,
        "num_occupancy_gaussians": int(occupancy.sum()),
        "occupancy_gaussian_ratio": float(occupancy.mean()) if xyz.shape[0] else 0.0,
        "num_outside_roi_gaussians": int((valid & ~roi).sum()),
        "num_empty_occupancy_gaussians": int((valid & ~occupancy).sum()),
        "grid_min": grid_min.tolist(),
        "grid_max": grid_max.tolist(),
    }
    with open(out_dir / "allocation_visualization_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    save_prior_panels(roi_mask, occupancy_mask, out_dir)
    save_gaussian_panels(indices, valid, roi, occupancy, roi_mask.shape, out_dir)
    print(json.dumps(summary, indent=2))
    print(f"Saved visualizations to {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
