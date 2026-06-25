import json
import os
import pickle
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

class ConeGeometry:
    def __init__(self, data):
        scale = 1.0
        self.DSD = data["DSD"] / scale
        self.DSO = data["DSO"] / scale
        self.nDetector = np.array(data["nDetector"])
        self.dDetector = np.array(data["dDetector"]) / scale
        self.sDetector = self.nDetector * self.dDetector
        self.nVoxel = np.array(data["nVoxel"])
        self.dVoxel = np.array(data["dVoxel"]) / scale
        self.sVoxel = self.nVoxel * self.dVoxel
        self.offOrigin = np.array(data["offOrigin"]) / scale
        self.offDetector = np.array(data["offDetector"]) / scale
        self.accuracy = data.get("accuracy")
        self.mode = data.get("mode")
        self.filter = data.get("filter")


def get_voxels(geo):
    n1, n2, n3 = geo.nVoxel
    s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2
    xyz = np.meshgrid(
        np.linspace(-s1, s1, int(n1)),
        np.linspace(-s2, s2, int(n2)),
        np.linspace(-s3, s3, int(n3)),
        indexing="ij",
    )
    return np.asarray(xyz, dtype=np.float32).transpose([1, 2, 3, 0])


def write_ascii_ply(path, points):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property float nx\n")
        handle.write("property float ny\n")
        handle.write("property float nz\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for x, y, z in points:
            handle.write(f"{float(x)} {float(y)} {float(z)} 0 0 0 255 255 255\n")


def normalize_volume(volume):
    volume = np.asarray(volume, dtype=np.float32)
    vmin = float(volume.min())
    vmax = float(volume.max())
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.float32), vmin, vmax
    return (volume - vmin) / (vmax - vmin), vmin, vmax


def load_xray_pickle(path):
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    if "image" not in data:
        raise KeyError(f"X-ray pickle does not contain an 'image' volume: {path}")
    return np.asarray(data["image"], dtype=np.float32), ConeGeometry(data)


def load_volume_and_geometry(input_path, geometry_pickle=None):
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix in {".pickle", ".pkl"}:
        return load_xray_pickle(input_path)

    if suffix == ".npy":
        volume = np.load(input_path).astype(np.float32)
    elif suffix == ".npz":
        payload = np.load(input_path)
        key = "volume" if "volume" in payload else payload.files[0]
        volume = payload[key].astype(np.float32)
    else:
        raise ValueError(f"Unsupported volume input format: {input_path}")

    geometry = None
    if geometry_pickle:
        _, geometry = load_xray_pickle(geometry_pickle)
    return volume, geometry


def threshold_volume(volume_norm, mode, value):
    if mode == "absolute":
        threshold = float(value)
    elif mode == "quantile":
        threshold = float(np.quantile(volume_norm, float(value)))
    else:
        raise ValueError(f"Unsupported threshold mode: {mode}")
    return volume_norm >= threshold, threshold


def dilate_mask(mask, radius):
    radius = int(radius)
    if radius <= 0:
        return mask.copy()

    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx * dx + dy * dy + dz * dz > radius * radius:
                    continue
                xs = radius + dx
                ys = radius + dy
                zs = radius + dz
                out |= padded[xs:xs + mask.shape[0], ys:ys + mask.shape[1], zs:zs + mask.shape[2]]
    return out


def voxel_positions_for_volume(volume_shape, geometry):
    if geometry is not None:
        return get_voxels(geometry)

    axes = [np.linspace(-1.0, 1.0, int(size), dtype=np.float32) for size in volume_shape]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.asarray(grid, dtype=np.float32).transpose([1, 2, 3, 0])


def sample_points(points, max_points, seed):
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=int(max_points), replace=False)
    return points[indices]


def build_volume_prior(
    volume,
    geometry=None,
    threshold_mode="quantile",
    threshold_value=0.98,
    sample_stride=1,
    max_points=50000,
    roi_dilation=2,
    seed=0,
):
    volume_norm, raw_min, raw_max = normalize_volume(volume)
    occupancy_mask, threshold = threshold_volume(volume_norm, threshold_mode, threshold_value)
    roi_mask = dilate_mask(occupancy_mask, roi_dilation)

    stride = max(1, int(sample_stride))
    sampled_mask = occupancy_mask[::stride, ::stride, ::stride]
    positions = voxel_positions_for_volume(volume_norm.shape, geometry)[::stride, ::stride, ::stride]
    points = positions[sampled_mask].reshape(-1, 3).astype(np.float32)
    points = sample_points(points, max_points, seed)

    stats = {
        "volume_shape": list(volume_norm.shape),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "threshold_mode": threshold_mode,
        "threshold_value": float(threshold_value),
        "threshold_applied": float(threshold),
        "occupancy_voxels": int(occupancy_mask.sum()),
        "occupancy_ratio": float(occupancy_mask.mean()) if occupancy_mask.size else 0.0,
        "roi_voxels": int(roi_mask.sum()),
        "roi_ratio": float(roi_mask.mean()) if roi_mask.size else 0.0,
        "sample_stride": int(stride),
        "num_points": int(points.shape[0]),
        "max_points": int(max_points) if max_points is not None else 0,
        "roi_dilation": int(roi_dilation),
        "seed": int(seed),
        "coordinate_source": "geometry" if geometry is not None else "normalized_grid",
    }

    return {
        "volume_norm": volume_norm,
        "occupancy_mask": occupancy_mask,
        "roi_mask": roi_mask,
        "points": points,
        "stats": stats,
    }


def save_slice_png(image, path):
    image = np.asarray(image, dtype=np.float32)
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
    else:
        image = np.zeros_like(image, dtype=np.float32)
    imageio.imwrite(path, (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))


def save_preview_slices(prior, out_dir):
    slices_dir = Path(out_dir) / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)

    volume = prior["volume_norm"]
    occupancy = prior["occupancy_mask"].astype(np.float32)
    roi = prior["roi_mask"].astype(np.float32)
    mids = [size // 2 for size in volume.shape]
    views = {
        "axial_z": (slice(None), slice(None), mids[2]),
        "coronal_y": (slice(None), mids[1], slice(None)),
        "sagittal_x": (mids[0], slice(None), slice(None)),
    }

    for name, index in views.items():
        save_slice_png(volume[index], slices_dir / f"{name}_volume.png")
        save_slice_png(occupancy[index], slices_dir / f"{name}_occupancy.png")
        save_slice_png(roi[index], slices_dir / f"{name}_roi.png")


def save_prior_outputs(prior, out_dir, source_path=None, source_type="gt"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    occupancy = prior["occupancy_mask"].astype(np.uint8)
    roi = prior["roi_mask"].astype(np.uint8)
    points = prior["points"].astype(np.float32)
    stats = dict(prior["stats"])
    stats.update({
        "source_path": str(source_path) if source_path is not None else None,
        "source_type": source_type,
    })

    np.save(out_dir / "occupancy_mask.npy", occupancy)
    np.save(out_dir / "roi_mask.npy", roi)
    np.save(out_dir / "point_cloud.npy", points)
    np.save(out_dir / "prior_volume.npy", prior["volume_norm"].astype(np.float32))
    np.savez_compressed(
        out_dir / "prior_data.npz",
        occupancy_mask=occupancy,
        roi_mask=roi,
        points=points,
        prior_volume=prior["volume_norm"].astype(np.float32),
        volume_shape=np.asarray(prior["volume_norm"].shape, dtype=np.int32),
    )

    if points.shape[0] > 0:
        write_ascii_ply(out_dir / "point_cloud.ply", points)

    save_preview_slices(prior, out_dir)

    with open(out_dir / "prior_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    return {
        "occupancy_mask": str(out_dir / "occupancy_mask.npy"),
        "roi_mask": str(out_dir / "roi_mask.npy"),
        "point_cloud_npy": str(out_dir / "point_cloud.npy"),
        "point_cloud_ply": str(out_dir / "point_cloud.ply") if points.shape[0] > 0 else None,
        "prior_volume": str(out_dir / "prior_volume.npy"),
        "prior_data": str(out_dir / "prior_data.npz"),
        "stats": str(out_dir / "prior_stats.json"),
        "slices": str(out_dir / "slices"),
    }
