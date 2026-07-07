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
    save_preview_slices,
    voxel_positions_for_volume,
    write_ascii_ply,
)


def read_volume(path, key=None):
    path = Path(path)
    if path.suffix.lower() in {".pickle", ".pkl"}:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if key:
            return np.asarray(payload[key], dtype=np.float32), key
        for candidate in ("image", "volume", "recon_volume", "fdk"):
            if candidate in payload:
                return np.asarray(payload[candidate], dtype=np.float32), candidate
        raise KeyError(f"No volume-like key found in pickle: {path}")
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32), "npy"
    if path.suffix.lower() == ".npz":
        payload = np.load(path)
        keys = [key] if key else [
            "soft_prior",
            "prior_volume",
            "vesselness",
            "vessel_prob",
            "volume",
            "recon_volume",
            "fdk",
            "image",
        ]
        for candidate in keys:
            if candidate and candidate in payload:
                return payload[candidate].astype(np.float32), candidate
        return payload[payload.files[0]].astype(np.float32), payload.files[0]
    raise ValueError(f"Unsupported volume path: {path}")


def load_prior(path, key=None):
    payload = np.load(path)
    keys = [key] if key else ["soft_prior", "prior_volume", "occupancy_mask"]
    for candidate in keys:
        if candidate and candidate in payload:
            prior = payload[candidate].astype(np.float32)
            return payload, prior, candidate
    raise KeyError(f"Prior file must contain one of {keys}: {path}")


def normalize_percentile(volume, pmin=0.5, pmax=99.5, eps=1e-6):
    volume = np.asarray(volume, dtype=np.float32)
    lo = float(np.percentile(volume, pmin))
    hi = float(np.percentile(volume, pmax))
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32), lo, hi
    normalized = (volume - lo) / max(hi - lo, eps)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32), lo, hi


def normalize_unit(volume, eps=1e-6):
    volume = np.asarray(volume, dtype=np.float32)
    vmin = float(volume.min())
    vmax = float(volume.max())
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - vmin) / max(vmax - vmin, eps)).astype(np.float32)


def hessian_eigenvalues(volume, sigma, chunk_size=500000):
    from scipy import ndimage

    volume = np.asarray(volume, dtype=np.float32)
    sigma = float(sigma)
    scale_norm = sigma * sigma
    hxx = ndimage.gaussian_filter(volume, sigma=sigma, order=(2, 0, 0), mode="nearest") * scale_norm
    hyy = ndimage.gaussian_filter(volume, sigma=sigma, order=(0, 2, 0), mode="nearest") * scale_norm
    hzz = ndimage.gaussian_filter(volume, sigma=sigma, order=(0, 0, 2), mode="nearest") * scale_norm
    hxy = ndimage.gaussian_filter(volume, sigma=sigma, order=(1, 1, 0), mode="nearest") * scale_norm
    hxz = ndimage.gaussian_filter(volume, sigma=sigma, order=(1, 0, 1), mode="nearest") * scale_norm
    hyz = ndimage.gaussian_filter(volume, sigma=sigma, order=(0, 1, 1), mode="nearest") * scale_norm

    flat_count = volume.size
    eigvals = np.empty((flat_count, 3), dtype=np.float32)
    for start in range(0, flat_count, int(chunk_size)):
        end = min(start + int(chunk_size), flat_count)
        hessian = np.empty((end - start, 3, 3), dtype=np.float32)
        hessian[:, 0, 0] = hxx.reshape(-1)[start:end]
        hessian[:, 1, 1] = hyy.reshape(-1)[start:end]
        hessian[:, 2, 2] = hzz.reshape(-1)[start:end]
        hessian[:, 0, 1] = hessian[:, 1, 0] = hxy.reshape(-1)[start:end]
        hessian[:, 0, 2] = hessian[:, 2, 0] = hxz.reshape(-1)[start:end]
        hessian[:, 1, 2] = hessian[:, 2, 1] = hyz.reshape(-1)[start:end]
        eigvals[start:end] = np.linalg.eigvalsh(hessian).astype(np.float32)
    return eigvals.reshape(volume.shape + (3,))


def frangi_vesselness_3d(volume, scales, alpha=0.5, beta=0.5, gamma=0.0, bright=True, chunk_size=500000):
    responses = []
    for sigma in scales:
        eig = hessian_eigenvalues(volume, sigma=sigma, chunk_size=chunk_size)
        abs_order = np.argsort(np.abs(eig), axis=-1)
        ordered = np.take_along_axis(eig, abs_order, axis=-1)
        l1 = ordered[..., 0]
        l2 = ordered[..., 1]
        l3 = ordered[..., 2]

        eps = 1e-6
        ra = np.abs(l2) / (np.abs(l3) + eps)
        rb = np.abs(l1) / np.sqrt(np.abs(l2 * l3) + eps)
        s = np.sqrt(l1 * l1 + l2 * l2 + l3 * l3)
        c = float(gamma) if float(gamma) > 0.0 else max(float(np.percentile(s, 95.0)), eps)

        plate = 1.0 - np.exp(-(ra * ra) / (2.0 * alpha * alpha))
        blob = np.exp(-(rb * rb) / (2.0 * beta * beta))
        struct = 1.0 - np.exp(-(s * s) / (2.0 * c * c))
        vessel = plate * blob * struct

        if bright:
            vessel[(l2 > 0.0) | (l3 > 0.0)] = 0.0
        else:
            vessel[(l2 < 0.0) | (l3 < 0.0)] = 0.0
        responses.append(vessel.astype(np.float32))

    return normalize_unit(np.maximum.reduce(responses))


def fuse_prior(prior, vesselness, mode, vessel_weight, blend_weight):
    prior = np.clip(prior.astype(np.float32), 0.0, 1.0)
    vesselness = np.clip(vesselness.astype(np.float32), 0.0, 1.0)
    if mode == "boost":
        fused = prior + float(vessel_weight) * vesselness * (1.0 - prior)
    elif mode == "max":
        fused = np.maximum(prior, float(vessel_weight) * vesselness)
    elif mode == "blend":
        w = float(blend_weight)
        fused = (1.0 - w) * prior + w * vesselness
    else:
        raise ValueError(f"Unsupported fusion mode: {mode}")
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def load_geometry(path):
    if not path:
        return None
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return ConeGeometry(payload)


def save_outputs(fused, vesselness, prior, out_dir, args, source_stats):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    occupancy_mask = fused >= float(args.threshold)
    roi_mask = dilate_mask(occupancy_mask, int(args.roi_dilation))

    geometry = load_geometry(args.geometry_pickle)
    positions = voxel_positions_for_volume(fused.shape, geometry)
    if positions.shape[:3] != fused.shape:
        axes = [np.linspace(-1.0, 1.0, int(size), dtype=np.float32) for size in fused.shape]
        grid = np.meshgrid(*axes, indexing="ij")
        positions = np.asarray(grid, dtype=np.float32).transpose([1, 2, 3, 0])

    points_source = roi_mask if args.points_source == "roi" else occupancy_mask
    points = positions[points_source].reshape(-1, 3).astype(np.float32)
    points = sample_points(points, int(args.max_points), int(args.seed))

    np.save(out_dir / "prior_volume.npy", fused.astype(np.float32))
    np.save(out_dir / "vesselness.npy", vesselness.astype(np.float32))
    np.save(out_dir / "source_prior.npy", prior.astype(np.float32))
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
        source_prior=prior.astype(np.float32),
        volume_shape=np.asarray(fused.shape, dtype=np.int32),
    )
    if points.shape[0] > 0:
        write_ascii_ply(out_dir / "point_cloud.ply", points)

    save_preview_slices(
        {
            "volume_norm": fused.astype(np.float32),
            "occupancy_mask": occupancy_mask.astype(np.float32),
            "roi_mask": roi_mask.astype(np.float32),
        },
        out_dir,
    )
    for name, volume in {
        "source_prior_mip_z.png": prior.max(axis=2),
        "vesselness_mip_z.png": vesselness.max(axis=2),
        "fused_prior_mip_z.png": fused.max(axis=2),
    }.items():
        imageio.imwrite(out_dir / name, (np.clip(volume, 0.0, 1.0) * 255).astype(np.uint8))

    stats = {
        "prior_path": str(args.prior),
        "prior_key": source_stats["prior_key"],
        "vessel_source": str(args.vessel_source or args.prior),
        "vessel_source_key": source_stats["vessel_source_key"],
        "shape": list(fused.shape),
        "scales": [float(v) for v in args.scales],
        "fusion_mode": args.fusion_mode,
        "vessel_weight": float(args.vessel_weight),
        "blend_weight": float(args.blend_weight),
        "threshold": float(args.threshold),
        "occupancy_voxels": int(occupancy_mask.sum()),
        "occupancy_ratio": float(occupancy_mask.mean()),
        "roi_voxels": int(roi_mask.sum()),
        "roi_ratio": float(roi_mask.mean()),
        "num_points": int(points.shape[0]),
        "source_prior_mean": float(prior.mean()),
        "vesselness_mean": float(vesselness.mean()),
        "fused_prior_mean": float(fused.mean()),
        "source_norm_low": float(source_stats["source_norm_low"]),
        "source_norm_high": float(source_stats["source_norm_high"]),
    }
    with open(out_dir / "vesselness_prior_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build a vesselness-fused soft structure prior.")
    parser.add_argument("--prior", required=True, help="Existing prior_data.npz.")
    parser.add_argument("--prior_key", default=None)
    parser.add_argument("--vessel_source", default="", help="Optional FDK/PriorNet volume. Defaults to --prior.")
    parser.add_argument("--vessel_source_key", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--geometry_pickle", default="data/aneurism_50.pickle")
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 1.5, 2.0, 2.5])
    parser.add_argument("--norm_percentiles", nargs=2, type=float, default=[0.5, 99.5])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--dark_tubes", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=500000)
    parser.add_argument("--fusion_mode", choices=["boost", "max", "blend"], default="boost")
    parser.add_argument("--vessel_weight", type=float, default=0.35)
    parser.add_argument("--blend_weight", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--roi_dilation", type=int, default=2)
    parser.add_argument("--points_source", default="roi", choices=["roi", "occupancy"])
    parser.add_argument("--max_points", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _, prior, prior_key = load_prior(args.prior, key=args.prior_key)
    prior = np.clip(prior.astype(np.float32), 0.0, 1.0)
    vessel_source_path = args.vessel_source or args.prior
    source, source_key = read_volume(vessel_source_path, key=args.vessel_source_key)
    if source.shape != prior.shape:
        raise ValueError(f"vessel source shape {source.shape} does not match prior shape {prior.shape}")

    pmin, pmax = args.norm_percentiles
    source_norm, norm_low, norm_high = normalize_percentile(source, pmin=pmin, pmax=pmax)
    vesselness = frangi_vesselness_3d(
        source_norm,
        scales=args.scales,
        alpha=float(args.alpha),
        beta=float(args.beta),
        gamma=float(args.gamma),
        bright=not bool(args.dark_tubes),
        chunk_size=int(args.chunk_size),
    )
    fused = fuse_prior(
        prior,
        vesselness,
        mode=args.fusion_mode,
        vessel_weight=float(args.vessel_weight),
        blend_weight=float(args.blend_weight),
    )
    stats = save_outputs(
        fused,
        vesselness,
        prior,
        args.out_dir,
        args,
        {
            "prior_key": prior_key,
            "vessel_source_key": source_key,
            "source_norm_low": norm_low,
            "source_norm_high": norm_high,
        },
    )
    print(json.dumps({"outputs": str(Path(args.out_dir) / "prior_data.npz"), "stats": stats}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
