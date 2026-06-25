import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import PriorNet3DUNet
from priors.volume_prior import (
    ConeGeometry,
    dilate_mask,
    sample_points,
    save_preview_slices,
    voxel_positions_for_volume,
    write_ascii_ply,
)


def parse_size(values, name):
    if values is None:
        return None
    if len(values) == 1:
        return (int(values[0]), int(values[0]), int(values[0]))
    if len(values) == 3:
        return tuple(int(v) for v in values)
    raise ValueError(f"{name} expects one value or three values")


def read_volume(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".pickle", ".pkl"}:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if "image" not in payload:
            raise KeyError(f"Pickle volume must contain key 'image': {path}")
        return np.asarray(payload["image"], dtype=np.float32)
    if suffix == ".npy":
        return np.load(path).astype(np.float32)
    if suffix == ".npz":
        payload = np.load(path)
        for key in ("volume", "recon_volume", "fdk", "image", "prior_volume", "soft_prior"):
            if key in payload:
                return payload[key].astype(np.float32)
        return payload[payload.files[0]].astype(np.float32)
    raise ValueError(f"Unsupported volume format: {path}")


def normalize_volume(volume, eps=1e-6, pmin=0.0, pmax=100.0):
    volume = np.asarray(volume, dtype=np.float32)
    lo = float(np.percentile(volume, pmin))
    hi = float(np.percentile(volume, pmax))
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - lo) / max(hi - lo, eps), 0.0, 1.0).astype(np.float32)


def resize_np(volume, target_size):
    if target_size is None or tuple(volume.shape) == tuple(target_size):
        return volume
    tensor = torch.from_numpy(volume)[None, None].float()
    tensor = F.interpolate(tensor, size=target_size, mode="trilinear", align_corners=False)
    return tensor[0, 0].numpy().astype(np.float32)


def load_geometry(path):
    if not path:
        return None
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return ConeGeometry(payload)


def build_model_from_checkpoint(checkpoint, args):
    ckpt_args = checkpoint.get("args", {})
    return PriorNet3DUNet(
        in_channels=1,
        out_channels=1,
        base_channels=int(args.base_channels or ckpt_args.get("base_channels", 16)),
        depth=int(args.depth or ckpt_args.get("depth", 3)),
        norm=args.norm or ckpt_args.get("norm", "instance"),
    )


def save_prior_outputs(soft_prior, out_dir, args):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    occupancy_mask = soft_prior >= float(args.threshold)
    roi_mask = dilate_mask(occupancy_mask, int(args.roi_dilation))

    geometry = load_geometry(args.geometry_pickle)
    positions = voxel_positions_for_volume(soft_prior.shape, geometry)
    if positions.shape[:3] != soft_prior.shape:
        axes = [np.linspace(-1.0, 1.0, int(size), dtype=np.float32) for size in soft_prior.shape]
        grid = np.meshgrid(*axes, indexing="ij")
        positions = np.asarray(grid, dtype=np.float32).transpose([1, 2, 3, 0])

    points_source = roi_mask if args.points_source == "roi" else occupancy_mask
    points = positions[points_source].reshape(-1, 3).astype(np.float32)
    points = sample_points(points, args.max_points, args.seed)

    np.save(out_dir / "prior_volume.npy", soft_prior.astype(np.float32))
    np.save(out_dir / "occupancy_mask.npy", occupancy_mask.astype(np.uint8))
    np.save(out_dir / "roi_mask.npy", roi_mask.astype(np.uint8))
    np.save(out_dir / "point_cloud.npy", points.astype(np.float32))
    np.savez_compressed(
        out_dir / "prior_data.npz",
        occupancy_mask=occupancy_mask.astype(np.uint8),
        roi_mask=roi_mask.astype(np.uint8),
        points=points.astype(np.float32),
        prior_volume=soft_prior.astype(np.float32),
        soft_prior=soft_prior.astype(np.float32),
        volume_shape=np.asarray(soft_prior.shape, dtype=np.int32),
    )
    if points.shape[0] > 0:
        write_ascii_ply(out_dir / "point_cloud.ply", points)

    preview_payload = {
        "volume_norm": soft_prior.astype(np.float32),
        "occupancy_mask": occupancy_mask.astype(np.float32),
        "roi_mask": roi_mask.astype(np.float32),
    }
    save_preview_slices(preview_payload, out_dir)

    stats = {
        "source_fdk": str(args.fdk),
        "checkpoint": str(args.checkpoint),
        "shape": list(soft_prior.shape),
        "threshold": float(args.threshold),
        "occupancy_voxels": int(occupancy_mask.sum()),
        "occupancy_ratio": float(occupancy_mask.mean()),
        "roi_voxels": int(roi_mask.sum()),
        "roi_ratio": float(roi_mask.mean()),
        "num_points": int(points.shape[0]),
        "max_points": int(args.max_points),
        "points_source": args.points_source,
        "geometry_pickle": args.geometry_pickle,
    }
    with open(out_dir / "prior_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Predict soft structure prior from V_fdk.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fdk", required=True, help="FDK coarse volume: .npy, .npz, or pickle with key 'image'.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--geometry_pickle", default=None, help="Optional pickle used only for physical point coordinates.")
    parser.add_argument("--input_size", nargs="+", type=int, default=None)
    parser.add_argument("--output_size", nargs="+", type=int, default=None)
    parser.add_argument("--norm_percentiles", nargs=2, type=float, default=[0.0, 100.0])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--roi_dilation", type=int, default=2)
    parser.add_argument("--points_source", default="roi", choices=["roi", "occupancy"])
    parser.add_argument("--max_points", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--norm", default=None, choices=["instance", "batch", "none"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model_from_checkpoint(checkpoint, args).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    original = read_volume(args.fdk)
    original_shape = original.shape
    input_size = parse_size(args.input_size, "--input_size")
    output_size = parse_size(args.output_size, "--output_size") or original_shape
    pmin, pmax = args.norm_percentiles
    volume = normalize_volume(original, pmin=pmin, pmax=pmax)
    volume = resize_np(volume, input_size)

    with torch.no_grad():
        tensor = torch.from_numpy(volume)[None, None].to(device=device, dtype=torch.float32)
        logits = model(tensor)
        prob = torch.sigmoid(logits)
        if tuple(prob.shape[2:]) != tuple(output_size):
            prob = F.interpolate(prob, size=output_size, mode="trilinear", align_corners=False)
        soft_prior = prob[0, 0].detach().cpu().numpy().astype(np.float32)

    stats = save_prior_outputs(soft_prior, args.out_dir, args)
    print(json.dumps({"outputs": str(Path(args.out_dir) / "prior_data.npz"), "stats": stats}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
