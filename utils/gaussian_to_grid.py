import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F


def load_prior_mask(prior_path, source="roi"):
    payload = np.load(prior_path)
    if source == "roi":
        key = "roi_mask"
    elif source in ("soft", "soft_prior", "prior_volume"):
        if "soft_prior" in payload:
            key = "soft_prior"
        elif "prior_volume" in payload:
            key = "prior_volume"
        else:
            key = "occupancy_mask"
    else:
        key = "occupancy_mask"
    if key not in payload:
        raise KeyError(f"prior file must contain {key}: {prior_path}")
    return payload[key].astype(np.float32)


def downsample_prior_mask(mask_np, grid_size, device):
    mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32)
    mask = mask[None, None]
    target_size = (int(grid_size), int(grid_size), int(grid_size))
    mask = F.interpolate(mask, size=target_size, mode="trilinear", align_corners=False)
    return mask[0, 0].clamp(0.0, 1.0)


def gaussian_centers_to_occupancy_grid(xyz, opacity, grid_min, grid_max, grid_size):
    grid_size = int(grid_size)
    device = xyz.device
    dtype = xyz.dtype
    grid_min = torch.as_tensor(grid_min, dtype=dtype, device=device)
    grid_max = torch.as_tensor(grid_max, dtype=dtype, device=device)
    denom = torch.clamp(grid_max - grid_min, min=1e-6)

    normalized = (xyz - grid_min[None, :]) / denom[None, :]
    valid = torch.all((normalized >= 0.0) & (normalized <= 1.0), dim=1)
    indices = torch.round(normalized * float(grid_size - 1)).long()
    indices = indices.clamp(0, grid_size - 1)

    flat_idx = indices[:, 0] * grid_size * grid_size + indices[:, 1] * grid_size + indices[:, 2]
    weights = opacity.reshape(-1).clamp(0.0, 1.0) * valid.to(dtype)

    flat = torch.zeros(grid_size ** 3, dtype=dtype, device=device)
    flat.scatter_add_(0, flat_idx, weights)
    grid = flat.reshape(grid_size, grid_size, grid_size)
    return grid.clamp(0.0, 1.0), valid


def save_grid_debug(gaussian_grid, prior_grid, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gaussian_np = gaussian_grid.detach().cpu().numpy().astype(np.float32)
    prior_np = prior_grid.detach().cpu().numpy().astype(np.float32)
    np.save(out_dir / "gaussian_occ.npy", gaussian_np)
    np.save(out_dir / "prior_occ.npy", prior_np)

    mids = [size // 2 for size in gaussian_np.shape]
    slices = {
        "axial_z": (slice(None), slice(None), mids[2]),
        "coronal_y": (slice(None), mids[1], slice(None)),
        "sagittal_x": (mids[0], slice(None), slice(None)),
    }
    for name, index in slices.items():
        imageio.imwrite(out_dir / f"{name}_gaussian_occ.png", (np.clip(gaussian_np[index], 0, 1) * 255).astype(np.uint8))
        imageio.imwrite(out_dir / f"{name}_prior_occ.png", (np.clip(prior_np[index], 0, 1) * 255).astype(np.uint8))
        diff = np.abs(gaussian_np[index] - prior_np[index])
        imageio.imwrite(out_dir / f"{name}_absdiff.png", (np.clip(diff, 0, 1) * 255).astype(np.uint8))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
