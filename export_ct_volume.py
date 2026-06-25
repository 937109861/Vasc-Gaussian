import argparse
import json
import os
import pickle

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scene.dataset_readers import ConeGeometry, get_voxels
from scene.gaussian_model import GaussianModel_Xray
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from utils.differentiable_volume import gaussian_centers_to_dgr_grid
from utils.export_volume import (
    calibrate_density_weights,
    gaussian_roi_gate,
    load_prior_payload,
    make_slicer_display_volume,
    save_histogram,
    save_mhd_uint16,
    save_region_preview_slices,
    save_tiff_stack_uint16,
    structure_preserve_volume,
    volume_region_statistics,
)


def resolve_iteration(model_path, iteration):
    if iteration == -1:
        return searchForMaxIteration(os.path.join(model_path, "point_cloud"))
    return iteration


def load_gaussians(model_path, iteration, sh_degree):
    gaussians = GaussianModel_Xray(sh_degree)
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Gaussian point cloud not found: {ply_path}")
    gaussians.load_ply(ply_path)
    return gaussians, ply_path


def gaussian_density_weights(gaussians, mode):
    opacity = gaussians.get_opacity.detach().reshape(-1).cpu().numpy().astype(np.float32)
    if mode == "opacity":
        return opacity

    dc = gaussians._features_dc.detach()[:, 0, :].mean(dim=1).cpu().numpy().astype(np.float32)
    dc = np.maximum(dc, 0.0)
    if dc.max() > dc.min():
        dc = (dc - dc.min()) / (dc.max() - dc.min())
    return opacity * dc


def gaussian_density_weights_extended(gaussians, mode):
    if mode in ("opacity", "opacity_dc"):
        return gaussian_density_weights(gaussians, mode)
    if mode in ("radiodensity", "attenuation"):
        if not hasattr(gaussians, "get_radiodensity"):
            return gaussians.get_opacity.detach().reshape(-1).cpu().numpy().astype(np.float32)
        return gaussians.get_radiodensity.detach().reshape(-1).cpu().numpy().astype(np.float32)
    if mode in ("opacity_radiodensity", "effective_density"):
        opacity = gaussians.get_opacity.detach().reshape(-1)
        if hasattr(gaussians, "get_radiodensity"):
            opacity = opacity * gaussians.get_radiodensity.detach().reshape(-1)
        return opacity.cpu().numpy().astype(np.float32)
    if mode == "opacity_scale":
        opacity = gaussians.get_opacity.detach().reshape(-1).cpu().numpy().astype(np.float32)
        scales = gaussians.get_scaling.detach().cpu().numpy().astype(np.float32)
        scale_score = 1.0 / np.maximum(np.mean(scales, axis=1), 1e-6)
        if scale_score.max() > scale_score.min():
            scale_score = (scale_score - scale_score.min()) / (scale_score.max() - scale_score.min())
        return opacity * scale_score.astype(np.float32)
    if mode == "opacity_feature_scale":
        opacity = gaussians.get_opacity.detach().reshape(-1).cpu().numpy().astype(np.float32)
        dc = gaussians._features_dc.detach()[:, 0, :].mean(dim=1).cpu().numpy().astype(np.float32)
        dc = np.maximum(dc, 0.0)
        if dc.max() > dc.min():
            dc = (dc - dc.min()) / (dc.max() - dc.min())
        scales = gaussians.get_scaling.detach().cpu().numpy().astype(np.float32)
        scale_score = 1.0 / np.maximum(np.mean(scales, axis=1), 1e-6)
        if scale_score.max() > scale_score.min():
            scale_score = (scale_score - scale_score.min()) / (scale_score.max() - scale_score.min())
        return opacity * (0.5 + 0.5 * dc.astype(np.float32)) * (0.5 + 0.5 * scale_score.astype(np.float32))
    raise ValueError(f"Unsupported density mapping mode: {mode}")


def normalize_volume(volume):
    volume = volume.astype(np.float32, copy=False)
    vmin = float(volume.min())
    vmax = float(volume.max())
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.float32), vmin, vmax
    return (volume - vmin) / (vmax - vmin), vmin, vmax


def smooth_volume_gaussian(volume, sigma):
    sigma = float(sigma)
    if sigma <= 0.0:
        return volume.astype(np.float32, copy=False)

    radius = max(1, int(np.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / torch.clamp(kernel.sum(), min=1e-8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vol = torch.from_numpy(np.asarray(volume, dtype=np.float32)).to(device=device)[None, None]
    kernel = kernel.to(device=device)

    vol = F.pad(vol, (radius, radius, 0, 0, 0, 0), mode="replicate")
    vol = F.conv3d(vol, kernel.view(1, 1, 1, 1, -1))
    vol = F.pad(vol, (0, 0, radius, radius, 0, 0), mode="replicate")
    vol = F.conv3d(vol, kernel.view(1, 1, 1, -1, 1))
    vol = F.pad(vol, (0, 0, 0, 0, radius, radius), mode="replicate")
    vol = F.conv3d(vol, kernel.view(1, 1, -1, 1, 1))
    return vol[0, 0].detach().cpu().numpy().astype(np.float32)


def smooth_volume_median(volume, kernel_size):
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return volume.astype(np.float32, copy=False)
    if kernel_size % 2 == 0:
        raise ValueError("post_median_size must be odd")

    radius = kernel_size // 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vol = torch.from_numpy(np.asarray(volume, dtype=np.float32)).to(device=device)[None, None]
    padded = F.pad(vol, (radius, radius, radius, radius, radius, radius), mode="replicate")

    patches = []
    d, h, w = vol.shape[-3:]
    for dz in range(kernel_size):
        for dy in range(kernel_size):
            for dx in range(kernel_size):
                patches.append(padded[:, :, dz:dz + d, dy:dy + h, dx:dx + w])
    stacked = torch.stack(patches, dim=0)
    filtered = torch.median(stacked, dim=0).values
    return filtered[0, 0].detach().cpu().numpy().astype(np.float32)


def apply_contrast_gamma(volume, gamma):
    gamma = float(gamma)
    if abs(gamma - 1.0) < 1e-8:
        return volume.astype(np.float32, copy=False)
    if gamma <= 0.0:
        raise ValueError("post_contrast_gamma must be positive")
    volume = np.asarray(volume, dtype=np.float32)
    return np.power(np.clip(volume, 0.0, None), gamma).astype(np.float32)


def center_voxelize(xyz, weights, geometry):
    voxel_grid = get_voxels(geometry)
    mins = voxel_grid[0, 0, 0]
    spacing = geometry.dVoxel.astype(np.float32)
    shape = tuple(int(v) for v in geometry.nVoxel)

    indices = np.rint((xyz - mins[None, :]) / spacing[None, :]).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < np.asarray(shape)[None, :]), axis=1)
    indices = indices[valid]
    weights = weights[valid]

    volume = np.zeros(shape, dtype=np.float32)
    np.add.at(volume, (indices[:, 0], indices[:, 1], indices[:, 2]), weights)
    return volume, int(valid.sum())


def gaussian_voxelize(xyz, scales, weights, geometry, cutoff, max_gaussians):
    voxel_grid = get_voxels(geometry)
    mins = voxel_grid[0, 0, 0].astype(np.float32)
    spacing = geometry.dVoxel.astype(np.float32)
    shape = tuple(int(v) for v in geometry.nVoxel)
    volume = np.zeros(shape, dtype=np.float32)

    order = np.argsort(weights)[::-1]
    if max_gaussians and max_gaussians > 0:
        order = order[:max_gaussians]

    for idx in tqdm(order, desc="Voxelizing gaussians"):
        weight = float(weights[idx])
        if weight <= 0:
            continue

        center = xyz[idx].astype(np.float32)
        sigma = np.maximum(scales[idx].astype(np.float32), spacing * 0.5)
        radius = cutoff * sigma

        lo = np.floor((center - radius - mins) / spacing).astype(np.int64)
        hi = np.ceil((center + radius - mins) / spacing).astype(np.int64)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.asarray(shape) - 1)
        if np.any(hi < lo):
            continue

        xs = mins[0] + np.arange(lo[0], hi[0] + 1, dtype=np.float32) * spacing[0]
        ys = mins[1] + np.arange(lo[1], hi[1] + 1, dtype=np.float32) * spacing[1]
        zs = mins[2] + np.arange(lo[2], hi[2] + 1, dtype=np.float32) * spacing[2]

        gx = np.exp(-0.5 * ((xs - center[0]) / sigma[0]) ** 2)
        gy = np.exp(-0.5 * ((ys - center[1]) / sigma[1]) ** 2)
        gz = np.exp(-0.5 * ((zs - center[2]) / sigma[2]) ** 2)
        patch = weight * gx[:, None, None] * gy[None, :, None] * gz[None, None, :]
        volume[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] += patch.astype(np.float32)

    return volume, int(order.shape[0])


def dgr_voxelize(gaussians, weights, geometry, args):
    shape = tuple(int(v) for v in geometry.nVoxel)
    if len(set(shape)) != 1:
        raise ValueError(f"DGR export currently requires a cubic volume, got nVoxel={shape}")
    grid_size = int(shape[0])

    voxel_grid = get_voxels(geometry)
    grid_min = voxel_grid[0, 0, 0].astype(np.float32)
    grid_max = voxel_grid[-1, -1, -1].astype(np.float32)
    weights_t = torch.from_numpy(np.asarray(weights, dtype=np.float32)).to(gaussians.get_xyz.device)

    if args.max_gaussians and args.max_gaussians > 0 and weights_t.numel() > args.max_gaussians:
        _, selected = torch.topk(weights_t, k=int(args.max_gaussians), largest=True, sorted=False)
        xyz = gaussians.get_xyz[selected]
        scales = gaussians.get_scaling[selected]
        rotations = gaussians.get_rotation[selected]
        weights_t = weights_t[selected]
    else:
        xyz = gaussians.get_xyz
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation

    reference_grid_size = int(args.dgr_reference_grid_size)
    resolution_scale = 1.0
    if reference_grid_size > 0:
        resolution_scale = float(grid_size) / float(reference_grid_size)
    splat_radius = max(1, int(round(float(args.dgr_splat_radius) * resolution_scale)))
    min_sigma_voxels = float(args.dgr_min_sigma_voxels) * resolution_scale
    max_sigma_voxels = float(args.dgr_max_sigma_voxels) * resolution_scale
    sigma_scale = float(args.dgr_sigma_scale)
    effective_splat_radius = splat_radius
    if args.dgr_kernel_cutoff and float(args.dgr_kernel_cutoff) > 0.0:
        voxel_size = torch.as_tensor(
            (grid_max - grid_min) / float(grid_size - 1),
            dtype=scales.dtype,
            device=scales.device,
        )
        with torch.no_grad():
            sigma_voxels = scales / voxel_size[None, :]
            sigma_voxels = torch.clamp(
                sigma_voxels * sigma_scale,
                min=min_sigma_voxels,
                max=max_sigma_voxels,
            )
            cutoff_radius = int(
                torch.ceil(torch.amax(sigma_voxels) * float(args.dgr_kernel_cutoff)).item()
            )
        effective_splat_radius = max(effective_splat_radius, cutoff_radius)
        if args.dgr_max_splat_radius and int(args.dgr_max_splat_radius) > 0:
            effective_splat_radius = min(
                effective_splat_radius,
                int(args.dgr_max_splat_radius),
            )

    batch_size = int(args.dgr_batch_size)
    num_gaussians = int(xyz.shape[0])
    if batch_size <= 0:
        batch_size = max(1, num_gaussians)

    volume_t = torch.zeros(
        (grid_size, grid_size, grid_size),
        dtype=xyz.dtype,
        device=xyz.device,
    )
    valid_count = 0
    batch_count = 0
    with torch.no_grad():
        for start in tqdm(
            range(0, num_gaussians, batch_size),
            desc="DGR voxelizing batches",
        ):
            end = min(start + batch_size, num_gaussians)
            batch_volume, batch_valid = gaussian_centers_to_dgr_grid(
                xyz[start:end],
                scales[start:end],
                rotations[start:end],
                weights_t[start:end],
                grid_min,
                grid_max,
                grid_size=grid_size,
                radius=effective_splat_radius,
                min_sigma_voxels=min_sigma_voxels,
                max_sigma_voxels=max_sigma_voxels,
                sigma_scale=sigma_scale,
                normalize_kernel=args.dgr_normalize_kernel,
                supersample=args.dgr_supersample,
                kernel_cutoff=0.0,
                max_splat_radius=args.dgr_max_splat_radius,
                normalize=False,
            )
            volume_t.add_(batch_volume)
            valid_count += int(batch_valid.sum().item())
            batch_count += 1
            del batch_volume, batch_valid

        if not args.dgr_no_normalize_internal:
            volume_min = torch.amin(volume_t)
            volume_max = torch.amax(volume_t)
            volume_t = (volume_t - volume_min) / torch.clamp(
                volume_max - volume_min,
                min=1e-6,
            )

    volume = volume_t.cpu().numpy().astype(np.float32)
    stats = {
        "dgr_grid_size": int(grid_size),
        "dgr_reference_grid_size": int(reference_grid_size),
        "dgr_resolution_scale": float(resolution_scale),
        "dgr_splat_radius": int(effective_splat_radius),
        "dgr_base_splat_radius": int(args.dgr_splat_radius),
        "dgr_min_sigma_voxels": float(min_sigma_voxels),
        "dgr_base_min_sigma_voxels": float(args.dgr_min_sigma_voxels),
        "dgr_max_sigma_voxels": float(max_sigma_voxels),
        "dgr_base_max_sigma_voxels": float(args.dgr_max_sigma_voxels),
        "dgr_sigma_scale": float(sigma_scale),
        "dgr_normalize_kernel": bool(args.dgr_normalize_kernel),
        "dgr_supersample": int(args.dgr_supersample),
        "dgr_kernel_cutoff": float(args.dgr_kernel_cutoff),
        "dgr_max_splat_radius": int(args.dgr_max_splat_radius),
        "dgr_internal_normalize": not bool(args.dgr_no_normalize_internal),
        "dgr_batch_size": int(batch_size),
        "dgr_batch_count": int(batch_count),
        "dgr_valid_gaussian_ratio": float(valid_count / num_gaussians) if num_gaussians else 0.0,
    }
    return volume, valid_count, stats


def save_preview_slices(volume, out_dir):
    preview_dir = os.path.join(out_dir, "preview_slices")
    os.makedirs(preview_dir, exist_ok=True)
    mids = [s // 2 for s in volume.shape]
    slices = {
        "axial_z.png": volume[:, :, mids[2]],
        "coronal_y.png": volume[:, mids[1], :],
        "sagittal_x.png": volume[mids[0], :, :],
    }
    for name, image in slices.items():
        imageio.imwrite(os.path.join(preview_dir, name), (np.clip(image, 0, 1) * 255).astype(np.uint8))


def save_mhd(volume, geometry, out_dir, basename):
    raw_name = basename + ".raw"
    mhd_name = basename + ".mhd"
    raw_path = os.path.join(out_dir, raw_name)
    mhd_path = os.path.join(out_dir, mhd_name)

    volume.astype(np.float32).tofile(raw_path)
    with open(mhd_path, "w", encoding="utf-8") as handle:
        handle.write("ObjectType = Image\n")
        handle.write("NDims = 3\n")
        handle.write(f"DimSize = {volume.shape[2]} {volume.shape[1]} {volume.shape[0]}\n")
        handle.write("ElementType = MET_FLOAT\n")
        handle.write(
            f"ElementSpacing = {float(geometry.dVoxel[2])} {float(geometry.dVoxel[1])} {float(geometry.dVoxel[0])}\n"
        )
        handle.write("ElementByteOrderMSB = False\n")
        handle.write(f"ElementDataFile = {raw_name}\n")
    return mhd_path, raw_path


def main():
    parser = argparse.ArgumentParser(description="Export trained X-Gaussian model as a real 3D CT volume.")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-s", "--source_path", required=True, help="Original CT pickle used for training.")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--basename", default="recon_volume")
    parser.add_argument("--mode", choices=["center", "gaussian", "dgr"], default="gaussian")
    parser.add_argument("--density", choices=["opacity", "opacity_dc"], default="opacity")
    parser.add_argument("--export_mode", choices=["baseline", "structure_preserved", "structure_density"], default="baseline")
    parser.add_argument("--prior_path", default="", help="prior_data.npz for structure-preserved export.")
    parser.add_argument("--use_roi_gate", action="store_true")
    parser.add_argument("--roi_gate_strength", type=float, default=0.2)
    parser.add_argument("--roi_gate_source", choices=["roi", "occupancy"], default="roi")
    parser.add_argument("--density_mapping_mode", choices=["opacity", "opacity_dc", "opacity_scale", "opacity_feature_scale", "radiodensity", "attenuation", "opacity_radiodensity", "effective_density"], default=None)
    parser.add_argument("--density_calibration", choices=["none", "percentile", "log"], default="percentile")
    parser.add_argument("--calibration_percentile", type=float, default=99.0)
    parser.add_argument("--structure_prior_blend", type=float, default=0.35)
    parser.add_argument("--structure_roi_boost", type=float, default=1.25)
    parser.add_argument("--structure_occupancy_boost", type=float, default=1.35)
    parser.add_argument("--structure_outside_roi_scale", type=float, default=0.08)
    parser.add_argument("--cutoff", type=float, default=3.0)
    parser.add_argument("--dgr_splat_radius", type=int, default=2)
    parser.add_argument("--dgr_min_sigma_voxels", type=float, default=0.5)
    parser.add_argument("--dgr_max_sigma_voxels", type=float, default=4.0)
    parser.add_argument("--dgr_sigma_scale", type=float, default=1.25)
    parser.add_argument("--dgr_reference_grid_size", type=int, default=0, help="Training grid size used to scale DGR voxel sigmas during higher-resolution export. 0 disables scaling.")
    parser.add_argument("--dgr_normalize_kernel", action="store_true")
    parser.add_argument("--dgr_supersample", type=int, default=1)
    parser.add_argument("--dgr_kernel_cutoff", type=float, default=0.0)
    parser.add_argument("--dgr_max_splat_radius", type=int, default=0)
    parser.add_argument("--dgr_batch_size", type=int, default=50000, help="Number of Gaussians processed per DGR export batch. Lower this value to reduce GPU memory.")
    parser.add_argument("--dgr_no_normalize_internal", action="store_true")
    parser.add_argument("--max_gaussians", type=int, default=0, help="0 means use all gaussians.")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--post_median_size", type=int, default=0, help="Optional odd 3D median filter size to suppress blocky black/white holes before Gaussian smoothing.")
    parser.add_argument("--post_smooth_sigma", type=float, default=0.0, help="Optional 3D Gaussian smoothing on the exported volume to reduce block artifacts.")
    parser.add_argument("--post_contrast_gamma", type=float, default=1.0, help="Optional gamma applied after normalization to recover display contrast. Values below 1 brighten low/mid densities.")
    parser.add_argument("--slicer_export", action="store_true", help="Also save a uint16 display volume for 3D Slicer.")
    parser.add_argument("--display_percentile_low", type=float, default=1.0)
    parser.add_argument("--display_percentile_high", type=float, default=99.5)
    parser.add_argument("--display_gamma", type=float, default=0.5)
    parser.add_argument("--no_tiff", action="store_true", help="Do not export the Slicer display TIFF stack.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    safe_state(args.quiet)
    iteration = resolve_iteration(args.model_path, args.iteration)
    out_dir = args.out_dir or os.path.join(args.model_path, "ct_volume", f"iteration_{iteration}")
    os.makedirs(out_dir, exist_ok=True)

    with open(args.source_path, "rb") as handle:
        data = pickle.load(handle)
    geometry = ConeGeometry(data)

    gaussians, ply_path = load_gaussians(args.model_path, iteration, args.sh_degree)
    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
    scales = gaussians.get_scaling.detach().cpu().numpy().astype(np.float32)
    density_mode = args.density_mapping_mode or args.density
    weights = gaussian_density_weights_extended(gaussians, density_mode)
    export_extra_stats = {
        "export_mode": args.export_mode,
        "density_mapping_mode": density_mode,
        "use_roi_gate": bool(args.use_roi_gate),
    }
    roi_mask = None
    occupancy_mask = None
    prior_volume = None
    if args.export_mode in ("structure_preserved", "structure_density"):
        if not args.prior_path:
            raise ValueError(f"{args.export_mode} export requires --prior_path")
        prior_payload = load_prior_payload(args.prior_path)
        roi_mask = prior_payload["roi_mask"]
        occupancy_mask = prior_payload["occupancy_mask"]
        prior_volume = prior_payload["prior_volume"]
        weights, calibration_stats = calibrate_density_weights(
            weights,
            mode=args.density_calibration,
            percentile=args.calibration_percentile,
        )
        export_extra_stats.update(calibration_stats)
        if args.use_roi_gate:
            gate, inside_gate, valid_gate, roi_mask, occupancy_mask = gaussian_roi_gate(
                xyz,
                geometry,
                args.prior_path,
                roi_gate_strength=args.roi_gate_strength,
                gate_source=args.roi_gate_source,
            )
            weights = weights * gate
            export_extra_stats.update({
                "prior_path": args.prior_path,
                "roi_gate_strength": float(args.roi_gate_strength),
                "roi_gate_source": args.roi_gate_source,
                "num_gate_inside_gaussians": int(inside_gate.sum()),
                "gate_inside_ratio": float(inside_gate.mean()) if inside_gate.size else 0.0,
                "num_gate_valid_gaussians": int(valid_gate.sum()),
            })
        else:
            export_extra_stats["prior_path"] = args.prior_path
        export_extra_stats["prior_volume_available"] = bool(prior_volume is not None)
    else:
        weights, calibration_stats = calibrate_density_weights(weights, mode="none")
        export_extra_stats.update(calibration_stats)

    mode_stats = {}
    if args.mode == "center":
        volume, used = center_voxelize(xyz, weights, geometry)
    elif args.mode == "dgr":
        volume, used, mode_stats = dgr_voxelize(gaussians, weights, geometry, args)
    else:
        volume, used = gaussian_voxelize(xyz, scales, weights, geometry, args.cutoff, args.max_gaussians)

    pre_smooth_min = float(volume.min())
    pre_smooth_max = float(volume.max())
    if args.post_median_size > 1:
        volume = smooth_volume_median(volume, args.post_median_size)
    if args.post_smooth_sigma > 0.0:
        volume = smooth_volume_gaussian(volume, args.post_smooth_sigma)

    raw_min = float(volume.min())
    raw_max = float(volume.max())
    structure_stats = {}
    if args.export_mode == "structure_density":
        volume, structure_stats = structure_preserve_volume(
            volume,
            roi_mask=roi_mask,
            occupancy_mask=occupancy_mask,
            prior_volume=prior_volume,
            prior_blend=args.structure_prior_blend,
            roi_boost=args.structure_roi_boost,
            occupancy_boost=args.structure_occupancy_boost,
            outside_roi_scale=args.structure_outside_roi_scale,
        )
        norm_min, norm_max = raw_min, raw_max
    elif not args.no_normalize:
        volume, norm_min, norm_max = normalize_volume(volume)
    else:
        norm_min, norm_max = raw_min, raw_max
    if args.post_contrast_gamma != 1.0:
        volume = apply_contrast_gamma(volume, args.post_contrast_gamma)

    npy_path = os.path.join(out_dir, args.basename + ".npy")
    np.save(npy_path, volume.astype(np.float32))
    mhd_path, raw_path = save_mhd(volume, geometry, out_dir, args.basename)
    save_preview_slices(volume, out_dir)
    region_stats = volume_region_statistics(volume, roi_mask=roi_mask, occupancy_mask=occupancy_mask)
    save_histogram(volume, out_dir, "volume_density")
    if roi_mask is not None or occupancy_mask is not None:
        save_region_preview_slices(volume, roi_mask, occupancy_mask, out_dir)

    slicer_outputs = {}
    if args.slicer_export or args.export_mode == "structure_density":
        display_volume, display_stats = make_slicer_display_volume(
            volume,
            percentile_low=args.display_percentile_low,
            percentile_high=args.display_percentile_high,
            gamma=args.display_gamma,
        )
        slicer_basename = args.basename + "_slicer"
        slicer_npy_path = os.path.join(out_dir, slicer_basename + ".npy")
        np.save(slicer_npy_path, display_volume.astype(np.float32))
        slicer_mhd_path, slicer_raw_path = save_mhd_uint16(display_volume, geometry, out_dir, slicer_basename)
        slicer_outputs = {
            "slicer_npy": slicer_npy_path,
            "slicer_mhd": slicer_mhd_path,
            "slicer_raw": slicer_raw_path,
            **display_stats,
        }
        if not args.no_tiff:
            slicer_outputs["slicer_tiff"] = save_tiff_stack_uint16(display_volume, out_dir, slicer_basename)

    stats = {
        "model_path": args.model_path,
        "source_path": args.source_path,
        "ply_path": ply_path,
        "iteration": int(iteration),
        "mode": args.mode,
        "density": args.density,
        **export_extra_stats,
        **mode_stats,
        "num_gaussians": int(xyz.shape[0]),
        "num_used_gaussians": int(used),
        "volume_shape": list(volume.shape),
        "dVoxel": geometry.dVoxel.tolist(),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "pre_smooth_min": pre_smooth_min,
        "pre_smooth_max": pre_smooth_max,
        "post_median_size": int(args.post_median_size),
        "post_smooth_sigma": float(args.post_smooth_sigma),
        "post_contrast_gamma": float(args.post_contrast_gamma),
        "saved_min": float(volume.min()),
        "saved_max": float(volume.max()),
        "normalization_input_min": norm_min,
        "normalization_input_max": norm_max,
        "structure_density_stats": structure_stats,
        "npy": npy_path,
        "mhd": mhd_path,
        "raw": raw_path,
        **slicer_outputs,
        "region_statistics": region_stats,
    }
    with open(os.path.join(out_dir, "export_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(f"Saved CT volume: {npy_path}")
    print(f"Saved MetaImage: {mhd_path}")
    if slicer_outputs:
        print(f"Saved Slicer MetaImage: {slicer_outputs['slicer_mhd']}")
    print(f"Saved previews: {os.path.join(out_dir, 'preview_slices')}")


if __name__ == "__main__":
    main()
