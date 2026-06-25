import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def load_prior_masks(prior_path):
    payload = np.load(prior_path)
    if "roi_mask" not in payload:
        raise KeyError(f"prior file must contain roi_mask: {prior_path}")
    if "occupancy_mask" not in payload:
        raise KeyError(f"prior file must contain occupancy_mask: {prior_path}")
    return payload["roi_mask"].astype(bool), payload["occupancy_mask"].astype(bool)


def load_prior_payload(prior_path):
    payload = np.load(prior_path)
    if "roi_mask" not in payload:
        raise KeyError(f"prior file must contain roi_mask: {prior_path}")
    if "occupancy_mask" not in payload:
        raise KeyError(f"prior file must contain occupancy_mask: {prior_path}")
    prior_volume = payload["prior_volume"].astype(np.float32) if "prior_volume" in payload else None
    return {
        "roi_mask": payload["roi_mask"].astype(bool),
        "occupancy_mask": payload["occupancy_mask"].astype(bool),
        "prior_volume": prior_volume,
    }


def normalize_minmax(values):
    values = np.asarray(values, dtype=np.float32)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32), vmin, vmax
    return ((values - vmin) / (vmax - vmin)).astype(np.float32), vmin, vmax


def normalize_percentile(values, low=1.0, high=99.5, mask=None):
    values = np.asarray(values, dtype=np.float32)
    sample = values[mask] if mask is not None and np.any(mask) else values.reshape(-1)
    lo = float(np.percentile(sample, low))
    hi = float(np.percentile(sample, high))
    if hi <= lo:
        return normalize_minmax(values)
    out = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float32), lo, hi


def xyz_to_prior_indices(xyz, geometry, mask_shape):
    shape = np.asarray(mask_shape, dtype=np.int64)
    mins = -geometry.sVoxel.astype(np.float32) / 2.0 + geometry.dVoxel.astype(np.float32) / 2.0
    spacing = geometry.dVoxel.astype(np.float32)
    indices = np.rint((xyz - mins[None, :]) / spacing[None, :]).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < shape[None, :]), axis=1)
    indices = np.clip(indices, 0, shape[None, :] - 1)
    return indices, valid


def gaussian_roi_gate(xyz, geometry, prior_path, roi_gate_strength=0.2, gate_source="roi"):
    roi_mask, occupancy_mask = load_prior_masks(prior_path)
    mask = occupancy_mask if gate_source == "occupancy" else roi_mask
    indices, valid = xyz_to_prior_indices(xyz, geometry, mask.shape)
    inside = np.zeros(xyz.shape[0], dtype=bool)
    inside[valid] = mask[indices[valid, 0], indices[valid, 1], indices[valid, 2]]

    gate = np.full(xyz.shape[0], float(roi_gate_strength), dtype=np.float32)
    gate[inside] = 1.0
    gate[~valid] = float(roi_gate_strength)
    return gate, inside, valid, roi_mask, occupancy_mask


def calibrate_density_weights(weights, mode="none", percentile=99.0):
    weights = np.asarray(weights, dtype=np.float32)
    if mode == "none":
        return weights, {"density_calibration": "none"}
    if mode == "percentile":
        high = float(np.percentile(weights, percentile))
        if high <= 1e-8:
            return weights, {"density_calibration": "percentile", "calibration_percentile": percentile, "scale": high}
        return np.clip(weights / high, 0.0, 1.0).astype(np.float32), {
            "density_calibration": "percentile",
            "calibration_percentile": float(percentile),
            "scale": high,
        }
    if mode == "log":
        out = np.log1p(np.maximum(weights, 0.0))
        vmax = float(out.max())
        if vmax > 1e-8:
            out = out / vmax
        return out.astype(np.float32), {"density_calibration": "log", "scale": vmax}
    raise ValueError(f"Unsupported density calibration mode: {mode}")


def structure_preserve_volume(
    volume,
    roi_mask=None,
    occupancy_mask=None,
    prior_volume=None,
    prior_blend=0.35,
    roi_boost=1.25,
    occupancy_boost=1.35,
    outside_roi_scale=0.08,
):
    base, scale_low, scale_high = normalize_percentile(volume, low=1.0, high=99.7, mask=roi_mask)
    stats = {
        "structure_input_percentile_low": float(scale_low),
        "structure_input_percentile_high": float(scale_high),
        "prior_volume_used": False,
        "prior_blend": float(prior_blend),
        "roi_boost": float(roi_boost),
        "occupancy_boost": float(occupancy_boost),
        "outside_roi_scale": float(outside_roi_scale),
    }

    out = base.astype(np.float32, copy=True)
    if prior_volume is not None and prior_volume.shape == out.shape and prior_blend > 0:
        prior_norm, prior_min, prior_max = normalize_minmax(prior_volume)
        blend_mask = roi_mask if roi_mask is not None and roi_mask.shape == out.shape else np.ones_like(out, dtype=bool)
        out[blend_mask] = (
            (1.0 - float(prior_blend)) * out[blend_mask]
            + float(prior_blend) * prior_norm[blend_mask]
        )
        stats.update({
            "prior_volume_used": True,
            "prior_volume_min": float(prior_min),
            "prior_volume_max": float(prior_max),
        })

    if roi_mask is not None and roi_mask.shape == out.shape:
        out[roi_mask] *= float(roi_boost)
        out[~roi_mask] *= float(outside_roi_scale)
    if occupancy_mask is not None and occupancy_mask.shape == out.shape:
        out[occupancy_mask] *= float(occupancy_boost)

    out, final_low, final_high = normalize_percentile(out, low=0.0, high=99.8, mask=roi_mask)
    stats.update({
        "structure_output_percentile_low": float(final_low),
        "structure_output_percentile_high": float(final_high),
    })
    return out.astype(np.float32), stats


def make_slicer_display_volume(
    volume,
    percentile_low=1.0,
    percentile_high=99.5,
    gamma=0.5,
    sample_mask=None,
):
    display, clip_low, clip_high = normalize_percentile(
        volume,
        low=percentile_low,
        high=percentile_high,
        mask=sample_mask,
    )
    gamma = max(float(gamma), 1e-6)
    display = np.power(np.clip(display, 0.0, 1.0), gamma).astype(np.float32)
    stats = {
        "display_percentile_low": float(percentile_low),
        "display_percentile_high": float(percentile_high),
        "display_clip_low": float(clip_low),
        "display_clip_high": float(clip_high),
        "display_gamma": float(gamma),
        "display_masked_percentiles": bool(sample_mask is not None and np.any(sample_mask)),
    }
    return display, stats


def save_mhd_uint16(volume, geometry, out_dir, basename):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_name = basename + ".raw"
    mhd_name = basename + ".mhd"
    raw_path = out_dir / raw_name
    mhd_path = out_dir / mhd_name
    volume_u16 = (np.clip(volume, 0.0, 1.0) * 65535.0).astype(np.uint16)
    volume_u16.tofile(raw_path)
    with open(mhd_path, "w", encoding="utf-8") as handle:
        handle.write("ObjectType = Image\n")
        handle.write("NDims = 3\n")
        handle.write(f"DimSize = {volume.shape[2]} {volume.shape[1]} {volume.shape[0]}\n")
        handle.write("ElementType = MET_USHORT\n")
        handle.write(
            f"ElementSpacing = {float(geometry.dVoxel[2])} {float(geometry.dVoxel[1])} {float(geometry.dVoxel[0])}\n"
        )
        handle.write("ElementByteOrderMSB = False\n")
        handle.write(f"ElementDataFile = {raw_name}\n")
    return str(mhd_path), str(raw_path)


def save_tiff_stack_uint16(volume, out_dir, basename):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tif_path = out_dir / (basename + ".tiff")
    volume_u16 = (np.clip(volume, 0.0, 1.0) * 65535.0).astype(np.uint16)
    imageio.mimwrite(tif_path, list(volume_u16), format="TIFF")
    return str(tif_path)


def summarize_values(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None, "sum": 0.0}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "sum": float(values.sum()),
    }


def volume_region_statistics(volume, roi_mask=None, occupancy_mask=None):
    stats = {"all": summarize_values(volume.reshape(-1))}
    if roi_mask is not None and roi_mask.shape == volume.shape:
        stats["roi"] = summarize_values(volume[roi_mask])
        stats["outside_roi"] = summarize_values(volume[~roi_mask])
    if occupancy_mask is not None and occupancy_mask.shape == volume.shape:
        stats["occupancy"] = summarize_values(volume[occupancy_mask])
        stats["empty_occupancy"] = summarize_values(volume[~occupancy_mask])
    return stats


def save_histogram(values, out_dir, name, bins=64):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    counts, edges = np.histogram(values, bins=bins)
    payload = {"counts": counts.astype(int).tolist(), "bin_edges": edges.astype(float).tolist()}
    with open(out_dir / f"{name}_histogram.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def save_region_preview_slices(volume, roi_mask, occupancy_mask, out_dir):
    out_dir = Path(out_dir) / "region_preview_slices"
    out_dir.mkdir(parents=True, exist_ok=True)
    mids = [s // 2 for s in volume.shape]
    slices = {
        "axial_z": (slice(None), slice(None), mids[2]),
        "coronal_y": (slice(None), mids[1], slice(None)),
        "sagittal_x": (mids[0], slice(None), slice(None)),
    }
    for name, index in slices.items():
        imageio.imwrite(out_dir / f"{name}_volume.png", (np.clip(volume[index], 0, 1) * 255).astype(np.uint8))
        if roi_mask is not None:
            imageio.imwrite(out_dir / f"{name}_roi.png", (roi_mask[index].astype(np.uint8) * 255))
        if occupancy_mask is not None:
            imageio.imwrite(out_dir / f"{name}_occupancy.png", (occupancy_mask[index].astype(np.uint8) * 255))
