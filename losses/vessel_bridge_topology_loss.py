from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from utils.differentiable_volume import (
    gaussian_centers_to_density_grid,
    gaussian_density_weights,
)


def _load_optional(payload, keys):
    for key in keys:
        if key in payload:
            return np.asarray(payload[key], dtype=np.float32)
    return None


def _normalize_np(array, eps=1e-6):
    array = np.asarray(array, dtype=np.float32)
    amin = float(array.min())
    amax = float(array.max())
    if amax <= amin:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - amin) / max(amax - amin, eps)).astype(np.float32)


class VesselBridgeTopologyLoss:
    """Patch-based topology bridge loss for second-stage density refinement.

    This loss is intentionally local and one-sided: it raises density on weak
    vessel bridge candidates supported by PriorNet/topology priors, but it does
    not punish density outside the prior. Geometry is expected to be frozen by
    the caller.
    """

    def __init__(
        self,
        topology_prior_path,
        volume_positions,
        patch_size=64,
        patch_count=2,
        density_mode="effective_density",
        splat_mode="trilinear",
        splat_radius=1,
        min_sigma_voxels=0.35,
        max_sigma_voxels=3.2,
        bridge_min_prior=0.20,
        bridge_target_density=0.035,
        bridge_weight_power=1.2,
        include_centerline=True,
        max_anchor_distance=24.0,
    ):
        if not topology_prior_path:
            raise ValueError("VesselBridgeTopologyLoss requires topology_prior_path")
        if volume_positions is None:
            raise ValueError("VesselBridgeTopologyLoss requires volume_positions")
        if splat_mode == "dgr":
            raise ValueError("DGR splatting is not supported during bridge refinement")

        self.topology_prior_path = str(topology_prior_path)
        self.patch_size = int(patch_size)
        self.patch_count = max(1, int(patch_count))
        self.density_mode = str(density_mode)
        self.splat_mode = str(splat_mode)
        self.splat_radius = int(splat_radius)
        self.min_sigma_voxels = float(min_sigma_voxels)
        self.max_sigma_voxels = float(max_sigma_voxels)
        self.bridge_target_density = float(bridge_target_density)
        self.bridge_weight_power = float(bridge_weight_power)

        positions = np.asarray(volume_positions, dtype=np.float32)
        self.grid_min = positions[0, 0, 0].astype(np.float32)
        self.grid_max = positions[-1, -1, -1].astype(np.float32)
        self.full_shape = np.asarray(positions.shape[:3], dtype=np.int64)

        payload = np.load(self.topology_prior_path)
        soft_prior = _load_optional(payload, ("soft_prior", "prior_volume", "vessel_prob"))
        vessel_mask = _load_optional(payload, ("vessel_mask", "occupancy_mask"))
        support_roi = _load_optional(payload, ("support_roi_mask", "support_mask", "roi_mask"))
        centerline = _load_optional(payload, ("centerline_mask", "skeleton_mask", "centerline"))

        if soft_prior is None:
            if vessel_mask is None:
                raise KeyError("topology prior needs soft_prior/prior_volume or vessel_mask")
            soft_prior = vessel_mask
        if soft_prior.shape != tuple(self.full_shape):
            raise ValueError(
                f"soft prior shape {soft_prior.shape} does not match volume shape {tuple(self.full_shape)}"
            )

        soft_prior = _normalize_np(soft_prior)
        vessel_bool = np.asarray(vessel_mask > 0.5, dtype=bool) if vessel_mask is not None else soft_prior >= 0.70
        if support_roi is None:
            support_bool = soft_prior >= float(bridge_min_prior)
        else:
            support_bool = np.asarray(support_roi > 0.5, dtype=bool)
            support_bool = np.logical_or(support_bool, soft_prior >= float(bridge_min_prior))

        bridge_mask = support_bool & (soft_prior >= float(bridge_min_prior))
        if include_centerline and centerline is not None:
            bridge_mask = np.logical_or(bridge_mask, np.asarray(centerline > 0.5, dtype=bool))

        if max_anchor_distance and float(max_anchor_distance) > 0.0 and np.any(vessel_bool):
            try:
                from scipy import ndimage

                dist_to_strict = ndimage.distance_transform_edt(~vessel_bool)
                bridge_mask = bridge_mask & (dist_to_strict <= float(max_anchor_distance))
            except ImportError:
                pass

        bridge_weight = (soft_prior ** float(bridge_weight_power)) * bridge_mask.astype(np.float32)
        candidate_coords = np.argwhere(bridge_weight > 0.0)
        if candidate_coords.size == 0:
            raise ValueError(f"empty bridge candidate map: {topology_prior_path}")

        self.soft_prior_np = soft_prior.astype(np.float32)
        self.bridge_mask_np = bridge_mask.astype(np.float32)
        self.bridge_weight_np = bridge_weight.astype(np.float32)
        self.candidate_coords = candidate_coords.astype(np.int64)

    def _sample_bounds(self):
        center = self.candidate_coords[np.random.randint(0, self.candidate_coords.shape[0])]
        size = int(self.patch_size)
        max_start = self.full_shape - size
        start = center - size // 2
        start = np.minimum(np.maximum(start, 0), max_start).astype(np.int64)
        end = start + size
        return start, end

    def _world_bounds(self, start, end):
        shape_minus_one = np.maximum(self.full_shape.astype(np.float32) - 1.0, 1.0)
        span = self.grid_max - self.grid_min
        start_frac = start.astype(np.float32) / shape_minus_one
        end_frac = (end.astype(np.float32) - 1.0) / shape_minus_one
        patch_min = self.grid_min + start_frac * span
        patch_max = self.grid_min + end_frac * span
        return patch_min.astype(np.float32), patch_max.astype(np.float32)

    def _patch_tensor(self, array, start, end, device):
        patch = array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        return torch.from_numpy(np.ascontiguousarray(patch)).to(device=device, dtype=torch.float32)

    def _pred_patch(self, gaussians, start, end):
        patch_min, patch_max = self._world_bounds(start, end)
        weights = gaussian_density_weights(gaussians, mode=self.density_mode)
        pred_grid, valid = gaussian_centers_to_density_grid(
            gaussians,
            weights,
            patch_min,
            patch_max,
            int(self.patch_size),
            splat_mode=self.splat_mode,
            splat_radius=self.splat_radius,
            min_sigma_voxels=self.min_sigma_voxels,
            max_sigma_voxels=self.max_sigma_voxels,
            normalize=True,
        )
        return pred_grid, valid

    def __call__(self, gaussians):
        losses = []
        raw_losses = []
        pred_means = []
        weight_means = []
        valid_ratios = []
        last_debug = None

        for _ in range(self.patch_count):
            start, end = self._sample_bounds()
            pred_grid, valid = self._pred_patch(gaussians, start, end)
            bridge_weight = self._patch_tensor(self.bridge_weight_np, start, end, pred_grid.device)
            denom = torch.clamp(bridge_weight.sum(), min=1.0)
            coverage = bridge_weight * F.relu(float(self.bridge_target_density) - pred_grid)
            loss = coverage.sum() / denom
            losses.append(loss)

            with torch.no_grad():
                mask = bridge_weight > 0.0
                raw_losses.append(float(loss.detach().item()))
                pred_means.append(float(pred_grid[mask].mean().detach().item()) if torch.any(mask) else 0.0)
                weight_means.append(float(bridge_weight.detach().mean().item()))
                valid_ratios.append(float(valid.float().mean().detach().item()) if valid.numel() else 0.0)
                last_debug = {
                    "pred_bridge_patch": pred_grid.detach(),
                    "bridge_weight_patch": bridge_weight.detach(),
                }

        total = torch.stack(losses).mean()
        stats = {
            "bridge_patch_size": int(self.patch_size),
            "bridge_patch_count": int(self.patch_count),
            "bridge_target_density": float(self.bridge_target_density),
            "bridge_loss_raw": float(np.mean(raw_losses)) if raw_losses else 0.0,
            "bridge_pred_mean": float(np.mean(pred_means)) if pred_means else 0.0,
            "bridge_weight_mean": float(np.mean(weight_means)) if weight_means else 0.0,
            "bridge_valid_gaussian_ratio": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
        }
        return total, stats, last_debug

    def save_debug(self, debug_payload, out_dir):
        if not debug_payload:
            return
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, tensor in debug_payload.items():
            array = tensor.detach().cpu().numpy().astype(np.float32)
            np.save(out_dir / f"{name}.npy", array)
            mid = array.shape[2] // 2
            imageio.imwrite(
                out_dir / f"{name}_axial_mid.png",
                (np.clip(array[:, :, mid], 0.0, 1.0) * 255).astype(np.uint8),
            )
