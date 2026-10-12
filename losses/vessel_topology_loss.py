import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from pathlib import Path

from utils.differentiable_volume import (
    gaussian_centers_to_density_grid,
    gaussian_density_weights,
)


def _load_topology_array(payload, keys, required=True):
    for key in keys:
        if key in payload:
            return payload[key].astype(np.float32)
    if required:
        raise KeyError(f"topology prior must contain one of: {keys}")
    return None


class VesselTopologyLoss:
    """Low-memory vessel topology regularizer on a low-res differentiable grid."""

    def __init__(
        self,
        prior_path,
        volume_positions,
        grid_size=32,
        patch_mode=False,
        patch_size=64,
        patch_count=1,
        patch_min_weight=0.05,
        density_mode="effective_density",
        splat_mode="trilinear",
        splat_radius=1,
        min_sigma_voxels=0.75,
        max_sigma_voxels=3.0,
        min_density=0.03,
        lambda_topology=1.0,
        lambda_centerline=0.002,
        lambda_support=0.0,
        lambda_offcenter=0.0005,
        lambda_outside=0.001,
        support_min_density=0.0,
        support_weight_power=1.0,
        outside_margin=0.0,
    ):
        if not prior_path:
            raise ValueError("VesselTopologyLoss requires topology_prior_path")
        if volume_positions is None:
            raise ValueError("VesselTopologyLoss requires scene.volume_positions")
        if splat_mode == "dgr":
            raise ValueError("VesselTopologyLoss intentionally disallows DGR during training")

        self.prior_path = str(prior_path)
        self.grid_size = int(grid_size)
        self.patch_mode = bool(patch_mode)
        self.patch_size = int(patch_size)
        self.patch_count = max(1, int(patch_count))
        self.patch_min_weight = float(patch_min_weight)
        self.density_mode = density_mode
        self.splat_mode = splat_mode
        self.splat_radius = int(splat_radius)
        self.min_sigma_voxels = float(min_sigma_voxels)
        self.max_sigma_voxels = float(max_sigma_voxels)
        self.min_density = float(min_density)
        self.lambda_topology = float(lambda_topology)
        self.lambda_centerline = float(lambda_centerline)
        self.lambda_support = float(lambda_support)
        self.lambda_offcenter = float(lambda_offcenter)
        self.lambda_outside = float(lambda_outside)
        self.support_min_density = float(support_min_density)
        self.support_weight_power = float(support_weight_power)
        self.outside_margin = float(outside_margin)

        positions = np.asarray(volume_positions, dtype=np.float32)
        self.grid_min = positions[0, 0, 0].astype("float32")
        self.grid_max = positions[-1, -1, -1].astype("float32")

        payload = np.load(self.prior_path)
        self.centerline_np = _load_topology_array(
            payload,
            ("centerline_mask", "skeleton_mask", "centerline", "skeleton"),
        )
        self.roi_np = _load_topology_array(
            payload,
            ("roi_mask", "vessel_roi", "vessel_mask", "occupancy_mask"),
        )
        self.support_roi_np = _load_topology_array(
            payload,
            ("support_roi_mask", "support_mask", "soft_roi_mask"),
            required=False,
        )
        self.distance_np = _load_topology_array(
            payload,
            ("distance_map", "distance_to_centerline", "centerline_distance"),
            required=False,
        )
        self.vessel_np = _load_topology_array(
            payload,
            ("soft_prior", "prior_volume", "vessel_prob", "vessel_mask", "occupancy_mask"),
            required=False,
        )

        if self.centerline_np.shape != positions.shape[:3]:
            raise ValueError(
                f"centerline shape {self.centerline_np.shape} does not match volume positions {positions.shape[:3]}"
            )
        if self.roi_np.shape != positions.shape[:3]:
            raise ValueError(
                f"roi shape {self.roi_np.shape} does not match volume positions {positions.shape[:3]}"
            )
        if self.support_roi_np is not None and self.support_roi_np.shape != positions.shape[:3]:
            raise ValueError(
                f"support roi shape {self.support_roi_np.shape} does not match volume positions {positions.shape[:3]}"
            )
        if self.distance_np is not None and self.distance_np.shape != positions.shape[:3]:
            raise ValueError(
                f"distance map shape {self.distance_np.shape} does not match volume positions {positions.shape[:3]}"
            )
        if self.vessel_np is not None and self.vessel_np.shape != positions.shape[:3]:
            raise ValueError(
                f"vessel mask shape {self.vessel_np.shape} does not match volume positions {positions.shape[:3]}"
            )

        self.centerline_np = (self.centerline_np > 0.5).astype(np.float32)
        self.roi_np = (self.roi_np > 0.5).astype(np.float32)
        if self.support_roi_np is not None:
            self.support_roi_np = (self.support_roi_np > 0.5).astype(np.float32)
        if self.vessel_np is not None:
            self.vessel_np = np.clip(self.vessel_np, 0.0, 1.0).astype(np.float32)
        if self.distance_np is None:
            self.distance_np = (1.0 - self.roi_np).astype(np.float32)
        else:
            self.distance_np = np.clip(self.distance_np, 0.0, None).astype(np.float32)
            max_distance = float(np.max(self.distance_np))
            if max_distance > 0.0:
                self.distance_np = self.distance_np / max_distance

        if float(self.centerline_np.sum()) <= 0.0:
            raise ValueError(f"topology prior has empty centerline: {self.prior_path}")

        self._prepare_patch_sampling()
        self._grid_cache = {}

    def can_reuse_pred_grid(self, volume_loss):
        return (
            not self.patch_mode
            and
            volume_loss is not None
            and int(getattr(volume_loss, "grid_size", -1)) == self.grid_size
            and getattr(volume_loss, "density_mode", None) == self.density_mode
            and getattr(volume_loss, "splat_mode", None) == self.splat_mode
        )

    def _support_weight_np(self):
        if self.support_roi_np is not None:
            support_roi = np.maximum(self.support_roi_np, self.roi_np).astype(np.float32)
        else:
            support_roi = self.roi_np.astype(np.float32)
        if self.vessel_np is None:
            support_weight = support_roi
        else:
            support_weight = np.clip(self.vessel_np, 0.0, 1.0) * support_roi
            if self.support_weight_power != 1.0:
                support_weight = np.power(
                    np.clip(support_weight, 0.0, 1.0),
                    float(self.support_weight_power),
                )
        return np.asarray(support_weight, dtype=np.float32)

    def _prepare_patch_sampling(self):
        self.patch_candidate_coords = None
        self.full_shape = np.asarray(self.centerline_np.shape, dtype=np.int64)
        self.full_grid_min = np.asarray(self.grid_min, dtype=np.float32)
        self.full_grid_max = np.asarray(self.grid_max, dtype=np.float32)
        if not self.patch_mode:
            return
        if len(set(int(v) for v in self.full_shape.tolist())) != 1:
            raise ValueError(f"patch topology requires cubic prior shape, got {self.full_shape.tolist()}")
        if self.patch_size < 8:
            raise ValueError("topology_patch_size must be at least 8")
        if self.patch_size > int(self.full_shape[0]):
            raise ValueError(
                f"topology_patch_size {self.patch_size} exceeds prior shape {self.full_shape.tolist()}"
            )
        support_weight = self._support_weight_np()
        candidates = np.argwhere(support_weight >= float(self.patch_min_weight))
        if candidates.size == 0:
            candidates = np.argwhere(support_weight > 0.0)
        if candidates.size == 0:
            candidates = np.argwhere(self.roi_np > 0.0)
        if candidates.size == 0:
            raise ValueError(f"topology patch mode found no candidate voxels: {self.prior_path}")
        self.patch_candidate_coords = candidates.astype(np.int64)

    def _downsample_np(self, array, device, mode="trilinear"):
        tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32)[None, None]
        tensor = F.interpolate(
            tensor,
            size=(self.grid_size, self.grid_size, self.grid_size),
            mode=mode,
            align_corners=False if mode == "trilinear" else None,
        )
        return tensor[0, 0].clamp(0.0, 1.0)

    def prior_grids(self, device):
        key = str(device)
        if key not in self._grid_cache:
            centerline = self._downsample_np(self.centerline_np, device)
            centerline = centerline.clamp(0.0, 1.0)
            roi = self._downsample_np(self.roi_np, device)
            roi = (roi > 0.25).to(torch.float32)
            if self.support_roi_np is not None:
                support_roi = self._downsample_np(self.support_roi_np, device)
                support_roi = (support_roi > 0.10).to(torch.float32)
                support_roi = torch.maximum(support_roi, roi)
            else:
                support_roi = roi
            distance = self._downsample_np(self.distance_np, device)
            distance = distance * (1.0 - support_roi) + 0.25 * distance * support_roi
            vessel = None
            if self.vessel_np is not None:
                vessel = self._downsample_np(self.vessel_np, device)
            support_weight = support_roi
            if vessel is not None:
                support_weight = vessel.clamp(0.0, 1.0) * support_roi
                if self.support_weight_power != 1.0:
                    support_weight = torch.pow(
                        support_weight.clamp(0.0, 1.0),
                        float(self.support_weight_power),
                    )
            self._grid_cache[key] = {
                "centerline": centerline,
                "roi": roi,
                "support_roi": support_roi,
                "support_weight": support_weight,
                "distance": distance,
                "outside": (1.0 - support_roi).clamp(0.0, 1.0),
                "vessel": vessel,
            }
        return self._grid_cache[key]

    def build_pred_grid(self, gaussians):
        weights = gaussian_density_weights(gaussians, mode=self.density_mode)
        pred_grid, valid = gaussian_centers_to_density_grid(
            gaussians,
            weights,
            self.grid_min,
            self.grid_max,
            self.grid_size,
            splat_mode=self.splat_mode,
            splat_radius=self.splat_radius,
            min_sigma_voxels=self.min_sigma_voxels,
            max_sigma_voxels=self.max_sigma_voxels,
            normalize=True,
        )
        return pred_grid, valid

    def _sample_patch_bounds(self):
        coords = self.patch_candidate_coords
        center = coords[np.random.randint(0, coords.shape[0])]
        size = int(self.patch_size)
        max_start = self.full_shape - size
        start = center - size // 2
        start = np.minimum(np.maximum(start, 0), max_start).astype(np.int64)
        end = start + size
        return start, end

    def _patch_world_bounds(self, start, end):
        shape_minus_one = np.maximum(self.full_shape.astype(np.float32) - 1.0, 1.0)
        grid_span = self.full_grid_max - self.full_grid_min
        start_frac = start.astype(np.float32) / shape_minus_one
        end_frac = (end.astype(np.float32) - 1.0) / shape_minus_one
        patch_min = self.full_grid_min + start_frac * grid_span
        patch_max = self.full_grid_min + end_frac * grid_span
        return patch_min.astype(np.float32), patch_max.astype(np.float32)

    def _patch_tensor(self, array, start, end, device, binary_threshold=None):
        patch = array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        tensor = torch.from_numpy(np.ascontiguousarray(patch)).to(device=device, dtype=torch.float32)
        tensor = tensor.clamp(0.0, 1.0)
        if binary_threshold is not None:
            tensor = (tensor > float(binary_threshold)).to(torch.float32)
        return tensor

    def _patch_grids(self, start, end, device):
        centerline = self._patch_tensor(self.centerline_np, start, end, device)
        roi = self._patch_tensor(self.roi_np, start, end, device, binary_threshold=0.5)
        if self.support_roi_np is not None:
            support_roi = self._patch_tensor(self.support_roi_np, start, end, device, binary_threshold=0.5)
            support_roi = torch.maximum(support_roi, roi)
        else:
            support_roi = roi
        distance = self._patch_tensor(self.distance_np, start, end, device)
        distance = distance * (1.0 - support_roi) + 0.25 * distance * support_roi
        vessel = None
        if self.vessel_np is not None:
            vessel = self._patch_tensor(self.vessel_np, start, end, device)
        support_weight = support_roi
        if vessel is not None:
            support_weight = vessel.clamp(0.0, 1.0) * support_roi
            if self.support_weight_power != 1.0:
                support_weight = torch.pow(
                    support_weight.clamp(0.0, 1.0),
                    float(self.support_weight_power),
                )
        return {
            "centerline": centerline,
            "roi": roi,
            "support_roi": support_roi,
            "support_weight": support_weight,
            "distance": distance,
            "outside": (1.0 - support_roi).clamp(0.0, 1.0),
            "vessel": vessel,
        }

    def _build_pred_patch(self, gaussians, start, end):
        weights = gaussian_density_weights(gaussians, mode=self.density_mode)
        patch_min, patch_max = self._patch_world_bounds(start, end)
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

    def __call__(self, gaussians, pred_grid=None):
        if self.patch_mode:
            return self._call_patch_mode(gaussians)
        reused_pred_grid = pred_grid is not None
        valid = None
        if pred_grid is None:
            pred_grid, valid = self.build_pred_grid(gaussians)
        grids = self.prior_grids(pred_grid.device)
        centerline = grids["centerline"]
        distance = grids["distance"]
        outside = grids["outside"]

        centerline_denom = torch.clamp(centerline.sum(), min=1.0)
        centerline_loss = (
            centerline * F.relu(float(self.min_density) - pred_grid)
        ).sum() / centerline_denom
        support_weight = grids["support_weight"]
        support_denom = torch.clamp(support_weight.sum(), min=1.0)
        support_loss = (
            support_weight * F.relu(float(self.support_min_density) - pred_grid)
        ).sum() / support_denom
        offcenter_loss = (pred_grid * distance).mean()
        outside_values = F.relu(pred_grid - float(self.outside_margin)) * outside
        outside_loss = outside_values.sum() / torch.clamp(outside.sum(), min=1.0)

        total = self.lambda_topology * (
            self.lambda_centerline * centerline_loss
            + self.lambda_support * support_loss
            + self.lambda_offcenter * offcenter_loss
            + self.lambda_outside * outside_loss
        )

        with torch.no_grad():
            centerline_values = pred_grid[centerline > 0.0]
            roi = grids["roi"]
            support_roi = grids["support_roi"]
            outside_ratio = float((pred_grid[outside > 0.5] > self.min_density).float().mean().item()) if torch.any(outside > 0.5) else 0.0
            stats = {
                "topology_grid_size": self.grid_size,
                "topology_density_mode": self.density_mode,
                "topology_splat_mode": self.splat_mode,
                "topology_reused_pred_grid": bool(reused_pred_grid),
                "topology_lambda": self.lambda_topology,
                "lambda_topology_centerline": self.lambda_centerline,
                "lambda_topology_support": self.lambda_support,
                "lambda_topology_offcenter": self.lambda_offcenter,
                "lambda_topology_outside": self.lambda_outside,
                "topology_min_density": self.min_density,
                "topology_support_min_density": self.support_min_density,
                "topology_centerline_loss_raw": float(centerline_loss.detach().item()),
                "topology_support_loss_raw": float(support_loss.detach().item()),
                "topology_offcenter_loss_raw": float(offcenter_loss.detach().item()),
                "topology_outside_loss_raw": float(outside_loss.detach().item()),
                "topology_centerline_ratio": float((centerline > 0.0).float().mean().detach().item()),
                "topology_centerline_weight_mean": float(centerline.float().mean().detach().item()),
                "topology_roi_ratio": float(roi.float().mean().detach().item()),
                "topology_support_roi_ratio": float(support_roi.float().mean().detach().item()),
                "topology_support_weight_mean": float(support_weight.float().mean().detach().item()),
                "topology_distance_mean": float(distance.detach().mean().item()),
                "topology_outside_pred_ratio": outside_ratio,
                "topology_centerline_pred_mean": float(centerline_values.mean().detach().item()) if centerline_values.numel() else 0.0,
            }
            if valid is not None:
                stats["topology_valid_gaussian_ratio"] = float(valid.float().mean().detach().item()) if valid.numel() else 0.0
        return total, stats, pred_grid, grids

    def _loss_from_grids(self, pred_grid, grids):
        centerline = grids["centerline"]
        distance = grids["distance"]
        outside = grids["outside"]
        centerline_denom = torch.clamp(centerline.sum(), min=1.0)
        centerline_loss = (
            centerline * F.relu(float(self.min_density) - pred_grid)
        ).sum() / centerline_denom
        support_weight = grids["support_weight"]
        support_denom = torch.clamp(support_weight.sum(), min=1.0)
        support_loss = (
            support_weight * F.relu(float(self.support_min_density) - pred_grid)
        ).sum() / support_denom
        offcenter_loss = (pred_grid * distance).mean()
        outside_values = F.relu(pred_grid - float(self.outside_margin)) * outside
        outside_loss = outside_values.sum() / torch.clamp(outside.sum(), min=1.0)
        total = self.lambda_topology * (
            self.lambda_centerline * centerline_loss
            + self.lambda_support * support_loss
            + self.lambda_offcenter * offcenter_loss
            + self.lambda_outside * outside_loss
        )
        return total, centerline_loss, support_loss, offcenter_loss, outside_loss

    def _call_patch_mode(self, gaussians):
        losses = []
        stats_accum = []
        last_pred = None
        last_grids = None
        valid_ratios = []
        for _ in range(int(self.patch_count)):
            start, end = self._sample_patch_bounds()
            pred_grid, valid = self._build_pred_patch(gaussians, start, end)
            grids = self._patch_grids(start, end, pred_grid.device)
            total, centerline_loss, support_loss, offcenter_loss, outside_loss = self._loss_from_grids(
                pred_grid,
                grids,
            )
            losses.append(total)
            last_pred = pred_grid
            last_grids = grids
            with torch.no_grad():
                support_weight = grids["support_weight"]
                stats_accum.append({
                    "centerline_loss": float(centerline_loss.detach().item()),
                    "support_loss": float(support_loss.detach().item()),
                    "offcenter_loss": float(offcenter_loss.detach().item()),
                    "outside_loss": float(outside_loss.detach().item()),
                    "centerline_ratio": float((grids["centerline"] > 0.0).float().mean().detach().item()),
                    "support_roi_ratio": float(grids["support_roi"].float().mean().detach().item()),
                    "support_weight_mean": float(support_weight.float().mean().detach().item()),
                    "support_pred_mean": float(pred_grid[support_weight > 0.0].mean().detach().item()) if torch.any(support_weight > 0.0) else 0.0,
                })
                valid_ratios.append(float(valid.float().mean().detach().item()) if valid.numel() else 0.0)
        total_loss = torch.stack(losses).mean() if len(losses) > 1 else losses[0]

        def mean_stat(name):
            return float(np.mean([item[name] for item in stats_accum])) if stats_accum else 0.0

        stats = {
            "topology_patch_mode": True,
            "topology_grid_size": int(self.patch_size),
            "topology_patch_size": int(self.patch_size),
            "topology_patch_count": int(self.patch_count),
            "topology_patch_min_weight": float(self.patch_min_weight),
            "topology_density_mode": self.density_mode,
            "topology_splat_mode": self.splat_mode,
            "topology_reused_pred_grid": False,
            "topology_lambda": self.lambda_topology,
            "lambda_topology_centerline": self.lambda_centerline,
            "lambda_topology_support": self.lambda_support,
            "lambda_topology_offcenter": self.lambda_offcenter,
            "lambda_topology_outside": self.lambda_outside,
            "topology_min_density": self.min_density,
            "topology_support_min_density": self.support_min_density,
            "topology_centerline_loss_raw": mean_stat("centerline_loss"),
            "topology_support_loss_raw": mean_stat("support_loss"),
            "topology_offcenter_loss_raw": mean_stat("offcenter_loss"),
            "topology_outside_loss_raw": mean_stat("outside_loss"),
            "topology_centerline_ratio": mean_stat("centerline_ratio"),
            "topology_support_roi_ratio": mean_stat("support_roi_ratio"),
            "topology_support_weight_mean": mean_stat("support_weight_mean"),
            "topology_support_pred_mean": mean_stat("support_pred_mean"),
            "topology_valid_gaussian_ratio": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
        }
        return total_loss, stats, last_pred, last_grids

    def save_debug(self, pred_grid, grids, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays = {
            "pred_topology_grid": pred_grid.detach().cpu().numpy().astype(np.float32),
            "centerline_grid": grids["centerline"].detach().cpu().numpy().astype(np.float32),
            "roi_grid": grids["roi"].detach().cpu().numpy().astype(np.float32),
            "support_roi_grid": grids["support_roi"].detach().cpu().numpy().astype(np.float32),
            "support_weight_grid": grids["support_weight"].detach().cpu().numpy().astype(np.float32),
            "distance_grid": grids["distance"].detach().cpu().numpy().astype(np.float32),
        }
        for name, array in arrays.items():
            np.save(out_dir / f"{name}.npy", array)

        mids = [size // 2 for size in arrays["pred_topology_grid"].shape]
        slices = {
            "axial_z": (slice(None), slice(None), mids[2]),
            "coronal_y": (slice(None), mids[1], slice(None)),
            "sagittal_x": (mids[0], slice(None), slice(None)),
        }
        for view, index in slices.items():
            for name, array in arrays.items():
                image = np.clip(array[index], 0.0, 1.0)
                imageio.imwrite(out_dir / f"{view}_{name}.png", (image * 255).astype(np.uint8))
