import json
from pathlib import Path

import numpy as np
import torch


class TopologyAwareGaussianController:
    def __init__(
        self,
        prior_path,
        volume_positions,
        topology_prior_path="",
        roi_source="support_roi",
        roi_densify_weight=1.25,
        outside_densify_weight=0.75,
        outside_grid_densify_weight=0.50,
        empty_prune_enabled=False,
        empty_prune_mask_source="vessel",
        empty_prune_density_mode="effective_density",
        empty_prune_threshold=0.002,
        centerline_enabled=False,
        centerline_protect_distance=0.08,
        centerline_far_distance=0.35,
        centerline_prune_threshold=0.0015,
        soft_empty_margin=0.05,
        soft_empty_tau=0.05,
        soft_empty_density_mode="opacity",
        component_small_voxels=260,
    ):
        self.prior_path = str(prior_path)
        self.topology_prior_path = str(topology_prior_path or "")
        self.roi_source = str(roi_source)
        self.roi_densify_weight = float(roi_densify_weight)
        self.outside_densify_weight = float(outside_densify_weight)
        self.outside_grid_densify_weight = float(outside_grid_densify_weight)
        self.empty_prune_enabled = bool(empty_prune_enabled)
        self.empty_prune_mask_source = str(empty_prune_mask_source)
        self.empty_prune_density_mode = str(empty_prune_density_mode)
        self.empty_prune_threshold = float(empty_prune_threshold)
        self.centerline_enabled = bool(centerline_enabled)
        self.centerline_protect_distance = float(centerline_protect_distance)
        self.centerline_far_distance = float(centerline_far_distance)
        self.centerline_prune_threshold = float(centerline_prune_threshold)
        self.soft_empty_margin = float(soft_empty_margin)
        self.soft_empty_tau = float(soft_empty_tau)
        self.soft_empty_density_mode = str(soft_empty_density_mode)
        self.component_small_voxels = int(component_small_voxels)
        self._tensor_cache = {}

        prior_payload = np.load(prior_path)
        topology_payload = np.load(topology_prior_path) if topology_prior_path else None

        self.roi_mask_np = self._select_roi_mask(prior_payload, topology_payload)
        self.vessel_mask_np = self._select_mask(
            topology_payload,
            ("vessel_mask", "roi_mask", "support_roi_mask"),
            fallback_payload=prior_payload,
            fallback_keys=("occupancy_mask", "roi_mask"),
            required=True,
        ).astype(bool)
        self.centerline_mask_np = self._select_mask(
            topology_payload,
            ("centerline_mask",),
            fallback_payload=None,
            fallback_keys=(),
            required=False,
        )
        if self.centerline_mask_np is not None:
            self.centerline_mask_np = self.centerline_mask_np.astype(bool)

        distance_map = self._select_mask(
            topology_payload,
            ("distance_map", "centerline_distance", "centerline_distance_map"),
            fallback_payload=None,
            fallback_keys=(),
            required=False,
        )
        self.distance_map_np = distance_map.astype(np.float32) if distance_map is not None else None
        if self.centerline_enabled and self.distance_map_np is None:
            raise ValueError(
                "centerline prior is enabled but topology_prior_path does not contain distance_map"
            )
        candidate_distance_map = self._select_mask(
            topology_payload,
            ("candidate_distance_map", "roi_distance_map", "empty_distance_map"),
            fallback_payload=None,
            fallback_keys=(),
            required=False,
        )
        self.candidate_distance_map_np = (
            candidate_distance_map.astype(np.float32)
            if candidate_distance_map is not None
            else None
        )

        positions = np.asarray(volume_positions, dtype=np.float32)
        if positions.shape[:3] != self.roi_mask_np.shape:
            raise ValueError(
                f"topology ROI shape {self.roi_mask_np.shape} does not match volume grid {positions.shape[:3]}"
            )
        self.grid_shape = np.asarray(self.roi_mask_np.shape, dtype=np.int64)
        self.grid_min = positions[0, 0, 0].astype(np.float32)
        self.grid_max = positions[-1, -1, -1].astype(np.float32)
        self.grid_spacing = (self.grid_max - self.grid_min) / np.maximum(
            self.grid_shape.astype(np.float32) - 1.0,
            1.0,
        )
        self.grid_spacing = np.where(
            np.abs(self.grid_spacing) < 1e-8,
            1.0,
            self.grid_spacing,
        ).astype(np.float32)

        self.mask_stats = self._component_stats(self.vessel_mask_np)
        self.mask_stats["centerline_voxels"] = (
            int(self.centerline_mask_np.sum()) if self.centerline_mask_np is not None else 0
        )

    @staticmethod
    def _select_mask(payload, keys, fallback_payload=None, fallback_keys=(), required=False):
        if payload is not None:
            for key in keys:
                if key in payload:
                    return np.asarray(payload[key])
        if fallback_payload is not None:
            for key in fallback_keys:
                if key in fallback_payload:
                    return np.asarray(fallback_payload[key])
        if required:
            raise KeyError(f"Could not find any mask key from {keys} or {fallback_keys}")
        return None

    def _select_roi_mask(self, prior_payload, topology_payload):
        if self.roi_source == "support_roi":
            keys = ("support_roi_mask", "roi_mask", "vessel_mask")
        elif self.roi_source == "vessel":
            keys = ("vessel_mask", "roi_mask", "support_roi_mask")
        elif self.roi_source == "roi":
            keys = ("roi_mask", "support_roi_mask", "vessel_mask")
        else:
            raise ValueError(f"Unsupported topology roi_source: {self.roi_source}")
        return self._select_mask(
            topology_payload,
            keys,
            fallback_payload=prior_payload,
            fallback_keys=("roi_mask", "occupancy_mask"),
            required=True,
        ).astype(bool)

    def _cached_tensor(self, name, array, device, dtype=None):
        key = (name, str(device), str(dtype))
        if key not in self._tensor_cache:
            tensor = torch.from_numpy(array).to(device=device)
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            self._tensor_cache[key] = tensor
        return self._tensor_cache[key]

    def _lookup_array(self, xyz, array, name):
        device = xyz.device
        grid_min = torch.tensor(self.grid_min, dtype=xyz.dtype, device=device)
        grid_spacing = torch.tensor(self.grid_spacing, dtype=xyz.dtype, device=device)
        grid_shape = torch.tensor(self.grid_shape, dtype=torch.long, device=device)

        indices = torch.round((xyz - grid_min[None, :]) / grid_spacing[None, :]).long()
        valid = torch.all((indices >= 0) & (indices < grid_shape[None, :]), dim=1)
        clamped = torch.minimum(
            torch.maximum(indices, torch.zeros_like(indices)),
            grid_shape[None, :] - 1,
        )
        dtype = xyz.dtype if np.issubdtype(array.dtype, np.floating) else None
        tensor = self._cached_tensor(name, array, device=device, dtype=dtype)
        values = tensor[clamped[:, 0], clamped[:, 1], clamped[:, 2]]
        return values, valid

    def region_masks(self, gaussians):
        xyz = gaussians.get_xyz.detach()
        roi_values, valid_mask = self._lookup_array(xyz, self.roi_mask_np, "roi_mask")
        vessel_values, _ = self._lookup_array(xyz, self.vessel_mask_np, "vessel_mask")
        roi_mask = torch.logical_and(roi_values.bool(), valid_mask)
        vessel_mask = torch.logical_and(vessel_values.bool(), valid_mask)
        outside_roi_mask = torch.logical_and(valid_mask, torch.logical_not(roi_mask))
        outside_vessel_mask = torch.logical_and(valid_mask, torch.logical_not(vessel_mask))
        outside_grid_mask = torch.logical_not(valid_mask)
        result = {
            "roi": roi_mask,
            "vessel": vessel_mask,
            "outside_roi": outside_roi_mask,
            "outside_vessel": outside_vessel_mask,
            "outside_grid": outside_grid_mask,
            "valid": valid_mask,
        }
        if self.distance_map_np is not None:
            distances, _ = self._lookup_array(xyz, self.distance_map_np, "distance_map")
            result["centerline_distance"] = distances.reshape(-1)
        if self.candidate_distance_map_np is not None:
            candidate_distances, _ = self._lookup_array(
                xyz,
                self.candidate_distance_map_np,
                "candidate_distance_map",
            )
            result["candidate_distance"] = candidate_distances.reshape(-1)
        return result

    def _density_values(self, gaussians, mode):
        mode = str(mode)
        opacity = gaussians.get_opacity.detach().reshape(-1)
        if mode == "opacity":
            return opacity
        radiodensity = gaussians.get_radiodensity.detach().reshape(-1)
        if mode == "radiodensity":
            return radiodensity
        if mode in ("opacity_radiodensity", "effective_density"):
            return opacity * radiodensity
        raise ValueError(f"Unsupported topology prune density mode: {mode}")

    @staticmethod
    def _ratio(count, total):
        return float(count / total) if total > 0 else 0.0

    def allocation_statistics(self, gaussians, iteration=None):
        masks = self.region_masks(gaussians)
        total = int(gaussians.get_xyz.shape[0])
        roi_count = int(masks["roi"].sum().item())
        vessel_count = int(masks["vessel"].sum().item())
        outside_roi_count = int(masks["outside_roi"].sum().item())
        outside_vessel_count = int(masks["outside_vessel"].sum().item())
        outside_grid_count = int(masks["outside_grid"].sum().item())
        density = self._density_values(gaussians, self.empty_prune_density_mode)
        opacity = gaussians.get_opacity.detach().reshape(-1)
        empty_mask = (
            masks["outside_vessel"]
            if self.empty_prune_mask_source == "vessel"
            else masks["outside_roi"]
        )
        prune_candidates = torch.logical_and(empty_mask, density < self.empty_prune_threshold)
        stats = {
            "num_gaussians": total,
            "num_inside_grid": int(masks["valid"].sum().item()),
            "num_outside_grid_gaussians": outside_grid_count,
            "num_roi_gaussians": roi_count,
            "roi_gaussian_ratio": self._ratio(roi_count, total),
            "num_vessel_gaussians": vessel_count,
            "vessel_gaussian_ratio": self._ratio(vessel_count, total),
            "num_outside_roi_gaussians": outside_roi_count,
            "outside_roi_gaussian_ratio": self._ratio(outside_roi_count, total),
            "num_outside_vessel_gaussians": outside_vessel_count,
            "outside_vessel_gaussian_ratio": self._ratio(outside_vessel_count, total),
            "num_extra_prune_candidates": int(prune_candidates.sum().item()),
            "empty_prune_mask_source": self.empty_prune_mask_source,
            "empty_prune_density_mode": self.empty_prune_density_mode,
            "empty_prune_threshold": self.empty_prune_threshold,
            "mean_opacity_inside_roi": float(opacity[masks["roi"]].mean().item()) if torch.any(masks["roi"]) else None,
            "mean_opacity_outside_roi": float(opacity[masks["outside_roi"]].mean().item()) if torch.any(masks["outside_roi"]) else None,
            **self.mask_stats,
        }
        if "centerline_distance" in masks:
            distance = masks["centerline_distance"]
            near_centerline = torch.logical_and(
                masks["valid"],
                distance <= self.centerline_protect_distance,
            )
            stats["num_centerline_protected_gaussians"] = int(near_centerline.sum().item())
            stats["centerline_protected_gaussian_ratio"] = self._ratio(
                stats["num_centerline_protected_gaussians"],
                total,
            )
        if "candidate_distance" in masks:
            candidate_distance = masks["candidate_distance"]
            outside_distance = candidate_distance[masks["outside_roi"]]
            stats["mean_candidate_distance_outside_roi"] = (
                float(outside_distance.mean().item()) if outside_distance.numel() > 0 else None
            )
        if iteration is not None:
            stats["iteration"] = int(iteration)
        return stats

    def soft_empty_loss(self, gaussians):
        if self.candidate_distance_map_np is None:
            device = gaussians.get_xyz.device
            return torch.zeros((), dtype=gaussians.get_xyz.dtype, device=device), {
                "topology_empty_loss_available": False,
            }
        masks = self.region_masks(gaussians)
        distance = masks["candidate_distance"]
        density = self._density_values(gaussians, self.soft_empty_density_mode)
        penalty = torch.sigmoid(
            (distance - self.soft_empty_margin) / max(self.soft_empty_tau, 1e-6)
        )
        valid = masks["valid"]
        if torch.any(valid):
            loss = (density[valid] * penalty[valid]).mean()
        else:
            loss = torch.zeros((), dtype=gaussians.get_xyz.dtype, device=gaussians.get_xyz.device)
        outside = masks["outside_roi"]
        return loss, {
            "topology_empty_loss_available": True,
            "topology_empty_loss_density_mode": self.soft_empty_density_mode,
            "topology_empty_margin": self.soft_empty_margin,
            "topology_empty_tau": self.soft_empty_tau,
            "topology_empty_penalty_mean": float(penalty[valid].mean().item()) if torch.any(valid) else 0.0,
            "topology_empty_penalty_outside_mean": float(penalty[outside].mean().item()) if torch.any(outside) else 0.0,
        }

    def build_iteration_bias(self, gaussians, centerline_active=False, empty_prune_active=None):
        xyz = gaussians.get_xyz.detach()
        masks = self.region_masks(gaussians)
        grad_multipliers = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
        grad_multipliers[masks["roi"]] = self.roi_densify_weight
        grad_multipliers[masks["outside_roi"]] = self.outside_densify_weight
        grad_multipliers[masks["outside_grid"]] = self.outside_grid_densify_weight

        density = self._density_values(gaussians, self.empty_prune_density_mode)
        empty_region = (
            masks["outside_vessel"]
            if self.empty_prune_mask_source == "vessel"
            else masks["outside_roi"]
        )
        extra_prune_mask = torch.zeros((xyz.shape[0],), dtype=torch.bool, device=xyz.device)
        if empty_prune_active is None:
            empty_prune_active = self.empty_prune_enabled
        if self.empty_prune_enabled and empty_prune_active:
            extra_prune_mask = torch.logical_and(
                empty_region,
                density < self.empty_prune_threshold,
            )

        prune_protect_mask = torch.zeros((xyz.shape[0],), dtype=torch.bool, device=xyz.device)
        far_centerline_prune_mask = torch.zeros((xyz.shape[0],), dtype=torch.bool, device=xyz.device)
        if centerline_active and self.centerline_enabled and "centerline_distance" in masks:
            distance = masks["centerline_distance"]
            prune_protect_mask = torch.logical_and(
                masks["vessel"],
                distance <= self.centerline_protect_distance,
            )
            far_centerline_prune_mask = torch.logical_and(
                masks["valid"],
                distance >= self.centerline_far_distance,
            )
            far_centerline_prune_mask = torch.logical_and(
                far_centerline_prune_mask,
                density < self.centerline_prune_threshold,
            )
            extra_prune_mask = torch.logical_or(extra_prune_mask, far_centerline_prune_mask)

        stats = self.allocation_statistics(gaussians)
        stats.update(
            {
                "topology_roi_densify_weight": self.roi_densify_weight,
                "topology_outside_densify_weight": self.outside_densify_weight,
                "topology_outside_grid_densify_weight": self.outside_grid_densify_weight,
                "topology_empty_prune_enabled": self.empty_prune_enabled,
                "topology_empty_prune_active": bool(self.empty_prune_enabled and empty_prune_active),
                "topology_centerline_active": bool(centerline_active and self.centerline_enabled),
                "num_topology_extra_prune": int(extra_prune_mask.sum().item()),
                "num_topology_prune_protected": int(prune_protect_mask.sum().item()),
                "num_topology_far_centerline_prune": int(far_centerline_prune_mask.sum().item()),
            }
        )
        return grad_multipliers, extra_prune_mask, prune_protect_mask, masks["roi"], stats

    def save_description(self, model_path):
        out_path = Path(model_path) / "topology_gaussian_prior" / "description.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "prior_path": self.prior_path,
            "topology_prior_path": self.topology_prior_path,
            "roi_source": self.roi_source,
            "roi_shape": list(self.roi_mask_np.shape),
            "roi_voxels": int(self.roi_mask_np.sum()),
            "vessel_voxels": int(self.vessel_mask_np.sum()),
            "centerline_voxels": int(self.centerline_mask_np.sum()) if self.centerline_mask_np is not None else 0,
            "grid_min": self.grid_min.tolist(),
            "grid_max": self.grid_max.tolist(),
            "grid_spacing": self.grid_spacing.tolist(),
            "roi_densify_weight": self.roi_densify_weight,
            "outside_densify_weight": self.outside_densify_weight,
            "empty_prune_enabled": self.empty_prune_enabled,
            "empty_prune_mask_source": self.empty_prune_mask_source,
            "empty_prune_density_mode": self.empty_prune_density_mode,
            "empty_prune_threshold": self.empty_prune_threshold,
            "centerline_enabled": self.centerline_enabled,
            "centerline_protect_distance": self.centerline_protect_distance,
            "centerline_far_distance": self.centerline_far_distance,
            "centerline_prune_threshold": self.centerline_prune_threshold,
            "candidate_distance_map_available": self.candidate_distance_map_np is not None,
            "soft_empty_margin": self.soft_empty_margin,
            "soft_empty_tau": self.soft_empty_tau,
            "soft_empty_density_mode": self.soft_empty_density_mode,
            **self.mask_stats,
        }
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return str(out_path)

    def _component_stats(self, mask):
        try:
            from scipy import ndimage
        except ImportError:
            return {
                "topology_mask_components": None,
                "topology_small_component_count": None,
                "topology_largest_component_ratio": None,
            }
        labels, num_labels = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 2))
        if num_labels == 0:
            return {
                "topology_mask_components": 0,
                "topology_small_component_count": 0,
                "topology_largest_component_ratio": 0.0,
            }
        sizes = np.bincount(labels.reshape(-1), minlength=num_labels + 1)
        sizes[0] = 0
        foreground = int(mask.sum())
        return {
            "topology_mask_components": int(num_labels),
            "topology_small_component_count": int(np.sum((sizes[1:] > 0) & (sizes[1:] < self.component_small_voxels))),
            "topology_largest_component_ratio": float(sizes.max() / max(foreground, 1)),
        }
