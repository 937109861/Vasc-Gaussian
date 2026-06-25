import json
from pathlib import Path

import numpy as np
import torch


class PriorRefinementController:
    def __init__(
        self,
        prior_path,
        volume_positions,
        roi_densify_weight=2.0,
        empty_prune_opacity=0.01,
        empty_prune_mask_source="roi",
    ):
        self.prior_path = prior_path
        self.roi_densify_weight = float(roi_densify_weight)
        self.empty_prune_opacity = float(empty_prune_opacity)
        self.empty_prune_mask_source = str(empty_prune_mask_source)

        payload = np.load(prior_path)
        if "roi_mask" not in payload:
            raise KeyError(f"prior file must contain roi_mask: {prior_path}")
        self.roi_mask_np = payload["roi_mask"].astype(bool)

        if "occupancy_mask" in payload:
            self.occupancy_mask_np = payload["occupancy_mask"].astype(bool)
        else:
            self.occupancy_mask_np = self.roi_mask_np

        positions = np.asarray(volume_positions, dtype=np.float32)
        if positions.shape[:3] != self.roi_mask_np.shape:
            raise ValueError(
                f"prior roi shape {self.roi_mask_np.shape} does not match volume grid {positions.shape[:3]}"
            )

        self.grid_shape = np.asarray(self.roi_mask_np.shape, dtype=np.int64)
        self.grid_min = positions[0, 0, 0].astype(np.float32)
        self.grid_max = positions[-1, -1, -1].astype(np.float32)
        self.grid_spacing = (self.grid_max - self.grid_min) / np.maximum(self.grid_shape.astype(np.float32) - 1.0, 1.0)
        self.grid_spacing = np.where(np.abs(self.grid_spacing) < 1e-8, 1.0, self.grid_spacing).astype(np.float32)

    @classmethod
    def from_args(cls, args, scene):
        prior_path = getattr(args, "refinement_prior_path", "") or getattr(args, "prior_path", "")
        if not prior_path:
            raise ValueError("use_prior_refinement requires refinement_prior_path or prior_path")
        return cls(
            prior_path=prior_path,
            volume_positions=scene.volume_positions,
            roi_densify_weight=getattr(args, "roi_densify_weight", 2.0),
            empty_prune_opacity=getattr(args, "empty_prune_opacity", 0.01),
        )

    def _lookup_mask(self, xyz, mask_np):
        device = xyz.device
        grid_min = torch.tensor(self.grid_min, dtype=xyz.dtype, device=device)
        grid_spacing = torch.tensor(self.grid_spacing, dtype=xyz.dtype, device=device)
        grid_shape = torch.tensor(self.grid_shape, dtype=torch.long, device=device)

        indices = torch.round((xyz - grid_min[None, :]) / grid_spacing[None, :]).long()
        valid = torch.all((indices >= 0) & (indices < grid_shape[None, :]), dim=1)
        clamped = torch.minimum(torch.maximum(indices, torch.zeros_like(indices)), grid_shape[None, :] - 1)

        mask_tensor = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool)
        values = mask_tensor[clamped[:, 0], clamped[:, 1], clamped[:, 2]]
        return torch.logical_and(values, valid), valid

    def region_masks(self, gaussians):
        xyz = gaussians.get_xyz.detach()
        roi_mask, valid_mask = self._lookup_mask(xyz, self.roi_mask_np)
        occupancy_mask, _ = self._lookup_mask(xyz, self.occupancy_mask_np)
        outside_roi_mask = torch.logical_and(valid_mask, torch.logical_not(roi_mask))
        empty_occupancy_mask = torch.logical_and(valid_mask, torch.logical_not(occupancy_mask))
        outside_grid_mask = torch.logical_not(valid_mask)
        return {
            "roi": roi_mask,
            "occupancy": occupancy_mask,
            "outside_roi": outside_roi_mask,
            "empty_occupancy": empty_occupancy_mask,
            "outside_grid": outside_grid_mask,
            "valid": valid_mask,
        }

    @staticmethod
    def _ratio(count, total):
        return float(count / total) if total > 0 else 0.0

    def allocation_statistics(self, gaussians, iteration=None):
        masks = self.region_masks(gaussians)
        xyz = gaussians.get_xyz.detach()
        opacity = gaussians.get_opacity.detach().reshape(-1)
        total = int(xyz.shape[0])

        roi_count = int(masks["roi"].sum().item())
        occupancy_count = int(masks["occupancy"].sum().item())
        outside_roi_count = int(masks["outside_roi"].sum().item())
        empty_occupancy_count = int(masks["empty_occupancy"].sum().item())
        outside_grid_count = int(masks["outside_grid"].sum().item())
        low_opacity_count = int((opacity < self.empty_prune_opacity).sum().item())
        prune_region = masks["empty_occupancy"] if self.empty_prune_mask_source == "occupancy" else torch.logical_not(masks["roi"])
        prune_candidate_mask = torch.logical_and(prune_region, opacity < self.empty_prune_opacity)

        stats = {
            "num_gaussians": total,
            "num_inside_grid": int(masks["valid"].sum().item()),
            "num_outside_grid_gaussians": outside_grid_count,
            "num_roi_gaussians": roi_count,
            "roi_gaussian_ratio": self._ratio(roi_count, total),
            "num_occupancy_gaussians": occupancy_count,
            "occupancy_gaussian_ratio": self._ratio(occupancy_count, total),
            "num_outside_roi_gaussians": outside_roi_count,
            "outside_roi_gaussian_ratio": self._ratio(outside_roi_count, total),
            "num_empty_occupancy_gaussians": empty_occupancy_count,
            "empty_occupancy_gaussian_ratio": self._ratio(empty_occupancy_count, total),
            "num_low_opacity_gaussians": low_opacity_count,
            "num_extra_prune_candidates": int(prune_candidate_mask.sum().item()),
            "empty_prune_mask_source": self.empty_prune_mask_source,
            "roi_densify_weight": self.roi_densify_weight,
            "empty_prune_opacity": self.empty_prune_opacity,
        }
        if iteration is not None:
            stats["iteration"] = int(iteration)
        return stats

    def build_iteration_bias(self, gaussians):
        xyz = gaussians.get_xyz.detach()
        masks = self.region_masks(gaussians)
        roi_mask = masks["roi"]
        valid_mask = masks["valid"]

        grad_multipliers = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
        grad_multipliers[roi_mask] = self.roi_densify_weight

        opacity = gaussians.get_opacity.detach().reshape(-1)
        if self.empty_prune_mask_source == "occupancy":
            empty_mask = masks["empty_occupancy"]
        else:
            empty_mask = torch.logical_not(roi_mask)
        low_opacity = opacity < self.empty_prune_opacity
        extra_prune_mask = torch.logical_and(empty_mask, low_opacity)

        stats = self.allocation_statistics(gaussians)
        stats["num_empty_gaussians"] = int(empty_mask.sum().item())
        return grad_multipliers, extra_prune_mask, stats

    def save_description(self, model_path):
        out_path = Path(model_path) / "refinement" / "prior_refinement.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "prior_path": self.prior_path,
            "roi_shape": list(self.roi_mask_np.shape),
            "roi_voxels": int(self.roi_mask_np.sum()),
            "occupancy_voxels": int(self.occupancy_mask_np.sum()),
            "roi_densify_weight": self.roi_densify_weight,
            "empty_prune_opacity": self.empty_prune_opacity,
            "empty_prune_mask_source": self.empty_prune_mask_source,
            "grid_min": self.grid_min.tolist(),
            "grid_max": self.grid_max.tolist(),
            "grid_spacing": self.grid_spacing.tolist(),
        }
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return str(out_path)
