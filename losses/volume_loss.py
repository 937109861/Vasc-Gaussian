import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from pathlib import Path

from utils.differentiable_volume import (
    gaussian_centers_to_density_grid,
    gaussian_density_weights,
)
from utils.gaussian_to_grid import downsample_prior_mask, load_prior_mask


def normalize_tensor_volume(volume, eps=1e-6):
    volume = volume.float()
    vmin = torch.amin(volume)
    vmax = torch.amax(volume)
    return (volume - vmin) / torch.clamp(vmax - vmin, min=eps)


def total_variation_3d(volume):
    dx = torch.abs(volume[1:, :, :] - volume[:-1, :, :]).mean()
    dy = torch.abs(volume[:, 1:, :] - volume[:, :-1, :]).mean()
    dz = torch.abs(volume[:, :, 1:] - volume[:, :, :-1]).mean()
    return (dx + dy + dz) / 3.0


class LowResVolumeLoss:
    def __init__(
        self,
        target_volume,
        volume_positions,
        grid_size=32,
        loss_type="l1",
        density_mode="opacity",
        splat_mode="trilinear",
        splat_radius=2,
        min_sigma_voxels=0.75,
        max_sigma_voxels=3.0,
        dgr_sigma_scale=1.0,
        dgr_normalize_kernel=False,
        dgr_supersample=1,
        dgr_kernel_cutoff=0.0,
        dgr_max_splat_radius=0,
        prior_path="",
        use_prior_mask=False,
        mask_source="roi",
        roi_weight=1.0,
        background_weight=0.1,
        tissue_balance=False,
        sparse_foreground_balance=False,
        foreground_weight=1.0,
        foreground_epsilon=1e-6,
        soft_tissue_weight=2.0,
        hard_tissue_weight=1.0,
        soft_tissue_min_quantile=0.05,
        soft_tissue_max_quantile=0.75,
        hard_tissue_min_quantile=0.90,
        background_suppression_weight=0.0,
        background_suppression_threshold=0.0,
        volume_tv_weight=0.0,
    ):
        self.grid_size = int(grid_size)
        self.loss_type = loss_type
        self.density_mode = density_mode
        self.splat_mode = splat_mode
        self.splat_radius = int(splat_radius)
        self.min_sigma_voxels = float(min_sigma_voxels)
        self.max_sigma_voxels = float(max_sigma_voxels)
        self.dgr_sigma_scale = float(dgr_sigma_scale)
        self.dgr_normalize_kernel = bool(dgr_normalize_kernel)
        self.dgr_supersample = int(dgr_supersample)
        self.dgr_kernel_cutoff = float(dgr_kernel_cutoff)
        self.dgr_max_splat_radius = int(dgr_max_splat_radius)
        self.use_prior_mask = bool(use_prior_mask)
        self.mask_source = mask_source
        self.roi_weight = float(roi_weight)
        self.background_weight = float(background_weight)
        self.tissue_balance = bool(tissue_balance)
        self.sparse_foreground_balance = bool(sparse_foreground_balance)
        self.foreground_weight = float(foreground_weight)
        self.foreground_epsilon = float(foreground_epsilon)
        self.soft_tissue_weight = float(soft_tissue_weight)
        self.hard_tissue_weight = float(hard_tissue_weight)
        self.soft_tissue_min_quantile = float(soft_tissue_min_quantile)
        self.soft_tissue_max_quantile = float(soft_tissue_max_quantile)
        self.hard_tissue_min_quantile = float(hard_tissue_min_quantile)
        self.background_suppression_weight = float(background_suppression_weight)
        self.background_suppression_threshold = float(background_suppression_threshold)
        self.volume_tv_weight = float(volume_tv_weight)

        if target_volume is None:
            raise ValueError("LowResVolumeLoss requires scene.image_3d target_volume")
        if volume_positions is None:
            raise ValueError("LowResVolumeLoss requires scene.volume_positions")

        target_np = np.asarray(target_volume, dtype=np.float32)
        positions = np.asarray(volume_positions, dtype=np.float32)
        if positions.shape[:3] != target_np.shape:
            raise ValueError(
                f"target volume shape {target_np.shape} does not match volume positions {positions.shape[:3]}"
            )

        self.grid_min = positions[0, 0, 0].astype("float32")
        self.grid_max = positions[-1, -1, -1].astype("float32")
        self.target_np = target_np
        self.prior_mask_np = None
        if self.use_prior_mask:
            if not prior_path:
                raise ValueError("use_prior_mask=True requires prior_path")
            self.prior_mask_np = load_prior_mask(prior_path, source=mask_source)
            if self.prior_mask_np.shape != target_np.shape:
                raise ValueError(
                    f"prior mask shape {self.prior_mask_np.shape} does not match target volume {target_np.shape}"
                )

        self._target_grid_cache = {}
        self._weight_grid_cache = {}
        self._tissue_mask_cache = {}

    def target_grid(self, device):
        key = str(device)
        if key not in self._target_grid_cache:
            target = torch.from_numpy(self.target_np).to(device=device, dtype=torch.float32)
            target = normalize_tensor_volume(target)[None, None]
            target = F.interpolate(
                target,
                size=(self.grid_size, self.grid_size, self.grid_size),
                mode="trilinear",
                align_corners=False,
            )
            self._target_grid_cache[key] = target[0, 0].clamp(0.0, 1.0)
        return self._target_grid_cache[key]

    def weight_grid(self, device):
        key = str(device)
        if key not in self._weight_grid_cache:
            weights = torch.ones_like(self.target_grid(device))
            if self.tissue_balance:
                soft_mask, hard_mask = self.tissue_masks(device)
                if self.sparse_foreground_balance:
                    foreground_mask = self._tissue_mask_cache[key]["foreground_mask"]
                    weights = torch.where(
                        foreground_mask,
                        torch.full_like(weights, self.foreground_weight),
                        weights,
                    )
                weights = torch.where(soft_mask, torch.full_like(weights, self.soft_tissue_weight), weights)
                weights = torch.where(hard_mask, torch.full_like(weights, self.hard_tissue_weight), weights)
            if self.use_prior_mask:
                mask = downsample_prior_mask(self.prior_mask_np, self.grid_size, device)
                prior_weights = torch.full_like(mask, self.background_weight)
                prior_weights = torch.where(mask > 0.5, torch.full_like(prior_weights, self.roi_weight), prior_weights)
                weights = weights * prior_weights
            self._weight_grid_cache[key] = weights
        return self._weight_grid_cache[key]

    def tissue_masks(self, device):
        key = str(device)
        if key not in self._tissue_mask_cache:
            target = self.target_grid(device)
            foreground_mask = target > self.foreground_epsilon
            values = target[foreground_mask] if self.sparse_foreground_balance else target.reshape(-1)
            if values.numel() > 0:
                soft_min = torch.quantile(values, self.soft_tissue_min_quantile)
                soft_max = torch.quantile(values, self.soft_tissue_max_quantile)
                hard_min = torch.quantile(values, self.hard_tissue_min_quantile)
                if self.sparse_foreground_balance:
                    soft_mask = foreground_mask & (target >= soft_min) & (target <= soft_max)
                    hard_mask = foreground_mask & (target >= hard_min)
                else:
                    soft_mask = (target >= soft_min) & (target <= soft_max)
                    hard_mask = target >= hard_min
            else:
                zero = torch.zeros((), dtype=target.dtype, device=target.device)
                soft_min = zero
                soft_max = zero
                hard_min = zero
                soft_mask = torch.zeros_like(target, dtype=torch.bool)
                hard_mask = torch.zeros_like(target, dtype=torch.bool)
            self._tissue_mask_cache[key] = {
                "soft_mask": soft_mask,
                "hard_mask": hard_mask,
                "foreground_mask": foreground_mask,
                "soft_min": soft_min.detach(),
                "soft_max": soft_max.detach(),
                "hard_min": hard_min.detach(),
            }
        payload = self._tissue_mask_cache[key]
        return payload["soft_mask"], payload["hard_mask"]

    def __call__(self, gaussians):
        xyz = gaussians.get_xyz
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
            dgr_sigma_scale=self.dgr_sigma_scale,
            dgr_normalize_kernel=self.dgr_normalize_kernel,
            dgr_supersample=self.dgr_supersample,
            dgr_kernel_cutoff=self.dgr_kernel_cutoff,
            dgr_max_splat_radius=self.dgr_max_splat_radius,
            normalize=True,
        )
        target_grid = self.target_grid(xyz.device)
        weight_grid = self.weight_grid(xyz.device)

        diff = pred_grid - target_grid
        if self.loss_type == "l1":
            loss_grid = torch.abs(diff)
        elif self.loss_type == "mse":
            loss_grid = diff * diff
        else:
            raise ValueError(f"Unsupported volume_loss_type: {self.loss_type}")

        loss = torch.sum(loss_grid * weight_grid) / torch.clamp(torch.sum(weight_grid), min=1e-6)
        background_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)
        if self.background_suppression_weight > 0.0:
            background_mask = target_grid <= self.foreground_epsilon
            if torch.any(background_mask):
                background_values = F.relu(
                    pred_grid[background_mask] - self.background_suppression_threshold
                )
                background_loss = torch.mean(background_values)
                loss = loss + self.background_suppression_weight * background_loss
        tv_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)
        if self.volume_tv_weight > 0.0:
            tv_loss = total_variation_3d(pred_grid)
            loss = loss + self.volume_tv_weight * tv_loss

        stats = {
            "volume_grid_size": self.grid_size,
            "volume_loss_type": self.loss_type,
            "volume_density_mode": self.density_mode,
            "volume_splat_mode": self.splat_mode,
            "volume_splat_radius": self.splat_radius,
            "volume_min_sigma_voxels": self.min_sigma_voxels,
            "volume_max_sigma_voxels": self.max_sigma_voxels,
            "volume_dgr_sigma_scale": self.dgr_sigma_scale,
            "volume_dgr_normalize_kernel": self.dgr_normalize_kernel,
            "volume_dgr_supersample": self.dgr_supersample,
            "volume_dgr_kernel_cutoff": self.dgr_kernel_cutoff,
            "volume_dgr_max_splat_radius": self.dgr_max_splat_radius,
            "volume_use_prior_mask": self.use_prior_mask,
            "volume_mask_source": self.mask_source,
            "volume_tissue_balance": self.tissue_balance,
            "volume_sparse_foreground_balance": self.sparse_foreground_balance,
            "volume_foreground_weight": self.foreground_weight,
            "volume_foreground_epsilon": self.foreground_epsilon,
            "volume_soft_tissue_weight": self.soft_tissue_weight,
            "volume_hard_tissue_weight": self.hard_tissue_weight,
            "volume_background_suppression_weight": self.background_suppression_weight,
            "volume_background_suppression_threshold": self.background_suppression_threshold,
            "volume_background_suppression_loss": float(background_loss.detach().item()),
            "volume_tv_weight": self.volume_tv_weight,
            "volume_tv_loss": float(tv_loss.detach().item()),
            "volume_valid_gaussian_ratio": float(valid.float().mean().detach().item()) if valid.numel() else 0.0,
            "pred_volume_mean": float(pred_grid.detach().mean().item()),
            "target_volume_mean": float(target_grid.detach().mean().item()),
            "pred_volume_nonzero_ratio": float((pred_grid.detach() > 1e-6).float().mean().item()),
            "volume_weight_mean": float(weight_grid.detach().mean().item()),
        }
        if self.use_prior_mask:
            stats.update(
                {
                    "volume_roi_weight": self.roi_weight,
                    "volume_background_weight": self.background_weight,
                }
            )
        if self.tissue_balance:
            cache = self._tissue_mask_cache[str(xyz.device)]
            soft_mask = cache["soft_mask"]
            hard_mask = cache["hard_mask"]
            stats.update(
                {
                    "volume_soft_tissue_ratio": float(soft_mask.float().mean().detach().item()),
                    "volume_hard_tissue_ratio": float(hard_mask.float().mean().detach().item()),
                    "volume_foreground_ratio": float(
                        cache["foreground_mask"].float().mean().detach().item()
                    ),
                    "volume_soft_tissue_min": float(cache["soft_min"].item()),
                    "volume_soft_tissue_max": float(cache["soft_max"].item()),
                    "volume_hard_tissue_min": float(cache["hard_min"].item()),
                }
            )
        return loss, stats, pred_grid, target_grid

    def save_debug(self, pred_grid, target_grid, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_np = pred_grid.detach().cpu().numpy().astype(np.float32)
        target_np = target_grid.detach().cpu().numpy().astype(np.float32)
        np.save(out_dir / "pred_volume_grid.npy", pred_np)
        np.save(out_dir / "target_volume_grid.npy", target_np)

        mids = [size // 2 for size in pred_np.shape]
        slices = {
            "axial_z": (slice(None), slice(None), mids[2]),
            "coronal_y": (slice(None), mids[1], slice(None)),
            "sagittal_x": (mids[0], slice(None), slice(None)),
        }
        for name, index in slices.items():
            pred_slice = np.clip(pred_np[index], 0.0, 1.0)
            target_slice = np.clip(target_np[index], 0.0, 1.0)
            diff = np.abs(pred_slice - target_slice)
            imageio.imwrite(out_dir / f"{name}_pred.png", (pred_slice * 255).astype(np.uint8))
            imageio.imwrite(out_dir / f"{name}_target.png", (target_slice * 255).astype(np.uint8))
            imageio.imwrite(out_dir / f"{name}_absdiff.png", (np.clip(diff, 0, 1) * 255).astype(np.uint8))
