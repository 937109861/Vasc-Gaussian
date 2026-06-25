import torch
import torch.nn.functional as F

from utils.gaussian_to_grid import (
    downsample_prior_mask,
    gaussian_centers_to_occupancy_grid,
    load_prior_mask,
    save_grid_debug,
)


class LowResPriorOccupancyLoss:
    def __init__(
        self,
        prior_path,
        volume_positions,
        grid_size=32,
        source="roi",
        loss_type="l1",
    ):
        self.prior_path = prior_path
        self.grid_size = int(grid_size)
        self.source = source
        self.loss_type = loss_type

        prior_mask_np = load_prior_mask(prior_path, source=source)
        positions = volume_positions
        if positions is None:
            raise ValueError("LowResPriorOccupancyLoss requires scene.volume_positions")
        if positions.shape[:3] != prior_mask_np.shape:
            raise ValueError(
                f"prior mask shape {prior_mask_np.shape} does not match volume grid {positions.shape[:3]}"
            )

        self.prior_mask_np = prior_mask_np
        self.grid_min = positions[0, 0, 0].astype("float32")
        self.grid_max = positions[-1, -1, -1].astype("float32")
        self._prior_grid_cache = {}

    def prior_grid(self, device):
        key = str(device)
        if key not in self._prior_grid_cache:
            self._prior_grid_cache[key] = downsample_prior_mask(self.prior_mask_np, self.grid_size, device)
        return self._prior_grid_cache[key]

    def __call__(self, gaussians):
        xyz = gaussians.get_xyz
        opacity = gaussians.get_opacity.reshape(-1)
        gaussian_grid, valid = gaussian_centers_to_occupancy_grid(
            xyz,
            opacity,
            self.grid_min,
            self.grid_max,
            self.grid_size,
        )
        prior_grid = self.prior_grid(xyz.device)

        if self.loss_type == "l1":
            loss = F.l1_loss(gaussian_grid, prior_grid)
        elif self.loss_type == "bce":
            loss = F.binary_cross_entropy(gaussian_grid.clamp(1e-6, 1.0 - 1e-6), prior_grid)
        elif self.loss_type == "dice":
            intersection = torch.sum(gaussian_grid * prior_grid)
            denom = torch.sum(gaussian_grid) + torch.sum(prior_grid)
            loss = 1.0 - (2.0 * intersection + 1e-6) / (denom + 1e-6)
        else:
            raise ValueError(f"Unsupported occ_loss_type: {self.loss_type}")

        stats = {
            "occ_grid_size": self.grid_size,
            "occ_source": self.source,
            "occ_loss_type": self.loss_type,
            "occ_valid_gaussian_ratio": float(valid.float().mean().detach().item()) if valid.numel() else 0.0,
            "gaussian_occ_mean": float(gaussian_grid.detach().mean().item()),
            "prior_occ_mean": float(prior_grid.detach().mean().item()),
            "gaussian_occ_nonzero_ratio": float((gaussian_grid.detach() > 1e-6).float().mean().item()),
            "prior_occ_nonzero_ratio": float((prior_grid.detach() > 0.5).float().mean().item()),
        }
        return loss, stats, gaussian_grid, prior_grid

    def save_debug(self, gaussian_grid, prior_grid, out_dir):
        save_grid_debug(gaussian_grid, prior_grid, out_dir)
