from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from utils.gaussian_to_grid import gaussian_centers_to_occupancy_grid


@dataclass
class PaperVesselManagementConfig:
    lambda_q: float = 1.0
    lambda_def: float = 1.0
    delta: float = 0.25
    lambda_m: float = 1.0
    lambda_eta: float = 1.0
    lambda_k: float = 1.0
    tau_keep: float = 0.05
    tau_den: float = 0.00004
    tau_s: float = 0.008
    eps: float = 1e-8
    prior_align_corners: bool = True


class FixedPriorVolume:
    """Fixed observation-derived P_fused sampler for Gaussian management."""

    def __init__(
        self,
        prior_path,
        volume_positions,
        source="soft",
        align_corners=True,
    ):
        if volume_positions is None:
            raise ValueError("FixedPriorVolume requires scene.volume_positions")
        payload = np.load(prior_path)
        prior_np, prior_key = self._select_prior(payload, source)
        positions = np.asarray(volume_positions, dtype=np.float32)
        if positions.shape[:3] != prior_np.shape:
            raise ValueError(
                f"P_fused shape {prior_np.shape} does not match volume grid {positions.shape[:3]}"
            )

        self.prior_path = str(prior_path)
        self.prior_key = prior_key
        self.source = str(source)
        self.align_corners = bool(align_corners)
        self.grid_shape = tuple(int(v) for v in prior_np.shape)
        self.grid_min_np = positions[0, 0, 0].astype(np.float32)
        self.grid_max_np = positions[-1, -1, -1].astype(np.float32)
        self._prior_np = np.clip(prior_np.astype(np.float32), 0.0, 1.0)
        self._cache = {}

    @staticmethod
    def _select_prior(payload, source):
        source = str(source)
        if source in ("soft", "soft_prior", "p_fused", "P_fused"):
            keys = ("soft_prior", "prior_volume", "occupancy_mask")
        elif source == "prior_volume":
            keys = ("prior_volume", "soft_prior", "occupancy_mask")
        elif source == "roi":
            keys = ("roi_mask",)
        elif source in ("occupancy", "occupancy_mask"):
            keys = ("occupancy_mask",)
        else:
            keys = (source,)
        for key in keys:
            if key in payload:
                return np.asarray(payload[key]), key
        raise KeyError(f"Could not find fixed P_fused source {source}; tried {keys}")

    def _volume_tensor(self, device, dtype):
        key = (str(device), str(dtype))
        if key not in self._cache:
            volume_xyz = torch.from_numpy(self._prior_np).to(device=device, dtype=dtype)
            # PyTorch 5D grid_sample expects [N, C, D, H, W] with grid order x,y,z.
            # The project stores volumes as [X, Y, Z], so convert to [Z, Y, X].
            self._cache[key] = volume_xyz.permute(2, 1, 0).contiguous()[None, None]
        return self._cache[key]

    def sample(self, xyz):
        xyz = xyz.detach()
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape [N, 3], got {tuple(xyz.shape)}")
        count = int(xyz.shape[0])
        if count == 0:
            empty = torch.empty((0,), dtype=xyz.dtype, device=xyz.device)
            return empty, torch.empty((0,), dtype=torch.bool, device=xyz.device)

        grid_min = torch.as_tensor(self.grid_min_np, dtype=xyz.dtype, device=xyz.device)
        grid_max = torch.as_tensor(self.grid_max_np, dtype=xyz.dtype, device=xyz.device)
        denom = torch.clamp(grid_max - grid_min, min=1e-6)
        normalized = (xyz - grid_min[None, :]) / denom[None, :]
        valid = torch.all((normalized >= 0.0) & (normalized <= 1.0), dim=1)

        if self.align_corners:
            grid_coords = normalized * 2.0 - 1.0
        else:
            shape = torch.as_tensor(self.grid_shape, dtype=xyz.dtype, device=xyz.device)
            index_coords = normalized * (shape[None, :] - 1.0)
            grid_coords = ((index_coords + 0.5) / shape[None, :]) * 2.0 - 1.0

        sample_grid = grid_coords.reshape(1, count, 1, 1, 3)
        values = F.grid_sample(
            self._volume_tensor(xyz.device, xyz.dtype),
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners,
        ).reshape(-1)
        values = torch.where(valid, values, torch.zeros_like(values))
        return values.clamp(0.0, 1.0), valid

    def description(self):
        return {
            "prior_path": self.prior_path,
            "prior_key": self.prior_key,
            "source": self.source,
            "grid_shape_xyz": list(self.grid_shape),
            "grid_min": self.grid_min_np.tolist(),
            "grid_max": self.grid_max_np.tolist(),
            "volume_order": "prior [X,Y,Z] -> grid_sample [Z,Y,X]",
            "align_corners": bool(self.align_corners),
            "fixed_prior": True,
        }


def _vector(name, tensor, count):
    tensor = tensor.reshape(-1)
    if int(tensor.shape[0]) != count:
        raise ValueError(f"{name} must have shape [N], got {tuple(tensor.shape)} for N={count}")
    return tensor


def _sanitize(tensor):
    bad = int((~torch.isfinite(tensor)).sum().item()) if tensor.numel() else 0
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0), bad


def _summary(prefix, tensor):
    tensor = tensor.detach().reshape(-1)
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return {
            f"{prefix}_min": None,
            f"{prefix}_mean": None,
            f"{prefix}_p50": None,
            f"{prefix}_p90": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_min": float(finite.min().item()),
        f"{prefix}_mean": float(finite.mean().item()),
        f"{prefix}_p50": float(torch.quantile(finite, 0.50).item()),
        f"{prefix}_p90": float(torch.quantile(finite, 0.90).item()),
        f"{prefix}_max": float(finite.max().item()),
    }


def _occupancy_deficit(gaussians, occupancy_loss):
    xyz = gaussians.get_xyz.detach()
    opacity = gaussians.get_opacity.detach().reshape(-1)
    gaussian_grid, valid = gaussian_centers_to_occupancy_grid(
        xyz,
        opacity,
        occupancy_loss.grid_min,
        occupancy_loss.grid_max,
        occupancy_loss.grid_size,
    )
    prior_grid = occupancy_loss.prior_grid(xyz.device).to(dtype=xyz.dtype)

    grid_min = torch.as_tensor(occupancy_loss.grid_min, dtype=xyz.dtype, device=xyz.device)
    grid_max = torch.as_tensor(occupancy_loss.grid_max, dtype=xyz.dtype, device=xyz.device)
    denom = torch.clamp(grid_max - grid_min, min=1e-6)
    normalized = (xyz - grid_min[None, :]) / denom[None, :]
    indices = torch.round(normalized * float(int(occupancy_loss.grid_size) - 1)).long()
    indices = indices.clamp(0, int(occupancy_loss.grid_size) - 1)

    prior_values = prior_grid[indices[:, 0], indices[:, 1], indices[:, 2]]
    pred_values = gaussian_grid[indices[:, 0], indices[:, 1], indices[:, 2]]
    deficit = torch.relu(prior_values - pred_values)
    return torch.where(valid, deficit, torch.zeros_like(deficit)), valid, gaussian_grid, prior_grid


def compute_paper_vessel_management(
    gaussians,
    prior_sampler,
    occupancy_loss,
    projection_support_controller,
    config,
):
    if occupancy_loss is None:
        raise ValueError("paper vessel management requires LowResPriorOccupancyLoss")
    if projection_support_controller is None:
        raise ValueError("paper vessel management requires ProjectionSupportController")

    with torch.no_grad():
        xyz = gaussians.get_xyz.detach()
        count = int(xyz.shape[0])
        dtype = xyz.dtype
        device = xyz.device

        g = gaussians.xyz_gradient_accum.detach() / gaussians.denom.detach()
        g = _vector("g", g, count)
        g, nan_g = _sanitize(g)

        q, prior_valid = prior_sampler.sample(xyz)
        q = _vector("q", q, count)
        q, nan_q = _sanitize(q)

        d, occ_valid, _, _ = _occupancy_deficit(gaussians, occupancy_loss)
        d = _vector("d", d, count)
        d, nan_d = _sanitize(d)

        support = projection_support_controller.support_ratio(gaussians)
        m = _vector("m", support["support_ratio"], count)
        observations = _vector("observations", support["observations"], count)
        m = torch.where(observations > 0.0, m, torch.zeros_like(m))
        m = m.clamp(0.0, 1.0)
        m, nan_m = _sanitize(m)

        r = _vector("r", gaussians.get_scaling.detach().max(dim=1).values, count)
        r, nan_r = _sanitize(r)

        alpha = gaussians.get_opacity.detach().reshape(-1)
        rho = gaussians.get_radiodensity.detach().reshape(-1)
        eta = _vector("eta", alpha * rho, count)
        eta, nan_eta = _sanitize(eta)
        eta_max = eta.max() if eta.numel() else torch.zeros((), dtype=dtype, device=device)
        eta_tilde = eta / (eta_max + float(config.eps))
        eta_tilde = _vector("eta_tilde", eta_tilde, count)
        eta_tilde, nan_eta_tilde = _sanitize(eta_tilde)

        S_den = (
            g
            * (1.0 + float(config.lambda_q) * q + float(config.lambda_def) * d)
            * (float(config.delta) + (1.0 - float(config.delta)) * m)
        )
        S_den = _vector("S_den", S_den, count)
        S_den, nan_s_den = _sanitize(S_den)

        S_keep = (
            float(config.lambda_m) * m
            + eta_tilde * (float(config.lambda_eta) + float(config.lambda_k) * q)
        )
        S_keep = _vector("S_keep", S_keep, count)
        S_keep, nan_s_keep = _sanitize(S_keep)

        prune_mask = S_keep < float(config.tau_keep)
        densify_mask = torch.logical_and(~prune_mask, S_den >= float(config.tau_den))
        clone_mask = torch.logical_and(densify_mask, r <= float(config.tau_s))
        split_mask = torch.logical_and(densify_mask, r > float(config.tau_s))
        retain_mask = ~(prune_mask | clone_mask | split_mask)

        decision_sum = (
            prune_mask.to(torch.int32)
            + clone_mask.to(torch.int32)
            + split_mask.to(torch.int32)
            + retain_mask.to(torch.int32)
        )
        if not torch.all(decision_sum == 1):
            raise RuntimeError("paper vessel management decisions are not mutually exclusive")

        num_nan = (
            nan_g
            + nan_q
            + nan_d
            + nan_m
            + nan_r
            + nan_eta
            + nan_eta_tilde
            + nan_s_den
            + nan_s_keep
        )

        stats = {
            "num_nan": int(num_nan),
            "num_zero_observation": int((observations <= 0.0).sum().item()),
            "num_prior_valid": int(prior_valid.sum().item()),
            "num_occ_valid": int(occ_valid.sum().item()),
            "num_support_hits": float(support["hits"].sum().item()),
            "num_support_observations": float(observations.sum().item()),
            "lambda_q": float(config.lambda_q),
            "lambda_def": float(config.lambda_def),
            "delta": float(config.delta),
            "lambda_m": float(config.lambda_m),
            "lambda_eta": float(config.lambda_eta),
            "lambda_k": float(config.lambda_k),
            "tau_keep": float(config.tau_keep),
            "tau_den": float(config.tau_den),
            "tau_s": float(config.tau_s),
            "prior_align_corners": bool(config.prior_align_corners),
        }
        for prefix, tensor in (
            ("q", q),
            ("g", g),
            ("m", m),
            ("d", d),
            ("eta", eta_tilde),
            ("S_den", S_den),
            ("S_keep", S_keep),
        ):
            stats.update(_summary(prefix, tensor))

    return {
        "q": q,
        "g": g,
        "m": m,
        "d": d,
        "r": r,
        "eta_tilde": eta_tilde,
        "S_den": S_den,
        "S_keep": S_keep,
        "observations": observations,
        "prune_mask": prune_mask,
        "clone_mask": clone_mask,
        "split_mask": split_mask,
        "retain_mask": retain_mask,
        "stats": stats,
    }
