import math

import torch
import torch.nn.functional as F


def gaussian_scale_regularization(
    gaussians,
    scale_floor=0.75,
    scale_max_axis=0.0,
    max_anisotropy=5.0,
    anisotropy_loss_mode="linear",
):
    scales = gaussians.get_scaling
    min_scale = torch.amin(scales, dim=1)
    max_scale = torch.amax(scales, dim=1)

    floor_loss = F.relu(float(scale_floor) - min_scale).mean()
    scale_max_axis = float(scale_max_axis)
    if scale_max_axis > 0.0:
        max_axis_loss = torch.square(F.relu(max_scale - scale_max_axis)).mean()
    else:
        max_axis_loss = torch.zeros((), dtype=scales.dtype, device=scales.device)

    ratio = max_scale / torch.clamp(min_scale, min=1e-6)
    max_anisotropy = max(float(max_anisotropy), 1.0)
    if anisotropy_loss_mode == "linear":
        aniso_loss = F.relu(ratio - max_anisotropy).mean()
    elif anisotropy_loss_mode == "log":
        # Log-ratio excess prevents a few collapsed axes from dominating the
        # sparse-vessel experiment while preserving the configured threshold.
        aniso_loss = F.relu(
            torch.log(torch.clamp(ratio, min=1.0)) - math.log(max_anisotropy)
        ).mean()
    else:
        raise ValueError(f"Unsupported anisotropy_loss_mode: {anisotropy_loss_mode}")

    stats = {
        "scale_floor": float(scale_floor),
        "scale_max_axis": float(scale_max_axis),
        "scale_aniso_max_ratio": max_anisotropy,
        "scale_aniso_loss_mode": anisotropy_loss_mode,
        "scale_floor_loss": float(floor_loss.detach().item()),
        "scale_max_axis_loss": float(max_axis_loss.detach().item()),
        "scale_aniso_loss": float(aniso_loss.detach().item()),
        "scale_min_mean": float(min_scale.detach().mean().item()),
        "scale_max_mean": float(max_scale.detach().mean().item()),
        "scale_max_median": float(torch.quantile(max_scale.detach(), 0.50).item()) if max_scale.numel() else 0.0,
        "scale_max_p95": float(torch.quantile(max_scale.detach(), 0.95).item()) if max_scale.numel() else 0.0,
        "scale_max_p99": float(torch.quantile(max_scale.detach(), 0.99).item()) if max_scale.numel() else 0.0,
        "scale_max_value": float(torch.amax(max_scale.detach()).item()) if max_scale.numel() else 0.0,
        "scale_ratio_mean": float(ratio.detach().mean().item()),
        "scale_ratio_median": float(torch.quantile(ratio.detach(), 0.50).item()) if ratio.numel() else 0.0,
        "scale_ratio_p95": float(torch.quantile(ratio.detach(), 0.95).item()) if ratio.numel() else 0.0,
        "scale_ratio_max": float(torch.amax(ratio.detach()).item()) if ratio.numel() else 0.0,
    }
    return floor_loss, max_axis_loss, aniso_loss, stats


def radiodensity_entropy_regularization(gaussians):
    if not hasattr(gaussians, "get_radiodensity"):
        density = gaussians.get_opacity.reshape(-1)
    else:
        density = gaussians.get_radiodensity.reshape(-1)

    density = torch.clamp(density, 1e-6, 1.0 - 1e-6)
    entropy = -(density * torch.log(density) + (1.0 - density) * torch.log(1.0 - density))
    normalized_entropy = entropy / math.log(2.0)
    anti_binary_loss = 1.0 - normalized_entropy.mean()

    stats = {
        "density_entropy_loss": float(anti_binary_loss.detach().item()),
        "radiodensity_mean": float(density.detach().mean().item()),
        "radiodensity_binary_ratio": float(
            ((density.detach() < 0.05) | (density.detach() > 0.95)).float().mean().item()
        ),
    }
    return anti_binary_loss, stats


def gaussian_density_sparsity_regularization(gaussians, mode="radiodensity"):
    opacity = gaussians.get_opacity.reshape(-1)
    if mode == "opacity":
        density = opacity
    elif mode in ("radiodensity", "attenuation"):
        density = (
            gaussians.get_radiodensity.reshape(-1)
            if hasattr(gaussians, "get_radiodensity")
            else opacity
        )
    elif mode in ("opacity_radiodensity", "effective_density"):
        radiodensity = (
            gaussians.get_radiodensity.reshape(-1)
            if hasattr(gaussians, "get_radiodensity")
            else torch.ones_like(opacity)
        )
        density = opacity * radiodensity
    else:
        raise ValueError(f"Unsupported density_sparsity_mode: {mode}")

    loss = density.mean()
    stats = {
        "density_sparsity_mode": mode,
        "density_sparsity_loss": float(loss.detach().item()),
        "effective_density_mean": float(density.detach().mean().item()),
        "effective_density_p95": float(torch.quantile(density.detach(), 0.95).item())
        if density.numel()
        else 0.0,
    }
    return loss, stats
