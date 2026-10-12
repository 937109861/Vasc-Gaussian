import torch
import torch.nn.functional as F

from .structure_losses import sobel_edges


def vessel_weighted_l1_loss(
    prediction,
    target,
    vessel_weight_map=None,
    vessel_projection_weight=0.0,
    eps=1e-6,
):
    if vessel_weight_map is None or float(vessel_projection_weight) == 0.0:
        return torch.abs(prediction - target).mean(), {
            "vessel_projection_active": False,
        }
    if prediction.shape != target.shape:
        raise ValueError(
            f"projection loss shape mismatch: prediction={prediction.shape}, target={target.shape}"
        )
    if vessel_weight_map.shape != prediction.shape:
        raise ValueError(
            f"vessel projection weight shape {vessel_weight_map.shape} does not match projection {prediction.shape}"
        )
    if not torch.isfinite(vessel_weight_map).all():
        raise ValueError("vessel projection weight map contains NaN or Inf")

    vessel_weight = torch.clamp(vessel_weight_map.to(dtype=prediction.dtype), 0.0, 1.0)
    residual_weight = 1.0 + float(vessel_projection_weight) * vessel_weight
    residual = torch.abs(prediction - target)
    loss = torch.sum(residual_weight * residual) / torch.clamp(
        torch.sum(residual_weight),
        min=float(eps),
    )
    stats = {
        "vessel_projection_active": True,
        "vessel_projection_map_min": vessel_weight.detach().min(),
        "vessel_projection_map_max": vessel_weight.detach().max(),
        "vessel_projection_map_mean": vessel_weight.detach().mean(),
        "vessel_projection_residual_weight_min": residual_weight.detach().min(),
        "vessel_projection_residual_weight_max": residual_weight.detach().max(),
        "vessel_projection_residual_weight_mean": residual_weight.detach().mean(),
    }
    return loss, stats


def _dilate_mask(mask, radius):
    radius = int(radius)
    if radius <= 0:
        return mask
    kernel_size = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=radius)


def vessel_weighted_sobel_loss(
    prediction,
    target,
    vessel_weight_map=None,
    vessel_edge_weight=0.0,
    vessel_edge_threshold=0.15,
    vessel_edge_dilation=1,
    eps=1e-6,
):
    if vessel_weight_map is None or float(vessel_edge_weight) == 0.0:
        return prediction.new_zeros(()), {
            "vessel_projection_edge_active": False,
        }
    if prediction.shape != target.shape:
        raise ValueError(
            f"projection edge loss shape mismatch: prediction={prediction.shape}, target={target.shape}"
        )
    if vessel_weight_map.shape != prediction.shape:
        raise ValueError(
            f"vessel projection weight shape {vessel_weight_map.shape} does not match projection {prediction.shape}"
        )
    if not torch.isfinite(vessel_weight_map).all():
        raise ValueError("vessel projection weight map contains NaN or Inf")

    vessel_weight = torch.clamp(vessel_weight_map.to(dtype=prediction.dtype), 0.0, 1.0)
    with torch.no_grad():
        vessel_mask = (vessel_weight >= float(vessel_edge_threshold)).to(dtype=prediction.dtype)
        vessel_mask = _dilate_mask(vessel_mask, vessel_edge_dilation)

    edge_residual = torch.abs(sobel_edges(prediction) - sobel_edges(target))
    residual_weight = 1.0 + float(vessel_edge_weight) * vessel_weight
    weighted_mask = vessel_mask * residual_weight
    loss = torch.sum(edge_residual * weighted_mask) / torch.clamp(
        torch.sum(weighted_mask),
        min=float(eps),
    )
    stats = {
        "vessel_projection_edge_active": True,
        "vessel_projection_edge_mask_ratio": vessel_mask.detach().mean(),
        "vessel_projection_edge_weight_min": residual_weight.detach().min(),
        "vessel_projection_edge_weight_max": residual_weight.detach().max(),
        "vessel_projection_edge_weight_mean": residual_weight.detach().mean(),
    }
    return loss, stats
