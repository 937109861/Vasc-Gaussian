import torch
import torch.nn.functional as F


def _sobel_kernels(device, dtype, channels):
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    return kernel_x.repeat(channels, 1, 1, 1), kernel_y.repeat(channels, 1, 1, 1)


def sobel_edges(image):
    if image.dim() == 3:
        image = image.unsqueeze(0)
    channels = image.shape[1]
    kernel_x, kernel_y = _sobel_kernels(image.device, image.dtype, channels)
    grad_x = F.conv2d(image, kernel_x, padding=1, groups=channels)
    grad_y = F.conv2d(image, kernel_y, padding=1, groups=channels)
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-8)


def sobel_edge_loss(prediction, target):
    """L_edge = mean(|Sobel(prediction) - Sobel(target)|)."""
    return torch.abs(sobel_edges(prediction) - sobel_edges(target)).mean()


def _dilate_mask(mask, radius):
    radius = int(radius)
    if radius <= 0:
        return mask
    kernel_size = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=radius)


def target_edge_mask(target, quantile=0.80, dilation=1):
    with torch.no_grad():
        target_edges = sobel_edges(target)
        flat = target_edges.reshape(target_edges.shape[0], -1)
        thresholds = torch.quantile(flat, float(quantile), dim=1).view(-1, 1, 1, 1)
        mask = (target_edges >= thresholds).to(dtype=target.dtype)
        mask = _dilate_mask(mask, dilation)
    return mask


def masked_sobel_edge_loss(prediction, target, quantile=0.80, dilation=1):
    """Sobel edge loss restricted to strong target-edge regions."""
    pred_edges = sobel_edges(prediction)
    target_edges = sobel_edges(target)
    mask = target_edge_mask(target, quantile=quantile, dilation=dilation)
    denom = torch.clamp(mask.sum(), min=1.0)
    return (torch.abs(pred_edges - target_edges) * mask).sum() / denom


def projection_support_losses(
    prediction,
    target,
    foreground_threshold=0.02,
    mask_dilation=2,
    background_margin=0.01,
):
    """Balance sparse projection foreground and suppress unsupported bright pixels."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"projection support loss shape mismatch: prediction={prediction.shape}, target={target.shape}"
        )
    if foreground_threshold < 0.0:
        raise ValueError("foreground_threshold must be non-negative")
    if background_margin < 0.0:
        raise ValueError("background_margin must be non-negative")

    with torch.no_grad():
        foreground_mask = (target >= float(foreground_threshold)).to(dtype=target.dtype)
        protected_mask = _dilate_mask(foreground_mask, mask_dilation)
        background_mask = (protected_mask < 0.5).to(dtype=target.dtype)

    foreground_denom = torch.clamp(foreground_mask.sum(), min=1.0)
    foreground_loss = (
        torch.abs(prediction - target) * foreground_mask
    ).sum() / foreground_denom

    false_positive = F.relu(prediction - target - float(background_margin))
    background_denom = torch.clamp(background_mask.sum(), min=1.0)
    background_loss = (false_positive * background_mask).sum() / background_denom

    stats = {
        "projection_foreground_ratio": foreground_mask.mean().detach(),
        "projection_background_ratio": background_mask.mean().detach(),
    }
    return foreground_loss, background_loss, stats
