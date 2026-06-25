import torch
import torch.nn.functional as F


def dice_loss(prob, target, eps=1e-6):
    prob = prob.float()
    target = target.float()
    dims = tuple(range(1, prob.dim()))
    intersection = torch.sum(prob * target, dim=dims)
    denom = torch.sum(prob, dim=dims) + torch.sum(target, dim=dims)
    return torch.mean(1.0 - (2.0 * intersection + eps) / (denom + eps))


def tv3d_loss(volume):
    dx = torch.abs(volume[:, :, 1:, :, :] - volume[:, :, :-1, :, :]).mean()
    dy = torch.abs(volume[:, :, :, 1:, :] - volume[:, :, :, :-1, :]).mean()
    dz = torch.abs(volume[:, :, :, :, 1:] - volume[:, :, :, :, :-1]).mean()
    return (dx + dy + dz) / 3.0


def prior_structure_loss(
    logits,
    target,
    bce_weight=1.0,
    dice_weight=1.0,
    tv_weight=1e-5,
    pos_weight=None,
):
    target = target.float()
    if pos_weight is not None and not torch.is_tensor(pos_weight):
        pos_weight = torch.tensor(float(pos_weight), dtype=logits.dtype, device=logits.device)

    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    prob = torch.sigmoid(logits)
    dice = dice_loss(prob, target)
    tv = tv3d_loss(prob)
    loss = float(bce_weight) * bce + float(dice_weight) * dice + float(tv_weight) * tv
    stats = {
        "loss": float(loss.detach().item()),
        "bce": float(bce.detach().item()),
        "dice": float(dice.detach().item()),
        "tv3d": float(tv.detach().item()),
        "prob_mean": float(prob.detach().mean().item()),
        "target_mean": float(target.detach().mean().item()),
    }
    return loss, stats
