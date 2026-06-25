import torch
from torch import nn
import torch.nn.functional as F


def _make_norm(num_channels, norm):
    if norm == "instance":
        return nn.InstanceNorm3d(num_channels, affine=True)
    if norm == "batch":
        return nn.BatchNorm3d(num_channels)
    if norm in ("none", None):
        return nn.Identity()
    raise ValueError(f"Unsupported norm type: {norm}")


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, norm="instance"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(out_channels, norm),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(out_channels, norm),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PriorNet3DUNet(nn.Module):
    """A compact 3D U-Net for V_fdk -> soft structure prior logits."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=16,
        depth=3,
        norm="instance",
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        channels = [int(base_channels) * (2 ** level) for level in range(int(depth) + 1)]
        self.encoders = nn.ModuleList()
        self.encoders.append(ConvBlock3D(in_channels, channels[0], norm=norm))
        for level in range(1, len(channels)):
            self.encoders.append(ConvBlock3D(channels[level - 1], channels[level], norm=norm))

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(len(channels) - 1, 0, -1):
            self.upconvs.append(nn.ConvTranspose3d(channels[level], channels[level - 1], kernel_size=2, stride=2))
            self.decoders.append(ConvBlock3D(channels[level], channels[level - 1], norm=norm))

        self.head = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    @staticmethod
    def _match_size(x, target):
        if x.shape[2:] == target.shape[2:]:
            return x
        return F.interpolate(x, size=target.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, x):
        skips = []
        out = x
        for encoder in self.encoders[:-1]:
            out = encoder(out)
            skips.append(out)
            out = self.pool(out)

        out = self.encoders[-1](out)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            out = upconv(out)
            out = self._match_size(out, skip)
            out = torch.cat([out, skip], dim=1)
            out = decoder(out)

        return self.head(out)
