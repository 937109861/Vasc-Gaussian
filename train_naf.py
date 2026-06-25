import argparse
import json
import os
import pickle
import time

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


def normalize_np(x):
    x = x.astype(np.float32, copy=False)
    xmin = float(x.min())
    xmax = float(x.max())
    if xmax <= xmin:
        return np.zeros_like(x, dtype=np.float32), xmin, xmax
    return (x - xmin) / (xmax - xmin), xmin, xmax


class FourierFeatures(nn.Module):
    def __init__(self, num_frequencies):
        super().__init__()
        self.num_frequencies = num_frequencies
        freq = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("freq", freq)

    @property
    def out_dim(self):
        return 3 + 3 * 2 * self.num_frequencies

    def forward(self, x):
        angles = x[..., None, :] * self.freq[:, None] * np.pi
        encoded = [x, torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)]
        return torch.cat(encoded, dim=-1)


class NAFField(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=6, num_frequencies=8):
        super().__init__()
        self.encoder = FourierFeatures(num_frequencies)
        self.skip_layer = max(1, num_layers // 2)
        self.layers = nn.ModuleList()
        in_dim = self.encoder.out_dim
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_in_dim = in_dim
            elif layer_idx == self.skip_layer:
                layer_in_dim = hidden_dim + in_dim
            else:
                layer_in_dim = hidden_dim
            self.layers.append(nn.Linear(layer_in_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, xyz_norm):
        feat = self.encoder(xyz_norm)
        x = feat
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == self.skip_layer:
                x = torch.cat([x, feat], dim=-1)
            x = F.relu(layer(x), inplace=True)
        return F.softplus(self.out(x) - 1.0)


def build_rays(angles, pixel_y, pixel_x, geometry, device):
    dsd = float(geometry["DSD"])
    dso = float(geometry["DSO"])
    d_detector = np.asarray(geometry["dDetector"], dtype=np.float32)
    off_detector = np.asarray(geometry["offDetector"], dtype=np.float32)
    h, w = int(geometry["detector_shape"][0]), int(geometry["detector_shape"][1])

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    radial = torch.stack([cos_a, sin_a, torch.zeros_like(cos_a)], dim=-1)
    tangent = torch.stack([-sin_a, cos_a, torch.zeros_like(cos_a)], dim=-1)
    vertical = torch.zeros_like(radial)
    vertical[:, 2] = 1.0

    source = dso * radial
    detector_center = source - dsd * radial

    u = (pixel_x.float() - (w - 1) * 0.5) * float(d_detector[0]) + float(off_detector[0])
    v = (pixel_y.float() - (h - 1) * 0.5) * float(d_detector[1]) + float(off_detector[1])
    detector = detector_center + u[:, None] * tangent + v[:, None] * vertical
    direction = F.normalize(detector - source, dim=-1)
    return source.to(device), direction.to(device)


def intersect_box(origin, direction, half_size):
    safe_direction = torch.where(
        torch.abs(direction) < 1e-8,
        torch.full_like(direction, 1e-8),
        direction,
    )
    inv_d = 1.0 / safe_direction
    t0 = (-half_size - origin) * inv_d
    t1 = (half_size - origin) * inv_d
    tmin = torch.minimum(t0, t1).amax(dim=-1)
    tmax = torch.maximum(t0, t1).amin(dim=-1)
    valid = tmax > torch.clamp(tmin, min=0.0)
    return torch.clamp(tmin, min=0.0), tmax, valid


def render_rays(field, origin, direction, half_size, samples_per_ray, projection_scale, projection_bias):
    near, far, valid = intersect_box(origin, direction, half_size)
    t = torch.linspace(0.0, 1.0, samples_per_ray, device=origin.device)
    depth = near[:, None] + (far - near)[:, None] * t[None, :]
    points = origin[:, None, :] + depth[:, :, None] * direction[:, None, :]
    points_norm = points / half_size[None, None, :]
    density = field(points_norm.reshape(-1, 3)).reshape(origin.shape[0], samples_per_ray)
    pred = density.mean(dim=-1) * F.softplus(projection_scale) + projection_bias
    return pred * valid.float(), valid


def save_preview_slices(volume, out_dir):
    preview_dir = os.path.join(out_dir, "preview_slices")
    os.makedirs(preview_dir, exist_ok=True)
    mids = [s // 2 for s in volume.shape]
    slices = {
        "axial_z.png": volume[mids[0], :, :],
        "coronal_y.png": volume[:, mids[1], :],
        "sagittal_x.png": volume[:, :, mids[2]],
    }
    for name, image in slices.items():
        imageio.imwrite(os.path.join(preview_dir, name), (np.clip(image, 0, 1) * 255).astype(np.uint8))


def save_mhd(volume, d_voxel, out_dir, basename):
    raw_name = basename + ".raw"
    mhd_name = basename + ".mhd"
    raw_path = os.path.join(out_dir, raw_name)
    mhd_path = os.path.join(out_dir, mhd_name)
    volume.astype(np.float32).tofile(raw_path)
    with open(mhd_path, "w", encoding="utf-8") as handle:
        handle.write("ObjectType = Image\n")
        handle.write("NDims = 3\n")
        handle.write(f"DimSize = {volume.shape[2]} {volume.shape[1]} {volume.shape[0]}\n")
        handle.write("ElementType = MET_FLOAT\n")
        handle.write(f"ElementSpacing = {float(d_voxel[2])} {float(d_voxel[1])} {float(d_voxel[0])}\n")
        handle.write("ElementByteOrderMSB = False\n")
        handle.write(f"ElementDataFile = {raw_name}\n")
    return mhd_path, raw_path


def save_tiff_stack(volume, out_dir, basename):
    tif_path = os.path.join(out_dir, basename + ".tiff")
    volume_norm, _, _ = normalize_np(volume)
    volume_u16 = (np.clip(volume_norm, 0, 1) * 65535).astype(np.uint16)
    imageio.mimwrite(tif_path, list(volume_u16), format="TIFF")
    return tif_path


@torch.no_grad()
def export_volume(field, half_size, resolution, chunk, device):
    coords_1d = torch.linspace(-1.0, 1.0, resolution, device=device)
    z, y, x = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    coords = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    values = []
    for start in tqdm(range(0, coords.shape[0], chunk), desc="Exporting volume"):
        values.append(field(coords[start:start + chunk]).detach().cpu())
    volume = torch.cat(values, dim=0).reshape(resolution, resolution, resolution).numpy().astype(np.float32)
    volume, _, _ = normalize_np(volume)
    return volume


def main():
    parser = argparse.ArgumentParser(description="NAF-style sparse-view CT reconstruction.")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-o", "--out_dir", default="output/naf")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--batch_rays", type=int, default=2048)
    parser.add_argument("--samples_per_ray", type=int, default=96)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_frequencies", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--l1_weight", type=float, default=0.1)
    parser.add_argument("--density_weight", type=float, default=0.00001)
    parser.add_argument("--export_resolution", type=int, default=256)
    parser.add_argument("--export_chunk", type=int, default=262144)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--no_tiff", action="store_true", help="Do not export 3D TIFF stack.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    with open(args.source_path, "rb") as handle:
        data = pickle.load(handle)

    projections, proj_min, proj_max = normalize_np(np.asarray(data["train"]["projections"], dtype=np.float32))
    angles = np.asarray(data["train"]["angles"], dtype=np.float32)
    h, w = projections.shape[-2:]
    geometry = {
        "DSD": data["DSD"],
        "DSO": data["DSO"],
        "dDetector": data["dDetector"],
        "offDetector": data["offDetector"],
        "detector_shape": [h, w],
    }
    n_voxel = np.asarray(data["nVoxel"], dtype=np.float32)
    d_voxel = np.asarray(data["dVoxel"], dtype=np.float32)
    half_size_np = n_voxel * d_voxel * 0.5
    half_size = torch.from_numpy(half_size_np.astype(np.float32)).to(device)

    projections_t = torch.from_numpy(projections).to(device)
    angles_t = torch.from_numpy(angles).to(device)

    field = NAFField(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_frequencies=args.num_frequencies,
    ).to(device)
    projection_scale = nn.Parameter(torch.tensor(1.0, device=device))
    projection_bias = nn.Parameter(torch.tensor(0.0, device=device))
    optimizer = torch.optim.Adam(list(field.parameters()) + [projection_scale, projection_bias], lr=args.lr)

    history = []
    start_time = time.time()
    progress = tqdm(range(1, args.iterations + 1), desc="Training NAF")
    for iteration in progress:
        view_idx = torch.randint(0, projections_t.shape[0], (args.batch_rays,), device=device)
        pixel_y = torch.randint(0, h, (args.batch_rays,), device=device)
        pixel_x = torch.randint(0, w, (args.batch_rays,), device=device)
        target = projections_t[view_idx, pixel_y, pixel_x]

        origin, direction = build_rays(angles_t[view_idx], pixel_y, pixel_x, geometry, device)
        pred, valid = render_rays(
            field,
            origin,
            direction,
            half_size,
            args.samples_per_ray,
            projection_scale,
            projection_bias,
        )
        mse = F.mse_loss(pred[valid], target[valid])
        l1 = F.l1_loss(pred[valid], target[valid])
        density_reg = torch.mean(field(torch.rand(args.batch_rays, 3, device=device) * 2.0 - 1.0))
        loss = mse + args.l1_weight * l1 + args.density_weight * density_reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iteration % 10 == 0:
            progress.set_postfix(loss=f"{loss.item():.5f}", mse=f"{mse.item():.5f}", scale=f"{F.softplus(projection_scale).item():.3f}")

        if iteration % args.save_every == 0 or iteration == args.iterations:
            iter_dir = os.path.join(args.out_dir, f"iteration_{iteration}")
            os.makedirs(iter_dir, exist_ok=True)
            volume = export_volume(field, half_size, args.export_resolution, args.export_chunk, device)
            np.save(os.path.join(iter_dir, "naf_volume.npy"), volume)
            save_mhd(volume, d_voxel, iter_dir, "naf_volume")
            tif_path = None if args.no_tiff else save_tiff_stack(volume, iter_dir, "naf_volume")
            save_preview_slices(volume, iter_dir)
            torch.save(
                {
                    "field": field.state_dict(),
                    "projection_scale": projection_scale.detach().cpu(),
                    "projection_bias": projection_bias.detach().cpu(),
                    "args": vars(args),
                },
                os.path.join(iter_dir, "naf_checkpoint.pth"),
            )
            history.append(
                {
                    "iteration": int(iteration),
                    "loss": float(loss.item()),
                    "mse": float(mse.item()),
                    "l1": float(l1.item()),
                    "tiff": tif_path,
                }
            )

    summary = {
        "source_path": args.source_path,
        "projection_normalization": {"min": proj_min, "max": proj_max},
        "num_train_views": int(projections.shape[0]),
        "projection_shape": [int(h), int(w)],
        "nVoxel": data["nVoxel"],
        "dVoxel": data["dVoxel"],
        "export_resolution": int(args.export_resolution),
        "tiff_export": not args.no_tiff,
        "iterations": int(args.iterations),
        "elapsed_sec": time.time() - start_time,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved NAF result to {args.out_dir}")


if __name__ == "__main__":
    main()
