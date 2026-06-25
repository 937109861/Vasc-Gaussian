import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from losses import prior_structure_loss
from models import PriorNet3DUNet


def parse_size(values, name):
    if values is None:
        return None
    if len(values) == 1:
        return (int(values[0]), int(values[0]), int(values[0]))
    if len(values) == 3:
        return tuple(int(v) for v in values)
    raise ValueError(f"{name} expects one value or three values")


def read_volume(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".pickle", ".pkl"}:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if "image" not in payload:
            raise KeyError(f"Pickle volume must contain key 'image': {path}")
        return np.asarray(payload["image"], dtype=np.float32)
    if suffix == ".npy":
        return np.load(path).astype(np.float32)
    if suffix == ".npz":
        payload = np.load(path)
        for key in ("volume", "recon_volume", "fdk", "image", "prior_volume", "soft_prior"):
            if key in payload:
                return payload[key].astype(np.float32)
        return payload[payload.files[0]].astype(np.float32)
    raise ValueError(f"Unsupported volume format: {path}")


def normalize_volume(volume, eps=1e-6, pmin=0.0, pmax=100.0):
    volume = np.asarray(volume, dtype=np.float32)
    lo = float(np.percentile(volume, pmin))
    hi = float(np.percentile(volume, pmax))
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - lo) / max(hi - lo, eps), 0.0, 1.0).astype(np.float32)


def resize_volume(volume, target_size):
    if target_size is None or tuple(volume.shape) == tuple(target_size):
        return volume
    tensor = torch.from_numpy(volume)[None, None].float()
    tensor = F.interpolate(tensor, size=target_size, mode="trilinear", align_corners=False)
    return tensor[0, 0].numpy().astype(np.float32)


def load_pairs(args):
    pairs = []
    if args.pairs_json:
        with open(args.pairs_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for item in payload:
            pairs.append((item["fdk"], item["gt"]))
    if args.fdk_paths or args.gt_paths:
        if len(args.fdk_paths) != len(args.gt_paths):
            raise ValueError("--fdk_paths and --gt_paths must have the same length")
        pairs.extend(zip(args.fdk_paths, args.gt_paths))
    if not pairs:
        raise ValueError("Provide --pairs_json or both --fdk_paths and --gt_paths")
    return pairs


class PriorPatchDataset(Dataset):
    def __init__(
        self,
        pairs,
        target_size=None,
        patch_size=None,
        steps_per_epoch=200,
        threshold_mode="quantile",
        threshold_value=0.98,
        norm_percentiles=(0.0, 100.0),
    ):
        self.samples = []
        self.patch_size = patch_size
        self.steps_per_epoch = int(steps_per_epoch)
        pmin, pmax = norm_percentiles

        for fdk_path, gt_path in pairs:
            fdk = normalize_volume(read_volume(fdk_path), pmin=pmin, pmax=pmax)
            gt = normalize_volume(read_volume(gt_path), pmin=pmin, pmax=pmax)
            fdk = resize_volume(fdk, target_size)
            gt = resize_volume(gt, target_size)
            if fdk.shape != gt.shape:
                raise ValueError(f"Shape mismatch: {fdk_path} {fdk.shape} vs {gt_path} {gt.shape}")
            if threshold_mode == "quantile":
                threshold = float(np.quantile(gt, float(threshold_value)))
            elif threshold_mode == "absolute":
                threshold = float(threshold_value)
            else:
                raise ValueError(f"Unsupported threshold mode: {threshold_mode}")
            target = (gt >= threshold).astype(np.float32)
            self.samples.append(
                {
                    "fdk": torch.from_numpy(fdk)[None],
                    "target": torch.from_numpy(target)[None],
                    "fdk_path": str(fdk_path),
                    "gt_path": str(gt_path),
                    "threshold": threshold,
                }
            )

    def __len__(self):
        return self.steps_per_epoch

    def foreground_ratio(self):
        fg = sum(float(sample["target"].sum().item()) for sample in self.samples)
        total = sum(float(sample["target"].numel()) for sample in self.samples)
        return fg / max(total, 1.0)

    def _random_crop(self, volume, target):
        if self.patch_size is None:
            return volume, target
        _, d, h, w = volume.shape
        pd, ph, pw = self.patch_size
        if pd > d or ph > h or pw > w:
            raise ValueError(f"Patch size {self.patch_size} exceeds volume shape {(d, h, w)}")
        z = np.random.randint(0, d - pd + 1)
        y = np.random.randint(0, h - ph + 1)
        x = np.random.randint(0, w - pw + 1)
        return volume[:, z:z + pd, y:y + ph, x:x + pw], target[:, z:z + pd, y:y + ph, x:x + pw]

    def __getitem__(self, _):
        sample = self.samples[np.random.randint(0, len(self.samples))]
        return self._random_crop(sample["fdk"], sample["target"])


def save_checkpoint(path, model, optimizer, args, epoch, stats):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "epoch": int(epoch),
            "stats": stats,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train PriorNet: V_fdk -> soft structure prior.")
    parser.add_argument("--pairs_json", default=None, help="JSON list with {'fdk': path, 'gt': path} entries.")
    parser.add_argument("--fdk_paths", nargs="*", default=[])
    parser.add_argument("--gt_paths", nargs="*", default=[])
    parser.add_argument("--out_dir", default="output/prior_net")
    parser.add_argument("--target_size", nargs="+", type=int, default=None)
    parser.add_argument("--patch_size", nargs="+", type=int, default=[96])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--norm", default="instance", choices=["instance", "batch", "none"])
    parser.add_argument("--threshold_mode", default="quantile", choices=["quantile", "absolute"])
    parser.add_argument("--threshold_value", type=float, default=0.98)
    parser.add_argument("--norm_percentiles", nargs=2, type=float, default=[0.0, 100.0])
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--tv_weight", type=float, default=1e-5)
    parser.add_argument("--auto_pos_weight", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    target_size = parse_size(args.target_size, "--target_size")
    patch_size = parse_size(args.patch_size, "--patch_size")
    pairs = load_pairs(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PriorPatchDataset(
        pairs,
        target_size=target_size,
        patch_size=patch_size,
        steps_per_epoch=args.steps_per_epoch,
        threshold_mode=args.threshold_mode,
        threshold_value=args.threshold_value,
        norm_percentiles=tuple(args.norm_percentiles),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = PriorNet3DUNet(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        depth=args.depth,
        norm=args.norm,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    fg_ratio = dataset.foreground_ratio()
    pos_weight = None
    if args.auto_pos_weight:
        pos_weight = min((1.0 - fg_ratio) / max(fg_ratio, 1e-6), 1000.0)

    summary = {
        "pairs": [{"fdk": fdk, "gt": gt} for fdk, gt in pairs],
        "target_size": target_size,
        "patch_size": patch_size,
        "foreground_ratio": fg_ratio,
        "pos_weight": pos_weight,
        "args": vars(args),
        "history": [],
    }
    with open(out_dir / "training_config.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    best_loss = float("inf")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {}
        progress = tqdm(loader, desc=f"PriorNet epoch {epoch}/{args.epochs}")
        for fdk, target in progress:
            fdk = fdk.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                logits = model(fdk)
                loss, stats = prior_structure_loss(
                    logits,
                    target,
                    bce_weight=args.bce_weight,
                    dice_weight=args.dice_weight,
                    tv_weight=args.tv_weight,
                    pos_weight=pos_weight,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            for key, value in stats.items():
                running[key] = running.get(key, 0.0) + float(value)
            progress.set_postfix(loss=f"{stats['loss']:.5f}", dice=f"{stats['dice']:.5f}")

        epoch_stats = {key: value / max(len(loader), 1) for key, value in running.items()}
        epoch_stats["epoch"] = epoch
        epoch_stats["elapsed_sec"] = time.time() - start_time
        summary["history"].append(epoch_stats)
        with open(out_dir / "training_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        save_checkpoint(out_dir / "last.pth", model, optimizer, args, epoch, epoch_stats)
        if epoch_stats["loss"] < best_loss:
            best_loss = epoch_stats["loss"]
            save_checkpoint(out_dir / "best.pth", model, optimizer, args, epoch, epoch_stats)

    print(f"Saved PriorNet checkpoints to {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
