import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.image_utils import psnr
from utils.loss_utils import ssim


def load_image(path):
    return tf.to_tensor(Image.open(path)).unsqueeze(0)[:, :3, :, :].cuda()


def resolve_method_dir(model_path, iteration, split):
    split_dir = Path(model_path) / split
    if iteration == -1:
        candidates = sorted(split_dir.glob("ours_*"), key=lambda p: int(p.name.split("_")[-1]))
        if not candidates:
            raise FileNotFoundError(f"No render directories found under {split_dir}")
        return candidates[-1]
    return split_dir / f"ours_{iteration}"


def evaluate_render_dir(method_dir):
    renders_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    if not renders_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f"Expected renders and gt directories under {method_dir}")

    image_names = sorted(name for name in os.listdir(renders_dir) if (gt_dir / name).exists())
    if not image_names:
        raise RuntimeError(f"No matched render/gt images found under {method_dir}")

    rows = []
    psnrs = []
    ssims = []
    start = time.time()

    for name in tqdm(image_names, desc=f"Evaluating {method_dir}"):
        render = load_image(renders_dir / name)
        gt = load_image(gt_dir / name)
        view_psnr = psnr(render, gt).mean().item()
        view_ssim = ssim(render, gt).mean().item()
        rows.append({"image": name, "psnr": view_psnr, "ssim": view_ssim})
        psnrs.append(view_psnr)
        ssims.append(view_ssim)

    elapsed = time.time() - start
    summary = {
        "method_dir": str(method_dir),
        "num_views": len(image_names),
        "psnr": float(torch.tensor(psnrs).mean().item()),
        "ssim": float(torch.tensor(ssims).mean().item()),
        "eval_time_sec": elapsed,
        "image_eval_fps": (len(image_names) / elapsed) if elapsed > 0 else 0.0,
    }
    return summary, rows


def save_outputs(method_dir, summary, rows):
    summary_path = method_dir / "nvs_metrics.json"
    per_view_path = method_dir / "nvs_per_view.csv"

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with open(per_view_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)

    return summary_path, per_view_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate X-Gaussian NVS renders with PSNR and SSIM.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--split", default="test", choices=["test", "train"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("eval_nvs.py currently requires CUDA because utils.loss_utils.ssim uses torch tensors on GPU.")

    method_dir = resolve_method_dir(args.model_path, args.iteration, args.split)
    summary, rows = evaluate_render_dir(method_dir)
    summary_path, per_view_path = save_outputs(method_dir, summary, rows)

    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {summary_path}")
    print(f"Saved per-view metrics: {per_view_path}")


if __name__ == "__main__":
    main()
