import argparse
import csv
import json
from pathlib import Path


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_json_or_empty(path):
    if not path.exists():
        return {}
    return read_json(path)


def read_last_jsonl(path):
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-1] if rows else {}


def collect_run(model_path):
    model_path = Path(model_path)
    init = read_json_or_empty(model_path / "init" / "init_stats.json")
    train = read_json_or_empty(model_path / "training_summary.json")

    nvs_candidates = sorted((model_path / "test").glob("ours_*/nvs_metrics.json"))
    nvs = read_json(nvs_candidates[-1]) if nvs_candidates else {}

    stats_candidates = sorted((model_path / "analysis").glob("iteration_*/gaussian_stats.json"))
    stats = read_json(stats_candidates[-1]) if stats_candidates else {}

    refinement_last = read_last_jsonl(model_path / "refinement" / "refinement_history.jsonl")
    allocation_last = read_last_jsonl(model_path / "analysis" / "gaussian_allocation_history.jsonl")
    allocation = stats.get("allocation", {}) or allocation_last
    return {
        "model_path": str(model_path),
        "init_mode": init.get("init_mode"),
        "num_acui_points": init.get("num_acui_points"),
        "num_prior_points": init.get("num_prior_points"),
        "psnr": nvs.get("psnr"),
        "ssim": nvs.get("ssim"),
        "eval_fps": nvs.get("image_eval_fps"),
        "train_time_sec": train.get("total_training_time_sec"),
        "num_gaussians": stats.get("num_gaussians"),
        "opacity_mean": stats.get("opacity", {}).get("summary", {}).get("mean"),
        "allocation_iteration": allocation.get("iteration"),
        "roi_gaussians": allocation.get("num_roi_gaussians"),
        "roi_gaussian_ratio": allocation.get("roi_gaussian_ratio"),
        "occupancy_gaussians": allocation.get("num_occupancy_gaussians"),
        "occupancy_gaussian_ratio": allocation.get("occupancy_gaussian_ratio"),
        "outside_roi_gaussians": allocation.get("num_outside_roi_gaussians"),
        "outside_roi_gaussian_ratio": allocation.get("outside_roi_gaussian_ratio"),
        "empty_occupancy_gaussians": allocation.get("num_empty_occupancy_gaussians"),
        "empty_occupancy_gaussian_ratio": allocation.get("empty_occupancy_gaussian_ratio"),
        "allocation_prune_candidates": allocation.get("num_extra_prune_candidates"),
        "empty_prune_mask_source": allocation.get("empty_prune_mask_source"),
        "refinement_iteration": refinement_last.get("iteration"),
        "refinement_roi_gaussians": refinement_last.get("num_roi_gaussians"),
        "refinement_extra_prune_candidates": refinement_last.get("num_extra_prune_candidates"),
        "roi_densify_enabled": refinement_last.get("roi_densify_enabled"),
        "empty_prune_enabled": refinement_last.get("empty_prune_enabled"),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize X-Gaussian stage experiment outputs.")
    parser.add_argument("--model_paths", nargs="+", required=True)
    parser.add_argument("--out_csv", default="stage_summary.csv")
    parser.add_argument("--out_json", default="stage_summary.json")
    args = parser.parse_args()

    rows = [collect_run(path) for path in args.model_paths]

    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['model_path']}: PSNR={row['psnr']}, SSIM={row['ssim']}, "
            f"Gaussians={row['num_gaussians']}, ROI={row['roi_gaussians']}"
        )
    print(f"Saved: {args.out_csv}")
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
