import json
import os
import time

import numpy as np
import torch
import torchvision


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def _safe_float(value):
    return float(value) if value is not None else None


def _summary_stats(values):
    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "median": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
        }

    return {
        "count": int(values.size),
        "min": _safe_float(values.min()),
        "max": _safe_float(values.max()),
        "mean": _safe_float(values.mean()),
        "std": _safe_float(values.std()),
        "median": _safe_float(np.median(values)),
        "p05": _safe_float(np.quantile(values, 0.05)),
        "p25": _safe_float(np.quantile(values, 0.25)),
        "p75": _safe_float(np.quantile(values, 0.75)),
        "p95": _safe_float(np.quantile(values, 0.95)),
    }


def _histogram(values, bins=20):
    if values.size == 0:
        return {"counts": [], "bin_edges": []}
    counts, bin_edges = np.histogram(values, bins=bins)
    return {
        "counts": counts.astype(int).tolist(),
        "bin_edges": bin_edges.tolist(),
    }


def save_json(path, data):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def append_jsonl(path, data):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(data) + "\n")


def save_dataset_split(model_path, scene):
    split_info = {
        "train_count": len(scene.getTrainCameras()),
        "test_count": len(scene.getTestCameras()),
        "additional_count": len(scene.getAddCameras()),
        "train_views": [cam.image_name for cam in scene.getTrainCameras()],
        "test_views": [cam.image_name for cam in scene.getTestCameras()],
        "additional_views": [cam.image_name for cam in scene.getAddCameras()],
    }
    save_json(os.path.join(model_path, "dataset_split.json"), split_info)


def save_init_statistics(model_path, init_info, gaussians=None):
    payload = dict(init_info or {})
    if gaussians is not None:
        xyz = _to_numpy(gaussians.get_xyz)
        payload["initial_num_gaussians"] = int(xyz.shape[0])
        payload["initial_bbox"] = {
            "min": xyz.min(axis=0).tolist() if xyz.size > 0 else [],
            "max": xyz.max(axis=0).tolist() if xyz.size > 0 else [],
            "mean": xyz.mean(axis=0).tolist() if xyz.size > 0 else [],
            "std": xyz.std(axis=0).tolist() if xyz.size > 0 else [],
        }
    save_json(os.path.join(model_path, "init", "init_stats.json"), payload)


def export_gaussian_statistics(model_path, iteration, gaussians, bins=20, allocation_stats=None):
    xyz = _to_numpy(gaussians.get_xyz)
    opacity = _to_numpy(gaussians.get_opacity).reshape(-1)
    radiodensity = _to_numpy(gaussians.get_radiodensity).reshape(-1) if hasattr(gaussians, "get_radiodensity") else opacity
    scaling = _to_numpy(gaussians.get_scaling)

    norms = np.linalg.norm(xyz, axis=1) if xyz.size > 0 else np.asarray([])
    if scaling.size > 0:
        scale_min = np.maximum(np.min(scaling, axis=1), 1e-9)
        scale_ratio = np.max(scaling, axis=1) / scale_min
    else:
        scale_ratio = np.asarray([])

    stats = {
        "iteration": int(iteration),
        "num_gaussians": int(xyz.shape[0]),
        "opacity": {
            "summary": _summary_stats(opacity),
            "histogram": _histogram(opacity, bins=bins),
        },
        "radiodensity": {
            "summary": _summary_stats(radiodensity),
            "histogram": _histogram(radiodensity, bins=bins),
        },
        "xyz": {
            "summary": {
                "x": _summary_stats(xyz[:, 0]) if xyz.size > 0 else _summary_stats(np.asarray([])),
                "y": _summary_stats(xyz[:, 1]) if xyz.size > 0 else _summary_stats(np.asarray([])),
                "z": _summary_stats(xyz[:, 2]) if xyz.size > 0 else _summary_stats(np.asarray([])),
                "radius": _summary_stats(norms),
            },
            "bbox": {
                "min": xyz.min(axis=0).tolist() if xyz.size > 0 else [],
                "max": xyz.max(axis=0).tolist() if xyz.size > 0 else [],
                "mean": xyz.mean(axis=0).tolist() if xyz.size > 0 else [],
                "std": xyz.std(axis=0).tolist() if xyz.size > 0 else [],
            },
            "histogram": {
                "x": _histogram(xyz[:, 0], bins=bins) if xyz.size > 0 else _histogram(np.asarray([]), bins=bins),
                "y": _histogram(xyz[:, 1], bins=bins) if xyz.size > 0 else _histogram(np.asarray([]), bins=bins),
                "z": _histogram(xyz[:, 2], bins=bins) if xyz.size > 0 else _histogram(np.asarray([]), bins=bins),
                "radius": _histogram(norms, bins=bins),
            },
        },
        "scale": {
            "x": _summary_stats(scaling[:, 0]) if scaling.size > 0 else _summary_stats(np.asarray([])),
            "y": _summary_stats(scaling[:, 1]) if scaling.size > 0 else _summary_stats(np.asarray([])),
            "z": _summary_stats(scaling[:, 2]) if scaling.size > 0 else _summary_stats(np.asarray([])),
            "anisotropy_ratio": {
                "summary": _summary_stats(scale_ratio),
                "histogram": _histogram(scale_ratio, bins=bins),
            },
        },
    }
    if allocation_stats is not None:
        stats["allocation"] = allocation_stats

    stats_dir = os.path.join(model_path, "analysis", f"iteration_{iteration}")
    save_json(os.path.join(stats_dir, "gaussian_stats.json"), stats)
    history_row = {
        "iteration": int(iteration),
        "num_gaussians": stats["num_gaussians"],
        "opacity_mean": stats["opacity"]["summary"]["mean"],
        "opacity_std": stats["opacity"]["summary"]["std"],
        "radiodensity_mean": stats["radiodensity"]["summary"]["mean"],
        "radiodensity_std": stats["radiodensity"]["summary"]["std"],
        "scale_ratio_median": stats["scale"]["anisotropy_ratio"]["summary"]["median"],
        "scale_ratio_p95": stats["scale"]["anisotropy_ratio"]["summary"]["p95"],
        "scale_ratio_max": stats["scale"]["anisotropy_ratio"]["summary"]["max"],
    }
    if allocation_stats is not None:
        history_row.update({
            "num_roi_gaussians": allocation_stats.get("num_roi_gaussians"),
            "roi_gaussian_ratio": allocation_stats.get("roi_gaussian_ratio"),
            "num_occupancy_gaussians": allocation_stats.get("num_occupancy_gaussians"),
            "occupancy_gaussian_ratio": allocation_stats.get("occupancy_gaussian_ratio"),
            "num_outside_roi_gaussians": allocation_stats.get("num_outside_roi_gaussians"),
            "outside_roi_gaussian_ratio": allocation_stats.get("outside_roi_gaussian_ratio"),
            "num_empty_occupancy_gaussians": allocation_stats.get("num_empty_occupancy_gaussians"),
            "empty_occupancy_gaussian_ratio": allocation_stats.get("empty_occupancy_gaussian_ratio"),
            "num_extra_prune_candidates": allocation_stats.get("num_extra_prune_candidates"),
            "empty_prune_mask_source": allocation_stats.get("empty_prune_mask_source"),
        })
    append_jsonl(
        os.path.join(model_path, "analysis", "gaussian_stats_history.jsonl"),
        history_row,
    )
    if allocation_stats is not None:
        append_jsonl(
            os.path.join(model_path, "analysis", "gaussian_allocation_history.jsonl"),
            allocation_stats,
        )


def save_render_snapshot(model_path, iteration, split_name, views, gaussians, render_func, render_args, max_views=-1):
    snapshot_root = os.path.join(model_path, "progress_renders", split_name, f"iteration_{iteration}")
    render_dir = os.path.join(snapshot_root, "renders")
    gt_dir = os.path.join(snapshot_root, "gt")
    ensure_dir(render_dir)
    ensure_dir(gt_dir)

    selected_views = views if max_views is None or max_views < 0 else views[:max_views]
    start = time.time()

    for idx, view in enumerate(selected_views):
        rendering = torch.clamp(render_func(view, gaussians, *render_args)["render"], 0.0, 1.0)
        gt = torch.clamp(view.normalized_image[0:3, :, :], 0.0, 1.0)
        torchvision.utils.save_image(rendering, os.path.join(render_dir, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt, os.path.join(gt_dir, f"{idx:05d}.png"))

    elapsed = time.time() - start
    save_json(
        os.path.join(snapshot_root, "render_summary.json"),
        {
            "iteration": int(iteration),
            "split": split_name,
            "num_views": int(len(selected_views)),
            "render_time_sec": elapsed,
            "fps": (len(selected_views) / elapsed) if elapsed > 0 and len(selected_views) > 0 else 0.0,
        },
    )
