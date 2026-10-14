import json
import pickle
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.build_vesselness_prior import (
    dilate_mask,
    frangi_vesselness_3d,
    fuse_prior,
    normalize_percentile,
    sample_points,
    save_preview_slices,
    voxel_positions_for_volume,
    write_ascii_ply,
)
from utils.export_volume import save_histogram, save_mhd_uint16, summarize_values


@dataclass
class PriorGeometry:
    shape: tuple
    spacing: tuple
    origin: tuple
    direction: tuple
    axis_order: str
    grid_min: tuple
    grid_max: tuple

    def as_dict(self):
        return {
            "shape": list(self.shape),
            "spacing": list(self.spacing),
            "origin": list(self.origin),
            "direction": list(self.direction),
            "axis_order": self.axis_order,
            "grid_min": list(self.grid_min),
            "grid_max": list(self.grid_max),
        }


def _as_path(path):
    return Path(str(path)).expanduser()


def _jsonable_path(path):
    return str(Path(path).as_posix())


def _stats(array):
    return summarize_values(np.asarray(array, dtype=np.float32).reshape(-1))


def _geometry_from_positions(volume_positions):
    positions = np.asarray(volume_positions, dtype=np.float32)
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError(f"volume_positions must have shape [X,Y,Z,3], got {positions.shape}")
    shape = tuple(int(v) for v in positions.shape[:3])
    origin = positions[0, 0, 0].astype(np.float32)
    grid_max = positions[-1, -1, -1].astype(np.float32)
    denom = np.maximum(np.asarray(shape, dtype=np.float32) - 1.0, 1.0)
    spacing = (grid_max - origin) / denom
    spacing = np.where(np.abs(spacing) < 1e-8, 1.0, spacing).astype(np.float32)
    return PriorGeometry(
        shape=shape,
        spacing=tuple(float(v) for v in spacing),
        origin=tuple(float(v) for v in origin),
        direction=(
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        axis_order="XYZ",
        grid_min=tuple(float(v) for v in origin),
        grid_max=tuple(float(v) for v in grid_max),
    )


def validate_prior_update_config(enable_dynamic_prior, update_iterations, total_iterations, beta):
    updates = [int(v) for v in (update_iterations or [])]
    if not bool(enable_dynamic_prior):
        return []
    if len(updates) != len(set(updates)):
        raise ValueError(f"prior_update_iterations contains duplicates: {updates}")
    if updates != sorted(updates):
        raise ValueError(f"prior_update_iterations must be strictly increasing: {updates}")
    if any(v <= 0 for v in updates):
        raise ValueError(f"prior_update_iterations must be positive: {updates}")
    if updates and updates[-1] >= int(total_iterations):
        raise ValueError(
            "last prior_update_iteration must be smaller than total training iterations: "
            f"last={updates[-1]}, iterations={int(total_iterations)}"
        )
    if not (0.0 <= float(beta) <= 1.0):
        raise ValueError(f"prior_fusion_beta must be in [0,1], got {beta}")
    return updates


def ema_fuse_prior(current, candidate, beta):
    beta = float(beta)
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"prior_fusion_beta must be in [0,1], got {beta}")
    current = np.asarray(current, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if current.shape != candidate.shape:
        raise ValueError(f"EMA prior shapes differ: current={current.shape}, candidate={candidate.shape}")
    if not np.all(np.isfinite(current)):
        raise ValueError("current prior contains NaN or Inf")
    if not np.all(np.isfinite(candidate)):
        raise ValueError("candidate prior contains NaN or Inf")
    fused = (1.0 - beta) * np.clip(current, 0.0, 1.0) + beta * np.clip(candidate, 0.0, 1.0)
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def should_trigger_prior_update(iteration, update_iterations, completed_iterations, enabled=True):
    if not bool(enabled):
        return False
    return int(iteration) in set(int(v) for v in update_iterations) and int(iteration) not in set(
        int(v) for v in completed_iterations
    )


def _select_prior(payload, source):
    if source in ("soft", "soft_prior", "p_fused", "P_fused"):
        keys = ("soft_prior", "prior_volume", "occupancy_mask")
    elif source == "prior_volume":
        keys = ("prior_volume", "soft_prior", "occupancy_mask")
    elif source in ("roi", "roi_mask"):
        keys = ("roi_mask",)
    elif source in ("occupancy", "occupancy_mask"):
        keys = ("occupancy_mask",)
    else:
        keys = (str(source),)
    for key in keys:
        if key in payload:
            return np.asarray(payload[key], dtype=np.float32), key
    raise KeyError(f"Could not find prior source {source}; tried {keys}")


def load_prior_array(prior_path, source="soft"):
    payload = np.load(prior_path)
    prior, key = _select_prior(payload, source)
    return np.clip(prior.astype(np.float32), 0.0, 1.0), key, payload


def validate_prior_geometry(prior, reference_geometry, name):
    if tuple(prior.shape) != tuple(reference_geometry.shape):
        raise ValueError(
            f"{name} shape {prior.shape} does not match reference volume grid {reference_geometry.shape}. "
            "Resampling is intentionally not guessed because the prior file does not carry a full physical affine."
        )
    if not np.all(np.isfinite(prior)):
        raise ValueError(f"{name} contains NaN or Inf")
    return {
        "name": name,
        "shape": list(prior.shape),
        "spacing": list(reference_geometry.spacing),
        "origin": list(reference_geometry.origin),
        "direction": list(reference_geometry.direction),
        "axis_order": reference_geometry.axis_order,
        "intensity": _stats(prior),
    }


def serialize_dynamic_prior_state(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["completed_prior_update_iterations"] = [
        int(v) for v in payload.get("completed_prior_update_iterations", [])
    ]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return str(path)


def load_dynamic_prior_state(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["current_prior_round"] = int(payload.get("current_prior_round", 0))
    payload["completed_prior_update_iterations"] = [
        int(v) for v in payload.get("completed_prior_update_iterations", [])
    ]
    payload["current_iteration"] = int(payload.get("current_iteration", 0))
    return payload


class DynamicPriorManager:
    def __init__(self, dataset, opt, volume_positions):
        self.enabled = bool(getattr(opt, "enable_dynamic_prior", False))
        self.update_iterations = validate_prior_update_config(
            self.enabled,
            getattr(opt, "prior_update_iterations", []),
            getattr(opt, "iterations", 0),
            getattr(opt, "prior_fusion_beta", 0.25),
        )
        self.beta = float(getattr(opt, "prior_fusion_beta", 0.25))
        self.mode = str(getattr(opt, "prior_update_mode", "ema"))
        self.source = str(getattr(opt, "prior_update_source", "gaussian_volume"))
        self.density_mode = str(getattr(opt, "prior_update_density_mode", "effective_density"))
        self.regenerate_projection_cache = bool(
            getattr(opt, "regenerate_prior_projection_cache", True)
        )
        self.save_artifacts = bool(getattr(opt, "save_prior_update_artifacts", True))
        self.resume_state_enabled = bool(getattr(opt, "resume_dynamic_prior_state", True))
        self.prior_source_key = str(getattr(opt, "paper_management_prior_source", "soft"))
        self.output_root = Path(getattr(dataset, "model_path")) / str(
            getattr(opt, "dynamic_prior_output_dir", "dynamic_prior")
        )
        self.volume_positions = volume_positions
        self.geometry = _geometry_from_positions(volume_positions) if volume_positions is not None else None
        self.initial_prior_path = (
            getattr(opt, "refinement_prior_path", "") or getattr(dataset, "prior_path", "")
        )
        self.current_prior_round = 0
        self.current_prior_path = self.initial_prior_path
        self.completed_prior_update_iterations = []
        self.current_projection_cache_path = getattr(opt, "vessel_projection_weight_path", "")
        self.state_path = self.output_root / "dynamic_prior_state.json"
        self.update_history_path = self.output_root / "prior_update_history.jsonl"

        if self.enabled:
            if self.mode != "ema":
                raise ValueError(f"Unsupported prior_update_mode: {self.mode}")
            if self.source != "gaussian_volume":
                raise ValueError(f"Unsupported prior_update_source: {self.source}")
            if not self.initial_prior_path:
                raise ValueError("enable_dynamic_prior requires prior_path or refinement_prior_path")
            if self.geometry is None:
                raise ValueError("enable_dynamic_prior requires scene.volume_positions")

    def state_dict(self, current_iteration=0):
        return {
            "enabled": bool(self.enabled),
            "current_prior_round": int(self.current_prior_round),
            "current_prior_path": str(self.current_prior_path or ""),
            "completed_prior_update_iterations": [
                int(v) for v in self.completed_prior_update_iterations
            ],
            "current_projection_cache_path": str(self.current_projection_cache_path or ""),
            "prior_fusion_beta": float(self.beta),
            "prior_update_mode": self.mode,
            "prior_update_iterations": [int(v) for v in self.update_iterations],
            "prior_update_source": self.source,
            "prior_update_density_mode": self.density_mode,
            "current_iteration": int(current_iteration),
            "state_path": str(self.state_path),
        }

    def apply_to_args(self, dataset, opt):
        if not self.enabled:
            return
        opt.dynamic_prior_current_round = int(self.current_prior_round)
        if self.current_prior_path:
            dataset.prior_path = str(self.current_prior_path)
            opt.refinement_prior_path = str(self.current_prior_path)
        if self.current_projection_cache_path:
            opt.vessel_projection_weight_path = str(self.current_projection_cache_path)

    def restore(self, checkpoint_state, first_iter):
        if not self.enabled:
            return
        if checkpoint_state:
            completed = [int(v) for v in checkpoint_state.get("completed_prior_update_iterations", [])]
            expected_completed = [v for v in self.update_iterations if v <= int(first_iter)]
            missing = [v for v in expected_completed if v not in completed]
            if missing:
                raise RuntimeError(
                    "dynamic prior checkpoint is missing completed update iterations "
                    f"{missing}; refusing to silently resume with an old prior"
                )
            prior_path = checkpoint_state.get("current_prior_path", "")
            if expected_completed and not prior_path:
                raise RuntimeError("dynamic prior checkpoint does not contain current_prior_path")
            if prior_path and not Path(prior_path).exists():
                raise FileNotFoundError(f"dynamic prior current_prior_path not found: {prior_path}")
            cache_path = checkpoint_state.get("current_projection_cache_path", "")
            if cache_path and not Path(cache_path).exists():
                raise FileNotFoundError(f"dynamic prior projection cache not found: {cache_path}")
            self.current_prior_round = int(checkpoint_state.get("current_prior_round", len(completed)))
            self.current_prior_path = prior_path or self.initial_prior_path
            self.current_projection_cache_path = cache_path or self.current_projection_cache_path
            self.completed_prior_update_iterations = completed
            return

        expected_completed = [v for v in self.update_iterations if v <= int(first_iter)]
        if expected_completed:
            raise RuntimeError(
                "checkpoint has no dynamic prior state after "
                f"completed update nodes {expected_completed}"
            )

    def should_update(self, iteration):
        return should_trigger_prior_update(
            iteration,
            self.update_iterations,
            self.completed_prior_update_iterations,
            enabled=self.enabled,
        )

    def save_state(self, iteration):
        return serialize_dynamic_prior_state(self.state_dict(current_iteration=iteration), self.state_path)

    def _round_dir(self, round_index):
        return self.output_root / f"round{int(round_index)}"

    def save_initial_snapshot(self, opt, iteration=0):
        if (
            not self.enabled
            or not self.save_artifacts
            or self.current_prior_round != 0
            or self.completed_prior_update_iterations
        ):
            return
        round_dir = self._round_dir(0)
        prior_dir = round_dir / "prior_initial"
        prior_dir.mkdir(parents=True, exist_ok=True)
        if self.current_prior_path and Path(self.current_prior_path).exists():
            shutil.copy2(self.current_prior_path, prior_dir / Path(self.current_prior_path).name)
        self._save_config_snapshot(opt, round_dir / "config_snapshot")
        self.save_state(iteration)

    def _save_config_snapshot(self, opt, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "opt_snapshot.json", "w", encoding="utf-8") as handle:
            json.dump(vars(opt), handle, indent=2, default=str)

    def _export_gaussian_volume(self, gaussians, opt, out_dir, iteration):
        from utils.differentiable_volume import (
            gaussian_centers_to_density_grid,
            gaussian_density_weights,
        )

        if self.geometry is None:
            raise RuntimeError("dynamic prior geometry is not initialized")
        shape = tuple(int(v) for v in self.geometry.shape)
        if len(set(shape)) != 1:
            raise ValueError(
                f"dynamic Gaussian volume export currently requires a cubic grid, got {shape}"
            )
        grid_size = shape[0]
        weights = gaussian_density_weights(gaussians, mode=self.density_mode)
        splat_mode = str(getattr(opt, "prior_update_splat_mode", getattr(opt, "volume_splat_mode", "dgr")))
        volume, valid = gaussian_centers_to_density_grid(
            gaussians,
            weights,
            self.geometry.grid_min,
            self.geometry.grid_max,
            grid_size,
            splat_mode=splat_mode,
            splat_radius=int(getattr(opt, "prior_update_splat_radius", getattr(opt, "volume_splat_radius", 3))),
            min_sigma_voxels=float(
                getattr(opt, "prior_update_min_sigma_voxels", getattr(opt, "volume_min_sigma_voxels", 0.40))
            ),
            max_sigma_voxels=float(
                getattr(opt, "prior_update_max_sigma_voxels", getattr(opt, "volume_max_sigma_voxels", 3.5))
            ),
            dgr_sigma_scale=float(
                getattr(opt, "prior_update_dgr_sigma_scale", getattr(opt, "volume_dgr_sigma_scale", 1.10))
            ),
            dgr_normalize_kernel=bool(
                getattr(opt, "prior_update_dgr_normalize_kernel", getattr(opt, "volume_dgr_normalize_kernel", False))
            ),
            dgr_supersample=int(
                getattr(opt, "prior_update_dgr_supersample", getattr(opt, "volume_dgr_supersample", 1))
            ),
            dgr_kernel_cutoff=float(
                getattr(opt, "prior_update_dgr_kernel_cutoff", getattr(opt, "volume_dgr_kernel_cutoff", 0.0))
            ),
            dgr_max_splat_radius=int(
                getattr(opt, "prior_update_dgr_max_splat_radius", getattr(opt, "volume_dgr_max_splat_radius", 0))
            ),
            normalize=True,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        volume_np = volume.detach().cpu().numpy().astype(np.float32)
        volume_path = out_dir / f"recon_volume_iter{int(iteration)}.npy"
        np.save(volume_path, volume_np)
        save_histogram(volume_np, out_dir, "reconstructed_volume_density")
        stats = {
            "iteration": int(iteration),
            "density_mode": self.density_mode,
            "splat_mode": splat_mode,
            "valid_gaussian_ratio": float(valid.float().mean().detach().item()) if valid.numel() else 0.0,
            "volume_path": _jsonable_path(volume_path),
            "volume": _stats(volume_np),
            "geometry": self.geometry.as_dict(),
        }
        with open(out_dir / "reconstructed_volume_stats.json", "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
        return volume_np, volume_path, stats

    def _save_prior_outputs(self, out_dir, prior, vesselness, source_prior, opt, iteration, label):
        out_dir.mkdir(parents=True, exist_ok=True)
        threshold = float(getattr(opt, "prior_update_threshold", 0.5))
        roi_dilation = int(getattr(opt, "prior_update_roi_dilation", 2))
        occupancy_mask = np.asarray(prior >= threshold, dtype=bool)
        roi_mask = dilate_mask(occupancy_mask, roi_dilation)
        positions = np.asarray(self.volume_positions, dtype=np.float32)
        if positions.shape[:3] != prior.shape:
            raise ValueError(
                f"volume_positions shape {positions.shape[:3]} does not match prior shape {prior.shape}"
            )
        points_source = roi_mask if str(getattr(opt, "prior_update_points_source", "roi")) == "roi" else occupancy_mask
        points = positions[points_source].reshape(-1, 3).astype(np.float32)
        points = sample_points(
            points,
            int(getattr(opt, "prior_update_max_points", getattr(opt, "prior_max_points", 50000))),
            int(getattr(opt, "prior_update_seed", 0)),
        )
        np.save(out_dir / "prior_volume.npy", prior.astype(np.float32))
        np.save(out_dir / "soft_prior.npy", prior.astype(np.float32))
        np.save(out_dir / "occupancy_mask.npy", occupancy_mask.astype(np.uint8))
        np.save(out_dir / "roi_mask.npy", roi_mask.astype(np.uint8))
        np.save(out_dir / "point_cloud.npy", points.astype(np.float32))
        if vesselness is not None:
            np.save(out_dir / "vesselness.npy", vesselness.astype(np.float32))
        if source_prior is not None:
            np.save(out_dir / "source_prior.npy", source_prior.astype(np.float32))
        prior_data_path = out_dir / "prior_data.npz"
        np.savez_compressed(
            prior_data_path,
            occupancy_mask=occupancy_mask.astype(np.uint8),
            roi_mask=roi_mask.astype(np.uint8),
            points=points.astype(np.float32),
            prior_volume=prior.astype(np.float32),
            soft_prior=prior.astype(np.float32),
            vesselness=np.zeros_like(prior, dtype=np.float32)
            if vesselness is None
            else vesselness.astype(np.float32),
            source_prior=np.zeros_like(prior, dtype=np.float32)
            if source_prior is None
            else source_prior.astype(np.float32),
            volume_shape=np.asarray(prior.shape, dtype=np.int32),
        )
        if points.shape[0] > 0:
            write_ascii_ply(out_dir / "point_cloud.ply", points)
        save_preview_slices(
            {
                "volume_norm": prior.astype(np.float32),
                "occupancy_mask": occupancy_mask.astype(np.float32),
                "roi_mask": roi_mask.astype(np.float32),
            },
            out_dir,
        )
        save_mhd_uint16(prior, self._geometry_for_mhd(), out_dir, "prior_volume")
        stats = {
            "label": label,
            "iteration": int(iteration),
            "threshold": threshold,
            "roi_dilation": roi_dilation,
            "num_points": int(points.shape[0]),
            "occupancy_voxels": int(occupancy_mask.sum()),
            "occupancy_ratio": float(occupancy_mask.mean()) if occupancy_mask.size else 0.0,
            "roi_voxels": int(roi_mask.sum()),
            "roi_ratio": float(roi_mask.mean()) if roi_mask.size else 0.0,
            "prior_data": _jsonable_path(prior_data_path),
            "prior": _stats(prior),
            "geometry": self.geometry.as_dict(),
        }
        with open(out_dir / "prior_stats.json", "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
        return prior_data_path, stats

    def _geometry_for_mhd(self):
        spacing = np.asarray(self.geometry.spacing, dtype=np.float32)
        shape = np.asarray(self.geometry.shape, dtype=np.int32)
        return SimpleNamespace(dVoxel=spacing, nVoxel=shape)

    def _build_candidate_prior(self, current_prior, gaussian_volume, opt, out_dir, iteration):
        pmin, pmax = getattr(opt, "prior_update_norm_percentiles", [0.5, 99.5])
        source_norm, norm_low, norm_high = normalize_percentile(
            gaussian_volume,
            pmin=float(pmin),
            pmax=float(pmax),
        )
        scales = getattr(opt, "prior_update_vesselness_scales", [1.0, 1.5, 2.0, 2.5])
        vesselness = frangi_vesselness_3d(
            source_norm,
            scales=[float(v) for v in scales],
            alpha=float(getattr(opt, "prior_update_vesselness_alpha", 0.5)),
            beta=float(getattr(opt, "prior_update_vesselness_beta", 0.5)),
            gamma=float(getattr(opt, "prior_update_vesselness_gamma", 0.0)),
            bright=not bool(getattr(opt, "prior_update_dark_tubes", False)),
            chunk_size=int(getattr(opt, "prior_update_vesselness_chunk_size", 500000)),
        )
        candidate = fuse_prior(
            current_prior,
            vesselness,
            mode=str(getattr(opt, "prior_update_candidate_fusion_mode", "boost")),
            vessel_weight=float(getattr(opt, "prior_update_candidate_vessel_weight", 0.35)),
            blend_weight=float(getattr(opt, "prior_update_candidate_blend_weight", 0.20)),
        )
        candidate_path, candidate_stats = self._save_prior_outputs(
            out_dir,
            candidate,
            vesselness,
            current_prior,
            opt,
            iteration,
            label="candidate",
        )
        candidate_stats.update(
            {
                "source_norm_low": float(norm_low),
                "source_norm_high": float(norm_high),
                "vesselness": _stats(vesselness),
            }
        )
        with open(out_dir / "candidate_prior_stats.json", "w", encoding="utf-8") as handle:
            json.dump(candidate_stats, handle, indent=2)
        return candidate, vesselness, candidate_path, candidate_stats

    def _build_projection_cache(self, source_path, prior_path, out_dir, opt):
        if not self.regenerate_projection_cache:
            return ""
        from scripts.build_vessel_projection_weights import (
            build_tigre_geometry,
            finite_stats,
            forward_project,
            match_projection_shape,
            normalize_weights,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "vessel_projection_weights.npz"
        with open(source_path, "rb") as handle:
            data = pickle.load(handle)
        prior, prior_key, _ = load_prior_array(prior_path, source="soft")
        expected_shape = tuple(map(int, data["nVoxel"]))
        if prior.shape != expected_shape:
            raise ValueError(f"Prior shape {prior.shape} does not match source nVoxel {expected_shape}")
        geo = build_tigre_geometry(data)
        outputs = {}
        stats = {
            "source": str(source_path),
            "prior_path": str(prior_path),
            "prior_key": prior_key,
            "normalization": str(getattr(opt, "vessel_projection_normalization", "per_view_percentile")),
            "percentile": float(getattr(opt, "vessel_projection_percentile", 99.5)),
            "transpose_detector": bool(getattr(opt, "vessel_projection_transpose_detector", False)),
            "prior": finite_stats("prior", prior),
        }
        for split in ("train", "val"):
            if split not in data:
                continue
            observed = np.asarray(data[split]["projections"], dtype=np.float32)
            angles = np.asarray(data[split]["angles"], dtype=np.float32)
            raw = forward_project(prior, geo, angles)
            weights = normalize_weights(
                raw,
                mode=stats["normalization"],
                percentile=stats["percentile"],
            )
            weights = match_projection_shape(
                weights,
                observed,
                split=split,
                transpose_detector=stats["transpose_detector"],
            )
            outputs[f"{split}_weights"] = weights.astype(np.float32)
            outputs[f"{split}_angles"] = angles.astype(np.float32)
            stats[f"{split}_raw"] = finite_stats(f"{split}_raw", raw)
            stats[f"{split}_weights"] = finite_stats(f"{split}_weights", weights)
            stats[f"{split}_observed"] = finite_stats(f"{split}_observed", observed)
        np.savez_compressed(
            out_path,
            **outputs,
            source_path=str(source_path),
            prior_path=str(prior_path),
            prior_key=prior_key,
            normalization=stats["normalization"],
            percentile=np.asarray(stats["percentile"], dtype=np.float32),
            transpose_detector=np.asarray(stats["transpose_detector"]),
        )
        with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
        return str(out_path)

    def update_after_iteration(self, iteration, gaussians, opt, source_path, checkpoint_callback):
        if not self.should_update(iteration):
            return None
        start = time.time()
        next_round = int(self.current_prior_round) + 1
        round_dir = self._round_dir(next_round)
        previous_gaussian_count = int(gaussians.get_xyz.shape[0])
        checkpoint_dir = round_dir / "checkpoint_before_update"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"chkpnt_before_prior_update_{int(iteration)}.pth"
        checkpoint_callback(checkpoint_path, iteration)

        previous_prior_path = str(self.current_prior_path)
        current_prior, current_key, _ = load_prior_array(previous_prior_path, self.prior_source_key)
        current_meta = validate_prior_geometry(current_prior, self.geometry, "current_prior")
        gaussian_volume, volume_path, volume_stats = self._export_gaussian_volume(
            gaussians,
            opt,
            round_dir / "reconstructed_volume",
            iteration,
        )
        candidate, vesselness, candidate_path, candidate_stats = self._build_candidate_prior(
            current_prior,
            gaussian_volume,
            opt,
            round_dir / "prior_candidate",
            iteration,
        )
        candidate_meta = validate_prior_geometry(candidate, self.geometry, "candidate_prior")
        fused = ema_fuse_prior(current_prior, candidate, self.beta)
        fused_path, fused_stats = self._save_prior_outputs(
            round_dir / "prior_fused",
            fused,
            vesselness,
            current_prior,
            opt,
            iteration,
            label="fused",
        )
        projection_cache_path = ""
        uses_projection_cache = bool(getattr(opt, "use_vessel_weighted_projection", False)) or bool(
            getattr(opt, "use_vessel_projection_edge_loss", False)
        )
        if uses_projection_cache:
            if not self.regenerate_projection_cache:
                raise RuntimeError(
                    "dynamic prior update invalidates vessel projection weights, but "
                    "regenerate_prior_projection_cache=False"
                )
            projection_cache_path = self._build_projection_cache(
                source_path,
                fused_path,
                round_dir / "projection_cache",
                opt,
            )

        self.current_prior_round = next_round
        self.current_prior_path = str(fused_path)
        if projection_cache_path:
            self.current_projection_cache_path = projection_cache_path
        self.completed_prior_update_iterations.append(int(iteration))
        self._save_config_snapshot(opt, round_dir / "config_snapshot")
        self.save_state(iteration)
        elapsed = time.time() - start
        event = {
            "iteration": int(iteration),
            "prior_round": int(self.current_prior_round),
            "beta": float(self.beta),
            "previous_prior_path": previous_prior_path,
            "current_prior_path": str(self.current_prior_path),
            "checkpoint_before_update": _jsonable_path(checkpoint_path),
            "reconstructed_volume_path": _jsonable_path(volume_path),
            "candidate_prior_path": _jsonable_path(candidate_path),
            "fused_prior_path": _jsonable_path(fused_path),
            "projection_cache_path": str(projection_cache_path),
            "projection_cache_regenerated": bool(projection_cache_path),
            "candidate_stats": candidate_stats.get("prior", _stats(candidate)),
            "fused_stats": fused_stats.get("prior", _stats(fused)),
            "current_prior_geometry": current_meta,
            "candidate_prior_geometry": candidate_meta,
            "reference_geometry": self.geometry.as_dict(),
            "resampled": False,
            "gaussian_count_before_update": previous_gaussian_count,
            "gaussian_count_after_update": int(gaussians.get_xyz.shape[0]),
            "checkpoint_source": "in_memory_gaussians_capture",
            "elapsed_sec": elapsed,
            "density_mode": self.density_mode,
            "reconstructed_volume_stats": volume_stats,
        }
        self.update_history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.update_history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        return event
