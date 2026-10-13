#!/usr/bin/env python3
"""Safe, profile-based bridge between Agnes Harness and Vasc-Gaussian.

The agent never supplies a shell command or arbitrary path. It selects a committed
profile and mode; this bridge constructs argument arrays, records provenance, and
runs the pipeline without a shell.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


BRIDGE_VERSION = "0.1.0"
RUN_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}$")
PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def repo_root() -> Path:
    configured = os.environ.get("VASC_GAUSSIAN_ROOT")
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[1]


ROOT = repo_root()
PROFILES_DIR = ROOT / "agh" / "profiles"
RUNS_DIR = Path(os.environ.get("VASC_AGH_RUNS_DIR", ROOT / "agh_runs")).expanduser().resolve()


class BridgeError(RuntimeError):
    def __init__(self, code: str, message: str, details: Any | None = None):
        super().__init__(message)
        self.code = code
        self.details = details


def emit(payload: dict[str, Any], exit_code: int = 0) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(exit_code)


def fail(code: str, message: str, details: Any | None = None) -> None:
    raise BridgeError(code, message, details)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def append_event(run_dir: Path, event: str, **data: Any) -> None:
    row = {"time": utc_now(), "event": event, **data}
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_repo_path(relative: str) -> Path:
    candidate = (ROOT / relative).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError as error:
        fail("PATH_OUTSIDE_REPOSITORY", f"Profile path escapes repository: {relative}")
    return candidate


def profile_path(profile_id: str) -> Path:
    if not PROFILE_ID_RE.fullmatch(profile_id):
        fail("INVALID_PROFILE", "Profile id may contain only lowercase letters, digits, and hyphens.")
    return PROFILES_DIR / f"{profile_id}.json"


def load_profile(profile_id: str) -> dict[str, Any]:
    path = profile_path(profile_id)
    if not path.is_file():
        fail("PROFILE_NOT_FOUND", f"Unknown profile: {profile_id}")
    profile = read_json(path)
    if profile.get("id") != profile_id:
        fail("PROFILE_ID_MISMATCH", f"Profile file does not declare id {profile_id}.")
    for key in ("base_config", "source_path", "export", "evaluation"):
        if key not in profile:
            fail("INVALID_PROFILE", f"Profile {profile_id} is missing {key}.")
    return profile


def list_profiles() -> dict[str, Any]:
    profiles = []
    if PROFILES_DIR.is_dir():
        for path in sorted(PROFILES_DIR.glob("*.json")):
            try:
                item = read_json(path)
                profiles.append({key: item.get(key) for key in ("id", "title", "description")})
            except (OSError, ValueError):
                continue
    return {"ok": True, "bridge_version": BRIDGE_VERSION, "profiles": profiles}


def python_bin() -> str:
    configured = os.environ.get("VASC_PYTHON", sys.executable)
    resolved = shutil.which(configured) if not Path(configured).is_absolute() else configured
    return str(resolved or configured)


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def module_probe(executable: str) -> dict[str, bool]:
    modules = ["torch", "yaml", "numpy", "scipy", "skimage", "imageio", "PIL"]
    code = "import importlib.util,json; print(json.dumps({m: importlib.util.find_spec(m) is not None for m in " + repr(modules) + "}))"
    try:
        result = subprocess.run(
            [executable, "-c", code], capture_output=True, text=True, timeout=20, check=False
        )
        return json.loads(result.stdout) if result.returncode == 0 else {module: False for module in modules}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {module: False for module in modules}


def gpu_probe() -> dict[str, Any]:
    command = shutil.which("nvidia-smi")
    if not command:
        return {"available": False, "reason": "nvidia-smi not found"}
    try:
        result = subprocess.run(
            [command, "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return {"available": result.returncode == 0 and bool(rows), "gpus": rows, "error": result.stderr.strip()[:500]}
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "reason": str(error)}


def preflight(profile_id: str, mode: str, gpu_id: str) -> dict[str, Any]:
    if mode not in {"quick", "full"}:
        fail("INVALID_MODE", "Mode must be quick or full.")
    if not re.fullmatch(r"[0-9]+", gpu_id):
        fail("INVALID_GPU", "GPU id must be a non-negative integer.")
    profile = load_profile(profile_id)
    executable = python_bin()
    required = [profile["base_config"], profile["source_path"]]
    if mode == "full":
        required.extend(profile.get("full_required_files", []))
    files = [{"path": item, "exists": safe_repo_path(item).is_file()} for item in required]
    modules = module_probe(executable)
    gpu = gpu_probe()
    free = shutil.disk_usage(RUNS_DIR.parent if RUNS_DIR.parent.exists() else ROOT).free
    issues = []
    if not Path(executable).exists() and not shutil.which(executable):
        issues.append(f"Python executable not found: {executable}")
    issues.extend(f"Missing required file: {item['path']}" for item in files if not item["exists"])
    issues.extend(f"Missing Python module: {name}" for name, available in modules.items() if not available)
    if not gpu.get("available"):
        issues.append("CUDA GPU was not detected through nvidia-smi")
    if free < 5 * 1024**3:
        issues.append("Less than 5 GiB free disk space")
    return {
        "ok": True,
        "ready": not issues,
        "profile": profile_id,
        "mode": mode,
        "gpu_id": gpu_id,
        "repository": str(ROOT),
        "git_commit": git_commit(),
        "python": executable,
        "required_files": files,
        "python_modules": modules,
        "gpu": gpu,
        "free_disk_gib": round(free / 1024**3, 2),
        "issues": issues,
    }


def run_dir_for(run_id: str) -> Path:
    if not RUN_ID_RE.fullmatch(run_id):
        fail("INVALID_RUN_ID", "Run id has an invalid format.")
    return RUNS_DIR / run_id


def read_status(run_id: str, tail_lines: int = 40) -> dict[str, Any]:
    run_dir = run_dir_for(run_id)
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        fail("RUN_NOT_FOUND", f"Unknown run: {run_id}")
    status = read_json(status_path)
    log_path = run_dir / "run.log"
    tail = []
    if log_path.is_file() and tail_lines > 0:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            tail = handle.readlines()[-min(tail_lines, 100):]
    return {"ok": True, "status": status, "log_tail": [line.rstrip()[:1000] for line in tail]}


def active_runs() -> list[dict[str, Any]]:
    rows = []
    if not RUNS_DIR.exists():
        return rows
    for path in sorted(RUNS_DIR.glob("*/status.json"), reverse=True):
        try:
            status = read_json(path)
            if status.get("state") in {"queued", "running", "stopping"}:
                pid = status.get("runner_pid")
                alive = False
                if isinstance(pid, int):
                    try:
                        os.kill(pid, 0)
                        alive = True
                    except (ProcessLookupError, PermissionError, OSError):
                        alive = False
                if alive:
                    rows.append(status)
                elif status.get("state") != "queued":
                    status.update(
                        state="failed",
                        stage="runner_disappeared",
                        error_code="RUNNER_DISAPPEARED",
                        error="The background runner exited without writing a terminal state.",
                        updated_at=utc_now(),
                    )
                    atomic_json(path, status)
        except (OSError, ValueError):
            continue
    return rows


def start_run(profile_id: str, mode: str, gpu_id: str) -> dict[str, Any]:
    check = preflight(profile_id, mode, gpu_id)
    if not check["ready"]:
        fail("PREFLIGHT_FAILED", "Preflight did not pass; no process was started.", check)
    existing = active_runs()
    if existing:
        fail("RUN_ALREADY_ACTIVE", "A Vasc-Gaussian run is already active.", existing)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = run_dir_for(run_id)
    run_dir.mkdir(parents=False)
    manifest = {
        "schema_version": 1,
        "bridge_version": BRIDGE_VERSION,
        "run_id": run_id,
        "profile": profile_id,
        "mode": mode,
        "gpu_id": gpu_id,
        "created_at": utc_now(),
        "repository": str(ROOT),
        "git_commit": check["git_commit"],
        "python": check["python"],
        "preflight": check,
    }
    atomic_json(run_dir / "manifest.json", manifest)
    status = {
        "run_id": run_id,
        "profile": profile_id,
        "mode": mode,
        "state": "queued",
        "stage": "initializing",
        "created_at": manifest["created_at"],
        "updated_at": utc_now(),
        "run_dir": str(run_dir),
    }
    atomic_json(run_dir / "status.json", status)
    append_event(run_dir, "run_queued", profile=profile_id, mode=mode)
    log_handle = (run_dir / "run.log").open("a", encoding="utf-8")
    command = [check["python"], str(Path(__file__).resolve()), "_execute", "--run-id", run_id]
    kwargs: dict[str, Any] = {"cwd": ROOT, "stdout": log_handle, "stderr": subprocess.STDOUT, "env": os.environ.copy()}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **kwargs)
    log_handle.close()
    status.update({"state": "running", "runner_pid": process.pid, "updated_at": utc_now()})
    atomic_json(run_dir / "status.json", status)
    append_event(run_dir, "runner_started", pid=process.pid)
    return {"ok": True, "started": True, "run_id": run_id, "status": status}


def classify_failure(log_text: str) -> str:
    lowered = log_text.lower()
    if "out of memory" in lowered or "cuda oom" in lowered:
        return "CUDA_OUT_OF_MEMORY"
    if "no such file" in lowered or "filenotfounderror" in lowered:
        return "MISSING_FILE"
    if "modulenotfounderror" in lowered or "importerror" in lowered:
        return "MISSING_DEPENDENCY"
    if "cuda" in lowered and "not available" in lowered:
        return "CUDA_UNAVAILABLE"
    return "STAGE_FAILED"


def update_status(run_dir: Path, **changes: Any) -> dict[str, Any]:
    status = read_json(run_dir / "status.json")
    status.update(changes)
    status["updated_at"] = utc_now()
    atomic_json(run_dir / "status.json", status)
    return status


def prepare_config(profile: dict[str, Any], mode: str, run_dir: Path) -> tuple[Path, Path, Path]:
    try:
        import yaml
    except ImportError as error:
        fail("MISSING_DEPENDENCY", "PyYAML is required to resolve the training configuration.")
    base_path = safe_repo_path(profile["base_config"])
    with base_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        fail("INVALID_CONFIG", f"Expected a mapping in {profile['base_config']}.")
    model_dir = run_dir / "model"
    export_dir = run_dir / "volume"
    eval_dir = run_dir / "evaluation"
    config["source_path"] = str(safe_repo_path(profile["source_path"]))
    config["model_path"] = str(model_dir)
    if mode == "quick":
        config.update(profile.get("quick_overrides", {}))
        config["model_path"] = str(model_dir)
        config["source_path"] = str(safe_repo_path(profile["source_path"]))
    resolved = run_dir / "resolved_config.yaml"
    with resolved.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True, allow_unicode=True)
    return resolved, export_dir, eval_dir


def stage_commands(profile: dict[str, Any], manifest: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    python = manifest["python"]
    mode = manifest["mode"]
    gpu_id = manifest["gpu_id"]
    resolved, export_dir, eval_dir = prepare_config(profile, mode, run_dir)
    model_dir = run_dir / "model"
    source = safe_repo_path(profile["source_path"])
    prediction = export_dir / "recon_volume.npy"
    export = profile["export"]
    evaluation = profile["evaluation"]
    export_args = [
        python, "export_ct_volume.py", "--model_path", str(model_dir), "--source_path", str(source),
        "--iteration", "-1", "--mode", str(export["mode"]), "--density_mapping_mode",
        str(export["density_mapping_mode"]), "--dgr_reference_grid_size", str(export["dgr_reference_grid_size"]),
        "--dgr_splat_radius", str(export["dgr_splat_radius"]), "--dgr_min_sigma_voxels",
        str(export["dgr_min_sigma_voxels"]), "--dgr_max_sigma_voxels", str(export["dgr_max_sigma_voxels"]),
        "--dgr_sigma_scale", str(export["dgr_sigma_scale"]), "--post_median_size",
        str(export["post_median_size"]), "--post_smooth_sigma", str(export["post_smooth_sigma"]),
        "--post_contrast_gamma", str(export["post_contrast_gamma"]), "--out_dir", str(export_dir),
    ]
    if export.get("slicer_export"):
        export_args.append("--slicer_export")
    if mode == "quick" and profile.get("quick_export_max_gaussians"):
        export_args.extend(["--max_gaussians", str(profile["quick_export_max_gaussians"])])
    prior = safe_repo_path(profile["full_required_files"][0]) if manifest["mode"] == "full" and profile.get("full_required_files") else None
    volume_eval = [
        python, "evaluation/eval_volume.py", "--pred", str(prediction), "--gt_pickle", str(source),
        "--out_dir", str(eval_dir / "volume"), "--label", f"{profile['id']}-{mode}",
    ]
    if prior:
        volume_eval.extend(["--prior_path", str(prior)])
    return [
        {"name": "train", "command": [python, "train.py", "--config", str(resolved), "--gpu_id", gpu_id]},
        {"name": "render", "command": [python, "render.py", "-m", str(model_dir), "-s", str(source), "--iteration", "-1", "--skip_train"]},
        {"name": "evaluate_nvs", "command": [python, "evaluation/eval_nvs.py", "--model_path", str(model_dir), "--iteration", "-1", "--split", "test"]},
        {"name": "export_volume", "command": export_args, "retry_on_oom": export_args + ["--dgr_batch_size", "10000"]},
        {"name": "evaluate_volume", "command": volume_eval},
        {"name": "evaluate_topology", "command": [
            python, "evaluation/eval_vessel_topology.py", "--pred", str(prediction), "--gt_pickle", str(source),
            "--out_dir", str(eval_dir / "topology"), "--label", f"{profile['id']}-{mode}",
            "--gt_positive_quantile", str(evaluation["gt_positive_quantile"]),
        ]},
        {"name": "evaluate_components", "command": [
            python, "evaluation/eval_threshold_topology_stats.py", "--pred", str(prediction), "--gt_pickle", str(source),
            "--out_dir", str(eval_dir / "components"), "--label", f"{profile['id']}-{mode}",
            "--gt_positive_quantile", str(evaluation["gt_positive_quantile"]),
            "--small_component_max_voxels", str(evaluation["small_component_max_voxels"]),
        ]},
    ]


def execute_stage(run_dir: Path, stage: dict[str, Any], env: dict[str, str]) -> None:
    command = stage["command"]
    append_event(run_dir, "stage_started", stage=stage["name"], argv=command)
    update_status(run_dir, state="running", stage=stage["name"], command=command)
    with (run_dir / "run.log").open("a", encoding="utf-8") as log:
        log.write(f"\n===== {stage['name']} @ {utc_now()} =====\n")
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=os.name != "nt")
        update_status(run_dir, stage_pid=process.pid)
        return_code = process.wait()
    if return_code == 0:
        append_event(run_dir, "stage_completed", stage=stage["name"], return_code=0)
        update_status(run_dir, stage_pid=None)
        return
    log_text = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")[-20000:]
    failure = classify_failure(log_text)
    if failure == "CUDA_OUT_OF_MEMORY" and stage.get("retry_on_oom"):
        append_event(run_dir, "stage_retry", stage=stage["name"], reason=failure, strategy="smaller DGR batch")
        retry = {"name": stage["name"] + "_retry", "command": stage["retry_on_oom"]}
        execute_stage(run_dir, retry, env)
        return
    raise BridgeError(failure, f"Stage {stage['name']} failed with exit code {return_code}.", {"stage": stage["name"], "return_code": return_code})


def load_metric(path: Path) -> dict[str, Any] | None:
    try:
        value = read_json(path)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def build_verification(run_dir: Path) -> dict[str, Any]:
    status = read_json(run_dir / "status.json")
    volume_path = run_dir / "volume" / "recon_volume.npy"
    volume_metrics_path = run_dir / "evaluation" / "volume" / "volume_metrics.json"
    topology_path = run_dir / "evaluation" / "topology" / "vessel_topology_metrics.json"
    components_path = run_dir / "evaluation" / "components" / "threshold_topology_stats.json"
    nvs_candidates = sorted((run_dir / "model" / "test").glob("ours_*/nvs_metrics.json"))
    volume_metrics = load_metric(volume_metrics_path)
    topology = load_metric(topology_path)
    components = load_metric(components_path)
    nvs = load_metric(nvs_candidates[-1]) if nvs_candidates else None
    checks = {
        "pipeline_completed": status.get("state") == "completed",
        "resolved_config_saved": (run_dir / "resolved_config.yaml").is_file(),
        "event_ledger_saved": (run_dir / "events.jsonl").is_file(),
        "volume_exists": volume_path.is_file() and volume_path.stat().st_size > 0,
        "volume_metrics_exist": volume_metrics is not None,
        "topology_metrics_exist": topology is not None,
        "component_metrics_exist": components is not None,
        "nvs_metrics_exist": nvs is not None,
    }
    metrics = {
        "nvs_psnr": nvs.get("psnr") if nvs else None,
        "nvs_ssim": nvs.get("ssim") if nvs else None,
        "volume_psnr": volume_metrics.get("volume_psnr") if volume_metrics else None,
        "volume_mae": volume_metrics.get("volume_mae") if volume_metrics else None,
        "dice": topology.get("dice") if topology else None,
        "cldice": topology.get("cldice") if topology else None,
        "centerline_recall": topology.get("centerline_recall") if topology else None,
        "hd95": topology.get("hd95") if topology else None,
        "largest_component_ratio": components.get("lcc_ratio") if components else None,
        "small_components": components.get("small_cc") if components else None,
    }
    finite_metrics = all(value is None or (isinstance(value, (int, float)) and math.isfinite(value)) for value in metrics.values())
    checks["metrics_are_finite"] = finite_metrics
    return {
        "run_id": status["run_id"],
        "verified": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "note": "Verified means the declared pipeline and evidence artifacts are complete; it is not a clinical-validity claim.",
    }


def execute_run(run_id: str) -> dict[str, Any]:
    run_dir = run_dir_for(run_id)
    manifest = read_json(run_dir / "manifest.json")
    profile = load_profile(manifest["profile"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = manifest["gpu_id"]
    env["PYTHONUNBUFFERED"] = "1"
    try:
        commands = stage_commands(profile, manifest, run_dir)
        manifest["stages"] = [{"name": item["name"], "argv": item["command"]} for item in commands]
        manifest["resolved_config_sha256"] = hashlib.sha256((run_dir / "resolved_config.yaml").read_bytes()).hexdigest()
        atomic_json(run_dir / "manifest.json", manifest)
        for stage in commands:
            execute_stage(run_dir, stage, env)
        update_status(run_dir, state="completed", stage="verification", stage_pid=None, completed_at=utc_now())
        verification = build_verification(run_dir)
        atomic_json(run_dir / "verification.json", verification)
        append_event(run_dir, "run_completed", verified=verification["verified"])
        update_status(run_dir, stage="done", verified=verification["verified"])
        return verification
    except Exception as error:
        code = error.code if isinstance(error, BridgeError) else "UNEXPECTED_FAILURE"
        details = error.details if isinstance(error, BridgeError) else None
        update_status(run_dir, state="failed", stage_pid=None, error_code=code, error=str(error), details=details)
        append_event(run_dir, "run_failed", code=code, message=str(error), details=details)
        raise


def verify_run(run_id: str) -> dict[str, Any]:
    run_dir = run_dir_for(run_id)
    if not (run_dir / "status.json").is_file():
        fail("RUN_NOT_FOUND", f"Unknown run: {run_id}")
    verification = build_verification(run_dir)
    return {"ok": True, "verification": verification}


def list_runs(limit: int) -> dict[str, Any]:
    runs = []
    if RUNS_DIR.exists():
        for path in sorted(RUNS_DIR.glob("*/status.json"), reverse=True)[: min(max(limit, 1), 50)]:
            try:
                runs.append(read_json(path))
            except (OSError, ValueError):
                continue
    return {"ok": True, "runs": runs}


def stop_run(run_id: str) -> dict[str, Any]:
    run_dir = run_dir_for(run_id)
    status = read_json(run_dir / "status.json")
    if status.get("state") not in {"queued", "running", "stopping"}:
        return {"ok": True, "stopped": False, "reason": f"Run is already {status.get('state')}."}
    update_status(run_dir, state="stopping")
    pids = [status.get("stage_pid"), status.get("runner_pid")]
    stopped = []
    for pid in pids:
        if not isinstance(pid, int):
            continue
        try:
            if os.name == "nt":
                os.kill(pid, signal.SIGTERM)
            else:
                os.killpg(pid, signal.SIGTERM)
            stopped.append(pid)
        except ProcessLookupError:
            continue
    update_status(run_dir, state="stopped", stage="stopped_by_user", stopped_at=utc_now(), stage_pid=None)
    append_event(run_dir, "run_stopped", pids=stopped)
    return {"ok": True, "stopped": True, "run_id": run_id, "pids": stopped}


def compare_runs(run_a: str, run_b: str) -> dict[str, Any]:
    first = verify_run(run_a)["verification"]
    second = verify_run(run_b)["verification"]
    deltas = {}
    for key, value in second["metrics"].items():
        before = first["metrics"].get(key)
        if isinstance(before, (int, float)) and isinstance(value, (int, float)):
            deltas[key] = value - before
    return {"ok": True, "baseline": first, "candidate": second, "candidate_minus_baseline": deltas}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AGH bridge for Vasc-Gaussian")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("profiles")
    pre = sub.add_parser("preflight")
    pre.add_argument("--profile", required=True)
    pre.add_argument("--mode", choices=["quick", "full"], default="quick")
    pre.add_argument("--gpu-id", default="0")
    start = sub.add_parser("start")
    start.add_argument("--profile", required=True)
    start.add_argument("--mode", choices=["quick", "full"], default="quick")
    start.add_argument("--gpu-id", default="0")
    status = sub.add_parser("status")
    status.add_argument("--run-id", required=True)
    status.add_argument("--tail-lines", type=int, default=40)
    runs = sub.add_parser("runs")
    runs.add_argument("--limit", type=int, default=10)
    verify = sub.add_parser("verify")
    verify.add_argument("--run-id", required=True)
    stop = sub.add_parser("stop")
    stop.add_argument("--run-id", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--run-a", required=True)
    compare.add_argument("--run-b", required=True)
    internal = sub.add_parser("_execute")
    internal.add_argument("--run-id", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "profiles":
            emit(list_profiles())
        if args.command == "preflight":
            emit(preflight(args.profile, args.mode, args.gpu_id))
        if args.command == "start":
            emit(start_run(args.profile, args.mode, args.gpu_id))
        if args.command == "status":
            emit(read_status(args.run_id, args.tail_lines))
        if args.command == "runs":
            emit(list_runs(args.limit))
        if args.command == "verify":
            emit(verify_run(args.run_id))
        if args.command == "stop":
            emit(stop_run(args.run_id))
        if args.command == "compare":
            emit(compare_runs(args.run_a, args.run_b))
        if args.command == "_execute":
            execute_run(args.run_id)
            return
    except BridgeError as error:
        emit({"ok": False, "error": {"code": error.code, "message": str(error), "details": error.details}}, 1)
    except Exception as error:
        emit({"ok": False, "error": {"code": "UNEXPECTED_FAILURE", "message": str(error)}}, 1)


if __name__ == "__main__":
    main()
