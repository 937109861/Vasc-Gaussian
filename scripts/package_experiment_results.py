import argparse
import json
import os
import zipfile
from pathlib import Path


TEXT_SUFFIXES = {
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".csv",
    ".log",
    ".md",
}

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

LARGE_SUFFIXES = {
    ".npy",
    ".npz",
    ".raw",
    ".ply",
    ".pth",
    ".pt",
    ".tiff",
    ".tif",
}

IMPORTANT_NAMES = {
    "cfg_args",
    "training_summary.json",
    "loss_history.jsonl",
    "metrics_history.jsonl",
    "densification_history.jsonl",
    "results.json",
    "per_view.json",
    "prior_stats.json",
    "init_stats.json",
    "gaussian_stats.json",
    "render_stats.json",
    "component_clean_stats.json",
}

IMPORTANT_DIR_NAMES = {
    "eval",
    "slices",
    "preview_slices",
    "analysis",
    "refinement",
    "occupancy_debug",
    "volume_debug",
    "progress_renders",
}


def should_include(path, include_large=False):
    suffix = path.suffix.lower()
    name = path.name

    if include_large and suffix in LARGE_SUFFIXES:
        return True
    if suffix in TEXT_SUFFIXES or suffix in IMAGE_SUFFIXES:
        return True
    if name in IMPORTANT_NAMES:
        return True
    return False


def iter_files(root, include_large=False, max_image_per_dir=80):
    root = Path(root)
    image_counts = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = set(path.relative_to(root).parts)
        in_important_dir = bool(rel_parts & IMPORTANT_DIR_NAMES)
        if not in_important_dir and path.name not in IMPORTANT_NAMES and path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if not should_include(path, include_large=include_large):
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES:
            parent = str(path.parent)
            image_counts[parent] = image_counts.get(parent, 0) + 1
            if image_counts[parent] > max_image_per_dir:
                continue
        yield path


def add_file(zf, path, arcname, manifest):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return
    zf.write(path, arcname)
    manifest.append(
        {
            "archive_path": arcname,
            "source_path": str(path),
            "size_bytes": int(path.stat().st_size),
        }
    )


def add_tree(zf, root, prefix, manifest, include_large=False):
    root = Path(root)
    if not root.exists():
        return
    for path in iter_files(root, include_large=include_large):
        rel = path.relative_to(root)
        add_file(zf, path, str(Path(prefix) / rel), manifest)


def add_if_exists(zf, path, prefix, manifest):
    path = Path(path)
    if path.exists() and path.is_file():
        add_file(zf, path, str(Path(prefix) / path.name), manifest)


def main():
    parser = argparse.ArgumentParser(description="Package experiment diagnostics for review.")
    parser.add_argument(
        "--run_dir",
        nargs="+",
        required=True,
        help="One or more output/volume_main experiment directories.",
    )
    parser.add_argument("--prior_dir", default=None, help="Optional output/priors/... directory.")
    parser.add_argument("--fdk_dir", default=None, help="Optional output/fdk directory.")
    parser.add_argument("--config", nargs="*", default=[], help="Config files to include.")
    parser.add_argument("--extra", nargs="*", default=[], help="Extra files or directories to include, e.g. screenshots.")
    parser.add_argument("--out", required=True, help="Output zip path.")
    parser.add_argument("--include_large", action="store_true", help="Also include .npy/.npz/.raw/.ply/.pth files.")
    parser.add_argument("--max_image_per_dir", type=int, default=80)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = []

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for run_dir in args.run_dir:
            run_path = Path(run_dir)
            prefix = Path("runs") / run_path.name
            add_tree(zf, run_path, prefix, manifest, include_large=args.include_large)

        if args.prior_dir:
            add_tree(zf, args.prior_dir, Path("priors") / Path(args.prior_dir).name, manifest, include_large=args.include_large)

        if args.fdk_dir:
            add_tree(zf, args.fdk_dir, Path("fdk") / Path(args.fdk_dir).name, manifest, include_large=args.include_large)

        for config in args.config:
            add_if_exists(zf, config, "configs", manifest)

        for extra in args.extra:
            extra_path = Path(extra)
            if extra_path.is_file():
                add_file(zf, extra_path, str(Path("extra") / extra_path.name), manifest)
            elif extra_path.is_dir():
                add_tree(zf, extra_path, Path("extra") / extra_path.name, manifest, include_large=args.include_large)

        zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))

    size_mb = out.stat().st_size / (1024.0 * 1024.0)
    print(json.dumps({"zip": str(out), "size_mb": size_mb, "num_files": len(manifest)}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
