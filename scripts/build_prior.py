import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from priors.volume_prior import build_volume_prior, load_volume_and_geometry, save_prior_outputs


def main():
    parser = argparse.ArgumentParser(description="Build reconstruction priors from a CT volume.")
    parser.add_argument("--input", required=True, help="X-Gaussian pickle, .npy volume, or .npz volume.")
    parser.add_argument("--out_dir", required=True, help="Output directory for prior files.")
    parser.add_argument("--source_type", default="gt", choices=["gt", "fdk", "naf", "volume"])
    parser.add_argument("--geometry_pickle", default=None, help="X-Gaussian pickle used for geometry when input is .npy/.npz.")
    parser.add_argument("--threshold_mode", default="quantile", choices=["quantile", "absolute"])
    parser.add_argument("--threshold_value", type=float, default=0.98)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--max_points", type=int, default=50000, help="0 means keep all sampled points.")
    parser.add_argument("--roi_dilation", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    volume, geometry = load_volume_and_geometry(args.input, args.geometry_pickle)
    prior = build_volume_prior(
        volume,
        geometry=geometry,
        threshold_mode=args.threshold_mode,
        threshold_value=args.threshold_value,
        sample_stride=args.sample_stride,
        max_points=args.max_points,
        roi_dilation=args.roi_dilation,
        seed=args.seed,
    )
    outputs = save_prior_outputs(prior, args.out_dir, source_path=args.input, source_type=args.source_type)

    print(json.dumps({"stats": prior["stats"], "outputs": outputs}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

