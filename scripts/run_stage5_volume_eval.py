import argparse
import subprocess
import sys


def run(command):
    print(" ".join(command))
    return subprocess.call(command)


def export_and_eval(
    python,
    model_path,
    source_path,
    label,
    iteration,
    mode,
    density,
    cutoff,
    max_gaussians,
    export_mode="baseline",
    prior_path="",
    use_roi_gate=False,
    roi_gate_strength=0.2,
    roi_gate_source="roi",
    density_mapping_mode=None,
    density_calibration="percentile",
):
    suffix = f"{mode}_{density}" if export_mode == "baseline" else f"{export_mode}_{mode}_{density_mapping_mode or density}_{roi_gate_source}"
    out_dir = f"{model_path}/ct_volume/iteration_{iteration if iteration != -1 else 'latest'}_{suffix}"
    export_cmd = [
        python,
        "export_ct_volume.py",
        "-m",
        model_path,
        "-s",
        source_path,
        "--iteration",
        str(iteration),
        "--out_dir",
        out_dir,
        "--mode",
        mode,
        "--density",
        density,
        "--cutoff",
        str(cutoff),
        "--max_gaussians",
        str(max_gaussians),
        "--export_mode",
        export_mode,
    ]
    if prior_path:
        export_cmd.extend(["--prior_path", prior_path])
    if use_roi_gate:
        export_cmd.append("--use_roi_gate")
        export_cmd.extend(["--roi_gate_strength", str(roi_gate_strength)])
        export_cmd.extend(["--roi_gate_source", roi_gate_source])
    if density_mapping_mode:
        export_cmd.extend(["--density_mapping_mode", density_mapping_mode])
    if export_mode == "structure_preserved":
        export_cmd.extend(["--density_calibration", density_calibration])
    code = run(export_cmd)
    if code != 0:
        return code

    eval_cmd = [
        python,
        "evaluation/eval_volume.py",
        "--pred",
        f"{out_dir}/recon_volume.npy",
        "--gt_pickle",
        source_path,
        "--out_dir",
        f"{out_dir}/eval",
        "--label",
        label,
    ]
    if prior_path:
        eval_cmd.extend(["--prior_path", prior_path])
    return run(eval_cmd)


def main():
    parser = argparse.ArgumentParser(description="Run Stage 5 volume export and evaluation.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source_path", default="data/chest_50.pickle")
    parser.add_argument("--baseline_model", default="output/stage2/chest_acui_5k")
    parser.add_argument("--best_model", default="output/stage3/chest_acui_refinement_5k")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--mode", choices=["center", "gaussian"], default="gaussian")
    parser.add_argument("--density", choices=["opacity", "opacity_dc"], default="opacity")
    parser.add_argument("--cutoff", type=float, default=3.0)
    parser.add_argument("--max_gaussians", type=int, default=0)
    parser.add_argument("--prior_path", default="output/priors/chest_gt_q98/prior_data.npz")
    parser.add_argument("--run_structure_preserved", action="store_true")
    parser.add_argument("--roi_gate_strength", type=float, default=0.2)
    parser.add_argument("--roi_gate_source", choices=["roi", "occupancy"], default="roi")
    parser.add_argument("--density_mapping_mode", choices=["opacity", "opacity_dc", "opacity_scale"], default="opacity")
    parser.add_argument("--density_calibration", choices=["none", "percentile", "log"], default="percentile")
    args = parser.parse_args()

    code = export_and_eval(
        args.python,
        args.baseline_model,
        args.source_path,
        "baseline",
        args.iteration,
        args.mode,
        args.density,
        args.cutoff,
        args.max_gaussians,
        prior_path=args.prior_path,
    )
    if code != 0:
        return code

    code = export_and_eval(
        args.python,
        args.best_model,
        args.source_path,
        "best_refinement",
        args.iteration,
        args.mode,
        args.density,
        args.cutoff,
        args.max_gaussians,
        prior_path=args.prior_path,
    )
    if code != 0 or not args.run_structure_preserved:
        return code

    return export_and_eval(
        args.python,
        args.best_model,
        args.source_path,
        "structure_preserved",
        args.iteration,
        args.mode,
        args.density,
        args.cutoff,
        args.max_gaussians,
        export_mode="structure_preserved",
        prior_path=args.prior_path,
        use_roi_gate=True,
        roi_gate_strength=args.roi_gate_strength,
        roi_gate_source=args.roi_gate_source,
        density_mapping_mode=args.density_mapping_mode,
        density_calibration=args.density_calibration,
    )


if __name__ == "__main__":
    raise SystemExit(main())
