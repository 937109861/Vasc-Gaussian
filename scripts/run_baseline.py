import argparse
import os
import shlex
import subprocess
import sys


def run_command(command, dry_run=False):
    printable = " ".join(shlex.quote(part) for part in command)
    print(printable)
    if dry_run:
        return 0
    return subprocess.call(command)


def main():
    parser = argparse.ArgumentParser(description="Run the X-Gaussian ACUI baseline on a cloud server.")
    parser.add_argument("--config", default="configs/baseline_chest.yaml", help="Baseline YAML config.")
    parser.add_argument("--gpu_id", default="0", help="CUDA device id passed to train.py.")
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    parser.add_argument("--eval", action="store_true", help="Pass --eval to train.py.")
    parser.add_argument("--skip_train", action="store_true", help="Do not run training.")
    parser.add_argument("--skip_render", action="store_true", help="Do not run render.py after training.")
    parser.add_argument("--skip_metrics", action="store_true", help="Do not run evaluation/eval_nvs.py after rendering.")
    parser.add_argument("--model_path", default=None, help="Override output model path for render/eval.")
    parser.add_argument("--source_path", default="data/chest_50.pickle", help="Source pickle used by render.py.")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration for render/eval. -1 resolves latest for render.py.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    train_cmd = [
        args.python,
        "train.py",
        "--config",
        args.config,
        "--gpu_id",
        str(args.gpu_id),
    ]
    if args.eval:
        train_cmd.append("--eval")

    if not args.skip_train:
        code = run_command(train_cmd, args.dry_run)
        if code != 0:
            return code

    model_path = args.model_path or "output/baseline/chest_acui"

    if not args.skip_render:
        render_cmd = [
            args.python,
            "render.py",
            "-m",
            model_path,
            "-s",
            args.source_path,
            "--iteration",
            str(args.iteration),
            "--skip_train",
        ]
        code = run_command(render_cmd, args.dry_run)
        if code != 0:
            return code

    if not args.skip_metrics:
        eval_cmd = [
            args.python,
            "evaluation/eval_nvs.py",
            "--model_path",
            model_path,
            "--iteration",
            str(args.iteration),
        ]
        code = run_command(eval_cmd, args.dry_run)
        if code != 0:
            return code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

