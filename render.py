import torch
from scene import Scene
import hashlib
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model import GaussianModel_Xray as GaussianModel
from utils.analysis_utils import save_json
import time


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    checkpoint_path,
    checkpoint_sha256,
):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    original_gts_path = os.path.join(
        model_path,
        name,
        "ours_{}".format(iteration),
        "gt_original",
    )

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(original_gts_path, exist_ok=True)

    start = time.time()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.normalized_image[0:3, :, :]
        gt_original = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(
            gt_original,
            os.path.join(original_gts_path, '{0:05d}'.format(idx) + ".png"),
        )
    elapsed = time.time() - start
    save_json(
        os.path.join(model_path, name, "ours_{}".format(iteration), "render_stats.json"),
        {
            "split": name,
            "iteration": int(iteration),
            "num_views": int(len(views)),
            "render_time_sec": elapsed,
            "fps": (len(views) / elapsed) if elapsed > 0 and len(views) > 0 else 0.0,
            "gt_space": "per_view_minmax_normalized",
            "original_gt_dir": original_gts_path,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
            "density_render_mode": getattr(
                pipeline,
                "density_render_mode",
                "legacy",
            ),
            "density_render_constant_color": bool(
                getattr(
                    pipeline,
                    "density_render_constant_color",
                    False,
                )
            ),
        },
    )


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        checkpoint_path = os.path.join(
            dataset.model_path,
            "point_cloud",
            "iteration_{}".format(scene.loaded_iter),
            "point_cloud.ply",
        )
        checkpoint_sha256 = sha256_file(checkpoint_path)
        print(
            "Loaded checkpoint: {} (sha256={})".format(
                checkpoint_path,
                checkpoint_sha256,
            )
        )

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                       background, checkpoint_path, checkpoint_sha256)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                       background, checkpoint_path, checkpoint_sha256)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
