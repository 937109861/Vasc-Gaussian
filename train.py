import os
import torch
import numpy as np
import imageio.v2 as imageio
from random import randint
from utils.loss_utils import l1_loss, ssim
# from pytorch_msssim import ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel_Xray
from refinement import (
    PriorRefinementController,
    ProjectionSupportController,
    TopologyAwareGaussianController,
)
from refinement.paper_vessel_management import (
    FixedPriorVolume,
    PaperVesselManagementConfig,
    compute_paper_vessel_management,
)
from refinement.dynamic_prior import DynamicPriorManager
from losses import (
    LowResPriorOccupancyLoss,
    LowResVolumeLoss,
    gaussian_density_sparsity_regularization,
    gaussian_scale_regularization,
    masked_sobel_edge_loss,
    projection_support_losses,
    radiodensity_entropy_regularization,
    sobel_edge_loss,
    vessel_weighted_l1_loss,
    vessel_weighted_sobel_loss,
)
from utils.general_utils import safe_state, gen_log
from tqdm import tqdm
from utils.image_utils import psnr, time2file_name
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import datetime
import time
import yaml
from utils.analysis_utils import (
    append_jsonl,
    export_gaussian_statistics,
    save_dataset_split,
    save_init_statistics,
    save_json,
    save_render_snapshot,
)

from pdb import set_trace as stx


def _to_numpy_image(image):
    array = image.detach().float().cpu().squeeze().numpy()
    if array.max() > array.min():
        array = (array - array.min()) / max(float(array.max() - array.min()), 1e-6)
    else:
        array = np.zeros_like(array, dtype=np.float32)
    return (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)


def _attach_vessel_projection_weights(scene, opt, exp_logger):
    use_weighted_residual = bool(getattr(opt, "use_vessel_weighted_projection", False)) and (
        float(getattr(opt, "vessel_projection_weight", 0.0)) != 0.0
    )
    use_vessel_edge = bool(getattr(opt, "use_vessel_projection_edge_loss", False)) and (
        float(getattr(opt, "lambda_vessel_projection_edge", 0.0)) != 0.0
    ) and (
        float(getattr(opt, "vessel_projection_edge_weight", 1.0)) != 0.0
    )
    if not use_weighted_residual and not use_vessel_edge:
        if getattr(opt, "use_vessel_weighted_projection", False):
            exp_logger.info("Vessel-weighted projection is enabled but vessel_projection_weight=0; using baseline L1.")
        return False

    weight_path = getattr(opt, "vessel_projection_weight_path", "")
    if not weight_path:
        raise ValueError("vessel projection weighting/edge loss requires vessel_projection_weight_path")
    payload = np.load(weight_path)
    if "train_weights" not in payload:
        raise KeyError(f"vessel projection cache must contain train_weights: {weight_path}")
    train_weights = np.asarray(payload["train_weights"], dtype=np.float32)
    if not np.all(np.isfinite(train_weights)):
        raise ValueError(f"vessel projection cache contains NaN or Inf: {weight_path}")

    attached = 0
    for camera in scene.getTrainCameras():
        index = int(getattr(camera, "colmap_id", camera.uid))
        if index < 0 or index >= train_weights.shape[0]:
            raise IndexError(
                f"Camera colmap_id={index} is outside train_weights with {train_weights.shape[0]} views"
            )
        weight = train_weights[index]
        expected_shape = (
            int(camera.image_height),
            int(camera.image_width),
        )
        if tuple(weight.shape) != expected_shape:
            raise ValueError(
                f"Vessel weight for camera {index} has shape {weight.shape}, expected {expected_shape}"
            )
        tensor = torch.from_numpy(weight).to(
            device=camera.normalized_image.device,
            dtype=camera.normalized_image.dtype,
        ).unsqueeze(0)
        camera.vessel_projection_weight = tensor
        attached += 1

    exp_logger.info(
        "Enabled vessel projection weight maps: "
        f"path={weight_path}, views={attached}, "
        f"lambda_v={float(getattr(opt, 'vessel_projection_weight', 0.0))}, "
        f"use_weighted_residual={use_weighted_residual}, "
        f"use_vessel_edge={use_vessel_edge}, "
        f"lambda_edge={float(getattr(opt, 'lambda_vessel_projection_edge', 0.0))}, "
        f"cache_shape={list(train_weights.shape)}, "
        f"min={float(train_weights.min()):.6f}, max={float(train_weights.max()):.6f}, "
        f"mean={float(train_weights.mean()):.6f}"
    )
    return True


def _save_vessel_projection_debug(
    model_path,
    iteration,
    camera,
    prediction,
    target,
    vessel_weight,
    vessel_projection_weight,
):
    out_dir = os.path.join(
        model_path,
        "vessel_projection_debug",
        f"iteration_{int(iteration)}",
        f"view_{int(getattr(camera, 'colmap_id', camera.uid)):03d}",
    )
    os.makedirs(out_dir, exist_ok=True)
    residual = torch.abs(prediction.detach() - target.detach())
    residual_weight = 1.0 + float(vessel_projection_weight) * vessel_weight.detach()
    weighted_residual = residual * residual_weight
    imageio.imwrite(os.path.join(out_dir, "observed.png"), _to_numpy_image(target))
    imageio.imwrite(os.path.join(out_dir, "vessel_weight.png"), _to_numpy_image(vessel_weight))
    imageio.imwrite(os.path.join(out_dir, "predicted.png"), _to_numpy_image(prediction))
    imageio.imwrite(os.path.join(out_dir, "absolute_residual.png"), _to_numpy_image(residual))
    imageio.imwrite(os.path.join(out_dir, "weighted_residual.png"), _to_numpy_image(weighted_residual))


def _paper_management_config(opt):
    return PaperVesselManagementConfig(
        lambda_q=getattr(opt, "paper_management_lambda_q", 1.0),
        lambda_def=getattr(opt, "paper_management_lambda_def", 1.0),
        delta=getattr(opt, "paper_management_delta", 0.25),
        lambda_m=getattr(opt, "paper_management_lambda_m", 1.0),
        lambda_eta=getattr(opt, "paper_management_lambda_eta", 1.0),
        lambda_k=getattr(opt, "paper_management_lambda_k", 1.0),
        tau_keep=getattr(opt, "paper_management_tau_keep", 0.05),
        tau_den=getattr(opt, "paper_management_tau_den", getattr(opt, "densify_grad_threshold", 0.00004)),
        tau_s=getattr(opt, "paper_management_tau_s", 0.008),
        eps=getattr(opt, "paper_management_eps", 1e-8),
        prior_align_corners=getattr(opt, "paper_management_prior_align_corners", True),
    )


def _load_training_checkpoint(checkpoint):
    payload = torch.load(checkpoint, weights_only=False)
    if isinstance(payload, dict) and "gaussians" in payload:
        return payload["gaussians"], int(payload["iteration"]), payload.get("dynamic_prior_state")
    model_params, iteration = payload
    return model_params, int(iteration), None


def _make_training_checkpoint(gaussians, iteration, dynamic_prior_manager=None):
    if dynamic_prior_manager is not None and dynamic_prior_manager.enabled:
        return {
            "gaussians": gaussians.capture(),
            "iteration": int(iteration),
            "dynamic_prior_state": dynamic_prior_manager.state_dict(current_iteration=iteration),
        }
    return (gaussians.capture(), int(iteration))


def _save_training_checkpoint(path, gaussians, iteration, dynamic_prior_manager=None):
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(_make_training_checkpoint(gaussians, iteration, dynamic_prior_manager), path)


def _build_prior_runtime(dataset, opt, scene, exp_logger):
    vessel_projection_active = _attach_vessel_projection_weights(scene, opt, exp_logger)
    refinement_controller = None
    prior_analysis_controller = None
    occupancy_loss = None
    volume_loss = None
    projection_support_controller = None
    topology_gaussian_controller = None
    paper_management_enabled = bool(getattr(opt, "use_paper_vessel_management", False))
    paper_prior_sampler = None
    paper_management_config = _paper_management_config(opt)
    if getattr(opt, "use_projection_support_loss", False):
        exp_logger.info(
            "Enabled projection support loss: "
            f"foreground_lambda={getattr(opt, 'lambda_projection_foreground', 0.0)}, "
            f"background_lambda={getattr(opt, 'lambda_projection_background', 0.0)}, "
            f"threshold={getattr(opt, 'projection_foreground_threshold', 0.02)}, "
            f"dilation={getattr(opt, 'projection_mask_dilation', 2)}, "
            f"margin={getattr(opt, 'projection_background_margin', 0.01)}"
        )
    if getattr(opt, "use_projection_support_refinement", False) or paper_management_enabled:
        projection_support_controller = ProjectionSupportController(
            cameras=scene.getTrainCameras(),
            foreground_threshold=getattr(
                opt,
                "projection_support_foreground_threshold",
                0.02,
            ),
            mask_dilation=getattr(opt, "projection_support_mask_dilation", 2),
            min_support_ratio=getattr(opt, "projection_support_min_ratio", 0.20),
            min_observations=getattr(
                opt,
                "projection_support_min_observations",
                8,
            ),
            densify_floor=getattr(opt, "projection_support_densify_floor", 0.25),
            prune_max_effective_density=getattr(
                opt,
                "projection_support_prune_max_effective_density",
                0.03,
            ),
            max_views=getattr(opt, "projection_support_max_views", 16),
            chunk_size=getattr(opt, "projection_support_chunk_size", 100000),
        )
        exp_logger.info(
            "Enabled projection support controller: "
            f"views={len(projection_support_controller.cameras)}, "
            f"paper_management={paper_management_enabled}, "
            f"legacy_refinement={bool(getattr(opt, 'use_projection_support_refinement', False))}, "
            f"from={getattr(opt, 'projection_support_from_iter', 4000)}, "
            f"until={getattr(opt, 'projection_support_until_iter', 15000)}, "
            f"interval={getattr(opt, 'projection_support_interval', 400)}, "
            f"min_ratio={getattr(opt, 'projection_support_min_ratio', 0.20)}, "
            f"prune_max_density={getattr(opt, 'projection_support_prune_max_effective_density', 0.03)}"
        )
    if getattr(opt, "use_prior_refinement", False):
        refinement_prior_path = getattr(opt, "refinement_prior_path", "") or getattr(dataset, "prior_path", "")
        if scene.volume_positions is None:
            raise ValueError("Prior refinement requires X-ray volume positions from the scene")
        refinement_controller = PriorRefinementController(
            prior_path=refinement_prior_path,
            volume_positions=scene.volume_positions,
            roi_densify_weight=getattr(opt, "roi_densify_weight", 2.0),
            empty_prune_opacity=getattr(opt, "empty_prune_opacity", 0.01),
            empty_prune_mask_source=getattr(opt, "empty_prune_mask_source", "roi"),
        )
        prior_analysis_controller = refinement_controller
        refinement_description = refinement_controller.save_description(dataset.model_path)
        exp_logger.info(f"Enabled prior refinement: {refinement_description}")
    elif getattr(dataset, "prior_path", ""):
        if scene.volume_positions is None:
            raise ValueError("Prior allocation analysis requires X-ray volume positions from the scene")
        prior_analysis_controller = PriorRefinementController(
            prior_path=getattr(dataset, "prior_path", ""),
            volume_positions=scene.volume_positions,
            roi_densify_weight=getattr(opt, "roi_densify_weight", 2.0),
            empty_prune_opacity=getattr(opt, "empty_prune_opacity", 0.01),
            empty_prune_mask_source=getattr(opt, "empty_prune_mask_source", "roi"),
        )
        analysis_description = prior_analysis_controller.save_description(dataset.model_path)
        exp_logger.info(f"Enabled prior allocation analysis: {analysis_description}")
    if getattr(opt, "use_topology_gaussian_prior", False):
        topology_prior_path = getattr(opt, "topology_prior_path", "")
        if (
            bool(getattr(opt, "enable_dynamic_prior", False))
            and int(getattr(opt, "dynamic_prior_current_round", 0)) > 0
            and getattr(opt, "refinement_prior_path", "")
        ):
            topology_prior_path = getattr(opt, "refinement_prior_path", "")
        topology_base_prior_path = getattr(opt, "refinement_prior_path", "") or getattr(dataset, "prior_path", "")
        if not topology_base_prior_path:
            raise ValueError("use_topology_gaussian_prior requires prior_path or refinement_prior_path")
        if scene.volume_positions is None:
            raise ValueError("Topology Gaussian prior requires X-ray volume positions from the scene")
        topology_gaussian_controller = TopologyAwareGaussianController(
            prior_path=topology_base_prior_path,
            topology_prior_path=topology_prior_path,
            volume_positions=scene.volume_positions,
            roi_source=getattr(opt, "topology_roi_source", "support_roi"),
            roi_densify_weight=getattr(opt, "topology_roi_densify_weight", 1.25),
            outside_densify_weight=getattr(opt, "topology_outside_densify_weight", 0.75),
            outside_grid_densify_weight=getattr(opt, "topology_outside_grid_densify_weight", 0.50),
            empty_prune_enabled=getattr(opt, "topology_empty_prune_enabled", False),
            empty_prune_mask_source=getattr(opt, "topology_empty_prune_mask_source", "vessel"),
            empty_prune_density_mode=getattr(opt, "topology_empty_prune_density_mode", "effective_density"),
            empty_prune_threshold=getattr(opt, "topology_empty_prune_threshold", 0.002),
            centerline_enabled=getattr(opt, "topology_centerline_enabled", False),
            centerline_protect_distance=getattr(opt, "topology_centerline_protect_distance", 0.08),
            centerline_far_distance=getattr(opt, "topology_centerline_far_distance", 0.35),
            centerline_prune_threshold=getattr(opt, "topology_centerline_prune_threshold", 0.0015),
            soft_empty_margin=getattr(opt, "topology_soft_empty_margin", 0.05),
            soft_empty_tau=getattr(opt, "topology_soft_empty_tau", 0.05),
            soft_empty_density_mode=getattr(opt, "topology_soft_empty_density_mode", "opacity"),
            component_small_voxels=getattr(opt, "topology_component_small_voxels", 260),
        )
        prior_analysis_controller = topology_gaussian_controller
        topology_description = topology_gaussian_controller.save_description(dataset.model_path)
        exp_logger.info(
            "Enabled topology-aware Gaussian management: "
            f"{topology_description}, "
            f"from={getattr(opt, 'topology_prior_from_iter', 5000)}, "
            f"centerline_from={getattr(opt, 'topology_centerline_from_iter', 8000)}, "
            f"roi_weight={getattr(opt, 'topology_roi_densify_weight', 1.25)}, "
            f"outside_weight={getattr(opt, 'topology_outside_densify_weight', 0.75)}"
        )
    if getattr(opt, "use_occ_loss", False):
        occ_prior_path = getattr(opt, "refinement_prior_path", "") or getattr(dataset, "prior_path", "")
        if not occ_prior_path:
            raise ValueError("use_occ_loss requires refinement_prior_path or prior_path")
        occupancy_loss = LowResPriorOccupancyLoss(
            prior_path=occ_prior_path,
            volume_positions=scene.volume_positions,
            grid_size=getattr(opt, "occ_grid_size", 32),
            source=getattr(opt, "occ_source", "roi"),
            loss_type=getattr(opt, "occ_loss_type", "l1"),
        )
        exp_logger.info(
            f"Enabled occupancy consistency loss: prior={occ_prior_path}, "
            f"source={getattr(opt, 'occ_source', 'roi')}, grid={getattr(opt, 'occ_grid_size', 32)}, "
            f"type={getattr(opt, 'occ_loss_type', 'l1')}, lambda={getattr(opt, 'lambda_occ', 0.001)}"
        )
    if paper_management_enabled:
        paper_prior_path = getattr(opt, "refinement_prior_path", "") or getattr(dataset, "prior_path", "")
        if not paper_prior_path:
            raise ValueError("use_paper_vessel_management requires refinement_prior_path or prior_path")
        if occupancy_loss is None:
            raise ValueError("use_paper_vessel_management requires use_occ_loss=true to reuse O_r/O_p")
        paper_prior_sampler = FixedPriorVolume(
            prior_path=paper_prior_path,
            volume_positions=scene.volume_positions,
            source=getattr(opt, "paper_management_prior_source", "soft"),
            align_corners=getattr(opt, "paper_management_prior_align_corners", True),
        )
        save_json(
            os.path.join(dataset.model_path, "paper_vessel_management", "fixed_prior_description.json"),
            paper_prior_sampler.description(),
        )
        exp_logger.info(
            "Enabled paper vessel management: "
            f"prior={paper_prior_path}, "
            f"prior_key={paper_prior_sampler.prior_key}, "
            f"occ_grid={getattr(opt, 'occ_grid_size', 32)}, "
            f"projection_views={len(projection_support_controller.cameras)}, "
            f"lambda_q={paper_management_config.lambda_q}, "
            f"lambda_def={paper_management_config.lambda_def}, "
            f"tau_keep={paper_management_config.tau_keep}, "
            f"tau_den={paper_management_config.tau_den}, "
            f"tau_s={paper_management_config.tau_s}"
        )
    if getattr(opt, "use_volume_loss", False):
        volume_prior_path = getattr(opt, "refinement_prior_path", "") or getattr(dataset, "prior_path", "")
        volume_loss = LowResVolumeLoss(
            target_volume=scene.image_3d,
            volume_positions=scene.volume_positions,
            grid_size=getattr(opt, "volume_grid_size", 32),
            loss_type=getattr(opt, "volume_loss_type", "l1"),
            density_mode=getattr(opt, "volume_density_mode", "opacity"),
            splat_mode=getattr(opt, "volume_splat_mode", "trilinear"),
            splat_radius=getattr(opt, "volume_splat_radius", 2),
            min_sigma_voxels=getattr(opt, "volume_min_sigma_voxels", 0.75),
            max_sigma_voxels=getattr(opt, "volume_max_sigma_voxels", 3.0),
            dgr_sigma_scale=getattr(opt, "volume_dgr_sigma_scale", 1.0),
            dgr_normalize_kernel=getattr(opt, "volume_dgr_normalize_kernel", False),
            dgr_supersample=getattr(opt, "volume_dgr_supersample", 1),
            dgr_kernel_cutoff=getattr(opt, "volume_dgr_kernel_cutoff", 0.0),
            dgr_max_splat_radius=getattr(opt, "volume_dgr_max_splat_radius", 0),
            prior_path=volume_prior_path,
            use_prior_mask=getattr(opt, "volume_use_prior_mask", False),
            mask_source=getattr(opt, "volume_mask_source", "roi"),
            roi_weight=getattr(opt, "volume_roi_weight", 1.0),
            background_weight=getattr(opt, "volume_background_weight", 0.1),
            tissue_balance=getattr(opt, "volume_tissue_balance", False),
            sparse_foreground_balance=getattr(opt, "volume_sparse_foreground_balance", False),
            foreground_weight=getattr(opt, "volume_foreground_weight", 1.0),
            foreground_epsilon=getattr(opt, "volume_foreground_epsilon", 1e-6),
            soft_tissue_weight=getattr(opt, "volume_soft_tissue_weight", 2.0),
            hard_tissue_weight=getattr(opt, "volume_hard_tissue_weight", 1.0),
            soft_tissue_min_quantile=getattr(opt, "volume_soft_tissue_min_quantile", 0.05),
            soft_tissue_max_quantile=getattr(opt, "volume_soft_tissue_max_quantile", 0.75),
            hard_tissue_min_quantile=getattr(opt, "volume_hard_tissue_min_quantile", 0.90),
            background_suppression_weight=getattr(opt, "volume_background_suppression_weight", 0.0),
            background_suppression_threshold=getattr(opt, "volume_background_suppression_threshold", 0.0),
            volume_tv_weight=getattr(opt, "volume_tv_weight", 0.0),
        )
        exp_logger.info(
            f"Enabled low-res volume loss: grid={getattr(opt, 'volume_grid_size', 32)}, "
            f"type={getattr(opt, 'volume_loss_type', 'l1')}, "
            f"density={getattr(opt, 'volume_density_mode', 'opacity')}, "
            f"splat={getattr(opt, 'volume_splat_mode', 'trilinear')}, "
            f"lambda={getattr(opt, 'lambda_volume', 0.0001)}, "
            f"interval={getattr(opt, 'volume_loss_interval', 100)}, "
            f"warmup={getattr(opt, 'volume_warmup_from_iter', 1000)}, "
            f"tissue_balance={getattr(opt, 'volume_tissue_balance', False)}"
        )
    return {
        "vessel_projection_active": vessel_projection_active,
        "refinement_controller": refinement_controller,
        "prior_analysis_controller": prior_analysis_controller,
        "occupancy_loss": occupancy_loss,
        "volume_loss": volume_loss,
        "projection_support_controller": projection_support_controller,
        "topology_gaussian_controller": topology_gaussian_controller,
        "paper_management_enabled": paper_management_enabled,
        "paper_prior_sampler": paper_prior_sampler,
        "paper_management_config": paper_management_config,
    }


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    render_iterations,
    analysis_iterations,
):
    first_iter = 0
    dataset.density_render_mode = getattr(pipe, "density_render_mode", "legacy")
    dataset.density_render_constant_color = getattr(
        pipe,
        "density_render_constant_color",
        False,
    )
    exp_logger = prepare_output_and_logger(dataset)
    exp_logger.info("Training parameters: {}".format(vars(opt)))
    exp_logger.info("Pipeline parameters: {}".format(vars(pipe)))

    gaussians = GaussianModel_Xray(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    save_dataset_split(dataset.model_path, scene)
    gaussians.training_setup(opt)
    checkpoint_dynamic_state = None
    if checkpoint:
        model_params, first_iter, checkpoint_dynamic_state = _load_training_checkpoint(checkpoint)
        gaussians.restore(model_params, opt)
    dynamic_prior_manager = DynamicPriorManager(dataset, opt, scene.volume_positions)
    dynamic_prior_manager.restore(checkpoint_dynamic_state, first_iter)
    dynamic_prior_manager.apply_to_args(dataset, opt)

    runtime = _build_prior_runtime(dataset, opt, scene, exp_logger)
    vessel_projection_active = runtime["vessel_projection_active"]
    refinement_controller = runtime["refinement_controller"]
    prior_analysis_controller = runtime["prior_analysis_controller"]
    occupancy_loss = runtime["occupancy_loss"]
    volume_loss = runtime["volume_loss"]
    projection_support_controller = runtime["projection_support_controller"]
    topology_gaussian_controller = runtime["topology_gaussian_controller"]
    paper_management_enabled = runtime["paper_management_enabled"]
    paper_prior_sampler = runtime["paper_prior_sampler"]
    paper_management_config = runtime["paper_management_config"]
    if getattr(dataset, "save_init_analysis", False):
        save_init_statistics(dataset.model_path, scene.init_info, gaussians)
    dynamic_prior_manager.save_initial_snapshot(opt, iteration=first_iter)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    vessel_projection_debug_saved = 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    training_wall_start = time.time()

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.normalized_image.cuda()

        vessel_weight_map = getattr(viewpoint_cam, "vessel_projection_weight", None)
        Ll1, vessel_projection_stats = vessel_weighted_l1_loss(
            image,
            gt_image,
            vessel_weight_map=vessel_weight_map if vessel_projection_active else None,
            vessel_projection_weight=getattr(opt, "vessel_projection_weight", 0.0),
        )
        ssim_term = 1.0 - ssim(image, gt_image)
        base_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_term
        projection_foreground_loss = torch.zeros((), dtype=image.dtype, device=image.device)
        projection_background_loss = torch.zeros((), dtype=image.dtype, device=image.device)
        projection_support_stats = {}
        if getattr(opt, "use_projection_support_loss", False):
            (
                projection_foreground_loss,
                projection_background_loss,
                projection_support_stats,
            ) = projection_support_losses(
                image,
                gt_image,
                foreground_threshold=getattr(opt, "projection_foreground_threshold", 0.02),
                mask_dilation=getattr(opt, "projection_mask_dilation", 2),
                background_margin=getattr(opt, "projection_background_margin", 0.01),
            )
        edge_loss_value = torch.zeros((), dtype=image.dtype, device=image.device)
        if getattr(opt, "use_edge_loss", False):
            edge_mask_mode = getattr(opt, "edge_loss_mask_mode", "none")
            if edge_mask_mode == "target_edge":
                edge_loss_value = masked_sobel_edge_loss(
                    image,
                    gt_image,
                    quantile=getattr(opt, "edge_mask_quantile", 0.80),
                    dilation=getattr(opt, "edge_mask_dilation", 1),
                )
            elif edge_mask_mode == "none":
                edge_loss_value = sobel_edge_loss(image, gt_image)
            else:
                raise ValueError(f"Unsupported edge_loss_mask_mode: {edge_mask_mode}")
        vessel_projection_edge_loss = torch.zeros((), dtype=image.dtype, device=image.device)
        vessel_projection_edge_stats = {}
        if getattr(opt, "use_vessel_projection_edge_loss", False):
            vessel_projection_edge_loss, vessel_projection_edge_stats = vessel_weighted_sobel_loss(
                image,
                gt_image,
                vessel_weight_map=vessel_weight_map if vessel_projection_active else None,
                vessel_edge_weight=getattr(opt, "vessel_projection_edge_weight", 1.0),
                vessel_edge_threshold=getattr(opt, "vessel_projection_edge_threshold", 0.15),
                vessel_edge_dilation=getattr(opt, "vessel_projection_edge_dilation", 1),
            )
        occ_loss_value = torch.zeros((), dtype=image.dtype, device=image.device)
        occ_stats = {}
        occ_gaussian_grid = None
        occ_prior_grid = None
        if occupancy_loss is not None and iteration >= getattr(opt, "occ_warmup_from_iter", 0):
            occ_loss_value, occ_stats, occ_gaussian_grid, occ_prior_grid = occupancy_loss(gaussians)
        volume_loss_value = torch.zeros((), dtype=image.dtype, device=image.device)
        volume_stats = {}
        volume_pred_grid = None
        volume_target_grid = None
        volume_interval = max(1, int(getattr(opt, "volume_loss_interval", 100)))
        volume_loss_active = (
            volume_loss is not None
            and iteration >= getattr(opt, "volume_warmup_from_iter", 0)
            and iteration % volume_interval == 0
        )
        if volume_loss_active:
            volume_loss_value, volume_stats, volume_pred_grid, volume_target_grid = volume_loss(gaussians)
        scale_floor_loss = torch.zeros((), dtype=image.dtype, device=image.device)
        scale_max_loss = torch.zeros((), dtype=image.dtype, device=image.device)
        scale_aniso_loss = torch.zeros((), dtype=image.dtype, device=image.device)
        density_entropy_loss = torch.zeros((), dtype=image.dtype, device=image.device)
        density_sparsity_loss = torch.zeros((), dtype=image.dtype, device=image.device)
        topology_empty_loss = torch.zeros((), dtype=image.dtype, device=image.device)
        topology_empty_stats = {}
        gaussian_reg_stats = {}
        if getattr(opt, "use_gaussian_reg", False):
            scale_floor_loss, scale_max_loss, scale_aniso_loss, scale_stats = gaussian_scale_regularization(
                gaussians,
                scale_floor=getattr(opt, "scale_floor", 0.75),
                scale_max_axis=getattr(opt, "scale_max_axis", 0.0),
                max_anisotropy=getattr(opt, "scale_aniso_max_ratio", 5.0),
                anisotropy_loss_mode=getattr(opt, "scale_aniso_loss_mode", "linear"),
            )
            density_entropy_loss, density_stats = radiodensity_entropy_regularization(gaussians)
            gaussian_reg_stats = {**scale_stats, **density_stats}
        density_sparsity_active = (
            getattr(opt, "lambda_density_sparsity", 0.0) > 0.0
            and iteration >= getattr(opt, "density_sparsity_from_iter", 0)
        )
        if density_sparsity_active:
            density_sparsity_loss, density_sparsity_stats = gaussian_density_sparsity_regularization(
                gaussians,
                mode=getattr(opt, "density_sparsity_mode", "radiodensity"),
            )
            gaussian_reg_stats.update(density_sparsity_stats)
        topology_empty_loss_active = (
            topology_gaussian_controller is not None
            and getattr(opt, "lambda_topology_empty", 0.0) > 0.0
            and iteration >= getattr(opt, "topology_empty_loss_from_iter", 10000)
        )
        if topology_empty_loss_active:
            topology_empty_loss, topology_empty_stats = topology_gaussian_controller.soft_empty_loss(gaussians)
        loss = (
            getattr(opt, "lambda_render", 1.0) * base_loss
            + getattr(opt, "lambda_projection_foreground", 0.0) * projection_foreground_loss
            + getattr(opt, "lambda_projection_background", 0.0) * projection_background_loss
            + getattr(opt, "edge_loss_weight", 0.0) * edge_loss_value
            + getattr(opt, "lambda_vessel_projection_edge", 0.0) * vessel_projection_edge_loss
            + getattr(opt, "lambda_occ", 0.0) * occ_loss_value
            + getattr(opt, "lambda_volume", 0.0) * volume_loss_value
            + getattr(opt, "lambda_scale_floor", 0.0) * scale_floor_loss
            + getattr(opt, "lambda_scale_max", 0.0) * scale_max_loss
            + getattr(opt, "lambda_scale_aniso", 0.0) * scale_aniso_loss
            + getattr(opt, "lambda_density_entropy", 0.0) * density_entropy_loss
            + getattr(opt, "lambda_density_sparsity", 0.0) * density_sparsity_loss
            + getattr(opt, "lambda_topology_empty", 0.0) * topology_empty_loss
        )
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == 1 or iteration % 100 == 0 or iteration in testing_iterations:
                append_jsonl(
                    os.path.join(dataset.model_path, "loss_history.jsonl"),
                    {
                        "iteration": int(iteration),
                        "l1_loss": float(Ll1.item()),
                        "ssim_loss": float(ssim_term.item()),
                        "base_loss": float(base_loss.item()),
                        "lambda_render": float(getattr(opt, "lambda_render", 1.0)),
                        "use_vessel_weighted_projection": bool(vessel_projection_active),
                        "vessel_projection_weight": float(getattr(opt, "vessel_projection_weight", 0.0)),
                        **{
                            key: (
                                bool(value)
                                if isinstance(value, bool)
                                else float(value.item())
                                if torch.is_tensor(value)
                                else float(value)
                            )
                            for key, value in vessel_projection_stats.items()
                        },
                        "use_projection_support_loss": bool(
                            getattr(opt, "use_projection_support_loss", False)
                        ),
                        "projection_foreground_loss": float(projection_foreground_loss.item()),
                        "projection_background_loss": float(projection_background_loss.item()),
                        "lambda_projection_foreground": float(
                            getattr(opt, "lambda_projection_foreground", 0.0)
                        ),
                        "lambda_projection_background": float(
                            getattr(opt, "lambda_projection_background", 0.0)
                        ),
                        "projection_foreground_threshold": float(
                            getattr(opt, "projection_foreground_threshold", 0.02)
                        ),
                        "projection_mask_dilation": int(
                            getattr(opt, "projection_mask_dilation", 2)
                        ),
                        "projection_background_margin": float(
                            getattr(opt, "projection_background_margin", 0.01)
                        ),
                        **{
                            key: float(value.item())
                            for key, value in projection_support_stats.items()
                        },
                        "edge_loss": float(edge_loss_value.item()),
                        "edge_loss_weight": float(getattr(opt, "edge_loss_weight", 0.0)),
                        "edge_loss_mask_mode": getattr(opt, "edge_loss_mask_mode", "none"),
                        "edge_mask_quantile": float(getattr(opt, "edge_mask_quantile", 0.80)),
                        "edge_mask_dilation": int(getattr(opt, "edge_mask_dilation", 1)),
                        "vessel_projection_edge_loss": float(vessel_projection_edge_loss.item()),
                        "use_vessel_projection_edge_loss": bool(
                            getattr(opt, "use_vessel_projection_edge_loss", False)
                        ),
                        "lambda_vessel_projection_edge": float(
                            getattr(opt, "lambda_vessel_projection_edge", 0.0)
                        ),
                        "vessel_projection_edge_weight": float(
                            getattr(opt, "vessel_projection_edge_weight", 1.0)
                        ),
                        "vessel_projection_edge_threshold": float(
                            getattr(opt, "vessel_projection_edge_threshold", 0.15)
                        ),
                        "vessel_projection_edge_dilation": int(
                            getattr(opt, "vessel_projection_edge_dilation", 1)
                        ),
                        **{
                            key: (
                                bool(value)
                                if isinstance(value, bool)
                                else float(value.item())
                                if torch.is_tensor(value)
                                else float(value)
                            )
                            for key, value in vessel_projection_edge_stats.items()
                        },
                        "occ_loss": float(occ_loss_value.item()),
                        "lambda_occ": float(getattr(opt, "lambda_occ", 0.0)),
                        "use_occ_loss": bool(getattr(opt, "use_occ_loss", False)),
                        **occ_stats,
                        "volume_loss": float(volume_loss_value.item()),
                        "lambda_volume": float(getattr(opt, "lambda_volume", 0.0)),
                        "use_volume_loss": bool(getattr(opt, "use_volume_loss", False)),
                        "volume_loss_active": bool(volume_loss_active),
                        **volume_stats,
                        "use_gaussian_reg": bool(getattr(opt, "use_gaussian_reg", False)),
                        "lambda_scale_floor": float(getattr(opt, "lambda_scale_floor", 0.0)),
                        "lambda_scale_max": float(getattr(opt, "lambda_scale_max", 0.0)),
                        "lambda_scale_aniso": float(getattr(opt, "lambda_scale_aniso", 0.0)),
                        "lambda_density_entropy": float(getattr(opt, "lambda_density_entropy", 0.0)),
                        "lambda_density_sparsity": float(getattr(opt, "lambda_density_sparsity", 0.0)),
                        "density_sparsity_mode": getattr(opt, "density_sparsity_mode", "radiodensity"),
                        "density_sparsity_active": bool(density_sparsity_active),
                        "density_sparsity_from_iter": int(getattr(opt, "density_sparsity_from_iter", 0)),
                        **gaussian_reg_stats,
                        "topology_empty_loss": float(topology_empty_loss.item()),
                        "lambda_topology_empty": float(getattr(opt, "lambda_topology_empty", 0.0)),
                        "topology_empty_loss_active": bool(topology_empty_loss_active),
                        **topology_empty_stats,
                        "total_loss": float(loss.item()),
                    },
                )
            if (
                occupancy_loss is not None
                and occ_gaussian_grid is not None
                and getattr(opt, "occ_debug_interval", 0) > 0
                and iteration % getattr(opt, "occ_debug_interval", 1000) == 0
            ):
                occupancy_loss.save_debug(
                    occ_gaussian_grid,
                    occ_prior_grid,
                    os.path.join(dataset.model_path, "occupancy_debug", f"iteration_{iteration}"),
                )
            debug_interval = int(getattr(opt, "vessel_projection_debug_interval", 0))
            debug_view_limit = int(getattr(opt, "vessel_projection_debug_views", 4))
            if (
                vessel_projection_active
                and debug_interval > 0
                and iteration % debug_interval == 0
                and vessel_weight_map is not None
                and vessel_projection_debug_saved < debug_view_limit
            ):
                _save_vessel_projection_debug(
                    dataset.model_path,
                    iteration,
                    viewpoint_cam,
                    image,
                    gt_image,
                    vessel_weight_map,
                    getattr(opt, "vessel_projection_weight", 0.0),
                )
                vessel_projection_debug_saved += 1
            if (
                volume_loss is not None
                and volume_pred_grid is not None
                and getattr(opt, "volume_debug_interval", 0) > 0
                and iteration % getattr(opt, "volume_debug_interval", 1000) == 0
            ):
                volume_loss.save_debug(
                    volume_pred_grid,
                    volume_target_grid,
                    os.path.join(dataset.model_path, "volume_debug", f"iteration_{iteration}"),
                )
            topology_prune_only_active = (
                topology_gaussian_controller is not None
                and not paper_management_enabled
                and getattr(opt, "topology_empty_prune_enabled", False)
                and iteration >= getattr(opt, "topology_empty_prune_from_iter", 10000)
                and iteration <= getattr(opt, "topology_prune_only_until_iter", opt.iterations)
                and iteration >= getattr(opt, "densify_until_iter", 15000)
                and iteration % max(1, int(getattr(opt, "topology_prune_only_interval", 400))) == 0
            )
            if topology_prune_only_active:
                centerline_active = iteration >= getattr(opt, "topology_centerline_from_iter", 8000)
                (
                    _,
                    topology_prune_mask,
                    topology_protect_mask,
                    topology_region_mask,
                    topology_stats,
                ) = topology_gaussian_controller.build_iteration_bias(
                    gaussians,
                    centerline_active=centerline_active,
                    empty_prune_active=True,
                )
                prune_mask = torch.logical_and(
                    topology_prune_mask,
                    torch.logical_not(topology_protect_mask),
                )
                count_before_prune = int(gaussians.get_xyz.shape[0])
                pruned_inside_roi = int(torch.logical_and(prune_mask, topology_region_mask).sum().item())
                pruned_outside_roi = int(
                    torch.logical_and(prune_mask, torch.logical_not(topology_region_mask)).sum().item()
                )
                pruned_count = int(prune_mask.sum().item())
                if pruned_count > 0:
                    gaussians.prune_points(prune_mask)
                    torch.cuda.empty_cache()
                append_jsonl(
                    os.path.join(dataset.model_path, "topology_gaussian_prior", "prune_only_history.jsonl"),
                    {
                        "iteration": int(iteration),
                        "topology_prune_only_active": True,
                        "count_before_prune": count_before_prune,
                        "pruned_count": pruned_count,
                        "count_after_prune": int(gaussians.get_xyz.shape[0]),
                        "pruned_inside_roi": pruned_inside_roi,
                        "pruned_outside_roi": pruned_outside_roi,
                        **topology_stats,
                    },
                )
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(
                exp_logger,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                render_iterations,
                analysis_iterations,
                scene,
                render,
                (pipe, background),
                dataset,
                prior_analysis_controller,
            )

            if iteration in saving_iterations:
                exp_logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if paper_management_enabled:
                        num_gaussians_before = int(gaussians.get_xyz.shape[0])
                        paper_result = compute_paper_vessel_management(
                            gaussians=gaussians,
                            prior_sampler=paper_prior_sampler,
                            occupancy_loss=occupancy_loss,
                            projection_support_controller=projection_support_controller,
                            config=paper_management_config,
                        )
                        num_prune = int(paper_result["prune_mask"].sum().item())
                        num_clone = int(paper_result["clone_mask"].sum().item())
                        num_split = int(paper_result["split_mask"].sum().item())
                        num_retain = int(paper_result["retain_mask"].sum().item())
                        split_children = int(getattr(opt, "paper_management_split_children", 2))
                        management_stats = gaussians.apply_vessel_aware_management(
                            paper_result["prune_mask"],
                            paper_result["clone_mask"],
                            paper_result["split_mask"],
                            paper_result["retain_mask"],
                            split_children=split_children,
                        )
                        num_gaussians_after = int(gaussians.get_xyz.shape[0])
                        append_jsonl(
                            os.path.join(dataset.model_path, "paper_vessel_management", "management_history.jsonl"),
                            {
                                "iteration": int(iteration),
                                "management_mode": "paper_vessel_management",
                                "num_gaussians_before": num_gaussians_before,
                                "num_gaussians_after": num_gaussians_after,
                                "num_prune": num_prune,
                                "num_clone": num_clone,
                                "num_split": num_split,
                                "num_retain": num_retain,
                                "split_children": split_children,
                                "legacy_opacity_pruning_enabled": False,
                                "legacy_effective_density_pruning_enabled": False,
                                "legacy_size_pruning_enabled": False,
                                "fixed_prior_key": paper_prior_sampler.prior_key,
                                **paper_result["stats"],
                                **management_stats,
                            },
                        )
                        exp_logger.info(
                            "[ITER {}] Paper vessel management: before={}, after={}, "
                            "prune={}, clone={}, split={}, retain={}".format(
                                iteration,
                                num_gaussians_before,
                                num_gaussians_after,
                                num_prune,
                                num_clone,
                                num_split,
                                num_retain,
                            )
                        )
                    else:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        grad_multipliers = None
                        extra_prune_mask = None
                        prune_protect_mask = None
                        densify_region_mask = None
                        projection_support_stats = {}
                        topology_stats = {}
                        if refinement_controller is not None and iteration >= getattr(opt, "refinement_from_iter", 0):
                            refinement_grad_multipliers, refinement_extra_prune_mask, refinement_stats = refinement_controller.build_iteration_bias(gaussians)
                            if getattr(opt, "roi_densify_enabled", False):
                                grad_multipliers = refinement_grad_multipliers
                            if getattr(opt, "empty_prune_enabled", False):
                                extra_prune_mask = refinement_extra_prune_mask
                            append_jsonl(
                                os.path.join(dataset.model_path, "refinement", "refinement_history.jsonl"),
                                {
                                    "iteration": int(iteration),
                                    "roi_densify_enabled": bool(getattr(opt, "roi_densify_enabled", False)),
                                    "empty_prune_enabled": bool(getattr(opt, "empty_prune_enabled", False)),
                                    **refinement_stats,
                                },
                            )
                        topology_prior_active = (
                            topology_gaussian_controller is not None
                            and iteration >= getattr(opt, "topology_prior_from_iter", 5000)
                        )
                        if topology_prior_active:
                            centerline_active = iteration >= getattr(opt, "topology_centerline_from_iter", 8000)
                            empty_prune_active = iteration >= getattr(opt, "topology_empty_prune_from_iter", 10000)
                            (
                                topology_grad_multipliers,
                                topology_prune_mask,
                                topology_protect_mask,
                                topology_region_mask,
                                topology_stats,
                            ) = topology_gaussian_controller.build_iteration_bias(
                                gaussians,
                                centerline_active=centerline_active,
                                empty_prune_active=empty_prune_active,
                            )
                            topology_grad_multipliers = topology_grad_multipliers.reshape(-1, 1)
                            if grad_multipliers is None:
                                grad_multipliers = topology_grad_multipliers
                            else:
                                grad_multipliers = grad_multipliers.reshape(-1, 1) * topology_grad_multipliers
                            if extra_prune_mask is None:
                                extra_prune_mask = topology_prune_mask
                            else:
                                extra_prune_mask = torch.logical_or(extra_prune_mask, topology_prune_mask)
                            if prune_protect_mask is None:
                                prune_protect_mask = topology_protect_mask
                            else:
                                prune_protect_mask = torch.logical_or(prune_protect_mask, topology_protect_mask)
                            densify_region_mask = topology_region_mask
                            append_jsonl(
                                os.path.join(dataset.model_path, "topology_gaussian_prior", "history.jsonl"),
                                {
                                    "iteration": int(iteration),
                                    "topology_prior_active": bool(topology_prior_active),
                                    **topology_stats,
                                },
                            )
                        projection_support_active = (
                            projection_support_controller is not None
                            and getattr(opt, "use_projection_support_refinement", False)
                            and iteration >= getattr(opt, "projection_support_from_iter", 4000)
                            and iteration <= getattr(opt, "projection_support_until_iter", 15000)
                            and iteration % max(
                                1,
                                int(getattr(opt, "projection_support_interval", 400)),
                            ) == 0
                        )
                        if projection_support_active:
                            (
                                support_grad_multipliers,
                                support_prune_mask,
                                projection_support_stats,
                            ) = projection_support_controller.evaluate(gaussians)
                            support_grad_multipliers = support_grad_multipliers.reshape(-1, 1)
                            if grad_multipliers is None:
                                grad_multipliers = support_grad_multipliers
                            else:
                                grad_multipliers = grad_multipliers.reshape(-1, 1)
                                grad_multipliers = (
                                    grad_multipliers * support_grad_multipliers
                                )
                            if extra_prune_mask is None:
                                extra_prune_mask = support_prune_mask
                            else:
                                extra_prune_mask = torch.logical_or(
                                    extra_prune_mask,
                                    support_prune_mask,
                                )
                        configured_prune_density_mode = getattr(opt, "prune_density_mode", "opacity")
                        prune_density_mode = (
                            configured_prune_density_mode
                            if iteration >= getattr(opt, "prune_density_from_iter", 0)
                            else "opacity"
                        )
                        densification_stats = gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            getattr(opt, "densify_min_opacity", 0.005),
                            scene.cameras_extent,
                            size_threshold,
                            grad_multipliers=grad_multipliers,
                            extra_prune_mask=extra_prune_mask,
                            density_aware_pruning=getattr(opt, "density_aware_pruning", False),
                            min_radiodensity=getattr(opt, "prune_min_radiodensity", 0.02),
                            prune_density_mode=prune_density_mode,
                            min_effective_density=getattr(opt, "prune_min_effective_density", 0.0),
                            prune_protect_mask=prune_protect_mask,
                            densify_region_mask=densify_region_mask,
                        )
                        append_jsonl(
                            os.path.join(dataset.model_path, "densification_history.jsonl"),
                            {
                                "iteration": int(iteration),
                                "grad_threshold": float(opt.densify_grad_threshold),
                                "configured_prune_density_mode": configured_prune_density_mode,
                                "prune_density_from_iter": int(getattr(opt, "prune_density_from_iter", 0)),
                                "projection_support_active": bool(projection_support_active),
                                "topology_prior_active": bool(topology_prior_active),
                                **projection_support_stats,
                                **topology_stats,
                                **densification_stats,
                            },
                        )

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            dynamic_event = dynamic_prior_manager.update_after_iteration(
                iteration,
                gaussians,
                opt,
                dataset.source_path,
                checkpoint_callback=lambda path, it: _save_training_checkpoint(
                    path,
                    gaussians,
                    it,
                    dynamic_prior_manager,
                ),
            )
            if dynamic_event is not None:
                dynamic_prior_manager.apply_to_args(dataset, opt)
                runtime = _build_prior_runtime(dataset, opt, scene, exp_logger)
                vessel_projection_active = runtime["vessel_projection_active"]
                refinement_controller = runtime["refinement_controller"]
                prior_analysis_controller = runtime["prior_analysis_controller"]
                occupancy_loss = runtime["occupancy_loss"]
                volume_loss = runtime["volume_loss"]
                projection_support_controller = runtime["projection_support_controller"]
                topology_gaussian_controller = runtime["topology_gaussian_controller"]
                paper_management_enabled = runtime["paper_management_enabled"]
                paper_prior_sampler = runtime["paper_prior_sampler"]
                paper_management_config = runtime["paper_management_config"]
                dynamic_checkpoint_dir = os.path.join(
                    dataset.model_path,
                    getattr(opt, "dynamic_prior_output_dir", "dynamic_prior"),
                    f"round{dynamic_prior_manager.current_prior_round}",
                    "checkpoint",
                )
                _save_training_checkpoint(
                    os.path.join(dynamic_checkpoint_dir, f"chkpnt_after_prior_update_{iteration}.pth"),
                    gaussians,
                    iteration,
                    dynamic_prior_manager,
                )
                exp_logger.info(
                    "[ITER {}] Dynamic prior update round {} complete: prior={}, projection_cache={}, elapsed={:.4f}s".format(
                        iteration,
                        dynamic_prior_manager.current_prior_round,
                        dynamic_prior_manager.current_prior_path,
                        dynamic_prior_manager.current_projection_cache_path,
                        dynamic_event["elapsed_sec"],
                    )
                )

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                _save_training_checkpoint(
                    os.path.join(scene.model_path, "chkpnt" + str(iteration) + ".pth"),
                    gaussians,
                    iteration,
                    dynamic_prior_manager,
                )

    total_training_time = time.time() - training_wall_start
    exp_logger.info(f"Training completed in {total_training_time:.4f} s")
    save_json(
        os.path.join(dataset.model_path, "training_summary.json"),
        {
            "scene": dataset.scene,
            "iterations": int(opt.iterations),
            "total_training_time_sec": total_training_time,
            "train_views": len(scene.getTrainCameras()),
            "test_views": len(scene.getTestCameras()),
            "render_iterations": sorted(render_iterations),
            "analysis_iterations": sorted(analysis_iterations),
            "testing_iterations": sorted(testing_iterations),
            "saving_iterations": sorted(saving_iterations),
        },
    )


def prepare_output_and_logger(args):
    if not args.model_path:
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
        args.model_path = os.path.join("./output/", args.scene, date_time)

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    exp_logger = gen_log(args.model_path)

    return exp_logger


def training_report(
    exp_logger,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    render_iterations,
    analysis_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    args,
    prior_analysis_controller=None,
):
    if exp_logger and (iteration == 0 or (iteration + 1) % 100 == 0):
        exp_logger.info(
            f"Iter:{iteration}, L1 loss={Ll1.item():.4g}, Total loss={loss.item():.4g}, Time:{int(elapsed)}")

    if iteration in analysis_iterations:
        allocation_stats = None
        if prior_analysis_controller is not None:
            allocation_stats = prior_analysis_controller.allocation_statistics(scene.gaussians, iteration=iteration)
        export_gaussian_statistics(
            args.model_path,
            iteration,
            scene.gaussians,
            bins=args.analysis_bins,
            allocation_stats=allocation_stats,
        )
        exp_logger.info(f"[ITER {iteration}] Exported Gaussian statistics")

    if iteration in render_iterations:
        if len(scene.getTestCameras()) > 0:
            save_render_snapshot(
                args.model_path,
                iteration,
                "test",
                scene.getTestCameras(),
                scene.gaussians,
                renderFunc,
                renderArgs,
                max_views=args.render_num_views,
            )
        if len(scene.getTrainCameras()) > 0:
            save_render_snapshot(
                args.model_path,
                iteration,
                "train",
                scene.getTrainCameras(),
                scene.gaussians,
                renderFunc,
                renderArgs,
                max_views=args.render_num_views,
            )
        exp_logger.info(f"[ITER {iteration}] Saved render snapshots")

    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0 and config['name'] == 'test':
                psnr_test = 0.0
                ssim_test = 0.0
                start = time.time()
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    image_backnorm = (viewpoint.max_value - viewpoint.min_value) * image + viewpoint.min_value

                    image = image.mean(dim=0, keepdim=True)
                    image_backnorm = image_backnorm.mean(dim=0, keepdim=True)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_image_norm = viewpoint.normalized_image.to("cuda")

                    ssim_test += ssim(image_backnorm, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image_norm).mean().double()

                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])

                end = time.time()
                fps = len(config['cameras']) / (end - start)
                exp_logger.info(f"Testing Speed: {len(config['cameras']) / (end - start)} fps")
                exp_logger.info(f"Testing Time: {end - start} s")
                exp_logger.info(
                    "\n[ITER {}] Evaluating {}: SSIM = {}, PSNR = {}".format(iteration, config['name'], ssim_test,
                                                                             psnr_test))
                append_jsonl(
                    os.path.join(args.model_path, "metrics_history.jsonl"),
                    {
                        "iteration": int(iteration),
                        "split": config["name"],
                        "ssim": float(ssim_test.item()),
                        "psnr": float(psnr_test.item()),
                        "fps": float(fps),
                        "eval_time_sec": float(end - start),
                        "num_views": int(len(config["cameras"])),
                        "num_gaussians": int(scene.gaussians.get_xyz.shape[0]),
                    },
                )

        if exp_logger:
            exp_logger.info(f'Iter:{iteration}, total_points:{scene.gaussians.get_xyz.shape[0]}')
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)  #
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='config/chest.yaml', help='Path to the configuration file')
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[100, 2_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20_000, ])
    parser.add_argument("--render_iterations", nargs="+", type=int, default=[100, 2_000, 20_000])
    parser.add_argument("--analysis_iterations", nargs="+", type=int, default=[100, 2_000, 20_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--gpu_id", default="0", help="gpu to use")
    args = parser.parse_args(sys.argv[1:])

    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        setattr(args, key, value)

    args.test_iterations = sorted(set(args.test_iterations))
    args.render_iterations = sorted(set(args.render_iterations))
    args.analysis_iterations = sorted(set(args.analysis_iterations))
    args.save_iterations = sorted(set(list(args.save_iterations) + [args.iterations]))

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.render_iterations,
        args.analysis_iterations,
    )

    print("\nTraining complete.")
