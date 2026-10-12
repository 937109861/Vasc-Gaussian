from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):

        group = parser.add_argument_group(name)

        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None

            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for key, value in vars(self).items():
            target_key = key[1:] if key.startswith("_") else key
            setattr(group, target_key, value)
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                if arg[1] is not None:
                    setattr(group, arg[0], arg[1])
        return group



class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.scene = ""
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.dump_pickle = False
        self.interval = 8
        self.acui_jitter = 0.0
        self.train_num = 50
        self.add_num = 50
        self.nview = 5
        self.render_num_views = -1
        self.analysis_bins = 20
        self.init_mode = "acui"
        self.prior_source = "gt"
        self.prior_path = ""
        self.prior_threshold_mode = "quantile"
        self.prior_threshold_value = 0.98
        self.prior_max_points = 50000
        self.hybrid_prior_points = 25000
        self.hybrid_acui_points = 25000
        self.init_anisotropy_ratio = 1.0
        self.save_init_analysis = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.density_render_mode = "legacy"
        self.density_render_constant_color = False
        super().__init__(parser, "Pipeline Parameters", sentinel)



class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 60_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.radiodensity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_render = 1.0
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.radiodensity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.densify_min_opacity = 0.005
        self.density_aware_pruning = False
        self.prune_min_radiodensity = 0.02
        self.prune_density_mode = "opacity"
        self.prune_min_effective_density = 0.0
        self.prune_density_from_iter = 0
        self.random_background = False
        self.use_prior_refinement = False
        self.refinement_prior_path = ""
        self.refinement_from_iter = 500
        self.roi_densify_enabled = False
        self.roi_densify_weight = 2.0
        self.empty_prune_enabled = False
        self.empty_prune_opacity = 0.01
        self.empty_prune_mask_source = "roi"
        self.use_topology_gaussian_prior = False
        self.topology_prior_path = ""
        self.topology_prior_from_iter = 5000
        self.topology_centerline_from_iter = 8000
        self.topology_roi_source = "support_roi"
        self.topology_roi_densify_weight = 1.25
        self.topology_outside_densify_weight = 0.75
        self.topology_outside_grid_densify_weight = 0.50
        self.topology_empty_prune_enabled = False
        self.topology_empty_prune_from_iter = 10000
        self.topology_prune_only_until_iter = 30000
        self.topology_prune_only_interval = 400
        self.topology_empty_prune_mask_source = "vessel"
        self.topology_empty_prune_density_mode = "effective_density"
        self.topology_empty_prune_threshold = 0.002
        self.topology_centerline_enabled = False
        self.topology_centerline_protect_distance = 0.08
        self.topology_centerline_far_distance = 0.35
        self.topology_centerline_prune_threshold = 0.0015
        self.topology_empty_loss_from_iter = 10000
        self.lambda_topology_empty = 0.0
        self.topology_soft_empty_margin = 0.05
        self.topology_soft_empty_tau = 0.05
        self.topology_soft_empty_density_mode = "opacity"
        self.topology_component_small_voxels = 260
        self.use_edge_loss = False
        self.edge_loss_weight = 0.05
        self.edge_loss_mask_mode = "none"
        self.edge_mask_quantile = 0.80
        self.edge_mask_dilation = 1
        self.use_projection_support_loss = False
        self.lambda_projection_foreground = 0.0
        self.lambda_projection_background = 0.0
        self.projection_foreground_threshold = 0.02
        self.projection_mask_dilation = 2
        self.projection_background_margin = 0.01
        self.use_projection_support_refinement = False
        self.projection_support_from_iter = 4000
        self.projection_support_until_iter = 15000
        self.projection_support_interval = 400
        self.projection_support_max_views = 16
        self.projection_support_foreground_threshold = 0.02
        self.projection_support_mask_dilation = 2
        self.projection_support_min_ratio = 0.20
        self.projection_support_min_observations = 8
        self.projection_support_densify_floor = 0.25
        self.projection_support_prune_max_effective_density = 0.03
        self.projection_support_chunk_size = 100000
        self.use_paper_vessel_management = False
        self.paper_management_prior_source = "soft"
        self.paper_management_prior_align_corners = True
        self.paper_management_lambda_q = 1.0
        self.paper_management_lambda_def = 1.0
        self.paper_management_delta = 0.25
        self.paper_management_lambda_m = 1.0
        self.paper_management_lambda_eta = 1.0
        self.paper_management_lambda_k = 1.0
        self.paper_management_tau_keep = 0.05
        self.paper_management_tau_den = 0.00004
        self.paper_management_tau_s = 0.008
        self.paper_management_eps = 0.00000001
        self.paper_management_split_children = 2
        self.use_vessel_weighted_projection = False
        self.vessel_projection_weight = 0.0
        self.vessel_projection_weight_path = ""
        self.vessel_projection_normalization = "per_view_percentile"
        self.vessel_projection_debug_interval = 0
        self.vessel_projection_debug_views = 4
        self.use_vessel_projection_edge_loss = False
        self.lambda_vessel_projection_edge = 0.0
        self.vessel_projection_edge_weight = 1.0
        self.vessel_projection_edge_threshold = 0.15
        self.vessel_projection_edge_dilation = 1
        self.enable_dynamic_prior = False
        self.prior_update_iterations = []
        self.prior_update_mode = "ema"
        self.prior_fusion_beta = 0.25
        self.prior_update_source = "gaussian_volume"
        self.prior_update_density_mode = "effective_density"
        self.regenerate_prior_projection_cache = True
        self.save_prior_update_artifacts = True
        self.resume_dynamic_prior_state = True
        self.dynamic_prior_output_dir = "dynamic_prior"
        self.prior_update_splat_mode = "dgr"
        self.prior_update_splat_radius = 3
        self.prior_update_min_sigma_voxels = 0.40
        self.prior_update_max_sigma_voxels = 3.5
        self.prior_update_dgr_sigma_scale = 1.10
        self.prior_update_dgr_normalize_kernel = False
        self.prior_update_dgr_supersample = 1
        self.prior_update_dgr_kernel_cutoff = 0.0
        self.prior_update_dgr_max_splat_radius = 0
        self.prior_update_threshold = 0.5
        self.prior_update_roi_dilation = 2
        self.prior_update_points_source = "roi"
        self.prior_update_max_points = 50000
        self.prior_update_seed = 0
        self.prior_update_candidate_fusion_mode = "boost"
        self.prior_update_candidate_vessel_weight = 0.35
        self.prior_update_candidate_blend_weight = 0.20
        self.prior_update_vesselness_alpha = 0.5
        self.prior_update_vesselness_beta = 0.5
        self.prior_update_vesselness_gamma = 0.0
        self.prior_update_dark_tubes = False
        self.prior_update_vesselness_chunk_size = 500000
        self.use_occ_loss = False
        self.occ_grid_size = 32
        self.occ_loss_type = "l1"
        self.lambda_occ = 0.001
        self.occ_source = "roi"
        self.occ_warmup_from_iter = 500
        self.occ_debug_interval = 1000
        self.use_volume_loss = False
        self.volume_grid_size = 32
        self.volume_loss_type = "l1"
        self.lambda_volume = 0.0001
        self.volume_density_mode = "opacity"
        self.volume_splat_mode = "trilinear"
        self.volume_splat_radius = 2
        self.volume_min_sigma_voxels = 0.75
        self.volume_max_sigma_voxels = 3.0
        self.volume_dgr_sigma_scale = 1.0
        self.volume_dgr_normalize_kernel = False
        self.volume_dgr_supersample = 1
        self.volume_dgr_kernel_cutoff = 0.0
        self.volume_dgr_max_splat_radius = 0
        self.volume_loss_interval = 100
        self.volume_warmup_from_iter = 1000
        self.volume_use_prior_mask = False
        self.volume_mask_source = "roi"
        self.volume_roi_weight = 1.0
        self.volume_background_weight = 0.1
        self.volume_tissue_balance = False
        self.volume_sparse_foreground_balance = False
        self.volume_foreground_weight = 1.0
        self.volume_foreground_epsilon = 1e-6
        self.volume_soft_tissue_weight = 2.0
        self.volume_hard_tissue_weight = 1.0
        self.volume_soft_tissue_min_quantile = 0.05
        self.volume_soft_tissue_max_quantile = 0.75
        self.volume_hard_tissue_min_quantile = 0.90
        self.volume_background_suppression_weight = 0.0
        self.volume_background_suppression_threshold = 0.0
        self.volume_tv_weight = 0.0
        self.volume_debug_interval = 0
        self.use_gaussian_reg = False
        self.scale_floor = 0.75
        self.scale_max_axis = 0.0
        self.scale_aniso_max_ratio = 5.0
        self.scale_aniso_loss_mode = "linear"
        self.lambda_scale_floor = 0.0
        self.lambda_scale_max = 0.0
        self.lambda_scale_aniso = 0.0
        self.lambda_density_entropy = 0.0
        self.lambda_density_sparsity = 0.0
        self.density_sparsity_mode = "radiodensity"
        self.density_sparsity_from_iter = 0
        super().__init__(parser, "Optimization Parameters")




def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass

    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()

    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
