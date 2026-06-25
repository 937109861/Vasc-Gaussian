import torch
import torch.nn.functional as F


class ProjectionSupportController:
    """Estimate Gaussian support from foreground regions in training projections."""

    def __init__(
        self,
        cameras,
        foreground_threshold=0.02,
        mask_dilation=2,
        min_support_ratio=0.20,
        min_observations=8,
        densify_floor=0.25,
        prune_max_effective_density=0.03,
        max_views=16,
        chunk_size=100000,
    ):
        if not cameras:
            raise ValueError("ProjectionSupportController requires training cameras")
        if foreground_threshold < 0.0:
            raise ValueError("foreground_threshold must be non-negative")
        if mask_dilation < 0:
            raise ValueError("mask_dilation must be non-negative")
        if not 0.0 <= min_support_ratio <= 1.0:
            raise ValueError("min_support_ratio must be in [0, 1]")
        if min_observations < 1:
            raise ValueError("min_observations must be positive")
        if not 0.0 <= densify_floor <= 1.0:
            raise ValueError("densify_floor must be in [0, 1]")
        if prune_max_effective_density < 0.0:
            raise ValueError("prune_max_effective_density must be non-negative")

        ordered_cameras = sorted(
            cameras,
            key=lambda camera: float(getattr(camera, "angle", 0.0)),
        )
        max_views = int(max_views)
        if max_views > 0 and len(ordered_cameras) > max_views:
            indices = torch.linspace(0, len(ordered_cameras) - 1, steps=max_views)
            indices = torch.round(indices).to(dtype=torch.long).tolist()
            self.cameras = [ordered_cameras[index] for index in indices]
        else:
            self.cameras = ordered_cameras

        self.foreground_threshold = float(foreground_threshold)
        self.mask_dilation = int(mask_dilation)
        self.min_support_ratio = float(min_support_ratio)
        self.min_observations = min(int(min_observations), len(self.cameras))
        self.densify_floor = float(densify_floor)
        self.prune_max_effective_density = float(prune_max_effective_density)
        self.chunk_size = max(1, int(chunk_size))

    def _foreground_mask(self, camera):
        mask = (
            camera.normalized_image >= self.foreground_threshold
        ).to(dtype=torch.float32)
        if self.mask_dilation > 0:
            kernel_size = 2 * self.mask_dilation + 1
            mask = F.max_pool2d(
                mask.unsqueeze(0),
                kernel_size=kernel_size,
                stride=1,
                padding=self.mask_dilation,
            )[0]
        return mask[0] > 0.5

    @staticmethod
    def _project_chunk(points, camera):
        ones = torch.ones(
            (points.shape[0], 1),
            dtype=points.dtype,
            device=points.device,
        )
        points_hom = torch.cat((points, ones), dim=1)
        clip = points_hom @ camera.full_proj_transform
        view = points_hom @ camera.world_view_transform
        ndc = clip[:, :3] / (clip[:, 3:4] + 1e-7)

        width = int(camera.image_width)
        height = int(camera.image_height)
        pixel_x = ((ndc[:, 0] + 1.0) * width - 1.0) * 0.5
        pixel_y = ((ndc[:, 1] + 1.0) * height - 1.0) * 0.5
        observed = (
            (view[:, 2] > 0.2)
            & (pixel_x >= 0.0)
            & (pixel_x < width)
            & (pixel_y >= 0.0)
            & (pixel_y < height)
        )
        return pixel_x, pixel_y, observed

    def evaluate(self, gaussians):
        xyz = gaussians.get_xyz.detach()
        count = int(xyz.shape[0])
        hits = torch.zeros(count, dtype=torch.float32, device=xyz.device)
        observations = torch.zeros_like(hits)

        with torch.no_grad():
            for camera in self.cameras:
                foreground_mask = self._foreground_mask(camera)
                for start in range(0, count, self.chunk_size):
                    end = min(start + self.chunk_size, count)
                    pixel_x, pixel_y, observed = self._project_chunk(
                        xyz[start:end],
                        camera,
                    )
                    observations[start:end].add_(observed.to(dtype=observations.dtype))
                    x_index = torch.round(pixel_x).to(dtype=torch.long)
                    y_index = torch.round(pixel_y).to(dtype=torch.long)
                    x_index = torch.clamp(x_index, 0, int(camera.image_width) - 1)
                    y_index = torch.clamp(y_index, 0, int(camera.image_height) - 1)
                    supported = observed & foreground_mask[y_index, x_index]
                    hits[start:end].add_(supported.to(dtype=hits.dtype))

            support_ratio = hits / torch.clamp(observations, min=1.0)
            sufficiently_observed = observations >= float(self.min_observations)
            low_support = sufficiently_observed & (
                support_ratio < self.min_support_ratio
            )

            effective_density = (
                gaussians.get_opacity.detach() * gaussians.get_radiodensity.detach()
            ).reshape(-1)
            prune_mask = low_support & (
                effective_density < self.prune_max_effective_density
            )
            densify_multipliers = self.densify_floor + (
                1.0 - self.densify_floor
            ) * support_ratio
            densify_multipliers = torch.where(
                sufficiently_observed,
                densify_multipliers,
                torch.ones_like(densify_multipliers),
            )

        stats = {
            "projection_support_num_views": int(len(self.cameras)),
            "projection_support_mean_ratio": float(support_ratio.mean().item()),
            "projection_support_median_ratio": float(support_ratio.median().item()),
            "projection_support_mean_observations": float(observations.mean().item()),
            "projection_support_low_ratio": float(low_support.float().mean().item()),
            "projection_support_prune_count": int(prune_mask.sum().item()),
            "projection_support_prune_ratio": float(prune_mask.float().mean().item()),
            "projection_support_min_ratio": self.min_support_ratio,
            "projection_support_min_observations": int(self.min_observations),
            "projection_support_densify_floor": self.densify_floor,
            "projection_support_prune_max_effective_density": (
                self.prune_max_effective_density
            ),
        }
        return densify_multipliers, prune_mask, stats
