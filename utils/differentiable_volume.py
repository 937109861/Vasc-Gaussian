import torch


def normalize_volume_grid(volume, eps=1e-6):
    volume_min = torch.amin(volume)
    volume_max = torch.amax(volume)
    denom = torch.clamp(volume_max - volume_min, min=eps)
    return (volume - volume_min) / denom


def quaternion_to_rotation(r):
    norm = torch.clamp(torch.linalg.norm(r, dim=1, keepdim=True), min=1e-8)
    q = r / norm
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rotation = torch.empty((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)

    rotation[:, 0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
    rotation[:, 0, 1] = 2.0 * (qx * qy - qw * qz)
    rotation[:, 0, 2] = 2.0 * (qx * qz + qw * qy)
    rotation[:, 1, 0] = 2.0 * (qx * qy + qw * qz)
    rotation[:, 1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
    rotation[:, 1, 2] = 2.0 * (qy * qz - qw * qx)
    rotation[:, 2, 0] = 2.0 * (qx * qz - qw * qy)
    rotation[:, 2, 1] = 2.0 * (qy * qz + qw * qx)
    rotation[:, 2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)
    return rotation


def gaussian_centers_to_trilinear_grid(
    xyz,
    weights,
    grid_min,
    grid_max,
    grid_size,
    normalize=True,
):
    """Softly splat Gaussian centers into a low-resolution 3D grid.

    The mapping is differentiable w.r.t. xyz and weights because each point is
    distributed to its 8 neighboring voxels with trilinear weights.
    """
    grid_size = int(grid_size)
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2 for trilinear splatting")

    device = xyz.device
    dtype = xyz.dtype
    weights = weights.reshape(-1).to(device=device, dtype=dtype)
    grid_min = torch.as_tensor(grid_min, dtype=dtype, device=device)
    grid_max = torch.as_tensor(grid_max, dtype=dtype, device=device)
    denom = torch.clamp(grid_max - grid_min, min=1e-6)

    normalized = (xyz - grid_min[None, :]) / denom[None, :]
    valid = torch.all((normalized >= 0.0) & (normalized <= 1.0), dim=1)
    coords = normalized * float(grid_size - 1)
    coords = torch.clamp(coords, 0.0, float(grid_size - 1))

    lower = torch.floor(coords)
    upper = torch.clamp(lower + 1.0, max=float(grid_size - 1))
    frac = coords - lower
    lower = lower.long()
    upper = upper.long()

    x0, y0, z0 = lower[:, 0], lower[:, 1], lower[:, 2]
    x1, y1, z1 = upper[:, 0], upper[:, 1], upper[:, 2]
    fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]

    wx0, wy0, wz0 = 1.0 - fx, 1.0 - fy, 1.0 - fz
    wx1, wy1, wz1 = fx, fy, fz
    weights = weights.clamp(min=0.0) * valid.to(dtype)

    flat = torch.zeros(grid_size ** 3, dtype=dtype, device=device)

    def add_corner(ix, iy, iz, corner_weight):
        flat_idx = ix * grid_size * grid_size + iy * grid_size + iz
        flat.scatter_add_(0, flat_idx, weights * corner_weight)

    add_corner(x0, y0, z0, wx0 * wy0 * wz0)
    add_corner(x0, y0, z1, wx0 * wy0 * wz1)
    add_corner(x0, y1, z0, wx0 * wy1 * wz0)
    add_corner(x0, y1, z1, wx0 * wy1 * wz1)
    add_corner(x1, y0, z0, wx1 * wy0 * wz0)
    add_corner(x1, y0, z1, wx1 * wy0 * wz1)
    add_corner(x1, y1, z0, wx1 * wy1 * wz0)
    add_corner(x1, y1, z1, wx1 * wy1 * wz1)

    grid = flat.reshape(grid_size, grid_size, grid_size)
    if normalize:
        grid = normalize_volume_grid(grid)
    return grid, valid


def gaussian_centers_to_scale_aware_grid(
    xyz,
    scales,
    weights,
    grid_min,
    grid_max,
    grid_size,
    radius=2,
    min_sigma_voxels=0.75,
    max_sigma_voxels=3.0,
    normalize=True,
):
    """Splat Gaussian centers into a low-res grid using scale-aware local kernels.

    This is scene-agnostic: the same implementation can be used for bones,
    vessels, organs, or any other structure. The anatomical behavior is
    controlled by config values such as radius and sigma clamps.
    """
    grid_size = int(grid_size)
    radius = int(radius)
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2 for gaussian splatting")
    if radius < 1:
        raise ValueError("radius must be at least 1 for gaussian splatting")

    device = xyz.device
    dtype = xyz.dtype
    weights = weights.reshape(-1).to(device=device, dtype=dtype).clamp(min=0.0)
    grid_min = torch.as_tensor(grid_min, dtype=dtype, device=device)
    grid_max = torch.as_tensor(grid_max, dtype=dtype, device=device)
    denom = torch.clamp(grid_max - grid_min, min=1e-6)

    normalized = (xyz - grid_min[None, :]) / denom[None, :]
    valid = torch.all((normalized >= 0.0) & (normalized <= 1.0), dim=1)
    coords = torch.clamp(normalized * float(grid_size - 1), 0.0, float(grid_size - 1))
    center_idx = torch.round(coords).long()

    voxel_size = denom / float(grid_size - 1)
    sigma = scales.to(device=device, dtype=dtype) / voxel_size[None, :]
    sigma = torch.clamp(sigma, min=float(min_sigma_voxels), max=float(max_sigma_voxels))

    offsets = range(-radius, radius + 1)
    kernel_sum = torch.zeros_like(weights)
    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                idx = center_idx + torch.tensor([dx, dy, dz], dtype=torch.long, device=device)[None, :]
                inside = torch.all((idx >= 0) & (idx < grid_size), dim=1) & valid
                grid_coord = idx.to(dtype)
                delta = (grid_coord - coords) / sigma
                kernel = torch.exp(-0.5 * torch.sum(delta * delta, dim=1)) * inside.to(dtype)
                kernel_sum = kernel_sum + kernel

    kernel_sum = torch.clamp(kernel_sum, min=1e-6)
    flat = torch.zeros(grid_size ** 3, dtype=dtype, device=device)
    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                idx = center_idx + torch.tensor([dx, dy, dz], dtype=torch.long, device=device)[None, :]
                inside = torch.all((idx >= 0) & (idx < grid_size), dim=1) & valid
                idx = torch.clamp(idx, 0, grid_size - 1)
                grid_coord = idx.to(dtype)
                delta = (grid_coord - coords) / sigma
                kernel = torch.exp(-0.5 * torch.sum(delta * delta, dim=1)) * inside.to(dtype)
                flat_idx = idx[:, 0] * grid_size * grid_size + idx[:, 1] * grid_size + idx[:, 2]
                flat.scatter_add_(0, flat_idx, weights * kernel / kernel_sum)

    grid = flat.reshape(grid_size, grid_size, grid_size)
    if normalize:
        grid = normalize_volume_grid(grid)
    return grid, valid


def gaussian_centers_to_dgr_grid(
    xyz,
    scales,
    rotations,
    weights,
    grid_min,
    grid_max,
    grid_size,
    radius=2,
    min_sigma_voxels=0.5,
    max_sigma_voxels=4.0,
    sigma_scale=1.0,
    normalize_kernel=False,
    supersample=1,
    kernel_cutoff=0.0,
    max_splat_radius=0,
    normalize=True,
):
    """Discretize anisotropic 3D Gaussians into an attenuation volume.

    Unlike center occupancy splatting, this treats each Gaussian as a local
    density kernel and evaluates it on nearby voxel centers. Rotation is used
    so elongated kernels can align with soft-tissue structures instead of only
    emphasizing point-like high-contrast regions.
    """
    grid_size = int(grid_size)
    radius = int(radius)
    supersample = int(supersample)
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2 for DGR volume splatting")
    if radius < 1:
        raise ValueError("radius must be at least 1 for DGR volume splatting")
    if supersample < 1:
        raise ValueError("supersample must be at least 1 for DGR volume splatting")

    device = xyz.device
    dtype = xyz.dtype
    weights = weights.reshape(-1).to(device=device, dtype=dtype).clamp(min=0.0)
    grid_min = torch.as_tensor(grid_min, dtype=dtype, device=device)
    grid_max = torch.as_tensor(grid_max, dtype=dtype, device=device)
    denom = torch.clamp(grid_max - grid_min, min=1e-6)
    voxel_size = denom / float(grid_size - 1)

    normalized = (xyz - grid_min[None, :]) / denom[None, :]
    valid = torch.all((normalized >= 0.0) & (normalized <= 1.0), dim=1)
    coords = torch.clamp(normalized * float(grid_size - 1), 0.0, float(grid_size - 1))
    center_idx = torch.round(coords).long()

    sigma_voxels = scales.to(device=device, dtype=dtype) / voxel_size[None, :]
    sigma_voxels = sigma_voxels * float(sigma_scale)
    sigma_voxels = torch.clamp(sigma_voxels, min=float(min_sigma_voxels), max=float(max_sigma_voxels))
    sigma_world = sigma_voxels * voxel_size[None, :]
    rotation = quaternion_to_rotation(rotations.to(device=device, dtype=dtype))

    if kernel_cutoff and float(kernel_cutoff) > 0.0:
        cutoff_radius = int(torch.ceil(torch.max(sigma_voxels).detach() * float(kernel_cutoff)).item())
        radius = max(radius, cutoff_radius)
        if max_splat_radius and int(max_splat_radius) > 0:
            radius = min(radius, int(max_splat_radius))

    offsets_1d = torch.arange(-radius, radius + 1, dtype=torch.long, device=device)
    offsets = torch.stack(torch.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij"), dim=-1).reshape(-1, 3)
    if supersample == 1:
        sub_offsets = torch.zeros((1, 3), dtype=dtype, device=device)
    else:
        sub_1d = (torch.arange(supersample, dtype=dtype, device=device) + 0.5) / float(supersample) - 0.5
        sub_offsets = torch.stack(torch.meshgrid(sub_1d, sub_1d, sub_1d, indexing="ij"), dim=-1).reshape(-1, 3)
    sub_offsets_world = sub_offsets * voxel_size[None, :]

    flat = torch.zeros(grid_size ** 3, dtype=dtype, device=device)
    kernel_sum = torch.zeros_like(weights) if normalize_kernel else None
    kernels = []
    flat_indices = []
    inside_masks = []

    for offset in offsets:
        idx = center_idx + offset[None, :]
        inside = torch.all((idx >= 0) & (idx < grid_size), dim=1) & valid
        idx_clamped = torch.clamp(idx, 0, grid_size - 1)
        voxel_world = grid_min[None, :] + idx_clamped.to(dtype) * voxel_size[None, :]
        kernel = torch.zeros_like(weights)
        for sub_offset_world in sub_offsets_world:
            diff_world = voxel_world + sub_offset_world[None, :] - xyz
            local = torch.bmm(rotation.transpose(1, 2), diff_world.unsqueeze(-1)).squeeze(-1)
            mahalanobis = torch.sum((local / torch.clamp(sigma_world, min=1e-6)) ** 2, dim=1)
            kernel = kernel + torch.exp(-0.5 * mahalanobis)
        kernel = (kernel / float(sub_offsets_world.shape[0])) * inside.to(dtype)
        flat_idx = idx_clamped[:, 0] * grid_size * grid_size + idx_clamped[:, 1] * grid_size + idx_clamped[:, 2]

        if normalize_kernel:
            kernel_sum = kernel_sum + kernel
            kernels.append(kernel)
            flat_indices.append(flat_idx)
            inside_masks.append(inside)
        else:
            flat.scatter_add_(0, flat_idx, weights * kernel)

    if normalize_kernel:
        kernel_sum = torch.clamp(kernel_sum, min=1e-6)
        for kernel, flat_idx, inside in zip(kernels, flat_indices, inside_masks):
            flat.scatter_add_(0, flat_idx, weights * kernel * inside.to(dtype) / kernel_sum)

    grid = flat.reshape(grid_size, grid_size, grid_size)
    if normalize:
        grid = normalize_volume_grid(grid)
    return grid, valid


def gaussian_centers_to_density_grid(
    gaussians,
    weights,
    grid_min,
    grid_max,
    grid_size,
    splat_mode="trilinear",
    splat_radius=2,
    min_sigma_voxels=0.75,
    max_sigma_voxels=3.0,
    dgr_sigma_scale=1.0,
    dgr_normalize_kernel=False,
    dgr_supersample=1,
    dgr_kernel_cutoff=0.0,
    dgr_max_splat_radius=0,
    normalize=True,
):
    if splat_mode == "trilinear":
        return gaussian_centers_to_trilinear_grid(
            gaussians.get_xyz,
            weights,
            grid_min,
            grid_max,
            grid_size,
            normalize=normalize,
        )
    if splat_mode == "gaussian":
        return gaussian_centers_to_scale_aware_grid(
            gaussians.get_xyz,
            gaussians.get_scaling,
            weights,
            grid_min,
            grid_max,
            grid_size,
            radius=splat_radius,
            min_sigma_voxels=min_sigma_voxels,
            max_sigma_voxels=max_sigma_voxels,
            normalize=normalize,
        )
    if splat_mode == "dgr":
        return gaussian_centers_to_dgr_grid(
            gaussians.get_xyz,
            gaussians.get_scaling,
            gaussians.get_rotation,
            weights,
            grid_min,
            grid_max,
            grid_size,
            radius=splat_radius,
            min_sigma_voxels=min_sigma_voxels,
            max_sigma_voxels=max_sigma_voxels,
            sigma_scale=dgr_sigma_scale,
            normalize_kernel=dgr_normalize_kernel,
            supersample=dgr_supersample,
            kernel_cutoff=dgr_kernel_cutoff,
            max_splat_radius=dgr_max_splat_radius,
            normalize=normalize,
        )
    raise ValueError(f"Unsupported volume splat mode: {splat_mode}")


def gaussian_density_weights(gaussians, mode="opacity"):
    opacity = gaussians.get_opacity.reshape(-1)
    if mode == "opacity":
        return opacity

    if mode in ("radiodensity", "attenuation"):
        if not hasattr(gaussians, "get_radiodensity"):
            return opacity
        return gaussians.get_radiodensity.reshape(-1)

    if mode in ("opacity_radiodensity", "effective_density"):
        if not hasattr(gaussians, "get_radiodensity"):
            return opacity
        return opacity * gaussians.get_radiodensity.reshape(-1)

    if mode in ("feature", "opacity_feature"):
        dc = gaussians._features_dc[:, 0, :].mean(dim=1)
        dc = torch.clamp(dc, min=0.0)
        dc = normalize_volume_grid(dc)
        if mode == "feature":
            return dc
        return opacity.reshape(-1) * (0.5 + 0.5 * dc)

    if mode == "opacity_feature_scale":
        dc = gaussians._features_dc[:, 0, :].mean(dim=1)
        dc = torch.clamp(dc, min=0.0)
        dc = normalize_volume_grid(dc)
        scales = gaussians.get_scaling
        scale_score = 1.0 / torch.clamp(torch.mean(scales, dim=1), min=1e-6)
        scale_score = normalize_volume_grid(scale_score)
        return opacity.reshape(-1) * (0.5 + 0.5 * dc) * (0.5 + 0.5 * scale_score)

    raise ValueError(f"Unsupported gaussian density weight mode: {mode}")
