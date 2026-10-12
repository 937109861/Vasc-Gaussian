import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from plyfile import PlyData
from tqdm import tqdm


scene = "chest"
ply_path = "output/chest_b/point_cloud/iteration_20000/point_cloud.ply"
output_dir = f"point_cloud_visualization/{scene}"

# Keep the original visual style, but filter obvious noise more carefully.
opacity_quantile = 0.82
scale_quantile = 0.95
radius_quantile = 0.995
max_points = 45000
point_size = 1.4
elevation = 2.0
proj_num = 120
fps = 45
crop_margin = 180
seed = 0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def set_axes_equal(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def load_point_cloud(path):
    plydata = PlyData.read(path)
    vertex = plydata.elements[0]
    xyz = np.stack(
        (
            np.asarray(vertex["x"]),
            np.asarray(vertex["y"]),
            np.asarray(vertex["z"]),
        ),
        axis=1,
    )
    opacity = sigmoid(np.asarray(vertex["opacity"]).reshape(-1))
    scales = np.stack(
        (
            np.asarray(vertex["scale_0"]),
            np.asarray(vertex["scale_1"]),
            np.asarray(vertex["scale_2"]),
        ),
        axis=1,
    )
    return xyz, opacity, np.exp(scales)


def filter_points(xyz, opacity, scales):
    scale_extent = scales.max(axis=1)
    centroid = np.average(xyz, axis=0, weights=np.clip(opacity, 1e-6, None))
    radius = np.linalg.norm(xyz - centroid[None, :], axis=1)

    opacity_threshold = np.quantile(opacity, opacity_quantile)
    scale_threshold = np.quantile(scale_extent, scale_quantile)
    radius_threshold = np.quantile(radius, radius_quantile)

    keep_mask = opacity >= opacity_threshold
    keep_mask &= scale_extent <= scale_threshold
    keep_mask &= radius <= radius_threshold

    if not np.any(keep_mask):
        raise RuntimeError("All points were filtered out. Lower the filtering quantiles.")

    filtered_points = xyz[keep_mask]
    filtered_opacity = opacity[keep_mask]
    filtered_radius = radius[keep_mask]

    if filtered_points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(filtered_points.shape[0], size=max_points, replace=False)
        filtered_points = filtered_points[sample_idx]
        filtered_opacity = filtered_opacity[sample_idx]
        filtered_radius = filtered_radius[sample_idx]

    return filtered_points, filtered_opacity, filtered_radius, keep_mask.sum(), xyz.shape[0]


def build_colors(points, opacity):
    z_norm = (points[:, 2] - points[:, 2].min()) / (np.ptp(points[:, 2]) + 1e-8)
    colors = plt.cm.bone(z_norm)
    opacity_norm = (opacity - opacity.min()) / (np.ptp(opacity) + 1e-8)
    colors[:, 3] = np.clip(0.2 + 0.8 * opacity_norm, 0.2, 1.0)
    return colors


def crop_image(image):
    if crop_margin <= 0:
        return image
    left = min(crop_margin, image.width // 4)
    top = min(crop_margin, image.height // 4)
    right = max(image.width - left, left + 1)
    bottom = max(image.height - top, top + 1)
    return image.crop((left, top, right, bottom))


def render_gif(points, opacity, save_path):
    os.makedirs(save_path, exist_ok=True)

    colors = build_colors(points, opacity)
    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=point_size, c=colors, depthshade=False)
    ax.axis("off")
    ax.invert_zaxis()
    set_axes_equal(ax, points)

    angle_interval = 360 / proj_num
    image_files = []
    for i in tqdm(range(proj_num), desc="Rendering frames"):
        angle = i * angle_interval
        ax.view_init(elev=elevation, azim=angle)
        frame_path = os.path.join(save_path, f"elev_{elevation}_azim_{angle:.1f}.png")
        plt.savefig(frame_path, dpi=280, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        image_files.append(frame_path)

    plt.close(fig)

    duration = 1000 / fps
    gif_filename = os.path.join(save_path, f"{scene}_fps_{fps}.gif")
    first = crop_image(Image.open(image_files[0]))
    gif_frames = [first]
    for filename in tqdm(image_files[1:], desc="Building gif"):
        gif_frames.append(crop_image(Image.open(filename)))

    gif_frames[0].save(
        gif_filename,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration,
        loop=0,
    )
    return gif_filename


def main():
    save_path = output_dir or os.path.join("point_cloud_visualization", scene)

    xyz, opacity, scales = load_point_cloud(ply_path)
    points, weights, _, kept_points, total_points = filter_points(xyz, opacity, scales)

    print(f"Loaded {total_points} gaussians from {ply_path}")
    print(f"Kept {kept_points} after filtering; rendering {points.shape[0]} points")
    print(
        "Filters:"
        f" opacity>={opacity_quantile:.2f} quantile,"
        f" scale<={scale_quantile:.2f} quantile,"
        f" radius<={radius_quantile:.2f} quantile"
    )

    gif_path = render_gif(points, weights, save_path)
    print(f"Saved visualization to {gif_path}")


if __name__ == "__main__":
    main()
