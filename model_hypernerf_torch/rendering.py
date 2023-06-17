import torch
import torch.nn as nn
import numpy as np
from model_hypernerf_torch._utils import noise_regularize, posenc


def render_samples(mlp, 
                   points, points_encoder_args, 
                   alpha_conditions, rgb_conditions,
                   z_vals, rays_d, return_weights,
                   use_white_background, sample_at_infinity, render_opts,
                   noise_std, use_stratified_sampling,
                   ):
    """
        mlp : coarse일 수도, fine 일 수도
    """
    points_embed = posenc(points[:, :3], **points_encoder_args)
    if points.size(-1)> 3:
        hyper_embed = posenc(points[:, 3:], **points_encoder_args)
        points_embed = torch.concat((points_embed, hyper_embed), dim=-1)
    
    raw = mlp(points_embed, alpha_conditions, rgb_conditions)
    raw = noise_regularize(raw, noise_std, use_stratified_sampling)
    rgb = nn.sigmoid(raw['rgb'])

    sigma = nn.relu(raw['alpha'].squeeze(-1))

    sigma = filter_sigma(points, sigma, render_opts)
    out = volumetric_rendering(rgb, sigma, z_vals, rays_d, use_white_background, sample_at_infinity, return_weights)
    return out


def volumetric_rendering(rgb,
                         sigma,
                         z_vals,
                         dirs,
                         use_white_background,
                         sample_at_infinity=True,
                         eps=1e-10):
    """Volumetric Rendering Function.

    Args:
        rgb: an array of size (B,S,3) containing the RGB color values.
        sigma: an array of size (B,S) containing the densities.
        z_vals: an array of size (B,S) containing the z-coordinate of the samples.
        dirs: an array of size (B,3) containing the directions of rays.
        use_white_background: whether to assume a white background or not.
        sample_at_infinity: if True adds a sample at infinity.
        eps: a small number to prevent numerical issues.

    Returns:
        A dictionary containing:
          rgb: an array of size (B,3) containing the rendered colors.
          depth: an array of size (B,) containing the rendered depth.
          acc: an array of size (B,) containing the accumulated density.
          weights: an array of size (B,S) containing the weight of each sample.
    """
    # TODO(keunhong): remove this hack.
    last_sample_z = 1e10 if sample_at_infinity else 1e-19
    dists = np.concatenate([z_vals[..., 1:] - z_vals[..., :-1], np.broadcast_to([last_sample_z], z_vals[..., :1].shape)], -1)
    dists = dists * np.linalg.norm(dirs[..., None, :], axis=-1)
    alpha = 1.0 - np.exp(-sigma * dists)
    # Prepend a 1.0 to make this an 'exclusive' cumprod as in `tf.math.cumprod`.
    accum_prod = np.concatenate([np.ones_like(alpha[..., :1], alpha.dtype), np.cumprod(1.0 - alpha[..., :-1] + eps, axis=-1), ], axis=-1)
    weights = alpha * accum_prod

    rgb = (weights[..., None] * rgb).sum(axis=-2)
    exp_depth = (weights * z_vals).sum(axis=-1)
    med_depth = compute_depth_map(weights, z_vals)
    acc = weights.sum(axis=-1)
    if use_white_background:
        rgb = rgb + (1. - acc[..., None])

    if sample_at_infinity:
        acc = weights[..., :-1].sum(axis=-1)

    out = {
      'rgb': rgb,
      'depth': exp_depth,
      'med_depth': med_depth,
      'acc': acc,
      'weights': weights,
    }
    return out


def compute_depth_map(weights, z_vals, depth_threshold=0.5):
    """Compute the depth using the median accumulation.

    Note that this differs from the depth computation in NeRF-W's codebase!

    Args:
        weights: the density weights from NeRF.
        z_vals: the z coordinates of the samples.
        depth_threshold: the accumulation threshold which will be used as the depth
          termination point.

    Returns:
        A tensor containing the depth of each input pixel.
    """
    opaqueness_mask = compute_opaqueness_mask(weights, depth_threshold)
    return np.sum(opaqueness_mask * z_vals, axis=-1)


def compute_opaqueness_mask(weights, depth_threshold=0.5):
    """Computes a mask which will be 1.0 at the depth point.

  Args:
    weights: the density weights from NeRF.
    depth_threshold: the accumulation threshold which will be used as the depth
      termination point.

  Returns:
    A tensor containing a mask with the same size as weights that has one
      element long the sample dimension that is 1.0. This element is the point
      where the 'surface' is.
    """
    cumulative_contribution = np.cumsum(weights, axis=-1)
    depth_threshold = np.array(depth_threshold, dtype=weights.dtype)
    opaqueness = cumulative_contribution >= depth_threshold
    false_padding = np.zeros_like(opaqueness[..., :1])
    padded_opaqueness = np.concatenate([false_padding, opaqueness[..., :-1]], axis=-1)
    opaqueness_mask = np.logical_xor(opaqueness, padded_opaqueness)
    opaqueness_mask = opaqueness_mask.astype(weights.dtype)
    return opaqueness_mask


def filter_sigma(points, sigma, render_opts):
    """Filters the density based on various rendering arguments.

    - `dust_threshold` suppresses any sigma values below a threshold.
    - `bounding_box` suppresses any sigma values outside of a 3D bounding box.

    Args:
        points: the input points for each sample.
        sigma: the array of sigma values.
        render_opts: a dictionary containing any of the options listed above.

    Returns:
        A filtered sigma density field.
    """
    if render_opts is None:
        return sigma

    # Clamp densities below the set threshold.
    if 'dust_threshold' in render_opts:
        dust_thres = render_opts.get('dust_threshold', 0.0)
        sigma = (sigma >= dust_thres).astype(np.float32) * sigma
    if 'bounding_box' in render_opts:
        xmin, xmax, ymin, ymax, zmin, zmax = render_opts['bounding_box']
        render_mask = ((points[..., 0] >= xmin) & (points[..., 0] <= xmax)
                       & (points[..., 1] >= ymin) & (points[..., 1] <= ymax)
                       & (points[..., 2] >= zmin) & (points[..., 2] <= zmax))
        sigma = render_mask.astype(np.float32) * sigma

    return sigma

