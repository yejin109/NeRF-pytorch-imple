import torch
import torch.nn as nn
from model_nerfies._utils import noise_regularize


def render_samples(mlp, 
                   points_embed, trunk_conditions, alpha_conditions, rgb_conditions,
                   z_vals, rays_d, return_weights,
                   use_white_background, sample_at_infinity, noise_std, use_stratified_sampling,
                   ):
    """
        mlp : coarse일 수도, fine 일 수도
    """
    raw = mlp(points_embed, trunk_conditions, alpha_conditions, rgb_conditions)
    raw = noise_regularize(raw, noise_std, use_stratified_sampling)
    rgb = nn.functional.sigmoid(raw['rgb'])
    # TODO: inappropriate naming
    sigma = nn.functional.relu(raw['alpha'].squeeze(-1))

    out = volumetric_rendering(rgb, sigma, z_vals, rays_d, use_white_background, sample_at_infinity, return_weights)
    return out


def volumetric_rendering(rgb, sigma, z_vals, dirs, use_white_background, sample_at_infinity, return_weights, eps=1e-10):
    """
        Volumetric Rendering Function.

        Args:
            rgb: an array of size (B,S,3) containing the RGB color values.
            sigma: an array of size (B,S,1) containing the densities.
            z_vals: an array of size (B,S) containing the z-coordinate of the samples.
            dirs: an array of size (B,3) containing the directions of rays.
            use_white_background: whether to assume a white background or not.
            sample_at_infinity: if True adds a sample at infinity.
            return_weights: if True returns the weights in the dictionary.
            eps: a small number to prevent numerical issues.

        Returns:
            A dictionary containing:
            rgb: an array of size (B,3) containing the rendered colors.
            depth: an array of size (B,) containing the rendered depth.
            acc: an array of size (B,) containing the accumulated density.
            weights: an array of size (B,S) containing the weight of each sample.
    """
    last_sample_z = 1e10 if sample_at_infinity else 1e-19
    last_sample_z = torch.Tensor([last_sample_z])
    dists = torch.concat([z_vals[..., 1:] - z_vals[..., :-1], torch.broadcast_to(last_sample_z, z_vals[..., :1].size())], dim=-1)
    dists = dists * torch.linalg.norm(dirs[..., None, :], dim=-1)
    alpha = 1.0 - torch.exp(-sigma * dists)
    # Prepend a 1.0 to make this an 'exclusive' cumprod as in `tf.math.cumprod`.
    accum_prod = torch.concat([torch.ones_like(alpha[..., :1]), torch.cumprod(1.0 - alpha[..., :-1] + eps, dim=-1)], dim=-1)
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
    }
    if return_weights:
        out['weights'] = weights
    
    return out


def compute_depth_map(weights, z_vals, depth_threshold=0.5):
    """
        Compute the depth using the median accumulation.

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
    return torch.sum(opaqueness_mask * z_vals, dim=-1)


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
    cumulative_contribution = torch.cumsum(weights, dim=-1)
    depth_threshold = torch.Tensor([depth_threshold])
    opaqueness = cumulative_contribution >= depth_threshold
    false_padding = torch.zeros_like(opaqueness[..., :1])
    padded_opaqueness = torch.concat([false_padding, opaqueness[..., :-1]], dim=-1)
    opaqueness_mask = torch.logical_xor(opaqueness, padded_opaqueness)
    opaqueness_mask = opaqueness_mask.to(dtype=weights.dtype)
    return opaqueness_mask
