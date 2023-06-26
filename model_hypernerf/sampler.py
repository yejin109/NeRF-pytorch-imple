import torch


def sample_along_rays(
        rays_o, rays_d, 
        num_coarse_samples: int, 
        near: float, far: float, 
        use_stratified_sampling, use_linear_disparity):
    """
        Stratified sampling along the rays.

        Args:
            origins: ray origins.
            directions: ray directions.
            num_coarse_samples: int.
            near: float, near clip.
            far: float, far clip.
            use_stratified_sampling: use stratified sampling.
            use_linear_disparity: sampling linearly in disparity rather than depth.

        Returns:
            z_vals: jnp.ndarray, [batch_size, num_coarse_samples], sampled z values.
            points: jnp.ndarray, [batch_size, num_coarse_samples, 3], sampled points.
    """
    batch_size = rays_o.shape[0]

    t_vals = torch.linspace(0., 1., num_coarse_samples)

    if not use_linear_disparity:
        z_vals = near * (1. - t_vals) + far * t_vals
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)


    if use_stratified_sampling:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.concat([mids, z_vals[..., -1:]], -1)
        lower = torch.concat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand([batch_size, num_coarse_samples])
        z_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast z_vals to make the returned shape consistent.
        # z_vals = jnp.broadcast_to(z_vals[None, ...],
        #                         [batch_size, num_coarse_samples])
        z_vals = torch.reshape(z_vals, (batch_size, num_coarse_samples))

    points = rays_o + z_vals * rays_d
    return z_vals, points


def sample_pdf(key, bins, weights, origins, directions, z_vals,
               num_coarse_samples, use_stratified_sampling):
    """Hierarchical sampling.

    Args:
        key: jnp.ndarray(float32), [2,], random number generator.
        bins: jnp.ndarray(float32), [batch_size, n_bins + 1].
        weights: jnp.ndarray(float32), [batch_size, n_bins].
        origins: ray origins.
        directions: ray directions.
        z_vals: jnp.ndarray(float32), [batch_size, n_coarse_samples].
        num_coarse_samples: int, the number of samples.
        use_stratified_sampling: bool, use use_stratified_sampling samples.

    Returns:
        z_vals: jnp.ndarray(float32),
        [batch_size, n_coarse_samples + num_fine_samples].
        points: jnp.ndarray(float32),
        [batch_size, n_coarse_samples + num_fine_samples, 3].
    """
    z_samples = piecewise_constant_pdf(key, bins, weights, num_coarse_samples,
                                        use_stratified_sampling)
    # Compute united z_vals and sample points
    z_vals = torch.sort(torch.concatenate([z_vals, z_samples], dim=-1), axis=-1)
    return z_vals, (origins[..., None, :] + z_vals[..., None] * directions[..., None, :])


def piecewise_constant_pdf(key, bins, weights, num_coarse_samples,
                           use_stratified_sampling):
    """Piecewise-Constant PDF sampling.

    Args:
    key: jnp.ndarray(float32), [2,], random number generator.
    bins: jnp.ndarray(float32), [batch_size, n_bins + 1].
    weights: jnp.ndarray(float32), [batch_size, n_bins].
    num_coarse_samples: int, the number of samples.
    use_stratified_sampling: bool, use use_stratified_sampling samples.

    Returns:
    z_samples: jnp.ndarray(float32), [batch_size, num_coarse_samples].
    """
    eps = 1e-5

    # Get pdf
    weights += eps  # prevent nans
    pdf = weights / weights.sum(axis=-1, keepdims=True)
    cdf = torch.cumsum(pdf, axis=-1)
    cdf = torch.concat([torch.zeros(list(cdf.shape[:-1]) + [1]), cdf], dim=-1)

    # Take uniform samples
    if use_stratified_sampling:
        u = torch.rand(list(cdf.size()[:-1]) + [num_coarse_samples])
    else:
        u = torch.linspace(0., 1., num_coarse_samples)
        u = torch.reshape(u, list(cdf.size()[:-1]) + [num_coarse_samples])

    # Invert CDF. This takes advantage of the fact that `bins` is sorted.
    mask = (u[..., None, :] >= cdf[..., :, None])

    bins_g0, bins_g1 = minmax(bins, mask)
    cdf_g0, cdf_g1 = minmax(cdf, mask)

    denom = (cdf_g1 - cdf_g0)
    denom = torch.where(denom < eps, 1., denom)
    t = (u - cdf_g0) / denom
    z_samples = bins_g0 + t * (bins_g1 - bins_g0)

    # TODO Prevent gradient from backprop-ing through samples
    # torch.stop_gradient(z_samples)
    return z_samples

def minmax(x, mask):
    x0 = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
    x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)
    x0 = torch.minimum(x0, x[..., -2:-1])
    x1 = torch.maximum(x1, x[..., 1:2])
    return x0, x1