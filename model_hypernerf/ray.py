import numpy as np


def sample_along_rays(key, origins, directions, num_coarse_samples, near, far,
                      use_stratified_sampling, use_linear_disparity):
    """Stratified sampling along the rays.

    Args:
        key: jnp.ndarray, random generator key.
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
    batch_size = origins.shape[0]

    t_vals = np.linspace(0., 1., num_coarse_samples)
    if not use_linear_disparity:
        z_vals = near * (1. - t_vals) + far * t_vals
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    if use_stratified_sampling:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = np.concatenate([mids, z_vals[..., -1:]], -1)
        lower = np.concatenate([z_vals[..., :1], mids], -1)
        t_rand = np.random.uniform(key, [batch_size, num_coarse_samples])
        z_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast z_vals to make the returned shape consistent.
        z_vals = np.broadcast_to(z_vals[None, ...],
                                 [batch_size, num_coarse_samples])

    return (z_vals, (origins[..., None, :] + z_vals[..., :, None] * directions[..., None, :]))
