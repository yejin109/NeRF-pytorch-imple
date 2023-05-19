# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions/classes for model definition."""
from typing import Optional


# class TrainState:
#   """Stores training state, including the optimizer and model params."""
#   optimizer: optim.Optimizer
#   nerf_alpha: Optional[jnp.ndarray] = None
#   warp_alpha: Optional[jnp.ndarray] = None
#   hyper_alpha: Optional[jnp.ndarray] = None
#   hyper_sheet_alpha: Optional[jnp.ndarray] = None
#
#   @property
#   def extra_params(self):
#     return {
#         'nerf_alpha': self.nerf_alpha,
#         'warp_alpha': self.warp_alpha,
#         'hyper_alpha': self.hyper_alpha,
#         'hyper_sheet_alpha': self.hyper_sheet_alpha,
#     }


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
  cdf = jnp.cumsum(pdf, axis=-1)
  cdf = jnp.concatenate([jnp.zeros(list(cdf.shape[:-1]) + [1]), cdf], axis=-1)

  # Take uniform samples
  if use_stratified_sampling:
    u = random.uniform(key, list(cdf.shape[:-1]) + [num_coarse_samples])
  else:
    u = jnp.linspace(0., 1., num_coarse_samples)
    u = jnp.broadcast_to(u, list(cdf.shape[:-1]) + [num_coarse_samples])

  # Invert CDF. This takes advantage of the fact that `bins` is sorted.
  mask = (u[..., None, :] >= cdf[..., :, None])

  def minmax(x):
    x0 = jnp.max(jnp.where(mask, x[..., None], x[..., :1, None]), -2)
    x1 = jnp.min(jnp.where(~mask, x[..., None], x[..., -1:, None]), -2)
    x0 = jnp.minimum(x0, x[..., -2:-1])
    x1 = jnp.maximum(x1, x[..., 1:2])
    return x0, x1

  bins_g0, bins_g1 = minmax(bins)
  cdf_g0, cdf_g1 = minmax(cdf)

  denom = (cdf_g1 - cdf_g0)
  denom = jnp.where(denom < eps, 1., denom)
  t = (u - cdf_g0) / denom
  z_samples = bins_g0 + t * (bins_g1 - bins_g0)

  # Prevent gradient from backprop-ing through samples
  return lax.stop_gradient(z_samples)


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
  z_vals = jnp.sort(jnp.concatenate([z_vals, z_samples], axis=-1), axis=-1)
  return z_vals, (
      origins[..., None, :] + z_vals[..., None] * directions[..., None, :])




def compute_depth_index(weights, depth_threshold=0.5):
  """Compute the sample index of the median depth accumulation."""
  opaqueness_mask = compute_opaqueness_mask(weights, depth_threshold)
  return jnp.argmax(opaqueness_mask, axis=-1)


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
  return jnp.sum(opaqueness_mask * z_vals, axis=-1)


def noise_regularize(key, raw, noise_std, use_stratified_sampling):
  """Regularize the density prediction by adding gaussian noise.

  Args:
    key: jnp.ndarray(float32), [2,], random number generator.
    raw: jnp.ndarray(float32), [batch_size, num_coarse_samples, 4].
    noise_std: float, std dev of noise added to regularize sigma output.
    use_stratified_sampling: add noise only if use_stratified_sampling is True.

  Returns:
    raw: jnp.ndarray(float32), [batch_size, num_coarse_samples, 4], updated raw.
  """
  if (noise_std is not None) and noise_std > 0.0 and use_stratified_sampling:
    unused_key, key = random.split(key)
    noise = random.normal(key, raw[..., 3:4].shape, dtype=raw.dtype) * noise_std
    raw = jnp.concatenate([raw[..., :3], raw[..., 3:4] + noise], axis=-1)
  return raw


def broadcast_feature_to(array: jnp.ndarray, shape: jnp.shape):
  """Matches the shape dimensions (everything except the channel dims).

  This is useful when you watch to match the shape of two features that have
  a different number of channels.

  Args:
    array: the array to broadcast.
    shape: the shape to broadcast the tensor to.

  Returns:
    The broadcasted tensor.
  """
  out_shape = (*shape[:-1], array.shape[-1])
  return jnp.broadcast_to(array, out_shape)


def metadata_like(rays, metadata_id):
  """Create a metadata array like a ray batch."""
  return jnp.full_like(rays[..., :1], fill_value=metadata_id, dtype=jnp.uint32)


def vmap_module(module, in_axes=0, out_axes=0, num_batch_dims=1):
  """Vectorize a module.

  Args:
    module: the module to vectorize.
    in_axes: the `in_axes` argument passed to vmap. See `jax.vmap`.
    out_axes: the `out_axes` argument passed to vmap. See `jax.vmap`.
    num_batch_dims: the number of batch dimensions (how many times to apply vmap
      to the module).

  Returns:
    A vectorized module.
  """
  for _ in range(num_batch_dims):
    module = nn.vmap(
        module,
        variable_axes={'params': None},
        split_rngs={'params': False},
        in_axes=in_axes,
        out_axes=out_axes)

  return module


def identity_initializer(_, shape):
  max_shape = max(shape)
  return jnp.eye(max_shape)[:shape[0], :shape[1]]


def posenc(x, min_deg, max_deg, use_identity=False, alpha=None):
    """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1]."""
    batch_shape = x.shape[:-1]
    scales = 2.0 ** jnp.arange(min_deg, max_deg)
    # (*, F, C).
    xb = x[..., None, :] * scales[:, None]
    # (*, F, 2, C).
    four_feat = jnp.sin(jnp.stack([xb, xb + 0.5 * jnp.pi], axis=-2))

    if alpha is not None:
        window = posenc_window(min_deg, max_deg, alpha)
        four_feat = window[..., None, None] * four_feat

    # (*, 2*F*C).
    four_feat = four_feat.reshape((*batch_shape, -1))

    if use_identity:
        return jnp.concatenate([x, four_feat], axis=-1)
    else:
        return four_feat


def posenc_window(min_deg, max_deg, alpha):
  """Windows a posenc using a cosiney window.

  This is equivalent to taking a truncated Hann window and sliding it to the
  right along the frequency spectrum.

  Args:
    min_deg: the lower frequency band.
    max_deg: the upper frequency band.
    alpha: will ease in each frequency as alpha goes from 0.0 to num_freqs.

  Returns:
    A 1-d numpy array with num_sample elements containing the window.
  """
  bands = jnp.arange(min_deg, max_deg)
  x = jnp.clip(alpha - bands, 0.0, 1.0)
  return 0.5 * (1 + jnp.cos(jnp.pi * x + jnp.pi))
