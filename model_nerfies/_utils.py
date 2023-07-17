import torch
import torch.nn as nn


def init_activation(act_str):
    if act_str == 'None':
        return None
    elif act_str == 'relu':
        return nn.ReLU()
    elif act_str == 'identity':
        return nn.Identity()

    else:
        raise KeyError(f'Cannot support {act_str} activation')


def get_value(val):
    if val == 'None':
        return None
    return val


def to_cpu(tensor):
    return tensor.detach().cpu().numpy()


def noise_regularize(raw, noise_std, use_stratified_sampling):
    """
    Regularize the density prediction by adding gaussian noise.

    Args:
        key: jnp.ndarray(float32), [2,], random number generator.
        raw: jnp.ndarray(float32), [batch_size, num_coarse_samples, 4].
        noise_std: float, std dev of noise added to regularize sigma output.
        use_stratified_sampling: add noise only if use_stratified_sampling is True.

    Returns:
        raw: jnp.ndarray(float32), [batch_size, num_coarse_samples, 4], updated raw.
    """
    if (noise_std is not None) and noise_std > 0.0 and use_stratified_sampling:
        noise = torch.rand(raw[..., 3:4].shape) * noise_std
        raw = torch.concat([raw[..., :3], raw[..., 3:4] + noise], dim=-1)
    return raw


def exponential_se3(S, theta):
    """
        Exponential map from Lie algebra so3 to Lie group SO3.

        Modern Robotics Eqn 3.88.

        Args:
            S: (6,) A screw axis of motion.
            theta: Magnitude of motion.

        Returns:
            a_X_b: (4, 4) The homogeneous transformation matrix attained by integrating
            motion of magnitude theta about S for one second.
    """
    theta = theta[..., None, None]
    w, v = torch.split(S, [3, 3], dim=-1) # w : # (Batch size, sample size, 3), v : # (Batch size, sample size, 3)
    W = skew(w) # (Batch size, sample size, 3, 3)
    R = exp_so3(w, theta) # (Batch size, sample size, 3, 3)
    p = (theta * torch.eye(3)[None, None, ...] + (1.0 - torch.cos(theta)) * W + (theta - torch.sin(theta)) * W @ W) @ v[..., None] # w : # (Batch size, sample size, 3, 1)
    return rp_to_se3(R, p)


def exp_so3(w, theta: float):
    """
        Exponential map from Lie algebra so3 to Lie group SO3.

        Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.

        Args:
        w: (3,) An axis of rotation.
        theta: An angle of rotation.

        Returns:
        R: (3, 3) An orthonormal rotation matrix representing a rotation of
            magnitude theta about axis w.
    """
    W = skew(w)
    return torch.eye(3)[None, None, :] + torch.sin(theta) * W + (1.0 - torch.cos(theta)) * W @ W


def skew(_w):
    """
        Build a skew matrix ("cross product matrix") for vector w.

        Modern Robotics Eqn 3.30.

        Args:
            w: (3,) A 3-vector

        Returns:
            W: (3, 3) A skew matrix such that W @ v == w x v
    """
    if len(_w.shape) == 2:
        _w = _w.unsqueeze(1)
    w = _w.clone().detach()
    res = torch.zeros(list(tuple(w.size()[:2]))+[3, 3])

    res[:, :, 0, 1] = -w[:, :, 2]
    res[:, :, 0, 2] = w[:, :, 1]
    res[:, :, 1, 0] = -w[:, :, 2]
    res[:, :, 1, 2] = w[:, :, 0]
    res[:, :, 2, 0] = -w[:, :, 1]
    res[:, :, 2, 1] = w[:, :, 0]
    return res


def rp_to_se3(R, p):
    """
        Rotation and translation to homogeneous transform.

        Args:
        R: (Batch size, sample size, 3, 3) An orthonormal rotation matrix.
        p: (Batch size, sample size,  3, 1) A 3-vector representing an offset.

        Returns:
        X: (4, 4) The homogeneous transformation matrix described by rotating by R
            and translating by p.
    """
    # p = torch.reshape(p, (3, 1))
    RT = torch.concat((R, p), dim=-1) # (Batch size, sample size, 3, 4)
    homogeneous = torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).repeat((R.size(0), R.size(1), 1, 1)) # (Batch size, sample size, 1, 4)
    return torch.concat((RT, homogeneous), dim=-2)


def from_homogenous(v):
    return v[..., :3] / v[..., -1:]


def to_homogenous(v):
    return torch.concat([v, torch.ones_like(v[..., :1])], dim=-1)


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


def general_loss_with_squared_residual(squared_x, alpha, scale, eps=1e-6):
    r"""The general loss that takes a squared residual.

    This fuses the sqrt operation done to compute many residuals while preserving
    the square in the loss formulation.

    This implements the rho(x, \alpha, c) function described in "A General and
    Adaptive Robust Loss Function", Jonathan T. Barron,
    https://arxiv.org/abs/1701.03077.

    Args:
    squared_x: The residual for which the loss is being computed. x can have
      any shape, and alpha and scale will be broadcasted to match x's shape if
      necessary.
    alpha: The shape parameter of the loss (\alpha in the paper), where more
      negative values produce a loss with more robust behavior (outliers "cost"
      less), and more positive values produce a loss with less robust behavior
      (outliers are penalized more heavily). Alpha can be any value in
      [-infinity, infinity], but the gradient of the loss with respect to alpha
      is 0 at -infinity, infinity, 0, and 2. Varying alpha allows for smooth
      interpolation between several discrete robust losses:
        alpha=-Infinity: Welsch/Leclerc Loss.
        alpha=-2: Geman-McClure loss.
        alpha=0: Cauchy/Lortentzian loss.
        alpha=1: Charbonnier/pseudo-Huber loss.
        alpha=2: L2 loss.
    scale: The scale parameter of the loss. When |x| < scale, the loss is an
      L2-like quadratic bowl, and when |x| > scale the loss function takes on a
      different shape according to alpha.

    Returns:
    The losses for each element of x, in the same shape as x.
    """

    # This will be used repeatedly.
    squared_scaled_x = squared_x / (scale ** 2)

    # The loss when alpha == 2.
    loss_two = 0.5 * squared_scaled_x
    # The loss when alpha == 0.
    loss_zero = torch.log1p(torch.min(0.5 * squared_scaled_x, torch.tensor([3e37])))
    # The loss when alpha == -infinity.
    loss_neginf = -torch.expm1(-0.5 * squared_scaled_x)
    # The loss when alpha == +infinity.
    loss_posinf = torch.expm1(torch.min(0.5 * squared_scaled_x, torch.Tensor([87.5])))

    # The loss when not in one of the above special cases.
    # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
    beta_safe = torch.max(torch.Tensor([eps]), torch.abs(torch.Tensor([alpha - 2.])))
    # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
    alpha_safe = torch.where(
      torch.greater_equal(torch.Tensor([alpha]), torch.Tensor([0.])), torch.ones_like(torch.Tensor([alpha])),
      -torch.ones_like(torch.Tensor([alpha]))) * torch.max(torch.Tensor([eps]), torch.abs(torch.Tensor([alpha])))
    loss_otherwise = (beta_safe / alpha_safe) * (
            torch.pow(squared_scaled_x / beta_safe + 1., 0.5 * alpha) - 1.)

    # Select which of the cases of the loss to return.
    if alpha == -torch.inf:
        loss =  loss_neginf
    elif alpha == 0 :
        loss = loss_zero
    elif alpha == 2:
        loss = loss_two
    else:
        loss = loss_otherwise
    # loss = torch.where(
    #   alpha == -torch.inf, loss_neginf,
    #   torch.where(
    #       alpha == 0, loss_zero,
    #       torch.where(
    #           alpha == 2, loss_two,
    #           torch.where(alpha == torch.inf, loss_posinf, loss_otherwise)
    #       )
    #   ))

    return scale * loss
