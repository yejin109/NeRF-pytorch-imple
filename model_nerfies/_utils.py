import torch
import torch.nn as nn


def init_activation(act_str):
    if act_str is None:
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
        raw = torch.concat([raw[..., :3], raw[..., 3:4] + noise], axis=-1)
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
    w, v = torch.split(S, 2)
    W = skew(w)
    R = exp_so3(w, theta)
    p = (theta * torch.eye(3) + (1.0 - torch.cos(theta)) * W +(theta - torch.sin(theta)) * W @ W) @ v
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
    return torch.eye(3) + torch.sin(theta) * W + (1.0 - torch.cos(theta)) * W @ W


def skew(w):
    """
        Build a skew matrix ("cross product matrix") for vector w.

        Modern Robotics Eqn 3.30.

        Args:
            w: (3,) A 3-vector

        Returns:
            W: (3, 3) A skew matrix such that W @ v == w x v
    """
    w = torch.reshape(w, (3))
    return torch.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])

def rp_to_se3(R, p):
    """
        Rotation and translation to homogeneous transform.

        Args:
        R: (3, 3) An orthonormal rotation matrix.
        p: (3,) A 3-vector representing an offset.

        Returns:
        X: (4, 4) The homogeneous transformation matrix described by rotating by R
            and translating by p.
    """
    p = torch.reshape(p, (3, 1))
    return torch.block_diag([[R, p], [torch([[0.0, 0.0, 0.0, 1.0]])]])

def from_homogenous(v):
    return v[..., :3] / v[..., -1:]


def to_homogenous(v):
    return torch.concat([v, torch.ones_like(v[..., :1])], dim=-1)
