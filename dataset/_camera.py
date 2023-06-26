import json
from typing import Tuple, Union, Optional

import numpy as np


class NerfiesCamera:
    """Class to handle camera geometry."""

    def __init__(self,
               orientation: np.ndarray,
               position: np.ndarray,
               focal_length: Union[np.ndarray, float],
               principal_point: np.ndarray,
               image_size: np.ndarray,
               skew: Union[np.ndarray, float] = 0.0,
               pixel_aspect_ratio: Union[np.ndarray, float] = 1.0,
               radial_distortion: Optional[np.ndarray] = None,
               tangential_distortion: Optional[np.ndarray] = None,
               dtype=np.float32):
        """Constructor for camera class."""
        if radial_distortion is None:
            radial_distortion = np.array([0.0, 0.0, 0.0], dtype)
        if tangential_distortion is None:
            tangential_distortion = np.array([0.0, 0.0], dtype)

        self.orientation = np.array(orientation, dtype)
        self.position = np.array(position, dtype)
        self.focal_length = np.array(focal_length, dtype)
        self.principal_point = np.array(principal_point, dtype)
        self.skew = np.array(skew, dtype)
        self.pixel_aspect_ratio = np.array(pixel_aspect_ratio, dtype)
        self.radial_distortion = np.array(radial_distortion, dtype)
        self.tangential_distortion = np.array(tangential_distortion, dtype)
        self.image_size = np.array(image_size, np.uint32)
        self.dtype = dtype

    @classmethod
    def from_json(cls, path):
        """Loads a JSON camera into memory."""
        with open(path, 'r') as fp:
            camera_json = json.load(fp)

        # Fix old camera JSON.
        if 'tangential' in camera_json:
            camera_json['tangential_distortion'] = camera_json['tangential']

        return cls(
            orientation=np.asarray(camera_json['orientation']),
            position=np.asarray(camera_json['position']),
            focal_length=camera_json['focal_length'],
            principal_point=np.asarray(camera_json['principal_point']),
            skew=camera_json['skew'],
            pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
            radial_distortion=np.asarray(camera_json['radial_distortion']),
            tangential_distortion=np.asarray(camera_json['tangential_distortion']),
            image_size=np.asarray(camera_json['image_size']),
        )

    def get_pixel_centers(self):
        """Returns the pixel centers."""
        xx, yy = np.meshgrid(np.arange(self.image_size[0], dtype=self.dtype), np.arange(self.image_size[1], dtype=self.dtype))
        return np.stack([xx, yy], axis=-1) + 0.5

    def pixels_to_rays(self, pixels: np.ndarray):
        """Returns the rays for the provided pixels.

        Args:
          pixels: [A1, ..., An, 2] tensor or np.array containing 2d pixel positions.

        Returns:
            An array containing the normalized ray directions in world coordinates.
        """
        if pixels.shape[-1] != 2:
            raise ValueError('The last dimension of pixels must be 2.')
        if pixels.dtype != self.dtype:
            raise ValueError(f'pixels dtype ({pixels.dtype!r}) must match camera '
                             f'dtype ({self.dtype!r})')

        batch_shape = pixels.shape[:-1]
        pixels = np.reshape(pixels, (-1, 2))

        local_rays_dir = self.pixel_to_local_rays(pixels)
        rays_dir = np.matmul(self.orientation.T, local_rays_dir[..., np.newaxis])
        rays_dir = np.squeeze(rays_dir, axis=-1)

        # Normalize rays.
        rays_dir /= np.linalg.norm(rays_dir, axis=-1, keepdims=True)
        rays_dir = rays_dir.reshape((*batch_shape, 3))
        return rays_dir

    def pixel_to_local_rays(self, pixels: np.ndarray):
        """Returns the local ray directions for the provided pixels."""
        y = ((pixels[..., 1] - self.principal_point[1]) / (self.focal_length * self.pixel_aspect_ratio) )
        x = ((pixels[..., 0] - self.principal_point[0] - y * self.skew) / self.focal_length)

        if any(self.radial_distortion != 0.0) or any(self.tangential_distortion != 0.0):
            x, y = _radial_and_tangential_undistort(
                x,
                y,
                k1=self.radial_distortion[0],
                k2=self.radial_distortion[1],
                k3=self.radial_distortion[2],
                p1=self.tangential_distortion[0],
                p2=self.tangential_distortion[1])

        dirs = np.stack([x, y, np.ones_like(x)], axis=-1)
        return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    def scale(self, scale: float):
        """Scales the camera."""
        if scale <= 0:
            raise ValueError('scale needs to be positive.')

        new_camera = NerfiesCamera(
            orientation=self.orientation.copy(),
            position=self.position.copy(),
            focal_length=self.focal_length * scale,
            principal_point=self.principal_point.copy() * scale,
            skew=self.skew,
            pixel_aspect_ratio=self.pixel_aspect_ratio,
            radial_distortion=self.radial_distortion.copy(),
            tangential_distortion=self.tangential_distortion.copy(),
            image_size=np.array((int(round(self.image_size[0] * scale)),
                                 int(round(self.image_size[1] * scale)))),
        )
        return new_camera

    def get_parameters(self):
        return {
            'orientation': self.orientation,
            'position': self.position,
            'focal_length': self.focal_length,
            'principal_point': self.principal_point,
            'skew': self.skew,
            'pixel_aspect_ratio': self.pixel_aspect_ratio,
            'radial_distortion': self.radial_distortion,
            'tangential_distortion': self.tangential_distortion,
            'image_size': self.image_size,
        }

def _compute_residual_and_jacobian(
        x: np.ndarray,
        y: np.ndarray,
        xd: np.ndarray,
        yd: np.ndarray,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        p1: float = 0.0,
        p2: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray]:
    """Auxiliary function of radial_and_tangential_undistort()."""
    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + k3 * r))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = (k1 + r * (2.0 * k2 + 3.0 * k3 * r))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float = 0,
    k2: float = 0,
    k3: float = 0,
    p1: float = 0,
    p2: float = 0,
    eps: float = 1e-9,
    max_iterations=10) -> Tuple[np.ndarray, np.ndarray]:
    """Computes undistorted (x, y) from (xd, yd)."""
    # Initialize from the distorted point.
    x = xd.copy()
    y = yd.copy()

    step_x, step_y = None, None
    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = np.where(
            np.abs(denominator) > eps, x_numerator / denominator,
            np.zeros_like(denominator))
        step_y = np.where(
            np.abs(denominator) > eps, y_numerator / denominator,
            np.zeros_like(denominator))

    x = x + step_x
    y = y + step_y

    return x, y