import os
import cv2
import json
import math
import torch
import numpy as np
from glob import glob
from ._dataset import Dataset
from itertools import permutations
from ._camera import NerfiesCamera
from collections import defaultdict
from functionals import log_internal


class NerfiesDataSet(Dataset):
    def __init__(self, data_type, dataset, image_scale, shuffle_pixels, test_camera_trajectory,
                 use_appearance_id, use_camera_id, use_warp_id, use_time, use_depth, **kwargs):
        super(NerfiesDataSet, self).__init__(data_type, dataset)
        self.train_ids, self.val_ids = _load_dataset_ids(self.data_dir)
        self.use_appearance_id = use_appearance_id
        self.use_camera_id = use_camera_id
        self.use_warp_id = use_warp_id
        self.use_time = use_time
        self.use_depth = use_depth
        self.scene_center, self.scene_scale, self._near, self._far = load_scene_info(self.data_dir)
        self.test_camera_trajectory = test_camera_trajectory

        self.image_scale = image_scale
        self.shuffle_pixels = shuffle_pixels

        self.rgb_dir = os.path.join(self.data_dir, 'rgb', f'{image_scale}x')
        self.depth_dir = os.path.join(self.data_dir, 'depth', f'{image_scale}x')
        self.camera_dir = os.path.join(self.data_dir, 'camera')

        metadata_path = f"{self.data_dir}/metadata.json"
        self.metadata_dict = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata_dict = json.load(f)

        self.n_images = None
        self.W = None
        self.H = None
        log_internal("[Dataset] Loaded")

    @classmethod
    def get_dataset(cls, dataset_cfg, model_cfg):
        _dset_cfg = dataset_cfg
        _dset_cfg['use_warp_id'] = model_cfg['warp']['use_warp']
        _dset_cfg['use_time'] = model_cfg['warp_metadata_encoder_type'] == 'time'
        return NerfiesDataSet(**_dset_cfg)

    def prepare_data(self):
        """

        Returns:
            origins(rays_o)
            directions(rays_d)
            pixels
            metadata
        """
        # 현재는 index : dictionary pair인 것.
        data_list = [self.get_item(idx) for idx in self.train_ids]
        data_dict = defaultdict(list)
        metadata_dict = defaultdict(dict)

        for idx, dat in enumerate(data_list):
            camera_parmas = dat.pop('camera_params')
            camera = NerfiesCamera(**camera_parmas)
            pixels = camera.get_pixel_centers()
            directions = camera.pixels_to_rays(pixels)
            origins = np.broadcast_to(camera.position[None, None, :], directions.shape)

            data_dict['rays_o'].append(origins)
            data_dict['rays_d'].append(directions)
            data_dict['rgb'].append(dat['rgb'])
            data_dict['pixels'].append(pixels)

            for k, v in dat['metadata'].items():
                metadata_dict[k][idx] = torch.full((dat['rgb'].shape[0], dat['rgb'].shape[1], 1), fill_value=v)

        # TODO: shuffling, size check
        for k, v in data_dict.items():
            data_dict[k] = torch.Tensor(v)

        for k, v in metadata_dict.items():
            metadata_dict[k] = torch.stack(tuple(v.values()))

        self.n_images = len(data_list)
        self.W = data_list[0]['rgb'].shape[0]
        self.H = data_list[0]['rgb'].shape[1]

        return data_dict, metadata_dict

    def get_item(self, item_id, scale_factor=1.0):
        """Load an example as a data dictionary.

        Args:
          item_id: the ID of the item to fetch.warp_ids
          scale_factor: a scale factor to apply to the camera.

        Returns:
          A dictionary containing one of more of the following items:
            `rgb`: the RGB pixel values of each ray.
            `rays_dir`: the direction of each ray.
            `rays_origin`: the origin of each ray.
            `rays_pixels`: the pixel center of each ray.
            `metadata`: a dictionary of containing various metadata arrays. Each
              item is an array containing metadata IDs for each ray.
        """
        rgb = self.load_imgs(os.path.join(self.rgb_dir, f'{item_id}.png'))
        if scale_factor != 1.0:
            rgb = rescale_image(rgb, scale_factor)

        camera = self.load_camera(item_id, scale_factor)
        data = {
            'camera_params': camera.get_parameters(),
            'rgb': rgb,
            'metadata': {},
        }

        if self.use_appearance_id:
            data['metadata']['appearance'] = (self.appearance_id.index(self.get_appearance_id(item_id)))
        if self.use_camera_id:
            data['metadata']['camera'] = (self.camera_id.index(self.get_camera_id(item_id)))
        if self.use_warp_id:
            data['metadata']['warp'] = self.warp_id.index(self.get_warp_id(item_id))
        if self.use_time:
            data['metadata']['time'] = self.get_time(item_id)

        if self.use_depth:
            raise NotImplementedError('Do not support use depth configuration for now!')
            # depth = self.load_depth(item_id)
            # if depth is not None:
            #     if scale_factor != 1.0:
            #         depth = image_utils.rescale_image(depth, scale_factor)
            # data['depth'] = depth[..., np.newaxis]

        return data

    @property
    def near(self):
        return self._near

    @property
    def far(self):
        return self._far

    @property
    def num_appearance_embeddings(self):
        if self.use_appearance_id:
            return max(self.appearance_id) + 1
        else:
            return 1

    @property
    def num_warp_embeddings(self):
        if self.use_warp_id:
            return max(self.warp_id) + 1
        else:
            return 1

    @property
    def num_camera_embeddings(self):
        if self.use_camera_id:
            return max(self.camera_id) + 1
        else:
            return 1

    def get_appearance_id(self, item_id):
        return self.metadata_dict[item_id]['appearance_id']

    def get_camera_id(self, item_id):
        return self.metadata_dict[item_id]['camera_id']

    def get_warp_id(self, item_id):
        return self.metadata_dict[item_id]['warp_id']

    def get_time_id(self, item_id):
        if 'time_id' in self.metadata_dict[item_id]:
            return self.metadata_dict[item_id]['time_id']
        else:
            # Fallback for older datasets.
            return self.metadata_dict[item_id]['warp_id']

    @property
    def appearance_id(self):
        if not self.use_appearance_id:
            return tuple()
        return tuple(sorted(set([self.get_appearance_id(i) for i in self.train_ids])))

    @property
    def camera_id(self):
        if not self.use_camera_id:
            return tuple()
        return tuple(sorted(set([self.get_camera_id(i) for i in self.train_ids])))

    @property
    def warp_id(self):
        if not self.use_warp_id:
            return tuple()
        return tuple(sorted(set([self.get_warp_id(i) for i in self.train_ids])))

    @property
    def time_id(self):
        if not self.use_time:
            return tuple()
        return tuple(sorted(set([self.get_time_id(i) for i in self.train_ids])))

    def load_imgs(self, path):
        with open(path, 'rb') as f:
            raw_im = np.asarray(bytearray(f.read()), dtype=np.uint8)
            image = cv2.imdecode(raw_im, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR -> RGB
            image = np.asarray(image).astype(np.float32) / 255.0
        return image

    def load_camera(self, item_id, scale_factor=1.0):
        return load_camera(os.path.join(self.camera_dir, f"{item_id}.json"),
                           scale_factor=scale_factor / self.image_scale,
                           scene_center=self.scene_center,
                           scene_scale=self.scene_scale)

    @staticmethod
    def glob_cameras(path):
        return sorted(glob(f'{path}/*.json'))

    def load_test_cameras(self, count=None):
        camera_dir = os.path.join(self.data_dir, 'camera-paths', self.test_camera_trajectory)
        if not os.path.exists(camera_dir):
            return []
        camera_paths = sorted(glob(f'{camera_dir}/*.json'))
        if count is not None:
            stride = max(1, len(camera_paths) // count)
            camera_paths = camera_paths[::stride]
        cameras = [self.load_camera(path) for path in camera_paths]
        return cameras

    def load_points(self, shuffle=False):
        with open(os.path.join(self.data_dir, 'points.npy'), 'rb') as f:
            points = np.load(f)
        points = (points - self.scene_center) * self.scene_scale
        points = points.astype(np.float32)
        if shuffle:
            shuffled_inds = list(permutations(range(len(points))))
            points = points[shuffled_inds]
        return points

    def get_time(self, item_id):
        max_time = max(self.time_id)
        return (self.get_time_id(item_id) / max_time) * 2.0 - 1.0


def load_scene_info(data_dir):
    """
        Loads the scene scale from scene_scale.npy.

        Args:
        data_dir: the path to the dataset.

        Returns:
        scene_center: the center of the scene (unscaled coordinates).
        scene_scale: the scale of the scene.
        near: the near plane of the scene (scaled coordinates).
        far: the far plane of the scene (scaled coordinates).

        Raises:
        ValueError if scene_scale.npy does not exist.
    """
    scene_json_path = os.path.join(data_dir, 'scene.json')
    with open(scene_json_path, 'r') as f:
        scene_json = json.load(f)

    scene_center = np.array(scene_json['center'])
    scene_scale = scene_json['scale']
    near = scene_json['near']
    far = scene_json['far']

    return scene_center, scene_scale, near, far



def _load_dataset_ids(data_dir):
    """Loads dataset IDs."""
    dataset_json_path = os.path.join(data_dir, 'dataset.json')
    with open(dataset_json_path, 'r') as f:
        dataset_json = json.load(f)
        train_ids = dataset_json['train_ids']
        val_ids = dataset_json['val_ids']

    train_ids = [str(i) for i in train_ids]
    val_ids = [str(i) for i in val_ids]

    return train_ids, val_ids


def load_camera(camera_path,
                scale_factor=1.0,
                scene_center=None,
                scene_scale=None):
    """Loads camera and rays defined by the center pixels of a camera.

    Args:
    camera_path: a path to a camera file.
    scale_factor: a factor to scale the camera image by.
    scene_center: the center of the scene where the camera will be centered to.
    scene_scale: the scale of the scene by which the camera will also be scaled
      by.

    Returns:
    A Camera instance.
    """
    camera = NerfiesCamera.from_json(camera_path)

    if scale_factor != 1.0:
        camera = camera.scale(scale_factor)

    if scene_center is not None:
        camera.position = camera.position - scene_center
    if scene_scale is not None:
        camera.position = camera.position * scene_scale

    return camera


def reshape_image(image, shape):
    """Reshapes the image to the given shape."""
    out_height, out_width = shape
    return cv2.resize(image, (out_width, out_height), interpolation=cv2.INTER_AREA)


def rescale_image(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """Resize an image by a scale factor, using integer resizing if possible."""
    scale_factor = float(scale_factor)
    if scale_factor <= 0.0:
        raise ValueError('scale_factor must be a non-negative number.')

    if scale_factor == 1.0:
        return image

    height, width = image.shape[:2]
    if scale_factor.is_integer():
        return upsample_image(image, int(scale_factor))

    inv_scale = 1.0 / scale_factor
    if (inv_scale.is_integer() and (scale_factor * height).is_integer() and (scale_factor * width).is_integer()):
        return downsample_image(image, int(inv_scale))

    height, width = image.shape[:2]
    out_height = math.ceil(height * scale_factor)
    out_height -= out_height % 2
    out_width = math.ceil(width * scale_factor)
    out_width -= out_width % 2

    return reshape_image(image, (out_height, out_width))


def upsample_image(image: np.ndarray, scale: int) -> np.ndarray:
    """Upsamples the image by an integer factor."""
    if scale == 1:
        return image

    height, width = image.shape[:2]
    out_height, out_width = height * scale, width * scale
    resized = cv2.resize(image, (out_width, out_height), cv2.INTER_AREA)
    return resized


def downsample_image(image: np.ndarray, scale: int) -> np.ndarray:
    """Downsamples the image by an integer factor to prevent artifacts."""
    if scale == 1:
        return image

    height, width = image.shape[:2]
    if height % scale > 0 or width % scale > 0:
        raise ValueError(f'Image shape ({height},{width}) must be divisible by the scale ({scale}).')
    out_height, out_width = height // scale, width // scale
    resized = cv2.resize(image, (out_width, out_height), cv2.INTER_AREA)
    return resized
