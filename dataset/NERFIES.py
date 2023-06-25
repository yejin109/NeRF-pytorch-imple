import os
import cv2
import json
import numpy as np
from glob import glob
from ._dataset import Dataset
from itertools import permutations
from ._camera import NerfiesCamera


class NerfiesDataSet(Dataset):
    def __init__(self, data_type, dataset, image_scale, shuffle_pixels, test_camera_trajectory,
                 use_appearance_id, use_camera_id, **kwargs):
        super(NerfiesDataSet, self).__init__(data_type, dataset)
        self.train_ids, self.val_ids = _load_dataset_ids(self.data_dir)
        self.use_appearance_id = use_appearance_id
        self.use_camera_id = use_camera_id
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
    def num_camera_embeddings(self):
        if self.use_camera_id:
            return max(self.camera_id) + 1
        else:
            return 1


    @property
    def appearance_id(self):
        if not self.use_appearance_id:
            return tuple()
        return tuple(
            sorted(set([self.get_appearance_id(i) for i in self.train_ids])))

    @property
    def camera_id(self):
        if not self.use_appearance_id:
            return tuple()
        return tuple(
            sorted(set([self.get_camera_id(i) for i in self.train_ids])))

    @property
    def warp_id(self):
        if not self.use_appearance_id:
            return tuple()
        return tuple(
            sorted(set([self.get_warp_id(i) for i in self.train_ids])))

    @property
    def time_id(self):
        if not self.use_appearance_id:
            return tuple()
        return tuple(
            sorted(set([self.get_time_id(i) for i in self.train_ids])))

    def get_rgb_path(self, item_id):
        return os.path.join(self.rgb_dir, f'{item_id}.png')

    def load_rgb(self, item_id):
        return _load_image(os.path.join(self.rgb_dir, f'{item_id}.png'))

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
        with (self.data_dir / 'points.npy').open('rb') as f:
            points = np.load(f)
        points = (points - self.scene_center) * self.scene_scale
        points = points.astype(np.float32)
        if shuffle:
            shuffled_inds = list(permutations(range(len(points))))
            points = points[shuffled_inds]
        return points

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


def _load_image(path):
    with open(path, 'rb') as f:
        raw_im = np.asarray(bytearray(f.read()), dtype=np.uint8)
        image = cv2.imdecode(raw_im, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR -> RGB
        image = np.asarray(image).astype(np.float32) / 255.0
    return image


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

