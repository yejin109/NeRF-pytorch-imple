import os
import json
import torch
import numpy as np

from ._dataset import Dataset
from ._utils import imread, recenter_poses, render_path_spherical, render_path_spiral, get_test_idx, get_train_idx, get_val_idx, get_boundary, log, _minify, inverse_w2c


class SyntheticDataset(Dataset):
    def __init__(self, data_type, run_type, dataset, path_zflat, factor=None, bd_factor=None, **kwargs):
        super(SyntheticDataset, self).__init__(data_type, run_type, dataset, path_zflat, factor, bd_factor)
        # 0. Setup
        sfx = ""
        if self.factor is not None:
            sfx = '_{}'.format(self.factor)
        self.img_dir = os.path.join(self.data_dir, run_type)
        self.img_paths = [os.path.join(self.img_dir, f) for f in sorted(os.listdir(self.img_dir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

        # 1. Load Image
        self.imgs = self.load_imgs()
        self.img_shape = self.imgs[0].shape

        # 2. Load pose matrix and all the attrs to be used
        self._pose_inv, self.bds, self.focal_length = self.load_matrices()
        assert (len(self.imgs) == len(self._pose_inv))
        self._render_poses = None
        self._test_i = None
        self._val_i = None
        self._train_i = None

        # 3. Value Update
        if kwargs['recenter']:
            if int(os.environ['VERBOSE']):
                log("Recentered : _poses updated\n")
            self._poses = recenter_poses(self.w2c)

        if kwargs['spherify']:
            if int(os.environ['VERBOSE']):
                log("Render path \n\tSpherified : _poses, _reder_poses, bds updated\n")
            self._poses, self._render_poses, self.bds = render_path_spherical(self._poses, self.bds)
        else:
            if int(os.environ['VERBOSE']):
                log("Render path \n\tSpherified : _poses, _reder_poses, bds updated\n")

            self._render_poses = render_path_spiral(self._poses, self.bds, path_zflat)

        self._render_poses = np.array(self._render_poses).astype(np.float32)
        self._test_i = get_test_idx(self._poses, -1, self.imgs.shape)
        self._val_i = get_val_idx(self._test_i)
        self._train_i = get_train_idx(self._val_i, self._test_i, self.imgs.shape)
        self._near, self._far = torch.FloatTensor(2.), torch.FloatTensor(6.)

    def __len__(self):
        return len(self.imgs)

    @Dataset.intrinsic_matrix.getter
    def intrinsic_matrix(self):
        H, W, focal = self.hwf

        intrinsic = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])
        return intrinsic

    @Dataset.hwf.getter
    def hwf(self):
        return np.array(list(self.img_shape[:2])+[self.focal_length])

    @Dataset.w2c.getter
    def w2c(self):
        return inverse_w2c(self.c2w)

    @Dataset.c2w.getter
    def c2w(self):
        return self._pose_inv

    @Dataset.render_pose.getter
    def render_pose(self):
        """
        Relevent function : spherify and load_llff_data

        NOTE: llff data에서 Spherify_poses를 사용하거나 기존 루틴을 사용함
        """
        return torch.Tensor(self._render_poses).to(os.environ['DEVICE'])

    @Dataset.test_i.getter
    def test_i(self):
        return self._test_i

    @Dataset.train_i.getter
    def train_i(self):
        return self._train_i

    @Dataset.val_i.getter
    def val_i(self):
        return self._val_i

    @Dataset.near.getter
    def near(self):
        return self._near

    @Dataset.far.getter
    def far(self):
        return self._far

    def load_imgs(self):
        # TODO: Factor 반영하기
        imgs = [imread(path) / 255. for path in self.img_paths]
        imgs = np.stack(imgs, 0)
        return imgs

    def load_matrices(self):
        with open(f'{self.data_dir}/transforms_{self.run_type}.json') as f:
            raw = json.load(f)
        field_of_view = raw['camera_angle_x']
        # NOTE: focal length는 {1\over 2} {W\over \tan({\theta \over 2})} 로 계산된다.
        # 참고 : https://velog.io/@gjghks950/NeRF-%EA%B5%AC%ED%98%84-%ED%86%BA%EC%95%84%EB%B3%B4%EA%B8%B0-feat.-Camera-to-World-Transformation
        focal = 0.5 * self.img_shape[1] / np.tan(0.5*field_of_view)

        frames = raw['frames']
        c2ws = []
        for frame in frames:
            c2ws.append(frame['transform_matrix'])
        c2ws = np.array(c2ws)
        c2ws = np.concatenate([c2ws[:, 1:2, :], -c2ws[:, 0:1, :], c2ws[:, 2:, :]], 1)
        bds = [None] * len(frames)
        return c2ws, bds, focal
