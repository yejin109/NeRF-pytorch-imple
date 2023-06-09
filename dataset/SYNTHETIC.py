"""
TODO

- skip: 만들어진 데이터셋이다 보니 사진이 많아서 학습할 때에는 다 쓰는데 validation이나 test에서는 조정을 하기도 한다.
e.g.
    if s=='train' or testskip==0:
        skip = 1
    else:
        skip = testskip
        
    for frame in meta['frames'][::skip]:
"""

import os
import cv2
import json
import torch
import numpy as np

from ._dataset import Dataset
from ._utils import imread


class SyntheticDataset(Dataset):
    def __init__(self, data_type, dataset, path_zflat, run_type=None, factor=None, bd_factor=None, **kwargs):
        super(SyntheticDataset, self).__init__(data_type, dataset, run_type, path_zflat, factor, bd_factor)
        # 1. Load pose matrix and all the attrs to be used
        self._poses, self.img_paths, field_of_view, idx = self.load_matrices()
        self.imgs = self.load_imgs()
        self.img_shape = self.imgs[0].shape

        # NOTE: focal length는 {1\over 2} {W\over \tan({\theta \over 2})} 로 계산된다.
        # 참고 : https://velog.io/@gjghks950/NeRF-%EA%B5%AC%ED%98%84-%ED%86%BA%EC%95%84%EB%B3%B4%EA%B8%B0-feat.-Camera-to-World-Transformation
        self.focal_length = 0.5 * self.img_shape[1] / np.tan(0.5 * field_of_view)

        assert (len(self.imgs) == len(self._poses))

        self._train_i, self._val_i, self._test_i = idx

        self._render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, kwargs['render_pose_num']+1)[:-1]], 0)

        self._near, self._far = torch.FloatTensor([2.]), torch.FloatTensor([6.])

        self.bds = None

        # 3. Dataset Specific
        # NOTE: durl
        if kwargs['half_res']:
            self.img_shape, self.focal_length, self.imgs = get_resize(self.img_shape, self.focal_length, self.imgs)

        if kwargs['white_bkgd']:
            self.imgs = self.imgs[..., :3] * self.imgs[..., -1:] + (1. - self.imgs[..., -1:])
        else:
            self.imgs = self.imgs[..., :3]

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
        w2cs = self._poses[:, :3, :4]
        return w2cs

    @Dataset.c2w.getter
    def c2w(self):
        return

    @Dataset.render_pose.getter
    def render_pose(self):
        """
        Relevent function : spherify and load_llff_data

        NOTE: llff data에서 Spherify_poses를 사용하거나 기존 루틴을 사용함
        """
        return self._render_poses.to(os.environ['DEVICE'])

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
        return self._near.to(os.environ['DEVICE'])

    @Dataset.far.getter
    def far(self):
        return self._far.to(os.environ['DEVICE'])

    def load_imgs(self):
        imgs = [imread(path) / 255. for path in self.img_paths]
        imgs = np.stack(imgs, 0).astype(np.float32)
        return imgs

    def load_matrices(self):
        """
        boundary 값을 따로 지정하지 않고 있음
        Field of view가 바뀌지 않는다고 가정
        이 데이터 셋은 (x,y,z) 그대로 들어오는 듯
        :return:
        """
        def enum(s, e):
            return list(range(s, e))
        idx = []
        lens = [0]
        c2ws, img_dir = [], []
        for run_type in ['train', 'val', 'test']:
            with open(f'{self.data_dir}/transforms_{run_type}.json') as f:
                raw = json.load(f)
            field_of_view = raw['camera_angle_x']

            frames = raw['frames']
            for frame in frames:
                c2ws.append(frame['transform_matrix'])
                img_dir.append('/'.join([self.data_dir]+frame['file_path'].split('/')[1:])+'.png')
            idx.append(enum(lens[-1], len(c2ws)))
            lens.append(len(c2ws))

        return np.array(c2ws), img_dir, field_of_view, tuple(idx)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()


rot_theta = lambda th : torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def get_resize(shape, focal, imgs):
    H, W = shape[0], shape[1]
    H = H // 2
    W = W // 2
    focal = focal / 2.

    imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
    for i, img in enumerate(imgs):
        imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    imgs = imgs_half_res
    return imgs[0].shape, focal, imgs
