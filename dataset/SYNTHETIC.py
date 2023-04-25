"""
TODO
- data loading : 기존 llff는 한 폴더에 다 있었는데 지금은 train,val,test 다 나눠져 있어서 for loop으로 불러오고 있음. 이것을 적용한 다음에 train val test에 대해서 다 load를 해야하도록 수정하기
- Support white_bkgd : imgs의 값에서 배경색의 색 조건에 따라서 다르게 적용해야함
e.g.
    if args.white_bkgd:
        images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    else:
        images = images[...,:3]

- half_res : 해당 데이터셋은 원본 이미지를 사용하는데 이걸 그대로 사용할지 말지에 따라서 값을 업데이트 해야함
e.g.
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
- skip: 만들어진 데이터셋이다 보니 사진이 많아서 학습할 때에는 다 쓰는데 validation이나 test에서는 조정을 하기도 한다.
e.g.
    if s=='train' or testskip==0:
        skip = 1
    else:
        skip = testskip
        
    for frame in meta['frames'][::skip]:
"""

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

        self._render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

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
        return self._near

    @Dataset.far.getter
    def far(self):
        return self._far

    def load_imgs(self):
        # TODO: Factor 반영하기
        imgs = [imread(path) / 255. for path in self.img_paths]
        imgs = np.stack(imgs, 0).astype(np.float32)
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
        # 이 데이터 셋은 (x,y,z) 그대로 들어오는 듯
        # c2ws = np.concatenate([c2ws[:, 1:2, :], -c2ws[:, 0:1, :], c2ws[:, 2:, :]], 1)

        # boundary값을 따로 지정하지 않고 있음
        bds = [None] * len(frames)
        return c2ws, bds, focal


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()