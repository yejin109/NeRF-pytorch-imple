import os
import json
import torch
import numpy as np

from ._dataset import Dataset
from ._utils import imread, recenter_poses, render_path_spherical, render_path_spiral, get_test_idx, get_train_idx, get_val_idx, get_boundary, log, _minify
from functionals import log_cfg, log_internal


class LLFFDataset(Dataset):
    def __init__(self, data_type, run_type, dataset, path_zflat, llffhold, factor=None, bd_factor=None, **kwargs):
        super(LLFFDataset, self).__init__(data_type, run_type, dataset, path_zflat, factor, bd_factor)
        # 0. Setup
        sfx = ""
        if self.factor is not None:
            sfx = '_{}'.format(self.factor)
        self.img_dir = os.path.join(self.data_dir, 'images'+sfx)
        self.img_paths = [os.path.join(self.img_dir, f) for f in sorted(os.listdir(self.img_dir)) if
                          f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

        # 1. Load Image
        self.imgs = self.load_imgs()
        self.img_shape = self.imgs[0].shape

        # 2. Load pose matrix and all the attrs to be used
        self._poses, self.bds, self.focal_length = self.load_matrices()
        # assert (len(self.imgs) == len(self._poses))
        self._render_poses = None
        self._test_i = None
        self._val_i = None
        self._train_i = None

        # 3. Dataset Specific
        if kwargs['recenter']:
            if int(os.environ['VERBOSE']):
                log("Recentered : _poses updated\n")
            self._poses = recenter_poses(self._poses)

        if kwargs['spherify']:
            if int(os.environ['VERBOSE']): 
                log("Render path \n\tSpherified : _poses, _reder_poses, bds updated\n")
            self._poses, self._render_poses, self.bds = render_path_spherical(self._poses, self.bds, kwargs['render_pose_num'])
        else:
            if int(os.environ['VERBOSE']): 
                log("Render path \n\tSpherified : _poses, _reder_poses, bds updated\n")

            self._render_poses = render_path_spiral(self._poses, self.bds, path_zflat, N_rots=kwargs['N_rots'], render_pose_num=kwargs['render_pose_num'], zrate=kwargs['zrate'])

        self._render_poses = np.array(self._render_poses).astype(np.float32)
        self._test_i = get_test_idx(self._poses, llffhold, self.imgs.shape)
        self._val_i = get_val_idx(self._test_i)
        self._train_i = get_train_idx(self._val_i, self._test_i, self.imgs.shape)
        self._near, self._far = get_boundary(kwargs['no_ndc'], self.bds)
        # 앞에서 numpy로 구현된 연산이 있어서
        self._poses = torch.Tensor(self._poses).to(os.environ['DEVICE'])

        if int(os.environ['VERBOSE']): 
            log(f"Attributes :\n")
            for name, attr in vars(self).items():
                msg = attr
                if isinstance(attr, list):
                    msg = np.array(attr).shape
                elif isinstance(attr, np.ndarray):
                    msg = attr.shape
                log(f"\t{name} : {msg}\n")
        log_internal(f"[Data] Dataset Loading DONE")

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
        return np.array(list(self.img_shape[:2])+[self.focal_length]).astype(int)

    @Dataset.w2c.getter
    def w2c(self):
        w2cs = self._poses[:, :3, :4]
        return w2cs

    @Dataset.c2w.getter
    def c2w(self):
        # TODO: 현재 w2c값만 가지고 있는 상태로, camera to world transform을 위한 inverse를 구현해야 함
        return

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
        imgs = [imread(path)[..., :3] / 255. for path in self.img_paths]
        imgs = np.stack(imgs, 0)
        imgs = torch.Tensor(imgs).to(os.environ['DEVICE'])
        return imgs

    def load_matrices(self):
        poses_arr = np.load(os.path.join(self.data_dir, 'poses_bounds.npy'))

        poses = poses_arr[:, :-2].reshape([-1, 3, 5])
        bds = poses_arr[:, -2:]
        # NOTE: Focal length는 하나인 것으로 생각하고 있는 것

        sh = poses[0, :2, 4]

        # Boundary scaling : boundary for integral which is along with z-axis
        sc = 1. if self.bd_factor is None else 1. / (bds.min() * self.bd_factor)
        poses[:, :3, 3] *= sc
        bds *= sc

        # Pose process: 아직 왜 이렇게 해야하는지 모르겠고 나중에 코드 최적화 해야할 듯
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
        if self.factor is not None:
            _minify(self.data_dir, factors=[self.factor])
            poses[2, 4, :] = poses[2, 4, :] * 1. / self.factor
        poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)

        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        focal = poses[0, 2, 4]
        return poses, bds, focal
