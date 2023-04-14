import os
import json
import numpy as np

from ._dataset import Dataset
from ._utils import imread, recenter_poses, render_path_spherical, render_path_spiral, get_test_idx, get_train_idx, get_val_idx, get_boundary, log

class LLFFDataset(Dataset):
    def __init__(self,data_type, run_type, dataset, path_zflat, llffhold, factor=None, bd_factor=None, **kwargs):
        super(LLFFDataset, self).__init__(data_type, run_type, dataset, path_zflat, factor, bd_factor)
        # 0. Setup
        self.img_dir = os.path.join(self.data_dir, 'images')
        self.img_paths = [os.path.join(self.img_dir, f) for f in sorted(os.listdir(self.img_dir)) if
                          f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        
        # 1. Load Image
        self.imgs = self.load_imgs()
        self.img_shape = self.imgs[0].shape

        # 2. Load pose matrix and all the attrs to be used
        self._poses, self.bds, self.focal_length = self.load_matrices()
        assert (len(self.imgs) == len(self._poses))
        self._reder_poses = None
        self._test_i = None
        self._val_i = None
        self._train_i = None

        # 3. Value Update
        if kwargs['recenter']:
            if int(os.environ['VERBOSE']):
                log("Recentered : _poses updated\n")
            self._poses = recenter_poses(self._poses)

        if kwargs['spherify']:
            if int(os.environ['VERBOSE']): 
                log("Render path \n\tSpherified : _poses, _reder_poses, bds updated\n")
            self._poses, self._render_poses, self.bds = render_path_spherical(self._poses, self.bds)
        else:
            if int(os.environ['VERBOSE']): 
                log("Render path \n\tSpherified : _poses, _reder_poses, bds updated\n")

            self._reder_poses = render_path_spiral(self._poses, self.bds, path_zflat)
        
        self._reder_poses = np.array(self._reder_poses).astype(np.float32)
        self._test_i = get_test_idx(self._poses, llffhold, self.imgs.shape)
        self._val_i = get_val_idx(self._test_i)
        self._train_i = get_train_idx(self._val_i, self._test_i, self.imgs.shape)
        self._near, self._far = get_boundary(kwargs['no_ndc'], self.bds)

        if int(os.environ['VERBOSE']): 
            log(f"Attributes :\n")
            for name, attr in vars(self).items():
                msg = attr
                if isinstance(attr, list):
                    msg = np.array(attr).shape
                elif isinstance(attr, np.ndarray):
                    msg = attr.shape
                log(f"\t{name} : {msg}\n")

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
        # TODO: 현재 w2c값만 가지고 있는 상태로, camera to world transform을 위한 inverse를 구현해야 함
        return

    @Dataset.render_pose.getter
    def render_pose(self):
        """
        Relevent function : spherify and load_llff_data

        NOTE: llff data에서 Spherify_poses를 사용하거나 기존 루틴을 사용함
        """
        return
    
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
        return imgs

    def load_matrices(self):
        poses_arr = np.load(os.path.join(self.data_dir, 'poses_bounds.npy'))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])
        poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)

        # NOTE: Focal length는 하나인 것으로 생각하고 있는 것
        focal = poses[0, 2, 4]
        bds = poses_arr[:, -2:]
        return poses, bds, focal
