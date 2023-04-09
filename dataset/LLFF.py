import os
import json
import numpy as np

from ._dataset import Dataset, imread


class LLFFDataset(Dataset):
    def __init__(self,data_type, run_type, dataset, path_zflat, factor=None, bd_factor=None):
        super(LLFFDataset, self).__init__(data_type, run_type, dataset, path_zflat, factor, bd_factor)
        self.img_dir = os.path.join(self.data_dir, 'images')
        self.img_paths = [os.path.join(self.img_dir, f) for f in sorted(os.listdir(self.img_dir)) if
                          f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

        self.imgs = self.load_imgs()
        self.img_shape = self.imgs[0].shape
        self._w2cs, self.bds, self.focal_length = self.load_matrices()
        assert (len(self.imgs) == len(self._w2cs))

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
        return self._w2cs

    @Dataset.c2w.getter
    def c2w(self):
        # TODO: 현재 w2c값만 가지고 있는 상태로, camera to world transform을 위한 inverse를 구현해야 함
        return

    def load_imgs(self):
        # TODO: Factor 반영하기
        imgs = [imread(path)[..., :3] / 255. for path in self.img_paths]
        imgs = np.stack(imgs, 0)
        return imgs

    def load_matrices(self):
        poses_arr = np.load(os.path.join(self.data_dir, 'poses_bounds.npy'))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])
        poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)

        w2cs = poses[:, :3, :4]
        # NOTE: Focal length는 하나인 것으로 생각하고 있는 것
        focal = poses[0, 2, 4]
        bds = poses_arr[:, -2:]
        return w2cs, bds, focal
