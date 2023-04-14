import os
import json
import numpy as np

from ._dataset import Dataset
from ._utils import imread


class SyntheticDataset(Dataset):
    def __init__(self, data_type, run_type, dataset, path_zflat, factor=None, bd_factor=None):
        super(SyntheticDataset, self).__init__(data_type, run_type, dataset, path_zflat, factor, bd_factor)
        self.img_dir = os.path.join(self.data_dir, run_type)
        self.img_paths = [os.path.join(self.img_dir, f) for f in sorted(os.listdir(self.img_dir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

        self.imgs = self.load_imgs()
        self.img_shape = self.imgs[0].shape
        self._pose_inv, self.bds, self.focal_length = self.load_matrices()
        assert (len(self.imgs) == len(self._pose_inv))

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
        # TODO: 현재 c2w값만 가지고 있는 상태로, camera to world transform을 위한 inverse를 구현해야 함
        return

    @Dataset.c2w.getter
    def c2w(self):
        return self._pose_inv


    def load_imgs(self):
        # TODO: Factor 반영하기
        imgs = [imread(path)[..., :3] / 255. for path in self.img_paths]
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
