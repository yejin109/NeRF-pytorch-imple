import torch
import cv2 as cv
import numpy as np
import os
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from ._dataset import Dataset
from functionals import log_internal


class NeusDataset(Dataset):
    def __init__(self, data_type, dataset, camera_outside_sphere, scale_mat_scale, **kwargs):
        super(NeusDataset, self).__init__(data_type, dataset, 'neus')
        self.camera_outside_sphere = camera_outside_sphere
        self.scale_mat_scale = scale_mat_scale
        self.imgs_cv, self.imgs_np, self.imgs, self.masks_cv, self.masks_np, self.masks = self.load_imgs()
        self.n_images = len(self.imgs_np)

        self._intrinsic_matrix, self._poses, self.object_bbox_min, self.object_bbox_max = self.load_matrices()
        self.focal = self._intrinsic_matrix[0][0, 0].to(os.environ['DEVICE'])
        self.H, self.W = self.imgs.size(1), self.imgs.size(2)
        self.image_pixels = self.H * self.W

        log_internal('[Data] Data loaded')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)

        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(torch.inverse(self.intrinsic_matrix)[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.w2c[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.w2c[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.imgs[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(torch.inverse(self.intrinsic_matrix)[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.w2c[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.w2c[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o, rays_v, color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(torch.inverse(self.intrinsic_matrix)[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.w2c[idx_0, :3, 3] * (1.0 - ratio) + self.w2c[idx_1, :3, 3] * ratio
        pose_0 = self.w2c[idx_0].detach().cpu().numpy()
        pose_1 = self.w2c[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    @staticmethod
    def near_far_from_sphere(rays_o, rays_d):
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = self.imgs_cv[idx]
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

    @Dataset.intrinsic_matrix.getter
    def intrinsic_matrix(self):
        return torch.stack(self._intrinsic_matrix).to(os.environ['DEVICE'])  # [n_images, 4, 4]

    @Dataset.w2c.getter
    def w2c(self):
        pose = torch.stack(self._poses).to(os.environ['DEVICE'])  # [n_images, 4, 4]
        w2cs = pose[:, :3, :4]
        return w2cs

    @Dataset.c2w.getter
    def c2w(self):
        # 이 데이터셋에선 사용하지 않음
        return torch.inverse(self.intrinsic_matrix)  # [n_images, 4, 4]

    def load_imgs(self):
        imgs_cv = [cv.imread(im_name) for im_name in sorted(glob(os.path.join(self.data_dir, 'image/*.png')))]
        imgs_np = np.stack(imgs_cv) / 256.0
        imgs_torch = torch.from_numpy(imgs_np.astype(np.float32)).to(os.environ['DEVICE'])  # [n_images, H, W, 3]

        masks_cv = [cv.imread(im_name) for im_name in sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))]
        masks_np = np.stack(masks_cv) / 256.0
        masks_torch = torch.from_numpy(masks_np.astype(np.float32)).to(os.environ['DEVICE'])  # [n_images, H, W, 3]
        return imgs_cv, imgs_np, imgs_torch, masks_cv, masks_np, masks_torch

    def load_matrices(self):
        camera_dict = np.load(os.path.join(self.data_dir, 'cameras_sphere.npz'))

        # world_mat is a projection matrix from world to image
        world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        intrinsics_all = []
        pose_all = []

        for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            intrinsics_all.append(torch.from_numpy(intrinsics).float())
            pose_all.append(torch.from_numpy(pose).float())

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, 'cameras_sphere.npz'))['scale_mat_0']
        object_bbox_min = np.linalg.inv(scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]

        object_bbox_min = object_bbox_min[:3, 0]
        object_bbox_max = object_bbox_max[:3, 0]
        return intrinsics_all, pose_all, object_bbox_min, object_bbox_max


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose
