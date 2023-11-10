import os
import imageio
import trimesh
import cv2 as cv
import numpy as np
from glob import glob
import skimage.transform

from pathlib import Path
from preprocess.interface_img import img_to_colmap
import preprocess.interface_colmap.colmap_read_model as read_model


def preprocess_neus(file_directory: Path, colmap_version, match_type='exhaustive_matcher'):
    """
    :param file_directory: directory of image files
    :param match_type: type of matcher used.
        exhaustive_matcher sequential_matcher.
        Other matchers not supported at this time
    :param colmap_version
    :return:
    """

    def save_poses(basedir, poses, pts3d, perm):
        pts_arr = []
        vis_arr = []
        for k in pts3d:
            pts_arr.append(pts3d[k].xyz)
            cams = [0] * poses.shape[-1]
            for ind in pts3d[k].image_ids:
                if len(cams) < ind - 1:
                    print('ERROR: the correct camera poses for current points cannot be accessed')
                    return
                cams[ind - 1] = 1
            vis_arr.append(cams)

        pts = np.stack(pts_arr, axis=0)
        pcd = trimesh.PointCloud(pts)
        pcd.export(os.path.join(basedir, 'sparse_points.ply'))

        pts_arr = np.array(pts_arr)
        vis_arr = np.array(vis_arr)
        print('Points', pts_arr.shape, 'Visibility', vis_arr.shape)

        poses = np.moveaxis(poses, -1, 0)
        poses = poses[perm]
        np.save(os.path.join(basedir, 'poses.npy'), poses)

    def gen_cameras(work_dir):
        poses_hwf = np.load(os.path.join(work_dir, 'poses.npy'))  # n_images, 3, 5
        poses_raw = poses_hwf[:, :, :4]
        hwf = poses_hwf[:, :, 4]
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose[:3, :4] = poses_raw[0]
        pts = []
        pts.append((pose @ np.array([0, 0, 0, 1])[:, None]).squeeze()[:3])
        pts.append((pose @ np.array([1, 0, 0, 1])[:, None]).squeeze()[:3])
        pts.append((pose @ np.array([0, 1, 0, 1])[:, None]).squeeze()[:3])
        pts.append((pose @ np.array([0, 0, 1, 1])[:, None]).squeeze()[:3])
        pts = np.stack(pts, axis=0)
        pcd = trimesh.PointCloud(pts)
        pcd.export(os.path.join(work_dir, 'pose.ply'))
        #

        cam_dict = dict()
        n_images = len(poses_raw)

        # Convert space
        convert_mat = np.zeros([4, 4], dtype=np.float32)
        convert_mat[0, 1] = 1.0
        convert_mat[1, 0] = 1.0
        convert_mat[2, 2] = -1.0
        convert_mat[3, 3] = 1.0

        for i in range(n_images):
            pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
            pose[:3, :4] = poses_raw[i]
            pose = pose @ convert_mat
            h, w, f = hwf[i, 0], hwf[i, 1], hwf[i, 2]
            intrinsic = np.diag([f, f, 1.0, 1.0]).astype(np.float32)
            intrinsic[0, 2] = (w - 1) * 0.5
            intrinsic[1, 2] = (h - 1) * 0.5
            w2c = np.linalg.inv(pose)
            world_mat = intrinsic @ w2c
            world_mat = world_mat.astype(np.float32)
            cam_dict['camera_mat_{}'.format(i)] = intrinsic
            cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(intrinsic)
            cam_dict['world_mat_{}'.format(i)] = world_mat
            cam_dict['world_mat_inv_{}'.format(i)] = np.linalg.inv(world_mat)

        # TODO [NeuS Err 1] 현재 manual 처리를 지원하지 않고 사용
        # pcd = trimesh.load(os.path.join(work_dir, 'sparse_points_interest.ply'))
        pcd = trimesh.load(os.path.join(work_dir, 'sparse_points.ply'))
        vertices = pcd.vertices
        bbox_max = np.max(vertices, axis=0)
        bbox_min = np.min(vertices, axis=0)
        center = (bbox_max + bbox_min) * 0.5
        radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
        scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
        scale_mat[:3, 3] = center

        for i in range(n_images):
            cam_dict['scale_mat_{}'.format(i)] = scale_mat
            cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)

        out_dir = os.path.join(work_dir)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'image'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'mask'), exist_ok=True)

        img_path = work_dir.parent
        image_list = glob(os.path.join(img_path, 'images/*.png'))
        image_list.sort()

        for i, image_path in enumerate(image_list):
            img = cv.imread(image_path)
            cv.imwrite(os.path.join(out_dir, 'image', '{:0>3d}.png'.format(i)), img)
            cv.imwrite(os.path.join(out_dir, 'mask', '{:0>3d}.png'.format(i)), np.ones_like(img) * 255)

        np.savez(os.path.join(out_dir, 'cameras_sphere.npz'), **cam_dict)
        print('Process done!')

    directory = Path(file_directory) / 'neus'
    os.makedirs(directory, exist_ok=True)

    # TODO : [NeuS Err 1] 현재 데이터 처리상 manually bounding box를 처리해야 하는데 일단 처리하지 않고 사용하도록
    if not (directory / 'sparse_points.ply').is_file():
        gen_poses(directory, match_type, save_poses, colmap_version)
        gen_cameras(directory)
    else:
        gen_cameras(directory)


def preprocess_nerf(file_directory, colmap_version, match_type='exhaustive_matcher'):
    def save_poses(basedir, poses, pts3d, perm):
        pts_arr = []
        vis_arr = []
        for k in pts3d:
            pts_arr.append(pts3d[k].xyz)
            cams = [0] * poses.shape[-1]
            for ind in pts3d[k].image_ids:
                if len(cams) < ind - 1:
                    print('ERROR: the correct camera poses for current points cannot be accessed')
                    return
                cams[ind - 1] = 1
            vis_arr.append(cams)

        pts_arr = np.array(pts_arr)
        vis_arr = np.array(vis_arr)
        print('Points', pts_arr.shape, 'Visibility', vis_arr.shape)

        zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2, 0, 1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
        valid_z = zvals[vis_arr == 1]
        print('Depth stats', valid_z.min(), valid_z.max(), valid_z.mean())

        save_arr = []
        for i in perm:
            vis = vis_arr[:, i]
            zs = zvals[:, i]
            zs = zs[vis == 1]
            close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
            # print( i, close_depth, inf_depth )

            save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
        save_arr = np.array(save_arr)

        np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)

    directory = Path(file_directory) / 'nerf'
    os.makedirs(directory, exist_ok=True)
    gen_poses(directory, match_type, save_poses, colmap_version)


def gen_poses(basedir, match_type, save_poses: callable, colmap_version, factors=None):
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []
    if not all([f in files_had for f in files_needed]):
        print('Need to run COLMAP')
        img_to_colmap(basedir, colmap_version, matching_method=match_type)
        # Deprecated
        # run_colmap(basedir, match_type)
    else:
        print('Don\'t need to run COLMAP')

    print('Post-colmap')

    poses, pts3d, perm = load_colmap_data(basedir)

    save_poses(basedir, poses, pts3d, perm)

    if factors is not None:
        print('Factors:', factors)
        minify(basedir, factors)

    print('Done with imgs2poses')

    return True


def load_colmap_data(realdir: Path):
    colmap_path = realdir.parent / 'colmap'

    camera_path = colmap_path / 'sparse/0/cameras.bin'
    img_path = colmap_path / 'sparse/0/images.bin'
    point3d_path = colmap_path / 'sparse/0/points3D.bin'

    # camerasfile = os.path.join(realdir, 'colmap/sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camera_path)

    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print('Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h, w, f]).reshape([3, 1])

    # imagesfile = os.path.join(realdir, 'colmap/sparse/0/images.bin')
    # imdata = read_model.read_images_binary(imagesfile)
    imdata = read_model.read_images_binary(img_path)

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])

    names = [imdata[k].name for k in imdata]
    print('Images #', len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)

    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)

    poses = c2w_mats[:, :3, :4].transpose([1, 2, 0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[-1]])], 1)

    # points3dfile = os.path.join(realdir, 'colmap/sparse/0/points3D.bin')
    # pts3d = read_model.read_points3d_binary(points3dfile)
    pts3d = read_model.read_points3d_binary(point3d_path)

    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]],
                           1)

    return poses, pts3d, perm


def minify_v0(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    def downsample(imgs, f):
        sh = list(imgs.shape)
        sh = sh[:-3] + [sh[-3] // f, f, sh[-2] // f, f, sh[-1]]
        imgs = np.reshape(imgs, sh)
        imgs = np.mean(imgs, (-2, -4))
        return imgs

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgs = np.stack([imageio.imread(img) / 255. for img in imgs], 0)

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
        print('Minifying', r, basedir)

        if isinstance(r, int):
            imgs_down = downsample(imgs, r)
        else:
            imgs_down = skimage.transform.resize(imgs, [imgs.shape[0], r[0], r[1], imgs.shape[-1]],
                                                 order=1, mode='constant', cval=0, clip=True, preserve_range=False,
                                                 anti_aliasing=True, anti_aliasing_sigma=None)

        os.makedirs(imgdir)
        for i in range(imgs_down.shape[0]):
            imageio.imwrite(os.path.join(imgdir, 'image{:03d}.png'.format(i)), (255 * imgs_down[i]).astype(np.uint8))


def minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(int(100. / r))
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''

    if factor is not None:
        sfx = '_{}'.format(factor)
        minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]))
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    if not load_imgs:
        return poses, bds

    # imgs = [imageio.imread(f, ignoregamma=True)[...,:3]/255. for f in imgfiles]
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs