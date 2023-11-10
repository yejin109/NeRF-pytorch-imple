import os
import imageio
import numpy as np

# Questions
# TODO: Pose recenter나 recenter pose에서 inv(c2w)와 기존 pose를 matmul하는 것으로 구현한 이유는?


########################################################################################
# 1.  Poses  Operation functionals
#   - c2w <-> w2c
def recenter_poses(poses):
    """
    TODO: input poses is c2w or w2c?
    """
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = inverse_w2c(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def inverse_w2c(wc2):
    """
    Original name : poses_avg in NeRF
        cf) w2c = pose matrix!
    """
    hwf = wc2[0, :3, -1:]

    center = wc2[:, :3, 3].mean(0)
    vec2 = normalize(wc2[:, :3, 2].sum(0))
    up = wc2[:, :3, 1].sum(0)
    c2w = np.concatenate([inverse_rot(vec2, up, center), hwf], 1)

    return c2w


def inverse_rot(z, up, pos):
    """
    Original Name : viewmatrix in NeRF

    Argument : Input 3 vectors 
        - z : z-axis in camera coordinate
        - up : y-axis in camera coordinate
        - pos : positional vector of camera
    """
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


########################################################################################
# 2. render pose functionals
def render_path_spherical(poses, bds, render_pose_num=120):
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    # z-axis direction 
    # TODO: Specify why it uses z-axis direction in d-vector
    rays_d = poses[:, :3, 2:3]

    # Translation becomes position of camera in camera coordinate
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    # Pose matrix(w2c) to c2w: Inverse operation!
    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    # Recenter pose와 같은 맥락에서 나온 것
    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []

    # NOTE: 현재 10개에 1분 정도 걸림
    # for th in np.linspace(0., 2. * np.pi, 120):
    for th in np.linspace(0., 2. * np.pi, render_pose_num):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, new_poses, bds


def render_path_spiral(poses, bds, path_zflat, N_rots=2, render_pose_num=120, zrate=.5):
    c2w = inverse_w2c(poses)

    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    if path_zflat:
        #             zloc = np.percentile(tt, 10, 0)[2]
        zloc = -close_depth * .1
        c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
        rads[2] = 0.
        N_rots = 1
        render_pose_num /= 2

    # NOTE: actual spiral path computation
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * N_rots, render_pose_num + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([inverse_rot(z, up, c), hwf], 1))
    return render_poses


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, f'images_{r}x')
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
            name = f'images_{r}'
            resizearg = '{}%'.format(100. / r)
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


# etc
def get_boundary(no_ndc, bds):
    if no_ndc:
        near = np.min(bds) * .9
        far = np.max(bds) * 1.
    else:
        near = 0.
        far = 1.    
    return near, far


def get_test_idx(poses, llffhold, img_shape):
    c2w = inverse_w2c(poses)
    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)

    if not isinstance(i_test, list):
        i_test = [i_test]
    
    if llffhold > 0:
        log(f"Auto LLFF holdout {llffhold}: \n")
        i_test = np.arange(img_shape[0])[::llffhold]
    return i_test


def get_val_idx(test_idx):
    return test_idx


def get_train_idx(val_idx, test_idx, img_shape):
    return np.array([i for i in np.arange(int(img_shape[0])) if (i not in test_idx and i not in val_idx)])
    

def normalize(x):
    return x / np.linalg.norm(x)


def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)
    

def log(msg):
    with open('logs/debug/pkg_dataset.txt', 'a') as f:
        f.write(msg)
    f.close()
