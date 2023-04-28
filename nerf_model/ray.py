import os
import tqdm
import torch
import numpy as np

from functionals import log_time, log_internal


def prepare_ray_batching(H, W, K, poses, images, i_train):
    _images = images
    if not isinstance(images, np.ndarray):
        _images = images.cpu().numpy()
    # For random ray batching
    rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
    rays_rgb = np.concatenate([rays, _images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
    rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.astype(np.float32)
    np.random.shuffle(rays_rgb)

    rays_rgb = torch.Tensor(rays_rgb).to(os.environ['DEVICE'])
    return rays_rgb


def sample_ray_batch(rays_rgb, i_batch, N_rand):
    # Random over all images
    batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3*?]
    batch = torch.transpose(batch, 0, 1)
    # 위의 _images로 np.concatenate된 것에서 보면 알 수 있듯이 만들어낸 ray와 실제 image의 값을 합치게 된다.
    batch_rays, target_s = batch[:2], batch[2]

    i_batch += N_rand
    if i_batch >= rays_rgb.shape[0]:
        log_internal(f"[Data] Shuffle data at {i_batch+1}th batch / Single batch size :{N_rand}")
        rand_idx = torch.randperm(rays_rgb.shape[0])
        rays_rgb = rays_rgb[rand_idx]
        i_batch = 0
    return rays_rgb, batch_rays, target_s, i_batch


def ray_generation(poses, images, N_rand, use_batching, H, W, focal, K, N_iters, i_train, i_val, i_test, precrop_iters, precrop_frac, iter_i):
    # Random from one image
    img_i = np.random.choice(i_train)
    target = images[img_i]
    target = torch.Tensor(target).to(os.environ['DEVICE'])
    pose = poses[img_i, :3,:4]

    rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

    if iter_i < precrop_iters:
        dH = int(H//2 * precrop_frac)
        dW = int(W//2 * precrop_frac)
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
            ), -1)
        if iter_i == 0:
            print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {precrop_iters}")
    else:
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

    coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
    select_coords = coords[select_inds].long()  # (N_rand, 2)
    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

    # NOTE : actual results are here
    batch_rays = torch.stack([rays_o, rays_d], 0)
    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    return target_s, batch_rays


def ray_post_processing(H, W, focal, K=None, c2w=None, ndc=True, rays=None, near=0., far=1., use_viewdirs=False, c2w_staticcam=None, **kwargs):
    """
    H: int. Height of image in pixels.
    W: int. Width of image in pixels.
    focal: float. Focal length of pinhole camera.
    rays: array of shape [2, batch_size, 3]. Ray origin and direction for each example in batch.
    c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
    ndc: bool. If True, represent ray origin, direction in NDC coordinates.
    near: float or array of shape [batch_size]. Nearest distance for a ray.
    far: float or array of shape [batch_size]. Farthest distance for a ray.
    use_viewdirs: bool. If True, use viewing direction of a point in space in model.
    c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for camera while using other c2w argument for viewing directions.
    :return:
    ray
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)

        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / torch.linalg.norm(viewdirs, dim=-1, keepdim=True)
        # viewdirs = tf.cast(tf.reshape(viewdirs, [-1, 3]), dtype=tf.float32)
        viewdirs = viewdirs.reshape([-1, 3]).float()

    # sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(
            H, W, focal, torch.Tensor([1.0]).float(), rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = torch.concat([rays_o, rays_d, near, far], dim=-1)
    if use_viewdirs:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = torch.concat([rays, viewdirs], dim=-1)

    return rays


def get_rays_np(H, W, K, c2w):
    _c2w = c2w
    _K = K
    if not isinstance(c2w, np.ndarray):
        _c2w = c2w.cpu().numpy()
    if not isinstance(K, np.ndarray):
        _K = K.cpu().numpy()

    # 이렇게 하면 안됨!!
    # """Get ray origins, directions from a pinhole camera."""
    # i, j = np.meshgrid(np.arange(W, dtype=np.float32),
    #                    np.arange(H, dtype=np.float32), indexing='xy')
    # dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # rays_d = np.sum(dirs[..., np.newaxis, :] * _c2w[:3, :3], -1)
    # rays_o = np.broadcast_to(_c2w[:3, -1], np.shape(rays_d))

    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - _K[0][2]) / _K[0][0], -(j - _K[1][2]) / _K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * _c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(_c2w[:3, -1], np.shape(rays_d))

    return rays_o, rays_d


def get_rays(H, W, K, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    dirs = dirs.to(os.environ['device'])
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.
    Space such that the canvas is a cube with sides [-1, 1] in each axis.
    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      _focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.
    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d
