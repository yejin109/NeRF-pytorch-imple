import os
import time
import tqdm
import imageio
import numpy as np

import torch
import torch.nn.functional as F

from .ray import get_rays, ndc_rays
from .sampler import sample_pdf
from ._utils import batchify, log, to8b
from functionals import log_cfg


device = os.environ['DEVICE']


def render_preprocess(H, W, focal, c2w=None, ndc=True, rays=None, near=0., far=1., use_viewdirs=False, c2w_staticcam=None, **kwargs):
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
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)

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


def render_rays(ray_batch,
                model_coarse,
                N_samples,
                embedder_ray,
                embedder_view=None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                model_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False, **kwargs):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      model_coarse: function. Model for predicting RGB and density at each point
        in space.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      model_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples).to(os.environ['device'])
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(os.environ['device'])

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    raw = run_network(pts, viewdirs, model_coarse, embedder_ray, embedder_view)
    # raw = network_query_fn(pts, viewdirs, model_coarse)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = model_coarse if model_fine is None else model_fine
        raw = run_network(pts, viewdirs, run_fn, embedder_ray, embedder_view)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
            log(f"! [Numerical Error] {k} contains nan or inf.\n")

    return ret


def run_network(inputs, viewdirs, fn, embedder_ray, embedder_view, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'."""

    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

    embedded = embedder_ray.embed(inputs_flat)
    if viewdirs is not None:
        input_dirs = torch.broadcast_to(viewdirs[:, None], inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embedder_view.embed(input_dirs_flat)
        embedded = torch.concat([embedded, embedded_dirs], dim=-1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(os.environ['device'])], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


# Visualization
@log_cfg
def render_from_pretrained(images, i_test, testsavedir, render_poses, hwf, K, render_kwargs_test, batch_size, chunk,
                           render_factor, models, embedder_ray, embedder_view, iter_i, **kwargs):
    log('RENDER ONLY\n')
    with torch.no_grad():
        image = images[i_test]
        os.makedirs(testsavedir, exist_ok=True)

        rgbs, _ = render_path(render_poses, hwf, K, chunk, render_kwargs_test, models, batch_size, embedder_ray, embedder_view,
                              savedir=testsavedir, render_factor=render_factor, **kwargs)
        log(f'Done rendering : {testsavedir}\n')
        imageio.mimwrite(os.path.join(testsavedir, f'{iter_i}th_video.mp4'), to8b(rgbs), fps=30, quality=8)

    fpath = os.path.join(testsavedir, f'params_{iter_i}th.pt')
    torch.save({
        'model_coarse':  models['model'],
        'model_fine': models['model_fine']
    }, fpath)


def render_path(render_poses, hwf, K, chunk, render_kwargs, models, batch_size,
                embedder_ray, embedder_view, savedir=None, render_factor=0, **kwargs):
    H, W, focal = hwf
    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    for i, c2w in enumerate(tqdm.tqdm(render_poses)):
        rays = render_preprocess(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        # - Volumetric Rendering
        model_coarse = models['model']
        model_fine = models['model_fine']
        all_ret = {}
        for i in range(0, rays.shape[0], chunk):
            ret = render_rays(rays[i:i + chunk], model_coarse=model_coarse, model_fine=model_fine, embedder_ray=embedder_ray, embedder_view=embedder_view, **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        # - Post process
        k_extract = ['rgb_map', 'disp_map', 'acc_map']
        all_ret = {k: torch.concat(all_ret[k], 0) for k in all_ret}
        for k in all_ret:
            # if all_ret[k].shape[0] // 3 == 0:
            #     sh = (H, W, 3)
            # else:
            sh = (H, W, -1)
            # k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            # NOTE: Manual
            all_ret[k] = torch.reshape(all_ret[k], sh)
        rgb, disp, acc, extras = all_ret['rgb_map'], all_ret['disp_map'], all_ret['acc_map'], {k: all_ret[k] for k in
                                                                                               all_ret if
                                                                                               k not in k_extract}
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())


        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """
        # if savedir is not None:
        #     rgb8 = to8b(rgbs[-1])
        #     filename = os.path.join(savedir, f'{i}.png')
        #     imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


@log_cfg
def get_render_kwargs(perturb, N_importance, N_samples, add_3d_view, white_bkgd, raw_noise_std, no_ndc, lindisp, data_type):
    render_kwargs_train = {
        'perturb': perturb,
        'N_importance': N_importance,
        'N_samples': N_samples,
        'use_viewdirs': add_3d_view,
        'white_bkgd': white_bkgd,
        'raw_noise_std': raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if data_type != 'nerf_llff_data' or no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    if int(os.environ['VERBOSE']):
        log("Train Render arguments : \n")
        log(f"\n\t {render_kwargs_train}\n\n")

        log("Test Render arguments : \n")
        log(f"\n\t {render_kwargs_test}\n\n")
    return render_kwargs_train, render_kwargs_test
