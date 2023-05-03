import os
import torch
import trimesh
import cv2 as cv
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from functionals import log_train, log_internal, total_grad_norm


def train(params_to_train, learning_rate, N_iters, dataset, white_bkgd, batch_size, mask_weight, renderer,
          igr_weight, i_print, i_weights, i_img, i_video, warm_up_end, learning_rate_alpha, anneal_end,
          validate_resolution_level,
          **kwargs):

    optimizer = torch.optim.Adam(params_to_train, lr=learning_rate)

    update_learning_rate(0, warm_up_end, learning_rate_alpha, N_iters, optimizer, learning_rate)
    image_perm = get_image_perm(dataset)

    for iter_step in tqdm(range(N_iters)):
        data = dataset.gen_random_rays_at(image_perm[iter_step % len(image_perm)], batch_size)

        rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
        near, far = dataset.near_far_from_sphere(rays_o, rays_d)

        background_rgb = None
        if white_bkgd:
            background_rgb = torch.ones([1, 3])

        if mask_weight > 0.0:
            mask = (mask > 0.5).float()
        else:
            mask = torch.ones_like(mask)

        mask_sum = mask.sum() + 1e-5
        render_out = renderer.render(rays_o, rays_d, near, far, background_rgb=background_rgb,
                                     cos_anneal_ratio=get_cos_anneal_ratio(iter_step, anneal_end))

        color_fine = render_out['color_fine']
        s_val = render_out['s_val']
        cdf_fine = render_out['cdf_fine']
        gradient_error = render_out['gradient_error']
        weight_max = render_out['weight_max']
        weight_sum = render_out['weight_sum']

        # Loss
        color_error = (color_fine - true_rgb) * mask
        color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
        psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())
        eikonal_loss = gradient_error
        mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
        loss = color_fine_loss + eikonal_loss * igr_weight + mask_loss * mask_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        cdf = (cdf_fine[:, :1] * mask).sum() / mask_sum
        weight_max = (weight_max * mask).sum() / mask_sum
        log_train(loss, color_fine_loss, eikonal_loss, s_val.mean(), cdf, weight_max, psnr, total_grad_norm(params_to_train))

        if (iter_step+1) % i_print == 0:
            log_internal('[Train] ')
            log_internal(
                f"[Train] Iteration {iter_step + 1} / {N_iters} DONE, Loss : {loss.item():.4f}, PSNR: {psnr.item():.4f}")

        if (iter_step+1) % i_weights == 0:
            save_checkpoint(iter_step, renderer, optimizer)

        if (iter_step+1) % i_img == 0:
            validate_image(iter_step, dataset, renderer, validate_resolution_level, batch_size, white_bkgd, anneal_end)

        if (iter_step+1) % i_video == 0:
            validate_mesh(dataset, iter_step, renderer)

        update_learning_rate(iter_step, warm_up_end, learning_rate_alpha, N_iters, optimizer, learning_rate)

        if (iter_step+1) % len(image_perm) == 0:
            image_perm = get_image_perm(dataset)


def get_image_perm(dataset):
    return torch.randperm(dataset.n_images)


def get_cos_anneal_ratio(iter_step, anneal_end):
    if anneal_end == 0.0:
        return 1.0
    else:
        return np.min([1.0, iter_step / anneal_end])


def update_learning_rate(iter_step, warm_up_end, learning_rate_alpha, N_iters, optimizer, learning_rate):
    if iter_step < warm_up_end:
        learning_factor = iter_step / warm_up_end
    else:
        alpha = learning_rate_alpha
        progress = (iter_step - warm_up_end) / (N_iters - warm_up_end)
        learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

    for g in optimizer.param_groups:
        g['lr'] = learning_rate * learning_factor


def save_checkpoint(iter_step, renderer, optimizer):
    nerf = renderer.nerf
    sdf_network = renderer.sdf_network
    deviation_network = renderer.deviation_network
    color_network = renderer.color_network
    checkpoint = {
        'nerf': nerf.state_dict(),
        'sdf_network_fine': sdf_network.state_dict(),
        'variance_network_fine': deviation_network.state_dict(),
        'color_network_fine': color_network.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_step': iter_step,
    }

    torch.save(checkpoint, os.path.join(os.environ['LOG_DIR'], 'ckpt_{:0>6d}.pth'.format(iter_step)))
    log_internal(f'[SAVE] {iter_step+1}th ckpt saved')


def validate_image(iter_step, dataset, renderer, validate_resolution_level, batch_size, use_white_bkgd, anneal_end,
                   idx=-1, resolution_level=-1):
    if idx < 0:
        idx = np.random.randint(dataset.n_images)

    log_internal(f'[Validate]: {iter_step}, camera at {idx}')

    if resolution_level < 0:
        resolution_level = validate_resolution_level
    rays_o, rays_d = dataset.gen_rays_at(idx, resolution_level=resolution_level)
    H, W, _ = rays_o.shape
    rays_o = rays_o.reshape(-1, 3).split(batch_size)
    rays_d = rays_d.reshape(-1, 3).split(batch_size)

    out_rgb_fine = []
    out_normal_fine = []

    for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
        near, far = dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
        background_rgb = torch.ones([1, 3]) if use_white_bkgd else None

        render_out = renderer.render(rays_o_batch,
                                          rays_d_batch,
                                          near,
                                          far,
                                          cos_anneal_ratio=get_cos_anneal_ratio(iter_step, anneal_end),
                                          background_rgb=background_rgb)

        def feasible(key):
            return (key in render_out) and (render_out[key] is not None)

        if feasible('color_fine'):
            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
        if feasible('gradients') and feasible('weights'):
            n_samples = renderer.n_samples + renderer.n_importance
            normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
            if feasible('inside_sphere'):
                normals = normals * render_out['inside_sphere'][..., None]
            normals = normals.sum(dim=1).detach().cpu().numpy()
            out_normal_fine.append(normals)
        del render_out
    log_internal(f'[Validate]: {iter_step}, Rendering done')

    img_fine = None
    if len(out_rgb_fine) > 0:
        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

    normal_img = None
    if len(out_normal_fine) > 0:
        normal_img = np.concatenate(out_normal_fine, axis=0)
        rot = np.linalg.inv(dataset.w2c[idx, :3, :3].detach().cpu().numpy())
        normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                      .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

    os.makedirs(os.path.join(os.environ['LOG_DIR'], 'validations_fine'), exist_ok=True)
    os.makedirs(os.path.join(os.environ['LOG_DIR'], 'normals'), exist_ok=True)

    for i in range(img_fine.shape[-1]):
        if len(out_rgb_fine) > 0:
            cv.imwrite(os.path.join(os.environ['LOG_DIR'],
                                    'validations_fine',
                                    '{:0>8d}_{}_{}.png'.format(iter_step, i, idx)),
                       np.concatenate([img_fine[..., i],
                                       dataset.image_at(idx, resolution_level=resolution_level)]))
        if len(out_normal_fine) > 0:
            cv.imwrite(os.path.join(os.environ['LOG_DIR'],
                                    'normals',
                                    '{:0>8d}_{}_{}.png'.format(iter_step, i, idx)),
                       normal_img[..., i])
    log_internal(f'[Validate]: {iter_step}, Image saved')


def render_novel_image(dataset, renderer, batch_size, use_white_bkgd, idx_0, idx_1, ratio, resolution_level):
    """
    Interpolate view between two cameras.
    """
    rays_o, rays_d = dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
    H, W, _ = rays_o.shape
    rays_o = rays_o.reshape(-1, 3).split(batch_size)
    rays_d = rays_d.reshape(-1, 3).split(batch_size)

    out_rgb_fine = []
    for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
        near, far = dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
        background_rgb = torch.ones([1, 3]) if use_white_bkgd else None

        render_out = renderer.render(rays_o_batch, rays_d_batch, near, far, cos_anneal_ratio=get_cos_anneal_ratio(),
                                          background_rgb=background_rgb)

        out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

        del render_out

    img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
    return img_fine


def validate_mesh(dataset, iter_step, renderer, world_space=False, resolution=64, threshold=0.0):
    bound_min = torch.tensor(dataset.object_bbox_min, dtype=torch.float32)
    bound_max = torch.tensor(dataset.object_bbox_max, dtype=torch.float32)

    vertices, triangles = renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
    os.makedirs(os.path.join(os.environ['LOG_DIR'], 'meshes'), exist_ok=True)

    if world_space:
        vertices = vertices * dataset.scale_mats_np[0][0, 0] + dataset.scale_mats_np[0][:3, 3][None]

    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.export(os.path.join(os.environ['LOG_DIR'], '{:0>8d}.ply'.format(iter_step)))

    log_internal('[Train] Validate mesh done')


def interpolate_view(img_idx_0, img_idx_1, iter_step):
    images = []
    n_frames = 60
    for i in range(n_frames):
        log_internal(f'[Rendering ] {iter_step+1}th Video frame {i+1}/{n_frames}')
        images.append(render_novel_image(img_idx_0, img_idx_1, np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5, resolution_level=4))
    for i in range(n_frames):
        images.append(images[n_frames - i - 1])

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    h, w, _ = images[0].shape
    writer = cv.VideoWriter(os.path.join(os.environ['LOG_DIR'],
                                         '{:0>8d}_{}_{}.mp4'.format(iter_step, img_idx_0, img_idx_1)),
                            fourcc, 30, (w, h))

    for image in images:
        writer.write(image)

    writer.release()
    log_internal(f'[Rendering] {iter_step+1}th Video Done')


def run(model_config, rendering_config, dataset_config, log_config, params, renderer, dataset):
    # Iteration :
    # - Step 3
    # - Step 4
    # - Step 5
    run_arguments = {
        'params_to_train': params,
        'renderer': renderer,
        'dataset': dataset
    }

    train_args = dict(rendering_config, **dict(dataset_config, **dict(run_arguments, **dict(model_config, **log_config))))
    train(**train_args)
