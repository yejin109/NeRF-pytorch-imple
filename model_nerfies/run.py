import os
import torch
import cv2 as cv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from dataset import NerfiesDataSet

from ._utils import get_value, compute_opaqueness_mask, to_cpu, general_loss_with_squared_residual
from .warping import create_warp_field
from .visualize import colorize
from functionals import log_train, log_internal, total_grad_norm

device = os.environ['DEVICE']


def train(model, dataset: NerfiesDataSet, run_cfg, render_cfg, model_cfg, log_cfg):
    data_dict, metadata_dict = dataset.prepare_data()
    # TODO : shuffle 시간이 오래 걸림. list indexing
    points = dataset.load_points(shuffle=False)
    # NOTE: torch에서는 params로 불러오려면 attr으로 module이 있어야 한다......
    params = model.get_params()
    # params = model.get_params() + list(model.mlps['coarse'].parameters()) + list(model.mlps['fine'].parameters())
    optimizer = torch.optim.Adam(params, lr=run_cfg['lr_schedule']['initial_value'])
    warp_param = {
        'alpha': None,
        'time_alpha': None
    }
    update_schedule(0, optimizer, run_cfg, warp_param)

    image_perm = torch.randperm(dataset.n_images)

    for iter_step in tqdm(range(run_cfg['N_iters'])):

        batch = get_batch(image_perm[iter_step % len(image_perm)], dataset, run_cfg, render_cfg, data_dict, metadata_dict, warp_param,
                          return_warp_jacobian=False, batch_size=run_cfg['batch_size'])

        update_schedule(iter_step, optimizer, run_cfg, warp_param)

        coarse_ret, fine_ret = model(**batch)
        stats = defaultdict(dict)
        fine_loss, stats['fine'] = compute_model_loss(fine_ret, batch, run_cfg)
        coarse_loss, stats['coarse'] = compute_model_loss(coarse_ret, batch, run_cfg)

        loss = fine_loss + coarse_loss
        if run_cfg['use_background_loss']:
            background_idx = np.random.randint(0, len(points), run_cfg['background_points_batch_size'])
            background_loss = compute_background_loss(
                dataset,
                points[background_idx],
                run_cfg['background_noise_std'], batch['warp_params'], model_cfg)
            background_loss = run_cfg['background_loss_weight'] * background_loss.mean()

            loss += background_loss
        else:
            background_loss = -1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        l2_grad_norm = total_grad_norm(params)
        train_log = {"Loss": loss, "L2_Grad_Norm": l2_grad_norm,
                     'Fine_loss': fine_loss.item(), 'Coarse_loss': coarse_loss.item(),
                     "Background_loss": background_loss}
        train_log = dict(train_log, **{f"{k}_fine": v for k, v in stats['fine'].items()})
        train_log = dict(train_log, **{f"{k}_coarse": v for k, v in stats['coarse'].items()})
        if iter_step == 0:
            log_train(tuple(train_log.keys()))
        log_train(tuple(train_log.values()))

        if (iter_step+1) % log_cfg['i_print'] == 0:
            log_internal(
                f"[Train] Iteration {iter_step + 1} / {run_cfg['N_iters']}, "
                f"Loss : {loss:.3f}, L2 Grad Norm : {l2_grad_norm: .3f} "
                f"Coarse PSNR: {stats['coarse']['metric/psnr'].item():.3f} "
                f"Fine PSNR: {stats['fine']['metric/psnr'].item():.3f} "
                f"LR : {optimizer.param_groups[0]['lr']}, warp alpha : {warp_param['alpha']} ")
        if (iter_step + 1) % log_cfg['i_img'] == 0:
            validate_image(model, dataset, run_cfg, render_cfg, data_dict, metadata_dict, warp_param, iter_step)

        if (iter_step + 1) % len(image_perm) == 0:
            image_perm = torch.randperm(dataset.n_images)

        if (iter_step+1) % log_cfg['i_weights'] == 0:
            # save_checkpoint(iter_step, model, optimizer)
            pass


# def save_checkpoint(iter_step, model, optimizer):
#     nerf = renderer.nerf
#     sdf_network = renderer.sdf_network
#     deviation_network = renderer.deviation_network
#     color_network = renderer.color_network
#     checkpoint = {
#         'nerf': nerf.state_dict(),
#         'sdf_network_fine': sdf_network.state_dict(),
#         'variance_network_fine': deviation_network.state_dict(),
#         'color_network_fine': color_network.state_dict(),
#         'optimizer': optimizer.state_dict(),
#         'iter_step': iter_step,
#     }
#
#     torch.save(checkpoint, os.path.join(os.environ['LOG_DIR'], 'ckpt_{:0>6d}.pth'.format(iter_step)))
#     log_internal(f'[SAVE] {iter_step+1}th ckpt saved')


def validate_image(model, dataset, run_cfg, render_cfg, data_dict, metadata_dict, warp_param, iter_step,
                   idx=-1):
    torch.cuda.empty_cache()
    if idx < 0:
        idx = np.random.randint(dataset.n_images)
        # idx = 0

    log_internal(f'[Validate]: iteration of {iter_step}, image of {idx}')

    with torch.no_grad():
        test_out = {'rgb_fine': torch.zeros((dataset.W * dataset.H, 3)),
                    'rgb_coarse': torch.zeros((dataset.W * dataset.H, 3)),
                    'med_depth': torch.zeros((dataset.W * dataset.H,))}
        test_batch = {'rgb': torch.zeros((dataset.W * dataset.H, 3))}
        test_batch_size = run_cfg['eval_batch_size']
        test_n_iters = int(dataset.W * dataset.H / test_batch_size)
        for test_iter in range(0, test_n_iters):
            if test_iter * test_batch_size > dataset.W * dataset.H:
                break
            batch_visual = get_batch(idx, dataset, run_cfg, render_cfg, data_dict,
                                     metadata_dict, warp_param,
                                     batch_size=test_batch_size, return_warp_jacobian=False,
                                     is_train=False, test_iter=test_iter)
            coarse_ret_visual, fine_ret_visual = model(**batch_visual)
            test_out['rgb_fine'][test_iter * test_batch_size:(test_iter + 1) * test_batch_size] = fine_ret_visual['rgb']
            test_out['rgb_coarse'][test_iter * test_batch_size:(test_iter + 1) * test_batch_size] = coarse_ret_visual['rgb']
            test_out['med_depth'][test_iter * test_batch_size:(test_iter + 1) * test_batch_size] = fine_ret_visual[
                'med_depth']
            test_batch['rgb'][test_iter * test_batch_size:(test_iter + 1) * test_batch_size] = batch_visual['rgb']

        test_out['rgb_fine'] = torch.reshape(test_out['rgb_fine'], (dataset.W, dataset.H, 3))
        test_out['rgb_coarse'] = torch.reshape(test_out['rgb_coarse'], (dataset.W, dataset.H, 3))
        test_out['med_depth'] = torch.reshape(test_out['med_depth'], (dataset.W, dataset.H))
        test_batch['rgb'] = torch.reshape(test_batch['rgb'], (dataset.W, dataset.H, 3))
        visualize(test_out, test_batch, dataset, idx, iter_step)


def visualize(model_out, batch, dataset: NerfiesDataSet, idx, iter_step):
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    rgb_fine = to_cpu(model_out['rgb_fine'])
    # rgb_fine = (rgb_fine*256).clip(0, 255)
    rgb_fine = to8b(rgb_fine)
    rgb_coarse = to_cpu(model_out['rgb_coarse'])
    rgb_coarse = to8b(rgb_coarse)
    # rgb_coarse = (rgb_coarse * 256).clip(0, 255)
    # acc = model_out['acc']
    # depth_exp = model_out['depth']
    depth_med = model_out['med_depth']
    rgb_target = to_cpu(batch['rgb'])
    rgb_target = to8b(rgb_target)
    # rgb_target = (rgb_target * 256).clip(0, 255)
    depth_med_viz = colorize(depth_med.detach().cpu().numpy(), cmin=dataset.near, cmax=dataset.far)
    depth_med_viz = (depth_med_viz * 256).clip(0, 255)
    cv.imwrite(os.path.join(os.environ['LOG_DIR'],
                            'validations',
                            'RGB_{:0>8d}_{}.png'.format(iter_step, idx)),
               np.concatenate([rgb_coarse, rgb_fine, rgb_target], axis=1))
    cv.imwrite(os.path.join(os.environ['LOG_DIR'],
                            'validations',
                            'Depth{:0>8d}_{}.png'.format(iter_step, idx)),
               depth_med_viz)


def compute_background_loss(dataset: NerfiesDataSet, points, noise_std, warp_params, model_cfg, alpha=-2, scale=0.001):
    """Compute the background regularization loss."""
    warp_cfg = model_cfg['warp']
    time_encoder_args = dict(warp_cfg['encoder_args'], **{
        'scale': warp_cfg['time_encoder_args']['scale'],
        'mlp_args': dict(warp_cfg['mlp_args'], **warp_cfg['time_encoder_args']['time_mlp_args'])
    })
    field_args = {
        'points_encoder_args': dict(warp_cfg['points_encoder_args'], **warp_cfg['encoder_args']),
        'metadata_encoder_type': model_cfg['warp_metadata_encoder_type'],
        'glo_encoder_args': {'num_embeddings': dataset.num_warp_embeddings,
                             'embedding_dim': warp_cfg['num_warp_features']},
        'time_encoder_args': time_encoder_args,
        'mlp_trunk_args': dict(warp_cfg['mlp_args'], **warp_cfg['mlp_trunk_args']),
        'mlp_branch_w_args': dict(warp_cfg['mlp_args'], **warp_cfg['mlp_branch_w_args']),
        'mlp_branch_v_args': dict(warp_cfg['mlp_args'], **warp_cfg['mlp_branch_v_args']),
        'use_pivot': False,
        'mlp_branch_p_args': dict(warp_cfg['mlp_args'], **warp_cfg['mlp_branch_p_args']),
        'use_translation': False,
        'mlp_branch_t_args': dict(warp_cfg['mlp_args'], **warp_cfg['mlp_branch_t_args']),

    }

    point_noise = noise_std * torch.randn(points.shape)
    points = torch.Tensor(points) + point_noise

    warp_field_args = {"field_type": model_cfg['warp']['warp_field_type'], 'field_args': field_args, 'num_batch_dims': 0}
    metadata = np.random.choice(dataset.warp_id, size=points.shape[0]).astype(np.int32)
    metadata = torch.LongTensor(metadata).to(device)

    warp_field = create_warp_field(**warp_field_args)
    warp_out = warp_field(points, metadata, warp_params, False, False)
    warped_points = warp_out['warped_points'][..., :3]
    sq_residual = torch.sum((warped_points - points)**2, dim=-1)
    loss = general_loss_with_squared_residual(sq_residual, alpha=alpha, scale=scale)
    return loss


def compute_model_loss(model_out, batch, run_cfg):
    # Original loss
    rgb_loss = ((model_out['rgb'] - batch['rgb']) ** 2).mean()
    stats = {
        'loss/rgb': rgb_loss,
    }
    loss = rgb_loss
    stats['loss/elastic'] = 0
    stats['residual/elastic'] = 0
    if run_cfg['use_elastic_loss']:
        # (Batch size, Sample size)
        weights = model_out['weights']
        # (Batch size, Sample size, 3, Batch size, Sample size, 3)
        jacobian = model_out['warp_jacobian']

        # Pick the median point Jacobian.
        if run_cfg['elastic_reduce_method'] == 'median':
            # Sample 들 중에서 가장 큰 것을 골랐으니
            depth_indices = torch.argmax(compute_opaqueness_mask(weights), dim=-1)
            depth_indices = depth_indices[..., None, None, None, None, None]
            # Sample size에서 indexing을 하도록 구현
            # Original Code :
            # jacobian = jnp.take_along_axis(
            #     # Unsqueeze axes: sample axis, Jacobian row, Jacobian col.
            #     jacobian, depth_indices[..., None, None, None], axis=-3)
            jacobian = torch.take_along_dim(jacobian, depth_indices, dim=-2)

            # Compute loss using Jacobian.
            elastic_loss, elastic_residual = compute_elastic_loss(jacobian)
        # Multiply weight if weighting by density.
        elif run_cfg['elastic_reduce_method'] == 'weight':
            # Compute loss using Jacobian.
            elastic_loss, elastic_residual = compute_elastic_loss(jacobian)
            elastic_loss = weights[..., None, None] * elastic_loss
        else:
            elastic_loss, elastic_residual = 0, 0
        elastic_loss = elastic_loss.sum(axis=-1).mean()
        stats['loss/elastic'] = run_cfg['elastic_loss_weight'] * elastic_loss
        stats['residual/elastic'] = torch.mean(elastic_residual)
        loss += run_cfg['elastic_loss_weight'] * elastic_loss

    # 현재 사용하는 config가 없음
    # if use_warp_reg_loss:
    #     weights = lax.stop_gradient(model_out['weights'])
    #     depth_indices = model_utils.compute_depth_index(weights)
    #     warp_mag = (
    #             (model_out['points'] - model_out['warped_points']) ** 2).sum(axis=-1)
    #     warp_reg_residual = jnp.take_along_axis(
    #         warp_mag, depth_indices[..., None], axis=-1)
    #     warp_reg_loss = utils.general_loss_with_squared_residual(
    #         warp_reg_residual,
    #         alpha=scalar_params.warp_reg_loss_alpha,
    #         scale=scalar_params.warp_reg_loss_scale).mean()
    #     stats['loss/warp_reg'] = warp_reg_loss
    #     stats['residual/warp_reg'] = jnp.mean(jnp.sqrt(warp_reg_residual))
    #     loss += scalar_params.warp_reg_loss_weight * warp_reg_loss

    # 현재 Jacobian 연산 지원하지 않음
    # if 'warp_jacobian' in model_out:
    #     jacobian = model_out['warp_jacobian']
    #     jacobian_det = torch.linalg.det(jacobian)
    #     jacobian_div = utils.jacobian_to_div(jacobian)
    #     jacobian_curl = utils.jacobian_to_curl(jacobian)
    #     stats['metric/jacobian_det'] = jnp.mean(jacobian_det)
    #     stats['metric/jacobian_div'] = jnp.mean(jacobian_div)
    #     stats['metric/jacobian_curl'] = jnp.mean(
    #         jnp.linalg.norm(jacobian_curl, axis=-1))

    stats['loss/total'] = loss
    stats['metric/psnr'] = -10. * torch.log(rgb_loss) / torch.log(torch.Tensor([10.]))
    return loss, stats


def compute_elastic_loss(jacobian, eps=1e-6):
    """
    log_sval type loss
    Args:
        jacobian:
        eps:

    Returns:

    """
    svals = torch.linalg.svd(jacobian).S
    log_svals = torch.log(torch.maximum(svals, torch.ones_like(svals)*eps))
    sq_residual = torch.sum(log_svals**2, dim=-1)

    residual = torch.sqrt(sq_residual)
    loss = general_loss_with_squared_residual(sq_residual, alpha=-2.0, scale=0.03)
    return loss, residual


def get_batch(idx, dataset: NerfiesDataSet, run_cfg, render_cfg, data_dict, metadata_dict, warp_param, batch_size, return_warp_jacobian,
              is_train=True, test_iter=None):
    if is_train:
        pixels_x = torch.randint(low=0, high=dataset.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=dataset.H, size=[batch_size])
        batch = {k: v[idx][(pixels_x, pixels_y)] for k, v in data_dict.items()}
        batch['metadata'] = {k: v[idx][(pixels_x, pixels_y)] for k, v in metadata_dict.items()}
    else:
        batch = {k: torch.flatten(v[idx], end_dim=1)[test_iter*batch_size:(test_iter+1)*batch_size] for k, v in data_dict.items()}
        batch['metadata'] = {k: torch.flatten(v[idx], end_dim=1)[test_iter*batch_size:(test_iter+1)*batch_size] for k, v in metadata_dict.items()}

    batch['viewdirs'] = None
    batch['warp_params'] = warp_param
    batch['is_metadata_encoded'] = run_cfg['is_metadata_encoded']
    batch['return_points'] = run_cfg['return_points']
    batch['return_weights'] = run_cfg['return_weights']
    batch['return_warp_jacobian'] = return_warp_jacobian

    batch['ray_sampling_args'] = {
        'near': dataset.near,
        'far': dataset.far,
        'use_stratified_sampling': render_cfg['use_stratified_sampling'],
        'use_linear_disparity': render_cfg['use_linear_disparity'],
    }
    batch['rendering_args'] = {
        'use_white_background': render_cfg['use_white_background'],
        'sample_at_infinity': render_cfg['use_sample_at_infinity'],
        'noise_std': get_value(render_cfg['noise_std']),
        'use_stratified_sampling': render_cfg['use_stratified_sampling'],
    }

    batch['num_coarse_samples'] = render_cfg['num_coarse_samples']
    batch['num_fine_samples'] = render_cfg['num_fine_samples']
    return batch


def update_schedule(iter_step, optimizer, run_cfg, warp_param):
    """Get the value for the given step."""

    for g in optimizer.param_groups:
        g['lr'] = update_param(iter_step, run_cfg['lr_schedule'])

    warp_param['alpha'] = update_param(iter_step, run_cfg['warp_alpha_schedule'])
    warp_param['time_alpha'] = schedule_const(run_cfg['constant_warp_time_alpha_schedule'])


def update_param(iter_step, cfg):
    if cfg['type'] == 'linear':
        return schedule_linear(iter_step, cfg)
    elif cfg['type'] == 'const':
        return schedule_const(cfg)
    elif cfg['type'] == 'exponential':
        return schedule_exp(iter_step, cfg)


def schedule_linear(iter_step, scheduler_cfg):
    """Get the value for the given step."""
    if scheduler_cfg['num_steps'] == 0:
        return np.full_like(iter_step, scheduler_cfg['final_value'], dtype=np.float32)
    alpha = np.minimum(iter_step / scheduler_cfg['num_steps'], 1.0)
    return (1.0 - alpha) * scheduler_cfg['initial_value'] + alpha * scheduler_cfg['final_value']


def schedule_const(scheduler_cfg):
    return scheduler_cfg['value']


def schedule_exp(iter_step, scheduler_cfg):
    if iter_step >= scheduler_cfg['num_steps']:
        new_val = scheduler_cfg['final_value']
    else:
        final_value = max(scheduler_cfg['final_value'], 1e-6)
        base = final_value / scheduler_cfg['initial_value']
        exponent = iter_step / (scheduler_cfg['num_steps'] - 1)
        new_val = scheduler_cfg['initial_value'] * base ** exponent
    return new_val


def run(model, dataset: NerfiesDataSet, run_cfg, render_cfg, model_cfg, log_cfg):
    train(model, dataset, run_cfg, render_cfg, model_cfg, log_cfg)
