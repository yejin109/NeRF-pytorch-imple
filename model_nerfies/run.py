# 1. Ray Generation
# 2. Encoding
# 3. Ray Sampling
# 4. Ray Colorization
# 5. Rendering
import os
import torch
import numpy as np
from tqdm import tqdm
from dataset import NerfiesDataSet
from ._utils import get_value, compute_opaqueness_mask
from .warping import create_warp_field


device = os.environ['DEVICE']


def train(model, dataset: NerfiesDataSet, run_cfg, render_cfg, model_cfg):
    # actual data to be used
    data_dict, metadata_dict = dataset.prepare_data()
    # TODO : shuffle 시간이 오래 걸림. list indexing
    points = dataset.load_points(shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=run_cfg['lr_schedule']['initial_value'])
    warp_param = {
        'alpha': schedule_const(run_cfg['constant_warp_alpha_schedule']),
        'time_alpha': schedule_const(run_cfg['constant_warp_time_alpha_schedule'])
    }
    update_schedule(0, optimizer, run_cfg, warp_param)

    image_perm = torch.randperm(dataset.n_images)

    for iter_step in tqdm(range(run_cfg['N_iters'])):
        batch = get_batch(image_perm[iter_step % len(image_perm)], dataset, run_cfg, render_cfg, data_dict, metadata_dict, warp_param)

        update_schedule(iter_step, optimizer, run_cfg, warp_param)

        coarse_ret, fine_ret = model(**batch)
        losses = dict()
        stats = dict()
        losses['coarse'], stats['coarse'] = compute_model_loss(coarse_ret, batch)
        losses['fine'], stats['fine'] = compute_model_loss(fine_ret, batch)

        if run_cfg['use_background_loss']:
            background_loss = compute_background_loss(
                dataset, points[run_cfg['background_points_batch_size']*iter_step: run_cfg['background_points_batch_size']*(iter_step+1)], run_cfg['background_noise_std'], batch['warp_params'], model_cfg)
            background_loss = background_loss.mean()
            losses['background'] = (
                    run_cfg['background_loss_weight'] * background_loss)
            stats['background_loss'] = background_loss

        # loss = torch.Tensor([0])
        # for _loss in losses.values():
        #      loss += _loss

        loss = losses['fine'] + losses['background']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print()


def inference():
    pass


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
        'glo_encoder_args': {'num_embeddings': dataset.num_appearance_embeddings,
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

    warp_field_args = {"field_type": model_cfg['warp']['warp_field_type'], 'field_args': field_args, 'num_batch_dims': 0}
    metadata = np.random.choice(dataset.warp_id, size=points.shape[0]).astype(np.int32)[:, None]
    point_noise = noise_std * torch.randn(points.shape)
    points = torch.Tensor(points) + point_noise

    warp_field = create_warp_field(**warp_field_args)
    warp_out = warp_field(points, torch.LongTensor(metadata).to(device), warp_params, False, False)
    warped_points = warp_out['warped_points'][..., :3]
    sq_residual = torch.sum((warped_points - points)**2, dim=-1)
    # TODO: general loss를 아직 구현하지 않음
    loss = sq_residual.pow(2)
    # loss = general_loss_with_squared_residual(
    #   sq_residual, alpha=alpha, scale=scale)
    return loss


def compute_model_loss(model_out, batch):
    rgb_loss = ((model_out['rgb'] - batch['rgb'][..., :3]) ** 2).mean()
    stats = {
        'loss/rgb': rgb_loss,
    }
    loss = rgb_loss
    # NOTE : 현재 config를 보면 결국 안쓰는 것 같다.
    # if render_cfg['use_elastic_loss']:
    #     weights = model_out['weights']
    #     jacobian = model_out['warp_jacobian']
    #     # Pick the median point Jacobian.
    #     if run_cfg['elastic_reduce_method'] == 'median':
    #         depth_indices = torch.argmax(compute_opaqueness_mask(weights))
    #         jacobian = torch.take_along_dim(
    #             # Unsqueeze axes: sample axis, Jacobian row, Jacobian col.
    #             jacobian, depth_indices[..., None, None, None], dim=-3)
    #     # Compute loss using Jacobian.
    #     elastic_loss, elastic_residual = v_elastic_fn(jacobian)
    #     # Multiply weight if weighting by density.
    #
    #     if elastic_reduce_method == 'weight':
    #         elastic_loss = weights * elastic_loss
    #     elastic_loss = elastic_loss.sum(axis=-1).mean()
    #     stats['loss/elastic'] = elastic_loss
    #     stats['residual/elastic'] = torch.mean(elastic_residual)
    #     loss += scalar_params.elastic_loss_weight * elastic_loss

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
    #     jacobian_det = jnp.linalg.det(jacobian)
    #     jacobian_div = utils.jacobian_to_div(jacobian)
    #     jacobian_curl = utils.jacobian_to_curl(jacobian)
    #     stats['metric/jacobian_det'] = jnp.mean(jacobian_det)
    #     stats['metric/jacobian_div'] = jnp.mean(jacobian_div)
    #     stats['metric/jacobian_curl'] = jnp.mean(
    #         jnp.linalg.norm(jacobian_curl, axis=-1))

    stats['loss/total'] = loss
    stats['metric/psnr'] = -10. * torch.log(rgb_loss) / torch.log(torch.Tensor([10.]))
    return loss, stats


def get_batch(idx, dataset: NerfiesDataSet, run_cfg, render_cfg, data_dict, metadata_dict, warp_param):
    batch_size = run_cfg['batch_size']
    pixels_x = torch.randint(low=0, high=dataset.W, size=[batch_size])
    pixels_y = torch.randint(low=0, high=dataset.H, size=[batch_size])

    batch = {k: v[idx][(pixels_x, pixels_y)] for k, v in data_dict.items()}
    batch['viewdirs'] = None
    batch['metadata'] = {k: v[idx][(pixels_x, pixels_y)] for k, v in metadata_dict.items()}
    batch['warp_params'] = warp_param
    batch['is_metadata_encoded'] = run_cfg['is_metadata_encoded']
    batch['return_points'] = run_cfg['return_points']
    batch['return_weights'] = run_cfg['return_weights']
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

    new_lr = schedule_exp(iter_step, run_cfg['lr_schedule'])

    for g in optimizer.param_groups:
        g['lr'] = new_lr

    warp_param['alpha'] = schedule_const(run_cfg['constant_warp_alpha_schedule'])
    warp_param['time_alpha'] = schedule_const(run_cfg['constant_warp_time_alpha_schedule'])


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


def run(model, dataset: NerfiesDataSet, run_cfg, render_cfg, model_cfg):
    train(model, dataset, run_cfg, render_cfg, model_cfg)
    print()
