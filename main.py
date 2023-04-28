"""
Search each step
Step 1 : Load Dataset
Step 2: Load Model
Step 3 : Load Rendering
Step 4 : Ray Generation
Step 5: Ray Colorization
"""

import _init_env
import datetime

import os
import time
import yaml
import tqdm
import torch
from torch.optim import Adam

import dataset
import nerf_model
from nerf_model import img2mse, mse2psnr
from functionals import TotalGradNorm, log_train, log_cfg, log_internal


def run(rendering_cfg, dataset_cfg, run_args):
    # Step 3 : Load Rendering
    render_args_train, render_kwargs_test = nerf_model.get_render_kwargs(
        rendering_cfg['perturb'],
        rendering_cfg['N_importance'],
        rendering_cfg['N_samples'],
        rendering_cfg['use_viewdirs'],
        rendering_cfg['raw_noise_std'],
        **dataset_cfg,
    )
    render_args_train['near'] = run_args['near']
    render_args_train['far'] = run_args['far']
    render_kwargs_test['near'] = run_args['near']
    render_kwargs_test['far'] = run_args['far']

    ray_cfg = {
        "poses": run_args["poses"],
        "images": run_args["images"],
        "N_rand": model_config['N_rand'],
        "use_batching": not model_config['no_batching'],
        "H": run_args['hwf'][0],
        "W": run_args['hwf'][1],
        "focal": run_args['hwf'][2],
        "K": run_args['K'],
        "i_train": run_args["i_train"],
        "i_val": run_args["i_val"],
        "i_test": run_args["i_test"],
        "precrop_iters": model_config['precrop_iters'],
        "precrop_frac": model_config['precrop_frac'],
        "N_iters": model_config['N_iters']
    }

    test_cfg = {
        'models': run_args['models'],
        'embedder_ray': run_args['embedder_ray'],
        'embedder_view': run_args['embedder_view'],

        'render_poses': run_args['render_poses'],
        'render_kwargs': render_kwargs_test,
        'render_factor': rendering_cfg['render_factor'],

        'K': run_args['K'],
        'hwf': run_args['hwf'].astype(int),
        "chunk": model_config['chunk'],
        'savedir': os.environ['LOG_DIR'],
    }

    if not model_config['no_batching']:
        batch_cfg = {
            "H": run_args['hwf'][0],
            "W": run_args['hwf'][1],
            'K': run_args['K'],
            "poses": run_args["poses"],
            "images": run_args["images"],
            "i_train": run_args["i_train"],
        }
        rays_rgb = nerf_model.ray.prepare_ray_batching(**batch_cfg)
        i_batch = 0

    optimizer = Adam(run_args["params"], lr=run_args["lrate"], betas=(0.9, 0.999))
    epoch_args = dict(run_args, **{'render_args_train': render_args_train})
    for iter_i in tqdm.tqdm(range(model_config['N_iters'])):
        # Step 4 : Ray Generation
        # - Pre process : Batch sampling or Random sampling
        if not model_config['no_batching']:
            rays_rgb, batch_rays, target_s, i_batch = nerf_model.ray.sample_ray_batch(rays_rgb, i_batch, model_config['N_rand'])
        else:
            target_s, batch_rays = nerf_model.ray_generation(**dict(ray_cfg, **{'iter_i': iter_i}))
        # - Post process : Update rays w.r.t ndc, use_viewdirs and etc.
        rays = nerf_model.ray.ray_post_processing(ray_cfg['H'], ray_cfg['W'], focal=ray_cfg['focal'], rays=batch_rays,
                                                  **render_args_train)

        # Forward
        loss_epoch, psnr_epoch = run_epoch(**dict(epoch_args, **{'iter_i': iter_i, 'rays': rays, "optimizer": optimizer,
                                                                 "target_s": target_s}))

        # Logging
        if (iter_i+1) % 100 == 0:
            log_internal(f"[Train] Iteration {iter_i + 1} / {model_config['N_iters']} DONE, Loss : {loss_epoch:.4f}, PSNR: {psnr_epoch:.4f}")

        # Image Visualization
        if (iter_i+1) % 300 == 0:
            render_cfg = dict(test_cfg, **{'iter_i': iter_i})
            with torch.no_grad():
                nerf_model.rendering.render_image(**render_cfg)

        grad_norm_epoch = TotalGradNorm(run_args['params'])
        log_train(iter_i, loss_epoch, psnr_epoch, grad_norm_epoch)


def run_epoch(models, lrate, lrate_decay, chunk, iter_i, render_args_train, batch_size,
              embedder_ray, optimizer, rays, target_s, embedder_view=None, **kwargs):
    # Step 5: Ray Colorization
    model_coarse = models['model']
    model_fine = models['model_fine']
    all_ret = {}
    for i in range(0, rays.shape[0], chunk):
        # - Preprocess : Make rays to be used for model
        rays_o, rays_d, z_vals, viewdirs = nerf_model.rendering.render_preprocess(rays[i:i+chunk], **render_args_train)

        # - Volumetric Rendering : Run network and then rendering(raw2output). Hierarchical sampling implemented.
        ret = nerf_model.rendering.render_rays(rays_o, rays_d, z_vals, viewdirs, model_coarse, model_fine, embedder_ray,
                                               embedder_view, **render_args_train)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
        del ret

    # - Post process
    sh = (batch_size, 3)
    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    all_ret = {k: torch.concat(all_ret[k], 0) for k in all_ret}
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)
    rgb, disp, acc, extras = all_ret['rgb_map'], all_ret['disp_map'], all_ret['acc_map'], {k: all_ret[k] for k in all_ret if k not in k_extract}

    optimizer.zero_grad()
    img_loss = img2mse(rgb, target_s)
    loss = img_loss
    psnr = mse2psnr(img_loss)

    if 'rgb0' in extras:
        img_loss0 = img2mse(extras['rgb0'], target_s)
        loss = loss + img_loss0
        psnr0 = mse2psnr(img_loss0)

    loss.backward()
    optimizer.step()

    # NOTE: IMPORTANT!
    ###   update learning rate   ###
    decay_rate = 0.1
    decay_steps = lrate_decay * 1000
    new_lrate = lrate * (decay_rate ** (iter_i / decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lrate
    return loss.item(), psnr.item()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    setting = yaml.safe_load(open('./config.yml'))
    # dataset_config = setting['llff']
    dataset_config = setting['synthetic']
    embedding_config = setting['embed']
    rendering_config = setting['rendering']
    model_config = setting['model']

    # Step 1 : Load Dataset
    # dset = dataset.LLFFDataset(**dict(dataset_config, **{'render_pose_num': rendering_config['render_pose_num'], 'N_rots': rendering_config['N_rots'], 'zrate': rendering_config['zrate']}))
    dset = dataset.SyntheticDataset(**dict(dataset_config, **{'render_pose_num': rendering_config['render_pose_num']}))

    # Step 2: Load Model
    model_config = dict(model_config, **rendering_config)
    model_config['embed_cfg'] = embedding_config
    models, params, embedder_ray, embedder_view = nerf_model.get_model(**model_config)

    # Iteration :
    # - Step 3
    # - Step 4
    # - Step 5
    run_arguments = {
        "poses": dset.w2c,
        "render_pose": dset.render_pose,
        "images": dset.imgs,
        "params": params,
        "H": dset.hwf[0],
        "W": dset.hwf[1],
        "K": dset.intrinsic_matrix,
        "focal": dset.hwf[2],
        "lrate": model_config['lrate'],
        "lrate_decay": model_config['lrate_decay'],
        "chunk": model_config['chunk'],
        'near': dset.near,
        'far': dset.far,

        # Ray Generation
        'i_train': dset.train_i,
        'i_val': dset.val_i,
        'i_test': dset.test_i,

        # Ray Colorization
        'N_samples': rendering_config['N_samples'],
        'batch_size': model_config['N_rand'],
        'models': models,
        'embedder_ray': embedder_ray,
        'embedder_view': embedder_view,

        # Export
        'render_poses': dset.render_pose,
        'hwf': dset.hwf,
    }
    run(rendering_config, dataset_config, run_arguments)

