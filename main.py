import os
import datetime

os.environ['EXP_NAME'] = '-'.join(['TEST', 'SYNC', str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))])
os.environ['LOG_DIR'] = f'./logs/{os.environ["EXP_NAME"]}'
os.mkdir(os.environ['LOG_DIR'])
os.environ['VERBOSE'] = "0"
os.environ['DEVICE'] = 'cuda:0'
# os.environ['DEVICE'] = 'cpu'

import yaml
import tqdm
import torch

torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torch.optim import Adam

import dataset
import nerf_model
from nerf_model import log as nerf_log
from nerf_model import img2mse, mse2psnr
from functionals import TotalGradNorm, log_train, log_cfg


def run(rendering_cfg, dataset_cfg, run_args):
    # Step 3 : Load Rendering
    render_args_train, render_kwargs_test = nerf_model.get_render_kwargs(
        rendering_cfg['perturb'],
        rendering_cfg['N_importance'],
        rendering_cfg['N_samples'],
        rendering_cfg['use_viewdirs'],
        rendering_cfg['white_bkgd'],
        rendering_cfg['raw_noise_std'],
        dataset_cfg['no_ndc'],
        dataset_cfg['lindisp'],
        dataset_cfg['data_type'],
    )

    ray_cfg = {
        "poses": run_args["poses"],
        "images": run_args["images"],
        "N_rand": model_config['N_rand'],
        "use_batching": model_config['no_batching'],
        "H": hwf[0],
        "W": hwf[1],
        "focal": hwf[2],
        "K": run_args['K'],
        "i_train": run_args["i_train"],
        "i_val": run_args["i_val"],
        "i_test": run_args["i_test"],
        "precrop_iters": model_config['precrop_iters'],
        "precrop_frac": model_config['precrop_frac'],
        "N_iters": model_config['N_iters']
    }

    test_cfg = {
        'images': run_args["images"],
        'i_test': run_args['i_test'],
        'testsavedir': os.environ['LOG_DIR'],
        'render_poses': run_args['render_poses'],
        'hwf': run_args['hwf'].astype(int),
        'K': run_args['K'],
        'render_kwargs_test': render_kwargs_test,
        'batch_size': run_args['batch_size'],
        "chunk": model_config['chunk'],
        'render_factor': rendering_cfg['render_factor'],
        'models': run_args['models'],
        'embedder_ray': run_args['embedder_ray'],
        'embedder_view': run_args['embedder_view'],
        'N_samples': rendering_config['N_samples'],

    }

    epoch_args = dict(run_args, **{'render_args_train': render_args_train, 'ray_cfg': ray_cfg})
    for iter_i in tqdm.tqdm(range(model_config['N_iters'])):
        loss_epoch, psnr_epohc = run_epoch(**dict(epoch_args, **{'iter_i': iter_i}))
        nerf_log(f"Iteration {iter_i+1} / {model_config['N_iters']} DONE")
        if (iter_i+1) % 100 == 0:
            nerf_model.rendering.render_from_pretrained(**dict(test_cfg, **{'iter_i': iter_i}))

        grad_norm_epoch = TotalGradNorm(run_args['params'])
        log_train(iter_i, loss_epoch, psnr_epohc, grad_norm_epoch)


@log_cfg
def run_epoch(models, params, H, W, K, lrate, lrate_decay, chunk, iter_i, render_args_train, ray_cfg, batch_size,
              embedder_ray, embedder_view=None, **kwargs):
    optimizer = Adam(params, lr=lrate, betas=(0.9, 0.999))

    # Step 4 : Ray Generation
    target_s, batch_rays = nerf_model.ray_generation(**dict(ray_cfg, **{'iter_i': iter_i}))

    # Step 5: Ray Colorization
    # - Preprocess : Make rays to be used for model
    rays = nerf_model.rendering.render_preprocess(H, W, K, rays=batch_rays, **render_args_train)

    # - Volumetric Rendering
    model_coarse = models['model']
    model_fine = models['model_fine']
    all_ret = {}
    for i in range(0, rays.shape[0], chunk):
        ret = nerf_model.rendering.render_rays(rays[i:i+chunk], model_coarse=model_coarse, model_fine=model_fine,
                                               embedder_ray=embedder_ray, embedder_view=embedder_view, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

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
    # config = yaml.safe_load(open('./config.yml'))['llff']
    # dset = dataset.SyntheticDataset(**config)

    torch.backends.cudnn.benchmark = True
    setting = yaml.safe_load(open('./config.yml'))
    dataset_config = setting['llff']
    # dataset_config = setting['synthetic']
    embedding_config = setting['embed']
    rendering_config = setting['rendering']
    model_config = setting['model']

    # Step 1 : Load Dataset
    dset = dataset.LLFFDataset(**dataset_config)
    # dset = dataset.SyntheticDataset(**dataset_config)
    #
    # Step 2: Load Model
    model_config = dict(model_config, **rendering_config)
    model_config['embed_cfg'] = embedding_config
    models, params, embedder_ray, embedder_view = nerf_model.get_model(**model_config)

    # Iteration :
    # - Step 3
    # - Step 4
    # - Step 5
    hwf = dset.hwf
    run_arguments = {
        "poses": dset.w2c,
        "images": dset.imgs,
        "params": params,
        "H": hwf[0],
        "W": hwf[1],
        "K": dset.intrinsic_matrix,
        "focal": hwf[2],
        "lrate": model_config['lrate'],
        "lrate_decay": model_config['lrate_decay'],
        "chunk": model_config['chunk'],

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
        'hwf': hwf,

    }
    run(rendering_config, dataset_config, run_arguments)

    print(dataset_config)
    print(dset.intrinsic_matrix)
    print(dset.hwf)
    print(dset.img_shape)

