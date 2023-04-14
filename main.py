import os
import time
import yaml
import tqdm
import torch
from torch.optim import Adam

import dataset
import nerf_model
from nerf_model import img2mse, mse2psnr, to8b


def train(run_args, render_args_train):
    global_step = 0
    args = dict(run_args, **render_args_train)
    for i in tqdm.tqdm(range(model_config['N_iters'])):
        time0 = time.time()
        run_epoch(dict(args, **{'global_step': global_step}))
        global_step += 1
    return


def volume_rendering():
    return


def run_epoch(params, target_s, batch_rays, H, W, K, lrate, lrate_decay, chunk, global_step, render_kwargs_train):
    optimizer = Adam(params, lr=lrate, betas=(0.9, 0.999))
    rgb, disp, acc, extras = nerf_model.render(H, W, K, chunk=chunk, rays=batch_rays, retraw=True, **render_kwargs_train)

    optimizer.zero_grad()
    img_loss = img2mse(rgb, target_s)
    trans = extras['raw'][..., -1]
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
    new_lrate = lrate * (decay_rate ** (global_step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lrate
    return loss.item()


if __name__ == '__main__':
    # config = yaml.safe_load(open('./config.yml'))['llff']
    # dset = dataset.SyntheticDataset(**config)

    os.environ['VERBOSE'] = "1"
    # os.environ['DEVICE'] = 'cuda:0'
    os.environ['DEVICE'] = 'cpu'

    setting = yaml.safe_load(open('./config.yml'))
    dataset_config = setting['llff']
    embedding_config = setting['embed']

    rendering_config = setting['rendering']
    model_config = setting['model']

    # Step 1 : Load Dataset
    dset = dataset.LLFFDataset(**dataset_config)

    # Step 2: Load Model
    model_config = dict(embedding_config, **rendering_config)
    models, params, embed_fn, embeddirs_fn = nerf_model.get_model()

    # Step 3 : Load Rendering
    # render_kwargs_train, render_kwargs_test = nerf_model.get_render_kwargs(
    #     rendering_config['perturb'],
    #     rendering_config['N_importance'],
    #     models['model_fine'],
    #     rendering_config['N_samples'],
    #     models['model'],
    #     rendering_config['use_viewdirs'],
    #     args_blender['white_bkgd'],
    #     rendering_config['raw_noise_std'],
    #     config['no_ndc'],
    #     config['lindisp'],
    #     args['dataset'],
    #     )
    # Step 4 : Ray Generation
    hwf = dset.hwf
    intrinsic_matrix = dset.intrinsic_matrix
    i_train, i_val, i_test = dset.i_train, dset.i_val, dset.i_test
    ray_cfg = { 
        "N_rand" : model_config['N_rand'], 
        "use_batching": model_config['no_batching'], 
        "H": hwf[0], 
        "W": hwf[1], 
        "focal": hwf[2], 
        "K": intrinsic_matrix, 
        "N_iters": model_config['N_iters'], 
        "i_train": i_train,
        "i_val": i_val, 
        "i_test": i_test, 
        "precrop_iters": model_config['precrop_iters'], 
        "precrop_frac": model_config['precrop_frac']
    }
    target_s, batch_rays = nerf_model.ray_generation(**ray_cfg)

    # Step 5: Ray Colorization
    # TODO: Make iteration loop
    # hwf = dset.hwf
    # run_args = {
    #     "params": params, 
    #     "target_s": target_s,
    #     'batch_rays': batch_rays,
    #     "H": hwf[0], 
    #     "W": hwf[1], 
    #     "focal": hwf[2], 
    #     "K": intrinsic_matrix,  
    #     "lrate": model_config['lrate'], 
    #     "lrate_decay": model_config['lrate_decay'], 
    #     "chunk": model_config['chunk'],     
    # }
    # train(run_args)


    print(dataset_config)
    print(dset.intrinsic_matrix)
    print(dset.hwf)
    print(dset.img_shape)

