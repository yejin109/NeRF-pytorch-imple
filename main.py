import os
import time
import yaml
import tqdm
import torch
import imageio
import numpy as np
from torch.optim import Adam

from src.load_llff import load_llff_data
from src.ray import get_rays_np, get_rays
from src.model_functional import create_nerf, run_network
from src.rendering import get_render_kwargs, render, render_path
from src.utils import init_log, img2mse, mse2psnr, to8b, debug, get_device


def load_data(dataset):
    K = None
    if dataset == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args['datadir'], config['factor'], recenter=config['recenter'], bd_factor=config['bd_factor'], spherify=config['spherify'])
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        if env['VERBOSE']:
            print('Loaded llff', images.shape, render_poses.shape, hwf, args['datadir'])
        if not isinstance(i_test, list):
            i_test = [i_test]

        if config['llffhold'] > 0:
            if env['VERBOSE']:
                print('Auto LLFF holdout,', config['llffhold'])
            i_test = np.arange(images.shape[0])[::config['llffhold']]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        if env['VERBOSE']:
            print('DEFINING BOUNDS')
        if config['no_ndc']:
            # TODO:  original code used tf so that both var type was tf.Tensor and temporarily using numpy ndarray.
            # near = tf.reduce_min(bds) * .9
            # far = tf.reduce_max(bds) * 1.
            near = np.min(bds) * .9
            far = np.max(bds) * 1.
        else:
            near = 0.
            far = 1.
        if env['VERBOSE']:
            print('NEAR FAR', near, far)
        return images, i_train, i_val, i_test, hwf, K, poses, render_poses, near, far
    else:
        print('Unknown dataset type', args['dataset'], 'exiting')
        return


def cast_intrinsics(poses, hwf, i_test, render_poses, K):
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args_render['render_test']:
        render_poses = np.array(poses[i_test])

    return H, W, hwf, focal, K


def create_nerf_model(near, far):
    models, embed_fn, embeddirs_fn = create_nerf(
        args_render['multires'],
        args_render['multires_views'],
        args_render['i_embed'],
        args_render['use_viewdirs'],
        args_model['netdepth'],
        args_model['netdepth_fine'],
        args_model['netwidth'],
        args_model['netwidth_fine'],
        args_render['N_importance'],
    )
    grad_vars = []
    for model in models.values():
        if model is not None:
            grad_vars += list(model.parameters())
    optimizer = Adam(grad_vars, lr=args_model['lrate'], betas=(0.9, 0.999))

    render_kwargs_train, render_kwargs_test = get_render_kwargs(
        args_render['perturb'],
        args_render['N_importance'],
        models['model_fine'],
        args_render['N_samples'],
        models['model'],
        args_render['use_viewdirs'],
        args_blender['white_bkgd'],
        args_render['raw_noise_std'],
        config['no_ndc'],
        config['lindisp'],
        args['dataset'],
        )
    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args_model['netchunk'])

    # bds_dict = {
    #     'near': tf.cast(near, tf.float32),
    #     'far': tf.cast(far, tf.float32),
    # }
    bds_dict = {
        'near': near,
        'far': far
    }
    render_kwargs_train['network_query_fn'] = network_query_fn
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    if env['VERBOSE']:
        print('Render Train args : ', render_kwargs_train)
        print('Render Test args : ', render_kwargs_test)

    return render_kwargs_train, render_kwargs_test, optimizer


def train(H, W, focal, images, i_train, i_test, i_val, render_kwargs_train, poses, optimizer, global_step=0):
    # Prepare raybatch tensor if batching random rays
    N_rand = args_model['N_rand']
    use_batching = not args_model['no_batching']
    if use_batching:
        # For random ray batching.
        #
        # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
        # interpreted as,
        #   axis=0: ray origin in world space
        #   axis=1: ray direction in world space
        #   axis=2: observed RGB color of pixel
        print('get rays')
        # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
        # for each pixel in the image. This stack() adds a new dimension.
        rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
        rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i]
                             for i in i_train], axis=0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)
        print('done')
        i_batch = 0

        # Move training data to GPU
        images = torch.Tensor(images).to(device)
        rays_rgb = torch.Tensor(rays_rgb).to(device)
    poses = torch.Tensor(poses).to(device)

    N_iters = args_model['N_iters']
    if env['VERBOSE']:
        print('Begin')
        print('TRAIN views are', i_train)
        print('TEST views are', i_test)
        print('VAL views are', i_val)

    for i in tqdm.tqdm(range(N_iters)):
        time0 = time.time()

        # Sample random ray batch

        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)

            # NOTE : actual results are here
            # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
            # target_s[n, rgb] = example_id, observed color.
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args_model['precrop_iters']:
                    dH = int(H//2 * args_model['precrop_frac'])
                    dW = int(W//2 * args_model['precrop_frac'])
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == 0:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args_model['precrop_iters']}")
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

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args_model['chunk'], rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

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
        decay_steps = args_model['lrate_decay'] * 1000
        new_lrate = args_model['lrate'] * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        global_step += 1
        ################################


def render_from_pretrained(images, i_test, basedir, expname, render_poses, hwf, K, render_kwargs_test):
    print('RENDER ONLY')
    with torch.no_grad():
        if args_render['render_test']:
            # render_test switches to test poses
            images = images[i_test]
        else:
            # Default is smoother render_poses path
            images = None

        testsavedir = os.path.join(basedir, expname,
                                   'renderonly_{}'.format('test' if args_render['render_test'] else 'path'))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        rgbs, _ = render_path(render_poses, hwf, K, args_model['chunck'], render_kwargs_test, gt_imgs=images,
                              savedir=testsavedir, render_factor=args_render['render_factor'])
        print('Done rendering', testsavedir)
        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)


if __name__ == '__main__':
    device = get_device()
    os.environ['device'] = device
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    env = yaml.safe_load(open('./env.yml'))
    args = env['args']
    args_render = args['rendering']
    args_model = args['model']
    args_blender = args['blender']

    config = yaml.safe_load(open('./config.yml'))[args['dataset']]

    # NOTE: @nerf-pytorch.run_nerf.py 539
    images, i_train, i_val, i_test, hwf, K, poses, render_poses, near, far = load_data(args['dataset'])

    # NOTE: @nerf-pytorch.run_nerf.py 610
    H, W, hwf, focal, K = cast_intrinsics(poses, hwf, i_test, render_poses, K)

    # NOTE: @nerf-pytorch.run_nerf.py 625
    init_log(env['log_dir'], env['exp_name'], dict(args, **config))

    # NOTE: @nerf-pytorch.run_nerf.py 639
    render_kwargs_train, render_kwargs_test, optimizer = create_nerf_model(near, far)

    # NOTE: @nerf-pytorch.run_nerf.py 650
    render_poses = torch.Tensor(render_poses).to(device)

    # NOTE: @nerf-pytorch.run_nerf.py 653
    # render_from_pretrained(images, i_test, env['log_dir'], env['exp_name'], render_poses, hwf, K, render_kwargs_test)

    # NOTE: @nerf-pytorch.run_nerf.py 674
    train(H, W, focal, images, i_train, i_test, i_val, render_kwargs_train, poses, optimizer)
