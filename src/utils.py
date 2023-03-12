import os
import torch
import numpy as np
import torch.nn.functional as F


def debug(func):
    def wrapper(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            return res
        except:
            print(f"Input \n\t args: \n{args} \n\t kwargs{kwargs}")
            assert False
    return wrapper


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device


def init_log(basedir, expname, args):
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(args):
            attr = args[arg]
            file.write('{} = {}\n'.format(arg, attr))


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn
    @debug
    def ret(inputs):
        # return tf.concat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


@debug
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


def log_train():
    if i % args.i_weights == 0:
        path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
        torch.save({
            'global_step': global_step,
            'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
            'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        print('Saved checkpoints at', path)

    if i % args.i_video == 0 and i > 0:
        # Turn on testing mode
        with torch.no_grad():
            rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
        print('Done, saving', rgbs.shape, disps.shape)
        moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
        imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        # if args.use_viewdirs:
        #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
        #     with torch.no_grad():
        #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
        #     render_kwargs_test['c2w_staticcam'] = None
        #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

    if i % args.i_testset == 0 and i > 0:
        testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', poses[i_test].shape)
        with torch.no_grad():
            render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test,
                        gt_imgs=images[i_test], savedir=testsavedir)
        print('Saved test set')

    if i % args.i_print == 0:
        tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
    """
        print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
        print('iter time {:.05f}'.format(dt))
        with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
            tf.contrib.summary.scalar('loss', loss)
            tf.contrib.summary.scalar('psnr', psnr)
            tf.contrib.summary.histogram('tran', trans)
            if args.N_importance > 0:
                tf.contrib.summary.scalar('psnr0', psnr0)
        if i%args.i_img==0:
            # Log a rendered validation view to Tensorboard
            img_i=np.random.choice(i_val)
            target = images[img_i]
            pose = poses[img_i, :3,:4]
            with torch.no_grad():
                rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                    **render_kwargs_test)
            psnr = mse2psnr(img2mse(rgb, target))
            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])
                tf.contrib.summary.scalar('psnr_holdout', psnr)
                tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])
            if args.N_importance > 0:
                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                    tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                    tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
    """


