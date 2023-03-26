import os
import torch
import numpy as np
from torch.optim import Adam

from src.positional_encode import get_embedder
from src.utils import batchify, profile
from src.model import NeRF


@profile
def create_nerf(multires, multires_views, i_embed, add_3d_view, netdepth, netdepth_fine, netwidth, netwidth_fine, N_importance):
    """Instantiate NeRF's MLP model."""

    embed_fn, input_ch = get_embedder(multires, i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if add_3d_view:
        embeddirs_fn, input_ch_views = get_embedder(multires_views, i_embed)
    output_ch = 5 if N_importance > 0 else 4
    skips = [4]
    model = NeRF(
        layer_num=netdepth,
        in_feature=input_ch,
        hidden_feature=netwidth,
        out_feature=output_ch,
        in_feature_view=input_ch_views,
        use_viewdirs=add_3d_view,
        skip_connection_layer_list=skips).to(os.environ['device'])
    # model = init_nerf_model(
    #     D=args.netdepth, W=args.netwidth,
    #     input_ch=input_ch, output_ch=output_ch, skips=skips,
    #     input_ch_views=input_ch_views, use_viewdirs=args.add_3d_view)
    models = {'model': model}
    grad_vars = list(model.parameters())

    model_fine = None
    if N_importance > 0:
        model_fine = NeRF(
            layer_num=netdepth_fine,
            in_feature=input_ch,
            hidden_feature=netwidth_fine,
            out_feature=output_ch,
            in_feature_view=input_ch_views,
            use_viewdirs=add_3d_view,
            skip_connection_layer_list=skips).to(os.environ['device'])
        # model_fine = init_nerf_model(
        #     D=args.netdepth_fine, W=args.netwidth_fine,
        #     input_ch=input_ch, output_ch=output_ch, skips=skips,
        #     input_ch_views=input_ch_views, use_viewdirs=args.add_3d_view)
        grad_vars += list(model_fine.parameters())
    models['model_fine'] = model_fine

    # start = 0
    # basedir = args.basedir
    # expname = args.expname
    #
    # if args.ft_path is not None and args.ft_path != 'None':
    #     ckpts = [args.ft_path]
    # else:
    #     ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
    #              ('model_' in f and 'fine' not in f and 'optimizer' not in f)]
    # print('Found ckpts', ckpts)
    # if len(ckpts) > 0 and not args.no_reload:
    #     ft_weights = ckpts[-1]
    #     print('Reloading from', ft_weights)
    #     model.set_weights(np.load(ft_weights, allow_pickle=True))
    #     start = int(ft_weights[-10:-4]) + 1
    #     print('Resetting step to', start)
    #
    #     if model_fine is not None:
    #         ft_weights_fine = '{}_fine_{}'.format(
    #             ft_weights[:-11], ft_weights[-10:])
    #         print('Reloading fine from', ft_weights_fine)
    #         model_fine.set_weights(np.load(ft_weights_fine, allow_pickle=True))

    return models, embed_fn, embeddirs_fn


@profile
def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'."""

    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

    embedded = embed_fn(inputs_flat)
    if viewdirs is not None:
        input_dirs = torch.broadcast_to(viewdirs[:, None], inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.concat([embedded, embedded_dirs], dim=-1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs
