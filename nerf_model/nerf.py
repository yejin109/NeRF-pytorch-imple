import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .embed import get_embedder
from ._utils import log
from functionals import log_cfg, log_internal


@log_cfg
def get_model(multires, multires_views, i_embed, use_viewdirs, layer_num, layer_num_fine, hidden_dim, hidden_feature_fine, N_importance, embed_cfg, **kwargs):
    embedder_ray = get_embedder(embed_cfg, multires, i_embed)
    input_ch_ray = embedder_ray.out_dim

    input_ch_views = 0
    embedder_view = None
    if use_viewdirs:
        embedder_view = get_embedder(embed_cfg, multires_views, i_embed)
        input_ch_views = embedder_view.out_dim

    output_ch = 5 if N_importance > 0 else 4
    skips = [4]
    model = NeRF(
        layer_num=layer_num,
        in_feature=input_ch_ray,
        hidden_dim=hidden_dim,
        out_feature=output_ch,
        in_feature_view=input_ch_views,
        use_viewdirs=use_viewdirs,
        skip_connection_layer_list=skips).to(os.environ['device'])
    model.model_vars('Coarse model')
    models = {'model': model}

    model_fine = None
    if N_importance > 0:
        model_fine = NeRF(
            layer_num=layer_num_fine,
            in_feature=input_ch_ray,
            hidden_dim=hidden_feature_fine,
            out_feature=output_ch,
            in_feature_view=input_ch_views,
            use_viewdirs=use_viewdirs,
            skip_connection_layer_list=skips).to(os.environ['device'])
        model.model_vars('Fine model')
    models['model_fine'] = model_fine

    params = []
    for model in models.values():
        if model is not None:
            params += list(model.parameters())
    log_internal(f"[Model] Init Model DONE")
    return models, params, embedder_ray, embedder_view


class Dense(nn.Module):
    def __init__(self, in_feature, out_feature, activation=None, use_bias=True, ):
        super(Dense, self).__init__()
        self.act = activation
        if activation is not None:
            self.act = activation()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.w = nn.Linear(in_feature, out_feature, bias=use_bias)

    def forward(self, x):
        out = self.w(x)
        if self.act is not None:
            out = self.act(out)
        return out


class NeRF(nn.Module):
    def __init__(self, layer_num, in_feature, hidden_dim, out_feature, in_feature_view, use_viewdirs=False, viewdir_layer_num=1, skip_connection_layer_list=[4]):
        super(NeRF, self).__init__()
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim
        self.in_feature = in_feature
        self.in_feature_view = in_feature_view
        self.use_viewdirs = use_viewdirs
        self.skips = skip_connection_layer_list

        self.pts_linears = nn.ModuleList([nn.Linear(in_feature, hidden_dim)] +
                                         [nn.Linear(hidden_dim, hidden_dim)
                                          if i not in self.skips
                                          else nn.Linear(hidden_dim + in_feature, hidden_dim)
                                          for i in range(layer_num - 1)])
        if use_viewdirs:
            self.feature_linear = nn.Linear(hidden_dim, hidden_dim)
            self.alpha_linear = nn.Linear(hidden_dim, 1)
            self.rgb_linear = nn.Linear(hidden_dim//2, 3)
        else:
            self.output_linear = nn.Linear(hidden_dim, out_feature)

        self.views_linears = nn.ModuleList([nn.Linear(in_feature_view + hidden_dim, hidden_dim // 2)])
        # self.viewdir_layer_num = viewdir_layer_num
        # self.bottleneck_out = 256
        #
        # dense_layer_list = [['Dense_0', Dense(in_feature, hidden_dim, activation=nn.ReLU)]]
        # for i in range(1, layer_num):
        #     dense_layer_list.append([f'Dense_{i}', Dense(hidden_dim, hidden_dim, activation=nn.ReLU)])
        #     if i-1 in self.skip:
        #         dense_layer_list[-1] = [f'Dense_{i}', Dense(hidden_dim + in_feature, hidden_dim, activation=nn.ReLU)]
        # self.dense_layers = nn.ModuleDict(dense_layer_list)
        # self.head = Dense(hidden_dim, out_feature)
        #
        # self.alpha_layer = None
        # self.bottleneck_layer = None
        # self.view_dir_layers = None
        # if use_viewdirs:
        #     self.alpha_layer = Dense(hidden_dim, 1)
        #     self.bottleneck_layer = Dense(hidden_dim, self.bottleneck_out)
        #
        #     viewdir_layer_list = [['ViewDir_0', Dense(in_feature_view+self.bottleneck_out, (in_feature_view+self.bottleneck_out)//2, activation=nn.ReLU)]]
        #     for i in range(1, viewdir_layer_num):
        #         viewdir_in_feature = viewdir_layer_list[-1][-1].out_feature
        #         viewdir_layer_list.append([f'ViewDir_{i}', Dense(viewdir_in_feature, viewdir_in_feature//2, activation=nn.ReLU)])
        #     viewdir_layer_list.append(['ViewDir_Head', Dense(viewdir_layer_list[-1][-1].out_feature, 3)])
        #     self.view_dir_layers = nn.ModuleDict(viewdir_layer_list)
        #
        #     del self.head

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.in_feature, self.in_feature_view], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        return outputs
    # def forward(self, x):
    #     x, x_view = torch.split(x, [self.in_feature, self.in_feature_view], dim=-1)
    #     out = x
    #     for i in range(self.layer_num):
    #         out = self.dense_layers[f'Dense_{i}'](out)
    #         if i in self.skip:
    #             out = torch.cat([out, x], dim=-1)
    #
    #     if self.uses_viewdirs:
    #         out_alpha = self.alpha_layer(out)
    #         out_bottleneck = self.bottleneck_layer(out)
    #         viewdirs = torch.cat([out_bottleneck, x_view], dim=-1)
    #         out_viewdirs = viewdirs
    #         for i in range(self.viewdir_layer_num):
    #             out_viewdirs = self.view_dir_layers[f'ViewDir_{i}'](out_viewdirs)
    #         out = self.view_dir_layers['ViewDir_Head'](out_viewdirs)
    #         out = torch.cat([out, out_alpha], axis=1)
    #     else:
    #         out = self.head(out)
    #     return out

    def model_vars(self, model_type):
        if int(os.environ['VERBOSE']):
            log(model_type)
            log(f"Attributes :\n")
            for name, attr in vars(self).items():
                msg = attr
                if isinstance(attr, list):
                    msg = np.array(attr).shape
                elif isinstance(attr, np.ndarray):
                    msg = attr.shape
                log(f"\t{name} : {msg}\n")
