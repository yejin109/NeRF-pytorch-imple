import os
import torch
import torch.nn as nn

from .embed import get_embed_func

def get_model(multires, multires_views, i_embed, add_3d_view, netdepth, netdepth_fine, netwidth, netwidth_fine, N_importance, **kwargs):
    coarse = NeRF()
    embed_fn, input_ch = get_embed_func(multires, i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if add_3d_view:
        embeddirs_fn, input_ch_views = get_embed_func(multires_views, i_embed)

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
    models = {'model': model}

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
    models['model_fine'] = model_fine

    params = []
    for model in models.values():
        if model is not None:
            params += list(model.parameters())
    return models, params, embed_fn, embeddirs_fn


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
    def __init__(self, layer_num, in_feature, hidden_feature, out_feature, in_feature_view, use_viewdirs=False, viewdir_layer_num=1, skip_connection_layer_list=[4]):
        super(NeRF, self).__init__()
        self.layer_num = layer_num
        self.skip = skip_connection_layer_list
        self.uses_viewdirs = use_viewdirs
        self.viewdir_layer_num = viewdir_layer_num
        self.bottleneck_out = 256
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.in_feature_view = in_feature_view

        dense_layer_list = [['Dense_0', Dense(in_feature, hidden_feature, activation=nn.ReLU)]]
        for i in range(1, layer_num):
            dense_layer_list.append([f'Dense_{i}', Dense(hidden_feature, hidden_feature, activation=nn.ReLU)])
            if i-1 in self.skip:
                dense_layer_list[-1] = [f'Dense_{i}', Dense(hidden_feature+in_feature, hidden_feature, activation=nn.ReLU)]
        self.dense_layers = nn.ModuleDict(dense_layer_list)
        self.head = Dense(hidden_feature, out_feature)

        self.alpha_layer = None
        self.bottleneck_layer = None
        self.view_dir_layers = None
        if use_viewdirs:
            self.alpha_layer = Dense(hidden_feature, 1)
            self.bottleneck_layer = Dense(hidden_feature, self.bottleneck_out)

            viewdir_layer_list = [['ViewDir_0', Dense(in_feature_view+self.bottleneck_out, (in_feature_view+self.bottleneck_out)//2, activation=nn.ReLU)]]
            for i in range(1, viewdir_layer_num):
                viewdir_in_feature = viewdir_layer_list[-1][-1].out_feature
                viewdir_layer_list.append([f'ViewDir_{i}', Dense(viewdir_in_feature, viewdir_in_feature//2, activation=nn.ReLU)])
            viewdir_layer_list.append(['ViewDir_Head', Dense(viewdir_layer_list[-1][-1].out_feature, 3)])
            self.view_dir_layers = nn.ModuleDict(viewdir_layer_list)

            del self.head

    def forward(self, x):
        x, x_view = torch.split(x, [self.in_feature, self.in_feature_view], dim=-1)
        out = x
        for i in range(self.layer_num):
            out = self.dense_layers[f'Dense_{i}'](out)
            if i in self.skip:
                out = torch.cat([out, x], dim=-1)

        if self.uses_viewdirs:
            out_alpha = self.alpha_layer(out)
            out_bottleneck = self.bottleneck_layer(out)
            viewdirs = torch.cat([out_bottleneck, x_view], dim=-1)
            out_viewdirs = viewdirs
            for i in range(self.viewdir_layer_num):
                out_viewdirs = self.view_dir_layers[f'ViewDir_{i}'](out_viewdirs)
            out = self.view_dir_layers['ViewDir_Head'](out_viewdirs)
            out = torch.cat([out, out_alpha], axis=1)
        else:
            out = self.head(out)
        return out