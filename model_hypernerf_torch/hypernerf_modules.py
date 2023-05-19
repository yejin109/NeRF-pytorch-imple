# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modules for NeRF models."""
# from hypernerf import model_utils
# from hypernerf import types

import torch
import torch.nn as nn


def get_norm_layer(norm_type, in_features):
    """Translates a norm type to a norm constructor."""
    if norm_type is None or norm_type == 'none':
        return None
    elif norm_type == 'layer':
        return nn.LayerNorm(in_features, elementwise_affine=False)
    # elif norm_type == 'group':
    #     return nn.GroupNorm(in_features, affine=False)
    elif norm_type == 'batch':
        return nn.BatchNorm1d(in_features, affine=False)
    else:
        raise ValueError(f'Unknown norm type {norm_type}')


class MLP(nn.Module):
    """Basic MLP class with hidden layers and an output layers."""
    def __init__(self, layer_num, in_feature, hidden_dim, out_feature=0, skip_connection_layer_list=(4,),
                 hidden_norm=None, hidden_activation='relu', use_bias=True, output_activation=None):
        super(MLP, self).__init__()
        self.skips = skip_connection_layer_list
        self.hidden_activation = hidden_activation

        self.layers = nn.ModuleList([nn.Linear(in_feature, hidden_dim, bias=use_bias)] +
                                    [nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
                                     if i not in self.skips
                                     else nn.Linear(hidden_dim + in_feature, hidden_dim, bias=use_bias)
                                     for i in range(layer_num - 1)])
        self.hidden_norm = get_norm_layer(hidden_norm, hidden_dim)
        self.hidden_activation = nn.ReLU() if hidden_activation == 'relu' else None

        self.output_layer = None
        if self.out_feature > 0:
            self.output_layer = nn.Linear(hidden_dim, out_feature, bias=use_bias)
            self.output_activation = nn.ReLU() if hidden_activation == 'relu' else None

    def forward(self, x):
        inputs = x
        for i, l in enumerate(self.layers):
            if i in self.skips:
                x = torch.cat([inputs, x], -1)
            x = self.layers[i](x)
            if self.hidden_norm is not None:
                x = self.hidden_norm(x)
            if self.hidden_activation is not None:
                x = self.hidden_activation(x)

        if self.output_layer is not None:
            x = self.output_layer(x)
            if self.output_activation is not None:
                x = self.output_activation(x)
        return x


class NeRFMLP(nn.Module):
    """A simple MLP.
        Attributes:
        nerf_trunk_depth: int, the depth of the first part of MLP.
        nerf_trunk_width: int, the width of the first part of MLP.
        nerf_rgb_branch_depth: int, the depth of the second part of MLP.
        nerf_rgb_branch_width: int, the width of the second part of MLP.
        activation: function, the activation function used in the MLP.
        skips: which layers to add skip layers to.
        alpha_channels: int, the number of alpha_channelss.
        rgb_channels: int, the number of rgb_channelss.
        condition_density: if True put the condition at the begining which
          conditions the density of the field.

        alpha_condition: a condition array provided to the alpha branch.
        rgb_condition: a condition array provided in the RGB branch.
    """
    def __init__(self, in_feature,
                 trunk_layer_num, trunk_hidden_dim,
                 rgb_layer_num, rgb_hidden_dim, rgb_channels,
                 alpha_layer_num, alpha_hidden_dim, alpha_channels,
                 skip_connection_layer_list=(4,),
                 rgb_condition=None, alpha_condition=None):
        super(NeRFMLP, self).__init__()

        self.rgb_condition = rgb_condition
        self.alpha_condition = alpha_condition

        # TODO: config로 옮기기
        self.trunk_depth: int = 8
        self.trunk_width: int = 256

        self.rgb_branch_depth: int = 1
        self.rgb_branch_width: int = 128
        self.rgb_channels: int = 3

        self.alpha_branch_depth: int = 0
        self.alpha_branch_width: int = 128
        self.alpha_channels: int = 1

        self.norm = None
        self.activation = 'relu'
        self.skips = skip_connection_layer_list

        self.trunk_mlp = None
        if trunk_layer_num > 0:
            self.trunk_mlp = MLP(trunk_layer_num, in_feature, trunk_hidden_dim, hidden_activation=self.activation,
                                 hidden_norm=self.norm, skip_connection_layer_list=skip_connection_layer_list)
        self.rgb_mlp = None
        if rgb_layer_num > 0:
            self.rgb_mlp = MLP(rgb_layer_num, trunk_hidden_dim, rgb_hidden_dim, hidden_activation=self.activation, hidden_norm=self.norm,
                               out_feature=rgb_channels, skip_connection_layer_list=skip_connection_layer_list)
        self.alpha_mlp = None
        if alpha_layer_num > 0:
            self.alpha_mlp = MLP(alpha_layer_num, trunk_hidden_dim, alpha_hidden_dim, hidden_activation=self.activation,
                               hidden_norm=self.norm, out_feature=alpha_channels, skip_connection_layer_list=skip_connection_layer_list)

        self.bottleneck = None
        if (alpha_condition is not None) or (rgb_condition is not None):
            self.bottleneck = nn.Linear(trunk_hidden_dim, trunk_hidden_dim)

    def forward(self, x):
        """Multi-layer perception for nerf.

            Args:
              x: sample points with shape [batch, num_coarse_samples, feature].


            Returns:
              raw: [batch, num_coarse_samples, rgb_channels+alpha_channels].
        """
        feature_dim = x.shape[-1]
        num_samples = x.shape[1]
        x = x.reshape([-1, feature_dim])
        if self.trunk_mlp is not None:
            x = self.trunk_mlp(x)

        if (self.alpha_condition is not None) or (self.rgb_condition is not None):
            bottleneck = self.bottleneck(x)

            if self.alpha_condition is not None:
                alpha_condition = broadcast_condition(self.alpha_condition, num_samples)
                alpha_input = torch.cat([bottleneck, alpha_condition], dim=-1)
            else:
                alpha_input = x
            alpha = self.alpha_mlp(alpha_input)

            if self.rgb_condition is not None:
                  rgb_condition = broadcast_condition(self.rgb_condition, num_samples)
                  rgb_input = torch.cat([bottleneck, rgb_condition], dim=-1)
            else:
              rgb_input = x
            rgb = self.rgb_mlp(rgb_input)

        return {
            'rgb': rgb.reshape((-1, num_samples, self.rgb_channels)),
            'alpha': alpha.reshape((-1, num_samples, self.alpha_channels)),
        }


def broadcast_condition(c, num_samples):
    # Broadcast condition from [batch, feature] to
    # [batch, num_coarse_samples, feature] since all the samples along the
    # same ray has the same viewdir.
    c = torch.tile(c[:, None, :], (1, num_samples, 1))
    # Collapse the [batch, num_coarse_samples, feature] tensor to
    # [batch * num_coarse_samples, feature] to be fed into nn.Dense.
    c = c.reshape([-1, c.shape[-1]])
    return c
