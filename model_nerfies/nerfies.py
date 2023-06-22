import torch
import torch.nn as nn
from model_nerfies import warping
from model_nerfies import _utils as utils
from model_nerfies.modules import NeRFMLP
from model_nerfies.sampler import sample_along_rays, sample_pdf
from model_nerfies.rendering import render_samples
from model_nerfies.embed import SinusoidalEncoder, GloEncoder


def get_model(model_cfg, render_cfg):
    mlp_args = {
        'mlp_trunk_args' : model_cfg['backbone']['trunk'],
        'mlp_rgb_args' : model_cfg['backbone']['rgb'],
        'mlp_alpha_args': model_cfg['bacbone']['alpha'],
        'use_rgb_condition': model_cfg['use_rgb_condition'],
        'use_alpha_condition': model_cfg['use_alpha_condition']
    }

    warp_field_args = {
        "field_type": model_cfg['warp']['warp_field_type'],
        'field_args': model_cfg['warp']['warp_field_args'],
        'num_batch_dims': 0,
    }

    _cfg = {
        "use_fine_samples": render_cfg['use_fine_samples'],
        "coarse_args": mlp_args,
        "fine_args": mlp_args,

        "use_warp": model_cfg['warp']['use_warp'],
        "warp_field_args": warp_field_args,
        "use_warp_jacobian": model_cfg['warp']['use_warp_jacobian'],

        "use_appearance_metadata": model_cfg['use_appearance_metadata'],
        "appearance_encoder_args": model_cfg['appearance_encoder_args'],

        "use_camera_metadata": model_cfg['use_camera_metadata'],
        "camera_encoder_args": camera_encoder_args,
        "use_trunk_condition": model_cfg['use_trunk_condition'],
        "use_alpha_condition": model_cfg['use_alpha_condition']
    }
    model = Nerfies(**_cfg)
    return model


class Nerfies(nn.Module):
    def __init__(self, use_viewdirs,
                 use_fine_samples, coarse_args, fine_args,
                 use_warp, warp_field_args, use_warp_jacobian,
                 use_appearance_metadata, appearance_encoder_args,
                 use_camera_metadata, camera_encoder_args,
                 use_trunk_condition, use_alpha_condition, 
                 ):
        super(Nerfies, self).__init__()

        self.use_viewdirs = use_viewdirs

        self.use_warp = use_warp
        self.use_warp_jacobian = use_warp_jacobian
        self.warp_field = None
        if use_warp:
            self.warp_field = create_warp_field(**warp_field_args)

        self.point_encoder = SinusoidalEncoder(
            # num_freqs
        )             
        self.viewdir_encoder = SinusoidalEncoder(
            # num_freqs
        )     
        self.use_trunk_condition = use_trunk_condition
        self.use_alpha_condition = use_alpha_condition        
        
        self.appearance_encoder = None
        if use_appearance_metadata:
            self.appearance_encoder = GloEncoder(**appearance_encoder_args)
        
        self.camera_encoder = None
        if use_camera_metadata:
            self.camera_encoder = GloEncoder(**camera_encoder_args)
        
        self.mlps = {'coarse': NeRFMLP(**coarse_args)}

        if use_fine_samples:
            self.mlps['fine'] = NeRFMLP(**fine_args)

    def get_condition(self, viewdirs, metadata, is_metadata_encoded):
        """Create the condition inputs for the NeRF template."""
        trunk_conditions = []
        alpha_conditions = []
        rgb_conditions = []            

        # Point attribute predictions
        if self.use_viewdirs:
            viewdirs_embed = self.viewdir_encoder(viewdirs)
            rgb_conditions.append(viewdirs_embed)

        if self.use_appearance_metadata:
            if is_metadata_encoded:
                appearance_code = metadata['appearance']
            else:
                appearance_code = self.appearance_encoder(metadata['appearance'])
            if self.use_trunk_condition:
                trunk_conditions.append(appearance_code)
            if self.use_alpha_condition:
                alpha_conditions.append(appearance_code)
            if self.use_alpha_condition:
                rgb_conditions.append(appearance_code)
        
        if self.use_camera_metadata:
            if is_metadata_encoded:
                camera_code = metadata['camera']
            else:
                camera_code = self.camera_encoder(metadata['camera'])
            rgb_conditions.append(camera_code)

        # The condition inputs have a shape of (B, C) now rather than (B, S, C)
        # since we assume all samples have the same condition input. We might want
        # to change this later.
        trunk_conditions = (torch.concat(trunk_conditions, dim=-1) if trunk_conditions else None)
        alpha_conditions = (torch.concat(alpha_conditions, dim=-1) if alpha_conditions else None)
        rgb_conditions = (torch.concat(rgb_conditions, dim=-1) if rgb_conditions else None)
        return trunk_conditions, alpha_conditions, rgb_conditions

    def forward(self, 
                rays_o, rays_d, viewdirs, metadata,
                warp_params, is_metadata_encoded,
                return_points, return_weights, return_warp_jacobian,
                deterministic,
                ray_sampling_args, rendering_args,           
                ):
        if viewdirs is None:
            viewdirs = rays_d

        out = {}
        if return_points:
            out['points'] = points
            
        # Ray Sampling : Coarse
        ray_sampling_args = dict(ray_sampling_args, **{'rays_o': rays_o, 'rays_d': rays_d})
        z_vals, points = sample_along_rays(**ray_sampling_args)

        trunk_conditions, alpha_conditions, rgb_conditions = self.get_condition(viewdirs, metadata, is_metadata_encoded)

        # Warp rays
        if self.use_warp:
            metadata_channels = self.num_warp_features if is_metadata_encoded else 1
            warp_metadata = (
                metadata['time']
                if self.warp_metadata_encoder_type == 'time' else metadata['warp'])
            warp_metadata = torch.reshape(warp_metadata, (points.size()[:2], metadata_channels))
            warp_out = self.warp_field(
                points,
                warp_metadata,
                warp_params,
                self.use_warp_jacobian,
                is_metadata_encoded)
            points = warp_out['warped_points']
        if 'jacobian' in warp_out:
            out['warp_jacobian'] = warp_out['jacobian']
        if return_points:
            out['warped_points'] = warp_out['warped_points']
        
        points_embed = self.point_encoder(points)

        # Ray Colorization : Coarse
        coarse_ret = render_samples(
            self.mlps['coarse'], 
            points_embed, trunk_conditions, alpha_conditions, rgb_conditions,
            z_vals, rays_d, return_weights,
            **rendering_args
        )
        out['coarse'] = coarse_ret

        if self.num_fine_samples > 0:
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            # Ray Sampling : Fine --> Hierarchical sampling
            z_vals, points = sample_pdf(z_vals_mid, coarse_ret['weights'][..., 1:-1],
                                        rays_o, rays_d, z_vals, self.num_fine_samples,
                                        rendering_args['use_stratified_sampling'])
            
            # Ray Colorization : Fine
            fine_ret = render_samples(
                self.mlps['fine'],
                points_embed, trunk_conditions, alpha_conditions, rgb_conditions,
                z_vals, rays_d, return_weights,
                **rendering_args
            )

        return coarse_ret, fine_ret


def create_warp_field(warp_field_type, field_args, num_batch_dims):
    return warping.create_warp_field(
        field_type=warp_field_type,
        field_args=field_args,
        num_batch_dims=num_batch_dims,
        # num_freqs=model.num_warp_freqs,
        # num_embeddings=model.num_warp_embeddings,
        # num_features=model.num_warp_features,
        # metadata_encoder_type=model.warp_metadata_encoder_type,
        )




