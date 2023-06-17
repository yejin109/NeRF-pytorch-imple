import torch
import torch.nn as nn
from model_hypernerf_torch import warping
from model_hypernerf_torch import _utils as utils
from model_hypernerf_torch.modules import HyperSheetMLP, NeRFMLP
from model_hypernerf_torch.sampler import sample_along_rays
from model_hypernerf_torch.rendering import render_samples
from model_hypernerf_torch.embed import GloEmbed


def get_model(model_cfg):
    model = HyperNeRF(**model_cfg)
    return model


class HyperNeRF(nn.Module):
    def __init__(self, 
                 use_viewdirs, positional_encoder_args,
                 use_nerf_embed, nerf_embed_args,
                 use_fine_samples, coarse_args, fine_args,
                 use_warp, warp_field_args, use_warp_jacobian,
                 hyper_slice_method, hyper_embed_args, hyper_use_warp_embed, hyper_sheet_args,

                 use_alpha_condition, 
                 ):
        super(HyperNeRF, self).__init__()
        self.positional_encoder_args = positional_encoder_args
        self.use_viewdirs = use_viewdirs

        self.use_nerf_embed = use_nerf_embed
        if use_nerf_embed:
            self.nerf_embedder = GloEmbed(**nerf_embed_args)

        self.warp_field = None
        self.use_warp = use_warp
        self.use_warp_jacobian = use_warp_jacobian
        if use_warp:
            self.warp_field = warping.SE3Field(**warp_field_args)

        self.hyper_slice_method = hyper_slice_method
        self.hyper_use_warp_embed = hyper_use_warp_embed
        if hyper_slice_method == 'axis_aligned_plane':
            self.hyper_embedder = GloEmbed(**hyper_embed_args)
        elif hyper_slice_method == 'bendy_sheet':
            if not hyper_use_warp_embed:
                self.hyper_embedder = GloEmbed(**hyper_embed_args)
            self.hyper_sheet_mlp = hyper_sheet_args(**hyper_sheet_args)
  
        self.use_alpha_condition = use_alpha_condition        
        
        self.mlps = {
            'coarse': NeRFMLP(coarse_args)
        }

        if use_fine_samples:
            self.mlps['fine'] = NeRFMLP(fine_args)

    def get_condition(self, viewdirs, metadata, is_metadata_encoded, nerf_embed_key):
        # TODO: nerf_embed_key?
        """Create the condition inputs for the NeRF template."""
        alpha_conditions = []
        rgb_conditions = []            

        # Point attribute predictions
        if self.use_viewdirs:
            viewdirs_embed = utils.posenc(viewdirs, **self.positional_encoder_args)
            rgb_conditions.append(viewdirs_embed)

        if self.use_nerf_embed:
            if is_metadata_encoded:
                nerf_embed = metadata['encoded_nerf']
            else:
                nerf_embed = metadata[nerf_embed_key]
                nerf_embed = self.nerf_embedder(nerf_embed)
            if self.use_alpha_condition:
                alpha_conditions.append(nerf_embed)
            if self.use_alpha_condition:
                rgb_conditions.append(nerf_embed)
        
       
        # The condition inputs have a shape of (B, C) now rather than (B, S, C)
        # since we assume all samples have the same condition input. We might want
        # to change this later.
        alpha_conditions = (torch.concat(alpha_conditions, axis=-1) if alpha_conditions else None)
        rgb_conditions = (torch.concat(rgb_conditions, axis=-1) if rgb_conditions else None)
        return alpha_conditions, rgb_conditions

    def forward(self, 
                # rays_dict
                rays_o, rays_d, viewdirs, metadata,
                warp_embed_key, hyper_embed_key,
                warp_params, is_metadata_encoded,
                return_points, return_weights,
                ray_sampling_args, rendering_args,    
                return_warp_jacobian,
                points_encoder_args   
                ):
        if viewdirs is None:
            viewdirs = rays_d

        out = {}
        if return_points:
            out['points'] = points
            
        # Ray Sampling : Coarse
        ray_sampling_args = dict(ray_sampling_args, **{'rays_o': rays_o, 'rays_d': rays_d})
        z_vals, points = sample_along_rays(**ray_sampling_args)

        alpha_conditions, rgb_conditions = self.get_condition(viewdirs, metadata, is_metadata_encoded)

        # Warp rays
        warp_embed = None
        if self.use_warp:
            if is_metadata_encoded:
                warp_embed = metadata['encoded_warp']
            else:
                warp_embed = metadata[warp_embed_key]
                warp_embed = self.warp_field(warp_embed)

        # Hyper Embedding
        hyper_embed = None
        if self.hyper_slice_method != 'none':
            if is_metadata_encoded:
                hyper_embed = metadata['encoded_hyper']
            elif self.hyper_use_warp_embed:
                hyper_embed = warp_embed
            else:
                hyper_embed = metadata[hyper_embed_key]
                hyper_embed = self.hyper_embedder(hyper_embed)
        
        if warp_embed is not None:
            warp_embed = torch.resahpe(warp_embed, (-1, warp_embed.size(-1)))
        
        if hyper_embed is not None:
            hyper_embed = torch.resahpe(hyper_embed, (-1, hyper_embed.size(-1)))
            
        warp_points, warp_jacobian = self.map_points(points, warp_embed, hyper_embed, warp_params,return_warp_jacobian)

        # Ray Colorization : Coarse
        coarse_ret = render_samples(
            self.mlps['coarse'], 
            warp_points, points_encoder_args, alpha_conditions, rgb_conditions,
            z_vals, rays_d, return_weights,
            **rendering_args
        )
        out['coarse'] = coarse_ret

        if self.num_fine_samples > 0:
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            # Ray Sampling : Fine --> Hierarchical sampling
            z_vals, points = utils.sample_pdf(z_vals_mid, coarse_ret['weights'][..., 1:-1],
                                        rays_o, rays_d, z_vals, self.num_fine_samples,
                                        rendering_args['use_stratified_sampling'])
            # Warp rays
            warp_embed = None
            if self.use_warp:
                if is_metadata_encoded:
                    warp_embed = metadata['encoded_warp']
                else:
                    warp_embed = metadata[warp_embed_key]
                    warp_embed = self.warp_field(warp_embed)

            # Hyper Embedding
            hyper_embed = None
            if self.hyper_slice_method != 'none':
                if is_metadata_encoded:
                    hyper_embed = metadata['encoded_hyper']
                elif self.hyper_use_warp_embed:
                    hyper_embed = warp_embed
                else:
                    hyper_embed = metadata[hyper_embed_key]
                    hyper_embed = self.hyper_embedder(hyper_embed)
            
            if warp_embed is not None:
                warp_embed = torch.resahpe(warp_embed, (-1, warp_embed.size(-1)))
            
            if hyper_embed is not None:
                hyper_embed = torch.resahpe(hyper_embed, (-1, hyper_embed.size(-1)))
                
            warp_points, warp_jacobian = self.map_points(points, warp_embed, hyper_embed, warp_params,return_warp_jacobian)

            # Ray Colorization : Fine
            fine_ret = render_samples(
                self.mlps['fine'], 
                warp_points, points_encoder_args, alpha_conditions, rgb_conditions,
                z_vals, rays_d, return_weights,
                **rendering_args
            )
            

        return coarse_ret, fine_ret
    
    def map_points(self, points, warp_embed, hyper_embed, extra_params,
                   return_warp_jacobian=False,
                   hyper_point_override=None):
        """
            Map input points to warped spatial and hyper points.

            Args:
                points: the input points to warp.
                warp_embed: the warp embeddings.
                hyper_embed: the hyper embeddings.
                extra_params: extra parameters to pass to the warp field/hyper field.
                use_warp: whether to use the warp or not.
                return_warp_jacobian: whether to return the warp jacobian or not.
                hyper_point_override: this may contain an override for the hyper points.
                Useful for rendering at specific hyper dimensions.

            Returns:
                A tuple containing `(warped_points, warp_jacobian)`.
        """
        use_warp = self.use_warp
        # Map input points to warped spatial and hyper points.
        spatial_points, warp_jacobian = self.map_spatial_points(
            points, warp_embed, extra_params, use_warp=use_warp,
            return_warp_jacobian=return_warp_jacobian)
        hyper_points = self.map_hyper_points(
            points, hyper_embed, extra_params,
            # Override hyper points if present in metadata dict.
            hyper_point_override=hyper_point_override)

        if hyper_points is not None:
            warped_points = torch.concat([spatial_points, hyper_points], dim=-1)
        else:
            warped_points = spatial_points

        return warped_points, warp_jacobian


    def map_hyper_points(self, points, hyper_embed, extra_params,
                    hyper_point_override=None):
        """Maps input points to hyper points.

        Args:
        points: the input points.
        hyper_embed: the hyper embeddings.
        extra_params: extra params to pass to the slicing MLP if applicable.
        hyper_point_override: this may contain an override for the hyper points.
            Useful for rendering at specific hyper dimensions.

        Returns:
        An array of hyper points.
        """
        if hyper_point_override is not None:
            hyper_points = torch.reshape(hyper_point_override, (*points.size(-1), hyper_point_override.size(-1)))
        elif self.hyper_slice_method == 'axis_aligned_plane':
            hyper_points = hyper_embed
        elif self.hyper_slice_method == 'bendy_sheet':
            hyper_points = self.hyper_sheet_mlp(points, hyper_embed, alpha=extra_params['hyper_sheet_alpha'])
        else:
            return None

        return hyper_points

    def map_spatial_points(self, points, warp_embed, extra_params, use_warp=True,
                                return_warp_jacobian=False):
        warp_jacobian = None
        if self.use_warp and use_warp:
            warp_out = self.warp_field(points, warp_embed, extra_params, return_warp_jacobian)
            if return_warp_jacobian:
                warp_jacobian = warp_out['jacobian']
            warped_points = warp_out['warped_points']
        else:
            warped_points = points

        return warped_points, warp_jacobian



def create_warp_field(model, num_batch_dims):
        return warping.create_warp_field(
            field_type=model.warp_field_type,
            num_freqs=model.num_warp_freqs,
            num_embeddings=model.num_warp_embeddings,
            num_features=model.num_warp_features,
            num_batch_dims=num_batch_dims,
            metadata_encoder_type=model.warp_metadata_encoder_type,
            **model.warp_kwargs)
        



