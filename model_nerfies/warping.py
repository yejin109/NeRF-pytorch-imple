import torch
import torch.nn as nn
from model_nerfies.modules import MLP
from model_nerfies.embed import AnnealedSinusoidalEncoder, GloEncoder, TimeEncoder
from model_nerfies._utils import exponential_se3, from_homogenous, to_homogenous, get_value


def create_warp_field(field_type, field_args, num_batch_dims):
    if field_type == 'translation':
        warp_field = TranslationField(**field_args)
    elif field_type == 'se3':
        warp_field = SE3Field(**field_args)
    else:
        raise ValueError(f'Unknown warp field type : {field_type}')
    
    if num_batch_dims > 0:
        # TODO : 아직 vectorize 지원 못함
        assert False, "Cannot Suuport vectorize"
    return warp_field


class TranslationField(nn.Module):
    def __init__(self, 
                 points_encoder_args, #  num_freqs, min_freq_log2, max_freq_log2, scale, use_identity, 
                 metadata_encoder_type, glo_encoder_args, time_encoder_args,
                 mlp_depth, mlp_hidden_dim, mlp_output_dim, mlp_skips, mlp_hidden_activation, mlp_output_actvation,
                 **kwargs):
        super(TranslationField, self).__init__()

        self.points_encoder = AnnealedSinusoidalEncoder(
            # {'num_freqs': num_freqs, 'min_freq_log2':min_freq_log2, 'max_freq_log2': max_freq_log2, 'scale': scale, 'use_identity': use_identity}
            **points_encoder_args
            )
        if metadata_encoder_type == 'glo':
            self.metadata_encoder = GloEncoder(**glo_encoder_args)
                # num_embeddings=self.num_embeddings,
                # features=self.num_embedding_features
        elif metadata_encoder_type == 'time':
            self.metadata_encoder = TimeEncoder(**time_encoder_args)
                # num_freqs=self.metadata_encoder_num_freqs,
                # features=self.num_embedding_features)
        elif metadata_encoder_type == 'blend':
            self.glo_encoder = GloEncoder(**glo_encoder_args)
                # num_embeddings=self.num_embeddings,
                # features=self.num_embedding_features)
            self.time_encoder = TimeEncoder(**time_encoder_args)
                # num_freqs=self.metadata_encoder_num_freqs,
                # features=self.num_embedding_features)
        else:
            raise ValueError(f'Unknown metadata encoder type {self.metadata_encoder_type}')

        self.mlp_output_dim = mlp_output_dim
        assert self.mlp_output_dim == 3
        assert mlp_hidden_activation is None
        assert mlp_output_actvation is None
        # TODO: in_feature?!
        self.mlp = MLP(
            in_feature=None,
            depth=mlp_depth,
            hidden_dim=mlp_hidden_dim,
            skips = mlp_skips,
            output_feature=mlp_output_dim,
            hidden_activation=mlp_hidden_activation,
            output_activation=mlp_output_actvation
        )

    def encode_metadata(self, metadata, time_alpha):
        if self.metadata_encoder_type == 'time':
            metadata_embed = self.metadata_encoder(metadata, time_alpha)  
        elif self.metadata_encoder_type == 'blend':
            glo_embed = self.glo_encoder(metadata)
            time_embed = self.time_encoder(metadata)
            metadata_embed = ((1.0 - time_alpha) * glo_embed +
                                time_alpha * time_embed)
        elif self.metadata_encoder_type == 'glo':
            metadata_embed = self.metadata_encoder(metadata)
        else:
            raise RuntimeError(f'Unknown metadata encoder type {self.metadata_encoder_type}')
        
        return metadata_embed
    
    def warp(self,points, metadata_embed, extra: dict):
        points_embed = self.points_encoder(points, alpha=extra['alpha'])
        inputs = torch.concat([points_embed, metadata_embed], dim=-1)
        translation = self.mlp(inputs)
        warped_points = points + translation

        return warped_points
            
    def forward(self, points, metadata, extra, return_jacobian, is_metadata_encoded):
        """
            Warp the given points using a warp field.

            Args:
            points: the points to warp.
            metadata: metadata indices if is_metadata_encoded is False else pre-encoded
                metadata.
            extra: extra parameters used in the warp field e.g., the warp alpha.
            return_jacobian: if True compute and return the Jacobian of the warp.
            is_metadata_encoded: if True assumes the metadata is already encoded.

            Returns:
            The warped points and the Jacobian of the warp if `return_jacobian` is
                True.
        """
        if is_metadata_encoded:
            metadata_embed = metadata
        else:
            metadata_embed = self.encode_metadata(metadata, extra.get('time_alpha'))

        out = {'warped_points': self.warp(points, metadata_embed, extra)}

        if return_jacobian:
            # TODO : 현재 jax에서는 jacobian을 구하고 torch에서도 autograd모듈에서 지원하긴 한다. 다만 구현을 확인해야 함
            assert False, "Cannot support jacobian "                
            jac_fn = jax.jacfwd(lambda *x: self.warp(*x)[..., :3], argnums=0)
            out['jacobian'] = jac_fn(points, metadata_embed, extra)            

        return out            


class SE3Field(nn.Module):
    def __init__(self, points_encoder_args,
                 metadata_encoder_type, glo_encoder_args, time_encoder_args,
                 mlp_trunk_args, mlp_branch_w_args, mlp_branch_v_args,
                 use_pivot, mlp_branch_p_args, 
                 use_translation, mlp_branch_t_args
                 ):
        super(SE3Field, self).__init__()

        self.points_encoder = AnnealedSinusoidalEncoder(
            **points_encoder_args
            )
        if metadata_encoder_type == 'glo':
            self.metadata_encoder = GloEncoder(**glo_encoder_args)
                # num_embeddings=self.num_embeddings,
                # features=self.num_embedding_features
        elif metadata_encoder_type == 'time':
            self.metadata_encoder = TimeEncoder(**time_encoder_args)
        else:
            raise ValueError(f'Unknown metadata encoder type {self.metadata_encoder_type}')
        self.trunk = MLP(**mlp_trunk_args)        
        self.branches = {
            'w': MLP(**mlp_branch_w_args),
            'v': MLP(**mlp_branch_v_args)
        }
        if use_pivot:
            self.branches['p'] = MLP(**mlp_branch_p_args)
        if use_translation:
            self.branches['t'] = MLP(**mlp_branch_t_args)
    
    def encode_metadata(self, metadata, time_alpha):
        if self.metadata_encoder_type == 'time':
            metadata_embed = self.metadata_encoder(metadata, time_alpha)
        elif self.metadata_encoder_type == 'glo':
            metadata_embed = self.metadata_encoder(metadata)
        else:
            raise RuntimeError(f'Unknown metadata encoder type {self.metadata_encoder_type}')

        return metadata_embed        
    
    def warp(self, points, metadata_embed, extra):
        points_embed = self.points_encoder(points, alpha=extra.get('alpha'))
        inputs = torch.concat([points_embed, metadata_embed], dim=-1)
        trunk_output = self.trunk(inputs)

        w = self.branches['w'](trunk_output)
        v = self.branches['v'](trunk_output)
        theta = torch.linalg.norm(w, dim=-1)
        w = w / theta.unsqueeze(-1)
        v = v / theta.unsqueeze(-1)
        screw_axis = torch.concat([w, v], dim=-1)
        transform = exponential_se3(screw_axis, theta)        
        
        warped_points = points
        if self.use_pivot:
            pivot = self.branches['p'](trunk_output)
            warped_points = warped_points + pivot

        warped_points = from_homogenous(transform @ to_homogenous(warped_points))

        if self.use_pivot:
            warped_points = warped_points - pivot

        if self.use_translation:
            t = self.branches['t'](trunk_output)
            warped_points = warped_points + t

        return warped_points
    
    def forward(self, points, metadata, extra, return_jacobian, is_metadata_encoded):
        """
            Warp the given points using a warp field.

            Args:
            points: the points to warp.
            metadata: metadata indices if metadata_encoded is False else pre-encoded
                metadata.
            extra: A dictionary containing
                'alpha': the alpha value for the positional encoding.
                'time_alpha': the alpha value for the time positional encoding
                (if applicable).
            return_jacobian: if True compute and return the Jacobian of the warp.
            metadata_encoded: if True assumes the metadata is already encoded.

            Returns:
            The warped points and the Jacobian of the warp if `return_jacobian` is
                True.
        """
        if is_metadata_encoded:
            metadata_embed = metadata
        else:
            metadata_embed = self.encode_metadata(metadata, extra.get('time_alpha'))

        out = {'warped_points': self.warp(points, metadata_embed, extra)}

        if return_jacobian:
            # TODO : 현재 jax에서는 jacobian을 구하고 torch에서도 autograd모듈에서 지원하긴 한다. 다만 구현을 확인해야 함
            assert False, "Cannot support jacobian "                
            # jac_fn = jax.jacfwd(self.warp, argnums=0)
            # out['jacobian'] = jac_fn(points, metadata_embed, extra)

        return out

