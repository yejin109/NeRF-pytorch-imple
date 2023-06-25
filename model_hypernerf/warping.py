import torch
import torch.nn as nn
from model_nerfies.modules import MLP
from model_nerfies.embed import AnnealedSinusoidalEncoder, GloEncoder, TimeEncoder
from model_nerfies._utils import exponential_se3, from_homogenous, to_homogenous, posenc


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
                 points_encoder_args,
                 mlp_depth, mlp_hidden_dim, mlp_output_dim, mlp_skips, mlp_hidden_activation, mlp_output_actvation,
                 **kwargs):
        super(TranslationField, self).__init__()
        self.points_encoder_args = points_encoder_args
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

    def warp(self,points, metadata_embed, extra: dict):
        # Nerfie version :
        # points_embed = self.points_encoder(points, alpha=extra.get('alpha'))
        points_embed = posenc(points, **self.points_encoder_args)
        inputs = torch.concat([points_embed, metadata_embed], dim=-1)
        translation = self.mlp(inputs)
        warped_points = points + translation

        return warped_points
            
    def forward(self, points, metadata_embed, extra, return_jacobian):
        """
            Warp the given points using a warp field.

            Args:
            points: the points to warp.
            metadata_embed: metadata indices if is_metadata_encoded is False else pre-encoded
                metadata.
            extra: extra parameters used in the warp field e.g., the warp alpha.
            return_jacobian: if True compute and return the Jacobian of the warp.

            Returns:
            The warped points and the Jacobian of the warp if `return_jacobian` is
                True.
        """

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

        self.points_encoder_args = points_encoder_args

        self.trunk = MLP(**mlp_trunk_args)        
        self.branches = {
            'w': MLP(**mlp_branch_w_args),
            'v': MLP(**mlp_branch_v_args)
        }      
    
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
        warped_points = from_homogenous(transform @ to_homogenous(warped_points))

        return warped_points
    
    def forward(self, points, metadata_embed, extra, return_jacobian):
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

            Returns:
            The warped points and the Jacobian of the warp if `return_jacobian` is
                True.
        """
        out = {'warped_points': self.warp(points, metadata_embed, extra)}

        if return_jacobian:
            # TODO : 현재 jax에서는 jacobian을 구하고 torch에서도 autograd모듈에서 지원하긴 한다. 다만 구현을 확인해야 함
            assert False, "Cannot support jacobian "                
            # jac_fn = jax.jacfwd(self.warp, argnums=0)
            # out['jacobian'] = jac_fn(points, metadata_embed, extra)

        return out

