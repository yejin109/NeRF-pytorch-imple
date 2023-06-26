import torch
import torch.nn as nn
from model_hypernerf import _utils as model_utils
from model_hypernerf.modules import MLP


class GloEmbed(nn.Module):
    """
        A GLO encoder module, which is just a thin wrapper around nn.Embed.

        Attributes:
        num_embeddings: The number of embeddings.
        embedding_dim: The dimensions of each embedding.
        [TBD] embedding_init: The initializer to use for each.
    """
    def __init__(self, num_embeddings, embedding_dim):        
        super(GloEmbed, self).__init__()
        self.embed = nn.Embed(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, x):
        return self.embed(x)
    

# class GLOEmbed(nn.Module):
#     """A GLO encoder module, which is just a thin wrapper around nn.Embed.

#     Attributes:
#         num_embeddings: The number of embeddings.
#         num_dims: The dimensions of each embedding.
#         embedding_init: The initializer to use for each.
#     """
#     def __init__(self, num_embeddings, num_dims):
#         super(GLOEmbed, self).__init__()
#         self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=num_dims)
#         # TODO 이거 해야 함.
#         # nn.init.uniform()

#     def forward(self, inputs):
#         """Method to get embeddings for specified indices.

#         Args:
#           inputs: The indices to fetch embeddings for.

#         Returns:
#           The embeddings corresponding to the indices provided.
#         """
#         if inputs.shape[-1] == 1:
#             inputs = torch.squeeze(inputs, dim=-1)

#         return self.embed(inputs)


class HyperSheetMLP(nn.Module):
    """An MLP that defines a bendy slicing surface through hyper space."""
    def __init__(self, out_feature, min_deg, max_deg, layer_num, hidden_dim, skips):
        super(HyperSheetMLP, self).__init__()
        # TODO infeature 계산해야 함
        in_feature = 10
        self.mlp = MLP(layer_num, in_feature, hidden_dim, out_feature=out_feature, skip_connection_layer_list=skips)
        self.min_deg = min_deg
        self.max_deg = max_deg

        # TODO initialization 해야 함
        # hidden_init: types.Initializer = jax.nn.initializers.glorot_uniform()
        # output_init: types.Initializer = jax.nn.initializers.normal(1e-5)

        use_residual: bool = False

    def forward(self, points, embed, alpha=None):
        # sinusoid featurization
        points_feat = model_utils.posenc(points, self.min_deg, self.max_deg, alpha=alpha)
        inputs = torch.concatenate([points_feat, embed], dim=-1)

        if self.use_residual:
            return self.mlp(inputs) + embed
        else:
            return self.mlp(inputs)
