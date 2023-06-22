import torch
import torch.nn as nn
from model_nerfies.modules import MLP

class SinusoidalEncoder(nn.Module):
    def __init__(self):
        super(SinusoidalEncoder, self).__init__()


class AnnealedSinusoidalEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(AnnealedSinusoidalEncoder, self).__init__()
        self.num_freqs = kwargs.num_freqs
        self.min_freq_log2 = kwargs.min_freq_log2
        self.max_freq_log2 = kwargs.max_freq_log2 # Optional
        self.scale = kwargs.scale
        self.use_identiy = kwargs.use_identity

    def cosine_easing_window(cls, min_freq_log2, max_freq_log2, num_bands, alpha):
        """Eases in each frequency one by one with a cosine.

        This is equivalent to taking a Tukey window and sliding it to the right
        along the frequency spectrum.

        Args:
        min_freq_log2: the lower frequency band.
        max_freq_log2: the upper frequency band.
        num_bands: the number of frequencies.
        alpha: will ease in each frequency as alpha goes from 0.0 to num_freqs.

        Returns:
        A 1-d numpy array with num_sample elements containing the window.
        """
        if max_freq_log2 is None:
            max_freq_log2 = num_bands - 1.0
        bands = torch.linspace(min_freq_log2, max_freq_log2, num_bands)
        x = torch.clip(alpha - bands, 0.0, 1.0)
        return 0.5 * (1 + torch.cos(torch.pi * x + torch.pi))

    def forward(self, x, alpha):
        if alpha is None:
            raise ValueError('alpha must be specified.')
        if self.num_freqs == 0:
            return x

        num_channels = x.shape[-1]

        base_encoder = SinusoidalEncoder(
            num_freqs=self.num_freqs,
            min_freq_log2=self.min_freq_log2,
            max_freq_log2=self.max_freq_log2,
            scale=self.scale,
            use_identity=self.use_identity)
        features = base_encoder(x)

        if self.use_identity:
            identity, features = torch.split(features, (x.shape[-1],), axis=-1)

        # Apply the window by broadcasting to save on memory.
        features = torch.reshape(features, (-1, 2, num_channels))
        window = self.cosine_easing_window(self.min_freq_log2, self.max_freq_log2, self.num_freqs, alpha)
        window = torch.reshape(window, (-1, 1, 1))
        features = window * features

        if self.use_identity:
            return torch.concat([identity, features.flatten()], axis=-1)
        else:
            return features.flatten()


class GloEncoder(nn.Module):
    """
        A GLO encoder module, which is just a thin wrapper around nn.Embed.

        Attributes:
        num_embeddings: The number of embeddings.
        embedding_dim: The dimensions of each embedding.
        [TBD] embedding_init: The initializer to use for each.
    """
    def __init__(self, num_embeddings, embedding_dim):        
        super(GloEncoder, self).__init__()
        self.embed = nn.Embed(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, x):
        return self.embed(x)


class TimeEncoder(nn.Module):
    def __init__(self, 
                 num_freqs,
                 mlp_args,
                #  mlp_depth, mlp_hidden_dim, mlp_output_dim, mlp_skips, mlp_hidden_activation, mlp_output_actvation,
                 ):
        self.num_freqs = num_freqs
        self.position_encoder = AnnealedSinusoidalEncoder(num_freqs=self.num_freqs)
        self.mlp =  MLP(**mlp_args)
    def forward(self, time, alpha=None):
        if alpha is None:
            alpha = self.num_freqs
        encoded_time = self.position_encoder(time, alpha)
        return self.mlp(encoded_time)