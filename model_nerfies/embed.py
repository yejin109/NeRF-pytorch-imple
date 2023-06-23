import torch
import torch.nn as nn
from model_nerfies.modules import MLP
from model_nerfies._utils import get_value


class SinusoidalEncoder(nn.Module):
    def __init__(self, num_freqs, min_freq_log2, max_freq_log2, scale, use_identity):
        super(SinusoidalEncoder, self).__init__()
        self.num_freqs = num_freqs
        self.min_freq_log2 = min_freq_log2
        self.max_freq_log2 = get_value(max_freq_log2)
        self.scale = scale
        self.use_identity = use_identity

        if self.max_freq_log2 is None:
            max_freq_log2 = self.num_freqs - 1.0
        else:
            max_freq_log2 = self.max_freq_log2
        self.freq_bands = 2.0 ** torch.linspace(self.min_freq_log2, max_freq_log2, int(self.num_freqs))

        self.freqs = torch.reshape(self.freq_bands, (self.num_freqs, 1))

    def forward(self, x, alpha):
        """A vectorized sinusoidal encoding.

        Args:
          x: the input features to encode.
          alpha: a dummy argument for API compatibility.

        Returns:
          A tensor containing the encoded features.
        """
        if self.num_freqs == 0:
            return x

        x_expanded = torch.unsqueeze(x, -1)  # (1, C).
        # Will be broadcasted to shape (F, C).
        angles = self.scale * x_expanded * self.freqs

        # The shape of the features is (F, 2, C) so that when we reshape it
        # it matches the ordering of the original NeRF code.
        # Vectorize the computation of the high-frequency (sin, cos) terms.
        # We use the trigonometric identity: cos(x) = sin(x + pi/2)
        features = torch.stack((angles, angles + torch.pi / 2), dim=-2)
        features = features.flatten()
        features = torch.sin(features)

        # Prepend the original signal for the identity.
        if self.use_identity:
            features = torch.concat([x, features], dim=-1)
        return features


class AnnealedSinusoidalEncoder(nn.Module):
    def __init__(self, num_freqs, min_freq_log2, max_freq_log2, scale, use_identity, **kwargs):
        super(AnnealedSinusoidalEncoder, self).__init__()
        self.num_freqs = num_freqs
        self.min_freq_log2 = min_freq_log2
        self.max_freq_log2 = get_value(max_freq_log2) # Optional
        self.scale = scale
        self.use_identiy = use_identity

        self.base_encoder = SinusoidalEncoder(
            num_freqs=self.num_freqs,
            min_freq_log2=self.min_freq_log2,
            max_freq_log2=self.max_freq_log2,
            scale=self.scale,
            use_identity=self.use_identity)

    @staticmethod
    def cosine_easing_window(min_freq_log2, max_freq_log2, num_bands, alpha):
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

        features = self.base_encoder(x)

        if self.use_identity:
            identity, features = torch.split(features, [x.size(-1), ], dim=-1)

        # Apply the window by broadcasting to save on memory.
        features = torch.reshape(features, (-1, 2, num_channels))
        window = self.cosine_easing_window(self.min_freq_log2, self.max_freq_log2, self.num_freqs, alpha)
        window = torch.reshape(window, (-1, 1, 1))
        features = window * features

        if self.use_identity:
            return torch.concat([identity, features.flatten()], dim=-1)
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
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, x):
        return self.embed(x)


class TimeEncoder(nn.Module):
    def __init__(self, num_freqs, min_freq_log2, max_freq_log2, scale, use_identity, mlp_args):
        super(TimeEncoder, self).__init__()
        self.num_freqs = num_freqs
        self.position_encoder = AnnealedSinusoidalEncoder(num_freqs, min_freq_log2, max_freq_log2, scale, use_identity)
        self.mlp = MLP(**mlp_args)

    def forward(self, time, alpha=None):
        if alpha is None:
            alpha = self.num_freqs
        encoded_time = self.position_encoder(time, alpha)
        return self.mlp(encoded_time)
