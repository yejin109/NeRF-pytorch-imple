import torch


class Embedder:
    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.kwargs['periodic_fns'] = [torch.sin, torch.cos]

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            # freq_bands = 2. ** tf.linspace(0., max_freq, N_freqs)
            freq_bands = 2.**torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim=-1)


def get_embedder(embed_cfg, multires, i=0):
    """
    Original Name : get_embedder
    """
    if i == -1:
        # TODO:  original code used tf so that both var type was tf.Tensor and temporarily using numpy ndarray.
        # return tf.identity, 3
        return torch.nn.Identity, 3

    embed_kwargs = dict(embed_cfg, **{
        'max_freq_log2': multires-1,
        'num_freqs': multires
    })

    return Embedder(**embed_kwargs)
