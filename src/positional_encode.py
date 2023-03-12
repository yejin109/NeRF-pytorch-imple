import torch


class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

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
        # return tf.concat([fn(inputs) for fn in self.embed_fns], -1)
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim=-1)


def get_embedder(multires, i=0):
    if i == -1:
        # TODO:  original code used tf so that both var type was tf.Tensor and temporarily using numpy ndarray.
        # return tf.identity, 3
        return torch.nn.Identity, 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        # 'periodic_fns': [tf.math.sin, tf.math.cos],
        'periodic_fns': [torch.sin, torch.cos],
    }
    if __name__ == '__main__': pprint.pprint(embed_kwargs)

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


if __name__ == '__main__':
    import pprint

    embed_func, out_dim = get_embedder(5)
    sample_input = torch.Tensor([[1, 2, 3]])
    emb = embed_func(sample_input)
    print(f'Out Dim : {out_dim}')

    print(f'Sample Input : {sample_input}')
    print(f'Embedding : \n {emb}')
