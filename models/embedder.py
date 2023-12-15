import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# code adapted from NeRF https://github.com/bmild/nerf and https://github.com/tancik/fourier-feature-networks

class Embedder(nn.Module):
    def __init__(self, num_freqs, max_freq_log2, include_input=True, log_sampling=True, periodic_fns=[torch.sin, torch.cos]):
        super(Embedder, self).__init__()

        embed_fns = []
        if include_input:
            embed_fns.append(lambda x : x)

        max_freq = max_freq_log2
        N_freqs = num_freqs

        if log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))

        self.embed_fns = embed_fns

    def forward(self, x):
        return torch.cat([fn(x) for fn in self.embed_fns], -1)


class FourierEmbedder(nn.Module):
    def __init__(self, mapping_size, input_size=3, sigma=10):
        super(FourierEmbedder, self).__init__()

        B = torch.randn(input_size, mapping_size)
        self.register_buffer('B', B)

    def forward(self, x):
        x_proj = torch.matmul((2 * np.pi * x), self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)


if __name__ == "__main__":
    embedder = Embedder(5, 4).cuda()
    i = torch.randn(2, 5, 3).cuda()

    out = embedder(i)
    print(out.shape)