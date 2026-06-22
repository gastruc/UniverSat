import numpy as np
import torch
from torch import nn as nn


class MPFourier(nn.Module):
    """Multiplicative-Fourier feature embedding (EDM2-style) with a learned projection.

    Lifts a scalar input (e.g. a wavelength or resolution) carried in a trailing
    singleton dimension to ``num_channels`` features: a frozen random cosine
    basis of ``num_channels // 4`` features, followed by a learned linear
    projection up to ``num_channels``.

    The random basis (``freqs``, ``phases``) is **frozen** (registered as
    buffers, not parameters). Uniform phases give ``E[cos^2] = 1/2``, so the
    ``sqrt(2)`` factor yields unit-variance cosine features; keeping the basis
    fixed preserves that magnitude-preserving property, which training the basis
    would let drift. The learned ``proj`` head then mixes those features up to
    ``num_channels``.
    """

    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels // 4) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels // 4))
        self.proj = nn.Linear(num_channels // 4, num_channels)

    def forward(self, x):
        original_dtype = x.dtype
        # x has a trailing singleton dim; broadcast it against the (num_channels // 4,)
        # basis so the encoded features become the new last dimension, then project up.
        y = x.to(torch.float32) * self.freqs.to(torch.float32) + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return self.proj(y.to(original_dtype))
