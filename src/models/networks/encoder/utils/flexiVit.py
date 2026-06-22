# List of flexible module inspired from FLexiVit and OlmoEarth
import math

import torch
import torch.nn as nn
from einops import rearrange, repeat


class FlexiViTLinear(nn.Module):
    """Frozen random-linear projection over a resolution-agnostic patch.

    Each ``(B, N, H*W*C)`` patch is reshaped to an image, bilinearly resampled to
    a fixed ``HW`` grid, then mapped to ``out_channels`` by a frozen orthogonal
    linear layer — so patches of any spatial size share one target projection.
    """

    def __init__(self, HW, in_channels, out_channels):
        super().__init__()

        self.HW = HW
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels*HW*HW, out_channels, bias=False)
        nn.init.orthogonal_(self.linear.weight)

    def forward(self, x):
        assert x.dim() == 3, f"Input tensor must be BxNxHWC, got a {x.dim()}D tensor"
        # x is (B, N, HWC)
        B, N, HWC = x.shape
        H = W = math.sqrt(HWC // self.in_channels)
        assert H.is_integer(), f"Input spatial dimensions {H}x{W} are not square integers"
        H = W = int(H)

        x = rearrange(x, 'b n (h w c) -> (b n) c h w', h=H, w=W, c=self.in_channels)
        # reshaped x to (B, H, W, C)
        x = torch.nn.functional.interpolate(x, size=(self.HW, self.HW), mode='bilinear', align_corners=False)
        x = rearrange(x, '(b n) c h w -> b n (h w c)', b=B, n=N)

        x = self.linear(x)

        return x

class FlexiViTTemporel(nn.Module):
    """Temporal variant of :class:`FlexiViTLinear`.

    Projects each patch to ``nb_times`` reference embeddings, then interpolates
    them at each timestamp's day-of-year before averaging over time.
    """

    def __init__(self, HW, in_channels, out_channels, nb_times=12):
        super().__init__()

        self.HW = HW
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nb_times = nb_times
        self.linear = nn.Linear(in_channels*HW*HW, nb_times*out_channels, bias=False)
        nn.init.orthogonal_(self.linear.weight)

    def forward(self, x, dates):
        assert x.dim() == 4, f"Input tensor must be BxNxTxHWC, got a {x.dim()}D tensor"
        # x is (B, N, T, HWC)
        B, N, T, HWC = x.shape
        H = W = math.sqrt(HWC // self.in_channels)
        assert H.is_integer(), f"Input spatial dimensions {H}x{W}x{self.in_channels} are not square integers"
        H = W = int(H)

        x = rearrange(x, 'b n t (h w c) -> (b n t) c h w', h=H, w=W, c=self.in_channels)
        # reshaped x to (B, H, W, C)
        x = torch.nn.functional.interpolate(x, size=(self.HW, self.HW), mode='bilinear', align_corners=False)
        x = rearrange(x, '(b n t) c h w -> b n t (h w c)', b=B, t=T, n=N)

        x = self.linear(x)
        x = rearrange(x, 'b n t (tref d) -> b n t tref d', tref=self.nb_times, d=self.out_channels)

        if not dates is None:
            #compute closest reference dates
            dates = dates/366 * self.nb_times # B,T
            dates_inf = torch.floor(dates) % self.nb_times
            dates_sup = (dates_inf + 1) % self.nb_times
            dates_remainder = repeat(dates-dates_inf, ' b t -> b n t d', n=N, d=self.out_channels)

            x_inf = x.gather(3, repeat(dates_inf, ' b t -> b n t tref d', n=N, tref=1, d=self.out_channels).long()).squeeze(3) #B,N,T,D
            x_sup = x.gather(3, repeat(dates_sup, ' b t -> b n t tref d', n=N, tref=1, d=self.out_channels).long()).squeeze(3) #B,N,T,D
            x = x_inf * (1 - dates_remainder) + x_sup * dates_remainder #B,N,T,D
        else:
            x = x.mean(dim=3)  # B,N,T,D

        return x.mean(dim=2)  # B,N,D
