import collections
from itertools import repeat
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final

from models.networks.encoder.utils.pos_embed import get_coords
from models.networks.encoder.utils.utils import DropPath, matching_grids


# Maximum batch size per scaled_dot_product_attention call. The fused
# (flash / memory-efficient) kernels have a grid limit on the batch dimension;
# above it we chunk and keep using the fused kernel rather than a slow fallback.
_SDPA_MAX_BATCH = 2**15


@torch.compiler.disable
def fused_sdpa(q, k, v, attn_mask=None):
    """Fastest available PyTorch attention.

    Delegates to ``F.scaled_dot_product_attention``, which auto-selects the
    fused (flash / memory-efficient) kernel. When the batch dimension exceeds
    the flash kernel's grid limit, the call is chunked so large (axial-flattened)
    batches stay on a fused kernel instead of a manual-softmax fallback.
    ``attn_mask`` (if given) is chunked along the batch dimension too.
    """
    B = q.shape[0]
    if B <= _SDPA_MAX_BATCH:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out = []
    for i in range(0, B, _SDPA_MAX_BATCH):
        m = attn_mask[i:i + _SDPA_MAX_BATCH] if attn_mask is not None else None
        out.append(F.scaled_dot_product_attention(
            q[i:i + _SDPA_MAX_BATCH], k[i:i + _SDPA_MAX_BATCH], v[i:i + _SDPA_MAX_BATCH], attn_mask=m))
    return torch.cat(out, dim=0)


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            n_modalities=1,
            n_registers=1,
            RoPe_2D=False,
            gating=False,
        ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.gating = gating
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.n_modalities = n_modalities
        self.n_registers = n_registers
        self.q_norm = norm_layer(dim // num_heads)
        self.k_norm = norm_layer(dim // num_heads)

        if self.gating:
            self.wq = nn.Linear(dim, dim * 2, bias=qkv_bias)
        else:
            self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if RoPe_2D:
            self.rope = Rope2D(self.head_dim)
        else:
            self.rope = Rope1D(self.head_dim, max_seq_len=750)
        self.RoPe_2D = RoPe_2D

        self._device_weight_init()

    def _device_weight_init(self):
        nn.init.kaiming_normal_(self.wq.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.wk.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.wv.weight, nonlinearity='relu')

    def forward(self, x, coords):
        B, N, C = x.shape

        if self.gating:
            q = self.wq(x).view(B, N, self.num_heads, -1)
            q, gate_score = torch.split(q, [C // self.num_heads, C // self.num_heads], dim=-1)
            gate_score = gate_score.reshape(B, N, -1, C // self.num_heads)
            q = q.reshape(B, N, -1, C // self.num_heads)
        else:
            q = self.wq(x).reshape(B, N, self.num_heads, C // self.num_heads)
        
        q = self.q_norm(q)
        
        k = self.k_norm(self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads))
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if coords is not None:
            if self.RoPe_2D:
                k = self.rope(k.permute(0, 2, 1, 3), coords[:, None, :, :])
                q = self.rope(q.permute(0, 2, 1, 3), coords[:, None, :, :])
            else:
                k = self.rope(k, input_pos=coords).permute(0, 2, 1, 3)
                q = self.rope(q, input_pos=coords).permute(0, 2, 1, 3)
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)

        x = fused_sdpa(q, k, v)

        x = x.transpose(1, 2).contiguous()
        if self.gating:
            x = x * torch.sigmoid(gate_score)

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            n_modalities=1,
            n_registers=1,
            RoPe_2D=True,
            gating=False
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            n_modalities=n_modalities,
            n_registers=n_registers,
            RoPe_2D=RoPe_2D,
            gating=gating
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_1 = nn.Parameter(1e-6 * torch.ones(dim)) # On commence très petit
        self.gamma_2 = nn.Parameter(1e-6 * torch.ones(dim)) # On commence très petit

    def forward(self, x, coords=None):
        x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), coords))
        x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class ACAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            n_queries=1,
            expand_dim=1,
            max_seq_len=750,
            RoPe=None,
            gating=False
        ):
        super().__init__()
        assert dim % num_heads == 0, f'dim({dim}) should be divisible by num_heads({num_heads})'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.expand_dim = expand_dim
        self.scale = (self.head_dim * expand_dim) ** -0.5
        self.n_queries = n_queries
        self.gating = gating

        self.q_norm = norm_layer((dim * expand_dim) // num_heads)
        self.k_norm = norm_layer((dim * expand_dim) // num_heads)

        if self.gating:
            self.wq = nn.Linear(dim * 2, dim * expand_dim * 2, bias=True)
        else:
            self.wq = nn.Linear(dim * 2, dim * expand_dim, bias=True)
        self.wk = nn.Linear(dim, dim * expand_dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim * expand_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        if RoPe == "2D":
            self.rope = Rope2D(self.head_dim * expand_dim)
        elif RoPe == "1D":
            self.rope = Rope1D(self.head_dim * expand_dim, max_seq_len=max_seq_len)
        else:
            self.rope = None
        self.RoPe = RoPe

        self._device_weight_init()

    def _device_weight_init(self):
        nn.init.kaiming_normal_(self.wq.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.wk.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.wv.weight, nonlinearity='relu')

    @torch.compiler.disable
    def get_qkv(self, x):
        B, N, C = x.shape
        C = C * self.expand_dim
        if N == 1: # if there is only one patch, we don't need to attend
            return self.wv(x)
        
        q = torch.cat([torch.mean(x, dim=1).unsqueeze(1), torch.max(x, dim=1).values.unsqueeze(1)], dim=2)
        
        if self.gating:
            q = self.wq(q).view(B, self.n_queries, self.num_heads, -1)
            q, gate_score = torch.split(q, [C // self.num_heads, C // self.num_heads], dim=-1)
            gate_score = gate_score.reshape(B, self.n_queries, self.num_heads, C // self.num_heads)
            q = q.reshape(B, self.n_queries, self.num_heads, C // self.num_heads)
        else:
            q = self.wq(q).view(B, self.n_queries, self.num_heads, C // self.num_heads)

        q = self.q_norm(q)

        k = self.k_norm(self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads))
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if self.gating:
            return q, k, v, gate_score
        return q, k, v


    def apply_rope(self, q, k, coords):
        if self.RoPe == "2D":
            k = self.rope(k.permute(0, 2, 1, 3), coords[:, None, :, :])
            q = q.permute(0, 2, 1, 3)
        elif self.RoPe == "1D":
            k = self.rope(k, input_pos=coords).permute(0, 2, 1, 3)
            q = self.rope(q).permute(0, 2, 1, 3)
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
        return q, k

    def forward(self, x, coords=None, return_attn=False):
        B, N, C = x.shape
        C = C * self.expand_dim

        if N == 1: # if there is only one patch, we don't need to attend
            return self.wv(x)

        if self.gating:
            q, k, v, gate_score = self.get_qkv(x)
        else:
            q,k,v = self.get_qkv(x)

        q,k = self.apply_rope(q, k, coords)

        if not return_attn:
            x = fused_sdpa(q, k, v)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
            
        x = x.transpose(1, 2).contiguous() 
        if self.gating:
            x = x * torch.sigmoid(gate_score)

        x = x.reshape(B, self.n_queries, C)
        if return_attn:
            return x, attn
        return x

class ACABlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            n_queries=1,
            expand_dim=1,
            RoPe=None,
            max_seq_len=750,
            gating=False
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ACAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            n_queries=n_queries,
            expand_dim=expand_dim,
            RoPe=RoPe,
            max_seq_len=max_seq_len,
            gating=gating
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.expand_dim = expand_dim

        dim = dim * expand_dim

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.gamma_1 = nn.Parameter(1e-6 * torch.ones(dim)) # On commence très petit
        self.gamma_2 = nn.Parameter(1e-6 * torch.ones(dim)) # On commence très petit

    def forward(self, x, coords=None):
        if self.expand_dim == 2:
            res = torch.cat([x.mean(dim=1).unsqueeze(1), x.max(dim=1).values.unsqueeze(1)], dim=2)
        elif self.expand_dim == 4:
            res = torch.cat([x.mean(dim=1).unsqueeze(1), x.min(dim=1).values.unsqueeze(1), x.max(dim=1).values.unsqueeze(1), x.std(dim=1).unsqueeze(1)], dim=2)
        elif self.expand_dim == 3:
            res = torch.cat([x.mean(dim=1).unsqueeze(1), x.min(dim=1).values.unsqueeze(1), x.max(dim=1).values.unsqueeze(1)], dim=2)
        else:
            res = x.max(dim=1).values.unsqueeze(1)

        x = res + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), coords))
        x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=None, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm, gating=False):
        super().__init__()
        self.num_heads = num_heads
        self.gating = gating
        head_dim = dim // num_heads
        self.scale = qk_scale or (head_dim) ** -0.5
        
        self.q_norm = norm_layer(dim // num_heads)
        self.k_norm = norm_layer(dim // num_heads)

        if self.gating:
            self.wq = nn.Linear(dim, dim * 2, bias=qkv_bias)
        else:
            self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.rope = Rope2D((dim) // self.num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self._device_weight_init()

    def _device_weight_init(self):
        nn.init.kaiming_normal_(self.wq.weight, nonlinearity='relu')
        self.wq.weight.data.mul_(0.1)
        if self.wq.bias is not None:
            self.wq.bias.data.fill_(1)
        nn.init.kaiming_normal_(self.wk.weight, nonlinearity='relu')
        self.wk.weight.data.mul_(0.1)
        if self.wk.bias is not None:
            self.wk.bias.data.fill_(1)
        nn.init.kaiming_normal_(self.wv.weight, nonlinearity='relu')
        self.wv.weight.data.copy_(torch.eye(self.wv.weight.shape[0], self.wv.weight.shape[1]))
        if self.wv.bias is not None:
            self.wv.bias.data.zero_()
        nn.init.eye_(self.proj.weight)
        self.proj.bias.data.zero_()


    def forward(self, q, coords_q, kv, coords_kv, n_registers, n_registers_q, n_modalities, masking=False):
        B, Nkv, C = kv.shape
        B, Nq, C = q.shape
        size_q = int((Nq - n_registers_q) ** .5)
        size_kv = int((Nkv // n_modalities - n_registers) ** .5)

        if self.gating:
            q = self.wq(q).view(B, Nq, self.num_heads, -1)
            q, gate_score = torch.split(q, [C // self.num_heads, C // self.num_heads], dim=-1)
            gate_score = gate_score.reshape(B, Nq, self.num_heads, C // self.num_heads)
            q = q.reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            q = self.wq(q).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        q = self.q_norm(q)
        k = self.k_norm(self.wk(kv).reshape(B, Nkv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3))
        v = self.wv(kv).reshape(B, Nkv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.rope(q, coords_q[:, None, :, :])
        k = self.rope(k, coords_kv[:, None, :, :])

        if masking:
            attn_mask = matching_grids(size_q, size_kv, n_registers=n_registers, n_registers_q=n_registers_q).to(q.device)
            attn_mask = attn_mask.repeat(1, n_modalities).repeat(B, 1, 1)
        else:
            attn_mask = None

        x = fused_sdpa(q, k, v, attn_mask=attn_mask)
            
        x = x.transpose(1, 2).contiguous()
        
        if self.gating:
            x = x * torch.sigmoid(gate_score)

        x = x.reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossBlockLearned(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=None, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_registers=4, gating=False):
        super().__init__()
        self.norm_kv = norm_layer(dim)
        self.norm_q = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer, gating=gating)
        self.q_learned = nn.Parameter(torch.empty(1, 1, dim))

        self.n_registers = n_registers
        self.registers = nn.Parameter(torch.empty(1, n_registers, dim))
        nn.init.normal_(self.registers, std=0.02)

        self.dim = dim

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, kv, coords_kv, n_registers, n_modalities, grid_size, mask_out=None):
        q_ = torch.max(kv, dim=1).values.unsqueeze(1)
        q = q_.expand(-1, grid_size, -1)

        if self.n_registers > 0:
            q = torch.cat((self.registers.repeat(kv.shape[0],1,1), q), dim=1)
        coords_q = get_coords(q, int(grid_size ** 0.5), 1, self.n_registers, res=1)
        x = self.drop_path(self.attn(self.norm_q(q), coords_q, self.norm_kv(kv), coords_kv, n_registers,
                                          self.n_registers, n_modalities))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=None, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, gating=False):
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop, proj_drop=drop, gating=gating)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, q, coords_q, kv, coords_kv, n_registers, n_registers_q, n_modalities):
        x = q + self.drop_path(self.attn(self.norm_q(q), coords_q, self.norm_kv(kv), coords_kv,
                                     n_registers, n_registers_q, n_modalities))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        to_2tuple = _ntuple(2)
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        # if x to big and not training, do in chunks to save memory
        if x.numel() > 2**24 and not torch.is_grad_enabled:
            chunk_size = 2**24 // (x.numel() // x.shape[0]) # keep approx 1G elements per chunk
            x_chunks = []
            print(f"MLP chunking with {x.shape=} and {chunk_size=}")
            for i in range(0, len(x), chunk_size):
                x_chunk = x[i:i+chunk_size]
                x_chunk = self.fc1(x_chunk)
                x_chunk = self.act(x_chunk)
                x_chunk = self.drop1(x_chunk)
                x_chunk = self.norm(x_chunk)
                x_chunk = self.fc2(x_chunk)
                x_chunk = self.drop2(x_chunk)
                x_chunks.append(x_chunk)
            x = torch.cat(x_chunks, dim=0)
            print(f"MLP chunking result: {x.shape=}")
            return x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

class Rope1D(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 367,
        base: int = 100_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        if input_pos is None:
            rope_cache = rope_cache.unsqueeze(0).unsqueeze(2)
        else:
            rope_cache = rope_cache.unsqueeze(2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)
    
def rotate_half_2d(x):
    x1, x2, x3, x4 = x.chunk(4, dim=-1)
    return torch.cat((-x2, x1, -x4, x3), dim=-1)

class Rope2D(nn.Module):
    def __init__(
        self,
        dim: int,
        max_freq: Union[float, int] = 7,
        min_freq: Union[float, int] = 7e-4,
    ):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.freqs = nn.Parameter(torch.empty(2, self.dim))
        self._device_weight_init()

    def _device_weight_init(self):
        # Create freqs in 1d
        freqs_1d = self.max_freq * (self.max_freq / self.min_freq) ** torch.linspace(0, -1, self.dim // 4)
        # duplicate freqs for rotation pairs of channels
        freqs_1d = torch.cat([freqs_1d, freqs_1d])
        # First half of channels do x, second half do y
        freqs_2d = torch.zeros(2, self.dim)
        freqs_2d[0, : self.dim // 2] = freqs_1d
        freqs_2d[1, -self.dim // 2 :] = freqs_1d
        # it's an angular freq here
        self.freqs.data.copy_(freqs_2d * 2 * torch.pi)

    def forward(self, x, coords, norm_coords=False):
        if norm_coords:
            coords = (coords - coords.amin(dim=(1,2), keepdim=True)) / (coords.amax(dim=(1,2), keepdim=True) - coords.amin(dim=(1,2), keepdim=True))
        angle = coords @ self.freqs
        return x * angle.cos() + rotate_half_2d(x) * angle.sin()
