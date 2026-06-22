import torch

def get_2d_sincos_pos_embed_with_scale(
    embed_dim, grid_size, scale, n_registers=0, modis=False
):
    """
    grid_size: int of the grid height and width
    res: array of size n, representing the resolution of a pixel (say, in meters),
    return:
    pos_embed: dict of [n,grid_size*grid_size, embed_dim] or [n,1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(
        grid_w, grid_h, indexing="xy"
    )  # here h goes first,direction reversed for numpy
    grid = torch.stack(grid, dim=0)  # 2 x h x w

    grid = torch.einsum("chw,n->cnhw", grid, torch.tensor([scale])) 
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(
        embed_dim, grid
    )  #  # (nxH*W, D/2)
    pos_embed = pos_embed.reshape(n, h * w, embed_dim)
    if n_registers > 0:
        pos_embed = torch.cat(
            [
                torch.zeros(
                    [n, n_registers, embed_dim], dtype=torch.float32, device=pos_embed.device
                ),
                pos_embed,
            ],
            dim=1,
        )
    if modis:
        pos_embed = torch.cat(
            [
                torch.zeros(
                    [n, 1, embed_dim], dtype=torch.float32, device=pos_embed.device
                ),
                pos_embed,
            ],
            dim=1,
        )
    return pos_embed

def get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    old_shape = pos
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

def get_edge_coordinates(n: int, dtype: torch.dtype, device: torch.device):
    side = n // 4
    assert n == 4 * side
    reg_coords = torch.zeros(1, n, 2, dtype=dtype, device=device)
    c = torch.arange(side, dtype=dtype, device=device) / side
    reg_coords[:, 0 * side : 1 * side, 0] = c
    reg_coords[:, 0 * side : 1 * side, 1] = 0
    reg_coords[:, 1 * side : 2 * side, 0] = 1
    reg_coords[:, 1 * side : 2 * side, 1] = c
    reg_coords[:, 2 * side : 3 * side, 0] = 1 - c
    reg_coords[:, 2 * side : 3 * side, 1] = 1
    reg_coords[:, 3 * side : 4 * side, 0] = 0
    reg_coords[:, 3 * side : 4 * side, 1] = 1 - c
    return reg_coords

def get_coords(x, grid_size, n_modalities, register, res=1, modis=False):
    """
    Generate 2D coordinates for patches, including optional register and MODIS tokens.

    Args:
        x (torch.Tensor): Input tensor used to determine batch size, device, and dtype.
        grid_size (int): Number of patches along one spatial dimension.
        n_modalities (int): Number of modalities to repeat the coordinates for.
        register (int): Number of register tokens to prepend coordinates for.
        res (float, optional): Resolution or scale factor for the coordinates. Defaults to 1.
        modis (bool, optional): If True, prepends an additional zero coordinate. Defaults to False.

    Returns:
        torch.Tensor: A tensor of coordinates with shape (b, total_tokens, 2).
    """
    b = x.shape[0]
    # compute coord for ropes

    coord_x = torch.linspace(0, 1, grid_size, device=x.device, dtype=torch.float32) * res
    coord_y = torch.linspace(0, 1, grid_size, device=x.device, dtype=torch.float32) * res
    
    coords_all = torch.cartesian_prod(coord_x, coord_y)
    coords_all = coords_all.repeat(n_modalities, 1)
    coords_all = coords_all[None].expand(b, -1, -1)
    if modis:
        coords_all = torch.cat([torch.zeros(b, 1, 2, device=x.device, dtype=torch.float32), coords_all], dim=1)
    if register > 3:
        reg_coords = get_edge_coordinates(register, torch.float32, x.device).expand(b, -1, -1)
    else:
        reg_coords = torch.zeros(b, register, 2, device=x.device, dtype=torch.float32)
    coords = torch.cat([reg_coords, coords_all], dim=1)
    
    return coords

def get_coords_1D(x, num_patches, n_modalities, register, res=1, modis=False):
    b = x.shape[0]
    # compute coord for ropes
    coords_all = torch.linspace(0, 1, num_patches, device=x.device, dtype=x.dtype) * res
    coords_all = coords_all.repeat(n_modalities)

    coords_all = coords_all[None].expand(b, -1)
    if modis:
        coords_all = torch.cat([torch.zeros(b, 1, device=x.device, dtype=x.dtype), coords_all], dim=1)
    reg_coords = torch.zeros(b, register, device=x.device, dtype=x.dtype)
    coords = torch.cat([reg_coords, coords_all], dim=1)
    
    return coords