from torch import nn


class MLPSemSeg(nn.Module):
    def __init__(
        self,
        initial_dim=512,
        hidden_dim=[128, 32, 2],
        final_dim=2,
        norm=nn.InstanceNorm1d,
        activation=nn.ReLU,
        patch_size: int = 1,
    ):
        """
        Initializes an MLP segmentation head.
        Args:
            initial_dim (int): dimension of input layer
            hidden_dim (list): list of hidden dimensions for the MLP
            final_dim (int): number of output classes
            norm (nn.Module): normalization layer
            activation (nn.Module): activation layer
            patch_size (int): each token is expanded to a patch_size x patch_size
                block of per-pixel logits
        """
        super().__init__()
        dim = [initial_dim] + hidden_dim + [final_dim * patch_size * patch_size]
        args = self.init_layers(dim, norm, activation)
        self.mlp = nn.Sequential(*args)
        self.patch_size = patch_size
        self.final_dim = final_dim

    def init_layers(self, dim, norm, activation):
        """Initializes the MLP layers."""
        args = [nn.LayerNorm(dim[0])]
        for i in range(len(dim) - 1):
            args.append(nn.Linear(dim[i], dim[i + 1]))
            if i < len(dim) - 2:
                args.append(norm(dim[i + 1]))
                args.append(activation())
        return args

    def forward(self, x):
        """Project each token to a patch_size x patch_size block of logits.

        Args:
            x: token features of shape (B, N, D) on a square N = H*W grid.
        Returns:
            Per-pixel logits of shape (B, final_dim, H*patch_size, W*patch_size).
        """
        x = self.mlp(x)
        B, N, D = x.shape
        num_patches = int(N**(1/2))
        size = num_patches * self.patch_size

        x = x.view(B, N, 1, D).view(B, N, 1, self.final_dim, self.patch_size * self.patch_size).permute(0, 2, 3, 1, 4)
        x = x.view(B, 1, self.final_dim, N, self.patch_size, self.patch_size)
        x = x.view(B, 1, self.final_dim, num_patches, num_patches, self.patch_size, self.patch_size).permute(0, 1, 2, 3, 5, 4, 6)
        x = x.reshape(B, 1, self.final_dim, size, size).flatten(0, 1)
        return x
