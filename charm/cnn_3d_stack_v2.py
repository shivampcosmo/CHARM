import torch
import torch.nn as nn
import numpy as np
from torch.utils.checkpoint import checkpoint as grad_ckpt


def _make_act(act: str) -> nn.Module:
    if act == 'tanh':
        return nn.Tanh()
    elif act == 'lrelu':
        return nn.LeakyReLU(0.2)
    raise ValueError(f"Unknown activation '{act}'; choose 'tanh' or 'lrelu'.")


class ResidualBlock(nn.Module):
    """
    Pre-activation 3D residual block: Conv3d -> Act -> Conv3d -> (+ skip) -> Act.

    Both convolutions use valid padding, so each reduces every spatial axis by
    (ksize - 1). Total shrinkage per block is 2*(ksize - 1), counted as
    n_cnn_tot += 2 in the parent stack.

    The skip connection is centre-cropped to match the post-conv spatial size.
    If nf_inp != nf_out, a bias-free Linear projects the skip channels (applied
    via a NDHWC → Linear → NCDHW transpose trick).
    """

    def __init__(self, nf_inp: int, nf_out: int, ksize: int, padding=None, act: str = 'tanh'):
        super().__init__()
        self.ksize = ksize
        self.conv1 = nn.Conv3d(nf_inp, nf_out, kernel_size=ksize, padding=padding)
        self.act1 = _make_act(act)
        self.conv2 = nn.Conv3d(nf_out, nf_out, kernel_size=ksize, padding=padding)
        self.act2 = _make_act(act)
        self.linear = nn.Linear(nf_inp, nf_out, bias=False) if nf_out != nf_inp else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.conv1(x))
        out = self.conv2(out)
        crop = (self.ksize + 1) // 2
        x_skip = x[..., crop:-crop, crop:-crop, crop:-crop]
        if self.linear is not None:
            x_skip = torch.moveaxis(self.linear(torch.moveaxis(x_skip, 1, 4)), 4, 1)
        return self.act2(out + x_skip)


class FiLMGenerator(nn.Module):
    """
    Maps a cosmology vector to per-channel FiLM parameters (gamma, beta).

    Applied as  F = (1 + gamma) * F + beta  so the network starts as an
    identity transform — the final linear layer is zero-initialised.

    Parameters
    ----------
    cosmo_dim : int
        Length of the cosmology parameter vector.
    n_channels : int
        Number of feature-map channels to modulate (C in the conv output).
    """

    def __init__(self, cosmo_dim: int, n_channels: int):
        super().__init__()
        hidden = max(2 * cosmo_dim, n_channels)
        self.net = nn.Sequential(
            nn.Linear(cosmo_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2 * n_channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cosmo: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cosmo : (N, cosmo_dim)
        x     : (N, C, D, H, W)
        """
        params = self.net(cosmo)                              # (N, 2C)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.view(gamma.shape[0], -1, 1, 1, 1)
        beta = beta.view(beta.shape[0], -1, 1, 1, 1)
        return (1.0 + gamma) * x + beta


class CNN3D_stackout_v2(nn.Module):
    """
    3D CNN encoder with multi-resolution skip readout and optional FiLM
    cosmology modulation.

    Improvements over CNN3D_stackout
    ---------------------------------
    1. **Multi-resolution skip readout.** After every conv block the
       intermediate feature map is pooled to (dim_out)^3 via
       AdaptiveAvgPool3d, projected to d_skip channels via a 1x1 Conv3d,
       and stored. All block skips are concatenated along the channel axis
       and a final 1x1 Conv3d projects them to nout. Downstream heads
       therefore see features from every receptive-field scale simultaneously.

    2. **FiLM cosmology modulation** (enabled when cosmo_dim > 0). After
       each block a FiLMGenerator maps the cosmology vector to per-channel
       (gamma, beta) and applies F = (1+gamma)*F + beta. This lets the
       encoder re-shape its features per cosmology rather than treating
       cosmology as just another input channel.
       When cosmo_dim > 0, cosmology must NOT be included in cond_mat —
       pass it via the cosmo argument to forward() instead.

    Output contract is identical to CNN3D_stackout: (nsim * dim_out**3, nout).

    Parameters
    ----------
    ksize : int
        Cubic kernel size for every Conv3d and ResidualBlock.
    nside_in : int
        Unpadded simulation cube side length.
    nside_out : int
        Desired output (halo-grid) cube side length.
    nbatch : int
        Sub-cube splits per side; dim_in = nside_in//nbatch,
        dim_out = nside_out//nbatch.
    ninp : int
        Input channel count (DM field only; exclude cosmology when using FiLM).
    nfeature : int
        Base channel width; blocks use 2*nfeature then 4*nfeature.
    nout : int
        Output features per voxel.
    layers_types : list[str]
        Ordered list of 'cnn' or 'res' block types.
    act : {'tanh', 'lrelu'}
        Activation used inside blocks.
    padding : str
        Forwarded to Conv3d; use 'valid' to match the padding arithmetic
        that callers apply to their inputs.
    cosmo_dim : int
        Length of the cosmology vector for FiLM. 0 disables FiLM.
    d_skip : int or None
        Channel width of each skip projection. Defaults to nout.

    Attributes
    ----------
    n_cnn_tot : int
        Total valid-conv shrinkage steps; callers compute required input
        padding as  dim_in + n_cnn_tot * (ksize - 1).
    """

    def __init__(
        self,
        ksize: int,
        nside_in: int,
        nside_out: int,
        nbatch: int,
        ninp: int,
        nfeature: int,
        nout: int,
        layers_types=None,
        act: str = 'tanh',
        padding: str = 'valid',
        cosmo_dim: int = 0,
        d_skip: int = None,
    ):
        super().__init__()
        if layers_types is None:
            layers_types = ['cnn', 'res', 'res', 'res']

        self.ksize = ksize
        self.nside_in = nside_in
        self.nside_out = nside_out
        self.nbatch = nbatch
        self.nfeature = nfeature
        self.nout = nout
        self.ninp = ninp
        self.cosmo_dim = cosmo_dim

        d_skip = nout if d_skip is None else d_skip

        self.n_cnn_tot = 0
        self.block_channels = []

        blocks = []
        for j, ltype in enumerate(layers_types):
            ch_in = ninp if j == 0 else (2 * nfeature if j == 1 else 4 * nfeature)
            ch_out = 2 * nfeature if j == 0 else 4 * nfeature

            if ltype == 'cnn':
                blocks.append(nn.Sequential(
                    nn.Conv3d(ch_in, ch_out, kernel_size=ksize, padding=padding),
                    _make_act(act),
                ))
                self.n_cnn_tot += 1
            elif ltype == 'res':
                blocks.append(ResidualBlock(ch_in, ch_out, ksize, padding=padding, act=act))
                self.n_cnn_tot += 2
            else:
                raise ValueError(f"Invalid layer type '{ltype}'; use 'cnn' or 'res'.")

            self.block_channels.append(ch_out)

        self.blocks = nn.ModuleList(blocks)

        self.film_gens = (
            nn.ModuleList([FiLMGenerator(cosmo_dim, ch) for ch in self.block_channels])
            if cosmo_dim > 0 else None
        )

        dim_out = nside_out // nbatch
        self.adaptive_pool = nn.AdaptiveAvgPool3d(dim_out)

        self.skip_projs = nn.ModuleList([
            nn.Conv3d(ch, d_skip, kernel_size=1)
            for ch in self.block_channels
        ])

        self.out_proj = nn.Conv3d(len(layers_types) * d_skip, nout, kernel_size=1)
        self.use_checkpoint = False

    def forward(self, cond_mat: torch.Tensor, cosmo: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        cond_mat : (nsim, ninp, padded_dim, padded_dim, padded_dim)
            Padded DM density sub-cube. When cosmo_dim > 0, do NOT include
            cosmology channels here — pass them via cosmo instead.
        cosmo : (nsim, cosmo_dim) or None
            Cosmology vector. Required when cosmo_dim > 0.

        Returns
        -------
        torch.Tensor, shape (nsim * dim_out**3, nout)
        """
        nsim = cond_mat.shape[0]
        dim_out = self.nside_out // self.nbatch
        dim_in = self.nside_in // self.nbatch
        padded_dim = dim_in + self.n_cnn_tot * (self.ksize - 1)

        if cond_mat.shape[2] != padded_dim:
            raise ValueError(
                f"Expected spatial input size {padded_dim}, got {cond_mat.shape[2]}. "
                f"Pad inputs by n_cnn_tot * (ksize - 1) = "
                f"{self.n_cnn_tot} * {self.ksize - 1} = "
                f"{self.n_cnn_tot * (self.ksize - 1)} voxels."
            )
        if self.cosmo_dim > 0 and cosmo is None:
            raise ValueError("cosmo_dim > 0 but cosmo was not passed to forward().")

        x = cond_mat
        skips = []
        for j, block in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                x = grad_ckpt(block, x, use_reentrant=False)
            else:
                x = block(x)
            if self.film_gens is not None:
                x = self.film_gens[j](cosmo, x)
            skip = self.adaptive_pool(x)        # (N, C_j, dim_out, dim_out, dim_out)
            skip = self.skip_projs[j](skip)     # (N, d_skip, dim_out, dim_out, dim_out)
            skips.append(skip)

        out = torch.cat(skips, dim=1)           # (N, n_blocks*d_skip, dim_out, dim_out, dim_out)
        out = self.out_proj(out)                # (N, nout, dim_out, dim_out, dim_out)
        out = out.permute(0, 2, 3, 4, 1)        # (N, dim_out, dim_out, dim_out, nout)
        return out.reshape(nsim * dim_out ** 3, self.nout)
