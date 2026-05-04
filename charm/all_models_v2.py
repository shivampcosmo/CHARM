import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

try:
    from .utils import unconstrained_RQS
except ImportError:
    from charm.utils import unconstrained_RQS

from torch.distributions import HalfNormal, Weibull, Gumbel


# ── Interpolation helper ──────────────────────────────────────────────────────

def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    Batched 1-D linear interpolation for monotonically increasing sample points.

    Args:
        x  : (B, N) query coordinates.
        xp : (B, M) knot x-coordinates (monotonically increasing).
        fp : (B, M) knot y-coordinates.

    Returns:
        (B, N) interpolated values.
    """
    m = (fp[:, 1:] - fp[:, :-1]) / (xp[:, 1:] - xp[:, :-1])
    b = fp[:, :-1] - m * xp[:, :-1]
    idx = torch.clamp(
        torch.sum(x[:, :, None] >= xp[:, None, :], dim=-1) - 1,
        0, m.shape[-1] - 1,
    )
    row = torch.arange(idx.shape[0], device=idx.device).unsqueeze(1).expand_as(idx)
    return m[row, idx] * x + b[row, idx]


# ── Shared MLP ────────────────────────────────────────────────────────────────

class FCNN(nn.Module):
    """Two-hidden-layer MLP with Tanh activations."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(dtype=self.net[0].weight.dtype))


# ── Module-level flow helpers ─────────────────────────────────────────────────

def _normalize_bounds(B, dim: int):
    """
    Normalise the tail-bound spec B into a list of (lo, hi) pairs of length dim.

    B may be:
      scalar s      → [(-s, s)] * dim
      [lo, hi]      → [(lo, hi)] * dim  (shared across all dims)
      [[lo0,hi0],…] → one pair per dim, last entry repeated if list is short
    """
    if isinstance(B, (int, float)):
        return [(-float(B), float(B))] * dim
    if isinstance(B[0], (int, float)):
        return [(float(B[0]), float(B[1]))] * dim
    pairs = [(float(b[0]), float(b[1])) for b in B]
    while len(pairs) < dim:
        pairs.append(pairs[-1])
    return pairs


@torch._dynamo.disable
def _apply_rqs(
    x: torch.Tensor,
    layer_out: torch.Tensor,
    K: int,
    bounds,
    inverse: bool,
):
    """
    Apply one RQS coupling layer.

    layer_out : (N, 3K-1)  or  (1, 3K-1) for the unconditional model.
    bounds    : (lo, hi) tuple from _normalize_bounds.
    Returns (z, log_det) both shape (N,).
    """
    lo, hi = bounds
    span = hi - lo
    W, H, D = torch.split(layer_out, K, dim=-1)
    W = torch.softmax(W, dim=-1) * span
    H = torch.softmax(H, dim=-1) * span
    D = F.softplus(D)
    # unconstrained_RQS allocates outputs as zeros_like(x) then index-puts
    # float32 RQS results into it — mismatches when x is bfloat16/float16.
    orig_dtype = x.dtype
    z, log_det = unconstrained_RQS(
        x.float(), W.float(), H.float(), D.float(),
        inverse=inverse, tail_bound=[lo, hi],
    )
    return z.to(orig_dtype), log_det.to(orig_dtype)


def _get_gauss_params(out, ngauss, mu_pos, base_dist_pwall, mu_fixed=None):
    """
    Parse batched MLP output into Gaussian base-distribution parameters.

    out           : (N, D) raw MLP output.
    ngauss        : number of mixture components.
    mu_pos        : if True squash predicted means to (0,1) via (1+tanh)/2.
    base_dist_pwall : 'pl_exp' or None.
    mu_fixed      : (ngauss,) buffer of fixed means, or None.

    Returns
    -------
    ngauss == 1 : (mu, var)  each (N,).
    ngauss  > 1 : (mu_all, var_all, pw_all)  each (N, ngauss).
    """
    if ngauss == 1:
        mu, alpha = out[:, 0], out[:, 1]
        if mu_pos:
            mu = (1.0 + torch.tanh(mu)) / 2.0
        return mu, torch.exp(alpha)

    if mu_fixed is not None:
        alpha_all = out[:, 0:ngauss]
        pw_raw    = out[:, ngauss:2 * ngauss]
        mu_all    = mu_fixed.unsqueeze(0).expand(out.shape[0], -1)
        al_idx, bt_idx = 2 * ngauss, 2 * ngauss + 1
    else:
        mu_all    = out[:, 0:ngauss]
        alpha_all = out[:, ngauss:2 * ngauss]
        pw_raw    = out[:, 2 * ngauss:3 * ngauss]
        mu_all    = (1.0 + torch.tanh(mu_all)) / 2.0 if mu_pos else torch.tanh(mu_all)
        al_idx, bt_idx = 3 * ngauss, 3 * ngauss + 1

    var_all = torch.exp(alpha_all)

    if base_dist_pwall == 'pl_exp':
        pw_raw = torch.exp(pw_raw)
        # alpha term is zeroed (effectively pure exponential prior)
        bt = torch.exp(out[:, bt_idx]) + 1.0
        base_pws = torch.stack(
            [torch.exp(-bt * mu_all[:, i]) for i in range(ngauss)], dim=1
        )
        pw_raw = pw_raw * base_pws

    return mu_all, var_all, torch.softmax(pw_raw, dim=1)


def _sample_gaussian_mixture(mu_all, var_all, pw_all, device):
    """
    Draw one sample per row from a Gaussian mixture.
    Preserves voxel ordering (fills a pre-allocated tensor by component index).

    Returns (N,) tensor.
    """
    N = pw_all.shape[0]
    counts = torch.distributions.Multinomial(total_count=1, probs=pw_all).sample()
    x = torch.zeros(N, device=device)
    for k in range(pw_all.shape[1]):
        ind = counts[:, k].bool()
        if ind.any():
            x[ind] = mu_all[ind, k] + torch.randn(ind.sum(), device=device) * torch.sqrt(var_all[ind, k])
    return x


# ── Distribution heads ────────────────────────────────────────────────────────

class BinaryMaskModel(nn.Module):
    """
    Per-voxel Bernoulli occupancy head.
    forward : BCE-with-logits loss, shape (N,).
    inverse : sigmoid probabilities, shape (N, 1).
    """

    def __init__(self, dim=1, hidden_dim=8, base_network=FCNN, num_cond=0):
        super().__init__()
        self.num_cond = num_cond
        self.layer_init = base_network(num_cond, 1, hidden_dim)

    def forward(self, x: torch.Tensor, cond_inp: torch.Tensor) -> torch.Tensor:
        logits = self.layer_init(cond_inp)
        return F.binary_cross_entropy_with_logits(
            logits.reshape(-1, 1), x.reshape(-1, 1), reduction='none'
        )[:, 0]

    def inverse(self, cond_inp: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.layer_init(cond_inp))


class MultiClassMaskModel(nn.Module):
    """
    Per-voxel categorical Nhalos head.
    forward : cross-entropy loss, shape (N,).
    inverse : softmax class probabilities, shape (N, num_classes).
    """

    def __init__(self, dim=1, hidden_dim=8, base_network=FCNN, num_cond=0, num_classes=2):
        super().__init__()
        self.num_cond = num_cond
        self.num_classes = num_classes
        self.layer_init = base_network(num_cond, num_classes, hidden_dim)

    def forward(self, x: torch.Tensor, cond_inp: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.layer_init(cond_inp), x[:, 0].long(), reduction='none')

    def inverse(self, cond_inp: torch.Tensor, mask=None) -> torch.Tensor:
        out = torch.softmax(self.layer_init(cond_inp), dim=1)
        if mask is not None:
            out = out * mask[:, 0:1]
        return out


class SumGaussModel(nn.Module):
    """
    Gaussian-mixture head for a scalar target in [0, 1].

    Two modes:
      Free  (mu_all / sig_all = None): MLP outputs means, log-vars, and logits.
      Fixed (both provided)           : only mixing weights are predicted;
                                        base_dist='pl_exp' adds an exponential
                                        prior over weights (alpha term zeroed).

    Buffers: mu_all, sig_all, var_all (registered so .to(device) propagates).
    """

    def __init__(
        self,
        dim=1,
        hidden_dim=8,
        base_network=FCNN,
        num_cond=0,
        ngauss=1,
        mu_all=None,
        sig_all=None,
        base_dist='normal',
    ):
        super().__init__()
        self.num_cond = num_cond
        self.ngauss = ngauss
        self.base_dist = base_dist
        self._free = (mu_all is None) or (sig_all is None)

        self.register_buffer('mu_all',  torch.tensor(mu_all,        dtype=torch.float32) if mu_all  is not None else None)
        self.register_buffer('sig_all', torch.tensor(sig_all,       dtype=torch.float32) if sig_all is not None else None)
        self.register_buffer('var_all', torch.tensor(sig_all ** 2,  dtype=torch.float32) if sig_all is not None else None)

        if self._free:
            out_dim = 3 * ngauss
        elif base_dist == 'normal':
            out_dim = ngauss
        elif base_dist == 'pl_exp':
            out_dim = ngauss + 2
        else:
            raise ValueError(f"base_dist '{base_dist}' not supported")
        self.layer_init = base_network(num_cond, out_dim, hidden_dim)

    def _mixing_weights(self, out):
        """Compute normalised mixing weights from raw MLP output."""
        if self._free:
            pw_raw = out[:, 2 * self.ngauss:]
        elif self.base_dist == 'normal':
            pw_raw = out
        else:  # pl_exp
            pw_raw   = torch.exp(out[:, :self.ngauss])
            bt       = torch.exp(out[:, self.ngauss + 1]) + 1.0
            base_pws = torch.stack(
                [torch.exp(-bt * self.mu_all[i]) for i in range(self.ngauss)], dim=1
            )
            pw_raw = torch.log(pw_raw * base_pws + 1e-30)
        return torch.softmax(pw_raw, dim=1)

    def forward(self, x: torch.Tensor, cond_inp: torch.Tensor) -> torch.Tensor:
        out = self.layer_init(cond_inp)
        dev = x.device

        if self._free:
            mu_all  = (1.0 + torch.tanh(out[:, :self.ngauss])) / 2.0
            var_all = torch.exp(out[:, self.ngauss:2 * self.ngauss])
        else:
            mu_all  = self.mu_all.to(dev).unsqueeze(0).expand(x.shape[0], -1)
            var_all = self.var_all.to(dev).unsqueeze(0).expand(x.shape[0], -1)

        pw_all = self._mixing_weights(out)
        Li = sum(
            pw_all[:, i]
            * (1.0 / torch.sqrt(2 * np.pi * var_all[:, i]))
            * torch.exp(-0.5 * (x[:, 0] - mu_all[:, i]) ** 2 / var_all[:, i])
            for i in range(self.ngauss)
        )
        return -torch.log(Li + 1e-30)

    def inverse(self, cond_inp: torch.Tensor) -> torch.Tensor:
        device = cond_inp.device
        out = self.layer_init(cond_inp)
        N = out.shape[0]

        if self._free:
            mu_all  = (1.0 + torch.tanh(out[:, :self.ngauss])) / 2.0
            var_all = torch.exp(out[:, self.ngauss:2 * self.ngauss])
        else:
            mu_all  = self.mu_all.to(device).unsqueeze(0).expand(N, -1)
            var_all = self.var_all.to(device).unsqueeze(0).expand(N, -1)

        pw_all = self._mixing_weights(out)
        return _sample_gaussian_mixture(mu_all, var_all, pw_all, device)

    def sample(self, cond_inp=None, mask=None):
        return self.inverse(cond_inp)


class NSF_1var_CNNcond(nn.Module):
    """
    Conditional Neural Spline Flow over a single scalar (e.g. M1).

    nflows RQS coupling layers; base distribution selectable via base_dist:
      'gauss', 'halfgauss', 'weibull', 'gumbel', 'physical_hmf'.
    ngauss > 1 enables a Gaussian mixture base with optional 'pl_exp' prior.
    Physical-HMF tables are stored as registered buffers.
    """

    def __init__(
        self,
        dim=1,
        K=5,
        B=3,
        hidden_dim=8,
        base_network=FCNN,
        num_cond=0,
        nflows=1,
        ngauss=1,
        base_dist='gauss',
        mu_all=None,
        mu_pos=False,
        base_dist_pwall=None,
        lgM_rs_tointerp=None,
        hmf_pdf_tointerp=None,
        hmf_cdf_tointerp=None,
    ):
        super().__init__()
        self.K = K
        self.num_cond = num_cond
        self.nflows = nflows
        self.ngauss = ngauss
        self.base_dist = base_dist
        self.mu_pos = mu_pos
        self.base_dist_pwall = base_dist_pwall
        self._bounds = _normalize_bounds(B, 1)   # [(lo, hi)]

        self.register_buffer('mu_all', torch.tensor(mu_all, dtype=torch.float32) if mu_all is not None else None)

        if base_dist in ('gauss', 'halfgauss'):
            if ngauss == 1:
                gauss_out = 2
            elif base_dist_pwall == 'pl_exp':
                gauss_out = (2 if mu_all is not None else 3) * ngauss + 2
            else:
                gauss_out = (2 if mu_all is not None else 3) * ngauss
            self.layer_init_gauss = base_network(num_cond, gauss_out, hidden_dim)
        elif base_dist in ('weibull', 'gumbel'):
            self.layer_init_gauss = base_network(num_cond, 2, hidden_dim)
        elif base_dist == 'physical_hmf':
            if lgM_rs_tointerp is None:
                raise ValueError("physical_hmf requires lgM_rs_tointerp, hmf_pdf_tointerp, hmf_cdf_tointerp")
            self.register_buffer('lgM_rs',   torch.tensor([lgM_rs_tointerp],                           dtype=torch.float32))
            self.register_buffer('hmf_logpdf', torch.log(torch.tensor([hmf_pdf_tointerp],              dtype=torch.float32)))
            self.register_buffer('hmf_cdf',    torch.tensor([hmf_cdf_tointerp],                        dtype=torch.float32))
        else:
            raise ValueError(f"base_dist '{base_dist}' not recognised")

        self.layers = nn.ModuleList([base_network(num_cond, 3 * K - 1, hidden_dim) for _ in range(nflows)])

    def _base_params(self, cond_inp):
        if self.base_dist in ('gauss', 'halfgauss'):
            out = self.layer_init_gauss(cond_inp)
            return _get_gauss_params(out, self.ngauss, self.mu_pos, self.base_dist_pwall, self.mu_all)
        out = self.layer_init_gauss(cond_inp)
        mu, alpha = out[:, 0], out[:, 1]
        if self.base_dist == 'weibull':
            return torch.exp(mu), torch.exp(alpha)          # scale, conc
        # gumbel
        if self.mu_pos:
            mu = (1.0 + torch.tanh(mu)) / 2.0
        return mu, torch.exp(alpha)                         # loc, scale

    def _base_logp(self, x, params):
        bd = self.base_dist
        if bd == 'gauss':
            if self.ngauss == 1:
                mu, var = params
                return -0.5 * math.log(2 * math.pi) - 0.5 * torch.log(var) - 0.5 * (x - mu) ** 2 / var
            mu_all, var_all, pw_all = params
            Li = sum(
                pw_all[:, i] / torch.sqrt(2 * np.pi * var_all[:, i])
                * torch.exp(-0.5 * (x - mu_all[:, i]) ** 2 / var_all[:, i])
                for i in range(self.ngauss)
            )
            lp = torch.log(Li)
        elif bd == 'halfgauss':
            mu, var = params
            x2 = torch.exp(x - mu)
            lp = HalfNormal(torch.sqrt(var)).log_prob(x2)
        elif bd == 'weibull':
            lp = Weibull(*params, validate_args=False).log_prob(x)
        elif bd == 'gumbel':
            lp = Gumbel(*params, validate_args=False).log_prob(x)
        elif bd == 'physical_hmf':
            lp = interpolate(x[None, :], self.lgM_rs, self.hmf_logpdf)[0, :]
        else:
            raise ValueError(f"base_dist '{bd}' not recognised")
        if bd != 'gauss':
            lp = torch.where(torch.isfinite(lp), lp, torch.full_like(lp, -100.0))
        return lp

    def _base_sample(self, params, N, device):
        bd = self.base_dist
        if bd == 'gauss':
            if self.ngauss == 1:
                mu, var = params
                return mu + torch.randn(N, device=device) * torch.sqrt(var)
            mu_all, var_all, pw_all = params
            return _sample_gaussian_mixture(mu_all, var_all, pw_all, device)
        elif bd == 'halfgauss':
            mu, var = params
            return torch.log(mu + torch.abs(torch.randn(N, device=device)) * torch.sqrt(var))
        elif bd == 'weibull':
            return Weibull(*params).sample()
        elif bd == 'gumbel':
            return Gumbel(*params).sample()
        elif bd == 'physical_hmf':
            u = torch.rand(N, device=device)
            return interpolate(u[None, :], self.hmf_cdf[:, 1:], self.lgM_rs[:, 1:])[0, :]
        raise ValueError(f"base_dist '{bd}' not recognised")

    def forward(self, x: torch.Tensor, cond_inp: torch.Tensor) -> torch.Tensor:
        params = self._base_params(cond_inp)
        if x.dim() > 1:
            x = x[:, 0]
        log_det = torch.zeros_like(x)
        for layer in self.layers:
            x, ld = _apply_rqs(x, layer(cond_inp), self.K, self._bounds[0], inverse=False)
            log_det += ld
        return log_det + self._base_logp(x, params)

    def inverse(self, cond_inp: torch.Tensor, mask=None):
        device = cond_inp.device
        params = self._base_params(cond_inp)
        x = self._base_sample(params, cond_inp.shape[0], device)
        log_det = torch.zeros_like(x)
        for layer in reversed(self.layers):
            x, ld = _apply_rqs(x, layer(cond_inp), self.K, self._bounds[0], inverse=True)
            log_det += ld
        if mask is not None:
            x = x * mask[:, 0]
        return x, log_det

    def sample(self, cond_inp=None, mask=None):
        x, _ = self.inverse(cond_inp, mask)
        return x


class NSF_Autoreg_CNNcond(nn.Module):
    """
    Conditional autoregressive Neural Spline Flow over dim scalar outputs
    (mass differences, velocities, concentrations, or positions).

    Component jd sees cond_inp concatenated with the already-sampled
    components z[:, :jd] as its conditioning. Each component has its own
    base-distribution MLP and nflows RQS MLPs.

    Supported base distributions: 'gauss', 'halfgauss', 'weibull', 'gumbel'.
    For 'gauss'/'halfgauss' with ngauss==1, mu is either zero (mu_pos=True)
    or tanh-squashed, and sigma is bounded to a fraction of the spline span.
    For ngauss > 1, a Gaussian mixture is used via _get_gauss_params.

    log_det returned from inverse() is the sum across all dim components.
    """

    def __init__(
        self,
        dim=None,
        K=5,
        B=3,
        hidden_dim=8,
        base_network=FCNN,
        num_cond=0,
        nflows=1,
        ngauss=1,
        base_dist='gumbel',
        mu_pos=False,
        base_dist_pwall='pl_exp',
    ):
        super().__init__()
        if dim is None:
            raise ValueError("dim must be specified")
        self.dim = dim
        self.K = K
        self.num_cond = num_cond
        self.nflows = nflows
        self.ngauss = ngauss
        self.base_dist = base_dist
        self.mu_pos = mu_pos
        self.base_dist_pwall = base_dist_pwall
        self._bounds = _normalize_bounds(B, dim)

        self.layers_all_dim_init = nn.ModuleList()
        self.layers_all_dim = nn.ModuleList()

        for jd in range(dim):
            cond_jd = num_cond + jd
            if base_dist in ('gauss', 'halfgauss'):
                if ngauss == 1:
                    init_out = 2
                elif base_dist_pwall == 'pl_exp':
                    init_out = 3 * ngauss + 2
                else:
                    init_out = 3 * ngauss
            elif base_dist in ('weibull', 'gumbel'):
                init_out = 2
            else:
                raise ValueError(f"base_dist '{base_dist}' not recognised")
            self.layers_all_dim_init.append(base_network(cond_jd, init_out, hidden_dim))
            self.layers_all_dim.append(
                nn.ModuleList([base_network(cond_jd, 3 * K - 1, hidden_dim) for _ in range(nflows)])
            )

    @torch._dynamo.disable
    def _base_params_jd(self, jd, cond_jd):
        out = self.layers_all_dim_init[jd](cond_jd)
        bd = self.base_dist
        lo, hi = self._bounds[jd]
        span = hi - lo

        if bd in ('gauss', 'halfgauss'):
            if self.ngauss == 1:
                mu_raw, sig_raw = out[:, 0], out[:, 1]
                mu = 0.0 * mu_raw if self.mu_pos else torch.tanh(mu_raw)
                sig = 0.25 * span * (1.0 + torch.tanh(sig_raw)) * 0.5
                return mu, sig
            return _get_gauss_params(out, self.ngauss, self.mu_pos, self.base_dist_pwall)

        # weibull / gumbel
        mu, alpha = out[:, 0], out[:, 1]
        if bd == 'weibull':
            return torch.exp(mu), torch.exp(alpha)
        if self.mu_pos:
            mu = (1.0 + torch.tanh(mu)) / 2.0
        return mu, torch.exp(alpha)

    def _base_logp_jd(self, jd, x, params):
        bd = self.base_dist
        if bd == 'gauss':
            if self.ngauss == 1:
                mu, sig = params
                return -0.5 * math.log(2 * math.pi) - torch.log(sig) - 0.5 * (x - mu) ** 2 / sig ** 2
            mu_all, var_all, pw_all = params
            Li = sum(
                pw_all[:, i] / torch.sqrt(2 * np.pi * var_all[:, i])
                * torch.exp(-0.5 * (x - mu_all[:, i]) ** 2 / var_all[:, i])
                for i in range(self.ngauss)
            )
            return torch.log(Li)
        if bd == 'halfgauss':
            mu, sig = params
            lp = HalfNormal(sig).log_prob(x - mu)
        elif bd == 'weibull':
            lp = Weibull(*params).log_prob(x)
        elif bd == 'gumbel':
            lp = Gumbel(*params).log_prob(x)
        else:
            raise ValueError(f"base_dist '{bd}' not recognised")
        return torch.where(torch.isfinite(lp), lp, torch.full_like(lp, -100.0))

    def _base_sample_jd(self, jd, params, device):
        bd = self.base_dist
        if bd == 'gauss':
            if self.ngauss == 1:
                mu, sig = params
                return mu + torch.randn(mu.shape[0], device=device) * sig
            mu_all, var_all, pw_all = params
            return _sample_gaussian_mixture(mu_all, var_all, pw_all, device)
        if bd == 'halfgauss':
            mu, sig = params
            return mu + HalfNormal(sig).sample()
        if bd == 'weibull':
            return Weibull(*params).sample()
        if bd == 'gumbel':
            return Gumbel(*params).sample()
        raise ValueError(f"base_dist '{bd}' not recognised")

    def forward(self, x_inp: torch.Tensor, cond_inp: torch.Tensor, mask=None) -> torch.Tensor:
        device = x_inp.device
        logp = torch.zeros_like(x_inp)
        for jd in range(self.dim):
            cond_jd = torch.cat([cond_inp, x_inp[:, :jd]], dim=1) if jd > 0 else cond_inp
            params = self._base_params_jd(jd, cond_jd)
            x = x_inp[:, jd]
            log_det = torch.zeros(x.shape[0], device=device)
            for layer in self.layers_all_dim[jd]:
                x, ld = _apply_rqs(x, layer(cond_jd), self.K, self._bounds[jd], inverse=False)
                log_det += ld
            logp[:, jd] = log_det + self._base_logp_jd(jd, x, params)
        if mask is not None:
            logp = logp * mask
        return logp.sum(dim=1)

    def inverse(self, cond_inp: torch.Tensor, mask=None):
        device = cond_inp.device
        N = cond_inp.shape[0]
        z_out = torch.zeros(N, self.dim, device=device)
        log_det_total = torch.zeros(N, device=device)
        for jd in range(self.dim):
            cond_jd = torch.cat([cond_inp, z_out[:, :jd]], dim=1) if jd > 0 else cond_inp
            params = self._base_params_jd(jd, cond_jd)
            x = self._base_sample_jd(jd, params, device)
            log_det = torch.zeros(N, device=device)
            for layer in reversed(self.layers_all_dim[jd]):
                x, ld = _apply_rqs(x, layer(cond_jd), self.K, self._bounds[jd], inverse=True)
                log_det += ld
            log_det_total += log_det
            z_out[:, jd] = x * (mask[:, jd] if mask is not None else 1.0)
        return z_out, log_det_total

    def sample(self, cond_inp=None, mask=None):
        x, _ = self.inverse(cond_inp, mask)
        return x


class NSF_M_all_uncond(nn.Module):
    """
    Unconditional NSF over a single scalar; parameters are nn.Parameter
    tensors rather than MLP outputs.

    forward(x) returns (logp, log_det, x_post_flow).
    inverse(ntot) returns (samples, log_det).
    """

    def __init__(
        self,
        dim=1,
        K=5,
        B=3,
        nflows=1,
        ngauss=1,
        base_dist='gauss',
        mu_pos=False,
        base_dist_pwall='pl_exp',
        lgM_rs_tointerp=None,
        hmf_pdf_tointerp=None,
        hmf_cdf_tointerp=None,
    ):
        super().__init__()
        self.K = K
        self.nflows = nflows
        self.ngauss = ngauss
        self.base_dist = base_dist
        self.mu_pos = mu_pos
        self.base_dist_pwall = base_dist_pwall
        self._bounds = _normalize_bounds(B, 1)

        if base_dist in ('gauss', 'halfgauss'):
            if ngauss == 1:
                p_size = 2
            elif base_dist_pwall == 'pl_exp':
                p_size = 3 * ngauss + 2
            else:
                p_size = 3 * ngauss
        elif base_dist in ('weibull', 'gumbel'):
            p_size = 2
        elif base_dist == 'physical_hmf':
            if lgM_rs_tointerp is None:
                raise ValueError("physical_hmf requires tabulated HMF arrays")
            self.register_buffer('lgM_rs',     torch.tensor([lgM_rs_tointerp],                   dtype=torch.float32))
            self.register_buffer('hmf_logpdf', torch.log(torch.tensor([hmf_pdf_tointerp],        dtype=torch.float32)))
            self.register_buffer('hmf_cdf',    torch.tensor([hmf_cdf_tointerp],                  dtype=torch.float32))
            p_size = 0
        else:
            raise ValueError(f"base_dist '{base_dist}' not recognised")

        if p_size > 0:
            self.initial_param = nn.Parameter(torch.zeros(p_size))
            init.uniform_(self.initial_param, -1.0, 1.0)

        self.layers = nn.ParameterList()
        for _ in range(nflows):
            p = nn.Parameter(torch.zeros(3 * K - 1))
            init.uniform_(p, -10.0, 10.0)
            self.layers.append(p)

    def _base_params(self):
        """Return scalar (or 1-D) base-distribution parameters from self.initial_param."""
        p = self.initial_param
        bd = self.base_dist
        if bd in ('gauss', 'halfgauss'):
            if self.ngauss == 1:
                mu, alpha = p[0], p[1]
                if self.mu_pos:
                    mu = (1.0 + torch.tanh(mu)) / 2.0
                return mu, torch.exp(alpha)
            # mixture — reuse batched helper with a fake batch dim then squeeze
            mu_all, var_all, pw_all = _get_gauss_params(
                p.unsqueeze(0), self.ngauss, self.mu_pos, self.base_dist_pwall
            )
            return mu_all[0], var_all[0], pw_all[0]   # each (ngauss,)
        mu, alpha = p[0], p[1]
        if bd == 'weibull':
            return torch.exp(mu), torch.exp(alpha)
        if self.mu_pos:
            mu = (1.0 + torch.tanh(mu)) / 2.0
        return mu, torch.exp(alpha)

    def _base_logp(self, x, params, dev):
        bd = self.base_dist
        if bd == 'gauss':
            if self.ngauss == 1:
                mu, var = params
                return -0.5 * math.log(2 * math.pi) - 0.5 * torch.log(var) - 0.5 * (x - mu) ** 2 / var
            mu_all, var_all, pw_all = params   # each (ngauss,) — broadcast over N
            Li = sum(
                pw_all[i] / torch.sqrt(2 * np.pi * var_all[i])
                * torch.exp(-0.5 * (x - mu_all[i]) ** 2 / var_all[i])
                for i in range(self.ngauss)
            )
            return torch.log(Li)
        if bd == 'halfgauss':
            mu, var = params
            lp = HalfNormal(torch.sqrt(var)).log_prob(torch.exp(x - mu))
        elif bd == 'weibull':
            lp = Weibull(*params, validate_args=False).log_prob(x)
        elif bd == 'gumbel':
            lp = Gumbel(*params, validate_args=False).log_prob(x)
        elif bd == 'physical_hmf':
            lp = interpolate(x[None, :], self.lgM_rs, self.hmf_logpdf)[0, :]
        else:
            raise ValueError(f"base_dist '{bd}' not recognised")
        return torch.where(torch.isfinite(lp), lp, torch.full_like(lp, -100.0))

    def forward(self, x: torch.Tensor):
        if x.dim() > 1:
            x = x[:, 0]
        params = self._base_params()
        log_det = torch.zeros_like(x)
        for p in self.layers:
            layer_out = p.unsqueeze(0).expand(x.shape[0], -1)
            x, ld = _apply_rqs(x, layer_out, self.K, self._bounds[0], inverse=False)
            log_det += ld
        logp = self._base_logp(x, params, x.device)
        return logp, log_det, x

    def inverse(self, ntot: int):
        dev = self.layers[0].device
        params = self._base_params()
        bd = self.base_dist
        if bd == 'gauss':
            if self.ngauss == 1:
                mu, var = params
                x = mu + torch.randn(ntot, device=dev) * torch.sqrt(var)
            else:
                mu_all, var_all, pw_all = params   # each (ngauss,)
                # expand to (ntot, ngauss) for _sample_gaussian_mixture
                x = _sample_gaussian_mixture(
                    mu_all.unsqueeze(0).expand(ntot, -1),
                    var_all.unsqueeze(0).expand(ntot, -1),
                    pw_all.unsqueeze(0).expand(ntot, -1),
                    dev,
                )
        elif bd == 'halfgauss':
            mu, var = params[0][0, 0], params[1][0, 0]
            x = torch.log(mu + torch.abs(torch.randn(ntot, device=dev)) * torch.sqrt(var))
        elif bd == 'weibull':
            x = Weibull(*params).sample([ntot])
        elif bd == 'gumbel':
            x = Gumbel(*params).sample([ntot])
        elif bd == 'physical_hmf':
            u = torch.rand(ntot, device=dev)
            x = interpolate(u[None, :], self.hmf_cdf[:, 1:], self.lgM_rs[:, 1:])[0, :]
        else:
            raise ValueError(f"base_dist '{bd}' not recognised")

        log_det = torch.zeros_like(x)
        for p in reversed(self.layers):
            layer_out = p.unsqueeze(0).expand(x.shape[0], -1)
            x, ld = _apply_rqs(x, layer_out, self.K, self._bounds[0], inverse=True)
            log_det += ld
        return x, log_det

    def sample(self, ntot: int):
        x, _ = self.inverse(ntot)
        return x


class M1_reg_model(nn.Module):
    """
    Deterministic regression head for M1 (rarely used; alternative to NSF_1var).
    Six-layer narrowing MLP with LeakyReLU(0.5).
    """

    def __init__(self, dim=1, hidden_dim=8, num_cond=0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_cond,        hidden_dim),      nn.LeakyReLU(0.5),
            nn.Linear(hidden_dim,      hidden_dim),      nn.LeakyReLU(0.5),
            nn.Linear(hidden_dim,      hidden_dim),      nn.LeakyReLU(0.5),
            nn.Linear(hidden_dim,      hidden_dim // 2), nn.LeakyReLU(0.5),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.LeakyReLU(0.5),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, cond_inp: torch.Tensor) -> torch.Tensor:
        return self.network(cond_inp)

    def inverse(self, cond_inp: torch.Tensor, mask=None) -> torch.Tensor:
        out = self.network(cond_inp)
        if mask is not None:
            out = out * mask
        return out[:, 0]
