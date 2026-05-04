import torch
import torch.nn as nn
import numpy as np


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
        return self.net(x)


class CHARM_Model(nn.Module):
    """
    Unified CHARM model: one shared encoder, up to seven property heads.

    The encoder is constructed externally (e.g. CNN3D_stackout_v2) and passed
    in; it is never built internally. All head models are optional — pass None
    to omit a head entirely.

    Property heads
    --------------
    binary_model      : per-voxel occupancy (BCE)
    multiclass_model  : per-voxel halo count (CE)
    m1_model          : heaviest-halo mass (NLL flow)
    mdiff_model       : autoregressive mass differences (NLL flow)
    vel_model         : per-halo 3D velocity  (NLL flow, ndim = 3*max_halos)
    conc_model        : per-halo concentration (NLL flow, ndim = max_halos)
    pos_model         : per-halo 3D sub-voxel position offset (NLL flow,
                        ndim = 3*max_halos), conditioned on Mhalos like vel.

    Conditioning
    ------------
    Explicit boolean flags replace the brittle num_cond arithmetic of the old
    combined models:
      cond_nhalos_on_m1   — prepend Nhalos to M1 conditioning
      cond_m1_on_mdiff    — prepend [Nhalos, M1] to Mdiff conditioning

    Cosmology routing (use_film):
      False (default) — encoder receives DM field only; cosmo vector is
                        concatenated to cond_out afterwards.
      True            — cosmo is passed to the encoder's FiLM generators;
                        nothing extra is concatenated. Requires the encoder to
                        accept a `cosmo` keyword argument (CNN3D_stackout_v2
                        with cosmo_dim > 0).

    cond_x_nsh (non-shifted local features) is always concatenated to
    cond_out after the encoder, regardless of use_film.

    Parameters
    ----------
    encoder : nn.Module
        Pre-built shared feature encoder.
    ndim : int
        Maximum halos modelled per voxel.  Vel and pos heads use ndim*3
        outputs; conc uses ndim; M1 uses 1; Mdiff uses ndim-1.
    cond_nhalos_on_m1, cond_m1_on_mdiff : bool
        See above.
    use_film : bool
        See above.
    sep_*_cond : bool
        If True, route each mass-head's conditioning through a small FCNN
        projector before the head sees it.
    num_cond_* : int or None
        Input/output width of the corresponding FCNN projector.  Required
        when sep_*_cond is True.
    """

    def __init__(
        self,
        encoder: nn.Module,
        ndim: int,
        # heads — all optional
        binary_model=None,
        multiclass_model=None,
        m1_model=None,
        mdiff_model=None,
        vel_model=None,
        conc_model=None,
        pos_model=None,
        # conditioning flags
        cond_nhalos_on_m1: bool = True,
        cond_m1_on_mdiff: bool = True,
        # cosmology routing
        use_film: bool = False,
        concat_cosmo_after_film: bool = False,
        # optional FCNN projectors for the four mass heads
        sep_binary_cond: bool = False,    num_cond_binary: int = None,
        sep_multi_cond: bool = False,     num_cond_multi: int = None,
        sep_m1_cond: bool = False,        num_cond_m1: int = None,
        sep_mdiff_cond: bool = False,     num_cond_mdiff: int = None,
        # ── binary head class-imbalance handling ───────────────────────
        # 'none'      : original behaviour (per-voxel mean over all selected voxels)
        # 'subsample' : keep all occupied voxels + an equal-size random
        #               draw of empty voxels, perfect 1:1 batch balance
        # 'alpha'     : per-voxel inverse-frequency weights, mean-1 normalised
        # 'focal'     : focal loss (1 - p_correct)^gamma weighting
        binary_loss_mode: str = 'none',
        binary_focal_gamma: float = 2.0,
        # Prior under which the binary head was trained. Subsample and
        # alpha both train under a balanced prior (0.5); the trained pw_occ
        # is therefore p(occ | x, π=0.5) and needs Bayesian renormalisation
        # at inference to recover the true posterior under π_target. If
        # None, no correction is applied (correct for 'none' / 'focal').
        binary_train_prior: float = None,
        # kept for downstream compatibility
        priors_all=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.ndim = ndim
        self.cond_nhalos_on_m1 = cond_nhalos_on_m1
        self.cond_m1_on_mdiff = cond_m1_on_mdiff
        self.use_film = use_film
        self.concat_cosmo_after_film = concat_cosmo_after_film
        self.priors_all = priors_all

        if binary_loss_mode not in ('none', 'subsample', 'alpha', 'focal'):
            raise ValueError(
                f"binary_loss_mode must be one of 'none', 'subsample', "
                f"'alpha', 'focal'; got {binary_loss_mode!r}"
            )
        self.binary_loss_mode = binary_loss_mode
        self.binary_focal_gamma = float(binary_focal_gamma)
        self.binary_train_prior = (
            None if binary_train_prior is None else float(binary_train_prior)
        )

        # heads
        self.binary_model = binary_model
        self.multiclass_model = multiclass_model
        self.m1_model = m1_model
        self.mdiff_model = mdiff_model
        self.vel_model = vel_model
        self.conc_model = conc_model
        self.pos_model = pos_model

        # FCNN projectors stored in a ModuleDict so parameters are registered
        proj = {}
        for name, active, dim in [
            ('binary',  sep_binary_cond, num_cond_binary),
            ('multi',   sep_multi_cond,  num_cond_multi),
            ('m1',      sep_m1_cond,     num_cond_m1),
            ('mdiff',   sep_mdiff_cond,  num_cond_mdiff),
        ]:
            if active:
                if dim is None:
                    raise ValueError(f"num_cond_{name} must be set when sep_{name}_cond=True")
                proj[name] = FCNN(dim, dim, dim)
        self.proj_layers = nn.ModuleDict(proj)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode(
        self,
        cond_x_jb: torch.Tensor,
        cond_x_nsh_jb: torch.Tensor,
        cond_cosmo_jb,
        LOCAL_BIASING: bool = False,
    ) -> torch.Tensor:
        """Run encoder and assemble per-voxel conditioning vector."""
        if LOCAL_BIASING:
            cond_out = cond_x_nsh_jb
        else:
            if self.use_film:
                # FiLM needs cosmo per sub-cube (nsims, ncosmo); cond_cosmo_jb
                # is (nsims * nvox, ncosmo) — all voxels in a sim share cosmo,
                # so stride by nvox = total_vox // n_sims to extract one row per sim.
                n_sims = cond_x_jb.shape[0]
                nvox = cond_cosmo_jb.shape[0] // n_sims
                cosmo_per_sim = cond_cosmo_jb[::nvox]          # (nsims, ncosmo)
                cond_out = self.encoder(cond_x_jb, cosmo=cosmo_per_sim)
                # Optionally also concatenate raw cosmo to give heads a direct
                # linear pathway to cosmological parameters (dual conditioning).
                if self.concat_cosmo_after_film:
                    cond_out = torch.cat([cond_out, cond_cosmo_jb], dim=1)
            else:
                cond_out = self.encoder(cond_x_jb)
                if cond_cosmo_jb is not None:
                    cond_out = torch.cat([cond_out, cond_cosmo_jb], dim=1)
            if cond_x_nsh_jb is not None:
                cond_out = torch.cat([cond_out, cond_x_nsh_jb], dim=1)
        return cond_out

    @staticmethod
    def _build_halo_mask(
        ntot: np.ndarray,
        ndim: int,
        vel_style: bool = False,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Build a float occupancy mask from per-voxel halo counts.

        ntot      : (N_voxels,) int array
        ndim      : slots per voxel
        vel_style : if True, repeat each halo slot 3× (for 3D properties)
        returns   : (N_voxels, ndim) or (N_voxels, 3*ndim) float tensor
        """
        ntot_t = torch.from_numpy(ntot.astype(np.int64))          # (N,)
        idx = torch.arange(ndim)                                    # (ndim,)
        mask = (idx.unsqueeze(0) < ntot_t.unsqueeze(1)).float()    # (N, ndim)
        if vel_style:
            mask = mask.unsqueeze(-1).expand(-1, -1, 3).reshape(mask.shape[0], -1)
        if device is not None:
            mask = mask.to(device)
        return mask

    @torch._dynamo.disable
    def _apply_proj(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name in self.proj_layers:
            return self.proj_layers[name](x)
        return x

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    @torch._dynamo.disable
    def forward(
        self,
        # targets — pass None for heads that are not trained
        x_binary,
        x_multi,
        x_m1,
        x_mdiff,
        x_vel,
        x_conc,
        x_pos,
        # per-halo flow masks
        mask_m1,
        mask_mdiff,
        mask_vel,
        mask_conc,
        mask_pos,
        # conditioning inputs  (leading axis = nbatches outer sub-cubes)
        cond_x,
        cond_x_nsh,
        cond_cosmo,
        # truth conditioning (used as input to downstream heads)
        nhalos_truth,   # (nbatches, N_voxels, 1)
        m1_truth,       # (nbatches, N_voxels, 1)
        mhalos_truth,   # (nbatches, N_voxels, ndim)  for vel/conc/pos
        # optional binary-head voxel selection
        mask_ntot=None,
        # which heads contribute to the loss
        heads_to_train=frozenset(['binary', 'multi', 'm1', 'mdiff']),
        LOCAL_BIASING: bool = False,
    ) -> dict:
        """
        Compute per-head NLL / CE / BCE losses.

        Returns
        -------
        dict mapping head name → scalar loss tensor.
        The caller sums (and optionally weights) the values.
        """
        device = cond_x.device
        nbatches = cond_x.shape[0]

        # Per-head accumulators: track Σ NLL and Σ #voxels separately so that
        # the final loss is a true voxel-weighted mean
        #     loss[h] = (Σ_b Σ_voxel NLL_voxel) / (Σ_b N_voxels_b)
        # rather than the previous Σ_b mean_b(NLL), which (i) scaled with the
        # outer batch count (coupling the loss magnitude to nsims_per_batch)
        # and (ii) implicitly down-weighted heads whose per-batch mask had
        # fewer voxels relative to heads that saw all voxels every batch.
        #
        # All head .forward()s in this codebase return per-voxel scalars of
        # shape (N,), so .numel() == N is the correct voxel count. Sum and
        # count are accumulated in float32 for stable reductions even when
        # the surrounding autocast region is bfloat16.
        loss_sum = {h: torch.zeros((), device=device, dtype=torch.float32)
                    for h in heads_to_train}
        count    = {h: 0 for h in heads_to_train}

        def _accum(name: str, L: torch.Tensor):
            loss_sum[name] = loss_sum[name] + L.sum()
            count[name] += L.numel()
        _accum = torch._dynamo.disable(_accum)

        for jb in range(nbatches):
            cond_out = self._encode(
                cond_x[jb], cond_x_nsh[jb],
                cond_cosmo[jb] if cond_cosmo is not None else None,
                LOCAL_BIASING,
            )

            n_vox = cond_out.shape[0]
            nhalos_jb = nhalos_truth[jb].to(device)          # (N_vox, 1)
            mask_occ = torch.where(nhalos_jb[:, 0] > 0)[0]   # voxels with ≥1 halo
            mask_gt1 = torch.where(nhalos_jb[:, 0] > 1)[0]   # voxels with ≥2 halos

            # ---- binary ------------------------------------------------
            # At low occupancy (e.g. Mmin=1e14, ~0.26%), an unbalanced
            # per-voxel mean lets the optimizer settle into a degenerate
            # plateau where pw_occ stays small for ALL voxels and the loss
            # gap to a perfect discriminator is only ~0.012. The four
            # modes below restore a usable gradient signal:
            #   subsample : exact 1:1 class balance via empty-voxel sub-draw
            #   alpha     : per-voxel inverse-frequency weights (mean-1)
            #   focal     : (1 - p_correct)^gamma weighting (Lin et al. 2017)
            #   none      : legacy mean-over-all-voxels behaviour
            if 'binary' in heads_to_train:
                sel = mask_ntot[jb].to(device) if mask_ntot is not None \
                      else torch.arange(n_vox, device=device)
                if sel.numel() > 0:
                    x_bin_sel = x_binary[jb][sel]            # (M, 1)
                    y_bin     = (x_bin_sel[:, 0] > 0.5)      # (M,) bool

                    if self.binary_loss_mode == 'subsample':
                        # Keep all occupied voxels + an equal-size random draw
                        # of empty voxels. The loss is averaged over the
                        # smaller balanced set, so per-voxel signal is fine.
                        occ_idx = torch.where(y_bin)[0]
                        emp_idx = torch.where(~y_bin)[0]
                        n_keep  = max(1, occ_idx.numel())
                        if emp_idx.numel() > n_keep:
                            perm = torch.randperm(
                                emp_idx.numel(), device=device
                            )[:n_keep]
                            emp_idx = emp_idx[perm]
                        sub = torch.cat([occ_idx, emp_idx])
                        if sub.numel() > 0:
                            cond_b = self._apply_proj(
                                'binary', cond_out[sel][sub]
                            )
                            _accum('binary',
                                   self.binary_model.forward(
                                       x_bin_sel[sub], cond_b))

                    elif self.binary_loss_mode == 'alpha':
                        # Per-voxel inverse-frequency weights, normalised so
                        # the mean weight = 1 in expectation.
                        cond_b = self._apply_proj('binary', cond_out[sel])
                        raw_nll = self.binary_model.forward(x_bin_sel, cond_b)
                        n_total = y_bin.numel()
                        n_occ   = y_bin.sum().clamp(min=1).float()
                        n_emp   = (~y_bin).sum().clamp(min=1).float()
                        w_occ   = n_total / (2.0 * n_occ)
                        w_emp   = n_total / (2.0 * n_emp)
                        alpha   = torch.where(y_bin, w_occ, w_emp).detach()
                        _accum('binary', raw_nll * alpha)

                    elif self.binary_loss_mode == 'focal':
                        cond_b = self._apply_proj('binary', cond_out[sel])
                        raw_nll = self.binary_model.forward(x_bin_sel, cond_b)
                        # Compute mixing weights for focal modulation under
                        # no_grad — the focal weight is a constant scaling
                        # factor; gradients flow only through raw_nll.
                        # This avoids a redundant forward pass through layer_init
                        # (forward() already called it above).
                        with torch.no_grad():
                            out_b = self.binary_model.layer_init(cond_b)
                            pw    = self.binary_model._mixing_weights(out_b)
                        # pw[:, 0] = pw_empty (mu=0); pw[:, 1] = pw_occ (mu=1)
                        p_correct = torch.where(y_bin, pw[:, 1], pw[:, 0])
                        focal_w = (1.0 - p_correct).pow(self.binary_focal_gamma)
                        _accum('binary', raw_nll * focal_w)

                    else:  # 'none' — legacy behaviour
                        cond_b = self._apply_proj('binary', cond_out[sel])
                        _accum('binary',
                               self.binary_model.forward(x_bin_sel, cond_b))

            # mask_m1[jb] has shape (N_vox, 1); use torch.where to get a 1D
            # index tensor — avoids the (M, 2) result from nonzero().squeeze()
            # which corrupts the MLP input when M > 1.  Also skip heads when
            # no occupied voxels exist in this batch (prevents nan from mean()).
            mask_sel_occ = torch.where(mask_m1[jb][:, 0] > 0)[0]  # (M,)

            # ---- multiclass --------------------------------------------
            if 'multi' in heads_to_train and mask_sel_occ.numel() > 0:
                cond_mc = self._apply_proj('multi', cond_out[mask_sel_occ])
                _accum('multi',
                       self.multiclass_model.forward(
                           x_multi[jb][mask_sel_occ], cond_mc))

            # ---- M1 ----------------------------------------------------
            if 'm1' in heads_to_train and mask_sel_occ.numel() > 0:
                cond_m1 = cond_out
                if self.cond_nhalos_on_m1:
                    cond_m1 = torch.cat([nhalos_jb, cond_m1], dim=1)
                cond_m1 = self._apply_proj('m1', cond_m1[mask_sel_occ])
                _accum('m1',
                       -self.m1_model.forward(x_m1[jb][mask_sel_occ], cond_m1))

            # ---- Mdiff -------------------------------------------------
            if 'mdiff' in heads_to_train and mask_gt1.numel() > 0:
                mask_sel = mask_gt1.to(device)
                m1_jb = m1_truth[jb].to(device)
                cond_md = cond_out
                if self.cond_m1_on_mdiff:
                    cond_md = torch.cat([nhalos_jb, m1_jb, cond_md], dim=1)
                cond_md = self._apply_proj('mdiff', cond_md[mask_sel])
                _accum('mdiff',
                       -self.mdiff_model.forward(
                           x_mdiff[jb][mask_sel], cond_md,
                           mask_mdiff[jb][mask_sel],
                       ))

            # ---- vel / conc / pos  (all share the same conditioning) ----
            if any(h in heads_to_train for h in ('vel', 'conc', 'pos')) \
                    and mask_occ.numel() > 0:
                mhalos_jb = mhalos_truth[jb].to(device)
                cond_prop = torch.cat([mhalos_jb, cond_out], dim=1)
                sel = mask_occ

                if 'vel' in heads_to_train:
                    _accum('vel',
                           -self.vel_model.forward(
                               x_vel[jb][sel], cond_prop[sel],
                               mask_vel[jb][sel],
                           ))

                if 'conc' in heads_to_train:
                    _accum('conc',
                           -self.conc_model.forward(
                               x_conc[jb][sel], cond_prop[sel],
                               mask_conc[jb][sel],
                           ))

                if 'pos' in heads_to_train:
                    _accum('pos',
                           -self.pos_model.forward(
                               x_pos[jb][sel], cond_prop[sel],
                               mask_pos[jb][sel],
                           ))

        # Finalise: per-voxel mean per head. If a head was active in
        # heads_to_train but its mask was empty in every batch (count == 0),
        # return a zero scalar — autograd graph is empty for that head, so it
        # contributes no gradient and the caller's sum-of-losses still works.
        losses = {}
        for h in heads_to_train:
            if count[h] > 0:
                losses[h] = loss_sum[h] / count[h]
            else:
                losses[h] = loss_sum[h]   # already a 0-d zero
        return losses

    # ------------------------------------------------------------------
    # Sample (inference)
    # ------------------------------------------------------------------

    def sample(
        self,
        cond_x,
        cond_x_nsh,
        cond_cosmo,
        # truth tensors (used when the corresponding head is not sampled)
        nhalos_truth=None,
        m1_truth=None,
        mhalos_truth=None,
        # truth values used when the corresponding head is not sampled
        mdiff_truth=None,
        # which heads to run; False → substitute truth
        sample_binary: bool = True,
        sample_multi: bool = True,
        sample_m1: bool = True,
        sample_mdiff: bool = True,
        sample_vel: bool = True,
        sample_conc: bool = True,
        sample_pos: bool = True,
        # condition vel/conc/pos on sampled masses (False → use truth masses)
        use_truth_masses: bool = False,
        # True occupancy fraction at inference (for Bayesian prior
        # correction). If both this and self.binary_train_prior are set,
        # pw_occ is renormalised before Bernoulli sampling. Required when
        # the binary head was trained with subsample / alpha (where the
        # trained pw_occ is calibrated under π=0.5, not under π_true).
        binary_target_prior: float = None,
        LOCAL_BIASING: bool = False,
    ) -> dict:
        """
        Run the full sampling pipeline.

        Returns
        -------
        dict with keys 'ntot', 'm1', 'mdiff', 'vel', 'conc', 'pos';
        each maps to a list (one entry per outer batch) of numpy arrays or
        torch tensors on CPU.
        """
        device = cond_x.device
        nbatches = cond_x.shape[0]

        out = {k: [] for k in ('ntot', 'm1', 'mdiff', 'vel', 'conc', 'pos')}

        for jb in range(nbatches):
            cond_out = self._encode(
                cond_x[jb], cond_x_nsh[jb],
                cond_cosmo[jb] if cond_cosmo is not None else None,
                LOCAL_BIASING,
            )
            n_vox = cond_out.shape[0]

            # ---- binary: which voxels are occupied ---------------------
            if sample_binary:
                # Apply sep_binary projector before inverse, matching forward().
                cond_b = self._apply_proj('binary', cond_out)

                if (self.binary_train_prior is not None
                        and binary_target_prior is not None):
                    # Bayesian prior correction. Subsample / alpha train
                    # under π_train=0.5; the trained pw_occ is therefore
                    # p(occ|x, π=0.5). To recover p(occ|x, π=π_target):
                    #     odds_target = odds_train * r,
                    #     r = (π_target/π_train) * ((1-π_train)/(1-π_target))
                    #     p_target = r·pw / (1 + (r-1)·pw)
                    # Then sample Bernoulli directly from the corrected
                    # probability — bypassing the GMM-noise sample of
                    # binary_model.inverse() (which uses uncorrected pw).
                    out_b   = self.binary_model.layer_init(cond_b)
                    pw      = self.binary_model._mixing_weights(out_b)
                    pw_occ  = pw[:, 1].clamp(1e-12, 1.0 - 1e-12)
                    p_tr    = float(self.binary_train_prior)
                    p_tg    = float(binary_target_prior)
                    r = (p_tg / p_tr) * ((1.0 - p_tr) / (1.0 - p_tg))
                    pw_occ_corr = r * pw_occ / (1.0 + (r - 1.0) * pw_occ)
                    occ_mask = torch.bernoulli(pw_occ_corr).float()
                else:
                    # binary_model.inverse() returns a GMM sample near 0
                    # (empty) or 1 (occupied) with sigma=0.05; threshold
                    # at 0.5 to obtain a hard occupancy mask.
                    samp_binary = self.binary_model.inverse(cond_b)
                    occ_mask = (samp_binary >= 0.5).float()
            else:
                occ_mask = (nhalos_truth[jb, :, 0] > 0).float().to(device)

            mask_occ_idx = torch.where(occ_mask > 0)[0]

            # ---- multiclass: Nhalos in occupied voxels -----------------
            ntot_samp = torch.zeros(n_vox, device=device)
            if sample_multi:
                # Apply sep_multi projector, matching forward().
                cond_mc = self._apply_proj('multi', cond_out[mask_occ_idx])
                mc_out = self.multiclass_model.inverse(cond_mc)
                ntot_samp[mask_occ_idx] = torch.clamp(torch.round(mc_out), min=1, max=self.ndim)
            else:
                ntot_samp = nhalos_truth[jb, :, 0].float().to(device)

            ntot_np = ntot_samp.cpu().numpy()
            out['ntot'].append(ntot_np)

            occ_gt0 = torch.where(ntot_samp > 0)[0]
            occ_gt1 = torch.where(ntot_samp > 1)[0]

            # ---- M1 ----------------------------------------------------
            m1_samp_all = torch.zeros(n_vox, device=device)
            if sample_m1:
                cond_m1 = cond_out
                if self.cond_nhalos_on_m1:
                    cond_m1 = torch.cat(
                        [ntot_samp.unsqueeze(1), cond_m1], dim=1
                    )
                # Apply sep_m1 projector, matching forward().
                cond_m1 = self._apply_proj('m1', cond_m1)
                mask_m1_samp = self._build_halo_mask(
                    ntot_np, 1, device=device
                )
                m1_samp, _ = self.m1_model.inverse(
                    cond_m1[occ_gt0], mask_m1_samp[occ_gt0]
                )
                m1_samp_all[occ_gt0] = m1_samp.reshape(-1)
            else:
                m1_samp_all = m1_truth[jb, :, 0].float().to(device)
            out['m1'].append(m1_samp_all.cpu())

            # ---- Mdiff -------------------------------------------------
            mdiff_samp_all = torch.zeros(n_vox, self.ndim - 1, device=device)
            if sample_mdiff:
                cond_md = cond_out
                if self.cond_m1_on_mdiff:
                    cond_md = torch.cat(
                        [ntot_samp.unsqueeze(1),
                         m1_samp_all.unsqueeze(1), cond_md], dim=1
                    )
                # Apply sep_mdiff projector, matching forward().
                cond_md = self._apply_proj('mdiff', cond_md)
                mask_mdiff_samp = self._build_halo_mask(
                    np.clip(ntot_np - 1, 0, None), self.ndim - 1, device=device
                )
                if occ_gt1.numel() > 0:
                    mdiff_samp, _ = self.mdiff_model.inverse(
                        cond_md[occ_gt1], mask_mdiff_samp[occ_gt1]
                    )
                    mdiff_samp_all[occ_gt1] = mdiff_samp
            else:
                if mdiff_truth is not None:
                    mdiff_samp_all = mdiff_truth[jb].to(device)
            out['mdiff'].append(mdiff_samp_all.cpu())

            # ---- build mass conditioning for vel/conc/pos --------------
            if use_truth_masses:
                mhalos_cond = mhalos_truth[jb].to(device)
            else:
                # Training conditions vel/conc/pos on M_norm, which stores
                # individual normalised masses [M1, M2, ..., MNmax].  Absent
                # halo slots are zero (raw catalog zero → rescale_sub=0 after
                # normalize_masses clamp). Mdiff_norm stores *differences*:
                #   Mdiff[j] = M_norm[j] - M_norm[j+1], zeroed for absent slots.
                # Two-step reconstruction to match the training representation:
                # 1. Cumulative subtraction to recover individual masses.
                # 2. Zero out absent halo slots (ntot_samp-based mask).
                m_all = torch.zeros(n_vox, self.ndim, device=device)
                m_all[:, 0] = m1_samp_all
                for j in range(1, self.ndim):
                    m_all[:, j] = (m_all[:, j - 1] - mdiff_samp_all[:, j - 1]).clamp(0.0, 1.0)
                halo_slot_mask = self._build_halo_mask(ntot_np, self.ndim, device=device)
                mhalos_cond = m_all * halo_slot_mask

            cond_prop = torch.cat([mhalos_cond, cond_out], dim=1)

            # ---- vel ---------------------------------------------------
            vel_samp_all = torch.zeros(n_vox, self.ndim * 3, device=device)
            if sample_vel and self.vel_model is not None:
                mask_vel_samp = self._build_halo_mask(
                    ntot_np, self.ndim, vel_style=True, device=device
                )
                if occ_gt0.numel() > 0:
                    vel_samp, _ = self.vel_model.inverse(
                        cond_prop[occ_gt0], mask_vel_samp[occ_gt0]
                    )
                    vel_samp_all[occ_gt0] = vel_samp
            out['vel'].append(vel_samp_all.cpu())

            # ---- conc --------------------------------------------------
            conc_samp_all = torch.zeros(n_vox, self.ndim, device=device)
            if sample_conc and self.conc_model is not None:
                mask_conc_samp = self._build_halo_mask(
                    ntot_np, self.ndim, device=device
                )
                if occ_gt0.numel() > 0:
                    conc_samp, _ = self.conc_model.inverse(
                        cond_prop[occ_gt0], mask_conc_samp[occ_gt0]
                    )
                    conc_samp_all[occ_gt0] = conc_samp
            out['conc'].append(conc_samp_all.cpu())

            # ---- pos ---------------------------------------------------
            pos_samp_all = torch.zeros(n_vox, self.ndim * 3, device=device)
            if sample_pos and self.pos_model is not None:
                mask_pos_samp = self._build_halo_mask(
                    ntot_np, self.ndim, vel_style=True, device=device
                )
                if occ_gt0.numel() > 0:
                    pos_samp, _ = self.pos_model.inverse(
                        cond_prop[occ_gt0], mask_pos_samp[occ_gt0]
                    )
                    pos_samp_all[occ_gt0] = pos_samp
            out['pos'].append(pos_samp_all.cpu())

        return out
