"""
Multi-panel histogram of all CHARM training shard quantities.
Masked (zero-padded) values are excluded from every panel.
"""
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Load shard ────────────────────────────────────────────────────────────────
ldir  = '/mnt/ceph/users/spandey/CHARM_v2/data/trial_Mmin1e14/shards/'
fname = ldir + 'CHARM_train_shard_0.h5'

with h5.File(fname, 'r') as df:
    M1_norm   = df['halos']['M1_norm'][:]      # (n_outer, N_vox, 1)
    N_halos   = df['halos']['N_halos'][:]      # (n_outer, N_vox)
    Mdiff_norm = df['halos']['Mdiff_norm'][:]  # (n_outer, N_vox, Nmax-1=3)
    c_norm    = df['halos']['c_norm'][:]       # (n_outer, N_vox, Nmax=4)
    pos_norm  = df['halos']['pos_norm'][:]     # (n_outer, N_vox, Nmax*3=12)
    v_norm    = df['halos']['v_norm'][:]       # (n_outer, N_vox, Nmax*3=12)

print(f"Shapes — M1:{M1_norm.shape}  N:{N_halos.shape}  Mdiff:{Mdiff_norm.shape}  "
      f"c:{c_norm.shape}  pos:{pos_norm.shape}  v:{v_norm.shape}")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
PALETTE   = plt.cm.tab10.colors
BG        = '#fafafa'
BINS      = 80
ALPHA     = 0.65
LW        = 1.3

def nonzero(arr):
    """Flatten and keep only strictly non-zero entries."""
    flat = arr.ravel().astype(np.float32)
    return flat[flat != 0.0]

def nonzero_component(arr, comp):
    """Return non-zero values of component `comp` (last axis)."""
    return nonzero(arr[..., comp])

# ── Figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.patch.set_facecolor(BG)
axes = axes.ravel()

for ax in axes:
    ax.set_facecolor(BG)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 0 — N_halos  (integer counts)
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0]
n_flat = N_halos.ravel().astype(np.int32)
n_masked = n_flat[n_flat > 0]          # 0 means empty voxel
vals, cnts = np.unique(n_masked, return_counts=True)
ax.bar(vals, cnts, color=PALETTE[0], alpha=0.8, edgecolor='white', linewidth=0.8,
       zorder=3)
ax.set_yscale('log')
ax.set_xlabel(r'$N_{\rm halos}$ per voxel', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Halo occupancy', fontsize=13, fontweight='bold')
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.text(0.97, 0.95, f'N = {len(n_masked):,}', transform=ax.transAxes,
        ha='right', va='top', fontsize=9, color='grey')

# ─────────────────────────────────────────────────────────────────────────────
# Panel 1 — M1_norm  (single component)
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1]
m1_flat = nonzero(M1_norm)
ax.hist(m1_flat, bins=BINS, color=PALETTE[1], alpha=ALPHA, edgecolor='none',
        log=True, zorder=3)
ax.set_xlabel(r'$M_1$ (normalised)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Heaviest halo mass', fontsize=13, fontweight='bold')
ax.text(0.97, 0.95, f'N = {len(m1_flat):,}', transform=ax.transAxes,
        ha='right', va='top', fontsize=9, color='grey')

# ─────────────────────────────────────────────────────────────────────────────
# Panel 2 — Mdiff_norm  (3 components = Δ1, Δ2, Δ3)
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[2]
Ncomp_mdiff = Mdiff_norm.shape[-1]
for i in range(Ncomp_mdiff):
    d = nonzero_component(Mdiff_norm, i)
    label = rf'$\Delta M_{i+1}$'
    ax.hist(d, bins=BINS, color=PALETTE[i+2], alpha=ALPHA, edgecolor='none',
            log=True, label=label, zorder=3 - i * 0.1)
ax.set_xlabel(r'$\Delta M_i$ (normalised)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Mass differences', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.7)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 3 — c_norm  (4 components = halos 1–4)
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[3]
Ncomp_c = c_norm.shape[-1]
for i in range(Ncomp_c):
    d = nonzero_component(c_norm, i)
    label = rf'halo {i+1}'
    ax.hist(d, bins=BINS, color=PALETTE[i], alpha=ALPHA, edgecolor='none',
            log=True, label=label, zorder=3 - i * 0.1)
ax.set_xlabel(r'$c_{\rm norm}$', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Concentration (normalised)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.7)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 4 — pos_norm  (12 = 4 halos × 3 directions; overlay x/y/z)
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[4]
dir_labels = ['x', 'y', 'z']
dir_colors = [PALETTE[0], PALETTE[1], PALETTE[2]]
Nmax = pos_norm.shape[-1] // 3
for d_idx, (dlabel, dcolor) in enumerate(zip(dir_labels, dir_colors)):
    # Gather this direction across all Nmax halos
    comp_indices = [d_idx + 3 * h for h in range(Nmax)]
    vals = np.concatenate([nonzero_component(pos_norm, ci) for ci in comp_indices])
    ax.hist(vals, bins=BINS, color=dcolor, alpha=ALPHA, edgecolor='none',
            log=True, label=dlabel, zorder=3 - d_idx * 0.1)
ax.set_xlabel(r'$\Delta x_i / L_{\rm vox}$', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Sub-voxel position (normalised)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.7, title='direction')

# ─────────────────────────────────────────────────────────────────────────────
# Panel 5 — v_norm  (12 = 4 halos × 3 directions; overlay vx/vy/vz)
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[5]
for d_idx, (dlabel, dcolor) in enumerate(zip(dir_labels, dir_colors)):
    comp_indices = [d_idx + 3 * h for h in range(Nmax)]
    vals = np.concatenate([nonzero_component(v_norm, ci) for ci in comp_indices])
    ax.hist(vals, bins=BINS, color=dcolor, alpha=ALPHA, edgecolor='none',
            log=True, label=f'$v_{dlabel}$', zorder=3 - d_idx * 0.1)
ax.set_xlabel(r'$v_i$ (normalised)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Velocity (normalised)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.7)

# ── Shared polish ─────────────────────────────────────────────────────────────
for ax in axes:
    ax.tick_params(labelsize=10)
    ax.grid(True, which='both', alpha=0.3, linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('CHARM shard data — distributions of halo properties\n'
             '(masked/empty-voxel zeros removed)',
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()

out_path = ldir + 'shard_histograms.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"Saved → {out_path}")
plt.show()
