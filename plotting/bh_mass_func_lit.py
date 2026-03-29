#!/usr/bin/env python3
"""
plot_bhmf.py
------------
Black Hole Mass Function plotter.

Overlays SAGE semi-analytic model predictions (solid lines, plasma colourmap)
against a curated library of observational/theoretical literature datasets
(coloured dashed lines + shaded uncertainty bands) across six redshift panels:
z ≈ 0, 1, 2, 4, 6, 8.

This version creates TWO plots:
1. A full view of all literature datasets.
2. A zoomed-in version of the same plot.

Directory layout expected (run the script from this folder):
    bh_mass_function_only/
    ├── plot_bhmf.py          ← this file
    ├── BHMF_txt/             ← literature .txt files
    │   ├── MH08.txt
    │   ├── HY24.txt
    │   └── ...
    └── output/               ← created automatically; figures saved here

If the SAGE HDF5 file is not found the script continues gracefully with
literature data only.
"""

import os
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Final, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings("ignore")


# ==============================================================================
# Constants
# ==============================================================================

MASS_UNIT_FACTOR: Final[float] = 1.0e10   # Internal mass unit → solar masses (pre-h)
COMMENT_CHAR:     Final[str]   = '#'

# All 64 Millennium snapshot redshifts (index 0 = z=127, index 63 = z=0)
MILLENNIUM_REDSHIFTS: Final[List[float]] = [
    127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343,
     14.086, 12.941, 11.897, 10.944, 10.073,  9.278,  8.550,  7.883,
      7.272,  6.712,  6.197,  5.724,  5.289,  4.888,  4.520,  4.179,
      3.866,  3.576,  3.308,  3.060,  2.831,  2.619,  2.422,  2.239,
      2.070,  1.913,  1.766,  1.630,  1.504,  1.386,  1.276,  1.173,
      1.078,  0.989,  0.905,  0.828,  0.755,  0.687,  0.624,  0.564,
      0.509,  0.457,  0.408,  0.362,  0.320,  0.280,  0.242,  0.208,
      0.175,  0.144,  0.116,  0.089,  0.064,  0.041,  0.020,  0.000,
]


# ==============================================================================
# Configuration
# ==============================================================================

# Paths are resolved relative to this script so the code runs from any CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class SimConfig:
    """Simulation file and cosmology settings."""
    dir_name:        str   = '../output/my_mini_millennium'
    file_name:       str   = 'model_0.hdf5'
    hubble_h:        float = 0.73
    box_size:        float = 500 #62.5           # h^-1 Mpc
    volume_fraction: float = 1.0
    redshifts:       List[float] = field(default_factory=lambda: MILLENNIUM_REDSHIFTS)

    @property
    def filepath(self) -> str:
        return os.path.join(self.dir_name, self.file_name)

    @property
    def volume(self) -> float:
        """Comoving volume in Mpc^3 (h-corrected)."""
        return (self.box_size / self.hubble_h) ** 3.0 * self.volume_fraction


@dataclass
class PlotConfig:
    """Plotting, I/O, and histogram settings."""
    # Output formats
    output_pdf:     bool  = True
    output_png:     bool  = True

    # Redshift panel centres
    bhmf_redshifts: List[float] = field(
        default_factory=lambda: [0.0, 1.0, 2.0, 4.0, 6.0, 8.0]
    )

    # ± tolerance for matching a literature z value to each panel
    z_half:         List[float] = field(
        default_factory=lambda: [0.4, 0.4, 0.6, 0.8, 1.0, 1.5]
    )

    # Histogram settings
    mass_bin_min:   float = 5.0
    mass_bin_max:   float = 11.5
    mass_bin_width: float = 0.25

    # Axis limits (both in log10 space)
    xlim:           tuple = (5.5, 10.5)
    ylim:           tuple = (-8.0, -1.0)

    # Colourmap used for SAGE model lines
    cmap_name:      str   = 'plasma'

    # Directories (relative to this script)
    lit_data_dir:   str   = os.path.join(_HERE, 'BHMF_txt')
    output_dir:     str   = os.path.join(_HERE, 'output')


# ==============================================================================
# Publication style
# ==============================================================================

def apply_publication_style() -> None:
    """Apply rcParams for a clean, journal-ready figure style."""
    plt.rcParams.update({
        'figure.facecolor':  'white',
        'figure.dpi':        150,
        'axes.facecolor':    'white',
        'axes.edgecolor':    'black',
        'axes.linewidth':    1.0,
        'axes.labelcolor':   'black',
        'axes.labelsize':    13,
        'xtick.color':       'black',
        'ytick.color':       'black',
        'xtick.direction':   'in',
        'ytick.direction':   'in',
        'xtick.top':         True,
        'ytick.right':       True,
        'xtick.major.size':  6,
        'ytick.major.size':  6,
        'xtick.minor.size':  3,
        'ytick.minor.size':  3,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.labelsize':   12,
        'ytick.labelsize':   12,
        'font.family':       'serif',
        'font.size':         13,
        'mathtext.fontset':  'stix',
        'legend.facecolor':  'white',
        'legend.edgecolor':  '0.7',
        'legend.fontsize':   9,
        'legend.framealpha': 0.9,
        'lines.linewidth':   1.8,
        'text.color':        'black',
    })


# ==============================================================================
# SAGE simulation I/O
# ==============================================================================

def _read_hdf(filepath: str, snap_key: str, param: str) -> np.ndarray:
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required to read SAGE output. "
                          "Install it with:  pip install h5py")
    with h5py.File(filepath, 'r') as f:
        if snap_key not in f:
            raise KeyError(f"Snapshot '{snap_key}' not found in {filepath}.")
        if param not in f[snap_key]:
            raise KeyError(f"Parameter '{param}' not found in '{snap_key}'.")
        return np.array(f[snap_key][param])


def find_nearest_snapshot(target_z: float, redshifts: List[float]) -> tuple:
    """Return (snap_index, actual_z) for the snapshot closest to target_z."""
    idx = int(np.argmin(np.abs(np.array(redshifts) - target_z)))
    return idx, redshifts[idx]


def load_sage_snapshots(
    cfg: SimConfig,
    target_redshifts: List[float],
) -> Dict[int, np.ndarray]:
    """
    Load BlackHoleMass arrays for only the required snapshots.
    Returns {snap_index: mass_array_in_solar_masses}.
    """
    m_unit = MASS_UNIT_FACTOR / cfg.hubble_h

    snap_indices = {
        find_nearest_snapshot(z, cfg.redshifts)[0]
        for z in target_redshifts
    }

    bh_masses: Dict[int, np.ndarray] = {}
    print(f"Reading SAGE output: {cfg.filepath}")
    print(f"  Snapshots needed: {sorted(snap_indices)}\n")

    for snap in sorted(snap_indices):
        raw = _read_hdf(cfg.filepath, f'Snap_{snap:d}', 'BlackHoleMass')
        bh_masses[snap] = raw * m_unit

    return bh_masses


def compute_bhmf(
    bh_mass_solar: np.ndarray,
    mass_bins: np.ndarray,
    volume: float,
) -> tuple:
    """
    Compute (bin_centres, log10_phi) where phi is in Mpc^-3 dex^-1.
    Returns NaN for empty bins so they are simply not plotted.
    """
    bin_width   = mass_bins[1] - mass_bins[0]
    bin_centers = mass_bins[:-1] + 0.5 * bin_width

    mask = bh_mass_solar > 0.0
    if not np.any(mask):
        return bin_centers, np.full(len(bin_centers), np.nan)

    counts, _ = np.histogram(np.log10(bh_mass_solar[mask]), bins=mass_bins)
    phi       = np.where(counts > 0, counts / (volume * bin_width), np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_phi = np.where(np.isfinite(phi), np.log10(phi), np.nan)
    return bin_centers, log_phi


# ==============================================================================
# Literature data I/O
# ==============================================================================

def load_txt(path: str) -> list:
    """Parse a comment-stripped whitespace file; preserves string tokens."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(COMMENT_CHAR):
                continue
            parts = line.split()
            parsed = []
            for p in parts:
                try:
                    parsed.append(float(p))
                except ValueError:
                    parsed.append(p)
            rows.append(parsed)
    return rows


def rows_to_arrays(rows: list, col_indices: list) -> list:
    """Extract numeric columns; skip rows where any requested column is non-numeric."""
    cols = [[] for _ in col_indices]
    for row in rows:
        try:
            vals = [float(row[i]) for i in col_indices]
        except (IndexError, ValueError, TypeError):
            continue
        for j, v in enumerate(vals):
            cols[j].append(v)
    return [np.array(c) for c in cols]


def _safe_log10(arr: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(arr > 0, np.log10(arr), np.nan)


def extract(ds: dict, rows: list) -> list:
    """
    Slice a loaded dataset into per-redshift chunks according to its 'kind'.

    Returns a list of dicts:
        { z, logM, logphi, logphi_lo (or None), logphi_hi (or None) }

    Supported kinds
    ---------------
    z_logM_logphi            col0=z  col1=logM  col2=log(phi)
    z_logM_phi               col0=z  col1=logM  col2=phi  [linear, no bounds]
    z_logM_phi_upper_lower   col0=z  col1=logM  col2=phi_up  col3=phi_lo  [linear phi]
    z_logM_logphi_lo_hi      col0=z  col1=logM  col2=log(phi)  col3=log(lo)  col4=log(hi)
    logM_logphi_lo_hi        no-z    col0=logM  col1=log(phi)  col2=log(lo)  col3=log(hi)
    logM_phi_upper_lower     no-z    col0=logM  col1=phi_cen   col2=phi_up   col3=phi_lo  [linear]
    z_Mpc_logM_phi           col0=z  col1=Mpc(skip)  col2=logM  col3=phi  [linear phi]
    z_logM_multiphi          col0=z  col1=logM  col2..N=log(phi) variants
    z_seed_logM_logphi       col0=z  col1=seed(str)  col2=logM  col3=log(phi)
    """
    kind    = ds['kind']
    results = []

    def _zvals(row_list):
        return sorted({float(r[0]) for r in row_list
                       if isinstance(r[0], (int, float))})

    if kind == 'z_logM_logphi':
        for z in _zvals(rows):
            sub = [r for r in rows if abs(float(r[0]) - z) < 1e-6]
            logM, logphi = rows_to_arrays(sub, [1, 2])
            if len(logM) == 0: continue
            results.append(dict(z=z, logM=logM, logphi=logphi,
                                logphi_lo=None, logphi_hi=None))

    elif kind == 'z_logM_phi':
        # col0=z  col1=logM  col2=phi (linear, no uncertainty)
        for z in _zvals(rows):
            sub = [r for r in rows if abs(float(r[0]) - z) < 1e-6]
            logM, phi = rows_to_arrays(sub, [1, 2])
            if len(logM) == 0: continue
            results.append(dict(z=z, logM=logM,
                                logphi=_safe_log10(phi),
                                logphi_lo=None, logphi_hi=None))

    elif kind == 'z_logM_phi_upper_lower':
        for z in _zvals(rows):
            sub = [r for r in rows if abs(float(r[0]) - z) < 1e-6]
            logM, phi_up, phi_lo = rows_to_arrays(sub, [1, 2, 3])
            if len(logM) == 0: continue
            results.append(dict(z=z, logM=logM,
                                logphi=_safe_log10(phi_up),
                                logphi_lo=_safe_log10(phi_lo),
                                logphi_hi=_safe_log10(phi_up)))

    elif kind == 'z_logM_logphi_lo_hi':
        for z in _zvals(rows):
            sub = [r for r in rows if abs(float(r[0]) - z) < 1e-6]
            logM, logphi, loglo, loghi = rows_to_arrays(sub, [1, 2, 3, 4])
            if len(logM) == 0: continue
            results.append(dict(z=z, logM=logM, logphi=logphi,
                                logphi_lo=loglo, logphi_hi=loghi))

    elif kind == 'logM_logphi_lo_hi':
        logM, logphi, loglo, loghi = rows_to_arrays(rows, [0, 1, 2, 3])
        results.append(dict(z=ds.get('fixed_z', 0.0), logM=logM,
                            logphi=logphi, logphi_lo=loglo, logphi_hi=loghi))

    elif kind == 'logM_phi_upper_lower':
        logM, phi_cen, phi_up, phi_lo = rows_to_arrays(rows, [0, 1, 2, 3])
        results.append(dict(z=ds.get('fixed_z', 0.0), logM=logM,
                            logphi=_safe_log10(phi_cen),
                            logphi_lo=_safe_log10(phi_lo),
                            logphi_hi=_safe_log10(phi_up)))

    elif kind == 'z_Mpc_logM_phi':
        for z in _zvals(rows):
            sub = [r for r in rows if abs(float(r[0]) - z) < 1e-6]
            logM, phi = rows_to_arrays(sub, [2, 3])
            if len(logM) == 0: continue
            results.append(dict(z=z, logM=logM,
                                logphi=_safe_log10(phi),
                                logphi_lo=None, logphi_hi=None))

    elif kind == 'z_logM_multiphi':
        phi_cols = [v[0] for v in ds.get('variants', [(2, '')])]
        for z in _zvals(rows):
            sub    = [r for r in rows if abs(float(r[0]) - z) < 1e-6]
            arrays = [rows_to_arrays(sub, [1, c]) for c in phi_cols]
            if len(arrays[0][0]) == 0: continue
            logM  = arrays[0][0]
            stack = np.array([a[1] for a in arrays])
            results.append(dict(z=z, logM=logM,
                                logphi=stack[0],
                                logphi_lo=np.nanmin(stack, axis=0),
                                logphi_hi=np.nanmax(stack, axis=0)))

    elif kind == 'logM_logphi_best_16_84':
        # No z column; col0=logM  col1=log(phi_best)  col2=log(phi_16th)  col3=log(phi_84th)
        # 16th = lower bound, 84th = upper bound
        logM, logphi, loglo, loghi = rows_to_arrays(rows, [0, 1, 2, 3])
        results.append(dict(z=ds.get('fixed_z', 0.0), logM=logM,
                            logphi=logphi,
                            logphi_lo=loglo,
                            logphi_hi=loghi))

    elif kind == 'z_seed_logM_logphi':
        sf  = ds.get('seed_filter')
        sub = [r for r in rows
               if sf is None or (len(r) > 1 and str(r[1]).strip() == sf)]
        for z in _zvals(sub):
            s = [r for r in sub if abs(float(r[0]) - z) < 1e-6]
            logM, logphi = rows_to_arrays(s, [2, 3])
            if len(logM) == 0: continue
            results.append(dict(z=z, logM=logM, logphi=logphi,
                                logphi_lo=None, logphi_hi=None))

    return results


def best_z_bin(z: float, z_bins: list, z_half: list) -> int:
    """Return the index of the nearest bin within its tolerance, else -1."""
    best_idx, best_dist = -1, np.inf
    for i, (zc, zh) in enumerate(zip(z_bins, z_half)):
        d = abs(z - zc)
        if d <= zh and d < best_dist:
            best_dist, best_idx = d, i
    return best_idx


# ==============================================================================
# Literature dataset catalogue
# ==============================================================================

PALETTE = {
    'MH08': '#E63946',
    'HY24': '#F4A261',
    'B25':  '#2A9D8F',
    'LM24': '#3A86FF',
    'NT09': '#8338EC',
    'CB10': '#FF006E',
    'S09':  '#FB5607',
    'NV12': '#06D6A0',
    'K10':  '#118AB2',
    'U14':  '#073B4C',
    'TV17': '#9B5DE5',
    'T22L': '#43AA8B',
    'T22H': '#F15BB5',
    'T22':  '#F15BB5',
    'S22':  '#00BBF9',
    'S15':  '#FF6B9D',
    'SC15': '#A8DADC',
    'RN18': '#E9C46A',
    'DS26': '#264653',
    'TNG':  '#E76F51',
    'C16':  '#6A994E',
    'S25':  '#BC6C25',
    'T23':  '#7B2FBE',
}

LINESTYLES = [
    '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))
]

# fmt: off
DATASETS = [

    # ── Observational / empirical ─────────────────────────────────────
    dict(name='Merloni & Heinz 2008', file='MH08.txt',
         kind='z_logM_phi_upper_lower', color=PALETTE['MH08'], category='obs'),
    dict(name='Hernandez-Yevenes 2024', file='HY24.txt',
         kind='z_Mpc_logM_phi', color=PALETTE['HY24'], category='obs'),
    dict(name='Burke et al. 2025', file='B25.txt',
         kind='z_logM_phi_upper_lower', color=PALETTE['B25'], category='obs'),
    dict(name='Liepold & Ma 2024', file='LM24.txt',
         kind='logM_logphi_lo_hi', fixed_z=0.0, color=PALETTE['LM24'], category='obs'),
    dict(name='Natarajan & Treister 2009', file='NT09.txt',
         kind='z_logM_phi_upper_lower', color=PALETTE['NT09'], category='obs'),
    dict(name='Caramete and Biermann 2010', file='CB10.txt',
         kind='logM_phi_upper_lower', fixed_z=0.0, color=PALETTE['CB10'], category='obs'),
    dict(name='Shankar et al. 2009', file='S09.txt',
         kind='z_logM_logphi',
         variants=[(2,''), (3,' (no CT)'), (4,' (dCT)'), (5,' (HRH)'), (6,' (Ext)')],
         color=PALETTE['S09'], category='obs'),
    dict(name='Natarajan & Volonteri 2012', file='NV12.txt',
         kind='z_logM_logphi', color=PALETTE['NV12'], category='obs'),
    dict(name='Kelly & Merloni 2010', file='K10.txt',
         kind='z_logM_logphi', color=PALETTE['K10'], category='obs'),
    dict(name='Ueda et al. 2014', file='U14.txt',
         kind='z_logM_logphi', color=PALETTE['U14'], category='obs'),
    dict(name='Tucci & Volonteri 2017', file='TV17.txt',
         kind='z_logM_logphi', color=PALETTE['TV17'], category='obs'),
    dict(name='Schulze et al. 2015', file='S15.txt',
         kind='z_logM_logphi', color=PALETTE['S15'], category='obs'),


    # ── Semi-analytic / simulation ────────────────────────────────────
    dict(name='Trinca et al. 2022', file='T22.txt',
         kind='z_logM_logphi', color=PALETTE['T22'], category='sim'),
    dict(name='Somerville et al. 2015 (SC SAM)', file='SC15.txt',
         kind='z_logM_phi', color=PALETTE['SC15'], category='sim'),
    dict(name='Ricarte and Natarajan 2018', file='RN18.txt',
         kind='z_logM_phi', color=PALETTE['RN18'], category='sim'),
    dict(name='Sicilia et al. 2022', file='S22.txt',
         kind='z_logM_logphi', color=PALETTE['S22'], category='sim'),
    dict(name='Dark Sage 2026', file='DS26.txt',
         kind='z_logM_logphi', color=PALETTE['DS26'], category='sim'),
    dict(name='IllustrisTNG TNG300', file='TNG300.txt',
         kind='z_logM_phi', color=PALETTE['TNG'], category='sim'),
    dict(name='Croton et al. 2016', file='C16.txt',
         kind='z_logM_logphi', color=PALETTE['C16'], category='sim'),
    dict(name='Bravo et al. 2025 (Shark)', file='S25.txt',
         kind='z_logM_phi', color=PALETTE['S25'], category='sim'),

    # ── TRINITY (keep only these, cleaner than Z23 bulk file) ─────────
    dict(name='Zhang et al. 2023 (TRINITY)', file='T23_z0.1.txt',
         kind='logM_logphi_best_16_84', fixed_z=0.1, color=PALETTE['T23'], category='sim'),
    dict(name='Zhang et al. 2023 (TRINITY)', file='T23_z1.0.txt',
         kind='logM_logphi_best_16_84', fixed_z=1.0, color=PALETTE['T23'], category='sim'),
    dict(name='Zhang et al. 2023 (TRINITY)', file='T23_z2.0.txt',
         kind='logM_logphi_best_16_84', fixed_z=2.0, color=PALETTE['T23'], category='sim'),
    dict(name='Zhang et al. 2023 (TRINITY)', file='T23_z4.0.txt',
         kind='logM_logphi_best_16_84', fixed_z=4.0, color=PALETTE['T23'], category='sim'),
    dict(name='Zhang et al. 2023 (TRINITY)', file='T23_z6.0.txt',
         kind='logM_logphi_best_16_84', fixed_z=6.0, color=PALETTE['T23'], category='sim'),
    dict(name='Zhang et al. 2023 (TRINITY)', file='T23_z8.0.txt',
         kind='logM_logphi_best_16_84', fixed_z=8.0, color=PALETTE['T23'], category='sim'),
]
# fmt: on


# ==============================================================================
# Figure builder
# ==============================================================================

def build_figure(
    sage_masses: Optional[Dict[int, np.ndarray]],
    sim_cfg: SimConfig,
    plot_cfg: PlotConfig,
) -> plt.Figure:

    z_bins = plot_cfg.bhmf_redshifts
    z_half = plot_cfg.z_half
    n_bins = len(z_bins)
    ncols  = 3
    nrows  = (n_bins + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()

    # Hide unused panels
    for ax in axes[n_bins:]:
        ax.set_visible(False)

    # Panel z-labels as in-panel text boxes
    for ax, zc in zip(axes[:n_bins], z_bins):
        ax.text(
            0.97, 0.93,
            rf'$\mathbf{{z \approx {zc}}}$',
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2.5),
        )

    # Colourmap for SAGE
    cmap = plt.get_cmap(plot_cfg.cmap_name)
    norm = mcolors.Normalize(vmin=min(z_bins), vmax=max(z_bins))

    mass_bins = np.arange(
        plot_cfg.mass_bin_min,
        plot_cfg.mass_bin_max + plot_cfg.mass_bin_width,
        plot_cfg.mass_bin_width,
    )

    sage_legend_handles: List[Line2D] = []
    lit_legend_handles:  dict         = {}

    # ── Literature (drawn first so SAGE sits on top) ──────────────────────────
    for i, ds in enumerate(DATASETS):
        path = os.path.join(plot_cfg.lit_data_dir, ds['file'])
        if not os.path.exists(path):
            continue

        rows   = load_txt(path)
        slices = extract(ds, rows)
        color  = ds['color']
        name   = ds['name']
        alpha  = ds.get('alpha_shade', 0.25)

        # Assign linestyle once per dataset to avoid UnboundLocalError
        ls = LINESTYLES[i % len(LINESTYLES)]

        for sl in slices:
            bin_idx = best_z_bin(sl['z'], z_bins, z_half)
            if bin_idx < 0:
                continue

            ax    = axes[bin_idx]
            logM  = sl['logM']
            lphi  = sl['logphi']
            order = np.argsort(logM)
            logM, lphi = logM[order], lphi[order]

            if sl['logphi_lo'] is not None and sl['logphi_hi'] is not None:
                lo = sl['logphi_lo'][order]
                hi = sl['logphi_hi'][order]
                ax.fill_between(logM, lo, hi,
                                color=color, alpha=alpha,
                                linewidth=0, zorder=2)

            ax.plot(
                logM, lphi,
                color=color,
                lw=1.6,
                ls=ls,
                alpha=0.95,
                zorder=3
            )

        if name not in lit_legend_handles:
            has_band = any(sl['logphi_lo'] is not None for sl in slices)
            if has_band:
                patch  = mpatches.Patch(facecolor=color, alpha=0.35,
                                        edgecolor=color)
                handle = (patch, Line2D([0], [0], color=color, lw=1.6, ls=ls))
                lit_legend_handles[name] = handle
            else:
                lit_legend_handles[name] = Line2D([0], [0], color=color, lw=1.6, ls=ls)

    # ── SAGE (drawn on top) ───────────────────────────────────────────────────
    if sage_masses is not None:
        for target_z in z_bins:
            snap_idx, _ = find_nearest_snapshot(target_z, sim_cfg.redshifts)
            if snap_idx not in sage_masses:
                continue

            bin_idx = best_z_bin(target_z, z_bins, z_half)
            if bin_idx < 0:
                continue

            color = cmap(norm(target_z))
            bin_centers, log_phi = compute_bhmf(
                sage_masses[snap_idx], mass_bins, sim_cfg.volume
            )

            valid = np.isfinite(log_phi)
            if np.any(valid):
                axes[bin_idx].plot(
                    bin_centers[valid], log_phi[valid],
                    color=color,
                    lw=3.2,              # thicker
                    ls='-',              # explicitly solid
                    alpha=1.0,           # no transparency
                    zorder=6,            # ensure always on top
                )
                sage_legend_handles.append(
                    Line2D([0], [0], color=color, lw=2.2,
                           label=f'SAGE $z={target_z:.1f}$')
                )

    # ── Axis formatting ───────────────────────────────────────────────────────
    for i, ax in enumerate(axes[:n_bins]):
        row = i // ncols
        col = i % ncols


        ax.set_xlim(*plot_cfg.xlim)
        ax.set_ylim(*plot_cfg.ylim)

        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))

        ax.grid(True, which='major', ls=':', lw=0.5, alpha=0.3)

        # Only show axis labels on outer edges
        if row == nrows - 1:
            ax.set_xlabel(r'$\log_{10}(M_\mathrm{BH}/\mathrm{M}_\odot)$')
        else:
            ax.set_xlabel('')

        if col == 0:
            ax.set_ylabel(
                r'$\log_{10}(\Phi)\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
        else:
            ax.set_ylabel('')



        ax.label_outer()

    # ── Legend ────────────────────────────────────────────────────────────────
    handler_map = {
        h: HandlerTuple(ndivide=None)
        for h in lit_legend_handles.values() if isinstance(h, tuple)
    }

    style_entries: List = []
    if sage_legend_handles:
        style_entries = sage_legend_handles + [
            Line2D([0], [0], color='none', label=''),
        ]

    all_handles = style_entries + list(lit_legend_handles.values())
    all_labels  = (
        [h.get_label() for h in style_entries]
        + list(lit_legend_handles.keys())
    )

    fig.legend(
        all_handles, all_labels,
        handler_map=handler_map,
        loc='lower center',
        ncol=4,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.20,
        top=0.95,
        wspace=0.0,
        hspace=0.0,
    )

    return fig


# ==============================================================================
# Save helper
# ==============================================================================

def save_figure(fig: plt.Figure, base_path: str, cfg: PlotConfig) -> None:
    if not cfg.output_pdf and not cfg.output_png:
        print('  [WARNING] Both output formats are False — figure not saved.')
        return
    if cfg.output_pdf:
        p = base_path + '.pdf'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f'  Saved: {p}')
    if cfg.output_png:
        p = base_path + '.png'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f'  Saved: {p}')


# ==============================================================================
# Entry point
# ==============================================================================

def main() -> None:
    print('=' * 60)
    print('  Black Hole Mass Function plotter')
    print('=' * 60 + '\n')

    apply_publication_style()

    sim_cfg  = SimConfig()

    # --- Create plot configurations for normal and zoomed views ---
    plot_cfg = PlotConfig()
    plot_cfg_zoom = deepcopy(plot_cfg)
    plot_cfg_zoom.xlim = (6.6, 10.0)
    plot_cfg_zoom.ylim = (-5.9, -2.0)

    os.makedirs(plot_cfg.output_dir, exist_ok=True)

    # ── Optionally load SAGE model data ───────────────────────────────────────
    sage_masses: Optional[Dict[int, np.ndarray]] = None
    if os.path.exists(sim_cfg.filepath):
        try:
            sage_masses = load_sage_snapshots(sim_cfg, plot_cfg.bhmf_redshifts)
        except Exception as exc:
            print(f'  [WARNING] Could not load SAGE data: {exc}')
            print('  Continuing with literature data only.\n')
    else:
        print(f'  [INFO] SAGE file not found at:\n    {sim_cfg.filepath}')
        print('  Plotting literature data only.\n')

    # ── Build and save STANDARD plot ──────────────────────────────────────────
    print('\nBuilding standard plot...')
    fig_std = build_figure(sage_masses, sim_cfg, plot_cfg)
    save_figure(
        fig_std,
        os.path.join(plot_cfg.output_dir, 'BlackHoleMassFunction_literature'),
        plot_cfg,
    )
    plt.close(fig_std)

    # ── Build and save ZOOMED plot ────────────────────────────────────────────
    print('Building zoomed plot...')
    fig_zoom = build_figure(sage_masses, sim_cfg, plot_cfg_zoom)
    save_figure(
        fig_zoom,
        os.path.join(plot_cfg_zoom.output_dir, 'BlackHoleMassFunction_literature_zoom'),
        plot_cfg_zoom,
    )
    plt.close(fig_zoom)

    print('\nDone.')


if __name__ == '__main__':
    main()
