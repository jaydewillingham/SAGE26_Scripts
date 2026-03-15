#!/usr/bin/env python
"""
Black Hole Mass Function (BHMF) plotter for SAGE semi-analytic model output.

Compares model predictions against observational data across multiple redshifts.
Produces a publication-ready figure with a redshift colorbar and clean formatting.
Optionally produces diagnostic M_BH vs M_* plots showing the effect of data cuts.
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Final, List, Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings("ignore")


# ==============================================================================
# Constants
# ==============================================================================

MASS_UNIT_FACTOR: Final[float] = 1.0e10   # Internal mass unit in solar masses (pre-h correction)
COMMENT_CHAR:     Final[str]   = '#'
OBS_DATA_NCOLS:   Final[int]   = 4        # Expected minimum columns in observational data files
BH_MSTAR_RATIO_CUT: Final[float] = 0.01  # Minimum M_BH / M_* ratio accepted in BHMF

# All 64 Millennium redshifts; index 0 = snap 0 (z=127), index 63 = snap 63 (z=0).
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

@dataclass
class SimConfig:
    """Simulation and file path configuration."""
    dir_name:        str   = /<absolute>/<root>/<path>/sage-model/output/millennium_full/
    file_name:       str   = 'model_0.hdf5'
    hubble_h:        float = 0.73
    box_size:        float = 500.0    # h^-1 Mpc
    volume_fraction: float = 0.3
    first_snap:      int   = 0
    last_snap:       int   = 63
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
    """Plotting and output configuration."""
    # --- Output format flags ---
    output_pdf:           bool  = False
    output_png:           bool  = True

    # --- Diagnostic plot flag ---
    plot_mbh_mstar:       bool  = False   # Set False to skip M_BH vs M_* scatter plots

    # --- File paths ---
    data_dir:             str   = '../data/'

    # --- Redshifts to plot ---
    bhmf_redshifts:       List[float] = field(
        default_factory=lambda: [0.1, 1.0, 2.0, 4.0, 6.0, 8.0]
    )

    # --- Observational data files keyed by redshift ---
    obs_files:            Dict[float, str] = field(default_factory=lambda: {
        0.1: 'fig4_bhmf_z0.1.txt',
        1.0: 'fig4_bhmf_z1.0.txt',
        2.0: 'fig4_bhmf_z2.0.txt',
        4.0: 'fig4_bhmf_z4.0.txt',
        6.0: 'fig4_bhmf_z6.0.txt',
        8.0: 'fig4_bhmf_z8.0.txt',
    })

    # --- BHMF histogram parameters ---
    mass_bin_min:         float = 5.0
    mass_bin_max:         float = 11.5
    mass_bin_width:       float = 0.25   # 0.5 for even smoother lines

    # --- BHMF axis limits ---
    xlim:                 tuple = (5.5, 10.0)
    ylim:                 tuple = (1e-5, 1e-2)

    # --- M_BH vs M_* axis limits ---
    mstar_lim:            tuple = (7.0, 12.0)
    mbh_lim:              tuple = (4.0, 10.5)

    # --- Colormap ---
    cmap_name:            str   = 'plasma'


# ==============================================================================
# I/O utilities
# ==============================================================================

def read_hdf(filepath: str, snap_key: str, param: str) -> np.ndarray:
    """
    Read a single parameter array from an HDF5 snapshot group.

    Parameters
    ----------
    filepath : str
        Full path to the HDF5 file.
    snap_key : str
        Snapshot group key (e.g. 'Snap_063').
    param : str
        Dataset name within the snapshot group.

    Returns
    -------
    np.ndarray
        Copy of the dataset as a NumPy array.

    Raises
    ------
    KeyError
        If the snapshot group or parameter is absent from the file.
    """
    with h5py.File(filepath, 'r') as f:
        if snap_key not in f:
            raise KeyError(f"Snapshot '{snap_key}' not found in {filepath}.")
        if param not in f[snap_key]:
            raise KeyError(f"Parameter '{param}' not found in '{snap_key}'.")
        return np.array(f[snap_key][param])


def make_snap_key(snap: int) -> str:
    """Return the HDF5 group key for a given snapshot index, e.g. 'Snap_063'."""
    return f'Snap_{snap:03d}'


def load_obs_data(filepath: str) -> Optional[np.ndarray]:
    """
    Load observational BHMF data from a whitespace-delimited text file.

    Expected columns: log10(M_BH), phi_best, phi_16th, phi_84th.
    Lines beginning with '#' are treated as comments.

    Parameters
    ----------
    filepath : str
        Path to the observational data file.

    Returns
    -------
    np.ndarray or None
        2-D array of shape (N, >=4), or None if the file cannot be loaded.
    """
    try:
        data = np.loadtxt(filepath, comments=COMMENT_CHAR)
        if data.ndim != 2 or data.shape[1] < OBS_DATA_NCOLS:
            raise ValueError(
                f"Expected >= {OBS_DATA_NCOLS} columns, got shape {data.shape}."
            )
        return data
    except Exception as exc:
        print(f"  [WARNING] Could not load '{filepath}': {exc}")
        return None


# ==============================================================================
# Data loading  (lazy — only snapshots needed for the requested redshifts)
# ==============================================================================

def load_snapshots_for_redshifts(
    cfg: SimConfig,
    target_redshifts: List[float],
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load StellarMass and BlackHoleMass for only the snapshots required.

    Performs a single HDF5 open per snapshot and reads only the two arrays
    needed, rather than loading all datasets across all 64 snapshots.

    Parameters
    ----------
    cfg : SimConfig
        Simulation configuration.
    target_redshifts : list of float
        The redshifts at which plots will be produced.

    Returns
    -------
    dict mapping snap_index -> {'BlackHoleMass': ndarray, 'StellarMass': ndarray}
        Masses are in solar masses (linear, not log).
    """
    h      = cfg.hubble_h
    m_unit = MASS_UNIT_FACTOR / h

    snap_indices = {
        find_nearest_snapshot(z, cfg.redshifts)[0]
        for z in target_redshifts
    }

    snap_data: Dict[int, Dict[str, np.ndarray]] = {}
    print(f"Reading galaxy properties from: {cfg.filepath}")
    print(f"  Snapshots required: {sorted(snap_indices)}\n")

    for snap in sorted(snap_indices):
        key = make_snap_key(snap)
        snap_data[snap] = {
            'BlackHoleMass': read_hdf(cfg.filepath, key, 'BlackHoleMass') * m_unit,
            'StellarMass':   read_hdf(cfg.filepath, key, 'StellarMass')   * m_unit,
        }

    return snap_data


# ==============================================================================
# BHMF computation
# ==============================================================================

def compute_bhmf(
    bh_mass_solar: np.ndarray,
    stellar_mass_solar: np.ndarray,
    mass_bins: np.ndarray,
    volume: float,
) -> tuple:
    """
    Compute the black hole mass function with a BH-to-stellar mass ratio cut.

    Only galaxies with M_BH / M_* > BH_MSTAR_RATIO_CUT are included, which
    removes unphysical seed black holes that have not grown appreciably.

    Parameters
    ----------
    bh_mass_solar : np.ndarray
        BH masses in solar masses (linear).
    stellar_mass_solar : np.ndarray
        Stellar masses in solar masses (linear).
    mass_bins : np.ndarray
        Bin edges in log10(M_BH / M_sun).
    volume : float
        Comoving volume in Mpc^3.

    Returns
    -------
    bin_centers : np.ndarray
    phi : np.ndarray
        Number density in Mpc^-3 dex^-1; zero where no galaxies exist.
    """
    bin_width   = mass_bins[1] - mass_bins[0]
    bin_centers = mass_bins[:-1] + 0.5 * bin_width

    # Safe division: ratio is zero wherever stellar mass is zero.
    ratio = np.zeros_like(bh_mass_solar, dtype=float)
    np.divide(bh_mass_solar, stellar_mass_solar,
              out=ratio, where=stellar_mass_solar > 0.0)

    mask = (bh_mass_solar > 0.0) & (ratio > BH_MSTAR_RATIO_CUT)
    if not np.any(mask):
        return bin_centers, np.zeros(len(bin_centers))

    counts, _ = np.histogram(np.log10(bh_mass_solar[mask]), bins=mass_bins)
    phi = np.where(counts > 0, counts / (volume * bin_width), 0.0)
    return bin_centers, phi


def find_nearest_snapshot(target_z: float, redshifts: List[float]) -> tuple:
    """
    Return (snap_index, actual_redshift) for the snapshot closest to target_z.
    """
    idx = int(np.argmin(np.abs(np.array(redshifts) - target_z)))
    return idx, redshifts[idx]


# ==============================================================================
# Shared plotting helpers
# ==============================================================================

def apply_publication_style() -> None:
    """Apply rcParams for a clean, journal-ready figure style."""
    plt.rcParams.update({
        'figure.facecolor':   'white',
        'figure.dpi':         150,
        'axes.facecolor':     'white',
        'axes.edgecolor':     'black',
        'axes.linewidth':     1.0,
        'axes.labelcolor':    'black',
        'axes.labelsize':     13,
        'xtick.color':        'black',
        'ytick.color':        'black',
        'xtick.direction':    'in',
        'ytick.direction':    'in',
        'xtick.top':          True,
        'ytick.right':        True,
        'xtick.major.size':   6,
        'ytick.major.size':   6,
        'xtick.minor.size':   3,
        'ytick.minor.size':   3,
        'xtick.major.width':  1.0,
        'ytick.major.width':  1.0,
        'xtick.labelsize':    12,
        'ytick.labelsize':    12,
        'font.family':        'serif',
        'font.size':          13,
        'mathtext.fontset':   'stix',
        'legend.facecolor':   'white',
        'legend.edgecolor':   '0.7',
        'legend.fontsize':    11,
        'legend.framealpha':  0.9,
        'lines.linewidth':    1.8,
        'text.color':         'black',
    })


def save_figure(fig: plt.Figure, base_path: str, cfg: PlotConfig) -> None:
    """
    Save a figure in the format(s) specified by PlotConfig.

    At least one of ``cfg.output_pdf`` or ``cfg.output_png`` must be True.
    If both are True the figure is written in both formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    base_path : str
        Full path *without* extension (e.g. '/results/bhplots/BlackHoleMassFunction').
    cfg : PlotConfig
    """
    if not cfg.output_pdf and not cfg.output_png:
        print("  [WARNING] Both output_pdf and output_png are False — figure not saved.")
        return

    formats: List[tuple] = []
    if cfg.output_pdf:
        formats.append(('.pdf', {'dpi': 300, 'bbox_inches': 'tight'}))
    if cfg.output_png:
        formats.append(('.png', {'dpi': 300, 'bbox_inches': 'tight'}))

    for ext, kwargs in formats:
        out_path = base_path + ext
        fig.savefig(out_path, **kwargs)
        print(f"  Saved: {out_path}")


# ==============================================================================
# Main BHMF plot
# ==============================================================================

def plot_bhmf(
    snap_data: Dict[int, Dict[str, np.ndarray]],
    sim_cfg: SimConfig,
    plot_cfg: PlotConfig,
    output_dir: str,
) -> None:
    """
    Plot the Black Hole Mass Function at multiple redshifts.

    Model predictions are shown as solid lines; observational constraints
    as dashed lines with shaded 16th–84th percentile uncertainty bands.
    A single legend explains both line style and redshift colour-coding.

    Parameters
    ----------
    snap_data : dict
        Mapping of snap_index -> {'BlackHoleMass': ndarray, 'StellarMass': ndarray}.
    sim_cfg : SimConfig
    plot_cfg : PlotConfig
    output_dir : str
    """
    print("Plotting black hole mass function...")

    apply_publication_style()

    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    cmap  = plt.get_cmap(plot_cfg.cmap_name)
    z_arr = np.array(plot_cfg.bhmf_redshifts)
    norm  = mcolors.Normalize(vmin=z_arr.min(), vmax=z_arr.max())

    mass_bins = np.arange(
        plot_cfg.mass_bin_min,
        plot_cfg.mass_bin_max + plot_cfg.mass_bin_width,
        plot_cfg.mass_bin_width,
    )

    redshift_handles: List[Line2D] = []

    for target_z in plot_cfg.bhmf_redshifts:
        snap_idx, _ = find_nearest_snapshot(target_z, sim_cfg.redshifts)
        color        = cmap(norm(target_z))
        arrays       = snap_data[snap_idx]

        # --- Model BHMF ---
        bin_centers, phi = compute_bhmf(
            arrays['BlackHoleMass'],
            arrays['StellarMass'],
            mass_bins,
            sim_cfg.volume,
        )
        valid = phi > 0
        if np.any(valid):
            ax.plot(bin_centers[valid], phi[valid],
                    color=color, lw=2.0, ls='-', zorder=3)
            redshift_handles.append(
                Line2D([0], [0], color=color, lw=2.0, ls='-',
                       label=f'z = {target_z:.1f}')
            )

        # --- Observational data ---
        obs_filename = plot_cfg.obs_files.get(target_z)
        if obs_filename:
            obs_path = os.path.join(plot_cfg.data_dir, obs_filename)
            obs_data = load_obs_data(obs_path)
            if obs_data is not None:
                obs_mass, obs_phi, obs_phi_16, obs_phi_84 = obs_data.T
                ax.plot(obs_mass, obs_phi,
                        color=color, lw=1.5, ls='--', alpha=0.85, zorder=2)
                ax.fill_between(obs_mass, obs_phi_16, obs_phi_84,
                                color=color, alpha=0.15, zorder=1)

    # --- Legend ---
    style_handles = [
        Line2D([0], [0], color='0.3', lw=2.0, ls='-',
               label=r'SAGE (M$_\mathrm{BH}$/M$_* > 0.01$)'),
        Line2D([0], [0], color='0.3', lw=1.5, ls='--',
               label=r'Zhang+23'),
    ]
    separator = Line2D([0], [0], color='none', label='')
    ax.legend(
        handles=style_handles + [separator] + redshift_handles,
        loc='upper right', fontsize=10, framealpha=0.9, edgecolor='0.7',
    )

    # --- Axes formatting ---
    ax.set_yscale('log')
    ax.set_xlim(*plot_cfg.xlim)
    ax.set_ylim(*plot_cfg.ylim)
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{BH}\ /\ \mathrm{M}_\odot)$', fontsize=13)
    ax.set_ylabel(r'$\Phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$', fontsize=13)
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(True, which='major', ls=':', lw=0.5, alpha=0.45, color='grey', zorder=0)

    fig.tight_layout()

    base_path = os.path.join(output_dir, 'BlackHoleMassFunctionCUTS')
    save_figure(fig, base_path, plot_cfg)
    plt.close(fig)
    print("")


# ==============================================================================
# Diagnostic M_BH vs M_* plots
# ==============================================================================

def plot_mbh_mstar_relation(
    snap_data: Dict[int, Dict[str, np.ndarray]],
    sim_cfg: SimConfig,
    plot_cfg: PlotConfig,
    output_dir: str,
) -> None:
    """
    Plot M_BH vs M_* to visualise the effect of the mass ratio cut.

    A separate scatter plot is generated for each redshift in
    ``plot_cfg.bhmf_redshifts``. Galaxies rejected by the BH_MSTAR_RATIO_CUT
    are shown in grey; accepted galaxies in red. The cut line is overlaid.

    This function is only called when ``plot_cfg.plot_mbh_mstar`` is True.

    Parameters
    ----------
    snap_data : dict
        Mapping of snap_index -> {'BlackHoleMass': ndarray, 'StellarMass': ndarray}.
    sim_cfg : SimConfig
    plot_cfg : PlotConfig
    output_dir : str
    """
    print("Plotting M_BH vs M_* relation (diagnostic cuts)...")

    apply_publication_style()

    for target_z in plot_cfg.bhmf_redshifts:
        snap_idx, actual_z = find_nearest_snapshot(target_z, sim_cfg.redshifts)
        arrays = snap_data[snap_idx]

        mbh_solar   = arrays['BlackHoleMass']
        mstar_solar = arrays['StellarMass']

        # Only plot galaxies with both a BH and stellar mass.
        plot_mask = (mbh_solar > 0) & (mstar_solar > 0)
        if not np.any(plot_mask):
            print(f"  Skipping z = {target_z:.1f} — no valid galaxies.")
            continue

        mbh_plot   = mbh_solar[plot_mask]
        mstar_plot = mstar_solar[plot_mask]

        ratio         = mbh_plot / mstar_plot
        accepted_mask = ratio > BH_MSTAR_RATIO_CUT
        rejected_mask = ~accepted_mask

        fig, ax = plt.subplots(figsize=(6.0, 6.0))

        # Rejected galaxies (background layer).
        ax.scatter(
            np.log10(mstar_plot[rejected_mask]),
            np.log10(mbh_plot[rejected_mask]),
            s=5, c='0.7', alpha=0.3, zorder=1,
            label=rf'Rejected (M$_\mathrm{{BH}}$/M$_* \leq {BH_MSTAR_RATIO_CUT}$)',
        )

        # Accepted galaxies (foreground layer).
        ax.scatter(
            np.log10(mstar_plot[accepted_mask]),
            np.log10(mbh_plot[accepted_mask]),
            s=15, c='crimson', alpha=0.7, zorder=2,
            label=rf'Accepted (M$_\mathrm{{BH}}$/M$_* > {BH_MSTAR_RATIO_CUT}$)',
        )

        # Ratio cut line.
        mstar_line = np.logspace(*plot_cfg.mstar_lim, 100)
        ax.plot(
            np.log10(mstar_line),
            np.log10(BH_MSTAR_RATIO_CUT * mstar_line),
            'k--', lw=1.5, zorder=3,
            label=rf'M$_\mathrm{{BH}} = {BH_MSTAR_RATIO_CUT} \times$ M$_*$',
        )

        ax.set_title(f'SAGE data at z = {actual_z:.2f}')
        ax.set_xlabel(r'$\log_{10}(M_*\ /\ \mathrm{M}_\odot)$')
        ax.set_ylabel(r'$\log_{10}(M_\mathrm{BH}\ /\ \mathrm{M}_\odot)$')
        ax.set_xlim(*plot_cfg.mstar_lim)
        ax.set_ylim(*plot_cfg.mbh_lim)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, which='major', ls=':', lw=0.5, alpha=0.45, color='grey', zorder=0)

        fig.tight_layout()

        base_path = os.path.join(output_dir, f'MBH_vs_Mstar_z{target_z:.1f}')
        save_figure(fig, base_path, plot_cfg)
        plt.close(fig)

    print("")


# ==============================================================================
# Entry point
# ==============================================================================

def main() -> None:
    print("=" * 60)
    print(" Black Hole Mass Function — SAGE model")
    print("=" * 60 + "\n")

    sim_cfg  = SimConfig()
    plot_cfg = PlotConfig()

    # Validate that at least one output format is requested.
    if not plot_cfg.output_pdf and not plot_cfg.output_png:
        print("[WARNING] Both output_pdf and output_png are False.\n"
              "         Set at least one to True in PlotConfig before running.")

    output_dir = os.path.join(sim_cfg.dir_name, 'bhplots')
    os.makedirs(output_dir, exist_ok=True)

    # Load only the snapshots required for the requested redshifts.
    try:
        snap_data = load_snapshots_for_redshifts(sim_cfg, plot_cfg.bhmf_redshifts)
    except (FileNotFoundError, KeyError) as exc:
        print(f"\n[ERROR] Failed to load simulation data: {exc}")
        print("Please verify 'SimConfig.dir_name' and 'SimConfig.file_name'.")
        return

    # Main BHMF figure.
    plot_bhmf(snap_data, sim_cfg, plot_cfg, output_dir)

    # Diagnostic M_BH vs M_* scatter plots (skipped when flag is False).
    if plot_cfg.plot_mbh_mstar:
        plot_mbh_mstar_relation(snap_data, sim_cfg, plot_cfg, output_dir)
    else:
        print("Skipping M_BH vs M_* plots (plot_mbh_mstar = False).\n")

    print("Done.")


if __name__ == '__main__':
    main()