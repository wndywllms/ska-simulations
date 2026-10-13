
#!/usr/bin/env python3
"""
Plot and animate direction-dependent phase solutions from an h5parm file.

Usage:
    python plot_phase_solutions.py solutions.h5
    python plot_phase_solutions.py solutions.h5 --solset sol000 --soltab phase000
    python plot_phase_solutions.py solutions.h5 --freq-idx 0 --time-idx 0
    python plot_phase_solutions.py solutions.h5 --movie
"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, FFMpegWriter
import losoto.h5parm as lh5
from astropy.coordinates import SkyCoord
import astropy.units as u


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_phase_solutions(
    h5parm_path: str,
    solset_name: str = "sol000",
    soltab_name: str = "phase000",
) -> dict:
    """
    Load phase solutions from an h5parm using losoto.

    Returns a dict with:
        phase       : (n_ant, n_time, n_freq, n_dir) — as stored
        times       : MJD seconds
        freqs       : Hz
        ant_names   : list of antenna names
        dir_names   : list of direction names
        directions  : (n_dir, 2) array of [ra, dec] in radians
    """
    h5     = lh5.h5parm(h5parm_path, readonly=True)
    solset = h5.getSolset(solset_name)
    soltab = solset.getSoltab(soltab_name)

    # getValues returns array in the axis order stored in the file
    # axes attribute tells us what order that is
    vals, axes = soltab.getValues(retAxesVals=True)
    axes_names = soltab.getAxesNames()
    print(f"Stored axis order: {axes_names}")
    print(f"Stored shape     : {vals.shape}")

    # Reorder to (n_ant, n_time, n_freq, n_dir) regardless of storage order
    target_order = ["ant", "time", "freq", "dir"]
    for ax in target_order:
        if ax not in axes_names:
            raise ValueError(f"Expected axis '{ax}' not found. Got: {axes_names}")

    perm  = [axes_names.index(ax) for ax in target_order]
    phase = np.transpose(vals, perm)

    # Direction coordinates from the source table
    dir_names  = list(axes["dir"])
    src_table  = solset.obj._f_get_child("source")
    directions = np.array([src_table.col("dir")[i] for i in range(len(dir_names))])

    times     = np.array(axes["time"])
    freqs     = np.array(axes["freq"])
    ant_names = list(axes["ant"])

    h5.close()

    print(f"Phase shape (ant, time, freq, dir): {phase.shape}")
    print(f"Antennas  : {ant_names}")
    print(f"Time steps: {len(times)}")
    print(f"Freqs     : {len(freqs)}")
    print(f"Directions: {dir_names}")

    return {
        "phase":      phase,
        "times":      times,
        "freqs":      freqs,
        "ant_names":  ant_names,
        "dir_names":  dir_names,
        "directions": directions,
    }


def load_phase_solutions_fulljones(
    h5parm_path: str,
    solset_name: str = "sol000",
    soltab_name: str = "phase000",
) -> dict:
    """
    Load phase solutions from an h5parm using losoto.

    Returns a dict with:
        phase       : (n_ant, n_time, n_freq, n_dir, n_pol) — as stored
        pols        : list of polarisations
        times       : MJD seconds
        freqs       : Hz
        ant_names   : list of antenna names
        dir_names   : list of direction names
        directions  : (n_dir, 2) array of [ra, dec] in radians
    """
    h5     = lh5.h5parm(h5parm_path, readonly=True)
    solset = h5.getSolset(solset_name)
    soltab = solset.getSoltab(soltab_name)

    # getValues returns array in the axis order stored in the file
    # axes attribute tells us what order that is
    vals, axes = soltab.getValues(retAxesVals=True)
    axes_names = soltab.getAxesNames()
    print(f"Stored axis order: {axes_names}")
    print(f"Stored shape     : {vals.shape}")

    # Reorder to (n_pol, n_ant, n_time, n_freq, n_dir) regardless of storage order
    target_order = ["ant", "time", "freq", "dir", "pol"]
    for ax in target_order:
        if ax not in axes_names:
            raise ValueError(f"Expected axis '{ax}' not found. Got: {axes_names}")

    perm  = [axes_names.index(ax) for ax in target_order]
    phase = np.transpose(vals, perm)

    # Direction coordinates from the source table
    dir_names  = list(axes["dir"])
    src_table  = solset.obj._f_get_child("source")
    directions = np.array([src_table.col("dir")[i] for i in range(len(dir_names))])

    times     = np.array(axes["time"])
    freqs     = np.array(axes["freq"])
    ant_names = list(axes["ant"])
    pol_names = list(axes["pol"])

    h5.close()

    print(f"Phase shape (ant, time, freq, dir, pol): {phase.shape}")
    print(f"Antennas  : {ant_names}")
    print(f"Time steps: {len(times)}")
    print(f"Freqs     : {len(freqs)}")
    print(f"Directions: {dir_names}")
    print(f"Polarisations: {pol_names}")

    return {
        "phase":      phase,
        "times":      times,
        "freqs":      freqs,
        "ant_names":  ant_names,
        "dir_names":  dir_names,
        "directions": directions,
        "pol_names":  pol_names,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def direction_axes(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert (ra, dec) in radians to offsets in arcminutes from field centre.
    Returns (dra_arcmin, ddec_arcmin).
    """
    ra  = np.degrees(directions[:, 0])
    dec = np.degrees(directions[:, 1])

    ra_centre  = np.mean(ra)
    dec_centre = np.mean(dec)

    # RA offset corrected for declination
    dra  = (ra  - ra_centre)  * np.cos(np.radians(dec_centre)) * 60   # arcmin
    ddec = (dec - dec_centre) * 60

    return dra, ddec


def make_phase_scatter(
    ax: plt.Axes,
    dra: np.ndarray,
    ddec: np.ndarray,
    phase_slice: np.ndarray,
    title: str,
    vmin: float = -np.pi,
    vmax: float = np.pi,
    cmap: str = "RdBu_r",
    triangulate: bool = True,
) -> plt.cm.ScalarMappable:
    """
    Plot a 2D phase map for one antenna/time/freq slice.

    Uses triangulated interpolation if enough directions exist (>3),
    otherwise falls back to a scatter plot.

    Returns the mappable for a colorbar.
    """
    ax.set_facecolor("0.15")

    if triangulate and len(dra) > 3:
        triang = mtri.Triangulation(dra, ddec)
        cf = ax.tripcolor(
            triang, phase_slice,
            cmap=cmap, vmin=vmin, vmax=vmax,
            shading="gouraud",
        )
        # Overlay direction positions
        ax.scatter(dra, ddec, c="white", s=15, zorder=5, linewidths=0)
    else:
        cf = ax.scatter(
            dra, ddec, c=phase_slice,
            cmap=cmap, vmin=vmin, vmax=vmax,
            s=80, zorder=5,
        )

    ax.set_xlabel("ΔRA (arcmin)")
    ax.set_ylabel("ΔDec (arcmin)")
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")

    return cf


# ---------------------------------------------------------------------------
# Single-time snapshot plots for all antennas
# ---------------------------------------------------------------------------

def plot_all_antennas(
    data1: dict,
    data2: dict,
    freq_idx: int = 0,
    output_dir: str = "phase_plots",
    cmap: str = "RdBu_r",
) -> None:
    """
    For a given time and frequency index, make one plot per antenna
    showing phase as a function of direction.

    Parameters
    ----------
    data       : dict returned by load_phase_solutions
    time_idx   : time index to plot
    freq_idx   : frequency index to plot
    output_dir : directory to save plots
    cmap       : matplotlib colormap
    """


def nearest_index(src, tgt):
    """For each value in tgt, return the index of the nearest value in src.
    src does not need to be sorted."""
    src = np.asarray(src, dtype=float)
    tgt = np.asarray(tgt, dtype=float)
    if src.size == 1:
        return np.zeros(tgt.shape, dtype=int)
    order = np.argsort(src)
    s = src[order]
    i = np.searchsorted(s, tgt)
    i = np.clip(i, 1, len(s) - 1)
    # step back one if the left neighbour is closer (ties go left)
    i -= (tgt - s[i - 1]) <= (s[i] - tgt)
    return order[i]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":


    #data_tec_fr = load_phase_solutions_fulljones("oskar_tec_phase_line1_nofr1.h5", "sol000", "phase000")
    #data_tec = load_phase_solutions("oskar_tec_phases_line1.h5", "sol000", "phase000")
    #data_rapthor = load_phase_solutions("field-solutions-line1.h5", "sol000", "phase000")
    data_tec = load_phase_solutions("oskar_tec_phases_scan0.h5", "sol000", "phase000")
    data_rapthor = load_phase_solutions("field-solutions.h5", "sol000", "phase000")
    
    data1 = data_tec
    data2 = data_rapthor

    
    phase1     = data1["phase"]        # (n_ant, n_time, n_freq, n_dir, n_pol)
    #phase1 = phase1[:,:,:,:,0]
    ant_names1 = data1["ant_names"]
    times1     = data1["times"]
    freqs1     = data1["freqs"]
    dra1, ddec1 = direction_axes(data1["directions"])
    phase2_r     = data2["phase"]        # (n_ant, n_time, n_freq, n_dir)
    ant_names2 = data2["ant_names"]
    times2     = data2["times"]
    freqs2     = data2["freqs"]
    dra2, ddec2 = direction_axes(data2["directions"])


    # Index maps from the phase1 grid onto the phase2 grid
    it = nearest_index(times2, times1)   # shape (n_time1,)
    jf = nearest_index(freqs2, freqs1)   # shape (n_freq1,)

    # Resample: (n_ant, n_time2, n_freq2, n_dir) -> (n_ant, n_time1, n_freq1, n_dir)
    phase2 = phase2_r[:, it[:, None], jf[None, :], :]


    dir_idx = 1
    
    ant_names = [aa.split("_")[1].split(")")[0] for aa in ant_names1]

    n_ant1 = len(ant_names1)
    #t_mjd1 = times1[time_idx]
    n_ant2 = len(ant_names2)
    #t_mjd2 = times2[time_idx]

    vmin, vmax = -np.pi, np.pi



    freq_idx = 5
    f_mhz1 = freqs1[freq_idx] / 1e6
    f_mhz2 = freqs2[freq_idx] / 1e6
    
        
    fig, axs = plt.subplots(figsize=(12, 10), nrows=8, ncols=9, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0, wspace=0)
    axs = axs.flatten()
    #time_idx2 = times2 < np.max(times1)
    for ant_idx, ant_name in enumerate(ant_names1):
        for dir_idxi in range(len(dra1)):
            phase_slice1 = phase1[ant_idx, :, :, dir_idxi]   # (n_time,)
            phase_slice2 = phase2[ant_idx, :, :, dir_idxi]   # (n_time,)
            ax = axs[ant_idx]
            
            X = (phase_slice1 - phase_slice2)
            #X = X/X.mean(axis=0,keepdims=True)
            im = ax.imshow(X.T, origin="lower", vmin=-0.25, vmax=0.25, cmap="twilight", extent=[times1[0],times1[-1],freqs1[0],freqs1[-1]], aspect="auto")
            ax.set_title(ant_names[ant_idx], y=0.75)
    fig.colorbar(im, ax=axs, label="Δphase [rad]", shrink=0.9)

    plt.suptitle("TEC screen - rapthor phase")
    
    
    fig, axs = plt.subplots(figsize=(12, 10), nrows=8, ncols=9, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0, wspace=0)
    axs = axs.flatten()
    #time_idx2 = times2 < np.max(times1)
    for ant_idx, ant_name in enumerate(ant_names1):
        for dir_idxi in range(len(dra1)):
            phase_slice1 = phase1[ant_idx, :, freq_idx, dir_idxi]   # (n_time,)
            ax = axs[ant_idx]
            
            ax.plot(times1 -times1[0], phase_slice1)
            ax.set_title(ant_names[ant_idx], y=0.75)
    ax.set_ylim(vmin,vmax)
    plt.suptitle("TEC screen phase")
    
    fig, axs = plt.subplots(figsize=(12, 10), nrows=8, ncols=9, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0, wspace=0)
    axs = axs.flatten()
    #time_idx2 = times2 < np.max(times1)
    #time_idx2 =
    for ant_idx, ant_name in enumerate(ant_names1):
        for dir_idxi in range(len(dra1)):
            phase_slice2 = phase2[ant_idx, :, freq_idx, dir_idxi]   # (n_time,)
            ax = axs[ant_idx]
            
            #ax.plot((times2 -times1[0])[time_idx2], phase_slice2)
            ax.plot(times1 -times1[0], phase_slice2)
            ax.set_title(ant_names[ant_idx], y=0.75)
    ax.set_ylim(vmin,vmax)
    plt.suptitle("RAPTHOR phase")
            


    freq_idx = 0
    f_mhz1 = freqs1[freq_idx] / 1e6
    f_mhz2 = freqs2[freq_idx] / 1e6
    fig, axs = plt.subplots(figsize=(14, 9), nrows=8, ncols=9, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0, wspace=0)
    axs = axs.flatten()
    
    #time_idx2 = times2 < np.max(times1)
    for ant_idx, ant_name in enumerate(ant_names1):
        phase_slice1 = phase1[ant_idx, :, freq_idx, dir_idx]   # (n_time,)
        phase_slice2 = phase2[ant_idx, :, freq_idx, dir_idx]   # (n_time,)
        ax = axs[ant_idx]
        
        ax.plot(times1 -times1[0], phase_slice1, c="C0", label="TEC screen")
        ax.plot(times1 -times1[0], phase_slice2, c="C1", label="rapthor")
        ax.set_title(ant_names[ant_idx], y=0.75)
                    
    ax.set_ylim(vmin,vmax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=len(labels))
    plt.suptitle(f"Phase v Time freq {f_mhz1}")
    fig.supxlabel("Time")
    fig.supylabel("Phase")
        

    time_idx = 10
    #print(f"time at time_idx2 of {time_idx2} is {times2[time_idx2]}")
    #time_idxs1 = np.where((times1 > times2[time_idx2]) & (times1 < times2[time_idx2+2]))[0]
    #print (time_idxs1)
    fig, axs = plt.subplots(figsize=(14, 9), nrows=8, ncols=9, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0, wspace=0)
    axs = axs.flatten()

    for ant_idx, ant_name in enumerate(ant_names1):
        ax = axs[ant_idx]
        
        phase_slice1 = phase1[ant_idx, time_idx, :, dir_idx]   # (n_time,)
        ax.plot(freqs1, phase_slice1, c="C0", label="TEC screen")
        
        phase_slice2 = phase2[ant_idx, time_idx, :, dir_idx]   # (n_time,)
        ax.plot(freqs1, phase_slice2, c="C1", label="rapthor")
        
        ax.set_title(ant_names[ant_idx], y=0.75)
        
        slope1, intercept = np.polyfit(1./freqs1, phase_slice1, 1)
        slope2, intercept = np.polyfit(1./freqs1, phase_slice2, 1)
        print(f"{ant_name} screen:{slope1} rapthor:{slope2} ")
    ax.set_ylim(vmin,vmax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=len(labels))
    plt.suptitle(f"Phase v Freq time {times2[time_idx]}")
    fig.supxlabel("Freq")
    fig.supylabel("Phase")

    '''
        cf = make_phase_scatter(
            ax, dra, ddec, phase_slice,
            title=f"{ant_name}  |  t_idx={time_idx}  f={f_mhz:.1f} MHz",
            vmin=vmin, vmax=vmax, cmap=cmap,
        )

        cbar = fig.colorbar(cf, ax=ax, pad=0.02)
        cbar.set_label("Phase (rad)")

        fig.tight_layout()
        outpath = os.path.join(output_dir, f"phase_{ant_name}_t{time_idx:04d}_f{freq_idx:04d}.png")
        fig.savefig(outpath, dpi=120)
        plt.close(fig)
        print(f"  Saved {outpath}")
    '''
