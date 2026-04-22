"""
Phase-space animation for 3D forward dynamics.

Four subplots in one high-resolution figure, animated over time:
  (x, vx)  |  (y, vy)
  (z, vz)  |  (r, vr)   ← spherical radial coordinate

Harmonic-approximation trapping ellipses are drawn for the static tweezers
at start and stop (once the aux is gone, the ellipse relevant to the final
trap is shown).

Atoms colour-coded: survived = green, lost = grey.

Output: a high-DPI GIF so the user can zoom in.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from typing import Dict

from utils3d import (
    COLOR_ALIVE, COLOR_LOST,
    cartesian_to_spherical_phasespace,
    trap_ellipse_xy, trap_ellipse_z,
)


def _draw_ellipses(ax, subplot, params, scales):
    """
    Draw 1-σ and 2-σ harmonic trapping ellipses appropriate for each subplot.
    """
    U0   = params["U0_static"]
    w    = params["w"]
    zR   = params["zR"]
    T_a  = scales["T_atom_dimless"]

    if subplot in ("x", "y"):
        xe1, ve1 = trap_ellipse_xy(U0, w, T_a)
        xe2, ve2 = trap_ellipse_xy(U0, w, 2 * T_a)
        return xe1, ve1, xe2, ve2
    elif subplot == "z":
        xe1, ve1 = trap_ellipse_z(U0, zR, T_a)
        xe2, ve2 = trap_ellipse_z(U0, zR, 2 * T_a)
        return xe1, ve1, xe2, ve2
    else:   # spherical — approximate as xy
        xe1, ve1 = trap_ellipse_xy(U0, w, T_a)
        xe2, ve2 = trap_ellipse_xy(U0, w, 2 * T_a)
        return xe1, ve1, xe2, ve2


def animate_phase_space_3d(data: Dict, output_dir: Path,
                            fps: int = 15, dpi: int = 180):
    """
    Parameters
    ----------
    dpi : 180 is high enough to zoom; keep fps low to keep file size manageable.
    """
    t       = data["t"]
    x       = data["x"];   vx = data["vx"]
    y       = data["y"];   vy = data["vy"]
    z_arr   = data["z"];   vz = data["vz"]
    is_lost = data["is_lost"]
    params  = data["params"]
    scales  = data["scales"]

    r_arr, vr_arr = cartesian_to_spherical_phasespace(x, y, z_arr, vx, vy, vz)

    n_steps, n_shots = x.shape

    skip   = max(1, n_steps // 150)
    frames = list(range(0, n_steps, skip))

    # pre-compute ellipses
    def _ell(subplot):
        return _draw_ellipses(None, subplot, params, scales)

    ell_x  = _ell("x")
    ell_y  = _ell("y")
    ell_z  = _ell("z")
    ell_r  = _ell("r")

    # global limits
    def lim(arr, pad=0.15):
        lo, hi = arr.min(), arr.max()
        d = (hi - lo) * pad or 0.5
        return lo - d, hi + d

    xlim  = lim(x);   vxlim = lim(vx)
    ylim  = lim(y);   vylim = lim(vy)
    zlim  = lim(z_arr); vzlim = lim(vz)
    rlim  = (0, lim(r_arr)[1]); vrlim = lim(vr_arr)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(hspace=0.35, wspace=0.30)

    ax_x, ax_y, ax_z, ax_r = axes.flat

    configs = [
        (ax_x, x,     vx,    xlim,  vxlim, "x  [μm]",  "vx [m/s]", ell_x),
        (ax_y, y,     vy,    ylim,  vylim, "y  [μm]",  "vy [m/s]", ell_y),
        (ax_z, z_arr, vz,    zlim,  vzlim, "z  [μm]",  "vz [m/s]", ell_z),
        (ax_r, r_arr, vr_arr, rlim, vrlim, "r  [μm]", "vr [m/s]", ell_r),
    ]

    scatters_alive = []
    scatters_lost  = []
    time_texts     = []

    for ax, pos_arr, vel_arr, plim, vlim, xlabel, ylabel, ell in configs:
        ax.set_xlim(*plim)
        ax.set_ylim(*vlim)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.grid(alpha=0.25, ls=":")

        # ellipses (drawn at origin; they represent the local trap geometry)
        xe1, ve1, xe2, ve2 = ell
        ax.plot(xe1, ve1, color="steelblue", lw=1.5, ls="--",
                alpha=0.7, label="1σ trap")
        ax.plot(xe2, ve2, color="steelblue", lw=1.0, ls=":",
                alpha=0.5, label="2σ trap")

        sc_alive = ax.scatter([], [], s=8, c=COLOR_ALIVE, alpha=0.6,
                               linewidths=0, zorder=5)
        sc_lost  = ax.scatter([], [], s=6, c=COLOR_LOST,  alpha=0.35,
                               linewidths=0, zorder=4)
        scatters_alive.append(sc_alive)
        scatters_lost.append(sc_lost)

        tt = ax.text(0.02, 0.96, "", transform=ax.transAxes,
                     fontsize=8, verticalalignment="top",
                     bbox=dict(boxstyle="round", fc="wheat", alpha=0.6))
        time_texts.append(tt)

        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)

    fig.suptitle("Phase-space evolution (x,y,z,r)", fontsize=13, fontweight="bold")

    data_arrays = [
        (x, vx), (y, vy), (z_arr, vz), (r_arr, vr_arr)
    ]

    def update(frame_idx):
        j = frames[frame_idx]
        alive = ~is_lost[j]
        lost  =  is_lost[j]

        for k, (pos_arr, vel_arr) in enumerate(data_arrays):
            p_a = pos_arr[j, alive]; v_a = vel_arr[j, alive]
            p_l = pos_arr[j, lost];  v_l = vel_arr[j, lost]

            if len(p_a):
                scatters_alive[k].set_offsets(np.c_[p_a, v_a])
            else:
                scatters_alive[k].set_offsets(np.empty((0, 2)))

            if len(p_l):
                scatters_lost[k].set_offsets(np.c_[p_l, v_l])
            else:
                scatters_lost[k].set_offsets(np.empty((0, 2)))

            time_texts[k].set_text(f"t = {t[j]:.2f} μs")

        return (*scatters_alive, *scatters_lost, *time_texts)

    ani = FuncAnimation(fig, update, frames=len(frames),
                        interval=int(1000 / fps), blit=True)

    out = output_dir / "phase_space_3d.gif"
    print(f"Saving phase-space animation → {out}  (dpi={dpi})")
    ani.save(str(out), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print("Done.")
