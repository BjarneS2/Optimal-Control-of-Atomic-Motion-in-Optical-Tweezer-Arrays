"""
3D transport animation.

Camera modes
------------
follow  : camera moves with the auxiliary tweezer (pulled back enough to keep
          both static traps in view), with an optional hold phase at the start
          where the atoms stay stationary while the camera orbits the initial
          trap.
static  : fixed wide-view camera throughout.

Potential rendering
-------------------
Each Gaussian beam surface is coloured by the actual potential amplitude at
every surface point, so the brightest / most-opaque regions coincide with the
trap focus.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from pathlib import Path
from typing import Dict

from utils3d import (
    COLOR_ALIVE, COLOR_LOST, COLOR_AUX,
    potential3d,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _gaussian_surface(cx, cy, cz, w, zR,
                       n_xi=40, n_phi=60, xi_range=None,
                       rho_scale=3.0):
    """
    Parametric hourglass surface for one Gaussian beam.

    The surface is drawn at rho_scale × w(ξ) so it is visible at scene scale.
    C ∈ [0,1] is the on-axis intensity factor — brightest / most opaque at
    the focus, fading toward the ends of the cone.
    """
    if xi_range is None:
        xi_range = 2.5 * zR

    xi_arr  = np.linspace(-xi_range, xi_range, n_xi)
    phi_arr = np.linspace(0, 2 * np.pi, n_phi)
    XI, PHI = np.meshgrid(xi_arr, phi_arr, indexing="ij")

    wxi2 = w**2 * (1.0 + (XI / zR)**2)
    rho  = rho_scale * np.sqrt(wxi2)   # visually scaled radius

    X = cx + rho * np.cos(PHI)
    Y = cy + rho * np.sin(PHI)
    Z = cz + XI

    # on-axis intensity profile: = 1 at focus, → 0 toward ends
    C = w**2 / wxi2

    return X, Y, Z, C


def _add_beam_surface(ax, cx, cy, cz, w, zR,
                      base_color, alpha_max=0.55,
                      n_xi=40, n_phi=60, xi_range=None,
                      rho_scale=3.0):
    """Draw one Gaussian beam as a coloured semi-transparent surface."""
    X, Y, Z, C = _gaussian_surface(cx, cy, cz, w, zR,
                                    n_xi=n_xi, n_phi=n_phi,
                                    xi_range=xi_range, rho_scale=rho_scale)
    rgba = np.zeros((*C.shape, 4))
    rgba[..., :3] = mcolors.to_rgb(base_color)
    rgba[..., 3]  = np.clip(C * alpha_max, 0.0, 1.0)
    surf = ax.plot_surface(X, Y, Z, facecolors=rgba,
                           linewidth=0, antialiased=True, shade=False)
    return surf


def _sigmoid(frac):
    """Smooth step ∈ [0,1] for frac ∈ [0,1]."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(frac, 0.0, 1.0))


def _interp(frac, v0, v1):
    return v0 + _sigmoid(frac) * (v1 - v0)


def _unwrap_azim(azim_arr):
    """Unwrap an azimuth sequence (degrees) to remove ±180 discontinuities."""
    unwrapped = np.array(azim_arr, dtype=float)
    for i in range(1, len(unwrapped)):
        diff = unwrapped[i] - unwrapped[i - 1]
        if diff > 180:
            unwrapped[i:] -= 360
        elif diff < -180:
            unwrapped[i:] += 360
    return unwrapped


# ── main entry ─────────────────────────────────────────────────────────────────

def create_transport_animation_3d(data: Dict, output_dir: Path,
                                   fps: int = 20,
                                   dpi: int = 120,
                                   total_duration_s: float = 10.0,
                                   hold_s: float = 3.0,
                                   camera_mode: str = "follow",
                                   show_potentials: bool = True):
    """
    Parameters
    ----------
    total_duration_s : total gif playback time in seconds.
    hold_s           : seconds of camera orbit at initial trap (atoms frozen).
                       Set to 0 to skip.
    camera_mode      : "follow" — camera tracks aux tweezer.
                       "static" — fixed wide view throughout.
    """
    t       = data["t"]
    x       = data["x"]
    y       = data["y"]
    z       = data["z"]
    ux_arr  = data["ux"]
    uy_arr  = data["uy"]
    ua_arr  = data["ua"]
    is_lost = data["is_lost"]

    p       = data["params"]
    x_start = p["x_start"]
    y_start = p["y_start"]
    x_stop  = p["x_stop"]
    y_stop  = p["y_stop"]
    w       = p["w"]
    w_aux   = p["w_aux"]
    zR      = p["zR"]
    zR_aux  = p["zR_aux"]
    n_steps, _ = x.shape

    x_mid = 0.5 * (x_start + x_stop)
    y_mid = 0.5 * (y_start + y_stop)

    # ── Axis limits (fixed for the whole animation) ────────────────────────────
    margin_xy = max(w, w_aux) * 4.0
    margin_z  = min(zR, zR_aux) * 0.5   # z range matched to visible cone height
    xlim = (min(x_start, x_stop) - margin_xy, max(x_start, x_stop) + margin_xy)
    ylim = (min(y_start, y_stop) - margin_xy, max(y_start, y_stop) + margin_xy)
    zlim = (-margin_z, margin_z)

    # ── Frame layout ──────────────────────────────────────────────────────────
    FLY_FRAC     = 0.15
    total_frames = max(60, int(fps * total_duration_s))
    hold_frames  = int(fps * hold_s)
    remaining    = total_frames - hold_frames
    fly_frames   = max(1, int(remaining * FLY_FRAC))
    run_frames   = remaining - fly_frames

    print(f"Total frames: {total_frames}  "
          f"(hold={hold_frames}, fly={fly_frames}, run={run_frames})")

    def _sim_step(frame_idx):
        if frame_idx < hold_frames + fly_frames:
            return 0
        run_frac = (frame_idx - hold_frames - fly_frames) / max(1, run_frames - 1)
        return int(np.clip(run_frac, 0.0, 1.0) * (n_steps - 1))

    # ── Camera azimuth schedule ────────────────────────────────────────────────
    # Pre-compute raw follow-azimuths for all sim steps, then unwrap.
    # Camera sits 120° behind-and-to-the-side of the line from mid → aux.
    raw_azim = np.array([
        np.degrees(np.arctan2(float(uy_arr[j]) - y_mid,
                              float(ux_arr[j]) - x_mid)) - 120.0
        for j in range(n_steps)
    ])
    follow_azim = _unwrap_azim(raw_azim)

    elev_near, azim_near, dist_near = 20.0, follow_azim[0], 6.0
    elev_wide, dist_wide            = 22.0, 10.5
    azim_orbit_end = azim_near + 60.0   # sweep 60° during hold

    # azimuth at the end of the fly-in = first run-phase follow azimuth
    azim_fly_target = follow_azim[0]

    def _azim_for_frame(frame_idx):
        if frame_idx < hold_frames:
            frac = frame_idx / max(1, hold_frames - 1)
            return azim_near + frac * (azim_orbit_end - azim_near)
        if frame_idx < hold_frames + fly_frames:
            frac = (frame_idx - hold_frames) / fly_frames
            return _interp(frac, azim_orbit_end, azim_fly_target)
        # run phase: interpolate along pre-computed unwrapped schedule
        j = _sim_step(frame_idx)
        return follow_azim[j]

    def _dist_for_frame(frame_idx):
        if frame_idx < hold_frames:
            return dist_near
        if frame_idx < hold_frames + fly_frames:
            frac = (frame_idx - hold_frames) / fly_frames
            return _interp(frac, dist_near, dist_wide)
        return dist_wide

    def _elev_for_frame(frame_idx):
        if frame_idx < hold_frames:
            return elev_near
        if frame_idx < hold_frames + fly_frames:
            frac = (frame_idx - hold_frames) / fly_frames
            return _interp(frac, elev_near, elev_wide)
        return elev_wide

    # ── Figure / axes ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8), facecolor="black")
    ax: Axes3D = fig.add_subplot(111, projection="3d", facecolor="black")  # type: ignore[assignment]
    ax.set_xlabel("x [μm]", color="white", labelpad=6)
    ax.set_ylabel("y [μm]", color="white", labelpad=6)
    ax.set_zlabel("z [μm]", color="white", labelpad=6)
    ax.tick_params(colors="white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("grey")

    def _fix_limits():
        """Re-apply fixed axis limits (matplotlib resets them on surface add/remove)."""
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)

    ax.autoscale(False)
    _fix_limits()

    # ── Static tweezer surfaces (drawn once) ───────────────────────────────────
    STATIC_COLOR = "#FF8070"   # light red / salmon
    cone_z = margin_z          # fill the full visible z range
    if show_potentials:
        _add_beam_surface(ax, x_start, y_start, 0.0, w, zR,
                          STATIC_COLOR, alpha_max=0.65, xi_range=cone_z)
        _add_beam_surface(ax, x_stop,  y_stop,  0.0, w, zR,
                          STATIC_COLOR, alpha_max=0.65, xi_range=cone_z)
        _fix_limits()

    # ── Aux tweezer surface (updated each frame) ───────────────────────────────
    aux_surf_container = [None]

    def _redraw_aux(j):
        if not show_potentials:
            return
        if aux_surf_container[0] is not None:
            try:
                aux_surf_container[0].remove()
            except Exception:
                pass
            aux_surf_container[0] = None
        ua_j = float(ua_arr[j])
        if ua_j < 5e-3:
            return
        alpha = 0.50 * float(np.clip(ua_j, 0.0, 1.0))
        surf = _add_beam_surface(ax,
                                  float(ux_arr[j]), float(uy_arr[j]), 0.0,
                                  w_aux, zR_aux,
                                  COLOR_AUX, alpha_max=alpha,
                                  xi_range=cone_z * (zR_aux / zR))
        aux_surf_container[0] = surf
        _fix_limits()

    # ── Atom scatter ───────────────────────────────────────────────────────────
    survived_sc = ax.scatter([], [], [], s=14, c=COLOR_ALIVE,  # type: ignore[call-arg]
                              alpha=0.75, depthshade=True, label="Survived")
    lost_sc     = ax.scatter([], [], [], s=8,  c=COLOR_LOST,   # type: ignore[call-arg]
                              alpha=0.40, depthshade=True, label="Lost")

    # Legend bottom-right; time counter top-right — no overlap
    ax.legend(loc="lower right", fontsize=8, labelcolor="white",
              facecolor="black", framealpha=0.6)
    time_text = ax.text2D(0.98, 0.96, "", transform=ax.transAxes,
                           color="white", fontsize=9, ha="right")

    # ── Animation update ───────────────────────────────────────────────────────
    def _set_dist(d):
        try:
            ax.dist = d  # type: ignore[attr-defined]
        except AttributeError:
            pass

    def update(frame_idx):
        j = _sim_step(frame_idx)

        # atoms
        alive_mask = ~is_lost[j]
        lost_mask  =  is_lost[j]
        survived_sc._offsets3d = (  # type: ignore[attr-defined]
            x[j, alive_mask], y[j, alive_mask], z[j, alive_mask])
        lost_sc._offsets3d = (      # type: ignore[attr-defined]
            x[j, lost_mask],  y[j, lost_mask],  z[j, lost_mask])

        # aux tweezer (also calls _fix_limits internally)
        _redraw_aux(j)

        # time label
        time_text.set_text(f"t = {t[j]:.2f} μs")

        # camera
        if camera_mode == "static":
            ax.view_init(elev=elev_wide, azim=follow_azim[0])  # type: ignore[attr-defined]
            _set_dist(dist_wide)
        else:
            ax.view_init(elev=_elev_for_frame(frame_idx),      # type: ignore[attr-defined]
                         azim=_azim_for_frame(frame_idx))
            _set_dist(_dist_for_frame(frame_idx))

        return survived_sc, lost_sc, time_text

    ani = FuncAnimation(fig, update, frames=total_frames,
                        interval=int(1000 / fps), blit=False)

    output_path = output_dir / "transport_3d.gif"
    print(f"Saving 3D transport animation → {output_path}")
    ani.save(str(output_path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print("Done.")
