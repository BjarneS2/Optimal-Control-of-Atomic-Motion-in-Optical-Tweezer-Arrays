import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
from pathlib import Path

COLOR_A    = "#2ecc71"   # initial trap (TiSaph)  — green
COLOR_B    = "#3498db"   # target trap (Mephisto) — blue
COLOR_LOST = "#e74c3c"   # lost                   — red
COLOR_SIG  = "#888888"   # control signal overlay


# ── Gaussian beam potential (dimensionless, same convention as pouring.jl) ────
def _beam_u(x, y, z, w, zR, U0_dl):
    """U(r) = -U0 * (w²/w(z)²) * exp(-2ρ²/w(z)²)  [dimensionless]"""
    wz2  = w**2 * (1.0 + (z / zR)**2)
    rho2 = x**2 + y**2
    return -U0_dl * (w**2 / wz2) * np.exp(-2.0 * rho2 / wz2)


# ── Per-timestep trapping classification ─────────────────────────────────────
def classify_atoms(x, y, z, vx, vy, vz,
                   w_A, zR_A, U_A,
                   w_B, zR_B, U_B,
                   trap_frac):
    """
    Returns bool arrays (N_atoms, N_t) for each category.
    An atom is 'trapped in X' when its total energy in X's potential is
    below TRAP_FRAC * depth, regardless of signal amplitude — the signal
    tells us which trap is nominally on, but we classify continuously.
    """
    E_kin = 0.5 * (vx**2 + vy**2 + vz**2)          # (N, T)
    U_A_r = _beam_u(x, y, z, w_A, zR_A, U_A)        # (N, T)
    U_B_r = _beam_u(x, y, z, w_B, zR_B, U_B)        # (N, T)
    in_A = (E_kin + U_A_r) < trap_frac * (-U_A)
    in_B = (E_kin + U_B_r) < trap_frac * (-U_B)
    lost = ~in_A & ~in_B
    return in_A, in_B, lost


def plot_survival(h5_path: str | Path, output_path: str | Path | None = None):
    h5_path = Path(h5_path)

    def _s(f, key):  # read scalar from h5py dataset
        return float(f[key][()]) # type: ignore[index]

    with h5py.File(h5_path, "r") as f:
        w_A       = _s(f, "constants/w_TiSaph_um")
        zR_A      = _s(f, "constants/zR_TiSaph_um")
        U_A       = _s(f, "constants/U_TiSaph_dl")
        w_B       = _s(f, "constants/w_Mephisto_um")
        zR_B      = _s(f, "constants/zR_Mephisto_um")
        U_B       = _s(f, "constants/U_Mephisto_dl")
        trap_frac = _s(f, "constants/trap_frac")
        T_uK      = _s(f, "constants/T_atom_uK")
        U_A_uK    = _s(f, "constants/U_TiSaph_uK")
        U_B_uK    = _s(f, "constants/U_Mephisto_uK")

        t_us    = f["protocol/t_us"][:]       # type: ignore[index]
        sig_A   = f["protocol/sig_A"][:]      # type: ignore[index]
        sig_B   = f["protocol/sig_B"][:]      # type: ignore[index]
        best_dt = _s(f, "protocol/best_dt_us")

        x  = f["trajectories/x_um"][:].T        # type: ignore[index]
        y  = f["trajectories/y_um"][:].T        # type: ignore[index]
        z  = f["trajectories/z_um"][:].T        # type: ignore[index]
        vx = f["trajectories/vx_ms"][:].T       # type: ignore[index]
        vy = f["trajectories/vy_ms"][:].T       # type: ignore[index]
        vz = f["trajectories/vz_ms"][:].T       # type: ignore[index]

    N, T = x.shape
    in_A, in_B, lost = classify_atoms(x, y, z, vx, vy, vz,
                                       w_A, zR_A, U_A,
                                       w_B, zR_B, U_B,
                                       trap_frac)

    n_A    = in_A.sum(axis=0)          # (T,)
    n_B    = in_B.sum(axis=0)
    n_lost = lost.sum(axis=0)
    survival_pct = 100.0 * (n_A + n_B) / N

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(11, 9),
                              gridspec_kw={"height_ratios": [3, 2, 1]},
                              sharex=True)
    ax1, ax2, ax3 = axes

    # — top: stacked atom count ------------------------------------------------
    base_B = n_A
    base_L = n_A + n_B

    ax1.fill_between(t_us, 0,      n_A,    color=COLOR_A,    alpha=0.35, label="In trap A (TiSaph)")
    ax1.fill_between(t_us, base_B, base_B + n_B, color=COLOR_B, alpha=0.35, label="In trap B (Mephisto)")
    ax1.fill_between(t_us, base_L, N,      color=COLOR_LOST, alpha=0.35, label="Lost")
    ax1.plot(t_us, n_A,            color=COLOR_A,    lw=2)
    ax1.plot(t_us, base_B + n_B,   color=COLOR_B,    lw=2)
    ax1.plot(t_us, base_L + n_lost, color=COLOR_LOST, lw=2)
    ax1.axhline(N, color="k", lw=1, ls=":", alpha=0.4, label=f"Total ({N})")

    final_in_B = int(n_B[-1])
    stats = (f"Final: {final_in_B}/{N} in trap B  ({100*final_in_B/N:.1f}%)\n"
             f"Optimal Δt = {best_dt:+.2f} μs")
    ax1.text(0.99, 0.97, stats, transform=ax1.transAxes,
             ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))
    ax1.set_ylabel("Number of atoms", fontsize=11)
    ax1.set_title(
        f"Atom survival during pouring  "
        f"(T={T_uK:.0f} μK,  U_A={U_A_uK:.0f} μK,  U_B={U_B_uK:.0f} μK)",
        fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25, ls=":")
    ax1.set_ylim(0, N * 1.05)

    # — middle: survival rate --------------------------------------------------
    ax2.plot(t_us, survival_pct, color=COLOR_A, lw=2.5)
    ax2.fill_between(t_us, 0, survival_pct, color=COLOR_A, alpha=0.25)
    ax2.axhline(100, color="gray", ls="--", lw=1, alpha=0.5)
    ax2.set_ylabel("Survival rate (%)", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.grid(alpha=0.25, ls=":")

    # vertical lines at individual loss events (first timestep atom leaves both traps)
    first_lost = np.argmax(lost, axis=1)          # index of first lost step per atom
    ever_lost  = lost.any(axis=1)
    loss_times = t_us[first_lost[ever_lost]]
    for tl in loss_times:
        ax2.axvline(tl, color=COLOR_LOST, alpha=0.08, lw=0.5)

    # — bottom: control signals ------------------------------------------------
    ax3.plot(t_us, sig_A, color=COLOR_A,    lw=1.8, label="Signal A (TiSaph off)")
    ax3.plot(t_us, sig_B, color=COLOR_B,    lw=1.8, label="Signal B (Mephisto on)")
    ax3.set_ylabel("AOM amplitude", fontsize=10)
    ax3.set_xlabel("Time (μs)", fontsize=11)
    ax3.set_ylim(-0.05, 1.15)
    ax3.legend(fontsize=9, loc="upper right")
    ax3.grid(alpha=0.25, ls=":")

    plt.tight_layout()

    if output_path is None:
        output_path = h5_path.with_name(h5_path.stem + "_survival.png")
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_potential_gif(h5_path: str | Path, output_path: str | Path | None = None,
                       n_frames: int = 120, fps: int = 20):
    h5_path = Path(h5_path)

    def _s(f, key):
        return float(f[key][()]) # type: ignore[index]

    with h5py.File(h5_path, "r") as f:
        w_A    = _s(f, "constants/w_TiSaph_um")
        zR_A   = _s(f, "constants/zR_TiSaph_um")
        U_A    = _s(f, "constants/U_TiSaph_dl")
        w_B    = _s(f, "constants/w_Mephisto_um")
        zR_B   = _s(f, "constants/zR_Mephisto_um")
        U_B    = _s(f, "constants/U_Mephisto_dl")
        U_A_uK = _s(f, "constants/U_TiSaph_uK")
        U_B_uK = _s(f, "constants/U_Mephisto_uK")
        T_uK   = _s(f, "constants/T_atom_uK")
        trap_frac = _s(f, "constants/trap_frac")
        best_dt   = _s(f, "protocol/best_dt_us")

        t_us  = f["protocol/t_us"][:]   # type: ignore[index]
        sig_A = f["protocol/sig_A"][:]  # type: ignore[index]
        sig_B = f["protocol/sig_B"][:]  # type: ignore[index]

        x  = f["trajectories/x_um"][:].T   # type: ignore[index]
        y  = f["trajectories/y_um"][:].T   # type: ignore[index]
        z  = f["trajectories/z_um"][:].T   # type: ignore[index]
        vx = f["trajectories/vx_ms"][:].T  # type: ignore[index]
        vy = f["trajectories/vy_ms"][:].T  # type: ignore[index]
        vz = f["trajectories/vz_ms"][:].T  # type: ignore[index]

    N, T = x.shape
    in_A, in_B, lost = classify_atoms(x, y, z, vx, vy, vz,
                                       w_A, zR_A, U_A,
                                       w_B, zR_B, U_B,
                                       trap_frac)
    n_A    = in_A.sum(axis=0)
    n_B    = in_B.sum(axis=0)
    n_lost = lost.sum(axis=0)
    base_B = n_A
    base_L = n_A + n_B

    x_um   = np.linspace(-30, 30, 800)
    U_A_x  = _beam_u(x_um, 0.0, 0.0, w_A, zR_A, U_A)
    U_B_x  = _beam_u(x_um, 0.0, 0.0, w_B, zR_B, U_B)
    U_full = U_A + U_B
    pot_ylim = (min(U_A_x.min(), U_B_x.min()) * 1.08, U_full * 0.06)

    frame_idx = np.linspace(0, T - 1, n_frames, dtype=int)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.fill_between(t_us, 0,      n_A,              color=COLOR_A,    alpha=0.35, label="In trap A (TiSaph)")
    ax1.fill_between(t_us, base_B, base_B + n_B,     color=COLOR_B,    alpha=0.35, label="In trap B (Mephisto)")
    ax1.fill_between(t_us, base_L, base_L + n_lost,  color=COLOR_LOST, alpha=0.35, label="Lost")
    ax1.plot(t_us, n_A,             color=COLOR_A,    lw=2)
    ax1.plot(t_us, base_B + n_B,    color=COLOR_B,    lw=2)
    ax1.plot(t_us, base_L + n_lost, color=COLOR_LOST, lw=2)
    ax1.axhline(N, color="k", lw=1, ls=":", alpha=0.4, label=f"Total ({N})")
    final_in_B = int(n_B[-1])
    stats = (f"Final: {final_in_B}/{N} in trap B  ({100*final_in_B/N:.1f}%)\n"
             f"Optimal Δt = {best_dt:+.2f} μs")
    ax1.text(0.99, 0.97, stats, transform=ax1.transAxes,
             ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))
    ax1.set_ylabel("Number of atoms", fontsize=11)
    ax1.set_title(
        f"Atom survival during pouring  "
        f"(T={T_uK:.0f} μK,  U_A={U_A_uK:.0f} μK,  U_B={U_B_uK:.0f} μK)",
        fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25, ls=":")
    ax1.set_ylim(0, N * 1.05)
    ax1.set_xlim(t_us[0], t_us[-1])

    vline1 = ax1.axvline(t_us[0], color="#aaaaaa", lw=1.5, ls="--", zorder=10)

    ax2.set_xlabel("x (μm)", fontsize=11)
    ax2.set_ylabel("Potential (dimensionless)", fontsize=11)
    ax2.set_xlim(x_um[0], x_um[-1])
    ax2.set_ylim(pot_ylim)
    ax2.axhline(0, color="k", lw=0.8, alpha=0.4)
    ax2.grid(alpha=0.25, ls=":")

    line_A,  = ax2.plot([], [], color=COLOR_A,   lw=2.2, label="Trap A (TiSaph)")
    line_B,  = ax2.plot([], [], color=COLOR_B,   lw=2.2, label="Trap B (Mephisto)")
    line_AB, = ax2.plot([], [], color="#9b59b6", lw=2.2, ls="--", label="A + B (combined)")
    fill_A  = ax2.fill_between(x_um, 0, 0, color=COLOR_A,   alpha=0.15)
    fill_B  = ax2.fill_between(x_um, 0, 0, color=COLOR_B,   alpha=0.15)
    fill_AB = ax2.fill_between(x_um, 0, 0, color="#9b59b6", alpha=0.10)
    ax2.legend(fontsize=9, loc="lower center", ncol=3)

    plt.tight_layout()

    def _update(fi):
        nonlocal fill_A, fill_B, fill_AB
        i = frame_idx[fi]
        vline1.set_xdata([t_us[i], t_us[i]])

        ua = U_A_x * sig_A[i]
        ub = U_B_x * sig_B[i]
        uab = ua + ub

        line_A.set_data(x_um, ua)
        line_B.set_data(x_um, ub)
        line_AB.set_data(x_um, uab)

        fill_A.remove()
        fill_A  = ax2.fill_between(x_um, ua,  0, color=COLOR_A,   alpha=0.15)
        fill_B.remove()
        fill_B  = ax2.fill_between(x_um, ub,  0, color=COLOR_B,   alpha=0.15)
        fill_AB.remove()
        fill_AB = ax2.fill_between(x_um, uab, 0, color="#9b59b6", alpha=0.10)

        return line_A, line_B, line_AB, vline1

    anim = animation.FuncAnimation(fig, _update, frames=n_frames,
                                    interval=1000 // fps, blit=False)

    if output_path is None:
        output_path = h5_path.with_name(h5_path.stem + "_potential.gif")
    output_path = Path(output_path)
    anim.save(str(output_path), writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    path = Path(__file__).parent / "results.h5"
    plot_survival(path)
    plot_potential_gif(path)
