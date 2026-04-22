import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    import seaborn as sns
    sns.set_style("dark")
except ImportError:
    pass


def load(filename):
    with h5py.File(filename, "r") as f:
        survival     = np.asarray(f["survival"]).T   # Julia col-major → restore [nU, nT, nO, nR]
        time_offsets = np.asarray(f["time_offsets_us"])
        depth_ratios = np.asarray(f["depth_ratios"])
        T_atoms_uK   = np.asarray(f["T_atoms_uK"])
        U0_fixed_uK  = np.asarray(f["U0_fixed_uK"])

        aom_A = dict(time=np.asarray(f["aom_A/time"]), amplitude=np.asarray(f["aom_A/amplitude"]))
        aom_B = dict(time=np.asarray(f["aom_B/time"]), amplitude=np.asarray(f["aom_B/amplitude"]))

        z_survival  = np.asarray(f["z_scan/survival"]).T  if "z_scan"  in f else None
        z_offsets   = np.asarray(f["z_scan/z_offsets"])   if "z_scan"  in f else None
        xy_survival = np.asarray(f["xy_scan/survival"]).T if "xy_scan" in f else None
        xy_offsets  = np.asarray(f["xy_scan/xy_offsets"]) if "xy_scan" in f else None

        attrs = dict(f.attrs)

    return dict(survival=survival, time_offsets=time_offsets, depth_ratios=depth_ratios,
                T_atoms_uK=T_atoms_uK, U0_fixed_uK=U0_fixed_uK,
                aom_A=aom_A, aom_B=aom_B,
                z_survival=z_survival, z_offsets=z_offsets,
                xy_survival=xy_survival, xy_offsets=xy_offsets,
                attrs=attrs)


def find_optimal(survival_2d, time_offsets, depth_ratios):
    """Returns (best_survival, best_time_offset, best_depth_ratio)."""
    flat = np.argmax(survival_2d)
    io, ir = np.unravel_index(flat, survival_2d.shape)
    best = survival_2d[io, ir]
    # tie-break: smallest |time_offset|
    candidates = np.argwhere(survival_2d == best)
    best_io = candidates[np.argmin(np.abs(time_offsets[candidates[:, 0]]))]
    return best, time_offsets[best_io[0]], depth_ratios[best_io[1]]


def plot_aom_responses(d, save=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(d["aom_A"]["time"], d["aom_A"]["amplitude"], color="C0")
    ax1.set_ylabel("Amplitude (norm.)")
    ax1.set_title("Trap A (1064 nm) — average AOM response")
    ax2.plot(d["aom_B"]["time"], d["aom_B"]["amplitude"], color="C1")
    ax2.set_ylabel("Amplitude (norm.)")
    ax2.set_xlabel("Time (μs)")
    ax2.set_title("Trap B (934 nm) — average AOM response")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def plot_heatmap_grid(d, save=None):
    survival    = d["survival"]          # [nU, nT, nO, nR]
    time_offsets = d["time_offsets"]
    depth_ratios = d["depth_ratios"]
    T_atoms_uK   = d["T_atoms_uK"]
    U0_fixed_uK  = d["U0_fixed_uK"]
    mode         = d["attrs"].get("mode", "?")
    trap_first   = d["attrs"].get("trap_first", "?")

    nU, nT = len(U0_fixed_uK), len(T_atoms_uK)
    fig, axes = plt.subplots(nU, nT, figsize=(4 * nT + 1, 3.5 * nU),
                             squeeze=False, constrained_layout=True)
    fig.suptitle(f"Survival rate  |  mode={mode}, trap_first={trap_first}", fontsize=13)

    vmin, vmax = 0.0, 1.0
    cmap = "viridis"
    im_ref = None

    for iu, U0 in enumerate(U0_fixed_uK):
        for it, T in enumerate(T_atoms_uK):
            ax = axes[iu, it]
            data = survival[iu, it, :, :]   # [nO, nR]

            im = ax.pcolormesh(time_offsets, depth_ratios, data.T,
                               cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
            im_ref = im

            best_s, best_t, best_r = find_optimal(data, time_offsets, depth_ratios)
            ax.plot(best_t, best_r, "*", color="white", markersize=10,
                    markeredgecolor="black", markeredgewidth=0.5,
                    label=f"max={best_s:.2f}\nΔt={best_t:.0f}μs\nr={best_r:.1f}")
            ax.legend(fontsize=7, loc="upper right", framealpha=0.6)

            ax.set_xlabel("Time offset Δt (μs)" if iu == nU - 1 else "")
            ax.set_ylabel("Depth ratio U_B/U_A" if it == 0 else "")
            ax.set_title(f"U₀={U0:.0f} μK,  T={T:.0f} μK", fontsize=9)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

    if im_ref is not None:
        fig.colorbar(im_ref, ax=axes[:, -1], label="Survival rate", shrink=0.8)

    if save:
        plt.savefig(save, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def plot_summary(d, save=None):
    survival    = d["survival"]
    time_offsets = d["time_offsets"]
    depth_ratios = d["depth_ratios"]
    T_atoms_uK   = d["T_atoms_uK"]
    U0_fixed_uK  = d["U0_fixed_uK"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    ax_s, ax_t, ax_r = axes

    colors = plt.colormaps["plasma"](np.linspace(0.2, 0.85, len(T_atoms_uK)))

    for it, (T, col) in enumerate(zip(T_atoms_uK, colors)):
        max_surv   = [np.max(survival[iu, it])             for iu in range(len(U0_fixed_uK))]
        opt_t      = [find_optimal(survival[iu, it], time_offsets, depth_ratios)[1]
                      for iu in range(len(U0_fixed_uK))]
        opt_r      = [find_optimal(survival[iu, it], time_offsets, depth_ratios)[2]
                      for iu in range(len(U0_fixed_uK))]
        label = f"T={T:.0f} μK"
        ax_s.plot(U0_fixed_uK, max_surv, "o-", color=col, label=label)
        ax_t.plot(U0_fixed_uK, opt_t,    "o-", color=col, label=label)
        ax_r.plot(U0_fixed_uK, opt_r,    "o-", color=col, label=label)

    ax_s.set_xlabel("Initial trap depth U₀ (μK)")
    ax_s.set_ylabel("Max survival rate")
    ax_t.set_xlabel("Initial trap depth U₀ (μK)")
    ax_t.set_ylabel("Optimal Δt (μs)")
    ax_r.set_xlabel("Initial trap depth U₀ (μK)")
    ax_r.set_ylabel("Optimal depth ratio")
    ax_s.set_title("Peak survival")
    ax_t.set_title("Optimal time offset")
    ax_r.set_title("Optimal depth ratio")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    if save:
        plt.savefig(save, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def plot_z_scan(d, save=None):
    if d["z_survival"] is None:
        return
    z_surv    = d["z_survival"]     # [nU, nT, nZ]
    z_offsets = d["z_offsets"]
    T_atoms_uK  = d["T_atoms_uK"]
    U0_fixed_uK = d["U0_fixed_uK"]

    nU, nT = len(U0_fixed_uK), len(T_atoms_uK)
    fig, axes = plt.subplots(nU, nT, figsize=(4 * nT + 1, 3 * nU),
                             squeeze=False, constrained_layout=True)
    fig.suptitle("Survival vs. z-offset between traps", fontsize=12)

    for iu, U0 in enumerate(U0_fixed_uK):
        for it, T in enumerate(T_atoms_uK):
            ax = axes[iu, it]
            ax.plot(z_offsets, z_surv[iu, it, :], "o-", color="C2")
            ax.set_ylim(0, 1)
            ax.set_xlabel("z offset (μm)" if iu == nU - 1 else "")
            ax.set_ylabel("Survival" if it == 0 else "")
            ax.set_title(f"U₀={U0:.0f} μK, T={T:.0f} μK", fontsize=9)
            ax.grid(True, alpha=0.3)

    if save:
        plt.savefig(save, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def plot_xy_scan(d, save=None):
    if d["xy_survival"] is None:
        return
    xy_surv   = d["xy_survival"]    # [nU, nT, nXY]
    xy_offsets = d["xy_offsets"]
    T_atoms_uK  = d["T_atoms_uK"]
    U0_fixed_uK = d["U0_fixed_uK"]

    nU, nT = len(U0_fixed_uK), len(T_atoms_uK)
    fig, axes = plt.subplots(nU, nT, figsize=(4 * nT + 1, 3 * nU),
                             squeeze=False, constrained_layout=True)
    fig.suptitle("Survival sensitivity to transverse (xy) offset of trap B", fontsize=12)

    for iu, U0 in enumerate(U0_fixed_uK):
        for it, T in enumerate(T_atoms_uK):
            ax = axes[iu, it]
            ax.plot(xy_offsets, xy_surv[iu, it, :], "o-", color="C3")
            ax.set_ylim(0, 1)
            ax.set_xlabel("Transverse offset (μm)" if iu == nU - 1 else "")
            ax.set_ylabel("Survival" if it == 0 else "")
            ax.set_title(f"U₀={U0:.0f} μK, T={T:.0f} μK", fontsize=9)
            ax.grid(True, alpha=0.3)

    if save:
        plt.savefig(save, dpi=150)
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_pouring.py <results.h5> [--save]")
        sys.exit(1)

    filename = sys.argv[1]
    save_figs = "--save" in sys.argv
    stem = filename.replace(".h5", "")

    d = load(filename)
    survival: np.ndarray = np.asarray(d["survival"])
    attrs: dict = dict(d["attrs"])  # type: ignore[arg-type]
    print(f"Loaded: {filename}")
    print(f"  survival shape: {survival.shape}  [nU0 x nT x nOffset x nRatio]")
    print(f"  mode={attrs.get('mode','?')}, trap_first={attrs.get('trap_first','?')}")

    plot_aom_responses(d, save=f"{stem}_aom.png"     if save_figs else None)
    plot_heatmap_grid (d, save=f"{stem}_heatmaps.png" if save_figs else None)
    plot_summary      (d, save=f"{stem}_summary.png"  if save_figs else None)
    plot_z_scan       (d, save=f"{stem}_zscan.png"    if save_figs else None)
    plot_xy_scan      (d, save=f"{stem}_xyscan.png"   if save_figs else None)
