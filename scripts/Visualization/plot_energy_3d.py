"""
Energy distribution plot: initial, mid-transfer, and final snapshots.

Shows KE, PE, and total energy histograms (with KDE) for survived and lost
atoms at three time points: t=0, t=T/2, t=T.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path
from typing import Dict

from utils3d import COLOR_ALIVE, COLOR_LOST


def plot_energy_distributions_3d(data: Dict, output_dir: Path):
    t       = data["t"]
    KE      = data["KE"]      # (n_steps, n_shots)
    PE      = data["PE"]
    E_tot   = data["E_tot"]
    is_lost = data["is_lost"]
    params  = data["params"]
    scales  = data["scales"]

    n_steps, n_shots = KE.shape
    U0_uK   = scales["U0_uK"]

    snap_indices = [0, n_steps // 2, n_steps - 1]
    snap_labels  = ["Initial (t = 0)", "Mid-transfer (t = T/2)", "Final (t = T)"]

    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex="col")
    plt.subplots_adjust(hspace=0.40, wspace=0.30)

    col_titles = ["Kinetic energy", "Potential energy", "Total energy"]
    row_colors = [("KE", KE), ("PE", PE), ("E_tot", E_tot)]

    for col_idx, (label, energy_arr) in enumerate(row_colors):
        # compute global xlim for this column
        all_vals = energy_arr.flatten()
        finite   = all_vals[np.isfinite(all_vals)]
        xlo, xhi = np.percentile(finite, 1), np.percentile(finite, 99)
        xpad = (xhi - xlo) * 0.05
        xgrid = np.linspace(xlo - xpad, xhi + xpad, 500)

        for row_idx, (snap_j, snap_lbl) in enumerate(zip(snap_indices, snap_labels)):
            ax = axes[row_idx, col_idx]

            alive_mask = ~is_lost[snap_j]
            lost_mask  =  is_lost[snap_j]

            vals_a = energy_arr[snap_j, alive_mask]
            vals_l = energy_arr[snap_j, lost_mask]

            ax.set_xlim(xlo - xpad, xhi + xpad)

            for vals, color, mlabel in [
                (vals_a, COLOR_ALIVE, f"Survived ({alive_mask.sum()})"),
                (vals_l, COLOR_LOST,  f"Lost ({lost_mask.sum()})"),
            ]:
                if len(vals) < 2:
                    continue
                ax.hist(vals, bins=min(30, max(5, len(vals) // 5)),
                        density=True, alpha=0.35, color=color, edgecolor="none")
                if len(vals) > 4:
                    try:
                        kde = gaussian_kde(vals, bw_method="scott")
                        ax.plot(xgrid, kde(xgrid), color=color, lw=2.0,
                                label=mlabel)
                    except Exception:
                        pass

            if col_idx == 0:
                ax.set_ylabel(snap_lbl, fontsize=9)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=10, fontweight="bold")
            if row_idx == 2:
                ax.set_xlabel(f"Energy [dimless  |  1={U0_uK:.0f} μK]", fontsize=8)

            ax.legend(fontsize=7, framealpha=0.7)
            ax.grid(alpha=0.25, ls=":")

            # vertical line: trap boundary
            U0 = params["U0_static"]
            if label == "E_tot":
                ax.axvline(-params["trap_fraction"] * U0, color="black",
                           ls="--", lw=1.0, alpha=0.6, label="Trap boundary")

    fig.suptitle("Energy distributions at three time points", fontsize=13,
                 fontweight="bold")
    out = output_dir / "energy_distributions_3d.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")
