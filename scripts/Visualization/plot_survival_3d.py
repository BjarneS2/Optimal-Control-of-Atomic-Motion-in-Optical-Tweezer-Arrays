"""
Survival-over-time plot for 3D forward dynamics.

Shows:
  - Number of survived / lost atoms vs time
  - Survival rate (%) vs time
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from utils3d import COLOR_ALIVE, COLOR_LOST


def plot_survival_3d(data: Dict, output_dir: Path):
    t       = data["t"]           # (n_steps,) [μs dimless]
    is_lost = data["is_lost"]     # (n_steps, n_shots) bool
    n_shots = is_lost.shape[1]

    n_lost_at_t     = np.sum(is_lost, axis=1)
    n_survived_at_t = n_shots - n_lost_at_t
    survival_pct    = 100.0 * n_survived_at_t / n_shots

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # ── Top: atom counts ───────────────────────────────────────────────────────
    ax1.fill_between(t, 0, n_survived_at_t,
                     color=COLOR_ALIVE, alpha=0.35, label="Survived")
    ax1.fill_between(t, n_survived_at_t, n_shots,
                     color=COLOR_LOST,  alpha=0.35, label="Lost")
    ax1.plot(t, n_survived_at_t, color=COLOR_ALIVE, lw=2.0)
    ax1.plot(t, [n_shots] * len(t), color="gray", lw=1, ls=":")
    ax1.set_ylabel("Number of atoms", fontsize=11)
    ax1.set_ylim(0, n_shots * 1.05)
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(alpha=0.3, ls=":")
    ax1.set_title(
        f"Atom survival  —  {n_shots} forward shots,  "
        f"final: {int(n_survived_at_t[-1])}/{n_shots} "
        f"({survival_pct[-1]:.1f}%)",
        fontsize=12, fontweight="bold")

    # ── Bottom: survival rate ──────────────────────────────────────────────────
    ax2.plot(t, survival_pct, color=COLOR_ALIVE, lw=2.5)
    ax2.fill_between(t, 0, survival_pct, color=COLOR_ALIVE, alpha=0.25)
    ax2.axhline(100, color="gray", ls="--", lw=1, alpha=0.5)
    ax2.axhline(survival_pct[-1], color=COLOR_ALIVE, ls=":", lw=1.5, alpha=0.8,
                label=f"Final {survival_pct[-1]:.1f}%")
    ax2.set_xlabel("Time [μs]", fontsize=11)
    ax2.set_ylabel("Survival rate [%]", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(alpha=0.3, ls=":")

    plt.tight_layout()
    out = output_dir / "survival_3d.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")
