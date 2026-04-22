"""
Physical constants and scale computations for 3D tweezer visualization.

Units: space [μm = dimless], time [μs = dimless], velocity [m/s = dimless].
The HDF5 files store everything in these dimensionless units.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class PhysicalConstants3D:
    kB: float = 1.380649e-23
    m: float = 2.20695e-25          # Cs-133 [kg]
    g_SI: float = 9.81              # [m/s²]
    w0_um: float = 1.0              # space unit [μm]
    t0_us: float = 1.0              # time unit [μs]


DEFAULTS = PhysicalConstants3D()


def compute_scales(T_tweezer: float, w0_um: float,
                   consts: PhysicalConstants3D = DEFAULTS) -> dict:
    """Return a dict of derived scale factors."""
    w0_SI = w0_um * 1e-6          # [m]
    t0_SI = consts.t0_us * 1e-6   # [s]
    v0    = w0_SI / t0_SI         # [m/s]  (= 1 m/s for default units)
    E0    = consts.m * v0**2      # [J]
    U0_J  = consts.kB * T_tweezer # [J]    trap depth
    U0_dimless = U0_J / E0        # dimensionless trap depth
    g_dimless  = consts.g_SI * t0_SI**2 / w0_SI
    return dict(
        w0_SI      = w0_SI,
        t0_SI      = t0_SI,
        v0         = v0,
        E0         = E0,
        U0_J       = U0_J,
        U0_dimless = U0_dimless,
        U0_uK      = T_tweezer * 1e6,     # trap depth in μK
        g_dimless  = g_dimless,
        t0_us      = consts.t0_us,
        w0_um      = w0_um,
    )
