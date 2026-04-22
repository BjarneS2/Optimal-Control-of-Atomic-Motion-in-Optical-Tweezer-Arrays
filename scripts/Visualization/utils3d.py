"""
Shared utilities: potential evaluation, atom categorization, color scheme.
"""

import numpy as np

# ── Color scheme ───────────────────────────────────────────────────────────────
COLOR_ALIVE  = "mediumseagreen"
COLOR_LOST   = "dimgrey"
COLOR_STATIC = "#8B0000"   # dark red — static tweezers
COLOR_AUX    = "#FF6347"   # tomato red — moving aux tweezer


# ── 3D Gaussian potential ──────────────────────────────────────────────────────

def _beam_potential(x, y, z, cx, cy, cz, U0, w, zR):
    """
    U = -U0 * (w²/w(ξ)²) * exp(-2ρ²/w(ξ)²)
    All inputs in dimensionless units. Returns array of same shape as x.
    """
    xi   = z - cz
    rho2 = (x - cx)**2 + (y - cy)**2
    wxi2 = w**2 * (1.0 + (xi / zR)**2)
    f    = (w**2 / wxi2) * np.exp(-2.0 * rho2 / wxi2)
    return -U0 * f


def potential3d(x, y, z, ux, uy, ua,
                x_start, y_start, x_stop, y_stop,
                w, w_aux, zR, zR_aux, U0_static, U0_aux_max,
                cz=0.0):
    """Total potential at position (x,y,z) for given control (ux,uy,ua)."""
    U1 = _beam_potential(x, y, z, x_start, y_start, cz, U0_static, w, zR)
    U2 = _beam_potential(x, y, z, x_stop,  y_stop,  cz, U0_static, w, zR)
    Ua = _beam_potential(x, y, z, ux, uy, cz, ua * U0_aux_max, w_aux, zR_aux)
    return U1 + U2 + Ua


def potential3d_grid(X, Y, Z, ux, uy, ua, params: dict, cz=0.0):
    """Vectorised potential on meshgrid arrays."""
    return potential3d(X, Y, Z, ux, uy, ua,
                       params["x_start"], params["y_start"],
                       params["x_stop"],  params["y_stop"],
                       params["w"], params["w_aux"],
                       params["zR"], params["zR_aux"],
                       params["U0_static"], params["U0_aux_max"],
                       cz)


# ── Loss / survival categorization ────────────────────────────────────────────

def compute_loss_mask(x, y, z, vx, vy, vz,
                      t, ux_arr, uy_arr, ua_arr,
                      params, trap_fraction):
    """
    Returns is_lost[n_steps, n_shots] bool array.
    An atom is lost at step j if total_energy > trap_fraction * U_total,
    and remains lost thereafter.
    """
    n_steps, n_shots = x.shape
    is_lost = np.zeros((n_steps, n_shots), dtype=bool)

    for j in range(n_steps):
        U = potential3d(x[j], y[j], z[j],
                        ux_arr[j], uy_arr[j], ua_arr[j],
                        params["x_start"], params["y_start"],
                        params["x_stop"],  params["y_stop"],
                        params["w"], params["w_aux"],
                        params["zR"], params["zR_aux"],
                        params["U0_static"], params["U0_aux_max"])
        KE = 0.5 * (vx[j]**2 + vy[j]**2) # + vz[j]**2)
        if np.any(U > 0):
            raise ValueError(f"Unexpected positive potential U={U} at step {j}")
        E_tot = KE + U
        newly_lost = E_tot > trap_fraction * U
        if j == 0:
            is_lost[j] = newly_lost
        else:
            is_lost[j] = is_lost[j-1] | newly_lost

    return is_lost


# ── Energy helpers ─────────────────────────────────────────────────────────────

def kinetic_energy(vx, vy, vz):
    return 0.5 * (vx**2 + vy**2 + vz**2)


def total_energy(x, y, z, vx, vy, vz,
                 ux_arr, uy_arr, ua_arr, params):
    """Returns (n_steps, n_shots) arrays: KE, PE, E_tot."""
    n_steps, n_shots = x.shape
    KE   = kinetic_energy(vx, vy, vz)
    PE   = np.zeros_like(KE)
    for j in range(n_steps):
        PE[j] = potential3d(x[j], y[j], z[j],
                            ux_arr[j], uy_arr[j], ua_arr[j],
                            params["x_start"], params["y_start"],
                            params["x_stop"],  params["y_stop"],
                            params["w"], params["w_aux"],
                            params["zR"], params["zR_aux"],
                            params["U0_static"], params["U0_aux_max"])
    return KE, PE, KE + PE


# ── Spherical phase-space coordinates ─────────────────────────────────────────

def cartesian_to_spherical_phasespace(x, y, z, vx, vy, vz):
    """
    r  = sqrt(x²+y²+z²)
    vr = (x*vx + y*vy + z*vz) / r   (radial velocity)
    """
    r  = np.sqrt(x**2 + y**2 + z**2)
    # avoid divide-by-zero at origin
    safe_r = np.where(r > 0, r, 1.0)
    vr = (x * vx + y * vy + z * vz) / safe_r
    vr = np.where(r > 0, vr, 0.0)
    return r, vr


# ── Phase-space ellipse (harmonic approximation) ──────────────────────────────

def trap_ellipse_xy(U0, w, T_atom, m_dimless=1.0, n_pts=400):
    """
    Harmonic approximation: ω² = 4U0/w²  →  σ_x = sqrt(T_atom/U0)*w/2
    Returns (x_ellipse, v_ellipse) for the 1-sigma contour in (x, vx) plane.
    T_atom, U0 both dimensionless (ratio to E0).
    """
    omega2 = 4.0 * U0 / w**2
    sigma_x = np.sqrt(T_atom / omega2)
    sigma_v = np.sqrt(T_atom)      # sigma_v = sqrt(kT/m) dimensionless
    theta = np.linspace(0, 2 * np.pi, n_pts)
    return 2 * sigma_x * np.cos(theta), 2 * sigma_v * np.sin(theta)


def trap_ellipse_z(U0, zR, T_atom, n_pts=400):
    """
    Axial harmonic: ω_z² = 2U0/zR²
    """
    omega2 = 2.0 * U0 / zR**2
    sigma_z = np.sqrt(T_atom / omega2)
    sigma_v = np.sqrt(T_atom)
    theta = np.linspace(0, 2 * np.pi, n_pts)
    return 2 * sigma_z * np.cos(theta), 2 * sigma_v * np.sin(theta)
