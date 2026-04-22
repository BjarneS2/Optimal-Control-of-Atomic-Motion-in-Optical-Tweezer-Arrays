module Models3D

export potential3d, forces3d, barrier_height3d

using ..Types3D

# Dimensionless 3D Gaussian beam potential.
#
# Space unit: 1 μm, time unit: 1 μs  →  velocity unit v0 = 1 m/s,
# energy unit E0 = m·v0² (all potentials are in units of E0).
#
# The tweezer beam propagates along z.  Radial coordinate in the x-y plane:
#   ρ² = (x - cx)² + (y - cy)²
# Beam waist at axial offset Ξ = z - cz:
#   w(Ξ)² = w²·(1 + (Ξ/zR)²)
# Potential (Grimm et al. 2000, Eq. 42):
#   U(ρ,Ξ) = -U0 · (w²/w(Ξ)²) · exp(-2ρ²/w(Ξ)²)

@inline function _beam_factor(x::Real, y::Real, z::Real,
                               cx::Real, cy::Real, cz::Real,
                               w::Float64, zR::Float64)
    Xi   = z - cz
    rho2 = (x - cx)^2 + (y - cy)^2
    wXi2 = w^2 * (1.0 + (Xi / zR)^2)
    f    = (w^2 / wXi2) * exp(-2.0 * rho2 / wXi2)
    return f, rho2, Xi, wXi2
end

function potential3d(x::Real, y::Real, z::Real,
                     ux::Real, uy::Real, ua::Real,
                     p::TweezerParams3D)
    w     = p.w
    w_aux = p.w * p.w_aux_factor
    zR    = p.zR
    zR_a  = p.zR_aux
    cz    = 0.0   # tweezer focus always at z=0 (transport in x-y plane)

    f_st1, _, _, _ = _beam_factor(x, y, z, p.x_start, 0.0, cz, w, zR)
    f_st2, _, _, _ = _beam_factor(x, y, z, p.x_stop,  p.y_stop, cz, w, zR)
    f_aux, _, _, _ = _beam_factor(x, y, z, ux, uy, cz, w_aux, zR_a)

    return -p.U0_static * f_st1 - p.U0_static * f_st2 - ua * p.U0_aux_max * f_aux
end

function potential3d(x::Real, y::Real, z::Real,
                     ux::Real, uy::Real, ua::Real,
                     p::TweezerParams3D, ::Val{:jump})
    w     = p.w
    w_aux = p.w * p.w_aux_factor
    zR    = p.zR
    zR_a  = p.zR_aux
    cz    = 0.0

    Xi1   = z - cz;  r1sq = (x - p.x_start)^2 + (y - 0.0)^2
    wXi1  = w^2 * (1.0 + (Xi1/zR)^2);  f1 = (w^2/wXi1)*exp(-2.0*r1sq/wXi1)

    Xi2   = z - cz;  r2sq = (x - p.x_stop)^2 + (y - p.y_stop)^2
    wXi2  = w^2 * (1.0 + (Xi2/zR)^2);  f2 = (w^2/wXi2)*exp(-2.0*r2sq/wXi2)

    Xia   = z - cz;  rasq = (x - ux)^2 + (y - uy)^2
    wXia  = w_aux^2 * (1.0 + (Xia/zR_a)^2);  fa = (w_aux^2/wXia)*exp(-2.0*rasq/wXia)

    return -p.U0_static*f1 - p.U0_static*f2 - ua*p.U0_aux_max*fa
end

# Forces: F = -∇U.  Analytical derivatives of the Gaussian beam potential.
#
# ∂U/∂x = -U0 · (-4(x-cx)/w(Ξ)²) · f   →  Fx = -4·U0·(x-cx)/w(Ξ)² · f
# ∂U/∂z: uses d/dΞ[(w²/w(Ξ)²)·exp(-2ρ²/w(Ξ)²)]
#       = f·(w²/w(Ξ)²)·(Ξ/zR²)·(4ρ²/w(Ξ)² - 2)
#   Fz = -∂U/∂z = U0·f·(w²/w(Ξ)²)·(Ξ/zR²)·(4ρ²/w(Ξ)² - 2)
#        + gravity term (below)

@inline function _beam_forces(x::Real, y::Real, z::Real,
                               cx::Real, cy::Real, cz::Real,
                               U0::Real, w::Float64, zR::Float64)
    Xi   = z - cz
    dx   = x - cx
    dy   = y - cy
    rho2 = dx^2 + dy^2
    wXi2 = w^2 * (1.0 + (Xi / zR)^2)
    alpha = w^2 / wXi2
    f     = alpha * exp(-2.0 * rho2 / wXi2)

    Fx = -4.0 * U0 * dx / wXi2 * f
    Fy = -4.0 * U0 * dy / wXi2 * f
    Fz =  U0  * f * alpha * (Xi / zR^2) * (4.0 * rho2 / wXi2 - 2.0)
    return Fx, Fy, Fz
end

function forces3d(x::Real, y::Real, z::Real,
                  ux::Real, uy::Real, ua::Real,
                  p::TweezerParams3D, g_dimless::Float64)
    w     = p.w
    w_aux = p.w * p.w_aux_factor
    zR    = p.zR
    zR_a  = p.zR_aux
    cz    = 0.0

    Fx1, Fy1, Fz1 = _beam_forces(x, y, z, p.x_start, 0.0,      cz, p.U0_static,           w,     zR)
    Fx2, Fy2, Fz2 = _beam_forces(x, y, z, p.x_stop,  p.y_stop, cz, p.U0_static,           w,     zR)
    Fxa, Fya, Fza = _beam_forces(x, y, z, ux,        uy,       cz, ua*p.U0_aux_max, w_aux, zR_a)

    Fx = Fx1 + Fx2 + Fxa
    Fy = Fy1 + Fy2 + Fya
    Fz = Fz1 + Fz2 + Fza - g_dimless   # gravity pulls in -z direction

    return Fx, Fy, Fz
end

end
