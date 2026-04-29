module Models2D

export potential2d, forces2d

using ..Types2D

@inline function _beam_factor(x::Real, z::Real,
                               cx::Real, cz::Real,
                               w::Float64, zR::Float64)
    dz  = z - cz
    dx2 = (x - cx)^2
    wz2 = w^2 * (1.0 + (dz / zR)^2)
    f   = (w^2 / wz2) * exp(-2.0 * dx2 / wz2)
    return f, dx2, dz, wz2
end

function potential2d(x::Real, z::Real,
                     ux::Real, uz::Real, ua::Real,
                     p::TweezerParams2D)
    w     = p.w
    w_aux = w * p.w_aux_factor
    zR    = p.zR
    zR_a  = p.zR_aux

    f_st1, _, _, _ = _beam_factor(x, z, p.x_start, p.z_start, w,     zR)
    f_st2, _, _, _ = _beam_factor(x, z, p.x_stop,  p.z_stop,  w,     zR)
    f_aux, _, _, _ = _beam_factor(x, z, ux,         uz,        w_aux, zR_a)

    return -p.U0_static * f_st1 - p.U0_static * f_st2 - ua * p.U0_aux_max * f_aux
end

function potential2d(x::Real, z::Real,
                     ux::Real, uz::Real, ua::Real,
                     p::TweezerParams2D, ::Val{:jump})
    w     = p.w
    w_aux = w * p.w_aux_factor
    zR    = p.zR
    zR_a  = p.zR_aux

    dz1  = z - p.z_start;  dx1sq = (x - p.x_start)^2
    wz1  = w^2 * (1.0 + (dz1/zR)^2);    f1 = (w^2/wz1)*exp(-2.0*dx1sq/wz1)

    dz2  = z - p.z_stop;   dx2sq = (x - p.x_stop)^2
    wz2  = w^2 * (1.0 + (dz2/zR)^2);    f2 = (w^2/wz2)*exp(-2.0*dx2sq/wz2)

    dza  = z - uz;          dxasq = (x - ux)^2
    wza  = w_aux^2 * (1.0 + (dza/zR_a)^2);  fa = (w_aux^2/wza)*exp(-2.0*dxasq/wza)

    return -p.U0_static*f1 - p.U0_static*f2 - ua*p.U0_aux_max*fa
end

@inline function _beam_forces(x::Real, z::Real,
                               cx::Real, cz::Real,
                               U0::Real, w::Float64, zR::Float64)
    dz    = z - cz
    dx    = x - cx
    dx2   = dx^2
    wz2   = w^2 * (1.0 + (dz / zR)^2)
    alpha = w^2 / wz2
    f     = alpha * exp(-2.0 * dx2 / wz2)

    Fx = -4.0 * U0 * dx / wz2 * f
    Fz =  U0  * f * alpha * (dz / zR^2) * (4.0 * dx2 / wz2 - 2.0)
    return Fx, Fz
end

function forces2d(x::Real, z::Real,
                  ux::Real, uz::Real, ua::Real,
                  p::TweezerParams2D, g_dimless::Float64)
    w     = p.w
    w_aux = w * p.w_aux_factor
    zR    = p.zR
    zR_a  = p.zR_aux

    Fx1, Fz1 = _beam_forces(x, z, p.x_start, p.z_start, p.U0_static,      w,     zR)
    Fx2, Fz2 = _beam_forces(x, z, p.x_stop,  p.z_stop,  p.U0_static,      w,     zR)
    Fxa, Fza = _beam_forces(x, z, ux,         uz,        ua*p.U0_aux_max,  w_aux, zR_a)

    Fx = Fx1 + Fx2 + Fxa
    Fz = Fz1 + Fz2 + Fza - g_dimless

    return Fx, Fz
end

end