module ThermalSampling2D

using ..Types2D
using ..Models2D: potential2d
using Random

export compute_scales2d_full, sample_initial_conditions2d, is_trapped2d

function compute_scales2d_full(p::TweezerParams2D;
                                consts::PhysicalConstants2D = default_constants2d())
    w0_SI  = consts.w0_um * 1e-6
    t0_SI  = consts.t0_us * 1e-6
    v0     = w0_SI / t0_SI
    E0     = consts.m * v0^2
    U0_SI  = consts.kB * p.T_tweezer

    omega_r = sqrt(4.0 * U0_SI / (consts.m * w0_SI^2))
    sigma_r = sqrt(consts.kB * p.T_atom / (consts.m * omega_r^2))
    sigma_r_dimless = sigma_r / w0_SI

    zR_SI   = p.zR * w0_SI
    omega_z = sqrt(2.0 * U0_SI / (consts.m * zR_SI^2))
    sigma_z = sqrt(consts.kB * p.T_atom / (consts.m * omega_z^2))
    sigma_z_dimless = sigma_z / w0_SI

    sigma_v = sqrt(consts.kB * p.T_atom / consts.m)
    sigma_v_dimless = sigma_v / v0

    g_dimless = consts.g_SI * t0_SI^2 / w0_SI

    return (
        w0_SI           = w0_SI,
        t0_SI           = t0_SI,
        v0              = v0,
        E0              = E0,
        sigma_r_dimless = sigma_r_dimless,
        sigma_z_dimless = sigma_z_dimless,
        sigma_v_dimless = sigma_v_dimless,
        omega_r         = omega_r,
        omega_z         = omega_z,
        g_dimless       = g_dimless,
    )
end

function is_trapped2d(x::Float64, z::Float64,
                       vx::Float64, vz::Float64,
                       ux::Float64, uz::Float64,
                       p::TweezerParams2D)
    U  = potential2d(x, z, ux, uz, 0.0, p)
    KE = 0.5 * (vx^2 + vz^2)
    return (U + KE) < p.starting_trap_fraction * U
end

function sample_initial_conditions2d(
    p::TweezerParams2D;
    consts::PhysicalConstants2D = default_constants2d(),
    check_trapped::Bool   = true,
    max_attempts::Int     = 2000,
)
    scales = compute_scales2d_full(p; consts = consts)

    sr = scales.sigma_r_dimless
    sz = scales.sigma_z_dimless
    sv = scales.sigma_v_dimless

    attempts = 0
    while attempts < max_attempts
        attempts += 1

        dx = randn() * sr
        dz = randn() * sz
        vx = randn() * sv
        vz = randn() * sv

        x = p.x_start + dx
        z = p.z_start + dz

        if !check_trapped || is_trapped2d(x, z, vx, vz, p.x_start, p.z_start, p)
            return (
                x = x, z = z,
                vx = vx, vz = vz,
                scales = scales,
            )
        end
    end
    error("Failed to sample trapped atom after $max_attempts attempts.")
end

end
