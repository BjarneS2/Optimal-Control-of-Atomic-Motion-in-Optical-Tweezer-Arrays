module Types2D

export compute_scales2d, ThermalControlResult2D
export PhysicalConstants2D, default_constants2d
export transport_direction2d, transport_length2d
export TweezerParams2D, ControlProtocol2D
export ControlBounds2D, default_bounds2d
export InitialG2D, Trajectory2D

Base.@kwdef struct PhysicalConstants2D
    kB::Float64        = 1.380649e-23
    hbar::Float64      = 1.054571817e-34
    m::Float64         = 2.20695e-25          # Cs-133 [kg]
    g_SI::Float64      = 9.81                 # gravitational acceleration [m/s^2]
    w0_um::Float64     = 1.0                  # reference waist [μm] — also the space unit
    t0_us::Float64     = 1.0                  # time unit [μs]
    wavelength_static_nm::Float64  = 1064.0
    wavelength_dynamic_nm::Float64 = 934.0
end
default_constants2d() = PhysicalConstants2D()

function compute_scales2d(consts::PhysicalConstants2D)
    w0_SI = consts.w0_um * 1e-6              # [m]
    t0_SI = consts.t0_us * 1e-6             # [s]
    v0    = w0_SI / t0_SI                   # [m/s]
    E0    = consts.m * v0^2                 # energy unit [J]
    g_dimless = consts.g_SI * t0_SI^2 / w0_SI
    return (w0_SI = w0_SI, t0_SI = t0_SI, v0 = v0, E0 = E0, g_dimless = g_dimless)
end


Base.@kwdef struct TweezerParams2D
    # geometry in dimensionless units (1 unit = 1 μm, 1 time unit = 1 μs)
    w::Float64                            # radial beam waist [dimless = μm]
    w_aux_factor::Float64  = 1.0          # aux tweezer waist = w * w_aux_factor
    zR::Float64                           # Rayleigh range of static tweezers [dimless]
    zR_aux::Float64                       # Rayleigh range of aux tweezer [dimless]
    # source and target positions in the x-y plane (z=0 is the beam focus plane)
    x_start::Float64       = 0.0
    z_start::Float64       = 0.0
    x_stop::Float64
    z_stop::Float64        = 0.0
    n::Int                                # number of time nodes
    maxT::Float64                         # maximum allowed transport time [dimless = μs]
    # trap depth as dimensionless ratio: U0_dimless = kB*T_tweezer / E0
    U0_static::Float64                    # static tweezer depth [dimless]
    U0_aux_max::Float64                   # maximum aux tweezer depth [dimless]
    T_atom::Float64        = 40e-6        # atom temperature [K]  (for thermal sampling)
    T_tweezer::Float64     = 287e-6       # trap depth temperature [K]
    lambda_jitter_pos::Float64 = 0.0
    lambda_jitter_ua::Float64  = 0.0
    max_sigma_position::Float64 = 2.0
    # some "constraints"
    trap_fraction::Union{Float64,Nothing} = nothing     # will be used for constraints during the move - nothing for deactivation
    final_trap_fraction::Float64 = 0.2                  # will be used for final energy constraints
    starting_trap_fraction::Float64 = 0.2               # will be used for sampling to reject initial cond.
end

Base.@kwdef struct ControlBounds2D
    T_min_fraction::Float64  = 0.05
    r_margin_w::Float64      = 0.5
    z_margin::Float64        = 2.0
    v_x_max::Float64         = 2.0
    v_z_max::Float64         = 2.0
    u_margin_w::Float64      = 1.0
    ua_min::Float64          = 0.0
    ua_max::Float64          = 1.0
    v_u_max_per_w::Float64   = 3.0
    v_ua_max::Float64        = 5.0
end

function transport_direction2d(p::TweezerParams2D)
    dx = p.x_stop - p.x_start
    dz = p.z_stop - p.z_start
    L  = sqrt(dx^2 + dz^2)
    return dx/L, dz/L, L
end

transport_length2d(p::TweezerParams2D) = sqrt((p.x_stop - p.x_start)^2 + (p.z_stop - p.z_start)^2)

default_bounds2d(::TweezerParams2D) = ControlBounds2D()

struct ControlProtocol2D
    t::Vector{Float64}     # time
    x::Vector{Float64}     # x position over the time for single atoms
    z::Vector{Float64}     # z pos over time for single atom
    vx::Vector{Float64}    # velocity over time
    vz::Vector{Float64}    # -||-
    ux::Vector{Float64}    # tweezer position x over t
    uz::Vector{Float64}    # tweezer position z over t
    ua::Vector{Float64}    # aux amplitude (relative to U0_aux_max)
end


struct Trajectory2D # trajectory of a single atom and classification if lost
    t::Vector{Float64}
    x::Vector{Float64}
    z::Vector{Float64}
    vx::Vector{Float64}
    vz::Vector{Float64}
    lost::Bool
end

struct ThermalControlResult2D # basically protocols but for many atoms
    protocol::ControlProtocol2D # average positions and velocities
    samples::Vector{NamedTuple}
    x_traj::Matrix{Float64}    # n_samples, time steps
    z_traj::Matrix{Float64}    # -||-
    vx_traj::Matrix{Float64}
    vz_traj::Matrix{Float64}
    seed::Union{Nothing,Int}   # seed from which initial conds. were sampled from
    n_samples::Int             # number of atoms sampled
    termination_status::String 
    objective_value::Float64   
end

struct InitialG2D
    t::Vector{Float64}
    x::Vector{Float64}
    z::Vector{Float64}
    vx::Vector{Float64}
    vz::Vector{Float64}
    ux::Vector{Float64}
    uz::Vector{Float64}
    ua::Vector{Float64}
end

end