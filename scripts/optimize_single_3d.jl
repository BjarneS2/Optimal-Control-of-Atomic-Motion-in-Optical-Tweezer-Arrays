using atomove
using HDF5
using Dates

# ── Physical constants and unit conversions ────────────────────────────────────
# Space unit: 1 μm,  time unit: 1 μs  →  velocity unit: 1 m/s
# Dimensionless trap depth: U0 = kB * T_tweezer / (m * v0^2)
#   with v0 = w0_SI/t0_SI = 1e-6/1e-6 = 1 m/s and m = 2.20695e-25 kg

consts = default_constants3d()

kB     = consts.kB
m      = consts.m
w0_SI  = consts.w0_um * 1e-6    # 1e-6 m
t0_SI  = consts.t0_us * 1e-6    # 1e-6 s
v0     = w0_SI / t0_SI           # 1 m/s
E0     = m * v0^2                # energy unit [J]

T_tweezer = 287e-6   # [K]
T_atom    = 40e-6    # [K]

U0_static  = kB * T_tweezer / E0   # dimensionless static trap depth
U0_aux_max = 3.0 * U0_static       # max aux depth = 3× static

# Rayleigh ranges (z_R = π w0² / λ, in dimensionless units = μm)
lambda_static_nm  = consts.wavelength_static_nm
lambda_dynamic_nm = consts.wavelength_dynamic_nm
w_static_um  = 1.2                          # static tweezer waist [μm]
w_dynamic_um = 1.1                          # dynamic tweezer waist [μm]
zR_static  = π * w_static_um^2  / (lambda_static_nm  * 1e-3)  # [μm]
zR_dynamic = π * w_dynamic_um^2 / (lambda_dynamic_nm * 1e-3)  # [μm]

# Transport: along x-axis from 0 to 4.6 μm
dist_um   = 4.6    # [μm] = [dimless]
x_stop    = dist_um
y_stop    = 0.0

params = TweezerParams3D(
    w              = w_static_um,           # static waist [μm dimless]
    w_aux_factor   = w_dynamic_um / w_static_um,
    zR             = zR_static,
    zR_aux         = zR_dynamic,
    x_start        = 0.0,
    y_start        = 0.0,
    x_stop         = x_stop,
    y_stop         = y_stop,
    n              = 501,                   # time nodes (small for single-atom)
    maxT           = 50.0,                  # [μs dimless]
    U0_static      = U0_static,
    U0_aux_max     = U0_aux_max,
    T_tweezer      = T_tweezer,
    T_atom         = T_atom,
    lambda_heat    = 0.0,
    lambda_jitter  = 0.0,
    trap_fraction  = 0.2,
    final_trap_fraction = 0.5,
)

bounds = ControlBounds3D(
    T_min_fraction  = 0.05,
    r_margin_w      = 0.5,
    z_margin        = 3.0 * zR_static,
    v_xy_max        = 5.0,
    v_z_max         = 1.0,
    u_margin_w      = 1.0,
    ua_min          = 0.0,
    ua_max          = 3.0,
    v_u_max_per_w   = 3.0,
    v_ua_max        = 10.0,
)

println("U0_static (dimless)  = $U0_static")
println("U0_aux_max (dimless) = $U0_aux_max")
println("zR_static (μm)       = $zR_static")
println("zR_dynamic (μm)      = $zR_dynamic")
println("g_dimless            = $(consts.g_SI * t0_SI^2 / w0_SI)")

# ── Initial guess ─────────────────────────────────────────────────────────────
# Options (uncomment one):
#   guess = nothing                          # built-in default (linear s, zero velocities)
#   guess = linear_sweep_guess(params)       # constant-velocity tweezer sweep
#   guess = sta_guess(params)                # 5th-order polynomial (zero jerk at endpoints)
  guess = load_guess_from_file("results/control3d_single_2026-04-14_11-49-44.h5", params)
# guess = sta_guess(params)

println("\nRunning single-atom 3D optimal control...")
protocol, status, obj = optimize_controls3d_single(
    params;
    guess                = guess,
    bounds               = bounds,
    max_iter             = 4000,
    hessian_approximation = true,
    print_level          = 5,
    consts               = consts,
    linear_r             = true,
)

println("Objective: $obj  |  Status: $status")

scales = compute_scales3d_full(params; consts = consts)

results_dir = joinpath(@__DIR__, "..", "results")
isdir(results_dir) || mkdir(results_dir)
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
filename  = joinpath(results_dir, "control3d_single_$(timestamp).h5")
println("Saving to $filename")

h5open(filename, "w") do file
    attrs(file)["w"]              = params.w
    attrs(file)["w_aux_factor"]   = params.w_aux_factor
    attrs(file)["zR"]             = params.zR
    attrs(file)["zR_aux"]         = params.zR_aux
    attrs(file)["x_start"]        = params.x_start
    attrs(file)["y_start"]        = params.y_start
    attrs(file)["x_stop"]         = params.x_stop
    attrs(file)["y_stop"]         = params.y_stop
    attrs(file)["n"]              = params.n
    attrs(file)["maxT"]           = params.maxT
    attrs(file)["U0_static"]      = params.U0_static
    attrs(file)["U0_aux_max"]     = params.U0_aux_max
    attrs(file)["T_tweezer"]      = params.T_tweezer
    attrs(file)["T_atom"]         = params.T_atom
    attrs(file)["trap_fraction"]  = params.trap_fraction
    attrs(file)["final_trap_fraction"] = params.final_trap_fraction
    attrs(file)["lambda_heat"]    = params.lambda_heat
    attrs(file)["lambda_jitter"]  = params.lambda_jitter
    attrs(file)["status"]         = status
    attrs(file)["objective"]      = obj
    attrs(file)["w0_um"]          = consts.w0_um
    attrs(file)["t0_us"]          = consts.t0_us
    attrs(file)["v0_m_s"]         = scales.v0
    attrs(file)["E0_J"]           = scales.E0
    attrs(file)["g_dimless"]      = scales.g_dimless

    write(file, "t",    protocol.t)
    write(file, "x",    protocol.x)
    write(file, "y",    protocol.y)
    write(file, "z",    protocol.z)
    write(file, "vx",   protocol.vx)
    write(file, "vy",   protocol.vy)
    write(file, "vz",   protocol.vz)
    write(file, "ux",   protocol.ux)
    write(file, "uy",   protocol.uy)
    write(file, "ua",   protocol.ua)

    write(file, "t_us",    protocol.t)
    write(file, "x_um",    protocol.x)
    write(file, "y_um",    protocol.y)
    write(file, "z_um",    protocol.z)
    write(file, "vx_m_s",  protocol.vx .* scales.v0)
    write(file, "vy_m_s",  protocol.vy .* scales.v0)
    write(file, "vz_m_s",  protocol.vz .* scales.v0)
    write(file, "ux_um",   protocol.ux)
    write(file, "uy_um",   protocol.uy)
end
