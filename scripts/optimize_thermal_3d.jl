using atomove
using HDF5
using Dates

consts = default_constants3d()

kB     = consts.kB
m      = consts.m
w0_SI  = consts.w0_um * 1e-6
t0_SI  = consts.t0_us * 1e-6
v0     = w0_SI / t0_SI
E0     = m * v0^2

T_tweezer = 287e-6
T_atom    = 40e-6

U0_static  = kB * T_tweezer / E0
U0_aux_max = 3.0 * U0_static

lambda_static_nm  = consts.wavelength_static_nm
lambda_dynamic_nm = consts.wavelength_dynamic_nm
w_static_um  = 1.4
w_dynamic_um = 1.2
zR_static  = π * w_static_um^2  / (lambda_static_nm  * 1e-3)
zR_dynamic = π * w_dynamic_um^2 / (lambda_dynamic_nm * 1e-3)

dist_um = 4.6
x_stop  = dist_um
y_stop  = 0.0

params = TweezerParams3D(
    w              = w_static_um,
    w_aux_factor   = w_dynamic_um / w_static_um,
    zR             = zR_static,
    zR_aux         = zR_dynamic,
    x_start        = 0.0,
    y_start        = 0.0,
    x_stop         = x_stop,
    y_stop         = y_stop,
    n              = 201,
    maxT           = 500.0,
    U0_static      = U0_static,
    U0_aux_max     = U0_aux_max,
    T_tweezer      = T_tweezer,
    T_atom         = T_atom,
    lambda_heat    = 0.1,
    lambda_jitter  = 0.5,
    trap_fraction  = 0.2,
    final_trap_fraction = 0.2,
)

bounds = ControlBounds3D(
    T_min_fraction  = 0.05,
    r_margin_w      = 0.5,
    z_margin        = 3.0 * zR_static,
    v_xy_max        = 0.5,
    v_z_max         = 0.1,
    u_margin_w      = 1.0,
    ua_min          = 0.0,
    ua_max          = 1.0,
    v_u_max_per_w   = 3.0,
    v_ua_max        = 10.0,
)

seed      = 37
n_samples = 20

println("U0_static (dimless)  = $U0_static")
println("U0_aux_max (dimless) = $U0_aux_max")
println("zR_static (μm)       = $zR_static")
println("zR_dynamic (μm)      = $zR_dynamic")
println("g_dimless            = $(consts.g_SI * t0_SI^2 / w0_SI)")
println("\nRunning 3D thermal optimal control with $n_samples samples (seed=$seed)...")

result = optimize_controls3d_thermal(
    params;
    bounds                = bounds,
    n_samples             = n_samples,
    seed                  = seed,
    max_iter              = 8000,
    hessian_approximation = true,
    print_level           = 5,
    consts                = consts,
)

println("Objective: $(result.objective_value)  |  Status: $(result.termination_status)")

scales = compute_scales3d_full(params; consts = consts)

results_dir = joinpath(@__DIR__, "..", "results")
isdir(results_dir) || mkdir(results_dir)
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
filename  = joinpath(results_dir, "control3d_thermal_$(timestamp).h5")
println("Saving to $filename")

proto = result.protocol

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
    attrs(file)["n_samples"]      = result.n_samples
    attrs(file)["seed"]           = result.seed === nothing ? -1 : result.seed
    attrs(file)["status"]         = result.termination_status
    attrs(file)["objective"]      = result.objective_value
    attrs(file)["w0_um"]          = consts.w0_um
    attrs(file)["t0_us"]          = consts.t0_us
    attrs(file)["v0_m_s"]         = scales.v0
    attrs(file)["E0_J"]           = scales.E0
    attrs(file)["g_dimless"]      = scales.g_dimless

    write(file, "t",    proto.t)
    write(file, "x",    proto.x)
    write(file, "y",    proto.y)
    write(file, "z",    proto.z)
    write(file, "vx",   proto.vx)
    write(file, "vy",   proto.vy)
    write(file, "vz",   proto.vz)
    write(file, "ux",   proto.ux)
    write(file, "uy",   proto.uy)
    write(file, "ua",   proto.ua)

    write(file, "t_us",   proto.t)
    write(file, "x_um",   proto.x)
    write(file, "y_um",   proto.y)
    write(file, "z_um",   proto.z)
    write(file, "ux_um",  proto.ux)
    write(file, "uy_um",  proto.uy)

    init_x  = [s.x  for s in result.samples]
    init_y  = [s.y  for s in result.samples]
    init_z  = [s.z  for s in result.samples]
    init_vx = [s.vx for s in result.samples]
    init_vy = [s.vy for s in result.samples]
    init_vz = [s.vz for s in result.samples]
    write(file, "init_x",  init_x)
    write(file, "init_y",  init_y)
    write(file, "init_z",  init_z)
    write(file, "init_vx", init_vx)
    write(file, "init_vy", init_vy)
    write(file, "init_vz", init_vz)

    write(file, "x_traj",  result.x_traj)
    write(file, "y_traj",  result.y_traj)
    write(file, "z_traj",  result.z_traj)
    write(file, "vx_traj", result.vx_traj)
    write(file, "vy_traj", result.vy_traj)
    write(file, "vz_traj", result.vz_traj)

    final_energies = zeros(result.n_samples)
    for i in 1:result.n_samples
        xf  = result.x_traj[i, end];  yf  = result.y_traj[i, end]
        zf  = result.z_traj[i, end]
        vxf = result.vx_traj[i, end]; vyf = result.vy_traj[i, end]; vzf = result.vz_traj[i, end]
        Uf  = potential3d(xf, yf, zf, proto.ux[end], proto.uy[end], proto.ua[end], params)
        KE  = 0.5 * (vxf^2 + vyf^2 + vzf^2)
        final_energies[i] = Uf + KE
    end
    write(file, "final_energies", final_energies)
    attrs(file)["barrier_height"] = barrier_height3d(params)
end
