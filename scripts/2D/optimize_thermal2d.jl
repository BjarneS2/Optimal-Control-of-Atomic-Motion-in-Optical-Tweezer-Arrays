using atomove
using HDF5
using Dates

consts = default_constants2d()

kB     = consts.kB
m      = consts.m
w0_SI  = consts.w0_um * 1e-6
t0_SI  = consts.t0_us * 1e-6
v0     = w0_SI / t0_SI
E0     = m * v0^2

T_tweezer = 180e-6
T_atom    = 4e-6

U0_static  = kB * T_tweezer / E0
UA_MAX = 280/180
U0_aux_max = UA_MAX * U0_static

lambda_static_nm  = consts.wavelength_static_nm
lambda_dynamic_nm = consts.wavelength_dynamic_nm
w_static_um  = 1.17
w_dynamic_um = 1.17
zR_static  = π * w_static_um^2  / (lambda_static_nm  * 1e-3)
zR_dynamic = π * w_dynamic_um^2 / (lambda_dynamic_nm * 1e-3)

# println(zR_dynamic, "\n", zR_static)
# println(1/zR_dynamic, "\n", 1/zR_static)

dist_um = 4.6
x_stop  = dist_um
z_stop  = 0.0

params = TweezerParams2D(
    w              = w_static_um,
    w_aux_factor   = w_dynamic_um / w_static_um,
    zR             = zR_static,
    zR_aux         = zR_dynamic,
    x_start        = 0.0,
    z_start        = 0.0,
    x_stop         = x_stop,
    z_stop         = z_stop,
    n              = 1001,
    maxT           = 500.0,
    U0_static      = U0_static,
    U0_aux_max     = U0_aux_max,
    T_tweezer      = T_tweezer,
    T_atom         = T_atom,
    lambda_jitter_pos = 10,
    lambda_jitter_ua  = 1,
    trap_fraction     = nothing,
    starting_trap_fraction = 0.2,
    final_trap_fraction    = 0.8,
)

function convert_velocity_to_dimless(
    v_physical::Float64, 
    unit::String
)
    if unit == "m/s"
        v_mps = v_physical
    elseif unit == "um/us" || unit == "μm/μs"
        v_mps = v_physical                      # μm/μs = m/s
    elseif unit == "um/ms" || unit == "μm/ms"
        v_mps = v_physical * 1e-3               # μm/ms = 1e-3 m/s
    else
        error("Unsupported velocity unit: $unit. Use 'm/s', 'um/us', or 'um/ms'")
    end
    return v_mps / v0
end

v_max = convert_velocity_to_dimless(10.0, "m/s") 
println("Velocity limit: $v_max in dimensionless units. Used for both v_s and ux slew rate constraints.")

bounds = ControlBounds2D(
    T_min_fraction  = 0.05,
    r_margin_w      = 0.5,
    z_margin        = 3.0 * zR_static,
    v_x_max         = v_max,
    v_z_max         = v_max,
    u_margin_w      = 1.0,
    ua_min          = 0.0,
    ua_max          = UA_MAX,
    v_u_max_per_w   = v_max,
    v_ua_max        = 26.0 * UA_MAX,
)

# load_from_file = nothing
load_from_file = "C:/dev/GitHub/MasterThesisJulia/results/control_protocol_adv_2026-04-28_10-42-56.h5"

if load_from_file !== nothing
    guess = load_guess_from_1d_file(load_from_file, params)
else
    guess = sta_guess2d(params; frac_time = 0.5)
end

seed      = 37
n_samples = 25

println("U0_static (dimless)  = $U0_static")
println("U0_aux_max (dimless) = $U0_aux_max")
println("zR_static (μm)       = $zR_static")
println("zR_dynamic (μm)      = $zR_dynamic")
println("g_dimless            = $(consts.g_SI * t0_SI^2 / w0_SI)")
println("\nRunning 2D thermal optimal control with $n_samples samples (seed=$seed)...")

result = optimize_controls2d_thermal(
    params;
    guess                 = guess,
    bounds                = bounds,
    n_samples             = n_samples,
    seed                  = seed,
    max_iter              = 8000,
    hessian_approximation = false,
    print_level           = 5,
    consts                = consts,
)

println("Objective: $(result.objective_value)  |  Status: $(result.termination_status)")

scales = compute_scales2d_full(params; consts = consts)

results_dir = joinpath(@__DIR__, "..", "results")
isdir(results_dir) || mkdir(results_dir)
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
filename  = joinpath(results_dir, "control2d_thermal_$(timestamp).h5")
println("Saving to $filename")

proto = result.protocol

h5open(filename, "w") do file
    attrs(file)["w"]              = params.w
    attrs(file)["w_aux_factor"]   = params.w_aux_factor
    attrs(file)["zR"]             = params.zR
    attrs(file)["zR_aux"]         = params.zR_aux
    attrs(file)["x_start"]        = params.x_start
    attrs(file)["z_start"]        = params.z_start
    attrs(file)["x_stop"]         = params.x_stop
    attrs(file)["z_stop"]         = params.z_stop
    attrs(file)["n"]              = params.n
    attrs(file)["maxT"]           = params.maxT
    attrs(file)["U0_static"]      = params.U0_static
    attrs(file)["U0_aux_max"]     = params.U0_aux_max
    attrs(file)["T_tweezer"]      = params.T_tweezer
    attrs(file)["T_atom"]         = params.T_atom
    attrs(file)["trap_fraction"]  = params.trap_fraction === nothing ? -1.0 : params.trap_fraction
    attrs(file)["final_trap_fraction"]    = params.final_trap_fraction
    attrs(file)["starting_trap_fraction"] = params.starting_trap_fraction
    attrs(file)["lambda_jitter_pos"] = params.lambda_jitter_pos
    attrs(file)["lambda_jitter_ua"]  = params.lambda_jitter_ua
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
    write(file, "z",    proto.z)
    write(file, "vx",   proto.vx)
    write(file, "vz",   proto.vz)
    write(file, "ux",   proto.ux)
    write(file, "uz",   proto.uz)
    write(file, "ua",   proto.ua)

    write(file, "t_us",  proto.t)
    write(file, "x_um",  proto.x)
    write(file, "z_um",  proto.z)
    write(file, "ux_um", proto.ux)
    write(file, "uz_um", proto.uz)

    init_x  = [s.x  for s in result.samples]
    init_z  = [s.z  for s in result.samples]
    init_vx = [s.vx for s in result.samples]
    init_vz = [s.vz for s in result.samples]
    write(file, "init_x",  init_x)
    write(file, "init_z",  init_z)
    write(file, "init_vx", init_vx)
    write(file, "init_vz", init_vz)

    write(file, "x_traj",  result.x_traj)
    write(file, "z_traj",  result.z_traj)
    write(file, "vx_traj", result.vx_traj)
    write(file, "vz_traj", result.vz_traj)

    final_energies = zeros(result.n_samples)
    for i in 1:result.n_samples
        xf  = result.x_traj[i, end]
        zf  = result.z_traj[i, end]
        vxf = result.vx_traj[i, end]
        vzf = result.vz_traj[i, end]
        Uf  = potential2d(xf, zf, proto.ux[end], proto.uz[end], proto.ua[end], params)
        KE  = 0.5 * (vxf^2 + vzf^2)
        final_energies[i] = Uf + KE
    end
    write(file, "final_energies", final_energies)
end
