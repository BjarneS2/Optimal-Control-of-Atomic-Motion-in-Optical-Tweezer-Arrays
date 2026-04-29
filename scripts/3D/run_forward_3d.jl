using atomove
using HDF5
using Dates
using Printf
using Random

const SEED             = 101
const DEFAULT_SHOTS    = 1000
const DEFAULT_EXTENSION_FACTOR = 0.5

const sim_consts = default_constants3d()

function latest_protocol_file(results_dir::String)
    candidates = filter(f -> occursin("control3d", basename(f)), readdir(results_dir; join = true))
    isempty(candidates) && error("No 3D control protocol found in $results_dir.")
    sort(candidates; by = f -> stat(f).mtime, rev = true)[1]
end

function load_protocol3d_and_params(path::String)
    h5open(path, "r") do file
        atr = attrs(file)
        get_attr(key, default) = haskey(atr, key) ? atr[key] : default

        params = TweezerParams3D(
            w              = get_attr("w",            1.4),
            w_aux_factor   = get_attr("w_aux_factor", 1.0),
            zR             = get_attr("zR",           8.0),
            zR_aux         = get_attr("zR_aux",       6.0),
            x_start        = get_attr("x_start",      0.0),
            y_start        = get_attr("y_start",      0.0),
            x_stop         = get_attr("x_stop",       4.6),
            y_stop         = get_attr("y_stop",       0.0),
            n              = get_attr("n",             201),
            maxT           = get_attr("maxT",          500.0),
            U0_static      = get_attr("U0_static",    0.01),
            U0_aux_max     = get_attr("U0_aux_max",   0.03),
            T_tweezer      = get_attr("T_tweezer",    287e-6),
            T_atom         = get_attr("T_atom",       40e-6),
            trap_fraction  = get_attr("trap_fraction", 0.5),
            final_trap_fraction = get_attr("final_trap_fraction", 0.5),
            lambda_heat    = get_attr("lambda_heat",  0.0),
            lambda_jitter  = get_attr("lambda_jitter",0.0),
        )

        ctrl = ControlProtocol3D(
            read(file["t"]),
            read(file["x"]),
            read(file["y"]),
            read(file["z"]),
            read(file["vx"]),
            read(file["vy"]),
            read(file["vz"]),
            read(file["ux"]),
            read(file["uy"]),
            read(file["ua"]),
        )
        return ctrl, params
    end
end

function extend_protocol3d(protocol::ControlProtocol3D, factor::Float64)
    factor <= 0.0 && return protocol
    t_total = protocol.t[end] - protocol.t[1]
    t_ext   = factor * t_total
    n_orig  = length(protocol.t)
    avg_dt  = t_total / (n_orig - 1)
    n_new   = max(2, round(Int, t_ext / avg_dt))
    t_tail  = range(protocol.t[end], protocol.t[end] + t_ext; length = n_new + 1)[2:end]
    return ControlProtocol3D(
        vcat(protocol.t, collect(t_tail)),
        vcat(protocol.x,  fill(protocol.x[end],  n_new)),
        vcat(protocol.y,  fill(protocol.y[end],  n_new)),
        vcat(protocol.z,  fill(protocol.z[end],  n_new)),
        vcat(protocol.vx, fill(0.0,              n_new)),
        vcat(protocol.vy, fill(0.0,              n_new)),
        vcat(protocol.vz, fill(0.0,              n_new)),
        vcat(protocol.ux, fill(protocol.ux[end], n_new)),
        vcat(protocol.uy, fill(protocol.uy[end], n_new)),
        vcat(protocol.ua, zeros(Float64, n_new)),
    )
end

function main()
    results_dir = joinpath(@__DIR__, "..", "results")
    protocol_file    = length(ARGS) >= 1 ? ARGS[1] : latest_protocol_file(results_dir)
    shots            = length(ARGS) >= 2 ? parse(Int,     ARGS[2]) : DEFAULT_SHOTS
    extension_factor = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : DEFAULT_EXTENSION_FACTOR
    Random.seed!(SEED)

    println("Loading protocol from $protocol_file")
    protocol, params = load_protocol3d_and_params(protocol_file)

    if extension_factor > 0.0
        println(@sprintf("Extending by %.2fx transport time", extension_factor))
        protocol = extend_protocol3d(protocol, extension_factor)
    end

    scales = compute_scales3d_full(params; consts = sim_consts)
    n = length(protocol.t)

    x_all  = zeros(n, shots);  y_all  = zeros(n, shots);  z_all  = zeros(n, shots)
    vx_all = zeros(n, shots);  vy_all = zeros(n, shots);  vz_all = zeros(n, shots)
    lost   = falses(shots)

    println(@sprintf("Running %d forward shots...", shots))
    for s in 1:shots
        traj = simulate_forward3d(protocol, params; thermal_sample = true, consts = sim_consts)
        x_all[:,s]  = traj.x;   y_all[:,s]  = traj.y;   z_all[:,s]  = traj.z
        vx_all[:,s] = traj.vx;  vy_all[:,s] = traj.vy;  vz_all[:,s] = traj.vz
        lost[s]     = traj.lost
    end

    survival = 1.0 - sum(lost) / shots
    println(@sprintf("Survival rate: %.3f  (%d/%d)", survival, shots - sum(lost), shots))

    isdir(results_dir) || mkdir(results_dir)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename  = joinpath(results_dir, "forward3d_$(timestamp).h5")
    println("Saving to $filename")

    h5open(filename, "w") do file
        attrs(file)["protocol_file"]     = protocol_file
        attrs(file)["timestamp"]         = timestamp
        attrs(file)["shots"]             = shots
        attrs(file)["extension_factor"]  = extension_factor
        attrs(file)["SEED"]              = SEED
        attrs(file)["survival_rate"]     = survival
        attrs(file)["w0_um"]             = sim_consts.w0_um
        attrs(file)["t0_us"]             = sim_consts.t0_us
        attrs(file)["v0_m_s"]            = scales.v0
        attrs(file)["g_dimless"]         = scales.g_dimless
        attrs(file)["w"]                 = params.w
        attrs(file)["x_start"]           = params.x_start
        attrs(file)["y_start"]           = params.y_start
        attrs(file)["x_stop"]            = params.x_stop
        attrs(file)["y_stop"]            = params.y_stop
        attrs(file)["T_atom"]            = params.T_atom
        attrs(file)["T_tweezer"]         = params.T_tweezer

        write(file, "t",    protocol.t)
        write(file, "t_us", protocol.t)
        write(file, "ux",   protocol.ux)
        write(file, "uy",   protocol.uy)
        write(file, "ua",   protocol.ua)

        write(file, "x",    x_all)
        write(file, "y",    y_all)
        write(file, "z",    z_all)
        write(file, "vx",   vx_all)
        write(file, "vy",   vy_all)
        write(file, "vz",   vz_all)
        write(file, "x_um", x_all)
        write(file, "y_um", y_all)
        write(file, "z_um", z_all)
        write(file, "lost", collect(lost))
    end
end

main()
