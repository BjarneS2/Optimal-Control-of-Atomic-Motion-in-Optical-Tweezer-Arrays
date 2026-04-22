"""
One needs minimally!:
# Trap waists [μm] and Rayleigh ranges [μm]
const w_TiSaph  = 1.2
const w_Mephisto  = 1.1
const λ_TiSaph = 1064.0
const λ_Mephisto = 934.0
const zR_TiSaph = π * w_A^2 / λ_Tisaph         # [μm]
const zR_Mephisto = π * w_B^2 / λ_Mephisto         # [μm]
const U0_Offset = 287.0 uK
const U_TiSaph = U0_Offset * UTiSaph
const U_Mephisto = U0_Offset * UMephisto
where UTiSaph and UMephisto are the trap depths in given by the trap ratio. E.g. TiSaph is 1.0 and Mephisto is 0.5.
Depth ratio to be checked; some number
const T_atoms = 30 uK -- always needs to be lower than the trap depth in both traps!

optional: 3 parameters for the offset in z direction, x and y direction.

const N = 5000 number of atoms to be sampled and evaluated
const Seed = 73 seed for sampling
const Trap fraction = 0.x so that atoms are always total energy < Trap fraction * max_Potential_of_trap <-- the trap that is turned on at that point in time (decided seen by the trigger signal)

Time offset intervals, to be checked; List of some e.g. +-5us interval.

"""

using Random, Printf, HDF5

# ── Data paths (hardcoded, never change) ─────────────────────────────────────
const PATH_TISAPH_OFF   = joinpath(@__DIR__, "data", "TiSaph")
const PATH_MEPHISTO_ON  = joinpath(@__DIR__, "data", "TiSaph1")
const PATH_TISAPH_ON    = joinpath(@__DIR__, "data", "TiSaph1")
const PATH_MEPHISTO_OFF = joinpath(@__DIR__, "data", "TiSaph")
const INITIAL_TRAP = :TiSaph

# ── Time cutoffs and edge modes per signal ────────────────────────────────────
# TiSaph dataset has both rising and falling edges; cut at 3 μs to isolate fall.
const TISAPH_OFF_CUTOFF  = 3.0;   const TISAPH_OFF_EDGE  = "fall"
const TISAPH_ON_CUTOFF   = Inf;   const TISAPH_ON_EDGE   = "rise"
const MEPHISTO_OFF_CUTOFF = Inf;  const MEPHISTO_OFF_EDGE = "fall"
const MEPHISTO_ON_CUTOFF  = 3.0;  const MEPHISTO_ON_EDGE  = "rise"

# ── Physical constants ────────────────────────────────────────────────────────
const kB_SI = 1.380649e-23
const m_Cs  = 2.20695e-25
const g_SI  = 9.81

# ── Trap parameters (hardcoded) ───────────────────────────────────────────────
const w_TiSaph      = 1.2
const w_Mephisto    = 1.1
const λ_TiSaph      = 1.064
const λ_Mephisto    = 0.934
const zR_TiSaph     = π * w_TiSaph^2  / λ_TiSaph
const zR_Mephisto   = π * w_Mephisto^2 / λ_Mephisto
const U0_Offset_K   = 287e-6
const U_TiSaph_K    = U0_Offset_K * 1.0
const U_Mephisto_K  = U0_Offset_K * 1.0
const T_atom_K      = 30e-6
const N_ATOMS       = 5000
const SEED          = 73
const TRAP_FRAC     = 0.75

# ── Unit system: 1 μm space, 1 μs time, v₀ = 1 m/s ──────────────────────────
const E0            = m_Cs * 1.0^2
const U_TiSaph_dl   = kB_SI * U_TiSaph_K  / E0
const U_Mephisto_dl = kB_SI * U_Mephisto_K / E0
const g_dl          = g_SI * (1e-6)^2 / 1e-6

# ── Progress bar ──────────────────────────────────────────────────────────────
function _progress(done::Int, total::Int, width::Int = 40)
    frac   = done / total
    filled = round(Int, frac * width)
    bar    = "█"^filled * "░"^(width - filled)
    pct    = round(Int, 100 * frac)
    print("\r  [$bar] $pct%  ($done/$total)")
    done == total && println()
    flush(stdout)
end

# ── Gaussian beam physics ─────────────────────────────────────────────────────
@inline function _beam_factor(x, y, z, cx, cy, cz, w, zR)
    Xi   = z - cz
    rho2 = (x - cx)^2 + (y - cy)^2
    wXi2 = w^2 * (1.0 + (Xi / zR)^2)
    f    = (w^2 / wXi2) * exp(-2.0 * rho2 / wXi2)
    return f, rho2, Xi, wXi2
end

@inline function _beam_forces(x, y, z, cx, cy, cz, U0, w, zR)
    Xi    = z - cz
    dx    = x - cx
    dy    = y - cy
    rho2  = dx^2 + dy^2
    wXi2  = w^2 * (1.0 + (Xi / zR)^2)
    alpha = w^2 / wXi2
    f     = alpha * exp(-2.0 * rho2 / wXi2)
    Fx    = -4.0 * U0 * dx / wXi2 * f
    Fy    = -4.0 * U0 * dy / wXi2 * f
    Fz    =  U0 * f * alpha * (Xi / zR^2) * (4.0 * rho2 / wXi2 - 2.0)
    return Fx, Fy, Fz
end

struct TrapGeom
    cx::Float64; cy::Float64; cz::Float64
    w::Float64;  zR::Float64
end

# ── CSV loading ───────────────────────────────────────────────────────────────
function load(filename::AbstractString)
    lines = readlines(filename)
    data_lines = filter(l -> !isempty(strip(l)), lines[4:end])
    n = length(data_lines)
    time = zeros(n); aom = zeros(n); trig = fill(Inf, n)
    for (i, line) in enumerate(data_lines)
        cols = split(line, ';')
        length(cols) < 3 && continue
        time[i] = parse(Float64, replace(strip(cols[1]), ',' => '.'))
        aom[i]  = parse(Float64, replace(strip(cols[2]), ',' => '.'))
        s3 = strip(cols[3])
        if !isempty(s3) && s3 != "∞"
            try; trig[i] = parse(Float64, replace(s3, ',' => '.')); catch; end
        end
    end
    return time, aom, trig
end

function load_folder(folderpath::AbstractString)
    files = sort(filter(f -> endswith(f, ".csv"), readdir(folderpath; join=true)))
    isempty(files) && error("No CSV files in $folderpath")
    return [load(f) for f in files]
end

function flatten(data)
    time = vcat([d[1] for d in data]...)
    aom  = vcat([d[2] for d in data]...)
    trig = vcat([d[3] for d in data]...)
    idx  = sortperm(time)
    return time[idx], aom[idx], trig[idx]
end

# ── Signal processing ─────────────────────────────────────────────────────────
function _quantile(v::AbstractVector, p::Float64)
    s = sort(filter(isfinite, Float64.(v)))
    isempty(s) && return 0.0
    n = length(s)
    i = 1.0 + p * (n - 1)
    lo = max(1, floor(Int, i)); hi = min(n, ceil(Int, i))
    lo == hi && return s[lo]
    return s[lo] + (i - lo) * (s[hi] - s[lo])
end

function normalize(signal::AbstractVector)
    s = copy(Float64.(signal))
    fin = isfinite.(s)
    maxv = maximum(s[fin])
    s[.!fin] .= maxv
    minv = minimum(s)
    d = maxv - minv
    d == 0.0 && return zeros(length(s))
    return (s .- minv) ./ d
end

normalize_all(data) = [(t, normalize(aom), normalize(trig)) for (t, aom, trig) in data]

function clamp_saturated(signal::AbstractVector; upper=0.95, lower=0.05)
    s = copy(Float64.(signal))
    q_up = _quantile(s, upper)
    q_lo = _quantile(s, lower)
    s[s .>= q_up] .= 0.95
    s[s .<= q_lo] .= 0.0
    return s
end

function bin_signal(time::AbstractVector, signal::AbstractVector, n_bins::Int=500)
    t0, t1 = Float64(time[1]), Float64(time[end])
    step = (t1 - t0) / n_bins
    sums = zeros(n_bins); counts = zeros(Int, n_bins)
    for (i, t) in enumerate(time)
        b = clamp(floor(Int, (t - t0) / step) + 1, 1, n_bins)
        sums[b] += signal[i]; counts[b] += 1
    end
    centers = [t0 + (b - 0.5) * step for b in 1:n_bins]
    means   = [counts[b] > 0 ? sums[b] / counts[b] : NaN for b in 1:n_bins]
    valid   = findall(isfinite, means)
    for b in 1:n_bins
        isfinite(means[b]) && continue
        means[b] = means[valid[argmin(abs.(valid .- b))]]
    end
    return centers, means
end

function interp_data(time::AbstractVector, signal::AbstractVector, t_query::AbstractVector)
    out = zeros(length(t_query))
    for (i, t) in enumerate(t_query)
        if t <= time[1]
            out[i] = signal[1]
        elseif t >= time[end]
            out[i] = signal[end]
        else
            ir = searchsortedfirst(time, t)
            il = ir - 1
            out[i] = signal[il] + (signal[ir] - signal[il]) * (t - time[il]) / (time[ir] - time[il])
        end
    end
    return out
end

function interpolate_signal(time, signal, t_query; edge="rise", upper=0.95, lower=0.05, n_bins=500)
    bt, bs  = bin_signal(time, signal, n_bins)
    clamped = clamp_saturated(bs; upper=upper, lower=lower)
    out = interp_data(bt, clamped, collect(t_query))
    left_fill  = edge == "rise" ? 0.0 : 1.0
    right_fill = edge == "rise" ? 1.0 : 0.0
    tq = collect(t_query)
    out[tq .< bt[1]]   .= left_fill
    out[tq .> bt[end]] .= right_fill
    return out
end

function moving_average(data::AbstractVector, window::Int)
    window < 1 && error("window must be ≥ 1")
    n = length(data)
    out = zeros(n - window + 1)
    for i in eachindex(out)
        out[i] = sum(data[i:i+window-1]) / window
    end
    return out
end

function shift_time_series(time::AbstractVector, dt::Float64; mode="p")
    mode == "p" && return time .+ dt
    mode == "m" && return time .- dt
    error("mode must be 'p' or 'm'")
end

function trig_time(time::AbstractVector, trigger::AbstractVector; threshold=0.3, mode="rise")
    if mode == "rise"
        crossings = findall(i -> trigger[i] < threshold && trigger[i+1] >= threshold, 1:length(trigger)-1)
    elseif mode == "fall"
        crossings = findall(i -> trigger[i] >= threshold && trigger[i+1] < threshold, 1:length(trigger)-1)
    else
        error("mode must be 'rise' or 'fall'")
    end
    isempty(crossings) && error("No trigger crossing found (mode='$mode', threshold=$threshold).")
    return time[crossings[1]]
end

function find_offset(time, signal, trigger; threshold=0.3, tmode="rise", smode="rise")
    return trig_time(time, signal; threshold=threshold, mode=smode) -
           trig_time(time, trigger; threshold=threshold, mode=tmode)
end

exp_func(t, A, tau, C) = A * (1.0 - exp(-t / tau)) + C

function fit_exponential(time::AbstractVector, signal::AbstractVector; guess::String="rise")
    C = signal[1]
    A = signal[end] - signal[1]
    if guess == "rise" && abs(A) < 1e-8; A = -1.0; end
    if guess == "fall" && abs(A) < 1e-8; A =  1.0; end
    lo, hi = 1.0, 500.0
    φ = (√5 - 1) / 2
    for _ in 1:80
        m1 = hi - φ*(hi - lo); m2 = lo + φ*(hi - lo)
        r1 = sum((exp_func.(time, A, m1, C) .- signal).^2)
        r2 = sum((exp_func.(time, A, m2, C) .- signal).^2)
        r1 < r2 ? (hi = m2) : (lo = m1)
    end
    return (A, (lo + hi) / 2, C)
end

function prepare(fpath::AbstractString; max_time::Float64=Inf, edge::String="rise")
    ntime, naom, ntrig = flatten(load_folder(fpath))
    if isfinite(max_time)
        mask  = ntime .<= max_time
        ntime = ntime[mask]; naom = naom[mask]; ntrig = ntrig[mask]
    end
    naom  = normalize(naom)
    ntrig = normalize(ntrig)
    crossings = findall(i -> ntrig[i] < 0.15 && ntrig[i+1] >= 0.15, 1:length(ntrig)-1)
    isempty(crossings) && error("No trigger crossing in $fpath")
    k = crossings[1]
    ntime_tc, naom_tc, ntrig_tc = ntime[k:end], naom[k:end], ntrig[k:end]
    popt = fit_exponential(ntime_tc, naom_tc; guess=edge)
    @printf("  %-28s  τ=%7.2f μs  A=%+.3f  C=%.3f\n", basename(fpath), popt[2], popt[1], popt[3])
    t_query    = collect(LinRange(ntime[1],    ntime[end],    5000))
    t_query_tc = collect(LinRange(ntime_tc[1], ntime_tc[end], 1000))
    return (ntime,    naom,    ntrig,
            t_query,    interp_data(ntime,    naom,    t_query),    interp_data(ntime,    ntrig,    t_query),
            t_query_tc, interp_data(ntime_tc, naom_tc, t_query_tc), interp_data(ntime_tc, ntrig_tc, t_query_tc),
            popt, edge)
end

# ── Thermal sampling — full Boltzmann (Grimm, Weidemüller, Ovchinnikov 2000) ─
# Grimm et al. Eq. 42: U(ρ,ξ) = -U₀·(w₀/w(ξ))²·exp(-2ρ²/w(ξ)²)
# Thermal distribution f(r,v) ∝ exp(-(½mv²+U(r))/(kBT)).
# Positions sampled with 2× harmonic-width proposal so anharmonic tails are covered;
# acceptance condition uses the exact non-parabolic potential.
@inline function _beam_u(x, y, z, w, zR, U0_dl)
    wXi2 = w^2 * (1.0 + (z / zR)^2)
    return -U0_dl * (w^2 / wXi2) * exp(-2.0 * (x^2 + y^2) / wXi2)
end

function sample_thermal(w::Float64, zR::Float64, U0_dl::Float64;
                        rng::AbstractRNG=Random.GLOBAL_RNG, max_attempts::Int=5000)
    U0_J  = U0_dl * E0
    ωr    = sqrt(4.0 * U0_J / (m_Cs * (w  * 1e-6)^2))
    ωz    = sqrt(2.0 * U0_J / (m_Cs * (zR * 1e-6)^2))
    σr    = 2.0 * sqrt(kB_SI * T_atom_K / (m_Cs * ωr^2)) / 1e-6
    σz    = 2.0 * sqrt(kB_SI * T_atom_K / (m_Cs * ωz^2)) / 1e-6
    σv    = sqrt(kB_SI * T_atom_K / m_Cs)
    thr   = TRAP_FRAC * (-U0_dl)
    for _ in 1:max_attempts
        x  = randn(rng) * σr;  y  = randn(rng) * σr;  z  = randn(rng) * σz
        vx = randn(rng) * σv;  vy = randn(rng) * σv;  vz = randn(rng) * σv
        0.5*(vx^2+vy^2+vz^2) + _beam_u(x, y, z, w, zR, U0_dl) < thr &&
            return (x, y, z, vx, vy, vz)
    end
    error("sample_thermal: max_attempts ($max_attempts) exceeded")
end

# ── Forward dynamics — 2-beam RK4 ────────────────────────────────────────────
# Both beams co-located at origin.  αA, αB ∈ [0,1] are instantaneous amplitudes.
@inline function _accel2(x, y, z, αA, αB)
    Fx1, Fy1, Fz1 = _beam_forces(x, y, z, 0.0, 0.0, 0.0, αA * U_TiSaph_dl,   w_TiSaph,   zR_TiSaph)
    Fx2, Fy2, Fz2 = _beam_forces(x, y, z, 0.0, 0.0, 0.0, αB * U_Mephisto_dl, w_Mephisto, zR_Mephisto)
    return Fx1 + Fx2, Fy1 + Fy2, Fz1 + Fz2 - g_dl
end

function simulate_atom(x0, y0, z0, vx0, vy0, vz0,
                       t_grid::AbstractVector,
                       sig_A::AbstractVector,
                       sig_B::AbstractVector)
    x, y, z    = x0, y0, z0
    vx, vy, vz = vx0, vy0, vz0
    n = length(t_grid)
    for j in 1:n-1
        dt            = t_grid[j+1] - t_grid[j]
        αA1, αB1      = sig_A[j],   sig_B[j]
        αA2, αB2      = sig_A[j+1], sig_B[j+1]
        αAm, αBm      = 0.5*(αA1+αA2), 0.5*(αB1+αB2)

        k1qx,k1qy,k1qz    = vx, vy, vz
        k1px,k1py,k1pz    = _accel2(x, y, z, αA1, αB1)

        x2  = x  + 0.5*dt*k1qx;  y2  = y  + 0.5*dt*k1qy;  z2  = z  + 0.5*dt*k1qz
        vx2 = vx + 0.5*dt*k1px;  vy2 = vy + 0.5*dt*k1py;  vz2 = vz + 0.5*dt*k1pz
        k2qx,k2qy,k2qz    = vx2, vy2, vz2
        k2px,k2py,k2pz    = _accel2(x2, y2, z2, αAm, αBm)

        x3  = x  + 0.5*dt*k2qx;  y3  = y  + 0.5*dt*k2qy;  z3  = z  + 0.5*dt*k2qz
        vx3 = vx + 0.5*dt*k2px;  vy3 = vy + 0.5*dt*k2py;  vz3 = vz + 0.5*dt*k2pz
        k3qx,k3qy,k3qz    = vx3, vy3, vz3
        k3px,k3py,k3pz    = _accel2(x3, y3, z3, αAm, αBm)

        x4  = x  + dt*k3qx;  y4  = y  + dt*k3qy;  z4  = z  + dt*k3qz
        vx4 = vx + dt*k3px;  vy4 = vy + dt*k3py;  vz4 = vz + dt*k3pz
        k4qx,k4qy,k4qz    = vx4, vy4, vz4
        k4px,k4py,k4pz    = _accel2(x4, y4, z4, αA2, αB2)

        x  += dt/6*(k1qx + 2k2qx + 2k3qx + k4qx)
        y  += dt/6*(k1qy + 2k2qy + 2k3qy + k4qy)
        z  += dt/6*(k1qz + 2k2qz + 2k3qz + k4qz)
        vx += dt/6*(k1px + 2k2px + 2k3px + k4px)
        vy += dt/6*(k1py + 2k2py + 2k3py + k4py)
        vz += dt/6*(k1pz + 2k2pz + 2k3pz + k4pz)
    end
    return x, y, z, vx, vy, vz
end

# ── Survival criterion ────────────────────────────────────────────────────────
function is_trapped_in(x, y, z, vx, vy, vz, w, zR, U0_dl)
    return 0.5*(vx^2+vy^2+vz^2) + _beam_u(x, y, z, w, zR, U0_dl) < TRAP_FRAC * (-U0_dl)
end

# ── Score function ────────────────────────────────────────────────────────────
function score_transfer(dt::Float64,
                        t_A::AbstractVector, sig_A::AbstractVector,
                        t_B::AbstractVector, sig_B::AbstractVector,
                        t_sim::AbstractVector;
                        rng::AbstractRNG=Random.GLOBAL_RNG,
                        target::Symbol=:Mephisto)
    sA = interp_data(t_A, sig_A, t_sim)
    sB = interp_data(t_B, sig_B, t_sim .- dt)

    w_init, zR_init, U_init =
        INITIAL_TRAP == :TiSaph ? (w_TiSaph,   zR_TiSaph,   U_TiSaph_dl) :
                                   (w_Mephisto, zR_Mephisto, U_Mephisto_dl)
    w_tgt, zR_tgt, U_tgt =
        target == :Mephisto ? (w_Mephisto, zR_Mephisto, U_Mephisto_dl) :
                               (w_TiSaph,   zR_TiSaph,   U_TiSaph_dl)

    survived = 0
    for i in 1:N_ATOMS
        x0,y0,z0,vx0,vy0,vz0 = sample_thermal(w_init, zR_init, U_init; rng=rng)
        xf,yf,zf,vxf,vyf,vzf = simulate_atom(x0,y0,z0,vx0,vy0,vz0, t_sim, sA, sB)
        is_trapped_in(xf,yf,zf,vxf,vyf,vzf, w_tgt, zR_tgt, U_tgt) && (survived += 1)
        i % 500 == 0 && _progress(i, N_ATOMS)
    end
    _progress(N_ATOMS, N_ATOMS)
    return survived / N_ATOMS
end

# ── HDF5 output ───────────────────────────────────────────────────────────────
function save_hdf5(path::AbstractString,
                   best_dt::Float64, best_score::Float64,
                   dts::AbstractVector, scores::AbstractVector,
                   t_sim::AbstractVector,
                   sig_A::AbstractVector, sig_B::AbstractVector,
                   rng::AbstractRNG)
    n = length(t_sim)
    xs  = zeros(N_ATOMS, n); ys  = zeros(N_ATOMS, n); zs  = zeros(N_ATOMS, n)
    vxs = zeros(N_ATOMS, n); vys = zeros(N_ATOMS, n); vzs = zeros(N_ATOMS, n)
    trapped = zeros(Bool, N_ATOMS)

    w_init, zR_init, U_init = w_TiSaph, zR_TiSaph, U_TiSaph_dl
    w_tgt,  zR_tgt,  U_tgt  = w_Mephisto, zR_Mephisto, U_Mephisto_dl

    println("\nRecording trajectories for all $N_ATOMS atoms at optimal dt...")
    for i in 1:N_ATOMS
        x0,y0,z0,vx0,vy0,vz0 = sample_thermal(w_init, zR_init, U_init; rng=rng)
        x, y, z    = x0, y0, z0
        vx, vy, vz = vx0, vy0, vz0
        xs[i,1]=x; ys[i,1]=y; zs[i,1]=z; vxs[i,1]=vx; vys[i,1]=vy; vzs[i,1]=vz
        for j in 1:n-1
            dt_s          = t_sim[j+1] - t_sim[j]
            αA1,αB1       = sig_A[j],   sig_B[j]
            αA2,αB2       = sig_A[j+1], sig_B[j+1]
            αAm,αBm       = 0.5*(αA1+αA2), 0.5*(αB1+αB2)
            k1qx,k1qy,k1qz = vx,vy,vz
            k1px,k1py,k1pz = _accel2(x,y,z,αA1,αB1)
            x2=x+0.5*dt_s*k1qx; y2=y+0.5*dt_s*k1qy; z2=z+0.5*dt_s*k1qz
            vx2=vx+0.5*dt_s*k1px; vy2=vy+0.5*dt_s*k1py; vz2=vz+0.5*dt_s*k1pz
            k2qx,k2qy,k2qz = vx2,vy2,vz2
            k2px,k2py,k2pz = _accel2(x2,y2,z2,αAm,αBm)
            x3=x+0.5*dt_s*k2qx; y3=y+0.5*dt_s*k2qy; z3=z+0.5*dt_s*k2qz
            vx3=vx+0.5*dt_s*k2px; vy3=vy+0.5*dt_s*k2py; vz3=vz+0.5*dt_s*k2pz
            k3qx,k3qy,k3qz = vx3,vy3,vz3
            k3px,k3py,k3pz = _accel2(x3,y3,z3,αAm,αBm)
            x4=x+dt_s*k3qx; y4=y+dt_s*k3qy; z4=z+dt_s*k3qz
            vx4=vx+dt_s*k3px; vy4=vy+dt_s*k3py; vz4=vz+dt_s*k3pz
            k4qx,k4qy,k4qz = vx4,vy4,vz4
            k4px,k4py,k4pz = _accel2(x4,y4,z4,αA2,αB2)
            x  += dt_s/6*(k1qx+2k2qx+2k3qx+k4qx)
            y  += dt_s/6*(k1qy+2k2qy+2k3qy+k4qy)
            z  += dt_s/6*(k1qz+2k2qz+2k3qz+k4qz)
            vx += dt_s/6*(k1px+2k2px+2k3px+k4px)
            vy += dt_s/6*(k1py+2k2py+2k3py+k4py)
            vz += dt_s/6*(k1pz+2k2pz+2k3pz+k4pz)
            xs[i,j+1]=x; ys[i,j+1]=y; zs[i,j+1]=z
            vxs[i,j+1]=vx; vys[i,j+1]=vy; vzs[i,j+1]=vz
        end
        trapped[i] = is_trapped_in(x,y,z,vx,vy,vz,w_tgt,zR_tgt,U_tgt)
        i % 500 == 0 && _progress(i, N_ATOMS)
    end
    _progress(N_ATOMS, N_ATOMS)

    h5open(path, "w") do f
        # constants
        g = create_group(f, "constants")
        for (k,v) in [("w_TiSaph_um",w_TiSaph), ("w_Mephisto_um",w_Mephisto),
                      ("lambda_TiSaph_um",λ_TiSaph), ("lambda_Mephisto_um",λ_Mephisto),
                      ("zR_TiSaph_um",zR_TiSaph), ("zR_Mephisto_um",zR_Mephisto),
                      ("U_TiSaph_uK",U_TiSaph_K*1e6), ("U_Mephisto_uK",U_Mephisto_K*1e6),
                      ("T_atom_uK",T_atom_K*1e6), ("U0_Offset_uK",U0_Offset_K*1e6),
                      ("trap_frac",TRAP_FRAC), ("N_atoms",Float64(N_ATOMS)),
                      ("seed",Float64(SEED)), ("kB_SI",kB_SI),
                      ("m_Cs_kg",m_Cs), ("g_SI",g_SI),
                      ("E0_J",E0), ("U_TiSaph_dl",U_TiSaph_dl),
                      ("U_Mephisto_dl",U_Mephisto_dl), ("g_dl",g_dl)]
            g[k] = v
        end
        # control protocol
        pg = create_group(f, "protocol")
        pg["t_us"]  = collect(Float64, t_sim)
        pg["sig_A"] = collect(Float64, sig_A)
        pg["sig_B"] = collect(Float64, sig_B)
        pg["best_dt_us"] = best_dt
        # grid search results
        rg = create_group(f, "grid_search")
        rg["dts_us"]  = collect(Float64, dts)
        rg["scores"]  = collect(Float64, scores)
        rg["best_score"] = best_score
        # trajectories  shape: (N_atoms, n_timesteps)
        tg = create_group(f, "trajectories")
        tg["x_um"]  = xs;  tg["y_um"]  = ys;  tg["z_um"]  = zs
        tg["vx_ms"] = vxs; tg["vy_ms"] = vys; tg["vz_ms"] = vzs
        tg["trapped"] = Int8.(trapped)
    end
    @printf("Saved results → %s\n", path)
end

# ── Single-transfer optimisation ─────────────────────────────────────────────
function optimize_single(t_A::AbstractVector, sig_A::AbstractVector,
                          t_B::AbstractVector, sig_B::AbstractVector;
                          n_grid::Int=40,
                          shift_range::Tuple{Float64,Float64}=(-50.0, 50.0),
                          t_sim_start::Float64=-50.0,
                          t_sim_end::Float64=100.0,
                          rng::AbstractRNG=Random.GLOBAL_RNG)
    t_sim  = collect(range(t_sim_start, t_sim_end; step=0.1))
    dts    = collect(LinRange(shift_range[1], shift_range[2], n_grid))
    scores = zeros(n_grid)
    println("Single-transfer grid search ($(n_grid) points):")
    for (i, dt) in enumerate(dts)
        scores[i] = score_transfer(Float64(dt), t_A, sig_A, t_B, sig_B, t_sim; rng=rng)
        @printf("  dt = %+7.1f μs  →  survival = %.4f\n", dt, scores[i])
    end
    bi = argmax(scores)
    best_dt = dts[bi]
    lo = bi > 1      ? dts[bi-1] : shift_range[1]
    hi = bi < n_grid ? dts[bi+1] : shift_range[2]
    φ  = (√5 - 1) / 2
    println("\nRefining with golden-section search...")
    for _ in 1:25
        m1 = hi - φ*(hi - lo); m2 = lo + φ*(hi - lo)
        s1 = score_transfer(m1, t_A, sig_A, t_B, sig_B, t_sim; rng=rng)
        s2 = score_transfer(m2, t_A, sig_A, t_B, sig_B, t_sim; rng=rng)
        if s1 >= s2; hi = m2; best_dt = m1
        else;         lo = m1; best_dt = m2
        end
    end
    best_score = score_transfer(best_dt, t_A, sig_A, t_B, sig_B, t_sim; rng=rng)
    @printf("\n▶ Single-transfer optimum:  dt = %+.3f μs   survival = %.4f\n\n", best_dt, best_score)
    return best_dt, best_score, dts, scores
end

# ── Double-transfer optimisation ─────────────────────────────────────────────
# Transfer sequence: TiSaph → Mephisto (at t≈0+dt1) → TiSaph (at t≈dead_time+dt2).
# sig_AO / sig_BO / sig_BF / sig_AF are the four AOM signals (all zeroed at trigger).
function optimize_double(t_AO, sig_AO, t_BO, sig_BO,
                          t_BF, sig_BF, t_AF, sig_AF;
                          n_grid::Int=20,
                          shift_range::Tuple{Float64,Float64}=(-50.0, 50.0),
                          dead_time::Float64=100.0,
                          rng::AbstractRNG=Random.GLOBAL_RNG)
    t_end = 50.0 + dead_time + 50.0
    t_sim = collect(range(-50.0, t_end; step=0.1))
    dts   = collect(LinRange(shift_range[1], shift_range[2], n_grid))

    best_score = -Inf; best_dt1 = 0.0; best_dt2 = 0.0
    w_tgt, zR_tgt, U_tgt = w_TiSaph, zR_TiSaph, U_TiSaph_dl

    println("Double-transfer grid search ($(n_grid)×$(n_grid)):")
    for dt1 in dts, dt2 in dts
        sA_off = interp_data(t_AO, sig_AO, t_sim)
        sB_on  = interp_data(t_BO, sig_BO, t_sim .- dt1)
        sB_off = interp_data(t_BF, sig_BF, t_sim .- (dead_time + dt1))
        sA_on  = interp_data(t_AF, sig_AF, t_sim .- (dead_time + dt1 + dt2))

        sA_tot = clamp.(sA_off .+ sA_on,  0.0, 1.0)
        sB_tot = clamp.(sB_on  .+ sB_off, 0.0, 1.0)

        survived = 0
        for _ in 1:N_ATOMS
            x0,y0,z0,vx0,vy0,vz0 = sample_thermal(w_TiSaph, zR_TiSaph, U_TiSaph_dl; rng=rng)
            xf,yf,zf,vxf,vyf,vzf = simulate_atom(x0,y0,z0,vx0,vy0,vz0, t_sim, sA_tot, sB_tot)
            is_trapped_in(xf,yf,zf,vxf,vyf,vzf, w_tgt, zR_tgt, U_tgt) && (survived += 1)
        end
        sc = survived / N_ATOMS
        sc > best_score && (best_score = sc; best_dt1 = dt1; best_dt2 = dt2)
        @printf("  dt1=%+6.1f  dt2=%+6.1f  survival=%.4f\n", dt1, dt2, sc)
    end
    @printf("\n▶ Double-transfer optimum:  dt1=%+.3f μs  dt2=%+.3f μs  survival=%.4f\n\n",
            best_dt1, best_dt2, best_score)
    return best_dt1, best_dt2, best_score
end

# ── Main ──────────────────────────────────────────────────────────────────────
function main()
    println("═"^60)
    println("Pouring simulation — thermal atom transfer optimisation")
    println("═"^60)

    println("\nLoading AOM datasets...")
    (_, _, _, tq_AO, ia_AO, _,
     tqtc_AO, _, ittc_AO, _, edge_AO) = prepare(PATH_TISAPH_OFF; max_time=TISAPH_OFF_CUTOFF, edge=TISAPH_OFF_EDGE)

    (_, _, _, tq_BO, ia_BO, _,
     tqtc_BO, _, ittc_BO, _, edge_BO) = prepare(PATH_MEPHISTO_ON; max_time=MEPHISTO_ON_CUTOFF, edge=MEPHISTO_ON_EDGE)

    t0_AO = trig_time(tqtc_AO, ittc_AO; threshold=0.3, mode="rise")
    t0_BO = trig_time(tqtc_BO, ittc_BO; threshold=0.3, mode="rise")
    t_AO  = tq_AO .- t0_AO
    t_BO  = tq_BO .- t0_BO

    @printf("\nPhysics:  T_atom=%.0f μK  U_TiSaph=%.0f μK  U_Mephisto=%.0f μK\n",
            T_atom_K*1e6, U_TiSaph_K*1e6, U_Mephisto_K*1e6)
    @printf("          N=%d atoms  seed=%d  trap_frac=%.2f\n\n", N_ATOMS, SEED, TRAP_FRAC)

    rng = MersenneTwister(SEED)
    println("── Single transfer: $(INITIAL_TRAP) → Mephisto ──────────────────────")
    best_dt, best_score, dts, scores =
        optimize_single(t_AO, ia_AO, t_BO, ia_BO; n_grid=40, rng=rng)

    t_sim  = collect(range(-50.0, 100.0; step=0.1))
    sig_A  = interp_data(t_AO, ia_AO, t_sim)
    sig_B  = interp_data(t_BO, ia_BO, t_sim .- best_dt)

    out_path = joinpath(@__DIR__, "results.h5")
    rng2 = MersenneTwister(SEED)
    save_hdf5(out_path, best_dt, best_score, dts, scores, t_sim, sig_A, sig_B, rng2)
end

main()
