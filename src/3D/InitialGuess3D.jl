module InitialGuess3D

using HDF5
using ..Types3D

export linear_sweep_guess, sta_guess, load_guess_from_file

# ── Helpers ────────────────────────────────────────────────────────────────────

function _zero_state_guess(p::TweezerParams3D, T::Float64,
                            s_profile::AbstractVector{Float64})
    n  = p.n
    ex, ey, _ = transport_direction(p)
    dt_vec = fill(T / (n - 1), n - 1)

    ux = p.x_start .+ s_profile .* ex
    uy = p.y_start .+ s_profile .* ey

    # Atom co-moving with tweezer centre, z=0, all velocities zero
    return InitialG3D(
        dt_vec,
        copy(ux), zeros(n), zeros(n),
        zeros(n), zeros(n), zeros(n),
        ux, uy, zeros(n),
    )
end

# ── Linear sweep ───────────────────────────────────────────────────────────────
# Tweezer moves at constant velocity from start to stop.
# T defaults to half of params.maxT if not provided.

function linear_sweep_guess(p::TweezerParams3D; T::Float64 = p.maxT / 2.0)
    n = p.n
    L = transport_length(p)
    s_profile = [L * (j - 1) / (n - 1) for j in 1:n]
    return _zero_state_guess(p, T, s_profile)
end

# ── Shortcut to adiabaticity (STA) ────────────────────────────────────────────
# Uses a 5th-order polynomial s(τ) with τ ∈ [0,1]:
#   s(τ) = L · (10τ³ - 15τ⁴ + 6τ⁵)
# This satisfies s(0)=0, s(1)=L, ṡ(0)=ṡ(1)=0, s̈(0)=s̈(1)=0 — the atom
# starts and ends at rest without acceleration, giving a smooth, minimal-jerk
# trajectory.  The atom trajectory is taken as co-moving with the tweezer.
# T defaults to half of params.maxT if not provided.

function sta_guess(p::TweezerParams3D; T::Float64 = p.maxT / 2.0)
    n = p.n
    L = transport_length(p)

    s_profile = Vector{Float64}(undef, n)
    for j in 1:n
        τ = (j - 1) / (n - 1)
        s_profile[j] = L * τ^3 * (10.0 - 15.0 * τ + 6.0 * τ^2)
    end

    return _zero_state_guess(p, T, s_profile)
end

# ── Load from HDF5 file ────────────────────────────────────────────────────────
# Reads a previously saved result (written by optimize_single_3d.jl or
# optimize_thermal_3d.jl) and resamples all trajectories onto the n-point grid
# of `p` using linear interpolation.  The file must contain datasets:
#   t, x, y, z, vx, vy, vz, ux, uy, ua
# (all in the same dimensionless units used by this codebase).

function load_guess_from_file(path::AbstractString, p::TweezerParams3D)
    t_src, x_src, y_src, z_src, vx_src, vy_src, vz_src, ux_src, uy_src, ua_src =
        h5open(path, "r") do f
            read(f, "t"),
            read(f, "x"), read(f, "y"), read(f, "z"),
            read(f, "vx"), read(f, "vy"), read(f, "vz"),
            read(f, "ux"), read(f, "uy"), read(f, "ua")
        end

    T_src = t_src[end]
    t_dst = [T_src * (j - 1) / (p.n - 1) for j in 1:p.n]

    interp(src, t_s, t_d) = [_linterp(t_s, src, t) for t in t_d]

    x_dst  = interp(x_src,  t_src, t_dst)
    y_dst  = interp(y_src,  t_src, t_dst)
    z_dst  = interp(z_src,  t_src, t_dst)
    vx_dst = interp(vx_src, t_src, t_dst)
    vy_dst = interp(vy_src, t_src, t_dst)
    vz_dst = interp(vz_src, t_src, t_dst)
    ux_dst = interp(ux_src, t_src, t_dst)
    uy_dst = interp(uy_src, t_src, t_dst)
    ua_dst = interp(ua_src, t_src, t_dst)

    dt_vec = fill(T_src / (p.n - 1), p.n - 1)

    return InitialG3D(
        dt_vec,
        x_dst, y_dst, z_dst,
        vx_dst, vy_dst, vz_dst,
        ux_dst, uy_dst, ua_dst,
    )
end

# Linear interpolation on a sorted grid (no external packages required).
function _linterp(t::AbstractVector{Float64}, v::AbstractVector{Float64}, t0::Float64)
    t0 <= t[1]   && return v[1]
    t0 >= t[end] && return v[end]
    # binary search for the interval
    lo, hi = 1, length(t)
    while hi - lo > 1
        mid = (lo + hi) >>> 1
        t[mid] <= t0 ? (lo = mid) : (hi = mid)
    end
    α = (t0 - t[lo]) / (t[hi] - t[lo])
    return v[lo] + α * (v[hi] - v[lo])
end

end
