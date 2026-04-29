module InitialGuess2D

using HDF5
using ..Types2D

export linear_sweep_guess2d, sta_guess2d, load_guess_from_2d_file, load_guess_from_1d_file

@inline function _linterp(t_old::AbstractVector{Float64}, y_old::AbstractVector{Float64},
                           t_new::AbstractVector{Float64})
    n_old = length(t_old)
    n_new = length(t_new)
    y_new = Vector{Float64}(undef, n_new)
    for k in 1:n_new
        t = t_new[k]
        i = searchsortedlast(t_old, t)
        if i == 0
            y_new[k] = y_old[1]
        elseif i >= n_old
            y_new[k] = y_old[n_old]
        else
            α = (t - t_old[i]) / (t_old[i+1] - t_old[i])
            y_new[k] = y_old[i] * (1 - α) + y_old[i+1] * α
        end
    end
    return y_new
end

function linear_sweep_guess2d(p::TweezerParams2D; frac_time::Float64 = 0.5)
    ex, ez, L = transport_direction2d(p)
    n = p.n
    T = frac_time * p.maxT
    t      = collect(range(0.0, T, length=n))
    fracs  = collect(range(0.0, 1.0, length=n))

    ux = p.x_start .+ fracs .* L .* ex
    uz = p.z_start .+ fracs .* L .* ez
    ua = ones(n)

    x  = copy(ux)
    z  = copy(uz)
    vx = fill(L * ex / T, n)
    vz = fill(L * ez / T, n)

    return InitialGuess2D(t, x, z, vx, vz, ux, uz, ua)
end

function sta_guess2d(p::TweezerParams2D; frac_time::Float64 = 0.5)
    ex, ez, L = transport_direction2d(p)
    n = p.n
    T = frac_time * p.maxT
    t     = collect(range(0.0, T, length=n))
    fracs = collect(range(0.0, 1.0, length=n))

    s_vals   = L .* (10 .* fracs.^3 .- 15 .* fracs.^4 .+ 6  .* fracs.^5)
    ds_dphi  = L .* (30 .* fracs.^2 .- 60 .* fracs.^3 .+ 30 .* fracs.^4)

    ux = p.x_start .+ s_vals .* ex
    uz = p.z_start .+ s_vals .* ez
    ua = ones(n)  # @. 4 * fracs * (1 - fracs)

    x  = copy(ux)
    z  = copy(uz)
    vx = (ds_dphi ./ T) .* ex
    vz = (ds_dphi ./ T) .* ez

    return InitialGuess2D(t, x, z, vx, vz, ux, uz, ua)
end

function load_guess_from_2d_file(filepath::String, p::TweezerParams2D;
                                  time_scale::Float64 = 1.0)
    h5open(filepath, "r") do f
        t_file  = read(f, "t")

        _read(keys...) = for k in keys; haskey(f, k) && return read(f, k); end;
                         zeros(length(t_file))

        ux_file = let v = nothing
            for k in ("ux", "control_ux", "guess_ux"); haskey(f, k) && (v = read(f, k); break); end
            v === nothing ? error("No ux key in $filepath") : v
        end
        uz_file = let v = nothing
            for k in ("uz", "control_uz", "guess_uz"); haskey(f, k) && (v = read(f, k); break); end
            v === nothing ? zeros(length(t_file)) : v
        end
        ua_file = let v = nothing
            for k in ("ua", "control_ua", "guess_ua"); haskey(f, k) && (v = read(f, k); break); end
            v === nothing ? error("No ua key in $filepath") : v
        end
        x_file  = haskey(f, "x")  ? read(f, "x")  : zeros(length(t_file))
        z_file  = haskey(f, "z")  ? read(f, "z")  : zeros(length(t_file))
        vx_file = haskey(f, "vx") ? read(f, "vx") : zeros(length(t_file))
        vz_file = haskey(f, "vz") ? read(f, "vz") : zeros(length(t_file))

        t_norm_old  = t_file ./ t_file[end]
        t_norm_new  = collect(range(0.0, 1.0, length=p.n))
        T_new       = t_file[end] * time_scale
        t_new       = t_norm_new .* T_new

        return InitialGuess2D(
            t_new,
            _linterp(t_norm_old, x_file,  t_norm_new),
            _linterp(t_norm_old, z_file,  t_norm_new),
            _linterp(t_norm_old, vx_file, t_norm_new),
            _linterp(t_norm_old, vz_file, t_norm_new),
            _linterp(t_norm_old, ux_file, t_norm_new),
            _linterp(t_norm_old, uz_file, t_norm_new),
            _linterp(t_norm_old, ua_file, t_norm_new),
        )
    end
end

function load_guess_from_1d_file(filepath::String, p::TweezerParams2D;
                                  time_scale::Float64 = 1.0)
    ex, ez, L = transport_direction2d(p)

    h5open(filepath, "r") do f
        t_file = read(f, "t")

        ux_1d = let v = nothing
            for k in ("ux", "control_ux", "guess_ux"); haskey(f, k) && (v = read(f, k); break); end
            v === nothing ? error("No ux key in $filepath") : v
        end
        ua_file = let v = nothing
            for k in ("ua", "control_ua", "guess_ua"); haskey(f, k) && (v = read(f, k); break); end
            v === nothing ? error("No ua key in $filepath") : v
        end
        x_1d = let v = nothing
            for k in ("x", "control_x", "guess_x"); haskey(f, k) && (v = read(f, k); break); end
            v === nothing ? copy(ux_1d) : v
        end
        v_1d = let v = nothing
            for k in ("v", "control_v", "guess_v"); haskey(f, k) && (v = read(f, k); break); end
            v === nothing ? zeros(length(t_file)) : v
        end

        attrs_f    = attrs(f)
        xStart_1d  = haskey(attrs_f, "xStart") ? read(attrs_f, "xStart") : ux_1d[1]
        xStop_1d   = haskey(attrs_f, "xStop")  ? read(attrs_f, "xStop")  : ux_1d[end]
        L_1d       = xStop_1d - xStart_1d

        s_ux  = (ux_1d .- xStart_1d) ./ L_1d
        s_x   = (x_1d  .- xStart_1d) ./ L_1d
        v_scaled = v_1d .* (L / L_1d)

        t_norm_old = t_file ./ t_file[end]
        t_norm_new = collect(range(0.0, 1.0, length=p.n))
        T_new      = t_file[end] * time_scale
        t_new      = t_norm_new .* T_new

        s_ux_n = _linterp(t_norm_old, s_ux,    t_norm_new)
        s_x_n  = _linterp(t_norm_old, s_x,     t_norm_new)
        v_n    = _linterp(t_norm_old, v_scaled, t_norm_new)
        ua_n   = _linterp(t_norm_old, ua_file,  t_norm_new)

        return InitialGuess2D(
            t_new,
            p.x_start .+ s_x_n  .* L .* ex,
            p.z_start .+ s_x_n  .* L .* ez,
            v_n .* ex,
            v_n .* ez,
            p.x_start .+ s_ux_n .* L .* ex,
            p.z_start .+ s_ux_n .* L .* ez,
            ua_n,
        )
    end
end

end
