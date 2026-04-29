module TweezerControls3DSingle

using JuMP
import Ipopt
using ..Types3D
using ..Models3D
using ..ThermalSampling3D

export optimize_controls3d_single

# Single-atom 3D optimal transport.
#
# The tweezer moves along the straight line from (x_start, y_start) to
# (x_stop, y_stop).  The scalar arc-length s[j] ∈ [0, L] parameterises the
# position along this line, so:
#   ux[j] = x_start + s[j]*ex,   uy[j] = y_start + s[j]*ey
# where (ex, ey) is the unit vector along the transport direction and L is
# the transport distance.  Only s[j] and ua[j] are optimised — the motion
# stays on the prescribed line by construction.
#
# Atom state: (x, y, z, vx, vy, vz) — all six degrees of freedom are free.
# Gravity acts in the -z direction (g_dimless = g_SI * t0^2 / w0).

function optimize_controls3d_single(
    p::TweezerParams3D;
    guess::Union{Nothing,InitialG3D} = nothing,
    bounds::Union{Nothing,ControlBounds3D} = nothing,
    silent::Bool   = true,
    max_iter::Int  = 4000,
    print_level::Int = 5,
    hessian_approximation::Bool = true,
    linear_r::Bool = false,
    consts::PhysicalConstants3D = default_constants3d(),
)
    bounds === nothing && (bounds = default_bounds3d(p))
    scales = compute_scales3d_full(p; consts = consts)
    g      = scales.g_dimless

    ex, ey, L = transport_direction(p)

    n    = p.n
    w    = p.w
    w_a  = p.w * p.w_aux_factor
    zR   = p.zR
    zR_a = p.zR_aux
    w2   = w^2
    wa2  = w_a^2
    cz   = 0.0

    model = Model(Ipopt.Optimizer)
    silent && set_silent(model)
    set_optimizer_attribute(model, "print_level", print_level)
    set_optimizer_attribute(model, "max_iter", max_iter)
    set_optimizer_attribute(model, "tol", 1e-6)
    set_optimizer_attribute(model, "acceptable_tol", 1e-5)
    hessian_approximation && set_optimizer_attribute(model, "hessian_approximation", "limited-memory")

    @variable(model, bounds.T_min_fraction * p.maxT <= T <= p.maxT)
    @expression(model, dt, T / (n - 1))

    # Arc-length along transport line (s=0 at start, s=L at stop)
    if linear_r
        @expression(model, s[j=1:n], (j - 1) / (n - 1) * L)
    else
        u_margin = bounds.u_margin_w * w
        @variable(model, -u_margin <= s[1:n] <= L + u_margin)
    end

    @variable(model, bounds.ua_min <= ua[1:n] <= bounds.ua_max)

    # Tweezer centre expressed through s
    @expression(model, ux[j=1:n], p.x_start + s[j] * ex)
    @expression(model, uy[j=1:n], p.y_start + s[j] * ey)

    # Atom state variables
    x_margin = bounds.r_margin_w * w
    @variable(model, p.x_start - x_margin <= x[1:n] <= p.x_stop + x_margin)
    @variable(model, -x_margin        <= y[1:n] <= x_margin)
    @variable(model, -bounds.z_margin <= z[1:n] <= bounds.z_margin)
    @variable(model, -bounds.v_xy_max <= vx[1:n] <= bounds.v_xy_max)
    @variable(model, -bounds.v_xy_max <= vy[1:n] <= bounds.v_xy_max)
    @variable(model, -bounds.v_z_max  <= vz[1:n] <= bounds.v_z_max)

    # ── Initial guess ──────────────────────────────────────────────────────────
    if guess !== nothing
        set_start_value(T, sum(guess.dt))
        for j in 1:n
            if !linear_r
                # recover s from ux/uy guess (pick whichever direction has larger component)
                s_guess = abs(ex) >= abs(ey) ?
                    (guess.ux[j] - p.x_start) / ex :
                    (guess.uy[j] - p.y_start) / ey
                set_start_value(s[j], clamp(s_guess, 0.0, L))
            end
            set_start_value(ua[j], guess.ua[j])
            set_start_value(x[j], guess.x[j])
            set_start_value(y[j], guess.y[j])
            set_start_value(z[j], guess.z[j])
            set_start_value(vx[j], guess.vx[j])
            set_start_value(vy[j], guess.vy[j])
            set_start_value(vz[j], guess.vz[j])
        end
    else
        set_start_value(T, 10.0) # 10us initial guess
        for j in 1:n
            frac = (j - 1) / (n - 1)
            !linear_r && set_start_value(s[j], frac * L)
            if j == 1 || j == n
                set_start_value(ua[j], 0.0)
            else
                set_start_value(ua[j], bounds.ua_max) # some initial auxiliary power in
            end
            set_start_value(x[j],  p.x_start + frac * (p.x_stop - p.x_start))
            set_start_value(y[j],  frac * p.y_stop)
            set_start_value(z[j],  0.0)
            set_start_value(vx[j], 0.0)
            set_start_value(vy[j], 0.0)
            set_start_value(vz[j], 0.0)
        end
    end

    # ── Boundary conditions ────────────────────────────────────────────────────
    if !linear_r
        @constraint(model, -bounds.u_margin_w * w <= s[1] <= bounds.u_margin_w * w)  # start at s=0 (with some margin)
    end
    @constraint(model, ua[1] == 0.0)
    @constraint(model, ua[n] == 0.0)

    @constraint(model, x[1]  == p.x_start)
    @constraint(model, y[1]  == 0.0)
    @constraint(model, z[1]  == 0.0)
    @constraint(model, vx[1] == 0.0)
    @constraint(model, vy[1] == 0.0)
    @constraint(model, vz[1] == 0.0)

    @constraint(model, x[n]  == p.x_stop)
    @constraint(model, y[n]  == p.y_stop)
    @constraint(model, vx[n] == 0.0)
    @constraint(model, vy[n] == 0.0)
    # @constraint(model, vz[n] == 0.0)

    # ── 3D Gaussian beam forces (analytic, inline for JuMP) ───────────────────
    # Static tweezer 1 at (x_start, 0, 0)
    @expression(model, XiSt[j=1:n], z[j] - cz)
    @expression(model, r1sq[j=1:n], (x[j] - p.x_start)^2 + y[j]^2)
    @expression(model, wXi1[j=1:n], w2 * (1.0 + (XiSt[j]/zR)^2))
    @expression(model, f1[j=1:n],   (w2/wXi1[j]) * exp(-2.0*r1sq[j]/wXi1[j]))

    # Static tweezer 2 at (x_stop, y_stop, 0)
    @expression(model, r2sq[j=1:n], (x[j] - p.x_stop)^2 + (y[j] - p.y_stop)^2)
    @expression(model, wXi2[j=1:n], w2 * (1.0 + (XiSt[j]/zR)^2))
    @expression(model, f2[j=1:n],   (w2/wXi2[j]) * exp(-2.0*r2sq[j]/wXi2[j]))

    # Aux tweezer at (ux, uy, 0)
    @expression(model, rasq[j=1:n], (x[j] - ux[j])^2 + (y[j] - uy[j])^2)
    @expression(model, wXia[j=1:n], wa2 * (1.0 + (XiSt[j]/zR_a)^2))
    @expression(model, fa[j=1:n],   (wa2/wXia[j]) * exp(-2.0*rasq[j]/wXia[j]))

    @expression(model, Fx[j=1:n],
        -4.0*p.U0_static*(x[j]-p.x_start)/wXi1[j]*f1[j]
        -4.0*p.U0_static*(x[j]-p.x_stop) /wXi2[j]*f2[j]
        -4.0*ua[j]*p.U0_aux_max*(x[j]-ux[j])/wXia[j]*fa[j]
    )
    @expression(model, Fy[j=1:n],
        -4.0*p.U0_static*y[j]/wXi1[j]*f1[j]
        -4.0*p.U0_static*(y[j]-p.y_stop)/wXi2[j]*f2[j]
        -4.0*ua[j]*p.U0_aux_max*(y[j]-uy[j])/wXia[j]*fa[j]
    )
    @expression(model, Fz_beam[j=1:n],
          p.U0_static*f1[j]*(w2/wXi1[j])*(XiSt[j]/zR^2)*(4.0*r1sq[j]/wXi1[j]-2.0)
        + p.U0_static*f2[j]*(w2/wXi2[j])*(XiSt[j]/zR^2)*(4.0*r2sq[j]/wXi2[j]-2.0)
        + ua[j]*p.U0_aux_max*fa[j]*(wa2/wXia[j])*(XiSt[j]/zR_a^2)*(4.0*rasq[j]/wXia[j]-2.0)
    )
    @expression(model, Fz_total[j=1:n], Fz_beam[j] - g)

    # ── Trapezoidal collocation ────────────────────────────────────────────────
    for j in 1:(n-1)
        @constraint(model, x[j+1]  - x[j]  == 0.5*dt*(vx[j] + vx[j+1]))
        @constraint(model, y[j+1]  - y[j]  == 0.5*dt*(vy[j] + vy[j+1]))
        @constraint(model, z[j+1]  - z[j]  == 0.5*dt*(vz[j] + vz[j+1]))
        @constraint(model, vx[j+1] - vx[j] == 0.5*dt*(Fx[j] + Fx[j+1]))
        @constraint(model, vy[j+1] - vy[j] == 0.5*dt*(Fy[j] + Fy[j+1]))
        @constraint(model, vz[j+1] - vz[j] == 0.5*dt*(Fz_total[j] + Fz_total[j+1]))
    end

    # ── Slew-rate constraints ──────────────────────────────────────────────────
    v_ua_max  = bounds.v_ua_max
    for j in 1:(n-1)
        if !linear_r
            v_s_max = bounds.v_u_max_per_w * w
            @constraint(model, s[j+1] - s[j] <= v_s_max * dt)
            @constraint(model, s[j] - s[j+1] <= v_s_max * dt)
        end
        @constraint(model, ua[j+1] - ua[j] <= v_ua_max * dt)
        @constraint(model, ua[j] - ua[j+1] <= v_ua_max * dt)
    end

    # ── Trapping constraint (energy) ───────────────────────────────────────────
    @expression(model, U_st1[j=1:n], -p.U0_static * f1[j])
    @expression(model, U_st2[j=1:n], -p.U0_static * f2[j])
    @expression(model, U_aux[j=1:n], -ua[j]*p.U0_aux_max * fa[j])
    @expression(model, U_tot[j=1:n], U_st1[j] + U_st2[j] + U_aux[j])
    @expression(model, KE[j=1:n],    0.5*(vx[j]^2 + vy[j]^2 + vz[j]^2))
    @expression(model, E_tot[j=1:n], KE[j] + U_tot[j])
    @constraint(model, [j=1:n], E_tot[j] <= p.trap_fraction * U_tot[j])
    @constraint(model, E_tot[n] <= p.final_trap_fraction * U_tot[n])

    # ── Objective ──────────────────────────────────────────────────────────────
    @expression(model, heat_term,   sum(-(U_st1[j]+U_st2[j]+U_aux[j]) for j=1:n))
    @expression(model, jitter_ua,   sum((ua[j+1]-ua[j])^2 for j=10:(n-10)))

    if linear_r
        @objective(model, Min,
            T
            + p.lambda_heat   * dt * heat_term
            + p.lambda_jitter * 10.0 * jitter_ua
        )
    else
        @expression(model, jitter_s, sum((s[j+1]-s[j])^2 for j=10:(n-10)))
        @objective(model, Min,
            T
            + p.lambda_heat   * dt * heat_term
            + p.lambda_jitter * (jitter_s + 10.0 * jitter_ua)
        )
    end

    optimize!(model)
    println("Termination status (single): ", termination_status(model))

    t_grid = collect(0.0:1.0:(n-1)) .* value(T) / (n - 1)
    s_val  = value.(s)
    ux_val = p.x_start .+ s_val .* ex
    uy_val = p.y_start .+ s_val .* ey

    protocol = ControlProtocol3D(
        t_grid,
        value.(x), value.(y), value.(z),
        value.(vx), value.(vy), value.(vz),
        ux_val, uy_val, value.(ua),
    )

    return protocol, string(termination_status(model)), objective_value(model)
end

end
