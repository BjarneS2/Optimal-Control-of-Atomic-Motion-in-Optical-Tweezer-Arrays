module TweezerControls3DThermal

using JuMP
import Ipopt
using ..Types3D
using ..Models3D
using ..ThermalSampling3D
using Random

export optimize_controls3d_thermal

function optimize_controls3d_thermal(
    p::TweezerParams3D;
    guess::Union{Nothing,InitialG3D} = nothing,
    bounds::Union{Nothing,ControlBounds3D} = nothing,
    silent::Bool   = true,
    max_iter::Int  = 4000,
    print_level::Int = 5,
    hessian_approximation::Bool = true,
    n_samples::Int = 10,
    seed::Union{Nothing,Int} = nothing,
    consts::PhysicalConstants3D = default_constants3d(),
)
    bounds === nothing && (bounds = default_bounds3d(p))
    seed !== nothing   && Random.seed!(seed)
    scales = compute_scales3d_full(p; consts = consts)
    g      = scales.g_dimless

    ex, ey, L = transport_direction(p)

    println("Sampling $n_samples thermal initial conditions (3D, only trapped)...")
    samples = [sample_initial_conditions3d(p; consts = consts, check_trapped = true)
               for _ in 1:n_samples]

    println("  x range: [$(minimum(s.x for s in samples)), $(maximum(s.x for s in samples))]")
    println("  z range: [$(minimum(s.z for s in samples)), $(maximum(s.z for s in samples))]")

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

    u_margin = bounds.u_margin_w * w
    @variable(model, -u_margin <= s[1:n] <= L + u_margin)
    @variable(model, bounds.ua_min <= ua[1:n] <= bounds.ua_max)

    @expression(model, ux[j=1:n], p.x_start + s[j] * ex)
    @expression(model, uy[j=1:n], p.y_start + s[j] * ey)

    # Per-sample atom state
    x_margin = bounds.r_margin_w * w
    @variable(model, x_s[1:n_samples, 1:n])
    @variable(model, y_s[1:n_samples, 1:n])
    @variable(model, z_s[1:n_samples, 1:n])
    @variable(model, vx_s[1:n_samples, 1:n])
    @variable(model, vy_s[1:n_samples, 1:n])
    @variable(model, vz_s[1:n_samples, 1:n])

    for i in 1:n_samples, j in 1:n
        set_lower_bound(x_s[i,j],  p.x_start - x_margin)
        set_upper_bound(x_s[i,j],  p.x_stop  + x_margin)
        set_lower_bound(y_s[i,j],  -x_margin)
        set_upper_bound(y_s[i,j],   x_margin)
        set_lower_bound(z_s[i,j],  -bounds.z_margin)
        set_upper_bound(z_s[i,j],   bounds.z_margin)
        set_lower_bound(vx_s[i,j], -bounds.v_xy_max)
        set_upper_bound(vx_s[i,j],  bounds.v_xy_max)
        set_lower_bound(vy_s[i,j], -bounds.v_xy_max)
        set_upper_bound(vy_s[i,j],  bounds.v_xy_max)
        set_lower_bound(vz_s[i,j], -bounds.v_z_max)
        set_upper_bound(vz_s[i,j],  bounds.v_z_max)
    end

    # ── Initial guess ──────────────────────────────────────────────────────────
    if guess !== nothing
        set_start_value(T, sum(guess.dt))
        for j in 1:n
            s_g = ex != 0.0 ? (guess.ux[j] - p.x_start) / ex : (guess.uy[j] - p.y_start) / ey
            set_start_value(s[j],  clamp(s_g, 0.0, L))
            set_start_value(ua[j], guess.ua[j])
        end
    else
        set_start_value(T, 0.5 * p.maxT)
        for j in 1:n
            set_start_value(s[j],  (j-1)/(n-1) * L)
            set_start_value(ua[j], 0.0)
        end
    end

    for i in 1:n_samples, j in 1:n
        frac = (j-1)/(n-1)
        set_start_value(x_s[i,j],  samples[i].x + frac*(p.x_stop  - samples[i].x))
        set_start_value(y_s[i,j],  samples[i].y + frac*(p.y_stop  - samples[i].y))
        set_start_value(z_s[i,j],  samples[i].z * (1 - frac))
        set_start_value(vx_s[i,j], samples[i].vx * (1 - frac))
        set_start_value(vy_s[i,j], samples[i].vy * (1 - frac))
        set_start_value(vz_s[i,j], samples[i].vz * (1 - frac))
    end

    # ── Boundary conditions ────────────────────────────────────────────────────
    @constraint(model, s[1]  == 0.0)
    @constraint(model, s[n]  == L)
    @constraint(model, ua[1] == 0.0)
    @constraint(model, ua[n] == 0.0)

    x_margin_final = bounds.r_margin_w * w
    for i in 1:n_samples
        @constraint(model, x_s[i,1]  == samples[i].x)
        @constraint(model, y_s[i,1]  == samples[i].y)
        @constraint(model, z_s[i,1]  == samples[i].z)
        @constraint(model, vx_s[i,1] == samples[i].vx)
        @constraint(model, vy_s[i,1] == samples[i].vy)
        @constraint(model, vz_s[i,1] == samples[i].vz)
        @constraint(model, p.x_stop - x_margin_final <= x_s[i,n] <= p.x_stop + x_margin_final)
        @constraint(model, p.y_stop - x_margin_final <= y_s[i,n] <= p.y_stop + x_margin_final)
    end

    # ── Forces (analytic, inline) ──────────────────────────────────────────────
    @expression(model, Xi_s[i=1:n_samples, j=1:n], z_s[i,j] - cz)

    @expression(model, r1sq_s[i=1:n_samples, j=1:n],
        (x_s[i,j]-p.x_start)^2 + y_s[i,j]^2)
    @expression(model, wXi1_s[i=1:n_samples, j=1:n],
        w2*(1.0+(Xi_s[i,j]/zR)^2))
    @expression(model, f1_s[i=1:n_samples, j=1:n],
        (w2/wXi1_s[i,j])*exp(-2.0*r1sq_s[i,j]/wXi1_s[i,j]))

    @expression(model, r2sq_s[i=1:n_samples, j=1:n],
        (x_s[i,j]-p.x_stop)^2 + (y_s[i,j]-p.y_stop)^2)
    @expression(model, wXi2_s[i=1:n_samples, j=1:n],
        w2*(1.0+(Xi_s[i,j]/zR)^2))
    @expression(model, f2_s[i=1:n_samples, j=1:n],
        (w2/wXi2_s[i,j])*exp(-2.0*r2sq_s[i,j]/wXi2_s[i,j]))

    @expression(model, rasq_s[i=1:n_samples, j=1:n],
        (x_s[i,j]-ux[j])^2 + (y_s[i,j]-uy[j])^2)
    @expression(model, wXia_s[i=1:n_samples, j=1:n],
        wa2*(1.0+(Xi_s[i,j]/zR_a)^2))
    @expression(model, fa_s[i=1:n_samples, j=1:n],
        (wa2/wXia_s[i,j])*exp(-2.0*rasq_s[i,j]/wXia_s[i,j]))

    @expression(model, Fx_s[i=1:n_samples, j=1:n],
        -4.0*p.U0_static*(x_s[i,j]-p.x_start)/wXi1_s[i,j]*f1_s[i,j]
        -4.0*p.U0_static*(x_s[i,j]-p.x_stop) /wXi2_s[i,j]*f2_s[i,j]
        -4.0*ua[j]*p.U0_aux_max*(x_s[i,j]-ux[j])/wXia_s[i,j]*fa_s[i,j]
    )
    @expression(model, Fy_s[i=1:n_samples, j=1:n],
        -4.0*p.U0_static*y_s[i,j]/(wXi1_s[i,j])*f1_s[i,j]
        -4.0*p.U0_static*(y_s[i,j]-p.y_stop)/wXi2_s[i,j]*f2_s[i,j]
        -4.0*ua[j]*p.U0_aux_max*(y_s[i,j]-uy[j])/wXia_s[i,j]*fa_s[i,j]
    )
    @expression(model, Fz_beam_s[i=1:n_samples, j=1:n],
          p.U0_static*f1_s[i,j]*(w2/wXi1_s[i,j])*(Xi_s[i,j]/zR^2)  *(4.0*r1sq_s[i,j]/wXi1_s[i,j]-2.0)
        + p.U0_static*f2_s[i,j]*(w2/wXi2_s[i,j])*(Xi_s[i,j]/zR^2)  *(4.0*r2sq_s[i,j]/wXi2_s[i,j]-2.0)
        + ua[j]*p.U0_aux_max*fa_s[i,j]*(wa2/wXia_s[i,j])*(Xi_s[i,j]/zR_a^2)*(4.0*rasq_s[i,j]/wXia_s[i,j]-2.0)
    )
    @expression(model, Fz_s[i=1:n_samples, j=1:n], Fz_beam_s[i,j] - g)

    # ── Trapezoidal collocation ────────────────────────────────────────────────
    for i in 1:n_samples, j in 1:(n-1)
        @constraint(model, x_s[i,j+1]  - x_s[i,j]  == 0.5*dt*(vx_s[i,j] + vx_s[i,j+1]))
        @constraint(model, y_s[i,j+1]  - y_s[i,j]  == 0.5*dt*(vy_s[i,j] + vy_s[i,j+1]))
        @constraint(model, z_s[i,j+1]  - z_s[i,j]  == 0.5*dt*(vz_s[i,j] + vz_s[i,j+1]))
        @constraint(model, vx_s[i,j+1] - vx_s[i,j] == 0.5*dt*(Fx_s[i,j] + Fx_s[i,j+1]))
        @constraint(model, vy_s[i,j+1] - vy_s[i,j] == 0.5*dt*(Fy_s[i,j] + Fy_s[i,j+1]))
        @constraint(model, vz_s[i,j+1] - vz_s[i,j] == 0.5*dt*(Fz_s[i,j] + Fz_s[i,j+1]))
    end

    # ── Slew-rate constraints ──────────────────────────────────────────────────
    v_s_max  = bounds.v_u_max_per_w * w
    v_ua_max = bounds.v_ua_max
    for j in 1:(n-1)
        @constraint(model,  s[j+1]-s[j] <=  v_s_max  * dt)
        @constraint(model,  s[j]-s[j+1] <=  v_s_max  * dt)
        @constraint(model,  ua[j+1]-ua[j] <= v_ua_max * dt)
        @constraint(model,  ua[j]-ua[j+1] <= v_ua_max * dt)
    end

    # ── Trapping constraints ───────────────────────────────────────────────────
    @expression(model, U_st1_s[i=1:n_samples, j=1:n], -p.U0_static * f1_s[i,j])
    @expression(model, U_st2_s[i=1:n_samples, j=1:n], -p.U0_static * f2_s[i,j])
    @expression(model, U_aux_s[i=1:n_samples, j=1:n], -ua[j]*p.U0_aux_max * fa_s[i,j])
    @expression(model, U_tot_s[i=1:n_samples, j=1:n], U_st1_s[i,j]+U_st2_s[i,j]+U_aux_s[i,j])
    @expression(model, KE_s[i=1:n_samples, j=1:n],
        0.5*(vx_s[i,j]^2 + vy_s[i,j]^2 + vz_s[i,j]^2))
    @expression(model, E_tot_s[i=1:n_samples, j=1:n], KE_s[i,j] + U_tot_s[i,j])
    @constraint(model, [i=1:n_samples, j=1:n],   E_tot_s[i,j] <= p.trap_fraction       * U_tot_s[i,j])
    @constraint(model, [i=1:n_samples],            E_tot_s[i,n] <= p.final_trap_fraction * U_tot_s[i,n])

    # ── Objective ──────────────────────────────────────────────────────────────
    println("λ_heat = $(p.lambda_heat), λ_jitter = $(p.lambda_jitter)")
    @expression(model, heat_s[i=1:n_samples],
        sum(-(U_st1_s[i,j]+U_st2_s[i,j]+U_aux_s[i,j]) for j=1:n))
    @objective(model, Min,
        T
        + p.lambda_heat   * dt * sum(heat_s[i] for i=1:n_samples) / n_samples
        + p.lambda_jitter * sum((ua[j+1]-ua[j])^2 for j=10:(n-10))
        + 10*p.lambda_jitter * sum((s[j+1]-s[j])^2 for j=1:(n-1))
    )

    println("Starting 3D thermal optimization with $n_samples samples...")
    optimize!(model)
    println("Termination status (thermal 3D): ", termination_status(model))

    t_grid = collect(0.0:1.0:(n-1)) .* value(T) / (n - 1)
    s_val  = value.(s)
    ux_val = p.x_start .+ s_val .* ex
    uy_val = p.y_start .+ s_val .* ey

    x_avg  = [sum(value(x_s[i,j]) for i=1:n_samples)/n_samples for j=1:n]
    y_avg  = [sum(value(y_s[i,j]) for i=1:n_samples)/n_samples for j=1:n]
    z_avg  = [sum(value(z_s[i,j]) for i=1:n_samples)/n_samples for j=1:n]
    vx_avg = [sum(value(vx_s[i,j]) for i=1:n_samples)/n_samples for j=1:n]
    vy_avg = [sum(value(vy_s[i,j]) for i=1:n_samples)/n_samples for j=1:n]
    vz_avg = [sum(value(vz_s[i,j]) for i=1:n_samples)/n_samples for j=1:n]

    protocol = ControlProtocol3D(
        t_grid, x_avg, y_avg, z_avg, vx_avg, vy_avg, vz_avg,
        ux_val, uy_val, value.(ua),
    )

    x_traj  = zeros(n_samples, n)
    y_traj  = zeros(n_samples, n)
    z_traj  = zeros(n_samples, n)
    vx_traj = zeros(n_samples, n)
    vy_traj = zeros(n_samples, n)
    vz_traj = zeros(n_samples, n)
    for i in 1:n_samples, j in 1:n
        x_traj[i,j]  = value(x_s[i,j])
        y_traj[i,j]  = value(y_s[i,j])
        z_traj[i,j]  = value(z_s[i,j])
        vx_traj[i,j] = value(vx_s[i,j])
        vy_traj[i,j] = value(vy_s[i,j])
        vz_traj[i,j] = value(vz_s[i,j])
    end

    return ThermalControlResult3D(
        protocol,
        samples,
        x_traj, y_traj, z_traj,
        vx_traj, vy_traj, vz_traj,
        seed,
        n_samples,
        string(termination_status(model)),
        objective_value(model),
    )
end

end
