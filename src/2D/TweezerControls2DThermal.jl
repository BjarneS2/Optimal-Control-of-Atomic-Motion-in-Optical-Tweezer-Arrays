module TweezerControls2DThermal

using JuMP
import Ipopt
using ..Types2D
using ..Models2D
using ..ThermalSampling2D
using Random

export optimize_controls2d_thermal

const N_EXT_CHECK = 100

function optimize_controls2d_thermal(
    p::TweezerParams2D;
    guess::Union{Nothing, InitialG2D}  = nothing,
    bounds::Union{Nothing, ControlBounds2D} = nothing,
    silent::Bool               = true,
    max_iter::Int              = 4000,
    print_level::Int           = 5,
    hessian_approximation::Bool = true,
    n_samples::Int             = 10,
    seed::Union{Nothing, Int}  = nothing,
    t_extension::Float64       = 0.0,
    consts::PhysicalConstants2D = default_constants2d(),
)
    bounds === nothing && (bounds = default_bounds2d(p))
    seed !== nothing   && Random.seed!(seed)
    scales = compute_scales2d_full(p; consts = consts)
    g      = scales.g_dimless

    if p.final_trap_fraction < p.starting_trap_fraction
        @warn "final_trap_fraction ($(p.final_trap_fraction)) < starting_trap_fraction ($(p.starting_trap_fraction)): final constraint is tighter than initial sampling — may be strongly infeasible"
    end
    if p.trap_fraction !== nothing && p.trap_fraction < p.starting_trap_fraction
        @warn "trap_fraction ($(p.trap_fraction)) < starting_trap_fraction ($(p.starting_trap_fraction)): mid-transport constraint is tighter than initial sampling — likely infeasible"
    end

    ex, ez, L = transport_direction2d(p)
    if abs(p.z_start) < 1e-10 && abs(p.z_stop) < 1e-10
        println("Confined z=0 mode: transport is purely along x, uz will be 0 throughout.")
    end

    println("Sampling $n_samples thermal initial conditions (2D, only trapped)...")
    samples = [sample_initial_conditions2d(p; consts = consts, check_trapped = true)
               for _ in 1:n_samples]
    println("  x range:  [$(minimum(s.x for s in samples)), $(maximum(s.x for s in samples))]")
    println("  z range:  [$(minimum(s.z for s in samples)), $(maximum(s.z for s in samples))]")
    println("  vx range: [$(minimum(s.vx for s in samples)), $(maximum(s.vx for s in samples))]")
    println("  vz range: [$(minimum(s.vz for s in samples)), $(maximum(s.vz for s in samples))]")
    let energies = [0.5*(s.vx^2 + s.vz^2) + potential2d(s.x, s.z, p.x_start, p.z_start, 0.0, p) for s in samples]
        println("  E range:  [$(minimum(energies)), $(maximum(energies))]")
    end

    n    = p.n
    w    = p.w
    w_a  = p.w * p.w_aux_factor
    zR   = p.zR
    zR_a = p.zR_aux
    w2   = w^2
    wa2  = w_a^2

    model = Model(Ipopt.Optimizer)
    silent && set_silent(model)
    set_optimizer_attribute(model, "print_level",    print_level)
    set_optimizer_attribute(model, "max_iter",       max_iter)
    set_optimizer_attribute(model, "tol",            1e-6)
    set_optimizer_attribute(model, "acceptable_tol", 1e-5)
    hessian_approximation && set_optimizer_attribute(model, "hessian_approximation", "limited-memory")

    @variable(model, bounds.T_min_fraction * p.maxT <= T <= p.maxT)
    @expression(model, dt, T / (n - 1))

    u_margin = bounds.u_margin_w * w
    @variable(model, -u_margin <= s[1:n] <= L + u_margin)
    @variable(model, bounds.ua_min <= ua[1:n] <= bounds.ua_max)

    @expression(model, ux[j=1:n], p.x_start + s[j] * ex)
    @expression(model, uz[j=1:n], p.z_start + s[j] * ez)

    x_margin = bounds.r_margin_w * w
    @variable(model, x_s[1:n_samples, 1:n])
    @variable(model, z_s[1:n_samples, 1:n])
    @variable(model, vx_s[1:n_samples, 1:n])
    @variable(model, vz_s[1:n_samples, 1:n])

    for i in 1:n_samples, j in 1:n
        set_lower_bound(x_s[i,j],  p.x_start - x_margin)
        set_upper_bound(x_s[i,j],  p.x_stop  + x_margin)
        set_lower_bound(z_s[i,j],  -bounds.z_margin)
        set_upper_bound(z_s[i,j],   bounds.z_margin)
        set_lower_bound(vx_s[i,j], -bounds.v_x_max)
        set_upper_bound(vx_s[i,j],  bounds.v_x_max)
        set_lower_bound(vz_s[i,j], -bounds.v_z_max)
        set_upper_bound(vz_s[i,j],  bounds.v_z_max)
    end

    if guess !== nothing
        set_start_value(T, guess.t[end])
        for j in 1:n
            s_g = abs(ex) > 1e-10 ? (guess.ux[j] - p.x_start) / ex :
                                     (guess.uz[j] - p.z_start) / ez
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
        set_start_value(x_s[i,j],  samples[i].x  + frac*(p.x_stop - samples[i].x))
        set_start_value(z_s[i,j],  samples[i].z  + frac*(p.z_stop - samples[i].z))
        set_start_value(vx_s[i,j], samples[i].vx * (1 - frac))
        set_start_value(vz_s[i,j], samples[i].vz * (1 - frac))
    end

    @constraint(model, s[1]  == 0.0)
    @constraint(model, s[n]  == L)
    @constraint(model, ua[1] == 0.0)
    @constraint(model, ua[n] == 0.0)

    x_margin_final = bounds.r_margin_w * w
    for i in 1:n_samples
        @constraint(model, x_s[i,1]  == samples[i].x)
        @constraint(model, z_s[i,1]  == samples[i].z)
        @constraint(model, vx_s[i,1] == samples[i].vx)
        @constraint(model, vz_s[i,1] == samples[i].vz)
        @constraint(model, p.x_stop - x_margin_final <= x_s[i,n] <= p.x_stop + x_margin_final)
        @constraint(model, p.z_stop - x_margin_final <= z_s[i,n] <= p.z_stop + x_margin_final)
    end

    @expression(model, dz1_s[i=1:n_samples, j=1:n], z_s[i,j] - p.z_start)
    @expression(model, wz1_s[i=1:n_samples, j=1:n], w2*(1.0+(dz1_s[i,j]/zR)^2))
    @expression(model, dx1sq_s[i=1:n_samples, j=1:n], (x_s[i,j]-p.x_start)^2)
    @expression(model, f1_s[i=1:n_samples, j=1:n], (w2/wz1_s[i,j])*exp(-2.0*dx1sq_s[i,j]/wz1_s[i,j]))

    @expression(model, dz2_s[i=1:n_samples, j=1:n], z_s[i,j] - p.z_stop)
    @expression(model, wz2_s[i=1:n_samples, j=1:n], w2*(1.0+(dz2_s[i,j]/zR)^2))
    @expression(model, dx2sq_s[i=1:n_samples, j=1:n], (x_s[i,j]-p.x_stop)^2)
    @expression(model, f2_s[i=1:n_samples, j=1:n], (w2/wz2_s[i,j])*exp(-2.0*dx2sq_s[i,j]/wz2_s[i,j]))

    @expression(model, dza_s[i=1:n_samples, j=1:n], z_s[i,j] - uz[j])
    @expression(model, wza_s[i=1:n_samples, j=1:n], wa2*(1.0+(dza_s[i,j]/zR_a)^2))
    @expression(model, dxasq_s[i=1:n_samples, j=1:n], (x_s[i,j]-ux[j])^2)
    @expression(model, fa_s[i=1:n_samples, j=1:n], (wa2/wza_s[i,j])*exp(-2.0*dxasq_s[i,j]/wza_s[i,j]))

    @expression(model, Fx_s[i=1:n_samples, j=1:n],
        -4.0*p.U0_static*(x_s[i,j]-p.x_start)/wz1_s[i,j]*f1_s[i,j]
        -4.0*p.U0_static*(x_s[i,j]-p.x_stop) /wz2_s[i,j]*f2_s[i,j]
        -4.0*ua[j]*p.U0_aux_max*(x_s[i,j]-ux[j])/wza_s[i,j]*fa_s[i,j]
    )
    @expression(model, Fz_beam_s[i=1:n_samples, j=1:n],
          p.U0_static*f1_s[i,j]*(w2/wz1_s[i,j])*(dz1_s[i,j]/zR^2)  *(4.0*dx1sq_s[i,j]/wz1_s[i,j]-2.0)
        + p.U0_static*f2_s[i,j]*(w2/wz2_s[i,j])*(dz2_s[i,j]/zR^2)  *(4.0*dx2sq_s[i,j]/wz2_s[i,j]-2.0)
        + ua[j]*p.U0_aux_max*fa_s[i,j]*(wa2/wza_s[i,j])*(dza_s[i,j]/zR_a^2)*(4.0*dxasq_s[i,j]/wza_s[i,j]-2.0)
    )
    @expression(model, Fz_s[i=1:n_samples, j=1:n], Fz_beam_s[i,j] - g)

    for i in 1:n_samples, j in 1:(n-1)
        @constraint(model, x_s[i,j+1]  - x_s[i,j]  == 0.5*dt*(vx_s[i,j] + vx_s[i,j+1]))
        @constraint(model, z_s[i,j+1]  - z_s[i,j]  == 0.5*dt*(vz_s[i,j] + vz_s[i,j+1]))
        @constraint(model, vx_s[i,j+1] - vx_s[i,j] == 0.5*dt*(Fx_s[i,j] + Fx_s[i,j+1]))
        @constraint(model, vz_s[i,j+1] - vz_s[i,j] == 0.5*dt*(Fz_s[i,j] + Fz_s[i,j+1]))
    end

    v_s_max  = bounds.v_u_max_per_w * w
    v_ua_max = bounds.v_ua_max
    for j in 1:(n-1)
        @constraint(model,  s[j+1] - s[j]   <=  v_s_max  * dt)
        @constraint(model,  s[j]   - s[j+1] <=  v_s_max  * dt)
        @constraint(model,  ua[j+1] - ua[j]  <=  v_ua_max * dt)
        @constraint(model,  ua[j]   - ua[j+1] <= v_ua_max * dt)
    end

    @expression(model, U_st1_s[i=1:n_samples, j=1:n], -p.U0_static * f1_s[i,j])
    @expression(model, U_st2_s[i=1:n_samples, j=1:n], -p.U0_static * f2_s[i,j])
    @expression(model, U_aux_s[i=1:n_samples, j=1:n], -ua[j]*p.U0_aux_max * fa_s[i,j])
    @expression(model, U_tot_s[i=1:n_samples, j=1:n], U_st1_s[i,j]+U_st2_s[i,j]+U_aux_s[i,j])
    @expression(model, KE_s[i=1:n_samples, j=1:n],    0.5*(vx_s[i,j]^2 + vz_s[i,j]^2))
    @expression(model, E_tot_s[i=1:n_samples, j=1:n], KE_s[i,j] + U_tot_s[i,j])

    @constraint(model, [i=1:n_samples], E_tot_s[i,1] <= p.starting_trap_fraction * U_tot_s[i,1])
    @constraint(model, [i=1:n_samples], E_tot_s[i,n] <= p.final_trap_fraction    * U_tot_s[i,n])
    if p.trap_fraction !== nothing
        @constraint(model, [i=1:n_samples, j=2:n-1], E_tot_s[i,j] <= p.trap_fraction * U_tot_s[i,j])
    end

    if t_extension > 0.0
        dt_ext = t_extension / (N_EXT_CHECK - 1)
        println("Extension period: $(t_extension) μs, $(N_EXT_CHECK) check nodes, dt_ext=$(round(dt_ext, digits=4)) μs")

        @variable(model, x_ext[1:n_samples,  1:N_EXT_CHECK])
        @variable(model, z_ext[1:n_samples,  1:N_EXT_CHECK])
        @variable(model, vx_ext[1:n_samples, 1:N_EXT_CHECK])
        @variable(model, vz_ext[1:n_samples, 1:N_EXT_CHECK])

        for i in 1:n_samples, k in 1:N_EXT_CHECK
            set_lower_bound(x_ext[i,k],  p.x_start - x_margin)
            set_upper_bound(x_ext[i,k],  p.x_stop  + x_margin)
            set_lower_bound(z_ext[i,k],  -bounds.z_margin)
            set_upper_bound(z_ext[i,k],   bounds.z_margin)
            set_lower_bound(vx_ext[i,k], -bounds.v_x_max)
            set_upper_bound(vx_ext[i,k],  bounds.v_x_max)
            set_lower_bound(vz_ext[i,k], -bounds.v_z_max)
            set_upper_bound(vz_ext[i,k],  bounds.v_z_max)
            set_start_value(x_ext[i,k],  p.x_stop)
            set_start_value(z_ext[i,k],  p.z_stop)
            set_start_value(vx_ext[i,k], 0.0)
            set_start_value(vz_ext[i,k], 0.0)
        end

        for i in 1:n_samples
            @constraint(model, x_ext[i,1]  == x_s[i,n])
            @constraint(model, z_ext[i,1]  == z_s[i,n])
            @constraint(model, vx_ext[i,1] == vx_s[i,n])
            @constraint(model, vz_ext[i,1] == vz_s[i,n])
        end

        @expression(model, dz1_ext[i=1:n_samples, k=1:N_EXT_CHECK], z_ext[i,k] - p.z_start)
        @expression(model, wz1_ext[i=1:n_samples, k=1:N_EXT_CHECK], w2*(1.0+(dz1_ext[i,k]/zR)^2))
        @expression(model, dx1sq_ext[i=1:n_samples, k=1:N_EXT_CHECK], (x_ext[i,k]-p.x_start)^2)
        @expression(model, f1_ext[i=1:n_samples, k=1:N_EXT_CHECK], (w2/wz1_ext[i,k])*exp(-2.0*dx1sq_ext[i,k]/wz1_ext[i,k]))

        @expression(model, dz2_ext[i=1:n_samples, k=1:N_EXT_CHECK], z_ext[i,k] - p.z_stop)
        @expression(model, wz2_ext[i=1:n_samples, k=1:N_EXT_CHECK], w2*(1.0+(dz2_ext[i,k]/zR)^2))
        @expression(model, dx2sq_ext[i=1:n_samples, k=1:N_EXT_CHECK], (x_ext[i,k]-p.x_stop)^2)
        @expression(model, f2_ext[i=1:n_samples, k=1:N_EXT_CHECK], (w2/wz2_ext[i,k])*exp(-2.0*dx2sq_ext[i,k]/wz2_ext[i,k]))

        @expression(model, Fx_ext[i=1:n_samples, k=1:N_EXT_CHECK],
            -4.0*p.U0_static*(x_ext[i,k]-p.x_start)/wz1_ext[i,k]*f1_ext[i,k]
            -4.0*p.U0_static*(x_ext[i,k]-p.x_stop) /wz2_ext[i,k]*f2_ext[i,k]
        )
        @expression(model, Fz_ext[i=1:n_samples, k=1:N_EXT_CHECK],
              p.U0_static*f1_ext[i,k]*(w2/wz1_ext[i,k])*(dz1_ext[i,k]/zR^2)  *(4.0*dx1sq_ext[i,k]/wz1_ext[i,k]-2.0)
            + p.U0_static*f2_ext[i,k]*(w2/wz2_ext[i,k])*(dz2_ext[i,k]/zR^2)  *(4.0*dx2sq_ext[i,k]/wz2_ext[i,k]-2.0)
            - g
        )

        for i in 1:n_samples, k in 1:(N_EXT_CHECK-1)
            @constraint(model, x_ext[i,k+1]  - x_ext[i,k]  == 0.5*dt_ext*(vx_ext[i,k] + vx_ext[i,k+1]))
            @constraint(model, z_ext[i,k+1]  - z_ext[i,k]  == 0.5*dt_ext*(vz_ext[i,k] + vz_ext[i,k+1]))
            @constraint(model, vx_ext[i,k+1] - vx_ext[i,k] == 0.5*dt_ext*(Fx_ext[i,k] + Fx_ext[i,k+1]))
            @constraint(model, vz_ext[i,k+1] - vz_ext[i,k] == 0.5*dt_ext*(Fz_ext[i,k] + Fz_ext[i,k+1]))
        end

        @expression(model, U_ext[i=1:n_samples, k=1:N_EXT_CHECK],
            -p.U0_static*f1_ext[i,k] - p.U0_static*f2_ext[i,k])
        @expression(model, E_ext[i=1:n_samples, k=1:N_EXT_CHECK],
            0.5*(vx_ext[i,k]^2 + vz_ext[i,k]^2) + U_ext[i,k])
        @constraint(model, [i=1:n_samples, k=1:N_EXT_CHECK],
            E_ext[i,k] <= p.final_trap_fraction * U_ext[i,k])
    end

    println("λ_jitter_pos = $(p.lambda_jitter_pos), λ_jitter_ua = $(p.lambda_jitter_ua)")
    @objective(model, Min,
        T
        + p.lambda_jitter_ua  * sum((ua[j+1]-ua[j])^2 for j=10:(n-10))
        + p.lambda_jitter_pos * sum((s[j+1]-s[j])^2  for j=1:(n-1))
    )

    println("Starting 2D thermal optimization with $n_samples samples...")
    optimize!(model)
    println("Termination status (thermal 2D): ", termination_status(model))

    t_grid  = collect(0.0:1.0:(n-1)) .* value(T) / (n - 1)
    s_val   = value.(s)
    ux_val  = p.x_start .+ s_val .* ex
    uz_val  = p.z_start .+ s_val .* ez

    x_avg  = [sum(value(x_s[i,j])  for i=1:n_samples)/n_samples for j=1:n]
    z_avg  = [sum(value(z_s[i,j])  for i=1:n_samples)/n_samples for j=1:n]
    vx_avg = [sum(value(vx_s[i,j]) for i=1:n_samples)/n_samples for j=1:n]
    vz_avg = [sum(value(vz_s[i,j]) for i=1:n_samples)/n_samples for j=1:n]

    protocol = ControlProtocol2D(
        t_grid, x_avg, z_avg, vx_avg, vz_avg,
        ux_val, uz_val, value.(ua),
    )

    x_traj  = zeros(n_samples, n)
    z_traj  = zeros(n_samples, n)
    vx_traj = zeros(n_samples, n)
    vz_traj = zeros(n_samples, n)
    for i in 1:n_samples, j in 1:n
        x_traj[i,j]  = value(x_s[i,j])
        z_traj[i,j]  = value(z_s[i,j])
        vx_traj[i,j] = value(vx_s[i,j])
        vz_traj[i,j] = value(vz_s[i,j])
    end

    return ThermalControlResult2D(
        protocol,
        samples,
        x_traj, z_traj,
        vx_traj, vz_traj,
        seed,
        n_samples,
        string(termination_status(model)),
        objective_value(model),
    )
end

end
