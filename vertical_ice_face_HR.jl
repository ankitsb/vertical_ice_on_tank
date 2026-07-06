using Oceananigans
using Oceananigans.Units: second, minutes, hour, hours, day
using Oceananigans.Operators: ∂xᶜᶜᶜ, ∂zᶜᶜᶜ
using CUDA
using Random
# using SeawaterPolynomials.TEOS10
using Printf
using NCDatasets

Random.seed!(1234) # for reproducible results

Nx, Ny, Nz = 512, 64, 256
Lx, Ly, Lz = 10, 5, 10

## Creates a grid with near-constant spacing `refinement * Lz / Nz`
## near the bottom:
refinement = 3 # controls spacing near surface (higher means finer spaced)
stretching = 10  # controls rate of stretching at bottom
## "Warped" height coordinate
h(i) = (Nx + 1 - i) / Nx
## Linear near-surface generator
ζ(i) = 1 + (h(i) - 1) / refinement
## Bottom-intensified stretching function
Σ(i) = (1 - exp(-stretching * h(i))) / (1 - exp(-stretching))
## Generating function
x_faces(i) = - Lx * (ζ(i) * Σ(i) - 1)

grid = RectilinearGrid(GPU(),
                        size = (Nx, Ny, Nz),
                        x = x_faces,
                        y = (0, Ly), z = (0, Lz),
                        topology=(Bounded, Periodic, Bounded))

@info "Build a grid:"
@show grid

# ## Buoyancy that depends on temperature and salinity
# We use the `SeawaterBuoyancy` model with the TEOS10 equation of state,
h_param = (
            ρₒ = 1027.44, # kg m⁻³, average density at the surface of the ocean
            cᴾ = 4180, # J K⁻¹ kg⁻¹, typical heat capacity for seawater
            ρᵢ = 920, # kg m⁻³, average density of sea-ice
            Lᶠ = 3.35e5, # J kg⁻¹, Latent heat of water: Ramudu et al., 2017
    
            a = - 6e-2, #5.72e-2, # ⁰C psu⁻¹, Freezing point coefficient
            b = 0.0, #9.39e-2, # ⁰C, Freezing point coefficient
    
            Sᵢ = 4, # ice core salinity 4
            Tᵢₙᵢₜ = - 0.0, # ⁰C, Interface temperature
            Sᵢₙᵢₜ = 5, # psu, from equation: Tᵢ = a*Sᵢ + b

            ν = 1e-5, # m²s⁻¹, Kinematic viscosity of seawater
            κₜ = 1.4e-6, # m²s⁻¹, molecular thermal diffusivity
            κₛ = 1.3e-8, # m²s⁻¹, molecular salt diffusivity
)

# equation_of_state = TEOS10EquationOfState(reference_density=h_param.ρₒ)
# buoyancy = SeawaterBuoyancy(; equation_of_state)

eos = LinearEquationOfState(thermal_expansion  = 3.8e-5,
                            haline_contraction = 7.8e-4)
buoyancy = SeawaterBuoyancy(equation_of_state=eos)

# ## Boundary conditions

@inline function boundary_salt(j, k, grid, clock, model_fields, p)
    
    x = xspacings(grid, Center(), Center(), Center())
    Δx = 0.5(x[1] + x[2])
    
    Sᵢₙₜ =  ((model_fields.S[1, j, k] - model_fields.S[1, j, k]) / Δx * (0.5*x[1]) + model_fields.S[1, j, k])

    return @inbounds Sᵢₙₜ
end

@inline function melt_rate(j, k, grid, clock, model_fields, p)
    # ρ = surf_dens(i, j, grid.Nz, grid, clock, model_fields, p)
    ρ = p.ρₒ
    
    ∂T_∂x = - ∂xᶜᶜᶜ(1, j, k, grid, model_fields.T)

    Qʰ = @. ρ * p.cᴾ * p.κₜ * ∂T_∂x

    wⁱ = Qʰ ./ (p.ρᵢ .* p.Lᶠ)

    return @inbounds wⁱ
end

@inline function salt_grad(j, k, grid, clock, model_fields, p)
    
    # ρ = surf_dens(i, j, grid.Nz, grid, clock, model_fields, p)
    ρ = p.ρₒ

    wⁱ = melt_rate(j, k, grid, clock, model_fields, p)
    
    Sᵢ = boundary_salt(j, k, grid, clock, model_fields, p)

    ∂S_∂x = @. p.ρᵢ .* wⁱ .* Sᵢ ./ (ρ * p.κₛ)

    return @inbounds -∂S_∂x  # [salinity unit] m⁻¹
end

S_boundary = GradientBoundaryCondition(salt_grad, discrete_form=true, parameters=h_param)
S_bcs = FieldBoundaryConditions(west = S_boundary)


@inline freez_T(S, p) = @. p.a * S + p.b

@inline function boundary_temperature(j, k, grid, clock, model_fields, p)

    Sᵢ =  boundary_salt(j, k, grid, clock, model_fields, p)
    Tᵢ = freez_T(Sᵢ, p)

    return @inbounds Tᵢ
end


T_boundary = ValueBoundaryCondition(boundary_temperature, discrete_form=true, parameters=h_param)
T_bcs = FieldBoundaryConditions(west = T_boundary)

u_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(0), bottom=ValueBoundaryCondition(0))
v_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(0), bottom=ValueBoundaryCondition(0),
                                east = ValueBoundaryCondition(0), west=ValueBoundaryCondition(0),)
w_bcs = FieldBoundaryConditions(east = ValueBoundaryCondition(0), west=ValueBoundaryCondition(0),)


molecular_diffusivity = ScalarDiffusivity(ν=h_param.ν, κ=(T = h_param.κₜ, S = h_param.κₛ))
closure = (AnisotropicMinimumDissipation(), molecular_diffusivity)

boundary_conditions = (T=T_bcs, S=S_bcs, u=u_bcs, v=v_bcs, w=w_bcs)

# coriolis = FPlane(rotation_rate=7.292115e-5, latitude=-65.5) ## for Earth
coriolis = nothing

model = NonhydrostaticModel(grid; advection = WENO(order=9),
                            timestepper = :RungeKutta3,
                            tracers = (:T, :S),
                            coriolis = coriolis,
                            buoyancy = buoyancy,
                            closure = closure,
                            boundary_conditions = boundary_conditions,
)

@info "Constructed a model"
@show model

# Velocity initial condition: random noise scaled by the friction velocity.
# Ξ(x) = randn() * x / Lx * (tanh(x/5)+1) / 0.05 # noise
Ξ(x) = randn() * exp(-(10*x)^2 / grid.Lx^2)

@inline vᵢₙᵢₜ(x, y, z) = @inbounds 1e-5
@inline uᵢₙᵢₜ(x, y, z) = @inbounds 1e-4 * Ξ(x)
@inline wᵢₙᵢₜ(x, y, z) = @inbounds 1e-4 * Ξ(x)

# Temperature initial condition: a stable density gradient with random noise superposed.
@inline Tᵢₙᵢₜ(x, y, z) = h_param.Tᵢₙᵢₜ + 1e-5 * Ξ(x)
@inline Sᵢₙᵢₜ(x, y, z) = h_param.Sᵢₙᵢₜ + 1e-3 * Ξ(x)

# `set!` the `model` fields using functions or constants:
set!(model, u=uᵢₙᵢₜ, w=wᵢₙᵢₜ, S=Sᵢₙᵢₜ, T=Tᵢₙᵢₜ)  

@info "model initial conditions are Set!!"


wall_clock = time_ns()
function progress_message(simulation)
    model = simulation.model
    u, v, w = model.velocities

    CFL_a = simulation.Δt / wizard.cell_advection_timescale(model)

    # Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax: (%.1e, %.1e, %.1e) ms⁻¹, CFL: %.2e, wall time: %s\n",
                   model.clock.iteration,
                   prettytime(model.clock.time),
                   prettytime(simulation.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w), CFL_a,
                   prettytime(1e-9 * (time_ns() - wall_clock))
                  )
        @info msg
    return nothing
end


simulation = Simulation(model, Δt=1e-4, stop_time=120minutes)

wizard = TimeStepWizard(cfl=0.9, max_change=1.01, max_Δt=1second)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(1))
simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(200))

u = @at (Center, Center, Center) model.velocities.u
v = @at (Center, Center, Center) model.velocities.v
w = @at (Center, Center, Center) model.velocities.w
T = model.tracers.T
S = model.tracers.S

dir = "."
simulation.output_writers[:threeD] =
NetCDFWriter(model, (; T, S),
                     schedule = TimeInterval(3minutes),
                     filename = joinpath(dir, "vertical_ice_face_3Dsolution_v1.nc"),
                     # overwrite_existing = true
)

simulation.output_writers[:plane] =
NetCDFWriter(model, (; u, v, w, T, S),
                     schedule = TimeInterval(0.25minutes),
                     filename = joinpath(dir, "vertical_ice_face_2Dsolution_v1.nc"),
                     indices = (:, 1, :),
                     # overwrite_existing = true
)


# #####
# ##### Build checkpointer and output writer
# #####
simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(10minutes),
                                                        prefix = "checkpoint",
                                                        cleanup=true)
@info "Output files gererated"

run!(simulation, pickup=true)
# run!(simulation)