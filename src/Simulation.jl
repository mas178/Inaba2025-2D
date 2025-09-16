module Simulation

using Base.Threads
using Agents
using Dates
using Random
using Statistics: mean, std

println("Julia: $(VERSION)")
println("Number of threads: ", nthreads())

#------------
# Agent
#------------
@enum Strategy C D

@agent struct Agent(GridAgent{2})
    s::Strategy
    π::Float64  # payoff
    ω::Float64  # fitness

    Agent(id::Int, pos::Dims{2}, s::Strategy) = new(id, pos, s, 0.0, 0.0)
    Agent(id::Int, pos::Dims{2}) = Agent(id, pos, D)
end

#------------
# Model
#------------
Base.@kwdef mutable struct ModelProperties
    # Space
    dims::Dims{2} = (100, 100)
    periodic_space::Bool = true
    distance_metric::Symbol = :euclidean  # :manhattan (Neumann (4)), :chebyshev (Moore (8)), :euclidean (Moore (8))

    # Agent
    N::Int = 7_000
    ϕC0::Float64 = 0.5

    # EV
    prob_EV::Float64 = 0.5
    prob_EV_t::Float64 = 0.5
    λ::Float64 = 0.0                      # inverse θ noise scale
    cycle::Int = 0                        # EV cycle
    sor_vec::Vector{Dims{2}} = [(1, 1)]   # SoR: Source(s) of resources
    θ_mat::Matrix{Float64} = zeros(0, 0)  # θ: survival threshold
    EV_spans::Vector{Tuple{Float64, Float64}} = Tuple{Float64, Float64}[]

    # Game
    T::Float64 = 1.1  # Temptation of defection
    S::Float64 = 0.0  # Sucker's payoff
    payoff_table::Dict{Tuple, Tuple} = Dict{Tuple, Tuple}()

    # Move
    prob_move::Float64 = 0.5
    sor_orientation::Float64 = 0.1

    # Strategy update
    x0::Float64 = 2.0
    μ::Float64 = 0.01 # probability for mutation

    # Simulation
    generation::Int = 1_000
    trial::Int = 10
    seed::Int = -1

    # Log
    snapshots::Vector{Int} = Vector{Int}()
end

function to_csv(p::ModelProperties)::Tuple{String, String}
    fields = fieldnames(ModelProperties)[[1:6; 8:10; 13:14; 16:21]]
    header = join(fields, ",")
    values = join([replace(string(getfield(p, field)), "," => " ") for field in fields], ",")

    return header, values
end

# Helper function to calculate spans
function calc_ev_spans(generation::Int, cycle::Int)::Vector{Tuple{Float64, Float64}}
    return [(generation * (c - 1) / cycle, generation * c / cycle) for c in 1:cycle]
end;

function initialize_model(; properties::ModelProperties)::ABM
    # update random seed by nano time and thread id
    properties.seed = properties.seed > 0 ? properties.seed : time_ns() + Threads.threadid()
    Random.seed!(properties.seed)
    # @show properties.seed

    # Population
    size_x, size_y = properties.dims
    N = size_x * size_y > properties.N ? properties.N : size_x * size_y
    nC = round(Int, N * properties.ϕC0)

    # Metric and neighborhood
    # - manhattan: Von Neumann neighborhood (4 neighbors)
    # - chebyshev: Moore neighborhood (8 neighbors)
    # - euclidean: Moore neighborhood (8 neighbors)
    @assert properties.distance_metric ∈ [:manhattan, :chebyshev, :euclidean]
    metric = properties.distance_metric == :manhattan ? :manhattan : :chebyshev

    # Initialize space and model
    space = GridSpaceSingle(properties.dims; periodic = properties.periodic_space, metric = metric)
    model = StandardABM(Agent, space, model_step! = run_one_generation!; properties)

    # Initialize agents (without duplicate position)
    strategies = vcat(fill(C, nC), fill(D, N - nC))
    for strategy in strategies
        add_agent_single!(Agent, model, strategy)
    end

    # EV
    model.EV_spans = calc_ev_spans(model.generation, model.cycle)
    model.prob_EV_t = model.prob_EV
    model.θ_mat = zeros(model.dims)

    # Game
    model.payoff_table = Dict(
        (C, C) => (1.0, 1.0),
        (D, C) => (model.T, model.S),
        (C, D) => (model.S, model.T),
        (D, D) => (0.0, 0.0),
        # (C, C) => (1.0, 1.0),
        # (D, C) => (1.0, 1.0),
        # (C, D) => (1.0, 1.0),
        # (D, D) => (1.0, 1.0),
    )

    return model
end

# Helpers for stats logging
ϕC(model::ABM)::Float64     = round(mean([a.s == C for a in allagents(model)]), digits=4)
mean_π(model::ABM)::Float64 = round(mean([a.π for a in allagents(model)]), digits=4)
std_π(model::ABM)::Float64  = round(std([a.π for a in allagents(model)]), digits=4)
mean_ω(model::ABM)::Float64 = round(mean([a.ω for a in allagents(model)]), digits=4)
std_ω(model::ABM)::Float64  = round(std([a.ω for a in allagents(model)]), digits=4)

#------------
# Environmental Variability
#------------
# Handle environmental variability (Stable/Variable Period)
function update_prob_EV!(model::ABM)::Nothing
    model.cycle <= 1 && return
    current_time = abmtime(model)
    
    for (i, (start_time, end_time)) in enumerate(model.EV_spans)
        if start_time <= current_time < end_time
            model.prob_EV_t = (i % 2 == 1) ? 0.0 : model.prob_EV
            return
        end
    end

    return
end

function move_sor!(model::ABM)::Nothing
    # calculate move count
    guaranteed_moves = floor(Int, model.prob_EV_t)        # Number of guaranteed moves
    additional_prob = model.prob_EV_t - guaranteed_moves  # Probability for one more move
    move_count = guaranteed_moves + (additional_prob > rand() ? 1 : 0)

    directions = if model.distance_metric == :manhattan
        [(0, 1), (0, -1), (1, 0), (-1, 0)]
    else
        [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    end

    for _ in 1:move_count
        model.sor_vec = [normalize_position(sor .+ rand(directions), model) for sor in model.sor_vec]
    end

    return
end

function calc_distance(
    sor::Tuple{Int, Int},
    size_x::Int,
    size_y::Int,
    metric::Symbol = :euclidean,  # :euclidean, :manhattan, or :chebyshev
    periodic::Bool = true,
)::Matrix{Float64}
    dx = abs.(collect(1:size_x) .- sor[1])
    dy = abs.(collect(1:size_y) .- sor[2])

    # Handle periodic boundaries
    if periodic
        dx = min.(dx, size_x .- dx)  
        dy = min.(dy, size_y .- dy)
    end

    if metric == :euclidean
        return sqrt.(dx.^2 .+ dy'.^2)
    elseif metric == :manhattan
        return dx .+ dy'
    elseif metric == :chebyshev
        return max.(dx, dy')
    else
        throw(ArgumentError("Unsupported metric: $metric. Use :euclidean, :manhattan, or :chebyshev."))
    end
end

function set_θ_mat!(model::ABM)::Nothing
    size_x, size_y = spacesize(model)

    d_mat = zeros(Float64, size_x, size_y)
    for sor in model.sor_vec
        d_mat .+= calc_distance(sor, size_x, size_y, model.distance_metric, model.periodic_space)
    end

    d_min = minimum(d_mat)
    d_max = maximum(d_mat)
    θ_mat = (d_mat .- d_min) ./ (d_max - d_min)

    θ_noise = model.λ > 0 ? (rand() * 2 - 1.0) / model.λ : 0.0 # λ: inverse θ noise scale

    model.θ_mat .= round.(θ_mat .+ θ_noise, digits=4)

    return
end

#------------
# Game
#------------
function play_game!(agent1::Agent, agent2::Agent, payoff_table::Dict)::Nothing
    π1, π2 = payoff_table[(agent1.s, agent2.s)]

    agent1.π += π1
    agent2.π += π2

    return
end

function calc_payoff_to_fitness!(model::ABM)::Nothing
    π_vec = [a.π for a in allagents(model)]
    k = 1.0
    ω_vec = 1 ./ (1 .+ exp.(-k .* (π_vec .- model.x0)))
    # ω_vec = 1 ./ (1 .+ exp.(-k .* (π_vec .- mean(π_vec))))

    for (agent, ω) in zip(allagents(model), ω_vec)
        agent.ω = round(ω, digits=4)
    end

    return
end

function reset_payoffs!(model::ABM)::Nothing
    foreach(a -> a.π = 0.0, allagents(model))

    return
end

function play_games!(model::ABM)::Nothing
    reset_payoffs!(model)

    # Each individual play games with all its neighbors (if any)
    for a_id in allids(model)
        for b_id in nearby_ids(model[a_id], model)
            if a_id < b_id
                play_game!(model[a_id], model[b_id], model.payoff_table)
            end
        end
    end

    calc_payoff_to_fitness!(model)

    return
end

#------------
# Move
#------------
function get_minimum_θ_destination(pos_vec::Vector{Dims{2}}, θ_mat::Matrix{Float64}, sor_orientation::Float64)::Dims{2}
    return if rand() > sor_orientation
        rand(pos_vec)
    else
        shuffled_pos_vec = shuffle(pos_vec)
        min_index = argmin([θ_mat[dest...] for dest in shuffled_pos_vec])
        shuffled_pos_vec[min_index]
    end
end

function move_one_agent!(agent::Agent, model::ABM, move_count::Int)::Nothing
    for _ in 1:move_count
        destinations = collect(empty_nearby_positions(agent, model))
        isempty(destinations) && return
        move_agent!(agent, get_minimum_θ_destination(destinations, model.θ_mat, model.sor_orientation), model)
    end

    return
end

function move!(model::ABM)::Nothing
    guaranteed_moves = floor(Int, model.prob_move)        # Number of guaranteed moves
    additional_prob = model.prob_move - guaranteed_moves  # Probability for one more move

    # Filter agents that meet the θ threshold
    agents = [a for a in allagents(model) if a.ω < model.θ_mat[a.pos...]]

    # Move each agent
    for a in agents
        move_count = guaranteed_moves + (additional_prob > rand() ? 1 : 0)
        move_one_agent!(a, model, move_count)
    end

    return
end

#------------
# Strategy update
#------------
flip(s::Strategy)::Strategy = s == C ? D : C

function update_strategy!(model::ABM)::Nothing
    # each player compares its resource with the ones of its neighbors and changes strategy,
    # following the one with the greatest payoff among them.
    agent_strategy_dict = Dict{Int, Strategy}()

    for agent in allagents(model)
        agent.ω > model.θ_mat[agent.pos...] && continue

        max_neighboring_agent = reduce(
            (a, b) -> a.ω > b.ω ? a : b,
            nearby_agents(agent, model),
            init = agent
        )

        agent_strategy_dict[agent.id] = model.μ < rand() ? max_neighboring_agent.s : flip(max_neighboring_agent.s)
    end

    for agent_id in keys(agent_strategy_dict)
        model[agent_id].s = agent_strategy_dict[agent_id]
    end

    return
end

#------------
# Integration
#------------
function run_one_generation!(model::ABM)::Nothing
    # Environmental variation
    update_prob_EV!(model)
    move_sor!(model)
    set_θ_mat!(model)

    # Agent Interaction
    play_games!(model)
    move!(model)
    update_strategy!(model)

    # Snapshot
    abmtime(model) ∈ model.snapshots && display_grid(model)

    return
end

function run_simulation!(properties::ModelProperties)::Matrix{Float64}
    mat_ϕC = fill(0.0, (properties.trial, properties.generation + 1))

    @threads for trial_num in 1:properties.trial
        model = initialize_model(; properties = deepcopy(properties)) # スレッド間でpropertiesを共有しないこと!!
        _, mdata = run!(model, model.generation, mdata = [ϕC])
        mat_ϕC[trial_num, :] .= mdata.ϕC
    end

    return mat_ϕC
end;

function calc_mean_std_ϕC(mat_c_rate::Matrix{Float64}, log_range::Float64 = 0.2)::Tuple{Float64, Float64}
    _, generation = size(mat_c_rate)
    start_gen = round(Int, (1 - log_range) * generation)
    selected_data = mat_c_rate[:, start_gen:generation]

    trial_means = mean(selected_data, dims=2)

    mean_ϕC = round(mean(trial_means), digits=4)
    std_ϕC = round(std(trial_means), digits=4)

    return mean_ϕC, std_ϕC
end

function run_simulation!(properties_vec::Vector{ModelProperties}, dir::String = "log")::Nothing
    # output file name
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    output_file = "$(dir)/$(timestamp).csv"
    println(output_file)

    open(output_file, "w") do file
        for (i, prop) in enumerate(properties_vec)
            print("$(i) ")

            # print header
            header, values = to_csv(prop)
            if i == 1
                header = "$(header),mean_ϕC,std_ϕC"
                println(file, header)
            end

            # execute simulation
            mat_ϕC = run_simulation!(prop)
            mean_ϕC, std_ϕC = calc_mean_std_ϕC(mat_ϕC)

            # print values
            result = "$(values),$(mean_ϕC),$(std_ϕC)"
            println(file, result)
            flush(file)
        end
    end
    println(Dates.format(now(), "\nyyyy/mm/dd_HH:MM:SS"))

    return
end

#------------
# Visualization
#------------
using CairoMakie

# Color Scheme
const RED = "#C7243A"
const BLUE = "#3261AB"

agent_color(agent::Agent)::String = agent.s == C ? BLUE : RED

# Display grid with agents and SoR
function display_grid(model::ABM; agent_size::Int = 5)::Nothing
    plotkwargs = (;
        agent_color = agent_color,
        agent_size = agent_size,
        agent_marker = :rect,
    )

    fig, ax, _ = abmplot(model; plotkwargs...)
    

    ax.title = "generation: $(abmtime(model)), prob_EV: $(model.prob_EV_t), prob_move: $(model.prob_move)"
    ax.xticksvisible = false  # x軸の目盛りを非表示
    ax.yticksvisible = false  # y軸の目盛りを非表示
    ax.xticklabelsvisible = false  # x軸目盛りのラベルを非表示
    ax.yticklabelsvisible = false  # y軸目盛りのラベルを非表示

    # draw circle around sor
    for (x, y) in model.sor_vec
        r = 4
        θ = range(0, 2π, length=100) # 円周を描くための角度の範囲
        xs = x .+ r .* cos.(θ)       # 円の x 座標
        ys = y .+ r .* sin.(θ)       # 円の y 座標
        grid_width, grid_height = spacesize(model) # Grid dimensions
        for dx in [-grid_width, 0, grid_width]
            for dy in [-grid_height, 0, grid_height]
                shifted_xs = xs .+ dx
                shifted_ys = ys .+ dy
                lines!(ax, shifted_xs, shifted_ys, color=:black, linewidth=1.5)
            end
        end
    end

    display(fig)
    save("img/$(abmtime(model)).png", fig)

    return
end

end