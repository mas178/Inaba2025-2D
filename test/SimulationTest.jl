module SimulationTest

using Base.Threads
using Agents
using Statistics: mean

using Test: @testset, @test

include("../src/Simulation.jl")

#------------
# Agent
#------------
using .Simulation: Agent, C, D

@testset "Agent and agent_color" begin
    agent = Agent(1, (1, 1))
    @test agent.id == 1
    @test agent.pos == (1, 1)
    @test agent.s == D
    @test agent.π == 0.0
    @test agent.ω == 0.0

    agent.s = C

    agent.π = 1.1
    @test agent.π == 1.1

    agent.ω = 2.2
    @test agent.ω == 2.2
end

#------------
# Model
#------------
using .Simulation: ModelProperties, to_csv, calc_ev_spans, initialize_model, ϕC, mean_π, std_π, mean_ω, std_ω

@testset "to_csv" begin
    header, values = to_csv(ModelProperties())
    @test header == "dims,periodic_space,distance_metric,N,ϕC0,prob_EV,λ,cycle,sor_vec,T,S,prob_move,sor_orientation,x0,μ,generation,trial"
    @test values == "(100  100),true,euclidean,7000,0.5,0.5,0.0,0,[(1  1)],1.1,0.0,0.5,0.1,2.0,0.01,1000,10"
end

@testset "calc_ev_spans" begin
    @test calc_ev_spans(100, 0) == Tuple{Float64, Float64}[]
    @test calc_ev_spans(100, 1) == [(0.0, 100.0)]
    @test calc_ev_spans(100, 3) == [(0.0, 33.333333333333336), (33.333333333333336, 66.66666666666667), (66.66666666666667, 100.0)]
end;

@testset "Model" begin
    model = initialize_model(properties = ModelProperties(sor_vec = [(25, 50), (75, 50)]))
    @show model
    @test nagents(model) == 7000
    @test ϕC(model) == 0.5
    @test model.prob_EV == 0.5
    @test model.prob_EV_t == 0.5
    @test model.prob_move == 0.5
    @test model.λ == 0.0
    @test model.payoff_table == Dict(
        (C, C) => (1.0, 1.0),
        (D, C) => (1.1, 0.0),
        (C, D) => (0.0, 1.1),
        (D, D) => (0.0, 0.0)
    )
    @test model.dims == size(model.θ_mat) == (100, 100)

    # エージェントの重複配置チェック
    @test length([a.pos for a in allagents(model)]) == length(Set([a.pos for a in allagents(model)])) == 7000
    
    @testset "Log" begin
        model = initialize_model(properties = ModelProperties(N = 6))

        for i in 1:6
            model[i].s = i % 3 == 1 ? C : D
            model[i].π = i / 6
            model[i].ω = i / 12
        end

        @test ϕC(model) == 0.3333
        @test mean_π(model) == 0.5833
        @test std_π(model) == 0.3118
        @test mean_ω(model) == 0.2917
        @test std_ω(model) == 0.1559
    end
end

@testset "Model (multithread)" begin
    thread_count = 10
    properties = ModelProperties()

    my_lock = ReentrantLock()
    
    objectid_model_vec = []
    
    @threads for thread_num in 1:thread_count
        # スレッド間でpropertiesを共有しないこと!!
        local_properties = deepcopy(properties)
        model = initialize_model(; properties = local_properties)
        @lock my_lock push!(objectid_model_vec, objectid(model))
    end

    @test length(objectid_model_vec) == length(Set(objectid_model_vec)) == thread_count
end

#------------
# Environmental Variability
#------------
using .Simulation: update_prob_EV!, move_sor!, calc_distance, calc_temperature, set_θ_mat!

@testset "update_prob_EV! tests" begin
    # cycle <= 1 の場合
    model = initialize_model(properties = ModelProperties(; prob_EV = 0.123, cycle = 1, generation = 10))
    @test model.EV_spans == [(0.0, 10.0)]

    @test abmtime(model) == 0
    update_prob_EV!(model)
    @test model.prob_EV_t == 0.123
    
    step!(model, 5)
    @test abmtime(model) == 5
    update_prob_EV!(model)
    @test model.prob_EV_t == 0.123

    step!(model, 5)
    @test abmtime(model) == 10
    update_prob_EV!(model)
    @test model.prob_EV_t == 0.123

    # cycle > 1 の場合
    model = initialize_model(properties = ModelProperties(; prob_EV = 0.123, cycle = 4, generation = 10))
    @test model.EV_spans == [(0.0, 2.5), (2.5, 5.0), (5.0, 7.5), (7.5, 10.0)]

    @test abmtime(model) == 0
    update_prob_EV!(model)
    @test model.prob_EV_t == 0

    step!(model, 2)
    @test abmtime(model) == 2
    update_prob_EV!(model)
    @test model.prob_EV_t == 0

    step!(model, 1)
    @test abmtime(model) == 3
    update_prob_EV!(model)
    @test model.prob_EV_t == 0.123

    step!(model, 2)
    @test abmtime(model) == 5
    update_prob_EV!(model)
    @test model.prob_EV_t == 0

    step!(model, 1)
    @test abmtime(model) == 6
    update_prob_EV!(model)
    @test model.prob_EV_t == 0

    step!(model, 1)
    @test abmtime(model) == 7
    update_prob_EV!(model)
    @test model.prob_EV_t == 0

    step!(model, 1)
    @test abmtime(model) == 8
    update_prob_EV!(model)
    @test model.prob_EV_t == 0.123

    step!(model, 2)
    @test abmtime(model) == 10
    update_prob_EV!(model)
    @test model.prob_EV_t == 0.123

    step!(model, 1)
    @test abmtime(model) == 11
    update_prob_EV!(model)
    @test model.prob_EV_t == 0.123
end

@testset "move_sor! (periodic_space = true, distance_metric = :manhattan)" begin
    @testset "prob_EV_t = 0.0" begin
        model = initialize_model(properties = ModelProperties(dims = (4, 4), seed = 1, prob_EV = 0.0, distance_metric = :manhattan))
        sor_vec_vec = [[(1, 1)]]
        for _ in 1:10
            move_sor!(model)
            push!(sor_vec_vec, model.sor_vec)
        end
        @test sor_vec_vec == [[(1, 1)], [(1, 1)], [(1, 1)], [(1, 1)], [(1, 1)], [(1, 1)], [(1, 1)], [(1, 1)], [(1, 1)], [(1, 1)], [(1, 1)]]
    end

    @testset "prob_EV_t = 0.5" begin
        model = initialize_model(properties = ModelProperties(dims = (4, 4), seed = 1, prob_EV = 0.5, distance_metric = :manhattan))
        sor_vec_vec = [[(1, 1)]]
        for _ in 1:10
            move_sor!(model)
            push!(sor_vec_vec, model.sor_vec)
        end
        @test sor_vec_vec == [[(1, 1)], [(1, 1)], [(4, 1)], [(4, 4)], [(4, 1)], [(4, 1)], [(1, 1)], [(1, 1)], [(2, 1)], [(3, 1)], [(3, 1)]]
    end

    @testset "prob_EV_t = 1.0" begin
        model = initialize_model(properties = ModelProperties(dims = (4, 4), seed = 1, prob_EV = 1.0, distance_metric = :manhattan))
        sor_vec_vec = [[(1, 1)]]
        for _ in 1:10
            move_sor!(model)
            push!(sor_vec_vec, model.sor_vec)
        end
        @test sor_vec_vec == [[(1, 1)], [(1, 4)], [(1, 3)], [(1, 4)], [(4, 4)], [(1, 4)], [(1, 1)], [(1, 2)], [(4, 2)], [(1, 2)], [(4, 2)]]
    end

    @testset "prob_EV_t = 1.5" begin
        model = initialize_model(properties = ModelProperties(dims = (4, 4), seed = 1, prob_EV = 1.5, distance_metric = :manhattan))
        sor_vec_vec = [[(1, 1)]]
        for _ in 1:10
            move_sor!(model)
            push!(sor_vec_vec, model.sor_vec)
        end
        @test sor_vec_vec == [[(1, 1)], [(1, 4)], [(1, 3)], [(1, 1)], [(1, 4)], [(2, 4)], [(3, 1)], [(2, 1)], [(3, 1)], [(2, 1)], [(3, 1)]]
    end

    @testset "prob_EV_t = 1.0" begin
        model = initialize_model(properties = ModelProperties(dims = (4, 4), seed = 1, prob_EV = 1.0, sor_vec = [(1, 1), (1, 2)], distance_metric = :manhattan))
        sor_vec_vec = [[(1, 1), (1, 2)]]
        for _ in 1:10
            move_sor!(model)
            push!(sor_vec_vec, model.sor_vec)
        end
        @test sor_vec_vec == [[(1, 1), (1, 2)], [(1, 4), (4, 2)], [(1, 3), (4, 3)], [(4, 3), (4, 2)], [(1, 3), (4, 3)],
            [(1, 4), (1, 3)], [(2, 4), (2, 3)], [(1, 4), (1, 3)], [(1, 1), (1, 2)], [(4, 1), (1, 3)], [(4, 4), (4, 3)]]
    end
end;

@testset "move_sor! (periodic_space = true or false, distance_metric = :manhattan, :chebyshev or :euclidean)" begin
    @testset "periodic_space = true, distance_metric = :manhattan" begin
        model = initialize_model(properties = ModelProperties(dims = (4, 4), seed = 1, prob_EV = 1.0, periodic_space = true, distance_metric = :manhattan))
        # _, mdata = run!(model, 10, mdata = [:sor_vec])
        sor_vec_vec = [[(1, 1)]]
        for _ in 1:10
            move_sor!(model)
            push!(sor_vec_vec, model.sor_vec)
        end
        @test sor_vec_vec == [[(1, 1)], [(1, 4)], [(1, 3)], [(1, 4)], [(4, 4)], [(1, 4)], [(1, 1)], [(1, 2)], [(4, 2)], [(1, 2)], [(4, 2)]]
    end

    @testset "periodic_space = false, distance_metric = :manhattan" begin
        model = initialize_model(properties = ModelProperties(dims = (4, 4), seed = 1, prob_EV = 1.0, periodic_space = false, distance_metric = :manhattan))
        sor_vec_vec = [[(1, 1)]]
        for _ in 1:10
            move_sor!(model)
            push!(sor_vec_vec, model.sor_vec)
        end
        @test sor_vec_vec == [[(1, 1)], [(1, 1)], [(1, 1)], [(1, 2)], [(1, 2)], [(2, 2)], [(2, 3)], [(2, 4)], [(1, 4)], [(2, 4)], [(1, 4)]]
    end

    @testset "periodic_space = true, distance_metric = :chebyshev" begin
        model = initialize_model(properties = ModelProperties(dims = (4, 4), seed = 1, prob_EV = 1.0, periodic_space = true, distance_metric = :chebyshev))
        sor_vec_vec = [[(1, 1)]]
        for _ in 1:10
            move_sor!(model)
            push!(sor_vec_vec, model.sor_vec)
        end
        @test sor_vec_vec == [[(1, 1)], [(4, 2)], [(3, 3)], [(4, 3)], [(1, 2)], [(4, 1)], [(1, 1)], [(2, 2)], [(2, 1)], [(1, 4)], [(1, 3)]]
    end

    @testset "periodic_space = false, distance_metric = :chebyshev" begin
        model = initialize_model(properties = ModelProperties(dims = (4, 4), seed = 1, prob_EV = 1.0, periodic_space = false, distance_metric = :chebyshev))
        sor_vec_vec = [[(1, 1)]]
        for _ in 1:10
            move_sor!(model)
            push!(sor_vec_vec, model.sor_vec)
        end
        @test sor_vec_vec == [[(1, 1)], [(1, 2)], [(1, 3)], [(2, 3)], [(3, 2)], [(2, 1)], [(3, 1)], [(4, 2)], [(4, 1)], [(3, 1)], [(3, 1)]]
    end

    @testset "periodic_space = true, distance_metric = :euclidean" begin
        model = initialize_model(properties = ModelProperties(dims = (4, 4), seed = 1, prob_EV = 1.0, periodic_space = true, distance_metric = :euclidean))
        sor_vec_vec = [[(1, 1)]]
        for _ in 1:10
            move_sor!(model)
            push!(sor_vec_vec, model.sor_vec)
        end
        @test sor_vec_vec == [[(1, 1)], [(4, 2)], [(3, 3)], [(4, 3)], [(1, 2)], [(4, 1)], [(1, 1)], [(2, 2)], [(2, 1)], [(1, 4)], [(1, 3)]]
    end

    @testset "periodic_space = false, distance_metric = :euclidean" begin
        model = initialize_model(properties = ModelProperties(dims = (4, 4), seed = 1, prob_EV = 1.0, periodic_space = false, distance_metric = :euclidean))
        sor_vec_vec = [[(1, 1)]]
        for _ in 1:10
            move_sor!(model)
            push!(sor_vec_vec, model.sor_vec)
        end
        @test sor_vec_vec == [[(1, 1)], [(1, 2)], [(1, 3)], [(2, 3)], [(3, 2)], [(2, 1)], [(3, 1)], [(4, 2)], [(4, 1)], [(3, 1)], [(3, 1)]]
    end
end

@testset "calc_distance" begin
    @test round.(calc_distance((1, 1), 4, 4), digits = 3) == [
        0.0 1.0   2.0   1.0
        1.0 1.414 2.236 1.414
        2.0 2.236 2.828 2.236
        1.0 1.414 2.236 1.414
    ]

    @test round.(calc_distance((1, 1), 3, 4), digits = 3) == [
        0.0 1.0   2.0   1.0
        1.0 1.414 2.236 1.414
        1.0 1.414 2.236 1.414
    ]

    @test round.(calc_distance((1, 1), 4, 4, :manhattan, true), digits = 3) == [
        0.0 1.0 2.0 1.0
        1.0 2.0 3.0 2.0
        2.0 3.0 4.0 3.0
        1.0 2.0 3.0 2.0
    ]

    @test round.(calc_distance((1, 1), 4, 4, :manhattan, false), digits = 3) == [
        0.0 1.0 2.0 3.0
        1.0 2.0 3.0 4.0
        2.0 3.0 4.0 5.0
        3.0 4.0 5.0 6.0
    ]

    @test round.(calc_distance((1, 1), 4, 4, :chebyshev, true), digits = 3) == [
        0.0 1.0 2.0 1.0
        1.0 1.0 2.0 1.0
        2.0 2.0 2.0 2.0
        1.0 1.0 2.0 1.0
    ]

    @test round.(calc_distance((1, 1), 4, 4, :chebyshev, false), digits = 3) == [
        0.0 1.0 2.0 3.0
        1.0 1.0 2.0 3.0
        2.0 2.0 2.0 3.0
        3.0 3.0 3.0 3.0
    ]
end

@testset "calc_temperature" begin
    @test round.(calc_temperature((1, 1), 2, 2), digits = 3) == [
        1.0   0.293
        0.293 0.0
    ]

    @test round.(calc_temperature((1, 1), 3, 3), digits = 3) == [
        1.0   0.293 0.293
        0.293 0.0   0.0
        0.293 0.0   0.0
    ]

    @test round.(calc_temperature((1, 1), 3, 3, :manhattan), digits = 3) == [
        1.0 0.5 0.5
        0.5 0.0 0.0
        0.5 0.0 0.0
    ]

    @test round.(calc_temperature((1, 1), 3, 3, :chebyshev), digits = 3) == [
        1.0 0.0 0.0
        0.0 0.0 0.0
        0.0 0.0 0.0
    ]

    @test round.(calc_temperature((5, 5), 10, 10), digits = 3) == [
        0.2   0.293 0.368 0.417 0.434 0.417 0.368 0.293 0.2   0.094;
        0.293 0.4   0.49  0.553 0.576 0.553 0.49  0.4   0.293 0.175;
        0.368 0.49  0.6   0.684 0.717 0.684 0.6   0.49  0.368 0.238;
        0.417 0.553 0.684 0.8   0.859 0.8   0.684 0.553 0.417 0.279;
        0.434 0.576 0.717 0.859 1.0   0.859 0.717 0.576 0.434 0.293;
        0.417 0.553 0.684 0.8   0.859 0.8   0.684 0.553 0.417 0.279;
        0.368 0.49  0.6   0.684 0.717 0.684 0.6   0.49  0.368 0.238;
        0.293 0.4   0.49  0.553 0.576 0.553 0.49  0.4   0.293 0.175;
        0.2   0.293 0.368 0.417 0.434 0.417 0.368 0.293 0.2   0.094;
        0.094 0.175 0.238 0.279 0.293 0.279 0.238 0.175 0.094 0.0
    ]
end

@testset "set_θ_mat!" begin
    model = initialize_model(properties = ModelProperties(dims = (2, 2), sor_vec = [(1, 1), (1, 2)], distance_metric = :manhattan))
    set_θ_mat!(model)
    @test model.θ_mat == [
        0.0 0.0
        1.0 1.0
    ]

    model = initialize_model(properties = ModelProperties(dims = (2, 2), sor_vec = [(1, 1), (1, 2)], λ = 10, seed = 1, distance_metric = :manhattan))
    set_θ_mat!(model)
    @test model.θ_mat == [
        -0.0614 -0.0614
         0.9386  0.9386
    ]

    model = initialize_model(properties = ModelProperties(dims = (2, 2), sor_vec = [(1, 1), (1, 2)], λ = 100, seed = 1, distance_metric = :manhattan))
    set_θ_mat!(model)
    @test model.θ_mat == [
        -0.0061 -0.0061
         0.9939  0.9939
    ]

    model = initialize_model(properties = ModelProperties(dims = (3, 4), distance_metric = :manhattan))
    set_θ_mat!(model)
    @test model.θ_mat == [
        0.0    0.3333 0.6667 0.3333
        0.3333 0.6667 1.0    0.6667
        0.3333 0.6667 1.0    0.6667
    ]

    model.sor_vec = [(2, 2)]
    set_θ_mat!(model)
    @test model.θ_mat == [
        0.6667 0.3333 0.6667 1.0
        0.3333 0.0    0.3333 0.6667
        0.6667 0.3333 0.6667 1.0
    ]

    model.distance_metric = :chebyshev
    set_θ_mat!(model)
    @test model.θ_mat == [
        0.5 0.5 0.5 1.0
        0.5 0.0 0.5 1.0
        0.5 0.5 0.5 1.0
    ]

    model.distance_metric = :euclidean
    set_θ_mat!(model)
    @test model.θ_mat == [
        0.6325 0.4472 0.6325 1.0
        0.4472 0.0    0.4472 0.8944
        0.6325 0.4472 0.6325 1.0
    ]

    for distance_metric in [:manhattan, :euclidean, :chebyshev]
        model = initialize_model(properties = ModelProperties(; dims = (400, 400), sor_vec = [(200, 200)], distance_metric))
        set_θ_mat!(model)
    end

    for distance_metric in [:manhattan, :euclidean, :chebyshev]
        model = initialize_model(properties = ModelProperties(; dims = (400, 400), sor_vec = [(200, 100), (200, 300)], distance_metric))
        set_θ_mat!(model)
    end
end

#------------
# Game
#------------
using .Simulation: play_game!, calc_payoff_to_fitness!, reset_payoffs!, play_games!

@testset "play_game" begin
    agent1 = Agent(1, (1, 1), C)
    agent2 = Agent(2, (2, 2), C)
    agent3 = Agent(3, (3, 3), D)
    agent4 = Agent(4, (4, 4), D)

    agent1.π = 0.8
    agent2.π = 1.0
    agent3.π = 1.0
    agent4.π = 1.0

    # C vs C
    b = 1.2
    payoff_table = Dict(
        (C, C) => (1.0, 1.0),
        (D, C) => (b, 0.0),
        (C, D) => (0.0, b),
        (D, D) => (0.0, 0.0)
    )
    play_game!(agent1, agent2, payoff_table)
    @test agent1.π == 0.8 + 1.0 == 1.8
    @test agent2.π == 1.0 + 1.0 == 2.0

    # C vs D
    b = 1.3
    payoff_table = Dict(
        (C, C) => (1.0, 1.0),
        (D, C) => (b, 0.0),
        (C, D) => (0.0, b),
        (D, D) => (0.0, 0.0)
    )
    play_game!(agent2, agent3, payoff_table)
    @test agent2.π == 2.0
    @test agent3.π == 1.0 + 1.3 == 2.3

    # D vs D
    b = 1.4
    payoff_table = Dict(
        (C, C) => (1.0, 1.0),
        (D, C) => (b, 0.0),
        (C, D) => (0.0, b),
        (D, D) => (0.0, 0.0)
    )
    play_game!(agent3, agent4, payoff_table)
    @test agent3.π == 2.3
    @test agent4.π == 1.0

    # D vs C
    b = 1.5
    payoff_table = Dict(
        (C, C) => (1.0, 1.0),
        (D, C) => (b, 0.0),
        (C, D) => (0.0, b),
        (D, D) => (0.0, 0.0)
    )
    play_game!(agent4, agent1, payoff_table)
    @test agent4.π == 1.0 + 1.5 == 2.5
    @test agent1.π == 1.8
end

@testset "calc_payoff_to_fitness!" begin
    model = initialize_model(properties = ModelProperties(dims = (3, 3), N = 9))
    foreach(a -> a.π = 0.0, allagents(model))
    calc_payoff_to_fitness!(model)
    @test all(a -> a.π == 0.0, allagents(model))
    @test all(a -> a.ω == 0.1192, allagents(model))

    foreach(a -> a.π = 2.0, allagents(model))
    calc_payoff_to_fitness!(model)
    @test all(a -> a.π == 2.0, allagents(model))
    @test all(a -> a.ω == 0.5, allagents(model))
    
    foreach(a -> a.π = a.id - 1, allagents(model))
    calc_payoff_to_fitness!(model)
    π_vec = sort([a.π for a in allagents(model)])
    ω_vec = sort([a.ω for a in allagents(model)])
    @test π_vec == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    @test ω_vec == [0.1192, 0.2689, 0.5, 0.7311, 0.8808, 0.9526, 0.982, 0.9933, 0.9975]
end

@testset "reset_payoffs!" begin
    model = initialize_model(properties = ModelProperties())
    foreach(a -> a.π = rand(), allagents(model))
    @test nagents(model) == 7_000
    @test all(a -> a.π > 0.0, allagents(model))
    reset_payoffs!(model)
    @test all(a -> a.π == 0.0, allagents(model))
end

@testset "play_games! (distance_metric = :manhattan)" begin
    model = initialize_model(properties = ModelProperties(dims = (3, 3), N = 6, ϕC0 = 0, distance_metric = :manhattan))
    play_games!(model)
    @test [a.π for a in allagents(model)] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test [a.ω for a in allagents(model)] == [0.1192, 0.1192, 0.1192, 0.1192, 0.1192, 0.1192]

    model = initialize_model(properties = ModelProperties(dims = (3, 3), N = 6, ϕC0 = 1, seed = 1, distance_metric = :manhattan))
    play_games!(model)
    @test [a.π for a in allagents(model)] == [2.0, 2.0, 1.0, 3.0, 3.0, 3.0]
    @test [a.ω for a in allagents(model)] == [0.5, 0.5, 0.2689, 0.7311, 0.7311, 0.7311]

    model = initialize_model(properties = ModelProperties(dims = (3, 3), N = 6, ϕC0 = 0.5, seed = 1, distance_metric = :manhattan))
    play_games!(model)
    @test [a.s for a in allagents(model)] == [D, D, D, C, C, C]
    @test [a.π for a in allagents(model)] == [1.1, 1.1, 1.1, 2.0, 2.0, 2.0]
    @test [a.ω for a in allagents(model)] == [0.2891, 0.2891, 0.2891, 0.5, 0.5, 0.5]
end

@testset "play_games! (distance_metric = :chebyshev)" begin
    model = initialize_model(properties = ModelProperties(dims = (3, 3), N = 6, ϕC0 = 0, distance_metric = :chebyshev))
    play_games!(model)
    @test [a.π for a in allagents(model)] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test [a.ω for a in allagents(model)] == [0.1192, 0.1192, 0.1192, 0.1192, 0.1192, 0.1192]

    model = initialize_model(properties = ModelProperties(dims = (3, 3), N = 6, ϕC0 = 1, seed = 1, distance_metric = :chebyshev))
    play_games!(model)
    @test [a.π for a in allagents(model)] == [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    @test [a.ω for a in allagents(model)] == [0.9526, 0.9526, 0.9526, 0.9526, 0.9526, 0.9526]

    model = initialize_model(properties = ModelProperties(dims = (3, 3), N = 6, ϕC0 = 0.5, seed = 1, distance_metric = :chebyshev))
    play_games!(model)
    @test [a.s for a in allagents(model)] == [D, D, D, C, C, C]
    @test [a.π for a in allagents(model)] == [3.3000000000000003, 3.3000000000000003, 3.3000000000000003, 2.0, 2.0, 2.0]
    @test [a.ω for a in allagents(model)] == [0.7858, 0.7858, 0.7858, 0.5, 0.5, 0.5]
end

@testset "play_games! (distance_metric = :euclidean)" begin
    model = initialize_model(properties = ModelProperties(dims = (3, 3), N = 6, ϕC0 = 0, distance_metric = :euclidean))
    play_games!(model)
    @test [a.π for a in allagents(model)] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test [a.ω for a in allagents(model)] == [0.1192, 0.1192, 0.1192, 0.1192, 0.1192, 0.1192]

    model = initialize_model(properties = ModelProperties(dims = (3, 3), N = 6, ϕC0 = 1, seed = 1, distance_metric = :euclidean))
    play_games!(model)
    @test [a.π for a in allagents(model)] == [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    @test [a.ω for a in allagents(model)] == [0.9526, 0.9526, 0.9526, 0.9526, 0.9526, 0.9526]

    model = initialize_model(properties = ModelProperties(dims = (3, 3), N = 6, ϕC0 = 0.5, seed = 1, distance_metric = :euclidean))
    play_games!(model)
    @test [a.s for a in allagents(model)] == [D, D, D, C, C, C]
    @test [a.π for a in allagents(model)] == [3.3000000000000003, 3.3000000000000003, 3.3000000000000003, 2.0, 2.0, 2.0]
    @test [a.ω for a in allagents(model)] == [0.7858, 0.7858, 0.7858, 0.5, 0.5, 0.5]
end

#------------
# Move
#------------
using .Simulation: get_minimum_θ_destination, move_one_agent!, move!

@testset "get_minimum_θ_destination" begin
    pos_vec = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
    θ_mat = [5. 5.; 4. 3.; 2. 1.]
    @test get_minimum_θ_destination(pos_vec, θ_mat, 1.0) == (3, 2)

    θ_mat = [5. 5.; 0.1 1.; 4. 3.]
    @test get_minimum_θ_destination(pos_vec, θ_mat, 1.0) == (2, 1)

    actual_value = mean([get_minimum_θ_destination(pos_vec, θ_mat, 0.3) == (2, 1) for _ in 1:10_000])
    expected_value = 0.3 + 0.7 * 1/6
    @test expected_value - 0.1 < actual_value < expected_value + 0.1
end

@testset "move_one_agent!" begin
    model = initialize_model(properties = ModelProperties(dims = (4, 5), N = 5, seed = 1, distance_metric = :manhattan))

    move_agent!(model[1], (1, 1), model)
    move_agent!(model[2], (1, 5), model)
    move_agent!(model[3], (1, 2), model)
    move_agent!(model[4], (2, 1), model)
    move_agent!(model[5], (4, 1), model)

    move_one_agent!(model[1], model, 5)
    @test model[1].pos == (1, 1)

    move_agent!(model[2], (3, 3), model)
    @test model[1].pos == (1, 1)
    move_one_agent!(model[1], model, 1)
    @test model[1].pos == (1, 5)

    move_agent!(model[1], (1, 1), model)
     @test model[1].pos == (1, 1)
    move_one_agent!(model[1], model, 2)
    @test model[1].pos == (2, 5)

    move_agent!(model[1], (1, 1), model)
    @test model[1].pos == (1, 1)
    move_one_agent!(model[1], model, 4)
    @test model[1].pos == (3, 1)
end;

@testset "move!" begin
    @testset "ω > θ, prob_move = 1.0" begin
        model = initialize_model(properties = ModelProperties(dims = (4, 4), N = 6, seed = 1, distance_metric = :manhattan))
        model.θ_mat .= 1/9
        foreach(a -> a.ω = a.id / 8, allagents(model))
        model.prob_move = 1.0

        @test [a.pos for a in allagents(model)] == [(3, 4), (3, 3), (4, 1), (2, 2), (4, 3), (2, 1)]
        move!(model)
        @test [a.pos for a in allagents(model)] == [(3, 4), (3, 3), (4, 1), (2, 2), (4, 3), (2, 1)]
    end

    @testset "ω < θ, prob_move = 0.0" begin
        model = initialize_model(properties = ModelProperties(dims = (4, 4), N = 6, seed = 1, distance_metric = :manhattan))
        foreach(a -> a.ω = a.id / 9, allagents(model))
        model.θ_mat .= 1.5
        model.prob_move = 0.0

        @test [a.pos for a in allagents(model)] == [(3, 4), (3, 3), (4, 1), (2, 2), (4, 3), (2, 1)]
        move!(model)
        @test [a.pos for a in allagents(model)] == [(3, 4), (3, 3), (4, 1), (2, 2), (4, 3), (2, 1)]
    end

    @testset "ω < θ, prob_move = 0.5" begin
        model = initialize_model(properties = ModelProperties(dims = (4, 4), N = 6, seed = 1, distance_metric = :manhattan))
        foreach(a -> a.ω = a.id / 9, allagents(model))
        model.θ_mat .= 1.5
        model.prob_move = 0.5

        @test [a.pos for a in allagents(model)] == [(3, 4), (3, 3), (4, 1), (2, 2), (4, 3), (2, 1)]
        move!(model)
        @test [a.pos for a in allagents(model)] == [(3, 4), (3, 3), (4, 1), (3, 2), (1, 3), (3, 1)]
    end

    @testset "ω < θ, prob_move = 1.0" begin
        model = initialize_model(properties = ModelProperties(dims = (4, 4), N = 6, seed = 1, distance_metric = :manhattan))
        foreach(a -> a.ω = a.id / 9, allagents(model))
        model.θ_mat .= 1.5
        model.prob_move = 1.0

        @test [a.pos for a in allagents(model)] == [(3, 4), (3, 3), (4, 1), (2, 2), (4, 3), (2, 1)]
        move!(model)
        @test [a.pos for a in allagents(model)] == [(3, 1), (2, 3), (1, 1), (1, 2), (1, 3), (2, 4)]
    end

    @testset "ω < θ, prob_move = 2.0" begin
        model = initialize_model(properties = ModelProperties(dims = (4, 4), N = 6, seed = 1, distance_metric = :manhattan))
        foreach(a -> a.ω = a.id / 9, allagents(model))
        model.θ_mat .= 1.5
        model.prob_move = 2.0

        @test [a.pos for a in allagents(model)] == [(3, 4), (3, 3), (4, 1), (2, 2), (4, 3), (2, 1)]
        move!(model)
        @test [a.pos for a in allagents(model)] == [(3, 2), (3, 3), (4, 1), (1, 1), (1, 2), (1, 4)]
    end

    @testset "ω < θ, prob_move = 2.5" begin
        model = initialize_model(properties = ModelProperties(dims = (4, 4), N = 6, seed = 1, distance_metric = :manhattan))
        foreach(a -> a.ω = a.id / 9, allagents(model))
        model.θ_mat .= 1.5
        model.prob_move = 2.5

        @test [a.pos for a in allagents(model)] == [(3, 4), (3, 3), (4, 1), (2, 2), (4, 3), (2, 1)]
        move!(model)
        @test [a.pos for a in allagents(model)] == [(3, 2), (2, 3), (4, 4), (2, 2), (4, 2), (3, 3)]
    end
end

#------------
# Strategy update
#------------
using .Simulation: update_strategy!, flip

@testset "update_strategy!" begin
    @testset "ωC = 0.1, ωD = 0.9, θ = 1.0" begin
        model = initialize_model(properties = ModelProperties(dims = (3, 3), N = 7, ϕC0 = 4 / 7, seed = 1, distance_metric = :manhattan))
        model.θ_mat .= 1.0
        foreach(a -> a.ω = a.s == D ? 0.9 : 0.1, allagents(model))
    
        @test [a.s for a in allagents(model)] == [D, C, D, D, C, C, C]
        update_strategy!(model)
        @test [a.s for a in allagents(model)] == [D, D, D, D, C, C, D]
    end

    @testset "ωC = 0.9, ωD = 0.1, θ = 1.0" begin
        model = initialize_model(properties = ModelProperties(dims = (3, 3), N = 7, ϕC0 = 4 / 7, seed = 1, distance_metric = :manhattan))
        model.θ_mat .= 1.0
        foreach(a -> a.ω = a.s == C ? 0.9 : 0.1, allagents(model))
    
        @test [a.s for a in allagents(model)] == [D, C, D, D, C, C, C]
        update_strategy!(model)
        @test [a.s for a in allagents(model)] == [C, C, C, C, C, D, C]
    end

    @testset "ωC = 0.9, ωD = 0.1, θ = 0.0" begin
        model = initialize_model(properties = ModelProperties(dims = (3, 3), N = 7, ϕC0 = 4 / 7, seed = 1, distance_metric = :manhattan))
        model.θ_mat .= 0.0
        foreach(a -> a.ω = a.s == C ? 0.9 : 0.1, allagents(model))
    
        @test [a.s for a in allagents(model)] == [D, C, D, D, C, C, C]
        update_strategy!(model)
        @test [a.s for a in allagents(model)] == [D, C, D, D, C, C, C]
    end
end

#------------
# Integration
#------------
using .Simulation: run_simulation!, calc_mean_std_ϕC
@testset "run!" begin
    model = initialize_model(properties = ModelProperties(dims = (3, 3), N = 7, ϕC0 = 3/7, seed = 4, distance_metric = :manhattan))
    _, mdata = run!(model, 3, mdata = [ϕC, mean_π, std_π, mean_ω, std_ω, :sor_vec])
    @test mdata.ϕC == [0.4286, 0.8571, 0.7143, 0.8571]
    @test mdata.mean_π == [0.0, 1.4857, 2.6286, 2.1857]
    @test mdata.std_π == [0.0, 0.4811, 1.0797, 0.6594]
    @test mdata.mean_ω == [0.0, 0.3795, 0.6256, 0.543]
    @test mdata.std_ω == [0.0, 0.1127, 0.2149, 0.1532]
    @test mdata.sor_vec == [[(1, 1)], [(1, 1)], [(3, 1)], [(3, 3)]]
end

@testset "run_simulation!(properties::ModelProperties)::Matrix{Float64}" begin
    properties = ModelProperties(generation = 30, trial = 3)

    mat_ϕC = run_simulation!(properties)

    @test size(mat_ϕC) == (3, 31)
end

@testset "calc_mean_std_ϕC" begin
    trial = 5
    generation = 30
    mat_ϕC = fill(0.0, (trial, generation))
    [mat_ϕC[t, g] = t * g / 100 for g in 1:generation, t in 1:trial]

    @test calc_mean_std_ϕC(mat_ϕC) == (0.81, 0.4269)
end;

@testset "run_simulation!(properties_vec::Vector{ModelProperties})::Nothing" begin
    properties_vec = [ModelProperties(prob_EV = prob_EV, dims = (10, 10), N = 50, generation = 100, trial = 10) for prob_EV in 0.1:0.1:0.9]
    run_simulation!(properties_vec, "log_test")
end

end