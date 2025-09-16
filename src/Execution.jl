module Execution

using Random

include("../src/Simulation.jl")
using .Simulation: ModelProperties, run_simulation!

const N_vec = [1_000]
const dims = (400, 200)
const distance_metric = :euclidean
const periodic_space = true
const ϕC0 = 0.5
const sor_vec_vec = [[(100, 100), (300, 100)]]
const T_vec = [1.2]
const S_vec = [-0.2]
const x0 = 4.0
const prob_EV_vec = 0:0.1:1
const λ = 0
const cycle = 1
const prob_move_vec = 0:0.1:1
const sor_orientation_vec = [0.1]
const μ = 0.0
const generation = 10_000
const trial = 50

const properties_vec = vec([ModelProperties(; N, dims, distance_metric, periodic_space, ϕC0, sor_vec, T, S, x0, prob_EV, λ, cycle, prob_move, sor_orientation, μ, generation, trial)
        for N in N_vec, sor_vec in sor_vec_vec, T in T_vec, S in S_vec, prob_EV in prob_EV_vec, prob_move in prob_move_vec, sor_orientation in sor_orientation_vec])

println("length(properties_vec): $(length(properties_vec))")

@time run_simulation!(shuffle(properties_vec))
end