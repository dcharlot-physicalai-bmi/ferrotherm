"""
    FerrothermQUBODriversExt

Two MathOptInterface drivers, so any JuMP model that `ToQUBO.jl` can reformulate becomes a
ferrotherm workload.

# Why this exists

`IsingSolvers.jl` — the JuliaQUBO ecosystem's own native Ising solver set — was archived on
2026-05-28, and the remaining annealing drivers reach a foreign runtime: `DWave.Neal` and `PySA`
depend on `PythonCall`, which puts a CPython interpreter and a Conda environment inside a Julia
stack. That is the two-language problem Julia exists to escape, and it is the whole reason the
native slot being empty matters.

# The sign convention, which is the trap

QUBOTools writes an Ising energy as `α (Σᵢ hᵢ sᵢ + Σᵢⱼ Jᵢⱼ sᵢ sⱼ + β)` and **minimises** it.
ferrotherm writes `E = -Σᵢⱼ Jᵢⱼ sᵢ sⱼ - Σᵢ hᵢ sᵢ`. The signs are opposite, so every coupling and
field is negated crossing this boundary.

That is exactly the kind of claim that should not be taken on faith, so the tests do not: they run
both drivers against `QUBODrivers.ExactSampler` on random instances and require the *same* optimum.
A sign error would still produce plausible-looking output and would be wrong on every problem.
"""
module FerrothermQUBODriversExt

import Ferrotherm
import QUBOTools
import QUBODrivers
import QUBODrivers: MOI, Sample, SampleSet

# ---- shared: build a ferrotherm model from the QUBOTools Ising form -------------------------------

"""
Build a ferrotherm simulation from a sampler's Ising form, and return it with the scale and offset
needed to map an energy back into the caller's units.

`QUBOTools.ising(...; sense = :min)` yields `(n, h, J, α, β)` for `α (Σ hᵢsᵢ + Σ Jᵢⱼsᵢsⱼ + β)`.
ferrotherm's energy carries the opposite sign, so both are negated here and nowhere else.
"""
function _model(sampler, beta::Float64, seed::Integer)
    n, h, J, α, β = QUBOTools.ising(sampler, :sparse; sense = :min)
    m = Ferrotherm.IsingModel(n)
    for (i, hi) in enumerate(h)
        hi == 0 && continue
        Ferrotherm.bias!(m, Int(i), -float(hi))     # sign flip: their +h s, our -h s
    end
    # `pairs` on the sparse quadratic form yields CartesianIndex keys, not tuples. Destructuring
    # them directly throws "iteration is deliberately unsupported for CartesianIndex", which is
    # Julia telling you to be explicit rather than lucky.
    for (idx, Jij) in pairs(J)
        Jij == 0 && continue
        i, j = Tuple(idx)
        i == j && (Ferrotherm.bias!(m, Int(i), -float(Jij)); continue)   # diagonal is a field
        i < j || continue                                                # upper triangle only
        Ferrotherm.couple!(m, Int(i), Int(j), -float(Jij))
    end
    (Ferrotherm.build(m; beta, seed), n, float(α), float(β))
end

"""
Their objective value, from a ferrotherm state.

Work the algebra rather than guessing the sign. With `J_ours = -J_theirs` and `h_ours = -h_theirs`,

    E_ours = -Σ J_ours s s - Σ h_ours s = Σ J_theirs s s + Σ h_theirs s

so our energy IS their bracket, and their objective is `α (E_ours + β)`. An extra negation here
still produces plausible numbers and is wrong on every problem, which is why the tests compare
against `QUBODrivers.ExactSampler` rather than against themselves. It caught exactly that.
"""
_value(sim, α, β) = α * (Ferrotherm.energy(sim) + β)

"""A ferrotherm ±1 state as the 0/1 vector QUBOTools wants, in `:spin` domain terms."""
_spin_sample(sim) = Int.(Ferrotherm.spins(sim))

# ---- the sampler ----------------------------------------------------------------------------------

module Ferrotherm_

import Ferrotherm
import QUBOTools
import QUBODrivers
import QUBODrivers: MOI, Sample, SampleSet
import ..FerrothermQUBODriversExt: _model, _value, _spin_sample

@doc raw"""
    Ferrotherm.Optimizer{T}

Simulated annealing by chromatic block-Gibbs sampling, in pure Rust, with no foreign runtime.

Attributes: `NumberOfReads` (independent restarts), `BetaMin`, `BetaMax`, `Stages`,
`SweepsPerStage`, `RandomSeed`.
"""
QUBODrivers.@setup Optimizer begin
    name    = "Ferrotherm"
    version = v"0.1.0"
    attributes = begin
        NumberOfReads["num_reads"]::Integer = 10
        BetaMin["beta_min"]::Float64        = 0.05
        BetaMax["beta_max"]::Float64        = 6.0
        Stages["stages"]::Integer           = 60
        SweepsPerStage["sweeps"]::Integer   = 40
        RandomSeed["seed"]::Union{Integer,Nothing} = nothing
    end
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    reads  = MOI.get(sampler, Ferrotherm_.NumberOfReads())
    bmin   = MOI.get(sampler, Ferrotherm_.BetaMin())
    bmax   = MOI.get(sampler, Ferrotherm_.BetaMax())
    stages = MOI.get(sampler, Ferrotherm_.Stages())
    per    = MOI.get(sampler, Ferrotherm_.SweepsPerStage())
    seed0  = MOI.get(sampler, QUBODrivers.RandomSeed())
    seed0  = seed0 === nothing ? 0 : seed0

    samples = Sample{T,Int}[]
    results = @timed for r in 0:(reads - 1)
        sim, n, α, β = _model(sampler, bmax, seed0 + r)
        try
            Ferrotherm.anneal!(sim, bmin, bmax; stages, per)
            push!(samples, Sample{T}(_spin_sample(sim), T(_value(sim, α, β))))
        finally
            Ferrotherm.close!(sim)
        end
    end

    metadata = QUBODrivers._sampler_metadata(
        origin                = "ferrotherm @ Institute for Physical AI @ BMI",
        algorithm_name        = "chromatic block-Gibbs simulated annealing",
        execution_mode        = "simulated_annealing",
        optimizer_evaluations = reads,
        number_of_reads       = reads,
        final_number_of_reads = reads,
        status                = "feasible",
        termination_status    = MOI.LOCALLY_SOLVED,
    )
    metadata["time"] = Dict{String,Any}("effective" => results.time)

    return SampleSet{T}(samples, metadata; sense = :min, domain = :spin)
end

end # module Ferrotherm_

# ---- the exact driver -------------------------------------------------------------------------------

module Exact

import Ferrotherm
import QUBOTools
import QUBODrivers
import QUBODrivers: MOI, Sample, SampleSet
import ..FerrothermQUBODriversExt: _model, _value, _spin_sample

@doc raw"""
    Ferrotherm.Exact.Optimizer{T}

An exact oracle by **variable elimination**, costing ``2^{w}`` in the induced width of the graph
rather than ``2^{n}`` in the number of variables.

This is a drop-in replacement for `QUBODrivers.ExactSampler`, which enumerates all ``2^{n}`` states
and is documented as being for small instances only. Width is a property of a model's *shape*: a
chain has width 1 at any length, a lattice strip has the width of its short side, and a dense model
has width ``n-1`` and is refused rather than attempted.

Set `MaxWidth` to choose the ceiling. If the model is wider, this returns an **empty sample set**
with `termination_status = MOI.OTHER_LIMIT` rather than running for a week — check
`Ferrotherm.treewidth` first if you want to know in advance.
"""
QUBODrivers.@setup Optimizer begin
    name    = "Ferrotherm Exact"
    version = v"0.1.0"
    attributes = begin
        MaxWidth["max_width"]::Integer = 22
    end
end

QUBODrivers.honors_final_reads(::Type{<:Optimizer}) = false

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    max_width = MOI.get(sampler, Exact.MaxWidth())

    samples = Sample{T,Int}[]
    local width, solved
    results = @timed begin
        sim, n, α, β = _model(sampler, 1.0, 0)
        try
            width = Ferrotherm.treewidth(sim)
            st = Ferrotherm.exact_ground_state(sim; max_width)
            solved = st !== nothing
            if solved
                push!(samples, Sample{T}(Int.(st), T(_exact_value(sim, st, α, β))))
            end
        finally
            Ferrotherm.close!(sim)
        end
    end

    metadata = QUBODrivers._sampler_metadata(
        origin                = "ferrotherm @ Institute for Physical AI @ BMI",
        algorithm_name        = "variable elimination (min-sum)",
        execution_mode        = "exact",
        optimizer_evaluations = 1,
        number_of_reads       = length(samples),
        final_number_of_reads = length(samples),
        status                = solved ? "optimal" : "too wide for exact inference",
        termination_status    = solved ? MOI.OPTIMAL : MOI.OTHER_LIMIT,
    )
    metadata["time"] = Dict{String,Any}("effective" => results.time)
    metadata["treewidth"] = width

    return SampleSet{T}(samples, metadata; sense = :min, domain = :spin)
end

"""Value of an explicit state, obtained by writing it back and reading the energy."""
function _exact_value(sim, st, α, β)
    α * (Ferrotherm.exact_ground_energy(sim) + β)
end

end # module Exact

# Re-export under the names the docstrings promise.
const Optimizer = Ferrotherm_.Optimizer

end # module FerrothermQUBODriversExt
