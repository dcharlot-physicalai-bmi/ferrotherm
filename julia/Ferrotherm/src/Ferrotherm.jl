"""
    Ferrotherm

Thermodynamic sampling over Ising models, from Julia.

A binding over the ferrotherm C ABI — the same sampler reachable from Rust, Python, Zig, a browser
and an HTTP API, with the same conventions and the same seeds. States are `-1`/`+1`, energy is
`-Σᵢⱼ Jᵢⱼ sᵢ sⱼ - Σᵢ hᵢ sᵢ`, and `beta` is inverse temperature.

# Why this package exists

Julia's Ising slot is empty. `IsingSolvers.jl`, the JuliaQUBO ecosystem's own native solver set, was
archived on 2026-05-28, and the recommended replacement for real annealing depends on `PythonCall` —
a CPython interpreter and a Conda environment inside a Julia stack, which is the two-language problem
this language exists to escape.

More to the point, nothing in Julia certifies a sample. The three communities that would each care
do not talk to each other: statistical physics reports autocorrelation time but no effective sample
size, the Bayesian stack reports ESS but has no Ising model, and the JuMP/QUBO stack reports success
rates but has no notion of a target distribution at all. **This review did not locate anything that
reports the inverse temperature a sampler actually achieved, nor anything that reports a
sampling-noise floor.** Those are [`certify`](@ref)'s job.

# Getting the library

This package calls a native library and does not build one. Either:

```julia
ENV["FERROTHERM_LIB"] = "/path/to/libferrotherm.dylib"   # before `using Ferrotherm`
```

or build it from the ferrotherm checkout with `cargo build --release` and let the default search
find `target/release/`.

# Example

```julia
using Ferrotherm

# a planted instance knows its own optimum, so the result is a measurement rather than a number
p = frustrated(8, 96; seed = 3)
anneal!(p, 0.05, 6.0; stages = 80, per = 40)
excess(p)            # how far above the true optimum, as a fraction

# and a run can be checked rather than trusted
sim = lattice2d(12; beta = 0.2)
sweep!(sim, 500)
certify(sim; draws = 800, thin = 4)
```
"""
module Ferrotherm

using Libdl

export IsingModel, Simulation, Certificate
export couple!, bias!, build
export lattice2d, ring, z1_grid, frustrated, wishart
export sweep!, anneal!, beta!, spins, spins!, energy, magnetization
export certify, findings, passed
export known_optimum, excess, solved
export treewidth, exact_ground_energy, exact_ground_state, exact_logz
export node_updates, joules_z1, onsager, library_path, close!
export tabu!, tabu_iterations, population_anneal!, branch!, bounds, gap
export PopulationRun, BranchResult, Bounds, trustworthy, best, tightest
export Problem, Variable, Literal, Answer
export categorical!, integer!, binary!, is
export not_equal!, equal!, fix!, exactly!, at_most!, at_least!, exactly_one!, at_most_one!
export all_different!
export maximize!, minimize!, penalty!, solve!, certify!, ftp, violated, feasible
export soften_last!, soft_cost, traded, amounts, ancillas, caveats, ommx, from_ommx

# ---- finding the library -----------------------------------------------------------------------

const LIB = Ref{String}("")

"""
The artifact shipped by `ferrotherm_jll`, if that package is installed.

A soft dependency on purpose. Making it a hard one would mean `Ferrotherm` could not be installed
without a registry carrying both, and the package is useful today from a repository URL. When the
JLL is present it is the right answer and comes first; when it is not, the search below still finds
a library built from a checkout.
"""
function _jll_library()
    id = Base.identify_package("ferrotherm_jll")
    id === nothing && return nothing
    m = try
        Base.require(id)
    catch
        return nothing
    end
    p = try
        getproperty(m, :libferrotherm)[]
    catch
        return nothing
    end
    return isempty(p) ? nothing : p
end

function _candidates()
    out = String[]
    # An explicit override wins over everything, including the JLL: someone who set this is
    # debugging a specific build and must get that build.
    haskey(ENV, "FERROTHERM_LIB") && push!(out, ENV["FERROTHERM_LIB"])
    # Then the artifact, if ferrotherm_jll is installed. This is the path an ordinary install takes.
    let jll = _jll_library()
        jll === nothing || push!(out, jll)
    end
    name = Sys.iswindows() ? "ferrotherm.dll" : (Sys.isapple() ? "libferrotherm.dylib" : "libferrotherm.so")
    # a checkout being developed against: julia/Ferrotherm/src -> ../../../target/release
    root = normpath(joinpath(@__DIR__, "..", "..", ".."))
    push!(out, joinpath(root, "target", "release", name))
    push!(out, joinpath(root, "target", "debug", name))
    push!(out, name)   # let the loader search
    out
end

function __init__()
    for p in _candidates()
        try
            Libdl.dlopen(p)
            LIB[] = p
            return
        catch
        end
    end
    # Name the fix for the situation the caller is ACTUALLY in.
    #
    # This used to print the candidate list and nothing else. For anyone who installed the package
    # rather than cloning it, that list is two paths under `.julia/packages/.../target/` -- which
    # can never contain a build, because there is no checkout there -- and it never once mentioned
    # `ferrotherm_jll`, which is the answer for that person. An error listing only paths that
    # cannot work, omitting the one thing that would, is worse than terse.
    jll_note = _jll_library() === nothing ?
        """
        `ferrotherm_jll` is not installed, which is how an installed (non-checkout) copy gets its
        library:

            pkg> add https://github.com/dcharlot-physicalai-bmi/ferrotherm.git:julia/ferrotherm_jll
        """ :
        "`ferrotherm_jll` is installed but its artifact did not load, which is a bug worth reporting."

    error("""
          could not load the ferrotherm native library.

          $(jll_note)
          From a checkout, build it and point at the result:

              cargo build --release
              ENV["FERROTHERM_LIB"] = "<checkout>/target/release/libferrotherm.dylib"

          set before `using Ferrotherm`. (.so on Linux, .dll on Windows.)

          Tried, in order:
          """ * join("  " .* _candidates(), "\n"))
end

"""Path of the native library actually loaded. Useful when a wrong one is on the path."""
library_path() = LIB[]

# ---- the C ABI ---------------------------------------------------------------------------------
#
# One Julia function per documented header symbol, same name, same argument order, no ergonomics.
# Everything above this line is the friendly API; everything at this line is the header.

macro cfn(name, ret, args...)
    quote
        (($(esc(name)))($([:($(Symbol("a$i"))) for i in 1:length(args)]...))) = ccall(
            ($(QuoteNode(name)), LIB[]), $(esc(ret)),
            ($(map(esc, args)...),), $([:($(Symbol("a$i"))) for i in 1:length(args)]...))
    end
end

const SimPtr = Ptr{Cvoid}
const BldPtr = Ptr{Cvoid}

@cfn ft_ising2d_new SimPtr Cuint Cdouble Cdouble Culonglong
@cfn ft_z1_new SimPtr Cuint Cuint Cdouble Cdouble Cdouble Culonglong
@cfn ft_builder_new BldPtr Cuint
@cfn ft_builder_couple Cuint BldPtr Cuint Cuint Cdouble
@cfn ft_builder_bias Cuint BldPtr Cuint Cdouble
@cfn ft_builder_build SimPtr BldPtr Cdouble Culonglong
@cfn ft_builder_free Cvoid BldPtr
@cfn ft_sweep Culonglong SimPtr Cuint
@cfn ft_anneal Cdouble SimPtr Cdouble Cdouble Cuint Cuint
@cfn ft_set_beta Cvoid SimPtr Cdouble
@cfn ft_len Cuint SimPtr
@cfn ft_spins Ptr{Int8} SimPtr
@cfn ft_set_spins Cuint SimPtr Ptr{Int8} Cuint
@cfn ft_energy Cdouble SimPtr
@cfn ft_magnetization Cdouble SimPtr
@cfn ft_ledger_updates Culonglong SimPtr
@cfn ft_ledger_joules_z1 Cdouble SimPtr
@cfn ft_planted_frustrated SimPtr Cuint Cuint Culonglong Cdouble
@cfn ft_planted_wishart SimPtr Cuint Cdouble Culonglong Cdouble
@cfn ft_ground_energy Cdouble SimPtr
@cfn ft_certify Cuint SimPtr Cuint Cuint
@cfn ft_cert_beta_eff Cdouble SimPtr
@cfn ft_cert_beta_lo Cdouble SimPtr
@cfn ft_cert_beta_hi Cdouble SimPtr
@cfn ft_cert_tau Cdouble SimPtr
@cfn ft_cert_ess Cdouble SimPtr
@cfn ft_cert_tv Cdouble SimPtr
@cfn ft_cert_floor Cdouble SimPtr
@cfn ft_cert_passed Cuint SimPtr
@cfn ft_cert_findings Cuint SimPtr
@cfn ft_cert_finding Cuint SimPtr Cuint Ptr{UInt8} Cuint
@cfn ft_exact_ground Cdouble SimPtr Cuint
@cfn ft_exact_log_z Cdouble SimPtr Cdouble Cuint
@cfn ft_exact_width Cuint SimPtr
@cfn ft_exact_ground_state Cuint SimPtr Cuint Ptr{Int8} Cuint
@cfn ft_tabu Cdouble SimPtr Cuint Cuint Cuint
@cfn ft_tabu_iterations Culonglong SimPtr
@cfn ft_popanneal Cdouble SimPtr Cuint Cuint Cdouble Cuint
@cfn ft_popanneal_ln_z Cdouble SimPtr
@cfn ft_popanneal_rho Cdouble SimPtr
@cfn ft_branch Cdouble SimPtr Culonglong
@cfn ft_branch_proved Cuint SimPtr
@cfn ft_branch_nodes Culonglong SimPtr
@cfn ft_bound_decoupled Cdouble SimPtr
@cfn ft_bound_forest Cdouble SimPtr Cuint
@cfn ft_bound_odd_cycle Cdouble SimPtr Cuint
@cfn ft_bound_sdp Cdouble SimPtr Cuint Culonglong
@cfn ft_onsager Cdouble Cdouble
@cfn ft_free Cvoid SimPtr

# the modelling layer
const ModPtr = Ptr{Cvoid}
@cfn ft_model_new ModPtr
@cfn ft_model_free Cvoid ModPtr
@cfn ft_model_categorical Cuint ModPtr Cuint
@cfn ft_model_categorical_as Cuint ModPtr Cuint Cuint
@cfn ft_model_integer_as Cuint ModPtr Clonglong Clonglong Cuint
@cfn ft_model_integer Cuint ModPtr Clonglong Clonglong
@cfn ft_model_binary Cuint ModPtr
@cfn ft_model_name Cuint ModPtr Cuint Ptr{UInt8} Cuint
@cfn ft_model_not_equal Cuint ModPtr Cuint Cuint
@cfn ft_model_equal Cuint ModPtr Cuint Cuint
@cfn ft_model_fix Cuint ModPtr Cuint Clonglong
@cfn ft_model_lits_clear Cuint ModPtr
@cfn ft_model_lit Cuint ModPtr Cuint Clonglong
@cfn ft_model_var Cuint ModPtr Cuint
@cfn ft_model_close Cuint ModPtr Cuint Cuint
@cfn ft_model_objective_term Cuint ModPtr Cuint Cdouble Cuint Clonglong
@cfn ft_model_objective_pair Cuint ModPtr Cuint Cdouble Cuint Clonglong Cuint Clonglong
@cfn ft_model_objective_product Cuint ModPtr Cuint Cdouble
@cfn ft_model_fixed_penalty Cuint ModPtr Cdouble
@cfn ft_model_close_soft Cuint ModPtr Cuint Cuint Cdouble
@cfn ft_model_soften_last Cuint ModPtr Cdouble
@cfn ft_model_soft_cost Cdouble ModPtr
@cfn ft_model_violation_is_hard Cuint ModPtr Cuint
@cfn ft_model_violation_amount Cdouble ModPtr Cuint
@cfn ft_model_ancillas Cuint ModPtr
@cfn ft_model_caveats Cuint ModPtr
@cfn ft_model_ommx Cuint ModPtr Ptr{UInt8} Cuint
@cfn ft_ommx_read SimPtr Ptr{UInt8} Cuint Cdouble Culonglong Ptr{Cdouble}
@cfn ft_ommx_error Cuint Ptr{UInt8} Cuint
@cfn ft_model_ommx_constant Cdouble ModPtr
@cfn ft_model_caveat Cuint ModPtr Cuint Ptr{UInt8} Cuint
@cfn ft_model_compile Cuint ModPtr
@cfn ft_model_solve Cuint ModPtr Cuint
@cfn ft_model_solve_with Cuint ModPtr Cuint Cdouble Cdouble Cuint Cuint
@cfn ft_model_value Clonglong ModPtr Cuint
@cfn ft_model_feasible Cuint ModPtr
@cfn ft_model_energy Cdouble ModPtr
@cfn ft_model_penalty Cdouble ModPtr
@cfn ft_model_violations Cuint ModPtr
@cfn ft_model_violation Cuint ModPtr Cuint Ptr{UInt8} Cuint
@cfn ft_model_error Cuint ModPtr Ptr{UInt8} Cuint
@cfn ft_model_ftp Cuint ModPtr Ptr{UInt8} Cuint
@cfn ft_model_certify Cuint ModPtr Cdouble Cuint Cuint
@cfn ft_model_cert_findings Cuint ModPtr
@cfn ft_model_cert_finding Cuint ModPtr Cuint Ptr{UInt8} Cuint
@cfn ft_model_cert_beta Cdouble ModPtr
@cfn ft_model_cert_ess Cdouble ModPtr
@cfn ft_model_cert_tau Cdouble ModPtr
@cfn ft_model_cert_tv Cdouble ModPtr
@cfn ft_model_cert_floor Cdouble ModPtr

# ---- models ------------------------------------------------------------------------------------

"""
    IsingModel(n)

A graph under construction. Add couplings and biases, then [`build`](@ref).

Indices are **1-based**, as Julia's are; the conversion to the C ABI's 0-based indices happens at
the boundary and nowhere else. A rejected entry throws rather than being dropped, because a coupling
that vanishes without complaint is a model that is quietly wrong.
"""
mutable struct IsingModel
    handle::BldPtr
    n::Int
    function IsingModel(n::Integer)
        n >= 1 || throw(ArgumentError("a model needs at least one node"))
        h = ft_builder_new(Cuint(n))
        h == C_NULL && error("could not allocate a model")
        m = new(h, Int(n))
        finalizer(m) do x
            x.handle != C_NULL && ft_builder_free(x.handle)
            x.handle = BldPtr(C_NULL)
        end
        m
    end
end

_live(m::IsingModel) = m.handle == C_NULL &&
    error("this model was already built and cannot be reused")

function _bounds(m::IsingModel, i::Integer, which::AbstractString)
    (1 <= i <= m.n) || throw(ArgumentError(
        "$which = $i is out of range for a model of $(m.n) nodes; indices here are 1-based"))
    nothing
end

"""
    couple!(m, i, j, J)

Add coupling `J` between nodes `i` and `j`. Returns the model, so calls chain.
"""
function couple!(m::IsingModel, i::Integer, j::Integer, J::Real)
    _live(m)
    # Range-check in Julia, before the conversion. Index 0 is valid in C and invalid here, and
    # letting it through produces `InexactError: trunc(UInt32, -1)` from deep in the conversion --
    # true, useless, and it names neither the argument nor the rule it broke.
    _bounds(m, i, "i"); _bounds(m, j, "j")
    ok = ft_builder_couple(m.handle, Cuint(i - 1), Cuint(j - 1), Cdouble(J))
    ok == 0 && throw(ArgumentError(
        "rejected coupling ($i, $j, $J): indices must be in 1:$(m.n), i must differ from j, " *
        "and the weight must be finite"))
    m
end

"""
    bias!(m, i, h)

Add external field `h` on node `i`. Repeated calls on one node accumulate.
"""
function bias!(m::IsingModel, i::Integer, h::Real)
    _live(m)
    _bounds(m, i, "i")
    ok = ft_builder_bias(m.handle, Cuint(i - 1), Cdouble(h))
    ok == 0 && throw(ArgumentError("rejected bias ($i, $h): index must be in 1:$(m.n), h finite"))
    m
end

Base.length(m::IsingModel) = m.n
Base.show(io::IO, m::IsingModel) =
    print(io, m.handle == C_NULL ? "IsingModel(spent)" : "IsingModel($(m.n) nodes)")

# ---- simulations -------------------------------------------------------------------------------

"""
    Simulation

A running sampler. Construct with [`build`](@ref), [`lattice2d`](@ref), [`ring`](@ref),
[`frustrated`](@ref) or [`wishart`](@ref).

The handle owns memory in the native library. A finalizer releases it, and [`close!`](@ref) does so
deterministically; both are safe to reach, and using a closed simulation throws rather than reading
freed memory.
"""
mutable struct Simulation
    handle::SimPtr
    function Simulation(h::SimPtr, what::AbstractString)
        h == C_NULL && error("could not build $what")
        s = new(h)
        finalizer(s) do x
            x.handle != C_NULL && ft_free(x.handle)
            x.handle = SimPtr(C_NULL)
        end
        s
    end
end

_live(s::Simulation) = s.handle == C_NULL && error("this simulation is closed")

"""Release the native handle now rather than at the next garbage collection."""
function close!(s::Simulation)
    s.handle != C_NULL && ft_free(s.handle)
    s.handle = SimPtr(C_NULL)
    nothing
end

"""
    build(m; beta = 1.0, seed = 0)

Consume a model into a simulation. The model is spent afterwards.
"""
function build(m::IsingModel; beta::Real = 1.0, seed::Integer = 0)
    _live(m)
    h = m.handle
    m.handle = BldPtr(C_NULL)      # the builder is consumed; do not free it twice
    Simulation(ft_builder_build(h, Cdouble(beta), Culonglong(seed)), "the simulation")
end

"""2D nearest-neighbour Ising lattice, periodic, side `l`."""
lattice2d(l::Integer; J::Real = 1.0, beta::Real = 0.44, seed::Integer = 0) =
    Simulation(ft_ising2d_new(Cuint(l), Cdouble(J), Cdouble(beta), Culonglong(seed)), "the lattice")

"""Z1-topology grid, degree 16, open boundaries."""
z1_grid(w::Integer, h::Integer; J::Real = 1.0, hb::Real = 0.0, beta::Real = 1.0, seed::Integer = 0) =
    Simulation(ft_z1_new(Cuint(w), Cuint(h), Cdouble(J), Cdouble(hb), Cdouble(beta),
                         Culonglong(seed)), "the grid")

"""A periodic chain of `n` nodes."""
function ring(n::Integer; J::Real = 1.0, h::Real = 0.0, beta::Real = 1.0, seed::Integer = 0)
    n >= 3 || throw(ArgumentError("a ring needs at least 3 nodes"))
    m = IsingModel(n)
    for i in 1:n
        couple!(m, i, mod1(i + 1, n), J)
    end
    h != 0 && for i in 1:n
        bias!(m, i, h)
    end
    build(m; beta, seed)
end

"""Run `n` chromatic block-Gibbs sweeps. Returns the simulation."""
function sweep!(s::Simulation, n::Integer = 1)
    _live(s)
    ft_sweep(s.handle, Cuint(n))
    s
end

"""
    anneal!(s, beta_min, beta_max; stages = 60, per = 40)

Anneal down a geometric ladder, leaving the simulation holding the best state found and returning
its energy.
"""
function anneal!(s::Simulation, beta_min::Real = 0.05, beta_max::Real = 4.0;
                 stages::Integer = 60, per::Integer = 40)
    _live(s)
    e = ft_anneal(s.handle, Cdouble(beta_min), Cdouble(beta_max), Cuint(stages), Cuint(per))
    isnan(e) && throw(ArgumentError("need 0 < beta_min < beta_max, stages >= 2, per >= 1"))
    e
end

"""Change temperature without disturbing the state."""
function beta!(s::Simulation, beta::Real)
    _live(s)
    ft_set_beta(s.handle, Cdouble(beta))
    s
end

Base.length(s::Simulation) = (_live(s); Int(ft_len(s.handle)))

"""
    spins(s) -> Vector{Int8}

The state, as a copy. It is a copy on purpose: the library owns that buffer and may move it on the
next sweep, so a view would alias freed memory the moment the simulation advanced.
"""
function spins(s::Simulation)
    _live(s)
    n = Int(ft_len(s.handle))
    copy(unsafe_wrap(Array, ft_spins(s.handle), n; own = false))
end

"""
    spins!(s, state)

Put a state INTO the simulation, so something computed elsewhere is scored, certified or
annealed by exactly the same code that handles a state this library produced.

It refuses rather than adapting: the length must match the graph and every value must be -1 or +1.
A shorter state means whatever produced it did not finish, and a value that is not a spin means the
buffer is not what the caller thinks it is. Both are cheap to launder into something plausible --
pad with -1, coerce with `v > 0` -- and a laundered state is then scored with full confidence, which
is how a dropped GPU dispatch turns into a believable energy.
"""
function spins!(s::Simulation, state::AbstractVector{<:Integer})
    _live(s)
    buf = Int8[]
    for (i, v) in enumerate(state)
        v in (-1, 1) || error("states are -1/+1; got $v at index $i")
        push!(buf, Int8(v))
    end
    ft_set_spins(s.handle, buf, Cuint(length(buf))) == 0 &&
        error("this simulation has $(Int(ft_len(s.handle))) nodes and that state has $(length(buf))")
    nothing
end

energy(s::Simulation) = (_live(s); ft_energy(s.handle))
magnetization(s::Simulation) = (_live(s); ft_magnetization(s.handle))
node_updates(s::Simulation) = (_live(s); Int(ft_ledger_updates(s.handle)))

"""
    joules_z1(s)

The energy ledger priced at Z1-class device figures. This prices the **modelled device**, not the
CPU that actually ran the sweep.
"""
joules_z1(s::Simulation) = (_live(s); ft_ledger_joules_z1(s.handle))

function Base.show(io::IO, s::Simulation)
    if s.handle == C_NULL
        print(io, "Simulation(closed)")
    else
        print(io, "Simulation($(length(s)) nodes, E = $(round(energy(s); digits = 4)), " *
                  "m = $(round(magnetization(s); digits = 3)))")
    end
end

# ---- planted instances -------------------------------------------------------------------------

"""
    frustrated(l, loops; seed = 0, beta = 1.0)

A planted instance on an `l`×`l` periodic lattice whose optimum is known by construction.

Difficulty is **not** monotonic in `loops`: it peaks near four planted loops per edge and falls away
at both ends, because a saturated instance's couplings concentrate toward their mean and relax back
into a gauged ferromagnet.
"""
frustrated(l::Integer, loops::Integer; seed::Integer = 0, beta::Real = 1.0) =
    Simulation(ft_planted_frustrated(Cuint(l), Cuint(loops), Culonglong(seed), Cdouble(beta)),
               "the planted instance")

"""
    wishart(n; alpha = 0.5, seed = 0, beta = 1.0)

The Wishart planted ensemble: dense, with a known optimum, and hard below `alpha` of 1.

A miss here is usually under 2% above the optimum because the landscape is dense with
near-degenerate minima, so report a **solve rate** rather than a mean excess or this family looks
easy when it is not.
"""
wishart(n::Integer; alpha::Real = 0.5, seed::Integer = 0, beta::Real = 1.0) =
    Simulation(ft_planted_wishart(Cuint(n), Cdouble(alpha), Culonglong(seed), Cdouble(beta)),
               "the Wishart instance")

"""The true ground energy for a planted instance, or `nothing` otherwise."""
function known_optimum(s::Simulation)
    _live(s)
    v = ft_ground_energy(s.handle)
    isnan(v) ? nothing : v
end

"""How far the current state sits above a planted optimum, as a fraction. `nothing` if not planted."""
function excess(s::Simulation)
    k = known_optimum(s)
    k === nothing && return nothing
    e = energy(s)
    abs(k) > 1e-12 ? (e - k) / abs(k) : e - k
end

"""Whether the current state reaches a planted optimum."""
function solved(s::Simulation)
    k = known_optimum(s)
    k === nothing ? false : energy(s) <= k + 1e-9
end

# ---- the certificate ---------------------------------------------------------------------------

"""
    Certificate

What a run actually did, computed from its samples rather than from the sampler's own account of
itself. `findings` is empty exactly when the run is sound — read that, not `beta_eff`.

Nothing else in Julia reports `beta_eff` or `noise_floor` at all.
"""
# ---- solvers and bounds -------------------------------------------------------------------------
#
# Each solver LEAVES ITS BEST STATE as the simulation's state, so `spins` reads the answer and
# `energy` recomputes it from that state rather than trusting the number returned. They compose:
# `anneal!`, then `tabu!` from where annealing stopped, then `branch!` with that as its incumbent.
# The `!` is not decoration -- these mutate the simulation.

"""A population-annealing run, with the diagnostic that says whether to believe it."""
struct PopulationRun
    energy::Float64
    "`ln Z` at the final β, or `nothing` when the ladder did not start at infinite temperature."
    ln_z::Union{Float64,Nothing}
    "1.0 when every ancestor still has a descendant; `population` on total collapse."
    rho::Float64
    population::Int
end

"""
    trustworthy(r::PopulationRun)

`rho` below a tenth of the population. A rule of thumb, not a theorem — the number is on the struct
so you can apply your own. A run whose `rho` spiked explored ONE basin with `population` copies of
one history, and its `ln_z` is worth nothing.
"""
trustworthy(r::PopulationRun) = r.rho <= max(1.0, r.population / 10)

"""What branch and bound found, and whether it proved it."""
struct BranchResult
    energy::Float64
    "True only when the tree was exhausted inside the node budget."
    proved::Bool
    nodes::Int
end

"""Lower bounds on the ground energy. All sound, so `best` is their maximum."""
struct Bounds
    decoupled::Float64
    forest::Float64
    odd_cycle::Float64
    "`nothing` when the certificate failed to re-verify. A refusal, not a missing feature."
    sdp::Union{Float64,Nothing}
end

_pairs(b::Bounds) = let v = [("decoupled", b.decoupled), ("forest", b.forest),
                             ("odd_cycle", b.odd_cycle)]
    b.sdp === nothing ? v : push!(v, ("sdp", b.sdp))
end

"""Taking the maximum of sound bounds is sound."""
best(b::Bounds) = maximum(last.(_pairs(b)))

"""Which bound set `best`. They disagree by a lot, and in both directions.

Named `tightest` rather than `which` because `Base.which` exists: exporting a second `which` makes
`using Ferrotherm` ambiguous at the call site, and the error a user gets says nothing about bounds.
"""
tightest(b::Bounds) = first(argmax(last, _pairs(b)))

"""
    tabu!(s; iterations = 50_000, tenure = 0, restart_after = 5_000)

Tabu search. Returns the best energy found and leaves that state on `s`.

`tenure = 0` scales the tenure to the graph; `restart_after = 0` never restarts. Check
`tabu_iterations` afterwards — a run shorter than its budget was truncated, and that is otherwise
invisible from outside.
"""
function tabu!(s::Simulation; iterations::Integer = 50_000, tenure::Integer = 0,
               restart_after::Integer = 5_000)
    _live(s)
    ft_tabu(s.handle, Cuint(max(1, iterations)), Cuint(tenure), Cuint(restart_after))
end

"""Iterations the last `tabu!` actually ran, or 0 if there was none."""
function tabu_iterations(s::Simulation)
    _live(s)
    Int(ft_tabu_iterations(s.handle))
end

"""
    population_anneal!(s; population = 1000, sweeps = 4, beta_max = 6.0, stages = 100)

Population annealing: `population` chains down one ladder, resampled at each rung.

Unlike a single annealed chain this also estimates the free energy, and reports whether to believe
it. The ladder starts at `β = 0`, where `Z = 2^n` exactly, which is what makes `ln_z` absolute
rather than a ratio.
"""
function population_anneal!(s::Simulation; population::Integer = 1000, sweeps::Integer = 4,
                            beta_max::Real = 6.0, stages::Integer = 100)
    _live(s)
    e = ft_popanneal(s.handle, Cuint(max(1, population)), Cuint(max(1, sweeps)),
                     Cdouble(beta_max), Cuint(max(1, stages)))
    isnan(e) && throw(ArgumentError("beta_max must be finite and non-negative"))
    z = ft_popanneal_ln_z(s.handle)
    PopulationRun(e, isnan(z) ? nothing : z, ft_popanneal_rho(s.handle), Int(population))
end

"""
    branch!(s; max_nodes = 20_000_000)

Branch and bound from `s`'s current state, which it uses as its incumbent.

The only solver here that returns a **proof**. A run that exhausts `max_nodes` returns the best
state it saw with `proved = false`; a flag meaning "optimal, or else we gave up" gets read as the
first thing and quoted as the second.
"""
function branch!(s::Simulation; max_nodes::Integer = 20_000_000)
    _live(s)
    e = ft_branch(s.handle, Culonglong(max(1, max_nodes)))
    BranchResult(e, ft_branch_proved(s.handle) != 0, Int(ft_branch_nodes(s.handle)))
end

"""
    bounds(s; forest_rounds = 40, max_cycle = 6, sdp_sweeps = 200, seed = 1)

Every lower bound on the ground energy this library computes, and the best of them.

All are sound on their own, so `best` is their maximum — not a tie-break, a result: `odd_cycle`
wins on sparse frustrated lattices and `sdp` wins by more the denser the instance gets. The SDP
bound is re-verified on the Rust side before it crosses, and is `nothing` if that fails.
"""
function bounds(s::Simulation; forest_rounds::Integer = 40, max_cycle::Integer = 6,
                sdp_sweeps::Integer = 200, seed::Integer = 1)
    _live(s)
    sdp = ft_bound_sdp(s.handle, Cuint(max(1, sdp_sweeps)), Culonglong(seed))
    Bounds(ft_bound_decoupled(s.handle),
           ft_bound_forest(s.handle, Cuint(forest_rounds)),
           ft_bound_odd_cycle(s.handle, Cuint(max_cycle)),
           isnan(sdp) ? nothing : sdp)
end

"""
    gap(s; kw...)

How far `s`'s current state is from optimal, at worst: `energy(s) - best(bounds(s))`.

Zero means the state is **proved** optimal without trusting the sampler that found it. Anything
above is an upper limit on what a better search could still win.
"""
gap(s::Simulation; kw...) = energy(s) - best(bounds(s; kw...))

struct Certificate
    draws::Int
    beta_eff::Float64
    beta_ci::Tuple{Float64,Float64}
    tau_int::Float64
    ess::Float64
    tv::Union{Float64,Nothing}
    noise_floor::Union{Float64,Nothing}
    findings::Vector{String}
end

"""Empty findings is the only thing that means the run is sound."""
passed(c::Certificate) = isempty(c.findings)
findings(c::Certificate) = c.findings

"""
    certify(s; draws = 512, thin = 1)

Sample and check the result.

Returns the inverse temperature actually achieved with a confidence interval, the integrated
autocorrelation time and the resulting effective sample size, and where the model is small enough to
enumerate, the total variation distance from the exact distribution **beside the sampling-noise
floor** — because a distance below that floor is agreement, not accuracy.
"""
function certify(s::Simulation; draws::Integer = 512, thin::Integer = 1)
    _live(s)
    draws >= 16 || throw(ArgumentError("certifying fewer than 16 draws says nothing"))
    ft_certify(s.handle, Cuint(draws), Cuint(max(1, thin))) == 0 &&
        error("could not certify this run")
    msgs = String[]
    for i in 0:(ft_cert_findings(s.handle) - 1)
        need = ft_cert_finding(s.handle, Cuint(i), Ptr{UInt8}(C_NULL), Cuint(0))
        buf = Vector{UInt8}(undef, need)
        got = ft_cert_finding(s.handle, Cuint(i), pointer(buf), Cuint(need))
        push!(msgs, String(buf[1:got]))
    end
    nan2n(x) = isnan(x) ? nothing : x
    Certificate(Int(draws), ft_cert_beta_eff(s.handle),
                (ft_cert_beta_lo(s.handle), ft_cert_beta_hi(s.handle)),
                ft_cert_tau(s.handle), ft_cert_ess(s.handle),
                nan2n(ft_cert_tv(s.handle)), nan2n(ft_cert_floor(s.handle)), msgs)
end

function Base.show(io::IO, c::Certificate)
    print(io, "Certificate ", passed(c) ? "PASSED" : "FAILED",
          "\n  beta      ", round(c.beta_eff; digits = 4),
          "  (95% ", round(c.beta_ci[1]; digits = 4), " … ", round(c.beta_ci[2]; digits = 4), ")",
          "\n  tau_int   ", round(c.tau_int; digits = 2),
          "\n  ess       ", round(c.ess; digits = 0), " of ", c.draws, " draws")
    # The floor is never printed apart from the distance. A distance without its floor is the
    # mistake this whole type exists to prevent.
    if c.tv !== nothing && c.noise_floor !== nothing
        print(io, "\n  tv        ", round(c.tv; digits = 4),
              "  against a noise floor of ", round(c.noise_floor; digits = 4))
    end
    for f in c.findings
        print(io, "\n  ! ", f)
    end
end

# ---- exact inference ---------------------------------------------------------------------------

"""
    treewidth(s)

Induced width of the elimination order. Exact inference costs `2^treewidth`, so this is the number
that decides whether to ask for it at all — a 2,000-spin chain has width 1.
"""
treewidth(s::Simulation) = (_live(s); Int(ft_exact_width(s.handle)))

"""
    exact_ground_energy(s; max_width = 22)

Exact ground energy by variable elimination, or `nothing` if the graph is too dense.

Cost is `2^width` in the graph's **shape** rather than `2^n` in its size, which is a strict upgrade
on brute-force enumeration for anything sparse.
"""
function exact_ground_energy(s::Simulation; max_width::Integer = 22)
    _live(s)
    v = ft_exact_ground(s.handle, Cuint(max_width))
    isnan(v) ? nothing : v
end

"""
    exact_ground_state(s; max_width = 22) -> Union{Vector{Int8},Nothing}

The exact ground state itself, not just its energy. `nothing` if the graph is too dense.

A caller that must return a *solution* rather than a bound needs this.
"""
function exact_ground_state(s::Simulation; max_width::Integer = 22)
    _live(s)
    n = length(s)
    out = Vector{Int8}(undef, n)
    ok = ft_exact_ground_state(s.handle, Cuint(max_width), pointer(out), Cuint(n))
    ok == 0 ? nothing : out
end

"""Exact `log Z` at `beta`, or `nothing` if the graph is too dense."""
function exact_logz(s::Simulation, beta::Real = 1.0; max_width::Integer = 22)
    _live(s)
    v = ft_exact_log_z(s.handle, Cdouble(beta), Cuint(max_width))
    isnan(v) ? nothing : v
end

# ---- reference ---------------------------------------------------------------------------------

"""Onsager's exact spontaneous magnetisation for the 2D Ising model. Ground truth; 0 above `betac`."""
onsager(beta::Real) = ft_onsager(Cdouble(beta))

"""The 2D square-lattice critical inverse temperature, `log(1+√2)/2`."""
const betac = log(1 + sqrt(2)) / 2


# ---- modelling ---------------------------------------------------------------------------------
#
# Everything above works in spins: couplings, biases, energies. That is the machine's language, not
# the problem's. This is the problem's — variables that hold values, constraints that say what must
# hold, an objective that says what is better — and it compiles down to the layer above.
#
# The shape a Julia user expects from JuMP, without the two-language problem: no Python, no Conda,
# one native library that the same code reaches from Rust, C, Zig, a browser and an HTTP API.

"""
    Variable

A declared variable. Ask it for a literal with [`is`](@ref).
"""
struct Variable
    idx::UInt32
    name::String
    domain::String
end

Base.show(io::IO, v::Variable) = print(io, v.name, "::", v.domain)

"""
    Literal

The claim that one variable takes one value. Build with [`is`](@ref).
"""
struct Literal
    var::Variable
    value::Int64
end

"""
    is(v, value)

The literal "`v` takes `value`", for counting constraints and objective terms.

Values are the variable's own. An integer over `10:20` takes the value 13; passing 3 is an error
naming the range, not the fourth slot.
"""
is(v::Variable, value::Integer) = Literal(v, Int64(value))

"""
    Answer

A solved problem, read by name. `answer["shift"]` gives a value.

`feasible` means every variable decoded **and** every HARD constraint held. When it is false,
`violated` describes each broken constraint in your own names and any variable that failed to decode
reads `nothing`. A penalty makes a constraint expensive, not impossible.

Soft constraints are separate: `traded` lists the preferences the solver chose to break and
`soft_cost` totals what they cost. Those do not make the answer infeasible — being traded away is
what asking for a preference means.
"""
struct Answer
    values::Dict{String, Union{Int64, Nothing}}
    feasible::Bool
    violated::Vector{String}
    energy::Float64
    spins::Int
    penalty::Float64
    soft_cost::Float64
    traded::Vector{String}
    by::Vector{Float64}
    ancillas::Int
    caveats::Vector{String}
end

Base.getindex(a::Answer, name::AbstractString) = a.values[String(name)]
Base.haskey(a::Answer, name::AbstractString) = haskey(a.values, String(name))
Base.keys(a::Answer) = keys(a.values)
Base.pairs(a::Answer) = pairs(a.values)
feasible(a::Answer) = a.feasible
violated(a::Answer) = a.violated

"""    traded(a)  — the soft constraints the solver chose to break, in your own names."""
traded(a::Answer) = a.traded

"""    soft_cost(a)  — what those trades cost, in the units the weights were given in."""
soft_cost(a::Answer) = a.soft_cost

"""
    ancillas(a)

Spins the higher-order lowering added, or zero if nothing named three or more variables.

Non-zero means the answer solves a model with **more** spins than the variables required. The extra
states are what makes the reduction exact for *optimisation* and not for sampling: the Boltzmann
distribution over the original variables is not preserved. Read this before drawing samples from a
solved model, rather than only its ground state.
"""
ancillas(a::Answer) = a.ancillas

"""    caveats(a)

What the compiler knows is wrong with the model and cannot fix. Today there is one kind: an encoding no penalty can make exact -- a binary encoding of k values spells 2^ceil(log2 k) codewords, and when k is not a power of two the spare ones decode to nothing while costing exactly what a valid state costs. Read them before trusting a result; empty is the normal case.
"""
caveats(a::Answer) = a.caveats

"""    amounts(a)  — how far outside each broken HARD constraint sits, parallel to `violated`.

"At most two, and four hold" is over by two, and that is a different problem from being over by one.
"""
amounts(a::Answer) = a.by

function Base.show(io::IO, a::Answer)
    print(io, "Answer ", a.feasible ? "feasible" : "INFEASIBLE",
          "  energy ", round(a.energy; digits = 4), "  ", a.spins, " spins")
    for k in sort(collect(keys(a.values)))
        v = a.values[k]
        print(io, "\n  ", k, " = ", v === nothing ? "(did not decode)" : v)
    end
    for v in a.traded
        print(io, "\n  traded: ", v)
    end
    a.soft_cost == 0 || print(io, "\n  soft cost: ", round(a.soft_cost; digits = 4))
    for (i, v) in enumerate(a.violated)
        print(io, "\n  broken: ", v, i <= length(a.by) ? " (by $(round(a.by[i]; digits = 4)))" : "")
    end
end

"""
    Problem()

A problem stated in its own terms, compiled to spins and sampled.

```julia
p = Problem()
days = [binary!(p, d) for d in ("mon", "tue", "wed", "thu", "fri")]
at_most!(p, days, 3)
maximize!(p, [(w, is(d, 1)) for (w, d) in zip((5, 4, 3, 2, 1), days)])
a = solve!(p)
[d.name for d in days if a[d.name] == 1]   # ["mon", "tue", "wed"]
```

Inequalities cost extra spins: each needs a slack variable to become an equality the sampler can
square. The slack never appears in the answer.
"""
mutable struct Problem
    handle::ModPtr
    vars::Vector{Variable}
    function Problem()
        h = ft_model_new()
        h == C_NULL && error("could not allocate a problem")
        p = new(h, Variable[])
        finalizer(close!, p)
        p
    end
end

function close!(p::Problem)
    if p.handle != C_NULL
        ft_model_free(p.handle)
        p.handle = ModPtr(C_NULL)
    end
    nothing
end

_live(p::Problem) = p.handle == C_NULL && error("this problem has been closed")

"""Why the library refused the last call."""
function _why(p::Problem)
    need = ft_model_error(p.handle, Ptr{UInt8}(C_NULL), Cuint(0))
    need == 0 && return ""
    buf = Vector{UInt8}(undef, need)
    got = ft_model_error(p.handle, pointer(buf), Cuint(need))
    String(buf[1:got])
end

_must(p::Problem, ok, what) =
    ok == 0 && error(isempty(_why(p)) ? "the library refused that $what" : _why(p))

function _declare(p::Problem, name::AbstractString, domain::AbstractString, idx::UInt32)
    idx == typemax(UInt32) && error(
        "$domain is not a usable domain for \"$name\" " *
        "(a categorical needs at least 2 values; an integer needs hi > lo)")
    raw = Vector{UInt8}(codeunits(String(name)))
    # Push the name down, so a refusal names the variable you declared rather than a handle you
    # never saw. It also refuses a name already taken: an answer is keyed by name, so a second
    # variable with the same one would replace the first rather than shadow it.
    _must(p, ft_model_name(p.handle, idx, pointer(raw), Cuint(length(raw))), "name")
    v = Variable(idx, String(name), String(domain))
    push!(p.vars, v)
    v
end

"""
    categorical!(p, name, values; encoding = :onehot)

A variable holding one of `values` distinct values. Needs at least two.

`encoding` decides how it is stored, and the trade is the difference between a model that fits a
machine and one that does not:

| encoding | spins for `k` values | usable in a constraint or objective |
|---|---|---|
| `:onehot` | `k` | yes |
| `:domainwall` | `k - 1` | yes |
| `:binary` | `ceil(log2 k)` | **no** |

A binary code's indicator is a product of every bit, so its degree grows with the domain. It is the
cheapest to store and is refused by name if it appears in a literal, rather than expanded into
something nobody wants to read.
"""
function categorical!(p::Problem, name::AbstractString, values::Integer;
                      encoding::Symbol = :onehot)
    _live(p)
    _declare(p, name, "categorical($values)",
             ft_model_categorical_as(p.handle, Cuint(values), _encoding(encoding)))
end

"""Encoding names as the codes the C ABI takes, naming the alternatives when it is not one."""
function _encoding(name::Symbol)
    name === :onehot && return Cuint(0)
    name === Symbol("one-hot") && return Cuint(0)
    name === :binary && return Cuint(1)
    name === :domainwall && return Cuint(2)
    name === Symbol("domain-wall") && return Cuint(2)
    error("unknown encoding $(repr(name)); try :onehot, :domainwall or :binary")
end

"""    integer!(p, name, range)

A variable over an inclusive integer range, written `integer!(p, "t", 10:20)`.

There is no machine integer here: this is a categorical over the range, and the name is for the
modeller rather than the fabric.

`:domainwall` is often the better encoding for an integer, which is an ORDERED domain: neighbouring
values sit one spin flip apart where one-hot puts them two apart, and it costs one spin fewer.
"""
function integer!(p::Problem, name::AbstractString, r::AbstractUnitRange{<:Integer};
                  encoding::Symbol = :onehot)
    _live(p)
    _declare(p, name, "$(first(r)):$(last(r))",
             ft_model_integer_as(p.handle, Clonglong(first(r)), Clonglong(last(r)),
                                 _encoding(encoding)))
end
integer!(p::Problem, name::AbstractString, lo::Integer, hi::Integer; kw...) =
    integer!(p, name, lo:hi; kw...)

"""    binary!(p, name)

A variable that is 0 or 1.
"""
function binary!(p::Problem, name::AbstractString)
    _live(p)
    _declare(p, name, "binary", ft_model_binary(p.handle))
end

# -- constraints ---------------------------------------------------------------------------------

"""
    soften_last!(p, weight)

Make the constraint added most recently a **preference**, priced at `weight`.

A hard constraint says which answers are answers at all. A soft one is a preference the solver may
trade away: breaking it costs `weight × amount²` and leaves the answer feasible. The square is not a
detail — a constraint becomes an energy term by squaring how far outside it sits, so missing by two
costs four times missing by one, and pricing a preference chooses that curve as well as its scale.

The weight is absolute rather than scaled. Automatic scaling exists to stop a hard constraint being
outbid by the objective, and a soft one is meant to be traded against it.

Every constraint here takes `soft = weight` as a shorthand for the same thing.
"""
soften_last!(p::Problem, weight::Real) =
    (_live(p); _must(p, ft_model_soften_last(p.handle, Cdouble(weight)), "soft constraint"); nothing)

_maybe_soft(p::Problem, soft) = soft === nothing ? nothing : soften_last!(p, soft)

"""    not_equal!(p, a, b; soft = nothing)  — `a` and `b` must differ, or had better."""
function not_equal!(p::Problem, a::Variable, b::Variable; soft::Union{Real, Nothing} = nothing)
    _live(p); _must(p, ft_model_not_equal(p.handle, a.idx, b.idx), "not_equal")
    _maybe_soft(p, soft); nothing
end

"""    equal!(p, a, b; soft = nothing)  — `a` and `b` must agree, or had better."""
function equal!(p::Problem, a::Variable, b::Variable; soft::Union{Real, Nothing} = nothing)
    _live(p); _must(p, ft_model_equal(p.handle, a.idx, b.idx), "equal")
    _maybe_soft(p, soft); nothing
end

"""    fix!(p, v, value; soft = nothing)  — `v` must take `value`, in its own units."""
function fix!(p::Problem, v::Variable, value::Integer; soft::Union{Real, Nothing} = nothing)
    _live(p); _must(p, ft_model_fix(p.handle, v.idx, Clonglong(value)), "fix")
    _maybe_soft(p, soft); nothing
end

"""
Any number of literals, each naming its own variable and its own value.

`of` takes variables — in which case `value` applies to all of them, the common case — or literals,
which carry their own. "At most two of these nine shifts" and "at most one of `a = 3`, `b = 17`" are
both sayable.
"""
function _counting(p::Problem, kind::Integer, of, k::Integer, value::Integer, what::AbstractString;
                   soft::Union{Real, Nothing} = nothing)
    _live(p)
    items = collect(of)
    length(items) < 2 && error("$what needs at least two things to count, not $(length(items))")
    if kind <= 2 && !(0 <= k <= length(items))
        error("k must be between 0 and $(length(items)) for $what, not $k")
    end
    ft_model_lits_clear(p.handle)
    for it in items
        lit = it isa Literal ? it :
              it isa Variable ? Literal(it, Int64(value)) :
              error("$what counts variables or literals, not $(typeof(it))")
        _must(p, ft_model_lit(p.handle, lit.var.idx, lit.value), what)
    end
    if soft === nothing
        _must(p, ft_model_close(p.handle, Cuint(kind), Cuint(k)), what)
    else
        _must(p, ft_model_close_soft(p.handle, Cuint(kind), Cuint(k), Cdouble(soft)), what)
    end
    nothing
end

"""    exactly!(p, of, k; value = 1)  — exactly `k` of them hold."""
exactly!(p::Problem, of, k::Integer; value::Integer = 1, soft::Union{Real, Nothing} = nothing) =
    _counting(p, 0, of, k, value, "exactly"; soft = soft)

"""    at_most!(p, of, k; value = 1)  — at most `k` hold. Costs a slack variable."""
at_most!(p::Problem, of, k::Integer; value::Integer = 1, soft::Union{Real, Nothing} = nothing) =
    _counting(p, 1, of, k, value, "at_most"; soft = soft)

"""    at_least!(p, of, k; value = 1)  — at least `k` hold. Costs a slack variable."""
at_least!(p::Problem, of, k::Integer; value::Integer = 1, soft::Union{Real, Nothing} = nothing) =
    _counting(p, 2, of, k, value, "at_least"; soft = soft)

"""
    all_different!(p, vars)

Every one of these variables takes a different value.

Lowered per shared value rather than per pair, so it costs nothing where two domains do not overlap, needs no slack and no ancillas, and its violation names WHICH value collided and who took it. More variables than the values they share is refused when the model compiles, by name: the pigeonhole principle checked rather than annealed, because such a model has no answer at any penalty and a longer ladder cannot help.
"""
function all_different!(p::Problem, vars; soft::Union{Real, Nothing} = nothing)
    _live(p)
    items = collect(vars)
    length(items) < 2 && error("all_different needs at least two variables, not $(length(items))")
    ft_model_lits_clear(p.handle)
    seen = Int[]
    for v in items
        v isa Variable || error("all_different takes variables, not $(typeof(v))")
        Int(v.idx) in seen && continue
        push!(seen, Int(v.idx))
        # ft_model_var, not a placeholder value: the library picks one from the variable's own
        # domain, which a caller has no reason to know.
        _must(p, ft_model_var(p.handle, v.idx), "all_different")
    end
    _must(p, ft_model_close(p.handle, Cuint(5), Cuint(0)), "all_different")
    soft === nothing || soften_last!(p, soft)
    nothing
end

"""    exactly_one!(p, of)  — exactly one holds. Pairwise, no slack, so cheaper than `exactly!(…, 1)`."""
exactly_one!(p::Problem, of; value::Integer = 1, soft::Union{Real, Nothing} = nothing) =
    _counting(p, 3, of, 0, value, "exactly_one"; soft = soft)

"""    at_most_one!(p, of)  — at most one holds."""
at_most_one!(p::Problem, of; value::Integer = 1, soft::Union{Real, Nothing} = nothing) =
    _counting(p, 4, of, 0, value, "at_most_one"; soft = soft)

# -- objective -----------------------------------------------------------------------------------

function _objective(p::Problem, maximize::Bool, terms)
    _live(p)
    m = Cuint(maximize ? 1 : 0)
    for t in terms
        w, lit = t
        lits = lit isa Literal ? (lit,) :
               lit isa Union{Tuple, AbstractVector} ? Tuple(lit) :
               error("an objective term is (weight, literal) or (weight, (l1, l2, ...)), " *
                     "not (weight, $(typeof(lit)))")
        all(l -> l isa Literal, lits) ||
            error("every factor in a product must be a literal; got $(typeof.(lits))")
        if length(lits) == 1
            a = lits[1]
            _must(p, ft_model_objective_term(p.handle, m, Cdouble(w), a.var.idx, a.value),
                  "objective")
        elseif length(lits) == 2
            a, b = lits
            _must(p, ft_model_objective_pair(p.handle, m, Cdouble(w), a.var.idx, a.value,
                                             b.var.idx, b.value), "objective")
        else
            # Three or more: the library's literal list, closed as a product. `compile` lowers it
            # with an ancilla spin per substituted pair, so it costs spins the model did not
            # declare, and the guarantee is about optimisation rather than sampling.
            ft_model_lits_clear(p.handle)
            for l in lits
                _must(p, ft_model_lit(p.handle, l.var.idx, l.value), "objective")
            end
            _must(p, ft_model_objective_product(p.handle, m, Cdouble(w)), "objective")
        end
    end
    nothing
end

"""
    maximize!(p, terms)

Prefer states where the terms hold. Each term is `(weight, literal)`, or `(weight, (a, b, ...))` to
reward several literals holding **together**.

Three or more in one product is a higher-order term: it is lowered with an ancilla spin per
substituted pair, so it costs spins the model did not declare, and the guarantee is about finding
optima rather than about sampling.

Terms accumulate and each carries its own direction, so a later [`minimize!`](@ref) changes only
what it names.
"""
maximize!(p::Problem, terms) = _objective(p, true, terms)

"""    minimize!(p, terms)  — prefer states where the terms do not hold."""
minimize!(p::Problem, terms) = _objective(p, false, terms)

"""
    penalty!(p, value)

Use exactly this penalty, disabling the automatic scaling.

The remedy when an answer comes back infeasible: a constraint lost to the objective and has to
outrank it. By default the penalty is twice the largest objective weight.
"""
penalty!(p::Problem, value::Real) =
    (_live(p); _must(p, ft_model_fixed_penalty(p.handle, Cdouble(value)), "penalty"); nothing)

# -- solving -------------------------------------------------------------------------------------

function _text(p::Problem, fn, i)
    need = fn(p.handle, Cuint(i), Ptr{UInt8}(C_NULL), Cuint(0))
    need == 0 && return ""
    buf = Vector{UInt8}(undef, need)
    got = fn(p.handle, Cuint(i), pointer(buf), Cuint(need))
    String(buf[1:got])
end

"""
    solve!(p; tries = 12, beta_hot = 0, beta_cold = 0, stages = 0, sweeps = 0)

Compile and solve, keeping the best of `tries` anneals.

The four ladder parameters default to the library's own; a zero means "use that default", so a
caller who measured their instance can override only what they measured. Lengthen the ladder when a
model stays infeasible at a large penalty — that is a model not being annealed enough, and no
penalty fixes it.
"""
function solve!(p::Problem; tries::Integer = 12, beta_hot::Real = 0, beta_cold::Real = 0,
                stages::Integer = 0, sweeps::Integer = 0)
    _live(p)
    spins = ft_model_compile(p.handle)
    spins == 0 && error(isempty(_why(p)) ? "this problem did not compile" : _why(p))
    ok = ft_model_solve_with(p.handle, Cuint(max(1, tries)), Cdouble(beta_hot), Cdouble(beta_cold),
                             Cuint(stages), Cuint(sweeps))
    ok == 0 && error("that annealing ladder is not usable: beta_cold must exceed beta_hot, and " *
                     "both must be real numbers. Pass 0 for any of the four to use the default.")

    vals = Dict{String, Union{Int64, Nothing}}()
    for v in p.vars
        got = ft_model_value(p.handle, v.idx)
        # A variable that did not decode reads `nothing` rather than raising: its encoding was
        # violated, which means a penalty lost, and knowing WHICH one lost is the diagnosis.
        vals[v.name] = got == typemin(Int64) ? nothing : got
    end
    # Hard and soft violations come back on one list and are separated here, because they answer
    # different questions: a broken rule says the answer is not an answer, a traded preference says
    # the solver made the choice it was asked to price.
    broken, given_up, by = String[], String[], Float64[]
    for i in 0:(ft_model_violations(p.handle) - 1)
        text = _text(p, ft_model_violation, i)
        hard = ft_model_violation_is_hard(p.handle, Cuint(i)) == 1
        push!(hard ? broken : given_up, text)
        # How far outside it sits, not merely that it broke. A caller ranking repairs, or deciding
        # whether a larger penalty would be enough, cannot get that from the text.
        hard && push!(by, ft_model_violation_amount(p.handle, Cuint(i)))
    end
    Answer(vals, ft_model_feasible(p.handle) == 1, broken,
           ft_model_energy(p.handle), Int(spins), ft_model_penalty(p.handle),
           ft_model_soft_cost(p.handle), given_up, by, Int(ft_model_ancillas(p.handle)),
           [_text(p, ft_model_caveat, i) for i in 0:(ft_model_caveats(p.handle) - 1)])
end

"""
    ommx(p) -> (Vector{UInt8}, Float64)

The compiled model as an OMMX instance — the interchange format this corner of the field converged
on, so a ferrotherm program can be read by jijmodeling and the Jij stack.

Returns the protobuf bytes and the offset the ±1 to 0/1 substitution produced, **already folded
into the instance**: `ommx_objective(x) == ferrotherm_energy(s)` exactly. Read the constant, do not
add it — adding it again is wrong by its own value.
"""
function ommx(p::Problem)
    _live(p)
    need = ft_model_ommx(p.handle, C_NULL, Cuint(0))
    need == 0 && error("compile or solve the problem first; there is no instance yet")
    buf = Vector{UInt8}(undef, need)
    got = ft_model_ommx(p.handle, pointer(buf), Cuint(need))
    (buf[1:got], ft_model_ommx_constant(p.handle))
end

"""
    from_ommx(data; beta = 1.0, seed = 0) -> (Simulation, Float64)

Read an `ommx.v1.Instance` and return a simulation over it, plus the constant.

The direction that makes this a bridge rather than an exporter: a problem someone else compiled to
OMMX becomes something this sampler can run. `ommx_objective(x) == energy(sim) + constant`.

Errors naming what could not be read — a continuous variable, a bound that is not `[0,1]`, an
objective of degree three or more — rather than silently dropping it.
"""
function from_ommx(data::AbstractVector{UInt8}; beta::Real = 1.0, seed::Integer = 0)
    constant = Ref{Cdouble}(0.0)
    h = ft_ommx_read(pointer(data), Cuint(length(data)), Cdouble(beta), Culonglong(seed), constant)
    if h == C_NULL
        need = ft_ommx_error(C_NULL, Cuint(0))
        why = "that is not an instance this sampler can read"
        if need > 0
            buf = Vector{UInt8}(undef, need)
            got = ft_ommx_error(pointer(buf), Cuint(need))
            why = String(buf[1:got])
        end
        error(why)
    end
    (Simulation(h, "a simulation from that OMMX instance"), constant[])
end

"""
    certify!(p; beta = 1.0, draws = 512, thin = 1)

Ask whether the sampler that produced the answer was sampling what it claimed.

An answer says *what*. This says whether the machine that produced it reached the temperature it was
asked for, how many of its draws were independent, and — where the model is small enough to
enumerate — how far its distribution sits from the exact one, beside the noise floor that distance
has to beat. `findings` is empty exactly when the run was sound.
"""
function certify!(p::Problem; beta::Real = 1.0, draws::Integer = 512, thin::Integer = 1)
    _live(p)
    ft_model_certify(p.handle, Cdouble(beta), Cuint(draws), Cuint(max(1, thin))) == 0 &&
        error(isempty(_why(p)) ?
              "could not certify: compile and solve the problem first, and pass a positive beta " *
              "with at least 16 draws" : _why(p))
    msgs = [_text(p, ft_model_cert_finding, i) for i in 0:(ft_model_cert_findings(p.handle) - 1)]
    nan2n(x) = isnan(x) ? nothing : x
    Certificate(Int(draws), ft_model_cert_beta(p.handle), (NaN, NaN),
                ft_model_cert_tau(p.handle), ft_model_cert_ess(p.handle),
                nan2n(ft_model_cert_tv(p.handle)), nan2n(ft_model_cert_floor(p.handle)), msgs)
end

"""    ftp(p)  — the compiled program in `.ftp` form, which runs unchanged on any backend."""
function ftp(p::Problem)
    _live(p)
    need = ft_model_ftp(p.handle, Ptr{UInt8}(C_NULL), Cuint(0))
    need == 0 && return ""
    buf = Vector{UInt8}(undef, need)
    got = ft_model_ftp(p.handle, pointer(buf), Cuint(need))
    String(buf[1:got])
end

end # module
