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
rates but has no notion of a target distribution at all. **Nothing anywhere reports the inverse
temperature a sampler actually achieved, and nothing anywhere reports a sampling-noise floor.**
Those are [`certify`](@ref)'s job.

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
export sweep!, anneal!, beta!, spins, energy, magnetization
export certify, findings, passed
export known_optimum, excess, solved
export treewidth, exact_ground_energy, exact_logz
export node_updates, joules_z1, onsager, library_path, close!

# ---- finding the library -----------------------------------------------------------------------

const LIB = Ref{String}("")

function _candidates()
    out = String[]
    haskey(ENV, "FERROTHERM_LIB") && push!(out, ENV["FERROTHERM_LIB"])
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
    error("""
          could not load the ferrotherm native library.

          Build it with `cargo build --release` in the ferrotherm checkout, or set
          ENV["FERROTHERM_LIB"] to the library path before `using Ferrotherm`.

          Tried:
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
@cfn ft_onsager Cdouble Cdouble
@cfn ft_free Cvoid SimPtr

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

end # module
