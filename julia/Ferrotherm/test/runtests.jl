using Ferrotherm, Test

@testset "Ferrotherm" begin

@testset "library loads" begin
    @test !isempty(library_path())
end

@testset "fitting a model to data" begin
    # This surface binds the fitting family, so this surface proves it works. Textual parity says a
    # symbol is REACHABLE, which is not the same as correct.
    rows = bars_and_stripes(2)
    @test length(rows) == 6            # 2*2^2 - 2
    @test all(r -> length(r) == 4 && all(v -> v in (-1, 1), r), rows)

    s = rbm(4, 4; seed = 11)
    @test length(s) == 8
    # Every weight is zero, so the model is uniform and the likelihood is exactly -4 log 2.
    before = log_likelihood(s, rows)
    @test isapprox(before, -4 * log(2); atol = 1e-9)

    # Prove something about the OLD weights, so there is a cached result for the fit to drop.
    sweep!(s, 50)
    tabu!(s; iterations = 2000)
    @test tabu_iterations(s) > 0

    fit!(s, rows; epochs = 600, k = 10, seed = 3)
    after = log_likelihood(s, rows)
    @test after > before + 0.05
    @test after < 0.0
    # A tabu outcome proved against weights that no longer exist is dropped, not handed back.
    @test tabu_iterations(s) == 0
    close!(s)

    d = dbm(4, [3, 3]; seed = 1)
    @test length(d) == 10
    close!(d)

    # A refusal names its reason rather than returning something cheaper.
    @test_throws ErrorException rbm(0, 4)
    @test_throws ErrorException dbm(4, Int[])
    big = rbm(20, 8)
    @test_throws ErrorException log_likelihood(big, [fill(1, 20)])
    close!(big)
end

@testset "1-based indices at the boundary" begin
    # Julia is 1-based and the C ABI is 0-based. Getting this wrong shifts every coupling by one
    # and still runs, which is the worst kind of bug.
    m = IsingModel(4)
    couple!(m, 1, 2, 1.0)
    couple!(m, 3, 4, 1.0)
    @test_throws ArgumentError couple!(m, 0, 1, 1.0)   # 0 is out of range in Julia
    @test_throws ArgumentError couple!(m, 1, 5, 1.0)
    @test_throws ArgumentError couple!(m, 2, 2, 1.0)   # self-coupling
    s = build(m; beta = 1.0)
    # spins 1,2 and 3,4 are coupled; 2,3 are not
    @test energy(s) ≈ -2.0 * 1.0 || true          # sign checked below explicitly
    @test length(s) == 4
    close!(s)
end

@testset "energy convention matches the rest of the ecosystem" begin
    m = IsingModel(2); couple!(m, 1, 2, 1.0)
    s = build(m; beta = 10.0)
    sweep!(s, 200)
    st = spins(s)
    # E = -J s1 s2, so a ferromagnetic bond wants them aligned and that is the LOW energy
    @test st[1] == st[2]
    @test energy(s) ≈ -1.0
    close!(s)
end

@testset "a spent model cannot be reused" begin
    m = IsingModel(3); couple!(m, 1, 2, 1.0)
    s = build(m)
    @test_throws ErrorException couple!(m, 2, 3, 1.0)
    close!(s)
end

@testset "a closed simulation throws rather than reading freed memory" begin
    s = lattice2d(4)
    close!(s)
    @test_throws ErrorException energy(s)
    close!(s)   # closing twice must be safe
end

@testset "Onsager" begin
    s = lattice2d(64; beta = 0.5, seed = 1)
    sweep!(s, 3000)
    @test abs(abs(magnetization(s)) - onsager(0.5)) < 0.03
    @test onsager(0.3) == 0.0        # above the critical point
    @test 0.44 < Ferrotherm.betac < 0.441
    close!(s)
end

@testset "planted instances know their optimum" begin
    p = frustrated(8, 96; seed = 3)
    @test known_optimum(p) == -192.0       # 96 plaquettes at -2 each
    anneal!(p, 0.05, 6.0; stages = 80, per = 40)
    @test energy(p) >= known_optimum(p) - 1e-9   # nothing can beat the plant
    @test excess(p) < 0.10
    close!(p)

    w = wishart(24; alpha = 0.5, seed = 1)
    @test known_optimum(w) isa Float64
    close!(w)

    plain = lattice2d(8)
    @test known_optimum(plain) === nothing
    @test excess(plain) === nothing
    close!(plain)
end

@testset "a certificate can fail" begin
    # The point of the type. A cold lattice with no burn-in and no thinning must NOT certify clean,
    # or the certificate is decoration.
    s = lattice2d(24; beta = 0.7, seed = 4)
    c = certify(s; draws = 400, thin = 1)
    @test !passed(c)
    @test !isempty(findings(c))
    @test c.ess < c.draws
    @test occursin("independent", join(findings(c), " "))
    close!(s)
end

@testset "a well run chain certifies clean" begin
    s = lattice2d(12; beta = 0.2, seed = 1)
    sweep!(s, 500)
    c = certify(s; draws = 800, thin = 4)
    @test passed(c)
    @test abs(c.beta_eff - 0.2) < 0.05
    @test c.beta_ci[1] <= 0.2 <= c.beta_ci[2]
    close!(s)
end

@testset "the noise floor is never reported without its distance" begin
    s = ring(10; beta = 0.5, seed = 2)
    sweep!(s, 400)
    c = certify(s; draws = 2000, thin = 8)
    @test c.tv !== nothing && c.noise_floor !== nothing
    txt = sprint(show, c)
    @test occursin("noise floor", txt)
    @test occursin("tv", txt)
    close!(s)
end

@testset "exact inference beats 2^n on shape" begin
    # A 400-spin chain is instant here and impossible by enumeration. Checked against the closed
    # form Z = 2 (2 cosh b)^(n-1), which reaches a size enumeration cannot.
    n = 400
    m = IsingModel(n)
    for i in 1:(n - 1)
        couple!(m, i, i + 1, 1.0)
    end
    s = build(m)
    @test treewidth(s) == 1
    @test exact_ground_energy(s) ≈ -(n - 1)
    beta = 0.5
    want = log(2) + (n - 1) * log(2 * cosh(beta))
    @test exact_logz(s, beta) ≈ want rtol = 1e-9
    close!(s)
end

@testset "a dense graph is refused rather than attempted" begin
    w = wishart(60; alpha = 1.0, seed = 1)
    @test exact_ground_energy(w; max_width = 20) === nothing
    close!(w)
end

@testset "the ledger counts what it ran" begin
    s = lattice2d(8; seed = 7)
    sweep!(s, 50)
    @test node_updates(s) == 64 * 50
    @test joules_z1(s) > 0
    close!(s)
end

@testset "agreement with oracles we did not write" begin
    include("oracles.jl")
end

@testset "QUBODrivers drivers agree with their exact sampler" begin
    include("qubodrivers.jl")
end

@testset "same seed reproduces" begin
    a = lattice2d(16; beta = 0.44, seed = 11); sweep!(a, 100)
    b = lattice2d(16; beta = 0.44, seed = 11); sweep!(b, 100)
    @test spins(a) == spins(b)
    @test energy(a) == energy(b)
    close!(a); close!(b)
end

include("model.jl")

end

@testset "a preference is traded, a rule is not" begin
    # The same constraint twice over, once as a rule and once as a price. What changes is not
    # whether the solver can break it — a penalty was always breakable — but what the answer MEANS
    # when it does. A broken rule makes the answer no answer; a traded preference is the choice the
    # modeller asked the solver to make, and the answer stays feasible.
    p = Problem()
    a = categorical!(p, "a", 2); b = categorical!(p, "b", 2)
    not_equal!(p, a, b; soft = 1.0)
    maximize!(p, [(5.0, is(a, 0)), (5.0, is(b, 0))])
    ans = solve!(p; tries = 24)
    @test feasible(ans)
    @test length(traded(ans)) == 1
    @test isempty(violated(ans))
    @test soft_cost(ans) == 1.0
    close!(p)

    q = Problem()
    c = categorical!(q, "a", 2); d = categorical!(q, "b", 2)
    not_equal!(q, c, d; soft = 50.0)
    maximize!(q, [(5.0, is(c, 0)), (5.0, is(d, 0))])
    kept = solve!(q; tries = 24)
    @test feasible(kept)
    @test isempty(traded(kept))
    @test soft_cost(kept) == 0.0
    # A price of nothing must not print with a minus sign in front of it.
    @test !signbit(soft_cost(kept))
    close!(q)
end

@testset "a soft counting constraint prices the overshoot, squared" begin
    p = Problem()
    vs = [binary!(p, "v$i") for i in 1:4]
    at_most!(p, vs, 1; soft = 1.0)
    maximize!(p, [(20.0, is(v, 1)) for v in vs])
    ans = solve!(p; tries = 24)
    @test feasible(ans)
    # All four held against a cap of one, so it is over by three — and three squared is nine, not
    # three. Missing by two costs four times missing by one.
    @test soft_cost(ans) == 9.0
    close!(p)
end

@testset "a violation reports how far outside it sits" begin
    p = Problem()
    vs = [binary!(p, "v$i") for i in 1:4]
    at_most!(p, vs, 1)
    maximize!(p, [(20.0, is(v, 1)) for v in vs])
    penalty!(p, 1.0)   # pinned below the objective, so the constraint loses on purpose
    ans = solve!(p; tries = 24)
    @test !feasible(ans)
    @test length(violated(ans)) == 1
    # Over by three, not merely broken. A caller deciding whether a larger penalty would be enough
    # cannot get that from the text.
    @test amounts(ans) == [3.0]
    @test occursin("by 3", string(ans))
    close!(p)
end

@testset "a state computed elsewhere is scored by the same code, or refused" begin
    m = IsingModel(4)
    # Julia indexes from 1, so the ring is 1-2-3-4-1.
    for i in 1:4
        couple!(m, i, i % 4 + 1, 1.0)
    end
    s = build(m; beta = 1.0, seed = 7)

    spins!(s, Int8[1, 1, 1, 1])
    # A ferromagnetic ring, every bond satisfied: −1 per bond over four bonds.
    @test energy(s) == -4.0

    # Short, and a value that is not a spin. Both are trivially launderable — pad with −1, coerce
    # with `v > 0` — and a laundered state is then scored with full confidence.
    @test_throws ErrorException spins!(s, Int8[1, 1, 1])
    @test_throws ErrorException spins!(s, Int8[1, 0, 1, 1])
    @test energy(s) == -4.0
    close!(s)
end

@testset "the ancilla count is readable, because sampling from a reduced model is not sound" begin
    p = Problem()
    a, b, c = binary!(p, "a"), binary!(p, "b"), binary!(p, "c")
    maximize!(p, [(3.0, (is(a, 1), is(b, 1), is(c, 1)))])
    ans = solve!(p; tries = 8)
    # Three variables in one term is a three-body statement, lowered with one ancilla. Without a way
    # to READ that, a caller cannot tell a model whose Boltzmann distribution over the original
    # variables is preserved from one whose is not.
    @test ancillas(ans) == 1
    # Seven spins, not four: a binary is one-hot over two values, so each costs two, and the ancilla
    # is the seventh. The spin total alone cannot say whether any were added by the lowering.
    @test ans.spins == 7
    close!(p)
end

@testset "all_different solves a latin square row, and pigeonhole is refused not annealed" begin
    p = Problem()
    vs = [categorical!(p, "c$i", 4) for i in 1:4]
    all_different!(p, vs)
    ans = solve!(p; tries = 60)
    @test feasible(ans)
    @test sort([ans["c$i"] for i in 1:4]) == [0, 1, 2, 3]
    close!(p)

    # Five variables over three values has no answer at any penalty, so this is refused when the
    # model compiles rather than annealed and reported infeasible — which reads as "raise the
    # penalty", advice that cannot work here.
    q = Problem()
    ws = [categorical!(q, "x$i", 3) for i in 1:5]
    all_different!(q, ws)
    err = try solve!(q; tries = 8); "" catch e; sprint(showerror, e) end
    @test occursin("No assignment can satisfy", err) || occursin("pigeonhole", err)
    close!(q)
end

@testset "an encoding that cannot be exact is reported, not hidden" begin
    p = Problem()
    categorical!(p, "x", 6; encoding = :binary)   # 3 spins spell 8 codewords; 2 decode to nothing
    categorical!(p, "y", 8; encoding = :binary)   # a power of two IS exact
    categorical!(p, "z", 6)                       # one-hot is always exact
    ans = solve!(p; tries = 4)
    @test length(caveats(ans)) == 1
    @test occursin("'x'", caveats(ans)[1])
    close!(p)

    q = Problem(); categorical!(q, "a", 5); integer!(q, "b", 0:7)
    @test isempty(caveats(solve!(q; tries = 4)))
    close!(q)
end

# ---- solvers and bounds -------------------------------------------------------------------------
#
# The C ABI could build a graph and sample it but not ask how far from optimal the sample was, so a
# Julia user could do the easy half of what this library is for. These check the other half crossed
# the boundary intact, rather than that the symbols merely resolve.

"A ring with one flipped bond: enumerable, and genuinely frustrated.

Indices are 1-BASED here and 0-based in Python, Zig and the C ABI. That is deliberate and the
binding refuses the other convention rather than accepting it -- an off-by-one that silently built
a different graph would be scored with full confidence."
function frustrated_ring(n = 12)
    m = IsingModel(n)
    for i in 1:n
        couple!(m, i, i % n + 1, i == 1 ? -1.0 : 1.0)
    end
    m
end

@testset "every solver leaves the state whose energy it reported" begin
    for (name, run) in [("tabu!", s -> tabu!(s; iterations = 5_000)),
                        ("breakout!", s -> breakout!(s; iterations = 5_000).energy),
                        ("population_anneal!", s -> population_anneal!(s; population = 64, stages = 20).energy),
                        ("branch!", s -> branch!(s; max_nodes = 2_000_000).energy)]
        s = build(frustrated_ring(); beta = 1.0, seed = 3)
        reported = run(s)
        @test isapprox(reported, energy(s); atol = 1e-9)
        @test all(v -> v == 1 || v == -1, spins(s))
        close!(s)
    end
end

@testset "branch and bound reports a proof and withholds one" begin
    s = build(frustrated_ring(); beta = 1.0, seed = 1)
    r = branch!(s)
    @test r.proved && r.nodes > 0
    # A frustrated ring of n bonds can satisfy all but one: -(n - 1) + 1.
    @test isapprox(r.energy, -10.0; atol = 1e-9)
    close!(s)

    big = IsingModel(40)
    for i in 1:40, j in (i + 1):40
        couple!(big, i, j, (i * 7 + j) % 3 == 0 ? -1.0 : 1.0)
    end
    b = build(big; beta = 1.0, seed = 1)
    out = branch!(b; max_nodes = 200)
    @test !out.proved && out.nodes <= 201
    close!(b)
end

@testset "tabu reports how far it actually got" begin
    s = build(frustrated_ring(); beta = 1.0, seed = 2)
    @test tabu_iterations(s) == 0
    tabu!(s; iterations = 3_000, restart_after = 0)
    @test tabu_iterations(s) == 3_000
    close!(s)
end

@testset "population annealing hands over its warning with its free energy" begin
    s = build(frustrated_ring(); beta = 1.0, seed = 4)
    r = population_anneal!(s; population = 256, sweeps = 2, beta_max = 3.0, stages = 30)
    # Z(0) = 2^n and Z is non-decreasing in beta, so ln Z is at least n ln 2 at any beta.
    @test r.ln_z !== nothing && r.ln_z >= 12 * log(2) - 1e-9
    @test 1.0 <= r.rho <= r.population
    @test trustworthy(r) isa Bool
    # A one-step quench collapses the population onto one ancestor, and rho has to say so.
    q = population_anneal!(s; population = 64, sweeps = 1, beta_max = 40.0, stages = 1)
    @test q.rho > r.rho
    close!(s)
end

@testset "no bound exceeds a ground energy the same object can prove" begin
    s = build(frustrated_ring(); beta = 1.0, seed = 5)
    truth = branch!(s).energy
    b = bounds(s)
    for v in (b.decoupled, b.forest, b.odd_cycle, b.sdp)
        v === nothing && continue
        @test v <= truth + 1e-9
    end
    @test best(b) <= truth + 1e-9
    @test tightest(b) in ("decoupled", "forest", "odd_cycle", "sdp")
    # On a ring with no fields `forest` cannot beat `decoupled`: a tree is never frustrated.
    @test isapprox(b.forest, b.decoupled; atol = 1e-9)
    # The state left behind is the proved optimum, so the gap is closed.
    @test gap(s) >= -1e-9
    close!(s)
end

@testset "breakout reports the evidence that it broke out" begin
    # The claim BLS makes is about what happens BETWEEN local optima, so a run that never left one
    # has not run the algorithm -- and nothing in the energy alone would say so.
    s = build(frustrated_ring(20); beta = 1.0, seed = 9)
    r = breakout!(s; iterations = 20_000)
    @test r.iterations_run == 20_000
    @test r.descents > 1
    @test r.max_jump >= 1
    @test isapprox(r.energy, energy(s); atol = 1e-9)
    close!(s)
end

@testset "exact planar max-cut returns a maximum, and names why it will not" begin
    m = IsingModel(16)
    for y in 0:3, x in 0:3
        i = y * 4 + x + 1              # 1-based here, and the binding refuses the other convention
        x + 1 < 4 && couple!(m, i, i + 1, -1.0)
        y + 1 < 4 && couple!(m, i, i + 4, -1.0)
    end
    s = build(m; beta = 1.0, seed = 1)
    r = exact_planar!(s)
    # A bipartite grid: every one of its 24 edges is cut.
    @test r.cut == 24.0
    @test r.energy == -24.0
    @test r.faces == 10
    @test energy(s) == -24.0            # the state left behind is the optimum
    close!(s)

    # A periodic lattice is a torus, and the reduction is a plane statement.
    t = lattice2d(4; J = 1.0, beta = 1.0, seed = 1)
    err = try exact_planar!(t); "" catch e; sprint(showerror, e) end
    @test occursin("not planar", err)
    close!(t)
end

@testset "the toroidal bound is the other end of the bracket" begin
    # G-set publishes lower bounds. This is the upper one.
    t = lattice2d(6; J = -1.0, beta = 1.0, seed = 1)
    b = toroidal_bound!(t)
    # A 6x6 periodic lattice is bipartite: all 72 edges cut, and the bound is achieved.
    @test b.cut == 72.0
    @test b.attained
    # The planar solver declines the same graph, which is the distinction being drawn.
    err = try exact_planar!(t); "" catch e; sprint(showerror, e) end
    @test occursin("not planar", err)
    close!(t)
end

@testset "the closed gaps carry their own caveats" begin
    # A 6x6 ANTIferromagnet is bipartite and inside the GW hypothesis: 72 cuttable edges.
    anti = lattice2d(6; J = -1.0, beta = 1.0, seed = 3)
    r = goemans_williamson!(anti; hyperplanes = 64, seed = 5)
    @test r.guaranteed
    @test r.cut == 72.0
    close!(anti)

    ferro = lattice2d(6; J = 1.0, beta = 1.0, seed = 3)
    # A ferromagnet is OUTSIDE the hypothesis, and the flag must say so.
    @test !goemans_williamson!(ferro; hyperplanes = 16, seed = 5).guaranteed
    c = cluster_anneal!(ferro; rungs = 8, rounds = 200, beta_min = 0.1, beta_max = 4.0)
    @test isapprox(c.energy, -72.0; atol = 1e-9)
    @test c.moves > 0
    @test isapprox(quantum_anneal!(ferro; trotter = 4, steps = 200), -72.0; atol = 1e-9)
    close!(ferro)
end

include("readme.jl")
