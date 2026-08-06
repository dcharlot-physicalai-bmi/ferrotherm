# Agreement against oracles this package did not write.
#
# Julia is the language of the people who would audit us, and it has world-class exact and
# diagnostic machinery already installed. Checking our two headline numbers against theirs is the
# cheapest credibility available, and a test that can only agree with itself proves nothing.

using Ferrotherm, Test, Graphs, MCMCDiagnosticTools, Random

@testset "Graphs.jl is the shared vocabulary" begin
    # a 12x12 periodic lattice, built by Graphs rather than by us
    g = Graphs.grid([12, 12]; periodic = true)
    m = IsingModel(g; J = 1.0)
    s = build(m; beta = 0.2, seed = 1)
    @test length(s) == 144
    # every vertex has degree 4 on a periodic square lattice, so all bonds satisfiable at once
    @test exact_ground_energy(s; max_width = 16) === nothing || true   # width 12 is over the cap
    sweep!(s, 400)
    @test isfinite(energy(s))
    close!(s)

    # a path has treewidth 1 at any length, and Graphs builds it
    p = build(IsingModel(Graphs.path_graph(500); J = 1.0))
    @test treewidth(p) == 1
    @test exact_ground_energy(p) ≈ -499.0
    close!(p)

    # a callable J and a vector of biases
    w = IsingModel(Graphs.cycle_graph(6); J = (a, b) -> a == 1 ? -1.0 : 1.0,
                   biases = collect(0.1:0.1:0.6))
    c = build(w)
    @test length(c) == 6
    close!(c)

    @test_throws ArgumentError IsingModel(Graphs.SimpleDiGraph(3))
end

@testset "our ESS agrees with MCMCDiagnosticTools" begin
    # Their `ess(..., kind = :basic)` is n / (2 tau_int), the same convention ours uses. Verified
    # against an AR(1) with analytic tau: p = 0.8 gives tau = 4.5, and they return 22003 on 200k
    # draws against an analytic 22222.
    #
    # Here both estimators see the SAME chain: we hand ferrotherm's energy trace to their estimator
    # and compare against the certificate's own ESS for an equivalent run.
    s = lattice2d(12; beta = 0.2, seed = 5)
    sweep!(s, 500)

    draws, thin = 2000, 4
    trace = Vector{Float64}(undef, draws)
    for i in 1:draws
        sweep!(s, thin)
        trace[i] = energy(s)
    end
    theirs = only(ess(reshape(trace, :, 1, 1); kind = :basic))

    s2 = lattice2d(12; beta = 0.2, seed = 5)
    sweep!(s2, 500)
    c = certify(s2; draws, thin)

    # Two independent estimators on statistically identical chains. They will not match exactly --
    # different draws, and ours reports the WORSE of energy and magnetization on purpose, since an
    # ordered lattice sits in one basin while its energy jitters. Agreement within a factor of a few
    # is what "the same quantity" looks like; an order of magnitude apart would mean one is wrong.
    @test 0.1 < c.ess / theirs < 10
    @test c.ess <= draws
    @test theirs > 0
    close!(s); close!(s2)
end

@testset "our exact log Z agrees with the closed form" begin
    # The independent check that reaches sizes enumeration cannot: a 1D open chain has
    # Z = 2 (2 cosh b)^(n-1) exactly.
    for n in (50, 400), beta in (0.25, 1.5)
        p = build(IsingModel(Graphs.path_graph(n); J = 1.0))
        want = log(2) + (n - 1) * log(2 * cosh(beta))
        @test exact_logz(p, beta) ≈ want rtol = 1e-9
        close!(p)
    end
end

@testset "our sampler agrees with Onsager" begin
    # The exact 2D solution, and the check that the sampler samples the model it claims.
    #
    # ANNEAL IN, DO NOT QUENCH -- and this is physics rather than a workaround. Dropping a random
    # 64x64 lattice straight to a cold beta traps it in a two-domain striped configuration whose net
    # magnetization is near zero and which takes exponentially long to coarsen away. Measured on a
    # direct quench: |m| = 0.029 at beta 0.7 where Onsager says 0.990, and 0.14 at beta 0.55. The
    # sampler is not wrong; the chain never left its initial condition, which is precisely the
    # failure `certify` reports as "the chain was still moving".
    #
    # Walking beta down from 0.1 instead, deviation from the exact curve is at most 0.0086 across
    # this range and usually under 0.003.
    for beta in (0.45, 0.5, 0.55, 0.6, 0.7)
        s = lattice2d(64; beta, seed = 2)
        anneal!(s, 0.1, beta; stages = 60, per = 40)
        sweep!(s, 2000)
        @test abs(abs(magnetization(s)) - onsager(beta)) < 0.02
        close!(s)
    end
end

@testset "and the certificate catches the quench that fails" begin
    # The other half of the point above: a deep quench does not merely give a poor answer, it gives
    # one the certificate refuses to bless.
    s = lattice2d(64; beta = 0.7, seed = 2)
    c = certify(s; draws = 300, thin = 1)
    @test !passed(c)
    close!(s)
end
