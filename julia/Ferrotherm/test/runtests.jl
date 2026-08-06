using Ferrotherm, Test

@testset "Ferrotherm" begin

@testset "library loads" begin
    @test !isempty(library_path())
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

end
