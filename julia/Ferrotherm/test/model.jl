# The modelling layer: variables, constraints, an objective, answered by name.
#
# Nothing here asserts on a particular random draw. Sampling is stochastic, so every assertion is
# about a property the answer must have for any correct solver.

@testset "a triangle needs three colours" begin
    p = Problem()
    r = Dict(n => categorical!(p, n, 3) for n in ("west", "middle", "east"))
    not_equal!(p, r["west"], r["middle"])
    not_equal!(p, r["middle"], r["east"])
    not_equal!(p, r["west"], r["east"])
    a = solve!(p; tries = 12)
    @test feasible(a)
    @test length(Set(a[n] for n in keys(r))) == 3
end

@testset "an integer is written in its own values, not in slots" begin
    p = Problem()
    t = integer!(p, "temperature", 10:20)
    maximize!(p, [(5.0, is(t, 13))])
    @test solve!(p; tries = 8)["temperature"] == 13

    # a slot index where a value belongs is refused, and the message names the range
    q = Problem()
    u = integer!(q, "temperature", 10:20)
    e = try fix!(q, u, 3); "" catch err; sprint(showerror, err) end
    @test occursin("temperature", e)
    @test occursin("10..=20", e)
end

@testset "a range below zero survives the boundary" begin
    p = Problem()
    d = integer!(p, "drift", -40:-10)
    fix!(p, d, -25)
    @test solve!(p; tries = 8)["drift"] == -25
end

@testset "at most k is a ceiling and at least k is a floor" begin
    # With every variable rewarded, `at most 2` and `exactly 2` agree. With every variable
    # penalised they do not: an inequality is satisfied by taking none, an equality is not.
    function run(f, sign)
        p = Problem()
        v = [binary!(p, "v$i") for i in 1:4]
        f(p, v, 2)
        maximize!(p, [(sign * (5 - i), is(v[i], 1)) for i in 1:4])
        a = solve!(p; tries = 16)
        @test feasible(a)
        count(i -> a["v$i"] == 1, 1:4)
    end

    @test run(at_most!, +1) == 2      # a ceiling binds against a reward pushing past it
    @test run(at_most!, -1) == 0      # and is satisfied by taking none
    @test run(exactly!, -1) == 2      # where an equality still has to take two
    @test run(at_least!, +1) == 4     # a floor does not forbid taking more
    @test run(at_least!, -1) == 2     # and holds against a reward pushing below it
end

@testset "a counting constraint can be any length" begin
    p = Problem()
    shifts = [binary!(p, "s$i") for i in 1:9]
    at_most!(p, shifts, 2)
    maximize!(p, [(10.0 - i, is(shifts[i], 1)) for i in 1:9])
    a = solve!(p; tries = 24)
    @test feasible(a)
    @test count(i -> a["s$i"] == 1, 1:9) == 2
end

@testset "literals in one constraint may name different values" begin
    p = Problem()
    x = categorical!(p, "x", 4)
    y = integer!(p, "y", 10:20)
    at_most!(p, [is(x, 3), is(y, 17)], 1)
    maximize!(p, [(5.0, is(x, 3)), (4.0, is(y, 17))])
    a = solve!(p; tries = 16)
    @test feasible(a)
    @test (a["x"] == 3) + (a["y"] == 17) <= 1
    @test a["x"] == 3          # and it keeps the more valuable of the two
end

@testset "exactly one and at most one" begin
    function run(f, sign)
        p = Problem()
        v = [binary!(p, "v$i") for i in 1:5]
        f(p, v)
        maximize!(p, [(sign * 1.0, is(x, 1)) for x in v])
        a = solve!(p; tries = 16)
        @test feasible(a)
        (count(i -> a["v$i"] == 1, 1:5), a.spins)
    end
    @test run(exactly_one!, -1)[1] == 1     # one, even pushed off
    @test run(at_most_one!, -1)[1] == 0     # none, when pushed off
    @test run(at_most_one!, +1)[1] == 1     # and one when pulled on

    # neither pays for a slack variable, where the inequality form does
    p = Problem()
    v = [binary!(p, "v$i") for i in 1:5]
    at_most!(p, v, 1)
    @test solve!(p; tries = 4).spins > run(at_most_one!, -1)[2]
end

@testset "feasible means the constraints hold" begin
    # A penalty makes a constraint EXPENSIVE, not impossible. Pinned below the objective, the
    # sampler pays it: every variable decodes cleanly and the constraint is broken.
    p = Problem()
    a = categorical!(p, "a", 3); b = categorical!(p, "b", 3)
    not_equal!(p, a, b)
    penalty!(p, 1.0)
    maximize!(p, [(40.0, is(a, 1)), (40.0, is(b, 1))])
    ans = solve!(p; tries = 16)
    @test ans["a"] == ans["b"] == 1
    @test !feasible(ans)
    @test length(violated(ans)) == 1
    @test occursin("must differ", violated(ans)[1])

    # raised, the same model is feasible
    q = Problem()
    x = categorical!(q, "a", 3); y = categorical!(q, "b", 3)
    not_equal!(q, x, y)
    penalty!(q, 200.0)
    maximize!(q, [(40.0, is(x, 1)), (40.0, is(y, 1))])
    ok = solve!(q; tries = 16)
    @test feasible(ok)
    @test ok["a"] != ok["b"]
end

@testset "objective terms accumulate and mixed directions compose" begin
    p = Problem()
    v = [binary!(p, "v$i") for i in 1:4]
    maximize!(p, [(1.0, is(v[i], 1)) for i in 1:3])
    minimize!(p, [(1.0, is(v[4], 1))])
    a = solve!(p; tries = 16)
    @test [a["v$i"] for i in 1:4] == [1, 1, 1, 0]
end

@testset "a quadratic term rewards agreement" begin
    p = Problem()
    a = categorical!(p, "a", 3); b = categorical!(p, "b", 3)
    maximize!(p, [(4.0, (is(a, 2), is(b, 2)))])
    ans = solve!(p; tries = 16)
    @test (ans["a"], ans["b"]) == (2, 2)
end

@testset "two variables cannot share a name" begin
    # An answer is keyed by name, so a second variable with the same one would replace the first
    # rather than shadow it, and one of the two would vanish from the result.
    p = Problem()
    binary!(p, "shift")
    @test_throws ErrorException binary!(p, "shift")
end

@testset "a caller's own ladder is used, and a bad one refused" begin
    p = Problem()
    a = categorical!(p, "a", 3); b = categorical!(p, "b", 3)
    not_equal!(p, a, b)
    @test feasible(solve!(p; tries = 8, beta_hot = 0.05, beta_cold = 6.0, stages = 60, sweeps = 20))
    @test feasible(solve!(p; tries = 8))      # zeros mean the library's own ladder
    @test_throws ErrorException solve!(p; tries = 8, beta_hot = 8.0, beta_cold = 0.05)
end

@testset "errors name what the caller wrote" begin
    p = Problem()
    x = categorical!(p, "colour", 3)
    @test_throws ErrorException categorical!(p, "colour", 3)
    @test_throws ErrorException fix!(p, x, 9)
    @test_throws ErrorException at_most!(p, [x], 1)
    @test_throws ErrorException categorical!(p, "bad", 1)
end

@testset "a certificate reports on the sampler, not the answer" begin
    p = Problem()
    x = categorical!(p, "x", 4)
    fix!(p, x, 2)
    solve!(p; tries = 8)
    c = certify!(p; beta = 1.0, draws = 512)
    @test !isnan(c.beta_eff)
    @test c.ess > 0
    @test passed(c) == isempty(findings(c))
    # a TV without its floor is not a measurement
    @test (c.tv === nothing) == (c.noise_floor === nothing) || c.noise_floor !== nothing
end

@testset "the compiled program exports as ftp" begin
    p = Problem()
    a = categorical!(p, "a", 3); b = categorical!(p, "b", 3)
    not_equal!(p, a, b)
    solve!(p; tries = 4)
    text = ftp(p)
    @test startswith(text, "ftp 1")
    @test occursin("spins 6", text)
end

@testset "a higher-order objective term" begin
    # Three literals together, which the binding could not express: it took one or two.
    p = Problem()
    a = categorical!(p, "a", 3); b = categorical!(p, "b", 3); c = categorical!(p, "c", 3)
    maximize!(p, [(9.0, (is(a, 2), is(b, 2), is(c, 2)))])
    ans = solve!(p; tries = 24)
    @test feasible(ans)
    @test (ans["a"], ans["b"], ans["c"]) == (2, 2, 2)
    @test ans.spins > 9      # three categoricals are 9 spins; the ancilla makes it more

    # a vector of literals works too, since that is how one gets built in a loop
    q = Problem()
    v = [binary!(q, "v$i") for i in 1:3]
    maximize!(q, [(6.0, [is(x, 1) for x in v])])
    minimize!(q, [(1.0, is(v[1], 1))])
    a2 = solve!(q; tries = 24)
    @test feasible(a2)
    @test all(a2["v$i"] == 1 for i in 1:3)

    # and something that is not a literal is refused by name
    r = Problem(); x = categorical!(r, "x", 3)
    @test_throws ErrorException maximize!(r, [(1.0, (is(x, 1), "not a literal"))])
end

@testset "an encoding can be chosen and costs what it says" begin
    function spins(enc)
        p = Problem()
        categorical!(p, "a", 6; encoding = enc)
        solve!(p; tries = 4).spins
    end
    @test spins(:onehot) == 6        # one spin per value
    @test spins(:domainwall) == 5    # one fewer
    @test spins(:binary) == 3        # log2 of the domain, and the cheapest by far

    # the two that can carry a literal both do
    for enc in (:onehot, :domainwall)
        p = Problem()
        a = categorical!(p, "a", 6; encoding = enc)
        fix!(p, a, 3)
        ans = solve!(p; tries = 16)
        @test feasible(ans)
        @test ans["a"] == 3
    end

    # an integer is an ORDERED domain, where a domain wall is often the better choice
    p = Problem()
    t = integer!(p, "t", 10:20; encoding = :domainwall)
    fix!(p, t, 17)
    ans = solve!(p; tries = 16)
    @test ans["t"] == 17
    @test ans.spins == 10            # eleven values in ten spins

    @test_throws ErrorException categorical!(Problem(), "x", 3; encoding = :rot13)
end

@testset "a weighted linear row is a constraint, not a preference" begin
    # 3a + 4b + 5c ≤ 7, which no counting constraint can say: they all count UNWEIGHTED literals.
    p = Problem()
    a = binary!(p, "a"); b = binary!(p, "b"); c = binary!(p, "c")
    linear!(p, [(a, 3), (b, 4), (c, 5)], :le, 7)
    maximize!(p, [(1.0, is(a, 1)), (1.0, is(b, 1)), (1.0, is(c, 1))])
    ans = solve!(p; tries = 32)
    @test feasible(ans)
    @test 3 * ans["a"] + 4 * ans["b"] + 5 * ans["c"] == 7   # 3 + 4 is the best that fits

    # A row nothing can satisfy is refused by arithmetic rather than annealed.
    q = Problem()
    x = binary!(q, "x"); y = binary!(q, "y")
    linear!(q, [(x, 3), (y, 4)], ">=", 9)
    @test_throws ErrorException solve!(q; tries = 4)

    # A relation nobody defined is refused before anything is built.
    r = Problem(); z = binary!(r, "z")
    @test_throws ErrorException linear!(r, [(z, 1)], :lt, 1)
end

@testset "a soft weighted row is priced in the modeller's own units" begin
    p = Problem()
    a = binary!(p, "a"); b = binary!(p, "b")
    linear!(p, [(a, 3), (b, 4)], :le, 3; soft = 0.5)
    maximize!(p, [(10.0, is(a, 1)), (10.0, is(b, 1))])
    ans = solve!(p; tries = 32)
    @test feasible(ans)                 # a soft row leaves the answer an answer
    @test ans["a"] == 1 && ans["b"] == 1
    # 3 + 4 = 7 against a bound of 3 is 4 over, priced at 0.5 × 4² = 8.
    @test soft_cost(ans) == 8.0
end
