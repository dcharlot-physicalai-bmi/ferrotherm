# Cross-checked against QUBODrivers' own ExactSampler on random instances.
#
# This is the only thing standing between us and a sign error, and a sign error here produces
# perfectly plausible numbers that are wrong on every problem. It caught one: our energy already
# equals their bracket, and an extra negation had crept into the value mapping.

using Test, QUBODrivers, QUBOTools, Ferrotherm, Random
using QUBODrivers: MOI

ext = Base.get_extension(Ferrotherm, :FerrothermQUBODriversExt)
@assert ext !== nothing "the extension did not load"

@testset "QUBODrivers extension" begin
    for (n, seed) in [(6, 1), (8, 2), (10, 3)]
        Random.seed!(seed)
        # build a spin model through MOI, the way a JuMP user's problem arrives
        model = MOI.instantiate(ext.Ferrotherm_.Optimizer; with_cache_type = Float64,
                                with_bridge_type = Float64)
        exact = MOI.instantiate(QUBODrivers.ExactSampler.Optimizer; with_cache_type = Float64,
                                with_bridge_type = Float64)
        mine  = MOI.instantiate(ext.Exact.Optimizer; with_cache_type = Float64,
                                with_bridge_type = Float64)

        J = Dict{Tuple{Int,Int},Float64}()
        h = randn(n)
        for i in 1:n, j in (i+1):n
            rand() < 0.5 && (J[(i, j)] = randn())
        end

        function build!(opt)
            x, _ = MOI.add_constrained_variables(opt, fill(Spin(), n))
            terms = MOI.ScalarQuadraticTerm{Float64}[]
            aff   = MOI.ScalarAffineTerm{Float64}[]
            for ((i, jj), v) in J
                push!(terms, MOI.ScalarQuadraticTerm(v, x[i], x[jj]))
            end
            for i in 1:n
                push!(aff, MOI.ScalarAffineTerm(h[i], x[i]))
            end
            MOI.set(opt, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
                    MOI.ScalarQuadraticFunction(terms, aff, 0.0))
            MOI.set(opt, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            opt
        end

        MOI.optimize!(build!(exact))
        MOI.optimize!(build!(mine))
        MOI.optimize!(build!(model))

        want = MOI.get(exact, MOI.ObjectiveValue(1))
        got_exact = MOI.get(mine, MOI.ObjectiveValue(1))
        got_anneal = MOI.get(model, MOI.ObjectiveValue(1))

        @test isapprox(got_exact, want; atol = 1e-8)     # our exact must MATCH their exact
        @test got_anneal >= want - 1e-8                  # annealing cannot beat the optimum
        @test isapprox(got_anneal, want; atol = 1e-6)    # and should find it at this size
    end
end
