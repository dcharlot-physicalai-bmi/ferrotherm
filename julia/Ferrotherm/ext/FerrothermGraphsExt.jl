"""
    FerrothermGraphsExt

Build a model from a `Graphs.jl` graph.

`Graphs.SimpleGraph` is the shared vocabulary of the Julia graph world — `GenericTensorNetworks`,
`SpinGlassNetworks` and essentially every lattice construction build on it — so accepting one means
`Graphs.grid([32, 32])` and D-Wave-style topologies work without anybody writing an adapter.
"""
module FerrothermGraphsExt

import Ferrotherm
import Graphs

"""
    IsingModel(g; J = 1.0, biases = nothing)

A model on the graph's edges. `J` is either a scalar applied to every edge, or a callable
`(src, dst) -> weight`. `biases` is either `nothing`, a scalar applied to every vertex, or a vector.

Vertices are 1-based in both Graphs.jl and here, so nothing is renumbered.
"""
function Ferrotherm.IsingModel(g::Graphs.AbstractGraph; J = 1.0, biases = nothing)
    Graphs.is_directed(g) &&
        throw(ArgumentError("an Ising coupling is symmetric; pass an undirected graph"))
    n = Graphs.nv(g)
    m = Ferrotherm.IsingModel(n)
    for e in Graphs.edges(g)
        s, d = Graphs.src(e), Graphs.dst(e)
        w = J isa Function ? J(s, d) : J
        w == 0 && continue
        Ferrotherm.couple!(m, s, d, w)
    end
    if biases !== nothing
        for i in 1:n
            h = biases isa AbstractVector ? biases[i] : biases
            h == 0 && continue
            Ferrotherm.bias!(m, i, h)
        end
    end
    m
end

end # module
