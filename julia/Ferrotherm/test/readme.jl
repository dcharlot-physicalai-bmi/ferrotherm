# The README's own code blocks, executed.
#
# `python/README.md` was found to be a verbatim copy of the Rust one -- shipped to PyPI, telling
# people to run `cargo add`. That is what a README nobody executes becomes. This runs every ```julia
# block in `julia/README.md` and holds each `println` to the value written beside it.
#
# Writing the block it checks caught the signature: `lattice2d` takes KEYWORD arguments
# (`lattice2d(64; J = 1.0, beta = 0.6)`), and the first draft passed J positionally.

const README = joinpath(@__DIR__, "..", "..", "README.md")

"Every ```julia fence in the README, in order."
function julia_blocks()
    text = read(README, String)
    [m.captures[1] for m in eachmatch(r"```julia\n(.*?)```"s, text)]
end

@testset "README" begin
    blocks = julia_blocks()
    # A floor: if the fences are renamed the loop below would iterate over nothing and pass.
    @test length(blocks) >= 2

    for (i, src) in enumerate(blocks)
        # The install block is a Pkg REPL line, not runnable Julia. Everything else must run.
        startswith(strip(src), "pkg>") && continue

        mod = Module(Symbol("ReadmeBlock", i))
        Core.eval(mod, :(using Ferrotherm))
        # A file, not an IOBuffer: `redirect_stdout` takes a stream, and Julia 1.12 rejects a
        # buffer with a MethodError rather than accepting it quietly.
        captured = mktemp() do path, io
            redirect_stdout(io) do
                Core.eval(mod, Meta.parseall(src))
            end
            flush(io)
            read(path, String)
        end
        printed = split(captured, '\n')

        # A `# 0.9736` on a println line is a promise about THAT line's output.
        #
        # Matched per line, not against the whole buffer: the Python version of this check searched
        # all of stdout, so a drifted |M| was satisfied by the NEXT line's Onsager value and the
        # test stayed green through the exact defect it was written for.
        n = 1
        for line in split(src, '\n')
            occursin("println(", line) || continue
            here = n <= length(printed) ? printed[n] : ""
            n += 1
            occursin("#", line) || continue
            claim = split(line, "#"; limit = 2)[2]
            for m in eachmatch(r"-?\d+\.\d+", split(claim, "—")[1])
                @test occursin(m.match, here)
            end
        end
    end
end
