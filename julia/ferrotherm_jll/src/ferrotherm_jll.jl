"""
    ferrotherm_jll

The compiled ferrotherm library, per platform.

Julia has no wheels, so a package needing a native library depends on a JLL: a package that carries
prebuilt binaries and hands you the path. This is that package for
[`Ferrotherm`](https://github.com/dcharlot-physicalai-bmi/ferrotherm).

The usual route to a JLL is a recipe submitted to Yggdrasil, built and reviewed by other people.
This one is built by the Institute's own CI, out of the same job that builds the Python wheels, and
the binaries are attached to the GitHub release. `Artifacts.toml` names each by URL **and by hash**,
so Julia refuses anything that does not match — which is what makes a self-hosted artifact exactly
as trustworthy as a registry-hosted one. The hash lives in the package; only the bytes live
elsewhere.

    using ferrotherm_jll
    ferrotherm_jll.libferrotherm        # the path, for ccall or Libdl.dlopen

Most people want `Ferrotherm` instead, which uses this and gives you a sampler.
"""
module ferrotherm_jll

using Artifacts, Libdl

export libferrotherm

"""Absolute path of the shared library for this platform."""
const libferrotherm = Ref{String}()

"""Was a library found? False only if this platform has no artifact."""
isavailable() = isassigned(libferrotherm) && !isempty(libferrotherm[])

function __init__()
    dir = try
        artifact"ferrotherm"
    catch err
        # A platform with no artifact is not a crash. Ferrotherm falls back to FERROTHERM_LIB and
        # says so, which is a better failure than a package that will not load at all.
        @debug "no ferrotherm artifact for this platform" exception = err
        libferrotherm[] = ""
        return
    end
    for sub in ("lib", "bin")
        d = joinpath(dir, sub)
        isdir(d) || continue
        for f in readdir(d)
            if startswith(f, "libferrotherm") || startswith(f, "ferrotherm.")
                libferrotherm[] = joinpath(d, f)
                return
            end
        end
    end
    libferrotherm[] = ""
end

end # module
