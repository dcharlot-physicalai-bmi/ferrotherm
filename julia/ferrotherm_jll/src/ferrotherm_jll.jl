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
# Required, not decorative. The artifact is declared `lazy` so `Pkg.instantiate` does not download
# it -- which is what lets a platform without one still load the package -- and Julia refuses to
# resolve a lazy artifact unless the package that owns it has imported this:
#
#   Artifact "ferrotherm" is a lazy artifact; package developers must call `using LazyArtifacts`
#
# It was missing, and CI failed with "the JLL resolved no library on this platform" while local
# runs passed -- because the artifact was already in ~/.julia/artifacts from a non-lazy build, so
# the lazy path was never taken here at all.
using LazyArtifacts

export libferrotherm

"""Absolute path of the shared library for this platform."""
const libferrotherm = Ref{String}()

"""Was a library found? False only if this platform has no artifact, or it could not be fetched."""
isavailable() = isassigned(libferrotherm) && !isempty(libferrotherm[])

"""
    why()

Why [`isavailable`](@ref) is false, as the error Julia actually raised.

Swallowing that error is what made a CI failure read as "the JLL resolved no library on this
platform" -- true, and silent about the platform mismatch or the failed download behind it. Empty
when a library was found.
"""
why() = REASON[]

const REASON = Ref{String}("")

function __init__()
    dir = try
        artifact"ferrotherm"
    catch err
        # A platform with no artifact is not a crash. Ferrotherm falls back to FERROTHERM_LIB and
        # says so, which is a better failure than a package that will not load at all -- but the
        # reason is KEPT, because "no library on this platform" without it is a dead end.
        REASON[] = sprint(showerror, err)
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
