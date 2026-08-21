#!/usr/bin/env julia
#
# Rebuild `julia/ferrotherm_jll/Artifacts.toml` from tarballs already on a GitHub release.
#
# `build-julia-artifacts.jl` builds the tarballs AND writes the manifest, which is right when it
# runs on a fresh release. It is the wrong tool when the tarballs uploaded fine and only the
# manifest commit failed -- which is what happened at v0.18.0: the release job built and uploaded
# all three platforms, then hit a rebase conflict in `Artifacts.toml` (a concurrent session had
# landed the v0.17.0 manifest) and printed "manifest rebase conflicted; commit it by hand".
#
# Committing it by hand means recomputing two hashes per platform from memory of how they were
# made, which is exactly the kind of step that goes subtly wrong and produces a JLL that RESOLVES
# and loads the wrong library. So this recomputes them from the published artifacts instead, with
# the same arithmetic the builder used -- extract the tree, `Pkg.GitTools.tree_hash` it, `sha256`
# the tarball -- and writes the same TOML.
#
#   julia scripts/rebuild-julia-manifest.jl <version> <dir-of-tarballs> [out]
#
# e.g. after `gh release download v0.18.0 -D /tmp/jll`:
#   julia scripts/rebuild-julia-manifest.jl 0.18.0 /tmp/jll julia/ferrotherm_jll/Artifacts.toml
#
# It NAMES the platforms it found rather than writing a manifest for whatever happened to be in the
# directory: a manifest silently missing a platform is a package that works everywhere you tested.

using Pkg, Tar, SHA, TOML, Downloads

if length(ARGS) < 2
    println(stderr, "usage: rebuild-julia-manifest.jl <version> <dir-of-tarballs> [out]")
    exit(2)
end
version, dir = ARGS[1], ARGS[2]
out = length(ARGS) >= 3 ? ARGS[3] : joinpath(dir, "Artifacts.toml")
baseurl = "https://github.com/dcharlot-physicalai-bmi/ferrotherm/releases/download/v$version"

# The platform set the release is expected to carry. Listed rather than globbed, so a missing
# tarball is an error instead of a manifest that quietly covers two platforms out of three.
const PLATFORMS = ["aarch64-apple-darwin", "x86_64-linux-gnu", "x86_64-w64-mingw32"]

entries = Dict{String, Any}[]
for platform in PLATFORMS
    tarball = joinpath(dir, "ferrotherm.v$version.$platform.tar.gz")
    isfile(tarball) || error("no tarball for $platform at $tarball -- download it from the release first")

    tree = mktempdir()
    open(`gzip -d -c $tarball`) do io
        Tar.extract(io, tree)
    end
    tree_hash = bytes2hex(Pkg.GitTools.tree_hash(tree))
    sha = bytes2hex(open(sha256, tarball))

    os, arch = let p = split(platform, '-')
        a = p[1]
        o = occursin("darwin", platform) ? "macos" :
            occursin("mingw", platform) || occursin("windows", platform) ? "windows" : "linux"
        (o, a)
    end

    println("  $platform")
    println("    tarball     $(basename(tarball))  $(filesize(tarball)) bytes")
    println("    git-tree    $tree_hash")
    println("    sha256      $sha")

    entry = Dict{String, Any}(
        "git-tree-sha1" => tree_hash,
        "arch"          => arch,
        "os"            => os,
        "lazy"          => true,
        "download"      => [Dict("url" => "$baseurl/$(basename(tarball))", "sha256" => sha)],
    )
    os == "linux" && (entry["libc"] = "glibc")
    push!(entries, entry)
end

open(out, "w") do io
    TOML.print(io, Dict("ferrotherm" => entries))
end
println("\nwrote $out with $(length(entries)) platform(s) for v$version")
