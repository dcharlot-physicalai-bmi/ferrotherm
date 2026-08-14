#!/usr/bin/env julia
#
# Build the Julia artifacts, and the Artifacts.toml that points at them.
#
# Julia has no wheels. A package needing a compiled library depends on a JLL: a package that carries
# prebuilt binaries per platform and hands you the path. The usual route is a recipe submitted to
# Yggdrasil, whose CI builds it and whose maintainers review it.
#
# We build our own instead, out of the same CI that builds the Python wheels, because the whole
# point of this stack is that we control every layer of it. Nothing here needs BinaryBuilder: a
# git-tree-sha1 and a sha256 are computable with stock Julia, and Rust already cross-compiles.
#
#   julia scripts/build-julia-artifacts.jl <out-dir> <base-url> <platform>=<library> ...
#
# e.g.
#   julia scripts/build-julia-artifacts.jl dist \
#       https://github.com/.../releases/download/v0.7.0 \
#       x86_64-linux-gnu=libferrotherm.so \
#       aarch64-apple-darwin=libferrotherm.dylib
#
# Writes one tarball per platform into <out-dir>, and an Artifacts.toml naming them by URL and hash.
# Run it once per release with every platform's library present.

using Pkg, Tar, SHA, TOML

if length(ARGS) < 3
    println(stderr, "usage: build-julia-artifacts.jl <out-dir> <base-url> <platform>=<lib> ...")
    exit(2)
end

outdir, baseurl = ARGS[1], rstrip(ARGS[2], '/')
mkpath(outdir)

"""What Julia calls each platform, and what the library is called there."""
function parse_pair(s)
    parts = split(s, '='; limit = 2)
    length(parts) == 2 || error("expected <platform>=<library>, got $s")
    String(parts[1]), String(parts[2])
end

entries = Dict{String, Any}[]

for arg in ARGS[3:end]
    platform, libpath = parse_pair(arg)
    isfile(libpath) || error("no library at $libpath for $platform")

    # A JLL artifact is a directory tree, not a bare file: `bin` on Windows because a DLL is loaded
    # from the binary path, `lib` everywhere else.
    subdir = occursin("mingw", platform) || occursin("windows", platform) ? "bin" : "lib"
    tree = mktempdir()
    mkpath(joinpath(tree, subdir))
    cp(libpath, joinpath(tree, subdir, basename(libpath)))

    tree_hash = bytes2hex(Pkg.GitTools.tree_hash(tree))
    tarball = joinpath(outdir, "ferrotherm.v$(ENV["FERROTHERM_VERSION"]).$platform.tar.gz")
    Tar.create(tree, pipeline(`gzip -9`, tarball))
    sha = bytes2hex(open(sha256, tarball))

    println("  $platform")
    println("    library     $(basename(libpath)) -> $subdir/")
    println("    tarball     $(basename(tarball))  $(filesize(tarball)) bytes")
    println("    git-tree    $tree_hash")
    println("    sha256      $sha")

    os, arch = let p = split(platform, '-')
        # x86_64-linux-gnu, aarch64-apple-darwin, x86_64-w64-mingw32
        a = p[1]
        o = occursin("darwin", platform) ? "macos" :
            occursin("mingw", platform) || occursin("windows", platform) ? "windows" : "linux"
        (o, a)
    end

    entry = Dict{String, Any}(
        "git-tree-sha1" => tree_hash,
        "arch"          => arch,
        "os"            => os,
        # Lazy on purpose. Without it `Pkg.instantiate` downloads every artifact eagerly and FAILS
        # when one cannot be fetched -- a platform with no build, or a release not yet published --
        # before the package's own __init__ gets a chance to fall back. Lazy defers the download to
        # first use, which is inside the try that handles exactly that.
        "lazy"          => true,
        "download"      => [Dict("url" => "$baseurl/$(basename(tarball))", "sha256" => sha)],
    )
    # Linux platforms carry a libc, and Julia's own host platform always names one. A JLL entry
    # that omits it is relying on missing-means-wildcard, which is not something to rely on when
    # the failure mode is a silent "no library on this platform".
    if os == "linux"
        entry["libc"] = "glibc"
    end
    push!(entries, entry)
end

# Artifacts.toml: one [[ferrotherm]] block per platform. Julia picks the one matching the host and
# refuses a hash mismatch, which is what makes a self-hosted artifact as trustworthy as a
# registry-hosted one -- the hash is in the package, and the package is in the registry.
open(joinpath(outdir, "Artifacts.toml"), "w") do io
    TOML.print(io, Dict("ferrotherm" => entries))
end

println("\nwrote $(joinpath(outdir, "Artifacts.toml")) with $(length(entries)) platform(s)")
