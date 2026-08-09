//! Build the Zig binding, and the Rust library it wraps.
//!
//! A Zig consumer already has a build step, so the cleanest answer to "where does the native
//! library come from" is: this builds it. `cargo build --release` runs as a build step, and the
//! module links against what it produces. No prebuilt artefact to go stale, no environment
//! variable to set, and the library is always the one that matches this source.
//!
//!     zig build test          # run the binding's tests
//!     zig build               # just build the library and the module
//!
//! Pass `-Dcargo=false` if you are building the Rust side yourself and only want the linking.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const run_cargo = b.option(bool, "cargo", "Build the Rust library first (default true)") orelse true;

    // The repository root, one level up from this file.
    const root = b.pathFromRoot("..");
    const lib_dir = b.pathJoin(&.{ root, "target", "release" });

    const cargo = b.addSystemCommand(&.{ "cargo", "build", "--release" });
    cargo.setCwd(.{ .cwd_relative = root });

    const mod = b.addModule("ferrotherm", .{
        .root_source_file = b.path("ferrotherm.zig"),
        .target = target,
        .optimize = optimize,
    });
    mod.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ root, "include" }) });
    mod.addLibraryPath(.{ .cwd_relative = lib_dir });
    mod.linkSystemLibrary("ferrotherm", .{});
    mod.link_libc = true;
    mod.addRPath(.{ .cwd_relative = lib_dir });

    // So the test binary finds the shared library at run time without LD_LIBRARY_PATH. A test that
    // cannot start is indistinguishable from a test that fails, and the message is worse.
    const tests = b.addTest(.{ .root_module = mod });
    if (run_cargo) tests.step.dependOn(&cargo.step);

    const run_tests = b.addRunArtifact(tests);
    b.step("test", "Run the binding's tests against the Rust library").dependOn(&run_tests.step);

    const install = b.addInstallArtifact(tests, .{});
    b.getInstallStep().dependOn(&install.step);
}
