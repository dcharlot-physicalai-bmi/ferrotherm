//! Thermodynamic sampling from Zig.
//!
//! A thin wrapper over the C ABI in `include/ferrotherm.h`. Zig reads that header directly, so this
//! file adds only what a C header cannot express: ownership through `defer`, errors instead of
//! sentinel returns, and slices instead of pointer-plus-length.
//!
//! Conventions match the rest of the ecosystem: states are -1/+1, energy is
//! -sum_ij J_ij s_i s_j - sum_i h_i s_i, and beta is inverse temperature.

const std = @import("std");
const c = @cImport({
    @cInclude("ferrotherm.h");
});

pub const Error = error{
    /// The library refused to allocate the handle.
    OutOfMemory,
    /// An index was out of range, i equalled j, or a weight was not finite.
    RejectedEntry,
    /// A ladder that is not 0 < beta_min < beta_max, or a degenerate stage count.
    BadSchedule,
    /// The builder was already consumed by build().
    BuilderSpent,
};

/// A graph under construction. Add couplings and biases, then `build`.
pub const Model = struct {
    h: ?*c.ft_builder,
    n: u32,

    pub fn init(n: u32) Error!Model {
        const h = c.ft_builder_new(n) orelse return Error.OutOfMemory;
        return .{ .h = h, .n = n };
    }

    /// Add coupling J_ij. A rejected entry is an error, never a silent no-op: a coupling that
    /// vanishes without complaint is a model that is quietly wrong.
    pub fn couple(self: *Model, i: u32, j: u32, w: f64) Error!void {
        const h = self.h orelse return Error.BuilderSpent;
        if (c.ft_builder_couple(h, i, j, w) == 0) return Error.RejectedEntry;
    }

    pub fn bias(self: *Model, i: u32, h_i: f64) Error!void {
        const h = self.h orelse return Error.BuilderSpent;
        if (c.ft_builder_bias(h, i, h_i) == 0) return Error.RejectedEntry;
    }

    /// Consume this model into a simulation. The model is spent afterwards.
    pub fn build(self: *Model, beta: f64, seed: u64) Error!Sim {
        const h = self.h orelse return Error.BuilderSpent;
        self.h = null;
        const s = c.ft_builder_build(h, beta, seed) orelse return Error.OutOfMemory;
        return .{ .h = s };
    }

    /// Safe to call whether or not `build` ran; pair it with `init` using `defer`.
    pub fn deinit(self: *Model) void {
        if (self.h) |h| c.ft_builder_free(h);
        self.h = null;
    }
};

/// A running simulation.
pub const Sim = struct {
    h: *c.ft_sim,

    /// 2D nearest-neighbour Ising lattice, periodic, side `l`.
    pub fn lattice2d(l: u32, j: f64, beta: f64, seed: u64) Error!Sim {
        return .{ .h = c.ft_ising2d_new(l, j, beta, seed) orelse return Error.OutOfMemory };
    }

    /// Z1-topology grid, degree 16, open boundaries.
    pub fn z1Grid(w: u32, ht: u32, j: f64, hb: f64, beta: f64, seed: u64) Error!Sim {
        return .{ .h = c.ft_z1_new(w, ht, j, hb, beta, seed) orelse return Error.OutOfMemory };
    }

    /// Run `n` chromatic block-Gibbs sweeps. Returns total sweeps so far.
    pub fn sweep(self: Sim, n: u32) u64 {
        return c.ft_sweep(self.h, n);
    }

    /// Anneal down a geometric ladder, keeping the best state found. Returns its energy.
    pub fn anneal(self: Sim, beta_min: f64, beta_max: f64, stages: u32, per: u32) Error!f64 {
        const e = c.ft_anneal(self.h, beta_min, beta_max, stages, per);
        if (std.math.isNan(e)) return Error.BadSchedule;
        return e;
    }

    pub fn setBeta(self: Sim, beta: f64) void {
        c.ft_set_beta(self.h, beta);
    }

    /// The state. Borrowed from the library and valid only until the next sweep; copy to keep it.
    pub fn spins(self: Sim) []const i8 {
        const n = c.ft_len(self.h);
        const p = c.ft_spins(self.h);
        return p[0..n];
    }

    pub fn len(self: Sim) u32 {
        return c.ft_len(self.h);
    }
    pub fn energy(self: Sim) f64 {
        return c.ft_energy(self.h);
    }
    pub fn magnetization(self: Sim) f64 {
        return c.ft_magnetization(self.h);
    }
    pub fn nodeUpdates(self: Sim) u64 {
        return c.ft_ledger_updates(self.h);
    }
    /// Ledger priced at Z1-class device figures. Prices the modelled device, not this CPU.
    pub fn joules(self: Sim) f64 {
        return c.ft_ledger_joules_z1(self.h);
    }

    pub fn deinit(self: Sim) void {
        c.ft_free(self.h);
    }
};

/// Onsager's exact spontaneous magnetisation for the 2D Ising model. Ground truth.
pub fn onsager(beta: f64) f64 {
    return c.ft_onsager(beta);
}

// ---- tests ---------------------------------------------------------------------------------

test "lattice agrees with Onsager" {
    var sim = try Sim.lattice2d(64, 1.0, 0.5, 1);
    defer sim.deinit();
    _ = sim.sweep(3000);
    const measured = @abs(sim.magnetization());
    const exact = onsager(0.5);
    try std.testing.expect(@abs(measured - exact) < 0.03);
}

test "frustrated ring reaches its optimum" {
    // An odd antiferromagnetic ring cannot be two-coloured, so exactly one bond stays unsatisfied.
    var m = try Model.init(5);
    defer m.deinit();
    for (0..5) |i| {
        try m.couple(@intCast(i), @intCast((i + 1) % 5), -1.0);
    }
    var sim = try m.build(0.1, 1);
    defer sim.deinit();
    const e = try sim.anneal(0.05, 6.0, 60, 40);
    try std.testing.expectEqual(@as(f64, -3.0), e);
}

test "bad entries are errors, not silent drops" {
    var m = try Model.init(3);
    defer m.deinit();
    try std.testing.expectError(Error.RejectedEntry, m.couple(0, 9, 1.0));
    try std.testing.expectError(Error.RejectedEntry, m.couple(1, 1, 1.0));
    try std.testing.expectError(Error.RejectedEntry, m.bias(7, 1.0));

    var spent = try Model.init(3);
    var sim = try spent.build(1.0, 0);
    defer sim.deinit();
    try std.testing.expectError(Error.BuilderSpent, spent.couple(0, 1, 1.0));
}

test "ledger counts what it ran" {
    var sim = try Sim.lattice2d(8, 1.0, 0.44, 7);
    defer sim.deinit();
    _ = sim.sweep(50);
    try std.testing.expectEqual(@as(u64, 64 * 50), sim.nodeUpdates());
    try std.testing.expect(sim.joules() > 0);
    try std.testing.expectEqual(@as(usize, 64), sim.spins().len);
}
