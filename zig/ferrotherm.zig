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
    /// Fewer than 16 draws; certifying that many says nothing.
    TooFewDraws,
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

// ---- instances with a known optimum -------------------------------------------------------------

/// An instance whose optimum was chosen before the couplings were built.
///
/// A result reported without one is a number nobody can judge. With one, the same run reports its
/// distance from the truth.
pub const Planted = struct {
    sim: Sim,
    optimum: f64,

    /// Frustrated plaquettes on an `l` by `l` periodic lattice.
    ///
    /// Difficulty is not monotonic in `loops`: it peaks near four per edge and falls away at both
    /// ends, so a very sparse or a saturated instance is easy.
    pub fn frustrated(l: u32, loops: u32, seed: u64, beta: f64) Error!Planted {
        const s = c.ft_planted_frustrated(l, loops, seed, beta) orelse return Error.OutOfMemory;
        return .{ .sim = .{ .h = s }, .optimum = c.ft_ground_energy(s) };
    }

    /// The Wishart ensemble: dense, and hard below alpha of 1.
    pub fn wishart(n: u32, alpha: f64, seed: u64, beta: f64) Error!Planted {
        const s = c.ft_planted_wishart(n, alpha, seed, beta) orelse return Error.OutOfMemory;
        return .{ .sim = .{ .h = s }, .optimum = c.ft_ground_energy(s) };
    }

    /// How far the current state sits above the optimum, as a fraction of it. Zero means solved.
    pub fn excess(self: Planted) f64 {
        const e = self.sim.energy();
        return if (@abs(self.optimum) > 1e-12)
            (e - self.optimum) / @abs(self.optimum)
        else
            e - self.optimum;
    }

    pub fn solved(self: Planted) bool {
        return self.sim.energy() <= self.optimum + 1e-9;
    }

    pub fn deinit(self: Planted) void {
        self.sim.deinit();
    }
};

// ---- certificate ---------------------------------------------------------------------------------

/// What a run actually did, computed from its samples rather than from its own account of itself.
pub const Certificate = struct {
    beta_eff: f64,
    beta_lo: f64,
    beta_hi: f64,
    tau_int: f64,
    ess: f64,
    tv: f64,
    noise_floor: f64,
    findings: usize,

    /// Zero findings is the only thing that means the run is sound.
    pub fn passed(self: Certificate) bool {
        return self.findings == 0;
    }
};

/// Sample and certify. `draws` must be at least 16; certifying fewer says nothing.
pub fn certify(sim: Sim, draws: u32, thin: u32) Error!Certificate {
    if (c.ft_certify(sim.h, draws, thin) == 0) return Error.TooFewDraws;
    return .{
        .beta_eff = c.ft_cert_beta_eff(sim.h),
        .beta_lo = c.ft_cert_beta_lo(sim.h),
        .beta_hi = c.ft_cert_beta_hi(sim.h),
        .tau_int = c.ft_cert_tau(sim.h),
        .ess = c.ft_cert_ess(sim.h),
        .tv = c.ft_cert_tv(sim.h),
        .noise_floor = c.ft_cert_floor(sim.h),
        .findings = c.ft_cert_findings(sim.h),
    };
}

/// Copy finding `i` into `buf`, returning the slice actually written.
pub fn finding(sim: Sim, i: u32, buf: []u8) []const u8 {
    const n = c.ft_cert_finding(sim.h, i, buf.ptr, @intCast(buf.len));
    return buf[0..n];
}

// ---- exact inference ------------------------------------------------------------------------------

/// Exact ground energy by variable elimination, or null if the graph is too dense.
///
/// Cost is `2^width` in the graph's shape rather than `2^n` in its size, so a long chain is instant
/// where a dense graph of the same node count is impossible. Check `exactWidth` first.
pub fn exactGround(sim: Sim, max_width: u32) ?f64 {
    const v = c.ft_exact_ground(sim.h, max_width);
    return if (std.math.isNan(v)) null else v;
}

/// Exact log partition function at `beta`, or null if too dense.
pub fn exactLogZ(sim: Sim, beta: f64, max_width: u32) ?f64 {
    const v = c.ft_exact_log_z(sim.h, beta, max_width);
    return if (std.math.isNan(v)) null else v;
}

/// Induced width of the elimination order. Exact inference costs `2^width`.
pub fn exactWidth(sim: Sim) u32 {
    return c.ft_exact_width(sim.h);
}

test "a planted instance knows its optimum" {
    var p = try Planted.frustrated(8, 96, 3, 1.0);
    defer p.deinit();
    try std.testing.expectEqual(@as(f64, -192.0), p.optimum);
    _ = try p.sim.anneal(0.05, 6.0, 80, 40);
    try std.testing.expect(p.sim.energy() >= p.optimum - 1e-9); // nothing beats the plant
    try std.testing.expect(p.excess() < 0.10);
}

test "a certificate can fail" {
    // A cold lattice with no burn-in and no thinning must not certify clean, or the type is
    // decoration.
    var sim = try Sim.lattice2d(24, 1.0, 0.7, 4);
    defer sim.deinit();
    const cert = try certify(sim, 400, 1);
    try std.testing.expect(!cert.passed());
    try std.testing.expect(cert.ess < 400);

    var buf: [512]u8 = undefined;
    const msg = finding(sim, 0, &buf);
    try std.testing.expect(msg.len > 10);
}

test "a well run chain certifies clean" {
    var sim = try Sim.lattice2d(12, 1.0, 0.2, 1);
    defer sim.deinit();
    _ = sim.sweep(500);
    const cert = try certify(sim, 800, 4);
    try std.testing.expect(cert.passed());
    try std.testing.expect(@abs(cert.beta_eff - 0.2) < 0.05);
}

test "exact inference matches the closed form on a chain" {
    // Z = 2 (2 cosh beta)^(n-1) for an open 1D chain. Checking against theory reaches sizes
    // enumeration cannot.
    const n: u32 = 300;
    var m = try Model.init(n);
    defer m.deinit();
    var i: u32 = 0;
    while (i + 1 < n) : (i += 1) try m.couple(i, i + 1, 1.0);
    var sim = try m.build(1.0, 0);
    defer sim.deinit();

    try std.testing.expectEqual(@as(u32, 1), exactWidth(sim));
    try std.testing.expectEqual(@as(f64, -@as(f64, n - 1)), exactGround(sim, 20).?);

    const beta: f64 = 0.5;
    const want = @log(2.0) + @as(f64, n - 1) * @log(2.0 * std.math.cosh(beta));
    const got = exactLogZ(sim, beta, 20).?;
    try std.testing.expect(@abs(got - want) < 1e-6 * @abs(want));
}
