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
    /// A domain with nothing in it: a categorical under two values, or an integer with hi <= lo.
    BadDomain,
    /// A value the variable cannot take. Call `lastError` for the range it can.
    BadValue,
    /// A constraint the library would not accept. `lastError` says why.
    RejectedConstraint,
    /// A model that does not compile. `lastError` says why.
    WillNotCompile,
    /// Reading an answer before solving one.
    NotSolved,
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

// ---- modelling ---------------------------------------------------------------------------------
//
// Everything above works in spins: couplings, biases, energies. That is the machine's language, not
// the problem's. This is the problem's — variables that hold values, constraints that say what must
// be true, an objective that says what is better — and it compiles down to the layer above.
//
// A value is the value you wrote. An integer over 10..=20 holds thirteen in its fourth slot, and
// every call here takes thirteen; passing three is `BadValue` and `lastError` names the range.

/// A declared variable. Ask it for a literal with `is`.
pub const Var = struct {
    idx: u32,

    /// The claim "this variable takes `value`".
    pub fn is(self: Var, value: i64) Lit {
        return .{ .v = self, .value = value };
    }
};

/// One variable taking one value, for counting constraints and objective terms.
pub const Lit = struct {
    v: Var,
    value: i64,
};

/// What to count in a `count` constraint.
pub const Counting = enum(u32) {
    /// Exactly `k` hold.
    exactly = 0,
    /// At most `k` hold. Costs a slack variable.
    at_most = 1,
    /// At least `k` hold. Costs a slack variable.
    at_least = 2,
    /// Exactly one holds. Lowers pairwise with no slack, so cheaper than `exactly` with k = 1.
    exactly_one = 3,
    /// At most one holds.
    at_most_one = 4,
};

/// How a variable is stored.
///
/// The trade is the difference between a model that fits a machine and one that does not.
///
/// | encoding | spins for `k` values | usable in a constraint or objective |
/// |---|---|---|
/// | `one_hot` | `k` | yes |
/// | `domain_wall` | `k - 1` | yes |
/// | `binary` | `ceil(log2 k)` | **no** |
///
/// A binary code's indicator is a product of every bit, so its degree grows with the domain. It is
/// the cheapest to store and is refused by name if it appears in a literal, rather than expanded
/// into something nobody wants to read.
pub const Encoding = enum(u32) {
    one_hot = 0,
    binary = 1,
    /// Often the better choice for an INTEGER, which is an ordered domain: neighbouring values sit
    /// one spin flip apart where one-hot puts them two apart, for one spin fewer.
    domain_wall = 2,
};

/// Which direction an objective term prefers.
pub const Sense = enum { maximize, minimize };

/// A problem stated in its own terms.
///
///     var p = try Problem.init();
///     defer p.deinit();
///     const west = try p.categorical("west", 3);
///     const east = try p.categorical("east", 3);
///     try p.notEqual(west, east);
///     _ = try p.compile();
///     try p.solve(12);
///     const c = try p.value(west);   // 0, 1 or 2 — and not whatever east got
pub const Problem = struct {
    h: *c.ft_model,

    pub fn init() Error!Problem {
        const h = c.ft_model_new() orelse return Error.OutOfMemory;
        return .{ .h = h };
    }

    pub fn deinit(self: *Problem) void {
        c.ft_model_free(self.h);
    }

    // -- variables -------------------------------------------------------------------------------

    /// One of `values` unordered values, encoded one-hot. Needs at least two.
    pub fn categorical(self: *Problem, name: []const u8, values: u32) Error!Var {
        return self.categoricalAs(name, values, .one_hot);
    }

    /// One of `values` values, stored the way you ask. See [`Encoding`] for what it costs.
    pub fn categoricalAs(
        self: *Problem,
        name: []const u8,
        values: u32,
        encoding: Encoding,
    ) Error!Var {
        return self.declare(
            name,
            c.ft_model_categorical_as(self.h, values, @intFromEnum(encoding)),
        );
    }

    /// An integer over the inclusive range `lo..=hi`.
    ///
    /// There is no machine integer here: this is a categorical over the range, and the name is for
    /// the modeller rather than the fabric. Values are the range's own, so `10..=20` takes 13.
    pub fn integer(self: *Problem, name: []const u8, lo: i64, hi: i64) Error!Var {
        return self.integerAs(name, lo, hi, .one_hot);
    }

    /// An integer over `lo..=hi`, stored the way you ask.
    ///
    /// `.domain_wall` is often the better choice here: an integer is an ORDERED domain, so
    /// neighbouring values sit one spin flip apart instead of two, and it costs one spin fewer.
    pub fn integerAs(
        self: *Problem,
        name: []const u8,
        lo: i64,
        hi: i64,
        encoding: Encoding,
    ) Error!Var {
        return self.declare(name, c.ft_model_integer_as(self.h, lo, hi, @intFromEnum(encoding)));
    }

    /// 0 or 1.
    pub fn binary(self: *Problem, name: []const u8) Error!Var {
        return self.declare(name, c.ft_model_binary(self.h));
    }

    fn declare(self: *Problem, name: []const u8, idx: u32) Error!Var {
        if (idx == std.math.maxInt(u32)) return Error.BadDomain;
        // Push the name down, so a refusal names the variable the caller declared rather than the
        // handle they were given back.
        _ = c.ft_model_name(self.h, idx, name.ptr, @intCast(name.len));
        return .{ .idx = idx };
    }

    // -- constraints -----------------------------------------------------------------------------

    pub fn notEqual(self: *Problem, a: Var, b: Var) Error!void {
        if (c.ft_model_not_equal(self.h, a.idx, b.idx) == 0) return Error.RejectedConstraint;
    }

    pub fn equal(self: *Problem, a: Var, b: Var) Error!void {
        if (c.ft_model_equal(self.h, a.idx, b.idx) == 0) return Error.RejectedConstraint;
    }

    pub fn fix(self: *Problem, v: Var, val: i64) Error!void {
        if (c.ft_model_fix(self.h, v.idx, val) == 0) return Error.BadValue;
    }

    /// Count how many of `lits` hold, and constrain that count.
    ///
    /// Any number of literals, each naming its own variable and its own value: "at most two of
    /// these nine shifts" and "at most one of a = 3, b = 17" are both sayable.
    pub fn count(self: *Problem, kind: Counting, k: u32, lits: []const Lit) Error!void {
        _ = c.ft_model_lits_clear(self.h);
        for (lits) |l| {
            if (c.ft_model_lit(self.h, l.v.idx, l.value) == 0) return Error.BadValue;
        }
        if (c.ft_model_close(self.h, @intFromEnum(kind), k) == 0) return Error.RejectedConstraint;
    }

    /// `count` over whole variables, each taking the same value. The common case.
    pub fn countVars(self: *Problem, kind: Counting, k: u32, vars: []const Var, val: i64) Error!void {
        _ = c.ft_model_lits_clear(self.h);
        for (vars) |v| {
            if (c.ft_model_lit(self.h, v.idx, val) == 0) return Error.BadValue;
        }
        if (c.ft_model_close(self.h, @intFromEnum(kind), k) == 0) return Error.RejectedConstraint;
    }

    // -- objective -------------------------------------------------------------------------------

    /// Prefer states where `lit` holds, by `weight`.
    ///
    /// Terms ACCUMULATE, and each carries its own sense: a minimising term added after maximising
    /// ones changes only itself.
    pub fn prefer(self: *Problem, sense: Sense, weight: f64, lit: Lit) Error!void {
        const max: u32 = if (sense == .maximize) 1 else 0;
        if (c.ft_model_objective_term(self.h, max, weight, lit.v.idx, lit.value) == 0) {
            return Error.BadValue;
        }
    }

    /// Prefer states where every literal in `lits` holds together.
    ///
    /// Three or more is a higher-order term: `compile` lowers it with an ancilla spin per
    /// substituted pair, so it costs spins the model did not declare. One or two literals are the
    /// ordinary linear and quadratic cases and go through the same call, so a caller building
    /// terms in a loop does not need three branches.
    pub fn preferAll(self: *Problem, sense: Sense, weight: f64, lits: []const Lit) Error!void {
        if (lits.len == 0) return Error.RejectedConstraint;
        _ = c.ft_model_lits_clear(self.h);
        for (lits) |l| {
            if (c.ft_model_lit(self.h, l.v.idx, l.value) == 0) return Error.BadValue;
        }
        const max: u32 = if (sense == .maximize) 1 else 0;
        if (c.ft_model_objective_product(self.h, max, weight) == 0) {
            return Error.RejectedConstraint;
        }
    }

    /// Prefer states where two literals hold together. Quadratic in the spins.
    pub fn preferPair(self: *Problem, sense: Sense, weight: f64, a: Lit, b: Lit) Error!void {
        const max: u32 = if (sense == .maximize) 1 else 0;
        if (c.ft_model_objective_pair(self.h, max, weight, a.v.idx, a.value, b.v.idx, b.value) == 0) {
            return Error.BadValue;
        }
    }

    /// Use exactly this penalty, disabling the automatic scaling.
    ///
    /// The remedy when `feasible` comes back false: a constraint lost to the objective and has to
    /// outrank it. By default the penalty is twice the largest objective weight.
    pub fn penalty(self: *Problem, p: f64) Error!void {
        if (c.ft_model_fixed_penalty(self.h, p) == 0) return Error.RejectedConstraint;
    }

    // -- solving ---------------------------------------------------------------------------------

    /// Compile to spins, returning how many were needed — including any slack an inequality wanted.
    pub fn compile(self: *Problem) Error!u32 {
        const n = c.ft_model_compile(self.h);
        if (n == 0) return Error.WillNotCompile;
        return n;
    }

    /// Anneal `tries` times and keep the best answer.
    pub fn solve(self: *Problem, tries: u32) Error!void {
        if (c.ft_model_solve(self.h, tries) == 0) return Error.NotSolved;
    }

    /// Anneal on your own ladder: `beta_hot` to `beta_cold` over `stages` of `sweeps` each.
    ///
    /// Zero for any of the four means the library's default, so you can override only what you
    /// measured. A ladder that runs backwards is refused rather than quietly replaced.
    pub fn solveWith(
        self: *Problem,
        tries: u32,
        beta_hot: f64,
        beta_cold: f64,
        stages: u32,
        sweeps: u32,
    ) Error!void {
        if (c.ft_model_solve_with(self.h, tries, beta_hot, beta_cold, stages, sweeps) == 0) {
            return Error.BadSchedule;
        }
    }

    /// The answer for one variable, in its own units.
    pub fn value(self: *Problem, v: Var) Error!i64 {
        const got = c.ft_model_value(self.h, v.idx);
        if (got == std.math.minInt(i64)) return Error.NotSolved;
        return got;
    }

    /// True when every variable decoded AND every constraint holds.
    ///
    /// Both halves matter and they fail differently. A penalty makes a constraint expensive, not
    /// impossible, so a sampler whose objective outbids it returns an answer that reads perfectly
    /// and breaks the request. `violations` says which.
    pub fn feasible(self: *Problem) bool {
        return c.ft_model_feasible(self.h) == 1;
    }

    pub fn energy(self: *Problem) f64 {
        return c.ft_model_energy(self.h);
    }

    /// The penalty actually used, after any automatic scaling.
    pub fn effectivePenalty(self: *Problem) f64 {
        return c.ft_model_penalty(self.h);
    }

    /// How many constraints the answer breaks. Zero when it keeps everything it was asked to.
    pub fn violations(self: *Problem) u32 {
        return c.ft_model_violations(self.h);
    }

    /// Violation `i`, described in your own names, written into `buf`.
    pub fn violation(self: *Problem, i: u32, buf: []u8) []const u8 {
        const need = c.ft_model_violation(self.h, i, null, 0);
        const n = @min(need, @as(u32, @intCast(buf.len)));
        const got = c.ft_model_violation(self.h, i, buf.ptr, n);
        return buf[0..got];
    }

    /// Why the last call was refused. Empty when nothing was.
    pub fn lastError(self: *Problem, buf: []u8) []const u8 {
        const need = c.ft_model_error(self.h, null, 0);
        const n = @min(need, @as(u32, @intCast(buf.len)));
        const got = c.ft_model_error(self.h, buf.ptr, n);
        return buf[0..got];
    }

    /// The compiled program in `.ftp` form, which runs unchanged on any backend.
    pub fn ftp(self: *Problem, buf: []u8) []const u8 {
        const need = c.ft_model_ftp(self.h, null, 0);
        const n = @min(need, @as(u32, @intCast(buf.len)));
        const got = c.ft_model_ftp(self.h, buf.ptr, n);
        return buf[0..got];
    }
};

test "two variables cannot share a name" {
    var p = try Problem.init();
    defer p.deinit();
    _ = try p.binary("shift");
    // the C ABI refuses the second name, so declare() reports the domain as unusable
    const second = c.ft_model_binary(p.h);
    try std.testing.expectEqual(@as(u32, 0), c.ft_model_name(p.h, second, "shift", 5));
    var buf: [256]u8 = undefined;
    try std.testing.expect(std.mem.indexOf(u8, p.lastError(&buf), "already") != null);
}

test "a triangle needs three colours" {
    var p = try Problem.init();
    defer p.deinit();
    const west = try p.categorical("west", 3);
    const middle = try p.categorical("middle", 3);
    const east = try p.categorical("east", 3);
    try p.notEqual(west, middle);
    try p.notEqual(middle, east);
    try p.notEqual(west, east);

    _ = try p.compile();
    try p.solve(12);
    try std.testing.expect(p.feasible());

    const a = try p.value(west);
    const b = try p.value(middle);
    const d = try p.value(east);
    try std.testing.expect(a != b and b != d and a != d);
}

test "an integer is written in its own values, not in slots" {
    var p = try Problem.init();
    defer p.deinit();
    const t = try p.integer("temperature", 10, 20);
    try p.prefer(.maximize, 5.0, t.is(13));
    _ = try p.compile();
    try p.solve(8);
    try std.testing.expectEqual(@as(i64, 13), try p.value(t));

    // and a slot index where a value belongs is refused, naming the range
    var q = try Problem.init();
    defer q.deinit();
    const u = try q.integer("temperature", 10, 20);
    try std.testing.expectError(Error.BadValue, q.fix(u, 3));
    var buf: [256]u8 = undefined;
    const e = q.lastError(&buf);
    try std.testing.expect(std.mem.indexOf(u8, e, "temperature") != null);
    try std.testing.expect(std.mem.indexOf(u8, e, "10..=20") != null);
}

test "at most two of nine, which the positional form cannot say" {
    var p = try Problem.init();
    defer p.deinit();
    var shifts: [9]Var = undefined;
    var names: [9][3]u8 = undefined;
    for (&shifts, 0..) |*s, i| {
        names[i] = .{ 's', @intCast('0' + i), 0 };
        s.* = try p.binary(names[i][0..2]);
        try p.prefer(.maximize, @as(f64, @floatFromInt(9 - i)), s.is(1));
    }
    try p.countVars(.at_most, 2, &shifts, 1);

    _ = try p.compile();
    try p.solve(24);
    try std.testing.expect(p.feasible());

    var on: u32 = 0;
    for (shifts) |s| {
        if (try p.value(s) == 1) on += 1;
    }
    try std.testing.expectEqual(@as(u32, 2), on);
}

test "literals in one constraint may name different values" {
    var p = try Problem.init();
    defer p.deinit();
    const a = try p.categorical("a", 4);
    const b = try p.integer("b", 10, 20);
    try p.count(.at_most, 1, &.{ a.is(3), b.is(17) });
    try p.prefer(.maximize, 5.0, a.is(3));
    try p.prefer(.maximize, 4.0, b.is(17));

    _ = try p.compile();
    try p.solve(16);
    try std.testing.expect(p.feasible());
    try std.testing.expectEqual(@as(i64, 3), try p.value(a));
    try std.testing.expect(try p.value(b) != 17);
}

test "feasible means the constraints hold, and says which one did not" {
    // A penalty makes a constraint expensive, not impossible. Pinned below the objective, the
    // sampler pays it: every variable decodes and the constraint is broken.
    var p = try Problem.init();
    defer p.deinit();
    const a = try p.categorical("a", 3);
    const b = try p.categorical("b", 3);
    try p.notEqual(a, b);
    try p.penalty(1.0);
    try p.prefer(.maximize, 40.0, a.is(1));
    try p.prefer(.maximize, 40.0, b.is(1));

    _ = try p.compile();
    try p.solve(16);
    try std.testing.expectEqual(try p.value(a), try p.value(b));
    try std.testing.expect(!p.feasible());
    try std.testing.expectEqual(@as(u32, 1), p.violations());

    var buf: [256]u8 = undefined;
    const v = p.violation(0, &buf);
    try std.testing.expect(std.mem.indexOf(u8, v, "must differ") != null);

    // raised, the same model is feasible
    var q = try Problem.init();
    defer q.deinit();
    const x = try q.categorical("a", 3);
    const y = try q.categorical("b", 3);
    try q.notEqual(x, y);
    try q.penalty(200.0);
    try q.prefer(.maximize, 40.0, x.is(1));
    try q.prefer(.maximize, 40.0, y.is(1));
    _ = try q.compile();
    try q.solve(16);
    try std.testing.expect(q.feasible());
    try std.testing.expect(try q.value(x) != try q.value(y));
}

test "objective terms accumulate, and a later sense does not rewrite earlier ones" {
    var p = try Problem.init();
    defer p.deinit();
    var v: [4]Var = undefined;
    var names: [4][2]u8 = undefined;
    for (&v, 0..) |*x, i| {
        // distinct names, because an answer is keyed by name and the library refuses a collision
        names[i] = .{ 'v', @intCast('0' + i) };
        x.* = try p.binary(&names[i]);
        try p.prefer(if (i < 3) .maximize else .minimize, 1.0, x.is(1));
    }
    _ = try p.compile();
    try p.solve(16);
    for (v, 0..) |x, i| {
        const want: i64 = if (i < 3) 1 else 0;
        try std.testing.expectEqual(want, try p.value(x));
    }
}

test "an encoding can be chosen and costs what it says" {
    const spins = struct {
        fn f(enc: Encoding) !u32 {
            var p = try Problem.init();
            defer p.deinit();
            _ = try p.categoricalAs("a", 6, enc);
            return p.compile();
        }
    }.f;
    try std.testing.expectEqual(@as(u32, 6), try spins(.one_hot));
    try std.testing.expectEqual(@as(u32, 5), try spins(.domain_wall));
    try std.testing.expectEqual(@as(u32, 3), try spins(.binary));

    // and a domain-wall variable really carries a constraint
    var p = try Problem.init();
    defer p.deinit();
    const a = try p.categoricalAs("a", 6, .domain_wall);
    try p.fix(a, 3);
    _ = try p.compile();
    try p.solve(16);
    try std.testing.expect(p.feasible());
    try std.testing.expectEqual(@as(i64, 3), try p.value(a));
}

test "an integer stored as a domain wall is one spin cheaper" {
    var p = try Problem.init();
    defer p.deinit();
    const t = try p.integerAs("t", 10, 20, .domain_wall);
    try p.fix(t, 17);
    try std.testing.expectEqual(@as(u32, 10), try p.compile()); // eleven values, ten spins
    try p.solve(16);
    try std.testing.expectEqual(@as(i64, 17), try p.value(t));
}

test "a higher-order objective term costs ancillas and finds the right answer" {
    // Three literals together, which the binding could not express at all: it offered one or two.
    var p = try Problem.init();
    defer p.deinit();
    const a = try p.categorical("a", 3);
    const b = try p.categorical("b", 3);
    const d = try p.categorical("d", 3);
    try p.preferAll(.maximize, 9.0, &.{ a.is(2), b.is(2), d.is(2) });

    const spins = try p.compile();
    try std.testing.expect(spins > 9); // three categoricals are 9; the ancilla makes it more
    try p.solve(24);
    try std.testing.expect(p.feasible());
    try std.testing.expectEqual(@as(i64, 2), try p.value(a));
    try std.testing.expectEqual(@as(i64, 2), try p.value(b));
    try std.testing.expectEqual(@as(i64, 2), try p.value(d));
}

test "an empty product is refused rather than silently ignored" {
    var p = try Problem.init();
    defer p.deinit();
    _ = try p.categorical("a", 3);
    try std.testing.expectError(Error.RejectedConstraint, p.preferAll(.maximize, 1.0, &.{}));
}

test "a caller's own ladder is used, and a backwards one refused" {
    var p = try Problem.init();
    defer p.deinit();
    const a = try p.categorical("a", 3);
    const b = try p.categorical("b", 3);
    try p.notEqual(a, b);
    _ = try p.compile();

    try p.solveWith(8, 0.05, 6.0, 60, 20);
    try std.testing.expect(p.feasible());

    // zeros mean the default, so only what was measured need be given
    try p.solveWith(8, 0.0, 0.0, 0, 0);
    try std.testing.expect(p.feasible());

    try std.testing.expectError(Error.BadSchedule, p.solveWith(8, 8.0, 0.05, 60, 20));
}

test "the compiled program exports as ftp" {
    var p = try Problem.init();
    defer p.deinit();
    const a = try p.categorical("a", 3);
    const b = try p.categorical("b", 3);
    try p.notEqual(a, b);
    _ = try p.compile();

    var buf: [4096]u8 = undefined;
    const text = p.ftp(&buf);
    try std.testing.expect(std.mem.startsWith(u8, text, "ftp 1"));
    try std.testing.expect(std.mem.indexOf(u8, text, "spins 6") != null);
}
