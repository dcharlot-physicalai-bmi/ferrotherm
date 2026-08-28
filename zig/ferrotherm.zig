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
    /// A state whose length does not match the graph, or that holds a value other than -1 or +1.
    BadState,
    /// An OMMX instance this sampler cannot represent. `ommxError` says which part.
    Unreadable,
    /// The exact planar solver cannot take this graph. `planarError` says which of the four
    /// reasons it was -- fields, non-planar, a cut vertex, or weights that do not scale to
    /// integers -- and they are four different things to do next.
    NotPlanar,
    /// A domain with nothing in it: a categorical under two values, or an integer with hi <= lo.
    BadDomain,
    /// A value the variable cannot take. Call `lastError` for the range it can.
    BadValue,
    /// A constraint the library would not accept. `lastError` says why.
    RejectedConstraint,
    /// A model that does not compile. `lastError` says why.
    WillNotCompile,
    /// Reading an answer before solving one.
    ///
    /// NOT the same as a variable that solved but did not decode -- `value` returns null for that,
    /// because the C ABI signals both with `i64::MIN` and only this side knows whether a solve has
    /// happened. Python and Julia make the same distinction; this used to collapse them.
    NotSolved,
    /// A name another variable already has.
    ///
    /// The C ABI refuses the rename and KEEPS the synthetic default (`v1`, `v2`, ...), so ignoring
    /// its return left the second variable silently carrying a name the caller never chose --
    /// which then appears in violation text and OMMX exports. Python raises and Julia throws.
    DuplicateName,
    /// There is no instance yet: compile or solve before asking for one.
    NotCompiled,
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

/// How many threads this machine can run at once, or 1 when that cannot be known.
///
/// An 18-core machine sampling on one core is the commonest way this library is left slow, and
/// this is where a caller learns the number instead of guessing it.
pub fn hardwareThreads() u32 {
    return c.ft_hardware_threads();
}

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

    /// Sweep across `threads` OS threads. Within a colour class no two nodes are adjacent, so the
    /// split is race-free by construction of the colouring.
    ///
    /// THE THREAD COUNT IS PART OF THE RUN. Bit-reproducible for a fixed (seed, threads), and a
    /// different count is a different, equally valid sample path -- record it beside the seed or
    /// the run is not reproducible from what you wrote down. `threads = 0` asks the machine, which
    /// is `hardwareThreads()`. `threadsUsed()` reports what actually ran.
    pub fn sweepPar(self: Sim, n: u32, threads: u32) u64 {
        return c.ft_sweep_par(self.h, n, threads);
    }

    /// How many threads the last `sweepPar` actually used, or 0 before one. Not the number asked
    /// for: a browser answers 1, and a colour class of three cannot occupy eight workers.
    pub fn threadsUsed(self: Sim) u32 {
        return c.ft_threads_used(self.h);
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

    /// Put a state INTO the simulation, so something computed elsewhere is scored, certified or
    /// annealed by exactly the same code that handles a state this library produced.
    ///
    /// It refuses rather than adapting: the length must match the graph and every value must be -1 or +1.
    /// A shorter state means whatever produced it did not finish, and a value that is not a spin means the
    /// buffer is not what the caller thinks it is. Both are cheap to launder into something plausible --
    /// pad with -1, coerce with `v > 0` -- and a laundered state is then scored with full confidence, which
    /// is how a dropped GPU dispatch turns into a believable energy.
    pub fn setSpins(self: Sim, state: []const i8) Error!void {
        if (c.ft_set_spins(self.h, state.ptr, @intCast(state.len)) == 0) return Error.BadState;
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

    // ---- solvers ------------------------------------------------------------------------------
    //
    // Each of these leaves its best state as the simulation's state, so `spins()` reads the answer
    // and `energy()` recomputes it from that state rather than trusting the number returned here.
    // They compose: anneal, then tabu from where annealing stopped, then `branch` with that as its
    // incumbent.

    /// Tabu search. Returns the energy of the best state found.
    ///
    /// `tenure = 0` scales the tenure to the graph; `restart_after = 0` never restarts. Check
    /// `tabuIterations` afterwards: a run shorter than its budget was truncated, which is
    /// otherwise invisible from outside.
    pub fn tabu(self: Sim, iterations: u32, tenure: u32, restart_after: u32) f64 {
        return c.ft_tabu(self.h, iterations, tenure, restart_after);
    }

    /// Iterations the last `tabu` actually ran, or 0 if there was none.
    pub fn tabuIterations(self: Sim) u64 {
        return c.ft_tabu_iterations(self.h);
    }

    /// EXACT max-cut on a planar graph, in polynomial time. Not a search.
    ///
    /// Max-cut is NP-hard in general and polynomial on a planar graph, and the difference is a
    /// theorem rather than an engineering margin. There is no budget to run out of and the answer
    /// is the maximum, not the best found. On success the simulation's state becomes the optimal
    /// partition, so `energy()` is then the PROVED minimum.
    ///
    /// Returns `error.NotPlanar` when the graph cannot be solved this way; `planarError` says which
    /// of the four reasons it was, and they are four different things to do next.
    pub fn exactPlanar(self: Sim, scale: f64) Error!PlanarCut {
        // Not named `c`: that is the @cImport alias, and shadowing it here compiles into a
        // recursive call to a float.
        const value = c.ft_planar_cut(self.h, scale);
        if (std.math.isNan(value)) return Error.NotPlanar;
        return .{
            .cut = value,
            .energy = self.energy(),
            .faces = c.ft_planar_faces(self.h),
            .odd_faces = c.ft_planar_odd_faces(self.h),
        };
    }

    /// Why the last `exactPlanar` refused, copied into `buf`.
    pub fn planarError(self: Sim, buf: []u8) []const u8 {
        const need = c.ft_planar_error(self.h, null, 0);
        const n = @min(need, @as(u32, @intCast(buf.len)));
        const got = c.ft_planar_error(self.h, buf.ptr, n);
        return buf[0..got];
    }

    /// An UPPER BOUND on the maximum cut of a toroidal grid, from the same dual reduction.
    ///
    /// The side of the G-set table nobody publishes: every figure there is a best cut FOUND, a
    /// lower bound. Measured, this closes the bracket on G11 and proves its best-known 564 optimal.
    ///
    /// `error.NotPlanar` unless the graph is a toroidal grid, whose structure is recovered from the
    /// edge list -- a match on all 2n edges rather than a guess.
    pub fn toroidalBound(self: Sim, scale: f64) Error!ToroidalBound {
        const v = c.ft_toroidal_bound(self.h, scale);
        if (std.math.isNan(v)) return Error.NotPlanar;
        return .{ .cut = v, .attained = c.ft_toroidal_attained(self.h) != 0 };
    }

    /// Goemans-Williamson: round the semidefinite relaxation to a state.
    ///
    /// THE ONLY WORST-CASE GUARANTEE IN MAX-CUT -- and `guaranteed` is false on most instances
    /// people care about, because the 0.87856 ratio is stated for non-negative edge weights.
    pub fn goemansWilliamson(self: Sim, hyperplanes: u32, seed: u64) Rounded {
        const cut = c.ft_gw_round(self.h, hyperplanes, seed);
        return .{ .cut = cut, .energy = self.energy(), .guaranteed = c.ft_gw_guaranteed(self.h) != 0 };
    }

    /// Parallel tempering with isoenergetic cluster moves -- the baseline the field measures
    /// against. `error.BadSchedule` on a graph with fields: the move preserves the pair's energy
    /// only at h = 0, and accepting it anyway would be silently wrong.
    pub fn clusterAnneal(self: Sim, rungs: u32, rounds: u32, beta_min: f64, beta_max: f64) Error!ClusterRun {
        const e = c.ft_icm(self.h, rungs, rounds, beta_min, beta_max);
        if (std.math.isNan(e)) return Error.BadSchedule;
        return .{ .energy = e, .moves = c.ft_icm_moves(self.h) };
    }

    /// Simulated quantum annealing. `trotter = 1` drops the Trotter coupling and is exactly
    /// classical annealing -- the honest control, not a degenerate case.
    pub fn quantumAnneal(self: Sim, trotter: u32, beta: f64, gamma_max: f64, gamma_min: f64, steps: u32) Error!f64 {
        const e = c.ft_sqa(self.h, trotter, beta, gamma_max, gamma_min, steps);
        if (std.math.isNan(e)) return Error.BadSchedule;
        return e;
    }

    /// Breakout local search -- the algorithm that holds the max-cut record on most of G-set.
    ///
    /// Steepest descent with an adaptive perturbation between local optima. One iteration is one
    /// SPIN FLIP, which is what `tabu` counts too, so giving both the same number is a
    /// matched-budget comparison -- the only comparison that is honest without a quiet machine.
    ///
    /// Check `descents`: a run with a handful of them spent its budget inside one basin.
    pub fn breakout(self: Sim, iterations: u32) BreakoutRun {
        const e = c.ft_bls(self.h, iterations);
        return .{
            .energy = e,
            .descents = c.ft_bls_descents(self.h),
            .iterations_run = c.ft_bls_iterations(self.h),
            .max_jump = c.ft_bls_max_jump(self.h),
        };
    }

    /// Population annealing on a linear ladder from beta = 0 to `beta_max`.
    ///
    /// Starting at zero, where `Z = 2^n` exactly, is what makes `PopulationRun.ln_z` an absolute
    /// free energy rather than a ratio.
    pub fn populationAnneal(self: Sim, population: u32, sweeps: u32, beta_max: f64, stages: u32) Error!PopulationRun {
        const e = c.ft_popanneal(self.h, population, sweeps, beta_max, stages);
        if (std.math.isNan(e)) return Error.BadSchedule;
        return .{
            .energy = e,
            .ln_z = c.ft_popanneal_ln_z(self.h),
            .rho = c.ft_popanneal_rho(self.h),
            .population = population,
        };
    }

    /// Branch and bound from the current state, which it uses as its incumbent.
    ///
    /// The only solver here that returns a PROOF. A run that exhausts `max_nodes` comes back with
    /// `proved = false` and the best state it saw.
    pub fn branch(self: Sim, max_nodes: u64) BranchResult {
        const e = c.ft_branch(self.h, max_nodes);
        return .{
            .energy = e,
            .proved = c.ft_branch_proved(self.h) != 0,
            .nodes = c.ft_branch_nodes(self.h),
        };
    }

    /// Every lower bound on the ground energy this library computes, and the best of them.
    ///
    /// All are sound on their own, so `Bounds.best()` is their maximum -- not a tie-break, a
    /// result: they disagree by a lot and in both directions.
    pub fn bounds(self: Sim, forest_rounds: u32, max_cycle: u32, sdp_sweeps: u32, seed: u64) Bounds {
        return .{
            .decoupled = c.ft_bound_decoupled(self.h),
            .forest = c.ft_bound_forest(self.h, forest_rounds),
            .odd_cycle = c.ft_bound_odd_cycle(self.h, max_cycle),
            .sdp = c.ft_bound_sdp(self.h, sdp_sweeps, seed),
        };
    }

    pub fn deinit(self: Sim) void {
        c.ft_free(self.h);
    }
};

/// An EXACT maximum cut on a planar graph. Not the best found -- the maximum.
pub const PlanarCut = struct {
    cut: f64,
    /// The proved MINIMUM energy of the partition this achieved.
    energy: f64,
    faces: u64,
    /// The size of the matching problem underneath, and the real cost driver. Zero is legitimate:
    /// a grid with uniform weights has every face of even degree and the whole cut comes free.
    odd_faces: u64,
};

/// An upper bound on the maximum cut of a torus, and whether it is achieved.
pub const ToroidalBound = struct {
    cut: f64,
    /// True when the relaxation's optimum is itself a genuine cut, so the bound IS the maximum.
    attained: bool,
};

/// A state rounded out of the semidefinite relaxation, and whether the guarantee covers it.
pub const Rounded = struct {
    cut: f64,
    energy: f64,
    /// False on most instances people care about: the ratio needs non-negative edge weights.
    guaranteed: bool,
};

/// A PT+ICM run. `moves` of zero means the replicas never disagreed and the move did nothing.
pub const ClusterRun = struct {
    energy: f64,
    moves: u64,
};

/// A breakout-local-search run, with the evidence that it actually broke out.
pub const BreakoutRun = struct {
    energy: f64,
    /// Local optima visited. A handful means the budget was spent inside one basin.
    descents: u64,
    iterations_run: u64,
    /// The largest jump reached. Above the initial L0 means the adaptive rule fired.
    max_jump: u32,
};

/// A population-annealing run, with the diagnostic that says whether to believe it.
pub const PopulationRun = struct {
    energy: f64,
    /// `ln Z` at the final beta, or NaN when the ladder did not start at infinite temperature.
    ln_z: f64,
    /// The family statistic: 1.0 when every ancestor still has a descendant, `population` when the
    /// population collapsed onto one and explored a single basin with N copies of one history.
    rho: f64,
    population: u32,

    /// `rho` below a tenth of the population. A rule of thumb, not a theorem -- the number is
    /// there so a caller can apply their own.
    pub fn trustworthy(self: PopulationRun) bool {
        return self.rho <= @max(1.0, @as(f64, @floatFromInt(self.population)) / 10.0);
    }
};

/// What branch and bound found, and whether it proved it.
pub const BranchResult = struct {
    energy: f64,
    /// True only when the tree was exhausted inside the node budget.
    proved: bool,
    nodes: u64,
};

/// Lower bounds on the ground energy. All sound, so `best()` is their maximum.
pub const Bounds = struct {
    decoupled: f64,
    forest: f64,
    odd_cycle: f64,
    /// NaN when the certificate failed to re-verify on the other side of the boundary. That is a
    /// refusal, not a missing feature.
    sdp: f64,

    pub fn best(self: Bounds) f64 {
        var b = @max(self.decoupled, @max(self.forest, self.odd_cycle));
        if (!std.math.isNan(self.sdp)) b = @max(b, self.sdp);
        return b;
    }
};

/// Onsager's exact spontaneous magnetisation for the 2D Ising model. Ground truth.
pub fn onsager(beta: f64) f64 {
    return c.ft_onsager(beta);
}

// ---- tests ---------------------------------------------------------------------------------

test "an answer is scored in the modeller's own units" {
    var p = try Problem.init();
    defer p.deinit();
    const mon = try p.categorical("mon", 3);
    const tue = try p.categorical("tue", 3);
    try p.notEqual(mon, tue);
    try p.prefer(.maximize, 5.0, mon.is(1));
    try p.prefer(.maximize, 4.0, tue.is(2));
    try p.solve(64);

    const obj = p.objective() orelse return error.TestUnexpectedResult;
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), obj, 1e-9);
    try std.testing.expect(p.feasible());
    // And it is NOT the compiled energy, which is the only number this used to hand back.
    try std.testing.expect(obj != p.energy());
}

test "a model with no objective reports none rather than zero" {
    var p = try Problem.init();
    defer p.deinit();
    const v = try p.categorical("v", 2);
    try p.fix(v, 1);
    try p.solve(8);
    // Zero would read as "worth nothing" instead of "not asked".
    try std.testing.expect(p.objective() == null);
}

test "a parallel sweep reproduces itself and reports what ran" {
    try std.testing.expect(hardwareThreads() >= 1);

    // The promise is per (seed, threads), not per seed.
    var ma = try Model.init(64);
    defer ma.deinit();
    var mb = try Model.init(64);
    defer mb.deinit();
    for (0..64) |i| {
        try ma.couple(@intCast(i), @intCast((i + 1) % 64), 1.0);
        try mb.couple(@intCast(i), @intCast((i + 1) % 64), 1.0);
    }
    var a = try ma.build(0.7, 0x2244);
    defer a.deinit();
    var b = try mb.build(0.7, 0x2244);
    defer b.deinit();

    _ = a.sweepPar(60, 4);
    _ = b.sweepPar(60, 4);
    try std.testing.expectApproxEqAbs(a.energy(), b.energy(), 1e-12);
    try std.testing.expect(a.threadsUsed() >= 1);

    // And 0 means ask the machine rather than run on nothing.
    var mc = try Model.init(64);
    defer mc.deinit();
    for (0..64) |i| try mc.couple(@intCast(i), @intCast((i + 1) % 64), 1.0);
    var d = try mc.build(0.7, 1);
    defer d.deinit();
    _ = d.sweepPar(10, 0);
    try std.testing.expect(d.threadsUsed() >= 1);
}

test "a higher-order term is solved without ancillas" {
    // The module doc's own example: a three-body parity term, minimised when the product is +1.
    var h = try Hubo.init(3);
    defer h.deinit();
    try h.add(&.{ 0, 1, 2 }, 1.0);

    const e = try h.anneal(0, 0, 0, 0, 7);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), e, 1e-9);

    var state: [3]i8 = undefined;
    try h.read(&state);
    try std.testing.expectEqual(@as(i32, 1), @as(i32, state[0]) * state[1] * state[2]);

    try std.testing.expectEqual(@as(u32, 1), h.terms());
    try std.testing.expectEqual(@as(u32, 3), h.maxArity());
    // The ceiling, not the cost: one substitution for one three-body term.
    try std.testing.expectEqual(@as(u32, 1), h.ancillasAvoided());
    try std.testing.expect(h.proposals() > 0);
}

test "a repeated variable is refused rather than silently changing the order" {
    var h = try Hubo.init(4);
    defer h.deinit();

    // s * s = 1, so [0, 0, 1] is a one-body term wearing a three-body's clothes.
    try std.testing.expectError(Error.RejectedEntry, h.add(&.{ 0, 0, 1 }, 1.0));
    try std.testing.expectError(Error.RejectedEntry, h.add(&.{ 0, 9 }, 1.0));
    try std.testing.expectEqual(@as(u32, 0), h.terms());

    // And a refused term leaves nothing pending for the next one to absorb.
    try h.add(&.{ 0, 1, 2 }, 1.0);
    try std.testing.expectEqual(@as(u32, 1), h.terms());
    try std.testing.expectEqual(@as(u32, 3), h.maxArity());
}

test "a lifted graph scores exactly as the pairwise path does" {
    var m = try Model.init(5);
    defer m.deinit();
    var i: u32 = 0;
    while (i < 4) : (i += 1) try m.couple(i, i + 1, if (i % 2 == 0) 1.0 else -1.0);
    try m.bias(0, 0.5);
    var sim = try m.build(0.9, 11);
    defer sim.deinit();
    _ = sim.sweep(20);

    var h = try Hubo.fromSim(sim);
    defer h.deinit();
    // The two paths must agree on the SAME state, or one of them has the sign convention wrong
    // and every later comparison inherits it silently.
    try std.testing.expectApproxEqAbs(sim.energy(), h.energy(), 1e-9);
    try std.testing.expectEqual(@as(u32, 2), h.maxArity());
    try std.testing.expectEqual(@as(u32, 0), h.ancillasAvoided());
}

test "a bad ladder is an error and NaN is not read as a default" {
    var h = try Hubo.init(3);
    defer h.deinit();
    try h.add(&.{ 0, 1, 2 }, 1.0);

    try std.testing.expectError(Error.BadSchedule, h.anneal(8.0, 0.05, 10, 10, 1));
    try std.testing.expectError(Error.BadSchedule, h.anneal(std.math.nan(f64), 8.0, 10, 10, 1));
    // Zeros DO mean "use the default", which is why NaN has to be refused before that test.
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), try h.anneal(0, 0, 0, 0, 1), 1e-9);
}

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

/// A certificate over a solved `Problem`.
///
/// Separate from `Certificate` because the numbers come from a different handle and the exact
/// bounds a simulation reports are not available here. Keeping one struct and leaving half its
/// fields NaN would say the bounds were computed and came out unknown, which is not what happened.
pub const ProblemCertificate = struct {
    beta_eff: f64,
    tau_int: f64,
    ess: f64,
    tv: f64,
    noise_floor: f64,
    findings: u32,
    h: ?*c.ft_model,

    /// Zero findings is the only thing that means the run is sound.
    pub fn passed(self: ProblemCertificate) bool {
        return self.findings == 0;
    }

    /// Finding `i`, written into `buf`.
    pub fn finding(self: ProblemCertificate, i: u32, buf: []u8) []const u8 {
        const need = c.ft_model_cert_finding(self.h, i, null, 0);
        const n = @min(need, @as(u32, @intCast(buf.len)));
        const got = c.ft_model_cert_finding(self.h, i, buf.ptr, n);
        return buf[0..got];
    }
};

/// Read an `ommx.v1.Instance` and return a simulation over it.
///
/// The direction that makes this a bridge rather than an exporter: a problem someone else compiled
/// to OMMX becomes something this sampler can run. `constant` receives the offset the 0/1 to +/-1
/// substitution introduces -- `ommx_objective(x) == sim.energy() + constant` -- and dropping it
/// leaves an energy that ranks states correctly and reports the wrong number.
///
/// Returns `error.Unreadable` for what this sampler cannot represent; `ommxError` says which.
pub fn ommxRead(bytes: []const u8, beta: f64, seed: u64, constant: *f64) Error!Sim {
    const h = c.ft_ommx_read(bytes.ptr, @intCast(bytes.len), beta, seed, constant);
    if (h == null) return Error.Unreadable;
    return Sim{ .h = h };
}

/// Why the last `ommxRead` on this thread failed, in the caller's own terms.
pub fn ommxError(buf: []u8) []const u8 {
    const need = c.ft_ommx_error(null, 0);
    const n = @min(need, @as(u32, @intCast(buf.len)));
    const got = c.ft_ommx_error(buf.ptr, n);
    return buf[0..got];
}

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
/// The exact ground STATE, written into `out`, or false if the graph is too dense for `max_width`.
///
/// `exactGround` says what the best energy is; this says which assignment reaches it, which is what
/// a caller checking a sampler against the truth actually needs to compare. `out` must be exactly
/// the node count long -- a wrong length is refused rather than truncated.
pub fn exactGroundState(sim: Sim, max_width: u32, out: []i8) bool {
    return c.ft_exact_ground_state(sim.h, max_width, out.ptr, @intCast(out.len)) == 1;
}

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
    /// Every VARIABLE named takes a different value; the literals' values are ignored.
    ///
    /// Lowered per shared value rather than per pair, so it costs nothing where two domains do not overlap, needs no slack and no ancillas, and its violation names WHICH value collided and who took it.
    /// More variables than the values they share is refused when the model compiles, by name: the pigeonhole principle checked rather than annealed, because such a model has no answer at any penalty and a longer ladder cannot help.
    all_different = 5,
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
/// A higher-order model, solved without quadratising it.
///
/// A term of any width contributes `-w * prod(s_i)`, so nothing here needs an ancilla. The other
/// route to a k-body term -- an objective product on `Problem` -- goes through Rosenberg's
/// reduction, which buys one ancilla per substituted pair and a penalty larger than the whole model
/// can pay. That penalty is what costs: on 60 three-body terms over 40 spins the reduced path at
/// 1024x the budget does not reach this one at 1x, because the landscape goes rigid rather than
/// merely larger.
pub const Hubo = struct {
    h: *c.ft_hubo,
    n: u32,

    /// A model over `n` spins. Zero is refused: a model with no variables can hold no term.
    pub fn init(n: u32) Error!Hubo {
        const h = c.ft_hubo_new(n) orelse return Error.OutOfMemory;
        return .{ .h = h, .n = n };
    }

    /// Lift a pairwise simulation, unchanged, so both paths can score the same state.
    pub fn fromSim(sim: Sim) Error!Hubo {
        const h = c.ft_hubo_from_sim(sim.h) orelse return Error.OutOfMemory;
        return .{ .h = h, .n = c.ft_hubo_len(h) };
    }

    pub fn deinit(self: Hubo) void {
        c.ft_hubo_free(self.h);
    }

    /// Add one term of any arity.
    ///
    /// A variable out of range or repeated within the term is refused: `s * s = 1`, so a repeat
    /// would silently change the term's order rather than mean what was written. The pending list
    /// is cleared either way, so a refused term cannot be absorbed by the next one.
    pub fn add(self: Hubo, vars: []const u32, weight: f64) Error!void {
        _ = c.ft_hubo_vars_clear(self.h);
        for (vars) |v| {
            if (c.ft_hubo_var(self.h, v) == 0) {
                _ = c.ft_hubo_vars_clear(self.h);
                return Error.RejectedEntry;
            }
        }
        if (c.ft_hubo_add(self.h, weight) == 0) return Error.RejectedEntry;
    }

    /// Anneal and return the best energy. Zero for any ladder parameter means its own default.
    pub fn anneal(
        self: Hubo,
        beta_min: f64,
        beta_max: f64,
        stages: u32,
        sweeps_per_stage: u32,
        seed: u64,
    ) Error!f64 {
        const e = c.ft_hubo_anneal(self.h, beta_min, beta_max, stages, sweeps_per_stage, seed);
        if (std.math.isNan(e)) return Error.BadSchedule;
        return e;
    }

    /// Copy the current state out. `out` must be exactly `n` long, and is never partly written.
    pub fn read(self: Hubo, out: []i8) Error!void {
        if (c.ft_hubo_read(self.h, out.ptr, @intCast(out.len)) == 0) return Error.BadState;
    }

    /// Put a state in, so something computed elsewhere is scored by this library. Refuses any
    /// element that is not -1 or +1, and refuses the whole write rather than part of it.
    pub fn setState(self: Hubo, spins: []const i8) Error!void {
        if (c.ft_hubo_set_spins(self.h, spins.ptr, @intCast(spins.len)) == 0) return Error.BadState;
    }

    /// Energy of the current state.
    pub fn energy(self: Hubo) f64 {
        return c.ft_hubo_energy(self.h);
    }

    /// The energy change from flipping spin `i`, in O(terms containing i).
    pub fn delta(self: Hubo, i: u32) Error!f64 {
        const d = c.ft_hubo_delta(self.h, i);
        if (std.math.isNan(d)) return Error.RejectedEntry;
        return d;
    }

    pub fn terms(self: Hubo) u32 {
        return c.ft_hubo_terms(self.h);
    }

    pub fn maxArity(self: Hubo) u32 {
        return c.ft_hubo_max_arity(self.h);
    }

    /// An UPPER BOUND on what a pairwise reduction would have spent, not the cost: `reduce`
    /// substitutes the commonest pair first, so one ancilla serves every term containing it.
    pub fn ancillasAvoided(self: Hubo) u32 {
        return c.ft_hubo_ancillas_avoided(self.h);
    }

    pub fn proposals(self: Hubo) u64 {
        return c.ft_hubo_proposals(self.h);
    }

    pub fn accepted(self: Hubo) u64 {
        return c.ft_hubo_accepted(self.h);
    }

    /// What the last run WOULD have cost on a Z1-class device (vendor SPICE, pre-silicon).
    pub fn joulesZ1(self: Hubo) f64 {
        return c.ft_hubo_joules_z1(self.h);
    }

    /// Why the last call was refused. Empty when nothing was.
    pub fn lastError(self: Hubo, buf: []u8) []const u8 {
        const need = c.ft_hubo_error(self.h, null, 0);
        const n = @min(need, @as(u32, @intCast(buf.len)));
        const got = c.ft_hubo_error(self.h, buf.ptr, n);
        return buf[0..got];
    }
};

pub const Problem = struct {
    /// Whether a solve has happened, so `value` can tell "never solved" from "did not decode".
    /// The C ABI cannot: it returns the same sentinel for both.
    solved: bool = false,
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
        // handle they were given back -- and CHECK it. Discarding this return let a duplicate name
        // through: the C ABI refuses the rename, keeps the synthetic `v1`, and returns 0, so
        // `p.binary("shift")` twice gave a second variable silently called "v1". Python raises and
        // Julia throws; this said nothing. `lastError` carries the reason, which also covers a
        // non-UTF-8 name.
        if (c.ft_model_name(self.h, idx, name.ptr, @intCast(name.len)) == 0) {
            return Error.DuplicateName;
        }
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

    /// `count`, but as a PREFERENCE priced at `weight` rather than a rule.
    ///
    /// A hard constraint says which answers are answers at all, so breaking one makes `feasible`
    /// false. A soft one may be traded away: breaking it costs and the answer stays feasible.
    /// `softCost` totals what was traded. See `softenLast` for the shape of the price.
    pub fn countSoft(self: *Problem, kind: Counting, k: u32, lits: []const Lit, weight: f64) Error!void {
        _ = c.ft_model_lits_clear(self.h);
        for (lits) |l| {
            if (c.ft_model_lit(self.h, l.v.idx, l.value) == 0) return Error.BadValue;
        }
        if (c.ft_model_close_soft(self.h, @intFromEnum(kind), k, weight) == 0) {
            return Error.RejectedConstraint;
        }
    }

    /// Every one of these variables takes a different value.
    ///
    /// See `Counting.all_different`. `k` is ignored and the values passed are irrelevant, so this
    /// takes variables directly rather than literals.
    pub fn allDifferent(self: *Problem, vars: []const Var) Error!void {
        _ = c.ft_model_lits_clear(self.h);
        for (vars) |v| {
            // ft_model_var, not a placeholder value: the library picks one from the variable's own
            // domain, which a caller has no reason to know and which a placeholder gets wrong for
            // any domain that does not contain it.
            if (c.ft_model_var(self.h, v.idx) == 0) return Error.BadValue;
        }
        if (c.ft_model_close(self.h, @intFromEnum(Counting.all_different), 0) == 0) {
            return Error.RejectedConstraint;
        }
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

    /// Make the constraint added most recently a preference, priced at `weight`.
    ///
    /// The price is `weight × amount²`, and the square is not a detail: a constraint becomes an
    /// energy term by squaring how far outside it sits, so missing by two costs FOUR times missing
    /// by one. Pricing a preference chooses that curve as well as its scale.
    ///
    /// The weight is absolute rather than scaled. Automatic scaling exists to stop a hard
    /// constraint being outbid by the objective; a soft one is meant to be traded against it.
    pub fn softenLast(self: *Problem, weight: f64) Error!void {
        if (c.ft_model_soften_last(self.h, weight) == 0) return Error.RejectedConstraint;
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

    /// Compile and solve, keeping the best of `tries` anneals.
    ///
    /// Compiles first, the way the Python and Julia bindings do. This used to call solve alone, so
    /// the same program written from those bindings' examples returned `Error.NotSolved` -- which
    /// says the sampler failed, when in fact nothing had been built for it to sample. One binding
    /// out of step with the others is a trap for anyone who reads more than one, and
    /// `ft_model_compile` is idempotent, so there is nothing to lose by making it the same.
    pub fn solve(self: *Problem, tries: u32) Error!void {
        if (c.ft_model_compile(self.h) == 0) return Error.WillNotCompile;
        if (c.ft_model_solve(self.h, tries) == 0) return Error.NotSolved;
        self.solved = true;
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
        if (c.ft_model_compile(self.h) == 0) return Error.WillNotCompile;
        if (c.ft_model_solve_with(self.h, tries, beta_hot, beta_cold, stages, sweeps) == 0) {
            return Error.BadSchedule;
        }
        self.solved = true;
    }

    /// The answer for one variable, in its own units, or null if it did not decode.
    ///
    /// The C ABI signals BOTH "no solution yet" and "solved, but this variable did not decode" with
    /// `i64::MIN`, and only this side knows which. Collapsing them into `Error.NotSolved` was
    /// actively misleading: a binary encoding of a non-power-of-two k can land on a spare codeword,
    /// so a perfectly good solve leaves one variable undecoded -- and the caller was told the model
    /// had never been solved. Python returns `None` and Julia `nothing` for exactly this case.
    pub fn value(self: *Problem, v: Var) Error!?i64 {
        if (!self.solved) return Error.NotSolved;
        const got = c.ft_model_value(self.h, v.idx);
        if (got == std.math.minInt(i64)) return null;
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

    /// The compiled Ising energy: the objective, every penalty and the constant, all folded in.
    ///
    /// A number about SPINS. It compares two answers to the same model and nothing else, and it
    /// moves when the penalty does. For what the answer is WORTH, see `objective`.
    pub fn energy(self: *Problem) f64 {
        return c.ft_model_energy(self.h);
    }

    /// The objective's value in your own units, in the direction you wrote it.
    ///
    /// Null when no objective was written, when both senses were used and there is no single
    /// direction to report, or when a variable did not decode and there is only half an answer to
    /// score. Write `maximize 5*mon + 4*tue`, get mon = 1 and tue = 2, and this reads 9 where
    /// `energy` reads a number in the hundreds with the penalties in it.
    pub fn objective(self: *Problem) ?f64 {
        if (c.ft_model_has_objective(self.h) == 0) return null;
        return c.ft_model_objective(self.h);
    }

    /// The penalty actually used, after any automatic scaling.
    pub fn effectivePenalty(self: *Problem) f64 {
        return c.ft_model_penalty(self.h);
    }

    /// How many constraints the answer breaks. Zero when it keeps everything it was asked to.
    pub fn violations(self: *Problem) u32 {
        return c.ft_model_violations(self.h);
    }

    /// The compiled model as an OMMX instance -- the interchange format this corner of the field converged on, so a ferrotherm program can be read by jijmodeling, Jij's stack, and anything else that speaks it.
    /// Returns the protobuf bytes and the constant the +/-1 to 0/1 substitution introduces: ferrotherm_energy(s) == ommx_objective(x) + constant.
    pub fn ommx(self: *Problem, buf: []u8) Error![]const u8 {
        const need = c.ft_model_ommx(self.h, null, 0);
        // A compiled instance is never empty, so need == 0 means "not compiled" unambiguously.
        // This used to return an empty slice with no signal, and a caller wrote a zero-byte .ommx
        // file believing it had serialised something. Python and Julia both raise here.
        if (need == 0) return Error.NotCompiled;
        const n = @min(need, @as(u32, @intCast(buf.len)));
        const got = c.ft_model_ommx(self.h, buf.ptr, n);
        return buf[0..got];
    }

    /// The offset the +/-1 to 0/1 substitution produced, ALREADY folded into the instance.
    /// Read it, do not add it: ommx_objective(x) == ferrotherm_energy(s) exactly.
    pub fn ommxConstant(self: *Problem) f64 {
        return c.ft_model_ommx_constant(self.h);
    }

    /// How many compile-time caveats this model carries.
    ///
    /// What the compiler knows is wrong with the model and cannot fix.
    /// Today there is one kind: an encoding no penalty can make exact -- a binary encoding of k values spells 2^ceil(log2 k) codewords, and when k is not a power of two the spare ones decode to nothing while costing exactly what a valid state costs.
    /// Read them before trusting a result; empty is the normal case.
    pub fn caveats(self: *Problem) u32 {
        return c.ft_model_caveats(self.h);
    }

    /// Caveat `i`, written into `buf`.
    pub fn caveat(self: *Problem, i: u32, buf: []u8) []const u8 {
        const need = c.ft_model_caveat(self.h, i, null, 0);
        const n = @min(need, @as(u32, @intCast(buf.len)));
        const got = c.ft_model_caveat(self.h, i, buf.ptr, n);
        return buf[0..got];
    }

    /// Spins the higher-order lowering added, or zero if nothing named three or more variables.
    ///
    /// Non-zero means the answer solves a model with MORE spins than the variables required. The
    /// extra states are what makes the reduction exact for OPTIMISATION and not for sampling: the
    /// Boltzmann distribution over the original variables is not preserved. Read this before
    /// drawing samples from a solved model, rather than only its ground state.
    pub fn ancillas(self: *Problem) u32 {
        return c.ft_model_ancillas(self.h);
    }

    /// What the traded-away preferences cost. Zero when none broke, or before solving.
    pub fn softCost(self: *Problem) f64 {
        return c.ft_model_soft_cost(self.h);
    }

    /// Whether violation `i` is a hard one, or a preference the solver traded away.
    pub fn violationIsHard(self: *Problem, i: u32) bool {
        return c.ft_model_violation_is_hard(self.h, i) == 1;
    }

    /// Violation `i`, described in your own names, written into `buf`.
    pub fn violation(self: *Problem, i: u32, buf: []u8) []const u8 {
        const need = c.ft_model_violation(self.h, i, null, 0);
        const n = @min(need, @as(u32, @intCast(buf.len)));
        const got = c.ft_model_violation(self.h, i, buf.ptr, n);
        return buf[0..got];
    }

    /// How far outside violation `i` sits -- not merely that it broke.
    ///
    /// A caller ranking repairs, or deciding whether a larger penalty would be enough, needs the
    /// magnitude: "at most two, and four hold" is over by two, and that is a different problem from
    /// being over by one.
    pub fn violationAmount(self: *Problem, i: u32) f64 {
        return c.ft_model_violation_amount(self.h, i);
    }

    /// Certify the sampling behind the answer, the same way `certify` does for a simulation.
    ///
    /// An answer is a state the sampler reached; a certificate says whether the chain that reached
    /// it had mixed. `draws` must be at least 16. Zero findings is the only thing that means sound.
    pub fn certify(self: *Problem, beta: f64, draws: u32, thin: u32) Error!ProblemCertificate {
        if (c.ft_model_certify(self.h, beta, draws, thin) == 0) return Error.TooFewDraws;
        return .{
            .beta_eff = c.ft_model_cert_beta(self.h),
            .tau_int = c.ft_model_cert_tau(self.h),
            .ess = c.ft_model_cert_ess(self.h),
            .tv = c.ft_model_cert_tv(self.h),
            .noise_floor = c.ft_model_cert_floor(self.h),
            .findings = c.ft_model_cert_findings(self.h),
            .h = self.h,
        };
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

test "the ancilla count is readable, because sampling from a reduced model is not sound" {
    var p = try Problem.init();
    defer p.deinit();
    const a = try p.binary("a");
    const b = try p.binary("b");
    const d = try p.binary("c");
    try p.preferAll(.maximize, 3.0, &.{ a.is(1), b.is(1), d.is(1) });
    const spins = try p.compile();
    try p.solve(8);
    // Three variables in one term is a three-body statement, lowered with one ancilla. Without a
    // way to READ that, a caller cannot tell a model whose Boltzmann distribution is preserved from
    // one whose is not -- and this file had a test named for ancillas that could not see them.
    try std.testing.expectEqual(@as(u32, 1), p.ancillas());
    // Seven, not four: a binary is one-hot over two values, so each costs two spins, and the
    // ancilla is the seventh. The ancilla count is what separates the two -- the spin total alone
    // cannot say whether any of them were added by the lowering.
    try std.testing.expectEqual(@as(u32, 7), spins);
}

test "the exact ground state is readable, not only its energy" {
    var m = try Model.init(6);
    var i: u32 = 0;
    while (i < 5) : (i += 1) try m.couple(i, i + 1, 1.0);
    var sim = try m.build(1.0, 1);
    defer sim.deinit();

    var out: [6]i8 = undefined;
    try std.testing.expect(exactGroundState(sim, 20, &out));
    // A ferromagnetic chain: every spin agrees, and the energy is one per bond.
    for (out) |s| try std.testing.expectEqual(out[0], s);
    try std.testing.expectEqual(@as(f64, -5.0), exactGround(sim, 20).?);
    // A wrong length is refused rather than truncated: a short buffer would otherwise be filled
    // partway and read as a whole answer.
    var short: [4]i8 = undefined;
    try std.testing.expect(!exactGroundState(sim, 20, &short));
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

test "a preference is traded, a rule is not" {
    // The same constraint twice over, once as a rule and once as a price. What changes is not
    // whether the solver can break it -- a penalty was always breakable -- but what the answer
    // MEANS when it does. A broken rule makes the answer no answer; a traded preference is the
    // choice the modeller asked the solver to make, and it stays feasible.
    var p = try Problem.init();
    defer p.deinit();
    const a = try p.categorical("a", 2);
    const b = try p.categorical("b", 2);
    try p.notEqual(a, b);
    try p.softenLast(1.0);
    try p.prefer(.maximize, 5.0, a.is(0));
    try p.prefer(.maximize, 5.0, b.is(0));
    _ = try p.compile();
    try p.solve(24);
    try std.testing.expect(p.feasible());
    try std.testing.expectEqual(@as(u32, 1), p.violations());
    try std.testing.expect(!p.violationIsHard(0));
    try std.testing.expectEqual(@as(f64, 1.0), p.softCost());

    // The identical model with the preference priced above the objective keeps it instead, and
    // costs nothing. Same code path, opposite trade.
    var q = try Problem.init();
    defer q.deinit();
    const c2 = try q.categorical("a", 2);
    const d = try q.categorical("b", 2);
    try q.notEqual(c2, d);
    try q.softenLast(50.0);
    try q.prefer(.maximize, 5.0, c2.is(0));
    try q.prefer(.maximize, 5.0, d.is(0));
    _ = try q.compile();
    try q.solve(24);
    try std.testing.expect(q.feasible());
    try std.testing.expectEqual(@as(u32, 0), q.violations());
    try std.testing.expectEqual(@as(f64, 0.0), q.softCost());
}

test "a soft counting constraint prices the overshoot, squared" {
    var p = try Problem.init();
    defer p.deinit();
    var vars: [4]Var = undefined;
    var lits: [4]Lit = undefined;
    for (0..4) |i| {
        var name: [8]u8 = undefined;
        vars[i] = try p.binary(std.fmt.bufPrint(&name, "v{d}", .{i}) catch unreachable);
        lits[i] = vars[i].is(1);
        try p.prefer(.maximize, 20.0, lits[i]);
    }
    try p.countSoft(.at_most, 1, &lits, 1.0);
    _ = try p.compile();
    try p.solve(24);
    try std.testing.expect(p.feasible());
    // All four held against a cap of one, so it is over by three -- and three squared is nine,
    // not three. Missing by two costs four times missing by one.
    try std.testing.expectEqual(@as(f64, 9.0), p.softCost());
}

test "a violation reports how far outside it sits, not only that it broke" {
    var p = try Problem.init();
    defer p.deinit();
    var vars: [4]Var = undefined;
    var lits: [4]Lit = undefined;
    for (0..4) |i| {
        var name: [8]u8 = undefined;
        vars[i] = try p.binary(std.fmt.bufPrint(&name, "v{d}", .{i}) catch unreachable);
        lits[i] = vars[i].is(1);
        try p.prefer(.maximize, 20.0, lits[i]);
    }
    try p.count(.at_most, 1, &lits);
    try p.penalty(1.0);  // pinned below the objective, so the constraint loses on purpose
    _ = try p.compile();
    try p.solve(24);
    try std.testing.expect(!p.feasible());
    try std.testing.expectEqual(@as(u32, 1), p.violations());
    // Over by three, not merely broken. A caller deciding whether a larger penalty would be enough
    // cannot get that from the text.
    try std.testing.expectEqual(@as(f64, 3.0), p.violationAmount(0));
}

test "a solved problem certifies its own sampling" {
    var p = try Problem.init();
    defer p.deinit();
    const a = try p.categorical("a", 3);
    const b = try p.categorical("b", 3);
    try p.notEqual(a, b);
    _ = try p.compile();
    try p.solve(8);

    // Too few draws says nothing, and the library refuses rather than returning a clean-looking
    // certificate over four samples.
    try std.testing.expectError(Error.TooFewDraws, p.certify(1.0, 4, 1));

    const cert = try p.certify(1.0, 512, 1);
    try std.testing.expect(cert.ess > 0);
    // Not `tau >= 1`: tau_int here measures 0.58, and an integrated autocorrelation time below one
    // is real for a chromatic sweep that decorrelates faster than one pass. Asserting the textbook
    // floor would have been asserting a convention this sampler does not obey.
    try std.testing.expect(cert.tau_int > 0);
    try std.testing.expect(cert.ess <= 512);
    try std.testing.expect(cert.beta_eff > 0);
    // This model DOES report a finding at these settings -- tv 0.097 sits under a noise floor of
    // 0.190, so the distance to the exact distribution cannot be told from sampling noise. Asserting
    // `passed()` would have made this test a check that the certifier stays quiet, which is the
    // opposite of what a certificate is for.
    try std.testing.expect(cert.findings > 0);
    var buf: [256]u8 = undefined;
    var i: u32 = 0;
    var saw = false;
    while (i < cert.findings) : (i += 1) {
        const text = cert.finding(i, &buf);
        try std.testing.expect(text.len > 0);
        saw = true;
    }
    try std.testing.expect(saw);
}

test "a state computed elsewhere is scored by the same code, or refused" {
    // The point of putting a state in is that whatever produced it -- a GPU sweep, another solver --
    // is then judged by the code that judges this library's own answers. That only means anything if
    // the state arrives intact, so the refusals matter more than the success.
    var b = try Model.init(4);
    var i: u32 = 0;
    while (i < 4) : (i += 1) try b.couple(i, (i + 1) % 4, 1.0);
    var sim = try b.build(1.0, 7);
    defer sim.deinit();

    try sim.setSpins(&.{ 1, 1, 1, 1 });
    // A ferromagnetic ring, every bond satisfied: -1 per bond over four bonds.
    try std.testing.expectEqual(@as(f64, -4.0), sim.energy());

    // Short, and a value that is not a spin. Both are trivially launderable -- pad with -1, coerce
    // with `v > 0` -- and a laundered state is scored with full confidence.
    try std.testing.expectError(Error.BadState, sim.setSpins(&.{ 1, 1, 1 }));
    try std.testing.expectError(Error.BadState, sim.setSpins(&.{ 1, 0, 1, 1 }));
    // and the refusals left the good state in place rather than half-writing over it
    try std.testing.expectEqual(@as(f64, -4.0), sim.energy());
}

test "the three places this binding used to disagree with Python and Julia" {
    // Cross-binding drift, found by auditing every surface against the C ABI. Each was a case where
    // Zig did something observably different for the same user intent -- the class of bug
    // `scripts/check-answers.sh` catches at the answer level, pinned here at the call level.
    var p = try Problem.init();
    defer p.deinit();

    // 1. A duplicate name was accepted and SILENTLY renamed. The C ABI refuses the rename and keeps
    //    the synthetic default, so the second variable quietly became "v1" -- a name that then
    //    appears in violation text and OMMX exports. Python raises, Julia throws.
    _ = try p.binary("shift");
    try std.testing.expectError(Error.DuplicateName, p.binary("shift"));

    // 2. ommx() before compile returned an EMPTY SLICE with no signal, so a caller wrote a
    //    zero-byte instance believing it had serialised one. Both other bindings raise.
    var buf: [256]u8 = undefined;
    try std.testing.expectError(Error.NotCompiled, p.ommx(&buf));

    // 3. value() before a solve is NotSolved -- but that error used to ALSO mean "solved, and this
    //    variable did not decode", because the C ABI signals both with i64::MIN. Only this side
    //    knows which, so they are different values now: an error, and null.
    const a = try p.binary("a");
    try std.testing.expectError(Error.NotSolved, p.value(a));

    try p.solve(16);
    try std.testing.expect((try p.value(a)) != null); // a binary always decodes
}

test "all_different solves a latin square row, and pigeonhole is refused not annealed" {
    var p = try Problem.init();
    defer p.deinit();
    var vars: [4]Var = undefined;
    for (0..4) |i| {
        var name: [8]u8 = undefined;
        vars[i] = try p.categorical(std.fmt.bufPrint(&name, "c{d}", .{i}) catch unreachable, 4);
    }
    try p.allDifferent(&vars);
    _ = try p.compile();
    try p.solve(60);
    try std.testing.expect(p.feasible());
    var seen = [_]bool{false} ** 4;
    for (vars) |v| {
        // `value` is optional now: null means solved-but-undecoded, which all_different over a
        // one-hot encoding never produces. Asserting it is non-null is the point -- if that ever
        // changes, this says so rather than silently indexing something else.
        const got = (try p.value(v)) orelse return error.TestUnexpectedResult;
        try std.testing.expect(!seen[@intCast(got)]);
        seen[@intCast(got)] = true;
    }

    // Five variables over three values has no answer at any penalty, so the compiler refuses
    // rather than annealing and reporting infeasible -- which would read as "raise the penalty".
    var q = try Problem.init();
    defer q.deinit();
    var few: [5]Var = undefined;
    for (0..5) |i| {
        var name: [8]u8 = undefined;
        few[i] = try q.categorical(std.fmt.bufPrint(&name, "x{d}", .{i}) catch unreachable, 3);
    }
    try q.allDifferent(&few);
    try std.testing.expectError(error.WillNotCompile, q.compile());
    // The error type alone says "it did not compile". What a modeller needs is WHY, and that it
    // is not a penalty they can raise -- so the message has to carry the counting argument.
    var buf: [512]u8 = undefined;
    const why = q.lastError(&buf);
    try std.testing.expect(std.mem.indexOf(u8, why, "pigeonhole") != null or
        std.mem.indexOf(u8, why, "No assignment can satisfy") != null);
}

// ---- solvers and bounds -------------------------------------------------------------------------
//
// The C ABI could build a graph and sample it but not ask how far from optimal the sample was, so a
// Zig user could do the easy half of what this library is for. These check the other half crossed
// the boundary intact, rather than that the symbols merely resolve.

/// A ring with one flipped bond: enumerable, and genuinely frustrated.
fn frustratedRing(n: u32) !Sim {
    var b = try Model.init(n);
    var i: u32 = 0;
    while (i < n) : (i += 1) try b.couple(i, (i + 1) % n, if (i == 0) -1.0 else 1.0);
    return b.build(1.0, 3);
}

test "every solver leaves the state whose energy it reported" {
    // The number returned is a claim about `spins()`, not a separate answer.
    {
        var sim = try frustratedRing(12);
        defer sim.deinit();
        const e = sim.tabu(5_000, 0, 1_000);
        try std.testing.expectApproxEqAbs(e, sim.energy(), 1e-9);
        try std.testing.expectEqual(@as(u64, 5_000), sim.tabuIterations());
    }
    {
        var sim = try frustratedRing(12);
        defer sim.deinit();
        const r = try sim.populationAnneal(64, 2, 3.0, 20);
        try std.testing.expectApproxEqAbs(r.energy, sim.energy(), 1e-9);
        // Z(0) = 2^n and Z is non-decreasing in beta, so ln Z is at least n ln 2 at any beta.
        try std.testing.expect(r.ln_z >= 12.0 * @log(2.0) - 1e-9);
        try std.testing.expect(r.rho >= 1.0 and r.rho <= 64.0);
    }
    {
        var sim = try frustratedRing(12);
        defer sim.deinit();
        const r = sim.branch(2_000_000);
        try std.testing.expectApproxEqAbs(r.energy, sim.energy(), 1e-9);
        try std.testing.expect(r.proved and r.nodes > 0);
        // A frustrated ring of n bonds can satisfy all but one: -(n - 1) + 1.
        try std.testing.expectApproxEqAbs(@as(f64, -10.0), r.energy, 1e-9);
    }
}

test "branch and bound withholds a proof when the budget runs out" {
    // The flag is the whole product. A search that gave up and still said `proved` would be worse
    // than one that returned nothing.
    var b = try Model.init(40);
    var i: u32 = 0;
    while (i < 40) : (i += 1) {
        var j: u32 = i + 1;
        while (j < 40) : (j += 1) try b.couple(i, j, if ((i * 7 + j) % 3 == 0) -1.0 else 1.0);
    }
    var sim = try b.build(1.0, 1);
    defer sim.deinit();
    const r = sim.branch(200);
    try std.testing.expect(!r.proved);
    try std.testing.expect(r.nodes <= 201);
}

test "the three closed gaps, and the caveats they carry" {
    // A 6x6 ANTIferromagnet: non-positive couplings, so the GW guarantee applies and the graph is
    // bipartite, so all 72 edges are cuttable.
    var anti = try Sim.lattice2d(6, -1.0, 1.0, 3);
    defer anti.deinit();
    const r = anti.goemansWilliamson(64, 5);
    try std.testing.expect(r.guaranteed);
    try std.testing.expectEqual(@as(f64, 72.0), r.cut);

    // A ferromagnet is OUTSIDE the hypothesis and the flag must say so.
    var ferro = try Sim.lattice2d(6, 1.0, 1.0, 3);
    defer ferro.deinit();
    try std.testing.expect(!ferro.goemansWilliamson(16, 5).guaranteed);

    // Cluster moves fire, and quantum annealing finds the ferromagnetic ground state.
    const cr = try ferro.clusterAnneal(8, 200, 0.1, 4.0);
    try std.testing.expectApproxEqAbs(@as(f64, -72.0), cr.energy, 1e-9);
    try std.testing.expect(cr.moves > 0);
    const q = try ferro.quantumAnneal(4, 10.0, 3.0, 0.05, 200);
    try std.testing.expectApproxEqAbs(@as(f64, -72.0), q, 1e-9);
}

test "the toroidal bound bounds, and a search cannot beat it" {
    // A 6x6 periodic lattice: a torus. Bipartite, so every one of its 72 edges is cut.
    var torus = try Sim.lattice2d(6, -1.0, 1.0, 3);
    defer torus.deinit();
    const b = try torus.toroidalBound(1.0);
    try std.testing.expectEqual(@as(f64, 72.0), b.cut);
    try std.testing.expect(b.attained);
    // And the planar solver correctly declines the same graph.
    try std.testing.expectError(Error.NotPlanar, torus.exactPlanar(1.0));
}

test "the exact planar solver, and the reason it refuses" {
    // A 4x4 antiferromagnetic grid: bipartite, so every one of its 24 edges is cut.
    var b = try Model.init(16);
    var y: u32 = 0;
    while (y < 4) : (y += 1) {
        var x: u32 = 0;
        while (x < 4) : (x += 1) {
            const i = y * 4 + x;
            if (x + 1 < 4) try b.couple(i, i + 1, -1.0);
            if (y + 1 < 4) try b.couple(i, i + 4, -1.0);
        }
    }
    var sim = try b.build(1.0, 1);
    defer sim.deinit();
    const r = try sim.exactPlanar(1.0);
    try std.testing.expectEqual(@as(f64, 24.0), r.cut);
    try std.testing.expectEqual(@as(f64, -24.0), r.energy);
    try std.testing.expectEqual(@as(u64, 10), r.faces);

    // A torus is genus 1, and the reduction is a plane statement. The REASON has to cross too:
    // "not planar" and "has a cut vertex" are different instructions to a caller.
    var torus = try Sim.lattice2d(4, 1.0, 1.0, 1);
    defer torus.deinit();
    try std.testing.expectError(Error.NotPlanar, torus.exactPlanar(1.0));
    var buf: [512]u8 = undefined;
    const msg = torus.planarError(&buf);
    try std.testing.expect(std.mem.indexOf(u8, msg, "not planar") != null);
}

test "breakout local search reports the evidence that it broke out" {
    // The claim BLS makes is about what happens BETWEEN local optima, so a run that never left one
    // has not run the algorithm -- and nothing in the energy alone would say so.
    var sim = try frustratedRing(24);
    defer sim.deinit();
    const r = sim.breakout(20_000);
    try std.testing.expectApproxEqAbs(r.energy, sim.energy(), 1e-9);
    try std.testing.expectEqual(@as(u64, 20_000), r.iterations_run);
    try std.testing.expect(r.descents > 1);
    try std.testing.expect(r.max_jump >= 1);
}

test "no bound exceeds a ground energy the same object can prove" {
    // One-sided on purpose: a bound may be loose by any amount and may never exceed the optimum.
    var sim = try frustratedRing(12);
    defer sim.deinit();
    const truth = sim.branch(2_000_000).energy;
    const b = sim.bounds(40, 6, 200, 1);
    try std.testing.expect(b.decoupled <= truth + 1e-9);
    try std.testing.expect(b.forest <= truth + 1e-9);
    try std.testing.expect(b.odd_cycle <= truth + 1e-9);
    try std.testing.expect(std.math.isNan(b.sdp) or b.sdp <= truth + 1e-9);
    try std.testing.expect(b.best() <= truth + 1e-9);
    // On a ring with no fields `forest` cannot beat `decoupled`: a tree is never frustrated.
    try std.testing.expectApproxEqAbs(b.decoupled, b.forest, 1e-9);
}
