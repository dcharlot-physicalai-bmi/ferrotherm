"""Tests for the modelling layer.

Run from the repository root, after `cargo build --release`:

    python3 -m pytest python/test_model.py -q

Each test names a way this layer can be wrong without looking wrong. Sampling is stochastic, so
nothing here asserts on a particular random draw: every assertion is about a property the answer
must have for any correct solver.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ferrotherm as ft  # noqa: E402


def test_a_triangle_needs_three_colours():
    p = ft.Problem()
    r = {n: p.categorical(n, 3) for n in ("west", "middle", "east")}
    p.not_equal(r["west"], r["middle"])
    p.not_equal(r["middle"], r["east"])
    p.not_equal(r["west"], r["east"])
    a = p.solve()
    assert a.feasible, a
    assert len({a["west"], a["middle"], a["east"]}) == 3, a


def test_an_integer_is_written_in_its_own_values():
    """The trap: a variable over 10..=20 holds 13 in its fourth slot. The literal says 13."""
    p = ft.Problem()
    t = p.integer("temperature", 10, 20)
    p.maximize(5 * t.is_(13))
    assert p.solve()["temperature"] == 13

    p = ft.Problem()
    t = p.integer("temperature", 10, 20)
    with pytest.raises(ValueError, match=r"temperature.*10\.\.=20"):
        p.fix(t, 3)


def test_a_range_below_zero_works():
    p = ft.Problem()
    d = p.integer("drift", -40, -10)
    p.fix(d, -25)
    assert p.solve()["drift"] == -25


def test_at_most_is_a_ceiling_and_at_least_is_a_floor():
    """The distinction an inequality exists for, in the only case that shows it.

    With every variable rewarded, `at most 2` and `exactly 2` agree. With every variable penalised,
    they do not: an inequality is satisfied by taking none, an equality is not.
    """
    def run(kind, sign):
        p = ft.Problem()
        vs = [p.binary(f"v{i}") for i in range(4)]
        getattr(p, kind)(vs, 2)
        p.maximize(sum(sign * (4 - i) * v.is_(1) for i, v in enumerate(vs)))
        a = p.solve()
        assert a.feasible, a
        return sum(1 for v in vs if a[v.name])

    assert run("at_most", +1) == 2, "a ceiling binds against a reward pushing past it"
    assert run("at_most", -1) == 0, "and is satisfied by taking none"
    assert run("exactly", -1) == 2, "where an equality still has to take two"
    assert run("at_least", +1) == 4, "a floor does not forbid taking more"
    assert run("at_least", -1) == 2, "and holds against a reward pushing below it"


def test_a_counting_constraint_can_be_any_length():
    """Nine shifts, at most two taken. The old binding capped this at four."""
    p = ft.Problem()
    sh = [p.binary(f"s{i}") for i in range(9)]
    p.at_most(sh, 2)
    p.maximize(sum((9 - i) * s.is_(1) for i, s in enumerate(sh)))
    a = p.solve()
    assert a.feasible, a
    assert sum(1 for s in sh if a[s.name]) == 2, a


def test_literals_in_one_constraint_may_name_different_values():
    """"At most one of a=3 and b=17" is not sayable with a single shared value."""
    p = ft.Problem()
    x = p.categorical("x", 4)
    y = p.integer("y", 10, 20)
    p.at_most([x.is_(3), y.is_(17)], 1)
    p.maximize(5 * x.is_(3) + 4 * y.is_(17))
    a = p.solve()
    assert a.feasible, a
    assert (a["x"] == 3) + (a["y"] == 17) <= 1, a
    assert a["x"] == 3, "and it keeps the more valuable of the two"


def test_exactly_one_and_at_most_one():
    def run(method, sign):
        p = ft.Problem()
        v = [p.binary(f"v{i}") for i in range(5)]
        getattr(p, method)(v)
        p.maximize(sum(sign * s.is_(1) for s in v))
        a = p.solve()
        assert a.feasible, a
        return sum(1 for s in v if a[s.name]), a.spins

    assert run("exactly_one", -1)[0] == 1, "one, even pushed off"
    assert run("at_most_one", -1)[0] == 0, "none, when pushed off"
    assert run("at_most_one", +1)[0] == 1, "and one when pulled on"

    # neither pays for a slack variable
    p = ft.Problem()
    v = [p.binary(f"v{i}") for i in range(5)]
    p.at_most(v, 1)
    assert p.solve().spins > run("at_most_one", -1)[1], "at_most k=1 costs slack; at_most_one does not"


def test_a_counting_constraint_refuses_what_it_cannot_count():
    p = ft.Problem()
    x = p.categorical("x", 3)
    with pytest.raises(ValueError, match="at least two"):
        p.at_most([x], 1)
    with pytest.raises(ValueError, match="k must be"):
        p.at_most([x, p.categorical("y", 3)], 9)
    with pytest.raises(TypeError, match="variables or literals"):
        p.at_most([x, "y"], 1)


def test_slack_costs_spins_but_never_appears_in_the_answer():
    def spins(kind):
        p = ft.Problem()
        vs = [p.binary(f"v{i}") for i in range(4)]
        getattr(p, kind)(vs, 2)
        a = p.solve()
        assert set(a.values) == {f"v{i}" for i in range(4)}, "no slack in the answer"
        return a.spins

    assert spins("at_most") > spins("exactly"), "an inequality needs a slack variable"


def test_a_quadratic_term_rewards_agreement():
    p = ft.Problem()
    a, b = p.categorical("a", 3), p.categorical("b", 3)
    p.maximize(4 * (a.is_(2) * b.is_(2)))
    ans = p.solve()
    assert (ans["a"], ans["b"]) == (2, 2), ans


def test_a_constraint_cannot_be_outbid_by_the_objective():
    """The penalty scales above the largest objective coefficient, or a constraint is decorative."""
    p = ft.Problem()
    a, b = p.categorical("a", 3), p.categorical("b", 3)
    p.not_equal(a, b)
    p.maximize(50 * a.is_(1) + 50 * b.is_(1))  # both want the same value
    ans = p.solve()
    assert ans.feasible, ans
    assert ans["a"] != ans["b"], ans
    assert p.solve().penalty >= 100, "the penalty must outrank the objective"


def test_feasible_means_the_constraints_hold():
    """A penalty makes a constraint expensive, not impossible.

    Pin it below the objective and the sampler pays it: every variable decodes cleanly and the
    constraint is broken. `feasible` used to be true for exactly that answer.
    """
    p = ft.Problem()
    a, b = p.categorical("a", 3), p.categorical("b", 3)
    p.not_equal(a, b)
    p.penalty(1.0)
    p.maximize(40 * a.is_(1) + 40 * b.is_(1))
    ans = p.solve()
    assert (ans["a"], ans["b"]) == (1, 1), ans
    assert ans.undecoded == [], "every variable decoded perfectly"
    assert not ans.feasible, "and it is still not feasible"
    assert len(ans.violated) == 1, ans.violated
    assert "must differ" in ans.violated[0].detail, ans.violated[0]
    assert ans.violated[0].by > 0, "and by how much, not only that it broke"


def test_raising_the_penalty_wins_a_constraint_back():
    def run(pin):
        p = ft.Problem()
        a, b = p.categorical("a", 3), p.categorical("b", 3)
        p.not_equal(a, b)
        if pin:
            p.penalty(pin)
        p.maximize(40 * a.is_(1) + 40 * b.is_(1))
        return p.solve()

    assert not run(1.0).feasible, "outbid"
    assert run(200.0).feasible, "and won back by raising it"
    assert run(None).feasible, "the automatic scaling already handles this one"

    for bad in (0.0, -1.0, float("nan")):
        p = ft.Problem()
        p.categorical("x", 3)
        with pytest.raises(ValueError, match="positive number"):
            p.penalty(bad)


def test_a_certificate_reports_on_the_sampler_not_the_answer():
    p = ft.Problem()
    x = p.categorical("x", 4)
    p.fix(x, 2)
    p.solve()
    c = p.certify(beta=1.0, draws=512)
    assert c.beta_eff == c.beta_eff, "beta_eff is a real number"
    assert c.ess > 0, c
    assert isinstance(c.findings, list)
    assert c.passed == (not c.findings), "passed is exactly an empty findings list"
    if c.tv is not None:
        assert c.noise_floor is not None, "a TV without its floor is not a measurement"


def test_a_grid_of_variables_models_an_assignment_problem():
    """The shape most real models have, and the one that needs an index to write at all."""
    workers, shifts = 3, 3
    p = ft.Problem()
    a = p.grid("assign", (workers, shifts))
    for w in range(workers):
        p.exactly_one(a.row(w))
    for s in range(shifts):
        p.at_most_one(a.column(s))
    p.maximize(3 * a[0, 2].is_(1) + 3 * a[1, 0].is_(1))

    ans = p.solve(tries=32)
    assert ans.feasible, ans
    for w in range(workers):
        taken = [s for s in range(shifts) if ans[f"assign[{w},{s}]"]]
        assert len(taken) == 1, f"worker {w} takes exactly one shift: {taken}"
    assert ans["assign[0,2]"] == 1, ans
    assert ans["assign[1,0]"] == 1, ans


def test_a_grid_index_outside_the_shape_is_refused_not_wrapped():
    """Python wraps a negative index. A wrap here is an off-by-one that reaches the answer."""
    p = ft.Problem()
    a = p.grid("x", (2, 3))
    assert len(a) == 6
    assert a.dims == (2, 3)
    assert a[1, 2].name == "x[1,2]"

    with pytest.raises(IndexError, match="outside dimension"):
        a[0, 3]
    with pytest.raises(IndexError, match="outside dimension"):
        a[-1, 0]
    with pytest.raises(IndexError, match="dimensions and was given"):
        a[1]

    # rows and columns pick out what a constraint is usually over
    assert [v.name for v in a.row(1)] == ["x[1,0]", "x[1,1]", "x[1,2]"]
    assert [v.name for v in a.column(2)] == ["x[0,2]", "x[1,2]"]


def test_a_grid_takes_any_domain():
    p = ft.Problem()
    t = p.grid("temp", (3,), lambda pr, n: pr.integer(n, 10, 20))
    p.fix(t[1], 17)
    assert p.solve()["temp[1]"] == 17


def test_an_encoding_can_be_chosen_and_costs_what_it_says():
    """The trade is the difference between a model that fits a machine and one that does not."""
    def spins(enc):
        p = ft.Problem()
        p.categorical("a", 6, encoding=enc)
        return p.solve(tries=4).spins

    assert spins("one-hot") == 6, "one spin per value"
    assert spins("domain-wall") == 5, "one fewer"
    assert spins("binary") == 3, "log2 of the domain, and the cheapest by far"

    # the two that can carry a literal both do, and decode to what they were fixed to
    for enc in ("one-hot", "domain-wall"):
        p = ft.Problem()
        a = p.categorical("a", 6, encoding=enc)
        p.fix(a, 3)
        ans = p.solve(tries=16)
        assert ans.feasible, (enc, ans)
        assert ans["a"] == 3, (enc, ans)

    # an integer is an ORDERED domain, where domain-wall is often the better choice
    p = ft.Problem()
    t = p.integer("t", 10, 20, encoding="domain-wall")
    p.fix(t, 17)
    ans = p.solve(tries=16)
    assert ans["t"] == 17, ans
    assert ans.spins == 10, "eleven values in ten spins"

    with pytest.raises(ValueError, match="unknown encoding"):
        ft.Problem().categorical("x", 3, encoding="rot13")


def test_a_binary_encoded_variable_cannot_carry_a_literal():
    """Its indicator is a product of every bit, so its degree grows with the domain."""
    p = ft.Problem()
    a = p.categorical("a", 8, encoding="binary")
    p.fix(a, 3)
    with pytest.raises(ValueError, match="OneHot|one-hot"):
        p.solve()


def test_errors_name_what_the_caller_wrote():
    p = ft.Problem()
    x = p.categorical("colour", 3)
    with pytest.raises(ValueError, match="already declared"):
        p.categorical("colour", 3)
    with pytest.raises(ValueError, match="colour"):
        p.fix(x, 9)
    with pytest.raises(ValueError, match="at least two"):
        p.at_most([x], 1)
    with pytest.raises(TypeError, match="not a bare number"):
        p.maximize(5)


def test_a_bad_annealing_ladder_is_refused_rather_than_defaulted():
    p = ft.Problem()
    x = p.categorical("x", 3)
    p.fix(x, 1)
    assert p.solve(beta_hot=0.05, beta_cold=6.0, stages=40, sweeps=20)["x"] == 1
    assert p.solve()["x"] == 1, "zeros mean the library's own ladder"
    with pytest.raises(ValueError, match="beta_cold must exceed"):
        p.solve(beta_hot=8.0, beta_cold=0.05)
    with pytest.raises(ValueError, match="beta_cold must exceed"):
        p.solve(beta_hot=float("nan"), beta_cold=6.0)


def test_summing_terms_in_a_loop_works():
    """`sum()` starts from 0, which is the natural Python idiom and must not be a type error."""
    p = ft.Problem()
    x = p.categorical("x", 6)
    p.maximize(sum(v * x.is_(v) for v in range(6)))
    assert p.solve()["x"] == 5


def test_the_compiled_program_is_readable():
    p = ft.Problem()
    a, b = p.categorical("a", 3), p.categorical("b", 3)
    p.not_equal(a, b)
    p.solve()
    text = p.ftp()
    assert text.startswith("ftp 1"), text[:40]
    assert "spins 6" in text, text[:80]


def test_a_preference_is_traded_and_a_rule_is_not():
    # The same model twice, differing only in whether the constraint is a rule or a price. What
    # changes is not whether the solver CAN break it -- a penalty was always breakable -- but what
    # the answer MEANS when it does. A broken rule makes the answer no answer; a traded preference
    # is the choice the modeller asked the solver to price, and the answer stays feasible.
    def build(**kw):
        p = ft.Problem()
        a, b = p.categorical("a", 2), p.categorical("b", 2)
        p.not_equal(a, b, **kw)
        p.maximize(5 * a.is_(0) + 5 * b.is_(0))
        return p.solve(tries=24)

    cheap = build(soft=1.0)
    assert cheap.feasible, cheap
    assert len(cheap.violated) == 1 and not cheap.violated[0].hard, cheap
    assert cheap.soft_cost == 1.0, cheap
    assert "traded" in str(cheap), str(cheap)

    dear = build(soft=50.0)
    assert dear.violated == [], dear
    assert dear.soft_cost == 0.0
    # A price of nothing must not print with a minus sign in front of it: Rust's f64 sum folds from
    # -0.0, and "-0" as a cost reads as a credit.
    assert not math.copysign(1.0, dear.soft_cost) < 0, repr(dear.soft_cost)

    rule = build()
    assert rule.feasible and rule.violated == [], rule
    assert rule.soft_cost == 0.0


def test_a_soft_price_is_squared_because_the_penalty_is():
    p = ft.Problem()
    vs = [p.binary(f"v{i}") for i in range(4)]
    p.at_most(vs, 1, soft=1.0)
    p.maximize(sum(20 * v.is_(1) for v in vs))
    a = p.solve(tries=24)
    assert a.feasible, a
    # All four held against a cap of one, so it is over by three -- and three squared is nine, not
    # three. Missing by two costs four times missing by one, and a linear price here would misstate
    # what the solver traded.
    assert a.soft_cost == 9.0, a


def test_a_state_computed_elsewhere_is_scored_by_the_same_code_or_refused():
    # The point of putting a state in is that whatever produced it -- a GPU sweep, another solver --
    # is then judged by the code that judges this library's own answers. That only means anything if
    # the state arrives intact, so the refusals matter more than the success.
    m = ft.Model(4)
    for i in range(4):
        m.couple(i, (i + 1) % 4, 1.0)
    sim = m.build(beta=1.0, seed=7)

    sim.spins = [1, 1, 1, 1]
    # A ferromagnetic ring, every bond satisfied: -1 per bond over four bonds.
    assert sim.energy == -4.0

    # Short, and a value that is not a spin. Both are trivially launderable -- pad with -1, coerce
    # with `v > 0` -- and a laundered state is then scored with full confidence.
    with pytest.raises(ValueError, match="4 nodes"):
        sim.spins = [1, 1, 1]
    with pytest.raises(ValueError, match=r"-1/\+1"):
        sim.spins = [1, 0, 1, 1]
    assert sim.energy == -4.0, "a refused write must not half-apply"


def test_the_ancilla_count_is_readable_because_sampling_a_reduced_model_is_not_sound():
    p = ft.Problem()
    a, b, c = (p.binary(n) for n in "abc")
    p.maximize(3 * a.is_(1) * b.is_(1) * c.is_(1))
    ans = p.solve(tries=8)
    # Three variables in one term is a three-body statement, lowered with one ancilla. Without a way
    # to READ that, a caller cannot tell a model whose Boltzmann distribution over the original
    # variables is preserved from one whose is not.
    assert ans.ancillas == 1, ans
    # Seven spins, not four: a binary is one-hot over two values, so each costs two, and the ancilla
    # is the seventh. The spin total alone cannot say whether any were added by the lowering.
    assert ans.spins == 7, ans


def test_the_exact_ground_state_is_readable_not_only_its_energy():
    m = ft.Model(6)
    for i in range(5):
        m.couple(i, i + 1, 1.0)
    sim = m.build(beta=1.0, seed=1)
    state = sim.exact_ground_state()
    # A ferromagnetic chain: every spin agrees, and the energy is one per bond.
    assert state is not None and len(set(state)) == 1, state
    assert sim.exact_ground_energy() == -5.0


def test_all_different_solves_a_latin_square_row_and_names_the_clash():
    p = ft.Problem()
    v = [p.categorical(f"c{i}", 4) for i in range(4)]
    p.all_different(v)
    a = p.solve(tries=60)
    assert a.feasible, a
    assert sorted(a[f"c{i}"] for i in range(4)) == [0, 1, 2, 3], a


def test_an_impossible_all_different_is_refused_rather_than_annealed():
    # Five variables over three values. No penalty makes this satisfiable, so reporting
    # feasible=False after a full anneal would send a modeller looking for a longer ladder that
    # cannot help. The pigeonhole principle is countable, so it is counted.
    p = ft.Problem()
    xs = [p.categorical(f"x{i}", 3) for i in range(5)]
    p.all_different(xs)
    with pytest.raises(ValueError, match="No assignment can satisfy"):
        p.solve(tries=8)


def test_all_different_over_disjoint_domains_is_free():
    # Two variables that cannot collide need no terms. Lowering per shared value notices that;
    # a pairwise not_equal sweep would emit them anyway.
    a = ft.Problem(); x = a.integer("a", 0, 3); y = a.integer("b", 10, 13)
    a.all_different([x, y])
    b = ft.Problem(); b.integer("a", 0, 3); b.integer("b", 10, 13)
    assert a.solve(tries=4).spins == b.solve(tries=4).spins


def test_an_encoding_that_cannot_be_exact_is_reported_not_hidden():
    # A binary encoding of 6 values uses 3 spins, spelling 8 codewords; the 2 spare ones decode to
    # nothing and cost exactly what a valid state costs, so nothing discourages the sampler from
    # landing on one. The compiler knows this before any sampling happens.
    p = ft.Problem()
    p.categorical("x", 6, encoding="binary")
    p.categorical("y", 8, encoding="binary")   # a power of two IS exact
    p.categorical("z", 6)                      # one-hot is always exact
    a = p.solve(tries=4)
    assert len(a.caveats) == 1, a.caveats
    assert "'x'" in a.caveats[0]
    assert "one-hot" in a.caveats[0] or "power of two" in a.caveats[0]
    assert "caveat:" in str(a)


def test_an_exact_model_carries_no_caveats():
    p = ft.Problem()
    p.categorical("a", 5)
    p.integer("b", 0, 7)
    assert p.solve(tries=4).caveats == []


# ---- solvers and bounds ----------------------------------------------------------------------
#
# The C ABI could build a graph and sample it, but not ask how far from optimal the sample was --
# so a Python user could do the easy half of what this library is for. These check the other half
# crossed the boundary intact, rather than that the symbols merely resolve.


def _frustrated_chain(n: int = 12) -> "ft.Model":
    """A ring with one flipped bond: enumerable, and genuinely frustrated."""
    m = ft.Model(n)
    for i in range(n):
        m.couple(i, (i + 1) % n, -1.0 if i == 0 else 1.0)
    return m


def test_every_solver_leaves_the_state_whose_energy_it_reported():
    """The number returned is a claim about ``spins``, not a separate answer."""
    for name, run in [
        ("tabu", lambda s: s.tabu(iterations=5_000)),
        ("breakout", lambda s: s.breakout(iterations=5_000).energy),
        ("population_anneal", lambda s: s.population_anneal(population=64, stages=20).energy),
        ("branch", lambda s: s.branch(max_nodes=2_000_000).energy),
    ]:
        sim = _frustrated_chain().build(beta=1.0, seed=3)
        reported = float(run(sim))
        assert math.isclose(reported, sim.energy, abs_tol=1e-9), (
            f"{name} returned {reported}, the state it left has {sim.energy}"
        )
        assert set(sim.spins) <= {-1, 1}, "`spins` is a property here, not a method"


def test_branch_and_bound_reports_a_proof_and_withholds_one():
    """``proved`` is the whole product. A search that gave up must not claim one."""
    sim = _frustrated_chain().build(beta=1.0, seed=1)
    done = sim.branch()
    assert done.proved and done.nodes > 0
    # A frustrated ring of n bonds can satisfy all but one: -(n - 1) + 1.
    assert math.isclose(done.energy, -10.0, abs_tol=1e-9), done

    big = ft.Model(40)
    for i in range(40):
        for j in range(i + 1, 40):
            big.couple(i, j, 1.0 if (i * 7 + j) % 3 else -1.0)
    out = big.build(beta=1.0, seed=1).branch(max_nodes=200)
    assert not out.proved and out.nodes <= 201, out


def test_tabu_reports_how_far_it_actually_got():
    """Truncation is invisible from outside without this, which is how it once shipped."""
    sim = _frustrated_chain().build(beta=1.0, seed=2)
    assert sim.tabu_iterations() == 0, "nothing has run yet"
    sim.tabu(iterations=3_000, restart_after=0)
    assert sim.tabu_iterations() == 3_000


def test_population_annealing_hands_over_its_warning_with_its_free_energy():
    sim = _frustrated_chain().build(beta=1.0, seed=4)
    run = sim.population_anneal(population=256, sweeps=2, beta_max=3.0, stages=30)
    # Z(0) = 2 ** n and Z is non-decreasing in beta, so ln Z is at least n ln 2 at any beta.
    assert run.ln_z is not None and run.ln_z >= 12 * math.log(2) - 1e-9, run
    assert 1.0 <= run.rho <= run.population
    assert isinstance(run.trustworthy, bool)
    f = run.free_energy(3.0, 12)
    assert f is not None and math.isclose(f, -run.ln_z / (3.0 * 12))
    # A one-step quench collapses the population onto one ancestor, and rho has to say so.
    quenched = sim.population_anneal(population=64, sweeps=1, beta_max=40.0, stages=1)
    assert quenched.rho > run.rho, (quenched, run)


def test_no_bound_exceeds_a_ground_energy_the_same_object_can_prove():
    """One-sided on purpose: a bound may be loose by any amount and may never exceed the optimum."""
    sim = _frustrated_chain().build(beta=1.0, seed=5)
    truth = sim.branch().energy
    b = sim.bounds()
    for name in ("decoupled", "forest", "odd_cycle", "sdp"):
        v = getattr(b, name)
        if v is None:
            continue
        assert v <= truth + 1e-9, f"{name} bound {v} exceeds the proved minimum {truth}"
    assert b.best <= truth + 1e-9
    assert b.which in {"decoupled", "forest", "odd_cycle", "sdp"}
    # On a ring with no fields, `forest` cannot beat `decoupled`: a tree is never frustrated.
    assert math.isclose(b.forest, b.decoupled, abs_tol=1e-9), (b.forest, b.decoupled)


def test_the_gap_is_zero_exactly_when_the_state_is_provably_optimal():
    sim = _frustrated_chain().build(beta=1.0, seed=6)
    sim.branch()  # leaves the proved optimum as the state
    assert sim.gap() >= -1e-9, "a gap below zero would mean a bound above the optimum"
    loose = _frustrated_chain().build(beta=1.0, seed=7)
    loose.spins = [1] * 12  # every bond satisfied except the frustrated one is NOT this state
    assert loose.gap() > sim.gap() - 1e-9


def test_breakout_reports_the_evidence_that_it_broke_out():
    """The claim BLS makes is about what happens BETWEEN local optima."""
    sim = _frustrated_chain(20).build(beta=1.0, seed=9)
    r = sim.breakout(iterations=20_000)
    assert r.iterations_run == 20_000
    assert r.descents > 1, "a run with one descent never left its first basin"
    assert r.max_jump >= 1
    assert math.isclose(r.energy, sim.energy, abs_tol=1e-9)
    # And it must not do worse than the descent it is built on.
    assert r.energy <= sim.bounds().best + abs(sim.bounds().best) + 1e-9


def test_exact_planar_returns_a_maximum_and_refuses_with_a_reason():
    """The one solver here that returns an optimum, and four refusals that are four instructions."""
    m = ft.Model(16)
    for y in range(4):
        for x in range(4):
            i = y * 4 + x
            if x + 1 < 4:
                m.couple(i, i + 1, -1.0)
            if y + 1 < 4:
                m.couple(i, i + 4, -1.0)
    sim = m.build(beta=1.0, seed=1)
    r = sim.exact_planar()
    # A bipartite grid: every one of its 24 edges is cut.
    assert r.cut == 24.0 and r.energy == -24.0
    assert r.faces == 10
    assert sim.energy == -24.0, "the state left behind is the optimum"

    # No search may beat an exact answer. That is what makes the word mean something.
    other = _frustrated_chain(12).build(beta=1.0, seed=1)
    exact = other.exact_planar()
    assert other.breakout(iterations=20_000).energy >= exact.energy - 1e-9

    # A periodic lattice is a torus, and the reduction is a plane statement.
    torus = ft.lattice2d(4, j=1.0, beta=1.0, seed=1)
    with pytest.raises(ValueError, match="not planar"):
        torus.exact_planar()


def test_the_toroidal_bound_is_the_other_end_of_the_bracket():
    """G-set publishes lower bounds. This is the upper one, and it must never be beatable."""
    torus = ft.lattice2d(6, j=-1.0, beta=1.0, seed=1)
    b = torus.toroidal_bound()
    # A 6x6 periodic lattice is bipartite: all 72 edges cut, and the bound is achieved.
    assert b.cut == 72.0 and b.attained

    # The planar solver declines the same graph. That distinction is the whole point.
    with pytest.raises(ValueError, match="not planar"):
        torus.exact_planar()

    # A frustrated torus: no search may exceed the bound.
    hard = ft.lattice2d(5, j=1.0, beta=1.0, seed=1)
    hb = hard.toroidal_bound()
    e = hard.breakout(iterations=100_000).energy
    cut = (-50.0 - e) / 2.0     # W = sum of -J over 50 edges of J = +1
    assert cut <= hb.cut + 1e-9, f"a search reached {cut}, above the bound {hb.cut}"


def test_the_closed_gaps_carry_their_own_caveats():
    """Three algorithms the toolchain survey named as missing, and the caveat each must carry."""
    # A 6x6 ANTIferromagnet is bipartite and inside the GW hypothesis: 72 cuttable edges.
    anti = ft.lattice2d(6, j=-1.0, beta=1.0, seed=3)
    r = anti.goemans_williamson(hyperplanes=64, seed=5)
    assert r.guaranteed and r.cut == 72.0
    assert anti.energy == -72.0, "the state left behind is the one that cut them"

    # A ferromagnet is OUTSIDE it. A guarantee that is always claimed is not a guarantee.
    ferro = ft.lattice2d(6, j=1.0, beta=1.0, seed=3)
    assert not ferro.goemans_williamson(hyperplanes=16, seed=5).guaranteed

    # Cluster moves fire and find the ground state; quantum annealing does too.
    c = ferro.cluster_anneal(rungs=8, rounds=200, beta_min=0.1, beta_max=4.0)
    assert abs(c.energy + 72.0) < 1e-9 and c.moves > 0
    assert abs(ferro.quantum_anneal(trotter=4, steps=200) + 72.0) < 1e-9

    # A field breaks the isoenergetic argument, and it is refused rather than accepted.
    m = ft.Model(8)
    for i in range(8):
        m.couple(i, (i + 1) % 8, 1.0)
    m.bias(3, 0.5)
    with pytest.raises(ValueError, match="h = 0"):
        m.build(beta=1.0, seed=1).cluster_anneal()


# ---- higher-order models -------------------------------------------------------------------------
#
# The other route to a k-body term is an objective product on Problem, which quadratises it with
# ancillas. These check the NATIVE path: no ancillas, and the same numbers every other binding gets.


def test_a_three_body_term_is_solved_without_ancillas():
    h = ft.Hubo(3)
    h.add([0, 1, 2], 1.0)
    e = h.anneal(seed=7)
    assert e == -1.0, e
    s = h.state
    assert s[0] * s[1] * s[2] == 1, s
    assert h.terms == 1
    assert h.max_arity == 3
    # The ceiling, not the cost: one substitution for one three-body term.
    assert h.ancillas_avoided == 1
    assert h.proposals > 0, "a run that proposed nothing is not a run"


def test_a_repeated_variable_is_refused_rather_than_changing_the_order():
    h = ft.Hubo(4)
    # s * s = 1, so [0, 0, 1] is a one-body term wearing a three-body's clothes.
    with pytest.raises(ValueError, match="already in this term"):
        h.add([0, 0, 1], 1.0)
    with pytest.raises(ValueError, match="no variable 9"):
        h.add([0, 9], 1.0)
    with pytest.raises(ValueError, match="finite"):
        h.add([0, 1], float("nan"))
    assert h.terms == 0, "nothing malformed was recorded"
    # And a refused term leaves nothing pending for the next one to absorb.
    h.add([0, 1, 2], 1.0)
    assert h.terms == 1 and h.max_arity == 3


def test_the_energy_returned_is_a_claim_about_the_state_left_behind():
    h = ft.Hubo(4)
    h.add([0, 1, 2], 1.5)
    h.add([1, 2, 3], -2.0)
    e = h.anneal(seed=3)
    assert abs(h.energy - e) < 1e-9, (h.energy, e)

    # And the incremental update agrees with recomputing, which is the check another language or a
    # GPU would run against this library.
    for i in range(4):
        before = h.energy
        d = h.delta(i)
        s = h.state
        s[i] = -s[i]
        h.state = s
        assert abs(h.energy - before - d) < 1e-9, (i, d, h.energy - before)
        s[i] = -s[i]
        h.state = s


def test_a_state_is_taken_whole_or_refused_whole():
    h = ft.Hubo(3)
    h.add([0, 1, 2], 1.0)
    h.state = [1, 1, 1]
    assert h.energy == -1.0
    with pytest.raises(ValueError, match="-1 or"):
        h.state = [1, 0, 1]
    assert h.energy == -1.0, "the refused write changed nothing"
    with pytest.raises(ValueError, match="3 spins"):
        h.state = [1, 1]


def test_a_lifted_graph_scores_exactly_as_the_pairwise_path_does():
    # If these disagree, one of them has the sign convention wrong and every later comparison
    # inherits it silently.
    m = ft.Model(5)
    for i in range(4):
        m.couple(i, i + 1, 1.0 if i % 2 == 0 else -1.0)
    m.bias(0, 0.5)
    sim = m.build(beta=0.9, seed=11)
    sim.sweep(20)
    h = ft.Hubo.from_sim(sim)
    assert abs(h.energy - sim.energy) < 1e-9, (h.energy, sim.energy)
    assert h.max_arity == 2, "a lifted pairwise graph is still pairwise"
    assert h.ancillas_avoided == 0, "nothing wider than two needs a substitution"


def test_a_bad_ladder_is_refused_and_a_nan_is_not_read_as_a_default():
    h = ft.Hubo(3)
    h.add([0, 1, 2], 1.0)
    with pytest.raises(ValueError):
        h.anneal(beta_min=8.0, beta_max=0.05)
    with pytest.raises(ValueError):
        h.anneal(beta_min=float("nan"), beta_max=8.0)
    # Zeros DO mean "use the default", which is why NaN has to be refused before that test.
    assert h.anneal(seed=1) == -1.0


# ---- parallel sweeps -------------------------------------------------------------------------
#
# The library has had a threaded chromatic sweep for a long time; it reached Rust and the HTTP API
# and no binding. On an 18-core machine that is 1/18th of the hardware, silently.


def test_a_parallel_sweep_reproduces_itself_at_a_fixed_thread_count():
    def ring(seed):
        m = ft.Model(128)
        for i in range(128):
            m.couple(i, (i + 1) % 128, 1.0)
        return m.build(beta=0.7, seed=seed)

    # The promise is per (seed, threads), not per seed. Asserting only the first would pass on a
    # sampler that ignored `threads` entirely.
    a, b = ring(0x2244), ring(0x2244)
    a.sweep(60, threads=4)
    b.sweep(60, threads=4)
    assert a.spins == b.spins, "same (seed, threads) must reproduce bit-identically"
    assert a.threads_used >= 1


def test_the_thread_count_is_part_of_the_run():
    def ring(seed):
        m = ft.Model(128)
        for i in range(128):
            m.couple(i, (i + 1) % 128, 1.0)
        return m.build(beta=0.7, seed=seed)

    a, b = ring(0x99), ring(0x99)
    a.sweep(60, threads=1)
    b.sweep(60, threads=4)
    assert a.spins != b.spins, "one thread and four are different sample paths, not the same one"


def test_zero_threads_asks_the_machine():
    assert ft.hardware_threads() >= 1
    m = ft.Model(64)
    for i in range(64):
        m.couple(i, (i + 1) % 64, 1.0)
    sim = m.build(beta=0.5, seed=3)
    assert sim.threads_used == 0, "nothing parallel has run yet"
    sim.sweep(20, threads=0)
    assert sim.threads_used == ft.hardware_threads() or sim.threads_used >= 1


def test_a_parallel_sweep_samples_the_same_physics():
    # Onsager is the referee: a sweep that raced would show up as a wrong magnetisation rather
    # than as a crash nobody sees. Started ORDERED, because below the critical point a random
    # start coarsens into domains and stays there far longer than a test will wait.
    L, beta = 32, 0.6
    for threads in (1, 4):
        m = ft.Model(L * L)
        for y in range(L):
            for x in range(L):
                i = y * L + x
                m.couple(i, y * L + (x + 1) % L, 1.0)
                m.couple(i, ((y + 1) % L) * L + x, 1.0)
        sim = m.build(beta=beta, seed=0x9A7)
        sim.spins = [1] * (L * L)
        sim.sweep(1500, threads=threads)
        acc = 0.0
        for _ in range(300):
            sim.sweep(1, threads=threads)
            acc += abs(sim.magnetization)
        got = acc / 300
        want = ft.onsager(beta)
        assert abs(got - want) < 0.03, f"threads={threads}: |M| {got:.4f} vs Onsager {want:.4f}"


def test_an_answer_is_scored_in_the_modellers_own_units():
    # `energy` is the compiled Ising energy with every penalty and the constant folded in, and it
    # was the only number an answer carried. A person cannot read what their schedule is worth out
    # of it, cannot compare two answers by it, and it moves when the penalty does.
    p = ft.Problem()
    mon = p.categorical("mon", 3)
    tue = p.categorical("tue", 3)
    p.not_equal(mon, tue)
    p.maximize(5 * mon.is_(1) + 4 * tue.is_(2))
    a = p.solve(tries=64)
    assert a.feasible, a
    assert a.objective == 9.0, (a.values, a.objective)
    assert a.objective != a.energy, "if these agree the test is measuring nothing"


def test_no_objective_reports_none_rather_than_zero():
    # Zero would read as "worth nothing" instead of "not asked".
    p = ft.Problem()
    v = p.categorical("v", 2)
    p.fix(v, 1)
    assert p.solve(tries=4).objective is None


# ---- the model layer can prove -------------------------------------------------------------------
#
# Every solve on the modelling layer was an anneal, so tabu, breakout and branch-with-proof were
# reachable only from a spin graph. The layer every document says to reach for first was the one
# that could not certify anything.


def test_the_model_layer_can_prove_an_answer_optimal():
    def problem():
        p = ft.Problem()
        a = p.categorical("a", 3)
        b = p.categorical("b", 3)
        p.not_equal(a, b)
        p.maximize(5 * a.is_(1) + 4 * b.is_(2))
        return p

    proved = problem().solve(method="branch", effort=5_000_000)
    assert proved.proved_optimal, proved
    assert proved.feasible
    # a != b permits a = 1 and b = 2, so the optimum in the modeller's units is 9.
    assert proved.objective == 9.0, proved.values

    # And nothing else claims a proof, whatever it finds.
    for method in ("anneal", "tabu", "breakout"):
        ans = problem().solve(method=method, effort=50_000)
        assert not ans.proved_optimal, method
        assert ans.feasible, method


def test_an_unknown_method_is_refused_by_name():
    p = ft.Problem()
    v = p.categorical("v", 2)
    p.fix(v, 1)
    with pytest.raises(ValueError, match="unknown method"):
        p.solve(method="magic")


def test_a_weighted_linear_row_is_a_constraint_not_a_preference():
    """`3a + 4b + 5c <= 7`, which no counting constraint can say.

    Every counting form counts UNWEIGHTED literals, so a weighted row could not be stated here at
    all -- and the advice the LP reader used to give, "add it to the objective", is the defect
    rather than the workaround: an objective term is not a constraint, so `feasible` and
    `violated` stop knowing about the row.
    """
    p = ft.Problem()
    a, b, c = (p.binary(n) for n in "abc")
    p.linear(3 * a.is_(1) + 4 * b.is_(1) + 5 * c.is_(1), "<=", 7)
    p.maximize(a.is_(1) + b.is_(1) + c.is_(1))
    ans = p.solve()
    assert ans.feasible, ans
    assert 3 * ans["a"] + 4 * ans["b"] + 5 * ans["c"] <= 7, ans
    assert ans.objective == 2, "3 + 4 = 7 fits and nothing better does: %s" % (ans,)

    # The pair form says the same thing, and a bare variable means "it holds".
    q = ft.Problem()
    a, b, c = (q.binary(n) for n in "abc")
    q.linear([(a, 3), (b, 4), (c, 5)], "<=", 7)
    q.maximize(a.is_(1) + b.is_(1) + c.is_(1))
    assert q.solve().objective == 2


def test_a_weighted_row_refuses_what_it_cannot_represent():
    # A non-integer coefficient on an INEQUALITY: there is no integer residual for the slack.
    p = ft.Problem()
    a, b = p.binary("a"), p.binary("b")
    p.linear([(a, 2.5), (b, 1)], "<=", 4)
    with pytest.raises(ValueError) as e:
        p.solve()
    assert "common denominator" in str(e.value)

    # A row nothing can satisfy is refused by arithmetic rather than annealed.
    q = ft.Problem()
    a, b = q.binary("a"), q.binary("b")
    q.linear([(a, 3), (b, 4)], ">=", 9)
    with pytest.raises(ValueError) as e:
        q.solve()
    assert "no answer" in str(e.value)

    # And a relation nobody defined is refused before anything is built.
    r = ft.Problem()
    a = r.binary("a")
    with pytest.raises(ValueError):
        r.linear([(a, 1)], "<", 1)


def test_a_soft_weighted_row_is_priced_in_the_modellers_own_units():
    """The identity that makes a soft row readable: cost == weight x amount squared."""
    p = ft.Problem()
    a, b = p.binary("a"), p.binary("b")
    p.linear([(a, 3), (b, 4)], "<=", 3, soft=0.5)
    p.maximize(10 * a.is_(1) + 10 * b.is_(1))
    ans = p.solve()
    assert ans.feasible, "a soft row leaves the answer an answer"
    assert (ans["a"], ans["b"]) == (1, 1), ans
    # 3 + 4 = 7 against a bound of 3 is 4 over, priced at 0.5 x 4^2 = 8, against the 10 that
    # taking the second one is worth.
    assert ans.soft_cost == 8.0, ans
    assert any("left side comes to 7" in str(v) for v in ans.violated), ans.violated
