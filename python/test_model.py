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
