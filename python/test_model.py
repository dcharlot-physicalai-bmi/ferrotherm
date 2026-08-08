"""Tests for the modelling layer.

Run from the repository root, after `cargo build --release`:

    python3 -m pytest python/test_model.py -q

Each test names a way this layer can be wrong without looking wrong. Sampling is stochastic, so
nothing here asserts on a particular random draw: every assertion is about a property the answer
must have for any correct solver.
"""

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
    assert "must differ" in ans.violated[0], ans.violated[0]


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
