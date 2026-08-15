"""Thermodynamic sampling from Python.

A thin, honest binding over the ferrotherm C ABI. There is no build step and no compiler
requirement: this loads the shared library through ctypes, so ``pip install`` never has to
compile anything on your machine.

    >>> import ferrotherm as ft
    >>> sim = ft.lattice2d(32, beta=0.44)
    >>> sim.sweep(500)
    >>> sim.magnetization           # doctest: +SKIP
    0.83...

Models you define yourself use the same shape the HTTP API and the MCP tools take, so a model
written here can be sent to a server or handed to an agent without translation::

    sim = ft.from_spec({"graph": {"n": 5, "couplings": [[0,1,-1],[1,2,-1],[2,3,-1],[3,4,-1],[4,0,-1]]}})
    sim.anneal()                    # -3.0, the frustrated optimum

Conventions match the rest of the ecosystem: states are -1/+1, energy is
``-sum_ij J_ij s_i s_j - sum_i h_i s_i``, and ``beta`` is inverse temperature.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import c_double, c_int8, c_uint32, c_uint64, POINTER
from typing import Any, Iterable, Sequence

__all__ = [
    "Answer",
    "Certificate",
    "Literal",
    "Model",
    "Problem",
    "Grid",
    "Term",
    "Variable",
    "Violation",
    "Sim",
    "frustrated",
    "wishart",
    "lattice2d",
    "ring",
    "z1_grid",
    "from_spec",
    "onsager",
    "library_path",
    "__version__",
]

# Tracks the native library it binds, because they are released together out of one repository and
# a binding whose version says nothing about the library underneath it is a version nobody can use.
__version__ = "0.8.0"


# ---- library loading ------------------------------------------------------------------------


def _candidates() -> list[str]:
    """Where the shared library might be, most specific first."""
    names = {
        "darwin": "libferrotherm.dylib",
        "win32": "ferrotherm.dll",
    }
    name = names.get(sys.platform, "libferrotherm.so")
    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    env = os.environ.get("FERROTHERM_LIB")
    if env:
        out.append(env)
    out.append(os.path.join(here, name))
    # a checkout being developed against, rather than an installed package
    root = os.path.abspath(os.path.join(here, "..", ".."))
    out.append(os.path.join(root, "target", "release", name))
    out.append(os.path.join(root, "target", "debug", name))
    out.append(name)  # let the loader search the system paths
    return out


def _load() -> ctypes.CDLL:
    tried = []
    for path in _candidates():
        try:
            return ctypes.CDLL(path)
        except OSError as exc:  # keep going; report everything if all fail
            tried.append(f"  {path}\n    {exc}")
    raise ImportError(
        "could not load the ferrotherm shared library. Build it with\n"
        "  cargo build --release\n"
        "in the ferrotherm checkout, or set FERROTHERM_LIB to the library path.\n"
        "Tried:\n" + "\n".join(tried)
    )


_lib = _load()


def library_path() -> str:
    """Path of the shared library actually loaded. Useful when a wrong one is on the path."""
    return getattr(_lib, "_name", "<unknown>")


def _sig(name: str, restype: Any, argtypes: Sequence[Any]) -> Any:
    fn = getattr(_lib, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


_p = ctypes.c_void_p
_ising2d_new = _sig("ft_ising2d_new", _p, [c_uint32, c_double, c_double, c_uint64])
_z1_new = _sig("ft_z1_new", _p, [c_uint32, c_uint32, c_double, c_double, c_double, c_uint64])
_builder_new = _sig("ft_builder_new", _p, [c_uint32])
_builder_couple = _sig("ft_builder_couple", c_uint32, [_p, c_uint32, c_uint32, c_double])
_builder_bias = _sig("ft_builder_bias", c_uint32, [_p, c_uint32, c_double])
_builder_build = _sig("ft_builder_build", _p, [_p, c_double, c_uint64])
_builder_free = _sig("ft_builder_free", None, [_p])
_sweep = _sig("ft_sweep", c_uint64, [_p, c_uint32])
_anneal = _sig("ft_anneal", c_double, [_p, c_double, c_double, c_uint32, c_uint32])
_set_beta = _sig("ft_set_beta", None, [_p, c_double])
_len = _sig("ft_len", c_uint32, [_p])
_spins = _sig("ft_spins", POINTER(c_int8), [_p])
_energy = _sig("ft_energy", c_double, [_p])
_magnetization = _sig("ft_magnetization", c_double, [_p])
_ledger_updates = _sig("ft_ledger_updates", c_uint64, [_p])
_ledger_joules = _sig("ft_ledger_joules_z1", c_double, [_p])
_onsager = _sig("ft_onsager", c_double, [c_double])
_planted_frustrated = _sig("ft_planted_frustrated", _p, [c_uint32, c_uint32, c_uint64, c_double])
_planted_wishart = _sig("ft_planted_wishart", _p, [c_uint32, c_double, c_uint64, c_double])
_ground_energy = _sig("ft_ground_energy", c_double, [_p])
_certify = _sig("ft_certify", c_uint32, [_p, c_uint32, c_uint32])
_cert_passed = _sig("ft_cert_passed", c_uint32, [_p])
_cert_findings = _sig("ft_cert_findings", c_uint32, [_p])
_cert_finding = _sig("ft_cert_finding", c_uint32, [_p, c_uint32, ctypes.POINTER(ctypes.c_ubyte), c_uint32])
_cert_f = {n: _sig("ft_cert_" + n, c_double, [_p]) for n in
           ("beta_eff", "beta_lo", "beta_hi", "tau", "ess", "tv", "floor")}
_exact_ground = _sig("ft_exact_ground", c_double, [_p, c_uint32])
_exact_log_z = _sig("ft_exact_log_z", c_double, [_p, c_double, c_uint32])
_exact_width = _sig("ft_exact_width", c_uint32, [_p])
_free = _sig("ft_free", None, [_p])

# the modelling layer
_u8p = ctypes.POINTER(ctypes.c_ubyte)
_model_new = _sig("ft_model_new", _p, [])
_model_free = _sig("ft_model_free", None, [_p])
_model_categorical = _sig("ft_model_categorical", c_uint32, [_p, c_uint32])
_model_integer = _sig("ft_model_integer", c_uint32, [_p, ctypes.c_int64, ctypes.c_int64])
_model_binary = _sig("ft_model_binary", c_uint32, [_p])
_model_not_equal = _sig("ft_model_not_equal", c_uint32, [_p, c_uint32, c_uint32])
_model_equal = _sig("ft_model_equal", c_uint32, [_p, c_uint32, c_uint32])
_model_fix = _sig("ft_model_fix", c_uint32, [_p, c_uint32, ctypes.c_int64])
_counting_args = [_p, c_uint32, c_uint32, ctypes.c_int64,
                  c_uint32, c_uint32, c_uint32, c_uint32]
_model_cardinality = _sig("ft_model_cardinality", c_uint32, _counting_args)
_model_at_most = _sig("ft_model_at_most", c_uint32, _counting_args)
_model_at_least = _sig("ft_model_at_least", c_uint32, _counting_args)
_model_objective_term = _sig("ft_model_objective_term", c_uint32,
                             [_p, c_uint32, c_double, c_uint32, ctypes.c_int64])
_model_objective_pair = _sig("ft_model_objective_pair", c_uint32,
                             [_p, c_uint32, c_double, c_uint32, ctypes.c_int64,
                              c_uint32, ctypes.c_int64])
_model_compile = _sig("ft_model_compile", c_uint32, [_p])
_model_solve_with = _sig("ft_model_solve_with", c_uint32,
                         [_p, c_uint32, c_double, c_double, c_uint32, c_uint32])
_model_value = _sig("ft_model_value", ctypes.c_int64, [_p, c_uint32])
_model_feasible = _sig("ft_model_feasible", c_uint32, [_p])
_model_energy = _sig("ft_model_energy", c_double, [_p])
_model_penalty = _sig("ft_model_penalty", c_double, [_p])
_model_name = _sig("ft_model_name", c_uint32, [_p, c_uint32, ctypes.c_char_p, c_uint32])
_model_lit = _sig("ft_model_lit", c_uint32, [_p, c_uint32, ctypes.c_int64])
_model_lits_clear = _sig("ft_model_lits_clear", c_uint32, [_p])
_model_objective_product = _sig("ft_model_objective_product", c_uint32, [_p, c_uint32, c_double])
_model_close = _sig("ft_model_close", c_uint32, [_p, c_uint32, c_uint32])
_model_fixed_penalty = _sig("ft_model_fixed_penalty", c_uint32, [_p, c_double])
_model_violations = _sig("ft_model_violations", c_uint32, [_p])
_model_violation = _sig("ft_model_violation", c_uint32, [_p, c_uint32, _u8p, c_uint32])
_model_violation_amount = _sig("ft_model_violation_amount", c_double, [_p, c_uint32])
_model_certify = _sig("ft_model_certify", c_uint32, [_p, c_double, c_uint32, c_uint32])
_model_cert_findings = _sig("ft_model_cert_findings", c_uint32, [_p])
_model_cert_finding = _sig("ft_model_cert_finding", c_uint32, [_p, c_uint32, _u8p, c_uint32])
_model_cert_f = {n: _sig("ft_model_cert_" + n, c_double, [_p])
                 for n in ("beta", "ess", "tau", "tv", "floor")}
_model_error = _sig("ft_model_error", c_uint32, [_p, _u8p, c_uint32])
_model_ftp = _sig("ft_model_ftp", c_uint32, [_p, _u8p, c_uint32])


def _read_text_idx(fn: Any, handle: Any, i: int) -> str:
    """The two-call protocol for getters that also take an index."""
    need = fn(handle, i, None, 0)
    if not need:
        return ""
    buf = (ctypes.c_ubyte * need)()
    got = fn(handle, i, buf, need)
    return bytes(buf[:got]).decode("utf-8", "replace")


def _read_text(fn: Any, handle: Any) -> str:
    """The two-call text protocol: ask how long it is, then ask for it."""
    need = fn(handle, None, 0)
    if not need:
        return ""
    buf = (ctypes.c_ubyte * need)()
    got = fn(handle, buf, need)
    return bytes(buf[:got]).decode("utf-8", "replace")


# ---- building -------------------------------------------------------------------------------


class Certificate:
    """What a run actually did, computed from its samples rather than from its own account.

    ``findings`` is empty exactly when the run is sound. Read that, not ``beta_eff``, to decide
    whether to trust a result.
    """

    __slots__ = ("draws", "beta_eff", "beta_ci", "tau_int", "ess", "tv", "noise_floor", "findings")

    def __init__(self, **kw: Any) -> None:
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def passed(self) -> bool:
        return not self.findings

    def __repr__(self) -> str:
        head = (
            f"<Certificate {'PASSED' if self.passed else 'FAILED'} "
            f"beta={self.beta_eff:.4f} ess={self.ess:.0f}"
        )
        if self.tv is not None and self.tv == self.tv:
            head += f" tv={self.tv:.4f}/floor={self.noise_floor:.4f}"
        if self.findings:
            head += "".join("\n  - " + f for f in self.findings)
        return head + ">"


class Model:
    """A graph under construction: add couplings and biases, then :meth:`build` it.

    Rejected entries raise immediately rather than being silently dropped, because a coupling that
    vanishes without complaint is a model that is quietly wrong.
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("a model needs at least one node")
        self.n = int(n)
        self._h = _builder_new(self.n)
        if not self._h:
            raise RuntimeError("could not allocate a model")

    def couple(self, i: int, j: int, w: float) -> "Model":
        """Add coupling J_ij. Returns self, so calls chain."""
        self._live()
        if _builder_couple(self._h, int(i), int(j), float(w)) == 0:
            raise ValueError(
                f"rejected coupling ({i}, {j}, {w}): indices must be below n={self.n}, "
                f"i must differ from j, and the weight must be finite"
            )
        return self

    def bias(self, i: int, h: float) -> "Model":
        """Add bias h_i. Returns self, so calls chain."""
        self._live()
        if _builder_bias(self._h, int(i), float(h)) == 0:
            raise ValueError(
                f"rejected bias ({i}, {h}): the index must be below n={self.n} and h must be finite"
            )
        return self

    def couple_many(self, edges: Iterable[Sequence[float]]) -> "Model":
        """Add many couplings from an iterable of ``(i, j, J)``."""
        for k, e in enumerate(edges):
            if len(e) != 3:
                raise ValueError(f"coupling {k} must be (i, j, J), got {len(e)} values")
            self.couple(int(e[0]), int(e[1]), float(e[2]))
        return self

    def build(self, beta: float = 1.0, seed: int = 0) -> "Sim":
        """Consume this model into a simulation. The model is spent afterwards."""
        self._live()
        h, self._h = self._h, None
        sim = _builder_build(h, float(beta), int(seed))
        if not sim:
            raise RuntimeError("could not build the simulation")
        return Sim(sim)

    def _live(self) -> None:
        if self._h is None:
            raise RuntimeError("this model was already built and cannot be reused")

    def __del__(self) -> None:
        if getattr(self, "_h", None):
            _builder_free(self._h)
            self._h = None


class Sim:
    """A running simulation. Use the constructors below rather than instantiating directly."""

    def __init__(self, handle: int) -> None:
        self._h = handle

    # -- running --

    def sweep(self, n: int = 1) -> int:
        """Run `n` chromatic block-Gibbs sweeps. Returns total sweeps so far."""
        self._live()
        return int(_sweep(self._h, int(n)))

    def anneal(
        self,
        beta_min: float = 0.05,
        beta_max: float = 4.0,
        stages: int = 60,
        sweeps_per_stage: int = 40,
    ) -> float:
        """Anneal down a geometric ladder, keeping the best state found. Returns its energy."""
        self._live()
        e = float(_anneal(self._h, beta_min, beta_max, stages, sweeps_per_stage))
        if e != e:  # NaN
            raise ValueError(
                "annealing needs 0 < beta_min < beta_max, stages >= 2, sweeps_per_stage >= 1"
            )
        return e

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        self._live()
        _set_beta(self._h, float(value))
        self._beta = float(value)

    # -- reading --

    def __len__(self) -> int:
        self._live()
        return int(_len(self._h))

    @property
    def spins(self) -> list[int]:
        """The state as a list of -1/+1."""
        self._live()
        n = len(self)
        p = _spins(self._h)
        return [int(p[i]) for i in range(n)]

    def numpy(self):
        """The state as a NumPy array.

        This is a **copy**, not a view: the library owns that buffer and may move it on the next
        sweep, so a view would silently alias freed memory.
        """
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise ImportError("numpy is not installed; use .spins for a plain list") from exc
        self._live()
        n = len(self)
        p = _spins(self._h)
        return np.ctypeslib.as_array(p, shape=(n,)).astype(np.int8).copy()

    @property
    def energy(self) -> float:
        self._live()
        return float(_energy(self._h))

    @property
    def magnetization(self) -> float:
        self._live()
        return float(_magnetization(self._h))

    @property
    def node_updates(self) -> int:
        """Node updates charged to the ledger so far."""
        self._live()
        return int(_ledger_updates(self._h))

    @property
    def joules(self) -> float:
        """Ledger priced at Z1-class device figures. This prices the modelled device, not your CPU."""
        self._live()
        return float(_ledger_joules(self._h))

    # -- lifetime --

    def certify(self, draws: int = 512, thin: int = 1) -> "Certificate":
        """Sample and check the result.

        Every commercial machine in this field returns "best found" and nothing else. This returns
        the temperature actually sampled at, how many of the samples were independent, and where
        the model is small enough, the distance from the exact distribution beside the sampling
        noise floor.
        """
        self._live()
        if draws < 16:
            raise ValueError("certifying fewer than 16 draws says nothing")
        if _certify(self._h, int(draws), int(max(1, thin))) == 0:
            raise RuntimeError("could not certify this run")
        n = _cert_findings(self._h)
        findings = []
        for i in range(n):
            need = _cert_finding(self._h, i, None, 0)
            buf = (ctypes.c_ubyte * (need + 1))()
            got = _cert_finding(self._h, i, buf, need)
            findings.append(bytes(buf[:got]).decode("utf-8", "replace"))
        g = _cert_f
        return Certificate(
            draws=int(draws),
            beta_eff=float(g["beta_eff"](self._h)),
            beta_ci=(float(g["beta_lo"](self._h)), float(g["beta_hi"](self._h))),
            tau_int=float(g["tau"](self._h)),
            ess=float(g["ess"](self._h)),
            tv=float(g["tv"](self._h)),
            noise_floor=float(g["floor"](self._h)),
            findings=findings,
        )

    @property
    def known_optimum(self) -> float | None:
        """The true ground energy, for a planted instance. ``None`` otherwise."""
        self._live()
        v = float(_ground_energy(self._h))
        return None if v != v else v

    def excess(self) -> float | None:
        """How far the current state sits above a planted instance's known optimum, as a fraction."""
        k = self.known_optimum
        if k is None:
            return None
        e = self.energy
        return (e - k) / abs(k) if abs(k) > 1e-12 else e - k

    @property
    def treewidth(self) -> int:
        """Induced width of the elimination order. Exact inference costs ``2 ** treewidth``."""
        self._live()
        return int(_exact_width(self._h))

    def exact_ground_energy(self, max_width: int = 22) -> float | None:
        """Exact ground energy by variable elimination, or ``None`` if the graph is too dense.

        Cost is ``2 ** width`` in the graph's shape, not ``2 ** n`` in its size, so a long chain is
        instant where a dense graph of the same node count is impossible. Check
        :attr:`treewidth` first.
        """
        self._live()
        v = float(_exact_ground(self._h, int(max_width)))
        return None if v != v else v

    def exact_log_z(self, beta: float = 1.0, max_width: int = 22) -> float | None:
        """Exact log partition function, or ``None`` if too dense."""
        self._live()
        v = float(_exact_log_z(self._h, float(beta), int(max_width)))
        return None if v != v else v

    def close(self) -> None:
        if getattr(self, "_h", None):
            _free(self._h)
            self._h = None

    def _live(self) -> None:
        if getattr(self, "_h", None) is None:
            raise RuntimeError("this simulation is closed")

    def __enter__(self) -> "Sim":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        if getattr(self, "_h", None) is None:
            return "<Sim closed>"
        return f"<Sim nodes={len(self)} energy={self.energy:.4g} m={self.magnetization:.3f}>"


# ---- constructors ------------------------------------------------------------------------------


def _wrap(handle: int, beta: float, what: str) -> Sim:
    if not handle:
        raise ValueError(f"could not build {what}")
    s = Sim(handle)
    s._beta = float(beta)
    return s


def lattice2d(l: int, j: float = 1.0, beta: float = 0.44, seed: int = 0) -> Sim:
    """2D nearest-neighbour Ising lattice, periodic, side `l`."""
    if l < 2:
        raise ValueError("l must be at least 2")
    return _wrap(_ising2d_new(int(l), float(j), float(beta), int(seed)), beta, "the lattice")


def z1_grid(w: int, h: int, j: float = 1.0, hb: float = 0.0, beta: float = 1.0, seed: int = 0) -> Sim:
    """Z1-topology grid, degree 16, open boundaries."""
    return _wrap(_z1_new(int(w), int(h), float(j), float(hb), float(beta), int(seed)), beta, "the grid")


def ring(n: int, j: float = 1.0, h: float = 0.0, beta: float = 1.0, seed: int = 0) -> Sim:
    """A periodic chain of `n` nodes."""
    if n < 3:
        raise ValueError("a ring needs at least 3 nodes")
    m = Model(n)
    for i in range(n):
        m.couple(i, (i + 1) % n, j)
    if h:
        for i in range(n):
            m.bias(i, h)
    return m.build(beta, seed)


def from_spec(spec: dict, beta: float | None = None, seed: int | None = None) -> Sim:
    """Build from the same JSON shape the HTTP API and the MCP tools accept.

    Accepts either the full request body (``{"graph": {...}, "beta": ..., "seed": ...}``) or the
    graph object alone.
    """
    g = spec.get("graph", spec)
    b = float(spec.get("beta", 1.0) if beta is None else beta)
    s = int(spec.get("seed", 0) if seed is None else seed)

    kind = g.get("builtin")
    if kind == "lattice2d":
        return lattice2d(g["l"], g.get("j", 1.0), b, s)
    if kind == "ring":
        return ring(g["n"], g.get("j", 1.0), g.get("h", 0.0), b, s)
    if kind is not None:
        raise ValueError(
            f'unknown builtin {kind!r}; use "lattice2d" or "ring", or give "n" with "couplings"'
        )
    if "n" not in g:
        raise ValueError('the graph needs either "builtin" or "n" with "couplings"')

    m = Model(int(g["n"]))
    m.couple_many(g.get("couplings", []))
    for k, e in enumerate(g.get("biases", [])):
        if len(e) != 2:
            raise ValueError(f"bias {k} must be (i, h), got {len(e)} values")
        m.bias(int(e[0]), float(e[1]))
    return m.build(b, s)


def frustrated(l: int, loops: int, seed: int = 0, beta: float = 1.0) -> Sim:
    """A planted instance on an ``l`` by ``l`` lattice whose optimum is known by construction.

    Difficulty is not monotonic in ``loops``: it peaks near four planted loops per edge and falls
    away at both ends, so a very sparse or a saturated instance is easy.
    """
    return _wrap(_planted_frustrated(int(l), int(loops), int(seed), float(beta)), beta,
                 "the planted instance")


def wishart(n: int, alpha: float = 0.5, seed: int = 0, beta: float = 1.0) -> Sim:
    """The Wishart planted ensemble: dense, with a known optimum, and hard below ``alpha`` of 1.

    A miss here is usually under 2% above the optimum, because the landscape is dense with
    near-degenerate minima. Report the solve rate rather than the mean excess, or this family looks
    easy when it is not.
    """
    return _wrap(_planted_wishart(int(n), float(alpha), int(seed), float(beta)), beta,
                 "the Wishart instance")


def onsager(beta: float) -> float:
    """Onsager's exact spontaneous magnetisation for the 2D Ising model. Ground truth."""
    return float(_onsager(float(beta)))


# ---- modelling ------------------------------------------------------------------------------
#
# Everything above works in spins: couplings, biases, energies. That is the machine's language, not
# the problem's. This layer is the problem's -- variables that hold values, constraints that say
# what must be true, an objective that says what is better -- and it compiles down to the layer
# above. Answers come back in the names you used.


class Term:
    """A piece of an objective. Build these with arithmetic, not by calling the constructor.

    ``5 * shift.is_(2)`` is one term; ``a.is_(1) * b.is_(1)`` is quadratic; a third factor makes it
    cubic, and so on. Adding them makes an objective. Terms are inert until handed to
    :meth:`Problem.maximize` or :meth:`Problem.minimize`.

    Three or more literals in one product is a **higher-order** term. The compiler lowers it with an
    ancilla spin per substituted pair, so it costs spins — and the guarantee that comes with it is
    about optimisation rather than sampling. See ``ferrotherm::reduce`` for the whole story.

    Each part is ``(coefficient, [literals])``. It used to be ``(coefficient, a, b)`` with ``b``
    optionally ``None``, which could not represent a product of three at all.
    """

    __slots__ = ("parts",)

    def __init__(self, parts: "list[tuple[float, list]]") -> None:
        self.parts = parts

    def __add__(self, other: Any) -> "Term":
        return Term(self.parts + _as_term(other).parts)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "Term":
        return self + (-_as_term(other))

    def __rsub__(self, other: Any) -> "Term":
        return _as_term(other) + (-self)

    def __neg__(self) -> "Term":
        return Term([(-c, lits) for c, lits in self.parts])

    def __mul__(self, k: Any) -> "Term":
        if isinstance(k, (int, float)):
            return Term([(c * float(k), lits) for c, lits in self.parts])
        if isinstance(k, Literal):
            # Another factor on every part: this is what makes `a * b * c` mean a cubic term
            # rather than a type error.
            return Term([(c, lits + [k]) for c, lits in self.parts])
        if isinstance(k, Term):
            if len(self.parts) != 1 or len(k.parts) != 1:
                raise TypeError(
                    "multiplying sums is not supported; expand it yourself, because doing it "
                    "silently would turn one term into many and change what the penalty scales to"
                )
            (c1, l1), (c2, l2) = self.parts[0], k.parts[0]
            return Term([(c1 * c2, l1 + l2)])
        return NotImplemented

    __rmul__ = __mul__

    @property
    def degree(self) -> int:
        """The widest product in this term. Three or more needs a higher-order reduction."""
        return max((len(lits) for _, lits in self.parts), default=0)

    def __repr__(self) -> str:
        return "<Term %d part%s, degree %d>" % (
            len(self.parts), "" if len(self.parts) == 1 else "s", self.degree)


class Literal:
    """The claim that one variable takes one value. Multiply it by a number to weight it."""

    __slots__ = ("var", "value")

    def __init__(self, var: "Variable", value: int) -> None:
        self.var, self.value = var, value

    def __mul__(self, other: Any) -> Term:
        if isinstance(other, (int, float)):
            return Term([(float(other), [self])])
        if isinstance(other, Literal):
            return Term([(1.0, [self, other])])     # two literals is quadratic
        if isinstance(other, Term):
            return Term([(1.0, [self])]) * other
        return NotImplemented

    __rmul__ = __mul__

    def __add__(self, other: Any) -> Term:
        return Term([(1.0, [self])]) + _as_term(other)

    __radd__ = __add__

    def __sub__(self, other: Any) -> Term:
        return Term([(1.0, [self])]) - _as_term(other)

    def __rsub__(self, other: Any) -> Term:
        return _as_term(other) - Term([(1.0, [self])])

    def __neg__(self) -> Term:
        return Term([(-1.0, [self])])

    def __repr__(self) -> str:
        return f"<{self.var.name} is {self.value}>"


def _prod(xs: "Sequence[int]") -> int:
    n = 1
    for x in xs:
        n *= x
    return n


def _as_term(x: Any) -> Term:
    if isinstance(x, Term):
        return x
    if isinstance(x, Literal):
        return Term([(1.0, [x])])
    # `sum()` starts from 0, and summing terms is the natural way to build an objective in a loop,
    # so zero is the empty term. Any other bare number is a mistake: an objective is a function of
    # the variables, and a constant added to it changes no answer.
    if isinstance(x, (int, float)) and x == 0:
        return Term([])
    raise TypeError(
        f"an objective is built from literals and terms, not {type(x).__name__}. "
        "Write `3 * x.is_(2)`, not a bare number; a constant would change no answer."
    )


class Variable:
    """A declared variable. Ask it for a literal with ``is_(value)``."""

    __slots__ = ("_problem", "_index", "name", "domain")

    def __init__(self, problem: "Problem", index: int, name: str, domain: str) -> None:
        self._problem, self._index, self.name, self.domain = problem, index, name, domain

    def is_(self, value: int) -> Literal:
        """The literal "this variable takes ``value``"."""
        return Literal(self, int(value))

    def __repr__(self) -> str:
        return f"<Variable {self.name}: {self.domain}>"


class Violation:
    """A constraint the answer breaks, and by how much.

    ``str(v)`` is the description; ``v.by`` is the magnitude in the constraint's own units — places
    over a ceiling, places under a floor, distance from a fixed value. Always positive.
    """

    __slots__ = ("detail", "by")

    def __init__(self, detail: str, by: float) -> None:
        self.detail, self.by = detail, float(by)

    def __str__(self) -> str:
        return self.detail

    def __repr__(self) -> str:
        return f"<Violation by {self.by:g}: {self.detail}>"


class Grid:
    """A grid of variables, subscripted like an array.

    ``a[w, s]`` is the variable at that position. Names are ``assign[2,3]``, so the answer still
    reads in your own words.

    Built by :meth:`Problem.grid`. The shape a real model has — one variable per worker per shift —
    and without it every model starts with a loop, an f-string and index arithmetic nobody checks.
    """

    __slots__ = ("name", "dims", "_vars")

    def __init__(self, name: str, dims: "tuple[int, ...]", vars: "list[Variable]") -> None:
        self.name, self.dims, self._vars = name, tuple(dims), vars

    def _offset(self, sub: "tuple[int, ...]") -> int:
        if len(sub) != len(self.dims):
            raise IndexError(
                f"{self.name} has {len(self.dims)} dimensions and was given {len(sub)}")
        off = 0
        for d, (i, n) in enumerate(zip(sub, self.dims)):
            # Negative indices are refused rather than wrapped. Python wraps, and a wrap here is an
            # off-by-one that reaches the answer looking like a result.
            if not 0 <= i < n:
                raise IndexError(f"{self.name}: index {i} is outside dimension {d}, which is {n}")
            off = off * n + i
        return off

    def __getitem__(self, sub: Any) -> "Variable":
        return self._vars[self._offset(sub if isinstance(sub, tuple) else (sub,))]

    def __len__(self) -> int:
        return len(self._vars)

    def __iter__(self):
        return iter(self._vars)

    @property
    def all(self) -> "list[Variable]":
        """Every variable, row-major."""
        return list(self._vars)

    def row(self, *prefix: int) -> "list[Variable]":
        """One row of the last dimension: ``a.row(w)`` is every shift for worker ``w``."""
        if len(prefix) >= len(self.dims):
            raise IndexError(f"{self.name}: a row needs fewer indices than the {len(self.dims)} it has")
        pad = tuple(prefix) + (0,) * (len(self.dims) - len(prefix) - 1)
        return [self[pad + (k,)] for k in range(self.dims[-1])]

    def column(self, *suffix: int) -> "list[Variable]":
        """One column of the first dimension: ``a.column(s)`` is every worker for shift ``s``."""
        if len(suffix) >= len(self.dims):
            raise IndexError(f"{self.name}: a column needs fewer indices than the {len(self.dims)} it has")
        pad = tuple(suffix) + (0,) * (len(self.dims) - len(suffix) - 1)
        return [self[(k,) + pad] for k in range(self.dims[0])]

    def __repr__(self) -> str:
        return f"<Grid {self.name}{list(self.dims)}>"


class Answer:
    """A solved problem, read by name.

    ``answer["shift"]`` gives a value; ``answer.feasible`` says whether every variable decoded AND
    every constraint held. An infeasible answer is still returned, because knowing *which* part was
    not delivered is more useful than a raised exception with nothing in it: a variable that failed
    to decode reads ``None`` and appears in :attr:`undecoded`, and a constraint the objective
    outbid is described in :attr:`violated`.
    """

    __slots__ = ("values", "feasible", "energy", "spins", "penalty", "violated")

    def __init__(self, **kw: Any) -> None:
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __getitem__(self, name: str) -> int:
        try:
            return self.values[name]
        except KeyError:
            raise KeyError(
                f"no variable named {name!r}; this problem has "
                + ", ".join(repr(k) for k in self.values)
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self.values

    def __iter__(self):
        return iter(self.values.items())

    @property
    def undecoded(self) -> "list[str]":
        """The variables whose encoding was violated, which read as ``None``."""
        return [k for k, v in self.values.items() if v is None]

    def __repr__(self) -> str:
        head = "feasible" if self.feasible else "INFEASIBLE"
        body = ", ".join(f"{k}={'?' if v is None else v}" for k, v in self.values.items())
        out = f"<Answer {head} energy={self.energy:.4f} [{body}]"
        for v in self.violated or ():
            out += f"\n  broken: {v.detail} (by {v.by:g})"
        return out + ">"


class Problem:
    """A problem stated in its own terms, compiled to spins and sampled.

    >>> p = Problem()
    >>> days = {d: p.binary(d) for d in ("mon", "tue", "wed", "thu", "fri")}
    >>> p.at_most(list(days.values()), 3)
    >>> p.maximize(sum(w * v.is_(1) for w, v in zip((5, 4, 3, 2, 1), days.values())))
    >>> ans = p.solve()
    >>> [d for d in days if ans[d]]
    ['mon', 'tue', 'wed']

    Inequalities cost extra spins: each needs a slack variable to become an equality the sampler can
    square. The slack never appears in the answer.
    """

    def __init__(self) -> None:
        self._h = _model_new()
        if not self._h:
            raise MemoryError("could not allocate a problem")
        self._vars: "dict[str, Variable]" = {}

    # -- variables ------------------------------------------------------------------------------

    def categorical(self, name: str, values: int) -> Variable:
        """A variable holding one of ``values`` distinct values, encoded one-hot."""
        return self._declare(name, "categorical", _model_categorical(self._h, int(values)))

    def integer(self, name: str, lo: int, hi: int) -> Variable:
        """A variable over the inclusive range ``lo``..``hi``.

        There is no machine integer here. This is a categorical over the range, encoded so that
        neighbouring values differ in one spin; the name is for the modeller, not the fabric.
        """
        return self._declare(name, f"integer {lo}..{hi}",
                             _model_integer(self._h, int(lo), int(hi)))

    def binary(self, name: str) -> Variable:
        """A variable that is 0 or 1."""
        return self._declare(name, "binary", _model_binary(self._h))

    def grid(self, name: str, dims: "Sequence[int]", declare: Any = None) -> Grid:
        """A grid of variables, indexed and named for you.

        ``declare`` builds one variable given the problem and a name; it defaults to a binary.

        >>> p = Problem()
        >>> a = p.grid("assign", (3, 3))
        >>> for w in range(3):
        ...     p.exactly_one(a.row(w))
        >>> a[0, 2].name
        'assign[0,2]'
        """
        dims = tuple(int(d) for d in dims)
        if not dims or any(d <= 0 for d in dims):
            raise ValueError(f"a grid needs positive dimensions, got {dims}")
        make = declare if declare is not None else (lambda p, n: p.binary(n))
        vars = []
        sub = [0] * len(dims)
        for _ in range(_prod(dims)):
            vars.append(make(self, f"{name}[{','.join(str(i) for i in sub)}]"))
            for d in reversed(range(len(dims))):
                sub[d] += 1
                if sub[d] < dims[d]:
                    break
                sub[d] = 0
        return Grid(name, dims, vars)

    def _declare(self, name: str, domain: str, index: int) -> Variable:
        if index == 0xFFFFFFFF:
            raise ValueError(
                f"{domain} is not a usable domain for {name!r} "
                "(a categorical needs at least 2 values; an integer needs hi > lo)"
            )
        if name in self._vars:
            raise ValueError(f"{name!r} is already declared")
        # Push the name down, so the library's errors name the variable the caller declared rather
        # than the handle it was given back.
        raw = name.encode("utf-8")
        _model_name(self._h, index, raw, len(raw))
        v = Variable(self, index, name, domain)
        self._vars[name] = v
        return v

    # -- constraints ----------------------------------------------------------------------------

    def not_equal(self, a: Variable, b: Variable) -> None:
        """``a`` and ``b`` must differ."""
        self._must(_model_not_equal(self._h, a._index, b._index), "not_equal")

    def equal(self, a: Variable, b: Variable) -> None:
        """``a`` and ``b`` must agree."""
        self._must(_model_equal(self._h, a._index, b._index), "equal")

    def fix(self, v: Variable, value: int) -> None:
        """``v`` must take ``value``."""
        self._must(_model_fix(self._h, v._index, int(value)), "fix")

    def exactly(self, of: "Sequence[Any]", k: int, value: int = 1) -> None:
        """Exactly ``k`` of them hold."""
        self._counting(0, of, k, value, "exactly")

    def at_most(self, of: "Sequence[Any]", k: int, value: int = 1) -> None:
        """At most ``k`` of them hold. Costs a slack variable."""
        self._counting(1, of, k, value, "at_most")

    def at_least(self, of: "Sequence[Any]", k: int, value: int = 1) -> None:
        """At least ``k`` of them hold. Costs a slack variable."""
        self._counting(2, of, k, value, "at_least")

    def exactly_one(self, of: "Sequence[Any]") -> None:
        """Exactly one of them holds. Cheaper than ``exactly(of, 1)``: pairwise, with no slack."""
        self._counting(3, of, 0, 1, "exactly_one")

    def at_most_one(self, of: "Sequence[Any]") -> None:
        """At most one of them holds."""
        self._counting(4, of, 0, 1, "at_most_one")

    def _counting(self, kind: int, of: "Sequence[Any]", k: int, value: int, what: str) -> None:
        """Any number of literals, each naming its own value.

        ``of`` takes variables -- in which case ``value`` applies to all of them, which is the common
        case -- or literals, which each carry their own. "at most two of these nine shifts" and "no
        more than one of a=3, b=17, c=0" are both sayable.
        """
        items = list(of)
        if len(items) < 2:
            raise ValueError(f"{what} needs at least two things to count, not {len(items)}")
        if kind <= 2 and not 0 <= k <= len(items):
            raise ValueError(f"k must be between 0 and {len(items)} for {what}, not {k}")
        _model_lits_clear(self._h)
        for it in items:
            lit = it if isinstance(it, Literal) else Literal(it, int(value))
            if not isinstance(lit.var, Variable):
                raise TypeError(
                    f"{what} counts variables or literals, not {type(it).__name__}. "
                    "Write `p.at_most([a, b, c], 2)` or `p.at_most([a.is_(3), b.is_(17)], 1)`."
                )
            self._must(_model_lit(self._h, lit.var._index, lit.value), what)
        self._must(_model_close(self._h, kind, int(k)), what)

    # -- objective ------------------------------------------------------------------------------

    def maximize(self, term: Any) -> None:
        """Prefer states where ``term`` is large."""
        self._objective(term, maximize=True)

    def minimize(self, term: Any) -> None:
        """Prefer states where ``term`` is small."""
        self._objective(term, maximize=False)

    def _objective(self, term: Any, maximize: bool) -> None:
        m = 1 if maximize else 0
        for coeff, lits in _as_term(term).parts:
            if not lits:
                continue    # a constant changes no answer
            if len(lits) == 1:
                a = lits[0]
                self._must(_model_objective_term(self._h, m, coeff, a.var._index, a.value),
                           "objective")
            elif len(lits) == 2:
                a, b = lits
                self._must(_model_objective_pair(self._h, m, coeff, a.var._index, a.value,
                                                 b.var._index, b.value), "objective")
            else:
                # Three or more: the library's literal list, then close it as a product. The
                # compiler lowers it with ancillas.
                _model_lits_clear(self._h)
                for l in lits:
                    self._must(_model_lit(self._h, l.var._index, l.value), "objective")
                self._must(_model_objective_product(self._h, m, coeff), "objective")

    # -- solving --------------------------------------------------------------------------------

    def solve(
        self,
        tries: int = 12,
        beta_hot: float = 0.0,
        beta_cold: float = 0.0,
        stages: int = 0,
        sweeps: int = 0,
    ) -> Answer:
        """Compile and solve, keeping the best of ``tries`` anneals.

        The four ladder parameters default to the library's own; a zero means "use that default", so
        a caller who has measured their own instance can override only what they measured.
        """
        spins = _model_compile(self._h)
        if spins == 0:
            raise ValueError(self._error() or "this problem did not compile")
        if not _model_solve_with(self._h, max(1, int(tries)), float(beta_hot), float(beta_cold),
                                 int(stages), int(sweeps)):
            raise ValueError(
                "that annealing ladder is not usable: beta_cold must exceed beta_hot, and both "
                "must be real numbers. Pass 0 for any of the four to use the default."
            )
        # A variable that did not decode is reported as None rather than raised on. Its encoding was
        # violated, which means a penalty lost to the objective -- and knowing WHICH variable lost is
        # the whole diagnosis. An exception would throw that away and leave the caller with nothing.
        vals = {}
        for name, v in self._vars.items():
            got = _model_value(self._h, v._index)
            vals[name] = None if got == -(2 ** 63) else int(got)
        # Each carries how far outside it sits, not only that it broke. "At most 2 of 5 and 3 hold"
        # is a near miss; "and 5 hold" is not, and only the magnitude separates them.
        broken = [
            Violation(_read_text_idx(_model_violation, self._h, i),
                      _model_violation_amount(self._h, i))
            for i in range(_model_violations(self._h))
        ]
        return Answer(values=vals, feasible=bool(_model_feasible(self._h)),
                      violated=broken, energy=float(_model_energy(self._h)),
                      spins=int(spins), penalty=float(_model_penalty(self._h)))

    def penalty(self, p: float) -> None:
        """Use exactly this penalty, disabling the automatic scaling.

        The remedy when :attr:`Answer.feasible` comes back false: a constraint lost to the objective
        and has to outrank it. By default the penalty is twice the largest objective coefficient,
        which is enough for most models and not for all of them.
        """
        self._must(_model_fixed_penalty(self._h, float(p)), "penalty")

    def certify(self, beta: float = 1.0, draws: int = 512, thin: int = 1) -> Certificate:
        """Ask whether the sampler that produced the answer was sampling what it claimed.

        An answer says *what*. This says whether the machine that produced it reached the
        temperature it was asked for, how many of its draws were independent, and -- where the model
        is small enough to enumerate -- how far its distribution sits from the exact one, beside the
        noise floor that distance has to beat.

        Read ``findings``. It is empty exactly when the run was sound.
        """
        if not _model_certify(self._h, float(beta), int(draws), int(thin)):
            raise ValueError(
                self._error()
                or "could not certify: compile and solve the problem first, and pass a positive "
                   "beta with at least 16 draws"
            )
        n = _model_cert_findings(self._h)
        findings = [_read_text_idx(_model_cert_finding, self._h, i) for i in range(n)]
        g = {k: fn(self._h) for k, fn in _model_cert_f.items()}
        nan = lambda x: None if x != x else x  # noqa: E731
        return Certificate(draws=int(draws), beta_eff=g["beta"], beta_ci=None,
                           tau_int=g["tau"], ess=g["ess"], tv=nan(g["tv"]),
                           noise_floor=nan(g["floor"]), findings=findings)

    def ftp(self) -> str:
        """The compiled program in ``.ftp`` form, for running on another fabric."""
        return _read_text(_model_ftp, self._h)

    def _error(self) -> str:
        return _read_text(_model_error, self._h)

    def _must(self, ok: int, what: str) -> None:
        if not ok:
            raise ValueError(self._error() or f"the library refused that {what}")

    def __del__(self) -> None:
        h, self._h = getattr(self, "_h", None), None
        if h:
            _model_free(h)
