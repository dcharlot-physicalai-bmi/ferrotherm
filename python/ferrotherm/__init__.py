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
    "Certificate",
    "Model",
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

__version__ = "0.1.0"


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
