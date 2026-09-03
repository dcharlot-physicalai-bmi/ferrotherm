"""Thermodynamic sampling from Python.

A thin, honest binding over the ferrotherm C ABI. There is no build step and no compiler
requirement: this loads the shared library through ctypes, so ``pip install`` never has to
compile anything on your machine.

    >>> import ferrotherm as ft
    >>> sim = ft.lattice2d(32, beta=0.44)
    >>> sim.sweep(500)
    500
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
    "Hubo",
    "hardware_threads",
    "Bounds",
    "BranchResult",
    "BreakoutRun",
    "Certificate",
    "pegasus",
    "zephyr",
    "Estimate",
    "SampleSet",
    "PlanarCut",
    "PopulationRun",
    "Rounded",
    "ClusterRun",
    "ToroidalBound",
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
    "rbm",
    "dbm",
    "bars_and_stripes",
    "onsager",
    "library_path",
    "__version__",
]

# Tracks the native library it binds, because they are released together out of one repository and
# a binding whose version says nothing about the library underneath it is a version nobody can use.
__version__ = "0.37.0"


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
_pegasus_new = _sig("ft_pegasus_new", _p, [c_uint32, c_double, c_double, c_uint64])
_zephyr_new = _sig("ft_zephyr_new", _p, [c_uint32, c_uint32, c_double, c_double, c_uint64])
_qubit = _sig("ft_qubit", c_uint32, [_p, c_uint32])
_sparsify = _sig("ft_sparsify", _p, [_p, c_uint32])
_sparsify_variables = _sig("ft_sparsify_variables", c_uint32, [_p])
_sparsify_copies = _sig("ft_sparsify_copies", c_uint32, [_p, c_uint32, POINTER(c_uint32), c_uint32])
_sparsify_offset = _sig("ft_sparsify_offset", c_double, [_p])
_sparsify_project = _sig("ft_sparsify_project", c_uint32, [_p, POINTER(c_int8), c_uint32])
_embed = _sig("ft_embed", c_uint32, [_p, _p, c_uint64, c_uint32, c_uint64])
_embed_sites = _sig("ft_embed_sites", c_uint32, [_p])
_embed_longest = _sig("ft_embed_longest", c_uint32, [_p])
_embed_chain = _sig("ft_embed_chain", c_uint32, [_p, c_uint32, POINTER(c_uint32), c_uint32])
_site_lower_bound = _sig("ft_site_lower_bound", c_uint32, [_p, _p])
_embed_apply = _sig("ft_embed_apply", _p, [_p, _p, c_double])
_clique_embed = _sig("ft_clique_embed", c_uint32, [_p, _p, POINTER(c_uint32)])
_unembed = _sig("ft_unembed", c_uint32, [_p, POINTER(c_int8), c_uint32])
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
_set_spins = _sig("ft_set_spins", c_uint32, [_p, POINTER(c_int8), c_uint32])
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
_collect = _sig("ft_collect", c_uint32, [_p, c_uint32, c_uint32, c_uint32])
_samples_len = _sig("ft_samples_len", c_uint32, [_p])
_samples_distinct = _sig("ft_samples_distinct", c_uint32, [_p])
_samples_best_energy = _sig("ft_samples_best_energy", c_double, [_p])
_samples_chain_tau = _sig("ft_samples_chain_tau", c_double, [_p])
_samples_degeneracy = _sig("ft_samples_degeneracy", c_uint32, [_p, c_double])
_samples_state = _sig("ft_samples_state", c_uint32, [_p, c_uint32, POINTER(c_int8), c_uint32])
_samples_mean_spin = _sig("ft_samples_mean_spin", c_uint32, [_p, c_uint32, POINTER(c_double)])
_samples_correlation = _sig("ft_samples_correlation", c_uint32, [_p, c_uint32, c_uint32, POINTER(c_double)])
_samples_magnetization = _sig("ft_samples_magnetization", c_uint32, [_p, POINTER(c_double)])
_samples_mean_energy = _sig("ft_samples_mean_energy", c_uint32, [_p, POINTER(c_double)])
_exact_ground = _sig("ft_exact_ground", c_double, [_p, c_uint32])
_exact_marginals = _sig("ft_exact_marginals", c_uint32,
                        [_p, c_double, c_uint32, ctypes.POINTER(c_double), c_uint32])
_exact_log_z = _sig("ft_exact_log_z", c_double, [_p, c_double, c_uint32])
_exact_width = _sig("ft_exact_width", c_uint32, [_p])
_exact_ground_state = _sig("ft_exact_ground_state", c_uint32, [_p, c_uint32, POINTER(c_int8), c_uint32])
_tabu = _sig("ft_tabu", c_double, [_p, c_uint32, c_uint32, c_uint32])
_tabu_iterations = _sig("ft_tabu_iterations", c_uint64, [_p])
_planar_cut = _sig("ft_planar_cut", c_double, [_p, c_double])
_toroidal_bound = _sig("ft_toroidal_bound", c_double, [_p, c_double])
_toroidal_attained = _sig("ft_toroidal_attained", c_uint32, [_p])
_planar_faces = _sig("ft_planar_faces", c_uint64, [_p])
_planar_odd_faces = _sig("ft_planar_odd_faces", c_uint64, [_p])
_planar_error = _sig("ft_planar_error", c_uint32, [_p, ctypes.POINTER(ctypes.c_ubyte), c_uint32])
_gw_round = _sig("ft_gw_round", c_double, [_p, c_uint32, c_uint64])
_gw_guaranteed = _sig("ft_gw_guaranteed", c_uint32, [_p])
_icm = _sig("ft_icm", c_double, [_p, c_uint32, c_uint32, c_double, c_double])
_icm_moves = _sig("ft_icm_moves", c_uint64, [_p])
_sqa = _sig("ft_sqa", c_double, [_p, c_uint32, c_double, c_double, c_double, c_uint32])
_bls = _sig("ft_bls", c_double, [_p, c_uint32])
_hfs = _sig("ft_hfs", c_double, [_p, c_uint32, c_uint32])
_hfs_moves = _sig("ft_hfs_moves", ctypes.c_uint64, [_p])
_hfs_improving = _sig("ft_hfs_improving", ctypes.c_uint64, [_p])
_bls_descents = _sig("ft_bls_descents", c_uint64, [_p])
_bls_iterations = _sig("ft_bls_iterations", c_uint64, [_p])
_bls_max_jump = _sig("ft_bls_max_jump", c_uint32, [_p])
_popanneal = _sig("ft_popanneal", c_double, [_p, c_uint32, c_uint32, c_double, c_uint32])
_popanneal_ln_z = _sig("ft_popanneal_ln_z", c_double, [_p])
_popanneal_rho = _sig("ft_popanneal_rho", c_double, [_p])
_branch = _sig("ft_branch", c_double, [_p, c_uint64])
_branch_proved = _sig("ft_branch_proved", c_uint32, [_p])
_branch_nodes = _sig("ft_branch_nodes", c_uint64, [_p])
_bound_decoupled = _sig("ft_bound_decoupled", c_double, [_p])
_bound_forest = _sig("ft_bound_forest", c_double, [_p, c_uint32])
_bound_odd_cycle = _sig("ft_bound_odd_cycle", c_double, [_p, c_uint32])
_bound_sdp = _sig("ft_bound_sdp", c_double, [_p, c_uint32, c_uint64])
_free = _sig("ft_free", None, [_p])

# the modelling layer
_u8p = ctypes.POINTER(ctypes.c_ubyte)
_sweep_par = _sig("ft_sweep_par", ctypes.c_uint64, [_p, c_uint32, c_uint32])
_threads_used = _sig("ft_threads_used", c_uint32, [_p])
_hardware_threads = _sig("ft_hardware_threads", c_uint32, [])
_hubo_new = _sig("ft_hubo_new", _p, [c_uint32])
_hubo_free = _sig("ft_hubo_free", None, [_p])
_hubo_from_sim = _sig("ft_hubo_from_sim", _p, [_p])
_hubo_vars_clear = _sig("ft_hubo_vars_clear", c_uint32, [_p])
_hubo_var = _sig("ft_hubo_var", c_uint32, [_p, c_uint32])
_hubo_add = _sig("ft_hubo_add", c_uint32, [_p, c_double])
_hubo_len = _sig("ft_hubo_len", c_uint32, [_p])
_hubo_terms = _sig("ft_hubo_terms", c_uint32, [_p])
_hubo_max_arity = _sig("ft_hubo_max_arity", c_uint32, [_p])
_hubo_ancillas_avoided = _sig("ft_hubo_ancillas_avoided", c_uint32, [_p])
_hubo_anneal = _sig("ft_hubo_anneal", c_double,
                    [_p, c_double, c_double, c_uint32, c_uint32, ctypes.c_uint64])
_hubo_read = _sig("ft_hubo_read", c_uint32, [_p, ctypes.POINTER(ctypes.c_int8), c_uint32])
_hubo_set_spins = _sig("ft_hubo_set_spins", c_uint32, [_p, ctypes.POINTER(ctypes.c_int8), c_uint32])
_hubo_energy = _sig("ft_hubo_energy", c_double, [_p])
_hubo_delta = _sig("ft_hubo_delta", c_double, [_p, c_uint32])
_hubo_proposals = _sig("ft_hubo_proposals", ctypes.c_uint64, [_p])
_hubo_accepted = _sig("ft_hubo_accepted", ctypes.c_uint64, [_p])
_hubo_joules_z1 = _sig("ft_hubo_joules_z1", c_double, [_p])
_hubo_error = _sig("ft_hubo_error", c_uint32, [_p, _u8p, c_uint32])

_model_new = _sig("ft_model_new", _p, [])
_model_free = _sig("ft_model_free", None, [_p])
_model_categorical = _sig("ft_model_categorical", c_uint32, [_p, c_uint32])
_model_categorical_as = _sig("ft_model_categorical_as", c_uint32, [_p, c_uint32, c_uint32])
_model_integer_as = _sig("ft_model_integer_as", c_uint32,
                         [_p, ctypes.c_int64, ctypes.c_int64, c_uint32])
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
_model_answers = _sig("ft_model_answers", c_uint32, [_p])
_model_optima = _sig("ft_model_optima", c_uint32, [_p, c_double])
_model_select_optimum = _sig("ft_model_select_optimum", c_uint32, [_p, c_uint32, c_double])
_model_solve_by = _sig("ft_model_solve_by", c_uint32, [_p, c_uint32, ctypes.c_uint64])
_model_proved = _sig("ft_model_proved", c_uint32, [_p])
_model_objective = _sig("ft_model_objective", c_double, [_p])
_model_has_objective = _sig("ft_model_has_objective", c_uint32, [_p])
_model_energy = _sig("ft_model_energy", c_double, [_p])
_model_penalty = _sig("ft_model_penalty", c_double, [_p])
_model_name = _sig("ft_model_name", c_uint32, [_p, c_uint32, ctypes.c_char_p, c_uint32])
_model_var = _sig("ft_model_var", c_uint32, [_p, c_uint32])
_model_lit = _sig("ft_model_lit", c_uint32, [_p, c_uint32, ctypes.c_int64])
_model_lit_weighted = _sig("ft_model_lit_weighted", c_uint32,
                           [_p, c_uint32, ctypes.c_int64, c_double])
_model_close_linear = _sig("ft_model_close_linear", c_uint32, [_p, c_uint32, c_double])
_model_close_linear_soft = _sig("ft_model_close_linear_soft", c_uint32,
                                [_p, c_uint32, c_double, c_double])
_model_lits_clear = _sig("ft_model_lits_clear", c_uint32, [_p])
_model_objective_product = _sig("ft_model_objective_product", c_uint32, [_p, c_uint32, c_double])
_model_close = _sig("ft_model_close", c_uint32, [_p, c_uint32, c_uint32])
_model_fixed_penalty = _sig("ft_model_fixed_penalty", c_uint32, [_p, c_double])
_model_close_soft = _sig("ft_model_close_soft", c_uint32, [_p, c_uint32, c_uint32, c_double])
_model_soften_last = _sig("ft_model_soften_last", c_uint32, [_p, c_double])
_model_soft_cost = _sig("ft_model_soft_cost", c_double, [_p])
_model_violation_is_hard = _sig("ft_model_violation_is_hard", c_uint32, [_p, c_uint32])
_model_ancillas = _sig("ft_model_ancillas", c_uint32, [_p])
_model_caveats = _sig("ft_model_caveats", c_uint32, [_p])
_ommx_read = _sig("ft_ommx_read", _p, [_u8p, c_uint32, c_double, c_uint64, POINTER(c_double)])
_ommx_error = _sig("ft_ommx_error", c_uint32, [_u8p, c_uint32])
_model_ommx = _sig("ft_model_ommx", c_uint32, [_p, _u8p, c_uint32])
_model_ommx_constant = _sig("ft_model_ommx_constant", c_double, [_p])
_model_caveat = _sig("ft_model_caveat", c_uint32, [_p, c_uint32, _u8p, c_uint32])
_model_violations = _sig("ft_model_violations", c_uint32, [_p])
_model_violation = _sig("ft_model_violation", c_uint32, [_p, c_uint32, _u8p, c_uint32])
_model_violation_amount = _sig("ft_model_violation_amount", c_double, [_p, c_uint32])
_model_certify = _sig("ft_model_certify", c_uint32, [_p, c_double, c_uint32, c_uint32])
_model_cert_findings = _sig("ft_model_cert_findings", c_uint32, [_p])
_model_cert_finding = _sig("ft_model_cert_finding", c_uint32, [_p, c_uint32, _u8p, c_uint32])
_model_cert_f = {n: _sig("ft_model_cert_" + n, c_double, [_p])
                 for n in ("beta", "ess", "tau", "tv", "floor")}
_model_error = _sig("ft_model_error", c_uint32, [_p, _u8p, c_uint32])
_ebm_rbm = _sig("ft_ebm_rbm", _p, [c_uint32, c_uint32, c_double, c_uint64])
_ebm_dbm = _sig("ft_ebm_dbm", _p,
                [c_uint32, POINTER(c_uint32), c_uint32, c_double, c_uint64])
_ebm_train = _sig("ft_ebm_train", c_uint32,
                  [_p, c_uint32, POINTER(c_int8), c_uint32, c_uint32, c_uint32, c_uint32,
                   c_double, c_uint32, c_uint64])
_ebm_log_likelihood = _sig("ft_ebm_log_likelihood", c_double,
                           [_p, c_uint32, POINTER(c_int8), c_uint32])
_ebm_bars_and_stripes = _sig("ft_ebm_bars_and_stripes", c_uint32, [c_uint32, POINTER(c_int8), c_uint32])
_ebm_error = _sig("ft_ebm_error", c_uint32, [_u8p, c_uint32])
_model_ftp = _sig("ft_model_ftp", c_uint32, [_p, _u8p, c_uint32])


def hardware_threads() -> int:
    """How many threads this machine can run at once, or 1 when that cannot be known.

    1 in a browser, which is the truth there rather than a failure to detect.
    """
    return int(_hardware_threads())


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


class Bounds:
    """Lower bounds on the ground energy. All sound, so :attr:`best` is their maximum.

    ``sdp`` is ``None`` when the certificate failed to re-verify — which is a refusal, not a
    missing feature. The others cannot fail.
    """

    __slots__ = ("decoupled", "forest", "odd_cycle", "sdp")

    def __init__(self, decoupled: float, forest: float, odd_cycle: float,
                 sdp: "float | None") -> None:
        self.decoupled, self.forest, self.odd_cycle, self.sdp = decoupled, forest, odd_cycle, sdp

    @property
    def best(self) -> float:
        """The tightest of them. Taking the maximum of sound bounds is sound."""
        vs = [self.decoupled, self.forest, self.odd_cycle]
        if self.sdp is not None:
            vs.append(self.sdp)
        return max(vs)

    @property
    def which(self) -> str:
        """Which bound set :attr:`best`. They disagree by a lot, and in both directions."""
        pairs = [("decoupled", self.decoupled), ("forest", self.forest),
                 ("odd_cycle", self.odd_cycle)]
        if self.sdp is not None:
            pairs.append(("sdp", self.sdp))
        return max(pairs, key=lambda kv: kv[1])[0]

    def __repr__(self) -> str:
        sdp = "refused" if self.sdp is None else f"{self.sdp:.6g}"
        return (f"<Bounds best={self.best:.6g} by {self.which}; decoupled={self.decoupled:.6g} "
                f"forest={self.forest:.6g} odd_cycle={self.odd_cycle:.6g} sdp={sdp}>")


class PlanarCut:
    """An **exact** maximum cut on a planar graph. Not the best found — the maximum.

    ``odd_faces`` is the size of the matching problem underneath, and the real cost driver: it is
    what makes the method cubic rather than exponential. Zero is legitimate and means the whole cut
    came free.
    """

    __slots__ = ("cut", "energy", "faces", "odd_faces")

    def __init__(self, cut: float, energy: float, faces: int, odd_faces: int) -> None:
        self.cut, self.energy = cut, energy
        self.faces, self.odd_faces = faces, odd_faces

    def __repr__(self) -> str:
        return (f"<PlanarCut EXACT cut={self.cut:.6g} energy={self.energy:.6g} "
                f"{self.faces} faces, {self.odd_faces} odd>")


class ToroidalBound:
    """An upper bound on the maximum cut of a torus, and whether it is achieved.

    ``attained`` means the relaxation's optimum is itself a genuine cut, so the bound *is* the
    maximum — proved. Not attained leaves the bound standing: every cut is such a subgraph, so a
    maximum over the larger set can only be larger.
    """

    __slots__ = ("cut", "attained")

    def __init__(self, cut: float, attained: bool) -> None:
        self.cut, self.attained = cut, attained

    def __repr__(self) -> str:
        what = "MAXIMUM (attained)" if self.attained else "upper bound"
        return f"<ToroidalBound {self.cut:.6g} — {what}>"


class Rounded:
    """A state rounded out of the semidefinite relaxation, and whether the guarantee covers it."""

    __slots__ = ("cut", "energy", "guaranteed")

    def __init__(self, cut: float, energy: float, guaranteed: bool) -> None:
        self.cut, self.energy, self.guaranteed = cut, energy, guaranteed

    def __repr__(self) -> str:
        g = "0.87856 guaranteed" if self.guaranteed else "no ratio (mixed-sign or fielded)"
        return f"<Rounded cut={self.cut:.6g} energy={self.energy:.6g} — {g}>"


class ClusterRun:
    """A PT+ICM run. ``moves`` is how many cluster moves actually fired — none means the replicas
    never disagreed, and the move was doing nothing."""

    __slots__ = ("energy", "moves")

    def __init__(self, energy: float, moves: int) -> None:
        self.energy, self.moves = energy, moves

    def __repr__(self) -> str:
        return f"<ClusterRun {self.energy:.6g} after {self.moves} cluster moves>"


class BreakoutRun:
    """A breakout-local-search run, with the evidence that it actually broke out.

    ``descents`` is the number of local optima visited and ``max_jump`` the largest perturbation it
    had to reach for. A run with few descents never left its first basin; a ``max_jump`` still at
    the initial value means the adaptive rule never fired.
    """

    __slots__ = ("energy", "descents", "iterations_run", "max_jump")

    def __init__(self, energy: float, descents: int, iterations_run: int, max_jump: int) -> None:
        self.energy, self.descents = energy, descents
        self.iterations_run, self.max_jump = iterations_run, max_jump

    def __repr__(self) -> str:
        return (f"<BreakoutRun {self.energy:.6g} after {self.descents} descents, "
                f"{self.iterations_run} flips, L<={self.max_jump}>")


class BranchResult:
    """What branch and bound found, and whether it proved it.

    ``proved`` is true only when the tree was exhausted inside the node budget. A run that hit the
    limit still returns its best state; it just cannot call it the minimum.
    """

    __slots__ = ("energy", "proved", "nodes")

    def __init__(self, energy: float, proved: bool, nodes: int) -> None:
        self.energy, self.proved, self.nodes = energy, proved, nodes

    def __repr__(self) -> str:
        head = "PROVED OPTIMAL" if self.proved else "best found (no proof: budget exhausted)"
        return f"<BranchResult {self.energy:.6g} {head}, {self.nodes} nodes>"


class PopulationRun:
    """A population-annealing run, with the diagnostic that says whether to believe it.

    ``rho`` is the family statistic: ``1.0`` when every ancestor still has one descendant,
    ``population`` when the population collapsed onto a single ancestor. A run whose ``rho`` spiked
    explored one basin with ``population`` copies of one history, and its ``ln_z`` is worth nothing.
    ``trustworthy`` applies the usual rule of thumb; the number is there so you can apply your own.
    """

    __slots__ = ("energy", "ln_z", "rho", "population")

    def __init__(self, energy: float, ln_z: "float | None", rho: float, population: int) -> None:
        self.energy, self.ln_z, self.rho, self.population = energy, ln_z, rho, population

    @property
    def trustworthy(self) -> bool:
        """``rho`` below a tenth of the population. A rule of thumb, not a theorem."""
        return self.rho <= max(1.0, self.population / 10.0)

    def free_energy(self, beta: float, n: int) -> "float | None":
        """``-ln Z / (beta * n)``, or ``None`` when ``ln_z`` is unavailable."""
        if self.ln_z is None or beta <= 0.0 or n <= 0:
            return None
        return -self.ln_z / (beta * n)

    def __repr__(self) -> str:
        z = "unavailable" if self.ln_z is None else f"{self.ln_z:.6g}"
        warn = "" if self.trustworthy else "  ** rho says do not trust ln_z **"
        return (f"<PopulationRun energy={self.energy:.6g} ln_z={z} "
                f"rho={self.rho:.3g}/{self.population}{warn}>")


class Estimate:
    """An expectation value with an error bar that accounts for how correlated the draws were.

    ``stderr`` is ``sqrt(var / ess)``, not ``sqrt(var / n)``. Chain draws are correlated, so ``n``
    of them are worth ``n / (2 * tau_int)`` independent ones, and the naive interval understates the
    error by ``sqrt(2 * tau)``. Measured against exact enumeration, on a chain with ``tau = 32`` the
    naive interval contains the true value for one site in four while announcing 95%.
    """

    __slots__ = ("value", "stderr", "ess", "tau_int")

    def __init__(self, value: float, stderr: float, ess: float, tau_int: float) -> None:
        self.value, self.stderr, self.ess, self.tau_int = value, stderr, ess, tau_int

    @property
    def ci95(self) -> "tuple[float, float]":
        return (self.value - 1.96 * self.stderr, self.value + 1.96 * self.stderr)

    def covers(self, truth: float) -> bool:
        lo, hi = self.ci95
        return lo <= truth <= hi

    def __repr__(self) -> str:
        tau = f", tau={self.tau_int:.1f}" if self.tau_int == self.tau_int else ""
        return f"<Estimate {self.value:.5f} +- {self.stderr:.5f} (ess {self.ess:.0f}{tau})>"


class SampleSet:
    """The states a run kept, and what may honestly be computed from them.

    Bound to one simulation: the accessors read whatever that simulation last collected, so keeping
    a ``SampleSet`` past the next :meth:`Sim.collect` gives you the new draws, not the old ones.

    >>> s = lattice2d(6, beta=0.5, seed=3)
    >>> d = s.collect(draws=512, thin=2, burn_in=200)
    >>> len(d)
    512
    >>> abs(d.mean_spin(0).value) <= 1.0
    True
    >>> d.mean_spin(0).stderr > 0.0
    True
    """

    def __init__(self, sim: "Sim") -> None:
        self._sim = sim

    def __len__(self) -> int:
        return int(_samples_len(self._sim._h))

    @property
    def distinct(self) -> int:
        """How many of the draws were DIFFERENT.

        A run returning 10,000 draws of which three are distinct has told you about three states,
        whatever its draw count says.
        """
        return int(_samples_distinct(self._sim._h))

    @property
    def best_energy(self) -> float:
        return float(_samples_best_energy(self._sim._h))

    @property
    def chain_tau(self) -> float:
        """The slowest autocorrelation the chain showed. Every estimate is deflated by it."""
        return float(_samples_chain_tau(self._sim._h))

    def degeneracy(self, tol: float = 1e-9) -> int:
        """Distinct states within ``tol`` of the lowest energy seen.

        Evidence of degeneracy, not a count of it: a chain proves the states it visited exist and
        nothing about the ones it did not.
        """
        return int(_samples_degeneracy(self._sim._h, float(tol)))

    def state(self, k: int) -> "list[int]":
        n = int(_samples_state(self._sim._h, int(k), None, 0))
        if n == 0:
            raise IndexError(f"no sample {k}")
        buf = (c_int8 * n)()
        got = _samples_state(self._sim._h, int(k), buf, n)
        return [int(v) for v in buf[:got]]

    def _four(self, call, *args) -> Estimate:
        out = (c_double * 4)()
        if call(self._sim._h, *args, out) == 0:
            raise RuntimeError("no samples collected, or the index is out of range")
        return Estimate(float(out[0]), float(out[1]), float(out[2]), float(out[3]))

    def mean_spin(self, i: int) -> Estimate:
        """``<s_i>``, in ``[-1, 1]``."""
        return self._four(_samples_mean_spin, int(i))

    def correlation(self, i: int, j: int) -> Estimate:
        """``<s_i s_j>``. This and :meth:`mean_spin` are the two moments contrastive divergence matches."""
        return self._four(_samples_correlation, int(i), int(j))

    def magnetization(self) -> Estimate:
        """The order parameter, with its error bar."""
        return self._four(_samples_magnetization)

    def mean_energy(self) -> Estimate:
        """``<E>``, the internal energy.

        :attr:`Sim.energy` reports the energy of the one configuration the machine is holding, and a
        draw from a distribution is not an estimate of its mean.
        """
        return self._four(_samples_mean_energy)

    def __repr__(self) -> str:
        return (f"<SampleSet {len(self)} draws, {self.distinct} distinct, "
                f"best {self.best_energy:.6g}, chain tau {self.chain_tau:.1f}>")


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


class Hubo:
    """A higher-order model, solved without quadratising it.

    A term of any width contributes ``-w * prod(s_i)``, so nothing here needs an ancilla. The other
    route to a k-body term -- an objective product on :class:`Problem` -- goes through Rosenberg's
    reduction, which introduces one ancilla per substituted pair and a penalty chosen larger than
    the whole model can pay. That penalty is what costs: measured on 60 three-body terms over 40
    spins, the reduced path at 1024x the budget does not reach this one at 1x.

    >>> h = Hubo(3)
    >>> h.add([0, 1, 2], 1.0)              # doctest: +ELLIPSIS
    <ferrotherm.Hubo ...>
    >>> round(h.anneal(seed=7), 6)
    -1.0
    >>> h.state[0] * h.state[1] * h.state[2]
    1
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("a model with no variables can hold no term")
        self.n = int(n)
        self._h = _hubo_new(self.n)
        if not self._h:
            raise RuntimeError("could not allocate a higher-order model")

    @classmethod
    def from_sim(cls, sim: "Sim") -> "Hubo":
        """Lift a pairwise simulation, unchanged, so both paths can score the same state."""
        h = cls.__new__(cls)
        h._h = _hubo_from_sim(sim._h)
        if not h._h:
            raise RuntimeError("could not lift this simulation")
        h.n = _hubo_len(h._h)
        return h

    def _live(self) -> None:
        if getattr(self, "_h", None) is None:
            raise RuntimeError("this model was already freed")

    def _why(self) -> str:
        return _read_text(_hubo_error, self._h)

    def add(self, variables: Iterable[int], weight: float) -> "Hubo":
        """Add one term of any arity. Returns self, so calls chain.

        A variable out of range or repeated within the term raises: ``s * s = 1``, so a repeat
        would silently change the term's order rather than mean what was written.
        """
        self._live()
        _hubo_vars_clear(self._h)
        vs = [int(v) for v in variables]
        for v in vs:
            if _hubo_var(self._h, v) == 0:
                raise ValueError(f"rejected term {vs}: {self._why()}")
        if _hubo_add(self._h, float(weight)) == 0:
            raise ValueError(f"rejected term {vs} at weight {weight}: {self._why()}")
        return self

    def anneal(self, beta_min: float = 0.0, beta_max: float = 0.0, stages: int = 0,
               sweeps_per_stage: int = 0, seed: int = 1) -> float:
        """Anneal and return the best energy. Zero for any ladder parameter means its default."""
        self._live()
        e = _hubo_anneal(self._h, float(beta_min), float(beta_max), int(stages),
                         int(sweeps_per_stage), int(seed))
        if e != e:  # NaN
            raise ValueError(f"could not anneal: {self._why()}")
        return e

    @property
    def state(self) -> List[int]:
        """The current state, copied out. Never a view into the library's memory."""
        self._live()
        buf = (ctypes.c_int8 * self.n)()
        if _hubo_read(self._h, buf, self.n) == 0:
            raise RuntimeError("could not read the state")
        return [int(v) for v in buf]

    @state.setter
    def state(self, spins: Sequence[int]) -> None:
        self._live()
        if len(spins) != self.n:
            raise ValueError(f"this model has {self.n} spins; {len(spins)} were offered")
        buf = (ctypes.c_int8 * self.n)(*[int(v) for v in spins])
        if _hubo_set_spins(self._h, buf, self.n) == 0:
            raise ValueError(self._why())

    @property
    def energy(self) -> float:
        """Energy of the current state.

        A property, like :attr:`Sim.energy`: a scalar readout of what the handle holds right now.
        """
        self._live()
        return float(_hubo_energy(self._h))

    def delta(self, i: int) -> float:
        """The energy change from flipping spin ``i``, in O(terms containing i)."""
        self._live()
        d = _hubo_delta(self._h, int(i))
        if d != d:
            raise IndexError(f"no variable {i}; {self.n} declared")
        return d

    @property
    def terms(self) -> int:
        self._live()
        return int(_hubo_terms(self._h))

    @property
    def max_arity(self) -> int:
        self._live()
        return int(_hubo_max_arity(self._h))

    @property
    def ancillas_avoided(self) -> int:
        """An UPPER BOUND on what a pairwise reduction would have spent, not the cost.

        ``reduce`` substitutes the commonest pair first, so one ancilla serves every term containing
        it: on three terms sharing one pair it spends one where this reports three.
        """
        self._live()
        return int(_hubo_ancillas_avoided(self._h))

    @property
    def proposals(self) -> int:
        self._live()
        return int(_hubo_proposals(self._h))

    @property
    def accepted(self) -> int:
        self._live()
        return int(_hubo_accepted(self._h))

    def joules_z1(self) -> float:
        """What the last run WOULD have cost on a Z1-class device (vendor SPICE, pre-silicon)."""
        self._live()
        return _hubo_joules_z1(self._h)

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return (f"<ferrotherm.Hubo {self.n} spins, {self.terms} terms, "
                f"max arity {self.max_arity}>")

    def __del__(self) -> None:
        if getattr(self, "_h", None):
            _hubo_free(self._h)
            self._h = None


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

    def sweep(self, n: int = 1, threads: int = 1) -> int:
        """Run `n` chromatic block-Gibbs sweeps. Returns total sweeps so far.

        ``threads`` above 1 spreads each colour class across OS threads -- no two nodes in a class
        are adjacent, so the split is race-free by construction of the colouring. ``threads=0``
        means *ask the machine*, which is :func:`hardware_threads`; on an 18-core box that is the
        difference between one core and eighteen, and one core is the commonest way this library
        is left slow.

        **The thread count is part of the run.** The result is bit-reproducible for a fixed
        ``(seed, threads)`` pair, and a different thread count is a different -- equally valid --
        sample path. Record the thread count beside the seed, or the run is not reproducible from
        what you wrote down. :attr:`threads_used` reports what actually ran.
        """
        self._live()
        if threads == 1:
            return int(_sweep(self._h, int(n)))
        return int(_sweep_par(self._h, int(n), int(threads)))

    @property
    def threads_used(self) -> int:
        """How many threads the last parallel sweep actually used, or 0 before one.

        Not the number you passed in: a browser has no threads to spread across, and a colour class
        of three nodes cannot occupy eight workers.
        """
        self._live()
        return int(_threads_used(self._h))

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

    @spins.setter
    def spins(self, state: "Sequence[int]") -> None:
        """Put a state INTO the simulation, so something computed elsewhere is scored, certified or
annealed by exactly the same code that handles a state this library produced.

        It refuses rather than adapting: the length must match the graph and every value must be -1 or +1.
A shorter state means whatever produced it did not finish, and a value that is not a spin means the
buffer is not what the caller thinks it is. Both are cheap to launder into something plausible --
pad with -1, coerce with `v > 0` -- and a laundered state is then scored with full confidence, which
is how a dropped GPU dispatch turns into a believable energy.
        """
        self._live()
        n = len(self)
        buf = (c_int8 * len(state))()
        for i, v in enumerate(state):
            v = int(v)
            if v not in (-1, 1):
                raise ValueError(f"states are -1/+1; got {v} at index {i}")
            buf[i] = v
        if not _set_spins(self._h, buf, len(state)):
            raise ValueError(
                f"this simulation has {n} nodes and that state has {len(state)}"
            )

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

    def _beta_or_default(self) -> float:
        """The beta to stamp on a Sim derived from this one.

        `_beta` is set by the module's own factories; a Sim built through :class:`Model` does not
        pass through one, so reading the attribute directly raised `AttributeError` on a perfectly
        ordinary path. Derived simulations carry the parent's beta where there is one and the
        library's default otherwise.
        """
        return float(getattr(self, "_beta", 1.0))

    def site_lower_bound(self, hardware: "Sim") -> int:
        """The fewest sites **any** embedding of this model onto ``hardware`` could use.

        A proof, not a heuristic. A chain of *L* sites on a degree-*d* machine offers at most
        ``L(d-2)+2`` ports, so a variable of degree *k* needs ``ceil((k-2)/(d-2))`` sites however
        cleverly it is placed. **When this exceeds the machine's site count, no embedding exists** —
        and asking costs microseconds where :meth:`embed` would spend its whole budget finding out.
        """
        self._live()
        hardware._live()
        return int(_site_lower_bound(self._h, hardware._h))

    def clique_embed(self, hardware: "Sim") -> int:
        """Store a **closed-form** structured clique embedding, where ``hardware`` has a known one.

        Where :meth:`embed` searches, this writes the answer down: ``K_n`` with uniform chains and no
        search. Supported today for **Pegasus** (``K_{12(m-2)+4}``, chains ≤ ``m+1`` — ``K_172`` on the
        Advantage's P₁₆, where the search reaches ``K_80``) and **Zephyr** (``K_{2t(2m-1)}``, chains
        ``m+1`` — ``K_232`` on Z₁₅, the busclique frontier size exactly).
        The clique size is fixed by the machine; this returns it. The placement lands on ``self``
        just as :meth:`embed` does, so :meth:`embed_apply`, :meth:`unembed` and the ``embed_*``
        readers work unchanged.

        >>> hw = zephyr(4)
        >>> problem = Model(56).build(beta=0.5)   # a K_56 on the 56 variables it will place
        >>> problem.clique_embed(hw)
        56
        >>> problem.embed_longest                 # uniform m+1 = 5
        5

        On Zephyr this IS the maximum ``busclique`` reaches on a perfect fabric, at the same chain
        length; on Pegasus it is within 5% (the last eight chains of ``K_{12(m-1)}`` need the
        staggered-fragment diagonal this crate does not perform). Either way it beats this crate's
        own search decisively and is instant.
        Raises when the topology has no known construction; :meth:`embed` is then the fallback.
        """
        self._live()
        hardware._live()
        n = c_uint32(0)
        if _clique_embed(self._h, hardware._h, ctypes.byref(n)) == 0:
            raise ValueError("no closed-form clique embedding is known for that topology; use embed()")
        return int(n.value)

    def embed(self, hardware: "Sim", seed: int = 0, rounds: int = 0, budget: int = 0) -> bool:
        """Place this model onto ``hardware``, keeping the placement for the accessors below.

        >>> k = Grid  # doctest: +SKIP

        **False never means "impossible."** It means this heuristic did not find a placement, which
        is a fact about the search. :meth:`site_lower_bound` is the question with a proof behind it.

        ``rounds`` of rip-up and reroute and ``budget`` shortest-path searches; 0 for either takes a
        default. A large machine wants a larger budget — saying "no" is not free.
        """
        self._live()
        hardware._live()
        return _embed(self._h, hardware._h, int(seed), int(rounds), int(budget)) == 1

    @property
    def embed_sites(self) -> int:
        """Physical sites the placement uses in total, or 0 if there is none."""
        self._live()
        return int(_embed_sites(self._h))

    @property
    def embed_longest(self) -> int:
        """The longest chain — the number that decides whether an answer survives.

        Sites are a budget and you either have them or you do not. A chain is a *failure mode*: it
        is held together by a coupling, and when that coupling loses, the sites of one variable
        disagree and the variable has no value at all. Halving this is worth more than halving
        :attr:`embed_sites`.
        """
        self._live()
        return int(_embed_longest(self._h))

    def chain(self, v: int) -> "list[int]":
        """The sites holding logical variable ``v``."""
        self._live()
        need = int(_embed_chain(self._h, int(v), None, 0))
        if need == 0:
            raise IndexError(f"no chain for variable {v}; embed() first, or it is out of range")
        buf = (c_uint32 * need)()
        got = _embed_chain(self._h, int(v), buf, need)
        return [int(x) for x in buf[:got]]

    def embed_apply(self, hardware: "Sim", chain_strength: float = 0.0) -> "Sim":
        """Build the model that actually **runs** on the hardware, from a placement already found.

        A simulation over the hardware's sites, each chain bound by a coupling strong enough to hold
        it; ``chain_strength`` of 0 takes the derived default. The placement rides along, so
        :meth:`unembed` works on the result.
        """
        self._live()
        hardware._live()
        return _wrap(_embed_apply(self._h, hardware._h, float(chain_strength)),
                     self._beta_or_default(),
                     "the embedded model (embed() first)")

    def unembed(self, variables: int) -> "tuple[list[int], int]":
        """Read this embedded state back as a logical one, with the count of chains that **broke**.

        A variable whose chain broke has two values at once and therefore none. The majority is
        returned so there is still a complete state, and the count says how much of it to distrust:
        non-zero means the chain coupling lost to the problem, and the answer is a stronger coupling
        or a shorter chain.
        """
        self._live()
        buf = (c_int8 * int(variables))()
        broken = int(_unembed(self._h, buf, int(variables)))
        if broken == 0xFFFFFFFF:
            raise ValueError("this simulation carries no placement, or the buffer is too small")
        return ([int(x) for x in buf], broken)

    def sparsify(self, budget: int) -> "Sim":
        """Rewrite this model so no variable has more than ``budget`` neighbours.

        A model denser than a fabric has two routes onto it: **embedding** places it onto one
        specific machine, and **sparsification** rewrites it — splitting each heavy variable into
        copies bound by a strong coupling — so any degree-``budget`` fabric can take it, with no
        machine involved.

        >>> dense = z1_grid(8, 8)          # degree 16 in the interior
        >>> sparse = dense.sparsify(6)
        >>> len(dense), len(sparse)
        (64, 148)
        >>> sparse.logical_variables
        64
        >>> len(sparse.copies(27))         # a centre node needs three
        3

        **Measured, and the answer is a negative one: where a placer exists, place.** `K_24` onto a
        Pegasus P16 costs 130 sites and a 14-site chain placed directly, against 758 sites and a
        55-site run through sparsification — the same tax paid twice, because copies are chosen
        before the machine is looked at and each is then chained anyway. This is for a fabric with a
        fixed sparse topology and *no* placer, where there is no direct route at all.

        The original is untouched. Raises for a budget below 3: a path of copies offers
        ``c(d-2)+2`` ports, which does not grow with ``c`` below 3.
        """
        self._live()
        if budget < 3:
            raise ValueError(
                f"a degree budget of {budget} cannot be met by splitting: a path of copies offers "
                "c(d-2)+2 ports and that is not increasing in c below d = 3"
            )
        return _wrap(_sparsify(self._h, int(budget)), self._beta_or_default(),
                     "the sparsified model")

    @property
    def logical_variables(self) -> int:
        """Logical variables this model stands for, or 0 if it was not produced by :meth:`sparsify`."""
        self._live()
        return int(_sparsify_variables(self._h))

    def copies(self, v: int) -> "list[int]":
        """The nodes representing logical variable ``v``."""
        self._live()
        need = int(_sparsify_copies(self._h, int(v), None, 0))
        if need == 0:
            raise IndexError(f"no logical variable {v}; this model has {self.logical_variables}")
        buf = (c_uint32 * need)()
        got = _sparsify_copies(self._h, int(v), buf, need)
        return [int(x) for x in buf[:got]]

    @property
    def sparsify_offset(self) -> float:
        """``E_logical = E_sparse + offset`` when every copy set agrees.

        The copy couplings contribute the same constant in every agreeing state, so reporting a
        sparsified energy without this compares a number from one model against a number from
        another.
        """
        self._live()
        return float(_sparsify_offset(self._h))

    def project(self) -> "tuple[list[int], int]":
        """Read the current state back as a logical one, with the count of variables that BROKE.

        A variable whose copies disagree has not been assigned a value; the majority is returned so
        there is still a complete state to look at, and the count says how much of it to distrust.
        Non-zero means the copy coupling lost.
        """
        self._live()
        n = self.logical_variables
        if n == 0:
            raise ValueError("this model was not produced by sparsify(), so it has no logical state")
        buf = (c_int8 * n)()
        broken = int(_sparsify_project(self._h, buf, n))
        if broken == 0xFFFFFFFF:
            raise RuntimeError("could not project this state")
        return ([int(x) for x in buf], broken)

    def qubit(self, i: int) -> "int | None":
        """The **vendor's** linear qubit index for node ``i``, or ``None`` if this graph has none.

        Two numbering systems meet here. Everything in this library indexes ``0..n`` densely;
        Pegasus drops the qubits outside its largest component, so its own numbering is sparse — a
        ``P16`` spreads 5,640 qubits over indices 30 to 5,729. A chain written in our indices and
        handed to a machine programs *different qubits*, and the answer comes back looking like a
        bad embedding rather than like the mistake it is.

        ``None`` rather than 0, because 0 is a valid qubit.
        """
        self._live()
        q = int(_qubit(self._h, int(i)))
        return None if q == 0xFFFFFFFF else q

    @property
    def joules(self) -> float:
        """Ledger priced at Z1-class device figures. This prices the modelled device, not your CPU."""
        self._live()
        return float(_ledger_joules(self._h))

    # -- lifetime --

    def collect(self, draws: int = 512, thin: int = 1, burn_in: int = 0) -> "SampleSet":
        """Draw states and keep them.

        Every solver here returns one state, which is an optimiser's answer. This is a sampler:
        expectation values, degeneracy evidence and the moments an energy-based model is trained on
        all need many. The run is certified at the same time -- read :attr:`certificate` after.

        It charges the ledger for the READBACK as well as the sweeps. A Z1-class read is 1.692 pJ
        per node against 7.09 fJ per Gibbs cycle, so a read is worth 239 updates; a collection loop
        that does not charge for it reports the larger half of its own energy bill as zero.
        """
        self._live()
        if draws < 16:
            raise ValueError("certifying fewer than 16 draws says nothing")
        if _collect(self._h, int(max(0, burn_in)), int(draws), int(max(1, thin))) == 0:
            raise RuntimeError("could not collect from this simulation")
        return SampleSet(self)

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

    def exact_ground_state(self, max_width: int = 22) -> "list[int] | None":
        """The exact ground STATE, or ``None`` if the graph is too dense.

        :meth:`exact_ground_energy` says what the best energy is; this says which assignment reaches
        it, which is what a caller checking a sampler against the truth actually needs to compare.
        Same elimination and same width limit.
        """
        self._live()
        n = len(self)
        out = (c_int8 * n)()
        if not _exact_ground_state(self._h, int(max_width), out, n):
            return None
        return [int(v) for v in out]

    def exact_log_z(self, beta: float = 1.0, max_width: int = 22) -> float | None:
        """Exact log partition function, or ``None`` if too dense."""
        self._live()
        v = float(_exact_log_z(self._h, float(beta), int(max_width)))
        return None if v != v else v

    def exact_marginals(self, beta: float = 1.0, max_width: int = 22) -> "list[float] | None":
        """Exact single-site ``P(s_i = +1)``, or ``None`` if the graph is too wide.

        **The referee.** Compare a sampler's histogram against these on a graph far past where
        enumeration stops: a 42-spin strip is 2^42 states and width 3. :meth:`verify` compares
        against exhaustive enumeration and stops near twenty spins; this does not.

        Costs ``2n`` eliminations -- ``O(n * 2^width)`` against the single ``O(2^width)`` of
        :meth:`exact_log_z` -- so read :attr:`treewidth` before asking for it on anything wide.
        """
        self._live()
        n = int(_len(self._h))
        buf = (c_double * n)()
        if _exact_marginals(self._h, float(beta), int(max_width), buf, n) == 0:
            return None
        return [float(v) for v in buf]

    # ---- solvers ---------------------------------------------------------------------------
    #
    # Each of these leaves its best state as this simulation's state, so :attr:`spins` reads the
    # answer and :attr:`energy` recomputes it. They compose: anneal, then tabu from where annealing
    # stopped, then :meth:`branch` with that as its incumbent.

    def tabu(self, iterations: int = 50_000, tenure: int = 0,
             restart_after: int = 5_000) -> float:
        """Tabu search. Returns the best energy found and leaves that state behind.

        ``tenure=0`` scales the tenure to the graph; ``restart_after=0`` never restarts. Check
        :meth:`tabu_iterations` afterwards — a run shorter than the budget was truncated, and that
        is otherwise invisible.
        """
        self._live()
        return float(_tabu(self._h, int(iterations), int(tenure), int(restart_after)))

    def hfs(self, steps: int = 400, block: int = 0) -> float:
        """Hamze-de Freitas-Selby: solve a low-treewidth **block** exactly, repeatedly.

        Every other local search here flips one spin and asks whether that helped.
        This takes the exact best assignment of a whole subgraph with everything outside it held
        fixed, so it steps over any barrier living entirely inside the block rather than paying to
        climb it. It is the algorithm that turned the first generation of quantum-annealer speedup
        claims.

        Starts from this simulation's **current** state, so it composes — anneal, then
        :meth:`tabu`, then this — and being a descent it can never undo what found that state.
        ``block=0`` takes the default. Blocks are grown as induced trees, width 1 by construction,
        so nothing here is refused for width.

        Read :meth:`hfs_improving` afterwards: a run whose blocks all land on a minimum they already
        sit in has stopped, and the energy alone does not show that.
        """
        self._live()
        return float(_hfs(self._h, int(steps), int(block)))

    def hfs_moves(self) -> int:
        """Block moves the last :meth:`hfs` ran, or 0 if there was none."""
        self._live()
        return int(_hfs_moves(self._h))

    def hfs_improving(self) -> int:
        """Block moves that strictly lowered the energy, or 0 if there was none."""
        self._live()
        return int(_hfs_improving(self._h))

    # ---- fitting a model to data ---------------------------------------------------------------

    def fit(self, rows: Sequence[Sequence[int]], visible: int = 0, epochs: int = 0,
            k: int = 0, positive_sweeps: int = 0, learning_rate: float = 0.0,
            batch: int = 0, seed: int = 0) -> "Sim":
        """Fit this model's weights to ``rows`` by contrastive divergence. Returns ``self``.

        Every other method here takes the model as given and samples, optimises or bounds it. This
        one **produces** one, and it is why a thermodynamic stack is a paradigm rather than a
        solver: the argument for this hardware is that it samples Boltzmann distributions cheaply,
        and the distributions anyone wants are *fitted*.

        ``rows`` is a sequence of ``±1`` rows; ``visible`` defaults to the row width. The edge set
        is kept and the weights are overwritten, so build the structure with :func:`rbm`,
        :func:`dbm`, or any graph of your own.

        **This replaces the model, so every cached result about the old one is dropped** —
        certificates, tabu and branch outcomes, the GPU model. A certificate proved against the
        weights before training is a true statement about a model that no longer exists. The spin
        state survives; it is a state of the same spins and a fine start for sampling the fit.

        Zero means the documented default for every knob. The learning rate **decays to a tenth**
        across training — without that the fit has a noise floor and never reaches its own fixed
        point.

        >>> import ferrotherm as ft
        >>> data = ft.bars_and_stripes(2)
        >>> m = ft.rbm(4, 4, seed=11)
        >>> before = m.log_likelihood(data)
        >>> after = m.fit(data, epochs=600, k=10, seed=3).log_likelihood(data)
        >>> after > before
        True
        """
        self._live()
        flat, width, n = _flatten_rows(rows, visible)
        ok = _ebm_train(self._h, width, flat, n, int(epochs), int(k), int(positive_sweeps),
                        float(learning_rate), int(batch), int(seed))
        if not ok:
            raise ValueError(_ebm_why() or "that dataset could not be fitted")
        return self

    def log_likelihood(self, rows: Sequence[Sequence[int]], visible: int = 0) -> float:
        """Mean log-likelihood per row under this model, **exact**, by enumeration.

        Never an ELBO, a reconstruction error or a pseudo-likelihood: each of those is worst
        exactly where sampling is worst, so comparing models on one reads the proxy's failure and
        calls it expressivity. Above 22 spins this **raises** rather than returning something
        cheaper.

        The scale has fixed ends and needs no calibration — a model that has learned nothing scores
        ``-visible * ln 2``, and one reproducing ``n`` equiprobable rows scores ``-ln n``.
        """
        self._live()
        flat, width, n = _flatten_rows(rows, visible)
        v = float(_ebm_log_likelihood(self._h, width, flat, n))
        if v != v:
            raise ValueError(_ebm_why() or "that likelihood could not be computed")
        return v

    def tabu_iterations(self) -> int:
        """Iterations the last :meth:`tabu` actually ran, or 0 if there was none."""
        self._live()
        return int(_tabu_iterations(self._h))

    def exact_planar(self, scale: float = 1.0) -> "PlanarCut":
        """**Exact** max-cut, in polynomial time, if this graph is planar.

        Not a search. Max-cut is NP-hard in general and polynomial on a planar graph, and the
        difference is a theorem: a cut in the graph is a cycle in the dual, so the problem becomes
        a minimum-weight T-join and then a minimum-weight perfect matching. There is no budget to
        run out of and the answer is the maximum, not the best found.

        Raises :class:`ValueError` with the reason if the graph cannot be solved this way — there
        are four, and they are four different things to do next.

        On success the simulation's state becomes the optimal partition, so :attr:`energy` is then
        the proved **minimum**.
        """
        self._live()
        cut = float(_planar_cut(self._h, float(scale)))
        if cut != cut:
            raise ValueError(_read_text(_planar_error, self._h) or "the planar solver refused")
        return PlanarCut(
            cut=cut,
            energy=self.energy,
            faces=int(_planar_faces(self._h)),
            odd_faces=int(_planar_odd_faces(self._h)),
        )

    def toroidal_bound(self, scale: float = 1.0) -> "ToroidalBound":
        """An **upper bound** on the maximum cut of a toroidal grid.

        The side of the G-set table nobody publishes. Every figure there is a best cut *found* — a
        lower bound. This is the other end of the bracket, from the same dual reduction run on a
        toroidal embedding: on a torus the cycle space of the dual is four times the cut space, so
        the relaxation ranges over sets that are not cuts and its optimum bounds the maximum above.

        Measured, it closes the bracket on G11: bound 564, published best 564, so 564 is optimal.

        Raises :class:`ValueError` unless the graph is a toroidal grid — the structure is recovered
        from the edge list, and a match on all ``2n`` edges is a proof rather than a guess.
        """
        self._live()
        cut = float(_toroidal_bound(self._h, float(scale)))
        if cut != cut:
            raise ValueError(
                "not a toroidal grid, or the weights do not scale to integers"
            )
        return ToroidalBound(cut=cut, attained=bool(_toroidal_attained(self._h)))

    def goemans_williamson(self, hyperplanes: int = 64, seed: int = 1) -> "Rounded":
        """Round the semidefinite relaxation to a state — the only guarantee in max-cut.

        :meth:`bounds` uses the relaxation from the dual side to produce a bound; this uses it from
        the primal side to produce a solution. Check :attr:`Rounded.guaranteed`: the 0.87856 ratio
        is stated for non-negative edge weights, which here means non-positive couplings and no
        fields, and it is false on most instances people care about.
        """
        self._live()
        cut = float(_gw_round(self._h, int(hyperplanes), int(seed)))
        return Rounded(cut=cut, energy=self.energy, guaranteed=bool(_gw_guaranteed(self._h)))

    def cluster_anneal(self, rungs: int = 16, rounds: int = 400,
                       beta_min: float = 0.1, beta_max: float = 6.0) -> "ClusterRun":
        """Parallel tempering with isoenergetic cluster moves — the field's baseline.

        Raises :class:`ValueError` on a graph with fields: the move preserves the pair energy only
        at ``h = 0``, and accepting it anyway would be silently wrong.
        """
        self._live()
        e = float(_icm(self._h, int(rungs), int(rounds), float(beta_min), float(beta_max)))
        if e != e:
            raise ValueError(
                "the isoenergetic argument holds only at h = 0, so a graph with fields is refused; "
                "use anneal() or tabu(), or check that 0 < beta_min < beta_max"
            )
        return ClusterRun(energy=e, moves=int(_icm_moves(self._h)))

    def quantum_anneal(self, trotter: int = 4, beta: float = 10.0, gamma_max: float = 3.0,
                       gamma_min: float = 0.05, steps: int = 200) -> float:
        """Simulated quantum annealing: path-integral Monte Carlo, not a quantum computer.

        ``trotter=1`` drops the Trotter coupling and is exactly classical annealing — the honest
        control to compare against, rather than a degenerate case.
        """
        self._live()
        e = float(_sqa(self._h, int(trotter), float(beta), float(gamma_max), float(gamma_min),
                       int(steps)))
        if e != e:
            raise ValueError("need beta > 0 and gamma_max >= gamma_min >= 0")
        return e

    def breakout(self, iterations: int = 50_000) -> "BreakoutRun":
        """Breakout local search: steepest descent, with an adaptive perturbation between optima.

        The algorithm that holds the max-cut record on most of G-set. One iteration is one spin
        flip — the same unit :meth:`tabu` counts — so giving both the same number is a
        matched-budget comparison, which is the only comparison that is honest without a quiet
        machine.

        Check :attr:`BreakoutRun.descents`: a run with a handful of them spent its budget inside
        one basin and is a descent, not a breakout search.
        """
        self._live()
        e = float(_bls(self._h, int(iterations)))
        return BreakoutRun(
            energy=e,
            descents=int(_bls_descents(self._h)),
            iterations_run=int(_bls_iterations(self._h)),
            max_jump=int(_bls_max_jump(self._h)),
        )

    def population_anneal(self, population: int = 1_000, sweeps: int = 4,
                          beta_max: float = 6.0, stages: int = 100) -> "PopulationRun":
        """Population annealing: ``population`` chains down one ladder, resampled at each rung.

        Unlike a single annealed chain this also estimates the free energy, and reports whether to
        believe it. The ladder starts at ``beta = 0``, where ``Z = 2 ** n`` exactly, which is what
        makes :attr:`PopulationRun.ln_z` absolute rather than a ratio.
        """
        self._live()
        e = float(_popanneal(self._h, int(population), int(sweeps), float(beta_max), int(stages)))
        ln_z = float(_popanneal_ln_z(self._h))
        return PopulationRun(
            energy=e,
            ln_z=None if ln_z != ln_z else ln_z,
            rho=float(_popanneal_rho(self._h)),
            population=int(population),
        )

    def branch(self, max_nodes: int = 20_000_000) -> "BranchResult":
        """Branch and bound from the current state, which it uses as its incumbent.

        The only solver here that returns a **proof**. A run that exhausts ``max_nodes`` returns the
        best state it saw with ``proved=False``; a flag meaning "optimal, or else we gave up" would
        get read as the first thing and quoted as the second.
        """
        self._live()
        e = float(_branch(self._h, int(max_nodes)))
        return BranchResult(
            energy=e,
            proved=bool(_branch_proved(self._h)),
            nodes=int(_branch_nodes(self._h)),
        )

    # ---- bounds ----------------------------------------------------------------------------

    def bounds(self, forest_rounds: int = 40, max_cycle: int = 6,
               sdp_sweeps: int = 200, seed: int = 1) -> "Bounds":
        """Every lower bound on the ground energy this library can compute, and the best of them.

        All four are sound on their own, so :attr:`Bounds.best` is their maximum — not a tie-break,
        a result: they disagree by a lot and in both directions. ``odd_cycle`` wins on sparse
        frustrated lattices and ``sdp`` wins by more the denser the instance gets.

        ``sdp`` is re-verified on the Rust side of the boundary before it crosses, and comes back
        as ``None`` if that verification fails.
        """
        self._live()
        sdp = float(_bound_sdp(self._h, int(sdp_sweeps), int(seed)))
        return Bounds(
            decoupled=float(_bound_decoupled(self._h)),
            forest=float(_bound_forest(self._h, int(forest_rounds))),
            odd_cycle=float(_bound_odd_cycle(self._h, int(max_cycle))),
            sdp=None if sdp != sdp else sdp,
        )

    def gap(self, **kw: Any) -> float:
        """How far this simulation's current state is from optimal, at worst.

        ``energy - best_bound``. Zero means the state is **proved** optimal without trusting the
        sampler that found it; anything above is an upper limit on what a better search could still
        win. Takes the same keyword arguments as :meth:`bounds`.
        """
        return self.energy - self.bounds(**kw).best

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


def pegasus(m: int = 16, j: float = 1.0, beta: float = 1.0, seed: int = 0) -> Sim:
    """The **Pegasus** graph ``P_m`` — the topology of every D-Wave *Advantage* processor.

    ``m = 16`` is the Advantage: 5,640 qubits, 40,484 couplers, degree 15.

    >>> s = pegasus(16)
    >>> len(s)
    5640
    >>> s.qubit(0)          # the VENDOR's number for our node 0, and it is not 0
    30

    This is the nominal full-yield graph, not a particular machine's working graph: a real QPU has
    qubits and couplers missing from fabrication, so a program that embeds here is not guaranteed to
    fit the machine in front of you. It is the right target for *could this fit at all*.
    """
    if m < 2:
        raise ValueError(f"P_{m} has no qubits; Pegasus starts at m = 2")
    return _wrap(_pegasus_new(int(m), float(j), float(beta), int(seed)), beta, f"P_{m}")


def zephyr(m: int = 15, t: int = 4, j: float = 1.0, beta: float = 1.0, seed: int = 0) -> Sim:
    """The **Zephyr** graph ``Z_{m,t}`` — the topology of D-Wave's *Advantage2* processors.

    ``m = 15, t = 4`` is the Advantage2: 7,440 qubits, 71,736 couplers, degree 20. ``t`` is 4 on
    every shipped machine.

    >>> s = zephyr(15)
    >>> len(s)
    7440

    Zephyr's higher degree is what it is for: the same problem embeds with shorter chains, and a
    chain that breaks leaves a variable with no value at all.
    """
    if m < 1 or t < 1:
        raise ValueError("Zephyr needs m >= 1 and t >= 1")
    return _wrap(_zephyr_new(int(m), int(t), float(j), float(beta), int(seed)), beta, f"Z_{m},{t}")


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


_ENCODINGS = {"one-hot": 0, "onehot": 0, "binary": 1, "domain-wall": 2, "domainwall": 2}


def _encoding(name: str) -> int:
    """An encoding name as its code, naming the alternatives when it is not one."""
    try:
        return _ENCODINGS[str(name).lower()]
    except KeyError:
        raise ValueError(
            f"unknown encoding {name!r}; try one-hot, domain-wall or binary"
        ) from None


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

    __slots__ = ("detail", "by", "hard")

    def __init__(self, detail: str, by: float, hard: bool = True) -> None:
        self.detail, self.by, self.hard = detail, float(by), bool(hard)

    def __str__(self) -> str:
        return self.detail

    def __repr__(self) -> str:
        kind = "Violation" if self.hard else "Traded"
        return f"<{kind} by {self.by:g}: {self.detail}>"


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

    :attr:`objective` is what the answer is WORTH, in your own units and the direction you wrote
    it in. :attr:`energy` is the compiled Ising energy with every penalty and the constant folded
    in — a number about spins, which compares two answers to one model and nothing else, and which
    moves when the penalty does. Write ``maximize 5*mon + 4*tue``, get ``mon = 1, tue = 2``, and
    ``objective`` reads 9 while ``energy`` reads some number in the hundreds. ``objective`` is
    ``None`` when no objective was written, when both senses were used and there is no single
    direction to report, or when a variable did not decode and there is only half an answer.

    :attr:`proved_optimal` is ``True`` only when :meth:`Problem.solve` was asked for ``"branch"``
    and it exhausted the tree. **Read it with** :attr:`feasible`: branch proves a statement about the
    compiled energy, and it becomes a statement about *your* model exactly when the answer is also
    feasible — a feasible assignment pays no penalty, so its compiled energy is the objective plus a
    constant. Proved and feasible is a real optimality proof and needs nothing from the penalty being
    large enough. Proved and *infeasible* says the penalty is too small and no longer search fixes it.

    :attr:`caveats` lists what the compiler knows is wrong with the model and cannot fix — today,
    an encoding no penalty can make exact. Empty is the normal case; a non-empty one means a value
    that reads back fine may still have come from a codeword the penalty never excluded.

    :attr:`ancillas` is non-zero when a term named three or more variables: the model that was
    solved has more spins than the variables required. Those extra states make the lowering exact
    for **optimisation** and not for sampling, so read it before drawing samples rather than only a
    ground state.
    """

    __slots__ = ("values", "feasible", "energy", "objective", "proved_optimal", "spins", "penalty",
                 "violated", "soft_cost", "ancillas", "caveats")

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
        for c in self.caveats or ():
            out += f"\n  caveat: {c}"
        for v in self.violated or ():
            word = "broken" if v.hard else "traded"
            out += f"\n  {word}: {v.detail} (by {v.by:g})"
        if self.soft_cost:
            out += f"\n  soft cost: {self.soft_cost:g}"
        return out + ">"


def from_ommx(data: bytes, beta: float = 1.0, seed: int = 0) -> "tuple[Sim, float]":
    """Read an ``ommx.v1.Instance`` and return a simulation over it, plus the constant.

    The direction that makes this a bridge rather than an exporter: a problem someone else compiled
    to OMMX — from jijmodeling, say — becomes something this sampler can run.

    ``ommx_objective(x) == sim.energy + constant``. Dropping the constant leaves an energy that ranks
    states correctly and reports the wrong number.

    Raises :class:`ValueError` naming what could not be read: a continuous variable, a bound that is
    not ``[0, 1]``, an objective of degree three or more, or **constraints** — ferrotherm expresses
    a constraint as a penalty whose weight changes the answer, so reading the objective alone would
    return the relaxation. This sampler samples spins, and a bridge that silently dropped what it
    could not represent would return a model solving a different problem.

    >>> import ferrotherm as ft                       # doctest: +SKIP
    >>> sim, constant = ft.from_ommx(open("p.ommx", "rb").read(), beta=1.0)   # doctest: +SKIP
    """
    buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    constant = c_double(0.0)
    h = _ommx_read(buf, len(data), float(beta), int(seed), ctypes.byref(constant))
    if not h:
        need = _ommx_error(None, 0)
        why = ""
        if need:
            eb = (ctypes.c_ubyte * need)()
            got = _ommx_error(eb, need)
            why = bytes(bytearray(eb)[:got]).decode("utf-8", "replace")
        raise ValueError(why or "that is not an instance this sampler can read")
    return Sim(h), float(constant.value)


# ---- fitting a model to data ---------------------------------------------------------------------


def _ebm_why() -> str:
    """The reason the last ``ft_ebm_*`` call refused, in the caller's own terms."""
    need = _ebm_error(None, 0)
    if not need:
        return ""
    buf = (ctypes.c_ubyte * need)()
    got = _ebm_error(buf, need)
    return bytes(bytearray(buf)[:got]).decode("utf-8", "replace")


def _flatten_rows(rows: Sequence[Sequence[int]], visible: int) -> tuple:
    """Rows of ±1 into one C array, with the width checked here rather than across the boundary."""
    rows = [list(r) for r in rows]
    if not rows:
        raise ValueError("no data rows; there is nothing to fit")
    width = int(visible) or len(rows[0])
    for i, r in enumerate(rows):
        if len(r) != width:
            raise ValueError(f"row {i} has {len(r)} entries, and the dataset declares {width}")
        for j, v in enumerate(r):
            if v not in (-1, 1):
                raise ValueError(f"row {i} position {j} is {v}, and a spin is -1 or +1")
    flat = (c_int8 * (len(rows) * width))()
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            flat[i * width + j] = int(v)
    return flat, width, len(rows)


def rbm(visible: int, hidden: int, beta: float = 1.0, seed: int = 0) -> Sim:
    """A restricted Boltzmann machine's **structure**: complete bipartite, every weight zero.

    Give the weights meaning with :meth:`Sim.fit`. Visible units are spins ``0..visible``, which is
    what :meth:`Sim.fit` and :meth:`Sim.log_likelihood` assume when they clamp a row on.
    """
    h = _ebm_rbm(int(visible), int(hidden), float(beta), int(seed))
    if not h:
        raise ValueError(_ebm_why() or "that is not a machine this library can build")
    return Sim(h)


def dbm(visible: int, layers: Sequence[int], beta: float = 1.0, seed: int = 0) -> Sim:
    """A deep Boltzmann machine's structure: ``visible`` spins, then each layer, chained.

    One layer is exactly :func:`rbm`. More layers add latent units **without** scaling any unit's
    connectivity, which is the arrangement the mixing-expressivity tradeoff is a claim about —
    ``examples/trained_tradeoff`` measures the two against each other and finds that the claim's
    two halves do not both survive.
    """
    widths = [int(w) for w in layers]
    if not widths:
        raise ValueError("no hidden layers were given")
    arr = (c_uint32 * len(widths))(*widths)
    h = _ebm_dbm(int(visible), arr, len(widths), float(beta), int(seed))
    if not h:
        raise ValueError(_ebm_why() or "that is not a machine this library can build")
    return Sim(h)


def bars_and_stripes(side: int) -> list:
    """The ``side x side`` bars-and-stripes dataset, the standard tiny benchmark for fitting an EBM.

    >>> import ferrotherm as ft
    >>> rows = ft.bars_and_stripes(3)
    >>> len(rows), len(rows[0])
    (14, 9)
    """
    n = _ebm_bars_and_stripes(int(side), None, 0)
    if not n:
        raise ValueError(_ebm_why() or "that side is not one this library can build")
    width = int(side) * int(side)
    buf = (c_int8 * (n * width))()
    got = _ebm_bars_and_stripes(int(side), buf, n * width)
    if not got:
        raise ValueError(_ebm_why() or "the dataset could not be written")
    return [[int(buf[r * width + c]) for c in range(width)] for r in range(got)]


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

    def categorical(self, name: str, values: int, encoding: str = "one-hot") -> Variable:
        """A variable holding one of ``values`` distinct values.

        ``encoding`` decides how it is stored, and the trade is the difference between a model that
        fits a machine and one that does not:

        ==============  ====================  ===================================
        encoding        spins for *k* values  usable in a constraint or objective
        ==============  ====================  ===================================
        ``one-hot``     *k*                   yes
        ``domain-wall`` *k* − 1               yes
        ``binary``      ``ceil(log2 k)``      **no**
        ==============  ====================  ===================================

        A binary code's indicator is a product of every bit, so its degree grows with the domain.
        It is the cheapest to store and is refused by name if it appears in a literal, rather than
        expanded into something nobody wants to read.
        """
        return self._declare(
            name, "categorical",
            _model_categorical_as(self._h, int(values), _encoding(encoding)))

    def integer(self, name: str, lo: int, hi: int, encoding: str = "one-hot") -> Variable:
        """A variable over the inclusive range ``lo``..``hi``.

        There is no machine integer here. This is a categorical over the range, encoded so that
        neighbouring values differ in one spin; the name is for the modeller, not the fabric.
        """
        return self._declare(name, f"integer {lo}..{hi}",
                             _model_integer_as(self._h, int(lo), int(hi), _encoding(encoding)))

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

    def not_equal(self, a: Variable, b: Variable, soft: "float | None" = None) -> None:
        """``a`` and ``b`` must differ, or — with ``soft`` — had better.

        A ``soft`` price makes this a preference the solver may trade away: breaking it costs and
        the answer stays feasible. See :meth:`soften_last`.
        """
        self._must(_model_not_equal(self._h, a._index, b._index), "not_equal")
        if soft is not None:
            self.soften_last(soft)

    def equal(self, a: Variable, b: Variable, soft: "float | None" = None) -> None:
        """``a`` and ``b`` must agree, or — with ``soft`` — had better."""
        self._must(_model_equal(self._h, a._index, b._index), "equal")
        if soft is not None:
            self.soften_last(soft)

    def fix(self, v: Variable, value: int) -> None:
        """``v`` must take ``value``."""
        self._must(_model_fix(self._h, v._index, int(value)), "fix")

    def exactly(self, of: "Sequence[Any]", k: int, value: int = 1,
                    soft: "float | None" = None) -> None:
        """Exactly ``k`` of them hold."""
        self._counting(0, of, k, value, "exactly", soft)

    def at_most(self, of: "Sequence[Any]", k: int, value: int = 1,
                    soft: "float | None" = None) -> None:
        """At most ``k`` of them hold. Costs a slack variable."""
        self._counting(1, of, k, value, "at_most", soft)

    def at_least(self, of: "Sequence[Any]", k: int, value: int = 1,
                     soft: "float | None" = None) -> None:
        """At least ``k`` of them hold. Costs a slack variable."""
        self._counting(2, of, k, value, "at_least", soft)

    def exactly_one(self, of: "Sequence[Any]", soft: "float | None" = None) -> None:
        """Exactly one of them holds. Cheaper than ``exactly(of, 1)``: pairwise, with no slack."""
        self._counting(3, of, 0, 1, "exactly_one", soft)

    def at_most_one(self, of: "Sequence[Any]", soft: "float | None" = None) -> None:
        """At most one of them holds."""
        self._counting(4, of, 0, 1, "at_most_one", soft)

    def all_different(self, of: "Sequence[Variable]", soft: "float | None" = None) -> None:
        """Every one of these variables takes a different value.

        The workhorse of assignment, scheduling, colouring and puzzles. Lowered per shared value
        rather than per pair, so it costs nothing where two domains do not overlap, needs no slack
        and no ancillas, and its violation names *which* value collided and who took it.

        More variables than the values they share between them is refused when the model compiles,
        by name. That is the pigeonhole principle checked rather than annealed: such a model has no
        answer at any penalty, and reporting ``feasible: False`` after a full anneal would send you
        looking for a longer ladder that cannot help.
        """
        _model_lits_clear(self._h)
        seen = []
        for v in of:
            if not isinstance(v, Variable):
                raise TypeError(f"all_different takes variables, not {type(v).__name__}")
            if v._index in seen:
                continue
            seen.append(v._index)
            # ft_model_var, not ft_model_lit with a placeholder value: the library picks a value
            # from the variable's OWN domain, because a caller has no reason to know one and a
            # placeholder is refused for any variable whose domain does not contain it.
            self._must(_model_var(self._h, v._index), "all_different")
        if len(seen) < 2:
            raise ValueError("all_different needs at least two variables")
        self._must(_model_close(self._h, 5, 0), "all_different")
        if soft is not None:
            self.soften_last(soft)

    #: How a weighted row compares, and the code the C ABI uses for each.
    _RELATIONS = {"<=": 0, "\u2264": 0, ">=": 1, "\u2265": 1, "=": 2, "==": 2}

    def linear(self, terms: Any, rel: str, rhs: float,
               soft: "float | None" = None) -> None:
        """A **weighted** linear row: ``3*a + 4*b + 5*c <= 7``.

        The constraint none of the counting forms can express. ``exactly``, ``at_most``,
        ``at_least``, ``exactly_one`` and ``at_most_one`` all count *unweighted* literals, so a row
        with coefficients could not be stated at all — and the advice the LP reader used to give,
        "add it to the objective", is the defect rather than the workaround: an objective term is
        not a constraint, so :attr:`Answer.feasible` and :attr:`Answer.violated` stop knowing
        about the row.

        ``terms`` is either an expression built with arithmetic — ``3*a.is_(1) + 4*b.is_(1)`` — or a
        sequence of ``(variable-or-literal, coefficient)`` pairs. ``rel`` is ``"<="``, ``">="`` or
        ``"="``. ``soft`` prices the row instead of enforcing it, at ``weight × amount²`` in your
        own units.

        **What it costs.** An equality adds no spins. An inequality adds ``ceil(log2(S+1))`` slack
        spins, where ``S`` is the residual span after dividing the row through by the gcd of its
        weights — so ``1000*a + 1000*b <= 1500`` costs one spin, not 1500. Either way the row is a
        clique on its own ``n`` literals plus its ``m`` slack bits: ``(n+m)(n+m-1)/2`` couplings.
        The bill is ``n``, not the weights.

        **What it refuses**, when the model compiles, by name: a non-integer coefficient or
        right-hand side on an *inequality* (there is no integer residual for the slack to range
        over, and rounding would change which answers are answers — an *equality* takes any finite
        coefficient, because it needs no slack); and a row nothing can satisfy, by arithmetic
        rather than by annealing. A row that constrains nothing compiles to nothing and says so in
        :attr:`Answer.caveats`.
        """
        code = self._RELATIONS.get(str(rel).strip())
        if code is None:
            raise ValueError(
                f"a linear row compares with '<=', '>=' or '=', not {rel!r}"
            )
        pairs: "list[tuple[Literal, float]]" = []
        if isinstance(terms, (Term, Literal)):
            for coeff, lits in _as_term(terms).parts:
                if len(lits) != 1:
                    raise ValueError(
                        "a linear row is a sum of single literals; "
                        f"one term here is a product of {len(lits)}. A quadratic constraint is a "
                        "different thing and this library does not read one."
                    )
                pairs.append((lits[0], float(coeff)))
        else:
            for item in terms:
                try:
                    lit, coeff = item
                except (TypeError, ValueError):
                    raise TypeError(
                        "linear takes an expression, or a sequence of (literal, coefficient) "
                        f"pairs; got {type(item).__name__}"
                    ) from None
                if isinstance(lit, Variable):
                    lit = Literal(lit, 1)
                if not isinstance(lit, Literal):
                    raise TypeError(
                        f"a linear row weights variables or literals, not {type(lit).__name__}"
                    )
                pairs.append((lit, float(coeff)))
        if not pairs:
            raise ValueError("a linear row needs at least one term")
        _model_lits_clear(self._h)
        for lit, coeff in pairs:
            self._must(
                _model_lit_weighted(self._h, lit.var._index, lit.value, coeff), "linear")
        if soft is None:
            self._must(_model_close_linear(self._h, code, float(rhs)), "linear")
        else:
            self._must(
                _model_close_linear_soft(self._h, code, float(rhs), float(soft)), "linear")

    def _counting(self, kind: int, of: "Sequence[Any]", k: int, value: int, what: str,
                  soft: "float | None" = None) -> None:
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
        if soft is None:
            self._must(_model_close(self._h, kind, int(k)), what)
        else:
            self._must(_model_close_soft(self._h, kind, int(k), float(soft)), what)

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

    #: The solvers :meth:`solve` can be pointed at, and the code the C ABI uses for each.
    _METHODS = {"anneal": 0, "tabu": 1, "breakout": 2, "branch": 3}

    def solve(
        self,
        tries: int = 12,
        beta_hot: float = 0.0,
        beta_cold: float = 0.0,
        stages: int = 0,
        sweeps: int = 0,
        method: str = "anneal",
        effort: int = 0,
    ) -> Answer:
        """Compile and solve, keeping the best of ``tries`` anneals.

        The four ladder parameters default to the library's own; a zero means "use that default", so
        a caller who has measured their own instance can override only what they measured.

        ``method`` points the solve at something other than annealing:

        ==============  ==========================================================================
        ``"anneal"``    Simulated annealing down a ladder. The default, and the only one there was.
        ``"tabu"``      Tabu search — the strongest single arm in this crate's own shootout.
        ``"breakout"``  Breakout local search: descent plus an adaptive perturbation.
        ``"branch"``    Branch and bound — **the only one that returns a proof**.
        ==============  ==========================================================================

        ``effort`` is that method's budget (iterations, or a node ceiling for ``"branch"``); 0 takes
        a default. The ladder parameters apply to ``"anneal"`` only, and ``"branch"`` warm-starts
        itself from a short anneal because a good incumbent prunes from the first node.

        Check :attr:`Answer.proved_optimal` after ``"branch"``.
        """
        if method not in self._METHODS:
            raise ValueError(
                f"unknown method {method!r}; one of "
                + ", ".join(repr(k) for k in self._METHODS)
            )
        spins = _model_compile(self._h)
        if spins == 0:
            raise ValueError(self._error() or "this problem did not compile")
        # Kept so `optima` can build its answers without recompiling -- a recompile clears the
        # answers it is about to read.
        self._spins = int(spins)
        if method != "anneal":
            if not _model_solve_by(self._h, self._METHODS[method], int(effort)):
                raise ValueError(self._error() or f"could not solve by {method}")
            return self._answer(spins)
        if not _model_solve_with(self._h, max(1, int(tries)), float(beta_hot), float(beta_cold),
                                 int(stages), int(sweeps)):
            raise ValueError(
                "that annealing ladder is not usable: beta_cold must exceed beta_hot, and both "
                "must be real numbers. Pass 0 for any of the four to use the default."
            )
        return self._answer(spins)

    def optima(self, tol: float = 1e-9) -> "list[Answer]":
        """Every distinct way to do the job that the last :meth:`solve` found, best first.

        A solve returns one answer and cannot say whether it was the only one. A model with a
        symmetry usually has several, and this is the question no surface in this field answers.

        >>> p = Problem()
        >>> a, b, c = (p.binary(n) for n in ("a", "b", "c"))
        >>> p.exactly_one([a.is_(1), b.is_(1), c.is_(1)])
        >>> _ = p.solve(tries=40)
        >>> len(p.optima())
        3

        Distinctness is on the DECODED VALUES, never on the spins: a compiled model carries slack
        and ancilla bits no variable reads, and the count has to be a statement about the model
        rather than about how the compiler chose to represent it.

        It is **evidence**, not a count: ``tries`` independent anneals prove the optima they landed
        on exist and prove nothing about the ones they missed. Only feasible answers are counted —
        an assignment that breaks a hard constraint is not a way to do the job. ``tol`` is on the
        compiled Ising energy, which folds in every penalty.
        """
        n = int(_model_optima(self._h, float(tol)))
        if n == 0:
            return []
        spins = getattr(self, "_spins", 0)
        out = []
        for i in range(n):
            if not _model_select_optimum(self._h, i, float(tol)):
                break
            out.append(self._answer(spins))
        # Put the handle back where the solve left it, so reading the alternatives does not change
        # what `solve` returned.
        _model_select_optimum(self._h, 0, float(tol))
        return out

    @property
    def answers_kept(self) -> int:
        """How many answers the last solve kept — one per try."""
        return int(_model_answers(self._h))

    def _answer(self, spins: int) -> Answer:
        """Read the last solve back, whichever method produced it."""
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
                      _model_violation_amount(self._h, i),
                      _model_violation_is_hard(self._h, i) == 1)
            for i in range(_model_violations(self._h))
        ]
        return Answer(values=vals, feasible=bool(_model_feasible(self._h)),
                      violated=broken, energy=float(_model_energy(self._h)),
                      objective=(float(_model_objective(self._h))
                                 if _model_has_objective(self._h) else None),
                      spins=int(spins), penalty=float(_model_penalty(self._h)),
                      soft_cost=float(_model_soft_cost(self._h)),
                      ancillas=int(_model_ancillas(self._h)),
                      proved_optimal=bool(_model_proved(self._h)),
                      caveats=[_read_text_idx(_model_caveat, self._h, i)
                               for i in range(_model_caveats(self._h))])

    def soften_last(self, weight: float) -> None:
        """Make the constraint added most recently a preference, priced at ``weight``.

        A hard constraint says which answers are answers at all. A soft one is a preference the
        solver may trade away: breaking it costs ``weight × amount²`` and leaves the answer
        feasible. The square is not a detail — a constraint becomes an energy term by squaring how
        far outside it sits, so missing by two costs four times missing by one.

        The weight is absolute rather than scaled: automatic scaling exists to stop a hard
        constraint being outbid by the objective, and a soft one is meant to be traded against it.
        """
        self._must(_model_soften_last(self._h, float(weight)), "soft constraint")

    def ommx(self) -> "tuple[bytes, float]":
        """The compiled model as an OMMX instance -- the interchange format this corner of the field converged on, so a ferrotherm program can be read by jijmodeling, Jij's stack, and anything else that speaks it. Returns the protobuf bytes and the offset the +/-1 to 0/1 substitution produced, ALREADY FOLDED INTO the instance -- ommx_objective(x) == ferrotherm_energy(s), so read the constant, do not add it.

        **Read the constant, do not add it.** The exporter writes it into the instance, so
        ``ommx_objective(x) == ferrotherm_energy(s)`` exactly and adding it again is wrong by its
        own value. It is returned so the substitution is visible.

        >>> raw, constant = problem.ommx()          # doctest: +SKIP
        >>> from ommx.v1 import Instance            # doctest: +SKIP
        >>> inst = Instance.from_bytes(raw)         # doctest: +SKIP
        """
        need = _model_ommx(self._h, None, 0)
        if not need:
            raise ValueError("compile or solve the problem first; there is no instance yet")
        buf = (ctypes.c_ubyte * need)()
        got = _model_ommx(self._h, buf, need)
        return bytes(bytearray(buf)[:got]), float(_model_ommx_constant(self._h))

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
