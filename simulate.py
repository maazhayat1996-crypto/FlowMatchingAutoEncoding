"""
Forward simulator for the Type-I seesaw mechanism.

Implements the parameter -> observable map G -> L of
Kitagawa, Nishimura, Otsuka, "Uncovering Hidden Leptonic Correlations with
Flow Matching and Autoencoders", arXiv:2608.15042 (Sec. 2 and Eq. 3.1-3.2).

Conventions (charged-lepton mass basis, diagonal Majorana mass matrix)
---------------------------------------------------------------------
Lagrangian:      L = Y_{i a} \bar N_i  l_a H  - 1/2 M_ij \bar N_i \bar N_j + h.c.
                 i = 1..3 right-handed neutrino index (rows of Y),
                 a = e, mu, tau flavour index (columns of Y).
Light masses:    (m_nu)_{ab} = <H>^2 (Y^T)_{a i} (M^-1)_{ij} Y_{j b}
                 M = diag(M_1, M_2, M_3), M_3 real positive,
                 M_k = |M_k| exp(i arg M_k) for k = 1, 2.
Diagonalisation: U^T m_nu U = diag(m_1, m_2, m_3), m_i >= 0 (Takagi),
                 masses sorted ascending, i.e. normal ordering labels.
PMNS:            PDG form, U = R(theta_ij, delta_CP) . diag(1, e^{i a21/2}, e^{i a31/2}).

Parameter vector G  (dim 23), in this order
--------------------------------------------
  G[0:9]    Re Y_{i a}, row-major (i, a)         in [-1, 1]
  G[9:18]   Im Y_{i a}, row-major (i, a)         in [-1, 1]
  G[18]     r_1 = |M_1| / M_3                    in [1e-5, 1]
  G[19]     r_2 = |M_2| / M_3                    in [1e-5, 1]
  G[20]     log10( <H>^2 / M_3 / meV )           in [-1.22, 2.78]
  G[21]     arg M_1                              in [-pi, pi]
  G[22]     arg M_2                              in [-pi, pi]

Observable vector L (dim 11), in this order
--------------------------------------------
  L[0]      log10( Delta m^2_21 / meV^2 )
  L[1]      log10( Delta m^2_31 / meV^2 )
  L[2:11]   |U_PMNS|_{a i}, row-major (a, i)

Everything is vectorised over a leading batch axis and computed in float64 /
complex128 with numpy. The public entry point is `simulate(theta)`.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Constants and prior
# --------------------------------------------------------------------------- #

DIM_G = 23
DIM_L = 11

# Names, mostly for plotting / debugging.
G_NAMES = (
    [f"ReY{i}{a}" for i in (1, 2, 3) for a in ("e", "mu", "tau")]
    + [f"ImY{i}{a}" for i in (1, 2, 3) for a in ("e", "mu", "tau")]
    + ["M1/M3", "M2/M3", "log10(v2/M3/meV)", "argM1", "argM2"]
)
L_NAMES = ["log10(dm21^2/meV^2)", "log10(dm31^2/meV^2)"] + [
    f"|U_{a}{i}|" for a in ("e", "mu", "tau") for i in (1, 2, 3)
]

# Eq. (3.2): box prior on G.  Sampling is uniform (linear) in each component.
_G_LOW = np.array([-1.0] * 18 + [1e-5, 1e-5, -1.22, -np.pi, -np.pi])
_G_HIGH = np.array([1.0] * 18 + [1.0, 1.0, 2.78, np.pi, np.pi])

# Floor on mass-squared differences before taking log10 (meV^2). Only matters
# for exactly degenerate or rank-deficient m_nu, which have measure zero.
_DM2_FLOOR = 1e-300


def prior_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return (low, high) arrays of shape (23,) for a BoxUniform prior on G."""
    return _G_LOW.copy(), _G_HIGH.copy()


def sample_prior(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw n parameter vectors uniformly from the box prior, Eq. (3.16)."""
    rng = np.random.default_rng() if rng is None else rng
    u = rng.random((n, DIM_G))
    return _G_LOW + (_G_HIGH - _G_LOW) * u


# --------------------------------------------------------------------------- #
# Forward model
# --------------------------------------------------------------------------- #

def _as_batch(theta) -> tuple[np.ndarray, bool]:
    theta = np.asarray(theta, dtype=np.float64)
    single = theta.ndim == 1
    if single:
        theta = theta[None, :]
    if theta.shape[-1] != DIM_G:
        raise ValueError(f"expected last dim {DIM_G}, got {theta.shape[-1]}")
    return theta, single


def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split G into (Y, Minv_scaled, scale).

    Returns
    -------
    Y      : (N, 3, 3) complex, Y[n, i, a]
    Minv   : (N, 3)    complex, M_3 / M_k  = exp(-i arg M_k) / r_k  (k = 1, 2, 3)
    scale  : (N,)      float,   <H>^2 / M_3 in meV
    """
    Y = (theta[:, 0:9] + 1j * theta[:, 9:18]).reshape(-1, 3, 3)
    r = np.stack([theta[:, 18], theta[:, 19], np.ones(len(theta))], axis=1)
    phase = np.stack([theta[:, 21], theta[:, 22], np.zeros(len(theta))], axis=1)
    Minv = np.exp(-1j * phase) / r
    scale = 10.0 ** theta[:, 20]
    return Y, Minv, scale


def light_mass_matrix(theta) -> np.ndarray:
    """
    Light neutrino mass matrix m_nu in meV, shape (N, 3, 3), complex symmetric.

    m_nu = (<H>^2 / M_3) * Y^T diag(M_3 / M_k) Y
    """
    theta, single = _as_batch(theta)
    Y, Minv, scale = unpack(theta)
    # (Y^T)_{a i} Minv_i Y_{i b}
    m = np.einsum("nia,ni,nib->nab", Y, Minv, Y)
    m = m * scale[:, None, None]
    return m[0] if single else m


def takagi(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Takagi factorisation of a batch of complex symmetric matrices.

    Finds unitary U and non-negative masses such that U^T m U = diag(masses),
    with masses sorted ascending.

    Parameters
    ----------
    m : (N, 3, 3) complex symmetric

    Returns
    -------
    masses : (N, 3) float, ascending
    U      : (N, 3, 3) complex unitary
    """
    # m = U^* D U^dag  =>  m^dag m = U D^2 U^dag.  eigh returns ascending eigvals.
    h = np.conj(np.swapaxes(m, -1, -2)) @ m
    _, V = np.linalg.eigh(h)
    # V = U P with P diagonal phases; V^T m V = P D P = diag(m_k e^{2 i phi_k}).
    d = np.einsum("nki,nkl,nli->ni", V, m, V)
    masses = np.abs(d)
    half_phase = np.exp(-0.5j * np.angle(d))
    U = V * half_phase[:, None, :]
    return masses, U


def simulate(theta) -> np.ndarray:
    """
    Forward map G -> L, Eq. (3.1).

    Parameters
    ----------
    theta : array-like, shape (23,) or (N, 23)

    Returns
    -------
    L : ndarray, shape (11,) or (N, 11)
        [log10(dm21^2/meV^2), log10(dm31^2/meV^2), |U|_{e1}, |U|_{e2}, ..., |U|_{tau3}]
    """
    theta, single = _as_batch(theta)
    m = light_mass_matrix(theta)
    masses, U = takagi(m)

    m1, m2, m3 = masses[:, 0], masses[:, 1], masses[:, 2]
    dm21 = np.maximum(m2**2 - m1**2, _DM2_FLOOR)
    dm31 = np.maximum(m3**2 - m1**2, _DM2_FLOOR)

    L = np.empty((len(theta), DIM_L), dtype=np.float64)
    L[:, 0] = np.log10(dm21)
    L[:, 1] = np.log10(dm31)
    L[:, 2:] = np.abs(U).reshape(-1, 9)
    return L[0] if single else L


def simulate_chunked(theta, chunk: int = 200_000) -> np.ndarray:
    """Same as `simulate` but processes the batch in chunks to bound memory."""
    theta, single = _as_batch(theta)
    out = np.concatenate([simulate(theta[i:i + chunk]) for i in range(0, len(theta), chunk)])
    return out[0] if single else out


# --------------------------------------------------------------------------- #
# Derived physical quantities (Eq. 3.14 and Eq. 4.1 of the paper)
# --------------------------------------------------------------------------- #

def pmns_parameters(U: np.ndarray) -> dict[str, np.ndarray]:
    """
    Extract PDG mixing angles and phases from a batch of PMNS matrices.

    Uses rephasing-invariant combinations so the result does not depend on
    the unphysical row phases of U.  Phases are returned in [0, 2 pi).

    Returns dict with keys s2_12, s2_13, s2_23, delta, alpha21, alpha31
    (sin^2 theta_ij and phases in radians), each of shape (N,).
    """
    Ue1, Ue2, Ue3 = U[:, 0, 0], U[:, 0, 1], U[:, 0, 2]
    Um1, Um2, Um3 = U[:, 1, 0], U[:, 1, 1], U[:, 1, 2]

    s2_13 = np.abs(Ue3) ** 2
    c2_13 = 1.0 - s2_13
    s2_12 = np.abs(Ue2) ** 2 / c2_13
    s2_23 = np.abs(Um3) ** 2 / c2_13

    s12, c12 = np.sqrt(s2_12), np.sqrt(1.0 - s2_12)
    s13, c13 = np.sqrt(s2_13), np.sqrt(c2_13)
    s23, c23 = np.sqrt(s2_23), np.sqrt(1.0 - s2_23)

    # Dirac phase from Jarlskog invariant (sin) and |U_mu1|^2 (cos).
    J = np.imag(Ue1 * Um2 * np.conj(Ue2) * np.conj(Um1))
    sin_d = J / (s23 * c23 * s12 * c12 * s13 * c2_13)
    cos_d = (np.abs(Um1) ** 2 - s2_12 * (1 - s2_23) - (1 - s2_12) * s2_23 * s2_13) / (
        2.0 * s12 * c12 * c23 * s23 * s13
    )
    delta = np.mod(np.arctan2(sin_d, cos_d), 2 * np.pi)

    # Majorana phases from Eq. (2.6)-(2.7):  arg(U_e1^* U_e2) = a21/2,
    # arg(U_e1^* U_e3) = a31/2 - delta.
    alpha21 = np.mod(2.0 * np.angle(np.conj(Ue1) * Ue2), 2 * np.pi)
    alpha31 = np.mod(2.0 * (np.angle(np.conj(Ue1) * Ue3) + delta), 2 * np.pi)

    return dict(s2_12=s2_12, s2_13=s2_13, s2_23=s2_23,
                delta=delta, alpha21=alpha21, alpha31=alpha31)


def derived_quantities(theta) -> dict[str, np.ndarray]:
    """
    Everything one usually wants from a parameter point, as a dict of (N,) arrays:

    m1, m2, m3 [meV], dm21, dm31 [meV^2], s2_12, s2_13, s2_23,
    delta, alpha21, alpha31 [rad], m_ee [meV] (effective 0nubb mass, Eq. 2.12),
    sum_m [meV].
    """
    theta, single = _as_batch(theta)
    m = light_mass_matrix(theta)
    masses, U = takagi(m)
    out = pmns_parameters(U)
    out["m1"], out["m2"], out["m3"] = masses[:, 0], masses[:, 1], masses[:, 2]
    out["dm21"] = masses[:, 1] ** 2 - masses[:, 0] ** 2
    out["dm31"] = masses[:, 2] ** 2 - masses[:, 0] ** 2
    out["sum_m"] = masses.sum(axis=1)
    out["m_ee"] = np.abs(np.einsum("ni,ni->n", U[:, 0, :] ** 2, masses))
    if single:
        out = {k: v[0] for k, v in out.items()}
    return out


def observables_T(theta) -> np.ndarray:
    """Eq. (3.14): [log10 dm21^2, log10 dm31^2, sin^2 th12, sin^2 th13, sin^2 th23]."""
    theta, single = _as_batch(theta)
    q = derived_quantities(theta)
    T = np.stack([
        np.log10(np.maximum(q["dm21"], _DM2_FLOOR)),
        np.log10(np.maximum(q["dm31"], _DM2_FLOOR)),
        q["s2_12"], q["s2_13"], q["s2_23"],
    ], axis=1)
    return T[0] if single else T


def observables_T_chunked(theta, chunk: int = 500_000) -> np.ndarray:
    """Same as `observables_T` for a batch, in chunks to bound memory.  Shape (N, 5)."""
    theta, _ = _as_batch(theta)
    if len(theta) == 0:
        return np.empty((0, 5))
    return np.concatenate([observables_T(theta[i:i + chunk]) for i in range(0, len(theta), chunk)])


# --------------------------------------------------------------------------- #
# NuFIT 6.0 (normal ordering, with SK atmospheric data), Table 1 of the paper
# --------------------------------------------------------------------------- #

NUFIT = {
    # name: (best fit, 3-sigma lower, 3-sigma upper)
    "s2_12": (0.308, 0.275, 0.345),
    "s2_13": (0.02215, 0.02030, 0.02388),
    "s2_23": (0.470, 0.435, 0.585),
    "delta_over_pi": (1.18, 0.69, 2.02),
    "dm21_meV2": (7.49e-5 * 1e6, 6.92e-5 * 1e6, 8.05e-5 * 1e6),   # eV^2 -> meV^2
    "dm31_meV2": (2.513e-3 * 1e6, 2.451e-3 * 1e6, 2.578e-3 * 1e6),
}


def pmns_matrix(s2_12, s2_13, s2_23, delta, alpha21=0.0, alpha31=0.0) -> np.ndarray:
    """PDG PMNS matrix, Eq. (2.4), for scalar inputs."""
    s12, c12 = np.sqrt(s2_12), np.sqrt(1 - s2_12)
    s13, c13 = np.sqrt(s2_13), np.sqrt(1 - s2_13)
    s23, c23 = np.sqrt(s2_23), np.sqrt(1 - s2_23)
    e = np.exp(1j * delta)
    V = np.array([
        [c12 * c13, s12 * c13, s13 / e],
        [-s12 * c23 - c12 * s23 * s13 * e, c12 * c23 - s12 * s23 * s13 * e, s23 * c13],
        [s12 * s23 - c12 * c23 * s13 * e, -c12 * s23 - s12 * c23 * s13 * e, c23 * c13],
    ])
    return V @ np.diag([1.0, np.exp(0.5j * alpha21), np.exp(0.5j * alpha31)])


def labels_from_nufit_bestfit() -> np.ndarray:
    """
    Experimental condition L_exp (shape (1, 11)) built from the NuFIT 6.0 best fit.

    |U| is evaluated at the best-fit angles and delta_CP; it does not depend on
    the Majorana phases.
    """
    U = pmns_matrix(NUFIT["s2_12"][0], NUFIT["s2_13"][0], NUFIT["s2_23"][0],
                    np.pi * NUFIT["delta_over_pi"][0])
    L = np.concatenate([
        [np.log10(NUFIT["dm21_meV2"][0]), np.log10(NUFIT["dm31_meV2"][0])],
        np.abs(U).reshape(9),
    ])
    return L[None, :]


def nufit_T() -> tuple[np.ndarray, np.ndarray]:
    """
    NuFIT 6.0 best fit of T (Eq. 3.14) and the half-width of its 3-sigma interval,
    each of shape (5,), in log10 space for the mass-squared differences.
    """
    keys = ["dm21_meV2", "dm31_meV2", "s2_12", "s2_13", "s2_23"]
    best = np.array([NUFIT[k][0] for k in keys])
    lo = np.array([NUFIT[k][1] for k in keys])
    hi = np.array([NUFIT[k][2] for k in keys])
    best[:2], lo[:2], hi[:2] = np.log10(best[:2]), np.log10(lo[:2]), np.log10(hi[:2])
    sig3 = 0.5 * (hi - lo)
    return best, sig3


def chi2_from_T(T: np.ndarray) -> np.ndarray:
    """Eq. (3.15) evaluated on precomputed observables T of shape (N, 5)."""
    best, sig3 = nufit_T()
    return 9.0 * np.sum((np.atleast_2d(T) - best) ** 2 / sig3**2, axis=1)


def chi2(theta) -> np.ndarray:
    """
    Eq. (3.15): chi^2 = 9 * sum_i (T_i - T_exp,i)^2 / (3 sigma_i)^2 over the five
    components of T, with 3 sigma_i taken as half the 3-sigma interval width
    (in log10 space for the mass-squared differences).
    """
    c = chi2_from_T(observables_T(theta))
    return c[0] if np.ndim(theta) == 1 else c


def labels_T_from_nufit_bestfit() -> np.ndarray:
    """NuFIT 6.0 best fit of T as a conditioning point, shape (1, 5)."""
    return nufit_T()[0][None, :]


def noisy_labels_T(theta, rng: np.random.Generator) -> np.ndarray:
    """
    T(theta) plus Gaussian noise with sigma_i = (3-sigma half-width)/3, i.e. the
    same sigma_i that define chi2() above, so that the likelihood implied by the
    training data is exp(-chi2/2).  Shape (N, 5).
    """
    T = np.atleast_2d(observables_T(theta))
    sigma = nufit_T()[1] / 3.0
    return T + sigma * rng.standard_normal(T.shape)


# --------------------------------------------------------------------------- #
# Reproducible chunked simulation (for multiprocessing / caching)
# --------------------------------------------------------------------------- #

def simulate_prior_chunk(seed: int, index: int, n: int, labels: str = "L",
                         label_noise: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw chunk number `index` of a training set: n prior samples and their labels,
    both float32.  The chunk's RNG depends only on (seed, index), so a data set
    built from chunks 0..K-1 is identical however the chunks are distributed over
    worker processes, and its first rows do not depend on the total size.

    labels = "L": the 11-dim deterministic label of `simulate` (paper default).
    labels = "T": the 5-dim observables of `observables_T`, plus Gaussian noise
                  from `noisy_labels_T` if label_noise is set.
    """
    rng = np.random.default_rng([seed, index])
    theta = sample_prior(n, rng)
    if labels == "L":
        if label_noise:
            raise ValueError("label_noise is only defined for labels='T'")
        x = simulate(theta)
    elif labels == "T":
        x = noisy_labels_T(theta, rng) if label_noise else observables_T(theta)
    else:
        raise ValueError(f"unknown labels {labels!r}")
    return theta.astype(np.float32), np.atleast_2d(x).astype(np.float32)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 1. Takagi factorisation is exact and U is unitary.
    th = sample_prior(2000, rng)
    m = light_mass_matrix(th)
    masses, U = takagi(m)
    D = np.einsum("nki,nkl,nlj->nij", U, m, U)
    err_diag = np.max(np.abs(D - masses[:, :, None] * np.eye(3)) / (masses[:, 2, None, None] + 1e-300))
    err_unit = np.max(np.abs(np.conj(np.swapaxes(U, 1, 2)) @ U - np.eye(3)))
    assert np.all(np.diff(masses, axis=1) >= 0), "masses not sorted"
    print(f"Takagi: max rel |U^T m U - diag| = {err_diag:.2e}, max |U^dag U - 1| = {err_unit:.2e}")

    # 2. Round trip: build m_nu from known PDG parameters (plus an unphysical
    #    charged-lepton rephasing P m P), recover them.
    for _ in range(200):
        p = dict(s2_12=rng.uniform(0.2, 0.4), s2_13=rng.uniform(0.01, 0.05),
                 s2_23=rng.uniform(0.3, 0.7), delta=rng.uniform(0, 2 * np.pi),
                 alpha21=rng.uniform(0, 2 * np.pi), alpha31=rng.uniform(0, 2 * np.pi))
        mass = np.sort(rng.uniform(1, 60, 3))
        U0 = pmns_matrix(**p)
        P = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, 3)))
        m0 = P @ (np.conj(U0) @ np.diag(mass) @ np.conj(U0).T) @ P
        mm, UU = takagi(m0[None])
        q = pmns_parameters(UU)
        assert np.allclose(mm[0], mass, rtol=1e-9)
        for k in ("s2_12", "s2_13", "s2_23"):
            assert abs(q[k][0] - p[k]) < 1e-9
        for k in ("delta", "alpha21", "alpha31"):
            d = np.mod(q[k][0] - p[k] + np.pi, 2 * np.pi) - np.pi
            assert abs(d) < 1e-7, (k, q[k][0], p[k])
    print("Round trip with charged-lepton rephasing: OK")

    # 3. Shapes, NuFIT labels, chi2 on prior samples.
    L = simulate(th)
    assert L.shape == (2000, DIM_L) and np.all(np.isfinite(L))
    Lexp = labels_from_nufit_bestfit()
    print("L_exp shape", Lexp.shape)
    print("L_exp =", np.array2string(Lexp[0], precision=4))
    print("|U|_exp =\n", np.array2string(Lexp[0, 2:].reshape(3, 3), precision=3))
    c = chi2(th)
    print(f"chi2 on 2000 prior samples: min={c.min():.1f}, median={np.median(c):.1f}")
    print("all checks passed")
