"""Free Claims + config bundle for the linear-Gaussian SCM.

Two `@claim` Free Claims:

- `gaussian_source(mu, sigma, noise) -> mu + sigma * noise`
- `linear_arrow(beta, parent, noise) -> beta * parent + noise`

One config bundle:

- `LinearGaussianSCM` — frozen dataclass holding the structural
  coefficients (mu_x, sigma_x, beta_xz, sigma_z, beta_zy, sigma_y,
  n_steps). `simulate(scm, *, seed)` runs the SCM forward by calling
  the Free Claims, returning a `LGSCMObservation` with per-step
  trajectories and scalar means.

The bundle's fields ARE the framework's leaves — `walk_paths` would
surface them at dotted paths (`mu_x`, `beta_xz`, ...). The
mechanics method `simulate(...)` is a plain function, NOT a Claim,
matching CLAUDE.md's "config-bundle-with-mechanics" pattern: the
theoretical content lives on the @claim Free Claims that
`simulate` calls.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corroborate import claim


# ============ Free Claims — structural arrows ============

@claim
def gaussian_source(mu: float, sigma: float, noise: float) -> float:
    """X = mu + sigma * noise. `noise` is the standardised
    unit-variance realisation supplied by the runner; the @claim
    wrapper records the call so a `trace_context()` sees one
    structural source per emission. Trivial arithmetically — its
    role is to be a recorded edge in the SCM's graph."""
    return mu + sigma * noise


@claim
def linear_arrow(beta: float, parent: float, noise: float) -> float:
    """Y = beta * parent + noise. Shared by every linear edge in
    the SCM (X→Z, Z→Y); only `beta` differs across edges, so the
    walker's leaf-signature distinguishes arms by which `beta`
    leaf was bound, not by which arrow function was used."""
    return beta * parent + noise


# ============ Config bundle — non-Claim mechanics ============

@dataclass(frozen=True, slots=True)
class LinearGaussianSCM:
    """Two-arrow linear-Gaussian SCM: X → Z → Y.

    Fields are leaves in the framework's vocabulary — the walker
    surfaces them at dotted topology paths (e.g. `beta_xz`). A
    paired two-arm intervention is expressed by constructing two
    `LinearGaussianSCM` instances that differ only in one
    coefficient (typically `beta_xz`); the runner stamps a
    distinct `arm_key` on each.

    Sharing a seed across arms gives identical noise stream
    realisations via `numpy.random.default_rng(seed)`, so the
    paired Δ on any downstream summary cancels the noise:

        Delta_Y(t)
            = beta_zy * (beta_xz_t - beta_xz_b) * X(t)

    The substrate's whole point is that this Δ has a closed form
    the test can assert against, rather than a large-N empirical
    bound that admits silent regressions."""
    mu_x: float
    sigma_x: float
    beta_xz: float
    sigma_z: float
    beta_zy: float
    sigma_y: float
    n_steps: int


@dataclass(frozen=True, slots=True)
class LGSCMObservation:
    """One run's observation. Trajectories are tuples of floats
    (immutable, hashable) so they round-trip through any
    `Mapping[str, MeasurementLeaf]`-shaped surface; scalar means
    are the canonical paired-g sources."""
    x: tuple[float, ...]
    z: tuple[float, ...]
    y: tuple[float, ...]
    x_mean: float
    z_mean: float
    y_mean: float


def simulate(scm: LinearGaussianSCM, *, seed: int) -> LGSCMObservation:
    """Run the SCM forward for `n_steps` and return trajectories +
    means.

    Uses `numpy.random.default_rng(seed)` for typed reproducible
    sampling. Pre-draws all three epsilon streams up-front so the
    sampling order is identical regardless of `beta_xz`,
    `beta_zy`, etc. — that's what guarantees the paired Δ
    cancellation on shared seeds.

    Calls each `@claim` arrow per step, so a `trace_context()`
    around `simulate(...)` records `n_steps` `gaussian_source`
    calls and `2 * n_steps` `linear_arrow` calls, suitable for
    Trace-extractor smoke tests in later slices.
    """
    rng = np.random.default_rng(seed)
    n = scm.n_steps
    eps_x = rng.standard_normal(n)
    eps_z_unit = rng.standard_normal(n)
    eps_y_unit = rng.standard_normal(n)

    xs: list[float] = []
    zs: list[float] = []
    ys: list[float] = []
    for t in range(n):
        x = gaussian_source(scm.mu_x, scm.sigma_x, float(eps_x[t]))
        z = linear_arrow(scm.beta_xz, x, float(eps_z_unit[t]) * scm.sigma_z)
        y = linear_arrow(scm.beta_zy, z, float(eps_y_unit[t]) * scm.sigma_y)
        xs.append(x)
        zs.append(z)
        ys.append(y)

    return LGSCMObservation(
        x=tuple(xs),
        z=tuple(zs),
        y=tuple(ys),
        x_mean=float(np.mean(xs)),
        z_mean=float(np.mean(zs)),
        y_mean=float(np.mean(ys)),
    )


# ============ Phased (multi-burst) simulation ============

@dataclass(frozen=True, slots=True)
class LGSCMPhasedObservation:
    """A multi-burst run's observation.

    `y` and `z` are the per-(burst, episode) value matrices as
    tuple-of-tuples — immutable + nested-list-decodable, matching
    the parquet round-trip shape on `TraceRow` (`List[List[Float]]`).
    Per-burst-mean Y / Z are computed eagerly so analyses can read
    either the raw matrices or the reduced arrays without rerunning.

    `x_mean_per_burst` carries the per-burst average of X. It's
    closed-form-tractable (E = mu_x for any burst with a finite
    mean), and the variance of the paired Δ on per-burst-mean Y
    reduces to `(Delta_beta * beta_zy)^2 * Var[x_mean_per_burst]`.
    Tests use it to size the analytical SE on per-burst Δ.

    Z is exposed as a separate trace to support link analyses
    (`paired_link_per_burst`): under shared-seed cancellation, the
    paired Δ_Z (predictor) and Δ_Y (target) per (burst, seed) are
    proportional with slope `beta_zy`, so the per-burst link r is
    exactly ±1 — a closed-form target the analysis must recover.
    """
    y: tuple[tuple[float, ...], ...]
    z: tuple[tuple[float, ...], ...]
    x: tuple[tuple[float, ...], ...]
    x_mean_per_burst: tuple[float, ...]
    y_mean_per_burst: tuple[float, ...]
    z_mean_per_burst: tuple[float, ...]
    y_mean_overall: float


def _validate_phased(scms_per_burst: tuple[LinearGaussianSCM, ...]) -> int:
    """Shared invariant: at least one burst, all bursts share the
    same `n_steps` (so the resulting Y matrix is rectangular and
    paired-arm comparisons across bursts make shape sense)."""
    if not scms_per_burst:
        raise ValueError(
            'simulate_phased: scms_per_burst must be non-empty',
        )
    n_steps_set = {scm.n_steps for scm in scms_per_burst}
    if len(n_steps_set) != 1:
        raise ValueError(
            f'simulate_phased: all bursts must share n_steps; '
            f'got {sorted(n_steps_set)}',
        )
    return n_steps_set.pop()


def simulate_phased(
    scms_per_burst: tuple[LinearGaussianSCM, ...],
    *,
    seed: int,
) -> LGSCMPhasedObservation:
    """Run a multi-burst SCM. Burst `b` runs `scms_per_burst[b]`
    for `n_steps` episodes; the resulting Y matrix has shape
    `(n_bursts, n_steps)`.

    Noise streams are pre-drawn up front against `seed` and shared
    across all bursts: eps_x has shape `(n_bursts, n_steps)`, same
    for eps_z and eps_y. Two arms with the same seed but different
    `scms_per_burst` consume the noise in the same order, so the
    per-(seed, burst, episode) paired Δ cancels every epsilon and
    reduces to `(beta_xz_t(b) - beta_xz_b(b)) * beta_zy(b) * X(b, e)`.
    This is what makes per-burst Δ analytically tractable.

    Calls each `@claim` arrow per (burst, episode) — within a
    `trace_context()` the trace records `n_bursts * n_steps`
    `gaussian_source` calls and `2 * n_bursts * n_steps`
    `linear_arrow` calls in lexicographic burst-major order.
    """
    n_steps = _validate_phased(scms_per_burst)
    n_bursts = len(scms_per_burst)
    rng = np.random.default_rng(seed)
    eps_x = rng.standard_normal((n_bursts, n_steps))
    eps_z_unit = rng.standard_normal((n_bursts, n_steps))
    eps_y_unit = rng.standard_normal((n_bursts, n_steps))

    y_per_burst: list[tuple[float, ...]] = []
    z_per_burst: list[tuple[float, ...]] = []
    x_per_burst: list[tuple[float, ...]] = []
    x_means: list[float] = []
    y_means: list[float] = []
    z_means: list[float] = []
    all_y: list[float] = []
    for b, scm in enumerate(scms_per_burst):
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for e in range(n_steps):
            x = gaussian_source(
                scm.mu_x, scm.sigma_x, float(eps_x[b, e]),
            )
            z = linear_arrow(
                scm.beta_xz, x, float(eps_z_unit[b, e]) * scm.sigma_z,
            )
            y = linear_arrow(
                scm.beta_zy, z, float(eps_y_unit[b, e]) * scm.sigma_y,
            )
            xs.append(x)
            ys.append(y)
            zs.append(z)
        y_per_burst.append(tuple(ys))
        z_per_burst.append(tuple(zs))
        x_per_burst.append(tuple(xs))
        x_means.append(float(np.mean(xs)))
        y_means.append(float(np.mean(ys)))
        z_means.append(float(np.mean(zs)))
        all_y.extend(ys)
    return LGSCMPhasedObservation(
        y=tuple(y_per_burst),
        z=tuple(z_per_burst),
        x=tuple(x_per_burst),
        x_mean_per_burst=tuple(x_means),
        y_mean_per_burst=tuple(y_means),
        z_mean_per_burst=tuple(z_means),
        y_mean_overall=float(np.mean(all_y)),
    )


__all__ = [
    'LGSCMObservation',
    'LGSCMPhasedObservation',
    'LinearGaussianSCM',
    'gaussian_source',
    'linear_arrow',
    'simulate',
    'simulate_phased',
]
