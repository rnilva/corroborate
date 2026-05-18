"""Designed LG-SCM-style sweep: empirical falsification of Theorem 3 / Cor 3.2.

Theorem 3 (THEORY_bootstrap_dominance.md §6.1): under (A2)+(A3) iid
Gaussian, DDQN's clip preserves the agent's one-step-bootstrapped
argmax at s when γ · σ_clip(s) · √(2(K-1)) < Δ_v(s).

Corollary 3.2: σ_clip²(s) ∝ Var_a[Λ_a(s'_a)] within the transition
band. Specifically, when destinations s'_a have heterogeneous
noise structure (σ_T(s'_a) varying across a), σ_clip grows.

This script constructs a clean test on a synthetic env where:
- (A2) holds by construction (Q_online, Q_target sampled
  independently each trial).
- (A3) holds by construction (iid Gaussian noise across actions
  at each destination).
- σ_T heterogeneity is the controlled axis.
- The four deep-RL open limitations (Polyak τ violation,
  FA correlation, A4'a one-step→converged, geometric-series
  accumulation) are eliminated by design.

Setup. One source state s with K=3 source-actions. Each source-
action a maps to a deterministic destination s'_a, where K=3
candidate next-actions are sampled. At destination s'_a:
- True next-action values Q*(s'_a, a') = (1.0, 0.5, 0.0) (so
  per-destination true gap Δ_T = 0.5).
- Per-action iid Gaussian noise SD = σ_T(s'_a). Heterogeneity
  level: σ_T(s'_a=1) = σ_T(s'_a=2) = 1.0, σ_T(s'_a=3) = σ_high.

For each MC trial:
- Q_online(s'_a, a') ~ N(Q*(s'_a, a'), σ_T(s'_a)²) iid
- Q_target(s'_a, a') ~ N(Q*(s'_a, a'), σ_T(s'_a)²) iid (independent of online)
- V_vanilla(s'_a) = max_{a'} Q_target(s'_a, a')
- V_ddqn(s'_a)    = Q_target(s'_a, argmax_{a'} Q_online(s'_a, a'))
- Δ_clip(s'_a) = V_vanilla(s'_a) - V_ddqn(s'_a) ≥ 0

After N trials per cell, compute E[Δ_clip(s'_a)] per destination,
then σ_clip² := Var_a[E[Δ_clip(s'_a)]].

Source-state argmax preservation: set Q_v(s, a) so that vanilla
argmax is a = 0 with Δ_v(s) (gap) = 1.0. Under DDQN at the source:
Q_d(s, a) = Q_v(s, a) - γ · E[Δ_clip(s'_a)]. Argmax preserved iff
max_a E[Δ_clip(s'_a)] - argmax-a E[Δ_clip(s'_a)] < Δ_v(s) / γ.

Sweep: σ_high ∈ {1.0, 1.5, 2.0, 3.0, 5.0, 10.0}. Per cell, 30
seeds × 5000 MC trials. Output: per-cell empirical σ_clip,
predicted σ_clip from Cor 3.2 proxy (Hasselt-asymptotic), and
preservation-rate. Falsification target: Spearman ρ(empirical
σ_clip, predicted σ_clip) ≥ 0.9 across heterogeneity levels.

Usage:
    PYTHONPATH=. uv run python scripts/theorem3_lg_scm_calibration.py
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


_GAMMA = 0.99
_K_ACTIONS = 3
_Q_STAR_DEST = (1.0, 0.5, 0.0)  # True Q* at each destination, fixed across destinations
_SIGMA_BASE = 1.0
_HETERO_LEVELS = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
_N_SEEDS_PER_CELL = 30
_N_MC_TRIALS_PER_SEED = 5000
_Q_V_SOURCE = (2.0, 1.0, 0.5)  # Vanilla source Q so argmax = 0, Δ_v = 1.0


@dataclass(frozen=True, slots=True)
class CellSpec:
    """Configuration of a single cell in the σ_T-heterogeneity sweep."""
    sigma_high: float  # σ_T at destination s'_0 (highest-bias dest)
    sigma_base: float = _SIGMA_BASE
    K: int = _K_ACTIONS
    gamma: float = _GAMMA

    @property
    def sigma_T(self) -> tuple[float, float, float]:
        """σ_T at each destination s'_a. Heterogeneity at s'_0."""
        return (self.sigma_high, self.sigma_base, self.sigma_base)


def hasselt_asymptotic(sigma: float, K: int) -> float:
    """Hasselt 2010 asymptotic max-bias bound: σ · √(2 ln K).
    Used as the analytical prediction for E[max_a Q̂ - max_a Q*]
    when Q* = 0 (constant) and noise iid N(0, σ²)."""
    return sigma * math.sqrt(2.0 * math.log(K))


def predicted_sigma_clip_proxy(spec: CellSpec) -> float:
    """Cor 3.2 prediction: σ_clip ∝ sqrt(Var_a[Λ_a(s'_a)]).
    With Δ_T constant across destinations and Λ_a(s'_a) ∝
    σ_T(s'_a) / Δ_T, σ_clip ∝ Var_a[σ_T(s'_a)] (up to multiplicative
    constants of order unity, per Theorem 2's φ(K)).

    Returns: SD of (σ_T(s'_a) · √(2 ln K)) across destinations a —
    the leading-order Hasselt-asymptotic clip-error vector across
    destinations, with cross-action mean removed."""
    K = spec.K
    sigma_T = np.array(spec.sigma_T, dtype=np.float64)
    # The per-destination expected clip error under (A2)+(A3) scales as
    # σ_T · η(K) where η(K) ≈ √(2 ln K) (Hasselt asymptotic).
    eta_K = math.sqrt(2.0 * math.log(K))
    per_dest_proxy = sigma_T * eta_K
    return float(np.std(per_dest_proxy, ddof=0))


def simulate_cell(spec: CellSpec, *, seed: int, n_trials: int) -> dict[str, float]:
    """Single-state bootstrap MC under (A2)+(A3). Returns per-cell
    empirical measurements:
    - sigma_clip_empirical: SD across destinations of E[Δ_clip]
    - argmax_preserved_rate: fraction of trials where DDQN's
      source-action argmax equals vanilla's source-action argmax
    - mean_clip_per_dest: list of E[Δ_clip(s'_a)] over destinations
    """
    rng = np.random.default_rng(seed)
    sigma_T = np.array(spec.sigma_T, dtype=np.float64)  # shape (K,)
    Q_star_dest = np.array(_Q_STAR_DEST, dtype=np.float64)  # shape (K,)
    Q_v_source = np.array(_Q_V_SOURCE, dtype=np.float64)  # shape (K,)
    K = spec.K
    # For each destination s'_a, sample n_trials MC bootstrap pairs.
    # Δ_clip(s'_a) per trial = max_a' Q_target(s'_a, a') - Q_target(s'_a, argmax_a' Q_online).
    delta_clip_per_dest = np.zeros((K, n_trials), dtype=np.float64)
    for a in range(K):  # source action, picks destination s'_a
        # Q_online[trial, a'] = Q*[a'] + sigma_T[a]*z_online[trial, a']
        # Q_target[trial, a'] = Q*[a'] + sigma_T[a]*z_target[trial, a']
        z_online = rng.standard_normal((n_trials, K))
        z_target = rng.standard_normal((n_trials, K))
        Q_online = Q_star_dest[None, :] + sigma_T[a] * z_online
        Q_target = Q_star_dest[None, :] + sigma_T[a] * z_target
        # vanilla: V_v = max_a' Q_target
        V_v = Q_target.max(axis=1)
        # DDQN: V_d = Q_target[argmax_a' Q_online]
        argmax_online = Q_online.argmax(axis=1)
        V_d = Q_target[np.arange(n_trials), argmax_online]
        delta_clip_per_dest[a, :] = V_v - V_d
    # E[Δ_clip(s'_a)] across trials
    mean_clip_per_dest = delta_clip_per_dest.mean(axis=1)  # shape (K,)
    sigma_clip_empirical = float(np.std(mean_clip_per_dest, ddof=0))
    # Source-action argmax preservation under DDQN.
    # Use per-trial Q_d(s, a) = Q_v(s, a) - γ * Δ_clip_per_trial(s'_a).
    Q_d_per_trial = Q_v_source[:, None] - spec.gamma * delta_clip_per_dest  # (K, n_trials)
    argmax_v = int(np.argmax(Q_v_source))
    argmax_d_per_trial = np.argmax(Q_d_per_trial, axis=0)
    preserved_rate = float(np.mean(argmax_d_per_trial == argmax_v))
    return {
        'sigma_clip_empirical': sigma_clip_empirical,
        'argmax_preserved_rate': preserved_rate,
        'mean_clip_per_dest_0': float(mean_clip_per_dest[0]),
        'mean_clip_per_dest_1': float(mean_clip_per_dest[1]),
        'mean_clip_per_dest_2': float(mean_clip_per_dest[2]),
    }


def main() -> int:
    print(f'Theorem 3 / Cor 3.2 calibration on designed LG-SCM-style sweep')
    print(f'(A2)+(A3) hold by construction; FA correlation + Polyak τ + '
          f'one-step→converged gaps eliminated.')
    print()
    print(f'{"σ_high":>7s} {"pred σ_clip":>12s} {"emp σ_clip (mean)":>20s} '
          f'{"emp σ_clip (sd)":>17s} {"preserved":>10s}')
    print('-' * 80)

    rows: list[dict[str, float]] = []
    for hetero in _HETERO_LEVELS:
        spec = CellSpec(sigma_high=hetero)
        pred = predicted_sigma_clip_proxy(spec)
        # Run n_seeds cells, aggregate
        per_seed = [
            simulate_cell(spec, seed=s, n_trials=_N_MC_TRIALS_PER_SEED)
            for s in range(_N_SEEDS_PER_CELL)
        ]
        sigma_clip_means = np.array([r['sigma_clip_empirical'] for r in per_seed])
        preserved_rates = np.array([r['argmax_preserved_rate'] for r in per_seed])
        rows.append({
            'sigma_high': hetero,
            'pred_sigma_clip': pred,
            'emp_sigma_clip_mean': float(sigma_clip_means.mean()),
            'emp_sigma_clip_sd': float(sigma_clip_means.std(ddof=1)),
            'preserved_rate_mean': float(preserved_rates.mean()),
            'preserved_rate_sd': float(preserved_rates.std(ddof=1)),
        })
        print(f'{hetero:>7.2f} {pred:>12.4f} {sigma_clip_means.mean():>20.4f} '
              f'{sigma_clip_means.std(ddof=1):>17.4f} {preserved_rates.mean():>10.4f}')

    # Cross-cell Spearman ρ(empirical σ_clip, predicted σ_clip)
    pred_arr = np.array([r['pred_sigma_clip'] for r in rows])
    emp_arr = np.array([r['emp_sigma_clip_mean'] for r in rows])
    pres_arr = np.array([r['preserved_rate_mean'] for r in rows])
    rho_pred_emp = spearmanr(pred_arr, emp_arr)
    rho_emp_pres = spearmanr(emp_arr, pres_arr)

    print()
    print(f'Cross-cell Cor 3.2 test: Spearman ρ(pred σ_clip, emp σ_clip) = '
          f'{rho_pred_emp.statistic:+.4f} (p={rho_pred_emp.pvalue:.4g})')
    print(f'Theorem 3 inequality test: Spearman ρ(emp σ_clip, preserved rate) = '
          f'{rho_emp_pres.statistic:+.4f} (p={rho_emp_pres.pvalue:.4g})  '
          f'[predicted negative]')

    out = {
        'spec': {
            'K': _K_ACTIONS,
            'gamma': _GAMMA,
            'Q_star_dest': list(_Q_STAR_DEST),
            'Q_v_source': list(_Q_V_SOURCE),
            'Delta_v_source': float(_Q_V_SOURCE[0] - _Q_V_SOURCE[1]),
            'sigma_base': _SIGMA_BASE,
            'hetero_levels': list(_HETERO_LEVELS),
            'n_seeds_per_cell': _N_SEEDS_PER_CELL,
            'n_mc_trials_per_seed': _N_MC_TRIALS_PER_SEED,
        },
        'rows': rows,
        'cross_cell_test': {
            'spearman_pred_vs_emp_sigma_clip': {
                'rho': float(rho_pred_emp.statistic),
                'p_value': float(rho_pred_emp.pvalue),
            },
            'spearman_emp_sigma_clip_vs_preserved': {
                'rho': float(rho_emp_pres.statistic),
                'p_value': float(rho_emp_pres.pvalue),
            },
        },
    }
    out_path = Path('docs/THEOREM3_LG_SCM_CALIBRATION.json')
    out_path.write_text(json.dumps(out, indent=2))
    print(f'\nwrote {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
