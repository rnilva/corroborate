"""Hasselt's max-bias primitives in the tabular regime.

Two `@claim` Free Claims that mirror the function-approximation
counterparts in `corroborate_rl.dqn.claims.bootstrap` but operate
on a Q-vector directly (no q_network, no params):

- `max_greedify_tabular(q)` — vanilla DQN's `max_a Q(s, a)` target.
- `double_greedify_tabular(q_online, q_target)` —
  DDQN's `Q_target[argmax_a Q_online(s, a)]` target.

These are the operators where Hasselt's max-bias theorem applies.
The tabular regime makes the bias **exactly computable** under
known noise distributions:

- For Q̂ = Q* + ε with ε iid N(0, σ²) across |A|=2 actions:
    E[max_a Q̂(s, a)] − max_a Q*(s, a) = σ / √π   (exact)
- For Q_online ⫫ Q_target both estimating Q*:
    E[Q_target[argmax_a Q_online]] − max_a Q*(s, a) = 0   (exact)
- For Q_online = Q_target (deterministic identity):
    DDQN reduces to max_greedify and inherits the same bias.

Closed-form helpers:
- `hasselt_n2_max_bias(sigma)`: returns σ/√π — the |A|=2 closed form.
- `hasselt_max_bias_asymptotic(sigma, n_actions)`: returns the
  σ·√(2 ln n) leading-order extreme-value-theory bias estimate
  (loose bound for small n; asymptotic for large n).

The framework's substrate-side `jensen_dormancy_gap` invariant
uses `σ_Q · √(2 log |A|)` as the structural-floor below which
mechanism dormancy is declared (auto-memory `findings_action_dim_sweep`)."""
from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from corroborate import claim


type _FArr = npt.NDArray[np.float64]


@claim
def max_greedify_tabular(q: _FArr) -> float:
    """Vanilla DQN's bootstrap target: V = max_a Q(s, a).

    Hasselt 2010, 2016: under noisy estimates Q̂ = Q* + ε with
    iid mean-zero ε across actions, the max is biased upward by
    Jensen's inequality (max is convex):
        E[max_a Q̂(s, a)] ≥ max_a E[Q̂(s, a)] = max_a Q*(s, a)
    For |A|=2 iid Gaussian, the exact bias is σ/√π.
    For larger |A| the leading-order extreme-value-theory term
    is σ · √(2 ln |A|)."""
    return float(np.max(q))


@claim
def double_greedify_tabular(
    q_online: _FArr, q_target: _FArr,
) -> float:
    """DDQN's bootstrap target: V = Q_target(s, argmax_a Q_online(s, a)).

    Hasselt 2016: when Q_online and Q_target are *independent*
    estimators of Q*, the action selection (online) is decoupled
    from the value evaluation (target), and Jensen's inequality
    no longer applies — the action chosen by Q_online is a random
    index w.r.t. Q_target's noise, so the value at that index
    samples Q_target's mean, which is unbiased:
        E[Q_target[argmax_a Q_online]] = max_a Q*(s, a) (when independent)

    When Q_online = Q_target (same source, deterministic identity),
    this reduces to max_greedify_tabular and inherits the same
    upward bias."""
    a_star = int(np.argmax(q_online))
    return float(q_target[a_star])


# ============ Closed-form bias helpers ============

def hasselt_n2_max_bias(sigma: float) -> float:
    """Exact closed-form upward bias of `max_a Q̂` for |A|=2 under
    iid N(0, σ²) noise.

    Derivation: max(X1, X2) = (X1 + X2)/2 + |X1 - X2|/2.
    For X1, X2 iid N(0, σ²): X1 - X2 ~ N(0, 2σ²), so
    E[|X1 - X2|] = 2σ/√π (folded-normal mean). Hence
        E[max] = 0 + (2σ/√π)/2 = σ/√π."""
    if sigma < 0.0:
        raise ValueError(f'sigma must be ≥ 0; got {sigma}')
    return sigma / math.sqrt(math.pi)


def hasselt_max_bias_asymptotic(
    sigma: float, n_actions: int,
) -> float:
    """Leading-order extreme-value-theory bias of `max_a Q̂` for
    |A| iid N(0, σ²) noise: σ · √(2 ln |A|).

    This is the dominant term in the Fisher-Tippett-Gnedenko limit
    for the maximum of n iid Gumbel-domain variables. For small
    |A| (≤ 5) it OVERESTIMATES the true bias by 30-60% (the
    correction term `−σ · ln(4π · ln |A|)/(2√(2 ln |A|))` brings
    it closer); for |A| ≥ 50 it's accurate to ~5%.

    The framework's implementation uses this as the structural floor
    in `jensen_dormancy_gap` — under-floor jensen_gap declares
    the Hasselt mechanism dormant on a cell."""
    if sigma < 0.0:
        raise ValueError(f'sigma must be ≥ 0; got {sigma}')
    if n_actions < 2:
        raise ValueError(f'n_actions must be ≥ 2; got {n_actions}')
    return sigma * math.sqrt(2.0 * math.log(n_actions))


__all__ = [
    'double_greedify_tabular',
    'hasselt_max_bias_asymptotic',
    'hasselt_n2_max_bias',
    'max_greedify_tabular',
]
