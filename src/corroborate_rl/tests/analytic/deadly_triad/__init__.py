"""Deadly-triad analytic test suite.

The deadly triad (Sutton & Barto §11.3): function approximation
+ bootstrapping + off-policy learning together can cause Q to
diverge unboundedly. The framework's primitives that detect the
divergence pattern:

- `q_divergence_score(record, r_max) = jensen_gap · (1 − γ) / r_max`
  — per-cell ratio against the Bellman fixed-point bound
  `|Q*| ≤ r_max / (1 − γ)`. Score < 1 means Q stays bounded (FQI
  regime corroborated); score >> 1 means Q has diverged orders of
  magnitude beyond the bound (CLAIM 11 in
  `findings_minatar_link_attenuation`).

- `fqi_decay_gap(sync_period, gamma)` — across-window sup-norm
  TD-error decay vs Munos 2003 FQI's `γ`-contraction. Gap ≈ 0 →
  contraction holds (FQI mechanism active); gap → (1 − γ) →
  no contraction (deadly triad active).

The closed-form story (auto-memory `findings_fqi_mechanism`):

    long sync  ⇒  FQI regime  ⇒  Q bounded by r_max / (1 − γ)
    short sync ⇒  deadly triad ⇒ Q can diverge unboundedly

Tests assert this against synthetic cells where the relationship
between (sync_period, jensen_gap) is encoded per the FQI
contraction theorem, then framework primitives normalize and
the cross-cell panel analyses recover the closed-form pattern."""
