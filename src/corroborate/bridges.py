"""Generic bridge factories — common assertion shapes turned into
typed `Bridge[R]` instances.

Each factory takes shape parameters (which record key, what
threshold, what comparison), returns a `Bridge[Mapping[str,
object]]` ready to attach to a `Hypothesis`. Substrate authors
who want one of these shapes don't hand-write the per-cell body
+ wire `BridgeResult.stats` themselves — they call the factory.

Ported from v10's `bridges.py` with verdict-naming adjusted to
corroborate's typology (`HELD` / `NO_EFFECT` / `INVARIANT_VIOLATION`
in place of v10's `ADMIT` / `REJECT` / `INCONCLUSIVE`).

The factory set covers four shapes:

- `monotonic(key, ...)` — late-half mean exceeds early-half mean
  by `threshold`. Standard "X grows over training."
- `correlation(a, b, ...)` — Spearman ρ has expected sign and
  exceeds magnitude threshold. Standard "X tracks Y."
- `mean_exceeds(key, ...)` — mean over the full series exceeds
  `threshold`. Coarse-grained version of `monotonic`.
- `variance_shrinks(key, ...)` — late-half variance / early-half
  variance below `ratio`. Standard "X stabilises over training."

Three deferred (joint-partial, feedback-lag, etc.) until a real
substrate consumer arrives — speculative without a use site."""
from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.stats import spearmanr  # type: ignore[reportMissingTypeStubs]

from corroborate.bridge import Bridge, BridgeResult, bridge
from corroborate.verdict import Verdict


def _as_1d(
    record: Mapping[str, object], key: str,
) -> npt.NDArray[np.float64]:
    """Pull `record[key]` as a 1-D float64 numpy array. Raises
    KeyError if the key is missing — bridges declare the keys
    they consume; missing data is a contract violation."""
    if key not in record:
        raise KeyError(
            f'bridge factory expected record[{key!r}] present',
        )
    arr = np.asarray(record[key], dtype=np.float64)
    if arr.ndim == 0:
        return arr[None]
    return arr.ravel() if arr.ndim > 1 else arr


def monotonic(
    key: str, *, threshold: float = 0.0,
    name: str | None = None,
) -> Bridge[Mapping[str, object]]:
    """Bridge: HELD iff `mean(later half) − mean(earlier half) >
    threshold`. NO_EFFECT otherwise.

    Standard 'X grows over training' assertion. The threshold
    is the minimum detectable monotonic increase — set to a small
    positive value to require strict growth, 0.0 to test
    direction only."""
    resolved_name = name or f'monotonic({key})'

    @bridge(targets=(key,), name=resolved_name)
    def fn(record: Mapping[str, object]) -> BridgeResult:
        arr = _as_1d(record, key)
        n = len(arr)
        early = float(arr[: n // 2].mean())
        late = float(arr[n // 2:].mean())
        diff = late - early
        verdict = Verdict.HELD if diff > threshold else Verdict.NO_EFFECT
        return BridgeResult(
            verdict=verdict,
            reason=(
                f'late−early = {diff:+.3f} vs threshold '
                f'{threshold:+.3f}'
            ),
            stats={'early': early, 'late': late, 'value': diff,
                   'threshold': threshold},
            name=resolved_name,
            targets=(key,),
        )
    return cast(Bridge[Mapping[str, object]], fn)


def correlation(
    a: str, b: str, *,
    expected_sign: int = 1, threshold: float = 0.3,
    name: str | None = None,
) -> Bridge[Mapping[str, object]]:
    """Bridge: HELD iff Spearman ρ(a, b) has `expected_sign` AND
    |ρ| ≥ threshold. NO_EFFECT on sign mismatch or magnitude below
    threshold. INVARIANT_VIOLATION when ρ undefined (constant
    series).

    Standard 'X tracks Y over training' assertion.
    `expected_sign` is +1 for predicted positive correlation, -1
    for negative, 0 for two-sided (any sign that exceeds
    threshold)."""
    resolved_name = name or f'correlation({a},{b})'

    @bridge(targets=(a, b), name=resolved_name)
    def fn(record: Mapping[str, object]) -> BridgeResult:
        xa = _as_1d(record, a)
        xb = _as_1d(record, b)
        if float(xa.std()) == 0.0 or float(xb.std()) == 0.0:
            return BridgeResult(
                verdict=Verdict.INVARIANT_VIOLATION,
                reason='constant series; ρ undefined',
                stats={'rho': float('nan')},
                name=resolved_name,
                targets=(a, b),
            )
        rho_raw, _ = spearmanr(xa, xb)  # type: ignore[reportUnknownMemberType]
        rho = float(rho_raw)  # type: ignore[reportUnknownArgumentType]
        if not np.isfinite(rho):
            return BridgeResult(
                verdict=Verdict.INVARIANT_VIOLATION,
                reason='ρ NaN',
                stats={'rho': rho},
                name=resolved_name,
                targets=(a, b),
            )
        sign_ok = (
            expected_sign == 0
            or int(np.sign(rho)) == int(np.sign(expected_sign))
        )
        mag_ok = abs(rho) >= threshold
        if sign_ok and mag_ok:
            return BridgeResult(
                verdict=Verdict.HELD,
                reason=(
                    f'ρ = {rho:+.3f} (sign={expected_sign:+d}, '
                    f'|ρ| ≥ {threshold:.2f})'
                ),
                stats={'rho': rho, 'threshold': threshold},
                name=resolved_name, targets=(a, b),
            )
        failures: list[str] = []
        if not sign_ok:
            failures.append(
                f'wrong sign (got {rho:+.3f}, want {expected_sign:+d})',
            )
        if not mag_ok:
            failures.append(f'|ρ| = {abs(rho):.3f} < {threshold:.2f}')
        return BridgeResult(
            verdict=Verdict.NO_EFFECT,
            reason='; '.join(failures),
            stats={'rho': rho, 'threshold': threshold},
            name=resolved_name, targets=(a, b),
        )
    return cast(Bridge[Mapping[str, object]], fn)


def mean_exceeds(
    key: str, *, threshold: float = 0.0,
    name: str | None = None,
) -> Bridge[Mapping[str, object]]:
    """Bridge: HELD iff mean(record[key]) > threshold."""
    resolved_name = name or f'mean_exceeds({key})'

    @bridge(targets=(key,), name=resolved_name)
    def fn(record: Mapping[str, object]) -> BridgeResult:
        arr = _as_1d(record, key)
        m = float(arr.mean())
        return BridgeResult(
            verdict=(
                Verdict.HELD if m > threshold else Verdict.NO_EFFECT
            ),
            reason=f'mean = {m:+.4f} vs threshold {threshold:+.4f}',
            stats={'mean': m, 'value': m, 'threshold': threshold},
            name=resolved_name, targets=(key,),
        )
    return cast(Bridge[Mapping[str, object]], fn)


def variance_shrinks(
    key: str, *, ratio: float = 0.8,
    name: str | None = None,
) -> Bridge[Mapping[str, object]]:
    """Bridge: HELD iff `var(later half) / var(earlier half) <
    ratio`. NO_EFFECT otherwise. INVARIANT_VIOLATION when early
    variance is zero (ratio undefined).

    Standard 'X stabilises over training' assertion. `ratio` is
    the maximum allowed late/early variance fraction; ratio=0.5
    requires variance to halve."""
    resolved_name = name or f'variance_shrinks({key})'

    @bridge(targets=(key,), name=resolved_name)
    def fn(record: Mapping[str, object]) -> BridgeResult:
        arr = _as_1d(record, key)
        n = len(arr)
        v_early = float(arr[: n // 2].var())
        v_late = float(arr[n // 2:].var())
        if v_early == 0:
            return BridgeResult(
                verdict=Verdict.INVARIANT_VIOLATION,
                reason='early variance is 0; ratio undefined',
                stats={'var_early': v_early, 'var_late': v_late},
                name=resolved_name, targets=(key,),
            )
        r = v_late / v_early
        verdict = Verdict.HELD if r < ratio else Verdict.NO_EFFECT
        cmp = '<' if r < ratio else '≥'
        return BridgeResult(
            verdict=verdict,
            reason=f'var ratio late/early = {r:.3f} {cmp} {ratio:.2f}',
            stats={'var_early': v_early, 'var_late': v_late,
                   'ratio': r, 'threshold': ratio},
            name=resolved_name, targets=(key,),
        )
    return cast(Bridge[Mapping[str, object]], fn)
