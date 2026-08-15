"""Layer 6 — is there a *better* DDQN? Deep van Hasselt 2010 at Asterix γ=0.999.

If the DDQN *clip* causes the harm (Layer 1), does decoupling selection
from valuation with a genuinely INDEPENDENT evaluator (the deep analogue of
van Hasselt 2010 — symmetric Double-Q, `y_A = r + γ·Q_{B⁻}(argmax Q_A)`) get
the bias fix *without* the harm? See `DEEP2010.md` for the algorithm + its
relationship to DDQL (Nagarajan, White & Machado 2024).

Three arms at Asterix γ=0.999, 1M, n_episodes=20 (matched HPs):
  vanilla (n=30) · DDQN (n=30) · DDQN-indp (n=15).

Two readouts, on the panel's primary RAW (undiscounted) eval:

(A) OUTCOME — DDQN HARMS vanilla (d=−0.80); DDQN-indp RECOVERS the harm
    to ≈ vanilla level (d=+0.68 vs DDQN; d=+0.15 vs vanilla, within noise)
    — bias removed, no residual harm. n=30; the first 15 seeds gave d=−0.21 vs
    vanilla, a small-n low draw the next 15 corrected. Raw is the fair metric —
    the discounted score inflates DDQN-indp because it earns reward earlier.

(B) MECHANISM — every bias / Q-explosion channel collapses MONOTONICALLY
    vanilla → DDQN → DDQN-indp, with DDQN-indp at the floor: it fully
    eliminates the bias (signed Jensen +280 → +127 → −4.4; REDQ rel-bias
    61 → 32 → −0.7) and the Q-explosion (Q-growth 673 → 281 → 2.6) that DDQN
    only halves. But the outcome does NOT track that monotone collapse — the
    necessary-not-sufficient theme (Layer 5) made sharp: DDQN-indp over-
    corrects into a tiny-Q, near-uniform-margin policy (argmax-margin 2.7 →
    1.4 → 0.02; argmax-entropy 0.34 → 0.36 → 1.36), which is why complete
    Q-control recovers the harm yet costs ~1 raw point vs vanilla. (A within-
    panel mediation is degenerate here — the arms separate so completely on
    the mechanism measurables that a mediator collinearly determines the
    arm; this is a contrast/profile story, not a mediation one.)

Input : experiments/data/cache/deep2010_g999_panel.parquet  (proper 3-arm cache,
        framework-recomputed measurables incl. raw eval + signed Jensen)
Output: papers/g999_harm/figures/06_better_ddqn_deep2010.{png,csv}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

_PAPER_DIR = Path(__file__).resolve().parents[1]
_REPO = _PAPER_DIR.parents[1]
CACHE = _REPO / 'experiments/data/cache/deep2010_g999_panel.parquet'
FIG_DIR = _PAPER_DIR / 'figures'

LAB = {
    'baseline': 'vanilla',
    'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)': 'DDQN',
    'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify_indep)': 'DDQN-indp',
}
ARMS = ('vanilla', 'DDQN', 'DDQN-indp')
COLOR = {'vanilla': '#2166ac', 'DDQN': '#b2182b', 'DDQN-indp': '#5e3c99'}
RAW = 'eval_best_burst_raw_mean'
# (column, display label) for the mechanism profile — bias then Q-explosion.
MECH = [
    ('jensen_signed', 'bias: signed Jensen'),
    ('normalized_bias_redq_late', 'bias: REDQ rel.'),
    ('q_late_mean', 'Q-magnitude: late-mean'),
    ('q_growth_max_minus_initial', 'Q-explosion: growth'),
    ('ddqn_bootstrap_gap', 'wedge: tgt − online-argmax'),
    ('q_argmax_margin_late', 'policy: argmax margin'),
]


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return float((a.mean()-b.mean())/sp) if sp > 0 else float('nan')


def main() -> None:
    cache = pl.read_parquet(CACHE).with_columns(pl.col('arm_key').replace(LAB).alias('arm'))
    by = {nm: cache.filter(pl.col('arm') == nm) for nm in ARMS}

    def col(nm: str, c: str) -> np.ndarray:
        return by[nm].get_column(c).to_numpy().astype(float)

    raw = {nm: col(nm, RAW) for nm in ARMS}
    d_harm = cohen_d(raw['DDQN'], raw['vanilla'])
    d_rescue = cohen_d(raw['DDQN-indp'], raw['DDQN'])
    d_resid = cohen_d(raw['DDQN-indp'], raw['vanilla'])

    fig, (axO, axM) = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # ── (A) raw outcome ──
    xs = np.arange(3)
    means = [np.nanmean(raw[nm]) for nm in ARMS]
    sds = [np.nanstd(raw[nm][np.isfinite(raw[nm])], ddof=1) for nm in ARMS]
    ns = [int(np.isfinite(raw[nm]).sum()) for nm in ARMS]
    axO.bar(xs, means, yerr=sds, color=[COLOR[a] for a in ARMS], alpha=0.85,
            capsize=5, edgecolor='black', linewidth=0.6)
    for x, m, n in zip(xs, means, ns):
        axO.text(x, m + 0.3, f'{m:.1f}\n(n={n})', ha='center', va='bottom', fontsize=9)
    axO.axhline(means[0], color=COLOR['vanilla'], ls=':', lw=1, alpha=0.7)
    axO.set_xticks(xs); axO.set_xticklabels(ARMS)
    axO.set_ylabel('RAW best-burst eval (undiscounted)')
    axO.set_ylim(0, max(means) + max(sds) + 4)
    axO.set_title(f'OUTCOME (raw) — DDQN harms (d={d_harm:+.2f});\n'
                  f'DDQN-indp ≈ vanilla (d={d_rescue:+.2f} vs DDQN · {d_resid:+.2f} vs vanilla)',
                  fontsize=9.5, fontweight='bold', color='#b2182b')
    axO.grid(axis='y', alpha=0.25)

    # ── (B) mechanism profile: normalized to |vanilla|, monotone to floor ──
    armx = np.arange(3)
    for i, (c, lab) in enumerate(MECH):
        vals = np.array([np.nanmean(col(nm, c)) for nm in ARMS])
        norm = vals / abs(vals[0]) if abs(vals[0]) > 1e-9 else vals
        axM.plot(armx, norm, marker='o', lw=1.4, color='#888', alpha=0.5,
                 label='6 harm channels: bias · Q-mag · Q-explosion · wedge · policy'
                 if i == 0 else None)
    # outcome (normalized) — the NON-monotone contrast line
    out_norm = np.array(means) / means[0]
    axM.plot(armx, out_norm, marker='s', lw=2.6, color='#1a7d3a', label='raw eval (outcome)')
    for x, v in zip(armx, out_norm):
        axM.annotate(f'{v:.2f}', (x, v), color='#1a7d3a', fontsize=8.5, fontweight='bold',
                     xytext=(0, 6), textcoords='offset points', ha='center')
    axM.axhline(0, color='black', lw=0.8); axM.axhline(1, color='#bbb', ls=':', lw=0.8)
    axM.set_xticks(armx); axM.set_xticklabels(ARMS)
    axM.set_xlim(-0.2, 2.7)
    axM.set_ylabel('value / |vanilla|  (lower = more controlled)')
    axM.set_title("MECHANISM — harm channels collapse to DDQN-indp's floor;\n"
                  'outcome is non-monotone (over-correction is outcome-neutral)',
                  fontsize=9.5, fontweight='bold')
    axM.legend(fontsize=8, loc='lower left')
    axM.grid(axis='y', alpha=0.25)

    plt.suptitle('Layer 6 — a better DDQN: deep van Hasselt 2010 (independent evaluator) '
                 'at Asterix γ=0.999', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / '06_better_ddqn_deep2010.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # CSV — per-arm profile of outcome + mechanism channels.
    prof_cols = [RAW, 'eval_best_burst_mean', 'jensen_gap', 'jensen_signed',
                 'normalized_bias_redq_late', 'q_late_mean', 'ddqn_bootstrap_gap',
                 'q_growth_max_minus_initial', 'q_max_growth', 'q_argmax_margin_late',
                 'argmax_entropy_late', 'q_action_std_late', 'q_mc_burst_correlation_late']
    rows = []
    for nm in ARMS:
        r: dict[str, object] = {'arm': nm, 'n': by[nm].height}
        for c in prof_cols:
            if c in cache.columns:
                r[c] = float(np.nanmean(col(nm, c)))
        rows.append(r)
    pl.DataFrame(rows).write_csv(FIG_DIR / '06_better_ddqn_deep2010.csv')
    print(pl.DataFrame(rows))
    print(f'\nraw-eval d: harm(DDQN vs V)={d_harm:+.2f} · rescue(deep vs DDQN)={d_rescue:+.2f} '
          f'· residual(deep vs V)={d_resid:+.2f}')
    print('saved figures/06_better_ddqn_deep2010.{png,csv}')


if __name__ == '__main__':
    main()
