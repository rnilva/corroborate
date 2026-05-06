# Sync-curve link-attenuation on Breakout-MinAtar

**Date:** 2026-05-05
**Predicate tested:** `findings_minatar_link_attenuation.md` predicted that
sync=10000 (Q-stabilized regime) should keep the per-burst link active
throughout 20 bursts, while sync=100 dies after Phase 1 (plc=0.05). The
intermediate sync ∈ {1000, 3000} points complete the curve.

## Data

Breakout-MinAtar, 30 paired seeds × 20 bursts × 1M steps × CNN.

| sync | source |
|---|---|
| 100 | `minatar_1M/per_burst_arrays.parquet` |
| 1000 | `minatar_sync_curve/ddqn_sync1k/{runs,traces}.parquet` |
| 3000 | `minatar_sync_curve/ddqn_sync3k/{runs,traces}.parquet` |
| 10000 | `minatar_sync_intervention/runs_with_bridge_cache.parquet` |

Per-burst MC and Q-at-start are mean-over-episodes of the
(n_bursts=20, n_eps=5) trace columns. Bias is Q − MC.
`paired_link_per_burst` with target=mc_per_burst, predictor=bias_per_burst,
paired by seed.

## Headline

The per-burst link is **monotone in sync_period** — higher sync = link
active for more of training. Three regimes emerge:

| sync | plc | β at burst 0 | β at burst 8 | β at burst 19 |
|---|---|---|---|---|
| 100 | **0.05** | +0.35 (r=0.43) | 3e-5 | 6e-7 |
| 1000 | **0.40** | +0.88 (r=0.97) | 0.0016 | 5e-6 |
| 3000 | **0.40** | +0.99 (r=1.00) | 0.040 (ns) | 3e-4 |
| 10000 | **0.85** | +1.00 (r=1.00) | **0.63 (r=0.88)** | 0.013 |

## Key observations

1. **β at burst 0 saturates near 1.0 for sync ≥ 1000.** Phase 1 has perfect
   1:1 conversion of bias-correction to outcome at every non-extreme sync.
   Only sync=100 has reduced β at burst 0 (β=0.35) — Q-explosion is already
   biting in the very first eval window.

2. **Phase-2 onset delays monotonically with sync.** Burst index where β
   drops below 0.1: sync=100 → burst 1, sync=1000 → burst 5, sync=3000 → burst 6,
   sync=10000 → burst 13+. Each 10× in sync buys ~5 more bursts of active link.

3. **Same plc at sync=1000 and sync=3000 (both 0.40)** — the active window is
   slightly longer at sync=3000 but doesn't add enough bursts to change plc.
   The plc transition is between sync=3000 and sync=10000, where it jumps
   0.40 → 0.85. Suggests the qualitative regime change is around sync=5k-7k,
   not a smooth function of log(sync).

4. **At sync=10000 burst 8, β=0.63 with r=0.88 (p=2e-12)** — strongly active
   where sync=100 has β=3e-5. The β ratio at this burst is **2 × 10⁴**, four
   orders of magnitude. This is the same magnitude prediction the original
   sync=100 finding made for the within-run β collapse — here it's the
   cross-sync amplitude at a fixed burst.

5. **All sync values still show some β collapse late.** Even sync=10000 has
   β dropping from 1.00 (burst 0) → 0.013 (burst 19). The Q magnitude isn't
   fully bounded; it just grows slower. plc=0.85 (not 1.00) reflects the
   late-burst tails where the link's r drops below significance even though
   the sign is still positive.

## What this says about the bridge

The original `findings_minatar_link_attenuation.md` deferred a bridge
(`ddqn_link_dies_after_q_divergence__minatar`) until cross-corroboration. The
sync curve provides that corroboration **on Breakout alone**, with monotone
plc(sync_period). The Q-explosion → β-collapse → link-attenuation chain is
not a property of "MinAtar at default sync" but a property of "MinAtar with
fast target sync" — and the dose-response shape on a single env at four
sync values is more interpretable than a binary contrast.

When Freeway+SI complete (resume sweep `minatar_sync_curve_resume.yaml`),
this becomes a 3-env × 4-sync panel and the bridge can be authored against
the per-(env, sync) plc trajectory.

## Next steps

- [ ] Wait for Freeway+SI resume run to finish; rerun this analysis with
      env_name iterating Freeway-MinAtar / SpaceInvaders-MinAtar.
- [ ] If plc(sync) shape is similar across all 3 envs, author the deferred
      bridge with a per-env plc threshold + sync_period as the moderator.
- [ ] Consider a sync=5000 / 7000 / 8000 mini-sweep on Breakout to localize
      the qualitative transition (plc 0.40 → 0.85).

## Reproduction

```bash
uv run python experiments/findings/sync_curve_breakout/run_analysis.py
```

Output: `experiments/findings/sync_curve_breakout/sync_curve_panel.json`.
