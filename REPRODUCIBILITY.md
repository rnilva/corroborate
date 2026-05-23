# REPRODUCIBILITY.md — what same-seed actually buys you

## What and why

A common assumption in JAX-based RL is that "same seed + same JIT'd
function = reproducible result." That assumption is **only correct
within a single run on a single GPU**. As soon as the operator
changes XLA flags (or upgrades the JAX/CUDA stack), the same seed
produces a different *scientific outcome*, not just bitwise noise.
This document records what we measured, the mechanism we traced it
to, and what the substrate exposes so authors can make informed
choices.

The findings here come from a controlled four-cell A/B on
`SpaceInvaders-MinAtar` (γ=0.999, n_step=2, lr=2e-5, 30 seeds × 1M
steps × baseline arm, plus 10-seed probes for two intermediate
modes). Corpora:

| Corpus | XLA settings | n | wall |
|---|---|---|---|
| `si_g0999_nstep2_lr2em5_det` | det-ops on, FP32, no cmdbuf | 30 | 5h |
| `si_g0999_nstep2_lr2em5` | det-ops off, TF32, cmdbuf | 30 | 1h |
| `si_g0999_nstep2_lr2em5_matmul_highest` | det-ops off, **FP32**, cmdbuf | 10 | 20m |
| `si_g0999_nstep2_lr2em5_no_cmdbuf` | det-ops off, TF32, **no cmdbuf** | 10 | 19m |

All four share environment, seeds, hyperparameters, and the
substrate commit. The only varying knobs are XLA-level.

## Two reproducibility kinds, only one of which "same seed" guarantees

1. **Bitwise reproducibility** — running the same script with the
   same seed on the same hardware with the same XLA flag set
   produces byte-identical traces. Same-seed + `--deterministic`
   gives this. Same-seed without `--deterministic` does **not**:
   `cuLaunchKernel` scheduling jitter and atomic-reduction ordering
   produce ~1e-9 per-op drift that compounds.

2. **Scientific reproducibility** — running the same script with the
   same seed under *different XLA configurations* produces an
   outcome distribution that supports the same scientific claim.
   This is what bridge / analysis verdicts depend on.

The two are independent. Bitwise reproducibility is preserved by
the determinism flag; scientific reproducibility is **not** — see
next section.

## Same seeds, same vmap, different basins

`SpaceInvaders` outcome distributions across the four configurations:

| Mode | `eval_final_mean` | `late_window_mean` | `jensen_gap` |
|---|---|---|---|
| det                                                  | 32.5 ± 15.4 | 32.8 ± 5.3 | 4.18 ± 1.7 |
| non-det (TF32, cmdbuf)                               | 24.0 ± 7.3  | 25.6 ± 2.2 | 4.79 ± 1.4 |
| non-det + `JAX_DEFAULT_MATMUL_PRECISION=highest`     | 23.1 ± 7.2  | 24.5 ± 2.0 | 5.26 ± 1.3 |
| non-det + `XLA_FLAGS=--xla_gpu_enable_command_buffer=` | 27.6 ± 10.5 | **31.8 ± 4.6** | 4.60 ± 0.9 |

The `late_window_mean` column is the clean signal — `eval_final_mean`
is averaged over fewer episodes per cell so its variance is larger
than the inter-mode shift.

**Two configurations land on the "det" basin (mean ~32) and two
land on the "non-det" basin (mean ~25).** The discriminator is *not*
the determinism flag and is *not* TF32 vs FP32. It is **whether
XLA captures the inner training loop as a CUDA Graph (command
buffer)**:

- det path: cmdbuf disabled (determinism forces no-capture) → det basin.
- non-det + cmdbuf disabled: still TF32, still atomic scatter, but no graph capture → **det basin**.
- non-det + matmul=highest: FP32 instead of TF32, but cmdbuf still on → **non-det basin**.

Cross-seed standard deviations follow the same pattern: cmdbuf-on
modes have sd ≈ 2.0–2.2 (tight loop attractor), cmdbuf-off modes
have sd ≈ 4.6–5.3 (broader exploration).

## Why seed-vmap doesn't save you

A natural intuition: "I vmap over seeds inside one JIT'd cell, so
all 30 seeds share the same compiled graph; whatever non-determinism
the graph has, it hits all seeds symmetrically, and cross-seed
averaging should erase it." This is wrong, in two ways:

1. **The systematic bias is not zero-mean across seeds.** A
   command-buffer captures one tile/reduction schedule. If that
   schedule has any directional rounding (e.g., a tree-reduction
   order that systematically accumulates positive partials earlier),
   *every* seed in the vmap inherits the same bias. 30 seeds vmap'd
   under cmdbuf converge to the same loop attractor; 30 seeds
   without cmdbuf explore independent neighborhoods.

2. **Argmax is a discontinuity.** At step ~100 the first gradient
   update lands. By step ~3000 the accumulated ε-scale numerical
   drift between two configurations crosses the Q-gap at some
   state and flips `argmax_a Q(s, a)`. From that step forward the
   two runs are exploring *different MDP trajectories*; the rest of
   training is no longer comparing the same algorithm under noise,
   it is two algorithms learning different state distributions.

The combination is what produces the 1.8 σ shift in `late_window_mean`
between cmdbuf-on and cmdbuf-off, even though both groups vmap 30
seeds in identical fashion.

## What the substrate records and exposes

To make this explicit at the data layer:

- **Provenance column on every RunRow**: `xla_deterministic_ops:
  bool` is stamped per cell at run time, recording the active
  `--xla_gpu_deterministic_ops` setting (read from `XLA_FLAGS`).
  Downstream consumers can filter / stratify / refuse cross-mode
  pooling via `pl.col('xla_deterministic_ops').is_in([True])`. See
  `corroborate_rl/cell_runner.py::_xla_deterministic_ops`.

- **CLI + YAML toggle**: `corroborate sweep run --no-deterministic …`
  or top-level `deterministic: false` in the sweep YAML disables
  the determinism stamp; both also opt out of the substrate's
  default `--xla_gpu_deterministic_ops=true` append to `XLA_FLAGS`.
  CLI > YAML > default (True). Operator-set `XLA_FLAGS` always
  wins (the substrate respects an explicit operator value). See
  `corroborate_rl/dqn_sweep.py::set_jax_env`.

- **Other knobs (operator-set via env, no substrate-level
  surface)**: `JAX_DEFAULT_MATMUL_PRECISION=highest` forces FP32
  matmul (no TF32 truncation); `XLA_FLAGS=--xla_gpu_enable_command_buffer=`
  disables CUDA Graph capture under non-det. These are not stamped
  to provenance today; if a sweep uses them, document the YAML
  comment block accordingly.

## Operational recipes

Pick the recipe that matches the use case:

| Use case | Config | Wall (SI 1M-step ref) |
|---|---|---|
| Cross-run bitwise reproducibility | `--deterministic` (default) | 5h |
| Pretty fast, scientifically comparable to det | `--no-deterministic` + `XLA_FLAGS=--xla_gpu_enable_command_buffer=` | 19m |
| Fastest, accepts a different scientific basin | `--no-deterministic` (substrate default when set) | 1h |

**The middle option is the operational sweet spot for most science:
~15× faster than full determinism, with outcome distributions that
match the deterministic basin within the cross-seed variance.** Sweeps
that mix this with the default non-det are not pooled cleanly — the
`xla_deterministic_ops` provenance column is necessary but not
sufficient for the cmdbuf distinction (no provenance column for
command buffer state exists today; encode it in the corpus name).

## Cross-mode analysis discipline

A panel of corpora that span multiple XLA configurations is
**not** a single population. Treat XLA configuration as a
nuisance covariate that interacts with the loop-attractor
dynamics:

- For univariate verdicts (e.g., DDQN-vs-baseline outcome Δ),
  the analysis must stratify on (or filter to) one XLA mode.
  Pooling cmdbuf-on with cmdbuf-off produces a Simpson-shaped
  artifact: the within-mode effect may be small but the
  between-mode shift can dominate the pooled effect size.

- For mediator analyses, the mediator's relationship to outcome
  is intact within a mode but the *absolute* mediator value
  (e.g., `jensen_gap` 4.18 vs 4.79 vs 5.26) is mode-dependent.
  Cross-mode mediation effects are not interpretable without an
  explicit conditional on mode.

- For pre-registered scope claims, the XLA mode is part of the
  scope. A pre-registered claim "DDQN helps at SI γ=0.999"
  measured under one configuration does NOT pre-register the
  same claim under a different configuration; that is a fresh
  empirical question.

## Pre-flight check before launching a sweep

The substrate's default is `--deterministic` (cmdbuf disabled,
FP32, slow). For long sweeps on a non-Jumanji-class env where
the docstring's "negligible at MinAtar scale" claim has been
empirically refuted (see `dqn_sweep.py::set_jax_env`), the
recommended invocation is:

```
XLA_FLAGS=--xla_gpu_enable_command_buffer= \
    corroborate sweep run \
        --substrate corroborate_rl.dqn_sweep \
        --device gpu --profile r2 \
        --no-deterministic \
        experiments/configs/<sweep>.yaml
```

Or set both flags in the YAML's comment header so the corpus
name self-documents the configuration. Per-cell provenance is
limited to `xla_deterministic_ops`; the cmdbuf state has to be
inferred from the corpus name + this document for now.

## Open items

- The cmdbuf state should also be stamped to RunRow provenance
  (parallel to `xla_deterministic_ops`). Today it isn't; the
  substrate's `set_jax_env` doesn't touch the command-buffer flag,
  so the operator's value flows through unchecked.
- The substrate's docstring still talks about "negligible perf
  overhead at MinAtar scale" with a 271 s vs 273 s 1M-step
  Asterix benchmark. That measurement was at a chunk size /
  hardware that didn't exercise the cmdbuf code path; it is
  empirically false at chunk=10–15 on SI / RTX 5090 (5h vs
  19m). The number in the docstring should be replaced with
  the measured 15× wall ratio.
- A cmdbuf-disabling YAML field (sibling of the new
  `deterministic` field) would be the natural next plumbing
  step. The lightweight pre-import YAML peek already exists in
  `_resolve_deterministic`; an analogous `_resolve_cmdbuf`
  could land alongside.
