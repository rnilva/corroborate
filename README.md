# corroborate — paper artifact

This branch (`submission`) is the self-contained archive for

> **Corroborate: A Framework for Testing Mechanism Claims in
> Reinforcement Learning** — *Finding the Frame* workshop,
> RLC 2026. The paper is at [`paper.pdf`](paper.pdf).

Everything the paper reports is reproducible from this branch
alone — the shipped per-hypothesis caches carry all required
measurables, so **no GPU, no cloud credentials, and no retraining
are needed**. The living framework is developed on the
[`main`](../../tree/main) branch; this branch is frozen at the
state of the study (bug-fix-only).

## Requirements

- Python ≥ 3.13 and [`uv`](https://docs.astral.sh/uv/)
- any OS / CPU (JAX runs on CPU here; the CUDA extra is
  Linux-gated and optional)

Every command below auto-builds the environment on first run.

## Reproduce the paper

```bash
# 1. Verdict surface — per-bridge + per-finding verdicts on the
#    cached 12-environment γ=0.99 panel (~1 min).
uv run corroborate hypothesis experiments.findings.hasselt_clean
#    Headline contrast (paper §results):
#      ddqn_reduces_bias__consistently_cross_env    held      (sign test p = 0.019, 10/12 envs)
#      intervention_reduces_bias__pool_inadequate   no_effect (same extent — the pooled
#                                                              estimator cannot see the effect)

# 2. γ=0.99 mediation case-study figures (~2 min). Regenerates
#    papers/g099_mediation/figures/*.csv byte-identical to the
#    committed copies.
bash papers/g099_mediation/run_all.sh

# 3. γ=0.999 harm-regime case-study figures (~2 min, fully
#    offline — the α-dose panels are frozen in papers/g999_harm/data/).
bash papers/g999_harm/run_all.sh

# 4. Framework test suite incl. the synthetic linear-Gaussian SCM
#    positive control (closed-form recovery of a known chain).
uv run pytest tests/
```

## Layout

| path | contents |
|---|---|
| `paper.pdf` | the workshop paper |
| `src/corroborate/` | the framework (typed claims → bridges → causal-graph verdicts) |
| `src/corroborate_rl/` | the DQN/JAX substrate the intervention re-executes |
| `experiments/findings/` | the hypothesis packages (bridges + findings), incl. `hasselt_clean/` — the worked DDQN chain |
| `experiments/data/cache/` | shipped per-hypothesis panels (the paper's data) |
| `experiments/data/*/_remote.json` | provenance manifests for the raw cloud-archived corpora (not needed to reproduce) |
| `experiments/findings/*.run.json` | committed verdict snapshots (drift sentinels) |
| `papers/g099_mediation/` | γ=0.99 five-layer case study — scripts + figures |
| `papers/g999_harm/` | γ=0.999 harm-regime case study — scripts + figures + frozen per-cell panels |
| `docs/HYPOTHESIS_AS_GRAPH.md` | the organizing principle: a hypothesis IS a causal graph |
| `REPRODUCIBILITY.md` | bitwise vs scientific reproducibility under XLA configuration |
| `CLAUDE.md` | contributor doc: typing discipline, vocabulary, canonical analyses |
| `tests/` | framework tests + `tests/analytic/lg_scm/` positive control |

## What the framework claims (one paragraph)

A mechanism claim — e.g. Hasselt 2010's *double estimation reduces
Jensen-gap overestimation bias* — is authored as a causal graph
whose edges are independently testable **bridges**: mechanism
(does the bias shrink?), outcome (does return improve?), and link
(does the one drive the other?) are three separate verdicts, each
per-stratum, each allowed to return `POWER_INSUFFICIENT` rather
than a false `NO_EFFECT`. The paper's demonstration: the mechanism
edge holds consistently across environments while the pooled
outcome estimator is structurally unable to see the effect — a
distinction scalar benchmark summaries cannot express.

## License

MIT — see [`LICENSE`](LICENSE).
