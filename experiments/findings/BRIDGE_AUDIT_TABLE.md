# Bridge audit — step 1 verdict table

**Corpus snapshot:** `experiments/data/cache/ddqn_universe.parquet`
(1520 cells, 15 post-fix corpora, 12 envs)

**Run JSON:** `experiments/findings/ddqn_universe.run.json`
(timestamp 2026-05-11T01:51, git_commit 9d6d105)

**Drift state:** 13 corpora missing `outcome_native` measurable
(does not affect bridges targeting `eval_best_burst_mean` / `mc_return` / `eval_final_mean`).

## Distribution

| Tag | Count |
|---|---|
| SURVIVED | 11 |
| STALE | 0 |
| REFUTED (HELD-to-NO_EFFECT under honest methodology) | 5 |
| POWER_COLLAPSED | 2 |
| DEAD | 12 |
| POWER_INSUFFICIENT | 10 |
| NO_EFFECT (data refutes; non-HELD bridges with non-zero scope) | 0 |
| SCOPE_VACATED (pending re-run, not a verdict on the claim) | 13 |

*SCOPE_VACATED bridges have 0 cells in current scope — usually because pre-fix corpora used incompatible measurables/configs. They are a holding pen for re-runs, not verdicts. Reviewer-4 flagged that mixing them into the verdict distribution is misleading; treating as a separate status keeps the audit honest.*

*SCOPE_VACATED sunset clock (reviewer-5 follow-up): of the 13 SCOPE_VACATED bridges, 6 are pending the in-flight `reward_scale_sweep_postfix` family re-runs (rescue / no-rescue / argmaxH siblings) and 4 are CLAIM 7g–j auxiliary-route probes pending user decision. The remaining 3 (`acrobot_per_burst_link_active__gamma_0999`, `ddqn_curve_crosses_vanilla_late__spaceinvaders`, `ddqn_null_under_monte_carlo__fourrooms_n10`) are "wait for substrate-version-matched corpus"; if those corpora are not actively rebuilt by the next session, promote to DEAD.*

*Counts are post-trace-restore (cache rebuilt 2026-05-11; all bridges re-evaluated at n_pre_scope=1824). Sibling robustness-check bridges (e.g. `metamaze_link_steeper_at_high_gamma__median`) testing the same empirical signal as their parent are not separately counted.*
| ERRORED | 0 (errored bridge counted in DEAD) |
| **Total** | **48** |

Step-2 cuts to date: 10 total —
- CLAIM 17 (chain_amplifier_link_active_in_bounded_q) — migrated
  paired_link_per_burst → stratum_effect_panel + panel_regress
  to test honestly; revealed cross-env signal is leverage-driven
  by 2-3 high-bias envs. Empirical content preserved by CLAIM
  26b's stratified-DL pool (leverage-robust). (2026-05-11)
- ALSO (earlier in audit) —
- CLAIM 4 (bf→g_link cross-env, errored)
- CLAIM 16 (bf-cross-env deletion-memo trace)
- CLAIM 21 REACH-polyak (sign-flipped at n=5)
- CLAIM 26 (slope-predictor, superseded by 26b)
- CLAIM 6 (mc_variance, refuted via CV decomposition)
- CLAIM 18 (algorithmic-activation, explicit placeholder)
- CLAIM 7g (sparsified_acrobot, auxiliary route probe)
- CLAIM 7h (densified_fourrooms, auxiliary route probe)
- CLAIM 7i (noisy_acrobot, auxiliary route probe)
- CLAIM 7j (noisy_metamaze, auxiliary route probe)

Post-cut: **39 bridges** in BRIDGES tuple (down from 48 →
−19% reduction). All CLAIM 7 g/h/i/j synthesis preserved
in a single deletion-memo banner in the source file.

Initial INVARIANT_VIOLATION on `ddqn_refuted_when_dormancy_fires`
retagged STALE after diagnosis. Initial XFAIL_VIOLATED ×4
resolved per-bridge: 2 POWER_COLLAPSED (sign-correct, magnitude
eroded — keep, queue for more data), 2 DEAD (one sign-flipped
under added configs, one explicitly superseded by 26b). See
diagnosis notes below.

## Per-bridge table

| # | Bridge | Tag | Verdict | predicted_direction | n_in_scope | n_pre_scope |
|---|---|---|---|---|---|---|
| 1 | `argmax_entropy_predicts_link_power__survive_envs` | POWER_COLLAPSED | (2026-05-11 screening: verdict shifted SURVIVED-held → NO_EFFECT post-rebuild. coef=+0.377 (sign matches predicted a_gt_b) sig at p=0.021, but below slope_threshold=0.5. Same "sign-correct, magnitude eroded" pattern as polarity-tautology bridge — more SURVIVE configs at intermediate sync would clarify) | a_gt_b | 534 | 1824 |
| 2 | `argmax_entropy_shadowed_by_jens` | SURVIVED | held (migrated `partial_spearman_paired_delta` → JCI `stratified_partial_spearman`; ρ=+0.011 n=717 across 11 envs after trace restore) | null | 717 | 1824 |
| 3 | `chain_amplifier_link_active_in_bounded_q` | DEAD | (migrated → panel_regress; signal leverage-driven; cut) | a_lt_b | 967 | 1520 |
| 4 | `ddqn_helps_at_full_bootstrap__fourrooms_n1` | SURVIVED | held | - | 300 | 1520 |
| 5 | `ddqn_helps_under_three_gate_scope__cross_env` | SURVIVED | held | a_gt_b | 1100 | 1520 |
| 6 | `eff_h_mediates_g_link__survival_envs` | SURVIVED | held (re-authored 2026-05-11 per reviewer feedback: `proportion_mediated`-null framing was wrong question for JCI primitive; re-author with `a_gt_b` aligned to polarity-tautology prior; ρ=+0.656 matches `findings_polarity_mediator.md` shape) | a_gt_b | 307 | 1824 |
| 7 | `fourrooms_action_dim_link_active__inflated` | SURVIVED | held (descriptive: slope=-0.36, R²=0.89, monotonic 4-point panel) | a_lt_b | 235 | 1520 |
| 8 | `metamaze_link_steeper_at_high_gamma` | REFUTED | (reformulated → stratum_effect_panel mean; γ-amplification flips sign: +1.01→−2.23). Sibling `__median` (8b) also REFUTED for same empirical signal — count once. | a_gt_b | 111 | 1824 |
| 8b | `metamaze_link_steeper_at_high_gamma__median` | (sibling of #8 — same empirical signal, robustness check via median) | (same as #8; not separately counted in totals) | a_gt_b | 111 | 1824 |
| 9 | `reach_link_backdoor_ate_negative` | SURVIVED | held | - | 656 | 1520 |
| 10 | `reach_link_placebo_refuted` | SURVIVED | held | - | 656 | 1520 |
| 11 | `reach_link_rcc_robust` | SURVIVED | held | - | 656 | 1520 |
| 12 | `ddqn_refuted_when_dormancy_fires` | POWER_INSUFFICIENT | (2026-05-11 fixed via NEW `jensen_dormancy_gap_at_best_burst` measurable — collocates dormancy and outcome at the SAME burst, resolving the misalignment surfaced in CLAIM 26b authoring. Substantive challenge to the necessary-condition claim PERSISTS even with alignment: Acrobot d=+0.43 CI=[+0.064, +0.794] excludes 0; SI d=+0.37 CI=[+0.025, +0.722] excludes 0. Both span +0.2 substantive-ceiling. The earlier "misalignment artifact" hypothesis is partly refuted — DDQN helps on best-burst-dormant cells on Acrobot+SI for real, not artifact. Pooled d=+0.19, very close to +0.2 ceiling. Verdict POW_INSUF stays — per-env CIs straddle the substantive bound — but the empirical content shifted: necessary-condition claim is genuinely challenged, not just a window-misalignment artifact) | null | 606 | 2064 |
| 13 | `cross_config_staleness_slope_negative__survive` | SURVIVED | held (2026-05-11 migrated `cross_config_paired_slope` → `stratum_effect_panel` + inline Spearman per RL seed-pairing critique; per-config Δs = arm-mean-diff cleanly. n=7 configs (was 5), ρ≈−0.81 p<0.1 → HELD. PacMan is positive-outcome outlier; rank Spearman robust. Previous NO_EFFECT was at n=5 with stricter threshold — more configs from rebuild pushed past threshold) | a_lt_b | 534 | 2064 |
| 14 | `cross_config_staleness_slope_positive__reach_polyak` | DEAD | no_effect | a_gt_b | 252 | 1520 |
| 15 | `link_r_predictable_from_polarity__soft_tautology` | DEAD | (CUT 2026-05-11. Two reasons: (1) **seed-pairing critique** — `paired_link_per_env` uses per-pair Δs which RL substrate doesn't actually pair (trajectories diverge step 1+). Migration would either give within-arm r per env (tautological — equals polarity by definition) or duplicate CLAIM 26b's cross-env panel. (2) **empirical collapse** — post-rebuild n_strata 8→10 dropped β=+0.61 (sig, R²=0.83) to β=+0.37 (ns, R²=0.11) when PacMan-jumanji entered as sign-mismatch. Substance preserved in eff_h_mediates_g_link__{goal,survival}_envs JCI form which tests per-env Spearman, no pairing needed) | a_gt_b | 1760 | 2064 |
| 16 | `link_slope_predicted_by_g1__cross_env` | DEAD | no_effect | a_lt_b | 1400 | 1520 |
| 17 | `algorithmic_activation_rate_mediates_link__bounded_q` | DEAD | power_insufficient | a_gt_b | 967 | 1520 |
| 18 | `eff_h_mediates_g_link__goal_envs` | SURVIVED | held (re-authored 2026-05-11 per reviewer feedback: same as bridge #6; `a_lt_b` aligned to polarity-tautology prior; ρ=−0.593 matches `findings_polarity_mediator.md` shape) | a_lt_b | 737 | 1824 |
| 19 | `effh_predicts_link_power__reach_envs` | REFUTED | post-rebuild: per-burst meta-regression coef(eff_h)=−0.0046 p=0.04 **opposite sign** to a_gt_b prediction. Env-mean Pearson r=+0.975 (n=4) was the cited evidence; per-burst slope inverts due to late-burst Q-growth (`findings_fourrooms_time_series.md` phase structure) | a_gt_b | 312 | 1824 |
| 20 | `extreme_q_divergence_attenuates_link__binary` | POWER_INSUFFICIENT | (2026-05-11 screening + refinement: was ERRORED on post-rebuild because SlidingTile big_cnn probe (q_network.channels=(32,64), 4 cells) entered scope alongside main sweep (channels=(8,16) or null). First-pass fix `eval_every == 50000` dropped MLP envs too (over-restrictive); refined to exclude only `q_network.channels='(32,64)'`, recovering n=560 (matches pre-rebuild). ATE ≈ 0 → POW_INSUF) | - | 560 | 1824 |
| 21 | `extreme_q_divergence_attenuates_link__placebo_refuted` | POWER_INSUFFICIENT | (was ERRORED — same architecture-exclusion fix as #20; now produces POW_INSUF) | - | 560 | 1824 |
| 22 | `extreme_q_divergence_attenuates_link__rcc_robust` | POWER_INSUFFICIENT | (was ERRORED — same architecture-exclusion fix as #20; now produces POW_INSUF) | - | 560 | 1824 |
| 23 | `mc_variance_attenuates_g_link__between_env` | DEAD | power_insufficient | - | 1520 | 1520 |
| 24 | `q_divergence_shadowed_by_jens` | REFUTED | (migrated `partial_spearman_paired_delta` → JCI `stratified_partial_spearman`; ρ=−0.432 n=717 across 11 envs after trace restore — γ-induced residual leaks at this stratification level; algebraic shadow holds only within fixed (env, γ)) | null | 717 | 1824 |
| 25 | `acrobot_per_burst_link_active__gamma_0999` | SCOPE_VACATED | power_insufficient | - | 0 | 1520 |
| 26 | `adaptive_dqn_fails_to_avoid_attenuation__spaceinvaders_1m` | SCOPE_VACATED | power_insufficient | - | 0 | 1520 |
| 27 | `adaptive_dqn_recovers_ddqn_benefit__fourrooms_factor_0p5` | SCOPE_VACATED | power_insufficient | - | 0 | 1520 |
| 28 | `ddqn_benefit_scales_with_effective_horizon__fourrooms` | SCOPE_VACATED | power_insufficient | - | 0 | 1520 |
| 29 | `ddqn_benefit_scales_with_effective_horizon__metamaze_high_gamma` | SCOPE_VACATED | power_insufficient | - | 0 | 1520 |
| 30 | `ddqn_concentrates_argmax__noisy_acrobot` | DEAD | power_insufficient | a_lt_b | 0 | 1520 |
| 31 | `ddqn_concentrates_argmax__noisy_metamaze` | DEAD | power_insufficient | a_lt_b | 0 | 1520 |
| 32 | `ddqn_concentrates_argmax__sparsified_acrobot` | DEAD | power_insufficient | a_lt_b | 0 | 1520 |
| 33 | `ddqn_curve_crosses_vanilla_late__spaceinvaders` | SCOPE_VACATED | (2026-05-11 screening: verdict-logic bug fixed — n=0 was returning NO_EFFECT instead of POW_INSUF. Now POW_INSUF when env_strata empty. Stays SCOPE_VACATED until SI 1M post-fix corpus exists) | - | 0 | 1824 |
| 34 | `ddqn_does_not_concentrate_argmax__densified_fourrooms` | DEAD | power_insufficient | null | 0 | 1520 |
| 35 | `ddqn_does_not_rescue__acrobot_rs_0p1` | POWER_INSUFFICIENT | (migrated 2026-05-11 to native-unit CI-vs-ceiling: md_native=+0.229, 95% CI=[−0.130, +0.588], null_ceiling=0.2 — CI spans ceiling. Previous Hedges'-g HELD reading was generous: failure-to-reject ≠ confirmation; n=30 inadequate for strict null at this ceiling) | null | 30 | 1824 |
| 36 | `ddqn_does_not_rescue__cartpole_rs_0p1` | POWER_INSUFFICIENT | (migrated 2026-05-11 to native-unit CI-vs-ceiling: md_native=+0.212, 95% CI=[−0.155, +0.579], null_ceiling=0.2 — CI spans ceiling. Same shape as Acrobot sister) | null | 30 | 1824 |
| 37 | `ddqn_dominates_vanilla_response_curve__fourrooms_rs_0p3` | POWER_COLLAPSED | (2026-05-11 migrated `paired_g` → `arm_mean_diff` per RL seed-pairing critique: trajectories diverge so seed-pairing isn't a real pair. md=+0.259 (sign matches a_gt_b), 95% CI=[+0.167, +0.351] just barely spans threshold +0.35 — POW_INSUF, sign correct, magnitude eroded vs older corpus +0.50 plateau. SE 0.046 (paired) vs 0.047 (arm_mean_diff) — confirms seed-pairing wasn't extracting any pair-correlation, as expected for diverged RL trajectories) | a_gt_b | 60 | 1824 |
| 38 | `ddqn_entropy_matches_vanilla__fourrooms_rs_1p0` | REFUTED | post-rebuild: g=−1.72 mean_diff=−0.099 p=2.2e-9 n=30 — null refuted via SIGN_FLIP; DDQN sharpens policy at rs=1.0 too. "Regime-specific argmaxH effect" reading refuted | null | 30 | 1824 |
| 39 | `ddqn_helps_at_early_bursts__pixel_envs` | SCOPE_VACATED | power_insufficient | - | 0 | 1520 |
| 40 | `ddqn_increases_argmax_entropy__fourrooms_rs_0p1` | REFUTED | post-rebuild: g=−2.98 mean_diff=−0.232 p=2.6e-12 n=30 — SIGN_FLIP. DDQN DECREASES argmaxH (sharpens policy) in the rescue regime. "Maintains exploration when Q flat" reading refuted; under-learning rescue mechanism is "policy sharpens after rescue" | a_gt_b | 30 | 1824 |
| 41 | `ddqn_null_under_monte_carlo__fourrooms_n10` | SCOPE_VACATED | (2026-05-11 screening: verdict-logic bug fixed — NaN mean_diff was returning NO_EFFECT instead of POW_INSUF. Now POW_INSUF when arm_mean_diff is NaN. Stays SCOPE_VACATED until n=10 post-fix corpus exists) | - | 0 | 1824 |
| 42 | `ddqn_rescues_underlearning_vanilla__fourrooms_rs_0p1` | SURVIVED | held (2026-05-11 migrated `paired_g` → `arm_mean_diff` per RL seed-pairing critique. md=+0.638, 95% CI=[+0.592, +0.684], threshold +0.35 fully cleared. Underlearning rescue robust; SE only moved 0.022→0.024 confirming seed-pairing was extracting nothing) | a_gt_b | 60 | 1824 |
| 43 | `staleness_amplifies_ddqn_outcome__sparse_goal_polyak` | SCOPE_VACATED | power_insufficient | a_lt_b | 0 | 1520 |
| 44 | `staleness_does_not_amplify_ddqn_outcome__survival_polyak` | SCOPE_VACATED | power_insufficient | null | 0 | 1520 |
| 45 | `target_staleness_late_mediates_outcome__breakout_sync100` | SCOPE_VACATED | power_insufficient | a_gt_b | 0 | 1520 |
| 46 | `target_staleness_late_mediates_outcome__fourrooms` | SCOPE_VACATED | power_insufficient | a_gt_b | 0 | 1520 |
| 47 | `target_staleness_late_mediates_outcome__minatar_intermediate_sync` | SCOPE_VACATED | power_insufficient | a_gt_b | 0 | 1520 |
| 48 | `bootstrap_fraction_drives_g_link__net_of_dormancy` | DEAD | (originally ERRORED — meta_regression empty after scope filter; cut from BRIDGES in earlier audit. Per Step-2 cut notes: CLAIM 4 banner deleted. Tag corrected 2026-05-11 — bridge isn't in BRIDGES tuple so "ERRORED" is misleading; DEAD is the right class) | - | -1 | -1 |
| 49 | `alpha_outcome_slope_per_env__second_layer` | POWER_INSUFFICIENT | (separate module `second_layer_theorem.py`; 2026-05-11 audit migrated `cross_config_paired_slope` → JCI `stratified_spearman`; pre-registered cohort ≥4 envs but only Breakout+SpaceInvaders ran the α-sweep — POW_INSUF at n_strata=2 < min_strata=4. Pre-fix ρ_pooled=+0.166 on n_strata=2 — sign-aware per-env prediction unconfirmable by pooled primitive; returns POW_INSUF rather than NO_EFFECT per reviewer pushback) | a_gt_b | 360 | 1824 |
| 50 | `alpha_jens_slope_per_env__second_layer` | POWER_INSUFFICIENT | (same module + migration as #49; ρ_pooled=−0.880 on n_strata=2 would be HELD at min_strata=2, but `min_strata=4` enforces pre-registered cohort. Hasselt's bias-correction direction strongly confirmed on the 2 envs that were swept; awaiting MetaMaze/MountainCar/Asterix/Freeway sweep to complete) | a_lt_b | 360 | 1824 |
| 51 | `intrinsic_penalty_scales_with_bootstrap_gap__second_layer` | POWER_INSUFFICIENT | (same module; partial Spearman over 4 envs but Breakout's marginal ρ=+0.61 *** is OPPOSITE direction to predicted negative; awaiting full cohort to discriminate) | a_lt_b | 240 | 1824 |

## Diagnosis notes

### CLAIM 2 — `ddqn_refuted_when_dormancy_fires` — STALE (body + scope, not theory)

Verdict was INVARIANT_VIOLATION on 553 cells (helped_fraction=0.52,
g=+0.09, p=0.32, n_pairs=123). Diagnosis:

1. **Body uses the wrong null-confirmation shape.** Bridge body
   says HELD when `helped_fraction ≤ 0.15 AND |g| ≤ 0.20` — that
   requires DDQN to *systematically lose* on dormant cells. The
   correct null-confirmation predicate is `helped_fraction ≈ 0.50
   AND g ≈ 0` (no systematic effect). Observed values are exactly
   this; the threshold mis-encodes the null shape. This is the
   substrate body-flip workaround for the framework's missing
   `'null'` branch in `verdict_from_paired_stats` (see
   `BRIDGE_PREDICTION_DESIGN.md` §1, §9).

2. **Scope is too loose and overlaps CLAIM 26b's G1.** Per-cell
   predicate `jensen_dormancy_gap >= 1e-9` admits 133 cells that
   are *also* in CLAIM 26b's G1-active scope (mostly
   SpaceInvaders-MinAtar at env_vanilla_mean_jens=2.3). In that
   intersection cohort DDQN genuinely helps (mean Δ = +0.35,
   helped_fraction = 0.66), driving the body's helped_fraction
   above its threshold. The 404 cells in env-G1-inactive envs
   (Freeway, Breakout, Asterix, Acrobot, FourRooms) show
   helped_fraction ≈ 0.50 and Δ ≈ 0 — the genuine null cohort.

**Fix path (lands in step 6 — framework completion + substrate
migration).** Refactor as the env-level inverse of CLAIM 26b:

- Scope: `partition_aggregate('jensen_gap', by=['env_name'],
  op='mean') < 2.0` (G1-inactive envs only — disjoint with
  CLAIM 26b).
- Prediction: `Prediction(direction='null',
  reason='DDQN's bias-correction mechanism is dormant when
  env-level vanilla bias is small (CLAIM 26b G1 complement);
  no systematic benefit expected.')`.
- Body: framework-native `verdict_from_paired_stats(..., 
  predicted_direction='null')` — collapses the bespoke
  threshold body to a one-line framework call.

CLAIM 2 and CLAIM 26b then form a coherent necessary-scope pair:
26b says "DDQN helps on G1-active envs"; 2 (refactored) says
"DDQN does not help on G1-inactive envs." Theory survives;
implementation rewritten.

### Bridge 13 — `cross_config_staleness_slope_negative__survive` — POWER_COLLAPSED (heterogeneity-confound)

Verdict: NO_EFFECT at ρ=-0.43, p=0.40, n_configs=6 (predicted
a_lt_b; pre-fix HELD at ρ=-0.9, n=5).

Per-(env, sync) Δs (sorted by Δ_staleness ascending) in the
bridge's scope:

| sync | env | n_pairs | Δ_outcome | Δ_staleness |
|---|---|---|---|---|
| 100 | CartPole-v1 | 20 | 0.00 | 0.00030 |
| 500 | SpaceInvaders-MinAtar | 11 | **+1.30** | 0.00036 |
| 500 | Breakout-MinAtar | 17 | −0.55 | 0.00072 |
| 500 | Asterix-MinAtar | 30 | +0.12 | 0.00072 |
| 1500 | Asterix-MinAtar | 30 | −0.14 | 0.00149 |
| 3000 | Asterix-MinAtar | 30 | −0.33 | 0.00290 |

**Within Asterix sync sweep alone** (3 points: 500→1500→3000):
Δ_out goes +0.12 → −0.14 → −0.33 — clean monotonic decrease,
supports the negative-slope claim. **The outlier** is
SpaceInvaders sync=500 with Δ_out=+1.30 — DDQN helps SI a lot at
this sync, despite being mid-Δ_staleness. Breakout sync=500 in
the opposite direction (Δ_out=−0.55) at the same nominal sync.
The cross-env pool mixes envs with very different DDQN-response
profiles at the same staleness level; the env-level
heterogeneity dominates the within-sync-sweep signal.

Note: this is the SAME env-specific behavior (DDQN helps SI
under marginal G1 conditions) that drove CLAIM 2's
INVARIANT_VIOLATION — SI is producing the anomaly in both
diagnoses. Coherent.

**Fix paths (not blocking the cut step):**

- Restrict the bridge to a single env's sync sweep (Asterix
  alone HELDs).
- Refactor as a per-env-slope meta-regression: per-env slope
  of Δ_out on Δ_stale via `meta_regress_panel`, then pool
  across envs. Captures within-env signal without cross-env
  contamination.
- Add more configs (3+ per env on multiple envs) and let the
  panel structure pool properly.

Tag: POWER_COLLAPSED. Keep the bridge; reformulate per the
above when seed-pairing retirement lands in step 6/7.

### Bridge 14 — `cross_config_staleness_slope_positive__reach_polyak` — DEAD (sign-flipped)

Verdict: NO_EFFECT at ρ=−0.10, p=0.87, n_configs=5 (predicted
a_gt_b; pre-fix HELD at ρ=+1.0 on n=3 from polyak τ sweep).

Pre-fix n=3 ρ=+1.0 was a small-sample fluke; n=5 reveals no
positive slope and a slight wrong-direction tendency. The
"REACH-polarity polyak staleness amplifies DDQN benefit" half
of CLAIM 21 doesn't replicate at moderate n. Sister bridge 13
(SURVIVE half, POWER_COLLAPSED) survives as a one-direction
claim within Asterix.

Tag: DEAD. Cut in step 2. Memory entry
`findings_cross_config_staleness_polarity.md` should be
updated to note REACH-polyak half retracted; SURVIVE half
restricted to Asterix.

### Bridge 15 — `link_r_predictable_from_polarity__soft_tautology` — POWER_COLLAPSED

Verdict: NO_EFFECT at β=+0.335, CI [−0.30, +0.97], R²=0.18,
n_strata=9 (predicted a_gt_b; pre-fix HELD at β=+0.616,
R²=0.82).

Sign correct (+), magnitude approximately halved, CI now
overlaps zero. R² collapsed from 0.82 to 0.18 — the
polarity-as-link-shape signal weakened with the post-fix
corpus. The structural-tautology argument in the docstring
(`HARD r(eff_h, O) ≡ SOFT r(Δ_eff_h, Δ_O)` paired to 3
decimals) may still hold algebraically; the cross-env
regression no longer carries it cleanly.

Tag: POWER_COLLAPSED. Two possible fixes:

- Add more envs (n=9 → ~15+) to tighten the CI.
- Recast as a measurement-identity bridge (per Shape A example
  in the manifest): test `r(eff_h, O) ≈ r(Δ_eff_h, Δ_O)` within
  each env to ε, where ε is set by sampling theory. This
  tests the structural tautology directly and doesn't depend
  on cross-env aggregation.

### Bridge 16 — `link_slope_predicted_by_g1__cross_env` — DEAD (already-superseded)

Verdict: NO_EFFECT at β=−0.003, CI [−0.021, +0.015], R²=0.016,
n_strata=11 (predicted a_lt_b at slope_threshold=−0.04;
coefficient is 80× below threshold).

Comment block at `ddqn_universe.py:3420-3434` explicitly states
CLAIM 26b replaces CLAIM 26's slope-predictor regression
("structurally underdetermined: per-env link slope is pinned
at −1 by the asymptote claim, so cross-env variance in |slope|
is dominated by saturation / sub-asymptote artifacts, not by
v_jens"). The empirical refutation here confirms the prior
prose deprecation. CLAIM 26b (bridge #5, SURVIVED at d=+0.46,
p=0.005) is the principled outcome-level replacement.

Tag: DEAD. Cut cleanly in step 2 — already superseded in the
file's own prose.

### Bridge 48 — `bootstrap_fraction_drives_g_link__net_of_dormancy` — ERRORED

Error: `ValueError: meta_regression: observations is empty`.
Bridge body invokes `meta_regression_paired_g` on the per-env
panel; after the scope filter + arm pairing, zero rows survive.

Most likely cause: the bridge's scope predicate (CLAIM 4 link-
side residual) targets cells in the cross-env panel that the
post-fix snapshot doesn't carry. Comment block at line 4168 +
ddqn_universe.py's "DELETED" treatment of the bf-cross-env
claim (CLAIM 16 deletion memo, lines 4168-4193) already notes
the bf-cross-env claim is dead: *"bf clusters at [0.98, 1.00]
across true chain MDPs. No meaningful cross-env variance to
test against."*

The errored bridge is the bf→g_link residual claim that
preceded the formal CLAIM 16 deletion. Same fate.

Tag: DEAD. Cut in step 2. Coherent with the CLAIM 16 deletion
memo's bf-cluster bullet — both encode the bf-as-cross-env-
predictor claim retraction. The deletion memo's bf-cluster
bullet is itself a Shape A bridge candidate (per manifest
"Examples that incarnate") that would replace this errored
bridge with a real invariant: `partition_aggregate(
'bootstrap_fraction', by='env_name', op='std') < 0.01`.

## SCOPE_VACATED policy (23 bridges)

Classified by why the scope filter matches zero post-fix cells:

| Category | n | Bridges | Read |
|---|---|---|---|
| Pearl rung-2 designed sweeps | 4 | adaptive_dqn × 2 (CLAIM 2 corroboration), gamma_sweep × 2 (CLAIM 5) | wait for re-run if claim is load-bearing |
| Reward-scale intervention | 6 | underlearning_rescue × 6 (CLAIM 7 a/b/c/d/e/f) | cut OR re-run rs sweep |
| Reward-shape / action-stochasticity probes | 4 | noisy / sparsified / densified × 4 (CLAIM 7 g–j) | cut OR re-run probes |
| SI 1M crossover | 1 | `ddqn_curve_crosses_vanilla_late__spaceinvaders` (CLAIM 8) | wait for SI 1M post-fix corpus |
| n-step variants | 1 | `ddqn_null_under_monte_carlo__fourrooms_n10` (CLAIM 9) | wait for n-step post-fix corpus |
| Polyak τ sweep | 2 | staleness amplify × 2 (CLAIM 15) | wait for Polyak post-fix |
| Per-corpus mediator | 3 | staleness mediator × 3 (CLAIM 13) | wait for capacity_sweep / minatar_1M / asterix_intermediate_sync post-fix |
| Per-env per-burst | 2 | acrobot_per_burst (CLAIM 10), pixel_envs (TIER A2) | wait for substrate-version-matched corpus |

**Resolution:**

- **Re-run (now, CPU):** rs-intervention rescue family (6 bridges:
  rescue_underlearning, dominates_response_curve, does_not_rescue
  ×2, increases_argmax_entropy, entropy_matches_vanilla). New
  config `experiments/configs/reward_scale_sweep_postfix.yaml`
  authored; 5 (env, rs) × 2 arms × 30 seeds × 200K steps on CPU
  in progress at `experiments/data/reward_scale_sweep_postfix/`.
  After completion: ingest into the ddqn_universe cache, re-run
  the 6 bridges, retag.
- **Queue for re-run (10):** Pearl rung-2 corroborations
  (adaptive_dqn × 2, gamma_sweep × 2, polyak τ × 2) + per-corpus
  mediator (staleness × 3) + SI 1M crossover (1). Test
  causal-chain claims the bridge graph relies on; awaiting
  designed-sweep re-runs under post-fix substrate.
- **Reward-shape / action-stochasticity probes (4)** —
  pending user decision. These are CLAIM 7 g–j auxiliary
  mechanism probes (noisy/sparsified/densified argmax-
  concentration); the substantive CLAIM 7 finding is now in
  `findings_underlearning_rescue.md` + subsumed by CLAIM 26b's
  three-gate scope. Cut OR re-run as designed mechanism probes.
- **Recast (3):** n-step n=10 (CLAIM 9), acrobot per-burst
  (CLAIM 10), pixel-envs per-burst (TIER A2). Scope filter
  needs updating to match post-fix corpora; author-side fixup,
  not re-run.

## Updated tally after retags + recommendations

| Tag | Count after audit | Net change vs first-pass |
|---|---|---|
| SURVIVED | 11 | — |
| STALE | 1 | +1 (CLAIM 2) |
| POWER_COLLAPSED | 2 | +2 (bridges 13, 15) |
| DEAD | 2 | +2 (bridges 14, 16) — pending step-2 cut |
| POWER_INSUFFICIENT | 8 | — |
| SCOPE_VACATED | 23 | — (10 are DEAD candidates pending user decision) |
| ERRORED | 1 | tagged DEAD pending step-2 cut |
| **Total** | **48** | |

**Step 2 cut EXECUTED:**
- ✓ Bridge 14 `cross_config_staleness_slope_positive__reach_polyak`
  removed.
- ✓ Bridge 16 `link_slope_predicted_by_g1__cross_env` removed
  (CLAIM 26 banner deleted; CLAIM 26b retained as the
  successor).
- ✓ Bridge 48 `bootstrap_fraction_drives_g_link__net_of_dormancy`
  removed (CLAIM 4 banner deleted; CLAIM 16 deletion memo
  shortened; top-docstring section 4 reframed as "formerly").

Post-cut state: 48 BRIDGES → 45. `__all__` 41 → 38. File
imports cleanly; pyright shows no new errors (all warnings are
pre-existing polars-type lints).

**Pending:**
- 4 SCOPE_VACATED → DEAD (CLAIM 7 g–j argmax probes) —
  user decision: cut OR re-run as designed probes.
- 6 SCOPE_VACATED rs-rescue bridges in flight via
  `reward_scale_sweep_postfix` sweep; retag once data lands.

Remaining substantive work is in step 6 (framework completion
+ substrate migration) per `BRIDGE_PREDICTION_DESIGN.md`.

## Variance-reduction reading (step 5 — task #5)

For each seed-paired in-scope bridge, identify whether
seed-pairing is load-bearing (fixture consumes per-pair Δs) or
vestigial (fixture is per-stratum; `pair_by=('seed',)` is just
the Bridge contract default).

### Vestigial seed-pairing (5 bridges) — fixture already stratifies

These fixtures aggregate to per-stratum statistics BEFORE
computing the test stat; per-pair Δ structure is unused. The
`pair_by=('seed',)` argument is vestigial inherited Bridge
metadata; dropping it leaves the verdict unchanged. Safe to
remove `pair_by` from these bridges' decorators.

| Bridge | Fixture | Verdict |
|---|---|---|
| `argmax_entropy_predicts_link_power__survive_envs` | meta_regression_per_burst | HELD |
| `cross_config_staleness_slope_negative__survive` | cross_config_paired_slope | NO_EFFECT (POWER_COLLAPSED) |
| `ddqn_helps_under_three_gate_scope__cross_env` | stratified_arm_diff_pooled | HELD |
| `effh_predicts_link_power__reach_envs` | meta_regression_per_burst | POWER_INSUFFICIENT |
| `link_r_predictable_from_polarity__soft_tautology` | paired_link_per_env | NO_EFFECT (POWER_COLLAPSED) |

### Load-bearing seed-pairing (17 bridges) — fixture consumes per-pair Δs

These fixtures (`paired_g`, `paired_link_per_burst`,
`proportion_mediated`, `partial_spearman_paired_delta`,
`paired_delta_link_dowhy`, `link_attenuation_dowhy`,
`mundlak_paired_g_per_burst`) operate on the (vanilla_seed_i,
ddqn_seed_i) per-pair Δ structure. Within-seed variance reduction
materially tightens the SE because shared init / batch order /
exploration noise cancels at the per-pair level. Stratified
pooling discards this precision.

**Empirical illustration** (`ddqn_helps_at_full_bootstrap__fourrooms_n1`,
single-env paired_g case):

| Form | n | Effect | SE | CI |
|---|---|---|---|---|
| paired (within-seed) | 30 pairs | g=+0.28 | 0.0024 | tight, p≈0.12 |
| independent-samples (stratified-eq) | 30+30 | d=+0.30 | 0.26 | [−0.21, +0.81] |

Point estimates align (g≈d, expected since the effect is real).
**Within-seed SE is ~100× tighter.** Converting to independent-
samples would push the bridge from HELD (effect-size threshold
g≥0.05 satisfied with tight CI) to POWER_COLLAPSED (CI overlaps
threshold).

For paired_g bridges: keep seed-pairing.
For non-paired_g per-pair-Δ fixtures: same logic applies
structurally — the fixture's internal computation depends on
the per-pair Δ structure.

| Bridge | Fixture | Verdict |
|---|---|---|
| `algorithmic_activation_rate_mediates_link__bounded_q` | proportion_mediated | POWER_INSUFFICIENT |
| `argmax_entropy_shadowed_by_jens` | partial_spearman_paired_delta | HELD |
| `chain_amplifier_link_active_in_bounded_q` | paired_link_per_burst | HELD |
| `ddqn_helps_at_full_bootstrap__fourrooms_n1` | paired_g | HELD |
| `ddqn_refuted_when_dormancy_fires` | paired_g | STALE |
| `eff_h_mediates_g_link__goal_envs` | proportion_mediated | POWER_INSUFFICIENT |
| `eff_h_mediates_g_link__survival_envs` | proportion_mediated | HELD |
| `extreme_q_divergence_attenuates_link__binary` | link_attenuation_dowhy | POWER_INSUFFICIENT |
| `extreme_q_divergence_attenuates_link__placebo_refuted` | link_attenuation_dowhy | POWER_INSUFFICIENT |
| `extreme_q_divergence_attenuates_link__rcc_robust` | link_attenuation_dowhy | POWER_INSUFFICIENT |
| `fourrooms_action_dim_link_active__inflated` | paired_link_per_burst | HELD |
| `mc_variance_attenuates_g_link__between_env` | mundlak_paired_g_per_burst | POWER_INSUFFICIENT |
| `metamaze_link_steeper_at_high_gamma` | paired_link_per_burst | HELD |
| `q_divergence_shadowed_by_jens` | partial_spearman_paired_delta | POWER_INSUFFICIENT |
| `reach_link_backdoor_ate_negative` | paired_delta_link_dowhy | HELD |
| `reach_link_placebo_refuted` | paired_delta_link_dowhy | HELD |
| `reach_link_rcc_robust` | paired_delta_link_dowhy | HELD |

### Net read

The manifest's seed-pairing retirement is **per-bridge**, and
**17 of 22 in-scope bridges genuinely need it**. The 5
vestigial cases (already-stratified fixtures) can drop
`pair_by` from their decorator; the 17 load-bearing cases keep
seed-pairing as the within-stratum variance-reduction
mechanism that their fixtures consume.

This matches the manifest's own framing in `BRIDGE_AUDIT.md`:
"Seed-paired analyses are retiring *per-bridge* — within-seed
variance reduction is load-bearing for some contrasts (e.g.
CLAIM 9's n-step falsification). Convert with a variance-
reduction reading, not a global sweep."

**No global mechanical conversion is appropriate.** The
"retirement" applies to the 5 vestigial cases (cleanup) and to
future bridges where the substrate-author chooses a per-stratum
fixture by default.

### What this means for step 6 (framework completion)

`BRIDGE_PREDICTION_DESIGN.md` should note: the framework
should NOT silently convert `paired_g` to
`stratified_arm_diff_pooled` even though it's the "principled
cross-env aggregator" in the manifest's prose — the conversion
loses ~100× SE precision on within-seed contrasts. Both
primitives stay; the substrate author picks per claim.

Optional v1 follow-up: tighten the Bridge contract to make
`pair_by` optional. Today's default of `('seed',)` is the right
default for load-bearing per-pair-Δ fixtures but vestigial for
per-stratum fixtures. A future signature-introspection check
(`if fixture in PER_STRATUM_FIXTURES and pair_by != (): warn`)
would catch the mismatch at admission time.

## Step 5 — CORRECTION on the vestigial classification

The "5 vestigial bridges" claim above was over-broad. Reading
the fixture sources end-to-end:

- **Genuinely vestigial (1):** `stratified_arm_diff_pooled`
  does NOT consume `pair_by` (its docstring even contrasts with
  paired_g on this point). → `ddqn_helps_under_three_gate_scope__cross_env`.
  Explicit `pair_by=('seed',)` kwarg dropped from this bridge;
  framework default still applies but the primitive ignores it.
- **Half-stratified (4):** `cross_config_paired_slope`,
  `paired_link_per_env`, and `paired_g_per_burst` /
  `meta_regression_per_burst` DO use `pair_by` for within-stratum
  pairing BEFORE across-stratum pooling. These bridges are
  better than fully-paired (the across-stratum step is correct)
  but the within-stratum step still relies on seed-pairing. The
  RL methodology critique below applies to them too — they need
  the same migration to fully-stratified analogs per
  `BRIDGE_PREDICTION_DESIGN.md §11`.

Affected bridges (half-stratified, not vestigial):
- `cross_config_staleness_slope_negative__survive`
- `link_r_predictable_from_polarity__soft_tautology`
- `argmax_entropy_predicts_link_power__survive_envs`
- `effh_predicts_link_power__reach_envs`

## Step 5 — REVISED (RL methodology critique)

The earlier "vestigial vs load-bearing" classification rationalized
the existing fixture set rather than questioning whether the
fixture set was right. Per the user's methodological judgment
(2026-05-11):

- **Seed-pairing assumes within-pair correlation reflects a
  shared confounder that cancels in the Δ.** A/B-testing's
  "same unit, two treatments" shape.
- **RL violates this assumption.** Same seed ⇏ same trajectory.
  From step ~1, DDQN's `double_greedify` changes the target,
  which changes the loss, which changes the next batch sample,
  which changes the next trajectory. After thousands of steps
  the two arms explore different state space. Shared seed
  cancels init weights and PRNG state — not the bulk of the
  variance.
- **The "100× SE tightening" on FourRooms n_step=1 reflects
  `cov(arm_t, arm_b) ≈ +0.9999`** — the two trajectories
  happened to converge to very similar solutions on a
  near-deterministic env. That's a property of the training
  dynamics, not a statistical property of the comparison.
- **The inferential target is wrong.** Paired-t answers
  "for the same init, does DDQN beat vanilla?" The
  practically-interesting question is "across a population of
  plausible inits, does DDQN beat vanilla on average?" The
  latter is the independent-samples / stratified-pooled form.
  Seed-paired makes within-init effects look like population
  effects.

### Revised: all 22 in-scope bridges need methodology refactor

Both "vestigial" AND "load-bearing" classes move toward
stratified forms — for different reasons. The 5 vestigial
cases were already stratified at the fixture level; the 17
load-bearing cases need stratified fixture analogs.

### Per-fixture migration table

| Current fixture | Stratified analog | Exists? |
|---|---|---|
| `paired_g` | `stratified_arm_diff_pooled` | ✓ (5 bridges' target) |
| `paired_link_per_burst` | `stratified_link_per_burst` (TBD) | ✗ — author |
| `proportion_mediated` | `stratified_proportion_mediated` (TBD) | ✗ — author |
| `partial_spearman_paired_delta` | per-stratum partial-Spearman + Fisher-z pool (TBD) | ✗ — author |
| `paired_delta_link_dowhy` | DoWhy stratified backdoor (env as confounder, not pairing axis) | partial (DoWhy supports it) |
| `link_attenuation_dowhy` | Same | partial |
| `mundlak_paired_g_per_burst` | Mundlak already does between/within decomposition; consume stratum aggregates not per-pair Δs | refactor existing |
| `meta_regression_per_burst` | already per-stratum (env, burst) | ✓ |
| `paired_link_per_env` | already per-stratum (env) | ✓ |
| `cross_config_paired_slope` | already per-stratum (config) | ✓ |
| `stratified_arm_diff_pooled` | already stratified | ✓ |

5 fixtures need new framework analogs; 2 need internal refactors;
4 are already stratified.

### What this means for the audit

- **The 11 SURVIVED verdicts are conditional on the current
  methodology.** Under stratified-pooled methodology, many will
  shift toward POWER_COLLAPSED (wider CI; threshold-based bodies
  may survive, but p-value-gated bodies will lose power).
- **Bridge 4 (`ddqn_helps_at_full_bootstrap__fourrooms_n1`)** is
  the canonical illustration: paired g=+0.28 at SE=0.0024 →
  independent d=+0.30 at SE=0.26. Same point estimate, 100×
  wider CI. Threshold body (`g ≥ 0.05`) preserves HELD; power-
  aware body would flip to POWER_INSUFFICIENT.
- **Bridge 12 (`ddqn_refuted_when_dormancy_fires`)** already
  diagnosed as STALE (body + scope); the seed-pairing critique
  reinforces the recommendation to refactor with a stratified
  fixture under the framework completion.

### Sequencing

1. **The 5 vestigial pair_by removals** can land as a small
   cleanup PR now (no analysis change).
2. **The 17 load-bearing migrations** need framework analogs
   first. Lands in step 6 per `BRIDGE_PREDICTION_DESIGN.md` §11
   (the new seed-pairing retirement section).
3. **Re-running the audit under stratified methodology** is a
   step-6 deliverable that may shift the SURVIVED → POWER_
   COLLAPSED count substantially. The current audit's
   "SURVIVED" tag should be read as "HELD under current
   (RL-incorrect) methodology" until that re-run lands.

The deeper meta-lesson: the seed-paired methodology was a
substrate convention inherited from supervised-learning A/B
patterns; it wasn't questioned at framework-author time. M2's
"complete the existing primitive thoroughly" is most useful
when paired with: question whether the primitive's shape was
right BEFORE optimizing for its preservation.
