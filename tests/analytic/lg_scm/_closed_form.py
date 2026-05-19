"""Shared closed-form helpers for LG-SCM analytic tests.

The same population-variance formula appears across multiple
files; centralizing it here keeps the algebra in ONE place so a
substrate-equation change updates one definition rather than
three. Each call site passes its module-level constants
explicitly — the helper is substrate-aware but parameter-free.
"""
from __future__ import annotations

import math


def y_mean_arm_variance(
    *,
    beta_xz: float,
    beta_zy: float,
    sigma_x: float,
    sigma_z: float,
    sigma_y: float,
    n_steps: int,
) -> float:
    """Population Var[y_mean_per_seed | arm] under the LG-SCM
    chain `X → Z → Y`. Decomposes into three structural-noise
    propagation terms:

        Var[y_mean] = (β_xz · β_zy)² · σ_x²/n_steps
                    + (β_zy · σ_z)²/n_steps
                    + σ_y²/n_steps

    Derivation: per step `Y_step = β_zy · (β_xz · X + σ_z · ε_z)
    + σ_y · ε_y`. Averaging over `n_steps` independent steps
    gives Var[Y_avg | one seed] = sum of each term's variance /
    n_steps. X_avg = (μ_x + σ_x·ε_x_avg) has Var σ_x²/n_steps.
    """
    return (
        (beta_xz * beta_zy) ** 2 * sigma_x ** 2 / n_steps
        + (beta_zy * sigma_z) ** 2 / n_steps
        + sigma_y ** 2 / n_steps
    )


def y_mean_arm_sd(
    *,
    beta_xz: float,
    beta_zy: float,
    sigma_x: float,
    sigma_z: float,
    sigma_y: float,
    n_steps: int,
) -> float:
    """sqrt of `y_mean_arm_variance`."""
    return math.sqrt(y_mean_arm_variance(
        beta_xz=beta_xz, beta_zy=beta_zy,
        sigma_x=sigma_x, sigma_z=sigma_z, sigma_y=sigma_y,
        n_steps=n_steps,
    ))
