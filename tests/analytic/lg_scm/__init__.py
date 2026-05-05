"""Linear-Gaussian SCM substrate.

Two-node chain with optional confounder:

    X ~ N(mu_x, sigma_x^2)
    Z = beta_xz * X + epsilon_z,    epsilon_z ~ N(0, sigma_z^2)
    Y = beta_zy * Z + epsilon_y,    epsilon_y ~ N(0, sigma_y^2)

Closed-form moments (no confounder):

    E[Y]   = beta_zy * beta_xz * mu_x
    Var[Y] = beta_zy^2 * (beta_xz^2 * sigma_x^2 + sigma_z^2) + sigma_y^2

Paired two-arm intervention. Arms differ in `beta_xz`. Within a
shared `seed`, both arms draw the same noise streams (epsilon_x,
epsilon_z, epsilon_y), so per-seed:

    Y_arm(t) = beta_zy * (beta_xz_arm * X(t) + epsilon_z(t)) + epsilon_y(t)
    Delta_Y(t)
        = (beta_xz_t - beta_xz_b) * beta_zy * X(t)
        = Delta_beta * beta_zy * X(t)

The epsilon noise cancels in the paired contrast, so per-cell
mean-Δ on `y_mean = mean_t Y(t)` has closed form:

    E[Delta_y_mean]  = Delta_beta * beta_zy * mu_x
    Var[Delta_y_mean | n_steps] = (Delta_beta * beta_zy)^2 * sigma_x^2 / n_steps

This makes paired-g's `mean_diff` an analytically tractable target.
"""
