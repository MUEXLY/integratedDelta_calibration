import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.stats import multivariate_normal, norm
from scipy.stats import invgamma
import json
import pandas as pd


def holdout_sensitivity_metric(
    y_true,
    y_pred,
    holdout_fractions=(0.1, 0.2, 0.3),
    n_repeats=20,
    random_state=0,
    x_domain=None,
    y_pred_std=None,
    parameter_corrections=None,
):
    """Estimate predictive sensitivity to withholding observations.

    Predictions are evaluated on randomly withheld observations for each
    requested fraction.  Application-domain minimum and maximum points are
    protected from holdout selection. ``sensitivity`` is the change in
    normalized RMSE relative to the score on all observations; positive values
    indicate degradation when observations are withheld.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size < 2:
        raise ValueError("At least two observations are required")
    if n_repeats < 1:
        raise ValueError("n_repeats must be positive")

    if x_domain is None:
        x_domain = np.arange(y_true.size, dtype=float)
    x_domain = np.asarray(x_domain, dtype=float)
    if x_domain.ndim == 1:
        x_domain = x_domain[:, None]
    if x_domain.shape[0] != y_true.size:
        raise ValueError("x_domain must have one row per observation")
    if y_pred_std is not None:
        y_pred_std = np.asarray(y_pred_std, dtype=float).ravel()
        if y_pred_std.shape != y_true.shape:
            raise ValueError("y_pred_std must have one value per observation")
    if parameter_corrections is not None:
        parameter_corrections = np.asarray(parameter_corrections, dtype=float)
        if parameter_corrections.shape[0] != y_true.size:
            raise ValueError(
                "parameter_corrections must have one row per observation"
            )

    domain_min = np.min(x_domain, axis=0)
    domain_max = np.max(x_domain, axis=0)
    endpoint_mask = np.any(
        np.isclose(x_domain, domain_min) | np.isclose(x_domain, domain_max),
        axis=1,
    )
    eligible_indices = np.flatnonzero(~endpoint_mask)
    if eligible_indices.size == 0:
        raise ValueError("No interior application-domain points are available")

    fractions = tuple(float(fraction) for fraction in holdout_fractions)
    if any(fraction <= 0 or fraction >= 1 for fraction in fractions):
        raise ValueError("holdout fractions must be between zero and one")

    scale = np.ptp(y_true)
    scale = scale if scale > 0 else 1.0
    full_nrmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / scale)
    rng = np.random.default_rng(random_state)
    results = {}
    cases = []

    for fraction in fractions:
        n_holdout = max(1, int(round(fraction * y_true.size)))
        if n_holdout > eligible_indices.size:
            raise ValueError(
                f"Holdout size ({n_holdout}) exceeds available interior points "
                f"({eligible_indices.size})"
            )
        scores = []
        for _ in range(n_repeats):
            holdout = rng.choice(eligible_indices, size=n_holdout, replace=False)
            score = np.sqrt(np.mean((y_true[holdout] - y_pred[holdout]) ** 2))
            scores.append(float(score / scale))
            case = {
                "fraction": fraction,
                "indices": holdout.tolist(),
                "x": x_domain[holdout].tolist(),
                "y_true": y_true[holdout].tolist(),
                "y_pred": y_pred[holdout].tolist(),
            }
            if y_pred_std is not None:
                case["y_pred_std"] = y_pred_std[holdout].tolist()
            if parameter_corrections is not None:
                case["parameter_corrections"] = (
                    parameter_corrections[holdout].tolist()
                )
            cases.append(case)
        mean_score = float(np.mean(scores))
        results[str(fraction)] = {
            "holdout_count": n_holdout,
            "nrmse": mean_score,
            "nrmse_std": float(np.std(scores)),
            "sensitivity": mean_score - full_nrmse,
        }

    return {
        "full_nrmse": full_nrmse,
        "holdout_results": results,
        "n_repeats": int(n_repeats),
        "scale": float(scale),
        "protected_indices": np.flatnonzero(endpoint_mask).tolist(),
        "eligible_indices": eligible_indices.tolist(),
        "cases": cases,
    }


def run_delta_sampler(
    x_obs,
    y_obs,
    theta,
    gp_eta,
    dtheta,
    n_mcmc,
    burn_in,
    mh_scale_delta,
    allow_singular_covariance=False,
):
    """Run the discrepancy sampler for one calibration training subset."""
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float).ravel()
    delta = np.zeros((dtheta, len(x_obs)))
    ell_delta = 0.3 * np.ones(dtheta)
    var_delta = 0.05 * np.ones(dtheta)
    sigma2 = 0.02**2
    delta_chain = np.zeros((n_mcmc, dtheta, len(x_obs)))

    prior_ell = {"mu": -1.0, "sigma": 1.0}
    prior_var = {"mu": -2.0, "sigma": 1.0}
    mh_scales = {"log_ell_delta": 0.1, "log_var_delta": 0.1}
    accept_delta = np.zeros(dtheta)
    target_accept = 0.35

    for iteration in range(n_mcmc):
        for k in range(dtheta):
            delta, accepted = mh_update_delta_k(
                k, delta, theta, ell_delta[k], var_delta[k],
                x_obs, y_obs, gp_eta, sigma2, mh_scale_delta
            )
            accept_delta[k] += accepted

        if iteration < burn_in and (iteration + 1) % 50 == 0:
            mean_accept = np.mean(accept_delta / (iteration + 1))
            mh_scale_delta *= np.exp(0.05 * (mean_accept - target_accept))

        for k in range(dtheta):
            ell_delta[k], var_delta[k], _ = mh_update_delta_hyperparams(
                delta[k], ell_delta[k], var_delta[k], x_obs,
                prior_ell, prior_var, mh_scales,
                allow_singular_covariance=allow_singular_covariance,
            )
        sigma2 = gibbs_sigma2(
            y_obs, x_obs, theta, delta, gp_eta, 2.0, 0.5
        )
        delta_chain[iteration] = delta

    return delta_chain, ell_delta, var_delta


def holdout_sensitivity_refit(
    x_obs,
    y_obs,
    x_display,
    y_display,
    theta,
    gp_eta,
    dtheta,
    holdout_fractions=(0.1, 0.2, 0.3),
    n_repeats=20,
    n_mcmc=2000,
    burn_in=500,
    mh_scale_delta=0.5,
    allow_singular_covariance=False,
    random_state=0,
):
    """Run true refit-based holdout sensitivity analysis.

    Each trial fits only retained observations and evaluates predictions only
    at withheld interior points. The returned cases retain the same structure
    as the plotting diagnostics while recording the refit predictions.
    """
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float).ravel()
    x_display = np.asarray(x_display, dtype=float)
    y_display = np.asarray(y_display, dtype=float).ravel()
    if x_display.ndim == 1:
        x_display = x_display[:, None]
    if len(x_obs) != len(y_obs) or len(x_display) != len(y_display):
        raise ValueError("Holdout arrays must have matching lengths")
    if len(x_obs) != len(x_display):
        raise ValueError("x_obs and x_display must describe the same observations")

    endpoint_mask = np.any(
        np.isclose(x_display, np.min(x_display, axis=0))
        | np.isclose(x_display, np.max(x_display, axis=0)),
        axis=1,
    )
    eligible = np.flatnonzero(~endpoint_mask)
    if not len(eligible):
        raise ValueError("No interior application-domain points are available")

    scale = np.ptp(y_display)
    scale = scale if scale > 0 else 1.0
    rng = np.random.default_rng(random_state)
    cases = []
    results = {}

    for fraction in holdout_fractions:
        fraction = float(fraction)
        n_holdout = max(1, int(round(fraction * len(y_obs))))
        if n_holdout > len(eligible):
            raise ValueError("Requested holdout exceeds available interior points")
        scores = []
        for _ in range(n_repeats):
            holdout = rng.choice(eligible, size=n_holdout, replace=False)
            retained = np.setdiff1d(np.arange(len(y_obs)), holdout)
            chain, ell, var = run_delta_sampler(
                x_obs[retained], y_obs[retained], theta, gp_eta, dtheta,
                n_mcmc, burn_in, mh_scale_delta, allow_singular_covariance
            )
            start = min(max(burn_in, 0), n_mcmc - 1)
            posterior = chain[start:]
            predictions = []
            corrections = []
            weights_by_parameter = []
            for k in range(dtheta):
                K = rbf_kernel(
                    x_obs[retained], x_obs[retained],
                    ell=ell[k], var=var[k]
                ) + 1e-8 * np.eye(len(retained))
                K_cross = rbf_kernel(
                    x_obs[holdout], x_obs[retained],
                    ell=ell[k], var=var[k]
                )
                weights_by_parameter.append(np.linalg.solve(K, K_cross.T))
            for index, point in enumerate(x_obs[holdout]):
                sample_predictions = []
                sample_corrections = []
                for delta_sample in posterior[::max(1, len(posterior) // 200)]:
                    correction = np.array([
                        delta_sample[k] @ weights_by_parameter[k][:, index]
                        for k in range(dtheta)
                    ])
                    mean, _ = eta_predict(
                        point, theta + correction, gp_eta
                    )
                    sample_predictions.append(mean)
                    sample_corrections.append(correction)
                predictions.append(float(np.mean(sample_predictions)))
                corrections.append(np.mean(sample_corrections, axis=0).tolist())

            predictions = np.asarray(predictions)
            score = np.sqrt(np.mean(
                (y_display[holdout] - predictions) ** 2
            )) / scale
            scores.append(float(score))
            cases.append({
                "fraction": fraction,
                "indices": holdout.tolist(),
                "x": x_display[holdout].tolist(),
                "y_true": y_display[holdout].tolist(),
                "y_pred": predictions.tolist(),
                "parameter_corrections": corrections,
                "refit": True,
            })
        mean_score = float(np.mean(scores))
        results[str(fraction)] = {
            "holdout_count": n_holdout,
            "nrmse": mean_score,
            "nrmse_std": float(np.std(scores)),
            "sensitivity": None,
        }

    return {
        "full_nrmse": None,
        "holdout_results": results,
        "n_repeats": int(n_repeats),
        "scale": float(scale),
        "protected_indices": np.flatnonzero(endpoint_mask).tolist(),
        "eligible_indices": eligible.tolist(),
        "cases": cases,
        "refit": True,
    }

def rbf_kernel(X, Y, ell=1.0, var=1.0):
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    sqdist = np.sum((X[:, None, :] - Y[None, :, :])**2, axis=2)
    return var * np.exp(-0.5 * sqdist / ell**2)

def log_likelihood(y, x, theta, delta, eta, sigma2):
    """
    y_i ~ N( eta(x_i, theta + delta(x_i)), sigma2 )
    """
    y_pred = np.zeros_like(y)

    for i in range(len(x)):
        theta_star = theta + delta[:, i]
        y_pred[i] = eta(x[i], theta_star)

    resid = y - y_pred
    return -0.5 * (
        np.sum(resid**2) / sigma2
        + len(y) * np.log(2 * np.pi * sigma2)
    )

def log_prior_delta(delta_k, K_delta_inv):
    return -0.5 * delta_k.T @ K_delta_inv @ delta_k

def gp_log_density(delta_k, K):
    """
    Stable log N(0,K) evaluation.
    """
    No = len(delta_k)
    L = np.linalg.cholesky(K)

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, delta_k))

    logdet = 2*np.sum(np.log(np.diag(L)))

    return -0.5 * (delta_k @ alpha + logdet + No*np.log(2*np.pi))


def log_prior_hyperparams(ell, var, prior_ell, prior_var):
    """
    Log-prior for kernel hyperparameters (ell, var).

    prior_ell = {"mu": ..., "sigma": ...} on log(ell)
    prior_var = {"mu": ..., "sigma": ...} on log(var)
    """

    log_ell = np.log(ell)
    log_var = np.log(var)

    lp_ell = norm.logpdf(
        log_ell,
        loc=prior_ell["mu"],
        scale=prior_ell["sigma"]
    )

    lp_var = norm.logpdf(
        log_var,
        loc=prior_var["mu"],
        scale=prior_var["sigma"]
    )

    return lp_ell + lp_var

def mh_update_delta_hyperparams(
    delta_k, ell, var, x_md,
    prior_ell, prior_var,
    mh_scales, allow_singular_covariance=False
):
    log_ell_prop = np.log(ell) + mh_scales["log_ell_delta"] * np.random.randn()
    log_var_prop = np.log(var) + mh_scales["log_var_delta"] * np.random.randn()

    ell_prop = np.exp(log_ell_prop)
    var_prop = np.exp(log_var_prop)

    K_curr = rbf_kernel(x_md, x_md, ell=ell, var=var)+ 1e-8*np.eye(len(x_md))
    K_prop = rbf_kernel(x_md, x_md, ell=ell_prop, var=var_prop) + 1e-8*np.eye(len(x_md))

    logp_curr = (
        multivariate_normal.logpdf(delta_k, mean=np.zeros(len(delta_k)), cov=K_curr, allow_singular=allow_singular_covariance)
        + log_prior_hyperparams(ell, var, prior_ell, prior_var)
    )

    logp_prop = (
        multivariate_normal.logpdf(delta_k, mean=np.zeros(len(delta_k)), cov=K_prop, allow_singular=allow_singular_covariance)
        + log_prior_hyperparams(ell_prop, var_prop, prior_ell, prior_var)
    )

    if np.log(np.random.rand()) < (logp_prop - logp_curr):
        return ell_prop, var_prop, True
    else:
        return ell, var, False
    
def gibbs_sigma2(y_obs, x_obs, theta, delta, gp_eta, a, b):

    resid = np.zeros_like(y_obs)

    for i in range(len(y_obs)):
        theta_star = theta + delta[:, i]
        m_i, _ = eta_predict(x_obs[i], theta_star, gp_eta)
        resid[i] = y_obs[i] - m_i

    a_post = a + len(y_obs)/2
    b_post = b + 0.5*np.sum(resid**2)

    return invgamma.rvs(a_post, scale=b_post)

def eta_predict(x, theta_star, gp_eta):

    x = np.atleast_1d(np.asarray(x))
    theta_star = np.atleast_1d(np.asarray(theta_star))

    z = np.hstack([x, theta_star]).reshape(1, -1)

    m, s2 = gp_eta.predict(z, return_std=True)

    return m[0], s2[0]**2


def log_likelihood_embedded(y_obs, x_obs, theta, delta, gp_eta, sigma2):
    """
    y_i ~ N( m_i , sigma2 + s_i^2 )
    where emulator provides (m_i, s_i^2)
    """
    N = len(y_obs)
    loglike = 0.0

    for i in range(N):

        theta_star = theta + delta[:, i]

        m_i, s2_i = eta_predict(
            x_obs[i],
            theta_star,
            gp_eta
        )

        total_var = sigma2 + s2_i

        resid = y_obs[i] - m_i

        loglike += -0.5 * (
            np.log(2*np.pi*total_var)
            + resid**2 / total_var
        )

    return loglike

def mh_update_delta_k(
    k, delta, theta,
    ell_k, var_k,
    x_obs, y_obs,
    gp_eta,
    sigma2,
    mh_scale
):
    No = len(x_obs)

    # --- GP prior covariance ---
    K = rbf_kernel(x_obs, x_obs, ell=ell_k, var=var_k) + 1e-8*np.eye(No)
    L = np.linalg.cholesky(K)

    # --- proposal ---
    proposal = delta[k] + mh_scale * (L @ np.random.randn(No))

    delta_prop = delta.copy()
    delta_prop[k] = proposal

    # --- log posterior current ---
    logpost_curr = (
        log_likelihood_embedded(y_obs, x_obs, theta, delta, gp_eta, sigma2)
        + gp_log_density(delta[k], K)
    )

    # --- log posterior proposed ---
    logpost_prop = (
        log_likelihood_embedded(y_obs, x_obs, theta, delta_prop, gp_eta, sigma2)
        + gp_log_density(proposal, K)
    )

    # print("mean proposal jump:", np.linalg.norm(delta_prop - delta))

    log_alpha = logpost_prop - logpost_curr

    # print(f"log posterior current: {logpost_curr:.3f}, proposed: {logpost_prop:.3f}, log alpha: {log_alpha:.3f}")

    if np.log(np.random.rand()) < log_alpha:
        delta[k] = proposal
        return delta, True
    else:
        return delta, False

def export_emulator_json(gp, filename,path="results/"):
    """
    Export sklearn GaussianProcessRegressor to JSON.
    """

    if isinstance(gp, pd.DataFrame):
        gp.to_json(f"{path}{filename}", orient="table", indent=4)
        print(f"Emulator exported to {filename}")
        return

    export_dict = {
        "kernel": str(gp.kernel_),
        "kernel_params": gp.kernel_.get_params(),
        "alpha": float(gp.alpha),
        "normalize_y": bool(gp.normalize_y),
        "X_train_shape": gp.X_train_.shape,
        "y_train_shape": gp.y_train_.shape,
    }

    # Convert numpy arrays to lists for JSON
    export_dict["X_train"] = gp.X_train_.tolist()
    export_dict["y_train"] = gp.y_train_.tolist()

    with open(f"{path}{filename}", "w") as f:
        json.dump(export_dict, f, indent=4)

    print(f"Emulator exported to {filename}")

def export_emulator_csv(gp, filename_prefix, path="results/"):
    """
    Export training data to CSV files.
    """

    if isinstance(gp, pd.DataFrame):
        gp.to_csv(f"{path}{filename_prefix}", index=False)
    else:
        df_X = pd.DataFrame(gp.X_train_)
        df_y = pd.DataFrame(gp.y_train_, columns=["y"])

        df_X.to_csv(f"{path}{filename_prefix}_X_train.csv", index=False)
        df_y.to_csv(f"{path}{filename_prefix}_y_train.csv", index=False)

    print("Training data exported to CSV.")

def build_known_theta_field(
    known_theta_form,
    form_config,
    x_phys,
    theta_idx
):
    """
    Returns known theta(x) field in physical units.
    """

    x = np.asarray(x_phys).ravel()

    if known_theta_form == "constant":

        values = form_config.get("values", [])

        if theta_idx < len(values):
            return np.ones_like(x) * values[theta_idx]

    elif known_theta_form == "trig_funct":

        funcs = form_config.get("functions", [])

        if theta_idx < len(funcs):

            f = funcs[theta_idx]

            if f == "sin":
                return np.sin(x)

            elif f == "cos":
                return np.cos(x)

    return np.zeros_like(x)