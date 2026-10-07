import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde


def _density(values, x_grid):
    """Return a KDE where possible, falling back for constant samples."""
    values = np.asarray(values, dtype=float)
    if np.ptp(values) == 0 or len(values) < 2:
        return np.zeros_like(x_grid)
    try:
        return gaussian_kde(values)(x_grid)
    except np.linalg.LinAlgError:
        return np.zeros_like(x_grid)


def plot_prior_posterior_densities(
    prior_samples,
    posterior_samples,
    parameter_labels,
    figures_directory,
    filename="prior_posterior_densities.png",
):
    """Plot empirical prior and posterior densities for each calibration parameter."""
    prior_samples = np.asarray(prior_samples, dtype=float)
    posterior_samples = np.asarray(posterior_samples, dtype=float)
    if prior_samples.ndim != 2 or posterior_samples.ndim != 2:
        raise ValueError("prior_samples and posterior_samples must be two-dimensional")
    if prior_samples.shape[1] != posterior_samples.shape[1]:
        raise ValueError("Prior and posterior must have the same parameter count")
    if len(parameter_labels) != prior_samples.shape[1]:
        raise ValueError("parameter_labels must match the parameter count")

    n_parameters = prior_samples.shape[1]
    fig, axes = plt.subplots(
        n_parameters, 1, figsize=(8, max(3, 2.8 * n_parameters)), squeeze=False
    )
    axes = axes[:, 0]
    for index, label in enumerate(parameter_labels):
        ax = axes[index]
        prior = prior_samples[:, index]
        posterior = posterior_samples[:, index]
        combined = np.concatenate((prior, posterior))
        x_grid = np.linspace(np.min(combined), np.max(combined), 300)
        if np.ptp(combined) > 0:
            ax.plot(x_grid, _density(prior, x_grid), label="Prior")
            ax.plot(x_grid, _density(posterior, x_grid), label="Posterior")
        else:
            ax.axvline(combined[0], label="Prior and posterior")
        ax.set_ylabel("Density")
        ax.set_title(str(label))
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Calibration parameter value")
    fig.tight_layout()
    os.makedirs(figures_directory, exist_ok=True)
    path = os.path.join(figures_directory, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved prior-posterior density figure to {path}")
    return path


def plot_holdout_predictions(
    holdout_cases,
    parameter_labels,
    figures_directory,
    full_x,
    full_y_true,
    full_y_pred,
    full_y_pred_std,
    full_parameter_corrections,
    filename="holdout_predictions.png",
):
    """Plot full-domain predictive checks and corrections for each trial.

    Each trial occupies one row. The first column shows the full posterior
    predictive over the application domain, with held-out points highlighted;
    the remaining columns show one correction field per calibration parameter.
    """
    if not holdout_cases:
        raise ValueError("holdout_cases must not be empty")
    full_x = np.asarray(full_x, dtype=float)
    full_x = full_x[:, 0] if full_x.ndim > 1 else full_x
    full_y_true = np.asarray(full_y_true, dtype=float).ravel()
    full_y_pred = np.asarray(full_y_pred, dtype=float).ravel()
    full_y_pred_std = np.asarray(full_y_pred_std, dtype=float).ravel()
    full_parameter_corrections = np.asarray(
        full_parameter_corrections, dtype=float
    )
    n_cases = len(holdout_cases)
    n_parameters = len(parameter_labels)
    if not (
        len(full_x)
        == len(full_y_true)
        == len(full_y_pred)
        == len(full_y_pred_std)
        == full_parameter_corrections.shape[0]
    ):
        raise ValueError("Full-domain arrays must have the same number of points")
    if full_parameter_corrections.shape[1] != n_parameters:
        raise ValueError("Correction columns must match parameter_labels")

    fig, axes = plt.subplots(
        n_cases,
        n_parameters + 1,
        figsize=(5 * (n_parameters + 1), max(4, 3.2 * n_cases)),
        squeeze=False,
        sharex="col",
    )

    for row, case in enumerate(holdout_cases):
        holdout_indices = np.asarray(case["indices"], dtype=int)
        prediction_ax = axes[row, 0]
        prediction_ax.fill_between(
            full_x,
            full_y_pred - 2 * full_y_pred_std,
            full_y_pred + 2 * full_y_pred_std,
            alpha=0.2,
            label="95% posterior interval",
        )
        prediction_ax.scatter(
            full_x, full_y_true, color="0.35", s=18, label="Observations"
        )
        prediction_ax.plot(full_x, full_y_pred, label="Posterior mean")
        prediction_ax.scatter(
            full_x[holdout_indices],
            full_y_true[holdout_indices],
            color="tab:red",
            marker="x",
            s=55,
            label="Held out",
        )
        prediction_ax.set_title(
            f"Full posterior\nfraction={case['fraction']:.2g}, trial={row + 1}"
        )
        prediction_ax.set_xlabel("Application domain")
        prediction_ax.set_ylabel("Response")
        prediction_ax.grid(True, alpha=0.3)
        prediction_ax.legend(fontsize="small")

        for parameter_index, label in enumerate(parameter_labels):
            correction_ax = axes[row, parameter_index + 1]
            correction_ax.plot(
                full_x,
                full_parameter_corrections[:, parameter_index],
                color=f"C{parameter_index}",
            )
            correction_ax.scatter(
                full_x[holdout_indices],
                full_parameter_corrections[holdout_indices, parameter_index],
                color="tab:red",
                marker="x",
                s=55,
                label="Held out",
            )
            correction_ax.axhline(0, color="black", linewidth=0.8)
            correction_ax.set_title(
                f"{label} correction\nfraction={case['fraction']:.2g}, "
                f"trial={row + 1}"
            )
            correction_ax.set_xlabel("Application domain")
            correction_ax.set_ylabel("Correction")
            correction_ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(figures_directory, exist_ok=True)
    path = os.path.join(figures_directory, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved holdout prediction figure to {path}")
    return path


def plot_holdout_nrmse_sensitivity(
    holdout_metrics,
    figures_directory,
    filename="holdout_nrmse_sensitivity.png",
):
    """Plot aggregate NRMSE sensitivity across the full holdout experiment."""
    holdout_results = holdout_metrics.get("holdout_results", {})
    if not holdout_results:
        raise ValueError("holdout_metrics contains no holdout results")

    sorted_results = sorted(
        ((float(fraction), result) for fraction, result in holdout_results.items()),
        key=lambda item: item[0],
    )
    fractions = np.array([item[0] for item in sorted_results], dtype=float)
    result_values = [item[1] for item in sorted_results]
    means = np.array([result["nrmse"] for result in result_values])
    spreads = np.array([result["nrmse_std"] for result in result_values])
    full_nrmse = float(holdout_metrics["full_nrmse"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(
        full_nrmse,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Full dataset (NRMSE={full_nrmse:.3f})",
    )
    ax.fill_between(
        fractions,
        means - spreads,
        means + spreads,
        color="tab:blue",
        alpha=0.18,
        label="Holdout mean +/- 1 SD",
    )
    ax.errorbar(
        fractions,
        means,
        yerr=spreads,
        color="tab:blue",
        marker="o",
        linewidth=2,
        capsize=4,
        label="Mean holdout NRMSE",
    )
    ax.set_xticks(fractions)
    ax.set_xticklabels([f"{fraction:.0%}" for fraction in fractions])
    ax.set_xlabel("Fraction of observations withheld")
    ax.set_ylabel("Normalized RMSE")
    ax.set_title("Holdout sensitivity across the application domain")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    os.makedirs(figures_directory, exist_ok=True)
    path = os.path.join(figures_directory, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved holdout NRMSE sensitivity figure to {path}")
    return path


def generate_rawData_figure(
    model_data,
    obs_data,
    figures_directory,
    observation_set_label="Observation",
    simulation_set_label="Simulation",
):
    """
    Create subplots for each theta parameter showing:
    - X-axis: application domain (x values)
    - Y-axis: calibration metric (y values)
    - Point color: theta parameter value
    - Observed data points
    """
    
    model_x_columns = [col for col in model_data.columns if col.startswith('x_')]
    model_y_columns = [col for col in model_data.columns if col.startswith('y_')]
    observation_x_columns = [col for col in obs_data.columns if col.startswith('x_')] if obs_data is not None else []
    observation_y_columns = [col for col in obs_data.columns if col.startswith('xi_')] if obs_data is not None else []
    
    theta_columns = [col for col in model_data.columns if col.startswith('theta_')]
    
    if not model_x_columns or not model_y_columns or not theta_columns:
        print("Missing required columns (x_, y_, or theta_)")
        return
    
    # Use first y column as calibration metric
    y_col = model_y_columns[0]
    x_col = model_x_columns[0]
    
    # Create subplot grid for theta parameters
    n_theta = len(theta_columns)
    n_cols = int(np.ceil(np.sqrt(n_theta)))
    n_rows = int(np.ceil(n_theta / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, theta_col in enumerate(theta_columns):
        ax = axes[idx]
        
        # Scatter plot colored by theta parameter value
        scatter = ax.scatter(model_data[x_col], model_data[y_col], 
                            c=model_data[theta_col], cmap='viridis', 
                            alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Plot observed data
        if obs_data is not None and observation_y_columns and observation_y_columns[0] in obs_data.columns and observation_x_columns and observation_x_columns[0] in obs_data.columns:
            ax.scatter(
                obs_data[observation_x_columns[0]],
                obs_data[observation_y_columns[0]],
                color='red',
                label=f'{observation_set_label} data',
                alpha=0.8,
                s=30,
                edgecolors='black',
            )
        
        ax.set_xlabel(f'{x_col}')
        ax.set_ylabel(f'{y_col}')
        ax.set_title(f'{y_col} colored by {theta_col}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add colorbar for each subplot
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(theta_col)
    
    # Hide unused subplots
    for idx in range(n_theta, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    # plt_path = os.path.join(figures_directory, f'rawData_theta_parameters.png')
    # # plt.savefig(plt_path, dpi=150)
    # # print(f'Saved raw data figure to {plt_path}')
    # plt.close()
    # plt.show()
    plt_path = os.path.join(figures_directory, f'rawData_theta_parameters.png')
    plt.savefig(plt_path, dpi=150)
    print(f'Saved raw data figure to {plt_path}')
    plt.close()

def plot_delta_acceptance_trajectory(accept_trace, window=50, figures_directory=None):
    """
    Plot acceptance-rate trajectories for each discrepancy GP delta_k.

    Parameters
    ----------
    accept_trace : list of lists
        accept_trace[k][t] = 1 if delta_k accepted at iteration t, else 0

    window : int
        Rolling window size for smoothing acceptance rates

    Output
    ------
    Displays a trajectory plot of rolling acceptance rates.
    """

    n_delta = len(accept_trace)

    plt.figure(figsize=(10, 6))

    for k in range(n_delta):
        accepts = np.array(accept_trace[k])

        # Rolling acceptance rate
        rolling = np.convolve(
            accepts,
            np.ones(window) / window,
            mode="valid"
        )

        plt.plot(
            rolling,
            label=f"delta[{k}]"
        )

    plt.axhline(0.25, linestyle="--", label="Target ~0.25")
    plt.axhline(0.50, linestyle="--", label="Target ~0.50")

    plt.xlabel("Iteration")
    plt.ylabel(f"Rolling Acceptance Rate (window={window})")
    plt.title("Delta GP Acceptance Rate Trajectories")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt_path = os.path.join(figures_directory, f"delta_acceptance_trajectory.png")
    plt.savefig(plt_path, dpi=150)
    print(f"Saved delta acceptance trajectory figure to {plt_path}")
    plt.close()

def plot_delta_jump_sizes(delta_chain, figures_directory=None):
    """
    Plot jump magnitudes ||delta^(t) - delta^(t-1)|| for each delta_k.

    Parameters
    ----------
    delta_chain : array
        Shape (Nmcmc, dtheta, n_obs)
        Stored delta latent fields across iterations.

    Output
    ------
    Line plot of jump norms for each discrepancy GP.
    """

    Nmcmc, dtheta, n_obs = delta_chain.shape

    plt.figure(figsize=(10, 6))

    for k in range(dtheta):

        # difference between successive iterations
        diffs = delta_chain[1:, k, :] - delta_chain[:-1, k, :]

        # norm of each jump
        jump_norms = np.linalg.norm(diffs, axis=1)

        plt.plot(jump_norms, label=f"delta[{k}] jump size")

    plt.xlabel("Iteration")
    plt.ylabel(r"$\|\delta^{(t)} - \delta^{(t-1)}\|$")
    plt.title("Delta GP Jump Magnitudes (Mixing Diagnostic)")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt_path = os.path.join(figures_directory, f"delta_jump_sizes.png")
    plt.savefig(plt_path, dpi=150)
    print(f"Saved delta jump size figure to {plt_path}")
    plt.close()