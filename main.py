import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


from plottingFuncts import *
from functs import *


def main():

    # Read in data for processing
    with open('config.json', 'r') as f:
            config = json.load(f)

    calibration_settings = config['calibration_settings']
    input_settings = config['input_settings']
    output_settings = config['output_settings']

    file_delimiter = input_settings['input_delimiter']

    model_data = pd.DataFrame()

    #populate x values from appDomain.txt
    # The first line of appDomain.txt contains the header names
    # The subsequent lines contain the x values
    with open('modelData/appDomain.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        headers = lines[0].strip().split(file_delimiter)
        for i, header in enumerate(headers):
            col_name = f'x_{header}'
            model_data[col_name] = [float(line.strip().split(file_delimiter)[i]) for line in lines[1:]]

    # #populate theta values from thetaVals.txt
    # The first line of thetaVals.txt contains the header names
    # The subsequent lines contain the theta values
    with open('modelData/thetaVals.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        headers = lines[0].strip().split(file_delimiter)
        for i, header in enumerate(headers):
            col_name = f'theta_{header}'
            model_data[col_name] = [float(line.strip().split(file_delimiter)[i]) for line in lines[1:]]

    #populate y values from modelPredictions.txt
    # The first line of modelPredictions.txt contains the header names
    # The subsequent lines contain the y values
    # If there are blank lines at the end of the file, ignore
    with open('modelData/modelPredictions.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        headers = lines[0].strip().split(file_delimiter)
        for i, header in enumerate(headers):
            col_name = f'y_{header}'
            model_data[col_name] = [float(line.strip().split(file_delimiter)[i]) for line in lines[1:]]


    # Create a dataframe to store the observation data
    # The DataFrame can have any number of rows and any number of columns
    # The columns will be identified by header names x_{label}, y_{label}, etc.
    # x will be read from obsData/appDomain.txt
    # y will be read from obsData/observationData.txt
    # Each row will correspond to a different observation
    obs_data = pd.DataFrame()
    with open('observationData/appDomain.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        headers = lines[0].strip().split(file_delimiter)
        for i, header in enumerate(headers):
            col_name = f'x_{header}'
            obs_data[col_name] = [float(line.strip().split(file_delimiter)[i]) for line in lines[1:]]

    with open('observationData/observationData.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        headers = lines[0].strip().split(file_delimiter)
        for i, header in enumerate(headers):
            col_name = f'xi_{header}'
            obs_data[col_name] = [float(line.strip().split(file_delimiter)[i]) for line in lines[1:]]


    model_domain = model_data[[col for col in model_data.columns if col.startswith('x_')]]
    calib_params = model_data[[col for col in model_data.columns if col.startswith('theta_')]]

    theta_labels = [col.replace('theta_', '') for col in calib_params.columns]
    app_labels = [col.replace('x_', '') for col in model_domain.columns]

    # create standardized/normalized dataframes
    model_normalized = model_data.copy()
    observation_normalized = obs_data.copy()

    # transform the simulator inputs (x,theta) to the unit hypercube [0,1] using min-max scaling
    for col in model_data.columns:
        if col.startswith('x_') or col.startswith('theta_'):
            min_val = model_data[col].min()
            max_val = model_data[col].max()
            model_normalized[col] = (model_data[col] - min_val) / (max_val - min_val) 

    # transform the simulator outputs to be zero mean and unit variance
    for col in model_data.columns:
        if col.startswith('y_'):
            mean_val = model_data[col].mean()
            std_val = model_data[col].std()
            model_normalized[col] = (model_data[col] - mean_val) / std_val
            
    # repeat the procedure for the observation data
    for col in obs_data.columns:
        if col.startswith('x_'):
            min_val = obs_data[col].min()
            max_val = obs_data[col].max()
            observation_normalized[col] = (obs_data[col] - min_val) / (max_val - min_val)
            
    for col in obs_data.columns:
        if col.startswith('xi_'):
            mean_val = obs_data[col].mean()
            std_val = obs_data[col].std()
            observation_normalized[col] = (obs_data[col] - mean_val) / std_val
        

    # Create figures directory if it doesn't exist
    figures_directory = output_settings['figures_directory']
    if not os.path.exists(figures_directory):
        os.makedirs(figures_directory)

    figure_options = output_settings['figure_options']
    
    if figure_options['data_priors']:
        generate_rawData_figure(model_data, obs_data, figures_directory)

    # Simulator inputs
    x_sim = model_normalized[[col for col in model_normalized.columns if col.startswith('x_')]]
    theta_sim = model_normalized[[col for col in model_normalized.columns if col.startswith('theta_')]]
    y_sim = model_normalized[[col for col in model_normalized.columns if col.startswith('y_')]]

        # Joint emulator input z = (x, theta)
    Z_sim = np.hstack([x_sim, theta_sim])

    # Simulator outputs
    y_sim = model_normalized["y_crss_DDD"].values


    d = Z_sim.shape[1]  # dimension = 1 + dtheta

    kernel_eta = C(1.0, (1e-2, 1e2)) * RBF(
        length_scale=np.ones(d),
        length_scale_bounds=(1e-2, 1e2)
    )

    gp_eta = GaussianProcessRegressor(
        kernel=kernel_eta,
        alpha=1e-6,           # nugget for stability
        normalize_y=True,
        n_restarts_optimizer=5
    )

    gp_eta.fit(Z_sim, y_sim)

    x_obs = observation_normalized[[col for col in observation_normalized.columns if col.startswith('x_')]].values
    y_obs = observation_normalized[[col for col in observation_normalized.columns if col.startswith('xi_')]].values.flatten()

    dtheta = theta_sim.shape[1]
    No = len(x_obs)

    # ============================================================
    # 2. Train probabilistic emulator GP η(x,θ)
    # ============================================================

    Z_sim = np.hstack([x_sim, theta_sim])

    kernel_eta = C(1.0, (1e-2, 1e2)) * RBF(
        length_scale=np.ones(Z_sim.shape[1]),
        length_scale_bounds=(1e-2, 1e2)
    )

    gp_eta = GaussianProcessRegressor(
        kernel=kernel_eta,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5
    )

    gp_eta.fit(Z_sim, y_sim)

    print("\nTrained emulator kernel:")
    print(gp_eta.kernel_)

    # ============================================================
    # 3. Initialize θ (fixed plug-in estimate)
    # ============================================================
    if calibration_settings['theta_initialization'] == 'fixed':
        theta_fixed = theta_sim.mean(axis=0)

        print("\nFixed theta used:")
        print(theta_fixed)
    else:
        print("Only fixed theta initialization is currently implemented.")

    # ============================================================
    # 4. Initialize discrepancy δₖ(x) latent vectors
    # ============================================================

    delta = np.zeros((dtheta, No))

    # Initial hyperparameters for each δₖ
    ell_delta = 0.3 * np.ones(dtheta)
    var_delta = 0.05 * np.ones(dtheta)

    # Initial noise variance
    sigma2 = 0.02**2

    Nmcmc=calibration_settings["N_mcmc"]
    mh_scale_delta=calibration_settings["mh_scale_delta_prior"]

    prior_ell = {"mu": -1.0, "sigma": 1.0}
    prior_var = {"mu": -2.0, "sigma": 1.0}

    mh_scales = {
        "log_ell_delta": 0.1,
        "log_var_delta": 0.1
    }

    a_sigma = 2.0
    b_sigma = 0.5


    # Storage
    delta_chain = np.zeros((Nmcmc, dtheta, No))
    sigma2_chain = np.zeros(Nmcmc)

    accept_delta = np.zeros(dtheta)

    # --- acceptance trace storage ---
    accept_trace = [[] for _ in range(dtheta)]

    # --- MH scaling adaptation settings ---
    burnin = 500
    adapt_interval = 50
    target_accept = 0.35

    mh_scale_delta = 0.5  # initial guess

    # store scale history (optional)
    mh_scale_trace = []



    # ============================================================
    # 6. Run Embedded Discrepancy Sampler
    # ============================================================

    print("\nRunning embedded δ-MH sampler...\n")

    for it in range(Nmcmc):

        # ---- update δₖ fields ----
        for k in range(dtheta):
            delta, acc = mh_update_delta_k(
                k,
                delta,
                theta_fixed,
                ell_delta[k],
                var_delta[k],
                x_obs,
                y_obs,
                gp_eta,
                sigma2,
                mh_scale_delta
            )
            accept_delta[k] += acc
            accept_trace[k].append(acc)
            
        # ---- adapt mh_scale_delta during burn-in ----
        if it < burnin and (it + 1) % adapt_interval == 0:

            # acceptance rate over last adapt_interval steps
            recent_accept = accept_delta / (it + 1)

            # average across delta dimensions
            mean_accept = np.mean(recent_accept)

            # log-scale update (stable + standard)
            gamma = 0.05  # adaptation speed

            mh_scale_delta *= np.exp(gamma * (mean_accept - target_accept))

            print(f"[ADAPT] Iter {it+1}: mean accept = {mean_accept:.3f}")
            print(f"[ADAPT] Updated mh_scale_delta = {mh_scale_delta:.4f}")


        # ---- update δ hyperparameters ----
        for k in range(dtheta):
            ell_delta[k], var_delta[k], _ = mh_update_delta_hyperparams(
                delta[k],
                ell_delta[k],
                var_delta[k],
                x_obs,
                prior_ell,
                prior_var,
                mh_scales
            )

        # ---- Gibbs update σ² ----
        sigma2 = gibbs_sigma2(
            y_obs,
            x_obs,
            theta_fixed,
            delta,
            gp_eta,
            a_sigma,
            b_sigma
        )

        # ---- store ----
        delta_chain[it] = delta
        sigma2_chain[it] = sigma2

        # ---- diagnostics ----
        if (it+1) % 50 == 0:
            print(f"Iter {it+1}/{Nmcmc}")
            print(" sigma2 =", sigma2)
            print(" delta acceptance rates:",
                accept_delta / (it+1))
            print(" ell_delta:", ell_delta)
            print(" var_delta:", var_delta)

            print("--------------------------------------------------")


    print("\nSampler complete.")


    # ============================================================
    # 7. Posterior Predictive Check
    # ============================================================

    print("\nBuilding posterior predictive mean + uncertainty...\n")

    y_post_mean = np.zeros(No)
    y_post_var  = np.zeros(No)

    Nsamp = calibration_settings['posterior_predictive_samples']
    idx = np.random.choice(Nmcmc, Nsamp, replace=False)

    for i in range(No):

        preds = []

        for s in idx:

            delta_s = delta_chain[s, :, i]
            theta_star = theta_fixed + delta_s

            m_i, s2_i = eta_predict(x_obs[i], theta_star, gp_eta)

            preds.append(m_i)

        preds = np.array(preds)
        

        y_post_mean[i] = preds.mean()
        y_post_var[i]  = preds.var()
        # print prediction mean and 95% credible interval
        print(f"x_obs[{i}] = {x_obs[i,0]:.3f}, y_obs = {y_obs[i]:.3f}, post pred mean = {y_post_mean[i]:.3f}, 95% CI = [{y_post_mean[i] - 2*np.sqrt(y_post_var[i]):.3f}, {y_post_mean[i] + 2*np.sqrt(y_post_var[i]):.3f}]")

    # ============================================================
    # 8. Convert back to physical units for plotting
    # ============================================================

    x_obs_col = [col for col in obs_data.columns if col.startswith('x_')][0]
    y_obs_col = [col for col in obs_data.columns if col.startswith('xi_')][0]
    x_sim_col = [col for col in model_data.columns if col.startswith('x_')][0]
    y_sim_col = [col for col in model_data.columns if col.startswith('y_')][0]
    theta_cols = [col for col in model_data.columns if col.startswith('theta_')]

    x_obs_min = obs_data[x_obs_col].min()
    x_obs_max = obs_data[x_obs_col].max()

    x_sim_min = model_data[x_sim_col].min()
    x_sim_max = model_data[x_sim_col].max()

    x_obs_phys = x_obs * (x_obs_max - x_obs_min) + x_obs_min
    x_sim_phys = x_sim * (x_sim_max - x_sim_min) + x_sim_min


    y_obs_mean = obs_data[y_obs_col].mean()
    y_obs_std = obs_data[y_obs_col].std()

    y_sim_mean = model_data[y_sim_col].mean()
    y_sim_std = model_data[y_sim_col].std()

    y_obs_phys = y_obs * y_obs_std + y_obs_mean
    y_sim_phys = y_sim * y_sim_std + y_sim_mean

    # denormalize posterior predictive (use obs scale to compare with obs)
    y_post_mean_phys = y_post_mean * y_obs_std + y_obs_mean
    y_post_std_phys = y_post_std * y_obs_std

    theta_min = model_data[theta_cols].min().values
    theta_max = model_data[theta_cols].max().values
    theta_range = theta_max - theta_min

    delta_chain_phys = delta_chain * theta_range[:, None]

    # Sort x for smooth plotting (physical)
    x_sorted_idx = np.argsort(x_obs_phys.ravel())
    x_sorted_phys = x_obs_phys.ravel()[x_sorted_idx]

    delta_mean_phys = delta_chain_phys.mean(axis=0)
    delta_std_phys = delta_chain_phys.std(axis=0)

    # ============================================================
    # 9. Plot Posterior Predictive Fit and other figures
    # ============================================================
    if figure_options['posterior_predict_normalized']:
        print('Plotting posterior predictive check (normalized)...')
    if figure_options['posterior_predict_physical']:
        print('Plotting posterior predictive check (physical)...')

        plt.figure(figsize=(8, 6))

        plt.scatter(
            x_obs_phys, y_obs_phys,
            label="Observed MD data (physical)",
            marker="o",
            s=60
        )

        plt.scatter(
            x_sim_phys, y_sim_phys,
            label="Simulator DDD runs (physical)",
            marker="x",
            alpha=0.7
        )

        plt.plot(
            x_obs_phys,
            y_post_mean_phys,
            label="Posterior mean η(x,θ+δ) (physical)",
            linewidth=2
        )

        plt.fill_between(
            x_obs_phys.ravel(),
            y_post_mean_phys - 2 * y_post_std_phys,
            y_post_mean_phys + 2 * y_post_std_phys,
            alpha=0.3,
            label="±2 posterior std"
        )



    if figure_options['delta_posterior_normalized']:
        print('Plotting delta posterior samples (normalized)...')
    if figure_options['delta_posterior_physical']:
        print('Plotting delta posterior samples (physical)...')
    if figure_options['acceptance_trajectory']:
        print('Plotting acceptance trajectory...')
    if figure_options['step_size_plot']:
        print('Plotting step size trajectory...')




    return


if __name__ == "__main__":
    main()