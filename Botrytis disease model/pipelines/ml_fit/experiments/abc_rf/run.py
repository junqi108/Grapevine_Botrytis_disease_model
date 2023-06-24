##############################################
# Imports

import arviz as az
import gpytorch
import mlflow
import numpy as np
import os
from os.path import join as path_join
import pandas as pd
import pymc as pm
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
import torch

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

from ml_fit_lib.config import get_config
from ml_fit_lib.constants import *
from ml_fit_lib.data import Data
from ml_fit_lib.gp import (
    MultitaskVariationalGPModel, get_gp_scores, log_predictive_dist,
    get_device, plot_results, get_grape_data, load_gp
)
from ml_fit_lib.mlops import MLFlowPipeline

##############################################
# Constants

if __name__ == "__main__":
    CONFIG = get_config(CONFIG_PATH)

    PIPELINE = MLFlowPipeline()

    NUM_LATENTS = CONFIG.get("num_latents")
    NUM_EPOCHS =  CONFIG.get("num_epochs")
    LEARNING_RATE =  CONFIG.get("lr")
    EPOCH_INTERVAL = int(NUM_EPOCHS / 5)

    DATA = Data()

    SEED = CONFIG.get("random_seed")

    X_train, Y_train, _, Y_test, SCALER, X_train_df, X_test_df = get_grape_data(
        DATA, GRAPEVINE_PARAMS_TRAINING, GRAPEVINE_PARAMS_TESTING, GRAPEVINE_DATA_TRAINING, 
        GRAPEVINE_DATA_TESTING, GRAPEVINE_RESP
    )

    Y_TEST = Y_test.cpu().numpy()
    MODEL, LIKELIHOOD = load_gp(X_train, Y_train, NUM_LATENTS, GP_MODEL_FILE)
    MODEL, LIKELIHOOD = MODEL.to("cpu"), LIKELIHOOD.to("cpu")
    MAX_TIME_STEP = Y_test.shape[0] 

    DISTRIBUTION = CONFIG.get("distribution")

    PRIORS = []

    STATISTICS = [ "mean_laplace", "median_laplace", "mean_gaussian", "median_gaussian" ]
    STATISTIC_FUNCTIONS = [ np.mean, np.median ]

##############################################
# Main

def simulator_model(
        rng, Vmax_berry, KM_BERRY, Cstar, coefLpmax, coefLpstar, coefLxmax, coefLxstar, 
        DegradationRateMalic_av, roMin, roMax, size = None
    ):

    sample = {
        "Vmax_berry": Vmax_berry, 
        "KM_BERRY": KM_BERRY,
        "Cstar": Cstar, 
        "coefLpmax": coefLpmax, 
        "coefLpstar": coefLpstar, 
        "coefLxmax": coefLxmax, 
        "coefLxstar": coefLxstar, 
        "DegradationRateMalic_av": DegradationRateMalic_av, 
        "roMin": roMin, 
        "roMax": roMax
    }

    samples = []
    for time_step in range(1, MAX_TIME_STEP + 1):
        sample_clone = sample.copy()
        sample_clone["time_step"] = time_step
        samples.append(sample_clone)

    samples_scaled = SCALER.transform(pd.DataFrame.from_records(samples))
    samples_tensor = torch.from_numpy(samples_scaled).float() 

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = LIKELIHOOD(MODEL(samples_tensor))
        mean = predictions.mean.cpu().numpy() 
        
    return mean

def set_prior(col, lb, ub, sigma = None):
    prior = { "parameter": col, "distribution": DISTRIBUTION }
    PRIORS.append(prior)

    if DISTRIBUTION == "triangular":
        c = (lb + ub) / 2
        prior["c"] = c
        prior["lower"] = lb
        prior["upper"] = ub
        return pm.Triangular(col, lower = lb, upper = ub, c = c)

    if DISTRIBUTION == "truncated_normal":
        mu = (lb + ub) / 2
        sigma = (ub - lb) / 6
        lower = 0
        prior["mu"] = mu
        prior["sigma"] = sigma
        prior["lower"] = lower
        return pm.TruncatedNormal(col, mu = mu, sigma = sigma, lower = lower)

    if DISTRIBUTION == "laplace":
        mu = (lb + ub) / 2
        b = (ub - lb) / 6
        prior["mu"] = mu
        prior["b"] = b
        return pm.Laplace(col, mu = mu, b = b)
    
    if DISTRIBUTION == "horseshoe":
        mu = (lb + ub) / 2
        lower = 0
        prior["mu"] = mu
        prior["lower"] = lower
        return pm.TruncatedNormal(col, mu = mu, sigma = sigma, lower = lower)
    
    prior["lower"] = lb
    prior["upper"] = ub
    return pm.Uniform(col, lb, ub)

def fit_model(
        X_train_bounds, sum_stat, distance
    ):
    with pm.Model() as model:
        σ = None
        if DISTRIBUTION == "horseshoe":
            λ = pm.HalfCauchy('lambda', beta = 1)
            τ = pm.HalfCauchy('tau', beta = 1)
            σ = pm.Deterministic(DISTRIBUTION, τ * τ* λ * λ)

        params = []
        for col in X_train_bounds.drop(columns = "time_step").columns:
            lb = X_train_bounds.iloc[0][col]
            ub = X_train_bounds.iloc[-1][col]
            param = set_prior(col, lb, ub, σ)
            params.append(param)
        params = tuple(params)

        pm.Simulator(
            "Y_obs",
            simulator_model,
            params = params,
            distance = distance,
            sum_stat = sum_stat,
            epsilon = 1,
            observed = Y_TEST,
        )

    chains = CONFIG.get("chains")

    with model:
        trace = pm.sample_smc(
            draws = CONFIG.get("draws"), 
            # kernel = "abc",
            chains = chains,
            cores = 1,
            compute_convergence_checks = True,
            return_inferencedata = True,
            random_seed = SEED,
            progressbar = True
        )

    return model, trace

def select_model(
    models, n_trees = 1000, f_max_features = 0.5 
):
    n_models = len(models)
    ref_table = []

    for model, trace in models.values():
        obs_name = model.observed_RVs[0].name
        pps = pm.sample_posterior_predictive(
            trace, model = model, return_inferencedata = False,
            progressbar = True, random_seed = SEED
        )

        # 3 dims if only one chain
        # 4 dims if multiple chains
        n_chains = 1
        posterior_predictive_draws = pps[obs_name].squeeze()
        if posterior_predictive_draws.ndim == 4:
            n_chains, n_draws, n_obs, n_features  = posterior_predictive_draws.shape
            posterior_predictive_draws = posterior_predictive_draws.reshape((n_chains * n_draws, n_obs, n_features))

        pps_sum = []
        for stat in STATISTIC_FUNCTIONS:
            # n_features = n_outputs * n_summary_statistics
            # Reduce to 2 dims: ndraws x n_features
            val = np.apply_along_axis(stat, 1, posterior_predictive_draws)
            if val.ndim > 1:
                for v in val.T:
                    pps_sum.append(v)
            else:
                pps_sum.append(val)

        pps_sum = np.array(pps_sum).T
        pps_sum = np.repeat((pps_sum,), n_chains, axis = 0).squeeze()
        ref_table.append(pps_sum)
    ref_table = np.concatenate(ref_table)

    # If multiple chains, then we have 3 dims 
    # So, we need to reduce this to 2 dims
    if n_chains > 1:
        # n_model_types = model_types * n_summary_statistics
        n_model_types, n_obs, n_features = ref_table.shape
        ref_table = ref_table.reshape((n_model_types * n_obs, n_features))

    obs_sum = []
    for stat in STATISTIC_FUNCTIONS:
        # Reduce to 1-dim: n_outputs * n_summary_statistics
        val = np.apply_along_axis(stat, 0, Y_TEST)
        if val.ndim > 1:
            for v in val.T:
                obs_sum.append(v)
        else:
            obs_sum.append(val)

    obs_sum = np.hstack(obs_sum)
    labels = np.repeat(np.arange(n_models), CONFIG.get("draws") * n_chains**2 )
    
    max_features = int(f_max_features * ref_table.shape[1])
    classifier = RandomForestClassifier(
        n_estimators = n_trees,
        max_features = max_features,
        bootstrap = True,
        random_state = SEED
    )

    classifier.fit(ref_table, labels)

    best_model_indx = int(classifier.predict([obs_sum]))
    best_model = STATISTICS[best_model_indx]
    pred_prob = classifier.predict_proba(ref_table)
    pred_error = 1 - np.take(pred_prob.T, labels)

    regressor = RandomForestRegressor(n_estimators = n_trees, random_state = SEED)
    regressor.fit(ref_table, pred_error)
    prob_best_model = 1 - regressor.predict([obs_sum])

    return best_model, prob_best_model.item()

def main():
    X_train_bounds = X_train_df.agg(['min', 'max'])

    models = {}
    for stat_distance in STATISTICS:
        sum_stat, distance = stat_distance.split("_")
        model = fit_model(X_train_bounds, sum_stat, distance)
        models[stat_distance] = model

    best_model, prob_best_model = select_model(models)
    
    outfile = path_join("out", "models_types.csv")
    pd.DataFrame({ "models": STATISTICS }).to_csv(outfile, index = False)
    PIPELINE.log_artifact(outfile)
    
    PIPELINE.log_param("draws", CONFIG.get("draws"))
    PIPELINE.log_param("chains", CONFIG.get("chains"))
    PIPELINE.log_param("seed", SEED)
    PIPELINE.log_param("distribution", CONFIG.get("distribution"))
    PIPELINE.log_param("best_model", best_model)
    PIPELINE.log_metric("prob_best_model", prob_best_model)

    config_out = path_join("out", CONFIG_FILE)
    CONFIG.export(config_out)
    PIPELINE.log_artifact(config_out)

if __name__ == "__main__":
    main()