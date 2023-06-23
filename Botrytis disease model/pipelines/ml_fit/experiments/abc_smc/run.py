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
        
    discrepency = np.linalg.norm(mean - Y_TEST).item()
    return np.array([0])

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
        X_train_bounds
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
            distance = "gaussian",
            sum_stat = "median",
            epsilon = 1,
            observed = np.array([0]),
        )

    pgm = pm.model_to_graphviz(model = model)
    pgm_out = path_join("out", "model_graph.png")
    pgm.render(format = "png", directory = "out", filename = "model_graph")
    PIPELINE.log_artifact(pgm_out)
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

        PIPELINE.log_param("draws", CONFIG.get("draws"))
        PIPELINE.log_param("chains", CONFIG.get("chains"))
        PIPELINE.log_param("seed", SEED)
        PIPELINE.log_param("distribution", CONFIG.get("distribution"))

        textsize = 7
        for plot in ["trace", "rank_vlines", "rank_bars"]:
            az.plot_trace(trace, kind = plot, plot_kwargs = {"textsize": textsize})
            outfile = path_join("out", f"{plot}.png")
            plt.tight_layout()
            plt.savefig(outfile)
            PIPELINE.log_artifact(outfile)

        def __create_plot(trace, plot_func, plot_name, kwargs):
            plot_func(trace, **kwargs)
            outfile = path_join("out", f"{plot_name}.png")
            plt.tight_layout()
            plt.savefig(outfile)
            PIPELINE.log_artifact(outfile)
        
        kwargs = {"figsize": (12, 12), "scatter_kwargs": dict(alpha = 0.01), "marginals": True, "textsize": textsize}
        __create_plot(trace, az.plot_pair, "marginals", kwargs)

        kwargs = {"figsize": (12, 12), "textsize": textsize}
        __create_plot(trace, az.plot_violin, "violin", kwargs)

        kwargs = {"figsize": (12, 12), "textsize": textsize}
        __create_plot(trace, az.plot_posterior, "posterior", kwargs)

        outfile = path_join("out", "priors.csv")
        pd.DataFrame.from_dict(PRIORS).to_csv(outfile, index = False)
        PIPELINE.log_artifact(outfile)

        outfile = path_join("out", "summary.csv")
        az.summary(trace).to_csv(outfile, index = False)
        PIPELINE.log_artifact(outfile)
        
def main():
    X_train_bounds = X_train_df.agg(['min', 'max'])
    fit_model(X_train_bounds)

    config_out = path_join("out", CONFIG_FILE)
    CONFIG.export(config_out)
    PIPELINE.log_artifact(config_out)

if __name__ == "__main__":
    main()