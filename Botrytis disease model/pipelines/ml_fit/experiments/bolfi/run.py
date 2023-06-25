##############################################
# Imports

import elfi
import gpytorch
import numpy as np
import os
from os.path import join as path_join
import pandas as pd
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

    N_DRAWS = CONFIG.get("draws")
    N_CHAINS = CONFIG.get("chains")
    DISTRIBUTION = CONFIG.get("distribution")
    
    PRIORS = []

    INITIAL_EVIDENCE = CONFIG.get("bolfi_initial_evidence")
    N_EVIDENCE = CONFIG.get("bolfi_n_evidence")
    SAMPLER = CONFIG.get("bolfi_sampler")

##############################################
# Main

def simulator_model(
        Vmax_berry, KM_BERRY, Cstar, coefLpmax, coefLpstar, coefLxmax, coefLxstar, 
        DegradationRateMalic_av, roMin, roMax, batch_size = 1, random_state = None
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

def set_prior(col, lb, ub):
    prior = { "parameter": col, "distribution": DISTRIBUTION }
    PRIORS.append(prior)

    if DISTRIBUTION == "triangular":
        c = (lb + ub) / 2
        prior["c"] = c
        mu = (lb + ub) / 2
        sigma = (ub - lb) / 6
        prior["mu"] = mu
        prior["sigma"] = sigma
        return elfi.Prior('triang', c = c, loc = mu, scale = sigma, name = col)
 
    if DISTRIBUTION == "normal":
        mu = (lb + ub) / 2
        sigma = (ub - lb) / 6
        prior["mu"] = mu
        prior["sigma"] = sigma
        return elfi.Prior('norm', mu, sigma, name = col)

    if DISTRIBUTION == "laplace":
        mu = (lb + ub) / 2
        b = (ub - lb) / 6
        prior["mu"] = mu
        prior["b"] = b
        return elfi.Prior('laplace', mu, b, name = col)
    
    prior["lower"] = lb
    prior["upper"] = ub
    return elfi.Prior('uniform', lb, ub - lb, name = col)

def fit_model(
        X_train_bounds
    ):

    priors = []
    bounds = {}
    for col in X_train_bounds.drop(columns = "time_step").columns:
        lb = X_train_bounds.iloc[0][col]
        ub = X_train_bounds.iloc[-1][col]
        param = set_prior(col, lb, ub)
        bounds[col] = (lb, ub)
        priors.append(param)

    simulator = elfi.Simulator(simulator_model, *priors, observed = Y_TEST, name = 'simulator')
    mean_stat = elfi.Summary(lambda y: np.mean(y, axis = 0), simulator, name = 'mean_stat')
    variance_stat = elfi.Summary(lambda y: np.var(y, axis = 0), simulator, name = 'variance_stat')
    
    euclidean_distance = elfi.Distance(
        lambda XA, XB: np.expand_dims(
            np.linalg.norm(XA.flatten() - XB.flatten()), axis = 0
        ), 
        mean_stat, variance_stat, name = 'euclidean_distance'
    )
    log_distance = elfi.Operation(np.log, euclidean_distance, name = 'log_distance')

    pgm = elfi.draw(log_distance)
    pgm_out = path_join("out", "model_graph.png")
    pgm.render(format = "png", directory = "out", filename = "model_graph")

    PIPELINE.log_artifact(pgm_out)

    bolfi = elfi.BOLFI(
        log_distance, batch_size = 1, initial_evidence = INITIAL_EVIDENCE, update_interval = int(INITIAL_EVIDENCE / 10), 
        bounds = bounds, seed = SEED
    )

    post = bolfi.fit(n_evidence = N_EVIDENCE)

    for plot_func in [bolfi.plot_state, bolfi.plot_discrepancy, bolfi.plot_gp]:
        plot_func()
        outfile = path_join("out", f"{plot_func.__name__}.png")
        plt.savefig(outfile)
        PIPELINE.log_artifact(outfile)

    # post.plot(logpdf = True)
    # plt.savefig(path_join("out", "posterior.png"))

    result_BOLFI = bolfi.sample(
        N_DRAWS, algorithm = SAMPLER, n_chains = N_CHAINS, # info_freq = int(N_DRAWS / 10)
    )

    for plot_func in [ result_BOLFI.plot_traces, result_BOLFI.plot_marginals ]:
        plot_func()
        outfile = path_join("out", f"{plot_func.__name__}.png")
        plt.savefig(outfile)
        PIPELINE.log_artifact(outfile)

    PIPELINE.log_param("draws", N_DRAWS)
    PIPELINE.log_param("chains", N_CHAINS)
    PIPELINE.log_param("seed", SEED)
    PIPELINE.log_param("distribution", DISTRIBUTION)
    PIPELINE.log_param("initial_evidence", INITIAL_EVIDENCE)
    PIPELINE.log_param("n_evidence", N_EVIDENCE)
    PIPELINE.log_param("sampler", SAMPLER)

    outfile = path_join("out", "priors.csv")
    pd.DataFrame.from_dict(PRIORS).to_csv(outfile, index = False)
    PIPELINE.log_artifact(outfile)
        
def main():
    X_train_bounds = X_train_df.agg(['min', 'max'])
    fit_model(X_train_bounds)

    outfile = path_join("out", "ground_truth.csv")
    X_test_df.query("time_step == 1").to_csv(outfile, index = False)
    PIPELINE.log_artifact(outfile)

    config_out = path_join("out", CONFIG_FILE)
    CONFIG.export(config_out)
    PIPELINE.log_artifact(config_out)

if __name__ == "__main__":
    main()