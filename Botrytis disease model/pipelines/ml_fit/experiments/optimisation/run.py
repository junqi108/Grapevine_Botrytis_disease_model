##############################################
# Imports

import gpytorch
import mlflow
from joblib import dump as optimiser_dump, load as optimiser_load
import numpy as np
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, NSGAIISampler, MOTPESampler, QMCSampler
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

    OPTUNA_SAMPLERS = {
        "tpes" : TPESampler, 
        "cmaes" : CmaEsSampler, 
        "nsga" : NSGAIISampler, 
        "motpes" : MOTPESampler,
        "qmc": QMCSampler
    }

    OPTIMISER_NAME = CONFIG.get("optimisation_sampler")
    SEED = CONFIG.get("random_seed")

##############################################
# Main

def get_sampler(sampler_key: str, seed: int):
    tpes = {
        "consider_prior": True,
        "prior_weight": 1.0,
        "consider_endpoints": False,
        "n_startup_trials": 10,
        "seed": seed,
    }

    cmaes = {
        "n_startup_trials": 1,
        "warn_independent_sampling": True,
        "consider_pruned_trials": True,
        "restart_strategy": "ipop",
        "seed": seed
    }

    nsga = {
        "population_size": 50,
        "crossover_prob": 0.9,
        "swapping_prob": 0.5,
        "seed": seed
    }

    motpes = {
        "consider_prior": True,
        "prior_weight": 1.0,
        "consider_endpoints": True,
        "n_startup_trials": 10,
        "seed": seed
    }

    qmc = {
        "qmc_type": "sobol",
        "scramble": True,
        "seed": seed
    }
    
    sampler_kwargs = {
        "tpes" : tpes, 
        "cmaes" : cmaes, 
        "nsga" : nsga, 
        "motpes" : motpes,
        "qmc": qmc
    }
    
    kwargs = sampler_kwargs[sampler_key]
    for k in kwargs:
        PIPELINE.log_param(k, kwargs.get(k))
    PIPELINE.log_param("sampler", sampler_key)

    sampler = OPTUNA_SAMPLERS.get(sampler_key)(**kwargs)
    return sampler

def objective(trial, model, likelihood, y_test, scaler, X_train_bounds):
    sample = {}
    for col in X_train_bounds.drop(columns = "time_step").columns:
        min_val = X_train_bounds.iloc[0][col]
        max_val = X_train_bounds.iloc[-1][col]
        sample[col] = trial.suggest_float(col, min_val, max_val) 

    samples = []
    max_time_step = X_train_bounds.iloc[-1]["time_step"].astype("int") + 1
    for time_step in range(1, max_time_step):
        sample_clone = sample.copy()
        sample_clone["time_step"] = time_step
        samples.append(sample_clone)

    samples_scaled = scaler.transform(pd.DataFrame.from_records(samples))
    device = get_device()
    samples_tensor = torch.from_numpy(samples_scaled).float().to(device)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(samples_tensor))
        mean = predictions.mean.cpu().numpy()
        
    discrepency = np.linalg.norm(mean - y_test).item()
    return discrepency

def save_model(study, X_test_df, output_dir: str):
    trial_out = path_join(output_dir, "trial_results.csv")
    trials_df = study.trials_dataframe()
    PIPELINE.log_series(trials_df["value"].values, "Discrepency")
    trials_df.sort_values("value", ascending = True).to_csv(trial_out, index = False)
    print(f"Trial results written to {trial_out}")
    PIPELINE.log_artifact(trial_out)

    optimiser_file = path_join(output_dir, "tree_optimiser.pkl") 
    optimiser_dump(study, optimiser_file)
    print(f"Optimiser written to {optimiser_file}")
    PIPELINE.log_artifact(optimiser_file)

    def __plot_results(plot_func, plot_name: str):
        img_file = path_join(output_dir, f"{plot_name}.png")
        plot_func(study).write_image(img_file)
        PIPELINE.log_artifact(img_file)

    # __plot_results(optuna.visualization.plot_contour, "contour")
    __plot_results(optuna.visualization.plot_edf, "edf")
    __plot_results(optuna.visualization.plot_optimization_history, "optimization_history")
    __plot_results(optuna.visualization.plot_parallel_coordinate, "parallel_coordinate")
    __plot_results(optuna.visualization.plot_param_importances, "param_importances")
    __plot_results(optuna.visualization.plot_slice, "slice")
    
    for col in X_test_df.drop(columns = "time_step").columns:
        PIPELINE.log_param(f"Ground truth {col}", X_test_df.iloc[0][col])
        
def optimise_sim(model, likelihood, X_test_df, Y_test, scaler, X_train_bounds):
    sampler = get_sampler(OPTIMISER_NAME, SEED)
    n_jobs = CONFIG.get("n_jobs")

    if CONFIG.get_as("load_optimiser", bool):
        optimiser_file = CONFIG.get("optimiser_file")
        if optimiser_file is None:
            raise Exception("Optimiser file must be provided when loading an existing optimiser.")
        study = optimiser_load(optimiser_file)
    else:
        study = optuna.create_study(sampler = sampler, study_name = CONFIG.get("MLFLOW_EXPERIMENT_NAME"), direction = "minimize")

    y_test = Y_test.cpu().numpy()
    study.optimize(lambda trial: objective(trial, model, likelihood, y_test, scaler, X_train_bounds), 
        n_trials = CONFIG.get("n_trials"), n_jobs = 1, gc_after_trial = True, 
        show_progress_bar = (n_jobs == 1))

    save_model(study, X_test_df, "out")

def main():
    X_train, Y_train, _, Y_test, scaler, X_train_df, X_test_df = get_grape_data(
        DATA, GRAPEVINE_PARAMS_TRAINING, GRAPEVINE_PARAMS_TESTING, GRAPEVINE_DATA_TRAINING, 
        GRAPEVINE_DATA_TESTING, GRAPEVINE_RESP
    )
    
    model, likelihood = load_gp(X_train, Y_train, NUM_LATENTS, GP_MODEL_FILE)

    X_train_bounds = X_train_df.agg(['min', 'max'])
    optimise_sim(model, likelihood, X_test_df, Y_test, scaler, X_train_bounds)

    config_out = path_join("out", CONFIG_FILE)
    CONFIG.export(config_out)
    PIPELINE.log_artifact(config_out)

if __name__ == "__main__":
    main()