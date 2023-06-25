##############################################
# Imports

import gpytorch
import math
from matplotlib import pyplot as plt
import os
from os.path import join as path_join
import pandas as pd
import sys
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi import utils as utils
from sbi import analysis as analysis

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

from ml_fit_lib.config import get_config
from ml_fit_lib.constants import *
from ml_fit_lib.data import Data
from ml_fit_lib.gp import get_grape_data, load_gp
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

    X_train, Y_train, X_test, Y_test, SCALER, X_train_df, X_test_df = get_grape_data(
        DATA, GRAPEVINE_PARAMS_TRAINING, GRAPEVINE_PARAMS_TESTING, GRAPEVINE_DATA_TRAINING, 
        GRAPEVINE_DATA_TESTING, GRAPEVINE_RESP
    )

    MODEL, LIKELIHOOD = load_gp(X_train, Y_train, NUM_LATENTS, GP_MODEL_FILE)
    MODEL, LIKELIHOOD = MODEL.to("cpu"), LIKELIHOOD.to("cpu")

    MAX_TIME_STEP = Y_test.shape[0] 
    Y_test = Y_test.reshape(1, -1).detach().cpu().float()
    
    MODEL_TYPE = CONFIG.get("nn_model")
    N_SIMULATIONS = CONFIG.get("nn_num_simulations")
    N_HIDDEN_FEATURES = CONFIG.get("nn_hidden_features")
    N_TRANSFORMS = CONFIG.get("nn_num_transforms")
    N_DRAWS = CONFIG.get("nn_draws")
    NN_OUT_FEATURES = CONFIG.get("nn_out_features")  
    N_SBC_RUNS = CONFIG.get("num_sbc_runs")

    DISTRIBUTION = CONFIG.get("nn_distribution")    
    SAMPLING_ALGO = CONFIG.get("nn_sampling_algo")
    PRIORS = []
    COLUMN_NAMES = list(X_train_df.drop(columns = "time_step").columns)

    Y_DIM = len(GRAPEVINE_RESP)

##############################################
# Main

def simulator_model(theta):
    theta = theta.detach().cpu().numpy()

    sample = {}
    for i, col in enumerate(COLUMN_NAMES):
        sample[col] = theta[i]

    samples = []
    for time_step in range(1, MAX_TIME_STEP + 1):
        sample_clone = sample.copy()
        sample_clone["time_step"] = time_step
        samples.append(sample_clone)

    samples_scaled = SCALER.transform(pd.DataFrame.from_records(samples))
    samples_tensor = torch.from_numpy(samples_scaled).float() 

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = LIKELIHOOD(MODEL(samples_tensor))
        mean = predictions.mean.detach().cpu().reshape(1, -1) 

    return mean

def set_prior(col, lb, ub, sigma = None):
    prior = { "parameter": col, "distribution": DISTRIBUTION }
    PRIORS.append(prior)

    if DISTRIBUTION == "student_t":
        mu = (lb + ub) / 2
        sigma = (ub - lb) / 6
        df = 3.0
        prior["df"] = df
        prior["mu"] = mu
        prior["sigma"] = sigma
        return dist.StudentT(torch.tensor([df]), torch.tensor([mu]), torch.tensor([sigma]))

    if DISTRIBUTION == "normal":
        mu = (lb + ub) / 2
        sigma = (ub - lb) / 6
        prior["mu"] = mu
        prior["sigma"] = sigma
        return dist.Normal(torch.tensor([mu]), torch.tensor([sigma]))

    if DISTRIBUTION == "laplace":
        mu = (lb + ub) / 2
        b = (ub - lb) / 6
        prior["mu"] = mu
        prior["b"] = b
        return dist.Laplace(torch.tensor([mu]), torch.tensor([b]))
    
    prior["lower"] = lb
    prior["upper"] = ub
    return dist.Uniform(torch.tensor([lb]), torch.tensor([ub]))

class SummaryStatNet(nn.Module):
    def __init__(self, input_dims, out_features = 8):
        super().__init__()
        self.x, self.y = input_dims
        
        stride = 4
        self.in_features = math.floor(self.x / stride)

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 5, padding = 2)
        self.pool = nn.MaxPool2d(kernel_size = 4, stride = stride)
        self.fc = nn.Linear(in_features = self.in_features, out_features = out_features)

    def forward(self, x):
        x = x.view(-1, 1, self.x, self.y)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, self.in_features)
        x = self.fc(x)
        x = F.relu(x)

        return x 

def fit_model(
        X_train_bounds
    ):

    priors = []
    limits = []
    for col in X_train_bounds.drop(columns = "time_step").columns:
        lb = X_train_bounds.iloc[0][col]
        ub = X_train_bounds.iloc[-1][col]
        prior = set_prior(col, lb, ub)
        priors.append(prior)
        limits.append((lb, ub))

    simulator, prior = prepare_for_sbi(simulator_model, priors)

    embedding_net = SummaryStatNet((MAX_TIME_STEP, Y_DIM), NN_OUT_FEATURES)
    neural_posterior = utils.posterior_nn(
        model = MODEL_TYPE, embedding_net = embedding_net, hidden_features = N_HIDDEN_FEATURES, num_transforms = N_TRANSFORMS
    )

    inference = SNPE(prior = prior, density_estimator = neural_posterior)
    theta, x = simulate_for_sbi(simulator, proposal = prior, num_simulations = N_SIMULATIONS)
    inference = inference.append_simulations(theta, x, data_device = "cpu")
    density_estimator = inference.train()
    # posterior = inference.build_posterior(density_estimator, sample_with = SAMPLING_ALGO, mcmc_method = "nuts", vi_method = "fKL")
    posterior = inference.build_posterior(density_estimator)
    posterior.set_default_x(Y_test)
    posterior_samples = posterior.sample((N_DRAWS,), x = Y_test)
    
    for plot_func in [analysis.pairplot, analysis.marginal_plot]:
        outfile = path_join("out", f"{plot_func.__name__}.png")
        plt.rcParams.update({'font.size': 8})
        fig, _ = plot_func(posterior_samples, figsize = (20, 12), labels = COLUMN_NAMES)
        fig.savefig(outfile)
        PIPELINE.log_artifact(outfile)

    for plot_func in [analysis.conditional_pairplot, analysis.conditional_marginal_plot]:
        outfile = path_join("out", f"{plot_func.__name__}.png")
        plt.rcParams.update({'font.size': 8})
        fig, _ = plot_func(
            density = posterior, condition = posterior.sample((1,)), figsize = (20, 12), 
            labels = COLUMN_NAMES, limits = limits
        )
        fig.savefig(outfile)
        PIPELINE.log_artifact(outfile)

    thetas = prior.sample((N_SBC_RUNS,))
    xs = simulator(thetas)

    ranks, dap_samples = analysis.run_sbc(
        thetas, xs, posterior, num_posterior_samples = N_DRAWS
    )

    check_stats = analysis.check_sbc(
        ranks, thetas, dap_samples, num_posterior_samples = N_DRAWS
    )

    check_stats_processed = []
    for metric in check_stats:
        metric_dict = { "metric": metric }
        check_stats_processed.append(metric_dict)
        scores = check_stats[metric].detach().cpu().numpy()
        for i, score in enumerate(scores):
            col_name = COLUMN_NAMES[i]
            metric_dict[col_name] = score
    
    outfile = path_join("out", "diagnostics.csv")
    pd.DataFrame(check_stats_processed).to_csv(outfile, index = False)
    PIPELINE.log_artifact(outfile)
   
    for plot_type in [ "hist", "cdf" ]:
        outfile = path_join("out", f"{analysis.sbc_rank_plot.__name__}_{plot_type}.png")
        plt.rcParams.update({'font.size': 8})

        fig, _ = analysis.sbc_rank_plot(
            ranks = ranks,
            num_posterior_samples = N_DRAWS,
            plot_type = plot_type,
            parameter_labels = COLUMN_NAMES
        )
        fig.savefig(outfile)
        PIPELINE.log_artifact(outfile)

    PIPELINE.log_param("nn_out_features", NN_OUT_FEATURES)
    PIPELINE.log_param("nn_sampling_algo", SAMPLING_ALGO)
    PIPELINE.log_param("nn_model", MODEL_TYPE)
    PIPELINE.log_param("nn_num_simulations", N_SIMULATIONS)
    PIPELINE.log_param("nn_hidden_features", N_HIDDEN_FEATURES)
    PIPELINE.log_param("nn_num_transforms", N_TRANSFORMS)
    PIPELINE.log_param("nn_draws", N_DRAWS)
    PIPELINE.log_param("num_sbc_runs", N_SBC_RUNS)

    PIPELINE.log_param("seed", SEED)
    PIPELINE.log_param("distribution", DISTRIBUTION)

    outfile = path_join("out", "priors.csv")
    pd.DataFrame.from_dict(PRIORS).to_csv(outfile, index = False)
    PIPELINE.log_artifact(outfile)

    return X_train_bounds
    
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