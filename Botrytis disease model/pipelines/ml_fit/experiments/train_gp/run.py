##############################################
# Imports

import gpytorch
import mlflow
import os
from os.path import join as path_join
import pandas as pd
from matplotlib import pyplot as plt
import sys
import torch
import tempfile

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
     
##############################################
# Main

def main():
    X_train, Y_train, X_test, Y_test, _, _, _ = get_grape_data(
        DATA, GRAPEVINE_PARAMS_TRAINING, GRAPEVINE_PARAMS_TESTING, GRAPEVINE_DATA_TRAINING, 
        GRAPEVINE_DATA_TESTING, GRAPEVINE_RESP
    )
    
    model = MultitaskVariationalGPModel(n_col = X_train.shape[-1], num_latents = NUM_LATENTS, num_tasks = Y_train.shape[-1])
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks = Y_train.shape[-1])

    device = get_device()
    model.train().to(device)
    likelihood.train().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data = Y_train.size(0))

    mlflow.log_param("Number Latent Variables", NUM_LATENTS)
    mlflow.log_param("Learning Rate", LEARNING_RATE)
    mlflow.log_param("Epochs", NUM_EPOCHS)
    for i in range(NUM_EPOCHS):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, Y_train)
        loss.backward()
        optimizer.step()

        if i % EPOCH_INTERVAL == 0:
            print(f"Epochs: {i}  Loss: {loss.item()}")
            mlflow.log_metric("loss", loss.item(), step = i)

    print(f"Final Loss: {loss.item()}")
    mlflow.log_metric("loss", loss.item(), step = i)

    torch.save(model.state_dict(), GP_MODEL_FILE)

    model.eval()
    likelihood.eval()

    f, ((y1_ax, y2_ax), (y3_ax, y4_ax), (y5_ax, y6_ax), (y7_ax, y8_ax)) = plt.subplots(4, 2, figsize=(10, 10), constrained_layout = True)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(X_test))
        mean = predictions.mean.cpu().numpy()
        lower, upper = predictions.confidence_region()
        lower, upper = lower.cpu().numpy(), upper.cpu().numpy()

    mlflow.log_metric("Test NLPD", gpytorch.metrics.negative_log_predictive_density(predictions, Y_test).detach().cpu().numpy())            

    X_test = X_test.cpu().numpy()
    Y_test = Y_test.cpu().numpy()

    log_predictive_dist(Y_test, mean, lower, upper, GRAPEVINE_RESP)

    plot_results(0, y1_ax, y2_ax, X_test, Y_test, mean, lower, upper, GRAPEVINE_RESP)
    plot_results(1, y3_ax, y4_ax, X_test, Y_test, mean, lower, upper, GRAPEVINE_RESP)
    plot_results(2, y5_ax, y6_ax, X_test, Y_test, mean, lower, upper, GRAPEVINE_RESP)
    plot_results(3, y7_ax, y8_ax, X_test, Y_test, mean, lower, upper, GRAPEVINE_RESP)

    out_preds = path_join("out", "gp_preds.png")

    plt.savefig(out_preds)
    mlflow.log_artifact(out_preds)

    config_out = path_join("out", CONFIG_FILE)
    CONFIG.export(config_out)
    PIPELINE.log_artifact(config_out)

if __name__ == "__main__":
    main()