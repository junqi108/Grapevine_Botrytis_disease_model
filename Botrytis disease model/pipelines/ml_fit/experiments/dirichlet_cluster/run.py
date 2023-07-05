##############################################
# Imports

import arviz as az
import numpy as np
import os
from os.path import join as path_join
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import sys
import scipy as sp
import xarray as xr

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
    DATA = Data()
    SEED = CONFIG.get("random_seed")

    K = CONFIG.get("dirichlet_n_components")
    VI_METHOD = CONFIG.get("dirichlet_vi_method")
    N_ADVI_SAMPLES = CONFIG.get("dirichlet_n_vi_samples")
    N_PARTICLES = CONFIG.get("dirichlet_n_particles")
    LR = CONFIG.get("dirichlet_lr")
    POSTERIOR_DRAWS = CONFIG.get("dirichlet_post_draws")

    ALPHA_SHAPE = CONFIG.get("dirichlet_alpha_shape")
    ALPHA_RATE = CONFIG.get("dirichlet_alpha_rate")
    
##############################################
# Main

def stick_breaking(beta):
    portion_remaining = pt.concatenate([[1], pt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

def fit_model(X_train_df, scaler):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled_X_train_df = pd.DataFrame(scaler.fit_transform(X_train_df), columns = X_train_df.columns)    
    
    pca = PCA(n_components = 1)
    X_train_df_reduced = pca.fit_transform(scaled_X_train_df)
    X_train_df_reduced = scaler.fit_transform(X_train_df_reduced).squeeze() 

    N = scaled_X_train_df.shape[0]
    
    with pm.Model(coords={"component": np.arange(K), "obs_id": np.arange(N)}) as model:
        alpha = pm.Gamma("alpha", ALPHA_SHAPE, ALPHA_RATE)
        beta = pm.Beta("beta", 1.0, alpha, dims="component")
        w = pm.Deterministic("w", stick_breaking(beta), dims="component")

        tau = pm.Gamma("tau", 1.0, 1.0, dims="component")
        lambda_ = pm.Gamma("lambda_", 10.0, 1.0, dims="component")
        mu = pm.Normal("mu", 0, tau=lambda_ * tau, dims="component")
        obs = pm.NormalMixture(
            "obs", w, mu, tau=lambda_ * tau, observed = X_train_df_reduced 
        )
    
        pgm = pm.model_to_graphviz(model = model)
        pgm_out = path_join("out", "model_graph.png")
        pgm.render(format = "png", directory = "out", filename = "model_graph")
        PIPELINE.log_artifact(pgm_out)
    
    with model:               
        approx = pm.fit(
            n = N_ADVI_SAMPLES, 
            method = VI_METHOD, 
            inf_kwargs= { "n_particles": N_PARTICLES },
            obj_optimizer = pm.adam(learning_rate = LR),
            random_seed = SEED
        )

        trace = approx.sample(POSTERIOR_DRAWS)

        PIPELINE.log_param("draws", POSTERIOR_DRAWS)
        PIPELINE.log_param("seed", SEED)
        PIPELINE.log_param("advi_samples", N_ADVI_SAMPLES)
        PIPELINE.log_param("n_particles", N_PARTICLES)
        PIPELINE.log_param("lr", LR)
        PIPELINE.log_param("vi_method", VI_METHOD)
        PIPELINE.log_param("alpha_rate", ALPHA_RATE)
        PIPELINE.log_param("alpha_shape", ALPHA_SHAPE)

        outfile = path_join("out", "num_components.png")
        plot_w = np.arange(K) + 1
        plt.figure()
        plt.bar(plot_w - 0.5, trace.posterior["w"].mean(("chain", "draw")), width=1.0, lw=0)
        plt.xlim(0.5, K)
        plt.xlabel("Component")
        plt.ylabel("Posterior expected mixture weight");
        plt.tight_layout()
        plt.savefig(outfile)
        PIPELINE.log_artifact(outfile)

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

        kwargs = {"figsize": (12, 12), "textsize": 5}
        __create_plot(trace, az.plot_posterior, "posterior", kwargs)

        outfile = path_join("out", "posterior_pdfs.png")

        xr_plot = xr.DataArray(np.sort(X_train_df_reduced))
        plt.figure()
        post_pdf_contribs = xr.apply_ufunc(
            sp.stats.norm.pdf,
            xr_plot,
            trace.posterior["mu"],
            1.0 / np.sqrt(trace.posterior["lambda_"] * trace.posterior["tau"]),
        )

        post_pdfs = (trace.posterior["w"] * post_pdf_contribs).sum(dim=("component"))
        post_pdf_expectation = post_pdfs.mean(dim=("chain", "draw")) 
        plt.hist(xr_plot, density=True, color="C0", lw=0, alpha=0.5)
        plt.plot(xr_plot, post_pdf_expectation, c="k", label="Posterior expected density")

        plt.tight_layout()
        plt.savefig(outfile)
        PIPELINE.log_artifact(outfile)

def main():
    X_train, Y_train, X_test, Y_test, scaler, X_train_df, X_test_df = get_grape_data(
        DATA, GRAPEVINE_PARAMS_TRAINING, GRAPEVINE_PARAMS_TESTING, GRAPEVINE_DATA_TRAINING, 
        GRAPEVINE_DATA_TESTING, GRAPEVINE_RESP
    )
    
    fit_model(X_train_df, scaler)


    config_out = path_join("out", CONFIG_FILE)
    CONFIG.export(config_out)
    PIPELINE.log_artifact(config_out)

if __name__ == "__main__":
    main()