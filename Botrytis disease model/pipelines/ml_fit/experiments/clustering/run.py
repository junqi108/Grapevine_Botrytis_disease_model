##############################################
# Imports

import os
from os.path import join as path_join
from pycaret.clustering import (setup, create_model, plot_model)
from matplotlib import pyplot as plt
from pathlib import Path
import sys

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

from ml_fit_lib.config import get_config
from ml_fit_lib.constants import *
from ml_fit_lib.data import Data
from ml_fit_lib.mlops import MLFlowPipeline

##############################################
# Constants

if __name__ == "__main__":
    CONFIG = get_config(CONFIG_PATH)

    PIPELINE = MLFlowPipeline()
    RUN_DISTRIBUTED = CONFIG.get("run_distributed")

    DATA = Data()
    
##############################################
# Main

def main():
    tmp_dir = PIPELINE.init(CONFIG)
    df = DATA.read_csv(DATA_FILE)
    
    kwargs = {}
    for k in ["log_profile", "log_data", "profile", "log_experiment", "log_plots", "use_gpu", "n_jobs", 
        "normalize_method", "normalize", "low_variance_threshold", "numeric_imputation", "imputation_type", 
        "ignore_features", "categorical_features"]:
       kwargs[k] = CONFIG.get(k)

    exp_setup = setup(
        df, preprocess = True, html = False, session_id = CONFIG.get("random_seed"), 
        experiment_name = CONFIG.get("MLFLOW_EXPERIMENT_NAME"), **kwargs
    )

    model = create_model(
        CONFIG.get("clustering_model"), num_clusters = CONFIG.get("num_clusters"), 
        ground_truth = CONFIG.get("clustering_ground_truth")
    )

    for plot in ["cluster", "tsne", "distribution"]:
        out_file = plot_model(model, plot = plot, save = tmp_dir)
        out_path = Path(out_file)
        if out_path.suffix == ".png":
            out_path = out_path.rename(out_path.with_suffix('.html'))
        PIPELINE.log_artifact(str(out_path))

    for plot in ["elbow", "silhouette", "distance"]:
        out_file = plot_model(model, plot = plot, save = tmp_dir)
        PIPELINE.log_artifact(out_file)

    config_out = path_join(tmp_dir, CONFIG_FILE)
    CONFIG.export(config_out)
    PIPELINE.log_artifact(config_out)

if __name__ == "__main__":
    main()