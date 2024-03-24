##############################################
# Imports

import os
from os.path import join as path_join
from pycaret.time_series import (
    setup, compare_models, tune_model, blend_models, plot_model, finalize_model,
    predict_model, create_model
)
from matplotlib import pyplot as plt
import sys

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

from ml_fit_lib.config import get_config
from ml_fit_lib.constants import *
from ml_fit_lib.data import Data
from ml_fit_lib.mlops import MLFlowPipeline
from ml_fit_lib.experiment import save_model

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
    df = DATA.read_csv(GRAPEVINE_DATA_TESTING)
    df = df[GRAPEVINE_RESP]
    target = GRAPEVINE_RESP[2]
    ts = df[target].squeeze()

    kwargs = {}
    for k in ["log_profile", "log_data", "profile", "log_experiment", "log_plots", "use_gpu", "n_jobs", 
              "fh", "fold", 
        # "numeric_imputation_target", "numeric_imputation_exogenous",  
        # "seasonal_period", "ignore_seasonality_test", "sp_detection", "max_sp_to_consider",
        # "remove_harmonics", "harmonic_order_method", "num_sps_to_use", "enforce_exogenous",
        "scale_target", "transform_target"]:
       kwargs[k] = CONFIG.get(k)

    exp_setup = setup(
        ts, html = False, session_id = CONFIG.get("random_seed"), 
        experiment_name = CONFIG.get("MLFLOW_EXPERIMENT_NAME"), target = target,
        fold_strategy = CONFIG.get("fold_strategy_ts"), **kwargs
    )

    # top_models = compare_models(
    #     include = ["arima", "exp_smooth", "ets", "auto_arima", "theta", "croston", "grand_means"],
    #     n_select = CONFIG.get("n_select"), fold = CONFIG.get("fold"),
    #     cross_validation = True, sort = CONFIG.get("target_metric_ts"), 
    #     turbo = CONFIG.get("turbo")
    # )
    
    # top_models = [ 
    #     create_model(model) for model in 
    #     ["arima", "exp_smooth", "ets", "theta", "croston", "polytrend"] 
    # ]
 
    # tuned_top = [tune_model(
    #     model, fold = CONFIG.get("fold"), n_iter = CONFIG.get("n_iter"),
    #     optimize = CONFIG.get("target_metric_ts"), search_algorithm = CONFIG.get("search_algorithm"),
    #     choose_better = True
    # ) for model in top_models ]

    # blended_model = blend_models(
    #     top_models, method = "median", fold = CONFIG.get("fold"),
    #     choose_better = False, 
    # )

    blended_model = create_model("arima")
    for plot in [
        "ts", "forecast"
    ]:
        out_file = plot_model(blended_model, plot = plot, save = tmp_dir)
    PIPELINE.log_artifacts(tmp_dir)

    final_model = finalize_model(blended_model)
    save_model(final_model, CONFIG) 
         
    config_out = path_join(tmp_dir, CONFIG_FILE)
    CONFIG.export(config_out)
    PIPELINE.log_artifact(config_out)

if __name__ == "__main__":
    main()
