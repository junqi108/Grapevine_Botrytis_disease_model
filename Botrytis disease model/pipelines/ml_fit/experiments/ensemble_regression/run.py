##############################################
# Imports

import os
from os.path import join as path_join
from matplotlib import pyplot as plt
import sys

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

from ml_fit_lib.config import get_config
from ml_fit_lib.constants import *
from ml_fit_lib.data import Data
from ml_fit_lib.experiment import (
    setup_experiment, regression_experiment, 
    get_conformal_regressor, get_prediction_intervals)
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

    setup_experiment(CONFIG, df)
    
    final_model = regression_experiment(CONFIG, RUN_DISTRIBUTED, tmp_dir)

    conformal_model = get_conformal_regressor(final_model, df , CONFIG)
    for alpha in [0.01, 0.05, 0.1]:
        preds, lb, ub = get_prediction_intervals(conformal_model, df, CONFIG, alpha = alpha)
        PIPELINE.log_prediction_intervals(
            lb, ub, target = CONFIG.get("target"), alpha = alpha
        )

    actual = df[CONFIG.get("target")].values
    PIPELINE.log_predictions(preds, actual, target = CONFIG.get("target"))
      
    config_out = path_join(tmp_dir, CONFIG_FILE)
    CONFIG.export(config_out)
    PIPELINE.log_artifact(config_out)

if __name__ == "__main__":
    main()