"""
Machine Learning Pipeline Experiment

A library for performing experiments within the machine learning pipeline.
"""

##########################################################################################################
### Imports  
##########################################################################################################

# External
import mlflow
from pycaret.regression import (
    setup, create_model, compare_models, tune_model, finalize_model,
    ensemble_model, blend_models, stack_models, automl, predict_model,
    interpret_model, get_config
)

from mapie.regression import MapieRegressor
from mapie.metrics import regression_coverage_score

##########################################################################################################
### Library  
##########################################################################################################

def _to_experiment_setup_kwargs(config):
    kwargs = {}
    args = ["target", "train_size", "categorical_features", "ignore_features", "preprocess", 
            "imputation_type", "numeric_imputation", "iterative_imputation_iters", "low_variance_threshold",
            "normalize", "normalize_method", "data_split_shuffle", "data_split_stratify", "fold_strategy", 
            "fold", "fold_shuffle", "fold_groups", "n_jobs", "use_gpu", "log_experiment", "log_plots", 
            "log_profile", "log_data", "profile"]

    for arg in args:
        kwargs[arg] = config.get(arg)
    return kwargs

def setup_experiment(config, data):
    kwargs = _to_experiment_setup_kwargs(config)

    setup(
        data, html = False, session_id = config.get("random_seed"), 
        experiment_name = config.get("MLFLOW_EXPERIMENT_NAME"),
        **kwargs
    )

def get_search_config(config, run_distributed):
    if run_distributed:
        search_algorithm = config.get("distributed_search_algorithm")
        search_library = config.get("distributed_search_library")
    else:
        search_algorithm = config.get("search_algorithm")
        search_library = config.get("search_library")
        
    return search_algorithm, search_library

def save_model(final_model, config):
    experiment_name = config.get("MLFLOW_EXPERIMENT_NAME")
    model_info = mlflow.sklearn.log_model(final_model, artifact_path = experiment_name)
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/{experiment_name}" 
    model_version = mlflow.register_model(model_uri, experiment_name)

    model_description = config.get("model_description")
    if model_description is not None:
        author = config.get("author")
        if author is not None:
            model_description += f" - {author}"
        client = mlflow.tracking.MlflowClient()
        client.update_model_version(experiment_name, model_version.version, model_description)

def regression_experiment(config, run_distributed, tmp_dir):
    evaluation_metric = config.get("target_metric")
    search_algorithm, search_library = get_search_config(config, run_distributed)
    n_iter = config.get("n_iter")
    n_estimators = config.get("n_estimators")
    fold = config.get("fold")
    ensemble_methods = config.get_as("ensemble_methods", set)
    
    top_models = compare_models(
        cross_validation = True, fold = fold, sort = evaluation_metric,
        n_select = config.get("n_select"), turbo = config.get("turbo"), groups = config.get("fold_groups"),
        errors = "ignore"
    )

    tuned_top = [ 
        tune_model(model, search_algorithm = search_algorithm, optimize = evaluation_metric,
            search_library = search_library, n_iter = n_iter, early_stopping = config.get("early_stopping"), 
            early_stopping_max_iters = config.get("early_stopping_max_iters"), choose_better = True) 
        for model in top_models
    ]

    if "boosting" in ensemble_methods:
        boosting_ensemble = ensemble_model(tuned_top[0], method = "Boosting", optimize = evaluation_metric, 
                choose_better = True, n_estimators = n_estimators)

    if "bagging" in ensemble_methods:   
        bagging_ensemble = ensemble_model(tuned_top[0], method = "Bagging", optimize = evaluation_metric, 
                choose_better = True, n_estimators = n_estimators)

    if "blending" in ensemble_methods:   
        blended_ensemble = blend_models(tuned_top, fold = fold, choose_better = True, optimize = evaluation_metric)

    if "stacking" in ensemble_methods:   
        meta_model = create_model(config.get("meta_model"))
        stacking_ensemble = stack_models(
            tuned_top, optimize = evaluation_metric, meta_model = meta_model, 
            choose_better = True, meta_model_fold = fold
        )
    
    best_model = automl(optimize = evaluation_metric, use_holdout = False, turbo = True)   

    for plot in ["summary", "correlation", "reason", "pdp", "msa"]:
        interpret_model(best_model, plot = plot, save = tmp_dir)
        mlflow.log_artifacts(tmp_dir)

    final_model = finalize_model(best_model)
    save_model(final_model, config)
    return final_model

def get_prediction_intervals(model, data, config):
    df = data.drop(columns = config.get("ignore_features"))
    target = config.get("target")
    X = df.drop(columns = target)
    y = df[target]

    mapie_model = MapieRegressor(model, method = "minmax", cv = config.get("fold"))
    mapie_model.fit(X, y)

    preds, intervals = mapie_model.predict(X, alpha = config.get("prediction_interval"))
    lb = intervals[:, 0, 0].ravel()
    ub = intervals[:, 1, 0].ravel()
    return preds, lb, ub

def reg_predict_model(model, df):
    preds = predict_model(model, data = df)
    return preds["prediction_label"].values