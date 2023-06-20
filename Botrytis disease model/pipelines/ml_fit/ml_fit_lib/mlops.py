"""
Machine Learning Pipeline MLOps

A library of MLOps tools within the machine learning pipeline.
"""

##########################################################################################################
### Imports
##########################################################################################################

# External
import mlflow
import tempfile

##########################################################################################################
### Library
##########################################################################################################

class MLFlowPipeline:
    """
    The MLFlow pipeline.
    """

    def init(self, config, prefix = '') -> str:
        """
        Initialise the MLFlow run.

        Args:
            config (Config): 
                The configuration object.
            prefix (str):
                The pipeline prefix.
                
        Returns:
            str: 
                The temporary directory name.
        """
        mlflow.set_tracking_uri(config.get("MLFLOW_TRACKING_URI")) # Enable tracking using MLFlow
        experiment_name = config.get("MLFLOW_EXPERIMENT_NAME")

        existing_exp = mlflow.get_experiment_by_name(experiment_name)
        if not existing_exp:
            mlflow.create_experiment(experiment_name)

        mlflow.set_tag("experiment_name", experiment_name)
        self.tmp_dir = tempfile.TemporaryDirectory()
        return self.tmp_dir.name

    def log_artifact(self, artifact_path: str) -> None:
        """
        Log the MLFlow artifact.

        Args:
            artifact_path (str): 
                The artifact file path.
        """
        mlflow.log_artifact(artifact_path)

    def log_artifacts(self, artifact_path: str) -> None:
        """
        Log the MLFlow artifact.

        Args:
            artifact_path (str): 
                The artifact file path.
        """
        mlflow.log_artifacts(artifact_path)

    def log_metric(self, name: str, metric, **kwargs) -> None:
        """
        Log the MLFlow metric.

        Args:
            name (str): 
                The metric name.
            metric (Any): 
                The metric value.
        """
        mlflow.log_metric(name, metric, **kwargs)

    def log_param(self, name: str, param, **kwargs) -> None:
        """
        Log the MLFlow parameter.

        Args:
            name (str): 
                The parameter name.
            metric (Any): 
                The parameter value.
        """
        mlflow.log_param(name, param, **kwargs)

    def end_run(self, config) -> None:
        """
        End the current MLFlow run.

        Args:
            config (Config): 
                The configuration object.
        """
        mlflow.end_run()
        self.tmp_dir.cleanup()    

    def log_series(self, series, name, interval = 0.01):
        series_interval = int(len(series) * interval)
        
        for i, value in enumerate(series):
            if i % series_interval != 0:
                continue
            self.log_metric(name, value, step = i)

    def log_predictions(self, preds, actual, target, interval = 0.01):
        pred_interval = int(len(preds) * interval)
        
        for i, pred in enumerate(preds):
            if i % pred_interval != 0:
                continue
            self.log_metric(f"Predicted {target}", pred, step = i)
            self.log_metric(f"Actual {target}", actual[i], step = i)

    def log_prediction_intervals(self, lb, ub, target, alpha, interval = 0.01):
        pred_interval = int(len(lb) * interval)
        pred_interval_percent = (1 - alpha) * 100

        for i, _ in enumerate(lb):
            if i % pred_interval != 0:
                continue
            self.log_metric(f"{pred_interval_percent} lower bound {target}", lb[i], step = i)
            self.log_metric(f"{pred_interval_percent} upper bound {target}", ub[i], step = i)