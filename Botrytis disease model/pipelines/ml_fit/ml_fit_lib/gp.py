"""
Machine Learning Pipeline Experiment

A library for fitting Gaussian Processes machine learning pipeline.
"""

##########################################################################################################
### Imports  
##########################################################################################################

# External
import gpytorch
import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

##########################################################################################################
### Library  
##########################################################################################################

def plot_results(res_indx, ax1, ax2, test_x, test_y, mean, lower, upper, resp_list):
    ax1.scatter(test_x[:, -1], test_y[:, res_indx], s = .1)
    ax1.set_title(f"{resp_list[res_indx]} Simulated")
    ax2.scatter(test_x[:, -1], mean[:, res_indx], s = .1)
    ax2.fill_between(test_x[:, -1], lower[:, res_indx], upper[:, res_indx], alpha=0.5)
    ax2.set_title(f"{resp_list[res_indx]} Emulated")

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_gp_scores(predictions, test_y, gp_scores_file):
    pd.DataFrame({
        "test_mse": np.array2string(gpytorch.metrics.mean_squared_error(predictions, test_y).cpu().numpy()),
        "test_nlpd": gpytorch.metrics.negative_log_predictive_density(predictions, test_y).cpu().detach().numpy(),
        "test_coverage_error_95": np.array2string(gpytorch.metrics.quantile_coverage_error(predictions, test_y).cpu().detach().numpy())
    }, 
    index = [0]
    ).to_csv(gp_scores_file, index = False)
        
class MultitaskVariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, n_col, num_latents, num_tasks):
        inducing_points = torch.rand(num_latents, n_col, n_col)

        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations = True
            ),
            num_tasks = num_tasks,
            num_latents = num_latents,
            latent_dim = -1
         )
        
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.MaternKernel(nu = 2.5, batch_shape=torch.Size([num_latents]), ard_num_dims = n_col)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class FeatureExtractor(torch.nn.Sequential):
    def __init__(self, n_col, num_tasks):
        super(FeatureExtractor, self).__init__()
        self.add_module("main", torch.nn.Sequential(
            torch.nn.Linear(n_col, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.Linear(8, num_tasks)
        ))

class DeepMultitaskVariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, n_col, num_latents, num_tasks):
        inducing_points = torch.rand(num_latents, n_col, n_col)

        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations = True
            ),
            num_tasks = num_tasks,
            num_latents = num_latents,
            latent_dim = -1
         )
        
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu = 2.5, batch_shape=torch.Size([num_latents]), ard_num_dims = n_col),
            batch_shape=torch.Size([num_latents]), ard_num_dims = n_col
        )

        self.feature_extractor = FeatureExtractor(n_col, n_col)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
        
    def forward(self, x):
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)   
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def log_predictive_dist(test_y, mean, lower, upper, resp_list, log_interval = 0.05):
    for row, col in np.ndindex(test_y.shape):
        if row % int(test_y.shape[0] * log_interval) != 0:
            continue
        
        mlflow.log_metric(f"Simulated {resp_list[col]}", test_y[row, col], step = row)
        mlflow.log_metric(f"Emulated Mean {resp_list[col]}", mean[row, col], step = row)
        mlflow.log_metric(f"Emulated Lower {resp_list[col]}", lower[row, col], step = row)
        mlflow.log_metric(f"Emulated Upper {resp_list[col]}", upper[row, col], step = row)

def add_time_step(df):
    df["time_step"] = df.reset_index().index + 1
    return df

def get_grape_data(
        data_obj, grapevine_params_train, grapevine_params_test, grapevine_data_train, grapevine_data_test,
        grapevine_resp, sample_frac = 0.6, random_state = 100
    ):
    X_train = data_obj.read_csv(grapevine_params_train).drop(columns = "factor") 
    Y_train = data_obj.read_csv(grapevine_data_train).drop(columns = "factor") 
    X_test = data_obj.read_csv(grapevine_params_test).drop(columns = "factor")  
    Y_test = data_obj.read_csv(grapevine_data_test).drop(columns = "factor")  

    uuids = pd.Series(X_train.uuid.unique()).sample(frac = sample_frac, random_state = random_state)
    X_train = X_train.query("uuid in @uuids")

    X_train["time_step"] = None
    X_cols = X_train.columns
    train = (
        pd.merge(X_train, Y_train, on = "uuid")
        .groupby("uuid", group_keys = False).apply(add_time_step)
    )

    test = (
        pd.merge(X_test, Y_test, on = "uuid")
        .groupby("uuid", group_keys = False).apply(add_time_step)
    )

    scaler = MinMaxScaler()
    X_train = train[X_cols].drop(columns = "uuid")
    scaler.fit(X_train)
    Y_train = train[grapevine_resp]
    X_test = test[X_cols].drop(columns = "uuid")
    Y_test = test[grapevine_resp]

    device = get_device()
    X_train_torch = torch.from_numpy(scaler.transform(X_train)).float().to(device)
    X_test_torch = torch.from_numpy(scaler.transform(X_test)).float().to(device)
    Y_train = torch.from_numpy(Y_train.values).float().to(device)
    Y_test = torch.from_numpy(Y_test.values).float().to(device)

    return X_train_torch, Y_train, X_test_torch, Y_test, scaler, X_train, X_test

def load_gp(X_train, Y_train, num_latents, gp_model_fit):
    device = get_device()
    model = MultitaskVariationalGPModel(n_col = X_train.shape[-1], num_latents = num_latents, num_tasks = Y_train.shape[-1])
    state_dict = torch.load(gp_model_fit)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks = Y_train.shape[-1])
    likelihood.eval().to(device)

    return model, likelihood