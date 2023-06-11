"""
Machine Learning Pipeline Configuration

A library of configuration tools for the machine learning pipeline.
"""

##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import os
from os.path import join as path_join
import yaml

from typing import Any

##########################################################################################################
### Library  
##########################################################################################################

ENV_VAR_WHITE_LIST = ["ML_PIPELINE_", "MLFLOW_", "RAY_"]

class Config:
    """Pipeline configuration class"""

    def __init__(self):
        self.config = {}
        self.params_override = {} 
        self.env_var_prefixes = ENV_VAR_WHITE_LIST

    def get(self, k: str, export: bool = True) -> Any:
        """Get a configuration value."""
        v = self.config.get(k)
        if export:
            self.params_override[k] = v
        return v

    def get_as(self, k: str, v_type: type, export: bool = True) -> Any:
        """Get and cast a configuration value."""
        v = v_type(self.config.get(k))
        if export:
            self.params_override[k] = v
        return v

    def set(self, k: str, v: Any, export: bool = True):
        """Set a configuration value."""
        self.config[k] = v
        if export:
            self.params_override[k] = v
        return self

    def from_env(self):
        """Add configuration values from environment variables."""
        environment_vars = dict(os.environ)
        for k, v in environment_vars.items():
            for prefix in self.env_var_prefixes:
                if k.startswith(prefix): 
                    self.config[k] = v
        return self
            
    def from_yaml(self, path: str):
        """Add configuration values from a YAML file."""
        with open(path) as f:
            options: dict = yaml.safe_load(f)
            if options is not None:
                self.extract_key_values(options)
        return self

    def to_yaml(self, config: dict, outfile: str, default_flow_style: bool = False):
        """Export a dictionary to a YAML file."""
        with open(outfile, 'w') as f:
            yaml.dump(config, f, default_flow_style = default_flow_style)
        return self

    def export(self, path: str = "", default_flow_style: bool = False) -> str:
        """Export active configuration to a params.override.yaml file."""
        config_path = path_join(path)
        self.to_yaml(self.params_override, config_path, default_flow_style)
        return config_path

    def from_parser(self, parser: argparse.ArgumentParser):
        """Add configuration values from an argument parser."""
        args = parser.parse_args()
        arg_dict = vars(args)
        return self.extract_key_values(arg_dict)

    def extract_key_values(self, options: dict):
        """Extract keys and values from a dictionary, and add them to the configuration."""
        for k, v in options.items():
            self.config[k] = v
        return self
    
def get_config(config_file: str) -> Config:
    config = Config()
    config.from_env()
    config.from_yaml(config_file)
    return config
