"""
Machine Learning Pipeline Data

A library for data loading and manipulation within the machine learning pipeline.
"""

##########################################################################################################
### Imports  
##########################################################################################################

# External
import os
import pandas as pd

##########################################################################################################
### Library  
##########################################################################################################

class Data:
    def read_csv(self, name: str, **kwargs) -> pd.DataFrame:
        """
        Wrapper method to load a csv file.

        Parameters
        --------------
        name: str
            The file name.
        **kwargs
            Additional arguments.

        Returns
        ---------
        df: DataFrame
            The dataframe.
        """
        return pd.read_csv(name, **kwargs)

    def train_test_split(self, df: pd.DataFrame, frac: float = 0.9, random_state: int = None) -> tuple:
        """
        Perform a train-test split.

        Parameters
        --------------
        df: DataFrame
            The dataframe.
        frac: float
            The fraction of training data.
        random_state: int
            The random seed.

        Returns
        ---------
        dfs: (DataFrame, DataFrame)
            The training and testing dataframes.
        """
        data = df.sample(frac = frac, random_state = random_state)
        data_unseen = df.drop(data.index)
        data.reset_index(drop = True, inplace = True)
        data_unseen.reset_index(drop = True, inplace = True)
        return data, data_unseen

    def query(self, df_query: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Wrapper method for performing dataframe queries.

        Parameters
        --------------
        df_query: str
            The dataframe query.
        df: DataFrame
            The dataframe.

        Returns
        ---------
        df: DataFrame
            The queried dataframe.
        """
        return df.query(df_query).reset_index()
    