import dask.dataframe as dd
import numpy as np
import pandas as pd

from easyexplore.utils import INVALID_VALUES
from typing import List


class MissingDataAnalysisException(Exception):
    """
    Class for handling exceptions for class MissingDataAnalysis
    """
    pass


class MissingDataAnalysis:
    """
    Class for analyzing missing data
    """
    def __init__(self,
                 df,
                 features: List[str] = None,
                 other_mis: list = None,
                 conv_invalid: bool = True,
                 percentages: bool = False,
                 digits: int = 2,
                 **kwargs
                 ):
        """
        :param df: Pandas or dask DataFrame
            Data set

        :param features: List[str]
            Name of the features

        :param other_mis: list
            Valid values which should be interpreted as missing values

        :param conv_invalid: bool
            Convert all invalid data into missing values

        :param percentages: bool
            Generate percentages instead of counters

        :param digits: int
            Number of decimals to round

        :param kwarg: dict
            Key-word arguments
        """
        self.partitions: int = 4 if kwargs.get('partitions') is None else kwargs.get('partitions')
        if isinstance(df, pd.DataFrame):
            self.df: dd.DataFrame = dd.from_pandas(data=df, npartitions=self.partitions)
        elif isinstance(df, dd.DataFrame):
            self.df: dd.DataFrame = df
        else:
            raise MissingDataAnalysisException('Format of data set ({}) not supported. Use Pandas or dask DataFrame instead'.format(type(df)))
        self.features: str = self.df.columns if features is None else features
        if len(self.features) == 0:
            raise MissingDataAnalysisException('No features found. Please check your parameter config')
        if other_mis is not None:
            if len(other_mis) > 0:
                self.df = self.df.replace(to_replace=other_mis, value=np.nan, regex=False)
        if conv_invalid:
            self.df = self.df.replace(to_replace=INVALID_VALUES, value=np.nan, regex=False)
        self.percentages: bool = percentages
        self.round: float = 2 if digits < 0 else digits

    def clean_nan(self) -> dd.DataFrame:
        """
        Clean all missing values from data set

        :return: pd.DataFrame
            Data set containing valid values only
        """
        return self.df.loc[~self.df.isnull().any(axis=1), self.features]

    def freq_nan_by_cases(self) -> dict:
        """
        Frequency of missing values by cases

        :return: dict:
            Frequency distribution of missing or invalid data value case-wise
        """
        if self.percentages:
            return self.df[self.features].isnull().astype(dtype=int).compute().apply(func=lambda x: sum(x) / len(self.features), axis=1).to_dict()
        else:
            return self.df[self.features].isnull().astype(dtype=int).compute().apply(func=lambda x: sum(x), axis=1).to_dict()

    def freq_nan_by_features(self) -> dict:
        """
        Frequency of missing values by features

        :return: dict:
            Frequency distribution of missing or invalid data value feature-wise
        """
        if self.percentages:
            return (self.df[self.features].isnull().sum().compute() / len(self.df)).to_dict()
        else:
            return self.df[self.features].isnull().sum().compute().to_dict()

    def get_nan_idx_by_cases(self) -> dict:
        """
        Get index of missing value by cases

        :return: dict:
            Missing or invalid data index value case-wise
        """
        _nan_idx_by_cases: dict = {}
        for case in range(0, len(self.df), 1):
            if len(self.df.shape) == 1:
                _nan_idx_by_cases.update({case: np.where(pd.isnull(self.df[self.features][case].compute()))[0].tolist()})
            else:
                _nan_idx_by_cases.update({case: np.where(pd.isnull(self.df.loc[case, self.features].values.compute()))[1].tolist()})
        return _nan_idx_by_cases

    def get_nan_idx_by_features(self) -> dict:
        """
        Get index of missing value by feature

        :return: dict:
            Missing or invalid data index value feature-wise
        """
        _nan_idx_by_features: dict = {}
        for feature in self.features:
            if len(self.df.shape) == 1:
                _nan_idx_by_features.update({feature: np.where(pd.isnull(self.df[feature].compute()))[0].tolist()})
            else:
                _nan_idx_by_features.update({feature: np.where(self.df[feature].isnull().compute())[0].tolist()})
        return _nan_idx_by_features

    def has_nan(self) -> bool:
        """
        Check whether given DataFrame contains missing values

        :return: bool:
            Has missing / invalid data
        """
        return self.df[self.features].isnull().astype(int).sum().sum().compute() > 0
