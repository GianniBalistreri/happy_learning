import dask.dataframe as dd
import pandas as pd
import random

from dask_ml.model_selection import KFold, train_test_split
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Union


class MLSamplerException(Exception):
    """
    Class for handling exceptions for class MLSampler
    """
    pass


class MLSampler:
    """
    Class for data sampling specialized for supervised machine learning
    """
    def __init__(self,
                 df: Union[dd.DataFrame, pd.DataFrame],
                 target: str,
                 features: List[str] = None,
                 train_size: float = 0.8,
                 random_sample: bool = True,
                 stratification: bool = False,
                 seed: int = 1234,
                 **kwargs
                 ):
        """
        :param df: Pandas or dask DataFrame
            Data set for sampling

        :param target: str
            Name of the target feature

        :param features: List[str]
            Name of the features

        :param train_size: float
            Size of the training set

        :param random_sample: bool
            Whether to sample randomly or not

        :param stratification: bool
            Whether to stratify data set or not

        :param seed: int
            Seed value

        :param kwargs: dict
            Key-word arguments for handling dask parameter configuration
        """
        self.seed: int = seed if seed >= 0 else 1234
        if isinstance(df, pd.DataFrame):
            self.df: dd.DataFrame = dd.from_pandas(data=df,
                                                   npartitions=4 if kwargs.get('partitions') is None else kwargs.get(
                                                       'partitions'))
        elif isinstance(df, dd.DataFrame):
            self.df: dd.DataFrame = df
        else:
            raise MLSamplerException(
                'Format of data set ({}) not supported. Use Pandas or dask DataFrame instead'.format(type(df)))
        if target not in self.df.columns:
            raise MLSamplerException('Target variable ({}) not found in data set'.format(target))
        self.target = target
        self.features: List[str] = list(self.df.columns) if (features is None or len(features) == 0) else features
        if self.target in self.features:
            del self.features[self.features.index(self.target)]
        for ft in self.features:
            if ft not in self.df.columns:
                raise MLSamplerException('Feature ({}) not found in data set'.format(ft))
        self.train_size: float = train_size
        self.test_size: float = 1 - self.train_size
        self.random_sample: bool = random_sample
        self.stratification: bool = stratification

    def train_test_sampling(self, validation_split: float = 0.1) -> dict:
        """
        Data sampling into train & test data

        :param validation_split: float
            Amount of training data to validate quality during training

        :return dict:
            Train and test split for both target and predictors
        """
        #if self.stratification:
        #    _stratification: np.array = self.df[self.target].values
        #else:
        #    _stratification = None
        _x_train, _x_test, _y_train, _y_test = train_test_split(self.df[self.features],
                                                                self.df[self.target],
                                                                test_size=self.test_size,
                                                                train_size=self.train_size,
                                                                random_state=self.seed,
                                                                shuffle=self.random_sample,
                                                                #stratify=_stratification
                                                                )
        if validation_split > 0:
            _x_train_, _x_val, _y_train_, _y_val = train_test_split(_x_train,
                                                                    _y_train,
                                                                    test_size=validation_split,
                                                                    train_size=1-validation_split,
                                                                    random_state=self.seed,
                                                                    shuffle=self.random_sample
                                                                    )
        else:
            _x_train_ = _x_train
            del _x_train
            _x_val = None
            _y_train_ = _y_train
            del _y_train
            _y_val = None
            return dict(x_train=_x_train_.compute(),
                        x_test=_x_test.compute(),
                        y_train=_y_train_.compute(),
                        y_test=_y_test.compute(),
                        x_val=_x_val,
                        y_val=_y_val
                        )
        return dict(x_train=_x_train_.compute(),
                    x_test=_x_test.compute(),
                    y_train=_y_train_.compute(),
                    y_test=_y_test.compute(),
                    x_val=_x_val.compute(),
                    y_val=_y_val.compute()
                    )

    def k_fold_cross_validation(self, k: int = 5) -> dict:
        """
        K-Fold Cross-validation

        :param k: int
            Number of k-folds

        :return dict:
            Cross-validated train and test split for both target and predictors
        """
        _kfold_sample: dict = {}
        _k: int = k if k > 0 else 5
        if self.stratification:
            _kfold: StratifiedKFold = StratifiedKFold(n_splits=_k, shuffle=self.random_sample, random_state=self.seed)
        else:
            _kfold: KFold = KFold(n_splits=_k, shuffle=self.random_sample, random_state=self.seed)
        for train, test in _kfold.split(X=self.df.loc[:, self.features], y=self.df.loc[:, self.target], groups=None):
            _kfold_sample.update({'x_train_k{}'.format(len(_kfold_sample.keys()) + 1): train,
                                  'x_test_k{}'.format(len(_kfold_sample.keys()) + 1): test
                                  })
        return _kfold_sample

    def time_series(self):
        pass


class SamplerException(Exception):
    """
    Class for handling exceptions for class Sampler
    """
    pass


class Sampler:
    """
    Class for general sampling purposes
    """
    def __init__(self, df, size: int = None, prop: float = None, **kwargs):
        """
        :param df: Pandas or dask DataFrame
            Data set

        :param size: int
            Sample size

        :param prop: float
            Sample proportion

        :param kwargs: dict
            Key-word arguments for handling dask parameter settings
        """
        if isinstance(df, pd.DataFrame):
            self.df: dd.DataFrame = dd.from_pandas(data=df, npartitions=4 if kwargs.get('partitions') is None else kwargs.get('partitions'))
        elif isinstance(df, dd.DataFrame):
            self.df: dd.DataFrame = df
        else:
            raise SamplerException('Format of data set ({}) not supported. Use Pandas or dask DataFrame instead'.format(type(df)))
        self.size: int = size
        self.prop: float = prop
        if size is None:
            if prop is None:
                raise SamplerException('Neither size nor proportion found')
            self.k: int = int(len(self.df) * prop)
        else:
            self.k: int = size

    def random(self) -> dd.DataFrame:
        """
        Draw random sampling

        :return dask DataFrame:
            Sampled data set
        """
        _sample_idx: list = random.sample(population=list(self.df.index.values.compute()), k=self.k)
        return self.df.loc[_sample_idx, :]

    def quota(self, features: List[str], quotas: Dict[str, Dict[str, float]] = None) -> dd.DataFrame:
        """
        Draw quota sampling

        :param features: List[str]
            List of string containing the name of the features

        :param quotas: Dict[str, Dict[str, float]]
            Dictionary containing the quotas for each given feature

        :return dask DataFrame:
            Sampled data set
        """
        _target_idx: List[int] = []
        for ft in features:
            if quotas is None:
                _counts: pd.Series = self.df[ft].compute().value_counts(normalize=True)
            else:
                if ft not in quotas.keys():
                    raise SamplerException('Feature not found in user defined quotas settings')
                _counts: pd.Series = pd.Series(data=quotas.get(ft))
            _size: List[int] = [int(round(s, ndigits=0)) for s in _counts.values * self.k]
            for i, val in enumerate(_counts.index.values):
                _sample_idx: list = random.sample(population=self.df.loc[self.df[ft] == val, ft].index.get_values().tolist(), k=_size[i])
                _target_idx = _target_idx + _sample_idx
        return self.df.loc[list(set(_target_idx)), :]
