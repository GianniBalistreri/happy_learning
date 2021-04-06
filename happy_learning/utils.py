import dask
import geocoder
import inspect
import itertools
import json
import logging
import numpy as np
import pandas as pd
import re
import subprocess
import sys

from dask.distributed import Client
from easyexplore.data_import_export import DataExporter, FileUtils
from easyexplore.utils import Log, SPECIAL_CHARACTERS
from itertools import accumulate, islice
from operator import mul
from scipy.stats import anderson, chi2, chi2_contingency, f_oneway, friedmanchisquare, mannwhitneyu, normaltest, kendalltau,\
                        kruskal, kstest, pearsonr, powerlaw, shapiro, spearmanr, stats, ttest_ind, ttest_rel, wilcoxon
from statsmodels.stats.weightstats import ztest
from typing import Dict, List, Tuple

PERCENTILES: List[float] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
if sys.platform.find('win') >= 0:
    CLIP: str = 'clip'
elif sys.platform.find('linux') >= 0:
    CLIP: str = 'xclip'
elif sys.platform.find('darwin') >= 0:
    CLIP: str = 'pbcopy'


def get_methods_of_class(cls_obj) -> List[str]:
    """
    Get all method names of given class object

    :param cls_obj: object
        Class object

    :return List[str]:
        Method names
    """
    return [meth[0] for meth in inspect.getmembers(object=cls_obj, predicate=inspect.ismethod)]


def get_parameters_of_class(cls_obj) -> List[str]:
    """
    Get all parameter names of given class object

    :param cls_obj: object
        Class object

    :return List[str]:
        Parameter names
    """
    return [param[0] for param in inspect.getfullargspec(func=cls_obj)]


def read_from_clipboard():
    """
    Read stored information from clipboard

    :return
        Clipboard information
    """
    pass


def write_to_clipboard(clip_text: str) -> subprocess:
    """
    Write information to clipboard

    :param clip_text: str
        Text to write to clipboard

    :return: subprocess
        Clipboard information
    """
    return subprocess.check_call('echo {}|{}'.format(clip_text.strip(), CLIP), shell=True)


class HappyLearningUtilsException(Exception):
    """
    Class for setting up exceptions for class HappyLearningUtils
    """
    pass


class HappyLearningUtils:
    """
    Class for applying general utility methods
    """
    @staticmethod
    def bayesian_blocks(df: pd.Series) -> dict:
        """
        Optimal univariate binning using Bayesian Blocks

        :param df: pd.Series
            Data set

        :return dict
            Edges and labels (Bayesian Blocks)
        """
        _x: np.array = df.sort_values(axis=0, ascending=True, inplace=False).values
        _edges: np.ndarray = np.concatenate([_x[:1], 0.5 * (_x[1:] + _x[:-1]), _x[-1:]])
        _block_length = _x[-1] - _edges
        _ones_arr: np.array = np.ones(_x.size)
        _best: list = []
        _last: list = []
        for k in range(0, _x.size, 1):
            # Compute the width and count of the final bin for all possible
            # locations of the K^th changepoint
            _width = _block_length[:k + 1] - _block_length[k + 1]
            _count_vec = np.cumsum(_ones_arr[:k + 1][::-1])[::-1]

            # evaluate fitness function for these possibilities
            _fit_vec: np.array = _count_vec * (np.log(_count_vec) - np.log(_width))
            _fit_vec -= 4  # 4 comes from the prior on the number of changepoints
            _fit_vec[1:] += _best[:k]

            # find the max of the fitness: this is the K^th changepoint
            _last.append(np.argmax(_fit_vec))
            _best.append(_fit_vec[np.argmax(_fit_vec)])

        # -----------------------------------------------------------------
        # Recover changepoints by iteratively peeling off the last block
        # -----------------------------------------------------------------
        _change_points = np.zeros(_x.size, dtype=int)
        _i: int = _x.size
        _idx: int = _x.size
        while True:
            _i -= 1
            _change_points[_i] = _idx
            if _idx == 0:
                break
            _idx = _last[_idx - 1]
        _bayesian_blocks: dict = dict(edges=_edges[_change_points[_idx:]], labels=[])
        _bayesian_blocks['labels'] = [np.argmin(np.abs(val - _bayesian_blocks.get('edges'))) for val in _x.tolist()]
        return _bayesian_blocks

    @staticmethod
    def cat_array(array_with_nan: np.array) -> np.array:
        """
        Convert categorical float typed Numpy array into integer typed array

        :param array_with_nan: np.array
            containing the categorical data with missing values (float typed)

        :return: np.array
            Categorical data as integer values without missing values
        """
        return np.array(array_with_nan[~pd.isnull(array_with_nan)], dtype=np.int_)

    @staticmethod
    def dask_setup(client_name: str,
                   client_address: str = None,
                   mode: str = 'threads',
                   show_warnings: bool = False,
                   **kwargs) -> Client:
        """
        Setup dask framework for parallel computation

        :param client_name: str
            Name of the client

        :param client_address: str
            Address end point

        :param mode: str
            Parallel computation mode:
                threads: Multi-Threading
                processes: Multi-Processing
                single-threaded: Single thread and process

        :param show_warnings: bool
            Print warnings or just errors

        :param kwargs: dict
            Key-word arguments for dask client implementation

        :return: Client
            Initialized dask client object
        """
        if mode == 'threads':
            dask.config.set(scheduler='threads')
        elif mode == 'processes':
            dask.config.set(scheduler='processes')
        else:
            dask.config.set(scheduler='single-threaded')
        if kwargs.get('memory_limit') is None:
            kwargs.update({'memory_limit': 'auto'})
        return Client(address=client_address,
                      loop=kwargs.get('loop'),
                      timeout=kwargs.get('timeout'),
                      set_as_default=True,
                      scheduler_file=kwargs.get('scheduler_file'),
                      security=kwargs.get('security'),
                      asynchronous=kwargs.get('asynchronous'),
                      name=client_name,
                      heartbeat_interval=kwargs.get('heartbeat_interval'),
                      serializers=kwargs.get('serializers'),
                      deserializers=kwargs.get('deserializers'),
                      direct_to_workers=kwargs.get('direct_to_workers'),
                      connection_limit=512 if kwargs.get('connection_limit') is None else kwargs.get('connection_limit'),
                      processes=False if kwargs.get('processes') is None else kwargs.get('processes'),
                      silence_logs=logging.WARNING if show_warnings else logging.ERROR,
                      **kwargs
                      )

    @staticmethod
    def extract_tuple_el_in_list(list_of_tuples: List[tuple], tuple_pos: int) -> list:
        """
        Extract specific tuple elements from list of tuples

        :param list_of_tuples: List[tuple]
            List of tuples

        :param tuple_pos: int
            Position of element in tuples to extract

        :return: list
            List of elements of tuple
        """
        if tuple_pos < 0:
            raise HappyLearningUtilsException('Position of element in tuple cannot be negative ({})'.format(tuple_pos))
        return next(islice(zip(*list_of_tuples), tuple_pos, None))

    @staticmethod
    def generate_time_series_arrays(data_set: np.ndarray, lag: int = 1, train: bool = True) -> dict:
        """
        Generate n-dimensional numpy arrays format for LSTM and Convolutional neural networks especially

        :param data_set: np.ndarray
            Data set as n-dimensional numpy array

        :param lag: int
            Number of previous values to look back

        :param train: bool
            Generate train data set or test data set

        :return: dict
            Time series train and test data sets
        """
        if lag < 1:
            _lag: int = 1
        else:
            _lag: int = lag
        _x_train: list = []
        _y_train: list = []
        _x_test: list = []
        _y_test: list = []
        for x in range(_lag, len(data_set)):
            if train:
                _x_train.append(data_set[x - _lag:x, : -2])
                _y_train.append(data_set[x - 1, -2])
                _x_test.append(data_set[x - _lag:x, 1:-1])
                _y_test.append(data_set[x - 1, -1])
            else:
                _x_train.append(data_set[x - _lag:x, : -1])
                _y_train.append(data_set[x - 1, -1])
                _x_test.append(data_set[x - _lag:x, 1:])
        return dict(x_train=np.array(_x_train),
                    y_train=np.array(_y_train),
                    x_test=np.array(_x_test),
                    y_test=np.array(_y_test)
                    )

    @staticmethod
    def geometric_progression(n: int = 10, ratio: int = 2) -> List[int]:
        """
        Generate list of geometric progression values

        n: int
            Amount of values of the geometric progression

        ratio: float
            Base ratio value of the geometric progression

        :return List[int]
            Geometric progression values
        """
        return list(accumulate([ratio] * n, mul))

    @staticmethod
    def get_analytical_type(df: pd.DataFrame,
                            feature: str,
                            dtype: List[np.dtype],
                            continuous: List[str] = None,
                            categorical: List[str] = None,
                            ordinal: List[str] = None,
                            date: List[str] = None,
                            id_text: List[str] = None,
                            date_edges: Tuple[str, str] = None,
                            max_categories: int = 100
                            ) -> Dict[str, str]:
        """
        Get analytical data type of feature using dask for parallel computing

        :param df:
            Pandas or dask DataFrame

        :param feature: str
            Name of the feature

        :param dtype: List[np.dtype]
            Numpy dtypes of each feature

        :param continuous: List[str]
            Name of the continuous features

        :param categorical: List[str]
            Name of the categorical features

        :param ordinal: List[str]
            Name of the ordinal features

        :param date: List[str]
            Name of the date features

        :param id_text: List[str]
            Name of the identifier or text features

        :param max_categories: int
            Maximum number of categories for identifying feature as categorical

        :return Dict[str, str]:
            Analytical data type and feature name
        """
        if date is not None:
            if feature in date:
                return {'date': feature}
        if ordinal is not None:
            if feature in ordinal:
                return {'ordinal': feature}
        if categorical is not None:
            if feature in categorical:
                return {'categorical': feature}
        if continuous is not None:
            if feature in continuous:
                return {'continuous': feature}
        if id_text is not None:
            if feature in id_text:
                return {'id_text': feature}
        _feature_data = df.loc[~df[feature].isnull(), feature]
        if str(dtype).find('float') >= 0:
            _unique = _feature_data.unique()
            if any(_feature_data.isnull()):
                if any(_unique[~pd.isnull(_unique)] % 1) != 0:
                    return {'continuous': feature}
                else:
                    if len(str(_feature_data.min()).split('.')[0]) > 4:
                        try:
                            assert pd.to_datetime(_feature_data)
                            if date_edges is None:
                                return {'date': feature}
                            else:
                                if (date_edges[0] < pd.to_datetime(_unique.min())) or (
                                        date_edges[1] > pd.to_datetime(_unique.max())):
                                    return {'id_text': feature}
                                else:
                                    return {'date': feature}
                        except (TypeError, ValueError):
                            return {'id_text': feature}
                    else:
                        if len(_unique) > max_categories:
                            return {'ordinal': feature}
                        else:
                            return {'categorical': feature}
            else:
                if any(_unique % 1) != 0:
                    return {'continuous': feature}
                else:
                    if len(str(_feature_data.min()).split('.')[0]) > 4:
                        try:
                            assert pd.to_datetime(_feature_data)
                            if date_edges is None:
                                return {'date': feature}
                            else:
                                if (date_edges[0] < pd.to_datetime(_unique.min())) or (
                                        date_edges[1] > pd.to_datetime(_unique.max())):
                                    return {'id_text': feature}
                                else:
                                    return {'date': feature}
                        except (TypeError, ValueError):
                            return {'id_text': feature}
                    else:
                        if len(_feature_data) == len(_feature_data.unique()):
                            return {'id_text': feature}
                        if len(_unique) > max_categories:
                            return {'ordinal': feature}
                        else:
                            return {'categorical': feature}
        elif str(dtype).find('int') >= 0:
            if len(_feature_data) == len(_feature_data.unique()):
                return {'id_text': feature}
            else:
                if len(_feature_data.unique()) > max_categories:
                    return {'ordinal': feature}
                else:
                    return {'categorical': feature}
        elif str(dtype).find('object') >= 0:
            _unique: np.array = _feature_data.unique()
            _digits: int = 0
            _dot: bool = False
            _max_dots: int = 0
            for text_val in _unique:
                if text_val == text_val:
                    if (str(text_val).find('.') >= 0) or (str(text_val).replace(',', '').isdigit()):
                        _dot = True
                    if str(text_val).replace('.', '').replace('-', '').isdigit() or str(text_val).replace(',',
                                                                                                          '').replace(
                            '-', '').isdigit():
                        if (len(str(text_val).split('.')) == 2) or (len(str(text_val).split(',')) == 2):
                            _digits += 1
                    if len(str(text_val).split('.')) > _max_dots:
                        _max_dots = len(str(text_val).split('.'))
            if _digits >= (len(_unique[~pd.isnull(_unique)]) * 0.5):
                if _dot:
                    try:
                        if any(_unique[~pd.isnull(_unique)] % 1) != 0:
                            return {'continuous': feature}
                        else:
                            if _max_dots == 2:
                                return {'continuous': feature}
                            else:
                                return {'id_text': feature}
                    except (TypeError, ValueError):
                        if _max_dots == 2:
                            return {'continuous': feature}
                        else:
                            return {'id_text': feature}
                else:
                    if len(_feature_data) == len(_feature_data.unique()):
                        return {'id_text': feature}
                    _len_of_feature = pd.DataFrame()
                    _len_of_feature[feature] = _feature_data[~_feature_data.isnull()]
                    _len_of_feature['len'] = _len_of_feature[feature].str.len()
                    _unique_values: np.array = _len_of_feature['len'].unique()
                    if len(_feature_data.unique()) >= (len(_feature_data) * 0.5):
                        return {'id_text': feature}
                    else:
                        if len(_feature_data.unique()) > max_categories:
                            return {'ordinal': feature}
                        else:
                            return {'categorical': feature}
            else:
                try:
                    _potential_date = _feature_data[~_feature_data.isnull()]
                    _unique_years = pd.to_datetime(_potential_date).dt.year.unique()
                    _unique_months = pd.to_datetime(_potential_date).dt.isocalendar().week.unique()
                    _unique_days = pd.to_datetime(_potential_date).dt.day.unique()
                    _unique_cats: int = len(_unique_years) + len(_unique_months) + len(_unique_days)
                    if _unique_cats > 4:
                        return {'date': feature}
                    else:
                        if len(_feature_data) == len(_feature_data.unique()):
                            return {'id_text': feature}
                        if len(_feature_data.unique().values) <= 3:
                            return {'categorical': feature}
                        else:
                            _len_of_feature = pd.DataFrame()
                            _len_of_feature[feature] = _feature_data[~_feature_data.isnull()]
                            _len_of_feature['len'] = _len_of_feature[feature].str.len()
                            _unique_values: np.array = _len_of_feature['len'].unique()
                            for val in _unique_values:
                                if len(re.findall(pattern=r'[a-zA-Z]', string=str(val))) > 0:
                                    if len(_feature_data.unique()) >= (len(_feature_data) * 0.5):
                                        return {'id_text': feature}
                                    else:
                                        return {'categorical': feature}
                            if np.min(_unique_values) > 3:
                                if len(_feature_data.unique()) >= (len(_feature_data) * 0.5):
                                    return {'id_text': feature}
                                else:
                                    if len(_feature_data.unique().values) > max_categories:
                                        return {'ordinal': feature}
                                    else:
                                        return {'categorical': feature}
                            else:
                                return {'categorical': feature}
                except (TypeError, ValueError):
                    if len(_feature_data) == len(_unique):
                        return {'id_text': feature}
                    if len(_feature_data.unique()) <= 3:
                        return {'categorical': feature}
                    else:
                        _len_of_feature = _feature_data[~_feature_data.isnull()]
                        _len_of_feature['len'] = _len_of_feature.str.len()
                        _unique_values: np.array = _len_of_feature['len'].unique()
                        for val in _feature_data.unique():
                            if len(re.findall(pattern=r'[a-zA-Z]', string=str(val))) > 0:
                                if len(_feature_data.unique()) >= (len(_feature_data) * 0.5):
                                    return {'id_text': feature}
                                else:
                                    if len(_feature_data.unique()) > max_categories:
                                        return {'ordinal': feature}
                                    else:
                                        return {'categorical': feature}
                        for ch in SPECIAL_CHARACTERS:
                            if any(_len_of_feature.str.find(ch) > 0):
                                if len(_feature_data.unique()) >= (len(_feature_data) * 0.5):
                                    return {'id_text': feature}
                                else:
                                    return {'categorical': feature}
                        # if np.mean(_unique_values) == np.median(_unique_values):
                        #    return {'id_text': feature}
                        # else:
                        return {'categorical': feature}
        elif str(dtype).find('date') >= 0:
            return {'date': feature}
        elif str(dtype).find('bool') >= 0:
            return {'categorical': feature}

    @staticmethod
    def get_ml_type(values: np.array) -> str:
        """
        Get supervised machine learning problem type from value of target feature

        :param values: np.array
            Value of target feature

        :return str:
            Supervised machine learning type
                -> reg: Regression
                -> clf_multi: Classification with multi class output
                -> clf_binary: Classification with binary class output
        """
        if isinstance(values, list):
            _values: np.array = np.array(values)
        else:
            _values: np.array = values
        if str(_values.dtype).find('int') < 0 and str(_values.dtype).find('float') < 0:
            _values = _values.astype(dtype=float)
        _unique: np.array = pd.unique(values=values)
        if any(_unique[~pd.isnull(_unique)] % 1) != 0:
            return 'reg'
        else:
            if len(_unique[~pd.isnull(_unique)]) == 2:
                return 'clf_binary'
            else:
                return 'clf_multi'

    @staticmethod
    def get_random_perm(shape: int) -> np.random:
        """
        Get random permutation

        :param shape: int: Maximum threshold of range to permutate randomly
        :return: np.random: Randomly permutated range
        """
        np.random.seed(seed=1234)
        return np.random.permutation(x=shape)

    @staticmethod
    def index_to_label(idx: List[int], labels: List[str]) -> List[str]:
        """
        Get list of labels based on a list of indices

        :param idx: List of integers containing the indices
        :param labels: List of integers containing the labels
        :return: List of string containing the subset of labels
        """
        return [labels[i] for i in idx]

    @staticmethod
    def label_to_index(all_labels: List[str], labels: List[str]) -> List[int]:
        """
        Get list of indices based on a list if labels

        :param labels: List of integers containing the indices
        :param all_labels: List of integers containing the labels
        :return: List of integers indicating the subset of indices
        """
        return [all_labels.index(i) for i in labels]

    @staticmethod
    def subset_array(arr: np.array, idx: List[int]) -> np.array:
        """
        Subset Numpy array

        :param arr:
            Data set

        :param idx:
            Indices to remove

        :return: Numpy array containing the data subset
        """
        return np.array(list(itertools.compress(data=arr, selectors=[i not in idx for i in range(len(idx))])))

    @staticmethod
    def rename_dict_keys(d: dict, old_keys: List[str], new_keys: List[str]) -> dict:
        """
        Rename keys of a dictionary

        :param d:
            Data

        :param old_keys:
            Old (current) key names

        :param new_keys:
            New key names

        :return: dict
            Renamed dictionary
        """
        if len(old_keys) != len(new_keys):
            raise HappyLearningUtilsException('Length of the two lists are unequal (old={}, new={}'.format(len(old_keys),
                                                                                                           len(new_keys)
                                                                                                           )
                                              )
        _d = d
        for i, k in enumerate(old_keys):
            _d = json.loads(json.dumps(_d).replace(k, new_keys[i]))
        return _d


class RequestUtilsException(Exception):
    """

    Class for handling exceptions for class RequestUtils

    """
    pass


class RequestUtils(FileUtils):
    """

    Class for handling requests

    """
    def __init__(self, url: List[str], parallel: bool = True, file_path: str = None):
        """
        Parameters
        ----------
        url: List[str]
            URL's for requesting payloads

        parallel: bool
            Request payloads in parallel or not

        file_path: str
            Complete file path either of imported files for sending or exported files for receiving payload
        """
        super().__init__(file_path=file_path, create_dir=True)
        self.url: List[str] = url
        self.parallel: bool = parallel
        self.file_path: str = file_path
        self.payload = None

    def get_payload(self):
        pass

    def send_payload(self):
        pass


class StatsUtils:
    """

    Class for calculating univariate and multivariate statistics

    """

    def __init__(self,
                 data: pd.DataFrame,
                 features: List[str]
                 ):
        """
        :param data:
        :param features:
        """
        self.data_set = data
        self.features = features
        self.nan_policy = 'omit'

    def _anderson_darling_test(self, feature: str, sig_level: float = 0.05) -> float:
        """

        Anderson-Darling test for normality tests

        :param feature:
        :param sig_level:
        :return: float: Probability describing significance level
        """
        _stat = anderson(x=self.data_set[feature], dist='norm')
        try:
            _i: int = _stat.significance_level.tolist().index(100 * sig_level)
            p: float = _stat.critical_values[_i]
        except ValueError:
            p: float = _stat.critical_values[2]
        return p

    def _bartlette_sphericity_test(self) -> dict:
        """

        Bartlette's test for sphericity

        """
        _n_cases, _n_features = self.data_set.shape
        _cor = self.data_set[self.features].corr('pearson')
        _cor_det = np.linalg.det(_cor.values)
        _statistic: np.ndarray = -np.log(_cor_det) * (_n_cases - 1 - (2 * _n_features + 5) / 6)
        _dof = _n_features * (_n_features - 1) / 2
        return dict(statistic=_statistic, p=chi2.pdf(_statistic, _dof))

    def _dagostino_k2_test(self, feature: str) -> float:
        """

        D'Agostino K² test for normality

        :param feature: String containing the name of the feature
        :return: Float indicating the statistical probability value (p-value)
        """
        stat, p = normaltest(a=self.data_set[feature], axis=0, nan_policy='propagate')
        return p

    def _kaiser_meyer_olkin_test(self) -> dict:
        """

        Kaiser-Meyer-Olkin test for unobserved features

        """
        _cor = self.correlation(meth='pearson').values
        _partial_cor = self.correlation(meth='partial').values
        np.fill_diagonal(_cor, 0)
        np.fill_diagonal(_partial_cor, 0)
        _cor = _cor ** 2
        _partial_cor = _partial_cor ** 2
        _cor_sum = np.sum(_cor)
        _partial_cor_sum = np.sum(_partial_cor)
        _cor_sum_feature = np.sum(_cor, axis=0)
        _partial_cor_sum_feature = np.sum(_partial_cor, axis=0)
        return dict(kmo=_cor_sum / (_cor_sum + _partial_cor_sum),
                    kmo_per_feature=_cor_sum_feature / (_cor_sum_feature + _partial_cor_sum_feature),
                    )

    def _shapiro_wilk_test(self, feature: str) -> float:
        """

        Shapiro-Wilk test for normality tests

        :param feature: String containing the name of the feature
        :return: Float indicating the statistical probability value (p-value)
        """
        return shapiro(x=self.data_set[feature])

    def curtosis_test(self) -> List[str]:
        """

        Test whether a distribution is tailed or not

        :return: List of strings containing the names of the tailed features
        """
        raise NotImplementedError('Method not supported')

    def correlation(self, meth: str = 'pearson', min_obs: int = 1) -> pd.DataFrame:
        """

        Calculate correlation coefficients

        :param meth: String containing the method to be used as correlation coefficient
                        -> pearson: Marginal Correlation based on Pearson's r
                        -> kendall: Rank Correlation based on Kendall
                        -> spearman: Rank Correlation based on Spearman
                        -> partial: Partial Correlation
        :param min_obs: Integer indicating the minimum amount of valid observations
        :return: Pandas DataFrame containing the correlation matrix
        """
        if meth in ['pearson', 'kendall', 'spearman']:
            _cor: pd.DataFrame = self.data_set[self.features].corr(method=meth, min_periods=min_obs)
        elif meth == 'partial':
            if self.data_set.shape[0] - self.data_set.isnull().astype(dtype=int).sum().sum() > 0:
                _cov: np.ndarray = np.cov(m=self.data_set[self.features].dropna())
                try:
                    assert np.linalg.det(_cov) > np.finfo(np.float32).eps
                    _inv_var_cov: np.ndarray = np.linalg.inv(_cov)
                except AssertionError:
                    _inv_var_cov: np.ndarray = np.linalg.pinv(_cov)
                    #warnings.warn('The inverse of the variance-covariance matrix '
                    #              'was calculated using the Moore-Penrose generalized '
                    #              'matrix inversion, due to its determinant being at '
                    #              'or very close to zero.')
                _std: np.ndarray = np.sqrt(np.diag(_inv_var_cov))
                _cov2cor: np.ndarray = _inv_var_cov / np.outer(_std, _std)
                _cor: pd.DataFrame = pd.DataFrame(data=np.nan_to_num(x=_cov2cor, copy=True) * -1,
                                                  columns=self.features,
                                                  index=self.features)
            else:
                _cor: pd.DataFrame = pd.DataFrame()
                Log(write=False, level='info').log(msg='Can not calculate coefficients for partial correlation because of the high missing data rate')
        else:
            raise HappyLearningUtilsException('Method for calculating correlation coefficient ({}) not supported'.format(meth))
        return _cor

    def correlation_test(self,
                         x: str,
                         y: str,
                         meth: str = 'pearson',
                         freq_table: List[float] = None,
                         yates_correction: bool = True,
                         power_divergence: str = 'cressie_read'
                         ) -> dict:
        """
        :param x:
        :param y:
        :param meth: String defining the hypothesis test method for correlation
                        -> pearson:
                        -> spearman:
                        -> kendall:
                        -> chi-squared:
        :param freq_table:
        :param yates_correction:
        :param power_divergence: String defining the power divergence test method used in chi-squared independent test
                                    -> pearson: Pearson's chi-squared statistic
                                    -> log-likelihood: Log-Likelihood ratio (G-test)
                                    -> freeman-tukey: Freeman-tukey statistic
                                    -> mod-log-likelihood: Modified log-likelihood ratio
                                    -> neyman: Neyman's statistic
                                    -> cressie-read: Cressie-Read power divergence test statistic
        :return:
        """
        _reject = None
        if meth == 'pearson':
            _correlation_test = pearsonr(x=self.data_set[x], y=self.data_set[y])
        elif meth == 'spearman':
            _correlation_test = spearmanr(a=self.data_set[x], b=self.data_set[y], axis=0, nan_policy=self.nan_policy)
        elif meth == 'kendall':
            _correlation_test = kendalltau(x=self.data_set[x], y=self.data_set[y], nan_policy=self.nan_policy)
        elif meth == 'chi-squared':
            _correlation_test = chi2_contingency(observed=freq_table, correction=yates_correction, lambda_=power_divergence)
        else:
            raise HappyLearningUtilsException('Method for correlation test not supported')
        if _correlation_test[1] <= self.p:
            _reject = False
        else:
            _reject = True
        return {'feature': ''.join(self.data_set.keys()),
                'cases': len(self.data_set.values),
                'test_statistic': _correlation_test[0],
                'p_value': _correlation_test[1],
                'reject': _reject}

    def factoriability_test(self, meth: str = 'kmo') -> dict:
        """

        Test whether a data set contains unobserved features required for factor analysis

        :param meth: String containing the name of the used method
                        -> kmo: Kaiser-Meyer-Olkin Criterion
                        -> bartlette: Bartlette's test of sphericity
        """
        _fac: dict = {}
        if meth == 'kmo':
            pass
        elif meth == 'bartlette':
            pass
        else:
            raise HappyLearningUtilsException('Method for testing "factoriability" ({}) not supported'.format(meth))
        raise NotImplementedError('Method not supported')

    def non_parametric_test(self,
                            x: str,
                            y: str,
                            meth: str = 'kruskal-wallis',
                            continuity_correction: bool = True,
                            alternative: str = 'two-sided',
                            zero_meth: str = 'pratt',
                            *args
                            ):
        """
        :param x:
        :param y:
        :param meth: String defining the hypothesis test method for non-parametric tests
                        -> kruskal-wallis: Kruskal-Wallis H test to test whether the distributions of two or more
                                           independent samples are equal or not
                        -> mann-whitney: Mann-Whitney U test to test whether the distributions of two independent
                                         samples are equal or not
                        -> wilcoxon: Wilcoxon Signed-Rank test for test whether the distributions of two paired samples
                                     are equal or not
                        -> friedman: Friedman test for test whether the distributions of two or more paired samples
                                     are equal or not
        :param continuity_correction:
        :param alternative: String defining the type of hypothesis test
                            -> two-sided:
                            -> less:
                            -> greater:
        :param zero_meth: String defining the method to handle zero differences in the ranking process (Wilcoxon test)
                            -> pratt: Pratt treatment that includes zero-differences (more conservative)
                            -> wilcox: Wilcox tratment that discards all zero-differences
                            -> zsplit: Zero rank split, just like Pratt, but spliting the zero rank between positive
                                       and negative ones
        :param args:
        :return:
        """
        _reject = None
        if meth == 'kruskal-wallis':
            _non_parametric_test = kruskal(args, self.nan_policy)
        elif meth == 'mann-whitney':
            _non_parametric_test = mannwhitneyu(x=self.data_set[x],
                                                y=self.data_set[y],
                                                use_continuity=continuity_correction,
                                                alternative=alternative)
        elif meth == 'wilcoxon':
            _non_parametric_test = wilcoxon(x=self.data_set[x],
                                            y=self.data_set[y],
                                            zero_method=zero_meth,
                                            correction=continuity_correction)
        elif meth == 'friedman':
            _non_parametric_test = friedmanchisquare(args)
        else:
            raise HappyLearningUtilsException('No non-parametric test found !')
        if _non_parametric_test[1] <= self.p:
            _reject = False
        else:
            _reject = True
        return {'feature': ''.join(self.data_set.keys()),
                'cases': len(self.data_set.values),
                'test_statistic': _non_parametric_test[0],
                'p_value': _non_parametric_test[1],
                'reject': _reject}

    def normality_test(self, alpha: float = 0.05, meth: str = 'shapiro-wilk') -> dict:
        """

        Test whether a distribution is normal distributed or not

        :param alpha: Float indicating the threshold that indicates whether a hypothesis can be rejected or not
        :param meth: String containing the method to test normality
                        -> shapiro-wilk:
                        -> anderson-darling:
                        -> dagostino:
        :return: dict: Results of normality test (statistic, p-value, p > alpha)
        """
        _alpha = alpha
        _normality: dict = {}
        for feature in self.features:
            if meth == 'shapiro-wilk':
                _stat, _p = self._shapiro_wilk_test(feature=feature)
            elif meth == 'anderson-darling':
                _stat, _p = self._anderson_darling_test(feature=feature, sig_level=alpha)
            elif meth == 'dagostino':
                _stat, _p = self._dagostino_k2_test(feature=feature)
            else:
                raise HappyLearningUtilsException('Method ({}) for testing normality not supported'.format(meth))
            _normality.update({feature: dict(stat=_stat, p=_p, normality=_p > _alpha)})
        return _normality

    def parametric_test(self, x: str, y: str, meth: str = 't-test', welch_t_test: bool = True, *args):
        """
        :param x:
        :param y:
        :param meth: String defining the hypothesis test method for parametric tests
                        -> z-test:
                        -> t-test:
                        -> t-test-paired:
                        -> anova:
        :param welch_t_test:
        :param args: Arguments containing samples from two or more groups for anova test
        :return:
        """
        _reject = None
        if meth == 't-test':
            _parametric_test = ttest_ind(a=self.data_set[x], b=self.data_set[y],
                                         axis=0, equal_var=not welch_t_test, nan_policy=self.nan_policy)
        elif meth == 't-test-paired':
            _parametric_test = ttest_rel(a=self.data_set[x], b=self.data_set[y], axis=0, nan_policy=self.nan_policy)
        elif meth == 'anova':
            _parametric_test = f_oneway(args)
        elif meth == 'z-test':
            _parametric_test = ztest(x1=x, x2=y, value=0, alternative='two-sided', usevar='pooled', ddof=1)
        else:
            raise ValueError('No parametric test found !')
        if _parametric_test[1] <= self.p:
            _reject = False
        else:
            _reject = True
        return {'feature': ''.join(self.data_set.keys()),
                'cases': len(self.data_set.values),
                'test_statistic': _parametric_test[0],
                'p_value': _parametric_test[1],
                'reject': _reject}

    def power_law_test(self,
                       tail_prob: List[float] = None,
                       shape_params: List[float] = None,
                       location_params: List[float] = None,
                       size: Tuple[int] = None,
                       moments: str = 'mvsk'
                       ):
        """
        :param tail_prob:
        :param shape_params:
        :param location_params:
        :param size:
        :param moments:
        :return:
        """
        raise NotImplementedError('Method not supported')

    def skewness_test(self, axis: str = 'col', threshold_interval: Tuple[float, float] = (-0.5, 0.5)) -> dict:
        """
        Test whether a distribution is skewed or not

        :param axis: String containing the name of the axis of the data frame to use
                        -> col: Test skewness of feature
                        -> row: test skewness of cases

        :param threshold_interval: Tuple of floats indicating the threshold interval for testing

        :return: List of strings containing the name of the skewed features
        """
        if axis == 'col':
            _axis = 0
        elif axis == 'row':
            _axis = 1
        else:
            raise HappyLearningUtilsException('Axis ({}) not supported'.format(axis))
        return self.data_set[self.features].skew(axis=_axis).to_dict()


class WebCrawlerUtils(FileUtils):
    """
    Class for crawling data from the internet
    """
    def __init__(self, location: List[str], file_path: str = None, stop: int = 15):
        """
        :param location:
        :param file_path:
        :param stop:
        """
        super().__init__(file_path=file_path, create_dir=True)
        self.location = location
        self.stop = stop

    def crawl_geo_data(self, provider: str = 'arcgis') -> dict:
        """
        Crawl continuous geo data based on categorical geo data

        :param provider: str
            Name of the provider to use:
                -> arcgis: ArcGis
                -> google: Google Maps

        :return: Dictionary containing the results of the geo-coding
        """
        _geo: dict = {}
        _status: str = ''
        for i, loc in enumerate(self.location):
            #while _status.find('REQUEST_DENIED') >= 0 or _status == '':
            if provider == 'arcgis':
                _g: geocoder = geocoder.arcgis(location=loc, maxRows=1, method='geocode')
            elif provider == 'google':
                _g: geocoder = geocoder.google(location=loc, maxRows=1, method='geocode')
            elif provider == 'bing':
                _g: geocoder = geocoder.bing(location=loc, maxRows=1, method='geocode')
            elif provider == 'baidu':
                _g: geocoder = geocoder.baidu(location=loc, maxRows=1, method='geocode')
            elif provider == 'freegeoip':
                _g: geocoder = geocoder.freegeoip(location=loc, maxRows=1, method='geocode')
            elif provider == 'osm':
                _g: geocoder = geocoder.osm(location=loc, maxRows=1, method='geocode')
            elif provider == 'tomtom':
                _g: geocoder = geocoder.tomtom(location=loc, maxRows=1, method='geocode')
            elif provider == 'yahoo':
                _g: geocoder = geocoder.yahoo(location=loc, maxRows=1, method='geocode')
            else:
                raise HappyLearningUtilsException('Provider "{}" for geocoding not supported'.format(provider))
            _status = _g.status
            if _status.find('OK') >= 0:
                _geo.update({loc: _g.json})
            elif _status.find('ERROR') >= 0:
                _geo.update({loc: 'NaN'})
            else:
                if _status.find('REQUEST_DENIED') < 0:
                    raise HappyLearningUtilsException('Unknown request error "{}"'.format(_g.status))
        if self.full_path is not None:
            DataExporter(obj=_geo, file_path=self.full_path).file()
        return _geo
