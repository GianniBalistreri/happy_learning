import pandas as pd
import unittest

from happy_learning.missing_data_analysis import MissingDataAnalysis

DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='data/avocado.csv')


class MissingDataAnalysisTest(unittest.TestCase):
    """
    Class for testing class MissingDataAnalysis
    """
    def test_clean_nan(self):
        self.assertTrue(expr=len(MissingDataAnalysis(df=DATA_SET).clean_nan()) <= DATA_SET.shape[0])

    def test_freq_nan_by_cases(self):
        _n_mis: int = DATA_SET.isnull().astype(int).sum().sum()
        _n_cases: int = DATA_SET.shape[0]
        _freq_nan_by_cases: dict = MissingDataAnalysis(df=DATA_SET, percentages=False).freq_nan_by_cases()
        _count_mis: int = 0
        for case in _freq_nan_by_cases.keys():
            _count_mis += _freq_nan_by_cases.get(case)
        self.assertTrue(expr=(_n_mis == _count_mis) and (_n_cases == len(_freq_nan_by_cases)))

    def test_freq_nan_by_features(self):
        _n_mis: int = DATA_SET.isnull().astype(int).sum().sum()
        _n_features: int = len(DATA_SET.columns)
        _freq_nan_by_features: dict = MissingDataAnalysis(df=DATA_SET, percentages=False).freq_nan_by_features()
        _count_mis: int = 0
        for feature in _freq_nan_by_features.keys():
            _count_mis += _freq_nan_by_features.get(feature)
        self.assertTrue(expr=(_n_mis == _count_mis) and (_n_features == len(_freq_nan_by_features)))

    def test_has_nan(self):
        self.assertTrue(expr=MissingDataAnalysis(df=DATA_SET).has_nan())

    def test_get_nan_idx_by_cases(self):
        _nan_idx_by_cases: dict = MissingDataAnalysis(df=DATA_SET).get_nan_idx_by_cases()
        _is_nan: bool = True
        for case in _nan_idx_by_cases.keys():
            for idx in _nan_idx_by_cases.get(case):
                if not pd.isnull(DATA_SET.loc[idx, :].values).all():
                    _is_nan = False
                    break
            if not _is_nan:
                break
        self.assertTrue(expr=_is_nan)

    def test_get_nan_idx_by_features(self):
        _nan_idx_by_features: dict = MissingDataAnalysis(df=DATA_SET).get_nan_idx_by_features()
        _is_nan: bool = True
        for feature in _nan_idx_by_features.keys():
            for idx in _nan_idx_by_features.get(feature):
                if not pd.isnull(DATA_SET.loc[idx, feature]):
                    _is_nan = False
                    break
            if not _is_nan:
                break
        self.assertTrue(expr=_is_nan)


if __name__ == '__main__':
    unittest.main()
