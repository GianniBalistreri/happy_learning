import dask.dataframe as dd
import unittest

from easyexplore.utils import EasyExploreUtils
from happy_learning.missing_data_analysis import MissingDataAnalysis
from happy_learning.multiple_imputation import MultipleImputation

DASK_CLIENT = EasyExploreUtils().dask_setup(client_name='test_multiple_imputation')
DATA_SET: dd.DataFrame = dd.read_csv(filepath_or_buffer='test/gun-violence-data_01-2013_03-2018.csv')


class MultipleImputationTest(unittest.TestCase):
    """
    Class for testing class MultipleImputation
    """
    def test_emb(self):
        pass

    def test_mice(self):
        _has_nan_before: bool = MissingDataAnalysis(df=DATA_SET).has_nan()
        _has_nan_after: bool = MissingDataAnalysis(df=MultipleImputation(df=DATA_SET).mice()).has_nan()
        self.assertGreater(a=int(_has_nan_before), b=int(_has_nan_after))


DASK_CLIENT.close()
