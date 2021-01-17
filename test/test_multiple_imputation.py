import pandas as pd
import unittest

from happy_learning.missing_data_analysis import MissingDataAnalysis
from happy_learning.multiple_imputation import MultipleImputation

DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='data/gun-violence-data_01-2013_03-2018.csv')


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


if __name__ == '__main__':
    unittest.main()
