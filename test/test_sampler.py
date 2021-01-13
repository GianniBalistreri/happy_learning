import pandas as pd
import unittest

from happy_learning.sampler import MLSampler, Sampler

DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='data/gun-violence-data_01-2013_03-2018.csv')


class MLSamplerTest(unittest.TestCase):
    """
    Class for testing class MLSampler
    """
    def test_train_test_sampling(self):
        _n_cases: int = len(DATA_SET)
        _train_test_sample: dict = MLSampler(df=DATA_SET,
                                             target='n_killed',
                                             features=None,
                                             train_size=0.8,
                                             random_sample=True,
                                             seed=1234
                                             ).train_test_sampling(validation_split=0.1)
        _train_test_cases: int = _train_test_sample.get('x_train').shape[0] + _train_test_sample.get('x_test').shape[0] + _train_test_sample.get('x_val').shape[0]
        self.assertTrue(expr=_n_cases == _train_test_cases)

    def test_k_fold_cross_validation(self):
        pass

    def test_time_series(self):
        pass


class SamplerTest(unittest.TestCase):
    """
    Class for testing class Sampler
    """
    def test_random(self):
        _sample_size: int = 100
        _sample: pd.DataFrame = Sampler(df=DATA_SET, size=_sample_size, prop=None).random()
        self.assertEqual(first=_sample_size, second=len(_sample))

    def test_quota(self):
        pass


if __name__ == '__main__':
    unittest.main()
