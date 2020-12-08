import pandas as pd
import unittest

from happy_learning.sampler import MLSampler, Sampler

DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='gun-violence-data_01-2013_03-2018.csv')




if __name__ == '__main__':
    unittest.main()
