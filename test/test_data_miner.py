import os
import pandas as pd
import unittest

from happy_learning.data_miner import DataMiner
from happy_learning.feature_engineer import FeatureEngineer
from typing import List

DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='data/avocado.csv')


class DataMinerTest(unittest.TestCase):
    """
    Class for testing class DataMiner
    """
    def test_supervised_clf(self):
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=DATA_SET, target_feature='type')
        _feature_engineer.set_predictors(exclude_original_data=False)
        DataMiner(df=_feature_engineer.get_data(dask_df=True),
                  file_path=None,
                  target=_feature_engineer.get_target(),
                  predictors=_feature_engineer.get_predictors(),
                  feature_engineer=_feature_engineer,
                  feature_generator=True,
                  train_critic=True,
                  plot=True,
                  output_path='data',
                  **dict(max_generations=2)
                  ).supervised(models=['cat', 'xgb'],
                               feature_selector='shapley',
                               top_features=0.5,
                               optimizer='ga',
                               force_target_type=None,
                               train=True,
                               train_size=0.8,
                               random=True,
                               clf_eval_metric='auc',
                               reg_eval_metric='rmse_norm',
                               save_train_test_data=True,
                               save_ga=True,
                               **dict(engineer_categorical=False)
                               )
        _found_results: List[bool] = []
        for result in ['feature_learning_data.parquet',
                       'feature_learning.p',
                       'feature_importance_shapley.html',
                       'feature_tournament_game_size.html',
                       'genetic.p',
                       'model.p'
                       ]:
            if os.path.isfile('data/{}'.format(result)):
                _found_results.append(True)
            else:
                _found_results.append(False)
        self.assertTrue(expr=all(_found_results))

    def test_supervised_reg(self):
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=DATA_SET, target_feature='AveragePrice')
        _feature_engineer.set_predictors(exclude_original_data=False)
        DataMiner(df=_feature_engineer.get_data(dask_df=True),
                  file_path=None,
                  target=_feature_engineer.get_target(),
                  predictors=_feature_engineer.get_predictors(),
                  feature_engineer=_feature_engineer,
                  feature_generator=True,
                  train_critic=True,
                  plot=True,
                  output_path='data',
                  **dict(max_generations=2)
                  ).supervised(models=['cat', 'xgb'],
                               feature_selector='shapley',
                               top_features=0.5,
                               optimizer='ga',
                               force_target_type=None,
                               train=True,
                               train_size=0.8,
                               random=True,
                               clf_eval_metric='auc',
                               reg_eval_metric='rmse_norm',
                               save_train_test_data=True,
                               save_ga=True,
                               **dict(engineer_categorical=False)
                               )
        _found_results: List[bool] = []
        for result in ['feature_learning_data.parquet',
                       'feature_learning.p',
                       'feature_importance_shapley.html',
                       'feature_tournament_game_size.html',
                       'genetic.p',
                       'model.p'
                       ]:
            if os.path.isfile('data/{}'.format(result)):
                _found_results.append(True)
            else:
                _found_results.append(False)
        self.assertTrue(expr=all(_found_results))


if __name__ == '__main__':
    unittest.main()
