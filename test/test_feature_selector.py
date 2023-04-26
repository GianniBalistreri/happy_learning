import pandas as pd
import unittest

from happy_learning.feature_selector import FeatureSelector
from happy_learning.sampler import MLSampler
from happy_learning.supervised_machine_learning import ModelGeneratorReg
from typing import List

DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='data/avocado.csv')


class FeatureSelectorTest(unittest.TestCase):
    """
    Class for testing class FeatureSelector
    """
    def test_select(self):
        _features: List[str] = ['Total Volume', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', '4046', '4225']
        _feature_selector: FeatureSelector = FeatureSelector(df=DATA_SET,
                                                             target='AveragePrice',
                                                             features=_features,
                                                             force_target_type=None,
                                                             model_name='cat',
                                                             init_pairs=3,
                                                             init_games=5,
                                                             increasing_pair_size_factor=0.5,
                                                             games=3,
                                                             penalty_factor=0.1,
                                                             max_iter=50,
                                                             max_players=-1,
                                                             evolutionary_algorithm='ga',
                                                             use_standard_params=True,
                                                             aggregate_feature_imp=None,
                                                             visualize_all_scores=False,
                                                             visualize_variant_scores=False,
                                                             visualize_core_feature_scores=False,
                                                             path='data/',
                                                             mlflow_log=True,
                                                             multi_threading=False
                                                             )
        _eval_result: dict = _feature_selector.select(imp_threshold=0.01,
                                                      redundant_threshold=0.02,
                                                      visualize_game_stats=True,
                                                      plot_type='bar'
                                                      )
        _imp_features: List[str] = _eval_result.get('important')
        _train_test_split: dict = MLSampler(df=DATA_SET,
                                            target='AveragePrice',
                                            features=_features,
                                            train_size=0.8,
                                            random_sample=True,
                                            stratification=False,
                                            seed=1234
                                            ).train_test_sampling(validation_split=0.1)
        _model_reg_all_features: ModelGeneratorReg = ModelGeneratorReg(model_name='xgb')
        _model_reg_all_features.generate_model()
        _model_reg_all_features.train(x=_train_test_split['x_train'], y=_train_test_split['y_train'])
        _pred_all_features = _model_reg_all_features.predict(x=_train_test_split['y_test'])
        _model_reg_all_features.eval(obs=_train_test_split['y_test'], pred=_pred_all_features)
        _model_reg_imp_features: ModelGeneratorReg = ModelGeneratorReg(model_name='xgb',
                                                                       reg_params=_model_reg_all_features.model_param
                                                                       )
        _model_reg_imp_features.generate_model()
        _model_reg_imp_features.train(x=_train_test_split['x_train'][_imp_features],
                                      y=_train_test_split['y_train'][_imp_features]
                                      )
        _pred_imp_features = _model_reg_imp_features.predict(x=_train_test_split['x_test'][_imp_features])
        _model_reg_all_features.eval(obs=_train_test_split['y_test'][_imp_features], pred=_pred_imp_features)
        self.assertAlmostEqual(first=_model_reg_all_features.fitness['test']['rmse_norm'],
                               second=_model_reg_imp_features.fitness['test']['rmse_norm']
                               )


if __name__ == '__main__':
    unittest.main()
