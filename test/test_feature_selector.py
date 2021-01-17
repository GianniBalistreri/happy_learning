import pandas as pd
import unittest

from happy_learning.feature_selector import FeatureSelector
from happy_learning.feature_tournament import FeatureTournament
from typing import List

DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='data/avocado.csv')


class FeatureSelectorTest(unittest.TestCase):
    """
    Class for testing class FeatureSelector
    """
    def test_get_imp_features_shapley(self):
        _features: List[str] = ['Total Volume', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', '4046', '4225']
        _tournament: FeatureTournament = FeatureTournament(df=DATA_SET,
                                                           features=_features,
                                                           target='AveragePrice',
                                                           force_target_type=None,
                                                           models=None,
                                                           init_pairs=3,
                                                           init_games=5,
                                                           increasing_pair_size_factor=0.5,
                                                           games=3,
                                                           penalty_factor=0.1,
                                                           max_iter=5,
                                                           multi_threading=True
                                                           )
        _shapley_additive_explanation: dict = _tournament.play()
        _rank_df: pd.DataFrame = pd.DataFrame(data=_shapley_additive_explanation.get('sum'), index=[0]).transpose()
        _rank_df.sort_values(by=list(_rank_df.keys())[0], axis=0, ascending=False, inplace=True)
        _feature_selector: FeatureSelector = FeatureSelector(df=DATA_SET,
                                                             target='AveragePrice',
                                                             features=_features,
                                                             force_target_type=None,
                                                             aggregate_feature_imp=None,
                                                             visualize_all_scores=False,
                                                             visualize_variant_scores=False,
                                                             visualize_core_feature_scores=False,
                                                             path='data/'
                                                             )
        _selected_features: List[str] = _feature_selector.get_imp_features(meth='shapley',
                                                                           model='cat',
                                                                           imp_threshold=0.01,
                                                                           plot_type='bar'
                                                                           ).get('imp_features')
        self.assertTrue(expr=_rank_df.index.values[0] in _selected_features and len(_selected_features) <= len(_features))


if __name__ == '__main__':
    unittest.main()
