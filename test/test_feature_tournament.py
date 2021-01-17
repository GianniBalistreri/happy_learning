import pandas as pd
import unittest

from happy_learning.feature_tournament import FeatureTournament
from typing import List

DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='data/avocado.csv')


class FeatureTournamentTest(unittest.TestCase):
    """
    Class for testing class FeatureTournament
    """
    def test_play(self):
        _iter: int = 5
        _games: int = 3
        _features: List[str] = ['Total Volume', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', '4046', '4225']
        _tournament: FeatureTournament = FeatureTournament(df=DATA_SET,
                                                           features=_features,
                                                           target='AveragePrice',
                                                           force_target_type=None,
                                                           models=None,
                                                           init_pairs=3,
                                                           init_games=5,
                                                           increasing_pair_size_factor=0.5,
                                                           games=_games,
                                                           penalty_factor=0.1,
                                                           max_iter=_iter,
                                                           multi_threading=True
                                                           )
        _shapley_additive_explanation: dict = _tournament.play()
        _test_feature: str = list(_shapley_additive_explanation.get('sum').keys())[0]
        self.assertTrue(expr=_iter * _games == len(_shapley_additive_explanation['game'][_test_feature]) and int(sum(_shapley_additive_explanation['game'][_test_feature])) <= int(_shapley_additive_explanation['sum'][_test_feature]))


if __name__ == '__main__':
    unittest.main()
