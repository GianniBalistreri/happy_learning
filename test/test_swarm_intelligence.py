import os
import pandas as pd
import unittest

from happy_learning.feature_engineer import FeatureEngineer
from happy_learning.swarm_intelligence import SwarmIntelligence
from typing import List

DATA_DIR: str = 'data/'
DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='{}avocado.csv'.format(DATA_DIR))


class SwarmIntelligenceTest(unittest.TestCase):
    """
    Class for testing class SwarmIntelligence
    """
    def test_optimize_modeling(self):
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=DATA_SET, target_feature='AveragePrice', temp_dir=DATA_DIR)
        _feature_engineer.set_predictors(exclude_original_data=False)
        _si: SwarmIntelligence = SwarmIntelligence(mode='model',
                                                   target=_feature_engineer.get_target(),
                                                   input_file_path=None,
                                                   train_data_file_path=None,
                                                   test_data_file_path=None,
                                                   valid_data_file_path=None,
                                                   df=None,
                                                   data_set=None,
                                                   features=_feature_engineer.get_predictors(),
                                                   re_split_data=False,
                                                   re_sample_cases=False,
                                                   re_sample_features=False,
                                                   re_populate=True,
                                                   max_trials=2,
                                                   max_features=-1,
                                                   labels=None,
                                                   models=['cat'],
                                                   model_params=None,
                                                   burn_in_adjustments=-1,
                                                   warm_start=True,
                                                   warm_start_strategy='monotone',
                                                   warm_start_constant_hidden_layers=0,
                                                   warm_start_constant_category='very_small',
                                                   max_adjustments=10,
                                                   pop_size=64,
                                                   adjustment_rate=0.1,
                                                   adjustment_prob=0.85,
                                                   early_stopping=0,
                                                   convergence=True,
                                                   convergence_measure='median',
                                                   timer_in_seconds=43200,
                                                   force_target_type=None,
                                                   plot=False,
                                                   output_file_path='data',
                                                   deep_learning_type='batch',
                                                   deep_learning_output_size=None,
                                                   log=False,
                                                   feature_engineer=_feature_engineer,
                                                   sampling_function=None
                                                   )
        _si.optimize()
        self.assertTrue(expr=_si.evolution_gradient.get('max')[0] <= _si.evolution_gradient.get('max')[-1])

    def test_optimize_feature_engineering(self):
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=DATA_SET, target_feature='AveragePrice', temp_dir=DATA_DIR)
        _feature_engineer.set_predictors(exclude_original_data=False)
        _si: SwarmIntelligence = SwarmIntelligence(mode='feature_engineer',
                                                   target=_feature_engineer.get_target(),
                                                   input_file_path=None,
                                                   train_data_file_path=None,
                                                   test_data_file_path=None,
                                                   valid_data_file_path=None,
                                                   df=None,
                                                   data_set=None,
                                                   features=_feature_engineer.get_predictors(),
                                                   re_split_data=False,
                                                   re_sample_cases=False,
                                                   re_sample_features=False,
                                                   re_populate=True,
                                                   max_trials=2,
                                                   max_features=-1,
                                                   labels=None,
                                                   models=['cat'],
                                                   model_params=None,
                                                   burn_in_adjustments=-1,
                                                   warm_start=True,
                                                   warm_start_strategy='monotone',
                                                   warm_start_constant_hidden_layers=0,
                                                   warm_start_constant_category='very_small',
                                                   max_adjustments=5,
                                                   pop_size=64,
                                                   adjustment_rate=0.1,
                                                   adjustment_prob=0.85,
                                                   early_stopping=0,
                                                   convergence=True,
                                                   convergence_measure='median',
                                                   timer_in_seconds=43200,
                                                   force_target_type=None,
                                                   plot=False,
                                                   output_file_path='data',
                                                   deep_learning_type='batch',
                                                   deep_learning_output_size=None,
                                                   log=False,
                                                   feature_engineer=_feature_engineer,
                                                   sampling_function=None
                                                   )
        _si.optimize()
        self.assertTrue(expr=_si.evolution_gradient.get('max')[0] <= _si.evolution_gradient.get('max')[-1])

    def test_optimize_sampling(self):
        pass

    def test_optimize_continue(self):
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=DATA_SET, target_feature='AveragePrice', temp_dir=DATA_DIR)
        _feature_engineer.set_predictors(exclude_original_data=False)
        _max_adjustments: int = 5
        _si: SwarmIntelligence = SwarmIntelligence(mode='model',
                                                   target=_feature_engineer.get_target(),
                                                   input_file_path=None,
                                                   train_data_file_path=None,
                                                   test_data_file_path=None,
                                                   valid_data_file_path=None,
                                                   df=None,
                                                   data_set=None,
                                                   features=_feature_engineer.get_predictors(),
                                                   re_split_data=False,
                                                   re_sample_cases=False,
                                                   re_sample_features=False,
                                                   re_populate=True,
                                                   max_trials=2,
                                                   max_features=-1,
                                                   labels=None,
                                                   models=['cat'],
                                                   model_params=None,
                                                   burn_in_adjustments=-1,
                                                   warm_start=True,
                                                   warm_start_strategy='monotone',
                                                   warm_start_constant_hidden_layers=0,
                                                   warm_start_constant_category='very_small',
                                                   max_adjustments=_max_adjustments,
                                                   pop_size=64,
                                                   adjustment_rate=0.1,
                                                   adjustment_prob=0.85,
                                                   early_stopping=0,
                                                   convergence=True,
                                                   convergence_measure='median',
                                                   timer_in_seconds=43200,
                                                   force_target_type=None,
                                                   plot=False,
                                                   output_file_path='data',
                                                   deep_learning_type='batch',
                                                   deep_learning_output_size=None,
                                                   log=False,
                                                   feature_engineer=_feature_engineer,
                                                   sampling_function=None
                                                   )
        _si.optimize()
        self.assertTrue(expr=_si.evolution_gradient.get('max')[0] <= _si.evolution_gradient.get('max')[-1] and len(_si.evolution_gradient.get('max')) > _max_adjustments)

    def test_visualize_clf(self):
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=DATA_SET, target_feature='type', temp_dir=DATA_DIR)
        _feature_engineer.set_predictors(exclude_original_data=False)
        _si: SwarmIntelligence = SwarmIntelligence(mode='model',
                                                   target=_feature_engineer.get_target(),
                                                   input_file_path=None,
                                                   train_data_file_path=None,
                                                   test_data_file_path=None,
                                                   valid_data_file_path=None,
                                                   df=None,
                                                   data_set=None,
                                                   features=_feature_engineer.get_predictors(),
                                                   re_split_data=False,
                                                   re_sample_cases=False,
                                                   re_sample_features=False,
                                                   re_populate=True,
                                                   max_trials=2,
                                                   max_features=-1,
                                                   labels=None,
                                                   models=None,
                                                   model_params=None,
                                                   burn_in_adjustments=-1,
                                                   warm_start=True,
                                                   warm_start_strategy='monotone',
                                                   warm_start_constant_hidden_layers=0,
                                                   warm_start_constant_category='very_small',
                                                   max_adjustments=1,
                                                   pop_size=64,
                                                   adjustment_rate=0.1,
                                                   adjustment_prob=0.85,
                                                   early_stopping=0,
                                                   convergence=False,
                                                   convergence_measure='min',
                                                   timer_in_seconds=43200,
                                                   force_target_type=None,
                                                   plot=False,
                                                   output_file_path='data',
                                                   deep_learning_type='batch',
                                                   deep_learning_output_size=None,
                                                   log=False,
                                                   feature_engineer=_feature_engineer,
                                                   sampling_function=None
                                                   )
        _si.optimize()
        _si.visualize(results_table=False,
                      model_distribution=True,
                      model_evolution=True,
                      param_distribution=False,
                      train_time_distribution=True,
                      breeding_map=True,
                      breeding_graph=False,
                      fitness_distribution=True,
                      fitness_evolution=True,
                      fitness_dimensions=True,
                      per_adjustment=True,
                      prediction_of_best_model=False,
                      epoch_stats=False
                      )
        _found_plot: List[bool] = []
        for plot in ['ga_metadata_table',
                     'ga_model_evolution',
                     'ga_model_distribution',
                     'ga_parameter_treemap',
                     'ga_training_time_distribution',
                     'ga_breeding_heatmap',
                     'ga_breeding_graph',
                     'ga_fitness_score_distribution_per_adjustment',
                     'ga_metadata_evolution_coords_actor_only',
                     'ga_evolution_fitness_score',
                     'ga_prediction_confusion_table',
                     'ga_prediction_confusion_heatmap',
                     'ga_prediction_confusion_normal_heatmap'
                     ]:
            if os.path.isfile('data/{}.html'.format(plot)):
                _found_plot.append(True)
            else:
                _found_plot.append(False)
        self.assertTrue(expr=all(_found_plot))

    def test_visualize_reg(self):
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=DATA_SET, target_feature='AveragePrice', temp_dir=DATA_DIR)
        _feature_engineer.set_predictors(exclude_original_data=False)
        _si: SwarmIntelligence = SwarmIntelligence(mode='model',
                                                   target=_feature_engineer.get_target(),
                                                   input_file_path=None,
                                                   train_data_file_path=None,
                                                   test_data_file_path=None,
                                                   valid_data_file_path=None,
                                                   df=None,
                                                   data_set=None,
                                                   features=_feature_engineer.get_predictors(),
                                                   re_split_data=False,
                                                   re_sample_cases=False,
                                                   re_sample_features=False,
                                                   re_populate=True,
                                                   max_trials=2,
                                                   max_features=-1,
                                                   labels=None,
                                                   models=None,
                                                   model_params=None,
                                                   burn_in_adjustments=-1,
                                                   warm_start=True,
                                                   warm_start_strategy='monotone',
                                                   warm_start_constant_hidden_layers=0,
                                                   warm_start_constant_category='very_small',
                                                   max_adjustments=5,
                                                   pop_size=4,
                                                   adjustment_rate=0.1,
                                                   adjustment_prob=0.85,
                                                   early_stopping=0,
                                                   convergence=True,
                                                   convergence_measure='median',
                                                   timer_in_seconds=43200,
                                                   force_target_type=None,
                                                   plot=False,
                                                   output_file_path='data',
                                                   deep_learning_type='batch',
                                                   deep_learning_output_size=None,
                                                   log=False,
                                                   feature_engineer=_feature_engineer,
                                                   sampling_function=None
                                                   )
        _si.optimize()
        _si.visualize(results_table=True,
                      model_distribution=True,
                      model_evolution=True,
                      param_distribution=True,
                      train_time_distribution=True,
                      breeding_map=True,
                      breeding_graph=True,
                      fitness_distribution=True,
                      fitness_evolution=True,
                      fitness_dimensions=True,
                      per_adjustment=True,
                      prediction_of_best_model=True,
                      epoch_stats=True
                      )
        _found_plot: List[bool] = []
        for plot in ['ga_metadata_table',
                     'ga_model_evolution',
                     'ga_model_distribution',
                     #'ga_parameter_treemap',
                     'ga_training_time_distribution',
                     #'ga_breeding_heatmap',
                     #'ga_breeding_graph',
                     'ga_fitness_score_distribution_per_adjustment',
                     'ga_metadata_evolution_coords_actor_only',
                     'ga_evolution_fitness_score',
                     'ga_prediction_evaluation_coords',
                     'ga_prediction_scatter_contour'
                     ]:
            if os.path.isfile('data/{}.html'.format(plot)):
                _found_plot.append(True)
            else:
                _found_plot.append(False)
        self.assertTrue(expr=all(_found_plot))

    def test_save_evolution(self):
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=DATA_SET, target_feature='AveragePrice', temp_dir=DATA_DIR)
        _feature_engineer.set_predictors(exclude_original_data=False)
        _si: SwarmIntelligence = SwarmIntelligence(mode='model',
                                                   target=_feature_engineer.get_target(),
                                                   input_file_path=None,
                                                   train_data_file_path=None,
                                                   test_data_file_path=None,
                                                   valid_data_file_path=None,
                                                   df=None,
                                                   data_set=None,
                                                   features=_feature_engineer.get_predictors(),
                                                   re_split_data=False,
                                                   re_sample_cases=False,
                                                   re_sample_features=False,
                                                   re_populate=True,
                                                   max_trials=2,
                                                   max_features=-1,
                                                   labels=None,
                                                   models=['cat'],
                                                   model_params=None,
                                                   burn_in_adjustments=-1,
                                                   warm_start=True,
                                                   warm_start_strategy='monotone',
                                                   warm_start_constant_hidden_layers=0,
                                                   warm_start_constant_category='very_small',
                                                   max_adjustments=5,
                                                   pop_size=64,
                                                   adjustment_rate=0.1,
                                                   adjustment_prob=0.85,
                                                   early_stopping=0,
                                                   convergence=True,
                                                   convergence_measure='median',
                                                   timer_in_seconds=43200,
                                                   force_target_type=None,
                                                   plot=False,
                                                   output_file_path='data',
                                                   deep_learning_type='batch',
                                                   deep_learning_output_size=None,
                                                   log=False,
                                                   feature_engineer=_feature_engineer,
                                                   sampling_function=None
                                                   )
        _si.optimize()
        _si.save_evolution(si=True,
                           model=True,
                           evolution_history=True,
                           adjustment_history=True,
                           final_adjustment=True
                           )
        _found_pickles: List[bool] = []
        for pickle in ['model', 'GeneticAlgorithm', 'evolution_history', 'adjustment_history', 'final_adjustment']:
            if os.path.isfile('data/{}.p'.format(pickle)):
                _found_pickles.append(True)
            else:
                _found_pickles.append(False)
        self.assertTrue(expr=all(_found_pickles))


if __name__ == '__main__':
    unittest.main()