import os
import pandas as pd
import unittest

from happy_learning.feature_engineer import FeatureEngineer
from happy_learning.genetic_algorithm import GeneticAlgorithm
from typing import List

DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='data/avocado.csv')


class GeneticAlgorithmTest(unittest.TestCase):
    """
    Class for testing class GeneticAlgorithm
    """
    def test_optimize_modeling(self):
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=DATA_SET, target_feature='AveragePrice')
        _feature_engineer.set_predictors(exclude_original_data=False)
        _ga: GeneticAlgorithm = GeneticAlgorithm(mode='model',
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
                                                 burn_in_generations=-1,
                                                 warm_start=True,
                                                 warm_start_strategy='monotone',
                                                 warm_start_constant_hidden_layers=0,
                                                 warm_start_constant_category='very_small',
                                                 max_generations=10,
                                                 pop_size=64,
                                                 mutation_rate=0.1,
                                                 mutation_prob=0.15,
                                                 parents_ratio=0.5,
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
        _ga.optimize()
        self.assertTrue(expr=_ga.evolution_gradient.get('max')[0] <= _ga.evolution_gradient.get('max')[-1])

    def test_optimize_feature_engineering(self):
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=DATA_SET, target_feature='AveragePrice')
        _feature_engineer.set_predictors(exclude_original_data=False)
        _ga: GeneticAlgorithm = GeneticAlgorithm(mode='feature_engineer',
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
                                                 burn_in_generations=-1,
                                                 warm_start=True,
                                                 warm_start_strategy='monotone',
                                                 warm_start_constant_hidden_layers=0,
                                                 warm_start_constant_category='very_small',
                                                 max_generations=5,
                                                 pop_size=64,
                                                 mutation_rate=0.1,
                                                 mutation_prob=0.15,
                                                 parents_ratio=0.5,
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
        _ga.optimize()
        self.assertTrue(expr=_ga.evolution_gradient.get('max')[0] <= _ga.evolution_gradient.get('max')[-1])

    def test_optimize_sampling(self):
        pass

    def test_optimize_continue(self):
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=DATA_SET, target_feature='AveragePrice')
        _feature_engineer.set_predictors(exclude_original_data=False)
        _max_generations: int = 5
        _ga: GeneticAlgorithm = GeneticAlgorithm(mode='model',
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
                                                 burn_in_generations=-1,
                                                 warm_start=True,
                                                 warm_start_strategy='monotone',
                                                 warm_start_constant_hidden_layers=0,
                                                 warm_start_constant_category='very_small',
                                                 max_generations=_max_generations,
                                                 pop_size=64,
                                                 mutation_rate=0.1,
                                                 mutation_prob=0.15,
                                                 parents_ratio=0.5,
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
        _ga.optimize()
        self.assertTrue(expr=_ga.evolution_gradient.get('max')[0] <= _ga.evolution_gradient.get('max')[-1] and len(_ga.evolution_gradient.get('max')) > _max_generations)

    def test_visualize_clf(self):
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=DATA_SET, target_feature='type')
        _feature_engineer.set_predictors(exclude_original_data=False)
        _ga: GeneticAlgorithm = GeneticAlgorithm(mode='model',
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
                                                 burn_in_generations=-1,
                                                 warm_start=True,
                                                 warm_start_strategy='monotone',
                                                 warm_start_constant_hidden_layers=0,
                                                 warm_start_constant_category='very_small',
                                                 max_generations=1,
                                                 pop_size=64,
                                                 mutation_rate=0.1,
                                                 mutation_prob=0.15,
                                                 parents_ratio=0.5,
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
        _ga.optimize()
        _ga.visualize(results_table=False,
                      model_distribution=True,
                      model_evolution=True,
                      param_distribution=False,
                      train_time_distribution=True,
                      breeding_map=True,
                      breeding_graph=False,
                      fitness_distribution=True,
                      fitness_evolution=True,
                      fitness_dimensions=True,
                      per_generation=True,
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
                     'ga_fitness_score_distribution_per_generation',
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
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=DATA_SET, target_feature='AveragePrice')
        _feature_engineer.set_predictors(exclude_original_data=False)
        _ga: GeneticAlgorithm = GeneticAlgorithm(mode='model',
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
                                                 burn_in_generations=-1,
                                                 warm_start=True,
                                                 warm_start_strategy='monotone',
                                                 warm_start_constant_hidden_layers=0,
                                                 warm_start_constant_category='very_small',
                                                 max_generations=5,
                                                 pop_size=4,
                                                 mutation_rate=0.1,
                                                 mutation_prob=0.15,
                                                 parents_ratio=0.5,
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
        _ga.optimize()
        _ga.visualize(results_table=True,
                      model_distribution=True,
                      model_evolution=True,
                      param_distribution=True,
                      train_time_distribution=True,
                      breeding_map=True,
                      breeding_graph=True,
                      fitness_distribution=True,
                      fitness_evolution=True,
                      fitness_dimensions=True,
                      per_generation=True,
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
                     'ga_fitness_score_distribution_per_generation',
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
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=DATA_SET, target_feature='AveragePrice')
        _feature_engineer.set_predictors(exclude_original_data=False)
        _ga: GeneticAlgorithm = GeneticAlgorithm(mode='model',
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
                                                 burn_in_generations=-1,
                                                 warm_start=True,
                                                 warm_start_strategy='monotone',
                                                 warm_start_constant_hidden_layers=0,
                                                 warm_start_constant_category='very_small',
                                                 max_generations=5,
                                                 pop_size=64,
                                                 mutation_rate=0.1,
                                                 mutation_prob=0.15,
                                                 parents_ratio=0.5,
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
        _ga.optimize()
        _ga.save_evolution(ga=True,
                           model=True,
                           evolution_history=True,
                           generation_history=True,
                           final_generation=True
                           )
        _found_pickles: List[bool] = []
        for pickle in ['model', 'GeneticAlgorithm', 'evolution_history', 'generation_history', 'final_generation']:
            if os.path.isfile('data/{}.p'.format(pickle)):
                _found_pickles.append(True)
            else:
                _found_pickles.append(False)
        self.assertTrue(expr=all(_found_pickles))


if __name__ == '__main__':
    unittest.main()
