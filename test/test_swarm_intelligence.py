import numpy as np
import os
import pandas as pd
import unittest

from happy_learning.feature_engineer import FeatureEngineer
from happy_learning.neural_network_generator_torch import NETWORK_TYPE
from happy_learning.sampler import MLSampler
from happy_learning.supervised_machine_learning import CLF_ALGORITHMS, REG_ALGORITHMS
from happy_learning.swarm_intelligence import SwarmIntelligence
from happy_learning.text_clustering_generator import CLUSTER_ALGORITHMS
from typing import List

DATA_DIR: str = 'data/'
DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='{}avocado.csv'.format(DATA_DIR))
DATA_FILE_PATH_CLUSTER: str = 'data/tweets_sample.csv'
PREDICTOR_CLUSTER: str = 'tweet'
pd.read_csv(filepath_or_buffer='data/tweets.csv', sep=',').loc[0:1000, ].to_csv(path_or_buf=DATA_FILE_PATH_CLUSTER, index=False)
N_CLUSTERS: int = 10
TARGET: str = 'AveragePrice'
PREDICTORS: List[str] = ['4046', '4225', '4770']
TARGET_TEXT: str = 'label'
PREDICTORS_TEXT: List[str] = ['text']
DATA_SET_REG: pd.DataFrame = pd.read_csv(filepath_or_buffer='data/avocado.csv').loc[0:1000, ]
TRAIN_TEST_REG: dict = MLSampler(df=DATA_SET_REG,
                                 target=TARGET,
                                 features=PREDICTORS,
                                 train_size=0.8,
                                 random_sample=True,
                                 stratification=False,
                                 seed=1234
                                 ).train_test_sampling(validation_split=0.1)
TRAIN_DATA_REG_PATH: str = 'data/reg_train.csv'
TEST_DATA_REG_PATH: str = 'data/reg_test.csv'
VALIDATION_DATA_REG_PATH: str = 'data/reg_val.csv'
TRAIN_DATA_PATH: str = 'data/text_train.csv'
TEST_DATA_PATH: str = 'data/text_test.csv'
VALIDATION_DATA_PATH: str = 'data/text_val.csv'
pd.concat(objs=[pd.DataFrame(data=TRAIN_TEST_REG.get('x_train')), pd.DataFrame(data=TRAIN_TEST_REG.get('y_train'))], axis=1).to_csv(path_or_buf=TRAIN_DATA_REG_PATH, index=False)
pd.concat(objs=[pd.DataFrame(data=TRAIN_TEST_REG.get('x_test')), pd.DataFrame(data=TRAIN_TEST_REG.get('y_test'))], axis=1).to_csv(path_or_buf=TEST_DATA_REG_PATH, index=False)
pd.concat(objs=[pd.DataFrame(data=TRAIN_TEST_REG.get('x_val')), pd.DataFrame(data=TRAIN_TEST_REG.get('y_val'))], axis=1).to_csv(path_or_buf=VALIDATION_DATA_REG_PATH, index=False)
DATA_SET_TEXT_CLF: pd.DataFrame = pd.read_csv(filepath_or_buffer='data/tripadvisor_hotel_reviews.csv').loc[0:100, ]
DATA_SET_TEXT_CLF = DATA_SET_TEXT_CLF.loc[~DATA_SET_TEXT_CLF.isnull().any(axis=1), :]
UNIQUE_LABELS: int = 3 # len(DATA_SET_TEXT_CLF['label'].unique())
TRAIN_TEST_TEXT_CLF: dict = MLSampler(df=DATA_SET_TEXT_CLF,
                                      target=TARGET_TEXT,
                                      features=PREDICTORS_TEXT,
                                      train_size=0.8,
                                      random_sample=True,
                                      stratification=False,
                                      seed=1234
                                      ).train_test_sampling(validation_split=0.1)
TRAIN_DATA_PATH_TEXT: str = 'data/text_train.csv'
TEST_DATA_PATH_TEXT: str = 'data/text_test.csv'
VALIDATION_DATA_PATH_TEXT: str = 'data/text_val.csv'
#pd.concat(objs=[pd.DataFrame(data=TRAIN_TEST_TEXT_CLF.get('x_train')), pd.DataFrame(data=TRAIN_TEST_TEXT_CLF.get('y_train'))], axis=1).to_csv(path_or_buf=TRAIN_DATA_PATH_TEXT, index=False)
#pd.concat(objs=[pd.DataFrame(data=TRAIN_TEST_TEXT_CLF.get('x_test')), pd.DataFrame(data=TRAIN_TEST_TEXT_CLF.get('y_test'))], axis=1).to_csv(path_or_buf=TEST_DATA_PATH_TEXT, index=False)
#pd.concat(objs=[pd.DataFrame(data=TRAIN_TEST_TEXT_CLF.get('x_val')), pd.DataFrame(data=TRAIN_TEST_TEXT_CLF.get('y_val'))], axis=1).to_csv(path_or_buf=VALIDATION_DATA_PATH_TEXT, index=False)


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
                                                   models=None,
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

    def test_optimize_modeling_text_classification(self):
        _si: SwarmIntelligence = SwarmIntelligence(mode='model',
                                                   target=TARGET_TEXT,
                                                   input_file_path=None,
                                                   train_data_file_path=TRAIN_DATA_PATH_TEXT,
                                                   test_data_file_path=TEST_DATA_PATH_TEXT,
                                                   valid_data_file_path=VALIDATION_DATA_PATH_TEXT,
                                                   df=None,
                                                   data_set=None,
                                                   features=PREDICTORS_TEXT,
                                                   re_split_data=False,
                                                   re_sample_cases=False,
                                                   re_sample_features=False,
                                                   re_populate=True,
                                                   max_trials=2,
                                                   max_features=-1,
                                                   labels=None,
                                                   models=['trans'],
                                                   model_params=None,
                                                   burn_in_adjustments=-1,
                                                   warm_start=True,
                                                   warm_start_strategy='monotone',
                                                   warm_start_constant_hidden_layers=0,
                                                   warm_start_constant_category='very_small',
                                                   max_adjustments=1,
                                                   pop_size=4,
                                                   adjustment_rate=0.5,
                                                   adjustment_prob=0.85,
                                                   early_stopping=0,
                                                   convergence=True,
                                                   convergence_measure='median',
                                                   timer_in_seconds=43200,
                                                   force_target_type=None,
                                                   plot=False,
                                                   output_file_path='data',
                                                   deep_learning_type='batch',
                                                   deep_learning_output_size=UNIQUE_LABELS,
                                                   log=False,
                                                   feature_engineer=None,
                                                   sampling_function=None,
                                                   **dict(sep=',')
                                                   )
        _si.optimize()
        self.assertTrue(expr=_si.evolution_gradient.get('max')[0] <= _si.evolution_gradient.get('max')[-1])

    def test_optimize_modeling_text_clustering(self):
        _si: SwarmIntelligence = SwarmIntelligence(mode='model',
                                                   target=None,
                                                   input_file_path=None,
                                                   train_data_file_path=DATA_FILE_PATH_CLUSTER,
                                                   test_data_file_path=None,
                                                   valid_data_file_path=None,
                                                   df=None,
                                                   data_set=None,
                                                   features=[PREDICTOR_CLUSTER],
                                                   re_split_data=False,
                                                   re_sample_cases=False,
                                                   re_sample_features=False,
                                                   re_populate=True,
                                                   max_trials=2,
                                                   max_features=-1,
                                                   labels=None,
                                                   models=['gsdmm'],#[np.random.choice(a=list(CLUSTER_ALGORITHMS.keys()))],
                                                   model_params=None,
                                                   burn_in_adjustments=-1,
                                                   warm_start=True,
                                                   warm_start_strategy='monotone',
                                                   warm_start_constant_hidden_layers=0,
                                                   warm_start_constant_category='very_small',
                                                   max_adjustments=2,
                                                   pop_size=3,
                                                   adjustment_rate=0.5,
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
                                                   feature_engineer=None,
                                                   sampling_function=None,
                                                   **dict(sep=',', tokenize=True)
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
