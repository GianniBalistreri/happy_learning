import pandas as pd
import unittest

from happy_learning.feature_engineer import FeatureEngineer
from happy_learning.feature_learning import FeatureLearning
from happy_learning.genetic_algorithm import GeneticAlgorithm

DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='data/avocado.csv')


class FeatureLearningTest(unittest.TestCase):
    """
    Class for testing class FeatureLearning
    """
    def test_ga_clf(self):
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=DATA_SET, target_feature='type')
        _feature_engineer.set_predictors(exclude=None, exclude_original_data=False)
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
        _feature_learning: FeatureLearning = FeatureLearning(feature_engineer=_feature_engineer,
                                                             df=None,
                                                             file_path=None,
                                                             target=_feature_engineer.get_target(),
                                                             force_target_type=None,
                                                             max_features=-1,
                                                             keep_fittest_only=True,
                                                             train_categorical_critic=False,
                                                             train_continuous_critic=False,
                                                             engineer_time_disparity=True,
                                                             engineer_categorical=True,
                                                             engineer_text=True,
                                                             output_path='data'
                                                             )
        _feature_learning_engineer = _feature_learning.ga()
        _ga_using_new_features: GeneticAlgorithm = GeneticAlgorithm(mode='model',
                                                                    target=_feature_learning_engineer.get_target(),
                                                                    input_file_path=None,
                                                                    train_data_file_path=None,
                                                                    test_data_file_path=None,
                                                                    valid_data_file_path=None,
                                                                    df=None,
                                                                    data_set=None,
                                                                    features=_feature_learning_engineer.get_predictors(),
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
                                                                    convergence_measure='min',
                                                                    timer_in_seconds=43200,
                                                                    force_target_type=None,
                                                                    plot=False,
                                                                    output_file_path='data',
                                                                    deep_learning_type='batch',
                                                                    deep_learning_output_size=None,
                                                                    log=False,
                                                                    feature_engineer=_feature_learning_engineer,
                                                                    sampling_function=None
                                                                    )
        _ga_using_new_features.optimize()
        self.assertTrue(expr=_ga_using_new_features.final_generation[_ga_using_new_features.best_individual_idx]['fitness_score'] >= _ga.final_generation[_ga.best_individual_idx]['fitness_score'])

    def test_ga_reg(self):
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=DATA_SET, target_feature='AveragePrice')
        _feature_engineer.set_predictors(exclude=None, exclude_original_data=False)
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
        _feature_learning: FeatureLearning = FeatureLearning(feature_engineer=_feature_engineer,
                                                             df=None,
                                                             file_path=None,
                                                             target=_feature_engineer.get_target(),
                                                             force_target_type=None,
                                                             max_features=-1,
                                                             keep_fittest_only=True,
                                                             train_categorical_critic=False,
                                                             train_continuous_critic=False,
                                                             engineer_time_disparity=True,
                                                             engineer_categorical=True,
                                                             engineer_text=True,
                                                             output_path='data'
                                                             )
        _feature_learning_engineer = _feature_learning.ga()
        _ga_using_new_features: GeneticAlgorithm = GeneticAlgorithm(mode='model',
                                                                    target=_feature_learning_engineer.get_target(),
                                                                    input_file_path=None,
                                                                    train_data_file_path=None,
                                                                    test_data_file_path=None,
                                                                    valid_data_file_path=None,
                                                                    df=None,
                                                                    data_set=None,
                                                                    features=_feature_learning_engineer.get_predictors(),
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
                                                                    convergence_measure='min',
                                                                    timer_in_seconds=43200,
                                                                    force_target_type=None,
                                                                    plot=False,
                                                                    output_file_path='data',
                                                                    deep_learning_type='batch',
                                                                    deep_learning_output_size=None,
                                                                    log=False,
                                                                    feature_engineer=_feature_learning_engineer,
                                                                    sampling_function=None
                                                                    )
        _ga_using_new_features.optimize()
        self.assertTrue(expr=_ga_using_new_features.final_generation[_ga_using_new_features.best_individual_idx]['fitness_score'] <= _ga.final_generation[_ga.best_individual_idx]['fitness_score'])


if __name__ == '__main__':
    unittest.main()
