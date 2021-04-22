import copy
import numpy as np
import os
import pandas as pd
import random
import torch
import warnings

from .evaluate_machine_learning import EvalClf, sml_score
from .neural_network_generator_torch import HIDDEN_LAYER_CATEGORY_EVOLUTION, NetworkGenerator, NETWORK_TYPE, NETWORK_TYPE_CATEGORY
from .sampler import MLSampler
from .supervised_machine_learning import CLF_ALGORITHMS, ModelGeneratorClf, ModelGeneratorReg, REG_ALGORITHMS
from .text_clustering_generator import CLUSTER_ALGORITHMS, ClusteringGenerator
from .utils import HappyLearningUtils
from datetime import datetime
from easyexplore.data_import_export import CLOUD_PROVIDER, DataExporter
from easyexplore.data_visualizer import DataVisualizer
from easyexplore.utils import Log
from multiprocessing.pool import ThreadPool
from typing import Dict, List

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# TODO:
#  1) Visualize:
#   -> Breeding Map & Graph
#   -> Parameter Configuration
#  2) Mode:
#   -> model_sampler


class SwarmIntelligenceException(Exception):
    """
    Class for managing exceptions for class SwarmIntelligence
    """
    pass


class SwarmIntelligence:
    """
    Class for reinforced optimizing machine learning algorithms and feature engineering using Swarm Intelligence
    """
    def __init__(self,
                 mode: str,
                 target: str = None,
                 input_file_path: str = None,
                 train_data_file_path: str = None,
                 test_data_file_path: str = None,
                 valid_data_file_path: str = None,
                 df: pd.DataFrame = None,
                 data_set: dict = None,
                 features: List[str] = None,
                 re_split_data: bool = False,
                 re_sample_cases: bool = False,
                 re_sample_features: bool = False,
                 re_populate: bool = True,
                 max_trials: int = 2,
                 max_features: int = -1,
                 labels: List[str] = None,
                 models: List[str] = None,
                 model_params: Dict[str, str] = None,
                 burn_in_adjustments: int = -1,
                 warm_start: bool = True,
                 warm_start_strategy: str = 'monotone',
                 warm_start_constant_hidden_layers: int = 0,
                 warm_start_constant_category: str = 'very_small',
                 max_adjustments: int = 50,
                 pop_size: int = 64,
                 adjustment_rate: float = 0.1,
                 adjustment_prob: float = 0.85,
                 early_stopping: int = 0,
                 convergence: bool = True,
                 convergence_measure: str = 'min',
                 timer_in_seconds: int = 43200,
                 force_target_type: str = None,
                 plot: bool = False,
                 output_file_path: str = None,
                 include_neural_networks: bool = False,
                 deep_learning_type: str = 'batch',
                 deep_learning_output_size: int = None,
                 cloud: str = None,
                 deploy_model: bool = True,
                 initial_population: list = None,
                 multi_threading: bool = False,
                 multi_processing: bool = False,
                 log: bool = False,
                 verbose: int = 0,
                 feature_engineer=None,
                 fitness_function=sml_score,
                 sampling_function=None,
                 **kwargs
                 ):
        """
        :param mode: str
            Optimization specification
                -> model: Optimize model or hyper parameter set
                -> model_sampler: Optimize model or hyper parameter set and resample data set each mutation
                -> feature_engineer: Optimize feature engineering
                -> feature_selector: Optimize feature selection

        :param input_file_path: str
            Complete file path of input file

        :param df: Pandas DataFrame
            Data set

        :param data_set: dict
            Already split data set into train, test and validation sets
                -> Keys-Naming: x_train / y_train / x_test / y_test / x_val / y_val
                -> Data set could be numpy arrays or images or audio or video streams

        :param target: str
            Name of the target feature

        :param features: List[str]
            Name of the features used as predictors

        :param re_split_data: bool
            Whether to re-split data set into train & test data every adjustment or not

        :param re_sample_cases: bool
            Whether to re-sample cases set every adjustment or not

        :param re_sample_features: bool
            Whether to re-sample features set every adjustment or not

        :param re_populate: bool
            Whether to re-populate adjustment 0 if all models achieve poor fitness results

        :param max_trials: int
            Maximum number of re-population trials for avoiding bad starts

        :param max_features: int
            Number of feature attributes of each individual (if re_sample_features == True or mode != "model")

        :param feature_engineer: object
            FeatureEngineer object for generating features

        :param models: dict
            Machine learning model objects

        :param model_params: dict
            Pre-defined machine learning model parameter config if mode == "feature"

        :param fitness_function: function
            User defined fitness function to evaluate machine learning models

        :param sampling_function: function
            User defined sampling function

        :param genes: dict
            Attributes of the individuals (genes)

        :param max_adjustments: int
            Maximum number of adjustments

        :param pop_size: int
            Population size of each adjustment

        :param adjustment_rate: float
            Adjustment rate

        :param adjustment_prob: float
            Adjustment probability

        :param warm_start: bool
            Whether to start evolution (adjustment 0) using standard parameter config for each model type once

        :param warm_start_strategy: str
            Name of the warm start strategy (deep learning only)
            -> random: Draw size of hidden layers randomly
            -> constant: Use constant size of hidden layers only
            -> monotone: Increase the range of hidden layers each adjustment by one category
            -> adaptive: Increase the range of hidden layers each strong individual mutation by one layer

        :param warm_start_constant_hidden_layers: int
            Number of hidden layers (constant for each individual)

        :param warm_start_constant_category: str
            Name of the hidden layer category (constant for each individual)
                -> very_small: 1 - 2 hidden layers
                -> small: 3 - 4 hidden layers
                -> medium: 5 - 7 hidden layers
                -> big: 8 - 10 hidden layers
                -> very_big: 11+ hidden layers

        :param early_stopping: int
            Number of adjustments for starting early stopping condition checks

        :param convergence: bool
            Whether to check convergence conditions for early stopping or not

        :param convergence_measure: str
            Measurement to use for applying convergence conditions:
                -> min:
                -> median:
                -> max:

        :param timer_in_seconds: int
            Maximum time exceeding to interrupt algorithm

        :param force_target_type: str
            Name of the target type to force (useful if target type is ordinal)
                -> reg: define target type as regression instead of multi classification
                -> clf_multi: define target type as multi classification instead of regression

        :param plot: bool
            Whether to visualize results or not

        :param output_file_path: str
            File path for exporting results (model, visualization, etc.)

        :param include_neural_networks: bool
            Include neural networks in random model sampling workaround for structured (tabular) data

        :param deep_learning_type: str
            Name of the learning type to use to train neural networks:
                -> batch: use hole data set as batches in each epoch
                -> stochastic: use sample of the data set in each epoch

        :param deep_learning_output_size: int
            Number of neurons of the last layer of the neural network (necessary if you have separate train / test / validatation data file as input)

        :param cloud: str
            Name of the cloud provider
                -> google: Google Cloud Storage
                -> aws: AWS Cloud

        :param deploy_model: bool
            Whether to deploy (save) evolved model or not

        :param adjustment_zero: list
            Pre-defined models for initial population

        :param multi_threading: bool
            Whether to run genetic algorithm using multiple threads (of one cpu core) or single thread

        :param multi_processing: bool
            Whether to run genetic algorithm using multiple processes (cpu cores) or single cpu core

        :param log: bool
            Write logging file or just print messages

        :param verbose: int
            Logging level:
                -> 0: Log basic info only
                -> 1: Log all info including algorithm results

        :param kwargs: dict
            Key-word arguments
        """
        self.mode = mode
        self.model = None
        self.model_params: dict = copy.deepcopy(model_params)
        self.deploy_model: bool = deploy_model
        self.n_training: int = 0
        self.cloud: str = cloud
        if self.cloud is None:
            self.bucket_name: str = None
        else:
            if self.cloud not in CLOUD_PROVIDER:
                raise SwarmIntelligenceException('Cloud provider ({}) not supported'.format(cloud))
            if output_file_path is None:
                raise SwarmIntelligenceException('Output file path is None')
            self.bucket_name: str = output_file_path.split("//")[1].split("/")[0]
        self.include_neural_networks: bool = include_neural_networks
        _neural_nets: List[str] = []
        _clustering: List[str] = []
        if models is None:
            self.text_clustering: bool = False
            self.deep_learning: bool = False
            self.models: List[str] = []
        else:
            for model in models:
                if model in NETWORK_TYPE.keys():
                    _neural_nets.append(model)
            for model in models:
                if model in CLUSTER_ALGORITHMS.keys():
                    _clustering.append(model)
            if len(_neural_nets) == 0:
                self.deep_learning: bool = False
                if len(_clustering) == 0:
                    self.text_clustering: bool = False
                    self.models: List[str] = copy.deepcopy(models)
                else:
                    self.text_clustering: bool = True
                    self.models: List[str] = _clustering
            else:
                self.text_clustering: bool = False
                self.deep_learning: bool = True
                self.models: List[str] = _neural_nets
        self.pop_size: int = pop_size if pop_size >= 3 else 64
        self.input_file_path: str = input_file_path
        self.train_data_file_path: str = train_data_file_path
        self.test_data_file_path: str = test_data_file_path
        self.valid_data_file_path: str = valid_data_file_path
        self.df: pd.DataFrame = df
        self.data_set: dict = data_set
        self.feature_engineer = feature_engineer
        self.target: str = target
        self.target_classes: int = 0
        self.target_values: np.array = None
        self.force_target_type: str = force_target_type
        self.features: List[str] = features
        self.max_features: int = max_features if max_features > 0 else len(self.features)
        self.n_cases: int = 0
        self.n_test_cases: int = 0
        self.n_train_cases: int = 0
        self.re_split_data: bool = re_split_data
        self.re_sample_cases: bool = re_sample_cases
        self.re_sample_features: bool = re_sample_features
        self.deep_learning_output_size: int = deep_learning_output_size
        if output_file_path is None:
            self.output_file_path: str = ''
        else:
            self.output_file_path: str = output_file_path.replace('\\', '/')
            if self.output_file_path[len(self.output_file_path) - 1] != '/':
                self.output_file_path = '{}/'.format(self.output_file_path)
        self.sampling_function = sampling_function
        self.kwargs: dict = kwargs
        self._input_manager()
        self.target_labels: List[str] = labels
        self.log: bool = log
        self.verbose: int = verbose
        self.warm_start: bool = warm_start
        self.warm_start_strategy: str = warm_start_strategy if warm_start_strategy in HIDDEN_LAYER_CATEGORY_EVOLUTION else 'monotone'
        self.warm_start_constant_hidden_layers: int = warm_start_constant_hidden_layers if warm_start_constant_hidden_layers > 0 else 0
        self.warm_start_constant_category: str = warm_start_constant_category if warm_start_constant_category in list(NETWORK_TYPE_CATEGORY.keys()) else 'very_small'
        self.re_populate: bool = re_populate
        self.max_trials: int = max_trials
        self.max_adjustments: int = max_adjustments if max_adjustments >= 0 else 50
        self.burn_in_adjustments: int = burn_in_adjustments if burn_in_adjustments >= 0 else round(0.1 * self.max_adjustments)
        self.population: List[object] = []
        self.adjustment_rate: float = adjustment_rate if adjustment_rate > 0 or adjustment_rate <= 1 else 0.1
        self.adjustment_prob: float = adjustment_prob if adjustment_prob > 0 or adjustment_prob <= 1 else 0.85
        self.plot: bool = plot
        self.fitness_function = fitness_function
        self.deep_learning_type: str = deep_learning_type
        self.initial_population: list = initial_population
        self.n_threads: int = self.pop_size
        self.multi_threading: bool = multi_threading
        self.multi_processing: bool = multi_processing
        self.n_individuals: int = -1
        self.best_global_idx: int = -1
        self.best_local_idx: int = -1
        self.best_global_local_idx: List[int] = []
        self.final_adjustment: dict = {}
        self.evolution: dict = {}
        self.evolved_features: List[str] = []
        self.adjusted_features: dict = dict(best=[], to=[], fitness=[], adjustment=[], action=[])
        self.current_adjustment_meta_data: dict = dict(adjustment=0,
                                                       id=[],
                                                       fitness_metric=[],
                                                       fitness_score=[],
                                                       model_name=[],
                                                       param=[],
                                                       param_adjusted=[],
                                                       features=[]
                                                       )
        self.adjustment_history: dict = dict(population={}, inheritance={}, time=[])
        self.evolution_history: dict = dict(id=[],
                                            model=[],
                                            adjustment=[],
                                            training=[],
                                            best=[],
                                            fitness_score=[],
                                            ml_metric=[],
                                            train_test_diff=[],
                                            train_time_in_seconds=[],
                                            original_ml_train_metric=[],
                                            original_ml_test_metric=[]
                                            )
        self.evolution_gradient: dict = dict(min=[], median=[], mean=[], max=[])
        self.evolution_continue: bool = False
        self.convergence: bool = convergence
        self.convergence_measure: str = convergence_measure
        self.early_stopping: int = early_stopping if early_stopping >= 0 else 0
        self.timer: int = timer_in_seconds if timer_in_seconds > 0 else 99999
        self._intro()
        self.start_time: datetime = datetime.now()

    def _adjust(self):
        """
        Adjust population towards best global and local individual
        """
        for idx in range(0, self.pop_size, 1):
            if idx != self.best_global_idx and idx != self.best_local_idx:
                if self.mode.find('model') >= 0:
                    if np.random.uniform(low=0, high=1) > self.adjustment_prob:
                        if self.mode == 'model_sampler':
                            self._sampling(features=self.population[idx].features)
                        if self.text_clustering:
                            self.population[idx] = ClusteringGenerator(predictor=self.features[0],
                                                                       models=self.models,
                                                                       model_name=self.models[0],
                                                                       tokenize=False if self.kwargs.get('tokenize') else self.kwargs.get('tokenize'),
                                                                       cloud=self.cloud,
                                                                       train_data_path=self.train_data_file_path,
                                                                       sep='\t' if self.kwargs.get('sep') is None else self.kwargs.get('sep'),
                                                                       sentence_embedding_model_path=self.kwargs.get('sentence_embedding_model_path')
                                                                       ).generate_model()
                        else:
                            if self.deep_learning:
                                #if self.warm_start_strategy == 'adaptive':
                                _hidden_layer_size: int = self.population[idx].hidden_layer_size
                                self.population[idx] = NetworkGenerator(target=self.target,
                                                                        predictors=self.features,
                                                                        output_layer_size=self.deep_learning_output_size,
                                                                        x_train=self.data_set.get('x_train').values if self.data_set is not None else self.data_set,
                                                                        y_train=self.data_set.get('y_train').values if self.data_set is not None else self.data_set,
                                                                        x_test=self.data_set.get('x_test').values if self.data_set is not None else self.data_set,
                                                                        y_test=self.data_set.get('y_test').values if self.data_set is not None else self.data_set,
                                                                        x_val=self.data_set.get('x_val').values if self.data_set is not None else self.data_set,
                                                                        y_val=self.data_set.get('y_val').values if self.data_set is not None else self.data_set,
                                                                        train_data_path=self.train_data_file_path,
                                                                        test_data_path=self.test_data_file_path,
                                                                        validation_data_path=self.valid_data_file_path,
                                                                        models=self.models,
                                                                        hidden_layer_size=_hidden_layer_size,
                                                                        hidden_layer_size_category=self.warm_start_constant_category,
                                                                        sep='\t' if self.kwargs.get('sep') is None else self.kwargs.get('sep'),
                                                                        cache_dir=self.kwargs.get('cache_dir')
                                                                        ).generate_model()
                            else:
                                self.population[idx] = ModelGeneratorReg(models=self.models).generate_model() if self.target_type == 'reg' else ModelGeneratorClf(models=self.models).generate_model()
                    else:
                        if self.text_clustering:
                            self.population[idx] = ClusteringGenerator(predictor=self.features[0],
                                                                       models=self.models,
                                                                       model_name=self.models[0],
                                                                       cluster_params=self.population[self.best_global_idx].model_param,
                                                                       tokenize=False if self.kwargs.get('tokenize') else self.kwargs.get('tokenize'),
                                                                       cloud=self.cloud,
                                                                       train_data_path=self.train_data_file_path,
                                                                       sep='\t' if self.kwargs.get('sep') is None else self.kwargs.get('sep'),
                                                                       sentence_embedding_model_path=self.kwargs.get('sentence_embedding_model_path')
                                                                       ).generate_model()
                            self.population[idx].generate_params(param_rate=self.adjustment_rate)
                        else:
                            if self.deep_learning:
                                self.population[idx] = NetworkGenerator(target=self.target,
                                                                        predictors=self.features,
                                                                        output_layer_size=self.deep_learning_output_size,
                                                                        x_train=self.data_set.get(
                                                                            'x_train').values if self.data_set is not None else self.data_set,
                                                                        y_train=self.data_set.get(
                                                                            'y_train').values if self.data_set is not None else self.data_set,
                                                                        x_test=self.data_set.get(
                                                                            'x_test').values if self.data_set is not None else self.data_set,
                                                                        y_test=self.data_set.get(
                                                                            'y_test').values if self.data_set is not None else self.data_set,
                                                                        x_val=self.data_set.get(
                                                                            'x_val').values if self.data_set is not None else self.data_set,
                                                                        y_val=self.data_set.get(
                                                                            'y_val').values if self.data_set is not None else self.data_set,
                                                                        train_data_path=self.train_data_file_path,
                                                                        test_data_path=self.test_data_file_path,
                                                                        validation_data_path=self.valid_data_file_path,
                                                                        models=self.models,
                                                                        input_param=self.population[idx].model_param,
                                                                        hidden_layer_size=self.population[idx].hidden_layer_size,
                                                                        hidden_layer_size_category=self.warm_start_constant_category,
                                                                        sep='\t' if self.kwargs.get('sep') is None else self.kwargs.get('sep'),
                                                                        cache_dir=self.kwargs.get('cache_dir')
                                                                        ).generate_model()
                                self.population[idx].generate_params(param_rate=self.adjustment_rate)
                            else:
                                self.population[idx] = ModelGeneratorReg(reg_params=self.population[self.best_global_idx].model_param,
                                                                         models=self.models,
                                                                         model_name=self.models[0]
                                                                         ).generate_model() if self.target_type == 'reg' else ModelGeneratorClf(clf_params=self.population[self.best_global_idx].model_param, models=self.models).generate_model()
                                self.population[idx].generate_params(param_rate=self.adjustment_rate)
                elif self.mode.find('feature') >= 0:
                    _new_features: List[str] = []
                    _feature_pool: List[str] = list(set(self.feature_pairs[np.random.choice(a=[self.best_global_idx, self.best_local_idx])]))
                    for feature in self.feature_pairs[idx]:
                        if self.mode == 'feature_engineer':
                            if np.random.uniform(low=0, high=1) > self.adjustment_prob:
                                self.feature_engineer.act(actor=feature,
                                                          inter_actors=_feature_pool,
                                                          force_action=None,
                                                          alternative_actions=None
                                                          )
                                _generated_feature: str = self.feature_engineer.get_last_generated_feature()
                                if _generated_feature == '':
                                    _new_features.append(feature)
                                else:
                                    _new_features.append(_generated_feature)
                                self.adjusted_features['best'].append(feature)
                                self.adjusted_features['to'].append(_new_features[-1])
                                self.adjusted_features['adjustment'].append(feature)
                                self.adjusted_features['action'].append(self.feature_engineer.get_last_action())
                            else:
                                _new_features.append(feature)
                        elif self.mode == 'feature_selector':
                            _new_features.append(feature)
                    self.feature_pairs[idx] = copy.deepcopy(_new_features)
                    # print('mutated new child', self.feature_pairs[child])

    def _collect_meta_data(self, current_adjustment: bool, idx: int = None):
        """
        Collect evolution meta data

        :param current_adjustment: bool
            Whether to write evolution meta data of each individual of current adjustment or not

        :param idx: int
            Index number of individual within population
        """
        if self.adjustment_history['population'].get('adjustment_{}'.format(self.current_adjustment_meta_data['adjustment'])) is None:
            self.adjustment_history['population'].update(
                {'adjustment_{}'.format(self.current_adjustment_meta_data['adjustment']): dict(id=[],
                                                                                               model=[],
                                                                                               best=[],
                                                                                               fitness=[]
                                                                                               )
                 })
        if current_adjustment:
            setattr(self.population[idx], 'fitness_score', self.evolution_history.get('fitness_score')[self.population[idx].id])
            if not self.deep_learning and not self.text_clustering:
                setattr(self.population[idx], 'features', list(self.data_set.get('x_train').columns))
            if self.current_adjustment_meta_data['adjustment'] == 0:
                self.current_adjustment_meta_data.get('id').append(copy.deepcopy(idx))
                if not self.deep_learning and not self.text_clustering:
                    self.current_adjustment_meta_data.get('features').append(copy.deepcopy(self.population[idx].features))
                self.current_adjustment_meta_data.get('model_name').append(copy.deepcopy(self.population[idx].model_name))
                self.current_adjustment_meta_data.get('param').append(copy.deepcopy(self.population[idx].model_param))
                self.current_adjustment_meta_data.get('param_adjusted').append(copy.deepcopy(self.population[idx].model_param_mutated))
                self.current_adjustment_meta_data.get('fitness_metric').append(copy.deepcopy(self.population[idx].fitness))
                self.current_adjustment_meta_data.get('fitness_score').append(copy.deepcopy(self.population[idx].fitness_score))
            else:
                self.current_adjustment_meta_data['id'][idx] = copy.deepcopy(self.population[idx].id)
                if not self.deep_learning and not self.text_clustering:
                    self.current_adjustment_meta_data['features'][idx] = copy.deepcopy(self.population[idx].features)
                self.current_adjustment_meta_data['model_name'][idx] = copy.deepcopy(self.population[idx].model_name)
                self.current_adjustment_meta_data['param'][idx] = copy.deepcopy(self.population[idx].model_param)
                self.current_adjustment_meta_data['param_adjusted'][idx] = copy.deepcopy(self.population[idx].model_param_mutated)
                self.current_adjustment_meta_data['fitness_metric'][idx] = copy.deepcopy(self.population[idx].fitness)
                self.current_adjustment_meta_data['fitness_score'][idx] = copy.deepcopy(self.population[idx].fitness_score)
        else:
            if idx is None:
                self.adjustment_history['population']['adjustment_{}'.format(self.current_adjustment_meta_data['adjustment'])]['fitness'] = copy.deepcopy(self.current_adjustment_meta_data.get('fitness'))
                self.evolution_gradient.get('min').append(copy.deepcopy(min(self.current_adjustment_meta_data.get('fitness_score'))))
                self.evolution_gradient.get('median').append(copy.deepcopy(np.median(self.current_adjustment_meta_data.get('fitness_score'))))
                self.evolution_gradient.get('mean').append(copy.deepcopy(np.mean(self.current_adjustment_meta_data.get('fitness_score'))))
                self.evolution_gradient.get('max').append(copy.deepcopy(max(self.current_adjustment_meta_data.get('fitness_score'))))
                Log(write=self.log, logger_file_path=self.output_file_path).log(
                    'Fitness: Max -> {}'.format(self.evolution_gradient.get('max')[-1]))
                Log(write=self.log, logger_file_path=self.output_file_path).log(
                    'Fitness: Median -> {}'.format(self.evolution_gradient.get('median')[-1]))
                Log(write=self.log, logger_file_path=self.output_file_path).log(
                    'Fitness: Mean -> {}'.format(self.evolution_gradient.get('mean')[-1]))
                Log(write=self.log, logger_file_path=self.output_file_path).log(
                    'Fitness: Min -> {}'.format(self.evolution_gradient.get('min')[-1]))
            else:
                if self.current_adjustment_meta_data['adjustment'] == 0:
                    self.evolution_history.get('best').append(-1)
                else:
                    self.evolution_history.get('best').append(copy.deepcopy(self.population[idx].id))
                self.adjustment_history['population']['adjustment_{}'.format(self.current_adjustment_meta_data['adjustment'])]['best'].append(copy.deepcopy(self.evolution_history.get('best')[-1]))
                self.n_individuals += 1
                setattr(self.population[idx], 'id', self.n_individuals)
                if not self.text_clustering:
                    setattr(self.population[idx], 'target', self.target)
                self.evolution_history.get('id').append(copy.deepcopy(self.population[idx].id))
                self.evolution_history.get('adjustment').append(copy.deepcopy(self.current_adjustment_meta_data['adjustment']))
                self.evolution_history.get('model').append(copy.deepcopy(self.population[idx].model_name))
                self.evolution_history.get('training').append(copy.deepcopy(self.n_training))
                self.adjustment_history['population']['adjustment_{}'.format(self.current_adjustment_meta_data['adjustment'])]['id'].append(copy.deepcopy(self.population[idx].id))
                self.adjustment_history['population']['adjustment_{}'.format(self.current_adjustment_meta_data['adjustment'])]['model'].append(copy.deepcopy(self.population[idx].model_name))

    def _fitness(self, individual: object, ml_metric: str):
        """
        Calculate fitness metric for evaluate individual ability to survive

        :param individual: object
            Object of individual to evaluating fitness metric

        :param ml_metric: str
            Name of the machine learning metric
                -> Regression - rmse_norm: Root-Mean-Squared Error normalized by standard deviation
                -> Classification Binary - auc: Area-Under-Curve (AUC)
                                           f1: F1-Score
                                           recall: Recall
                                           accuracy: Accuracy
                -> Classification Multi - auc: Area-Under-Curve (AUC) multi classes summarized
                                          auc_multi: Area-Under-Curve (AUC) multi classes separately
                                          cohen_kappa: Cohen's Kappa
                -> Clustering - nmi: Normalized Mutual Information
        """
        _best_score: float = 0.0 if ml_metric == 'rmse_norm' else 1.0
        _ml_metric: str = 'roc_auc' if ml_metric == 'auc' else ml_metric
        if self.fitness_function.__name__ == 'sml_score':
            if self.text_clustering:
                _scores: dict = dict(fitness_score=individual.fitness)
            else:
                _scores: dict = sml_score(ml_metric=tuple([_best_score, individual.fitness['test'].get(_ml_metric)]),
                                          train_test_metric=tuple([individual.fitness['train'].get(_ml_metric),
                                                                   individual.fitness['test'].get(_ml_metric)]
                                                                  ),
                                          train_time_in_seconds=individual.train_time
                                          )
        else:
            _scores: dict = self.fitness_function(**dict(ml_metric=tuple([_best_score, individual.fitness['test'].get(_ml_metric)]),
                                                         train_test_metric=tuple([individual.fitness['train'].get(_ml_metric),
                                                                                  individual.fitness['test'].get(_ml_metric)
                                                                                  ]),
                                                         train_time_in_seconds=individual.train_time,
                                                         )
                                                  )
        for score in _scores.keys():
            self.evolution_history.get(score).append(copy.deepcopy(_scores.get(score)))

    def _gather_final_adjustment(self):
        """
        Gather information about each individual of final adjustment
        """
        for i, individual in enumerate(self.population):
            self.final_adjustment.update({i: dict(id=copy.deepcopy(individual.id),
                                                  model_name=copy.deepcopy(individual.model_name),
                                                  param=copy.deepcopy(individual.model_param),
                                                  fitness=copy.deepcopy(individual.fitness),
                                                  fitness_score=copy.deepcopy(individual.fitness_score),
                                                  hidden_layer_size=copy.deepcopy(individual.hidden_layer_size) if self.deep_learning else None
                                                  )
                                          })

    def _input_manager(self):
        """
        Manage input options
        """
        _train_size: float = 0.8 if self.kwargs.get('train_size') is None else self.kwargs.get('train_size')
        if self.mode in ['feature_engineer', 'feature_selector', 'model']:
            if self.mode.find('feature') >= 0:
                if self.target not in self.feature_engineer.get_features():
                    raise SwarmIntelligenceException('Target feature ({}) not found in data set'.format(self.target))
                self.target_values: np.array = self.feature_engineer.get_target_values()
                if self.mode == 'feature_engineer':
                    if self.feature_engineer is None:
                        raise SwarmIntelligenceException('FeatureEngineer object not found')
                    else:
                        self.feature_engineer.activate_actor()
                        self.n_cases = self.feature_engineer.get_n_cases()
                        self.n_test_cases: int = round(self.n_cases * (1 - _train_size))
                        self.n_train_cases: int = round(self.n_cases * _train_size)
                        self.feature_pairs: list = [random.sample(self.feature_engineer.get_predictors(), self.max_features) for _ in range(0, self.pop_size, 1)]
                elif self.mode == 'feature_selector':
                    self.feature_pairs: list = [random.sample(self.features, self.max_features) for _ in range(0, self.pop_size, 1)]
            else:
                if self.df is None:
                    if self.feature_engineer is None:
                        if self.data_set is None:
                            if self.text_clustering:
                                if self.train_data_file_path is None:
                                    raise SwarmIntelligenceException('No training data set found')
                            else:
                                if self.train_data_file_path is None or self.test_data_file_path is None or self.valid_data_file_path is None:
                                    raise SwarmIntelligenceException('No training, testing, validation data set found')
                        else:
                            if self.text_clustering:
                                if 'x_train' not in self.data_set.keys():
                                    raise SwarmIntelligenceException('x_train not found in data dictionary')
                            else:
                                if 'x_train' not in self.data_set.keys():
                                    raise SwarmIntelligenceException('x_train not found in data dictionary')
                                if 'y_train' not in self.data_set.keys():
                                    raise SwarmIntelligenceException('y_train not found in data dictionary')
                                if 'x_test' not in self.data_set.keys():
                                    raise SwarmIntelligenceException('x_test not found in data dictionary')
                                if 'y_test' not in self.data_set.keys():
                                    raise SwarmIntelligenceException('y_test not found in data dictionary')
                    else:
                        self.df = self.feature_engineer.get_training_data()
                        self.target = self.feature_engineer.get_target()
                        self.target_values = self.feature_engineer.get_target_values()
                        self.features = self.feature_engineer.get_predictors()
                        self.n_cases = self.feature_engineer.get_n_cases()
                        self.n_test_cases: int = round(self.n_cases * (1 - _train_size))
                        self.n_train_cases: int = round(self.n_cases * _train_size)
                else:
                    if self.target not in self.df.columns and not self.text_clustering:
                        raise SwarmIntelligenceException('Target feature ({}) not found in data set'.format(self.target))
                    if self.features is None:
                        self.features = list(self.df.columns)
                        if not self.text_clustering:
                            del self.features[self.features.index(self.target)]
                    if not self.text_clustering:
                        self.df = self.df[self.features + [self.target]]
                        self.target_values: np.array = self.df[self.target].unique()
                        self.feature_pairs = None
                        self.n_cases = len(self.df)
                        self.n_test_cases: int = round(self.n_cases * (1 - _train_size))
                        self.n_train_cases: int = round(self.n_cases * _train_size)
                if self.re_sample_features:
                    if self.text_clustering:
                        _features: List[str] = []
                    else:
                        _features: List[str] = random.sample(self.features, self.max_features)
                else:
                    _features: List[str] = self.features
                if self.data_set is None:
                    if self.deep_learning:
                        self.n_cases = 0
                        self.n_test_cases = 0
                        self.n_train_cases = 0
                    else:
                        if self.text_clustering:
                            self.n_cases = 0
                            self.n_test_cases = 0
                            self.n_train_cases = 0
                        else:
                            self._sampling(features=_features)
                            self.n_cases = len(self.df)
                            self.n_test_cases: int = len(self.data_set['x_test'])
                            self.n_train_cases: int = len(self.data_set['x_train'])
        else:
            raise SwarmIntelligenceException('Optimization mode ({}) not supported. Use "model", "feature_engineer" or "feature_selector" instead.'.format(self.mode))
        if self.deep_learning:
            if self.target_values is None:
                self.target_classes = self.deep_learning_output_size
                if self.deep_learning_output_size is None:
                    raise SwarmIntelligenceException('Size of the output layer of the neural network is missing')
                else:
                    if self.deep_learning_output_size < 0:
                        raise SwarmIntelligenceException('Size of the output layer of the neural network is missing')
                    elif self.deep_learning_output_size == 1:
                        self.target_type: str = 'reg'
                    elif self.deep_learning_output_size == 2:
                        self.target_type: str = 'clf_binary'
                    else:
                        self.target_type: str = 'clf_multi'
            else:
                self.target_classes = len(self.target_values)
                self.target_type: str = HappyLearningUtils().get_ml_type(values=self.target_values)
                if self.target_type == 'reg':
                    self.deep_learning_output_size = 1
                else:
                    self.deep_learning_output_size = self.target_classes
        else:
            if self.text_clustering:
                self.target_type: str = 'cluster'
            else:
                self.target_classes = len(self.target_values)
                self.target_type: str = HappyLearningUtils().get_ml_type(values=self.target_values)
                if self.models is None and self.target_type == 'reg':
                    self.models = np.random.choice(a=list(REG_ALGORITHMS.keys()))
                if self.models is None and self.target_type != 'reg':
                    self.models = np.random.choice(a=list(CLF_ALGORITHMS.keys()))
        if self.force_target_type is not None:
            if self.force_target_type == 'reg' and self.target_type == 'clf_multi':
                self.target_type = 'reg'
            elif self.force_target_type == 'clf_multi' and self.target_type == 'reg':
                self.target_type = 'clf_multi'
        if self.pop_size is None:
            self.pop_size = 64

    def _intro(self):
        """
        Print Genetic Algorithm configuration
        """
        _intro: str = 'Reinforcement learning environment started ...\nSwarm Intelligence: Optimizing mode -> {}\n' \
                      'Environment setup:\n-> Machine learning model(s): {}\n-> Model parameter: {}\n' \
                      '-> Population: {}\n-> Adjustment Rate: {}\n-> Adjustment Probability: {}\n' \
                      '-> Early stopping: {}\n-> Convergence: {}\n-> Target feature: {}\n-> Target type: {}\n' \
                      '-> Features per model: {}\n'.format(self.mode,
                                                           self.models,
                                                           self.model_params,
                                                           self.pop_size,
                                                           self.adjustment_rate,
                                                           self.adjustment_prob,
                                                           self.early_stopping,
                                                           self.convergence,
                                                           self.target,
                                                           self.target_type,
                                                           self.max_features
                                                           )
        Log(write=self.log, logger_file_path=self.output_file_path).log(msg=_intro)

    def _is_gradient_converged(self, compare: str = 'min', threshold: float = 0.05) -> bool:
        """
        Check whether evolutionary gradient has converged into optimum

        :param compare: str
            Measurement to compare maximum fitness score with:
                -> min: Compare maximum and minimum fitness score of adjustment
                -> median: Compare maximum and median fitness score of adjustment
                -> mean: Compare maximum and mean fitness score of adjustment

        :param threshold: float
            Conversion threshold of relative difference between maximum fitness score and comparison fitness score

        :return bool
            Whether to stop evolution because the hole adjustment achieve very similar gradient score or not
        """
        _threshold: float = threshold if threshold > 0 else 0.05
        _threshold_score: float = self.evolution_gradient.get('max')[-1] - (self.evolution_gradient.get('max')[-1] * _threshold)
        if compare == 'median':
            if self.evolution_gradient.get('median')[-1] >= _threshold_score:
                return True
            else:
                return False
        elif compare == 'mean':
            if self.evolution_gradient.get('mean')[-1] >= _threshold_score:
                return True
            else:
                return False
        else:
            if self.evolution_gradient.get('min')[-1] >= _threshold_score:
                return True
            else:
                return False

    def _is_gradient_stagnating(self,
                                min_fitness: bool = True,
                                median_fitness: bool = True,
                                mean_fitness: bool = True,
                                max_fitness: bool = True
                                ) -> bool:
        """
        Check whether evolutionary gradient (best fitness metric of adjustment) has not increased a certain amount of adjustments

        :param min_fitness: bool
            Use minimum fitness score each adjustment to evaluate stagnation

        :param median_fitness: bool
            Use median fitness score each adjustment to evaluate stagnation

        :param mean_fitness: bool
            Use mean fitness score each adjustment to evaluate stagnation

        :param max_fitness: bool
            Use maximum fitness score each adjustment to evaluate stagnation

        :return bool
            Whether to stop evolution early because of the stagnation of gradient or not
        """
        _gradients: int = 0
        _stagnating: int = 0
        if min_fitness:
            _gradients += 1
            _stagnating = int(len(self.evolution_gradient.get('min')) - np.array(self.evolution_gradient.get('min')).argmax() >= self.early_stopping)
        if median_fitness:
            _gradients += 1
            _stagnating = int(len(self.evolution_gradient.get('median')) - np.array(self.evolution_gradient.get('median')).argmax() >= self.early_stopping)
        if mean_fitness:
            _gradients += 1
            _stagnating = int(len(self.evolution_gradient.get('mean')) - np.array(self.evolution_gradient.get('mean')).argmax() >= self.early_stopping)
        if max_fitness:
            _gradients += 1
            _stagnating = int(len(self.evolution_gradient.get('max')) - np.array(self.evolution_gradient.get('max')).argmax() >= self.early_stopping)
        if _gradients == _stagnating:
            return True
        else:
            return False

    def _modeling(self, pop_idx: int):
        """
        Generate, train and evaluate supervised & unsupervised machine learning model

        :param pop_idx: int
            Population index number
        """
        self._collect_meta_data(current_adjustment=False, idx=pop_idx)
        _re: int = 0
        _re_generate: bool = False
        _re_generate_max: int = 50
        while True:
            _re += 1
            try:
                if self.mode == 'model':
                    if _re_generate:
                        self.population[pop_idx] = copy.deepcopy(self.population[pop_idx].generate_params(param_rate=self.adjustment_rate))
                elif self.mode == 'feature_engineer':
                    self._sampling(features=self.feature_pairs[pop_idx])
                    if self.deep_learning:
                        self.population[pop_idx].update_data(x_train=self.data_set.get('x_train'),
                                                             y_train=self.data_set.get('y_train'),
                                                             x_test=self.data_set.get('x_test'),
                                                             y_test=self.data_set.get('x_test'),
                                                             x_valn=self.data_set.get('x_val'),
                                                             y_valn=self.data_set.get('y_val')
                                                             )
                if self.deep_learning:
                    self.population[pop_idx].train()
                else:
                    if self.text_clustering:
                        self.population[pop_idx].train()
                    else:
                        self.population[pop_idx].train(x=copy.deepcopy(self.data_set.get('x_train').values),
                                                       y=copy.deepcopy(self.data_set.get('y_train').values),
                                                       validation=dict(x_val=copy.deepcopy(self.data_set.get('x_val').values),
                                                                       y_val=copy.deepcopy(self.data_set.get('y_val').values)
                                                                       )
                                                       )
                _re = 0
                break
            except Exception as e:
                if _re == _re_generate_max:
                    break
                else:
                    _re_generate = True
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg='Error while training model ({} : {})\n{}'.format(self.population[pop_idx].model_name, self.population[pop_idx].model_param, e))
        if _re == _re_generate_max:
            raise SwarmIntelligenceException('Maximum number of errors occurred. Check last error message ...')
        if self.text_clustering:
            self._fitness(individual=self.population[pop_idx], ml_metric='nmi')
        else:
            if self.target_type == 'reg':
                if self.deep_learning:
                    self.population[pop_idx].predict()
                else:
                    _pred: np.array = self.population[pop_idx].predict(x=self.data_set.get('x_test').values)
                    self.population[pop_idx].eval(obs=self.data_set.get('y_test').values, pred=_pred, eval_metric=None)
                self._fitness(individual=self.population[pop_idx], ml_metric='rmse_norm')
            else:
                if self.deep_learning:
                    self.population[pop_idx].predict()
                else:
                    _pred: np.array = self.population[pop_idx].predict(x=self.data_set.get('x_test').values, probability=False)
                    self.population[pop_idx].eval(obs=self.data_set.get('y_test').values, pred=_pred, eval_metric=None)
                if self.target_type == 'clf_multi':
                    self._fitness(individual=self.population[pop_idx], ml_metric='cohen_kappa')
                else:
                    self._fitness(individual=self.population[pop_idx], ml_metric='auc')
        self._collect_meta_data(current_adjustment=True, idx=pop_idx)

    def _populate(self):
        """
        Populate adjustment zero with individuals
        """
        if self.text_clustering:
            _warm_model: dict = {}
            if self.warm_start:
                _warm_model = ClusteringGenerator(predictor=self.features[0],
                                                  models=self.models,
                                                  tokenize=False if self.kwargs.get('tokenize') else self.kwargs.get('tokenize'),
                                                  cloud=self.cloud,
                                                  train_data_path=self.train_data_file_path,
                                                  sep='\t' if self.kwargs.get('sep') is None else self.kwargs.get('sep'),
                                                  sentence_embedding_model_path=self.kwargs.get('sentence_embedding_model_path')
                                                  ).get_model_parameter()
            for p in range(0, self.pop_size, 1):
                if self.evolution_continue:
                    _params: dict = self.final_adjustment.get('param')
                else:
                    if self.initial_population is not None:
                        if p < len(self.initial_population):
                            self.population.append(self.initial_population[p])
                            continue
                    if len(_warm_model.keys()) > 0:
                        if p + 1 > len(_warm_model.keys()):
                            _params: dict = self.model_params
                        else:
                            _params: dict = _warm_model.get(list(_warm_model.keys())[p])
                    else:
                        _params: dict = self.model_params
                self.population.append(ClusteringGenerator(predictor=self.features[0],
                                                           models=self.models,
                                                           cluster_params=_params,
                                                           tokenize=False if self.kwargs.get('tokenize') else self.kwargs.get('tokenize'),
                                                           cloud=self.cloud,
                                                           train_data_path=self.train_data_file_path,
                                                           sep='\t' if self.kwargs.get('sep') is None else self.kwargs.get('sep'),
                                                           sentence_embedding_model_path=self.kwargs.get('sentence_embedding_model_path')
                                                           ).generate_model()
                                       )
        else:
            _warm_model: dict = {}
            if self.warm_start:
                if self.target_type == 'reg':
                    _warm_model = ModelGeneratorReg(models=self.models).get_model_parameter()
                else:
                    _warm_model = ModelGeneratorClf(models=self.models).get_model_parameter()
            for p in range(0, self.pop_size, 1):
                if self.mode.find('feature') >= 0:
                    self._sampling(features=self.feature_pairs[p])
                if self.evolution_continue:
                    _params: dict = self.final_adjustment.get('param')
                else:
                    if self.initial_population is not None:
                        if p < len(self.initial_population):
                            self.population.append(self.initial_population[p])
                            continue
                    if len(_warm_model.keys()) > 0:
                        if p + 1 > len(_warm_model.keys()):
                            _params: dict = self.model_params
                        else:
                            _params: dict = _warm_model.get(list(_warm_model.keys())[p])
                    else:
                        _params: dict = self.model_params
                if self.target_type == 'reg':
                    self.population.append(ModelGeneratorReg(reg_params=_params, models=self.models).generate_model())
                else:
                    self.population.append(ModelGeneratorClf(clf_params=_params, models=self.models).generate_model())

    def _populate_networks(self):
        """
        Populate adjustment zero (with neural networks only)
        """
        _model_param = None
        if self.warm_start:
            _n_vanilla_networks: int = int(self.pop_size * 0.5)
            _n_vanilla_networks_per_model: int = int(_n_vanilla_networks / len(self.models))
            for model in self.models:
                _p: int = 0
                while _p <= _n_vanilla_networks_per_model:
                    if self.evolution_continue:
                        _model_param: dict = self.final_adjustment[str(_p)].get('param')
                    _p += 1
                    self.population.append(NetworkGenerator(target=self.target,
                                                            predictors=self.features,
                                                            output_layer_size=self.deep_learning_output_size,
                                                            x_train=self.data_set.get('x_train').values if self.data_set is not None else self.data_set,
                                                            y_train=self.data_set.get('y_train').values if self.data_set is not None else self.data_set,
                                                            x_test=self.data_set.get('x_test').values if self.data_set is not None else self.data_set,
                                                            y_test=self.data_set.get('y_test').values if self.data_set is not None else self.data_set,
                                                            x_val=self.data_set.get('x_val').values if self.data_set is not None else self.data_set,
                                                            y_val=self.data_set.get('y_val').values if self.data_set is not None else self.data_set,
                                                            train_data_path=self.train_data_file_path,
                                                            test_data_path=self.test_data_file_path,
                                                            validation_data_path=self.valid_data_file_path,
                                                            model_name=model,
                                                            input_param=_model_param,
                                                            hidden_layer_size=self.warm_start_constant_hidden_layers,
                                                            hidden_layer_size_category=self.warm_start_constant_category,
                                                            sep='\t' if self.kwargs.get('sep') is None else self.kwargs.get('sep'),
                                                            cache_dir=self.kwargs.get('cache_dir')
                                                            ).get_vanilla_model()
                                           )
        else:
            _n_vanilla_networks: int = 0
        for p in range(0, self.pop_size - _n_vanilla_networks - 1, 1):
            if self.evolution_continue:
                _model_param: dict = self.final_adjustment[str(p + len(self.population))].get('param')
            else:
                if self.initial_population is not None:
                    if p < len(self.initial_population):
                        self.population.append(self.initial_population[p])
                        continue
            if self.mode.find('feature') >= 0:
                self._sampling(features=self.feature_pairs[p])
            _net_gen: NetworkGenerator = NetworkGenerator(target=self.target,
                                                          predictors=self.features,
                                                          output_layer_size=self.deep_learning_output_size,
                                                          x_train=self.data_set.get('x_train').values if self.data_set is not None else self.data_set,
                                                          y_train=self.data_set.get('y_train').values if self.data_set is not None else self.data_set,
                                                          x_test=self.data_set.get('x_test').values if self.data_set is not None else self.data_set,
                                                          y_test=self.data_set.get('y_test').values if self.data_set is not None else self.data_set,
                                                          x_val=self.data_set.get('x_val').values if self.data_set is not None else self.data_set,
                                                          y_val=self.data_set.get('y_val').values if self.data_set is not None else self.data_set,
                                                          train_data_path=self.train_data_file_path,
                                                          test_data_path=self.test_data_file_path,
                                                          validation_data_path=self.valid_data_file_path,
                                                          models=self.models,
                                                          input_param=_model_param,
                                                          hidden_layer_size=self.warm_start_constant_hidden_layers,
                                                          hidden_layer_size_category=self.warm_start_constant_category,
                                                          sep='\t' if self.kwargs.get('sep') is None else self.kwargs.get('sep'),
                                                          cache_dir=self.kwargs.get('cache_dir')
                                                          )
            self.population.append(_net_gen.generate_model())

    def _re_populate(self):
        """
        Re-populate adjustment 0 to increase likelihood for a good evolution start
        """
        Log(write=self.log, logger_file_path=self.output_file_path).log(msg='Re-populate adjustment 0 because of the poor fitness scoring of all individuals')
        self.n_individuals = -1
        for gen_history in self.adjustment_history.keys():
            self.adjustment_history[gen_history] = {}
        for evo_history in self.evolution_history.keys():
            self.evolution_history[evo_history] = []
        for gen_cur in self.current_adjustment_meta_data.keys():
            if isinstance(self.current_adjustment_meta_data[gen_cur], list):
                self.current_adjustment_meta_data[gen_cur] = []
            elif isinstance(self.current_adjustment_meta_data[gen_cur], int):
                self.current_adjustment_meta_data[gen_cur] = 0
        for evo_gradient in self.evolution_gradient.keys():
            self.evolution_gradient[evo_gradient] = []
        if self.deep_learning:
            self._populate_networks()
        else:
            self._populate()

    def _sampling(self, features: List[str] = None):
        """
        Sample data set
        """
        if self.sampling_function is None:
            self.data_set = MLSampler(df=self.df if self.feature_engineer is None else self.feature_engineer.get_data(),
                                      target=self.target,
                                      features=self.features if features is None else features,
                                      train_size=0.8 if self.kwargs.get('train_size') is None else self.kwargs.get('train_size'),
                                      stratification=False if self.kwargs.get('stratification') is None else self.kwargs.get('stratification')
                                      ).train_test_sampling(validation_split=0.1 if self.kwargs.get('validation_split') is None else self.kwargs.get('validation_split'))
        else:
            self.data_set = self.sampling_function()

    def _select_best_individual(self):
        """
        Select current best global and local individual
        """
        self.best_global_idx = np.array(self.current_adjustment_meta_data['fitness_score']).argmax()
        _other_idx: List[float] = copy.deepcopy(self.current_adjustment_meta_data['fitness_score'])
        del _other_idx[self.best_global_idx]
        self.best_local_idx = np.array(_other_idx).argmax()
        if self.best_global_idx <= self.best_local_idx:
            self.best_local_idx += 1
        self.best_global_local_idx.append(self.best_global_idx)
        self.best_global_local_idx.append(self.best_local_idx)

    def _trainer(self):
        """
        Prepare data set, start training and collect meta data
        """
        _trials: int = 0
        while True:
            if self.re_split_data or self.re_sample_cases or self.re_sample_features:
                _features: List[str] = self.features
                if self.re_sample_features:
                    _features: List[str] = random.sample(self.features, self.max_features)
                _sample_trials: int = 0
                __i = 0
                while True:
                    __i += 1
                    self._sampling(features=_features)
                    _s: int = copy.deepcopy(_sample_trials)
                    for s in self.data_set.keys():
                        if s.find('x_') >= 0 and len(self.data_set[s].shape) != 2:
                            _sample_trials += 1
                            break
                    if _s == _sample_trials:
                        break
                    else:
                        self.data_set = None
                    if _sample_trials == self.max_trials:
                        break
            _threads: dict = {}
            _thread_pool: ThreadPool = ThreadPool(processes=self.n_threads) if self.multi_threading else None
            for i in range(0, self.pop_size, 1):
                if i not in [self.best_global_idx, self.best_local_idx]:
                    if self.multi_threading:
                        _threads.update({i: _thread_pool.apply_async(func=self._modeling, args=[i])})
                    else:
                        self._modeling(pop_idx=i)
            if self.multi_threading:
                for thread in _threads.keys():
                    _threads.get(thread).get()
            self._collect_meta_data(current_adjustment=False, idx=None)
            if self.current_adjustment_meta_data.get('adjustment') == 0:
                if self.re_populate:
                    if _trials == 1: #self.max_trials:
                        break
                    if self.evolution_gradient.get('min')[0] == self.evolution_gradient.get('max')[0] and self.evolution_gradient.get('max')[0] < 1:
                        _trials += 1
                        self._re_populate()
                    else:
                        break
                else:
                    break
            else:
                break

    @staticmethod
    def get_models() -> dict:
        """
        Get all implemented supervised and unsupervised machine learning models

        :return: dict
            Model overview for each machine learning case (classification / regression / neural network / clustering)
        """
        return dict(clf=CLF_ALGORITHMS, reg=REG_ALGORITHMS, nn=NETWORK_TYPE, cl=CLUSTER_ALGORITHMS)

    def inject_data(self):
        """
        Inject new data set (continue evolution using new data set)
        """
        pass

    def optimize(self):
        """
        Optimize attribute configuration of supervised machine learning models in order to select best model, parameter set or feature set
        """
        self.n_training += 1
        if self.evolution_continue:
            self.current_adjustment_meta_data['adjustment'] += 1
        else:
            self.current_adjustment_meta_data['adjustment'] = 0
        _evolve: bool = True
        _stopping_reason: str = ''
        if self.deep_learning:
            self._populate_networks()
        else:
            self._populate()
        while _evolve:
            Log(write=self.log, logger_file_path=self.output_file_path).log('Adjustment: {} / {}'.format(self.current_adjustment_meta_data['adjustment'], self.max_adjustments))
            if self.current_adjustment_meta_data['adjustment'] > 0:
                self.n_threads = self.max_adjustments - 1
            if self.deep_learning:
                if self.warm_start:
                    if self.warm_start_strategy == 'monotone':
                        self.warm_start_constant_hidden_layers += 1
            self._trainer()
            if (self.mode.find('model') >= 0) and (self.current_adjustment_meta_data['adjustment'] > self.burn_in_adjustments):
                if self.convergence:
                    if self._is_gradient_converged(compare=self.convergence_measure, threshold=0.05):
                        _evolve = False
                        _stopping_reason = 'gradient_converged'
                        Log(write=self.log).log('Fitness metric (gradient) has converged. Therefore the evolution stops at adjustment {}'.format(self.current_adjustment_meta_data.get('adjustment')))
                if self.early_stopping > 0:
                    if self._is_gradient_stagnating(min_fitness=True, median_fitness=True, mean_fitness=True, max_fitness=True):
                        _evolve = False
                        _stopping_reason = 'gradient_stagnating'
                        Log(write=self.log).log('Fitness metric (gradient) per adjustment has not increased a certain amount of adjustments ({}). Therefore the evolution stops early at adjustment {}'.format(self.early_stopping, self.current_adjustment_meta_data.get('adjustment')))
            if (datetime.now() - self.start_time).seconds >= self.timer:
                _evolve = False
                _stopping_reason = 'time_exceeded'
                Log(write=self.log).log('Time exceeded:{}'.format(self.timer))
            if _evolve:
                self._select_best_individual()
                self._adjust()
            self.current_adjustment_meta_data['adjustment'] += 1
            if self.current_adjustment_meta_data['adjustment'] > self.max_adjustments:
                _evolve = False
                _stopping_reason = 'max_adjustment_evolved'
                Log(write=self.log).log('Maximum number of adjustments reached: {}'.format(self.max_adjustments))
        if self.mode.find('feature') >= 0:
            self._select_best_individual()
            for idx in list(set(self.best_global_local_idx)):
                self.evolved_features.extend(self.feature_pairs[idx])
            self.evolved_features = list(set(self.evolved_features))
        if self.deep_learning:
            _net_gen = NetworkGenerator(target=self.target,
                                        predictors=self.features,
                                        output_layer_size=self.deep_learning_output_size,
                                        x_train=self.data_set.get('x_train').values if self.data_set is not None else self.data_set,
                                        y_train=self.data_set.get('y_train').values if self.data_set is not None else self.data_set,
                                        x_test=self.data_set.get('x_test').values if self.data_set is not None else self.data_set,
                                        y_test=self.data_set.get('y_test').values if self.data_set is not None else self.data_set,
                                        x_val=self.data_set.get('x_val').values if self.data_set is not None else self.data_set,
                                        y_val=self.data_set.get('y_val').values if self.data_set is not None else self.data_set,
                                        train_data_path=self.train_data_file_path,
                                        test_data_path=self.test_data_file_path,
                                        validation_data_path=self.valid_data_file_path,
                                        models=[self.current_adjustment_meta_data['model_name'][self.best_global_idx]],
                                        model_name=self.current_adjustment_meta_data['model_name'][self.best_global_idx],
                                        input_param=self.current_adjustment_meta_data['param'][self.best_global_idx],
                                        model_param=self.current_adjustment_meta_data['param'][self.best_global_idx],
                                        hidden_layer_size=self.warm_start_constant_hidden_layers,
                                        hidden_layer_size_category=self.warm_start_constant_category,
                                        cloud=self.cloud,
                                        sep='\t' if self.kwargs.get('sep') is None else self.kwargs.get('sep'),
                                        cache_dir=self.kwargs.get('cache_dir')
                                        ).generate_model()
            _net_gen.train()
            self.model = _net_gen.model
        else:
            if self.text_clustering:
                _cluster_gen = ClusteringGenerator(predictor=self.features[0],
                                                   models=[self.current_adjustment_meta_data['model_name'][self.best_global_idx]],
                                                   model_name=self.current_adjustment_meta_data['model_name'][self.best_global_idx],
                                                   cluster_params=self.current_adjustment_meta_data['param'][self.best_global_idx],
                                                   tokenize=False if self.kwargs.get('tokenize') else self.kwargs.get('tokenize'),
                                                   cloud=self.cloud,
                                                   train_data_path=self.train_data_file_path,
                                                   sep='\t' if self.kwargs.get('sep') is None else self.kwargs.get('sep'),
                                                   sentence_embedding_model_path=self.kwargs.get('sentence_embedding_model_path')
                                                   ).generate_model()
                _cluster_gen.train()
                self.model = _cluster_gen.model
            else:
                if self.target_type == 'reg':
                    _model_gen = ModelGeneratorReg(reg_params=self.current_adjustment_meta_data['param'][self.best_global_idx],
                                                   model_name=self.current_adjustment_meta_data['model_name'][self.best_global_idx]
                                                   ).generate_model()
                else:
                    _model_gen = ModelGeneratorClf(clf_params=self.current_adjustment_meta_data['param'][self.best_global_idx],
                                                   model_name=self.current_adjustment_meta_data['model_name'][self.best_global_idx]
                                                   ).generate_model()
                _model_gen.train(x=copy.deepcopy(self.data_set.get('x_train').values),
                                 y=copy.deepcopy(self.data_set.get('y_train').values),
                                 validation=dict(x_val=copy.deepcopy(self.data_set.get('x_val').values),
                                                 y_val=copy.deepcopy(self.data_set.get('y_val').values)
                                                 )
                                 )
                self.model = _model_gen.model
        Log(write=self.log, logger_file_path=self.output_file_path).log(msg='Best model: {} - {}'.format(self.current_adjustment_meta_data['model_name'][self.best_global_idx], self.current_adjustment_meta_data['param'][self.best_global_idx]))
        Log(write=self.log, logger_file_path=self.output_file_path).log(msg='Fitness score: {}'.format(self.current_adjustment_meta_data['fitness_score'][self.best_global_idx]))
        Log(write=self.log, logger_file_path=self.output_file_path).log(msg='Fitness metric: {}'.format(self.current_adjustment_meta_data['fitness_metric'][self.best_global_idx]))
        self._gather_final_adjustment()
        if self.deep_learning:
            self.data_set = dict(y_test=self.population[self.best_global_idx].obs,
                                 pred=self.population[self.best_global_idx].pred
                                 )
        else:
            if self.mode.find('model') >= 0 and self.plot:
                self.data_set.update({'pred': self.model.predict(self.data_set.get('x_test'))})
        self.evolution: dict = dict(model_name=self.current_adjustment_meta_data['model_name'][self.best_global_idx],
                                    param=self.current_adjustment_meta_data['param'][self.best_global_idx],
                                    param_adjusted=self.current_adjustment_meta_data['param_adjusted'][self.best_global_idx],
                                    fitness_score=self.current_adjustment_meta_data['fitness_score'][self.best_global_idx],
                                    fitness_metric=self.current_adjustment_meta_data['fitness_metric'][self.best_global_idx],
                                    epoch_metric_score=self.population[self.best_global_idx].epoch_eval if self.deep_learning else None,
                                    features=self.features if self.text_clustering or self.deep_learning else self.current_adjustment_meta_data['features'][self.best_global_idx],
                                    target=self.target,
                                    target_type=self.target_type,
                                    re_split_data=self.re_split_data,
                                    re_split_cases=self.re_sample_cases,
                                    re_sample_features=self.re_sample_features,
                                    id=self.current_adjustment_meta_data['id'][self.best_global_idx],
                                    mode=self.mode,
                                    adjustments=self.current_adjustment_meta_data['adjustment'],
                                    adjustment_prob=self.adjustment_prob,
                                    adjustment_rate=self.adjustment_rate,
                                    adjusted_features=self.adjusted_features,
                                    adjustment_history=self.adjustment_history,
                                    evolution_history=self.evolution_history,
                                    evolution_gradient=self.evolution_gradient,
                                    convergence_check=self.convergence,
                                    convergence_measure=self.convergence_measure,
                                    early_stopping=self.early_stopping,
                                    max_time=self.timer,
                                    start_time=self.start_time,
                                    end_time=str(datetime.now()),
                                    stopping_reason=_stopping_reason
                                    )
        if self.plot:
            self.visualize(results_table=True,
                           model_distribution=True,
                           model_evolution=True,
                           param_distribution=False,
                           train_time_distribution=True,
                           breeding_map=True,
                           breeding_graph=True,
                           fitness_distribution=True,
                           fitness_evolution=True if self.current_adjustment_meta_data['adjustment'] > 0 else False,
                           fitness_dimensions=True,
                           per_adjustment=True if self.current_adjustment_meta_data['adjustment'] > 0 else False,
                           prediction_of_best_model=True,
                           epoch_stats=True
                           )
        if self.output_file_path is not None:
            if len(self.output_file_path) > 0:
                self.save_evolution(si=True,
                                    model=self.deploy_model,
                                    evolution_history=False,
                                    adjustment_history=False,
                                    final_adjustment=False
                                    )

    def optimize_continue(self, deploy_model: bool = True, max_adjustments: int = 5):
        """
        Continue evolution by using last adjustment of previous evolution as new adjustment 0

        :param deploy_model: bool
            Deploy fittest model to cloud platform

        :param max_adjustments: int
            Maximum number of adjustments
        """
        self.data_set = None
        self.evolution_continue = True
        self.deploy_model = deploy_model
        _max_gen: int = max_adjustments if max_adjustments > 0 else 5
        self.max_adjustments: int = self.max_adjustments + _max_gen
        self.burn_in_adjustments += self.current_adjustment_meta_data['adjustment']
        self.optimize()

    def save_evolution(self,
                       si: bool = True,
                       model: bool = True,
                       evolution_history: bool = False,
                       adjustment_history: bool = False,
                       final_adjustment: bool = False
                       ):
        """
        Save evolution meta data generated by genetic algorithm to local hard drive as pickle file

        :param si: bool
            Save SwarmIntelligence class object (required for continuing evolution / optimization)

        :param model: bool
            Save evolved model

        :param evolution_history: bool
            Save evolution history meta data

        :param adjustment_history: bool
            Save adjustment history meta data

        :param final_adjustment: bool
            Save settings of each individual of final adjustment
        """
        # Export evolution history data:
        if evolution_history:
            DataExporter(obj=self.evolution_history,
                         file_path=os.path.join(self.output_file_path, 'evolution_history.p'),
                         create_dir=False,
                         overwrite=True,
                         cloud=self.cloud,
                         bucket_name=self.bucket_name
                         ).file()
        # Export adjustment history data:
        if adjustment_history:
            DataExporter(obj=self.adjustment_history,
                         file_path=os.path.join(self.output_file_path, 'adjustment_history.p'),
                         create_dir=False,
                         overwrite=True,
                         cloud=self.cloud,
                         bucket_name=self.bucket_name
                         ).file()
        if final_adjustment:
            DataExporter(obj=self.final_adjustment,
                         file_path=os.path.join(self.output_file_path, 'final_adjustment.p'),
                         create_dir=True,
                         overwrite=True,
                         cloud=self.cloud,
                         bucket_name=self.bucket_name
                         ).file()
        # Export evolved model:
        if model:
            _file_name_extension: str = '' if self.kwargs.get('model_file_name_extension') is None else '_{}'.format(self.kwargs.get('model_file_name_extension'))
            _file_name: str = 'model{}.p'.format(_file_name_extension)
            if self.deep_learning:
                if self.current_adjustment_meta_data['model_name'][self.best_global_idx] == 'trans':
                    torch.save(obj=self.model.model, f=os.path.join(self.output_file_path, _file_name))
                else:
                    torch.save(obj=self.model, f=os.path.join(self.output_file_path, _file_name))
            else:
                DataExporter(obj=self.model,
                             file_path=os.path.join(self.output_file_path, _file_name),
                             create_dir=False,
                             overwrite=True,
                             cloud=self.cloud,
                             bucket_name=self.bucket_name
                             ).file()
        # Export SwarmIntelligence class object:
        if si:
            self.df = None
            self.model = None
            self.population = []
            #self.data_set = None
            self.feature_engineer = None
            _file_name_extension: str = '' if self.kwargs.get('si_file_name_extension') is None else '_{}'.format(self.kwargs.get('si_file_name_extension'))
            _file_name: str = 'swarm{}.p'.format(_file_name_extension)
            DataExporter(obj=self,
                         file_path=os.path.join(self.output_file_path, _file_name),
                         create_dir=False,
                         overwrite=True,
                         cloud=self.cloud,
                         bucket_name=self.bucket_name
                         ).file()

    def visualize(self,
                  results_table: bool = True,
                  model_distribution: bool = False,
                  model_evolution: bool = True,
                  param_distribution: bool = False,
                  train_time_distribution: bool = True,
                  breeding_map: bool = False,
                  breeding_graph: bool = False,
                  fitness_distribution: bool = True,
                  fitness_evolution: bool = True,
                  fitness_dimensions: bool = True,
                  per_adjustment: bool = True,
                  prediction_of_best_model: bool = True,
                  epoch_stats: bool = True
                  ):
        """
        Visualize evolutionary activity

        :param results_table: bool
            Evolution results table
                -> Table Chart

        :param model_evolution: bool
            Evolution of individuals
                -> Scatter Chart

        :param model_distribution: bool
            Distribution of used model types
                -> Bar Chart / Pie Chart

        :param param_distribution: bool
            Distribution of used model parameter combination
                -> Tree Map / Sunburst

        :param train_time_distribution: bool
            Distribution of training time
                -> Violin

        :param breeding_map: bool
            Breeding evolution as
                -> Heat Map

        :param breeding_graph: bool
            Breeding evolution as
                -> Network Graph

        :param fitness_distribution: bool
            Distribution of fitness metric
                -> Ridge Line Chart

        :param fitness_evolution: bool
            Evolution of fitness metric
                -> Line Chart

        :param fitness_dimensions: bool
            Calculated loss value for each dimension in fitness metric
                -> Radar Chart
                -> Tree Map

        :param per_adjustment: bool
            Visualize results of each adjustment in detail or visualize just evolutionary results

        :param prediction_of_best_model: bool
            Evaluation of prediction of the fittest model of evolution
                -> Parallel Coordinate Chart
                -> Joint Chart

        :param epoch_stats: bool
            Visualize train and validation error for each training epoch (deep learning only)
        """
        _charts: dict = {}
        _evolution_history_data: pd.DataFrame = pd.DataFrame(data=self.evolution_history)
        _m: List[str] = ['fitness_score', 'ml_metric', 'train_test_diff']
        _evolution_history_data[_m] = _evolution_history_data[_m].round(decimals=2)
        _evolution_gradient_data: pd.DataFrame = pd.DataFrame(data=self.evolution_gradient)
        _evolution_gradient_data['adjustment'] = [i for i in range(0, len(self.evolution_gradient.get('max')), 1)]
        _best_model_results: pd.DataFrame = pd.DataFrame(data=dict(obs=self.data_set.get('y_test'),
                                                                   pred=self.data_set.get('pred')
                                                                   )
                                                         )
        if self.target_type == 'reg':
            _best_model_results['abs_diff'] = _best_model_results['obs'] - _best_model_results['pred']
            _best_model_results['rel_diff'] = _best_model_results['obs'] / _best_model_results['pred']
        elif self.target_type == 'clf_multi':
            _best_model_results['abs_diff'] = _best_model_results['obs'] - _best_model_results['pred']
        _best_model_results = _best_model_results.round(decimals=4)
        if results_table:
            _charts.update({'Results of Genetic Algorithm:': dict(data=_evolution_history_data,
                                                                  plot_type='table',
                                                                  file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'si_metadata_table.html')
                                                                  )
                            })
        if model_evolution:
            _charts.update({'Evolution of used ML Models:': dict(data=_evolution_history_data,
                                                                 features=['fitness_score', 'adjustment'],
                                                                 color_feature='model',
                                                                 plot_type='scatter',
                                                                 melt=True,
                                                                 file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'si_model_evolution.html')
                                                                 )
                            })
        if model_distribution:
            if self.models is None or len(self.models) > 1:
                _charts.update({'Distribution of used ML Models:': dict(data=_evolution_history_data,
                                                                        features=['model'],
                                                                        group_by=['adjustment'] if per_adjustment else None,
                                                                        plot_type='pie',
                                                                        file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'si_model_distribution.html')
                                                                        )
                                })
        #if param_distribution:
        #    _charts.update({'Distribution of ML Model parameters:': dict(data=_evolution_history_data,
        #                                                                 features=['model_param'],
        #                                                                 group_by=['adjustment'] if per_adjustment else None,
        #                                                                 plot_type='tree',
        #                                                                 file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'si_parameter_treemap.html')
        #                                                                 )
        #                    })
        if train_time_distribution:
            _charts.update({'Distribution of elapsed Training Time:': dict(data=_evolution_history_data,
                                                                           features=['train_time_in_seconds'],
                                                                           group_by=['model'],
                                                                           plot_type='violin',
                                                                           melt=False,
                                                                           file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'si_training_time_distribution.html')
                                                                           )
                            })
        if breeding_map:
            _breeding_map: pd.DataFrame = pd.DataFrame(data=dict(adjustment_0=self.adjustment_history['population']['adjustment_0'].get('fitness')), index=[0])
            for g in self.adjustment_history['population'].keys():
                if g != 'adjustment_0':
                    _breeding_map[g] = self.adjustment_history['population'][g].get('fitness')
            _charts.update({'Breeding Heat Map:': dict(data=_breeding_map,
                                                       plot_type='heat',
                                                       file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'si_breeding_heatmap.html')
                                                       )
                            })
        if breeding_graph:
            _charts.update({'Breeding Network Graph:': dict(data=_evolution_history_data,
                                                            features=['adjustment', 'fitness_score'],
                                                            graph_features=dict(node='id', edge='best'),
                                                            color_feature='model',
                                                            plot_type='network',
                                                            file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'si_breeding_graph.html')
                                                            )
                            })
        if fitness_distribution:
            _charts.update({'Distribution of Fitness Metric:': dict(data=_evolution_history_data,
                                                                    features=['fitness_score'],
                                                                    time_features=['adjustment'],
                                                                    plot_type='ridgeline',
                                                                    file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'si_fitness_score_distribution_per_adjustment.html')
                                                                    )
                            })
        if fitness_dimensions:
            _charts.update({'Evolution Meta Data:': dict(data=_evolution_history_data,
                                                         features=['train_time_in_seconds',
                                                                   'ml_metric',
                                                                   'train_test_diff',
                                                                   'fitness_score',
                                                                   'best',
                                                                   'id',
                                                                   'adjustment',
                                                                   'model'
                                                                   ],
                                                         color_feature='model',
                                                         plot_type='parcoords',
                                                         file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'si_metadata_evolution_coords_actor_only.html')
                                                         )
                            })
        if fitness_evolution:
            _charts.update({'Fitness Evolution:': dict(data=_evolution_gradient_data,
                                                       features=['min', 'median', 'mean', 'max'],
                                                       time_features=['adjustment'],
                                                       plot_type='line',
                                                       file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'si_evolution_fitness_score.html')
                                                       )
                            })
        if epoch_stats:
            if self.deep_learning:
                _epoch_metric_score: pd.DataFrame = pd.DataFrame(data=self.evolution.get('epoch_metric_score'))
                _epoch_metric_score['epoch'] = [epoch + 1 for epoch in range(0, _epoch_metric_score.shape[0], 1)]
                print(_epoch_metric_score)
                _charts.update({'Epoch Evaluation of fittest neural network': dict(data=_epoch_metric_score,
                                                                                   features=['train', 'val'],
                                                                                   time_features=['epoch'],
                                                                                   plot_type='line',
                                                                                   file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'si_epoch_metric_score.html')
                                                                                   )
                                })
        if prediction_of_best_model:
            if self.target_type == 'reg':
                _charts.update({'Prediction Evaluation of final adjusted ML Model:': dict(data=_best_model_results,
                                                                                          features=['obs', 'abs_diff', 'rel_diff', 'pred'],
                                                                                          color_feature='pred',
                                                                                          plot_type='parcoords',
                                                                                          file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'si_prediction_evaluation_coords.html')
                                                                                          ),
                                'Prediction vs. Observation of final adjusted ML Model:': dict(data=_best_model_results,
                                                                                               features=['obs', 'pred'],
                                                                                               plot_type='joint',
                                                                                               file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'si_prediction_scatter_contour.html')
                                                                                               )
                                })
            else:
                _confusion_matrix: pd.DataFrame = pd.DataFrame(data=EvalClf(obs=self.data_set.get('y_test'),
                                                                            pred=self.data_set.get('pred')
                                                                            ).confusion(),
                                                               index=self.target_labels,
                                                               columns=self.target_labels
                                                               )
                _cf_row_sum = pd.DataFrame()
                _cf_row_sum[' '] = _confusion_matrix.sum()
                _confusion_matrix = pd.concat([_confusion_matrix, _cf_row_sum.transpose()], axis=0)
                _cf_col_sum = pd.DataFrame()
                _cf_col_sum[' '] = _confusion_matrix.transpose().sum()
                _confusion_matrix = pd.concat([_confusion_matrix, _cf_col_sum], axis=1)
                _charts.update({'Confusion Matrix': dict(data=_confusion_matrix,
                                                         plot_type='table',
                                                         file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'si_prediction_confusion_table.html')
                                                         )
                                })
                _charts.update({'Confusion Matrix Heatmap': dict(data=_best_model_results,
                                                                 features=['obs', 'pred'],
                                                                 plot_type='heat',
                                                                 file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'si_prediction_confusion_heatmap.html')
                                                                 )
                                })
                _confusion_matrix_normalized: pd.DataFrame = pd.DataFrame(data=EvalClf(obs=self.data_set.get('y_test'),
                                                                                       pred=self.data_set.get('pred')
                                                                                       ).confusion(normalize='pred'),
                                                                          #index=['obs', 'pred'],
                                                                          #columns=['obs', 'pred']
                                                                          )
                _charts.update({'Confusion Matrix Normalized Heatmap:': dict(data=_confusion_matrix_normalized,
                                                                             features=self.target_labels,
                                                                             plot_type='heat',
                                                                             file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'si_prediction_confusion_normal_heatmap.html')
                                                                             )
                                })
                _charts.update({'Classification Report:': dict(data=_best_model_results,
                                                               plot_type='table',
                                                               file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'si_prediction_clf_report_table.html')
                                                               )
                                })
                if self.target_type == 'clf_multi':
                    _charts.update({'Prediction Evaluation of final inherited ML Model:': dict(data=_best_model_results,
                                                                                               features=['obs', 'abs_diff', 'pred'],
                                                                                               color_feature='pred',
                                                                                               plot_type='parcoords',
                                                                                               brushing=True,
                                                                                               file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'si_prediction_evaluation_category.html')
                                                                                               )
                                    })
                else:
                    _roc_curve = pd.DataFrame()
                    _roc_curve_values: dict = EvalClf(obs=_best_model_results['obs'],
                                                      pred=_best_model_results['pred']
                                                      ).roc_curve()
                    _roc_curve['roc_curve'] = _roc_curve_values['true_positive_rate'][1]
                    _roc_curve['baseline'] = _roc_curve_values['false_positive_rate'][1]
                    _charts.update({'ROC-AUC Curve': dict(data=_roc_curve,
                                                          features=['roc_curve', 'baseline'],
                                                          time_features=['baseline'],
                                                          #xaxis_label=['False Positive Rate'],
                                                          #yaxis_label=['True Positive Rate'],
                                                          plot_type='line',
                                                          file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'si_prediction_roc_auc_curve.html')
                                                          )
                                    })
        if len(_charts.keys()) > 0:
            DataVisualizer(subplots=_charts,
                           interactive=True,
                           file_path=self.output_file_path,
                           render=True if self.output_file_path is None else False,
                           height=750,
                           width=750,
                           unit='px'
                           ).run()
