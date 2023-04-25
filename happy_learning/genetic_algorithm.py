"""

Reinforcement Learning Environment using genetic algorithm

"""

import copy
import mlflow
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
from typing import Dict, List, Union

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class GeneticAlgorithmException(Exception):
    """
    Class for managing exceptions for class GeneticAlgorithm
    """
    pass


class GeneticAlgorithm:
    """
    Class for reinforced optimizing machine learning algorithms and feature engineering using Genetic Algorithm
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
                 burn_in_generations: int = -1,
                 warm_start: bool = True,
                 warm_start_strategy: str = 'monotone',
                 warm_start_constant_hidden_layers: int = 0,
                 warm_start_constant_category: str = 'very_small',
                 max_generations: int = 50,
                 pop_size: int = 64,
                 mutation_rate: float = 0.1,
                 mutation_prob: float = 0.85,
                 parents_ratio: float = 0.5,
                 early_stopping: int = 0,
                 convergence: bool = False,
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
                 generation_zero: list = None,
                 multi_threading: bool = False,
                 multi_processing: bool = False,
                 log: bool = False,
                 verbose: bool = False,
                 checkpoint: bool = True,
                 mlflow_log: bool = True,
                 feature_engineer=None,
                 fitness_function=sml_score,
                 sampling_function=None,
                 **kwargs
                 ):
        """
        :param mode: str
            Optimization specification
                -> model: Optimize model or hyperparameter set
                -> model_sampler: Optimize model or hyperparameter set and resample data set each mutation
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
            Whether to re-split data set into train & test data every generation or not

        :param re_sample_cases: bool
            Whether to re-sample cases set every generation or not

        :param re_sample_features: bool
            Whether to re-sample features set every generation or not

        :param re_populate: bool
            Whether to re-populate generation 0 if all models achieve poor fitness results

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

        :param max_generations: int
            Maximum number of generations

        :param pop_size: int
            Population size of each generation

        :param mutation_rate: float
            Mutation rate

        :param mutation_prob: float
            Mutation probability

        :param parents_ratio: float
            Ratio of parents to generate

        :param warm_start: bool
            Whether to start evolution (generation 0) using standard parameter config for each model type once

        :param warm_start_strategy: str
            Name of the warm start strategy (deep learning only)
            -> random: Draw size of hidden layers randomly
            -> constant: Use constant size of hidden layers only
            -> monotone: Increase the range of hidden layers each generation by one category
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
            Number of generations for starting early stopping condition checks

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

        :param generation_zero: list
            Pre-defined models for initial generation

        :param multi_threading: bool
            Whether to run genetic algorithm using multiple threads (of one cpu core) or single thread

        :param multi_processing: bool
            Whether to run genetic algorithm using multiple processes (cpu cores) or single cpu core

        :param log: bool
            Write logging file or just print messages

        :param verbose: bool
            Log all processes (extended logging)

        :param checkpoint: bool
            Save checkpoint after each adjustment

        :param mlflow_log: bool
            Track experiment results using mlflow

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
                raise GeneticAlgorithmException('Cloud provider ({}) not supported'.format(cloud))
            if output_file_path is None:
                raise GeneticAlgorithmException('Output file path is None')
            self.bucket_name: str = output_file_path.split("//")[1].split("/")[0]
        self.include_neural_networks: bool = include_neural_networks
        _neural_nets: List[str] = []
        _clustering: List[str] = []
        if models is None:
            self.text_clustering: bool = False
            self.deep_learning: bool = False
            self.models: List[str] = models
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
        self.parents_ratio: float = parents_ratio
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
        self.sep: str = self.kwargs.get('sep')
        self.tokenize: bool = self.kwargs.get('tokenize')
        self.eval_method: str = self.kwargs.get('eval_method')
        self.cache_dir: str = self.kwargs.get('cache_dir')
        self.language_model_path: str = self.kwargs.get('language_model_path')
        self.sentence_embedding_model_path: str = self.kwargs.get('sentence_embedding_model_path')
        self.kwargs.pop('sep', None)
        self.kwargs.pop('tokenize', None)
        self.kwargs.pop('eval_method', None)
        self.kwargs.pop('cache_dir', None)
        self.kwargs.pop('language_model_path', None)
        self.kwargs.pop('sentence_embedding_model_path', None)
        self._input_manager()
        if labels is None:
            self.target_labels: List[str] = [f'{label}' for label in range(0, self.target_classes, 1)]
        else:
            if len(labels) == self.target_classes:
                self.target_labels: List[str] = labels
            else:
                self.target_labels: List[str] = [f'{label}' for label in range(0, self.target_classes, 1)]
        self.log: bool = log
        self.verbose: bool = verbose
        self.mlflow_log: bool = mlflow_log
        if self.mlflow_log:
            self.mlflow_client: mlflow.tracking.MlflowClient = mlflow.tracking.MlflowClient(
                tracking_uri=self.kwargs.get('tracking_uri'),
                registry_uri=self.kwargs.get('registry_uri')
            )
        else:
            self.mlflow_client = None
        self.warm_start: bool = warm_start
        self.warm_start_strategy: str = warm_start_strategy if warm_start_strategy in HIDDEN_LAYER_CATEGORY_EVOLUTION else 'monotone'
        self.warm_start_constant_hidden_layers: int = warm_start_constant_hidden_layers if warm_start_constant_hidden_layers > 0 else 0
        self.warm_start_constant_category: str = warm_start_constant_category if warm_start_constant_category in list(NETWORK_TYPE_CATEGORY.keys()) else 'very_small'
        self.re_populate: bool = re_populate
        self.max_trials: int = max_trials
        self.max_generations: int = max_generations if max_generations >= 0 else 50
        self.parents_ratio = self.parents_ratio if (self.parents_ratio > 0) and (self.parents_ratio < 1) else 0.5
        self.burn_in_generations: int = burn_in_generations if burn_in_generations >= 0 else round(0.1 * self.max_generations)
        self.population: List[object] = []
        self.mutation_rate: float = mutation_rate if mutation_rate > 0 or mutation_rate <= 1 else 0.1
        self.mutation_prob: float = mutation_prob if mutation_prob > 0 or mutation_prob <= 1 else 0.85
        self.plot: bool = plot
        self.fitness_function = fitness_function
        self.deep_learning_type: str = deep_learning_type
        self.generation_zero: list = generation_zero
        self.n_threads: int = self.pop_size
        self.multi_threading: bool = multi_threading
        self.multi_processing: bool = multi_processing
        self.n_individuals: int = -1
        self.child_idx: List[int] = []
        self.parents_idx: List[int] = []
        self.best_individual_idx: int = -1
        self.stopping_reason: str = None
        self.checkpoint: bool = checkpoint
        self.final_generation: dict = {}
        self.evolution: dict = {}
        self.evolved_features: List[str] = []
        self.mutated_features: dict = dict(parent=[], child=[], fitness=[], generation=[], action=[])
        self.current_generation_meta_data: dict = dict(generation=0,
                                                       id=[],
                                                       fitness_metric=[],
                                                       fitness_score=[],
                                                       model_name=[],
                                                       param=[],
                                                       param_mutated=[],
                                                       features=[]
                                                       )
        self.generation_history: dict = dict(population={},
                                             inheritance={},
                                             time=[]
                                             )
        self.evolution_history: dict = dict(id=[],
                                            model=[],
                                            generation=[],
                                            training=[],
                                            parent=[],
                                            mutation_type=[],
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

    def _collect_meta_data(self, current_gen: bool, idx: int = None):
        """
        Collect evolution metadata

        :param current_gen: bool
            Whether to write evolution metadata of each individual of current generation or not

        :param idx: int
            Index number of individual within population
        """
        if self.generation_history['population'].get('gen_{}'.format(self.current_generation_meta_data['generation'])) is None:
            self.generation_history['population'].update(
                {'gen_{}'.format(self.current_generation_meta_data['generation']): dict(id=[],
                                                                                        model=[],
                                                                                        parent=[],
                                                                                        fitness=[]
                                                                                        )
                 })
        if current_gen:
            setattr(self.population[idx], 'fitness_score', self.evolution_history.get('fitness_score')[self.population[idx].id])
            if not self.deep_learning and not self.text_clustering:
                setattr(self.population[idx], 'features', list(self.data_set.get('x_train').columns))
            if self.current_generation_meta_data['generation'] == 0:
                self.current_generation_meta_data.get('id').append(copy.deepcopy(idx))
                if not self.deep_learning and not self.text_clustering:
                    self.current_generation_meta_data.get('features').append(copy.deepcopy(self.population[idx].features))
                self.current_generation_meta_data.get('model_name').append(copy.deepcopy(self.population[idx].model_name))
                self.current_generation_meta_data.get('param').append(copy.deepcopy(self.population[idx].model_param))
                self.current_generation_meta_data.get('param_mutated').append(copy.deepcopy(self.population[idx].model_param_mutated))
                self.current_generation_meta_data.get('fitness_metric').append(copy.deepcopy(self.population[idx].fitness))
                self.current_generation_meta_data.get('fitness_score').append(copy.deepcopy(self.population[idx].fitness_score))
                if self.verbose:
                    Log(write=self.log, logger_file_path=self.output_file_path).log('Fitness score {} of individual {}'.format(self.population[idx].fitness_score, idx))
                    Log(write=self.log, logger_file_path=self.output_file_path).log('Fitness metric {} of individual {}'.format(self.population[idx].fitness, idx))
            else:
                self.current_generation_meta_data['id'][idx] = copy.deepcopy(self.population[idx].id)
                if not self.deep_learning and not self.text_clustering:
                    self.current_generation_meta_data['features'][idx] = copy.deepcopy(self.population[idx].features)
                self.current_generation_meta_data['model_name'][idx] = copy.deepcopy(self.population[idx].model_name)
                self.current_generation_meta_data['param'][idx] = copy.deepcopy(self.population[idx].model_param)
                self.current_generation_meta_data['param_mutated'][idx] = copy.deepcopy(self.population[idx].model_param_mutated)
                self.current_generation_meta_data['fitness_metric'][idx] = copy.deepcopy(self.population[idx].fitness)
                self.current_generation_meta_data['fitness_score'][idx] = copy.deepcopy(self.population[idx].fitness_score)
                if self.verbose:
                    Log(write=self.log, logger_file_path=self.output_file_path).log('Fitness score {} of individual {}'.format(self.population[idx].fitness_score, idx))
                    Log(write=self.log, logger_file_path=self.output_file_path).log('Fitness metric {} of individual {}'.format(self.population[idx].fitness, idx))
        else:
            if idx is None:
                self.generation_history['population']['gen_{}'.format(self.current_generation_meta_data['generation'])]['fitness'] = copy.deepcopy(self.current_generation_meta_data.get('fitness'))
                self.evolution_gradient.get('min').append(copy.deepcopy(min(self.current_generation_meta_data.get('fitness_score'))))
                self.evolution_gradient.get('median').append(copy.deepcopy(np.median(self.current_generation_meta_data.get('fitness_score'))))
                self.evolution_gradient.get('mean').append(copy.deepcopy(np.mean(self.current_generation_meta_data.get('fitness_score'))))
                self.evolution_gradient.get('max').append(copy.deepcopy(max(self.current_generation_meta_data.get('fitness_score'))))
                Log(write=self.log, logger_file_path=self.output_file_path).log(msg='Fitness: Max -> {}'.format(self.evolution_gradient.get('max')[-1]))
                Log(write=self.log, logger_file_path=self.output_file_path).log(msg='Fitness: Median -> {}'.format(self.evolution_gradient.get('median')[-1]))
                Log(write=self.log, logger_file_path=self.output_file_path).log(msg='Fitness: Mean -> {}'.format(self.evolution_gradient.get('mean')[-1]))
                Log(write=self.log, logger_file_path=self.output_file_path).log(msg='Fitness: Min -> {}'.format(self.evolution_gradient.get('min')[-1]))
            else:
                if self.current_generation_meta_data['generation'] == 0:
                    self.evolution_history.get('parent').append(-1)
                else:
                    self.evolution_history.get('parent').append(copy.deepcopy(self.population[idx].id))
                self.generation_history['population']['gen_{}'.format(self.current_generation_meta_data['generation'])][
                    'parent'].append(copy.deepcopy(self.evolution_history.get('parent')[-1]))
                self.n_individuals += 1
                setattr(self.population[idx], 'id', self.n_individuals)
                if self.text_clustering:
                    setattr(self.population[idx], 'target', self.target)
                self.evolution_history.get('id').append(copy.deepcopy(self.population[idx].id))
                self.evolution_history.get('generation').append(copy.deepcopy(self.current_generation_meta_data['generation']))
                self.evolution_history.get('model').append(copy.deepcopy(self.population[idx].model_name))
                self.evolution_history.get('mutation_type').append(copy.deepcopy(self.population[idx].model_param_mutation))
                self.evolution_history.get('training').append(copy.deepcopy(self.n_training))
                self.generation_history['population']['gen_{}'.format(self.current_generation_meta_data['generation'])]['id'].append(copy.deepcopy(self.population[idx].id))
                self.generation_history['population']['gen_{}'.format(self.current_generation_meta_data['generation'])]['model'].append(copy.deepcopy(self.population[idx].model_name))

    def _crossover(self, parent: int, child: int):
        """
        Mutate individuals after mixing the individual genes using cross-over method

        :param parent: int
            Index number of parent in population

        :param child: int
            Index number of child in population
        """
        _inherit_genes: List[str] = copy.deepcopy(self.feature_pairs[parent])
        _x: int = 0
        for x in range(0, round(len(self.feature_pairs[parent]) * self.parents_ratio), 1):
            _new_pair: List[str] = copy.deepcopy(self.feature_pairs[child])
            while True:
                _gene: str = np.random.choice(_inherit_genes)
                if _gene not in self.feature_pairs[child] or _x == 20:
                    _x = 0
                    _new_pair[x] = copy.deepcopy(_gene)
                    break
                else:
                    _x += 1
            self.feature_pairs[child] = copy.deepcopy(_new_pair)

    def _fitness(self, individual: object, ml_metric: str, track_metric: bool = True):
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

        :param track_metric: bool
            Whether to track metric or not
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
        if track_metric:
            for score in _scores.keys():
                self.evolution_history.get(score).append(copy.deepcopy(_scores.get(score)))

    def _gather_final_generation(self):
        """
        Gather information about each individual of final generation
        """
        for i, individual in enumerate(self.population):
            self.final_generation.update({i: dict(id=copy.deepcopy(individual.id),
                                                  model_name=copy.deepcopy(individual.model_name),
                                                  param=copy.deepcopy(individual.model_param),
                                                  fitness=copy.deepcopy(individual.fitness),
                                                  fitness_score=copy.deepcopy(individual.fitness_score),
                                                  hidden_layer_size=copy.deepcopy(individual.hidden_layer_size) if self.deep_learning else None
                                                  )
                                          })

    def _generate_final_model(self):
        """
        Generate final model based on the evolved parameters / features
        """
        if self.verbose and self.stopping_reason is not None:
            Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Generate final model {self.best_individual_idx}')
        if self.deep_learning:
            _network_generator: NetworkGenerator = NetworkGenerator(target=self.target,
                                                                    predictors=self.features,
                                                                    output_layer_size=self.deep_learning_output_size,
                                                                    train_data_path=self.train_data_file_path,
                                                                    test_data_path=self.test_data_file_path,
                                                                    validation_data_path=self.valid_data_file_path,
                                                                    models=[self.current_generation_meta_data['model_name'][self.best_individual_idx]],
                                                                    model_name=self.current_generation_meta_data['model_name'][self.best_individual_idx],
                                                                    input_param=self.current_generation_meta_data['param'][self.best_individual_idx],
                                                                    model_param=self.current_generation_meta_data['param'][self.best_individual_idx],
                                                                    hidden_layer_size=self.warm_start_constant_hidden_layers,
                                                                    hidden_layer_size_category=self.warm_start_constant_category,
                                                                    cloud=self.cloud,
                                                                    sep='\t' if self.sep is None else self.sep,
                                                                    cache_dir=self.cache_dir,
                                                                    **self.kwargs
                                                                    )
            _network_generator.generate_model()
            _network_generator.train()
            _generator: NetworkGenerator = _network_generator
            self.model = _network_generator.model
        else:
            if self.text_clustering:
                _cluster_generator: ClusteringGenerator = ClusteringGenerator(predictor=self.features[0],
                                                                              models=[self.current_generation_meta_data['model_name'][self.best_individual_idx]],
                                                                              model_name=self.current_generation_meta_data['model_name'][self.best_individual_idx],
                                                                              cluster_params=self.current_generation_meta_data['param'][self.best_individual_idx],
                                                                              tokenize=False if self.tokenize is None else self.tokenize,
                                                                              cloud=self.cloud,
                                                                              df=self.df,
                                                                              train_data_path=self.train_data_file_path,
                                                                              sep='\t' if self.sep is None else self.sep,
                                                                              eval_method='c_umass' if self.eval_method is None else self.eval_method,
                                                                              language_model_path=self.language_model_path,
                                                                              sentence_embedding_model_path=self.sentence_embedding_model_path,
                                                                              **self.kwargs
                                                                              )
                _cluster_generator.generate_model()
                _cluster_generator.train()
                _generator: ClusteringGenerator = _cluster_generator
                self.model = _cluster_generator.model
            else:
                if self.target_type == 'reg':
                    _model_generator: ModelGeneratorReg = ModelGeneratorReg(reg_params=self.current_generation_meta_data['param'][self.best_individual_idx],
                                                                            model_name=
                                                                            self.current_generation_meta_data['model_name'][self.best_individual_idx],
                                                                            **self.kwargs
                                                                            )
                else:
                    _model_generator: ModelGeneratorClf = ModelGeneratorClf(clf_params=self.current_generation_meta_data['param'][self.best_individual_idx],
                                                                            model_name=
                                                                            self.current_generation_meta_data['model_name'][self.best_individual_idx],
                                                                            labels=self.target_labels,
                                                                            **self.kwargs
                                                                            )
                _model_generator.generate_model()
                _model_generator.train(x=copy.deepcopy(self.data_set.get('x_train').values),
                                       y=copy.deepcopy(self.data_set.get('y_train').values),
                                       validation=dict(x_val=copy.deepcopy(self.data_set.get('x_val').values),
                                                       y_val=copy.deepcopy(self.data_set.get('y_val').values)
                                                       )
                                       )
                _generator: Union[ModelGeneratorClf, ModelGeneratorReg] = _model_generator
                self.model = _model_generator.model
        if self.mlflow_log:
            if self.text_clustering:
                self._fitness(individual=_generator, ml_metric='nmi', track_metric=False)
            else:
                if self.deep_learning:
                    _generator.predict()
                else:
                    if self.target_type == 'reg':
                        self.data_set.update({'pred': _generator.predict(x=self.data_set.get('x_test').values)})
                    else:
                        self.data_set.update({'pred': _generator.predict(x=self.data_set.get('x_test').values,
                                                                         probability=False
                                                                         )
                                              })
                    _generator.eval(obs=self.data_set.get('y_test').values,
                                    pred=self.data_set.get('pred'),
                                    eval_metric=None
                                    )
                if self.target_type == 'reg':
                    self._fitness(individual=_generator, ml_metric='rmse_norm', track_metric=False)
                elif self.target_type == 'clf_multi':
                    self._fitness(individual=_generator, ml_metric='cohen_kappa', track_metric=False)
                else:
                    self._fitness(individual=_generator, ml_metric='auc', track_metric=False)
            try:
                mlflow.set_experiment(experiment_name='GA: Evolved model',
                                      experiment_id=None
                                      )
                self._mlflow_tracking(individual=_generator, register=True)
                Log(write=self.log,
                    logger_file_path=self.output_file_path
                    ).log(msg='Register model artifact on MLflow')
            except:
                Log(write=self.log,
                    logger_file_path=self.output_file_path
                    ).log(msg='Model artifact could not be registered on MLflow')

    def _input_manager(self):
        """
        Manage input options
        """
        _train_size: float = 0.8 if self.kwargs.get('train_size') is None else self.kwargs.get('train_size')
        if self.mode in ['feature_engineer', 'feature_selector', 'model']:
            if self.mode.find('feature') >= 0:
                if self.target not in self.feature_engineer.get_features():
                    raise GeneticAlgorithmException('Target feature ({}) not found in data set'.format(self.target))
                self.target_values: np.array = self.feature_engineer.get_target_values()
                if self.mode == 'feature_engineer':
                    if self.feature_engineer is None:
                        raise GeneticAlgorithmException('FeatureEngineer object not found')
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
                                    raise GeneticAlgorithmException('No training data set found')
                            else:
                                if self.train_data_file_path is None or self.test_data_file_path is None or self.valid_data_file_path is None:
                                    raise GeneticAlgorithmException('No training, testing, validation data set found')
                        else:
                            if self.text_clustering:
                                if 'x_train' not in self.data_set.keys():
                                    raise GeneticAlgorithmException('x_train not found in data dictionary')
                            else:
                                if 'x_train' not in self.data_set.keys():
                                    raise GeneticAlgorithmException('x_train not found in data dictionary')
                                if 'y_train' not in self.data_set.keys():
                                    raise GeneticAlgorithmException('y_train not found in data dictionary')
                                if 'x_test' not in self.data_set.keys():
                                    raise GeneticAlgorithmException('x_test not found in data dictionary')
                                if 'y_test' not in self.data_set.keys():
                                    raise GeneticAlgorithmException('y_test not found in data dictionary')
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
                        raise GeneticAlgorithmException('Target feature ({}) not found in data set'.format(self.target))
                    if self.features is None:
                        self.features = list(self.df.columns)
                        if not self.text_clustering:
                            del self.features[self.features.index(self.target)]
                    if not self.text_clustering:
                        _used_features: List[str] = self.features
                        _used_features.append(self.target)
                        _used_features = list(set(_used_features))
                        self.df = self.df[_used_features]
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
            raise GeneticAlgorithmException('Optimization mode ({}) not supported. Use "model", "feature_engineer" or "feature_selector" instead.'.format(self.mode))
        if self.deep_learning:
            if self.target_values is None:
                self.target_classes = self.deep_learning_output_size
                if self.deep_learning_output_size is None:
                    raise GeneticAlgorithmException('Size of the output layer of the neural network is missing')
                else:
                    if self.deep_learning_output_size < 0:
                        raise GeneticAlgorithmException('Size of the output layer of the neural network is missing')
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
        if self.force_target_type is not None:
            if self.force_target_type == 'reg' and self.target_type == 'clf_multi':
                self.target_type = 'reg'
            elif self.force_target_type == 'clf_multi' and self.target_type == 'reg':
                self.target_type = 'clf_multi'
        if self.pop_size is None:
            self.pop_size = 64
        if self.parents_ratio is None:
            self.parents_ratio = 0.5

    def _intro(self):
        """
        Print Genetic Algorithm configuration
        """
        _intro: str = 'Reinforcement learning environment started ...\nGenetic Algorithm: Optimizing mode -> {}\n' \
                      'Environment setup:\n-> Machine learning model(s): {}\n-> Model parameter: {}\n' \
                      '-> Population: {}\n-> Mutation Rate: {}\n-> Mutation Probability: {}\n-> Parent Rate: {}\n' \
                      '-> Early stopping: {}\n-> Convergence: {}\n-> Target feature: {}\n-> Target type: {}\n' \
                      '-> Features per model: {}\n'.format(self.mode,
                                                           self.models,
                                                           self.model_params,
                                                           self.pop_size,
                                                           self.mutation_rate,
                                                           self.mutation_prob,
                                                           self.parents_ratio,
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
                -> min: Compare maximum and minimum fitness score of generation
                -> median: Compare maximum and median fitness score of generation
                -> mean: Compare maximum and mean fitness score of generation

        :param threshold: float
            Conversion threshold of relative difference between maximum fitness score and comparison fitness score

        :return bool
            Whether to stop evolution because the hole generation achieve very similar gradient score or not
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
        Check whether evolutionary gradient (best fitness metric of generation) has not increased a certain amount of generations

        :param min_fitness: bool
            Use minimum fitness score each generation to evaluate stagnation

        :param median_fitness: bool
            Use median fitness score each generation to evaluate stagnation

        :param mean_fitness: bool
            Use mean fitness score each generation to evaluate stagnation

        :param max_fitness: bool
            Use maximum fitness score each generation to evaluate stagnation

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

    def _inherit(self) -> List[tuple]:
        """
        Inheritance from one parent to one child

        :return List[tuple]
            Selected combination of parent id and child id of population
        """
        _parent_child_combination: List[tuple] = []
        _min_matching_length: int = len(self.parents_idx) if len(self.parents_idx) <= len(self.child_idx) else len(self.child_idx)
        for c in range(0, _min_matching_length, 1):
            _parent_child_combination.append(tuple([self.parents_idx[c], self.child_idx[c]]))
        return _parent_child_combination

    def _mating_pool(self, crossover: bool = False):
        """
        Mutate genes of chosen parents

        :param crossover: bool
            Allow individuals to include crossover before mutating single genes as additional mutation strategy
        """
        for parent, child in self._inherit():
            if crossover:
                # Mutation including crossover strategy (used for optimizing feature engineering and selection)
                self._crossover(parent=parent, child=child)
                self._mutate(parent=parent, child=child)
            else:
                # Mutation without crossover strategy (used for optimizing ml models / parameters)
                self._mutate(parent=parent, child=child)

    def _mlflow_tracking(self, individual, register: bool = False):
        """
        Track training information using mlflow
        """
        _tags: dict = dict(n_cases=self.n_cases,
                           n_features=1 if self.text_clustering else len(self.features),
                           train_cases=self.n_cases if self.text_clustering else self.n_train_cases,
                           test_cases=0 if self.text_clustering else self.n_test_cases,
                           target=self.target,
                           target_type=self.target_type,
                           model_name=individual.model_name,
                           individual=individual.id,
                           train_time=individual.train_time
                           )
        with mlflow.start_run():
            if self.deep_learning:
                mlflow.pytorch.log_model(pytorch_model=individual.model,
                                         artifact_path='model',
                                         registered_model_name=individual.model_name if register else None
                                         )
                _params: dict = copy.deepcopy(individual.model_param)
                del _params['weights']
                mlflow.log_params(params=_params)
                for p, predictor in enumerate(individual.predictors):
                    _tags.update({f'text_feature_{p}': predictor})
                #mlflow.pytorch.log_state_dict(state_dict=individual.model.state_dict(),
                #                              artifact_path='state_dict.yml'
                #                              )
            if self.text_clustering:
                mlflow.sklearn.log_model(sk_model=individual.model,
                                         artifact_path='model',
                                         registered_model_name=individual.model_name if register else None
                                         )
                mlflow.log_metric(key=individual.eval_method, value=individual.fitness)
                mlflow.log_params(params=individual.model_param)
                _tags.update({'text_feature': individual.predictor})
            if not self.deep_learning and not self.text_clustering:
                mlflow.log_dict(dictionary=dict(features=individual.features),
                                artifact_file='features.yaml'
                                )
                mlflow.sklearn.log_model(sk_model=individual.model,
                                         artifact_path='model',
                                         registered_model_name=individual.model_name if register else None
                                         )
                mlflow.log_params(params=individual.model_param)
            #if self.current_generation_meta_data['generation'] > 0:
            #    mlflow.log_dict(dictionary=dict(features=individual.model_param_mutated),
            #                    artifact_file='param_mutation.txt'
            #                    )
            mlflow.set_tags(tags=_tags)
            if not self.text_clustering:
                for metric_context in individual.fitness:
                    for metric in individual.fitness[metric_context]:
                        if self.target_type.find('clf') >= 0:
                            if isinstance(individual.fitness[metric_context][metric], dict):
                                for clf_metric in individual.fitness[metric_context][metric]:
                                    if isinstance(individual.fitness[metric_context][metric][clf_metric], dict):
                                        for clf_metric_element in individual.fitness[metric_context][metric][clf_metric]:
                                            mlflow.log_metric(key=f'{clf_metric}_{clf_metric_element}_{metric_context}',
                                                              value=individual.fitness[metric_context][metric][clf_metric][clf_metric_element]
                                                              )
                                    else:
                                        mlflow.log_metric(key=f'{clf_metric}_{metric_context}',
                                                          value=individual.fitness[metric_context][metric][clf_metric]
                                                          )
                            else:
                                if metric == 'confusion':
                                    continue
                                mlflow.log_metric(key=f'{metric}_{metric_context}',
                                                  value=individual.fitness[metric_context][metric]
                                                  )
                        else:
                            mlflow.log_metric(key=f'{metric}_{metric_context}',
                                              value=individual.fitness[metric_context][metric]
                                              )
            mlflow.log_metric(key='sml_score', value=individual.fitness_score)
            self._track_visualization()

    def _modeling(self, pop_idx: int):
        """
        Generate, train and evaluate supervised & unsupervised machine learning model

        :param pop_idx: int
            Population index number
        """
        self._collect_meta_data(current_gen=False, idx=pop_idx)
        _re: int = 0
        _re_generate: bool = False
        _re_generate_max: int = 50
        while True:
            _re += 1
            try:
                if self.mode == 'model':
                    if _re_generate:
                        if self.verbose:
                            Log(write=self.log, logger_file_path=self.output_file_path).log('Re-generate individual {}'.format(pop_idx))
                        if self.deep_learning:
                            self.population[pop_idx].generate_model()
                        else:
                            self.population[pop_idx].generate_params(param_rate=self.mutation_rate)
                elif self.mode == 'feature_engineer':
                    if self.verbose:
                        Log(write=self.log, logger_file_path=self.output_file_path).log('Re-sample features for individual {}'.format(pop_idx))
                    self._sampling(features=self.feature_pairs[pop_idx])
                    if self.deep_learning:
                        self.population[pop_idx].update_data(x_train=self.data_set.get('x_train'),
                                                             y_train=self.data_set.get('y_train'),
                                                             x_test=self.data_set.get('x_test'),
                                                             y_test=self.data_set.get('x_test'),
                                                             x_valn=self.data_set.get('x_val'),
                                                             y_valn=self.data_set.get('y_val')
                                                             )
                if self.verbose:
                    Log(write=self.log, logger_file_path=self.output_file_path).log('Train individual {}'.format(pop_idx))
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
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg='Error while training model ({})\n{}'.format(self.population[pop_idx].model_name, e))
        if _re == _re_generate_max:
            raise GeneticAlgorithmException('Maximum number of errors occurred. Check last error message ...')
        if self.verbose:
            Log(write=self.log, logger_file_path=self.output_file_path).log('Evaluate training of individual {}'.format(pop_idx))
        if self.text_clustering:
            self._fitness(individual=self.population[pop_idx], ml_metric='nmi')
        else:
            if self.target_type == 'reg':
                if self.deep_learning:
                    self.population[pop_idx].predict()
                    self.data_set = dict(y_test=self.population[pop_idx].obs,
                                         pred=self.population[pop_idx].pred
                                         )
                else:
                    self.data_set.update({'pred': self.population[pop_idx].predict(x=self.data_set.get('x_test').values)})
                    self.population[pop_idx].eval(obs=self.data_set.get('y_test').values,
                                                  pred=self.data_set.get('pred'),
                                                  eval_metric=None
                                                  )
                self._fitness(individual=self.population[pop_idx], ml_metric='rmse_norm')
            else:
                if self.deep_learning:
                    self.population[pop_idx].predict()
                    self.data_set = dict(y_test=self.population[pop_idx].obs,
                                         pred=self.population[pop_idx].pred
                                         )
                else:
                    self.data_set.update({'pred': self.population[pop_idx].predict(x=self.data_set.get('x_test').values,
                                                                                   probability=False
                                                                                   )
                                          })
                    self.population[pop_idx].eval(obs=self.data_set.get('y_test').values,
                                                  pred=self.data_set.get('pred'),
                                                  eval_metric=None
                                                  )
                if self.target_type == 'clf_multi':
                    self._fitness(individual=self.population[pop_idx], ml_metric='cohen_kappa')
                else:
                    self._fitness(individual=self.population[pop_idx], ml_metric='auc')
        self._collect_meta_data(current_gen=True, idx=pop_idx)

    def _mutate(self, parent: int, child: int, force_param: dict = None):
        """
        Mutate individual

        :param parent: int
            Index number of parent in population

        :param child: int
            Index number of child in population

        :param force_param: dict
            Model parameter config to force during mutation
        """
        if self.mode.find('model') >= 0:
            if np.random.uniform(low=0, high=1) > self.mutation_prob:
                # Generate new individual:
                if self.verbose:
                    Log(write=self.log, logger_file_path=self.output_file_path).log('Generate new model for individual {}'.format(child))
                if self.mode == 'model_sampler':
                    self._sampling(features=self.population[child].features)
                if self.text_clustering:
                    _cluster_generator: ClusteringGenerator = ClusteringGenerator(predictor=self.features[0],
                                                                                  models=self.models,
                                                                                  tokenize=False if self.tokenize is None else self.tokenize,
                                                                                  cloud=self.cloud,
                                                                                  df=self.df,
                                                                                  train_data_path=self.train_data_file_path,
                                                                                  sep='\t' if self.sep is None else self.sep,
                                                                                  eval_method='c_umass' if self.eval_method is None else self.eval_method,
                                                                                  language_model_path=self.language_model_path,
                                                                                  sentence_embedding_model_path=self.sentence_embedding_model_path,
                                                                                  **self.kwargs
                                                                                  )
                    _cluster_generator.generate_model()
                    self.population[child] = _cluster_generator
                else:
                    if self.deep_learning:
                        # if self.warm_start_strategy == 'adaptive':
                        _hidden_layer_size: int = self.population[child].hidden_layer_size
                        _network_generator: NetworkGenerator = NetworkGenerator(target=self.target,
                                                                                predictors=self.features,
                                                                                output_layer_size=self.deep_learning_output_size,
                                                                                train_data_path=self.train_data_file_path,
                                                                                test_data_path=self.test_data_file_path,
                                                                                validation_data_path=self.valid_data_file_path,
                                                                                models=self.models,
                                                                                hidden_layer_size=_hidden_layer_size,
                                                                                hidden_layer_size_category=self.warm_start_constant_category,
                                                                                sep='\t' if self.sep is None else self.sep,
                                                                                cache_dir=self.cache_dir,
                                                                                **self.kwargs
                                                                                )
                        _network_generator.generate_model()
                        self.population[child] = _network_generator
                    else:
                        if self.target_type == 'reg':
                            _model_generator: ModelGeneratorReg = ModelGeneratorReg(models=self.models, **self.kwargs)
                        else:
                            _model_generator: ModelGeneratorClf = ModelGeneratorClf(models=self.models, **self.kwargs)
                        _model_generator.generate_model()
                        self.population[child] = _model_generator
            else:
                # Mutate individual:
                if self.text_clustering:
                    _cluster_generator: ClusteringGenerator = ClusteringGenerator(predictor=self.features[0],
                                                                                  models=self.models,
                                                                                  model_name=self.population[parent].model_name,
                                                                                  cluster_params=self.population[parent].model_param,
                                                                                  tokenize=False if self.tokenize is None else self.tokenize,
                                                                                  cloud=self.cloud,
                                                                                  df=self.df,
                                                                                  train_data_path=self.train_data_file_path,
                                                                                  sep='\t' if self.sep is None else self.sep,
                                                                                  eval_method='c_umass' if self.eval_method is None else self.eval_method,
                                                                                  language_model_path=self.language_model_path,
                                                                                  sentence_embedding_model_path=self.sentence_embedding_model_path,
                                                                                  **self.kwargs
                                                                                  )
                    _cluster_generator.generate_model()
                    _cluster_generator.generate_params(param_rate=self.mutation_rate, force_param=force_param)
                    self.population[child] = _cluster_generator
                else:
                    if self.verbose:
                        Log(write=self.log, logger_file_path=self.output_file_path).log('Mutate individual {} inherited from parent {}'.format(child, parent))
                    if self.deep_learning:
                        _network_generator: NetworkGenerator = NetworkGenerator(target=self.target,
                                                                                predictors=self.features,
                                                                                output_layer_size=self.deep_learning_output_size,
                                                                                train_data_path=self.train_data_file_path,
                                                                                test_data_path=self.test_data_file_path,
                                                                                validation_data_path=self.valid_data_file_path,
                                                                                models=self.models,
                                                                                model_name=self.population[parent].model_name,
                                                                                input_param=self.population[parent].model_param,
                                                                                hidden_layer_size=self.population[parent].hidden_layer_size,
                                                                                hidden_layer_size_category=self.warm_start_constant_category,
                                                                                sep='\t' if self.sep is None else self.sep,
                                                                                cache_dir=self.cache_dir,
                                                                                **self.kwargs
                                                                                )
                        _network_generator.generate_model()
                        _network_generator.generate_params(param_rate=self.mutation_rate)
                        self.population[child] = _network_generator
                    else:
                        if self.target_type == 'reg':
                            _model_generator: ModelGeneratorReg = ModelGeneratorReg(model_name=self.population[parent].model_name,
                                                                                    reg_params=self.population[parent].model_param,
                                                                                    models=self.models,
                                                                                    **self.kwargs
                                                                                    )
                        else:
                            _model_generator: ModelGeneratorClf = ModelGeneratorClf(model_name=self.population[parent].model_name,
                                                                                    clf_params=self.population[parent].model_param,
                                                                                    models=self.models,
                                                                                    **self.kwargs
                                                                                    )
                        _model_generator.generate_model()
                        _model_generator.generate_params(param_rate=self.mutation_rate, force_param=force_param)
                        self.population[child] = _model_generator
            if self.verbose:
                Log(write=self.log, logger_file_path=self.output_file_path).log('Hyperparameter setting of individual {}: {}'.format(child, self.population[child].model_param))
        elif self.mode.find('feature') >= 0:
            _new_features: List[str] = []
            _feature_pool: List[str] = self.feature_pairs[np.random.choice(a=self.parents_idx)]
            for feature in self.feature_pairs[child]:
                if feature in self.feature_pairs[parent]:
                    if self.mode == 'feature_engineer':
                        if np.random.uniform(low=0, high=1) <= self.mutation_prob:
                            if self.verbose:
                                Log(write=self.log, logger_file_path=self.output_file_path).log('Generate new feature for individual {} using feature: {}'.format(child, feature))
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
                            self.mutated_features['parent'].append(feature)
                            self.mutated_features['child'].append(_new_features[-1])
                            self.mutated_features['generation'].append(feature)
                            self.mutated_features['action'].append(self.feature_engineer.get_last_action())
                        else:
                            _new_features.append(feature)
                    elif self.mode == 'feature_selector':
                        _new_features.append(feature)
                else:
                    _new_features.append(feature)
            self.feature_pairs[child] = copy.deepcopy(_new_features)
            if self.verbose:
                Log(write=self.log, logger_file_path=self.output_file_path).log('Feature setting of individual {}: {}'.format(child, self.feature_pairs[child]))

    def _natural_selection(self):
        """
        Select best individuals of population as parents for next generation
        """
        # Calculate number of parents within current generation:
        _count_parents: int = int(self.pop_size * self.parents_ratio)
        # Rank individuals according to their fitness score:
        _sorted_fitness_matrix: pd.DataFrame = pd.DataFrame(data=dict(fitness=self.current_generation_meta_data.get('fitness_score'))).sort_values(by='fitness', axis=0, ascending=False)
        # Set parents:
        self.parents_idx = _sorted_fitness_matrix[0:_count_parents].index.values.tolist()
        # Set children:
        self.child_idx = _sorted_fitness_matrix[_count_parents:].index.values.tolist()

    def _populate(self):
        """
        Populate generation zero with individuals
        """
        if self.text_clustering:
            _warm_model: dict = {}
            if self.warm_start:
                _warm_model = ClusteringGenerator(predictor=self.features[0],
                                                  models=self.models,
                                                  tokenize=False if self.tokenize is None else self.tokenize,
                                                  cloud=self.cloud,
                                                  df=self.df,
                                                  train_data_path=self.train_data_file_path,
                                                  sep='\t' if self.sep is None else self.sep,
                                                  eval_method='c_umass' if self.eval_method is None else self.eval_method,
                                                  language_model_path=self.language_model_path,
                                                  sentence_embedding_model_path=self.sentence_embedding_model_path,
                                                  **self.kwargs
                                                  ).get_model_parameter()
            for p in range(0, self.pop_size, 1):
                if self.evolution_continue:
                    _params: dict = self.final_generation.get('param')
                else:
                    if self.generation_zero is not None:
                        if p < len(self.generation_zero):
                            self.population.append(self.generation_zero[p])
                            continue
                    if len(_warm_model.keys()) > 0:
                        if p + 1 > len(_warm_model.keys()):
                            _params: dict = self.model_params
                        else:
                            _params: dict = _warm_model.get(list(_warm_model.keys())[p])
                    else:
                        _params: dict = self.model_params
                if self.verbose:
                    Log(write=self.log, logger_file_path=self.output_file_path).log('Populate individual {}'.format(p))
                _cluster_generator: ClusteringGenerator = ClusteringGenerator(predictor=self.features[0],
                                                                              models=self.models,
                                                                              cluster_params=_params,
                                                                              tokenize=False if self.tokenize is None else self.tokenize,
                                                                              cloud=self.cloud,
                                                                              df=self.df,
                                                                              train_data_path=self.train_data_file_path,
                                                                              sep='\t' if self.sep is None else self.sep,
                                                                              eval_method='c_umass' if self.eval_method is None else self.eval_method,
                                                                              language_model_path=self.language_model_path,
                                                                              sentence_embedding_model_path=self.sentence_embedding_model_path,
                                                                              **self.kwargs
                                                                              )
                _cluster_generator.generate_model()
                self.population.append(_cluster_generator)
                if self.verbose:
                    Log(write=self.log, logger_file_path=self.output_file_path).log('Hyperparameter setting of individual {}: {}'.format(p, self.population[p].model_param))
        else:
            _warm_model: dict = {}
            if self.warm_start:
                if self.target_type == 'reg':
                    _warm_model = ModelGeneratorReg(models=self.models, **self.kwargs).get_model_parameter()
                else:
                    _warm_model = ModelGeneratorClf(models=self.models, labels=self.target_labels, **self.kwargs).get_model_parameter()
            for p in range(0, self.pop_size, 1):
                if self.mode.find('feature') >= 0:
                    if self.verbose:
                        Log(write=self.log, logger_file_path=self.output_file_path).log('Sample features for individual {}'.format(p))
                    self._sampling(features=self.feature_pairs[p])
                if self.evolution_continue:
                    _params: dict = self.final_generation.get('param')
                else:
                    if self.generation_zero is not None:
                        if p < len(self.generation_zero):
                            self.population.append(self.generation_zero[p])
                            continue
                    if len(_warm_model.keys()) > 0:
                        if p + 1 > len(_warm_model.keys()):
                            _params: dict = self.model_params
                        else:
                            _params: dict = _warm_model.get(list(_warm_model.keys())[p])
                    else:
                        _params: dict = self.model_params
                if self.verbose:
                    Log(write=self.log, logger_file_path=self.output_file_path).log('Populate individual {}'.format(p))
                if self.target_type == 'reg':
                    _model_generator: ModelGeneratorReg = ModelGeneratorReg(reg_params=_params, models=self.models, **self.kwargs)
                else:
                    _model_generator: ModelGeneratorClf = ModelGeneratorClf(clf_params=_params, models=self.models, labels=self.target_labels, **self.kwargs)
                _model_generator.generate_model()
                self.population.append(_model_generator)
                if self.verbose:
                    Log(write=self.log, logger_file_path=self.output_file_path).log('Hyperparameter setting of individual {}: {}'.format(p, self.population[p].model_param))

    def _populate_networks(self):
        """
        Populate generation zero (with neural networks only)
        """
        _model_param = None
        if self.warm_start:
            _n_vanilla_networks: int = int(self.pop_size * 0.5)
            _n_vanilla_networks_per_model: int = int(_n_vanilla_networks / len(self.models))
            for model in self.models:
                _p: int = 0
                while _p <= _n_vanilla_networks_per_model:
                    if self.evolution_continue:
                        _model_param: dict = self.final_generation[str(_p)].get('param')
                    if self.verbose:
                        Log(write=self.log, logger_file_path=self.output_file_path).log('Populate individual {}'.format(_p))
                    _p += 1
                    _network_generator: NetworkGenerator = NetworkGenerator(target=self.target,
                                                                            predictors=self.features,
                                                                            output_layer_size=self.deep_learning_output_size,
                                                                            train_data_path=self.train_data_file_path,
                                                                            test_data_path=self.test_data_file_path,
                                                                            validation_data_path=self.valid_data_file_path,
                                                                            model_name=model,
                                                                            input_param=_model_param,
                                                                            hidden_layer_size=self.warm_start_constant_hidden_layers,
                                                                            hidden_layer_size_category=self.warm_start_constant_category,
                                                                            sep='\t' if self.sep is None else self.sep,
                                                                            cache_dir=self.cache_dir,
                                                                            **self.kwargs
                                                                            )
                    _network_generator.get_vanilla_model()
                    self.population.append(_network_generator)
                    if self.verbose:
                        Log(write=self.log, logger_file_path=self.output_file_path).log('Hyperparameter setting of individual {}: {}'.format(_p - 1, self.population[_p - 1].model_param))
        else:
            _n_vanilla_networks: int = 0
        for p in range(0, self.pop_size - _n_vanilla_networks - 1, 1):
            if self.evolution_continue:
                _model_param: dict = self.final_generation[str(p + len(self.population))].get('param')
            else:
                if self.generation_zero is not None:
                    if p < len(self.generation_zero):
                        self.population.append(self.generation_zero[p])
                        continue
            if self.mode.find('feature') >= 0:
                if self.verbose:
                    Log(write=self.log, logger_file_path=self.output_file_path).log('Sample features for individual {}'.format(p))
                self._sampling(features=self.feature_pairs[p])
            if self.verbose:
                Log(write=self.log, logger_file_path=self.output_file_path).log('Populate individual {}'.format(p))
            _network_generator: NetworkGenerator = NetworkGenerator(target=self.target,
                                                                    predictors=self.features,
                                                                    output_layer_size=self.deep_learning_output_size,
                                                                    train_data_path=self.train_data_file_path,
                                                                    test_data_path=self.test_data_file_path,
                                                                    validation_data_path=self.valid_data_file_path,
                                                                    models=self.models,
                                                                    input_param=_model_param,
                                                                    hidden_layer_size=self.warm_start_constant_hidden_layers,
                                                                    hidden_layer_size_category=self.warm_start_constant_category,
                                                                    sep='\t' if self.sep is None else self.sep,
                                                                    cache_dir=self.cache_dir,
                                                                    **self.kwargs
                                                                    )
            _network_generator.generate_model()
            self.population.append(_network_generator)
            if self.verbose:
                Log(write=self.log, logger_file_path=self.output_file_path).log('Hyperparameter setting of individual {}: {}'.format(p, self.population[p].model_param))

    def _post_processing(self):
        """
        Post-process evolution
        """
        if self.mode.find('feature') >= 0:
            for parent in self.parents_idx:
                self.evolved_features.extend(self.feature_pairs[parent])
            self.evolved_features = list(set(self.evolved_features))
        self.best_individual_idx = np.array(self.current_generation_meta_data['fitness_score']).argmax()
        if self.deploy_model:
            self._generate_final_model()
        if self.stopping_reason is not None:
            Log(write=self.log, logger_file_path=self.output_file_path).log(msg='Best model: {} - {}'.format(self.current_generation_meta_data['model_name'][self.best_individual_idx], self.current_generation_meta_data['param'][self.best_individual_idx]))
            Log(write=self.log, logger_file_path=self.output_file_path).log(msg='Fitness score: {}'.format(self.current_generation_meta_data['fitness_score'][self.best_individual_idx]))
            Log(write=self.log, logger_file_path=self.output_file_path).log(msg='Fitness metric: {}'.format(self.current_generation_meta_data['fitness_metric'][self.best_individual_idx]))
        self._gather_final_generation()
        if self.deep_learning:
            self.data_set = dict(y_test=self.population[self.best_individual_idx].obs,
                                 pred=self.population[self.best_individual_idx].pred
                                 )
        else:
            if self.mode.find('model') >= 0 and self.plot:
                self.data_set.update({'pred': self.model.predict(self.data_set.get('x_test'))})
        self.evolution: dict = dict(
            model_name=self.current_generation_meta_data['model_name'][self.best_individual_idx],
            param=self.current_generation_meta_data['param'][self.best_individual_idx],
            param_mutated=self.current_generation_meta_data['param_mutated'][self.best_individual_idx],
            fitness_score=self.current_generation_meta_data['fitness_score'][self.best_individual_idx],
            fitness_metric=self.current_generation_meta_data['fitness_metric'][self.best_individual_idx],
            epoch_metric_score=self.population[self.best_individual_idx].epoch_eval if self.deep_learning else None,
            features=self.features if self.text_clustering or self.deep_learning else
            self.current_generation_meta_data['features'][self.best_individual_idx],
            target=self.target,
            target_type=self.target_type,
            re_split_data=self.re_split_data,
            re_split_cases=self.re_sample_cases,
            re_sample_features=self.re_sample_features,
            id=self.current_generation_meta_data['id'][self.best_individual_idx],
            mode=self.mode,
            generations=self.current_generation_meta_data['generation'],
            parent_ratio=self.parents_ratio,
            mutation_prob=self.mutation_prob,
            mutation_rate=self.mutation_rate,
            mutated_features=self.mutated_features,
            generation_history=self.generation_history,
            evolution_history=self.evolution_history,
            evolution_gradient=self.evolution_gradient,
            convergence_check=self.convergence,
            convergence_measure=self.convergence_measure,
            early_stopping=self.early_stopping,
            max_time=self.timer,
            start_time=self.start_time,
            end_time=str(datetime.now()),
            stopping_reason=self.stopping_reason
            )
        if self.plot and self.stopping_reason is not None:
            self.visualize(results_table=True,
                           model_distribution=True,
                           model_evolution=True,
                           param_distribution=False,
                           train_time_distribution=True,
                           breeding_map=True,
                           breeding_graph=True,
                           fitness_distribution=True,
                           fitness_evolution=True if self.current_generation_meta_data['generation'] > 0 else False,
                           fitness_dimensions=True,
                           per_generation=True if self.current_generation_meta_data['generation'] > 0 else False,
                           prediction_of_best_model=True,
                           epoch_stats=True
                           )
        if self.output_file_path is not None:
            if len(self.output_file_path) > 0:
                self.save_evolution(ga=True,
                                    model=self.deploy_model,
                                    evolution_history=False,
                                    generation_history=False,
                                    final_generation=False
                                    )

    def _re_populate(self):
        """
        Re-populate generation 0 to increase likelihood for a good evolution start
        """
        Log(write=self.log, logger_file_path=self.output_file_path).log(msg='Re-populate generation 0 because of the poor fitness scoring of all individuals')
        self.n_individuals = -1
        for gen_history in self.generation_history.keys():
            self.generation_history[gen_history] = {}
        for evo_history in self.evolution_history.keys():
            self.evolution_history[evo_history] = []
        for gen_cur in self.current_generation_meta_data.keys():
            if isinstance(self.current_generation_meta_data[gen_cur], list):
                self.current_generation_meta_data[gen_cur] = []
            elif isinstance(self.current_generation_meta_data[gen_cur], int):
                self.current_generation_meta_data[gen_cur] = 0
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

    def _save_checkpoint(self):
        """
        Save checkpoint
        """
        self._post_processing()

    def _track_visualization(self):
        """
        Track model visualization using mlflow
        """
        if not self.text_clustering:
            _charts: dict = {}
            _file_paths: List[str] = []
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
            if self.target_type == 'reg':
                _file_paths.append(os.path.join(self.output_file_path, 'evaluation_coords.html'))
                DataVisualizer(df=_best_model_results,
                               title='Prediction Evaluation of final inherited ML Model:',
                               features=['obs', 'abs_diff', 'rel_diff', 'pred'],
                               color_feature='pred',
                               plot_type='parcoords',
                               file_path=self.output_file_path if self.output_file_path is None else os.path.join(
                                   self.output_file_path, 'evaluation_coords.html'),
                               ).run()
                _file_paths.append(os.path.join(self.output_file_path, 'scatter_contour.html'))
                DataVisualizer(df=_best_model_results,
                               title='Prediction vs. Observation of final inherited ML Model:',
                               features=['obs', 'pred'],
                               plot_type='joint',
                               file_path=self.output_file_path if self.output_file_path is None else os.path.join(
                                   self.output_file_path, 'scatter_contour.html'),
                               ).run()
            else:
                _eval_clf: EvalClf = EvalClf(obs=self.data_set.get('y_test'),
                                             pred=self.data_set.get('pred'),
                                             labels=self.target_labels
                                             )
                _confusion_matrix: pd.DataFrame = pd.DataFrame(data=_eval_clf.confusion(),
                                                               index=[f'{label}_obs' for label in self.target_labels],
                                                               columns=[f'{label}_pred' for label in self.target_labels]
                                                               )
                _cf_row_sum = pd.DataFrame()
                _cf_row_sum[' '] = _confusion_matrix.sum()
                _confusion_matrix = pd.concat([_confusion_matrix, _cf_row_sum.transpose()], axis=0)
                _cf_col_sum = pd.DataFrame()
                _cf_col_sum[' '] = _confusion_matrix.transpose().sum()
                _confusion_matrix = pd.concat([_confusion_matrix, _cf_col_sum], axis=1)
                _file_paths.append(os.path.join(self.output_file_path, 'confusion_table.html'))
                DataVisualizer(df=_confusion_matrix,
                               title='Confusion Matrix:',
                               features=_confusion_matrix.columns.to_list(),
                               plot_type='table',
                               file_path=self.output_file_path if self.output_file_path is None else os.path.join(
                                   self.output_file_path, 'confusion_table.html'),
                               ).run()
                _file_paths.append(os.path.join(self.output_file_path, 'confusion_heatmap.html'))
                DataVisualizer(df=_confusion_matrix,
                               title='Confusion Matrix Heatmap:',
                               features=_confusion_matrix.columns.to_list(),
                               plot_type='heat',
                               file_path=self.output_file_path if self.output_file_path is None else os.path.join(
                                   self.output_file_path, 'confusion_heatmap.html'),
                               ).run()
                _confusion_matrix_normalized: pd.DataFrame = pd.DataFrame(data=EvalClf(obs=self.data_set.get('y_test'),
                                                                                       pred=self.data_set.get('pred')
                                                                                       ).confusion(normalize='pred'),
                                                                          # index=['obs', 'pred'],
                                                                          # columns=['obs', 'pred']
                                                                          )
                _file_paths.append(os.path.join(self.output_file_path, 'confusion_normal_heatmap.html'))
                DataVisualizer(df=_confusion_matrix_normalized,
                               title='Confusion Matrix Normalized Heatmap:',
                               features=_confusion_matrix_normalized.columns.to_list(),
                               plot_type='heat',
                               file_path=self.output_file_path if self.output_file_path is None else os.path.join(
                                   self.output_file_path, 'confusion_normal_heatmap.html'),
                               ).run()
                _file_paths.append(os.path.join(self.output_file_path, 'clf_report_table.html'))
                DataVisualizer(df=_best_model_results,
                               title='Classification Report:',
                               features=_best_model_results.columns.to_list(),
                               plot_type='table',
                               file_path=self.output_file_path if self.output_file_path is None else os.path.join(
                                   self.output_file_path, 'clf_report_table.html'),
                               ).run()
                _classification_report: dict = _eval_clf.classification_report()
                _confusion_metrics: dict = dict(precision=[], recall=[], f1=[])
                for label in self.target_labels:
                    _confusion_metrics['precision'].append(_classification_report.get(label)['precision'])
                    _confusion_metrics['recall'].append(_classification_report.get(label)['recall'])
                    _confusion_metrics['f1'].append(_classification_report.get(label)['f1-score'])
                _file_paths.append(os.path.join(self.output_file_path, 'clf_metrics_table.html'))
                _clf_metrics: pd.DataFrame = pd.DataFrame(data=_confusion_metrics,
                                                          index=self.target_labels,
                                                          columns=list(_confusion_metrics.keys())
                                                          )
                DataVisualizer(df=_clf_metrics,
                               title='Classification Metrics:',
                               features=_clf_metrics.columns.to_list(),
                               plot_type='table',
                               file_path=self.output_file_path if self.output_file_path is None else os.path.join(
                                   self.output_file_path, 'clf_metrics_table.html'),
                               ).run()
                if self.target_type == 'clf_multi':
                    _file_paths.append(os.path.join(self.output_file_path, 'evaluation_category.html'))
                    DataVisualizer(df=_best_model_results,
                                   title='Prediction Evaluation of final inherited ML Model:',
                                   features=['obs', 'abs_diff', 'pred'],
                                   color_feature='pred',
                                   plot_type='parcoords',
                                   file_path=self.output_file_path if self.output_file_path is None else os.path.join(
                                       self.output_file_path, 'evaluation_category.html'),
                                   ).run()
                else:
                    _roc_curve = pd.DataFrame()
                    _roc_curve_values: dict = EvalClf(obs=_best_model_results['obs'],
                                                      pred=_best_model_results['pred']
                                                      ).roc_curve()
                    _roc_curve['roc_curve'] = _roc_curve_values['true_positive_rate'][1]
                    _roc_curve['baseline'] = _roc_curve_values['false_positive_rate'][1]
                    _file_paths.append(os.path.join(self.output_file_path, 'roc_auc_curve.html'))
                    DataVisualizer(df=_roc_curve,
                                   title='ROC-AUC Curve:',
                                   features=['roc_curve', 'baseline'],
                                   time_features=['baseline'],
                                   plot_type='line',
                                   # xaxis_label=['False Positive Rate'],
                                   # yaxis_label=['True Positive Rate'],
                                   use_auto_extensions=False,
                                   file_path=self.output_file_path if self.output_file_path is None else os.path.join(
                                       self.output_file_path, 'roc_auc_curve.html'),
                                   ).run()
            for path in _file_paths:
                _file_name: str = path.split('/')[-1].replace('.html', '')
                try:
                    mlflow.log_artifact(local_path=path, artifact_path=_file_name)
                except FileNotFoundError:
                    Log(write=self.log, logger_file_path=self.output_file_path).log(f'File artifact {path} not found')

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
                if i not in self.parents_idx:
                    if self.multi_threading:
                        _threads.update({i: _thread_pool.apply_async(func=self._modeling, args=[i])})
                    else:
                        self._modeling(pop_idx=i)
                    if self.mlflow_log:
                        self._mlflow_tracking(individual=self.population[i], register=False)
            if self.multi_threading:
                for thread in _threads.keys():
                    _threads.get(thread).get()
            self._collect_meta_data(current_gen=False, idx=None)
            if self.current_generation_meta_data.get('generation') == 0:
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
            self.current_generation_meta_data['generation'] += 1
        else:
            self.current_generation_meta_data['generation'] = 0
        _evolve: bool = True
        if self.deep_learning:
            self._populate_networks()
        else:
            self._populate()
        while _evolve:
            if self.mlflow_log:
                mlflow.set_experiment(experiment_name=f'GA: Gen {self.current_generation_meta_data["generation"]}',
                                      experiment_id=None
                                      )
            Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Generation: {self.current_generation_meta_data["generation"]} / {self.max_generations}')
            if self.current_generation_meta_data['generation'] > 0:
                self.n_threads = len(self.child_idx)
            if self.deep_learning:
                if self.warm_start:
                    if self.warm_start_strategy == 'monotone':
                        self.warm_start_constant_hidden_layers += 1
            self._trainer()
            self.current_generation_meta_data['generation'] += 1
            if (self.mode.find('model') >= 0) and (self.current_generation_meta_data['generation'] > self.burn_in_generations):
                if self.convergence:
                    if self._is_gradient_converged(compare=self.convergence_measure, threshold=0.05):
                        _evolve = False
                        self.stopping_reason = 'gradient_converged'
                        Log(write=self.log, logger_file_path=self.output_file_path).log(msg='Fitness metric (gradient) has converged. Therefore the evolution stops at generation {}'.format(self.current_generation_meta_data.get('generation')))
                if self.early_stopping > 0:
                    if self._is_gradient_stagnating(min_fitness=True, median_fitness=True, mean_fitness=True, max_fitness=True):
                        _evolve = False
                        self.stopping_reason = 'gradient_stagnating'
                        Log(write=self.log, logger_file_path=self.output_file_path).log(msg='Fitness metric (gradient) per generation has not increased a certain amount of generations ({}). Therefore the evolution stops early at generation {}'.format(self.early_stopping, self.current_generation_meta_data.get('generation')))
            if (datetime.now() - self.start_time).seconds >= self.timer:
                _evolve = False
                self.stopping_reason = 'time_exceeded'
                Log(write=self.log, logger_file_path=self.output_file_path).log('Time exceeded:{}'.format(self.timer))
            if self.current_generation_meta_data['generation'] > self.max_generations:
                _evolve = False
                self.stopping_reason = 'max_generation_evolved'
                Log(write=self.log, logger_file_path=self.output_file_path).log(msg='Maximum number of generations reached: {}'.format(self.max_generations))
            self._natural_selection()
            if _evolve:
                self._mating_pool(crossover=False if self.mode == 'model' else True)
            if self.checkpoint and _evolve and self.mode.find('feature') < 0:
                if self.verbose:
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg='Save checkpoint ...')
                self._save_checkpoint()
        self._post_processing()

    def optimize_continue(self, deploy_model: bool = True, max_generations: int = 5):
        """
        Continue evolution by using last generation of previous evolution as new generation 0

        :param deploy_model: bool
            Deploy fittest model to cloud platform

        :param max_generations: int
            Maximum number of generations
        """
        self.data_set = None
        self.evolution_continue = True
        self.deploy_model = deploy_model
        _max_gen: int = max_generations if max_generations > 0 else 5
        self.max_generations: int = self.max_generations + _max_gen
        self.burn_in_generations += self.current_generation_meta_data['generation']
        self.optimize()

    def save_evolution(self,
                       ga: bool = True,
                       model: bool = True,
                       evolution_history: bool = False,
                       generation_history: bool = False,
                       final_generation: bool = False
                       ):
        """
        Save evolution metadata generated by genetic algorithm to local hard drive as pickle file

        :param ga: bool
            Save GeneticAlgorithm class object (required for continuing evolution / optimization)

        :param model: bool
            Save evolved model

        :param evolution_history: bool
            Save evolution history metadata

        :param generation_history: bool
            Save generation history metadata

        :param final_generation: bool
            Save settings of each individual of final generation
        """
        # Export evolution history data:
        if evolution_history:
            DataExporter(obj=self.evolution_history,
                         file_path=os.path.join(self.output_file_path, 'evolution_history.p'),
                         create_dir=False,
                         overwrite=True,
                         cloud=self.cloud,
                         bucket_name=self.bucket_name,
                         region=self.kwargs.get('region')
                         ).file()
        # Export generation history data:
        if generation_history:
            DataExporter(obj=self.generation_history,
                         file_path=os.path.join(self.output_file_path, 'generation_history.p'),
                         create_dir=False,
                         overwrite=True,
                         cloud=self.cloud,
                         bucket_name=self.bucket_name,
                         region=self.kwargs.get('region')
                         ).file()
        if final_generation:
            DataExporter(obj=self.final_generation,
                         file_path=os.path.join(self.output_file_path, 'final_generation.p'),
                         create_dir=False,
                         overwrite=True,
                         cloud=self.cloud,
                         bucket_name=self.bucket_name,
                         region=self.kwargs.get('region')
                         ).file()
        # Export evolved model:
        if model:
            _file_name_extension: str = '' if self.kwargs.get('model_file_name_extension') is None else '_{}'.format(self.kwargs.get('model_file_name_extension'))
            _file_name: str = 'model{}.p'.format(_file_name_extension)
            if self.cloud is None:
                if self.deep_learning:
                    torch.save(obj=self.model, f=os.path.join(self.output_file_path, _file_name))
                else:
                    DataExporter(obj=self.model,
                                 file_path=os.path.join(self.output_file_path, _file_name),
                                 create_dir=False,
                                 overwrite=True
                                 ).file()
            else:
                if self.current_generation_meta_data['model_name'][self.best_individual_idx] == 'trans':
                    DataExporter(obj=self.model.model,
                                 file_path=os.path.join(self.output_file_path, _file_name),
                                 create_dir=False,
                                 overwrite=True,
                                 cloud=self.cloud,
                                 bucket_name=self.bucket_name,
                                 region=self.kwargs.get('region')
                                 ).file()
                else:
                    DataExporter(obj=self.model,
                                 file_path=os.path.join(self.output_file_path, _file_name),
                                 create_dir=False,
                                 overwrite=True,
                                 cloud=self.cloud,
                                 bucket_name=self.bucket_name,
                                 region=self.kwargs.get('region')
                                 ).file()
        # Export GeneticAlgorithm class object:
        if ga:
            if not self.deep_learning:
                if self.stopping_reason is None:
                    self.feature_engineer = None
                else:
                    self.df = None
                    self.model = None
                    self.population = []
                    self.feature_engineer = None
                _file_name_extension: str = '' if self.kwargs.get('ga_file_name_extension') is None else '_{}'.format(self.kwargs.get('ga_file_name_extension'))
                _file_name: str = 'genetic{}.p'.format(_file_name_extension)
                DataExporter(obj=self,
                             file_path=os.path.join(self.output_file_path, _file_name),
                             create_dir=False,
                             overwrite=True,
                             cloud=self.cloud,
                             bucket_name=self.bucket_name,
                             region=self.kwargs.get('region')
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
                  per_generation: bool = True,
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

        :param per_generation: bool
            Visualize results of each generation in detail or visualize just evolutionary results

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
        _evolution_gradient_data['generation'] = [i for i in range(0, len(self.evolution_gradient.get('max')), 1)]
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
            DataVisualizer(df=_evolution_history_data,
                           title='Results of Genetic Algorithm:',
                           features=_evolution_history_data.columns.to_list(),
                           plot_type='table',
                           file_path=self.output_file_path if self.output_file_path is None else os.path.join(self.output_file_path, 'ga_metadata_table.html'),
                           ).run()
        if model_evolution:
            DataVisualizer(df=_evolution_history_data,
                           title='Evolution of used ML Models:',
                           features=['fitness_score', 'generation'],
                           color_feature='model',
                           plot_type='scatter',
                           melt=True,
                           file_path=self.output_file_path if self.output_file_path is None else os.path.join(self.output_file_path, 'ga_model_evolution.html'),
                           ).run()
        if model_distribution:
            if self.models is None or len(self.models) > 1:
                DataVisualizer(df=_evolution_history_data,
                               title='Distribution of used ML Models:',
                               features=['model'],
                               group_by=['generation'] if per_generation else None,
                               plot_type='pie',
                               file_path=self.output_file_path if self.output_file_path is None else os.path.join(self.output_file_path, 'ga_model_distribution.html'),
                               ).run()
        #if param_distribution:
        #    _charts.update({'Distribution of ML Model parameters:': dict(data=_evolution_history_data,
        #                                                                 features=['model_param'],
        #                                                                 group_by=['generation'] if per_generation else None,
        #                                                                 plot_type='tree',
        #                                                                 file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(self.output_file_path, 'ga_parameter_treemap.html')
        #                                                                 )
        #                    })
        if train_time_distribution:
            DataVisualizer(df=_evolution_history_data,
                           title='Distribution of elapsed Training Time:',
                           features=['train_time_in_seconds'],
                           group_by=['model'],
                           melt=True,
                           plot_type='violin',
                           use_auto_extensions=False,
                           file_path=self.output_file_path if self.output_file_path is None else os.path.join(self.output_file_path, 'ga_training_time_distribution.html'),
                           ).run()
        if breeding_map:
            _breeding_map: pd.DataFrame = pd.DataFrame(data=dict(gen_0=self.generation_history['population']['gen_0'].get('fitness')), index=[0])
            for g in self.generation_history['population'].keys():
                if g != 'gen_0':
                    _breeding_map[g] = self.generation_history['population'][g].get('fitness')
            DataVisualizer(df=_breeding_map,
                           title='Breeding Heat Map:',
                           features=_breeding_map.columns.to_list(),
                           plot_type='heat',
                           file_path=self.output_file_path if self.output_file_path is None else os.path.join(self.output_file_path, 'ga_breeding_heatmap.html'),
                           ).run()
        if breeding_graph:
            DataVisualizer(df=_evolution_history_data,
                           title='Breeding Network Graph:',
                           features=['generation', 'fitness_score'],
                           graph_features=dict(node='id', edge='parent'),
                           color_feature='model',
                           plot_type='network',
                           file_path=self.output_file_path if self.output_file_path is None else os.path.join(self.output_file_path, 'ga_breeding_graph.html'),
                           ).run()
        if fitness_distribution:
            DataVisualizer(df=_evolution_history_data,
                           title='Distribution of Fitness Metric:',
                           features=['fitness_score'],
                           time_features=['generation'],
                           plot_type='ridgeline',
                           file_path=self.output_file_path if self.output_file_path is None else os.path.join(self.output_file_path, 'ga_fitness_score_distribution_per_generation.html'),
                           ).run()
        if fitness_dimensions:
            DataVisualizer(df=_evolution_history_data,
                           title='Evolution Meta Data:',
                           features=['train_time_in_seconds',
                                     'ml_metric',
                                     'train_test_diff',
                                     'fitness_score',
                                     'parent',
                                     'id',
                                     'generation',
                                     'model'
                                     ],
                           color_feature='model',
                           plot_type='parcoords',
                           file_path=self.output_file_path if self.output_file_path is None else os.path.join(self.output_file_path, 'ga_metadata_evolution_coords.html'),
                           ).run()
        if fitness_evolution:
            DataVisualizer(df=_evolution_gradient_data,
                           title='Fitness Evolution:',
                           features=['min', 'median', 'mean', 'max'],
                           time_features=['generation'],
                           melt=True,
                           plot_type='line',
                           file_path=self.output_file_path if self.output_file_path is None else os.path.join(self.output_file_path, 'ga_evolution_fitness_score.html'),
                           ).run()
        if epoch_stats:
            if self.deep_learning:
                _epoch_metric_score: pd.DataFrame = pd.DataFrame(data=self.evolution.get('epoch_metric_score'))
                _epoch_metric_score['epoch'] = [epoch + 1 for epoch in range(0, _epoch_metric_score.shape[0], 1)]
                DataVisualizer(df=_epoch_metric_score,
                               title='Epoch Evaluation of fittest neural network:',
                               features=['train', 'val'],
                               time_features=['epoch'],
                               melt=True,
                               plot_type='line',
                               file_path=self.output_file_path if self.output_file_path is None else os.path.join(self.output_file_path, 'ga_epoch_metric_score.html'),
                               ).run()
        if prediction_of_best_model:
            if self.target_type == 'reg':
                DataVisualizer(df=_best_model_results,
                               title='Prediction Evaluation of final inherited ML Model:',
                               features=['obs', 'abs_diff', 'rel_diff', 'pred'],
                               color_feature='pred',
                               plot_type='parcoords',
                               file_path=self.output_file_path if self.output_file_path is None else os.path.join(self.output_file_path, 'ga_prediction_evaluation_coords.html'),
                               ).run()
                DataVisualizer(df=_best_model_results,
                               title='Prediction vs. Observation of final inherited ML Model:',
                               features=['obs', 'pred'],
                               plot_type='joint',
                               file_path=self.output_file_path if self.output_file_path is None else os.path.join(self.output_file_path, 'ga_prediction_scatter_contour.html'),
                               ).run()
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
                DataVisualizer(df=_confusion_matrix,
                               title='Confusion Matrix:',
                               features=_confusion_matrix.columns.to_list(),
                               plot_type='table',
                               file_path=self.output_file_path if self.output_file_path is None else os.path.join(self.output_file_path, 'ga_prediction_confusion_table.html'),
                               ).run()
                DataVisualizer(df=_best_model_results,
                               title='Confusion Matrix Heatmap:',
                               features=_best_model_results.columns.to_list(),
                               plot_type='heat',
                               file_path=self.output_file_path if self.output_file_path is None else os.path.join(self.output_file_path, 'ga_prediction_confusion_heatmap.html'),
                               ).run()
                _confusion_matrix_normalized: pd.DataFrame = pd.DataFrame(data=EvalClf(obs=self.data_set.get('y_test'),
                                                                                       pred=self.data_set.get('pred')
                                                                                       ).confusion(normalize='pred'),
                                                                          index=self.target_labels,
                                                                          columns=self.target_labels
                                                                          )
                DataVisualizer(df=_confusion_matrix_normalized,
                               title='Confusion Matrix Normalized Heatmap:',
                               features=_confusion_matrix_normalized.columns.to_list(),
                               plot_type='heat',
                               file_path=self.output_file_path if self.output_file_path is None else os.path.join(self.output_file_path, 'ga_prediction_confusion_normal_heatmap.html'),
                               ).run()
                DataVisualizer(df=_best_model_results,
                               title='Classification Report:',
                               features=_best_model_results.columns.to_list(),
                               plot_type='table',
                               file_path=self.output_file_path if self.output_file_path is None else os.path.join(self.output_file_path, 'ga_prediction_clf_report_table.html'),
                               ).run()
                if self.target_type == 'clf_multi':
                    DataVisualizer(df=_best_model_results,
                                   title='Prediction Evaluation of final inherited ML Model:',
                                   features=['obs', 'abs_diff', 'pred'],
                                   color_feature='pred',
                                   plot_type='parcoords',
                                   brushing=True,
                                   file_path=self.output_file_path if self.output_file_path is None else os.path.join(self.output_file_path, 'ga_prediction_evaluation_category.html'),
                                   ).run()
                else:
                    _roc_curve = pd.DataFrame()
                    _roc_curve_values: dict = EvalClf(obs=_best_model_results['obs'],
                                                      pred=_best_model_results['pred']
                                                      ).roc_curve()
                    _roc_curve['roc_curve'] = _roc_curve_values['true_positive_rate'][1]
                    _roc_curve['baseline'] = _roc_curve_values['false_positive_rate'][1]
                    DataVisualizer(df=_roc_curve,
                                   title='ROC-AUC Curve:',
                                   features=['roc_curve', 'baseline'],
                                   time_features=['baseline'],
                                   plot_type='line',
                                   melt=True,
                                   use_auto_extensions=False,
                                   # xaxis_label=['False Positive Rate'],
                                   # yaxis_label=['True Positive Rate'],
                                   file_path=self.output_file_path if self.output_file_path is None else os.path.join(self.output_file_path, 'ga_prediction_roc_auc_curve.html'),
                                   ).run()
