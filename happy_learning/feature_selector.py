"""

Feature selection of structured (tabular) features

"""

import copy
import numpy as np
import mlflow
import os
import pandas as pd
import random
import warnings

from .evaluate_machine_learning import sml_fitness_score
from .genetic_algorithm import GeneticAlgorithm
from .sampler import MLSampler
from .supervised_machine_learning import ModelGeneratorClf, ModelGeneratorReg
from .swarm_intelligence import SwarmIntelligence
from .utils import HappyLearningUtils
from easyexplore.data_visualizer import DataVisualizer
from easyexplore.utils import Log
from multiprocessing.pool import ThreadPool
from typing import Dict, List

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class FeatureSelectorException(Exception):
    """
    CLass for handling exceptions for class FeatureSelector
    """
    pass


class FeatureSelector:
    """
    Class for calculating shapley values (shapley additive explanations) for feature importance evaluation and feature selection
    """
    def __init__(self,
                 df: pd.DataFrame,
                 features: List[str],
                 target: str,
                 force_target_type: str = None,
                 model_name: str = 'cat',
                 init_pairs: int = 3,
                 init_games: int = 5,
                 increasing_pair_size_factor: float = 0.5,
                 games: int = 3,
                 penalty_factor: float = 0.1,
                 max_iter: int = 50,
                 max_players: int = -1,
                 evolutionary_algorithm: str = 'ga',
                 use_standard_params: bool = True,
                 aggregate_feature_imp: Dict[str, dict] = None,
                 visualize_all_scores: bool = True,
                 visualize_variant_scores: bool = True,
                 visualize_core_feature_scores: bool = True,
                 path: str = None,
                 mlflow_log: bool = True,
                 multi_threading: bool = False,
                 **kwargs
                 ):
        """
        :param df: Pandas or dask DataFrame
            Data set

        :param features: List[str]
            Names of the predictor features

        :param target: str
            Name of the target feature

        :param force_target_type: str
            Force specific target type
                -> clf_multi: Multi-Classification
                -> reg: Regression

        :param model_name: str
            Name of the model

        :param init_pairs: int
            Number of players in each starting game of the tournament

        :param init_games: int
            Number of penalty games to qualify players for the tournament

        :param increasing_pair_size_factor: float
            Factor for increasing amount of player in each game in each step

        :param games: int
            Number of games to play in each step of the tournament

        :param penalty_factor: float
            Amount of players to exclude from the tournament because of their poor contribution capabilities

        :param max_iter: int
            Maximum number of steps of the tournament

        :param max_players: int
            Maximum number of features used for training machine learning model

        :param evolutionary_algorithm: str
            Name of the reinforced evolutionary algorithm
                -> ga: Genetic Algorithm
                -> si: Swarm Intelligence

        :param aggregate_feature_imp: Dict[str, dict]
            Name of the aggregation method and the feature names to aggregate
                -> core: Aggregate feature importance score by each core (original) feature
                -> level: Aggregate feature importance score by the processing level of each feature

        :param visualize_all_scores: bool
            Whether to visualize all feature importance scores or not

        :param visualize_variant_scores: bool
            Whether to visualize all variants of feature processing importance scores separately or not

        :param visualize_core_feature_scores: bool
            Whether to visualize summarized core feature importance scores or not

        :param path: str
            Path or directory to export visualization to

        :param mlflow_log: bool
            Track experiment results using mlflow

        :param multi_threading: bool
            Whether to run each game multi- or single-threaded during each iteration

        :param kwargs: dict
            Key-word arguments
        """
        self.df: pd.DataFrame = df
        self.target: str = target
        self.features: List[str] = features
        if self.target in self.features:
            del self.features[self.features.index(self.target)]
        self.df = self.df[self.features + [self.target]]
        self.n_cases: int = self.df.shape[0]
        self.n_features: int = len(self.features)
        self.force_target_type: str = force_target_type
        if self.force_target_type is None:
            self.ml_type: str = HappyLearningUtils().get_ml_type(values=self.df[self.target].values) if kwargs.get('ml_type') is None else kwargs.get('ml_type')
        else:
            self.ml_type: str = self.force_target_type
        _stratify: bool = False
        if self.ml_type == 'reg':
            self.ml_metric: str = 'rmse_norm'
        elif self.ml_type == 'clf_binary':
            self.ml_metric: str = 'roc_auc'
            _stratify = True if kwargs.get('stratification') is None else kwargs.get('stratification')
        elif self.ml_type == 'clf_multi':
            self.ml_metric: str = 'cohen_kappa'
            _stratify = False if kwargs.get('stratification') is None else kwargs.get('stratification')
        self.train_test: dict = MLSampler(df=self.df,
                                          target=self.target,
                                          features=self.features,
                                          train_size=0.8 if kwargs.get('train_size') is None else kwargs.get('train_size'),
                                          random_sample=True if kwargs.get('random') is None else kwargs.get('random'),
                                          stratification=_stratify
                                          ).train_test_sampling(validation_split=0 if kwargs.get('validation_size') is None else kwargs.get('validation_size'))
        self.init_pairs: int = init_pairs
        self.init_games: int = init_games
        self.pair_size_factor: float = increasing_pair_size_factor
        self.game: int = 0
        self.games: int = games
        self.penalty_factor: float = penalty_factor
        self.max_iter: int = max_iter
        self.max_players: int = max_players if max_players > 1 else len(self.features)
        self.pairs: List[np.array] = []
        self.threads: dict = {}
        self.multi_threading: bool = multi_threading
        self.tournament: bool = False
        self.shapley_additive_explanation: dict = dict(sum={}, game={}, tournament={})
        self.model_name: str = model_name
        self.evolutionary_algorithm: str = evolutionary_algorithm
        self.imp_score: Dict[str, float] = {}
        self.use_standard_params: bool = use_standard_params
        self.aggregate_feature_imp: Dict[str, dict] = aggregate_feature_imp
        self.visualize_all_scores: bool = visualize_all_scores
        self.visualize_variant_scores: bool = visualize_variant_scores
        self.visualize_core_features_scores: bool = visualize_core_feature_scores
        self.path: str = path
        if self.path is not None:
            self.path = self.path.replace('\\', '/')
            if os.path.isfile(self.path):
                self.path = self.path.replace(self.path.split('/')[-1], '')
            else:
                if self.path.split('/')[-1] != '':
                    self.path = '{}/'.format(self.path)
        self.mlflow_log: bool = mlflow_log
        if self.mlflow_log:
            self.mlflow_client: mlflow.tracking.MlflowClient = mlflow.tracking.MlflowClient(tracking_uri=kwargs.get('tracking_uri'),
                                                                                            registry_uri=kwargs.get('registry_uri')
                                                                                            )
        else:
            self.mlflow_client = None
        self.kwargs: dict = kwargs
        if self.use_standard_params:
            if self.ml_type == 'reg':
                _model_param: dict = ModelGeneratorReg(models=[self.model_name]).get_model_parameter()
            else:
                _model_param: dict = ModelGeneratorClf(models=[self.model_name]).get_model_parameter()
            self.feature_tournament_ai: dict = dict(model_name=self.model_name,
                                                    param=_model_param
                                                    )
        else:
            if self.kwargs.get('model_param') is None:
                self.feature_tournament_ai: dict = {}
                self._evolve_feature_tournament_ai()
                if self.target in self.features:
                    del self.features[self.features.index(self.target)]
            else:
                _model_name: str = list(self.kwargs.get('model_param').keys())[0]
                self.feature_tournament_ai: dict = dict(model_name=_model_name,
                                                        param=self.kwargs.get('model_param')[_model_name]
                                                        )

    def _evolve_feature_tournament_ai(self):
        """
        Evolve machine learning model using evolutionary algorithm
        """
        Log(write=False, level='info').log(msg='Evolve feature tournament ai ...')
        if self.evolutionary_algorithm == 'ga':
            _feature_tournament_ai_learning: GeneticAlgorithm = GeneticAlgorithm(mode='model',
                                                                                 df=self.df,
                                                                                 target=self.target,
                                                                                 features=self.features,
                                                                                 re_split_data=False if self.kwargs.get('re_split_data') is None else self.kwargs.get('re_split_data'),
                                                                                 re_sample_cases=False if self.kwargs.get('re_sample_cases') is None else self.kwargs.get('re_sample_cases'),
                                                                                 re_sample_features=True,
                                                                                 max_features=self.n_features,
                                                                                 labels=self.kwargs.get('labels'),
                                                                                 models=[self.model_name],
                                                                                 model_params=None,
                                                                                 burn_in_generations=-1 if self.kwargs.get('burn_in_generations') is None else self.kwargs.get('burn_in_generations'),
                                                                                 warm_start=True if self.kwargs.get('warm_start') is None else self.kwargs.get('warm_start'),
                                                                                 max_generations=2 if self.kwargs.get('max_generations_ai') is None else self.kwargs.get('max_generations_ai'),
                                                                                 pop_size=64 if self.kwargs.get('pop_size') is None else self.kwargs.get('pop_size'),
                                                                                 mutation_rate=0.1 if self.kwargs.get('mutation_rate') is None else self.kwargs.get('mutation_rate'),
                                                                                 mutation_prob=0.5 if self.kwargs.get('mutation_prob') is None else self.kwargs.get('mutation_prob'),
                                                                                 parents_ratio=0.5 if self.kwargs.get('parents_ratio') is None else self.kwargs.get('parents_ratio'),
                                                                                 early_stopping=0 if self.kwargs.get('early_stopping') is None else self.kwargs.get('early_stopping'),
                                                                                 convergence=False if self.kwargs.get('convergence') is None else self.kwargs.get('convergence'),
                                                                                 timer_in_seconds=10000 if self.kwargs.get('timer_in_secondes') is None else self.kwargs.get('timer_in_secondes'),
                                                                                 force_target_type=self.force_target_type,
                                                                                 plot=False if self.kwargs.get('plot') is None else self.kwargs.get('plot'),
                                                                                 output_file_path=self.kwargs.get('output_file_path'),
                                                                                 multi_threading=False if self.kwargs.get('multi_threading') is None else self.kwargs.get('multi_threading'),
                                                                                 multi_processing=False if self.kwargs.get('multi_processing') is None else self.kwargs.get('multi_processing'),
                                                                                 log=False if self.kwargs.get('log') is None else self.kwargs.get('log'),
                                                                                 verbose=False if self.kwargs.get('verbose') is None else self.kwargs.get('verbose'),
                                                                                 mlflow_log=False if self.kwargs.get('mlflow_log_evolutionary') is None else self.kwargs.get('mlflow_log_evolutionary')
                                                                                 )
        elif self.evolutionary_algorithm == 'si':
            _feature_tournament_ai_learning: SwarmIntelligence = SwarmIntelligence(mode='model',
                                                                                   df=self.df,
                                                                                   target=self.target,
                                                                                   features=self.features,
                                                                                   re_split_data=False if self.kwargs.get('re_split_data') is None else self.kwargs.get('re_split_data'),
                                                                                   re_sample_cases=False if self.kwargs.get('re_sample_cases') is None else self.kwargs.get('re_sample_cases'),
                                                                                   re_sample_features=True,
                                                                                   max_features=self.n_features,
                                                                                   labels=self.kwargs.get('labels'),
                                                                                   models=[self.model_name],
                                                                                   model_params=None,
                                                                                   burn_in_adjustments=-1 if self.kwargs.get('burn_in_adjustments') is None else self.kwargs.get('burn_in_adjustments'),
                                                                                   warm_start=True if self.kwargs.get('warm_start') is None else self.kwargs.get('warm_start'),
                                                                                   max_adjustments=2 if self.kwargs.get('max_adjustments_ai') is None else self.kwargs.get('max_adjustments_ai'),
                                                                                   pop_size=64 if self.kwargs.get('pop_size') is None else self.kwargs.get('pop_size'),
                                                                                   adjustment_rate=0.1 if self.kwargs.get('adjustment_rate') is None else self.kwargs.get('adjustment_rate'),
                                                                                   adjustment_prob=0.5 if self.kwargs.get('adjustment_prob') is None else self.kwargs.get('adjustment_prob'),
                                                                                   early_stopping=0 if self.kwargs.get('early_stopping') is None else self.kwargs.get('early_stopping'),
                                                                                   convergence=False if self.kwargs.get('convergence') is None else self.kwargs.get('convergence'),
                                                                                   timer_in_seconds=10000 if self.kwargs.get('timer_in_secondes') is None else self.kwargs.get('timer_in_secondes'),
                                                                                   force_target_type=self.force_target_type,
                                                                                   plot=False if self.kwargs.get('plot') is None else self.kwargs.get('plot'),
                                                                                   output_file_path=self.kwargs.get('output_file_path'),
                                                                                   multi_threading=False if self.kwargs.get('multi_threading') is None else self.kwargs.get('multi_threading'),
                                                                                   multi_processing=False if self.kwargs.get('multi_processing') is None else self.kwargs.get('multi_processing'),
                                                                                   log=False if self.kwargs.get('log') is None else self.kwargs.get('log'),
                                                                                   verbose=False if self.kwargs.get('verbose') is None else self.kwargs.get('verbose'),
                                                                                   mlflow_log=False if self.kwargs.get('mlflow_log_evolutionary') is None else self.kwargs.get('mlflow_log_evolutionary')
                                                                                   )
        else:
            raise FeatureSelectorException('Reinforced evolutionary algorithm ({}) not supported'.format(self.evolutionary_algorithm))
        _feature_tournament_ai_learning.optimize()
        self.feature_tournament_ai = _feature_tournament_ai_learning.evolution
        Log(write=False, level='info').log(msg='Feature tournament ai evolved -> {}'.format(self.feature_tournament_ai.get('model_name')))

    def _game(self, iteration: int):
        """
        Play tournament game

        :param iteration: int
            Number of current iteration
        """
        for pair in self.pairs:
            if self.ml_type == 'reg':
                _game: ModelGeneratorReg = ModelGeneratorReg(model_name=self.feature_tournament_ai.get('model_name'),
                                                             reg_params=self.feature_tournament_ai.get('param')
                                                             )
                _game.generate_model()
                _game.train(x=self.train_test.get('x_train')[pair].values,
                            y=self.train_test.get('y_train').values,
                            #validation=dict(x_val=self.train_test.get('x_val')[pair].values,
                            #                y_val=self.train_test.get('y_val').values
                            #                )
                            )
                _pred = _game.predict(x=self.train_test.get('x_test')[pair].values)
                _game.eval(obs=self.train_test.get('y_test').values, pred=_pred, train_error=False)
                _game_score: float = sml_fitness_score(ml_metric=tuple([0, _game.fitness['test'].get(self.ml_metric)]),
                                                       train_test_metric=tuple([_game.fitness['train'].get(self.ml_metric), _game.fitness['test'].get(self.ml_metric)]),
                                                       train_time_in_seconds=_game.train_time,
                                                       capping_to_zero=True
                                                       )
            else:
                _game: ModelGeneratorClf = ModelGeneratorClf(model_name=self.feature_tournament_ai.get('model_name'),
                                                             clf_params=self.feature_tournament_ai.get('param')
                                                             )
                _game.generate_model()
                _game.train(x=self.train_test.get('x_train')[pair].values,
                            y=self.train_test.get('y_train').values,
                            #validation=dict(x_val=self.train_test.get('x_val')[pair].values,
                            #                y_val=self.train_test.get('y_val').values
                            #                )
                            )
                _pred = _game.predict(x=self.train_test.get('x_test')[pair].values, probability=False)
                _game.eval(obs=self.train_test.get('y_test').values, pred=_pred, train_error=False)
                _game_score: float = sml_fitness_score(ml_metric=tuple([1, _game.fitness['test'].get(self.ml_metric)]),
                                                       train_test_metric=tuple([_game.fitness['train'].get(self.ml_metric), _game.fitness['test'].get(self.ml_metric)]),
                                                       train_time_in_seconds=_game.train_time,
                                                       capping_to_zero=True
                                                       )
            for j, imp in enumerate(_game.model.feature_importances_):
                _shapley_value: float = imp * _game_score
                if _shapley_value != _shapley_value:
                    _shapley_value = 0.0
                if pair[j] in self.shapley_additive_explanation['sum'].keys():
                    self.shapley_additive_explanation['sum'][pair[j]] += _shapley_value
                else:
                    self.shapley_additive_explanation['sum'].update({pair[j]: _shapley_value})
                if iteration >= self.init_games:
                    if pair[j] in self.shapley_additive_explanation['game'].keys():
                        self.shapley_additive_explanation['game'][pair[j]].append(_shapley_value)
                    else:
                        self.shapley_additive_explanation['game'].update({pair[j]: [_shapley_value]})
            if self.mlflow_log:
                with mlflow.start_run():
                    for j, imp in enumerate(_game.model.feature_importances_):
                        _shapley_value: float = imp * _game_score
                        mlflow.log_metric(key='sml_score', value=_game_score, step=None)
                        for metric_context in _game.fitness:
                            for metric in _game.fitness[metric_context]:
                                mlflow.log_metric(key=f'{metric}_{metric_context}',
                                                  value=_game.fitness[metric_context][metric]
                                                  )
                        _tags: dict = dict(model_name=self.feature_tournament_ai.get('model_name'),
                                           target_type=self.ml_type,
                                           game=self.game,
                                           iteration=iteration,
                                           init_game=True if iteration < self.init_games else False,
                                           tree_importance=imp,
                                           shapley_value=_shapley_value
                                           )
                        mlflow.set_tags(tags=_tags)
                        for k, feature in enumerate(pair):
                            mlflow.set_tag(key=f'feature_{k}', value=feature)

    @staticmethod
    def _mlflow_tracking(stats: Dict[str, pd.DataFrame], file_paths: List[str]):
        """
        Track model performance using mlflow
        """
        for i, stat in enumerate(stats.keys()):
            _df: pd.DataFrame = stats.get(stat)
            mlflow.set_experiment(experiment_name=stat, experiment_id=None)
            for case in range(0, _df.shape[0], 1):
                with mlflow.start_run():
                    for feature in range(0, _df.shape[1], 1):
                        mlflow.set_tag(key=_df.columns.tolist()[feature], value=_df.iloc[case, feature])
            with mlflow.start_run():
                try:
                    _file_name: str = file_paths[i].split('/')[-1].replace('.html', '')
                    mlflow.log_artifact(local_path=file_paths[i], artifact_path=_file_name)
                except (FileNotFoundError, IndexError):
                    pass

    def _permutation(self, n: int):
        """
        Permute combination of players

        :param n: int
            Number of players in each game
        """
        _shuffle: np.array = np.array(tuple(random.sample(population=self.features, k=self.n_features)))
        try:
            _pairs: np.array = np.array_split(ary=_shuffle, indices_or_sections=int(self.n_features / n))
        except ValueError:
            _pairs: np.array = self.pairs
        if self.tournament:
            for pair in _pairs:
                for feature in pair:
                    if feature in self.shapley_additive_explanation['tournament'].keys():
                        self.shapley_additive_explanation['tournament'][feature].append(len(pair))
                    else:
                        self.shapley_additive_explanation['tournament'].update({feature: [len(pair)]})
        self.pairs = _pairs

    def _play_tournament(self):
        """
        Play unreal tournament to extract the fittest or most important players based on the concept of shapley values
        """
        Log(write=False, level='info').log(msg='Start penalty with {} players...'.format(self.n_features))
        if self.mlflow_log:
            mlflow.set_experiment(experiment_name='Shapley Additive Explanation (Feature Tournament)',
                                  experiment_id=None
                                  )
        _game_scores: List[float] = []
        _permutation_space: int = self.init_pairs
        _pair_size_factor: float = self.max_iter * self.pair_size_factor
        for i in range(0, self.max_iter + self.init_games, 1):
            if i == self.init_games:
                Log(write=False, level='info').log(msg='Start feature tournament with {} players ...'.format(self.n_features))
                self.tournament = True
            elif i > self.init_games:
                _pair_size: int = _permutation_space + int(_pair_size_factor)
                if self.n_features >= _pair_size:
                    _permutation_space = _pair_size
                    #_permutation_space = int(_permutation_space + (_permutation_space * self.pair_size_factor))
            else:
                if i == 0:
                    _permutation_space = self.init_pairs
            if _permutation_space > self.max_players:
                _permutation_space = self.max_players
            self._permutation(n=_permutation_space)
            _pool: ThreadPool = ThreadPool(processes=len(self.pairs)) if self.multi_threading else None
            for g in range(0, self.games, 1):
                Log(write=False, level='info').log(msg='Iteration {} - Game {} ~ {} players each game'.format(i + 1,
                                                                                                              g + 1,
                                                                                                              _permutation_space
                                                                                                              )
                                                   )
                self.game = g
                if self.multi_threading:
                    self.threads.update({g: _pool.apply_async(func=self._game, args=[i])})
                else:
                    self._game(iteration=i)
                if i < self.init_games:
                    break
                self._permutation(n=_permutation_space)
            for thread in self.threads.keys():
                self.threads.get(thread).get()
            if i + 1 == self.init_games:
                _shapley_matrix: pd.DataFrame = pd.DataFrame(data=self.shapley_additive_explanation['sum'], index=['score']).transpose()
                _sorted_shapley_matrix = _shapley_matrix.sort_values(by='score', axis=0, ascending=False, inplace=False)
                _all_features: int = _sorted_shapley_matrix.shape[0]
                _sorted_shapley_matrix = _sorted_shapley_matrix.loc[_sorted_shapley_matrix['score'] > 0, :]
                if _sorted_shapley_matrix.shape[0] == 0:
                    raise FeatureSelectorException('No feature scored higher than 0 during penalty phase')
                _n_features: int = _sorted_shapley_matrix.shape[0]
                Log(write=False, level='info').log(msg='Excluded {} features with score 0'.format(_all_features - _n_features))
                _exclude_features: int = int(_n_features * self.penalty_factor)
                self.features = _sorted_shapley_matrix.index.values.tolist()[0:(_n_features - _exclude_features)]
                self.n_features = len(self.features)
                Log(write=False, level='info').log(msg='Excluded {} lowest scored features from tournament'.format(_exclude_features))
            if i + 1 == self.max_iter + self.init_games:
                _shapley_values: dict = {}
                for sv in self.shapley_additive_explanation['game'].keys():
                    _shapley_values.update({sv: self.shapley_additive_explanation['sum'][sv] / len(self.shapley_additive_explanation['game'][sv])})
                self.shapley_additive_explanation.update({'total': _shapley_values})
            if self.n_features <= (self.pair_size_factor * _permutation_space):
                if i + 1 == self.max_iter:
                    break

    def select(self,
               imp_threshold: float = 0.01,
               redundant_threshold: float = 0.02,
               visualize_game_stats: bool = True,
               plot_type: str = 'bar'
               ) -> dict:
        """
        Select most important features based on shapley values

        :param imp_threshold: float
            Threshold of importance score

        :param redundant_threshold: float
            Threshold for defining redundant features in percent

        :param visualize_game_stats: bool
            Whether to visualize game statistics or not

        :param plot_type: str
            Name of the plot type
                -> pie: Pie Chart
                -> bar: Bar Chart

        :return dict
            Redundant features, important features and reduction scores
        """
        self._play_tournament()
        _imp_plot: dict = {}
        _core_features: List[str] = []
        _processed_features: List[str] = []
        _imp_threshold: float = imp_threshold if (imp_threshold >= 0) and (imp_threshold <= 1) else 0.7
        _df: pd.DataFrame = pd.DataFrame(data=self.shapley_additive_explanation.get('total'), index=['score']).transpose()
        _df = _df.sort_values(by='score', axis=0, ascending=False, inplace=False)
        _df['feature'] = _df.index.values
        _imp_features: List[str] = _df['feature'].values.tolist()
        for s, feature in enumerate(_imp_features):
            self.imp_score.update({feature: _df['score'].values.tolist()[s]})
        _rank: List[int] = []
        _sorted_scores: List[float] = _df['score'].values.tolist()
        for r, val in enumerate(_sorted_scores):
            if r == 0:
                _rank.append(r + 1)
            else:
                if val == _sorted_scores[r - 1]:
                    _rank.append(_rank[-1])
                else:
                    _rank.append(r + 1)
        _df['rank'] = _rank
        _game_df: pd.DataFrame = pd.DataFrame(data=self.shapley_additive_explanation.get('game'))
        # _game_df['game'] = _game_df.index.values
        _tournament_df: pd.DataFrame = pd.DataFrame(data=self.shapley_additive_explanation.get('tournament'))
        # _tournament_df['game'] = _tournament_df.index.values
        _file_paths: List[str] = []
        if self.visualize_all_scores:
            if visualize_game_stats:
                _file_paths.append(os.path.join(str(self.path), 'feature_tournament_game_stats.html'))
                _file_paths.append(os.path.join(str(self.path), 'feature_tournament_game_size.html'))
                _game_plot: dict = {'Feature Tournament Game Stats (Shapley Scores)': dict(data=_game_df,
                                                                                           features=list(
                                                                                               _game_df.columns),
                                                                                           plot_type='violin',
                                                                                           melt=True,
                                                                                           render=True,
                                                                                           file_path=_file_paths[
                                                                                               0] if self.path is not None else None
                                                                                           ),
                                    'Feature Tournament Stats (Game Size)': dict(data=_tournament_df,
                                                                                 features=list(_tournament_df.columns),
                                                                                 plot_type='heat',
                                                                                 render=True,
                                                                                 file_path=_file_paths[
                                                                                     1] if self.path is not None else None
                                                                                 )
                                    }
                DataVisualizer(subplots=_game_plot,
                               height=500,
                               width=500
                               ).run()
            _file_paths.append(os.path.join(str(self.path), 'feature_importance_shapley.html'))
            _imp_plot: dict = {'Feature Importance (Shapley Scores)': dict(df=_df,
                                                                           plot_type=plot_type,
                                                                           render=True if self.path is None else False,
                                                                           file_path=_file_paths[
                                                                               -1] if self.path is not None else None,
                                                                           kwargs=dict(layout={},
                                                                                       y=_df['score'].values,
                                                                                       x=_df.index.values.tolist(),
                                                                                       marker=dict(color=_df['score'],
                                                                                                   colorscale='rdylgn',
                                                                                                   autocolorscale=True
                                                                                                   )
                                                                                       )
                                                                           )
                               }
        if self.mlflow_log:
            self._mlflow_tracking(stats={'Feature Score (Feature Tournament)': _df,
                                         'Game Size (Feature Tournament)': _tournament_df,
                                         'Game Score (Feature Tournament)': _game_df
                                         },
                                  file_paths=_file_paths)
        if self.aggregate_feature_imp is not None:
            _aggre_score: dict = {}
            for core_feature in self.aggregate_feature_imp.keys():
                _feature_scores: dict = {}
                _aggre_score.update({core_feature: 0.0 if self.imp_score.get(core_feature) is None else self.imp_score.get(core_feature)})
                if self.imp_score.get(core_feature) is not None:
                    _feature_scores.update({core_feature: self.imp_score.get(core_feature)})
                for proc_feature in self.aggregate_feature_imp[core_feature]:
                    _feature_scores.update({proc_feature: 0.0 if self.imp_score.get(proc_feature) is None else self.imp_score.get(proc_feature)})
                    if self.imp_score.get(proc_feature) is not None:
                        _aggre_score[core_feature] += self.imp_score.get(proc_feature)
                if len(self.aggregate_feature_imp[core_feature]) < 2:
                    continue
                _aggre_score[core_feature] = _aggre_score[core_feature] / len(self.aggregate_feature_imp[core_feature])
                _processed_feature_matrix: pd.DataFrame = pd.DataFrame(data=_feature_scores, index=['score']).transpose()
                _processed_feature_matrix.sort_values(by='score', axis=0, ascending=False, inplace=True)
                _processed_features.append(_processed_feature_matrix.index.values.tolist()[0])
                if self.visualize_variant_scores:
                    _imp_plot.update(
                        {'Feature Importance (Preprocessing Variants {})'.format(core_feature): dict(data=_processed_feature_matrix,
                                                                                                     plot_type=plot_type,
                                                                                                     melt=True,
                                                                                                     render=True if self.path is None else False,
                                                                                                     file_path='{}feature_importance_processing_variants.html'.format(self.path) if self.path is not None else None,
                                                                                                     kwargs=dict(layout={},
                                                                                                                 y=_processed_feature_matrix['score'].values,
                                                                                                                 x=_processed_feature_matrix.index.values,
                                                                                                                 marker=dict(color=_processed_feature_matrix['score'],
                                                                                                                             colorscale='rdylgn',
                                                                                                                             autocolorscale=True
                                                                                                                             )
                                                                                                                 )
                                                                                                     )
                         })
            _core_imp_matrix: pd.DataFrame = pd.DataFrame(data=_aggre_score, index=['abs_score']).transpose()
            _core_imp_matrix['rel_score'] = _core_imp_matrix['abs_score'] / sum(_core_imp_matrix['abs_score'])
            _core_imp_matrix.sort_values(by='abs_score', axis=0, ascending=False, inplace=True)
            _raw_core_features: List[str] = _core_imp_matrix.loc[_core_imp_matrix['rel_score'] >= _imp_threshold, :].index.values.tolist()
            for core in _raw_core_features:
                _core_features.extend(self.aggregate_feature_imp[core])
                _core_features = list(set(_core_features))
            if self.visualize_core_features_scores:
                _imp_plot.update({'Feature Importance (Core Feature Aggregation)': dict(data=_core_imp_matrix,
                                                                                        plot_type=plot_type,
                                                                                        melt=False,
                                                                                        render=True if self.path is None else False,
                                                                                        file_path='{}feature_importance_core_aggregation.html'.format(self.path) if self.path is not None else None,
                                                                                        kwargs=dict(layout={},
                                                                                                    y=_core_imp_matrix['abs_score'].values,
                                                                                                    x=_core_imp_matrix['abs_score'].index.values,
                                                                                                    marker=dict(
                                                                                                        color=_core_imp_matrix['abs_score'],
                                                                                                        colorscale='rdylgn',
                                                                                                        autocolorscale=True
                                                                                                        )
                                                                                                    )
                                                                                        )
                                  })
        if self.visualize_all_scores or self.visualize_variant_scores or self.visualize_core_features_scores:
            DataVisualizer(subplots=_imp_plot,
                           height=500,
                           width=500
                           ).run()
        if self.ml_type == 'reg':
            _model_generator: ModelGeneratorReg = ModelGeneratorReg(model_name=self.feature_tournament_ai.get('model_name'),
                                                                    reg_params=self.feature_tournament_ai.get('param')
                                                                    )
        else:
            _model_generator: ModelGeneratorClf = ModelGeneratorClf(model_name=self.feature_tournament_ai.get('model_name'),
                                                                    clf_params=self.feature_tournament_ai.get('param')
                                                                    )
        _model_generator.generate_model()
        _train_test_split: dict = MLSampler(df=self.df,
                                            target=self.target,
                                            features=_imp_features
                                            ).train_test_sampling(validation_split=0.1)
        _model_generator.train(x=_train_test_split.get('x_train').values, y=_train_test_split.get('y_train').values)
        _pred = _model_generator.predict(x=_train_test_split.get('x_test').values)
        _model_generator.eval(obs=_train_test_split.get('y_test').values, pred=_pred)
        _model_test_score: float = _model_generator.fitness['test'].get(self.ml_metric)
        if self.ml_type == 'reg':
            _threshold: float = _model_test_score * (1 + redundant_threshold)
        else:
            _threshold: float = _model_test_score * (1 - redundant_threshold)
        print('Metric', _model_test_score)
        print('Threshold', _threshold)
        _features: List[str] = copy.deepcopy(_imp_features)
        print(_features)
        _result: dict = dict(redundant=[], important=[], reduction={})
        for i in range(len(_imp_features) - 1, 0, -1):
            print(_imp_features[i])
            if len(_features) == 1:
                _result['important'] = _features
                break
            del _features[i]
            _model_generator.train(x=_train_test_split.get('x_train')[_features].values, y=_train_test_split.get('y_train').values)
            _pred = _model_generator.predict(x=_train_test_split.get('x_test')[_features].values)
            _model_generator.eval(obs=_train_test_split.get('y_test').values, pred=_pred)
            _new_model_test_score: float = _model_generator.fitness['test'].get(self.ml_metric)
            print('New model score', _new_model_test_score)
            if self.ml_type == 'reg':
                if _threshold <= _new_model_test_score:
                    _features.append(_imp_features[i])
                    _result['important'] = _features
                    break
                else:
                    _result['redundant'].append(_imp_features[i])
                    _result['reduction'].update({_imp_features[i]: _model_test_score - _new_model_test_score})
            else:
                if _threshold >= _new_model_test_score:
                    _features.append(_imp_features[i])
                    _result['important'] = _features
                    break
                else:
                    _result['redundant'].append(_imp_features[i])
                    _result['reduction'].update({_imp_features[i]: _model_test_score - _new_model_test_score})
        Log(write=False,
            level='info'
            ).log(msg=f'Number of redundant features: {len(_result["redundant"])}\nNumber of important features: {len(_result["important"])}')
        return dict(imp_features=_imp_features,
                    imp_score=self.imp_score,
                    imp_threshold=imp_threshold,
                    imp_core_features=_core_features,
                    imp_processed_features=_processed_features,
                    redundant=_result['redundant'],
                    important=_result['important'],
                    reduction=_result['reduction']
                    )
