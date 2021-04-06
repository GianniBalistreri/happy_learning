import numpy as np
import dask.dataframe as dd
import pandas as pd
import random

from .evaluate_machine_learning import sml_fitness_score
from .genetic_algorithm import GeneticAlgorithm
from .sampler import MLSampler
from .supervised_machine_learning import ModelGeneratorClf, ModelGeneratorReg
from .swarm_intelligence import SwarmIntelligence
from .utils import HappyLearningUtils
from easyexplore.utils import Log
from multiprocessing.pool import ThreadPool
from typing import List, Union


class FeatureTournamentException(Exception):
    """
    CLass for handling exceptions for class FeatureTournament
    """
    pass


class FeatureTournament:
    """
    Class for calculating shapley values (shapley additive explanations) for feature importance evaluation
    """
    def __init__(self,
                 df: Union[dd.DataFrame, pd.DataFrame],
                 features: List[str],
                 target: str,
                 force_target_type: str = None,
                 models: List[str] = None,
                 init_pairs: int = 3,
                 init_games: int = 5,
                 increasing_pair_size_factor: float = 0.5,
                 games: int = 3,
                 penalty_factor: float = 0.1,
                 max_iter: int = 50,
                 evolutionary_algorithm: str = 'ga',
                 multi_threading: bool = True,
                 **kwargs
                 ):
        """
        :param df: Pandas or dask DataFrame
            Data set

        :param features: List[str]
            Names of the predictor features

        :param target: str
            Name of the target feature

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

        :param evolutionary_algorithm: str
            Name of the reinforced evolutionary algorithm
                -> ga: Genetic Algorithm
                -> si: Swarm Intelligence

        :param multi_threading: bool
            Whether to run each game multi- or single-threaded during each iteration

        :param kwargs: dict
            Key-word arguments
        """
        self.dask_client = HappyLearningUtils().dask_setup(client_name='feature_tournament',
                                                           client_address=kwargs.get('client_address'),
                                                           mode='threads' if kwargs.get('client_mode') is None else kwargs.get('client_mode')
                                                           )
        self.feature_tournament_ai: dict = {}
        if isinstance(df, pd.DataFrame):
            self.df: dd.DataFrame = dd.from_pandas(data=df, npartitions=4 if kwargs.get('partitions') is None else kwargs.get('partitions'))
        elif isinstance(df, dd.DataFrame):
            self.df: dd.DataFrame = df
        else:
            raise FeatureTournamentException('Format of data set ({}) not supported. Use Pandas or dask DataFrame instead'.format(type(df)))
        self.target: str = target
        self.features: List[str] = features
        if self.target in self.features:
            del self.features[self.features.index(self.target)]
        self.df = self.df[self.features + [self.target]]
        self.n_cases: int = len(self.df)
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
        self.games: int = games
        self.penalty_factor: float = penalty_factor
        self.max_iter: int = max_iter
        self.pairs: List[np.array] = []
        self.threads: dict = {}
        self.multi_threading: bool = multi_threading
        self.tournament: bool = False
        self.shapley_additive_explanation: dict = dict(sum={}, game={}, tournament={})
        self.models: List[str] = models
        self.evolutionary_algorithm: str = evolutionary_algorithm
        self.kwargs: dict = kwargs
        self._evolve_feature_tournament_ai()

    def _evolve_feature_tournament_ai(self):
        """
        Evolve ai for feature tournament using genetic algorithm
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
                                                                                 models=['cat'] if self.models is None else self.models,
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
                                                                                 verbose=0 if self.kwargs.get('verbose') is None else self.kwargs.get('verbose'),
                                                                                 **self.kwargs
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
                                                                                   models=['cat'] if self.models is None else self.models,
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
                                                                                   verbose=0 if self.kwargs.get('verbose') is None else self.kwargs.get('verbose'),
                                                                                   **self.kwargs
                                                                                   )
        else:
            raise FeatureTournamentException('Reinforced evolutionary algorithm ({}) not supported'.format(self.evolutionary_algorithm))
        _feature_tournament_ai_learning.optimize()
        self.feature_tournament_ai = _feature_tournament_ai_learning.evolution
        Log(write=False, level='error').log(msg='Feature tournament ai evolved -> {}'.format(self.feature_tournament_ai.get('model_name')))

    def _game(self, iteration: int):
        """
        Play tournament game

        :param iteration: int
            Number of current iteration
        """
        for pair in self.pairs:
            if self.ml_type == 'reg':
                _game = ModelGeneratorReg(model_name=self.feature_tournament_ai.get('model_name'),
                                          reg_params=self.feature_tournament_ai.get('param')
                                          ).generate_model()
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
                                                       train_time_in_seconds=_game.train_time
                                                       )
            else:
                _game = ModelGeneratorClf(model_name=self.feature_tournament_ai.get('model_name'),
                                          clf_params=self.feature_tournament_ai.get('param')
                                          ).generate_model()
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
                                                       train_time_in_seconds=_game.train_time
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

    def play(self) -> dict:
        """
        Play unreal tournament to extract the fittest or most important players based on the concept of shapley values
        """
        Log(write=False, level='info').log(msg='Start penalty with {} players...'.format(self.n_features))
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
            self._permutation(n=_permutation_space)
            _pool: ThreadPool = ThreadPool(processes=len(self.pairs)) if self.multi_threading else None
            for g in range(0, self.games, 1):
                Log(write=False, level='info').log(msg='Iteration {} - Game {} ~ {} players each game'.format(i + 1,
                                                                                                              g + 1,
                                                                                                              _permutation_space
                                                                                                              )
                                                   )
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
                    raise FeatureTournamentException('No feature scored higher than 0 during penalty phase')
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
        return self.shapley_additive_explanation
