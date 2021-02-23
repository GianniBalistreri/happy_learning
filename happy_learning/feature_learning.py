from .feature_engineer import FeatureEngineer
from .feature_selector import FeatureSelector
from .genetic_algorithm import GeneticAlgorithm
from easyexplore.utils import Log
from typing import Dict, List


class FeatureLearningException(Exception):
    """
    Class for handling exceptions for class FeatureLearning
    """
    pass


class FeatureLearning:
    """
    Class for reinforcement feature engineering using supervised machine learning
    """
    def __init__(self,
                 feature_engineer: FeatureEngineer = None,
                 df=None,
                 file_path: str = None,
                 target: str = None,
                 force_target_type: str = None,
                 max_features: int = -1,
                 keep_fittest_only: bool = True,
                 train_continuous_critic: bool = False,
                 train_categorical_critic: bool = False,
                 engineer_time_disparity: bool = True,
                 engineer_categorical: bool = True,
                 engineer_text: bool = True,
                 output_path: str = None,
                 **kwargs
                 ):
        """
        :param feature_engineer: FeatureEngineer
            Pre-defined FeatureEngineer object

        :param df: Pandas or dask DataFrame
            Data set

        :param file_path: str
            Complete file path

        :param target: str
            Name of the target feature

        :param max_features: int
            Number of finally engineered features

        :param keep_fittest_only: bool
            Whether to keep the fittest features only or all (generated) features

        :param engineer_time_disparity: bool
            Whether to process time disparity of datetime features or not

        :param engineer_categorical: bool
            Whether to process categorical features or not

        :param engineer_text: bool
            Whether to process text features or not

        :param output_path: str
            Complete path to write temporary data sets

        :param kwargs: dict
            Key-word arguments
        """
        self.imp_features: List[str] = []
        self.max_features: int = -1
        self.user_defined_max_features: int = max_features
        self.evolved_features: List[str] = []
        self.keep_fittest_only: bool = keep_fittest_only
        self.train_continuous_critic: bool = train_continuous_critic
        self.train_categorical_critic: bool = train_categorical_critic
        self.engineer_text: bool = engineer_text
        self.engineer_categorical: bool = engineer_categorical
        self.engineer_time_disparity: bool = engineer_time_disparity
        if output_path is None:
            self.output_path: str = ''
        else:
            self.output_path: str = output_path.replace('\\', '/')
            if self.output_path[len(self.output_path) - 1] != '/':
                self.output_path = '{}/'.format(self.output_path)
        if feature_engineer is None:
            self.feature_engineer = FeatureEngineer(df=df,
                                                    target_feature=target,
                                                    generate_new_feature=True,
                                                    id_features=kwargs.get('id_features'),
                                                    date_features=kwargs.get('date_features'),
                                                    ordinal_features=kwargs.get('ordinal_features'),
                                                    keep_original_data=False,
                                                    unify_invalid_values=True,
                                                    encode_missing_data=False,
                                                    date_edges=kwargs.get('date_edges'),
                                                    max_level_processing=4 if kwargs.get('max_level_processing') is None else kwargs.get('max_level_processing'),
                                                    auto_cleaning=True if kwargs.get('auto_cleaning') is None else kwargs.get('auto_cleaning'),
                                                    auto_typing=True,
                                                    auto_engineering=False,
                                                    multi_threading=True,
                                                    file_path=file_path,
                                                    sep=',' if kwargs.get('sep') is None else kwargs.get('sep'),
                                                    print_msg=True if kwargs.get('print_msg') is None else kwargs.get('print_msg'),
                                                    seed=1234
                                                    )
        else:
            self.feature_engineer = feature_engineer
        self.feature_engineer.impute(multiple=True, multiple_meth='random', m=25, convergence_threshold=0.99)
        self.feature_engineer.reset_multi_threading()
        if self.engineer_time_disparity:
            self.feature_engineer.disparity(years=True if kwargs.get('years') is None else kwargs.get('years'),
                                            months=True if kwargs.get('months') is None else kwargs.get('months'),
                                            weeks=True if kwargs.get('weeks') is None else kwargs.get('weeks'),
                                            days=True if kwargs.get('days') is None else kwargs.get('days'),
                                            hours=True if kwargs.get('hours') is None else kwargs.get('hours'),
                                            minutes=True if kwargs.get('minutes') is None else kwargs.get('minutes'),
                                            seconds=True if kwargs.get('seconds') is None else kwargs.get('seconds')
                                            )
        #if self.engineer_text:
        #    self.feature_engineer.text_occurances()
        #    self.feature_engineer.linguistic_features()
        self.force_target_type: str = force_target_type
        self.kwargs: dict = kwargs
        self.continuous_learning = None
        self.categorical_learning = None

    def _evolve_feature_learning_ai(self, feature_type: str):
        """
        Evolve ai for feature learning using genetic algorithm

        :param feature_type: str
            Name of the feature type to engineer
                -> continuous: (semi-) continuous features
                -> categorical: categorical (nominal) features
        """
        if feature_type == 'continuous':
            Log(write=False, level='info').log(msg='Evolve feature learning ai for engineering (semi-) continuous features ...')
        else:
            Log(write=False, level='info').log(msg='Evolve feature learning ai for engineering categorical (one-hot encoded) features ...')
        _feature_learner: GeneticAlgorithm = GeneticAlgorithm(mode='model',
                                                              target=self.feature_engineer.get_target(),
                                                              features=self.feature_engineer.get_predictors(),
                                                              re_split_data=False if self.kwargs.get('re_split_data') is None else self.kwargs.get('re_split_data'),
                                                              re_sample_cases=False if self.kwargs.get('re_sample_cases') is None else self.kwargs.get('re_sample_cases'),
                                                              re_sample_features=True,
                                                              max_features=self.max_features,
                                                              labels=self.kwargs.get('labels'),
                                                              models=['cat'] if self.kwargs.get('models') is None else self.kwargs.get('models'),
                                                              model_params=None,
                                                              burn_in_generations=-1 if self.kwargs.get('burn_in_generations') is None else self.kwargs.get('burn_in_generations'),
                                                              warm_start=True if self.kwargs.get('warm_start') is None else self.kwargs.get('warm_start'),
                                                              max_generations=2 if self.kwargs.get('max_generations_ai') is None else self.kwargs.get('max_generations_ai'),
                                                              pop_size=64 if self.kwargs.get('pop_size') is None else self.kwargs.get('pop_size'),
                                                              mutation_rate=0.1 if self.kwargs.get('mutation_rate') is None else self.kwargs.get('mutation_rate'),
                                                              mutation_prob=0.8 if self.kwargs.get('mutation_prob') is None else self.kwargs.get('mutation_prob'),
                                                              parents_ratio=0.5 if self.kwargs.get('parents_ratio') is None else self.kwargs.get('parents_ratio'),
                                                              early_stopping=0 if self.kwargs.get('early_stopping') is None else self.kwargs.get('early_stopping'),
                                                              convergence=False if self.kwargs.get('convergence') is None else self.kwargs.get('convergence'),
                                                              timer_in_seconds=43200 if self.kwargs.get('timer_in_secondes') is None else self.kwargs.get('timer_in_secondes'),
                                                              force_target_type=self.force_target_type,
                                                              plot=False if self.kwargs.get('plot') is None else self.kwargs.get('plot'),
                                                              output_file_path=self.kwargs.get('output_file_path'),
                                                              multi_threading=False if self.kwargs.get('multi_threading') is None else self.kwargs.get('multi_threading'),
                                                              multi_processing=False if self.kwargs.get('multi_processing') is None else self.kwargs.get('multi_processing'),
                                                              log=False if self.kwargs.get('log') is None else self.kwargs.get('log'),
                                                              verbose=0 if self.kwargs.get('verbose') is None else self.kwargs.get('verbose'),
                                                              feature_engineer=self.feature_engineer
                                                              )
        _feature_learner.optimize()
        if feature_type == 'categorical':
            self.categorical_learning = _feature_learner.evolution
        else:
            self.continuous_learning = _feature_learner.evolution
        Log(write=False, level='error').log(msg='Feature learning ai evolved -> {}'.format(_feature_learner.evolution.get('model_name')))

    def _feature_learning(self, feature_type: str):
        """
        Run reinforcement feature learning based on feature types (categorical or continuous)

        :param feature_type: str
            Name of the feature type to engineer
                -> continuous: (semi-) continuous features
                -> categorical: categorical (nominal) features
        """
        Log(write=False, level='error').log(msg='Start feature engineering using {} features'.format('continuous original' if feature_type == 'continuous' else 'categorical one-hot-encoded'))
        if self.kwargs.get('mutation_prob') is None:
            self.kwargs.update(dict(mutation_prob=0.5))
        if self.kwargs.get('max_generations') is None:
            self.kwargs.update(dict(max_generations=5))
        if self.kwargs.get('parents_ratio') is None:
            self.kwargs.update(dict(parents_ratio=0.5))
        _feature_learning_evolution: GeneticAlgorithm = GeneticAlgorithm(mode='feature_engineer',
                                                                         feature_engineer=self.feature_engineer,
                                                                         df=self.feature_engineer.get_data(),
                                                                         target=self.feature_engineer.get_target(),
                                                                         features=self.feature_engineer.get_predictors(),
                                                                         re_split_data=False if self.kwargs.get('re_split_data') is None else self.kwargs.get('re_split_data'),
                                                                         re_sample_cases=False if self.kwargs.get('re_sample_cases') is None else self.kwargs.get('re_sample_cases'),
                                                                         re_sample_features=False,
                                                                         max_features=self.max_features,
                                                                         labels=self.kwargs.get('labels'),
                                                                         models=[self.categorical_learning.get('model_name')] if feature_type == 'categorical' else [self.continuous_learning.get('model_name')],
                                                                         model_params=self.categorical_learning.get('param') if feature_type == 'categorical' else self.continuous_learning.get('param'),
                                                                         burn_in_generations=-1,
                                                                         warm_start=False,
                                                                         max_generations=self.kwargs.get('max_generations'),
                                                                         pop_size=64 if self.kwargs.get('pop_size') is None else self.kwargs.get('pop_size'),
                                                                         mutation_rate=0.1,
                                                                         mutation_prob=self.kwargs.get('mutation_prob'),
                                                                         parents_ratio=self.kwargs.get('parents_ratio'),
                                                                         early_stopping=0,
                                                                         convergence=False,
                                                                         timer_in_seconds=43200 if self.kwargs.get('timer_in_secondes') is None else self.kwargs.get('timer_in_secondes'),
                                                                         force_target_type=self.force_target_type,
                                                                         plot=False if self.kwargs.get('plot') is None else self.kwargs.get('plot'),
                                                                         output_file_path=self.kwargs.get('output_file_path'),
                                                                         multi_threading=False if self.kwargs.get('multi_threading') is None else self.kwargs.get('multi_threading'),
                                                                         multi_processing=False if self.kwargs.get('multi_processing') is None else self.kwargs.get('multi_processing'),
                                                                         log=False if self.kwargs.get('log') is None else self.kwargs.get('log'),
                                                                         verbose=0 if self.kwargs.get('verbose') is None else self.kwargs.get('verbose')
                                                                         )
        _feature_learning_evolution.optimize()
        self.evolved_features.extend(_feature_learning_evolution.evolved_features)
        self.feature_engineer = _feature_learning_evolution.feature_engineer
        Log(write=False, level='error').log(msg='Generated {} engineered features'.format(len(_feature_learning_evolution.mutated_features.get('child'))))
        if self.keep_fittest_only:
            Log(write=False, level='error').log(msg='Selected {} fittest features'.format(len(_feature_learning_evolution.evolved_features)))
            _erase: Dict[str, List[str]] = dict(features=list(set(_feature_learning_evolution.mutated_features.get('child')).difference(_feature_learning_evolution.evolved_features)))
            if len(_erase.get('features')) > 0:
                self.feature_engineer.clean(markers=_erase)
        del _feature_learning_evolution

    def _generate_categorical_features(self):
        """
        Generate additional categorical features by processing continuous, date and text features
        """
        self.feature_engineer.label_encoder(encode=True)
        self.feature_engineer.date_categorizer()
        self.feature_engineer.binning(supervised=True, optimal=True, optimal_meth='bayesian_blocks')
        self.feature_engineer.one_hot_encoder(threshold=self.kwargs.get('threshold'))
        _features: List[str] = self.feature_engineer.get_features(feature_type='ordinal') + self.feature_engineer.get_features(feature_type='categorical')
        self.feature_engineer.set_predictors(features=_features, exclude_original_data=True)

    def _generate_continuous_features(self):
        """
        Generate additional continuous features by processing date and text features
        """
        self.feature_engineer.disparity()
        _features: List[str] = self.feature_engineer.get_features(feature_type='ordinal') + self.feature_engineer.get_features(feature_type='continuous')
        self.feature_engineer.set_predictors(features=_features, exclude_original_data=False)

    def _feature_critic(self):
        """
        Get feature importance for criticizing feature learning
        """
        self.imp_features = FeatureSelector(df=self.feature_engineer.get_training_data(),
                                            target=self.feature_engineer.get_target(),
                                            features=self.feature_engineer.get_predictors(),
                                            aggregate_feature_imp=None,
                                            visualize_all_scores=False,
                                            visualize_variant_scores=False,
                                            visualize_core_feature_scores=False,
                                            path=None
                                            ).get_imp_features(meth='shapley', imp_threshold=0.0)
        self.feature_engineer.set_imp_features(imp_features=self.imp_features.get('imp_features'))

    def _pre_define_max_features(self, feature_type: str, scale: bool = True):
        """
        Pre-define (maximum) number of features of each model

        :param feature_type: str
            Name of the feature type to use for pre-defining number of features

        :param scale: bool
            Whether to scale number of features per model by number of unique features and the population size or just use pre-defined number of features
        """
        _pop_size: int = 64 if self.kwargs.get('pop_size') is None else self.kwargs.get('pop_size')
        if feature_type == 'categorical':
            if scale:
                self.max_features = 2 * round(10 + (len(self.feature_engineer.get_predictors()) / _pop_size))
                if self.max_features > len(self.feature_engineer.get_predictors()):
                    self.max_features = 50
                if self.max_features <= 2:
                    self.max_features = 10
            else:
                self.max_features = self.user_defined_max_features
        elif feature_type == 'continuous':
            if scale:
                self.max_features = round(10 + (len(self.feature_engineer.get_predictors()) / _pop_size))
                if self.max_features > len(self.feature_engineer.get_predictors()):
                    self.max_features = 4
                if self.max_features <= 2:
                    self.max_features = 4
            else:
                self.max_features = self.user_defined_max_features

    def ga(self) -> FeatureEngineer:
        """
        Run genetic algorithm for optimizing semi-continuous, continuous and (one-hot-encoded) categorical feature engineering

        :return: FeatureEngineer
            FeatureEngineer object containing the hole feature engineering, meta data included
        """
        self._generate_continuous_features()
        if len(self.feature_engineer.get_predictors()) >= 4:
            if self.train_continuous_critic:
                self._feature_critic()
            self._pre_define_max_features(feature_type='continuous', scale=True if self.user_defined_max_features <= 1 else False)
            self._evolve_feature_learning_ai(feature_type='continuous')
            self._feature_learning(feature_type='continuous')
        else:
            Log(write=False, env='dev').log(msg='Not enough continuous or ordinal features to efficiently run reinforcement feature learning framework')
        if self.engineer_categorical:
            if self.output_path is None:
                Log(write=False, level='info').log(msg='No output path found for writing temporary data for applying one-hot merging')
            else:
                self.feature_engineer.save(file_path='{}feature_learning.p'.format(self.output_path),
                                           cls_obj=True,
                                           overwrite=True,
                                           create_dir=False
                                           )
                del self.feature_engineer
                self.feature_engineer: FeatureEngineer = FeatureEngineer(feature_engineer_file_path='{}feature_learning.p'.format(self.output_path))
                _continuous_features: List[str] = self.feature_engineer.get_feature_types().get('continuous')
                self.feature_engineer.clean(markers=dict(features=_continuous_features))
                self._generate_categorical_features()
                _remaining_non_categorical_features: List[str] = self.feature_engineer.get_feature_types().get('date')# + self.feature_engineer.get_feature_types().get('text')
                self.feature_engineer.clean(markers=dict(features=_remaining_non_categorical_features))
                if len(self.feature_engineer.get_predictors()) >= 4:
                    if self.train_categorical_critic:
                        self._feature_critic()
                    self._pre_define_max_features(feature_type='categorical', scale=True if self.user_defined_max_features <= 1 else False)
                    self._evolve_feature_learning_ai(feature_type='categorical')
                    self._feature_learning(feature_type='categorical')
                    self.feature_engineer.merge_engineer(feature_engineer_file_path='{}feature_learning.p'.format(self.output_path))
                else:
                    Log(write=False, env='dev').log(msg='Not enough categorical features to efficiently run reinforcement feature learning framework')
        if len(self.evolved_features) > 0:
            self.feature_engineer.set_predictors(features=list(set(self.evolved_features)), exclude_original_data=False)
        return self.feature_engineer

    def nas(self):
        """
        Run neural architecture search for continuous feature engineering
        """
        raise NotImplementedError('Neural architecture search (NAS) for feature learning not implemented')
