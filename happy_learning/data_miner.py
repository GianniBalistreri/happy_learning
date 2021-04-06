import dask.dataframe as dd
import numpy as np
import os
import pandas as pd

from .evaluate_machine_learning import EvalClf
from .feature_learning import FeatureEngineer, FeatureLearning
from .feature_selector import FeatureSelector
from .genetic_algorithm import GeneticAlgorithm
from .sampler import MLSampler
from .supervised_machine_learning import ModelGeneratorClf, ModelGeneratorReg
from .swarm_intelligence import SwarmIntelligence
from .utils import HappyLearningUtils
from easyexplore.data_import_export import DataExporter
from easyexplore.data_visualizer import DataVisualizer
from easyexplore.utils import EasyExploreUtils, Log
from typing import List, Union

# TODO:
#   Visualization:
#       -> visualize parameter importance analysis for developing / evaluating genetic algorithm critic
#   Sampling:
#       -> kfold cross-validation for using manual model development


class DataMinerException(Exception):
    """
    Class for handling exceptions
    """
    pass


class DataMiner:
    """
    Class for running reinforced prototyping using structured (tabular) data
    """
    def __init__(self,
                 temp_dir: str,
                 df: Union[dd.DataFrame, pd.DataFrame] = None,
                 file_path: str = None,
                 target: str = None,
                 predictors: List[str] = None,
                 feature_engineer: FeatureEngineer = None,
                 feature_generator: bool = True,
                 train_critic: bool = False,
                 cpu_cores: int = 0,
                 plot: bool = True,
                 output_path: str = None,
                 **kwargs
                 ):
        """
        :param temp_dir: str
            File path of the temporary feature files

        :param df: Pandas or dask DataFrame
            Data set

        :param file_path: str
            Complete file path of data input file or db

        :param target: str
            Name of the target

        :param feature_engineer: FeatureEngineer
            FeatureEngineer object

        :param feature_generator: bool
            Whether to generate features by using machine learning

        :param train_critic: bool
            Whether to train critic based on the relative feature importance or not

        :param cpu_cores: int
            Number of cpu cores to use
                -> 0: if there are more than 1 cpu core available then all cores will be used except by 1

        :param plot: bool
            Whether to plot evaluation results or not

        :param output_path: str
            Path or directory to export visualization to

        :param kwargs: dict
            Key-word arguments for class FeatureEngineer if feature_engineer is None
        """
        if not os.path.isdir(output_path):
            raise DataMinerException('Invalid output path ({})'.format(output_path))
        if feature_engineer is None:
            if df is None:
                if file_path is None:
                    raise DataMinerException('No data set found')
                if not os.path.isfile(file_path):
                    raise DataMinerException('Invalid data file path ({})'.format(file_path))
            if target is None:
                raise DataMinerException('No target feature found')
            self.feature_engineer: FeatureEngineer = FeatureEngineer(df=df,
                                                                     target_feature=target,
                                                                     generate_new_feature=True if kwargs.get('generate_new_feature') is None else kwargs.get('generate_new_feature'),
                                                                     id_text_features=kwargs.get('id_text_features'),
                                                                     date_features=kwargs.get('date_features'),
                                                                     ordinal_features=kwargs.get('ordinal_features'),
                                                                     keep_original_data=False if kwargs.get('keep_original_data') is None else kwargs.get('keep_original_data'),
                                                                     unify_invalid_values=True if kwargs.get('unify_invalid_values') is None else kwargs.get('unify_invalid_values'),
                                                                     encode_missing_data=False if kwargs.get('encode_missing_data') is None else kwargs.get('encode_missing_data'),
                                                                     date_edges=kwargs.get('date_edges'),
                                                                     max_level_processing=5 if kwargs.get('max_level_processing') is None else kwargs.get('max_level_processing'),
                                                                     auto_cleaning=True if kwargs.get('auto_cleaning') is None else kwargs.get('auto_cleaning'),
                                                                     auto_typing=True if kwargs.get('auto_typing') is None else kwargs.get('auto_typing'),
                                                                     auto_engineering=False if kwargs.get('auto_engineering') is None else kwargs.get('auto_engineering'),
                                                                     multi_threading=False,
                                                                     file_path=file_path,
                                                                     temp_dir=temp_dir,
                                                                     sep=',' if kwargs.get('sep') is None else kwargs.get('sep'),
                                                                     print_msg=True if kwargs.get('print_msg') is None else kwargs.get('print_msg'),
                                                                     seed=1234 if kwargs.get('seed') is None else kwargs.get('seed'),
                                                                     )
        else:
            if feature_engineer.get_target() is None:
                if target is None:
                    raise DataMinerException('No target feature set')
                else:
                    if target in feature_engineer.get_features():
                        feature_engineer.set_target(feature=target)
                    else:
                        raise DataMinerException('Target feature ({}) not found in data set'.format(target))
            self.feature_engineer: FeatureEngineer = feature_engineer
        self.force_target_type = None
        self.feature_generator: bool = feature_generator
        self.train_critic: bool = train_critic
        if cpu_cores <= 0:
            self.cpu_cores: int = os.cpu_count() - 1 if os.cpu_count() > 1 else os.cpu_count()
        else:
            if cpu_cores <= os.cpu_count():
                self.cpu_cores: int = cpu_cores
            else:
                self.cpu_cores: int = os.cpu_count() - 1 if os.cpu_count() > 1 else os.cpu_count()
        self.seed: int = 1234
        self.plot: bool = plot
        self.output_path: str = output_path
        if self.output_path is not None:
            self.output_path = self.output_path.replace('\\', '/')
            if os.path.isfile(self.output_path):
                self.output_path = self.output_path.replace(self.output_path.split('/')[-1], '')
            else:
                if self.output_path.split('/')[-1] != '':
                    self.output_path = '{}/'.format(self.output_path)
        self.kwargs: dict = kwargs

    def supervised(self,
                   models: List[str] = None,
                   feature_selector: str = 'shapley',
                   top_features: float = 0.5,
                   optimizer: str = 'ga',
                   force_target_type: str = None,
                   train: bool = True,
                   train_size: float = 0.8,
                   random: bool = True,
                   stratification: bool = False,
                   clf_eval_metric: str = 'auc',
                   reg_eval_metric: str = 'rmse_norm',
                   save_train_test_data: bool = True,
                   save_optimizer: bool = True,
                   **kwargs
                   ):
        """
        Run supervised machine learning models

        :param models: List[str]
            Name of the supervised machine learning models to use

        :param feature_selector: str
            Feature selection method:
                -> shapley: Shapley Value based on the FeatureTournament framework

        :param top_features: float
            Amount of top features to select

        :param optimizer: str
            Model optimizer method:
                -> ga: Genetic Algorithm
                -> si: Swarm Intelligence
                -> None: Develop model manually using pre-defined parameter config without optimization

        :param force_target_type: str
            Name of the target type to force (useful if target type is ordinal)
                -> reg: define target type as regression instead of multi classification
                -> clf_multi: define target type as multi classification instead of regression

        :param train: bool
            Whether to train or to predict from supervised machine learning models

        :param train_size: float
            Proportion of cases in the training data set

        :param random: bool
            Whether to sample randomly or by index

        :param stratification: bool
            Whether to stratify train and test data sets

        :param clf_eval_metric: str
            Name of the metric to use for evaluate classification models

        :param reg_eval_metric: str
            Name of the metric to use for evaluate regression models

        :param save_train_test_data: bool
            Whether to save train-test data split or not

        :param save_optimizer: bool
            Whether to save "GeneticAlgorithm" or "SwarmIntelligence" object or not

        :param kwargs: dict
            Key-word arguments of classes FeatureSelector / DataExporter / GeneticAlgorithm / SwarmIntelligence / MLSampler / DataVisualizer
        """
        self.force_target_type = force_target_type
        if train:
            _train_size: float = train_size if (train_size > 0) and (train_size < 1) else 0.8
            if self.feature_generator:
                self.feature_engineer = FeatureLearning(feature_engineer=self.feature_engineer,
                                                        target=self.feature_engineer.get_target(),
                                                        force_target_type=force_target_type,
                                                        max_features=0,
                                                        keep_fittest_only=True if kwargs.get('keep_fittest_only') is None else kwargs.get('keep_fittest_only'),
                                                        train_continuous_critic=False if kwargs.get('train_continuous_critic') is None else kwargs.get('train_continuous_critic'),
                                                        train_categorical_critic=False if kwargs.get('train_categorical_critic') is None else kwargs.get('train_categorical_critic'),
                                                        engineer_time_disparity=True if kwargs.get('engineer_time_disparity') is None else kwargs.get('engineer_time_disparity'),
                                                        engineer_categorical=False if kwargs.get('engineer_categorical') is None else kwargs.get('engineer_categorical'),
                                                        output_path=self.output_path,
                                                        **self.kwargs
                                                        ).ga()
            else:
                self.feature_engineer.set_predictors(exclude_original_data=False)
            if feature_selector is not None:
                _imp_features: dict = FeatureSelector(df=self.feature_engineer.get_training_data(output='df_dask'),
                                                      target=self.feature_engineer.get_target(),
                                                      features=self.feature_engineer.get_predictors(),
                                                      force_target_type=force_target_type,
                                                      aggregate_feature_imp=self.feature_engineer.get_processing()['features']['raw'],
                                                      visualize_all_scores=self.plot if kwargs.get('visualize_all_scores') is None else kwargs.get('visualize_all_scores'),
                                                      visualize_variant_scores=self.plot if kwargs.get('visualize_variant_scores') is None else kwargs.get('visualize_variant_scores'),
                                                      visualize_core_feature_scores=self.plot if kwargs.get('visualize_core_feature_scores') is None else kwargs.get('visualize_core_feature_scores'),
                                                      path=self.output_path
                                                      ).get_imp_features(meth=feature_selector,
                                                                         imp_threshold=0.001 if kwargs.get('imp_threshold') is None else kwargs.get('imp_threshold')
                                                                         )
                _ratio: float = top_features if (top_features > 0) and (top_features <= 1) else 0.5
                _top_n_features: int = round(self.feature_engineer.get_n_predictors() * _ratio)
                self.feature_engineer.set_predictors(features=_imp_features.get('imp_features')[0:_top_n_features],
                                                     exclude_original_data=False
                                                     )
                if self.output_path is not None or kwargs.get('file_path') is not None:
                    DataExporter(obj=_imp_features,
                                 file_path='{}feature_importance.pkl'.format(self.output_path) if kwargs.get('file_path') is None else kwargs.get('file_path'),
                                 create_dir=True if kwargs.get('create_dir') is None else kwargs.get('create_dir'),
                                 overwrite=False if kwargs.get('overwrite') is None else kwargs.get('overwrite')
                                 ).file()
            if optimizer == 'ga':
                _optimizer: GeneticAlgorithm = GeneticAlgorithm(mode='model',
                                                                df=self.feature_engineer.get_training_data(),
                                                                target=self.feature_engineer.get_target(),
                                                                force_target_type=force_target_type,
                                                                features=self.feature_engineer.get_predictors(),
                                                                stratify=stratification,
                                                                labels=None if kwargs.get('labels') is None else kwargs.get('labels'),
                                                                models=models,
                                                                burn_in_generations=10 if kwargs.get('burn_in_generations') is None else kwargs.get('burn_in_generations'),
                                                                max_generations=25 if kwargs.get('max_generations') is None else kwargs.get('max_generations'),
                                                                pop_size=64 if kwargs.get('pop_size') is None else kwargs.get('pop_size'),
                                                                mutation_rate=0.1 if kwargs.get('mutation_rate') is None else kwargs.get('mutation_rate'),
                                                                mutation_prob=0.15 if kwargs.get('mutation_prob') is None else kwargs.get('mutation_prob'),
                                                                parents_ratio=0.5 if kwargs.get('parents_ratio') is None else kwargs.get('parents_ratio'),
                                                                early_stopping=0 if kwargs.get('early_stopping') is None else kwargs.get('early_stopping'),
                                                                convergence=False if kwargs.get('convergence') is None else kwargs.get('convergence'),
                                                                convergence_measure='median' if kwargs.get('convergence_measure') is None else kwargs.get('convergence_measure'),
                                                                timer_in_seconds=43200 if kwargs.get('timer_in_seconds') is None else kwargs.get('timer_in_seconds'),
                                                                plot=self.plot,
                                                                output_file_path=self.output_path
                                                                )
                _optimizer.optimize()
                if save_train_test_data:
                    DataExporter(obj=_optimizer.data_set,
                                 file_path='{}train_test_data.pkl'.format(self.output_path),
                                 create_dir=False if kwargs.get('create_dir') is None else kwargs.get('create_dir'),
                                 overwrite=False if kwargs.get('overwrite') is None else kwargs.get('overwrite')
                                 ).file()
                if save_optimizer:
                    _optimizer.save_evolution(ga=True,
                                              model=False if kwargs.get('model') is None else kwargs.get('model'),
                                              evolution_history=False if kwargs.get('evolution_history') is None else kwargs.get('evolution_history'),
                                              generation_history=False if kwargs.get('generation_history') is None else kwargs.get('generation_history'),
                                              final_generation=False if kwargs.get('final_generation') is None else kwargs.get('final_generation'),
                                              )
            elif optimizer == 'si':
                _optimizer: SwarmIntelligence = SwarmIntelligence(mode='model',
                                                                  df=self.feature_engineer.get_training_data(),
                                                                  target=self.feature_engineer.get_target(),
                                                                  force_target_type=force_target_type,
                                                                  features=self.feature_engineer.get_predictors(),
                                                                  stratify=stratification,
                                                                  labels=None if kwargs.get('labels') is None else kwargs.get('labels'),
                                                                  models=models,
                                                                  burn_in_adjustments=-1 if kwargs.get('burn_in_adjustments') is None else kwargs.get('burn_in_adjustments'),
                                                                  max_adjustments=10 if kwargs.get('max_adjustments') is None else kwargs.get('max_adjustments'),
                                                                  pop_size=64 if kwargs.get('pop_size') is None else kwargs.get('pop_size'),
                                                                  adjustment_rate=0.1 if kwargs.get('adjustment_rate') is None else kwargs.get('adjustment_rate'),
                                                                  adjustment_prob=0.15 if kwargs.get('adjustment_prob') is None else kwargs.get('adjustment_prob'),
                                                                  early_stopping=0 if kwargs.get('early_stopping') is None else kwargs.get('early_stopping'),
                                                                  convergence=False if kwargs.get('convergence') is None else kwargs.get('convergence'),
                                                                  convergence_measure='median' if kwargs.get('convergence_measure') is None else kwargs.get('convergence_measure'),
                                                                  timer_in_seconds=43200 if kwargs.get('timer_in_seconds') is None else kwargs.get('timer_in_seconds'),
                                                                  plot=self.plot,
                                                                  output_file_path=self.output_path
                                                                  )
                _optimizer.optimize()
                if save_train_test_data:
                    DataExporter(obj=_optimizer.data_set,
                                 file_path='{}train_test_data.pkl'.format(self.output_path),
                                 create_dir=False if kwargs.get('create_dir') is None else kwargs.get('create_dir'),
                                 overwrite=False if kwargs.get('overwrite') is None else kwargs.get('overwrite')
                                 ).file()
                if save_optimizer:
                    _optimizer.save_evolution(si=True,
                                              model=False if kwargs.get('model') is None else kwargs.get('model'),
                                              evolution_history=False if kwargs.get('evolution_history') is None else kwargs.get('evolution_history'),
                                              adjustment_history=False if kwargs.get('adjustmenmt_history') is None else kwargs.get('generation_history'),
                                              final_adjustment=False if kwargs.get('final_adjustment') is None else kwargs.get('final_adjustment'),
                                              )
            else:
                _model_eval_plot: dict = {}
                _data_set: dict = MLSampler(df=self.feature_engineer.get_data(),
                                            target=self.feature_engineer.get_target(),
                                            features=self.feature_engineer.get_predictors(),
                                            train_size=_train_size,
                                            random_sample=random,
                                            stratification=stratification
                                            ).train_test_sampling(validation_split=0.1 if kwargs.get('validation_split') is None else kwargs.get('validation_split'))
                if save_train_test_data:
                    DataExporter(obj=_data_set,
                                 file_path='{}train_test_data.pkl'.format(self.output_path),
                                 create_dir=True if kwargs.get('create_dir') is None else kwargs.get('create_dir'),
                                 overwrite=False if kwargs.get('overwrite') is None else kwargs.get('overwrite')
                                 ).file()
                for model in models:
                    if HappyLearningUtils().get_ml_type(values=self.feature_engineer.get_target_values()) == 'reg':
                        _model = ModelGeneratorReg(model_name=model, reg_params=None).generate_model()
                        _model.train(x=_data_set.get('x_train').values,
                                     y=_data_set.get('y_train').values,
                                     validation=dict(x_val=_data_set.get('x_val').values,
                                                     y_val=_data_set.get('y_val').values
                                                     )
                                     )
                        _pred: np.array = _model.predict(x=_data_set.get('x_test').values)
                        _model.eval(obs=_data_set.get('y_test').values, pred=_pred, eval_metric=[reg_eval_metric])
                        _perc_table: pd.DataFrame = EasyExploreUtils().get_perc_eval(pred=_pred,
                                                                                     obs=_data_set.get('y_test').values.tolist(),
                                                                                     aggregation='median',
                                                                                     percentiles=10
                                                                                     )
                        _min_table: pd.DataFrame = EasyExploreUtils().get_perc_eval(pred=_pred,
                                                                                    obs=_data_set.get('y_test').values.tolist(),
                                                                                    aggregation='min',
                                                                                    percentiles=10
                                                                                    )
                        _max_table: pd.DataFrame = EasyExploreUtils().get_perc_eval(pred=_pred,
                                                                                    obs=_data_set.get('y_test').values.tolist(),
                                                                                    aggregation='max',
                                                                                    percentiles=10
                                                                                    )
                        _multi: dict = {'bar_obs': dict(y=_perc_table['obs'].values,
                                                        name='obs',
                                                        error_y=dict(type='data',
                                                                     array=_max_table['obs'].values - _min_table[
                                                                         'obs'].values)
                                                        ),
                                        'bar_preds': dict(y=_perc_table['preds'].values,
                                                          name='pred',
                                                          error_y=dict(type='data',
                                                                       array=_max_table['preds'].values - _min_table[
                                                                           'preds'].values)
                                                          )
                                        }
                        _model_eval_df: pd.DataFrame(data={'obs': _data_set.get('y_test').values, 'preds': _pred})
                        _model_eval_df['abs_diff'] = _model_eval_df['obs'] - _model_eval_df['preds']
                        _model_eval_df['rel_diff'] = _model_eval_df['obs'] / _model_eval_df['preds']
                        # TODO: Add train & test error to plot
                        _model_eval_plot.update({'Prediction vs. Observation (Value Based)': dict(data=_model_eval_df,
                                                                                                  features=['obs', 'preds'],
                                                                                                  plot_type='joint',
                                                                                                  render=True,
                                                                                                  file_path='{}prediction_scatter_{}.html'.format(self.output_path, model)
                                                                                                  ),
                                                 'Prediction vs. Observation (Range Based)': dict(data=_model_eval_df,
                                                                                                  features=['obs', 'preds', 'abs_diff', 'rel_diff'],
                                                                                                  plot_type='parcoords',
                                                                                                  render=True,
                                                                                                  file_path='{}prediction_coords_{}.html'.format(self.output_path, model)
                                                                                                  ),
                                                 'Prediction vs. Observation (Percentile Based)': dict(data=_perc_table,
                                                                                                       plot_type='multi',
                                                                                                       render=True,
                                                                                                       file_path='{}prediction_percentiles_{}.html'.format(self.output_path, model),
                                                                                                       kwargs=dict(layout=dict(barmode='group',
                                                                                                                               xaxis=dict(tickmode='array',
                                                                                                                                          tickvals=[p for p in range(0, 10, 1)],
                                                                                                                                          ticktext=[str(label) for label in _perc_table['obs'].values.tolist()]
                                                                                                                                          )
                                                                                                                               ),
                                                                                                                   multi=_multi
                                                                                                                   )
                                                                                                       )
                                                 })
                    else:
                        _model = ModelGeneratorClf(model_name=model, clf_params={}).generate_model()
                        _model.train(x=_data_set.get('x_train').values,
                                     y=_data_set.get('y_train').values,
                                     validation=dict(x_val=_data_set.get('x_val').values,
                                                     y_val=_data_set.get('y_val').values
                                                     )
                                     )
                        _pred: np.array = _model.predict(x=_data_set.get('x_test').values)
                        _model.eval(obs=_data_set.get('y_test').values, pred=_pred, eval_metric=[clf_eval_metric])
                        _confusion_matrix: pd.DataFrame = EvalClf(obs=_data_set.get('y_test').values.tolist(),
                                                                  pred=_pred,
                                                                  probability=True
                                                                  ).confusion(normalize='true')
                        _model_eval_plot.update({'Confusion Matrix': dict(data=_confusion_matrix,
                                                                          plot_type='heat',
                                                                          kwargs={'layout': {'xaxis': {'title': 'Observation'},
                                                                                             'yaxis': {'title': 'Prediction'}
                                                                                             },
                                                                                  'text': _confusion_matrix.values.tolist()
                                                                                  }
                                                                          )
                                                 })
                    if self.output_path is not None:
                        DataExporter(obj=_model.model,
                                     file_path='{}model_{}'.format(self.output_path, model),
                                     create_dir=True,
                                     overwrite=False
                                     ).file()
        else:
            raise DataMinerException('Prediction method not implemented yet')
