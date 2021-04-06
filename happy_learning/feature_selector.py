import dask.dataframe as dd
import os
import pandas as pd

from .feature_tournament import FeatureTournament
from .sampler import MLSampler
from .supervised_machine_learning import ModelGeneratorClf, ModelGeneratorReg
from .utils import HappyLearningUtils
from easyexplore.data_visualizer import DataVisualizer
from sklearn.feature_selection import SelectFromModel
from typing import Dict, List, Union


class FeatureSelectorException(Exception):
    """
    Class for setting up exception for class FeatureSelection
    """
    pass


class FeatureSelector:
    """
    Class for analyzing the importance of each feature in the data set
    """
    def __init__(self,
                 df: Union[dd.DataFrame, pd.DataFrame],
                 target: str,
                 features: List[str] = None,
                 force_target_type: str = None,
                 aggregate_feature_imp: Dict[str, dict] = None,
                 visualize_all_scores: bool = True,
                 visualize_variant_scores: bool = True,
                 visualize_core_feature_scores: bool = True,
                 path: str = None,
                 **kwargs
                 ):
        """
        :param df: pd.DataFrame
            Data set

        :param target: str
            Name of the target variable

        :param features: List[str]
            Name of the features

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

        :param kwargs: dict
            Key-word arguments
        """
        self.seed: int = 1234
        self.cpu_cores: int = os.cpu_count() - 1 if os.cpu_count() > 1 else os.cpu_count()
        if isinstance(df, pd.DataFrame):
            self.df: dd.DataFrame = dd.from_pandas(data=df, npartitions=4 if kwargs.get('partitions') is None else kwargs.get('partitions'))
        elif isinstance(df, dd.DataFrame):
            self.df: dd.DataFrame = df
        else:
            raise FeatureSelectorException('Format of data set ({}) not supported. Use Pandas or dask DataFrame instead'.format(type(df)))
        self.target: str = target
        self.imp_score: Dict[str, float] = {}
        self.aggregate_feature_imp: Dict[str, dict] = aggregate_feature_imp
        self.features: List[str] = list(self.df.keys()) if features is None else features
        if self.target in self.features:
            del self.features[self.features.index(self.target)]
        self.force_target_type: str = force_target_type
        if self.force_target_type is None:
            self.ml_type: str = HappyLearningUtils().get_ml_type(values=self.df[self.target].values) if kwargs.get('ml_type') is None else kwargs.get('ml_type')
        else:
            self.ml_type: str = self.force_target_type
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
        self.kwargs: dict = kwargs

    def get_imp_features(self,
                         meth: str = 'shapley',
                         model: str = 'cat',
                         imp_threshold: float = 0.01,
                         plot_type: str = 'bar'
                         ) -> dict:
        """
        Get important features

        :param meth: str
            Feature selection method
                -> rf: Random Forest for multi-scaled features
                -> xgb: Extreme Gradient Boosting Decision Tree for multi-scaled features
                -> lasso: Lasso Regression for continuous features
                -> pca: Principle Component Analysis for continuous features
                -> shapley: Shapley Value approximation using feature tournament framework

        :param model: str
            Name of the supervised learning model to use for ai evolution

        :param imp_threshold: float
            Threshold of importance score

        :param plot_type: str
            Name of the plot type
                -> pie: Pie Chart
                -> bar: Bar Chart

        :return: dict:
            Important features and importance score
        """
        _imp_plot: dict = {}
        _imp_features: List[str] = []
        _core_features: List[str] = []
        _processed_features: List[str] = []
        _imp_threshold: float = imp_threshold if (imp_threshold >= 0) and (imp_threshold <= 1) else 0.7
        if meth == 'shapley':
            _imp_scores: dict = FeatureTournament(df=self.df,
                                                  features=self.features,
                                                  target=self.target,
                                                  force_target_type=self.force_target_type,
                                                  models=['cat'] if model is None else [model],
                                                  init_pairs=3 if self.kwargs.get('init_pairs') is None else self.kwargs.get('init_pairs'),
                                                  init_games=5 if self.kwargs.get('init_games') is None else self.kwargs.get('init_games'),
                                                  increasing_pair_size_factor=0.05 if self.kwargs.get('increasing_pair_size_factor') is None else self.kwargs.get('increasing_pair_size_factor'),
                                                  games=3 if self.kwargs.get('games') is None else self.kwargs.get('games'),
                                                  penalty_factor=0.1 if self.kwargs.get('penalty_factor') is None else self.kwargs.get('penalty_factor'),
                                                  evolutionary_algorithm='si' if self.kwargs.get('evolutionary_algorithm') is None else self.kwargs.get('evolutionary_algorithm'),
                                                  max_iter=50 if self.kwargs.get('max_iter') is None else self.kwargs.get('max_iter'),
                                                  **self.kwargs
                                                  ).play()
            _df: pd.DataFrame = pd.DataFrame(data=_imp_scores.get('total'), index=['score']).transpose()
            _df = _df.sort_values(by='score', axis=0, ascending=False, inplace=False)
            _df['feature'] = _df.index.values
            _imp_features = _df['feature'].values.tolist()
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
            _game_df: pd.DataFrame = pd.DataFrame(data=_imp_scores.get('game'))
            #_game_df['game'] = _game_df.index.values
            _tournament_df: pd.DataFrame = pd.DataFrame(data=_imp_scores.get('tournament'))
            #_tournament_df['game'] = _tournament_df.index.values
            if self.visualize_all_scores:
                _imp_plot: dict = {'Feature Tournament Game Stats (Shapley Scores)': dict(data=_game_df,
                                                                                          features=list(_game_df.columns),
                                                                                          plot_type='violin',
                                                                                          melt=True,
                                                                                          render=True,
                                                                                          file_path='{}feature_tournament_game_stats.html'.format(self.path) if self.path is not None else None
                                                                                          ),
                                   'Feature Tournament Stats (Game Size)': dict(data=_tournament_df,
                                                                                features=list(_tournament_df.columns),
                                                                                #color_feature='game',
                                                                                plot_type='heat',
                                                                                render=True,
                                                                                file_path='{}feature_tournament_game_size.html'.format(self.path) if self.path is not None else None
                                                                                ),
                                   'Feature Importance (Shapley Scores)': dict(df=_df,
                                                                               plot_type=plot_type,
                                                                               render=True if self.path is None else False,
                                                                               file_path='{}feature_importance_shapley.html'.format(self.path) if self.path is not None else None,
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
        elif meth == 'pca':
            raise NotImplementedError('Feature selection using Principle Component Analysis (PCA) not implemented')
        elif meth in ['lasso', 'cat', 'rf', 'gbo', 'xgb']:
            if self.ml_type == 'reg':
                _ml = ModelGeneratorReg(model_name=meth).generate_model()
            else:
                _ml = ModelGeneratorClf(model_name=meth).generate_model()
            _train_test_data: dict = MLSampler(df=self.df, target=self.target, features=self.features, stratification=False).train_test_sampling()
            _ml.train(x=_train_test_data.get('x_train').values, y=_train_test_data.get('y_train').values)
            _imp_features = self.df[self.features].columns[SelectFromModel(estimator=_ml.model,
                                                                           threshold='median' if imp_threshold is None else imp_threshold,
                                                                           prefit=True,
                                                                           norm_order=1,
                                                                           max_features=None
                                                                           ).get_support()]
            _imp_features = list(set(_imp_features))
            self.imp_score.update({self.features[i]: ft_imp for i, ft_imp in enumerate(_ml.model.feature_importances_)})
            if plot_type not in ['pie', 'bar']:
                raise FeatureSelectorException(
                    'Plot type ({}) for visualizing feature importance not supported'.format(plot_type))
            _df: pd.DataFrame = pd.DataFrame(data=self.imp_score.values(), index=list(self.imp_score.keys()),
                                             columns=['importance'])
            _df = _df.sort_values(by=['importance'], axis=0, ascending=False, inplace=False, kind='quicksort')
            if self.visualize_all_scores:
                _imp_plot: dict = {'Feature Importance ({})'.format(meth.upper()): dict(data=_df,
                                                                                        plot_type=plot_type,
                                                                                        melt=False,
                                                                                        interactive=True,
                                                                                        render=True if self.path is None else False,
                                                                                        file_path='{}feature_importance_{}.html'.format(self.path, meth.upper()) if self.path is not None else None,
                                                                                        kwargs=dict(layout={})
                                                                                        )
                                   }
                if plot_type == 'pie':
                    _imp_plot[list(_imp_plot.keys())[0]]['kwargs'].update({'values': _df['importance'].values * 100,
                                                                           'labels': _df.index.values.tolist(),
                                                                           'textposition': 'inside',
                                                                           'marker': dict(colors=_df['importance'])
                                                                           })
                elif plot_type == 'bar':
                    _imp_plot[list(_imp_plot.keys())[0]]['kwargs'].update({'y': _df['importance'].values,
                                                                           'x': _df.index.values.tolist(),
                                                                           'marker': dict(color=_df['importance'],
                                                                                          colorscale='rdylgn',
                                                                                          autocolorscale=True
                                                                                          )
                                                                           })
        else:
            raise FeatureSelectorException('Method ({}) for scoring and selecting important features not supported'.format(meth))
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
        return dict(imp_features=_imp_features,
                    imp_score=self.imp_score,
                    imp_threshold=imp_threshold,
                    imp_core_features=_core_features,
                    imp_processed_features=_processed_features
                    )
