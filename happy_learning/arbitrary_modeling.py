"""

Generate Hyper-Parameter Configuration purely arbitrary (for benchmarking purposes)

"""

import numpy as np
import pandas as pd

from happy_learning.evaluate_machine_learning import sml_fitness_score, SML_SCORE
from happy_learning.sampler import MLSampler
from happy_learning.supervised_machine_learning import ModelGeneratorClf, ModelGeneratorReg
from happy_learning.utils import HappyLearningUtils, Log
from typing import List


class ArbitraryModeling:
    """
    Class for generating supervised machine learning models arbitrary
    """
    def __init__(self, sml_algorithm: str, n_models: int):
        """
        :param sml_algorithm: str
            Abbreviate name of the supervised machine learning algorithm

        :param n_models: int
            Number of models to generate
        """
        self.sml_algorithm: str = sml_algorithm
        self.n_models: int = n_models
        self.observations: dict = dict(sml_score=[], fitness=[], model_param=[])

    def run(self, df: pd.DataFrame, target: str, features: List[str], **kwargs):
        """
        Run arbitrary modeling

        :param df: pd.DataFrame
            Data set

        :param target: str
            Name of the target feature

        :param features: List[str]
            Names of the features used as predictors

        :param kwargs: dict
            Key-word arguments
        """
        if features is None:
            _features: List[str] = list(df.columns)
        else:
            _features: List[str] = features
        if target in _features:
            del _features[_features.index(target)]
        data_sets: dict = MLSampler(df=df,
                                    target=target,
                                    features=_features if features is None else features,
                                    train_size=0.8 if kwargs.get('train_size') is None else kwargs.get(
                                        'train_size'),
                                    stratification=False if kwargs.get(
                                        'stratification') is None else kwargs.get('stratification')
                                    ).train_test_sampling(
            validation_split=0.1 if kwargs.get('validation_split') is None else kwargs.get(
                'validation_split'))
        _sml_problem: str = HappyLearningUtils().get_ml_type(values=df[target].values)
        for i in range(0, self.n_models, 1):
            Log().log(msg=f'Model: {i}')
            if _sml_problem.find('clf') >= 0:
                _model = ModelGeneratorClf(model_name=self.sml_algorithm, models=[self.sml_algorithm]).generate_model()
            else:
                _model = ModelGeneratorReg(model_name=self.sml_algorithm, models=[self.sml_algorithm]).generate_model()
            _model.train(x=data_sets.get('x_train').values, y=data_sets.get('y_train').values)
            _pred_train: np.array = _model.predict(x=data_sets.get('x_train').values)
            _model.eval(obs=data_sets.get('y_train').values, pred=_pred_train, eval_metric=None, train_error=True)
            _pred_test: np.array = _model.predict(x=data_sets.get('x_test').values)
            _model.eval(obs=data_sets.get('y_test').values, pred=_pred_test, eval_metric=None, train_error=False)
            _ml_metric: str = SML_SCORE['ml_metric'][_sml_problem]
            _best_score: float = SML_SCORE['ml_metric_best'][_sml_problem]
            _score: float = sml_fitness_score(ml_metric=tuple([_best_score, _model.fitness['test'].get(_ml_metric)]),
                                              train_test_metric=tuple([_model.fitness['train'].get(_ml_metric),
                                                                       _model.fitness['test'].get(_ml_metric)]
                                                                      ),
                                              train_time_in_seconds=_model.train_time
                                              )
            self.observations['sml_score'].append(_score)
            self.observations['fitness'].append(_model.fitness)
            self.observations['model_param'].append(_model.model_param)
            Log().log(msg=f'Parameter: {_model.model_param}')
            Log().log(msg=f'SML Score: {_score}')
            Log().log(msg=f'Fitness: {_model.fitness}')
