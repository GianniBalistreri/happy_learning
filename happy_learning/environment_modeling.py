"""

Reinforcement Learning Environment for Hyper-Parameter Optimization

"""

import numpy as np

from .evaluate_machine_learning import sml_fitness_score, SML_SCORE
from .supervised_machine_learning import (
    CLF_ALGORITHMS, ModelGeneratorClf, ModelGeneratorReg, REG_ALGORITHMS, PARAM_SPACE_CLF, PARAM_SPACE_REG
)


class EnvironmentModelingException(Exception):
    """
    Class for handling exception for class EnvironmentModeling
    """
    pass


class EnvironmentModeling:
    """
    Class for setting up reinforcement learning environment for modeling
    """
    def __init__(self, sml_problem: str, sml_algorithm: str = None):
        """
        :param sml_problem: str
            Abbreviate name of the supervised machine learning problem

        :param sml_algorithm: str
            Abbreviate name of the supervised machine learning algorithm
        """
        if sml_problem not in ['clf_binary', 'clf_multi', 'reg']:
            raise EnvironmentModelingException(f'Machine learning problem ({sml_problem}) not supported')
        self.action_space: dict = {}
        if sml_algorithm is None:
            if sml_problem.find('clf') >= 0:
                for algorithm in CLF_ALGORITHMS.keys():
                    self.action_space.update({algorithm: PARAM_SPACE_CLF.get(algorithm)})
            else:
                for algorithm in REG_ALGORITHMS.keys():
                    self.action_space.update({algorithm: PARAM_SPACE_REG.get(algorithm)})
        else:
            if sml_problem.find('clf') >= 0:
                if sml_algorithm in CLF_ALGORITHMS.keys():
                    self.action_space.update({sml_algorithm: PARAM_SPACE_CLF.get(sml_algorithm)})
                else:
                    raise EnvironmentModelingException(f'Classification algorithm ({sml_algorithm}) not supported')
            else:
                if sml_algorithm in REG_ALGORITHMS.keys():
                    self.action_space.update({sml_algorithm: PARAM_SPACE_REG.get(sml_algorithm)})
                else:
                    raise EnvironmentModelingException(f'Regression algorithm ({sml_algorithm}) not supported')
        self.sml_problem: str = sml_problem
        self.sml_algorithm: str = sml_algorithm
        self.n_steps: int = 0
        self.n_actions: int = 0
        for algorithm in self.action_space.keys():
            for param in self.action_space.get(algorithm):
                if isinstance(self.action_space[algorithm][param], list):
                    self.n_actions += len(self.action_space[algorithm][param])
                else:
                    self.n_actions += 1
        self.last_reward: float = 0.0
        self.profit: float = 0

    def reset(self):
        """
        Reset reinforcement learning environment settings
        """
        self.profit = 0

    def step(self, act: dict, data_sets: dict) -> tuple:
        """
        Reaction step of the environment to given action

        :param act: dict
            Hyper-parameter configuration

        :param data_sets: dict
            Train, test and validation data set

        :return: tuple
            State, reward (sml-score), clipped reward
        """
        self.n_steps += 1
        _model_name: str = list(act.keys())[0]
        if self.sml_problem.find('clf') >= 0:
            _model: ModelGeneratorClf = ModelGeneratorClf(model_name=_model_name,
                                                          clf_params=act.get(_model_name),
                                                          models=[_model_name]
                                                          )
        else:
            _model: ModelGeneratorReg = ModelGeneratorReg(model_name=_model_name,
                                                          reg_params=act.get(_model_name),
                                                          models=[_model_name]
                                                          )
        # TODO: change old state into new state using current action
        _state: dict = {}

        _model.train(x=data_sets.get('x_train').values, y=data_sets.get('y_train').values)
        _pred_train: np.array = _model.predict(x=data_sets.get('x_train').values, probability=False)
        _model.eval(obs=data_sets.get('x_train').values, pred=_pred_train, eval_metric=None, train_error=True)
        _pred_test: np.array = _model.predict(x=data_sets.get('x_test').values, probability=False)
        _model.eval(obs=data_sets.get('x_test').values, pred=_pred_test, eval_metric=None, train_error=False)
        _ml_metric: str = SML_SCORE['ml_metric'][self.sml_problem]
        _best_score: float = SML_SCORE['ml_metric_best'][self.sml_problem]
        _score: float = sml_fitness_score(ml_metric=tuple([_best_score, _model.fitness['test'].get(_ml_metric)]),
                                          train_test_metric=tuple([_model.fitness['train'].get(_ml_metric),
                                                                   _model.fitness['test'].get(_ml_metric)]
                                                                  ),
                                          train_time_in_seconds=_model.train_time
                                          )
        _reward_clipped: int = 1 if _score > self.last_reward else -1
        self.profit += _score - self.last_reward
        return _state, _score, _reward_clipped
