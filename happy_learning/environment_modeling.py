"""

Reinforcement Learning Environment for Hyper-Parameter Optimization of Supervised Machine Learning Algorithms

"""

import numpy as np
import random
import torch

from .evaluate_machine_learning import sml_fitness_score, SML_SCORE
from .supervised_machine_learning import (
    ModelGeneratorClf, ModelGeneratorReg, PARAM_SPACE_CLF, PARAM_SPACE_REG, Q_TABLE_PARAM_SPACE_CLF, Q_TABLE_PARAM_SPACE_REG
)
from typing import List

DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        self.action_space: dict = dict(param_to_value={},
                                       action=[],
                                       categorical_action={},
                                       ordinal_action={},
                                       continuous_action={},
                                       model_name=[],
                                       original_param={},
                                       categorical_param={}
                                       )
        if sml_algorithm is None:
            if sml_problem.find('clf') >= 0:
                for algorithm in Q_TABLE_PARAM_SPACE_CLF.keys():
                    self.action_space['param_to_value'].update({algorithm: Q_TABLE_PARAM_SPACE_CLF.get(algorithm)})
                    if algorithm in PARAM_SPACE_CLF.keys():
                        self.action_space['original_param'].update({algorithm: {}})
                        for param in PARAM_SPACE_CLF.get(algorithm).keys():
                            self.action_space['original_param'][algorithm].update({param: []})
                            for sub_param in Q_TABLE_PARAM_SPACE_CLF[sml_algorithm].keys():
                                if sub_param.find(param) >= 0:
                                    if isinstance(self.action_space['param_to_value'][algorithm][sub_param], list):
                                        for cat in self.action_space['param_to_value'][algorithm][sub_param]:
                                            self.action_space['categorical_param'].update({cat: param})
                                        self.action_space['original_param'][algorithm][param].extend(self.action_space['param_to_value'][algorithm][param])
                                    else:
                                        self.action_space['original_param'][algorithm][param].append(sub_param)
                    else:
                        raise EnvironmentModelingException(f'Classification algorithm ({algorithm}) not supported')
            else:
                for algorithm in Q_TABLE_PARAM_SPACE_REG.keys():
                    self.action_space['param_to_value'].update({algorithm: Q_TABLE_PARAM_SPACE_REG.get(algorithm)})
                    if algorithm in PARAM_SPACE_REG.keys():
                        self.action_space['original_param'].update({algorithm: {}})
                        for param in PARAM_SPACE_REG.get(algorithm).keys():
                            self.action_space['original_param'][algorithm].update({param: []})
                            for sub_param in Q_TABLE_PARAM_SPACE_REG[sml_algorithm].keys():
                                if sub_param.find(param) >= 0:
                                    if isinstance(self.action_space['param_to_value'][algorithm][sub_param], list):
                                        for cat in self.action_space['param_to_value'][algorithm][sub_param]:
                                            self.action_space['categorical_param'].update({cat: param})
                                        self.action_space['original_param'][algorithm][param].extend(self.action_space['param_to_value'][algorithm][param])
                                    else:
                                        self.action_space['original_param'][algorithm][param].append(sub_param)
                    else:
                        raise EnvironmentModelingException(f'Regression algorithm ({algorithm}) not supported')
        else:
            if sml_problem.find('clf') >= 0:
                if sml_algorithm in Q_TABLE_PARAM_SPACE_CLF.keys():
                    self.action_space['param_to_value'].update({sml_algorithm: Q_TABLE_PARAM_SPACE_CLF.get(sml_algorithm)})
                else:
                    raise EnvironmentModelingException(f'Classification algorithm ({sml_algorithm}) not supported')
                if sml_algorithm in PARAM_SPACE_CLF.keys():
                    self.action_space['original_param'].update({sml_algorithm: {}})
                    for param in PARAM_SPACE_CLF.get(sml_algorithm).keys():
                        self.action_space['original_param'][sml_algorithm].update({param: []})
                        for sub_param in Q_TABLE_PARAM_SPACE_CLF[sml_algorithm].keys():
                            if sub_param.find(param) >= 0:
                                if isinstance(self.action_space['param_to_value'][sml_algorithm][sub_param], list):
                                    for cat in self.action_space['param_to_value'][sml_algorithm][sub_param]:
                                        self.action_space['categorical_param'].update({cat: param})
                                    self.action_space['original_param'][sml_algorithm][param].extend(self.action_space['param_to_value'][sml_algorithm][param])
                                else:
                                    self.action_space['original_param'][sml_algorithm][param].append(sub_param)
                else:
                    raise EnvironmentModelingException(f'Classification algorithm ({sml_algorithm}) not supported')
            else:
                if sml_algorithm in Q_TABLE_PARAM_SPACE_REG.keys():
                    self.action_space['param_to_value'].update({sml_algorithm: Q_TABLE_PARAM_SPACE_REG.get(sml_algorithm)})
                else:
                    raise EnvironmentModelingException(f'Regression algorithm ({sml_algorithm}) not supported')
                if sml_algorithm in PARAM_SPACE_REG.keys():
                    self.action_space['original_param'].update({sml_algorithm: {}})
                    for param in PARAM_SPACE_REG.get(sml_algorithm).keys():
                        self.action_space['original_param'][sml_algorithm].update({param: []})
                        for sub_param in Q_TABLE_PARAM_SPACE_REG[sml_algorithm].keys():
                            if sub_param.find(param) >= 0:
                                if isinstance(self.action_space['param_to_value'][sml_algorithm][sub_param], list):
                                    for cat in self.action_space['param_to_value'][sml_algorithm][sub_param]:
                                        self.action_space['categorical_param'].update({cat: param})
                                    self.action_space['original_param'][sml_algorithm][param].extend(self.action_space['param_to_value'][sml_algorithm][param])
                                else:
                                    self.action_space['original_param'][sml_algorithm][param].append(sub_param)
                else:
                    raise EnvironmentModelingException(f'Regression algorithm ({sml_algorithm}) not supported')
        self.sml_problem: str = sml_problem
        self.sml_algorithm: str = sml_algorithm
        self.n_steps: int = 0
        self.n_actions: int = 0
        _n_params: int = 0
        for algorithm in self.action_space['param_to_value'].keys():
            for param in self.action_space['param_to_value'].get(algorithm):
                if isinstance(self.action_space['param_to_value'][algorithm][param], list):
                    for value in self.action_space['param_to_value'][algorithm][param]:
                        self.action_space['categorical_action'].update({_n_params: value})
                        self.action_space['action'].append(f'{param}_{value}')
                        _n_params += 1
                    self.action_space['model_name'].extend([algorithm] * len(self.action_space['param_to_value'][algorithm][param]))
                    self.n_actions += len(self.action_space['param_to_value'][algorithm][param])
                elif isinstance(self.action_space['param_to_value'][algorithm][param], tuple):
                    self.action_space['action'].append(param)
                    self.action_space['model_name'].append(algorithm)
                    self.n_actions += 1
                    if isinstance(self.action_space['param_to_value'][algorithm][param][0], float):
                        self.action_space['continuous_action'].update({_n_params: tuple([self.action_space['param_to_value'][algorithm][param][0],
                                                                                         self.action_space['param_to_value'][algorithm][param][1]
                                                                                         ]
                                                                                        )
                                                                       })
                    elif isinstance(self.action_space['param_to_value'][algorithm][param][0], int):
                        self.action_space['ordinal_action'].update({_n_params: tuple([self.action_space['param_to_value'][algorithm][param][0],
                                                                                      self.action_space['param_to_value'][algorithm][param][1]
                                                                                      ]
                                                                                     )
                                                                    })
                    _n_params += 1
                else:
                    raise EnvironmentModelingException(f'Type of action ({self.action_space["param_to_value"][algorithm][param]}) not supported')
        self.state: torch.tensor = None
        self.observations: dict = dict(action=[],
                                       state=[],
                                       reward=[],
                                       reward_clipped=[],
                                       fitness=[],
                                       transition_gain=[],
                                       model_param=[],
                                       model_name=[],
                                       sml_score=[],
                                       action_learning_type=[],
                                       action_value=[],
                                       episode=[],
                                       loss=[],
                                       policy_update=[],
                                       target_update=[]
                                       )

    def _initial_state(self, model_name: str) -> dict:
        """
        Initialize state

        :param model_name: str
            Abbreviated name of the model

        :return: dict
            Initial hyper-parameter configuration (state)
        """
        _state: dict = {model_name: {}}
        for param in self.action_space['original_param'][model_name].keys():
            _action: str = random.choice(seq=self.action_space['original_param'][model_name][param])
            if _action in self.action_space['categorical_param'].keys():
                _state[model_name].update({f'{param}_{_action}': 1.0})
                for action in self.action_space['categorical_param'].keys():
                    if action != _action and self.action_space['categorical_param'][action] == param:
                        _state[model_name].update({f'{param}_{action}': 0.0})
            else:
                if isinstance(self.action_space['param_to_value'][model_name].get(_action), tuple):
                    if isinstance(self.action_space['param_to_value'][model_name].get(_action)[0], float):
                        _state[model_name].update({_action: random.uniform(a=self.action_space['param_to_value'][model_name].get(_action)[0],
                                                                           b=self.action_space['param_to_value'][model_name].get(_action)[1]
                                                                           )
                                                   })
                    elif isinstance(self.action_space['param_to_value'][model_name].get(_action)[0], int):
                        _state[model_name].update({_action: random.randint(a=self.action_space['param_to_value'][model_name].get(_action)[0],
                                                                           b=self.action_space['param_to_value'][model_name].get(_action)[1]
                                                                           )
                                                   })
                    else:
                        raise EnvironmentModelingException(f'Type of action ({self.action_space["param_to_value"][model_name].get(_action)}) not supported')
                else:
                    raise EnvironmentModelingException(f'Type of action ({self.action_space["param_to_value"][model_name].get(_action)}) not supported')
                for action in self.action_space['original_param'][model_name][param]:
                    if action != _action:
                        _state[model_name].update({action: 0.0})
        self.observations['state'].append(_state)
        return _state

    def _state_to_param(self, model_name: str, state: dict) -> dict:
        """
        Convert state (hyper-parameter configuration compatible for q-table) to hyper-parameter configuration compatible for modeling

        :param model_name: str
            Abbreviate name of the model

        :param state: dict
            Hyper-parameter configuration compatible for q-table (state)

        :return: dict
            Hyper-parameter configuration compatible for modeling
        """
        _model_param: dict = {model_name: {}}
        for param in self.action_space['original_param'][model_name].keys():
            for sub_param in self.action_space['original_param'][model_name][param]:
                if sub_param in self.action_space['categorical_param'].keys():
                    if state[model_name][f'{param}_{sub_param}'] != 0.0:
                        _model_param[model_name].update({param: sub_param})
                        break
                else:
                    if state[model_name][sub_param] != 0.0:
                        _model_param[model_name].update({param: state[model_name][sub_param]})
                        break
        return _model_param

    def action_to_state(self, action: dict) -> dict:
        """
        Convert action to state

        :param action: dict
            Selected action by agent

        :return: dict
            New hyper-parameter configuration (state)
        """
        _model_name: str = list(action.keys())[0]
        _idx: int = action[_model_name]['idx']
        _model_param: str = list(action[_model_name].keys())[0]
        _current_state: dict = self.observations['state'][-1]
        _current_state[_model_name].update({_model_param: action[_model_name][_model_param]})
        _same_param_other_value: List[str] = []
        for original_param in self.action_space['original_param'][_model_name].keys():
            if _model_param in self.action_space['original_param'][_model_name][original_param]:
                _same_param_other_value.extend(self.action_space['original_param'][_model_name][original_param])
                break
        for other_value in _same_param_other_value:
            if other_value != _model_param:
                _current_state[_model_name].update({other_value: 0.0})
        return _current_state

    def param_to_state(self, model_name: str, param: dict) -> dict:
        """
        Convert hyper-parameter configuration to state

        :param model_name: str
            Abbreviated name of the supervised machine learning model

        :param param: dict
            Hyper-parameter configuration

        :return: dict
            State
        """
        _state: dict = {model_name: {}}
        for hyper_param in param.keys():
            for sub_param in self.action_space['original_param'][model_name][hyper_param]:
                if sub_param in self.action_space['categorical_param'].keys():
                    if sub_param == param[hyper_param]:
                        _state[model_name].update({f'{hyper_param}_{sub_param}': 1.0})
                    else:
                        _state[model_name].update({f'{hyper_param}_{sub_param}': 0.0})
                else:
                    if self.action_space['param_to_value'][model_name][sub_param][1] >= param[hyper_param] >= self.action_space['param_to_value'][model_name][sub_param][0]:
                        _state[model_name].update({sub_param: param[hyper_param]})
                    else:
                        _state[model_name].update({sub_param: 0.0})
        return _state

    def state_to_tensor(self, model_name: str, state: dict) -> torch.tensor:
        """
        Convert state to tensor

        :param model_name: str
            Abbreviate name of the model

        :param state: dict
            Hyper-parameter configuration compatible for q-table (state)

        :return: torch.tensor
            Tensor representing new state
        """
        _new_state: List[float] = []
        for action in self.action_space['param_to_value'][model_name].keys():
            if isinstance(self.action_space['param_to_value'][model_name].get(action), tuple):
                if state[model_name][action] >= 1:
                    _new_state.append(state[model_name][action] / 1000)
                else:
                    _new_state.append(state[model_name][action])
            elif isinstance(self.action_space['param_to_value'][model_name].get(action), list):
                for cat in self.action_space['param_to_value'][model_name].get(action):
                    _new_state.append(state[model_name][f'{self.action_space["categorical_param"][cat]}_{cat}'])
            else:
                raise EnvironmentModelingException(
                    f'Type of action ({self.action_space["param_to_value"][model_name].get(action)}) not supported')
        return torch.tensor([_new_state], device=DEVICE)

    def step(self, action: dict, data_sets: dict) -> tuple:
        """
        Reaction step of the environment to given action

        :param action: dict
            Hyper-parameter configuration

        :param data_sets: dict
            Train, test and validation data set

        :return: tuple
            State, reward (sml-score), clipped reward
        """
        self.n_steps += 1
        if action is None:
            _model_name: str = random.choice(seq=self.action_space['model_name'])
            _action: str = '*'
            _current_state: torch.tensor = torch.tensor(data=[[0.0] * self.n_actions], device=DEVICE)
            _state: dict = self._initial_state(model_name=_model_name)
        else:
            _model_name: str = list(action.keys())[0]
            _action: str = list(action.get(_model_name).keys())[0]
            _current_state: torch.tensor = self.state_to_tensor(model_name=_model_name,
                                                                state=self.observations['state'][-1]
                                                                )
            _state: dict = self.action_to_state(action=action)
        _model_param: dict = self._state_to_param(model_name=_model_name, state=_state)
        if self.sml_problem.find('clf') >= 0:
            _model = ModelGeneratorClf(model_name=_model_name,
                                       clf_params=_model_param.get(_model_name),
                                       models=[_model_name]
                                       ).generate_model()
        else:
            _model = ModelGeneratorReg(model_name=_model_name,
                                       reg_params=_model_param.get(_model_name),
                                       models=[_model_name]
                                       ).generate_model()
        _model.train(x=data_sets.get('x_train').values, y=data_sets.get('y_train').values)
        _pred_train: np.array = _model.predict(x=data_sets.get('x_train').values)
        _model.eval(obs=data_sets.get('y_train').values, pred=_pred_train, eval_metric=None, train_error=True)
        _pred_test: np.array = _model.predict(x=data_sets.get('x_test').values)
        _model.eval(obs=data_sets.get('y_test').values, pred=_pred_test, eval_metric=None, train_error=False)
        _ml_metric: str = SML_SCORE['ml_metric'][self.sml_problem]
        _best_score: float = SML_SCORE['ml_metric_best'][self.sml_problem]
        _score: float = sml_fitness_score(ml_metric=tuple([_best_score, _model.fitness['test'].get(_ml_metric)]),
                                          train_test_metric=tuple([_model.fitness['train'].get(_ml_metric),
                                                                   _model.fitness['test'].get(_ml_metric)]
                                                                  ),
                                          train_time_in_seconds=_model.train_time
                                          )
        if len(self.observations['reward']) <= 0:
            _reward_clipped: int = -1
        else:
            _reward_clipped: int = 1 if _score > self.observations['sml_score'][-1] else -1
        self.state = self.state_to_tensor(model_name=_model_name, state=_state)
        if self.n_steps == 1:
            self.observations['transition_gain'].append(0)
        else:
            self.observations['transition_gain'].append(_score - self.observations['sml_score'][-1])
        if _score == 0.00001:
            _reward: float = -1
        else:
            if self.observations['transition_gain'][-1] == 0:
                _reward: float = -1
            else:
                _reward: float = self.observations['transition_gain'][-1] * 0.01
        self.observations['action'].append(_action)
        self.observations['action_value'].append(action[_model_name][_action] if _action != '*' else 0)
        self.observations['state'].append(_state)
        self.observations['reward'].append(_reward)
        self.observations['reward_clipped'].append(_reward_clipped)
        self.observations['fitness'].append(_model.fitness)
        self.observations['model_param'].append(_model_param)
        self.observations['model_name'].append(_model_name)
        self.observations['sml_score'].append(_score)
        _transition_elements: tuple = tuple([_current_state,
                                             self.state,
                                             torch.tensor([[_reward]], device=DEVICE),
                                             torch.tensor([[_reward_clipped]], device=DEVICE)
                                             ])
        return _transition_elements

    def train_final_model(self,
                          data_sets: dict,
                          model_name: str,
                          param: dict,
                          eval: bool = True
                          ) -> object:
        """
        Train machine learning model based on best experienced model parameters

        :param data_sets: dict
             Train, test and validation data set

        :param model_name: str
            Abbreviated name of the supervised machine learning model

        :param param: dict
            Hyper-parameter configuration

        :param eval: bool
            Evaluate trained supervised machine learning model

        :return: object
            Trained model object
        """
        if self.sml_problem.find('clf') >= 0:
            _model = ModelGeneratorClf(model_name=model_name,
                                       clf_params=param,
                                       models=[model_name]
                                       ).generate_model()
        else:
            _model = ModelGeneratorReg(model_name=model_name,
                                       reg_params=param,
                                       models=[model_name]
                                       ).generate_model()
        _model.train(x=data_sets.get('x_train').values, y=data_sets.get('y_train').values)
        if eval:
            _pred_train: np.array = _model.predict(x=data_sets.get('x_train').values)
            _model.eval(obs=data_sets.get('y_train').values, pred=_pred_train, eval_metric=None, train_error=True)
            _pred_test: np.array = _model.predict(x=data_sets.get('x_test').values)
            _model.eval(obs=data_sets.get('y_test').values, pred=_pred_test, eval_metric=None, train_error=False)
            _ml_metric: str = SML_SCORE['ml_metric'][self.sml_problem]
            _best_score: float = SML_SCORE['ml_metric_best'][self.sml_problem]
            _score: float = sml_fitness_score(ml_metric=tuple([_best_score, _model.fitness['test'].get(_ml_metric)]),
                                              train_test_metric=tuple([_model.fitness['train'].get(_ml_metric),
                                                                       _model.fitness['test'].get(_ml_metric)]
                                                                      ),
                                              train_time_in_seconds=_model.train_time
                                              )
            _model.fitness_score = _score
        return _model
