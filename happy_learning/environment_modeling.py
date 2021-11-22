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
                                       model_param=[]
                                       )

    def _action_to_state(self, action: dict) -> dict:
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
                #elif isinstance(self.action_space['param_to_value'][model_name].get(_action), list):
                #    _state[model_name].update({_action: random.choice(seq=self.action_space['param_to_value'][model_name].get(_action))})
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

    def _state_to_tensor(self, model_name: str, state: dict) -> torch.tensor:
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
            _current_state: torch.tensor = torch.tensor(data=[[0.0] * self.n_actions], device=DEVICE)
            _state: dict = self._initial_state(model_name=_model_name)
        else:
            _model_name: str = list(action.keys())[0]
            _current_state: torch.tensor = self._state_to_tensor(model_name=_model_name,
                                                                 state=self.observations['state'][-1]
                                                                 )
            _state: dict = self._action_to_state(action=action)
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
        if len(self.observations['reward']) == 0:
            _reward_clipped: int = -1
        else:
            _reward_clipped: int = 1 if _score > self.observations['reward'][-1] else -1
        self.state = self._state_to_tensor(model_name=_model_name, state=_state)
        self.observations['action'].append(action)
        self.observations['state'].append(_state)
        self.observations['reward'].append(_score)
        self.observations['reward_clipped'].append(_reward_clipped)
        self.observations['fitness'].append(_model.fitness)
        self.observations['transition_gain'].append(_score - self.observations['reward'][-1])
        self.observations['model_param'].append(_model_param)
        return _current_state,\
               self.state,\
               torch.tensor([[_score]], device=DEVICE),\
               torch.tensor([[_reward_clipped]], device=DEVICE)

    def train_final_model(self, experience_id: int, data_sets: dict) -> object:
        """
        Train machine learning model based on best experienced model parameters

        :param experience_id: int
            Number of experienced state

        :param data_sets: dict
             Train, test and validation data set

        :return: object
            Trained model object
        """
        _model_name: str = self.action_space['model_name'][experience_id]
        if self.sml_problem.find('clf') >= 0:
            _model = ModelGeneratorClf(model_name=_model_name,
                                       clf_params=self.observations['model_param'][experience_id].get(_model_name),
                                       models=[_model_name]
                                       ).generate_model()
        else:
            _model = ModelGeneratorReg(model_name=_model_name,
                                       reg_params=self.observations['model_param'][experience_id].get(_model_name),
                                       models=[_model_name]
                                       ).generate_model()
        _model.train(x=data_sets.get('x_train').values, y=data_sets.get('y_train').values)
        return _model.model
