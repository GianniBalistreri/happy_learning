"""

Train Reinforcement Learning Algorithm: Deep-Q-Learning Network

"""

import numpy as np
import math
import os
import pandas as pd
import random
import torch

from .environment_modeling import EnvironmentModeling
from .neural_network_torch import DQNFC
from .sampler import MLSampler
from .utils import HappyLearningUtils
from collections import deque
from datetime import datetime
from easyexplore.data_import_export import CLOUD_PROVIDER, DataExporter, DataImporter
from easyexplore.data_visualizer import DataVisualizer
from easyexplore.utils import Log
from typing import List, NamedTuple

DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class Transition(NamedTuple):
    """
    Class for defining transition parameter
    """
    state: torch.tensor
    action: torch.tensor
    next_state: torch.tensor
    reward: torch.tensor


class TransitionMemory:
    """
    Class for storing transitions that the agent observes
    """
    def __init__(self, capacity: int):
        """
        :param capacity: int
            Maximum length of memory (memory capacity)
        """
        self.capacity: int = capacity
        self.memory: deque = deque([], maxlen=capacity)

    def push(self, *args):
        """
        Save transition

        :param args: dict
            Transition arguments
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        """
        Memory sample

        :param batch_size: int
            Batch size

        :return: xxx
            Sample
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """
        Get length of memory

        :return: int
            Length of memory
        """
        return len(self.memory)


class DQNAgentException(Exception):
    """
    Class for handling exceptions for class DQNAgent
    """
    pass


class DQNAgent:
    """
    Class for training deep-q-learning network
    """
    def __init__(self,
                 episodes: int = 50,
                 batch_size: int = 128,
                 gamma: float = 0.999,
                 eps_start: float = 0.9,
                 eps_end: float = 0.5,
                 eps_decay: int = 200,
                 target_update: int = 10,
                 optimizer: str = 'rmsprop',
                 hidden_layer_size: int = 100,
                 use_clipped_reward: bool = False,
                 timer_in_seconds: int = 43200,
                 force_target_type: str = None,
                 plot: bool = False,
                 output_file_path: str = None,
                 deploy_model: bool = True,
                 cloud: str = None,
                 log: bool = False,
                 verbose: bool = False,
                 checkpoint: bool = True,
                 checkpoint_episode_interval: int = 5,
                 **kwargs
                 ):
        """
        :param batch_size: int
            Batch size

        :param episodes: int
            Number of episodes to train

        :param gamma: float
            Gamma value for calculating expected Q values

        :param eps_start: float
            Start value of the epsilon-greedy algorithm

        :param eps_end: float
            End value of the epsilon-greedy algorithm

        :param eps_decay: float
            Decay value of the epsilon-greedy algorithm each step

        :param target_update: int
            Interval for updating target net

        :param optimizer: str
            Abbreviated name of the optimizer
                -> rmsprop: RMSprop
                -> adam: Adam
                -> sgd: Stochastic Gradient Descent

        :param hidden_layer_size: int
            Number of neurons of the fully connected hidden layer (policy and target network)

        :param use_clipped_reward: bool
            Whether to use clipped reward (categorical) or metric reward

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

        :param cloud: str
            Name of the cloud provider
                -> google: Google Cloud Storage
                -> aws: AWS Cloud

        :param deploy_model: bool
            Whether to deploy (save) evolved model or not

        :param log: bool
            Write logging file or just print messages

        :param verbose: bool
            Log all processes (extended logging)

        :param checkpoint: bool
            Save checkpoint after each adjustment

        :parm checkpoint_episode_interval: int
            Episode interval for saving checkpoint

        :param kwargs: dict
            Key-word arguments
        """
        self.env = None
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.memory = None
        self.episodes: int = episodes
        self.batch_size: int = batch_size
        self.gamma: float = gamma
        self.eps_start: float = eps_start
        self.eps_end: float = eps_end
        self.eps_decay: float = eps_decay
        self.target_update: int = target_update
        self.hidden_layer_size: int = hidden_layer_size if hidden_layer_size > 1 else 100
        self.use_clipped_reward: bool = use_clipped_reward
        self.deploy_model: bool = deploy_model
        self.n_training: int = 0
        self.n_optimization: int = 0
        self.n_update: int = 0
        if optimizer in ['rmsprop', 'adam', 'sgd']:
            self.optimizer_name: str = optimizer
        else:
            self.optimizer_name: str = 'rmsprop'
        self.cloud: str = cloud
        if self.cloud is None:
            self.bucket_name: str = None
        else:
            if self.cloud not in CLOUD_PROVIDER:
                raise DQNAgentException('Cloud provider ({}) not supported'.format(cloud))
            if output_file_path is None:
                raise DQNAgentException('Output file path is None')
            self.bucket_name: str = output_file_path.split("//")[1].split("/")[0]
        if output_file_path is None:
            self.output_file_path: str = ''
        else:
            self.output_file_path: str = output_file_path.replace('\\', '/')
            if self.output_file_path[len(self.output_file_path) - 1] != '/':
                self.output_file_path = '{}/'.format(self.output_file_path)
        self.force_target_type: str = force_target_type
        self.n_cases: int = 0
        self.experience_idx: int = 0
        self.plot: bool = plot
        self.log: bool = log
        self.verbose: bool = verbose
        self.checkpoint: bool = checkpoint
        self.checkpoint_episode_interval: int = checkpoint_episode_interval if checkpoint_episode_interval > 0 else 5
        self.timer: int = timer_in_seconds if timer_in_seconds > 0 else 99999
        self.start_time: datetime = datetime.now()
        self.kwargs: dict = kwargs

    def _save_checkpoint(self):
        """
        Save checkpoint of the agent
        """
        self.save(agent=True,
                  model=False,
                  data_sets={},
                  experience=False
                  )

    def _select_action(self, state: torch.tensor) -> dict:
        """
        Select action by the agent

        :param state: torch.tensor
            Tensor representing current state

        :return: dict
            New action
        """
        _eps_threshold: float = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * self.env.n_steps / self.eps_decay)
        _sample: float = random.random()
        if self.verbose:
            Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'EPS threshold: {_eps_threshold}')
        _model_name: str = random.choice(seq=list(self.env.action_space['param_to_value'].keys()))
        _action_names: List[str] = self.env.action_space['action']
        if _sample > _eps_threshold:
            self.env.observations['action_learning_type'].append('exploitation')
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                _new_action_exploitation: torch.tensor = self.policy_net(state).max(1)[1].view(1, 1)
                _idx: int = _new_action_exploitation.tolist()[0][0]
                if self.verbose:
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Exploitation action: {_idx}')
                if _idx in list(self.env.action_space['categorical_action'].keys()):
                    _action_name: str = _action_names[_idx].replace(f'_{self.env.action_space["categorical_action"][_idx]}', '')
                    return {_model_name: {_action_name: self.env.action_space['categorical_action'][_idx],
                                          'idx': _idx
                                          }
                            }
                elif _idx in list(self.env.action_space['ordinal_action'].keys()):
                    return {_model_name: {_action_names[_idx]: random.randint(a=self.env.action_space['ordinal_action'][_idx][0],
                                                                              b=self.env.action_space['ordinal_action'][_idx][1]
                                                                              ),
                                          'idx': _idx
                                          }
                            }
                elif _idx in list(self.env.action_space['continuous_action'].keys()):
                    return {_model_name: {_action_names[_idx]: random.uniform(a=self.env.action_space['continuous_action'][_idx][0],
                                                                              b=self.env.action_space['continuous_action'][_idx][1]
                                                                              ),
                                          'idx': _idx
                                          }
                            }
                else:
                    raise DQNAgentException(f'Type of action ({_action_names[_idx]}) not supported')
        else:
            self.env.observations['action_learning_type'].append('exploration')
            _new_state: List[float] = []
            _idx: int = random.randint(a=0, b=len(_action_names) - 1)
            if self.verbose:
                Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Exploration action: {_idx}')
            if _idx in list(self.env.action_space['categorical_action'].keys()):
                _action_name: str = _action_names[_idx].replace(f'_{self.env.action_space["categorical_action"][_idx]}',
                                                                '')
                return {_model_name: {_action_name: self.env.action_space['categorical_action'][_idx],
                                      'idx': _idx
                                      }
                        }
            elif _idx in list(self.env.action_space['ordinal_action'].keys()):
                return {_model_name: {
                    _action_names[_idx]: random.randint(a=self.env.action_space['ordinal_action'][_idx][0],
                                                        b=self.env.action_space['ordinal_action'][_idx][1]
                                                        ),
                    'idx': _idx
                    }
                        }
            elif _idx in list(self.env.action_space['continuous_action'].keys()):
                return {_model_name: {
                    _action_names[_idx]: random.uniform(a=self.env.action_space['continuous_action'][_idx][0],
                                                        b=self.env.action_space['continuous_action'][_idx][1]
                                                        ),
                    'idx': _idx
                    }
                        }
            else:
                raise DQNAgentException(f'Type of action ({_action_names[_idx]}) not supported')

    def _train(self, data_sets: dict):
        """
        Train deep-q-learning network

        :param data_sets: dict
            Train, test and validation data set
        """
        _early_stopping: bool = False
        for episode in range(0, self.episodes, 1):
            if len(self.env.observations['episode']) == 0:
                _episode: int = episode
            else:
                _episode: int = self.env.observations['episode'][-1] + 1
            Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Episode: {_episode}')
            for _ in range(0, self.env.n_actions, 1):
                if (datetime.now() - self.start_time).seconds >= self.timer:
                    _early_stopping = False
                    Log(write=self.log, logger_file_path=self.output_file_path).log('Time exceeded:{}'.format(self.timer))
                    break
                self.env.observations['episode'].append(_episode)
                if self.verbose:
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Step: {self.env.n_steps}')
                # Select and perform an action
                if self.env.n_steps == 0:
                    _action: dict = None
                    self.env.observations['action_learning_type'].append('exploration')
                else:
                    _action: dict = self._select_action(state=self.env.state)
                    if self.verbose:
                        Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'New action: {_action}')
                # Observe new state and reward
                _state, _next_state, _reward, _clipped_reward = self.env.step(action=_action, data_sets=data_sets)
                if self.verbose:
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Current state: {self.env.observations["state"][-2]}')
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'New state: {self.env.observations["state"][-1]}')
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Reward: {self.env.observations["reward"][-1]}')
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Reward (clipped): {self.env.observations["reward_clipped"][-1]}')
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Transition gain: {self.env.observations["transition_gain"][-1]}')
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Fitness: {self.env.observations["fitness"][-1]}')
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Score: {self.env.observations["sml_score"][-1]}')
                # Store the transition in memory
                if _action is not None:
                    _model_name: str = list(_action.keys())[0]
                    self.memory.push(_state,
                                     torch.tensor([[_action[_model_name]['idx']]]),
                                     _next_state,
                                     _clipped_reward if self.use_clipped_reward else _reward
                                     )
                # Move to the next state
                _state = _next_state
                # Perform one step of the optimization (on the policy network)
                if self.verbose:
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Memory length: {self.memory.__len__()}')
                if self.memory.__len__() >= self.batch_size:
                    self.n_optimization += 1
                    self.env.observations['policy_update'].append(self.n_optimization)
                    self.env.observations['target_update'].append(self.n_update)
                    if self.verbose:
                        Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Optimization step of policy network: {self.n_optimization}')
                    transitions = self.memory.sample(self.batch_size)
                    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                    # detailed explanation). This converts batch-array of Transitions
                    # to Transition of batch-arrays.
                    batch = Transition(*zip(*transitions))
                    # Compute a mask of non-final states and concatenate the batch elements
                    # (a final state would've been the one after which simulation ended)
                    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                            batch.next_state)), device=DEVICE, dtype=torch.bool)
                    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
                    state_batch = torch.cat(batch.state)
                    action_batch = torch.cat(batch.action)
                    reward_batch = torch.cat(batch.reward)
                    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                    # columns of actions taken. These are the actions which would've been taken
                    # for each batch state according to policy_net
                    state_action_values = self.policy_net(state_batch).gather(1, action_batch)
                    # Compute V(s_{t+1}) for all next states.
                    # Expected values of actions for non_final_next_states are computed based
                    # on the "older" target_net; selecting their best reward with max(1)[0].
                    # This is merged based on the mask, such that we'll have either the expected
                    # state value or 0 in case the state was final.
                    next_state_values = torch.zeros(self.batch_size, device=DEVICE)
                    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
                    # Compute the expected Q values
                    expected_state_action_values = (next_state_values * self.gamma) + reward_batch
                    # Compute Huber loss
                    criterion = torch.nn.SmoothL1Loss()
                    loss = criterion(state_action_values.to(dtype=torch.float64),
                                     expected_state_action_values.unsqueeze(1).resize_((self.batch_size, 1)).to(dtype=torch.float64)
                                     )
                    self.env.observations['loss'].append(loss.detach().numpy().tolist())
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Loss: {loss}')
                    # Optimize the model
                    self.optimizer.zero_grad()
                    loss.backward()
                    for param in self.policy_net.parameters():
                        param.grad.data.clamp_(-1, 1)
                    self.optimizer.step()
                else:
                    self.env.observations['loss'].append(0)
                    self.env.observations['target_update'].append(self.n_update)
                    self.env.observations['policy_update'].append(self.n_optimization)
            if _early_stopping:
                break
            # Update the target network, copying all weights and biases in DQN
            if episode % self.target_update == 0:
                self.n_update += 1
                if self.verbose:
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Update step of target network: {self.n_update}')
                self.target_net.load_state_dict(self.policy_net.state_dict())
            if self.checkpoint and _episode > 0 and _episode % self.checkpoint_episode_interval == 0:
                self._save_checkpoint()

    def apply_learning(self,
                       model_name: str,
                       param: dict,
                       file_path_policy_network: str = None,
                       file_path_environment: str = None,
                       file_path_model: str = None,
                       data_sets: dict = None,
                       max_actions: int = 1,
                       force_all_actions: bool = False,
                       **kwargs
                       ):
        """
        Apply learning to generate optimized hyper-parameter configuration

        :param model_name: str
            Abbreviate name of the supervised machine learning model

        :param param: dict
            Initial hyper-parameter configuration (state)

        :param file_path_policy_network: str
            Complete file path of the saved policy network

        :param file_path_environment: str
            Complete file path of the saved environment

        :param file_path_model: str
            Complete file path to save optimized model

        :param data_sets: dict
            Train, test and validation data sets

        :param max_actions: int
            Maximum number of actions to apply

        :param force_all_actions: bool
            Whether to force all number of action and choose best hyper-parameter configuration

        :param kwargs: dict
            Key-word arguments for saving model

        :return: dict
            Optimized hyper-parameter configuration
        """
        if file_path_environment is not None:
            self.env = DataImporter(file_path=file_path_environment,
                                    as_data_frame=False,
                                    use_dask=False,
                                    create_dir=False,
                                    sep=',',
                                    cloud=self.cloud,
                                    bucket_name=self.bucket_name,
                                    region=self.kwargs.get('region')
                                    ).file()
        if file_path_policy_network is not None:
            self.policy_net = DataImporter(file_path=file_path_policy_network,
                                           as_data_frame=False,
                                           use_dask=False,
                                           create_dir=False,
                                           sep=',',
                                           cloud=self.cloud,
                                           bucket_name=self.bucket_name,
                                           region=self.kwargs.get('region')
                                           ).file()
        _current_state: dict = {}
        _max_actions: int = max_actions if max_actions > 0 else 1
        if data_sets is not None:
            _model: object = self.env.train_final_model(data_sets=data_sets,
                                                        model_name=model_name,
                                                        param=param,
                                                        eval=True
                                                        )
        _param: dict = param
        _observations: dict = dict(model=[], sml_score=[])
        _model_idx: int = 0
        for i in range(0, _max_actions, 1):
            if self.verbose:
                Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Current state: {_param}')
            _state: torch.tensor = self.env.state_to_tensor(model_name=model_name,
                                                            state=self.env.param_to_state(model_name=model_name,
                                                                                          param=_param
                                                                                          )
                                                            )
            with torch.no_grad():
                _new_action_exploitation: torch.tensor = self.policy_net(_state).max(1)[1].view(1, 1)
                _idx: int = _new_action_exploitation.tolist()[0][0]
                _action_names: List[str] = self.env.action_space['action']
                if _idx in list(self.env.action_space['categorical_action'].keys()):
                    _action_name: str = _action_names[_idx].replace(f'_{self.env.action_space["categorical_action"][_idx]}',
                                                                    '')
                    _action: dict = {model_name: {_action_name: self.env.action_space['categorical_action'][_idx],
                                                  'idx': _idx
                                                  }
                                     }
                elif _idx in list(self.env.action_space['ordinal_action'].keys()):
                    _action: dict = {model_name: {
                        _action_names[_idx]: random.randint(a=self.env.action_space['ordinal_action'][_idx][0],
                                                            b=self.env.action_space['ordinal_action'][_idx][1]
                                                            ),
                        'idx': _idx
                        }
                            }
                elif _idx in list(self.env.action_space['continuous_action'].keys()):
                    _action: dict = {model_name: {
                        _action_names[_idx]: random.uniform(a=self.env.action_space['continuous_action'][_idx][0],
                                                            b=self.env.action_space['continuous_action'][_idx][1]
                                                            ),
                        'idx': _idx
                        }
                            }
                else:
                    raise DQNAgentException(f'Type of action ({_action_names[_idx]}) not supported')
                if self.verbose:
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Exploited action: {_action_names[_idx]} - {_action[model_name][_action_name]}')
            _current_state: dict = _param
            _model_param: str = list(_action[model_name].keys())[0]
            _same_param_other_value: List[str] = []
            for original_param in self.env.action_space['original_param'][model_name].keys():
                if _model_param in self.env.action_space['original_param'][model_name][original_param]:
                    _current_state.update({original_param: _action[model_name][_model_param]})
                    break
            if data_sets is None:
                break
            else:
                _new_model: object = self.env.train_final_model(data_sets=data_sets,
                                                                model_name=model_name,
                                                                param=_current_state,
                                                                eval=True
                                                                )
                _observations['model'].append(_new_model)
                _observations['sml_score'].append(_new_model.fitness_score)
                if self.verbose:
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Current Model Fitness: {_model.fitness_score} | New Model Fitness:{_new_model.fitness_score}')
                if _model.fitness_score < _new_model.fitness_score:
                    if force_all_actions:
                        if i + 1 == max_actions:
                            _best_model_idx: int = np.array(_observations['sml_score']).argmax()
                            _best_model: object = _observations['model'][_best_model_idx]
                            _current_state = _best_model.model_param
                            if file_path_model is not None:
                                DataExporter(obj=_best_model.model,
                                             file_path=file_path_model,
                                             create_dir=False,
                                             overwrite=False if kwargs.get('overwrite') is None else kwargs.get('overwrite'),
                                             cloud=kwargs.get('cloud'),
                                             bucket_name=kwargs.get('bucket_name'),
                                             region=kwargs.get('region')
                                             ).file()
                    else:
                        if file_path_model is not None:
                            DataExporter(obj=_new_model.model,
                                         file_path=file_path_model,
                                         create_dir=False,
                                         overwrite=False if kwargs.get('overwrite') is None else kwargs.get('overwrite'),
                                         cloud=kwargs.get('cloud'),
                                         bucket_name=kwargs.get('bucket_name'),
                                         region=kwargs.get('region')
                                         ).file()
                        break
                else:
                    _param = _current_state
        return _current_state

    def optimize(self,
                 df: pd.DataFrame,
                 target: str,
                 features: List[str] = None,
                 memory_capacity: int = 10000,
                 ):
        """
        Optimize hyper-parameter configuration

        :param df: pd.DataFrame
            Data set

        :param target: str
            Name of the target feature

        :param features: List[str]
            Names of the features used as predictors

        :param memory_capacity: int
            Maximum capacity of the agent memory
        """
        if features is None:
            _features: List[str] = list(df.columns)
        else:
            _features: List[str] = features
        if target in _features:
            del _features[_features.index(target)]
        _n_features: int = len(_features)
        data_sets: dict = MLSampler(df=df,
                                    target=target,
                                    features=_features if features is None else features,
                                    train_size=0.8 if self.kwargs.get('train_size') is None else self.kwargs.get('train_size'),
                                    stratification=False if self.kwargs.get('stratification') is None else self.kwargs.get('stratification')
                                    ).train_test_sampling(validation_split=0.1 if self.kwargs.get('validation_split') is None else self.kwargs.get('validation_split'))
        self.env: EnvironmentModeling = EnvironmentModeling(sml_problem=HappyLearningUtils().get_ml_type(values=df[target].values),
                                                            sml_algorithm=self.kwargs.get('sml_algorithm')
                                                            )
        self.policy_net: DQNFC = DQNFC(input_size=self.env.n_actions,
                                       hidden_size=self.hidden_layer_size,
                                       output_size=self.env.n_actions
                                       ).to(DEVICE)
        self.target_net: DQNFC = DQNFC(input_size=self.env.n_actions,
                                       hidden_size=self.hidden_layer_size,
                                       output_size=self.env.n_actions
                                       ).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        if self.optimizer_name == 'rmsprop':
            self.optimizer: torch.optim = torch.optim.RMSprop(self.policy_net.parameters())
        elif self.optimizer_name == 'adam':
            self.optimizer: torch.optim = torch.optim.Adam(self.policy_net.parameters())
        elif self.optimizer_name == 'sgd':
            self.optimizer: torch.optim = torch.optim.SGD(self.policy_net.parameters())
        _memory_capacity: int = memory_capacity if memory_capacity >= 10 else 10000
        self.memory: TransitionMemory = TransitionMemory(capacity=_memory_capacity)
        self.n_update = len(self.env.observations['target_update'])
        self.n_optimization = len(self.env.observations['policy_update'])
        self._train(data_sets=data_sets)
        self.experience_idx = np.array(self.env.observations['sml_score']).argmax()
        _score: float = self.env.observations['sml_score'][self.experience_idx]
        _state: dict = self.env.observations['state'][self.experience_idx]
        _model_param: dict = self.env.observations['model_param'][self.experience_idx]
        _fitness: dict = self.env.observations['fitness'][self.experience_idx]
        Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Best model state: {_state}')
        Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Best model param: {_model_param}')
        Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Best model fitness: {_fitness}')
        Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Best model score: {_score}')
        self.save(agent=True if self.kwargs.get('save_agent') is None else self.kwargs.get('save_agent'),
                  model=self.deploy_model,
                  data_sets=data_sets,
                  experience=False if self.kwargs.get('save_experience') is None else self.kwargs.get('save_experience')
                  )
        if self.plot:
            self.visualize()

    def optimize_continue(self,
                          df: pd.DataFrame,
                          target: str,
                          features: List[str] = None,
                          ):
        """
        Continue hyper-parameter optimization

        :param df: pd.DataFrame
            Data set

        :param target: str
            Name of the target feature

        :param features: List[str]
            Names of the features used as predictors
        """
        if features is None:
            _features: List[str] = list(df.columns)
        else:
            _features: List[str] = features
        if target in _features:
            del _features[_features.index(target)]
        _n_features: int = len(_features)
        data_sets: dict = MLSampler(df=df,
                                    target=target,
                                    features=_features if features is None else features,
                                    train_size=0.8 if self.kwargs.get('train_size') is None else self.kwargs.get('train_size'),
                                    stratification=False if self.kwargs.get('stratification') is None else self.kwargs.get('stratification')
                                    ).train_test_sampling(validation_split=0.1 if self.kwargs.get('validation_split') is None else self.kwargs.get('validation_split'))
        self._train(data_sets=data_sets)
        self.experience_idx = np.array(self.env.observations['sml_score']).argmax()
        _score: float = self.env.observations['sml_score'][self.experience_idx]
        _state: dict = self.env.observations['state'][self.experience_idx]
        _model_param: dict = self.env.observations['model_param'][self.experience_idx]
        _fitness: dict = self.env.observations['fitness'][self.experience_idx]
        Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Best model state: {_state}')
        Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Best model param: {_model_param}')
        Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Best model fitness: {_fitness}')
        Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Best model score: {_score}')
        self.save(agent=True if self.kwargs.get('save_agent') is None else self.kwargs.get('save_agent'),
                  model=self.deploy_model,
                  data_sets=data_sets,
                  experience=False if self.kwargs.get('save_experience') is None else self.kwargs.get('save_experience')
                  )
        if self.plot:
            self.visualize()

    def save(self, agent: bool, model: bool, data_sets: dict, experience: bool):
        """
        Save learnings of the agent

        :param agent: bool
            Save necessary parts of the agent (policy network, target network, environment)

        :param model: bool
            Save best observed model

        :param data_sets: dict
            Train, test and validation data sets

        :param experience: bool
            Save experiences of the agent
        """
        if agent:
            # Save agent class DQNAgent itself:
            DataExporter(obj=self,
                         file_path=os.path.join(self.output_file_path, 'rl_agent.p'),
                         create_dir=False,
                         overwrite=True,
                         cloud=self.cloud,
                         bucket_name=self.bucket_name,
                         region=self.kwargs.get('region')
                         ).file()
            # Save reinforcement learning environment:
            DataExporter(obj=self.env,
                         file_path=os.path.join(self.output_file_path, 'rl_env.p'),
                         create_dir=False,
                         overwrite=True,
                         cloud=self.cloud,
                         bucket_name=self.bucket_name,
                         region=self.kwargs.get('region')
                         ).file()
            # Save trained policy network:
            DataExporter(obj=self.policy_net,
                         file_path=os.path.join(self.output_file_path, 'rl_policy_net.p'),
                         create_dir=False,
                         overwrite=True,
                         cloud=self.cloud,
                         bucket_name=self.bucket_name,
                         region=self.kwargs.get('region')
                         ).file()
            # Save trained target network:
            DataExporter(obj=self.target_net,
                         file_path=os.path.join(self.output_file_path, 'rl_target_net.p'),
                         create_dir=False,
                         overwrite=True,
                         cloud=self.cloud,
                         bucket_name=self.bucket_name,
                         region=self.kwargs.get('region')
                         ).file()
        if model:
            _model = self.env.train_final_model(data_sets=data_sets,
                                                model_name=self.kwargs.get('sml_algorithm'),
                                                param=self.env.observations['model_param'][self.experience_idx],
                                                eval=True
                                                )
            DataExporter(obj=_model,
                         file_path=os.path.join(self.output_file_path, 'model.p'),
                         create_dir=False,
                         overwrite=True,
                         cloud=self.cloud,
                         bucket_name=self.bucket_name,
                         region=self.kwargs.get('region')
                         ).file()
        if experience:
            DataExporter(obj=self.env.observations,
                         file_path=os.path.join(self.output_file_path, 'rl_agent_experience.p'),
                         create_dir=False,
                         overwrite=True,
                         cloud=self.cloud,
                         bucket_name=self.bucket_name,
                         region=self.kwargs.get('region')
                         ).file()

    def visualize(self,
                  results_table: bool = True,
                  rl_experience_metadata: bool = True,
                  rl_experience_time_step: bool = True,
                  reward_distribution: bool = True,
                  reward_distribution_per_episode: bool = True,
                  reward_distribution_grouped_by_action_learning: bool = True,
                  reward_score_distribution: bool = True,
                  score_distribution: bool = True,
                  score_distribution_per_episode: bool = True,
                  score_distribution_grouped_by_action_learning: bool = True,
                  action_distribution: bool = True,
                  action_learning_type_distribution: bool = True,
                  action_distribution_grouped_by_action_type: bool = True,
                  loss_distribution: bool = True,
                  loss_distribution_per_episode: bool = False,
                  ):
        """
        Visualize statistics generated by the interaction of agent and reinforcement learning environment

        :param results_table: bool
            Visualize reinforcement learning results as table chart

        :param rl_experience_metadata: bool
            Visualize experiences of agent and environment observations

        :param rl_experience_time_step: bool
            Visualize experiences of the agent as time series

        :param reward_distribution: bool
            Visualize reward distribution

        :param reward_distribution_per_episode: bool
            Visualize reward distribution per episode

        :param reward_distribution_grouped_by_action_learning: bool
            Visualize reward distribution grouped by action learning types

        :param reward_score_distribution: bool
            Visualize reward supervised machine learning score joint distribution

        :param score_distribution: bool
            Visualize supervised machine learning score distribution

        :param score_distribution_per_episode: bool
            Visualize supervised machine learning score distribution per episode

        :param score_distribution_grouped_by_action_learning: bool
            Visualize supervised machine learning score distribution grouped by action learning type

        :param action_distribution: bool
            Visualize action distribution

        :param action_distribution_grouped_by_action_type: bool
            Visualize action distribution grouped by action learning type

        :param action_learning_type_distribution: bool
            Visualize distribution of action learning type

        :param loss_distribution: bool
            Visualize loss distribution of the neural network of the agent

        :param loss_distribution_per_episode: bool
            Visualize loss distribution of the neural network of the agent per episode
        """
        _df: pd.DataFrame = pd.DataFrame()
        _df['model_name'] = self.env.observations.get('model_name')
        _df['action'] = self.env.observations.get('action')
        _df['action_value'] = self.env.observations.get('action_value')
        _df['sml_score'] = self.env.observations.get('sml_score')
        _df['reward'] = self.env.observations.get('reward')
        _df['reward_scaled'] = _df['reward'] * 100
        _df['reward_clipped'] = self.env.observations.get('reward_clipped')
        _df['transition_gain'] = self.env.observations.get('transition_gain')
        _df['action_learning_type'] = self.env.observations.get('action_learning_type')
        _df['episode'] = self.env.observations.get('episode')
        _df['loss'] = self.env.observations.get('loss')
        _df['policy_update'] = self.env.observations.get('policy_update')
        _df['target_update'] = self.env.observations.get('target_update')
        _df['step'] = [step for step in range(0, self.env.n_steps, 1)]
        _train_error: List[float] = []
        _test_error: List[float] = []
        _metric_name: str = ''
        for metric in self.env.observations.get('fitness'):
            _metric_name = list(metric['train'].keys())[0]
            _train_error.append(metric['train'][_metric_name])
            _test_error.append(metric['test'][_metric_name])
        _df[f'train_error_{_metric_name}'] = _train_error
        _df[f'test_error_{_metric_name}'] = _test_error
        _charts: dict = {}
        if results_table:
            _charts.update({'Results of Reinforcement Learning:': dict(data=_df,
                                                                       plot_type='table',
                                                                       file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(
                                                                           self.output_file_path,
                                                                           'rl_metadata_table.html'
                                                                       )
                                                                       )
                            })
        if rl_experience_metadata:
            _charts.update({'Reinforcement Learning Experience:': dict(data=_df,
                                                                       features=['model_name',
                                                                                 'action_learning_type',
                                                                                 'step',
                                                                                 'episode',
                                                                                 'policy_update',
                                                                                 'target_update',
                                                                                 f'train_error_{_metric_name}',
                                                                                 f'test_error_{_metric_name}',
                                                                                 'reward_clipped',
                                                                                 'reward',
                                                                                 'transition_gain',
                                                                                 'sml_score'
                                                                                 ],
                                                                       color_feature='sml_score',
                                                                       plot_type='parcoords',
                                                                       file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(
                                                                           self.output_file_path,
                                                                           'rl_experience_metadata.html'
                                                                       )
                                                                       )
                            })
        if rl_experience_time_step:
            _charts.update({'Reinforcement Learning Experience Time Steps:': dict(data=_df,
                                                                                  features=['reward_scaled',
                                                                                            'transition_gain',
                                                                                            'sml_score'
                                                                                            ],
                                                                                  time_features=['step'],
                                                                                  plot_type='line',
                                                                                  file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(
                                                                                      self.output_file_path,
                                                                                      'rl_experience_time_step.html'
                                                                                  )
                                                                                  )
                            })
        if reward_distribution:
            _charts.update({'Distribution of Reward:': dict(data=_df,
                                                            features=['reward'],
                                                            plot_type='violin',
                                                            file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(
                                                                self.output_file_path,
                                                                'rl_reward_distribution.html'
                                                            )
                                                            )
                            })
        if reward_distribution_per_episode:
            _charts.update({'Distribution of Reward per Episode:': dict(data=_df,
                                                                        features=['reward'],
                                                                        time_features=['episode'],
                                                                        plot_type='ridgeline',
                                                                        file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(
                                                                            self.output_file_path,
                                                                            'rl_reward_distribution_per_episode.html'
                                                                        )
                                                                        )
                            })
        if reward_distribution_grouped_by_action_learning:
            _charts.update({'Reward grouped by Action Learning Type:': dict(data=_df,
                                                                            features=['reward'],
                                                                            group_by=['action_learning_type'],
                                                                            plot_type='hist',
                                                                            melt=True,
                                                                            file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(
                                                                                self.output_file_path,
                                                                                'rl_reward_by_action_learning_type.html'
                                                                            )
                                                                            )
                            })
        if reward_score_distribution:
            _charts.update({'Reward SML Score Joint Distribution:': dict(data=_df,
                                                                         features=['reward', 'sml_score'],
                                                                         plot_type='joint',
                                                                         file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(
                                                                             self.output_file_path,
                                                                             'rl_reward_score_joint_distribution.html'
                                                                         )
                                                                         )
                            })
        if score_distribution:
            _charts.update({'Distribution of SML Score:': dict(data=_df,
                                                               features=['sml_score'],
                                                               plot_type='violin',
                                                               file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(
                                                                   self.output_file_path,
                                                                   'rl_sml_score_distribution.html'
                                                               )
                                                               )
                            })
        if score_distribution_per_episode:
            _charts.update({'Distribution of SML Score per Episode:': dict(data=_df,
                                                                           features=['sml_score'],
                                                                           time_features=['episode'],
                                                                           plot_type='ridgeline',
                                                                           file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(
                                                                               self.output_file_path,
                                                                               'rl_sml_score_distribution_per_episode.html'
                                                                           )
                                                                           )
                            })
        if score_distribution_grouped_by_action_learning:
            _charts.update({'SML Score grouped by Action Learning Type:': dict(data=_df,
                                                                               features=['sml_score'],
                                                                               group_by=['action_learning_type'],
                                                                               plot_type='hist',
                                                                               melt=True,
                                                                               file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(
                                                                                   self.output_file_path,
                                                                                   'rl_reward_by_action_learning_type.html'
                                                                               )
                                                                               )
                            })
        if action_distribution:
            _charts.update({'Distribution of Action:': dict(data=_df,
                                                            features=['action'],
                                                            plot_type='bar',
                                                            melt=True,
                                                            file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(
                                                                self.output_file_path,
                                                                'rl_action_distribution.html'
                                                            )
                                                            )
                            })
        if action_learning_type_distribution:
            _charts.update({'Distribution of Action Learning Type:': dict(data=_df,
                                                                          features=['action_learning_type'],
                                                                          plot_type='pie',
                                                                          melt=True,
                                                                          file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(
                                                                                self.output_file_path,
                                                                                'rl_action_learning_type_distribution.html'
                                                                          )
                                                                          )
                            })
        if action_distribution_grouped_by_action_type:
            _charts.update({'Distribution of Action grouped by Action Type:': dict(data=_df,
                                                                                   features=['action'],
                                                                                   group_by=['action_learning_type'],
                                                                                   plot_type='bar',
                                                                                   melt=True,
                                                                                   file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(
                                                                                       self.output_file_path,
                                                                                       'rl_action_distribution_grouped_by_action_type.html'
                                                                                   )
                                                                                   )
                            })
        if loss_distribution:
            _charts.update({'Loss of Policy Network:': dict(data=_df,
                                                            features=['loss'],
                                                            time_features=['step'],
                                                            plot_type='line',
                                                            file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(
                                                                self.output_file_path,
                                                                'rl_network_loss.html'
                                                            )
                                                            )
                            })
        if loss_distribution_per_episode:
            _charts.update({'Loss of Policy Network per Episode:': dict(data=_df,
                                                                        features=['loss'],
                                                                        group_by=['episode'],
                                                                        plot_type='hist',
                                                                        melt=False,
                                                                        file_path=self.output_file_path if self.output_file_path is None else '{}{}'.format(
                                                                            self.output_file_path,
                                                                            'rl_network_loss_per_episode.html'
                                                                        )
                                                                        )
                            })
        if len(_charts.keys()) > 0:
            DataVisualizer(subplots=_charts,
                           interactive=True,
                           file_path=self.output_file_path,
                           render=True if self.output_file_path is None else False,
                           height=750,
                           width=750,
                           unit='px'
                           ).run()
