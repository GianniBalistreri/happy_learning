"""

Train Reinforcement Learning Algorithm: Deep-Q-Learning Network

"""

import numpy as np
import math
import pandas as pd
import random
import torch

from .environment_modeling import EnvironmentModeling
from .neural_network_torch import DQNFC
from .sampler import MLSampler
from .utils import HappyLearningUtils
from collections import namedtuple, deque
from datetime import datetime
from easyexplore.data_import_export import CLOUD_PROVIDER, DataExporter
from easyexplore.data_visualizer import DataVisualizer
from easyexplore.utils import Log
from typing import List

DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
TRANSITION: namedtuple = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# TODO:
#  categorize metric parameters into very small, small, medium, high, very high


class ReplayMemory:
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
        self.memory.append(TRANSITION(*args))

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
                 **kwargs
                 ):
        """
        :param batch_size: int
            Batch size

        :param episodes: int
            Number of episodes to train

        :param gamma: float

        :param eps_start: float

        :param eps_end: float

        :param eps_decay: float

        :param target_update: int
            Interval for updating target net

        :param optimizer: str
            Abbreviate name of the optimizer

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
        self.use_clipped_reward: bool = use_clipped_reward
        self.deploy_model: bool = deploy_model
        self.n_training: int = 0
        self.n_optimization: int = 0
        self.n_update: int = 0
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
        self.timer: int = timer_in_seconds if timer_in_seconds > 0 else 99999
        self.log: bool = log
        self.verbose: bool = verbose
        self.start_time: datetime = datetime.now()
        self.kwargs: dict = kwargs

    def _save_checkpoint(self):
        """
        Save checkpoint of the agent
        """
        pass

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
        #_action_names: List[str] = list(self.env.action_space['param_to_value'][_model_name].keys())
        if _sample > _eps_threshold:
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
        for episode in range(0, self.episodes, 1):
            Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Episode: {episode}')
            for _ in range(0, self.env.n_actions, 1):
                if self.verbose:
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Step: {self.env.n_steps}')
                # Select and perform an action
                if self.env.n_steps == 0:
                    _action: dict = None
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
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Fitness: {self.env.observations["fitness"][-1]}')
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Transition gain: {self.env.observations["transition_gain"][-1]}')
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
                    if self.verbose:
                        Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Optimization step of policy network: {self.n_optimization}')
                    transitions = self.memory.sample(self.batch_size)
                    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                    # detailed explanation). This converts batch-array of Transitions
                    # to Transition of batch-arrays.
                    batch = TRANSITION(*zip(*transitions))

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
                                     expected_state_action_values.unsqueeze(1).to(dtype=torch.float64)
                                     )

                    # Optimize the model
                    self.optimizer.zero_grad()
                    loss.backward()
                    for param in self.policy_net.parameters():
                        param.grad.data.clamp_(-1, 1)
                    self.optimizer.step()
            # Update the target network, copying all weights and biases in DQN
            if episode % self.target_update == 0:
                self.n_update += 1
                if self.verbose:
                    Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Update step of target network: {self.n_update}')
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize(self,
                 df: pd.DataFrame,
                 target: str,
                 features: List[str] = None,
                 memory_capacity: int = 10000,
                 hidden_layer_size: int = 100,
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

        :param hidden_layer_size: int
            Number of neurons of the fully connected hidden layer
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
        _hidden_layer_size: int = hidden_layer_size if hidden_layer_size > 1 else 100
        self.policy_net: DQNFC = DQNFC(input_size=self.env.n_actions,
                                       hidden_size=_hidden_layer_size,
                                       output_size=self.env.n_actions
                                       ).to(DEVICE)
        self.target_net: DQNFC = DQNFC(input_size=self.env.n_actions,
                                       hidden_size=_hidden_layer_size,
                                       output_size=self.env.n_actions
                                       ).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer: torch.optim = torch.optim.RMSprop(self.policy_net.parameters())
        _memory_capacity: int = memory_capacity if memory_capacity >= 10 else 10000
        self.memory: ReplayMemory = ReplayMemory(capacity=_memory_capacity)
        self._train(data_sets=data_sets)
        self.experience_idx = np.array(self.env.observations['reward']).argmax()
        _reward: float = self.env.observations['reward'][self.experience_idx]
        _state: dict = self.env.observations['state'][self.experience_idx]
        _model_param: dict = self.env.observations['model_param'][self.experience_idx]
        _fitness: dict = self.env.observations['fitness'][self.experience_idx]
        Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Best model state: {_state}')
        Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Best model param: {_model_param}')
        Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Best model fitness: {_fitness}')
        Log(write=self.log, logger_file_path=self.output_file_path).log(msg=f'Best model reward: {_reward}')
        self.save(agent=True, model=self.deploy_model, experience=True)
        if self.plot:
            self.visualize()

    def save(self, agent: bool, model: bool, experience: bool):
        """
        Save learnings of the agent
        """
        if agent:
            pass
        if model:
            _model = self.env.train_final_model(experience_id=self.experience_idx, data_sets=None)
        if experience:
            pass

    def visualize(self):
        """
        Visualize statistics generated by reinforcement learning environment
        """
        pass
