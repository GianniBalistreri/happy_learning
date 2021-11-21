"""

Train Reinforcement Learning Algorithm: Deep-Q-Learning Network

"""

import math
import pandas as pd
import random
import torch

from .environment_modeling import EnvironmentModeling
from .neural_network_torch import DQNFC
from .sampler import MLSampler
from .utils import HappyLearningUtils
from collections import namedtuple, deque
from easyexplore.data_visualizer import DataVisualizer
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
        self.exploration_phase: bool = True
        self.exploration_phase_end: int = -1
        self.kwargs: dict = kwargs

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
        print('Random Sample', _sample)
        print('EPS Threshold', _eps_threshold)
        _model_name: str = random.choice(seq=list(self.env.action_space['param_to_value'].keys()))
        _action_names: List[str] = self.env.action_space['action']
        #_action_names: List[str] = list(self.env.action_space['param_to_value'][_model_name].keys())
        if _sample > _eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                _new_action_exploitation: torch.tensor = self.policy_net(state).max(1)[1].view(1, 1)
                print('Exploitation action', _new_action_exploitation)
                _idx: int = _new_action_exploitation.tolist()[0][0]
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
        for i_episode in range(0, self.episodes, 1):
            print('Episode:', i_episode)
            for a in range(0, self.env.n_actions, 1):
                # Select and perform an action
                if self.env.n_steps == 0:
                    _action: dict = None
                else:
                    _action: dict = self._select_action(state=self.env.state)
                # Observe new state and reward
                _state, _next_state, _reward, _clipped_reward = self.env.step(action=_action, data_sets=data_sets)
                print('Reward', _reward)
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
                print('Length Memory', len(self.memory))
                if len(self.memory) >= self.batch_size:
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
                    loss = criterion(state_action_values.to(dtype=torch.float64), expected_state_action_values.unsqueeze(1))

                    # Optimize the model
                    self.optimizer.zero_grad()
                    loss.backward()
                    for param in self.policy_net.parameters():
                        param.grad.data.clamp_(-1, 1)
                    self.optimizer.step()
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize(self, df: pd.DataFrame, target: str, features: List[str] = None):
        """
        Optimize hyper-parameter configuration

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
        self.env: EnvironmentModeling = EnvironmentModeling(sml_problem=HappyLearningUtils().get_ml_type(values=df[target].values),
                                                            sml_algorithm=self.kwargs.get('sml_algorithm')
                                                            )
        self.policy_net: DQNFC = DQNFC(input_size=self.env.n_actions,
                                       hidden_size=100,
                                       output_size=self.env.n_actions
                                       ).to(DEVICE)
        self.target_net: DQNFC = DQNFC(input_size=self.env.n_actions,
                                       hidden_size=100,
                                       output_size=self.env.n_actions
                                       ).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer: torch.optim = torch.optim.RMSprop(self.policy_net.parameters())
        self.memory: ReplayMemory = ReplayMemory(10000)
        print('Training started')
        self._train(data_sets=data_sets)

    def save(self):
        """
        Save learnings of the agent
        """
        pass

    def visualize(self):
        """
        Visualize statistics generated by reinforcement learning environment
        """
        pass
