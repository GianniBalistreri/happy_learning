import pandas as pd
import torch
import unittest

from happy_learning.environment_modeling import EnvironmentModeling
from happy_learning.sampler import MLSampler
from happy_learning.supervised_machine_learning import ModelGeneratorClf
from typing import List

MODEL_NAME: str = 'cat'
ENV: EnvironmentModeling = EnvironmentModeling(sml_problem='clf_binary', sml_algorithm=MODEL_NAME)
PARAM: dict = ModelGeneratorClf(model_name=MODEL_NAME).generate_model().model_param
DF: pd.DataFrame = pd.read_csv(filepath_or_buffer='./data/avocado.csv', sep=',')
DF = DF.replace({'conventional': 0, 'organic': 1})
DF['type'] = DF['type'].astype(int)
CLF_TARGET: str = 'type'
FEATURES: List[str] = ['Total Volume',
                       'Total Bags',
                       '4046',
                       '4225',
                       '4770',
                       'Total Bags',
                       'Small Bags',
                       'Large Bags',
                       'XLarge Bags'
                       ]
DATA_SETS: dict = MLSampler(df=DF,
                            target=CLF_TARGET,
                            features=FEATURES,
                            train_size=0.8,
                            stratification=False
                            ).train_test_sampling(validation_split=0.1)


class TestEnvironmentModeling(unittest.TestCase):
    """
    Class for testing class EnvironmentModeling
    """
    def test_action_to_state(self):
        """
        Test action to state conversion function
        """
        _action: dict = {'cat': {'n_estimators': 5, 'idx': 0}}
        _state, _next_state, _reward, _clipped_reward = ENV.step(action=None, data_sets=DATA_SETS)
        __state: dict = ENV.action_to_state(action=_action)
        self.assertTrue(expr=isinstance(__state, dict)) and self.assertTrue(expr=len(__state.keys()) == len(PARAM.keys()))

    def test_param_to_state(self):
        """
        Test param to state conversion function
        """
        _state: dict = ENV.param_to_state(model_name=MODEL_NAME, param=PARAM)
        self.assertTrue(expr=isinstance(_state, dict)) and self.assertGreater(a=len(_state.keys()), b=len(PARAM.keys()))

    def test_state_to_tensor(self):
        """
        Test state to tensor conversion function
        """
        _state: dict = ENV.param_to_state(model_name=MODEL_NAME, param=PARAM)
        __state: torch.tensor = ENV.state_to_tensor(model_name=MODEL_NAME, state=_state)
        self.assertTrue(expr=isinstance(__state, torch.Tensor)) and self.assertTrue(expr=len(_state.keys()) == __state.shape[0])

    def test_step(self):
        """
        Test step function of reinforcement learning environment for hyper-parameter optimization
        """
        _state, _next_state, _reward, _clipped_reward = ENV.step(action=None, data_sets=DATA_SETS)
        self.assertTrue(expr=isinstance(_state, torch.Tensor)) and self.assertTrue(expr=isinstance(_next_state, torch.Tensor)) and\
        self.assertTrue(expr=isinstance(_reward, torch.Tensor)) and self.assertTrue(expr=isinstance(_clipped_reward, torch.Tensor))

    def test_train_final_model(self):
        """
        Test train final model function
        """
        self.assertTrue(expr=isinstance(ENV.train_final_model(data_sets=DATA_SETS,
                                                              model_name=MODEL_NAME,
                                                              param=PARAM,
                                                              eval=True
                                                              ),
                                        ModelGeneratorClf
                                        )
                        )


if __name__ == '__main__':
    unittest.main()
