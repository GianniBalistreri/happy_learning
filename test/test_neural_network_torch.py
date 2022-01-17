import numpy as np
import pandas as pd
import unittest

from happy_learning.neural_network_generator_torch import NetworkGenerator
from typing import List


DATA_FILE_PATH: dict = dict(train='data/text_train.csv',
                            test='data/text_test.csv',
                            val='data/text_val.csv',
                            )

DATA_SET_MLP: pd.DataFrame = pd.DataFrame(data=dict(x1=np.random.choice(a=[0, 1], size=1000),
                                                    x2=np.random.choice(a=[0, 1], size=1000),
                                                    x3=np.random.choice(a=[0, 1], size=1000),
                                                    x4=np.random.choice(a=[0, 1], size=1000),
                                                    y=np.random.choice(a=[0, 1], size=1000)
                                                    )
                                          )


class AttentionTest(unittest.TestCase):
    """
    Class for testing class Attention (torch)
    """
    def test_forward(self):
        _network_generator: NetworkGenerator = NetworkGenerator(target='label',
                                                                predictors=['text'],
                                                                output_layer_size=5,
                                                                train_data_path=DATA_FILE_PATH.get('train'),
                                                                test_data_path=DATA_FILE_PATH.get('test'),
                                                                validation_data_path=DATA_FILE_PATH.get('val'),
                                                                model_name='att',
                                                                sep=','
                                                                )
        _network_generator.get_vanilla_model()
        _network_generator.train()
        self.assertTrue(expr=len(_network_generator.fitness.keys()) > 0)


class GRUTest(unittest.TestCase):
    """
    Class for testing class GRU (torch)
    """
    def test_forward(self):
        _network_generator: NetworkGenerator = NetworkGenerator(target='label',
                                                                predictors=['text'],
                                                                output_layer_size=5,
                                                                train_data_path=DATA_FILE_PATH.get('train'),
                                                                test_data_path=DATA_FILE_PATH.get('test'),
                                                                validation_data_path=DATA_FILE_PATH.get('val'),
                                                                model_name='gru',
                                                                sep=','
                                                                )
        _network_generator.get_vanilla_model()
        _network_generator.train()
        self.assertTrue(expr=len(_network_generator.fitness.keys()) > 0)


class LSTMTest(unittest.TestCase):
    """
    Class for testing class LSTM (torch)
    """
    def test_forward(self):
        _network_generator: NetworkGenerator = NetworkGenerator(target='label',
                                                                predictors=['text'],
                                                                output_layer_size=5,
                                                                train_data_path=DATA_FILE_PATH.get('train'),
                                                                test_data_path=DATA_FILE_PATH.get('test'),
                                                                validation_data_path=DATA_FILE_PATH.get('val'),
                                                                model_name='lstm',
                                                                sep=','
                                                                )
        _network_generator.get_vanilla_model()
        _network_generator.train()
        self.assertTrue(expr=len(_network_generator.fitness.keys()) > 0)


class MLPTest(unittest.TestCase):
    """
    Class for testing class MLP (torch)
    """
    def test_forward(self):
        _predictors: List[str] = ['x1', 'x2', 'x3', 'x4']
        _network_generator: NetworkGenerator = NetworkGenerator(target='y',
                                                                predictors=_predictors,
                                                                output_layer_size=2,
                                                                x_train=DATA_SET_MLP[_predictors].values,
                                                                y_train=DATA_SET_MLP['y'].values,
                                                                x_test=DATA_SET_MLP[_predictors].values,
                                                                y_test=DATA_SET_MLP['y'].values,
                                                                x_val=DATA_SET_MLP[_predictors].values,
                                                                y_val=DATA_SET_MLP['y'].values,
                                                                #models=['mlp'],
                                                                model_name='mlp',
                                                                sequential_type='numeric'
                                                                )
        _network_generator.generate_model()
        _network_generator.train()
        self.assertTrue(expr=len(_network_generator.fitness.keys()) > 0)


class RCNNTest(unittest.TestCase):
    """
    Class for testing class RCNN (torch)
    """
    def test_forward(self):
        _network_generator: NetworkGenerator = NetworkGenerator(target='label',
                                                                predictors=['text'],
                                                                output_layer_size=5,
                                                                train_data_path=DATA_FILE_PATH.get('train'),
                                                                test_data_path=DATA_FILE_PATH.get('test'),
                                                                validation_data_path=DATA_FILE_PATH.get('val'),
                                                                model_name='rcnn',
                                                                sep=','
                                                                )
        _network_generator.get_vanilla_model()
        _network_generator.train()
        self.assertTrue(expr=len(_network_generator.fitness.keys()) > 0)


class RNNTest(unittest.TestCase):
    """
    Class for testing class RNN (torch)
    """
    def test_forward(self):
        _network_generator: NetworkGenerator = NetworkGenerator(target='label',
                                                                predictors=['text'],
                                                                output_layer_size=5,
                                                                train_data_path=DATA_FILE_PATH.get('train'),
                                                                test_data_path=DATA_FILE_PATH.get('test'),
                                                                validation_data_path=DATA_FILE_PATH.get('val'),
                                                                model_name='rnn',
                                                                sep=','
                                                                )
        _network_generator.get_vanilla_model()
        _network_generator.train()
        self.assertTrue(expr=len(_network_generator.fitness.keys()) > 0)


class SelfAttentionTest(unittest.TestCase):
    """
    Class for testing class SelfAttention (torch)
    """
    def test_forward(self):
        _network_generator: NetworkGenerator = NetworkGenerator(target='label',
                                                                predictors=['text'],
                                                                output_layer_size=5,
                                                                train_data_path=DATA_FILE_PATH.get('train'),
                                                                test_data_path=DATA_FILE_PATH.get('test'),
                                                                validation_data_path=DATA_FILE_PATH.get('val'),
                                                                model_name='self',
                                                                sep=','
                                                                )
        _network_generator.get_vanilla_model()
        _network_generator.train()
        self.assertTrue(expr=len(_network_generator.fitness.keys()) > 0)


class TransformerTest(unittest.TestCase):
    """
    Class for testing class Transformer (torch)
    """
    def test_forward(self):
        _network_generator: NetworkGenerator = NetworkGenerator(target='label',
                                                                predictors=['text'],
                                                                output_layer_size=5,
                                                                train_data_path=DATA_FILE_PATH.get('train'),
                                                                test_data_path=DATA_FILE_PATH.get('test'),
                                                                validation_data_path=DATA_FILE_PATH.get('val'),
                                                                model_name='trans',
                                                                sep=','
                                                                )
        _network_generator.get_vanilla_model()
        _network_generator.train()
        self.assertTrue(expr=len(_network_generator.fitness.keys()) > 0)


if __name__ == '__main__':
    unittest.main()
