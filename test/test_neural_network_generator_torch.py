import numpy as np
import pandas as pd
import unittest

from happy_learning.neural_network_torch import MLP, RCNN
from happy_learning.neural_network_generator_torch import NeuralNetwork, NetworkGenerator
from happy_learning.sampler import MLSampler
from typing import List


DATA_SET_CLF: pd.DataFrame = pd.DataFrame(data=dict(x1=np.random.choice(a=[0, 1], size=1000),
                                                    x2=np.random.choice(a=[0, 1], size=1000),
                                                    x3=np.random.choice(a=[0, 1], size=1000),
                                                    x4=np.random.choice(a=[0, 1], size=1000),
                                                    y=np.random.choice(a=[0, 1], size=1000)
                                                    )
                                          )
TARGET_MLP: str = 'y'
PREDICTORS_MLP: List[str] = ['x1', 'x2', 'x3', 'x4']
TRAIN_TEST_CLF: dict = MLSampler(df=DATA_SET_CLF,
                                 target=TARGET_MLP,
                                 features=PREDICTORS_MLP,
                                 train_size=0.8,
                                 random_sample=True,
                                 stratification=False,
                                 seed=1234
                                 ).train_test_sampling(validation_split=0.1)
TARGET_TEXT: str = 'n_killed'
PREDICTORS_TEXT: List[str] = ['notes']
DATA_SET_TEXT_CLF: pd.DataFrame = pd.read_csv(filepath_or_buffer='data/gun-violence-data_01-2013_03-2018.csv')
DATA_SET_TEXT_CLF = DATA_SET_TEXT_CLF.loc[~DATA_SET_TEXT_CLF.isnull().any(axis=1), :]
TRAIN_TEST_TEXT_CLF: dict = MLSampler(df=DATA_SET_TEXT_CLF,
                                      target=TARGET_TEXT,
                                      features=PREDICTORS_TEXT,
                                      train_size=0.8,
                                      random_sample=True,
                                      stratification=False,
                                      seed=1234
                                      ).train_test_sampling(validation_split=0.1)
TRAIN_DATA_PATH_TEXT: str = 'data/text_train.csv'
TEST_DATA_PATH_TEXT: str = 'data/text_test.csv'
VALIDATION_DATA_PATH_TEXT: str = 'data/text_val.csv'
pd.concat(objs=[pd.DataFrame(data=TRAIN_TEST_TEXT_CLF.get('x_train')), pd.DataFrame(data=TRAIN_TEST_TEXT_CLF.get('y_train'))]).to_csv(path_or_buf=TRAIN_DATA_PATH_TEXT)
pd.concat(objs=[pd.DataFrame(data=TRAIN_TEST_TEXT_CLF.get('x_test')), pd.DataFrame(data=TRAIN_TEST_TEXT_CLF.get('y_test'))]).to_csv(path_or_buf=TEST_DATA_PATH_TEXT)
pd.concat(objs=[pd.DataFrame(data=TRAIN_TEST_TEXT_CLF.get('x_val')), pd.DataFrame(data=TRAIN_TEST_TEXT_CLF.get('y_val'))]).to_csv(path_or_buf=VALIDATION_DATA_PATH_TEXT)


class NeuralNetworkTest(unittest.TestCase):
    """
    Class for testing class NeuralNetwork
    """
    def test_multi_layer_perceptron(self):
        self.assertTrue(expr=isinstance(NeuralNetwork(target=TARGET_MLP,
                                                      predictors=PREDICTORS_MLP,
                                                      output_layer_size=2,
                                                      x_train=DATA_SET_CLF[PREDICTORS_MLP],
                                                      y_train=DATA_SET_CLF[TRAIN_TEST_CLF],
                                                      x_test=DATA_SET_CLF[PREDICTORS_MLP],
                                                      y_test=DATA_SET_CLF[TRAIN_TEST_CLF],
                                                      x_val=DATA_SET_CLF[PREDICTORS_MLP],
                                                      y_val=DATA_SET_CLF[TRAIN_TEST_CLF],
                                                      models=['mlp']
                                                      ).multi_layer_perceptron(),
                                        MLP
                                        )
                        )

    def test_multi_layer_perceptron_param(self):
        _mlp_param: dict = NeuralNetwork(target=TARGET_MLP,
                                         predictors=PREDICTORS_MLP,
                                         x_train=TRAIN_TEST_CLF.get('x_train'),
                                         y_train=TRAIN_TEST_CLF.get('y_train'),
                                         x_test=TRAIN_TEST_CLF.get('x_test'),
                                         y_test=TRAIN_TEST_CLF.get('y_test'),
                                         x_val=TRAIN_TEST_CLF.get('x_val'),
                                         y_val=TRAIN_TEST_CLF.get('y_val')
                                         ).multi_layer_perceptron_param()
        self.assertTrue(expr=_mlp_param.get(list(_mlp_param.keys())[0]) != NeuralNetwork(target=TARGET_MLP,
                                                                                         predictors=PREDICTORS_MLP,
                                                                                         x_train=TRAIN_TEST_CLF.get('x_train'),
                                                                                         y_train=TRAIN_TEST_CLF.get('y_train'),
                                                                                         x_test=TRAIN_TEST_CLF.get('x_test'),
                                                                                         y_test=TRAIN_TEST_CLF.get('y_test'),
                                                                                         x_val=TRAIN_TEST_CLF.get('x_val'),
                                                                                         y_val=TRAIN_TEST_CLF.get('y_val')
                                                                                         ).multi_layer_perceptron_param().get(list(_mlp_param.keys())[0]))

    def test_recurrent_convolutional_network_param(self):
        _rcnn_param: dict = NeuralNetwork(target=TARGET_TEXT,
                                          predictors=PREDICTORS_TEXT,
                                          x_train=TRAIN_TEST_TEXT_CLF.get('x_train'),
                                          y_train=TRAIN_TEST_TEXT_CLF.get('y_train'),
                                          x_test=TRAIN_TEST_TEXT_CLF.get('x_test'),
                                          y_test=TRAIN_TEST_TEXT_CLF.get('y_test'),
                                          x_val=TRAIN_TEST_TEXT_CLF.get('x_val'),
                                          y_val=TRAIN_TEST_TEXT_CLF.get('y_val')
                                          ).recurrent_convolutional_neural_network_param()
        self.assertTrue(expr=_rcnn_param.get(list(_rcnn_param.keys())[0]) != NeuralNetwork(target=TARGET_TEXT,
                                                                                           predictors=PREDICTORS_TEXT,
                                                                                           x_train=TRAIN_TEST_TEXT_CLF.get('x_train'),
                                                                                           y_train=TRAIN_TEST_TEXT_CLF.get('y_train'),
                                                                                           x_test=TRAIN_TEST_TEXT_CLF.get('x_test'),
                                                                                           y_test=TRAIN_TEST_TEXT_CLF.get('y_test'),
                                                                                           x_val=TRAIN_TEST_TEXT_CLF.get('x_val'),
                                                                                           y_val=TRAIN_TEST_TEXT_CLF.get('y_val')
                                                                                           ).recurrent_convolutional_neural_network_param().get(list(_rcnn_param.keys())[0]))


class NetworkGeneratorTest(unittest.TestCase):
    """
    Class for testing class NetworkGenerator
    """
    def test_generate_model(self):
        _net_gen: object = NetworkGenerator(target=TARGET_TEXT,
                                            predictors=PREDICTORS_TEXT,
                                            output_layer_size=2,
                                            train_data_path=TRAIN_DATA_PATH_TEXT,
                                            test_data_path=TEST_DATA_PATH_TEXT,
                                            validation_data_path=VALIDATION_DATA_PATH_TEXT,
                                            models=['rcnn']
                                            ).generate_model()
        self.assertTrue(expr=isinstance(_net_gen.model, RCNN))

    def test_generate_params(self):
        _net_gen: object = NetworkGenerator(target=TARGET_TEXT,
                                            predictors=PREDICTORS_TEXT,
                                            output_layer_size=2,
                                            train_data_path=TRAIN_DATA_PATH_TEXT,
                                            test_data_path=TEST_DATA_PATH_TEXT,
                                            validation_data_path=VALIDATION_DATA_PATH_TEXT,
                                            models=['rcnn']
                                            ).generate_model()
        print('model param', _net_gen.model_param)
        _net_gen.generate_params()
        print('mutated', _net_gen.model_param_mutated)
        print('new param', _net_gen.model_param)

    def test_get_vanilla_network(self):
        pass

    def test_eval(self):
        pass

    def test_predict(self):
        pass

    def test_save(self):
        pass

    def test_train(self):
        pass

    def test_update_data(self):
        pass

    def test_update_model_param(self):
        pass


if __name__ == '__main__':
    unittest.main()
