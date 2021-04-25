import copy
import pandas as pd
import unittest

from happy_learning.neural_network_torch import Attention, GRU, LSTM, MLP, RCNN, RNN, SelfAttention, Transformers
from happy_learning.neural_network_generator_torch import NETWORK_TYPE, NetworkGenerator
from happy_learning.sampler import MLSampler
from typing import List


TARGET: str = 'AveragePrice'
PREDICTORS: List[str] = ['4046', '4225', '4770']
TARGET_TEXT: str = 'label'
PREDICTORS_TEXT: List[str] = ['text']
DATA_SET_REG: pd.DataFrame = pd.read_csv(filepath_or_buffer='data/avocado.csv').loc[0:1000, ]
TRAIN_TEST_REG: dict = MLSampler(df=DATA_SET_REG,
                                 target=TARGET,
                                 features=PREDICTORS,
                                 train_size=0.8,
                                 random_sample=True,
                                 stratification=False,
                                 seed=1234
                                 ).train_test_sampling(validation_split=0.1)
TRAIN_DATA_REG_PATH: str = 'data/reg_train.csv'
TEST_DATA_REG_PATH: str = 'data/reg_test.csv'
VALIDATION_DATA_REG_PATH: str = 'data/reg_val.csv'
TRAIN_DATA_PATH: str = 'data/text_train.csv'
TEST_DATA_PATH: str = 'data/text_test.csv'
VALIDATION_DATA_PATH: str = 'data/text_val.csv'
pd.concat(objs=[pd.DataFrame(data=TRAIN_TEST_REG.get('x_train')), pd.DataFrame(data=TRAIN_TEST_REG.get('y_train'))], axis=1).to_csv(path_or_buf=TRAIN_DATA_REG_PATH, index=False)
pd.concat(objs=[pd.DataFrame(data=TRAIN_TEST_REG.get('x_test')), pd.DataFrame(data=TRAIN_TEST_REG.get('y_test'))], axis=1).to_csv(path_or_buf=TEST_DATA_REG_PATH, index=False)
pd.concat(objs=[pd.DataFrame(data=TRAIN_TEST_REG.get('x_val')), pd.DataFrame(data=TRAIN_TEST_REG.get('y_val'))], axis=1).to_csv(path_or_buf=VALIDATION_DATA_REG_PATH, index=False)
DATA_SET_TEXT_CLF: pd.DataFrame = pd.read_csv(filepath_or_buffer='data/tripadvisor_hotel_reviews.csv').loc[0:1000, ]
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
pd.concat(objs=[pd.DataFrame(data=TRAIN_TEST_TEXT_CLF.get('x_train')), pd.DataFrame(data=TRAIN_TEST_TEXT_CLF.get('y_train'))], axis=1).to_csv(path_or_buf=TRAIN_DATA_PATH_TEXT, index=False)
pd.concat(objs=[pd.DataFrame(data=TRAIN_TEST_TEXT_CLF.get('x_test')), pd.DataFrame(data=TRAIN_TEST_TEXT_CLF.get('y_test'))], axis=1).to_csv(path_or_buf=TEST_DATA_PATH_TEXT, index=False)
pd.concat(objs=[pd.DataFrame(data=TRAIN_TEST_TEXT_CLF.get('x_val')), pd.DataFrame(data=TRAIN_TEST_TEXT_CLF.get('y_val'))], axis=1).to_csv(path_or_buf=VALIDATION_DATA_PATH_TEXT, index=False)


class NetworkGeneratorTest(unittest.TestCase):
    """
    Class for testing class NetworkGenerator
    """
    def test_generate_attention_network_model(self):
        _net_gen: object = NetworkGenerator(target=TARGET_TEXT,
                                            predictors=PREDICTORS_TEXT,
                                            output_layer_size=5,
                                            train_data_path=TRAIN_DATA_PATH_TEXT,
                                            test_data_path=TEST_DATA_PATH_TEXT,
                                            validation_data_path=VALIDATION_DATA_PATH_TEXT,
                                            models=['att'],
                                            sep=','
                                            ).generate_model()
        self.assertTrue(expr=isinstance(_net_gen.model, Attention))

    def test_generate_gated_recurrent_neural_network_model(self):
        _net_gen: object = NetworkGenerator(target=TARGET_TEXT,
                                            predictors=PREDICTORS_TEXT,
                                            output_layer_size=5,
                                            train_data_path=TRAIN_DATA_PATH_TEXT,
                                            test_data_path=TEST_DATA_PATH_TEXT,
                                            validation_data_path=VALIDATION_DATA_PATH_TEXT,
                                            models=['gru'],
                                            sep=','
                                            ).generate_model()
        self.assertTrue(expr=isinstance(_net_gen.model, GRU))

    def test_generate_long_short_term_memory_network_model(self):
        _net_gen: object = NetworkGenerator(target=TARGET_TEXT,
                                            predictors=PREDICTORS_TEXT,
                                            output_layer_size=5,
                                            train_data_path=TRAIN_DATA_PATH_TEXT,
                                            test_data_path=TEST_DATA_PATH_TEXT,
                                            validation_data_path=VALIDATION_DATA_PATH_TEXT,
                                            models=['lstm'],
                                            sep=','
                                            ).generate_model()
        self.assertTrue(expr=isinstance(_net_gen.model, LSTM))

    def test_generate_multi_layer_perceptron_model(self):
        _net_gen: object = NetworkGenerator(target=TARGET_TEXT,
                                            predictors=PREDICTORS_TEXT,
                                            output_layer_size=5,
                                            train_data_path=TRAIN_DATA_PATH_TEXT,
                                            test_data_path=TEST_DATA_PATH_TEXT,
                                            validation_data_path=VALIDATION_DATA_PATH_TEXT,
                                            models=['mlp'],
                                            sep=','
                                            ).generate_model()
        self.assertTrue(expr=isinstance(_net_gen.model, MLP))

    def test_generate_recurrent_convolutional_neural_network_model(self):
        _net_gen: object = NetworkGenerator(target=TARGET_TEXT,
                                            predictors=PREDICTORS_TEXT,
                                            output_layer_size=5,
                                            train_data_path=TRAIN_DATA_PATH_TEXT,
                                            test_data_path=TEST_DATA_PATH_TEXT,
                                            validation_data_path=VALIDATION_DATA_PATH_TEXT,
                                            models=['rcnn'],
                                            sep=','
                                            ).generate_model()
        self.assertTrue(expr=isinstance(_net_gen.model, RCNN))

    def test_generate_recurrent_neural_network_model(self):
        _net_gen: object = NetworkGenerator(target=TARGET_TEXT,
                                            predictors=PREDICTORS_TEXT,
                                            output_layer_size=5,
                                            train_data_path=TRAIN_DATA_PATH_TEXT,
                                            test_data_path=TEST_DATA_PATH_TEXT,
                                            validation_data_path=VALIDATION_DATA_PATH_TEXT,
                                            models=['rnn'],
                                            sep=','
                                            ).generate_model()
        self.assertTrue(expr=isinstance(_net_gen.model, RNN))

    def test_generate_self_attention_network_model(self):
        _net_gen: object = NetworkGenerator(target=TARGET_TEXT,
                                            predictors=PREDICTORS_TEXT,
                                            output_layer_size=5,
                                            train_data_path=TRAIN_DATA_PATH_TEXT,
                                            test_data_path=TEST_DATA_PATH_TEXT,
                                            validation_data_path=VALIDATION_DATA_PATH_TEXT,
                                            models=['self'],
                                            sep=','
                                            ).generate_model()
        self.assertTrue(expr=isinstance(_net_gen.model, SelfAttention))

    def test_generate_transformer_model(self):
        _net_gen: object = NetworkGenerator(target=TARGET_TEXT,
                                            predictors=PREDICTORS_TEXT,
                                            output_layer_size=5,
                                            train_data_path=TRAIN_DATA_PATH_TEXT,
                                            test_data_path=TEST_DATA_PATH_TEXT,
                                            validation_data_path=VALIDATION_DATA_PATH_TEXT,
                                            models=['trans'],
                                            sep=','
                                            ).generate_model()
        self.assertTrue(expr=isinstance(_net_gen.model, Transformers))

    def test_generate_params(self):
        _net_gen: object = NetworkGenerator(target=TARGET_TEXT,
                                            predictors=PREDICTORS_TEXT,
                                            output_layer_size=5,
                                            train_data_path=TRAIN_DATA_PATH_TEXT,
                                            test_data_path=TEST_DATA_PATH_TEXT,
                                            validation_data_path=VALIDATION_DATA_PATH_TEXT,
                                            models=list(NETWORK_TYPE.keys())
                                            ).generate_model()
        _model = _net_gen.generate_model()
        _mutated_param: dict = copy.deepcopy(_model.model_param_mutated)
        _net_gen.generate_params(param_rate=0.1, force_param=None)
        self.assertTrue(expr=len(_mutated_param.keys()) < len(_net_gen.model_param_mutated.keys()))

    def test_get_vanilla_transformer(self):
        _net_gen: object = NetworkGenerator(target=TARGET_TEXT,
                                            predictors=PREDICTORS_TEXT,
                                            output_layer_size=5,
                                            train_data_path=TRAIN_DATA_PATH_TEXT,
                                            test_data_path=TEST_DATA_PATH_TEXT,
                                            validation_data_path=VALIDATION_DATA_PATH_TEXT,
                                            models=['trans'],
                                            model_name='trans',
                                            sep=','
                                            )
        _model = _net_gen.get_vanilla_model()

    def test_eval(self):
        _net_gen: object = NetworkGenerator(target=TARGET_TEXT,
                                            predictors=PREDICTORS_TEXT,
                                            output_layer_size=5,
                                            train_data_path=TRAIN_DATA_PATH_TEXT,
                                            test_data_path=TEST_DATA_PATH_TEXT,
                                            validation_data_path=VALIDATION_DATA_PATH_TEXT,
                                            models=list(NETWORK_TYPE.keys()),
                                            sep=','
                                            )
        _model = _net_gen.generate_model()
        _model.train()
        _model.eval(validation=True)
        _model.eval(validation=False)
        self.assertTrue(expr=_model.fitness.get('val') is not None and _model.fitness.get('test') is not None)

    def test_predict(self):
        _net_gen: object = NetworkGenerator(target=TARGET_TEXT,
                                            predictors=PREDICTORS_TEXT,
                                            output_layer_size=5,
                                            train_data_path=TRAIN_DATA_PATH_TEXT,
                                            test_data_path=TEST_DATA_PATH_TEXT,
                                            validation_data_path=VALIDATION_DATA_PATH_TEXT,
                                            models=list(NETWORK_TYPE.keys()),
                                            sep=','
                                            )
        _model = _net_gen.generate_model()
        _model.train()
        _model.predict()
        self.assertTrue(expr=_model.fitness.get('test') is not None)

    def test_save(self):
        pass

    def test_train(self):
        _net_gen: object = NetworkGenerator(target=TARGET_TEXT,
                                            predictors=PREDICTORS_TEXT,
                                            output_layer_size=5,
                                            train_data_path=TRAIN_DATA_PATH_TEXT,
                                            test_data_path=TEST_DATA_PATH_TEXT,
                                            validation_data_path=VALIDATION_DATA_PATH_TEXT,
                                            models=['trans'],#list(NETWORK_TYPE.keys()),
                                            sep=','
                                            )
        _model = _net_gen.generate_model()
        _model.train()
        self.assertTrue(expr=_model.fitness.get('train') is not None)

    def test_update_data(self):
        pass

    def test_update_model_param(self):
        pass


if __name__ == '__main__':
    unittest.main()
