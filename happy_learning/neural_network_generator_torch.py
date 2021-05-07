import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional

from .evaluate_machine_learning import EvalClf, EvalReg, ML_METRIC, SML_SCORE
from .neural_network_torch import Attention, GRU, MLP, LSTM, RCNN, RNN, SelfAttention, Transformers
from .utils import HappyLearningUtils
from datetime import datetime
from easyexplore.data_import_export import CLOUD_PROVIDER, DataImporter
from torch.utils.data import DataLoader, TensorDataset
from torchtext.data import BucketIterator, Field, TabularDataset
from torchtext.vocab import FastText
from typing import Dict, List, Tuple


MAX_HIDDEN_LAYERS: int = 50
NETWORK_TYPE: Dict[str, str] = dict(att='attention_network',
                                    #cnn='convolutional_neural_network',
                                    gru='gated_recurrent_unit_network',
                                    lstm='long_short_term_memory_network',
                                    mlp='multi_layer_perceptron',
                                    rnn='recurrent_neural_network',
                                    rcnn='recurrent_convolutional_neural_network',
                                    self='self_attention_network',
                                    trans='transformer'
                                    )
NETWORK_TYPE_CATEGORY: Dict[str, List[str]] = dict(tabular=['mlp'],
                                                   seq=['att', 'cnn', 'lstm', 'rnn', 'rcnn', 'lstm', 'self', 'trans']
                                                   )
HIDDEN_LAYER_CATEGORY_EVOLUTION: List[str] = ['random',
                                              'constant',
                                              'monotone',
                                              'adaptive'
                                              ]
HIDDEN_LAYER_CATEGORY: Dict[str, Tuple[int, int]] = dict(very_small=(1, 2),
                                                         small=(3, 4),
                                                         medium=(5, 7),
                                                         big=(8, 10),
                                                         very_big=(11, MAX_HIDDEN_LAYERS)
                                                         )
ACTIVATION: dict = dict(weighted_sum=dict(celu=torch.nn.CELU,
                                          elu=torch.nn.ELU,
                                          gelu=torch.nn.GELU,
                                          hard_shrink=torch.nn.Hardshrink,
                                          hard_sigmoid=torch.nn.Hardsigmoid,
                                          hard_swish=torch.nn.Hardswish,
                                          hard_tanh=torch.nn.Hardtanh,
                                          leakly_relu=torch.nn.LeakyReLU,
                                          linear=torch.nn.Linear,
                                          log_sigmoid=torch.nn.LogSigmoid,
                                          prelu=torch.nn.PReLU,
                                          rrelu=torch.nn.RReLU,
                                          relu=torch.nn.ReLU,
                                          selu=torch.nn.SELU,
                                          sigmoid=torch.nn.Sigmoid,
                                          silu=torch.nn.SiLU,
                                          soft_plus=torch.nn.Softplus,
                                          soft_shrink=torch.nn.Softshrink,
                                          soft_sign=torch.nn.Softsign,
                                          tanh=torch.nn.Tanh,
                                          tanh_shrink=torch.nn.Tanhshrink
                                          ),
                        others=dict(log_softmax=torch.nn.LogSoftmax,
                                    softmin=torch.nn.Softmin,
                                    softmax=torch.nn.Softmax,
                                    )
                        )
LOSS: dict = dict(reg=dict(mse=torch.nn.MSELoss(),
                           l1=torch.nn.L1Loss(),
                           l1_smooth=torch.nn.SmoothL1Loss(),
                           cosine_embedding=torch.nn.CosineEmbeddingLoss()
                           ),
                  clf_binary=dict(binary_cross_entropy=torch.nn.CrossEntropyLoss(),
                                  #hinge_embedding=torch.nn.functional.hinge_embedding_loss
                                  ),
                  clf_multi=dict(cross_entropy=torch.nn.CrossEntropyLoss(),
                                 #multilabel_margin=torch.nn.MultiLabelMarginLoss(),
                                 #multilabel_soft_margin=torch.nn.MultiLabelSoftMarginLoss()
                                 )
                  )
OPTIMIZER: dict = dict(adam=torch.optim.Adam,
                       rmsprop=torch.optim.RMSprop,
                       sgd=torch.optim.SGD
                       )
EMBEDDING: dict = dict(fast_text=FastText)
TRANSFORMER: dict = {'roberta': 'roberta-large',
                     'xlm': 'xlm-mlm-100-1280',
                     'xlmroberta': 'xlm-roberta-large'
                     }
IGNORE_PARAM_FOR_OPTIMIZATION: List[str] = ['embedding_len',
                                            'weights',
                                            'vocab_size',
                                            'bidirectional',
                                            'early_stopping',
                                            'recurrent_network_type',
                                            'loss_torch',
                                            'optimizer_torch',
                                            'initializer_torch'
                                            ]
TORCH_OBJECT_PARAM: Dict[str, List[str]] = dict(activation=['activation'],
                                                initializer=['initializer'],
                                                optimizer=['optimizer',
                                                           'learning_rate',
                                                           'momentum',
                                                           'dampening',
                                                           'weight_decay',
                                                           'nesterov',
                                                           'alpha',
                                                           'eps',
                                                           'centered',
                                                           'betas',
                                                           'amsgrad'
                                                           ]
                                                )


def find_category_for_hidden_layers(hidden_layer: int) -> str:
    """
    Find hidden layer size category for given number of hidden layers

    :param hidden_layer: int
        Number of hidden layers

    :return: str
        Name of the hidden layer size category
    """
    for category in HIDDEN_LAYER_CATEGORY.keys():
        if hidden_layer <= HIDDEN_LAYER_CATEGORY.get(category)[1]:
            return category


class NeuralNetworkException(Exception):
    """
    Class for handling exceptions for class NeuralNetwork
    """
    pass


class NeuralNetwork:
    """
    Class for handling neural networks
    """
    def __init__(self,
                 target: str,
                 predictors: List[str],
                 output_layer_size: int = None,
                 x_train: np.ndarray = None,
                 y_train: np.ndarray = None,
                 x_test: np.ndarray = None,
                 y_test: np.ndarray = None,
                 x_val: np.ndarray = None,
                 y_val: np.ndarray = None,
                 train_data_path: str = None,
                 test_data_path: str = None,
                 validation_data_path: str = None,
                 sequential_type: str = 'text',
                 input_param: dict = None,
                 model_param: dict = None,
                 cache_dir: str = None,
                 seed: int = 1234,
                 **kwargs
                 ):
        """
        :param target: str
            Name of the target feature

        :param output_layer_size: int
            Number of neurons in output layer
                -> 1: Regression
                -> 2: Binary Classification
                -> 3: Multi-Classification

        :param x_train: np.array

        :param predictors: List[str]
            Name of the predictor features

        :param sequential_type: str
            Name of the sequence type:
                -> text: Text data
                -> numeric: Numeric data

        :param train_data_path: str
            Complete file path of the training data

        :param test_data_path: str
            Complete file path of the testing data

        :param validation_data_path: str
            Complete file path of the validation data

        :param model_param: dict
            Pre-configured model parameter

        :param cache_dir: str
            Cache directory for loading pre-trained language (embedding) models from disk (locally)

        :param seed: int
            Seed

        :param kwargs: dict
            Key-word arguments for configuring PyTorch parameter settings
        """
        self.target: str = target
        self.output_size: int = output_layer_size
        self.predictors: List[str] = predictors
        self.x_train: np.ndarray = x_train
        self.y_train: np.array = y_train
        self.x_test: np.ndarray = x_test
        self.y_test: np.array = y_test
        self.x_val: np.ndarray = x_val
        self.y_val: np.array = y_val
        self.train_data_df: pd.DataFrame = None
        self.test_data_df: pd.DataFrame = None
        self.val_data_df: pd.DataFrame = None
        self.train_data_path: str = train_data_path
        self.test_data_path: str = test_data_path
        self.validation_data_path: str = validation_data_path
        if self.x_train is None or self.y_train is None:
            if self.train_data_path is None:
                raise NeuralNetworkException('No training data found')
        if self.x_val is None or self.y_val is None:
            if self.validation_data_path is None:
                raise NeuralNetworkException('No validation data found')
        self.model = None
        self.input_param: dict = {} if input_param is None else input_param
        self.model_param: dict = {} if model_param is None else model_param
        self.model_param_mutated: dict = {}
        self.model_param_mutation: str = ''
        self.train_iter = None
        self.test_iter = None
        self.validation_iter = None
        self.sequential_type: str = sequential_type
        self.cache_dir: str = cache_dir
        self.seed: int = 1234 if seed <= 0 else seed

    def attention_network(self) -> Attention:
        """
        Config Attention Network

        :return: Attention
            Model object
        """
        return Attention(parameters=self.model_param, output_size=self.output_size)

    @staticmethod
    def attention_network_param() -> dict:
        """
        Generate Long-Short Term Memory Network + Attention classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(hidden_states=np.random.choice(a=HappyLearningUtils().geometric_progression()),
                    bidirectional=np.random.choice(a=[False, True])
                    )

    @staticmethod
    def convolutional_neural_network_param() -> dict:
        """
        Generate Convolutional Neural Network (CNN) classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(padding=np.random.choice(a=HappyLearningUtils().geometric_progression()),
                    stride=np.random.choice(a=HappyLearningUtils().geometric_progression())
                    )

    def gated_recurrent_unit_network(self) -> GRU:
        """
        Config GRU Network

        :return: LSTM
            Model object
        """
        return GRU(parameters=self.model_param, output_size=self.output_size)

    @staticmethod
    def gated_recurrent_unit_network_param() -> dict:
        """
        Generate Gated Recurrent Unit Network (GRU) classifier parameter randomly

        :return: dict
            Parameter config
        """
        return dict(hidden_states=np.random.choice(a=HappyLearningUtils().geometric_progression()))

    def multi_layer_perceptron(self) -> MLP:
        """
        Generate Multi-Layer Perceptron (MLP) classifier parameter configuration randomly

        :return dict:
            Configured Multi-Layer Perceptron (MLP) hyper parameter set
        """
        return MLP(input_size=len(self.predictors), output_size=self.output_size, parameters=self.model_param)

    @staticmethod
    def multi_layer_perceptron_param() -> dict:
        """
        Generate Multi-Layer Perceptron (MLP) classifier parameter configuration randomly

        :return dict:
            Configured Multi-Layer Perceptron (MLP) hyper parameter set
        """
        return dict(hidden_neurons=np.random.choice(a=HappyLearningUtils().geometric_progression()))

    def long_short_term_memory_network(self) -> LSTM:
        """
        Config LSTM Network

        :return: LSTM
            Model object
        """
        return LSTM(parameters=self.model_param, output_size=self.output_size)

    @staticmethod
    def long_short_term_memory_network_param() -> dict:
        """
        Generate Long-Short Term Memory Network (LSTM) classifier parameter randomly

        :return: dict
            Parameter config
        """
        return dict(hidden_states=np.random.choice(a=HappyLearningUtils().geometric_progression()),
                    bidirectional=np.random.choice(a=[False, True])
                    )

    def recurrent_neural_network(self) -> RNN:
        """
        Config RNN Network

        :return: RNN
            Model object
        """
        return RNN(parameters=self.model_param, output_size=self.output_size)

    @staticmethod
    def recurrent_neural_network_param() -> dict:
        """
        Generate Recurrent Neural Network (RNN) classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(hidden_states=np.random.choice(a=HappyLearningUtils().geometric_progression()))

    def recurrent_convolutional_neural_network(self) -> RCNN:
        """
        Generate Recurrent Convolutional Neural Network (RCNN) classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return RCNN(parameters=self.model_param, output_size=self.output_size)

    def recurrent_convolutional_neural_network_param(self) -> dict:
        """
        Generate Recurrent Convolutional Neural Network (RCNN) classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(hidden_states=np.random.choice(a=HappyLearningUtils().geometric_progression(n=2 if len(self.model_param_mutated.keys()) == 0 else 8)),
                    recurrent_network_type='lstm', #np.random.choice(a=['gru', 'lstm', 'rnn']),
                    bidirectional=np.random.choice(a=[False, True])
                    )

    def self_attention_network(self) -> SelfAttention:
        """
        Config Self-Attention Network

        :return: SelfAttention
            Model object
        """
        return SelfAttention(parameters=self.model_param, output_size=self.output_size)

    @staticmethod
    def self_attention_network_param() -> dict:
        """
        Generate Attention Network classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(hidden_states=np.random.choice(a=HappyLearningUtils().geometric_progression()))

    def transformer(self) -> Transformers:
        """
        Config Transformer Network

        :return: Transformers
            Model object
        """
        return Transformers(parameters=self.model_param, output_size=self.output_size, cache_dir=self.cache_dir)

    def transformer_param(self) -> dict:
        """
        Generate Transformer Network classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        _model_type: str = np.random.choice(a=list(TRANSFORMER.keys())) if self.cache_dir is None else self.cache_dir.split('/')[-2].split('-')[0]
        _num_attention_heads: int = np.random.randint(low=2, high=20)
        _hidden_size: int = _num_attention_heads * np.random.randint(low=100, high=1000)
        _batch_size: int = int(np.random.choice(a=HappyLearningUtils().geometric_progression(n=6))) if torch.cuda.is_available() else int(np.random.choice(a=HappyLearningUtils().geometric_progression(n=8)))
        return dict(model_type=_model_type,
                    model_name=TRANSFORMER.get(_model_type) if self.cache_dir is None else self.cache_dir,
                    epoch=np.random.randint(low=1, high=20),
                    batch_size=_batch_size,
                    learning_rate=np.random.uniform(low=0.00001, high=0.4),
                    #adafactor_beta1=np.random.uniform(low=0.0, high=1),
                    #adafactor_clip_threshold=np.random.uniform(low=0.001, high=1),
                    #adafactor_decay_rate=np.random.uniform(low=-0.001, high=0.999),
                    #adafactor_eps=(np.random.uniform(low=1e-50, high=1e-10), np.random.uniform(low=1e-50, high=1e-10)),
                    #adafactor_relative_step=False,#np.random.choice(a=[False, True]),
                    #adafactor_scale_parameter=np.random.choice(a=[False, True]),
                    #adafactor_warmup_init=False,#np.random.choice(a=[False, True]),
                    adam_epsilon=np.random.uniform(low=1e-10, high=1e-5),
                    cosine_schedule_num_cycles=np.random.uniform(low=0.3, high=0.7),
                    dynamic_quantize=False,#np.random.choice(a=[False, True]),
                    early_stopping_consider_epochs=np.random.choice(a=[False, True]),
                    use_early_stopping=np.random.choice(a=[False, True]),
                    early_stopping_delta=np.random.uniform(low=0, high=0.3),
                    early_stopping_patience=np.random.randint(low=3, high=10),
                    attention_probs_dropout_prob=np.random.uniform(low=0.05, high=0.95),
                    hidden_size=_hidden_size,
                    hidden_dropout_prob=np.random.uniform(low=0.05, high=0.95),
                    gradient_accumulation_steps=1,#np.random.randint(low=1, high=3),
                    initializer_range=np.random.uniform(low=0.05, high=0.95),
                    layer_norm_eps=np.random.uniform(low=0.05, high=0.5),
                    num_attention_heads=_num_attention_heads,
                    num_hidden_layers=np.random.randint(low=2, high=30),
                    warmup_ratio=np.random.uniform(low=0.03, high=0.15),
                    optimizer='AdamW',#np.random.choice(a=['AdamW', 'Adafactor']),
                    scheduler='linear_schedule_with_warmup',#np.random.choice(a=['constant_schedule', 'constant_schedule_with_warmup', 'linear_schedule_with_warmup', 'cosine_schedule_with_warmup', 'cosine_with_hard_restarts_schedule_with_warmup', 'polynomial_decay_schedule_with_warmup']),
                    polynomial_decay_schedule_lr_end=np.random.uniform(low=1e-8, high=1e-4),
                    polynomial_decay_schedule_power=np.random.uniform(low=0.5, high=1.0),
                    max_grad_norm=np.random.uniform(low=0.01, high=1.0),
                    weight_decay=np.random.randint(low=0, high=1)
                    )


class NetworkGenerator(NeuralNetwork):
    """
    Class for generating neural networks using PyTorch
    """
    def __init__(self,
                 target: str,
                 predictors: List[str],
                 output_layer_size: int = None,
                 x_train: np.ndarray = None,
                 y_train: np.ndarray = None,
                 x_test: np.ndarray = None,
                 y_test: np.ndarray = None,
                 x_val: np.ndarray = None,
                 y_val: np.ndarray = None,
                 train_data_path: str = None,
                 test_data_path: str = None,
                 validation_data_path: str = None,
                 sequential_type: str = 'text',
                 learning_type: str = 'batch',
                 model_name: str = None,
                 model_param: dict = None,
                 input_param: dict = None,
                 hidden_layer_size: int = 0,
                 hidden_layer_size_category: str = None,
                 models: List[str] = None,
                 sep: str = '\t',
                 cloud: str = None,
                 cache_dir: str = None,
                 seed: int = 1234
                 ):
        """
        :param models: List[str]
            Names of the possible models to sample from

        :param learning_type: str
            Name of the learning method:
                -> batch: using hole data set using batches in each epoch
                -> stochastic: using sample of the data set in each epoch

        :param hidden_layer_size_category: str
            Name of the hidden layer size category
                -> very_small: 1 - 2 hidden layers
                -> small: 3 - 4 hidden layers
                -> medium: 5 - 7 hidden layers
                -> big: 8 - 10 hidden layers
                -> very_big: 11+ hidden layers

        :param hidden_layer_size: int
            Number of hidden layers

        :param sep: str
            Separator

        :param cloud: str
            Name of the cloud provider
                -> google: Google Cloud Storage
                -> aws: AWS Cloud

        :param cache_dir: str
            Cache directory for loading pre-trained language (embedding) models from disk (locally)

        :param seed: int
            Seed
        """
        super().__init__(target=target,
                         predictors=predictors,
                         output_layer_size=output_layer_size,
                         x_train=x_train,
                         y_train=y_train,
                         x_test=x_test,
                         y_test=y_test,
                         x_val=x_val,
                         y_val=y_val,
                         train_data_path=train_data_path,
                         test_data_path=test_data_path,
                         validation_data_path=validation_data_path,
                         sequential_type=sequential_type,
                         model_param=model_param,
                         input_param=input_param,
                         seed=seed
                         )
        self.id: int = 0
        self.fitness_score: float = 0.0
        self.fitness: dict = {}
        if self.output_size == 1:
            self.target_type: str = 'reg'
        elif self.output_size == 2:
            self.target_type: str = 'clf_binary'
        else:
            self.target_type: str = 'clf_multi'
        self.models: List[str] = models
        self.model_name: str = model_name
        self.transformer: bool = True if model_name == 'trans' else False
        if self.model_name is None:
            self.random: bool = True
            if self.models is not None:
                for model in self.models:
                    if model not in list(NETWORK_TYPE.keys()):
                        self.random: bool = False
                        raise NeuralNetworkException('Model ({}) is not supported. Supported classification models are: {}'.format(model, list(NETWORK_TYPE.keys())))
        else:
            if self.model_name not in list(NETWORK_TYPE.keys()):
                raise NeuralNetworkException('Model ({}) is not supported. Supported classification models are: {}'.format(self.model_name, list(NETWORK_TYPE.keys())))
            else:
                self.random: bool = False
        self.features: List[str] = []
        self.train_time = None
        self.obs: list = []
        self.pred: list = []
        self.sep: str = sep
        self.cloud: str = cloud
        self.cache_dir: str = cache_dir
        if self.cloud is None:
            self.bucket_name: str = None
        else:
            if self.cloud not in CLOUD_PROVIDER:
                raise NeuralNetworkException('Cloud provider ({}) not supported'.format(cloud))
            self.bucket_name: str = self.train_data_path.split("//")[1].split("/")[0]
        self.embedding_text = None
        self.embedding_label = None
        self.hidden_layer_size: int = hidden_layer_size
        self.hidden_layer_size_category: str = hidden_layer_size_category
        self.learning_type: str = learning_type if learning_type in ['batch', 'stochastic'] else 'batch'
        self.batch_eval: Dict[str, dict] = dict(train=dict(batch_loss=[],
                                                           batch_metric=[],
                                                           total_batch_loss=0,
                                                           total_batch_metric=0
                                                           ),
                                                val=dict(batch_loss=[],
                                                         batch_metric=[],
                                                         total_batch_loss=0,
                                                         total_batch_metric=0
                                                         ),
                                                test=dict(batch_loss=[],
                                                          batch_metric=[],
                                                          total_batch_loss=0,
                                                          total_batch_metric=0
                                                          )
                                                )
        self.epoch_eval: Dict[str, dict] = dict(train=dict(), val=dict(), test=dict())

    def _batch_evaluation(self, iter_type: str, loss_value: float, metric_value: float):
        """
        Evaluation of each training batch

        :param iter_type: str
            Name of the iteration process:
                -> train: Training iteration
                -> test: Testing iteration
                -> val: Validation iteration

        :param loss_value: float
            Loss value of current batch

        :param metric_value: float
            Metric value of current batch
        """
        self.batch_eval[iter_type]['batch_loss'].append(loss_value)
        self.batch_eval[iter_type]['batch_metric'].append(metric_value)
        self.batch_eval[iter_type]['total_batch_loss'] += self.batch_eval[iter_type]['batch_loss'][-1]
        self.batch_eval[iter_type]['total_batch_metric'] += self.batch_eval[iter_type]['batch_metric'][-1]

    def _batch_learning(self, train: bool = True, eval_set: str = 'val'):
        """
        Train gradient using batch learning

        :param train: bool
            Training or evaluation mode

        :param eval_set: str
            Evaluation data set:
                -> val: Validation data set
                -> test: Test data set
        """
        if train:
            _predictions: List[int] = []
            _observations: List[int] = []
            if self.target_type == 'reg':
                _eval = EvalReg
            else:
                _eval = EvalClf
            if torch.cuda.is_available():
                self.model.cuda()
            self.model.train()
            self._config_params(loss=True, optimizer=True)
            _optim: torch.optim = self.model_param.get('optimizer_torch')
            for idx, batch in enumerate(self.train_iter):
                if self.sequential_type == 'text':
                    _predictors = batch.text[0]
                    _target = batch.label
                else:
                    _predictors, _target = batch
                if _target.size()[0] != self.model_param.get('batch_size'):
                    continue
                if torch.cuda.is_available():
                    _target = _target.cuda()
                    _predictors = _predictors.cuda()
                _optim.zero_grad()
                _prediction = torch.softmax(input=self.model(_predictors), dim=1)
                _, _pred = torch.max(input=_prediction, dim=1)
                _loss = self.model_param.get('loss_torch')(_prediction, _target)
                _loss.backward()
                self._clip_gradient(self.model, 1e-1)
                _optim.step()
                _predictions.extend(_pred.detach().tolist())
                _observations.extend(_target.detach().numpy().tolist())
                self._batch_evaluation(iter_type='train',
                                       loss_value=_loss.item(),
                                       metric_value=getattr(_eval(obs=copy.deepcopy(_target.detach().numpy().tolist()),
                                                                  pred=copy.deepcopy(_pred.detach().tolist())
                                                                  ),
                                                            SML_SCORE['ml_metric'][self.target_type]
                                                            )()
                                       )
            if len(_observations) == len(_predictions):
                self._eval(iter_type='train', obs=_observations, pred=_predictions)
        else:
            if eval_set == 'val':
                _iter_type: str = 'val'
                _eval_iter = self.validation_iter
            else:
                _iter_type: str = 'test'
                _eval_iter = self.test_iter
            if self.target_type == 'reg':
                _eval = EvalReg
            else:
                _eval = EvalClf
            if torch.cuda.is_available():
                self.model.cuda()
            self.model.eval()
            with torch.no_grad():
                _predictions: List[int] = []
                _observations: List[int] = []
                for idx, batch in enumerate(_eval_iter):
                    if self.sequential_type == 'text':
                        _predictors = batch.text[0]
                        _target = batch.label
                    else:
                        _predictors, _target = batch
                    if torch.cuda.is_available():
                        _target = _target.cuda()
                        _predictors = _predictors.cuda()
                    if _target.size()[0] != self.model_param.get('batch_size'):
                        continue
                    _prediction = torch.softmax(input=self.model(_predictors), dim=1)
                    _, _pred = torch.max(input=_prediction, dim=1)
                    _loss = self.model_param.get('loss_torch')(_prediction, _target)
                    _predictions.extend(_pred.detach().tolist())
                    _observations.extend(_target.detach().numpy().tolist())
                    self._batch_evaluation(iter_type=_iter_type,
                                           loss_value=_loss.item(),
                                           metric_value=getattr(
                                               _eval(obs=copy.deepcopy(_target.detach().numpy().tolist()),
                                                     pred=copy.deepcopy(_pred.detach().tolist())
                                                     ),
                                               SML_SCORE['ml_metric'][self.target_type]
                                               )()
                                           )
                if len(_observations) == len(_predictions):
                    self.obs = copy.deepcopy(_observations)
                    self.pred = copy.deepcopy(_predictions)
                    self._eval(iter_type=_iter_type, obs=_observations, pred=_predictions)

    @staticmethod
    def _clip_gradient(model, clip_value: float):
        """
        Clip gradient during network training

        :param model:
        :param clip_value:
        """
        params = list(filter(lambda p: p.grad is not None, model.parameters()))
        for p in params:
            p.grad.data.clamp_(-clip_value, clip_value)

    def _config_activation_functions(self, hidden_layers: int = 0):
        """
        Configure activation functions

        :param hidden_layers: int
            Number of hidden layers
        """
        if hidden_layers > 0:
            _layers: int = hidden_layers
            _param_names: List[str] = ['hidden_layer_{}_activation'.format(hl) for hl in range(1, hidden_layers + 1, 1)]
        else:
            _layers: int = 1
            _param_names: List[str] = ['activation_torch']
        _activation_torch: dict = dict()
        for layer in range(0, _layers, 1):
            if self.model_param.get('activation') == 'celu':
                _activation_torch.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')](alpha=1.0 if self.model_param.get('alpha') is None else self.model_param.get('alpha'))})
            elif self.model_param.get('activation') == 'elu':
                _activation_torch.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')](alpha=1.0 if self.model_param.get('alpha') is None else self.model_param.get('alpha'))})
            elif self.model_param.get('activation') == 'hard_shrink':
                _activation_torch.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')](lambd=0.5 if self.model_param.get('lambd') is None else self.model_param.get('lambd'))})
            elif self.model_param.get('activation') == 'hard_tanh':
                _activation_torch.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')](min_val=-1.0 if self.model_param.get('min_val') is None else self.model_param.get('min_val'),
                                                                                                                              max_val=1.0 if self.model_param.get('max_val') is None else self.model_param.get('max_val')
                                                                                                                              )
                                          })
            elif self.model_param.get('activation') == 'leaky_relu':
                _activation_torch.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')](negative_slope=0.01 if self.model_param.get('negative_slope') is None else self.model_param.get('negative_slope'))})
            elif self.model_param.get('activation') == 'prelu':
                _activation_torch.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')](num_parameters=1 if self.model_param.get('num_parameters') is None else self.model_param.get('num_parameters'),
                                                                                                                              init=0.25 if self.model_param.get('init') is None else self.model_param.get('init')
                                                                                                                              )
                                          })
            elif self.model_param.get('activation') == 'rrelu':
                _activation_torch.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')](lower=0.125 if self.model_param.get('lower') is None else self.model_param.get('lower'),
                                                                                                                              upper=0.3333333333333333 if self.model_param.get('upper') is None else self.model_param.get('upper')
                                                                                                                              )
                                          })
            elif self.model_param.get('activation') == 'soft_plus':
                _activation_torch.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')](beta=1 if self.model_param.get('beta') is None else self.model_param.get('beta'),
                                                                                                                              threshold=20 if self.model_param.get('threshold') is None else self.model_param.get('threshold')
                                                                                                                              )
                                          })
            elif self.model_param.get('activation') == 'soft_shrink':
                _activation_torch.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')](lambd=0.5 if self.model_param.get('lambd') is None else self.model_param.get('lambd'))})
            else:
                self.model_param.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')]()})
        self.model_param.update(_activation_torch)

    def _config_hidden_layers(self):
        """
        Configure hyper parameters of the hidden layers
        """
        if self.hidden_layer_size_category is None:
            self.hidden_layer_size_category = np.random.choice(a=list(HIDDEN_LAYER_CATEGORY.keys()))
            _hidden_layer_size: tuple = HIDDEN_LAYER_CATEGORY.get(self.hidden_layer_size_category)
        else:
            _hidden_layer_size: tuple = HIDDEN_LAYER_CATEGORY.get(self.hidden_layer_size_category)
        if self.hidden_layer_size is None or self.hidden_layer_size <= 0:
            self.hidden_layer_size = np.random.randint(low=_hidden_layer_size[0], high=_hidden_layer_size[1])
        self.model_param.update({'num_hidden_layers': self.hidden_layer_size + 1})
        for hidden in range(1, self.hidden_layer_size + 1, 1):
            _hidden_layer_settings: dict = self._get_param_space(general=True)
            self.model_param.update({'hidden_layer_{}_neurons'.format(hidden): _hidden_layer_settings.get('hidden_neurons')})
            self.model_param.update({'hidden_layer_{}_dropout'.format(hidden): _hidden_layer_settings.get('dropout')})
            self.model_param.update({'hidden_layer_{}_alpha_dropout'.format(hidden): _hidden_layer_settings.get('alpha_dropout')})
            self.model_param.update({'hidden_layer_{}_activation'.format(hidden): _hidden_layer_settings.get('activation')})
            self.model_param.update({'hidden_layer_{}_rnn_network_type'.format(hidden): _hidden_layer_settings.get('rnn_network_type')})

    def _config_params(self,
                       loss: bool = False,
                       optimizer: bool = False,
                       activation: bool = False,
                       hidden_layers: bool = False,
                       natural_language: bool = False
                       ):
        """
        Finalize configuration of hyper parameter settings of the neural network

        :param loss: bool
            Configure loss function initially based on the size of the output layer

        :param optimizer: bool
            Configure optimizer

        :param activation: bool
            Configure activation functions for all layers

        :param hidden_layers: bool
            Configure hidden_layers initially

        :param natural_language: bool
            Configure pre-trained embedding or transformer models
        """
        if loss:
            if self.model_param.get('loss') is None:
                _loss: str = np.random.choice(a=list(LOSS.get(self.target_type).keys()))
                self.model_param.update(dict(loss_torch=LOSS[self.target_type][_loss]))
        if optimizer:
            if self.model_param.get('optimizer') == 'sgd':
                self.model_param.update({'optimizer_torch': torch.optim.SGD(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                                                                            lr=self.model_param.get('learning_rate'),
                                                                            momentum=0 if self.model_param.get('momentum') is None else self.model_param.get('momentum'),
                                                                            dampening=0 if self.model_param.get('dampening') is None else self.model_param.get('dampening'),
                                                                            weight_decay=0 if self.model_param.get('weight_decay') is None else self.model_param.get('weight_decay'),
                                                                            nesterov=False if self.model_param.get('nesterov') is None else self.model_param.get('nesterov')
                                                                            )
                                         })
            elif self.model_param.get('optimizer') == 'rmsprop':
                self.model_param.update({'optimizer_torch': torch.optim.RMSprop(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                                                                                lr=self.model_param.get('learning_rate'),
                                                                                alpha=0.99 if self.model_param.get('alpha') is None else self.model_param.get('alpha'),
                                                                                eps=1e-08 if self.model_param.get('eps') is None else self.model_param.get('eps'),
                                                                                weight_decay=0 if self.model_param.get('weight_decay') is None else self.model_param.get('weight_decay'),
                                                                                momentum=0 if self.model_param.get('momentum') is None else self.model_param.get('momentum'),
                                                                                centered=False if self.model_param.get('centered') is None else self.model_param.get('centered')
                                                                                )
                                         })
            elif self.model_param.get('optimizer') == 'adam':
                self.model_param.update({'optimizer_torch': torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                                                                             lr=self.model_param.get('learning_rate'),
                                                                             betas=(0.9, 0.999) if self.model_param.get('betas') is None else self.model_param.get('betas'),
                                                                             eps=1e-08 if self.model_param.get('eps') is None else self.model_param.get('eps'),
                                                                             weight_decay=0.0 if self.model_param.get('weight_decay') is None else self.model_param.get('weight_decay'),
                                                                             amsgrad=False if self.model_param.get('amsgrad') is None else self.model_param.get('amsgrad')
                                                                             )
                                         })
        if activation:
            self._config_activation_functions()
        if hidden_layers:
            self._config_hidden_layers()
        if natural_language:
            _special_settings: dict = self._get_param_space(general=False)
            self.model_param.update(_special_settings['embedding'])

    def _epoch_eval(self, iter_types: List[str]):
        """
        Evaluation of each training epoch

        :param iter_types: List[str]
            Names of the iteration process:
                -> train: Training iteration
                -> test: Testing iteration
                -> val: Validation iteration
        """
        for iter_type in iter_types:
            for metric in self.fitness[iter_type].keys():
                if metric in self.epoch_eval[iter_type].keys():
                    self.epoch_eval[iter_type][metric].append(self.fitness[iter_type][metric])
                else:
                    self.epoch_eval[iter_type].update({metric: [self.fitness[iter_type][metric]]})

    def _eval(self, iter_type: str, obs: list, pred: list):
        """
        Internal evaluation for applying machine learning metric methods

        :param iter_type: str
            Name of the iteration process:
                -> train: Training iteration
                -> test: Testing iteration
                -> val: Validation iteration

        :param obs: list
            Observations of target feature

        :param pred: list
            Predictions
        """
        self.fitness.update({iter_type: {}})
        _target_type: str = 'clf' if self.output_size > 1 else 'reg'
        if self.output_size == 1:
            for metric in ML_METRIC.get('reg'):
                self.fitness[iter_type].update({metric: copy.deepcopy(getattr(EvalReg(obs=obs, pred=pred), metric)())})
        else:
            if self.output_size == 2:
                _eval_metric: List[str] = ML_METRIC.get('clf_binary')
            else:
                _eval_metric: List[str] = ML_METRIC.get('clf_multi')
            for metric in _eval_metric:
                self.fitness[iter_type].update({metric: copy.deepcopy(getattr(EvalClf(obs=obs, pred=pred), metric)())})

    def _get_param_space(self, general: bool = True) -> dict:
        """
        Get randomly drawn hyper parameter settings

        :param general: bool
            Get settings either of general hyper parameters or special hyper parameters like embeddings or transformers

        :return dict:
            Hyper parameter settings
        """
        if general:
            return dict(hidden_neurons=np.random.choice(a=HappyLearningUtils().geometric_progression(n=2 if len(self.model_param_mutated.keys()) > 0 else 2)),
                        learning_rate=np.random.uniform(low=0.00001, high=0.4),
                        batch_size=np.random.choice(a=HappyLearningUtils().geometric_progression(n=8)),
                        sample_size=np.random.choice(a=HappyLearningUtils().geometric_progression(n=8)),
                        epoch=np.random.randint(low=3, high=10) if len(self.model_param_mutated.keys()) == 0 else np.random.randint(low=5, high=15),
                        #early_stopping=[False, True],
                        #patience=np.random.uniform(low=2, high=20),
                        dropout=np.random.uniform(low=0.05, high=0.95),
                        alpha_dropout=np.random.choice(a=[False, True]),
                        activation=np.random.choice(a=list(ACTIVATION['weighted_sum'].keys())),
                        optimizer=np.random.choice(a=list(OPTIMIZER.keys()))
                        )
        else:
            return dict(embedding=dict(embedding_len=300,
                                       embedding_model=np.random.choice(a=list(EMBEDDING.keys()))
                                       )
                        )

    def _import_data_torch(self):
        """
        Import data sets (Training, Testing, Validation) from file
        """
        if self.transformer:
            self.train_data_df = DataImporter(file_path=self.train_data_path,
                                              as_data_frame=True,
                                              use_dask=False,
                                              create_dir=False,
                                              sep=self.sep,
                                              cloud=self.cloud,
                                              bucket_name=self.bucket_name
                                              ).file(table_name=None)
            self.train_data_df[self.target] = pd.to_numeric(self.train_data_df[self.target])
            self.test_data_df = DataImporter(file_path=self.test_data_path,
                                             as_data_frame=True,
                                             use_dask=False,
                                             create_dir=False,
                                             sep=self.sep,
                                             cloud=self.cloud,
                                             bucket_name=self.bucket_name
                                             ).file(table_name=None)
            self.test_data_df[self.target] = pd.to_numeric(self.test_data_df[self.target])
            self.val_data_df = DataImporter(file_path=self.validation_data_path,
                                            as_data_frame=True,
                                            use_dask=False,
                                            create_dir=False,
                                            sep=self.sep,
                                            cloud=self.cloud,
                                            bucket_name=self.bucket_name
                                            ).file(table_name=None)
            self.val_data_df[self.target] = pd.to_numeric(self.val_data_df[self.target])
        else:
            if self.learning_type == 'batch':
                if self.sequential_type == 'text':
                    _unique_labels: list = pd.read_csv(filepath_or_buffer=self.train_data_path, sep=self.sep)[self.target].unique().tolist()
                    _data_fields: List[tuple] = []
                    self.embedding_text: Field = Field(sequential=True,
                                                       tokenize=lambda x: x.split(),
                                                       lower=True,
                                                       include_lengths=True,
                                                       batch_first=True,
                                                       fix_length=300 if self.model_param.get('embedding_len') is None else self.model_param.get('embedding_len')
                                                       )
                    self.embedding_label: Field = Field(sequential=False, is_target=True, unk_token=None)
                    for predictor in self.predictors:
                        _data_fields.append((predictor, self.embedding_text))
                    _data_fields.append((self.target, self.embedding_label))
                    _train_data: TabularDataset = TabularDataset(path=self.train_data_path,
                                                                 format='csv',
                                                                 fields=_data_fields,
                                                                 skip_header=True
                                                                 )
                    if self.test_data_path is None:
                        _test_data = None
                    else:
                        _test_data: TabularDataset = TabularDataset(path=self.test_data_path,
                                                                    format='csv',
                                                                    fields=_data_fields,
                                                                    skip_header=True
                                                                    )
                    _validation_data: TabularDataset = TabularDataset(path=self.validation_data_path,
                                                                      format='csv',
                                                                      fields=_data_fields,
                                                                      skip_header=True
                                                                      )
                    if self.model_param.get('embedding_model') == 'fast_text':
                        self.embedding_text.build_vocab(_train_data,
                                                        vectors=EMBEDDING[self.model_param.get('embedding_model')](language='de' if self.model_param.get('lang') is None else self.model_param.get('lang'))
                                                        )
                    self.embedding_label.build_vocab()
                    if 0 in _unique_labels:
                        for label in _unique_labels:
                            self.embedding_label.vocab.stoi.update({str(label): label})
                    else:
                        for label in _unique_labels:
                            self.embedding_label.vocab.stoi.update({str(label): label - 1})
                    self.embedding_label.build_vocab(_train_data)
                    self.model_param.update(dict(weights=self.embedding_text.vocab.vectors, vocab_size=len(self.embedding_text.vocab)))
                    if self.test_data_path is None:
                        self.train_iter, self.validation_iter = BucketIterator.splits((_train_data, _validation_data),
                                                                                      batch_size=int(self.model_param.get('batch_size')),
                                                                                      sort_key=lambda x: len(x.text),
                                                                                      repeat=False,
                                                                                      shuffle=True
                                                                                      )
                    else:
                        self.train_iter, self.validation_iter, self.test_iter = BucketIterator.splits((_train_data, _validation_data, _test_data),
                                                                                                      batch_size=int(self.model_param.get('batch_size')),
                                                                                                      sort_key=lambda x: len(x.text),
                                                                                                      repeat=False,
                                                                                                      shuffle=True
                                                                                                      )
                else:
                    self.train_data_df = DataImporter(file_path=self.train_data_path,
                                                      as_data_frame=True,
                                                      use_dask=False,
                                                      create_dir=False,
                                                      sep=self.sep,
                                                      cloud=self.cloud,
                                                      bucket_name=self.bucket_name
                                                      ).file(table_name=None)
                    self.test_data_df = DataImporter(file_path=self.test_data_path,
                                                     as_data_frame=True,
                                                     use_dask=False,
                                                     create_dir=False,
                                                     sep=self.sep,
                                                     cloud=self.cloud,
                                                     bucket_name=self.bucket_name
                                                     ).file(table_name=None)
                    self.val_data_df = DataImporter(file_path=self.validation_data_path,
                                                    as_data_frame=True,
                                                    use_dask=False,
                                                    create_dir=False,
                                                    sep=self.sep,
                                                    cloud=self.cloud,
                                                    bucket_name=self.bucket_name
                                                    ).file(table_name=None)
                    _train_predictor_tensor: torch.tensor = torch.tensor(data=self.train_data_df['text'].values)
                    _test_predictor_tensor: torch.tensor = torch.tensor(data=self.test_data_df['text'].values)
                    _val_predictor_tensor: torch.tensor = torch.tensor(data=self.val_data_df['text'].values)
                    _train_target_tensor: torch.tensor = torch.tensor(data=self.train_data_df['label'].values.astype(np.float32))
                    _test_target_tensor: torch.tensor = torch.tensor(data=self.test_data_df['label'].values.astype(np.float32))
                    _val_target_tensor: torch.tensor = torch.tensor(data=self.val_data_df['label'].values.astype(np.float32))
                    _train_data_tensor: TensorDataset = TensorDataset(_train_predictor_tensor, _train_target_tensor)
                    _test_data_tensor: TensorDataset = TensorDataset(_test_predictor_tensor, _test_target_tensor)
                    _val_data_tensor: TensorDataset = TensorDataset(_val_predictor_tensor, _val_target_tensor)
                    self.train_iter = DataLoader(dataset=_train_data_tensor, shuffle=True, batch_size=int(self.model_param.get('batch_size')))
                    self.test_iter = DataLoader(dataset=_test_data_tensor, shuffle=True, batch_size=int(self.model_param.get('batch_size')))
                    self.validation_iter = DataLoader(dataset=_val_data_tensor, shuffle=True, batch_size=int(self.model_param.get('batch_size')))
                    self.train_data_df = None
                    self.test_data_df = None
                    self.val_data_df = None
                    del _train_predictor_tensor, _train_target_tensor
                    del _test_predictor_tensor, _test_target_tensor
                    del _val_predictor_tensor, _val_target_tensor
                    del _train_data_tensor, _test_data_tensor, _val_data_tensor
            elif self.learning_type == 'stochastic':
                raise NeuralNetworkException('Importing data set for stochastic learning not implemented yet')
            else:
                raise NeuralNetworkException('Learning type ({}) not supported'.format(self.learning_type))
            self.x_train = None
            self.y_train = None
            self.x_test = None
            self.y_test = None
            self.x_val = None
            self.y_val = None
            del _train_data
            del _test_data
            del _validation_data

    def _predict_transformer(self, text_data: str):
        """
        Get prediction from pre-trained neural network (transformer only)

        :param text_data: str
            Text data sequence
        """
        _predictions, _raw_output = self.model.model.predict(to_predict=[text_data], multi_label=False if self.output_size <= 2 else True)
        return _predictions

    def _stochastic_learning(self, train: bool = True, eval_set: str = 'val'):
        """
        Train gradient using stochastic learning

        :param train: bool
            Training or evaluation mode

        :param eval_set: str
            Evaluation data set:
                -> val: Validation data set
                -> test: Test data set
        """
        if train:
            _predictions: List[int] = []
            _observations: List[int] = []
            if self.target_type == 'reg':
                _eval = EvalReg
            else:
                _eval = EvalClf
            if torch.cuda.is_available():
                self.model.cuda()
            self.model.train()
            self._config_params(loss=True, optimizer=True)
            _optim: torch.optim = self.model_param.get('optimizer_torch')
            for idx, sample in enumerate(self.train_iter):
                if self.sequential_type == 'text':
                    _predictors = sample.text[0]
                    _target = sample.label
                else:
                    _predictors, _target = sample
                if _target.size()[0] != self.model_param.get('sample_size'):
                    continue
                if torch.cuda.is_available():
                    _target = _target.cuda()
                    _predictors = _predictors.cuda()
                _optim.zero_grad()
                _prediction = torch.softmax(input=self.model(_predictors), dim=1)
                _, _pred = torch.max(input=_prediction, dim=1)
                _loss = self.model_param.get('loss_torch')(_prediction, _target)
                _loss.backward()
                self._clip_gradient(self.model, 1e-1)
                _optim.step()
                _predictions.extend(_pred.detach().tolist())
                _observations.extend(_target.detach().numpy().tolist())
                break
            if len(_observations) == len(_predictions):
                self._eval(iter_type='train', obs=_observations, pred=_predictions)
        else:
            if eval_set == 'val':
                _iter_type: str = 'val'
                _eval_iter = self.validation_iter
            else:
                _iter_type: str = 'test'
                _eval_iter = self.test_iter
            if self.target_type == 'reg':
                _eval = EvalReg
            else:
                _eval = EvalClf
            self.model.eval()
            with torch.no_grad():
                _predictions: List[int] = []
                _observations: List[int] = []
                for idx, sample in enumerate(_eval_iter):
                    if self.sequential_type == 'text':
                        _predictors = sample.text[0]
                        _target = sample.label
                    else:
                        _predictors, _target = sample
                    if torch.cuda.is_available():
                        _target = _target.cuda()
                        _predictors = _predictors.cuda()
                    if _target.size()[0] != self.model_param.get('sample_size'):
                        continue
                    _prediction = self.model(_predictors)
                    _, _pred = torch.softmax(input=torch.max(input=_prediction, dim=1))
                    _loss = self.model_param.get('loss_torch')(_prediction, _target)
                    _predictions.extend(_pred.detach().tolist())
                    _observations.extend(_target.detach().numpy().tolist())
                    break
                if len(_observations) == len(_predictions):
                    self.obs = copy.deepcopy(_observations)
                    self.pred = copy.deepcopy(_predictions)
                    self._eval(iter_type=_iter_type, obs=_observations, pred=_predictions)

    def _train_transformer(self):
        """
        Train neural network using deep learning framework 'PyTorch' (transformer only)
        """
        self.model.model.train_model(train_df=self.train_data_df,
                                     multi_label=False,
                                     output_dir=None,
                                     show_running_loss=False,
                                     eval_df=self.val_data_df,
                                     verbose=False
                                     )
        _predictions, _raw_output = self.model.model.predict(to_predict=self.train_data_df[self.predictors[0]].values.tolist())
        self._eval(iter_type='train', obs=self.train_data_df[self.target].values.tolist(), pred=_predictions.tolist())
        del _predictions
        del _raw_output

    def generate_model(self) -> object:
        """
        Generate supervised machine learning model with randomized parameter configuration

        :return object
            Model object itself (self)
        """
        if self.random:
            if self.models is None:
                self.model_name = copy.deepcopy(np.random.choice(a=list(NETWORK_TYPE.keys())))
            else:
                self.model_name = copy.deepcopy(np.random.choice(a=self.models))
            _model = copy.deepcopy(NETWORK_TYPE.get(self.model_name))
        else:
            _model = copy.deepcopy(NETWORK_TYPE.get(self.model_name))
        self.transformer = True if self.model_name == 'trans' else False
        if len(self.input_param.keys()) == 0:
            self.model_param = getattr(NeuralNetwork(target=self.target,
                                                     predictors=self.predictors,
                                                     output_layer_size=self.output_size,
                                                     x_train=self.x_train,
                                                     y_train=self.y_train,
                                                     x_test=self.x_test,
                                                     y_test=self.y_test,
                                                     x_val=self.x_val,
                                                     y_val=self.y_val,
                                                     train_data_path=self.train_data_path,
                                                     test_data_path=self.test_data_path,
                                                     validation_data_path=self.validation_data_path,
                                                     model_param=self.model_param
                                                     ),
                                       '{}_param'.format(_model)
                                       )()
            if not self.transformer:
                self.model_param.update(self._get_param_space(general=True))
                self._config_params(hidden_layers=True)
                if self.sequential_type == 'text':
                    self._config_params(natural_language=True)
        else:
            self.model_param = copy.deepcopy(self.input_param)
        _idx: int = 0 if len(self.model_param_mutated.keys()) == 0 else len(self.model_param_mutated.keys()) + 1
        self.model_param_mutated.update({str(_idx): {copy.deepcopy(self.model_name): {}}})
        for param in list(self.model_param.keys()):
            self.model_param_mutated[str(_idx)][copy.deepcopy(self.model_name)].update({param: copy.deepcopy(self.model_param.get(param))})
        self.model_param_mutation = 'params'
        if len(self.predictors) > 0:
            if self.target != '':
                self._import_data_torch()
            else:
                raise NeuralNetworkException('No target feature found')
        else:
            raise NeuralNetworkException('No predictors found')
        self.model = getattr(NeuralNetwork(target=self.target,
                                           predictors=self.predictors,
                                           output_layer_size=self.output_size,
                                           x_train=self.x_train,
                                           y_train=self.y_train,
                                           x_test=self.x_test,
                                           y_test=self.y_test,
                                           x_val=self.x_val,
                                           y_val=self.y_val,
                                           train_data_path=self.train_data_path,
                                           test_data_path=self.test_data_path,
                                           validation_data_path=self.validation_data_path,
                                           model_param=self.model_param
                                           ),
                             _model
                             )()
        return self

    def generate_params(self, param_rate: float = 0.1, force_param: dict = None) -> object:
        """
        Generate parameter for supervised learning models

        :param param_rate: float
            Rate of parameters of each model to mutate

        :param force_param: dict
            Parameter config to force explicitly

        :return object
            Model object itself (self)
        """
        if param_rate > 1:
            _rate: float = 1.0
        else:
            if param_rate > 0:
                _rate: float = param_rate
            else:
                _rate: float = 0.1
        _params: dict = getattr(NeuralNetwork(target=self.target,
                                              predictors=self.predictors,
                                              output_layer_size=self.output_size,
                                              x_train=self.x_train,
                                              y_train=self.y_train,
                                              x_test=self.x_test,
                                              y_test=self.y_test,
                                              x_val=self.x_val,
                                              y_val=self.y_val,
                                              train_data_path=self.train_data_path,
                                              test_data_path=self.test_data_path,
                                              validation_data_path=self.validation_data_path,
                                              sequential_type=self.sequential_type,
                                              input_param=self.input_param,
                                              model_param=self.model_param,
                                              seed=self.seed
                                              ),
                                '{}_param'.format(NETWORK_TYPE.get(self.model_name))
                                )()
        if self.transformer:
            _force_param: dict = {} if force_param is None else force_param
            _param_choices: List[str] = [p for p in list(_params.keys()) if p not in list(_force_param.keys())]
            _gen_n_params: int = round(len(_params.keys()) * _rate)
            if _gen_n_params == 0:
                _gen_n_params = 1
            self.model_param_mutated.update({len(self.model_param_mutated.keys()) + 1: {copy.deepcopy(self.model_name): {}}})
            for param in list(_force_param.keys()):
                self.model_param.update({param: copy.deepcopy(_force_param.get(param))})
            _old_model_param: dict = copy.deepcopy(self.model_param)
            for _ in range(0, _gen_n_params, 1):
                while True:
                    _param: str = np.random.choice(a=_param_choices)
                    if _old_model_param.get(_param) is not None:
                        if self.model_param.get(_param) is not None:
                            break
                self.model_param_mutated[list(self.model_param_mutated.keys())[-1]][copy.deepcopy(self.model_name)].update({_param: self.model_param.get(_param)})
            self.model_param_mutation = 'new_model'
        else:
            for fixed in ['hidden_layers', 'hidden_layer_size_category']:
                if fixed in list(self.model_param.keys()):
                    del _params[fixed]
            _force_param: dict = {} if force_param is None else force_param
            _param_choices: List[str] = [p for p in list(_params.keys()) if p not in list(_force_param.keys())]
            _gen_n_params: int = round(len(_params.keys()) * _rate)
            if _gen_n_params == 0:
                _gen_n_params = 1
            self.model_param_mutated.update({len(self.model_param_mutated.keys()) + 1: {copy.deepcopy(self.model_name): {}}})
            for param in list(_force_param.keys()):
                self.model_param.update({param: copy.deepcopy(_force_param.get(param))})
            _old_model_param: dict = copy.deepcopy(self.model_param)
            _ignore_param: List[str] = IGNORE_PARAM_FOR_OPTIMIZATION
            if self.learning_type == 'batch':
                _ignore_param.append('batch_size')
            elif self.learning_type == 'stochastic':
                _ignore_param.append('sample_size')
            _parameters: List[str] = [p for p in _param_choices if p not in _ignore_param]
            for _ in range(0, _gen_n_params, 1):
                while True:
                    _param: str = np.random.choice(a=_parameters)
                    if _old_model_param.get(_param) is not None:
                        if self.model_param.get(_param) is not None:
                            break
                if _param == 'loss':
                    self._config_params(loss=True)
                elif _param == 'optimizer':
                    self._config_params(optimizer=True)
                elif _param == 'hidden_layers':
                    self._config_params(hidden_layers=True)
                else:
                    if _param in self._get_param_space(general=True).keys():
                        self.model_param.update({_param: copy.deepcopy(self._get_param_space(general=True).get(_param))})
                    elif _param in self._get_param_space(general=False).keys():
                        self.model_param.update({_param: copy.deepcopy(self._get_param_space(general=False).get(_param))})
                self.model_param_mutated[list(self.model_param_mutated.keys())[-1]][copy.deepcopy(self.model_name)].update({_param: self.model_param.get(_param)})
            self.model_param_mutation = 'new_model'
        if len(self.predictors) > 0:
            if self.target != '':
                self._import_data_torch()
            else:
                raise NeuralNetworkException('No target feature found')
        else:
            raise NeuralNetworkException('No predictors found')
        self.model = getattr(NeuralNetwork(target=self.target,
                                           predictors=self.predictors,
                                           output_layer_size=self.output_size,
                                           x_train=self.x_train,
                                           y_train=self.y_train,
                                           x_test=self.x_test,
                                           y_test=self.y_test,
                                           x_val=self.x_val,
                                           y_val=self.y_val,
                                           train_data_path=self.train_data_path,
                                           test_data_path=self.test_data_path,
                                           validation_data_path=self.validation_data_path,
                                           model_param=self.model_param
                                           ),
                             NETWORK_TYPE.get(self.model_name)
                             )()
        return self

    def get_vanilla_model(self) -> object:
        """
        Get 'vanilla' typed neural network (one hidden layer only)

        :return object
            Model object itself (self)
        """
        if self.model_name is None:
            return self
        else:
            if self.transformer:
                if len(self.input_param.keys()) == 0:
                    self.model_param = getattr(NeuralNetwork(target=self.target,
                                                             predictors=self.predictors,
                                                             output_layer_size=self.output_size,
                                                             x_train=self.x_train,
                                                             y_train=self.y_train,
                                                             x_test=self.x_test,
                                                             y_test=self.y_test,
                                                             x_val=self.x_val,
                                                             y_val=self.y_val,
                                                             train_data_path=self.train_data_path,
                                                             test_data_path=self.test_data_path,
                                                             validation_data_path=self.validation_data_path
                                                             ),
                                               '{}_param'.format(NETWORK_TYPE.get(self.model_name))
                                               )()
                else:
                    self.model_param = copy.deepcopy(self.input_param)
                if len(self.predictors) > 0:
                    if self.target != '':
                        self._import_data_torch()
                    else:
                        raise NeuralNetworkException('No target feature found')
                else:
                    raise NeuralNetworkException('No predictors found')
                self.model = getattr(NeuralNetwork(target=self.target,
                                                   predictors=self.predictors,
                                                   output_layer_size=self.output_size,
                                                   x_train=self.x_train,
                                                   y_train=self.y_train,
                                                   x_test=self.x_test,
                                                   y_test=self.y_test,
                                                   x_val=self.x_val,
                                                   y_val=self.y_val,
                                                   train_data_path=self.train_data_path,
                                                   test_data_path=self.test_data_path,
                                                   validation_data_path=self.validation_data_path,
                                                   model_param=self.model_param
                                                   ),
                                     NETWORK_TYPE.get(self.model_name)
                                     )()
            else:
                if len(self.input_param.keys()) == 0:
                    self.model_param = getattr(NeuralNetwork(target=self.target,
                                                             predictors=self.predictors,
                                                             output_layer_size=self.output_size,
                                                             x_train=self.x_train,
                                                             y_train=self.y_train,
                                                             x_test=self.x_test,
                                                             y_test=self.y_test,
                                                             x_val=self.x_val,
                                                             y_val=self.y_val,
                                                             train_data_path=self.train_data_path,
                                                             test_data_path=self.test_data_path,
                                                             validation_data_path=self.validation_data_path
                                                             ),
                                               '{}_param'.format(NETWORK_TYPE.get(self.model_name))
                                               )()
                    self.model_param.update(self._get_param_space(general=True))
                    self.model_param.update(learning_rate=0.001)
                    if self.sequential_type == 'text':
                        self._config_params(natural_language=True)
                else:
                    self.model_param = copy.deepcopy(self.input_param)
                if len(self.predictors) > 0:
                    if self.target != '':
                        self._import_data_torch()
                    else:
                        raise NeuralNetworkException('No target feature found')
                else:
                    raise NeuralNetworkException('No predictors found')
            self.model = getattr(NeuralNetwork(target=self.target,
                                               predictors=self.predictors,
                                               output_layer_size=self.output_size,
                                               x_train=self.x_train,
                                               y_train=self.y_train,
                                               x_test=self.x_test,
                                               y_test=self.y_test,
                                               x_val=self.x_val,
                                               y_val=self.y_val,
                                               train_data_path=self.train_data_path,
                                               test_data_path=self.test_data_path,
                                               validation_data_path=self.validation_data_path,
                                               model_param=self.model_param
                                               ),
                                 NETWORK_TYPE.get(self.model_name)
                                 )()
        return self

    def eval(self, validation: bool = True):
        """
        Evaluate supervised machine learning classification model

        :param validation: bool
            Whether to run validation or testing iteration
        """
        if self.transformer:
            if validation:
                _predictions, _raw_output = self.model.model.predict(to_predict=self.val_data_df[self.predictors[0]].values.tolist())
                self._eval(iter_type='val', obs=self.val_data_df[self.target].values.tolist(), pred=_predictions)
            else:
                _predictions, _raw_output = self.model.model.predict(to_predict=self.test_data_df[self.predictors[0]].values.tolist())
                self._eval(iter_type='test', obs=self.test_data_df[self.target].values.tolist(), pred=_predictions)
            # _result, _predictions_val, _wrong_predictions = self.model.model.eval_model(eval_df=self.val_data_df,
            #                                                                            multi_label=False,
            #                                                                            output_dir=None,
            #                                                                            show_running_loss=True,
            #                                                                            verbose=True
            #                                                                            )
            del _predictions
            del _raw_output
        else:
            if self.learning_type == 'batch':
                self._batch_learning(train=False, eval_set='val' if validation else 'test')
            elif self.learning_type == 'stochastic':
                self._stochastic_learning()

    def predict(self):
        """
        Get prediction from pre-trained neural network using PyTorch
        """
        if self.test_iter is None:
            self.eval(validation=True)
        else:
            self.eval(validation=False)

    def save(self, file_path: str):
        """
        Save PyTorch model to disk

        :param file_path: str
            Complete file path of the PyTorch model to save
        """
        if self.transformer:
            self.model.save_model(output_dir=file_path,
                                  optimizer=None,
                                  scheduler=None,
                                  model=None,
                                  results=None
                                  )
        else:
            torch.save(obj=self.model, f=file_path)

    def train(self):
        """
        Train neural network using deep learning framework 'PyTorch'
        """
        _t0: datetime = datetime.now()
        if self.transformer:
            self._train_transformer()
            self.train_time = (datetime.now() - _t0).seconds
            self.eval(validation=True)
            self.eval(validation=False)
        else:
            for _ in range(0, self.model_param.get('epoch'), 1):
                print('\nEpoch: {}'.format(_))
                if self.learning_type == 'batch':
                    self._batch_learning(train=True)
                    self.eval(validation=True)
                elif self.learning_type == 'stochastic':
                    self._stochastic_learning()
                self._epoch_eval(iter_types=['train', 'val'])
            self.train_time = (datetime.now() - _t0).seconds

    def update_data(self,
                    x_train: np.ndarray,
                    y_train: np.array,
                    x_test: np.ndarray,
                    y_test: np.array,
                    x_val: np.ndarray,
                    y_val: np.array
                    ):
        """
        Update training, testing and validation data
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val
        self._import_data_torch()

    def update_model_param(self, param: dict):
        """
        Update model parameter config

        :param param: dict
        """
        if len(param.keys()) > 0:
            self.model_param.update(param)
