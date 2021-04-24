import numpy as np
import torch
import torch.nn as nn

from simpletransformers.model import ClassificationModel
from torch.autograd import Variable
from torch.nn import functional
from typing import List

INITIALIZER: dict = dict(#constant=torch.nn.init.constant_,
                         #eye=torch.nn.init.eye_,
                         #dirac=torch.nn.init.dirac_,
                         #kaiming_normal=torch.nn.init.kaiming_normal_,
                         #kaiming_uniform=torch.nn.init.kaiming_uniform_,
                         #normal=torch.nn.init.normal_,
                         ones=torch.nn.init.ones_,
                         #orthogonal=torch.nn.init.orthogonal_,
                         #sparse=torch.nn.init.sparse_,
                         #trunc_normal=torch.nn.init.trunc_normal_,
                         uniform=torch.nn.init.uniform_,
                         xavier_normal=torch.nn.init.xavier_normal_,
                         xavier_uniform=torch.nn.init.xavier_uniform_,
                         zeros=torch.nn.init.zeros_
                         )
CACHE_DIR: str = 'data/cache'
BEST_MODEL_DIR: str = 'data/best_model'
OUTPUT_DIR: str = 'data/output'


def set_initializer(x, **kwargs):
    """
    Setup weights and bias initialization

    :param x:
        Data set

    :param kwargs: dict
        Key-word arguments for configuring initializer parameters
    """
    if kwargs.get('initializer') == 'constant':
        INITIALIZER.get(kwargs.get('initializer'))(x, val=kwargs.get('val'))
    elif kwargs.get('initializer') == 'dirac':
        INITIALIZER.get(kwargs.get('initializer'))(x, groups=1 if kwargs.get('groups') is None else kwargs.get('groups'))
    elif kwargs.get('initializer') == 'kaiming_normal':
        INITIALIZER.get(kwargs.get('initializer'))(x,
                                                   a=0 if kwargs.get('a') is None else kwargs.get('a'),
                                                   mode='fan_in' if kwargs.get('mode') is None else kwargs.get('mode'),
                                                   nonlinearity='leaky_relu' if kwargs.get('nonlinearity') is None else kwargs.get('nonlinearity')
                                                   )
    elif kwargs.get('initializer') == 'kaiming_uniform':
        INITIALIZER.get(kwargs.get('initializer'))(x,
                                                   a=0 if kwargs.get('a') is None else kwargs.get('a'),
                                                   mode='fan_in' if kwargs.get('mode') is None else kwargs.get('mode'),
                                                   nonlinearity='leaky_relu' if kwargs.get('nonlinearity') is None else kwargs.get('nonlinearity')
                                                   )
    elif kwargs.get('initializer') == 'normal':
        INITIALIZER.get(kwargs.get('initializer'))(x,
                                                   mean=0.0 if kwargs.get('mean') is None else kwargs.get('mean'),
                                                   std=1.0 if kwargs.get('std') is None else kwargs.get('std'),
                                                   )
    elif kwargs.get('initializer') == 'orthogonal':
        INITIALIZER.get(kwargs.get('initializer'))(x, gain=1 if kwargs.get('mean') is None else kwargs.get('mean'))
    elif kwargs.get('initializer') == 'sparse':
        INITIALIZER.get(kwargs.get('initializer'))(x,
                                                   sparsity=np.random.uniform(low=0.01, high=0.99) if kwargs.get('sparsity') is None else kwargs.get('sparsity'),
                                                   std=0.01 if kwargs.get('std') is None else kwargs.get('std'),
                                                   )
    elif kwargs.get('initializer') == 'uniform':
        INITIALIZER.get(kwargs.get('initializer'))(x,
                                                   a=0.0 if kwargs.get('a') is None else kwargs.get('a'),
                                                   b=1.0 if kwargs.get('b') is None else kwargs.get('b'),
                                                   )
    elif kwargs.get('initializer') == 'xavier_normal':
        INITIALIZER.get(kwargs.get('initializer'))(x, gain=1.0 if kwargs.get('gain') is None else kwargs.get('gain'))
    elif kwargs.get('initializer') == 'xavier_uniform':
        INITIALIZER.get(kwargs.get('initializer'))(x, gain=1.0 if kwargs.get('gain') is None else kwargs.get('gain'))
    else:
        INITIALIZER.get(kwargs.get('initializer'))(x)


class PyTorchNetworkException(Exception):
    """
    Class for handling exceptions for class Attention, MLP, LSTM, RCNN, RNN, SelfAttention, Transformers
    """
    pass


class Attention(nn.Module):
    """
    Class for building deep learning attention network model using PyTorch
    """
    def __init__(self, parameters: dict, output_size: int):
        """
        :param parameters: dict
			Parameter settings

        :param output_size: int
            Output size:
                -> 1: Float value (Regression)
                -> 2: Classes (Binary Classification)
                -> >2: Classes (Multi-Class Classification)
        """
        super(Attention, self).__init__()
        self.params: dict = parameters
        self.output_size: int = output_size
        self.batch_size: int = self.params.get('batch_size')
        self.hidden_size: int = self.params.get('hidden_states')
        self.vocab_size: int = self.params.get('vocab_size')
        self.embedding_length: int = self.params.get('embedding_len')
        self.word_embeddings: torch.nn.Embedding = torch.nn.Embedding(self.vocab_size, self.embedding_length)
        self.word_embeddings.weights = nn.Parameter(self.params.get('weights'), requires_grad=False)
        self.lstm_layer = nn.LSTM(self.embedding_length, self.hidden_size)
        self.output_layer = nn.Linear(in_features=self.hidden_size, out_features=output_size)

    @staticmethod
    def attention_network(lstm_output, final_state):
        """
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        :param lstm_output: torch.tensor
            Final output of the LSTM which contains hidden layer outputs for each sequence

        :param final_state: torch.tensor
            Final time-step hidden state (h_n) of the LSTM

        :return: torch.tensor
            First computing weights for each of the sequence present in lstm_output and and then finally computing the new hidden state
        """
        _hidden = final_state.squeeze(0)
        _attention_weights = torch.bmm(lstm_output, _hidden.unsqueeze(2)).squeeze(2)
        _soft_attention_weights = torch.softmax(_attention_weights, 1)
        return torch.bmm(lstm_output.transpose(1, 2), _soft_attention_weights.unsqueeze(2)).squeeze(2)

    def forward(self, input_sentence):
        """
        :param input_sentence:
            Input text (batch_size, num_sequences)

        :return torch.tensor:
            Logits from the attention network
        """
        _input = self.word_embeddings(input_sentence)
        _input = _input.permute(1, 0, 2)
        _h_0 = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        _c_0 = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        _output, (final_hidden_state, final_cell_state) = self.lstm_layer(_input, (_h_0, _c_0))
        _output = _output.permute(1, 0, 2)
        _attn_output = self.attention_network(lstm_output=_output, final_state=final_hidden_state)
        return self.output_layer(_attn_output)


class GRU(nn.Module):
    """
	Class for building deep learning gated recurrent unit network model using PyTorch
	"""
    def __init__(self, parameters: dict, output_size: int):
        """
		:param parameters: dict
			Parameter settings

        :param output_size: int
            Output size:
                -> 1: Float value (Regression)
                -> 2: Classes (Binary Classification)
                -> >2: Classes (Multi-Class Classification)
		"""
        super(GRU, self).__init__()
        self.params: dict = parameters
        self.output_size: int = output_size
        self.batch_size: int = self.params.get('batch_size')
        self.hidden_size: int = self.params.get('hidden_states')
        self.vocab_size: int = self.params.get('vocab_size')
        self.embedding_length: int = self.params.get('embedding_len')
        self.word_embeddings: torch.nn.Embedding = torch.nn.Embedding(self.vocab_size, self.embedding_length)
        self.word_embeddings.weights = nn.Parameter(self.params.get('weights'), requires_grad=False)
        self.gru_layer = nn.GRU(self.embedding_length, self.hidden_size)
        self.output_layer = nn.Linear(in_features=self.hidden_size, out_features=output_size)

    def forward(self, input_sentence):
        """
		:param input_sentence:
			Input text (batch_size, num_sequences)

		:return torch.tensor:
			Logits
		"""
        _input = self.word_embeddings(input_sentence)
        _input = _input.permute(1, 0, 2)
        _h_0 = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        torch.nn.init.ones_(_h_0)
        _output, final_hidden_state = self.gru_layer(_input, _h_0)
        return self.output_layer(_output)


class MLP(torch.nn.Module):
    """
    Class for applying Multi Layer Perceptron (Fully Connected Layer) using PyTorch as a Deep Learning Framework
    """
    def __init__(self, parameters: dict, input_size: int, output_size: int
                 ):
        """
        :param parameters: dict
			Parameter settings

        :param input_size: int
            Number of input features

        :param output_size: int
            Output size:
                -> 1: Float value (Regression)
                -> 2: Classes (Binary Classification)
                -> >2: Classes (Multi-Class Classification)
        """
        super(MLP, self).__init__()
        self.params: dict = parameters
        self.hidden_layers: List[torch.torch.nn.Linear] = []
        self.dropout_layers: List[functional] = []
        self.use_alpha_dropout: bool = False
        self.activation_functions: List[functional] = []
        self.batch_size: int = self.params.get('batch_size')
        self.output_size: int = output_size
        if output_size == 1:
            self.activation_output: functional = functional.relu
        elif output_size == 2:
            self.activation_output: functional = functional.sigmoid
        else:
            self.activation_output: functional = functional.softmax
        if self.params is None:
            self.fully_connected_layer: torch.nn = torch.torch.nn.Linear(in_features=input_size,
                                                                         out_features=output_size,
                                                                         bias=True
                                                                         )
        else:
            _l: int = 0
            for layer in range(0, len(self.params.keys()), 1):
                if self.params.get('hidden_layer_{}_neurons'.format(layer)) is not None:
                    if self.params['hidden_layer_{}_alpha_dropout'.format(layer)]:
                        self.use_alpha_dropout = True
                    self.dropout_layers.append(self.params['hidden_layer_{}_dropout'.format(layer)])
                    self.activation_functions.append(self.params['hidden_layer_{}_activation'.format(layer)])
                    if len(self.hidden_layers) == 0:
                        self.hidden_layers.append(torch.torch.nn.Linear(in_features=input_size,
                                                                        out_features=self.params['hidden_layer_{}_neurons'.format(layer)],
                                                                        bias=True
                                                                        )
                                                  )
                    else:
                        if layer + 1 < len(self.params.keys()):
                            _l += 1
                            self.hidden_layers.append(torch.torch.nn.Linear(in_features=self.params['hidden_layer_{}_neurons'.format(layer - 1)],
                                                                            out_features=self.params['hidden_layer_{}_neurons'.format(layer)],
                                                                            bias=True
                                                                            )
                                                      )
                        else:
                            _l = layer
                #else:
                #    break
            if len(self.hidden_layers) == 0:
                self.fully_connected_layer: torch.nn = torch.torch.nn.Linear(in_features=input_size,
                                                                             out_features=output_size,
                                                                             bias=True
                                                                             )
            else:
                self.fully_connected_layer: torch.nn = torch.torch.nn.Linear(in_features=self.params['hidden_layer_{}_neurons'.format(_l)],
                                                                             out_features=output_size,
                                                                             bias=True
                                                                             )

    def forward(self, x):
        """
        Feed forward algorithm

        :param x:
            Input

        :return: Configured neural network
        """
        if self.params.get('initializer') is None:
            self.params.update(dict(initializer=np.random.choice(a=list(INITIALIZER.keys()))))
        set_initializer(x=x, **self.params)
        x = x.float()
        for l, layer in enumerate(self.hidden_layers):
            x = self.activation_functions[l](layer(x))
            if self.use_alpha_dropout:
                x = functional.alpha_dropout(input=x, p=self.dropout_layers[l], training=True, inplace=False)
            else:
                x = functional.dropout(input=x, p=self.dropout_layers[l], training=True, inplace=False)
        return self.activation_output(self.fully_connected_layer(x))


class LSTM(nn.Module):
    """
	Class for building deep learning long-short term memory network model using PyTorch
	"""
    def __init__(self, parameters: dict, output_size: int):
        """
		:param parameters: dict
			Parameter settings

        :param output_size: int
            Output size:
                -> 1: Float value (Regression)
                -> 2: Classes (Binary Classification)
                -> >2: Classes (Multi-Class Classification)
		"""
        super(LSTM, self).__init__()
        self.params: dict = parameters
        self.output_size: int = output_size
        self.batch_size: int = self.params.get('batch_size')
        self.hidden_size: int = self.params.get('hidden_states')
        self.vocab_size: int = self.params.get('vocab_size')
        self.embedding_length: int = self.params.get('embedding_len')
        self.word_embeddings: torch.nn.Embedding = torch.nn.Embedding(self.vocab_size, self.embedding_length)
        self.word_embeddings.weights = nn.Parameter(self.params.get('weights'), requires_grad=False)
        self.lstm_layer = nn.LSTM(self.embedding_length, self.hidden_size)
        self.output_layer = nn.Linear(in_features=self.hidden_size, out_features=output_size)

    def forward(self, input_sentence):
        """
		:param input_sentence:
			Input text (batch_size, num_sequences)

		:return torch.tensor:
			Logits
		"""
        _input = self.word_embeddings(input_sentence)
        _input = _input.permute(1, 0, 2)
        _h_0 = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        _c_0 = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        torch.nn.init.ones_(_h_0)
        torch.nn.init.ones_(_c_0)
        _output, (final_hidden_state, final_cell_state) = self.lstm_layer(_input, (_h_0, _c_0))
        return self.output_layer(final_hidden_state[-1])


class RCNN(torch.nn.Module):
    """
    Class for applying Recurrent Convolutional Neural Network using PyTorch as a Deep Learning Framework
    """
    def __init__(self, parameters: dict, output_size: int):
        """
		:param parameters: dict
			Parameter settings

        :param output_size: int
            Output size:
                -> 1: Float value (Regression)
                -> 2: Classes (Binary Classification)
                -> >2: Classes (Multi-Class Classification)
        """
        super(RCNN, self).__init__()
        self.params: dict = parameters
        self.output_size: int = output_size
        self.batch_size: int = self.params.get('batch_size')
        self.hidden_size: int = self.params.get('hidden_states')
        self.vocab_size: int = self.params.get('vocab_size')
        self.embedding_length: int = self.params.get('embedding_len')
        self.word_embeddings: torch.nn.Embedding = torch.nn.Embedding(self.vocab_size, self.embedding_length)
        self.word_embeddings.weight = torch.nn.Parameter(self.params.get('weights'), requires_grad=False)
        self.dropout: float = self.params.get('dropout')
        self.use_alpha_dropout: bool = False
        self.recurrent_network_type: str = 'lstm' if self.params.get('recurrent_network_type') is None else self.params.get('recurrent_network_type')
        if self.recurrent_network_type == 'gru':
            self.recurrent_input_cell: torch.nn.GRU = torch.nn.GRU(input_size=self.embedding_length,
                                                                   hidden_size=self.hidden_size,
                                                                   num_layers=1,
                                                                   dropout=0,
                                                                   bidirectional=True
                                                                   )
            self.recurrent_output_cell: torch.nn.GRU = torch.nn.GRU(input_size=2 * self.hidden_size,
                                                                    hidden_size=self.hidden_size,
                                                                    num_layers=1,
                                                                    dropout=0,
                                                                    bidirectional=True
                                                                    )
        elif self.recurrent_network_type == 'lstm':
            self.recurrent_input_cell: torch.nn.LSTM = torch.nn.LSTM(input_size=self.embedding_length,
                                                                     hidden_size=self.hidden_size,
                                                                     num_layers=1,
                                                                     dropout=0,
                                                                     bidirectional=True
                                                                     )
            self.recurrent_output_cell: torch.nn.LSTM = torch.nn.LSTM(input_size=2 * self.hidden_size,
                                                                      hidden_size=self.hidden_size,
                                                                      num_layers=1,
                                                                      dropout=0,
                                                                      bidirectional=True
                                                                      )
        elif self.recurrent_network_type == 'rnn':
            self.recurrent_input_cell: torch.nn.RNN = torch.nn.RNN(input_size=2 * self.embedding_length,
                                                                   hidden_size=self.hidden_size,
                                                                   num_layers=1,
                                                                   dropout=0,
                                                                   bidirectional=True
                                                                   )
            self.recurrent_output_cell: torch.nn.RNN = torch.nn.RNN(input_size=2 * self.hidden_size,
                                                                    hidden_size=self.hidden_size,
                                                                    num_layers=1,
                                                                    dropout=0,
                                                                    bidirectional=True
                                                                    )
        else:
            raise PyTorchNetworkException('Recurrent network type ({}) not supported'.format(self.recurrent_network_type))
        self.W2: torch.nn.Linear = torch.nn.Linear(in_features=2 * self.hidden_size + self.embedding_length,
                                                   out_features=self.hidden_size,
                                                   bias=True
                                                   )
        self.output_layer: torch.nn.Linear = torch.nn.Linear(in_features=self.hidden_size,
                                                             out_features=output_size,
                                                             bias=True
                                                             )
        self.hidden_layers: List[torch.nn] = []
        self.dropout_layers: List[float] = []
        if self.params is not None:
            _l: int = 0
            for layer in range(1, len(self.params.keys()) + 1, 1):
                if self.params.get('hidden_layer_{}_neurons'.format(layer)) is not None:
                    if self.params['hidden_layer_{}_alpha_dropout'.format(layer)]:
                        self.use_alpha_dropout = True
                    self.dropout_layers.append(self.params['hidden_layer_{}_dropout'.format(layer)])
                    if len(self.hidden_layers) == 0:
                        self.hidden_layers.append(self.recurrent_output_cell)
                    else:
                        if layer + 1 < len(self.params.keys()):
                            if self.params['hidden_layer_{}_rnn_network_type'.format(layer)] == 'gru':
                                self.hidden_layers.append(torch.nn.GRU(input_size=2 * self.hidden_size,
                                                                       hidden_size=self.hidden_size,
                                                                       num_layers=1,
                                                                       dropout=0,
                                                                       bidirectional=True
                                                                       )
                                                          )
                            elif self.params['hidden_layer_{}_rnn_network_type'.format(layer - 1)] == 'lstm':
                                self.hidden_layers.append(torch.nn.LSTM(input_size=2 * self.hidden_size,
                                                                        hidden_size=self.hidden_size,
                                                                        num_layers=1,
                                                                        dropout=0,
                                                                        bidirectional=True
                                                                        )
                                                          )
                            elif self.params['hidden_layer_{}_rnn_network_type'.format(layer - 1)] == 'rnn':
                                self.hidden_layers.append(torch.nn.RNN(input_size=2 * self.hidden_size,
                                                                       hidden_size=self.hidden_size,
                                                                       num_layers=1,
                                                                       dropout=0,
                                                                       bidirectional=True
                                                                       )
                                                          )
                        else:
                            _l = layer

    def forward(self, input_sentence, batch_size=None):
        """
        :param input_sentence:
        :param batch_size:
        :return:
        """
        if self.params.get('initializer') is None:
            self.params.update(dict(initializer=np.random.choice(a=list(INITIALIZER.keys()))))
        _word_embedding = self.word_embeddings(input_sentence)
        _word_embedding = _word_embedding.permute(1, 0, 2)
        _batch_size: int = self.batch_size if batch_size is None else batch_size
        _hidden_state = Variable(torch.zeros(2, _batch_size, self.hidden_size))
        _cell_state = Variable(torch.zeros(2, _batch_size, self.hidden_size))
        set_initializer(x=_hidden_state, **self.params)
        set_initializer(x=_cell_state, **self.params)
        _input, (_, _) = self.recurrent_input_cell(_word_embedding, (_hidden_state, _cell_state))
        _input = functional.dropout(input=_input, p=self.dropout, training=True, inplace=False)
        _output = _input
        if len(self.hidden_layers) == 0:
            pass
        else:
            for l, layer in enumerate(self.hidden_layers):
                __hidden_state = Variable(torch.zeros(2, _batch_size, self.hidden_size))
                __cell_state = Variable(torch.zeros(2, _batch_size, self.hidden_size))
                set_initializer(x=__hidden_state, **self.params)
                set_initializer(x=__cell_state, **self.params)
                _output, (_, _) = layer(_output, (__hidden_state, __cell_state))
                _output = functional.dropout(input=_output, p=self.dropout_layers[l], training=True, inplace=False)
        _output, (final_hidden_state, final_cell_state) = self.recurrent_output_cell(_output, (_hidden_state, _cell_state))
        _final_encoding = torch.cat((_output, _word_embedding), 2).permute(1, 0, 2)
        _target_output = self.W2(_final_encoding)
        _target_output = _target_output.permute(0, 2, 1)
        _target_output = functional.max_pool1d(_target_output, _target_output.size()[2])
        _target_output = _target_output.squeeze(2)
        return self.output_layer(_target_output)


class RNN(nn.Module):
    """
	Class for building deep learning rnn model using PyTorch
	"""
    def __init__(self, parameters: dict, output_size: int):
        """
		:param parameters: dict
			Parameter settings

        :param output_size: int
            Output size:
                -> 1: Float value (Regression)
                -> 2: Classes (Binary Classification)
                -> >2: Classes (Multi-Class Classification)
		"""
        super(RNN, self).__init__()
        self.params: dict = parameters
        self.output_size: int = output_size
        self.batch_size: int = self.params.get('batch_size')
        self.hidden_size: int = self.params.get('hidden_states')
        self.vocab_size: int = self.params.get('vocab_size')
        self.embedding_length: int = self.params.get('embedding_len')
        self.word_embeddings: torch.nn.Embedding = torch.nn.Embedding(self.vocab_size, self.embedding_length)
        self.word_embeddings.weights = nn.Parameter(self.params.get('weights'), requires_grad=False)
        self.rnn = nn.RNN(self.embedding_length, self.hidden_size, num_layers=1, bidirectional=True)
        self.output_layer = nn.Linear(in_features=4 * self.hidden_size, out_features=output_size)

    def forward(self, input_sentence):
        """
		:param input_sentence:
			Input text (batch_size, num_sequences)

		:return torch.tensor:
			Logits
		"""
        _input = self.word_embeddings(input_sentence)
        _input = _input.permute(1, 0, 2)
        _h_0 = torch.autograd.Variable(torch.zeros(4, self.batch_size, self.hidden_size))
        _output, _h_n = self.rnn(_input, _h_0)
        _h_n = _h_n.permute(1, 0, 2)
        _h_n = _h_n.contiguous().view(_h_n.size()[0], _h_n.size()[1] * _h_n.size()[2])
        return self.output_layer(_h_n)


class SelfAttention(nn.Module):
    """
	Class for building deep learning self attention model using PyTorch
	"""
    def __init__(self, parameters: dict, output_size: int):
        """
		:param parameters: dict
			Parameter settings

        :param output_size: int
            Output size:
                -> 1: Float value (Regression)
                -> 2: Classes (Binary Classification)
                -> >2: Classes (Multi-Class Classification)
		"""
        super(SelfAttention, self).__init__()
        self.params: dict = parameters
        self.output_size: int = output_size
        self.batch_size: int = self.params.get('batch_size')
        self.hidden_size: int = self.params.get('hidden_states')
        self.vocab_size: int = self.params.get('vocab_size')
        self.embedding_length: int = self.params.get('embedding_len')
        self.word_embeddings: torch.nn.Embedding = torch.nn.Embedding(self.vocab_size, self.embedding_length)
        self.word_embeddings.weights = nn.Parameter(self.params.get('weights'), requires_grad=False)
        self.dropout = 0.5
        self.lstm_layer = nn.LSTM(self.embedding_length, self.hidden_size, dropout=self.dropout, bidirectional=True)
        self.W_s1 = nn.Linear(2 * self.hidden_size, 350)
        self.W_s2 = nn.Linear(350, 30)
        self.fully_connected_layer = nn.Linear(30 * 2 * self.hidden_size, 2000)
        self.output_layer = nn.Linear(in_features=2000, out_features=output_size)

    def attention_network(self, lstm_output):
        """
		Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an
		encoding of the inout sentence but giving an attention to a specific part of the sentence. We will use 30 such embedding of
		the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully
		connected layer of size 2000 which will be connected to the output layer of size 2 returning logits for our two classes i.e.,
		pos & neg.

		Arguments
		---------

		lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.
		---------

		Returns : Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give
				  attention to different parts of the input sentence.

		Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
					  attn_weight_matrix.size() = (batch_size, 30, num_seq)

		"""
        _attention_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
        _attention_weight_matrix = _attention_weight_matrix.permute(0, 2, 1)
        return torch.softmax(_attention_weight_matrix, dim=2)

    def forward(self, input_sentence):
        """
		:param input_sentence:
			Input text (batch_size, num_sequences)

		:return torch.tensor:
			Logits
		"""
        _input = self.word_embeddings(input_sentence)
        _input = _input.permute(1, 0, 2)
        _h_0 = torch.autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_size))
        _c_0 = torch.autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_size))
        _output, (_h_n, _c_n) = self.lstm_layer(_input, (_h_0, _c_0))
        _output = _output.permute(1, 0, 2)
        _attention_weight_matrix = self.attention_network(lstm_output=_output)
        _hidden_matrix = torch.bmm(_attention_weight_matrix, _output)
        _fully_connected_output = self.fc_layer(_hidden_matrix.view(-1, _hidden_matrix.size()[1] * _hidden_matrix.size()[2]))
        return self.output_layer(_fully_connected_output)


class Transformers:
    """
    Class for building encoder decoder transformer networks using simpletransformers based on hugging face
    """
    def __init__(self, parameters: dict, output_size: int, cache_dir: str = None):
        """
        :param parameters: dict
			Parameter settings

        :param output_size: int
            Output size:
                -> 1: Float value (Regression)
                -> 2: Classes (Binary Classification)
                -> >2: Classes (Multi-Class Classification)

        :param cache_dir: str
            Cache directory for loading pre-trained language (embedding) models from disk
        """
        self.args: dict = dict(model_type=parameters.get('model_type'),
                               model_name=parameters.get('model_name'),
                               regression=False if output_size > 1 else True,
                               num_train_epochs=parameters.get('epoch'),
                               learning_rate=parameters.get('learning_rate'),
                               train_batch_size=parameters.get('batch_size'),
                               eval_batch_size=parameters.get('batch_size'),
                               max_seq_length=512,
                               adafactor_beta1=parameters.get('adafactor_beta1'),
                               adafactor_clip_threshold=parameters.get('adafactor_clip_threshold'),
                               adafactor_decay_rate=parameters.get('adafactor_decay_rate'),
                               adafactor_eps=parameters.get('adafactor_eps'),
                               adafactor_relative_step=parameters.get('adafactor_relative_step'),
                               adafactor_scale_parameter=parameters.get('adafactor_scale_parameter'),
                               adafactor_warmup_init=parameters.get('adafactor_warmup_init'),
                               adam_epsilon=parameters.get('adam_epsilon'),
                               cosine_schedule_num_cycles=parameters.get('cosine_schedule_num_cycles'),
                               dynamic_quantize=parameters.get('dynamic_quantize'),
                               early_stopping_consider_epochs=parameters.get('early_stopping_consider_epochs'),
                               use_early_stopping=parameters.get('use_early_stopping'),
                               early_stopping_delta=parameters.get('early_stopping_delta'),
                               early_stopping_patience=parameters.get('early_stopping_patience'),
                               attention_probs_dropout_prob=parameters.get('attention_probs_dropout_prob'),
                               hidden_size=parameters.get('hidden_size'),
                               hidden_dropout_prob=parameters.get('hidden_dropout_prob'),
                               initializer_range=parameters.get('initializer_range'),
                               layer_norm_eps=parameters.get('layer_norm_eps'),
                               num_attention_heads=parameters.get('num_attention_heads'),
                               num_hidden_layers=parameters.get('num_hidden_layers'),
                               optimizer=parameters.get('optimizer'),
                               scheduler=parameters.get('scheduler'),
                               polynomial_decay_schedule_lr_end=parameters.get('polynomial_decay_schedule_lr_end'),
                               polynomial_decay_schedule_power=parameters.get('polynomial_decay_schedule_power'),
                               weight_decay=parameters.get('weight_decay'),
                               gradient_accumulation_steps=parameters.get('gradient_accumulation_steps'),
                               max_grad_norm=parameters.get('max_grad_norm'),
                               early_stopping_metric='eval_loss',
                               early_stopping_metric_minimize=True,
                               sliding_window=False,
                               manual_seed=1234,
                               warmup_ratio=parameters.get('warmup_ratio'),
                               warmup_step=parameters.get('warmup_step'),
                               save_steps=2000,
                               logging_steps=100,
                               evaluate_during_training=True,
                               eval_all_checkpoints=False,
                               use_tensorboard=True,
                               overwrite_output_dir=True,
                               reprocess_input_data=True,
                               do_lower_case=True,
                               no_save=True,
                               no_cache=False,
                               silent=True,
                               best_model_dir=BEST_MODEL_DIR,
                               output_dir=OUTPUT_DIR,
                               cache_dir=CACHE_DIR,
                               fp16=parameters.get('fp16'),
                               fp16_opt_level=parameters.get('fp16_opt_level')
                               )
        _kwargs: dict = dict(cache_dir=cache_dir, local_files_only=False if cache_dir is None else True)
        self.model = ClassificationModel(model_type=parameters.get('model_type'),
                                         model_name=parameters.get('model_name'),
                                         tokenizer_type=None,
                                         tokenizer_name=None,
                                         num_labels=output_size,
                                         weight=None,
                                         args=self.args,
                                         use_cuda=torch.cuda.is_available(),
                                         cuda_device=0 if torch.cuda.is_available() else -1,
                                         onnx_execution_provider=None,
                                         **_kwargs
                                         )

    def forward(self) -> ClassificationModel:
        return self.model
