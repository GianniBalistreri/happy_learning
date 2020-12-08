import numpy as np
import torch

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
    Class for handling exceptions for class MLP, RCNN
    """
    pass


class MLP(torch.nn.Module):
    """
    Class for applying Multi Layer Perceptron (Fully Connected Layer) using PyTorch as a Deep Learning Framework
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 parameters: dict = None,
                 ):
        """
        :param input_size: int
            Number of predictors

        :param output_size: int
            Output size:
                -> 1: Float value (Regression)
                -> 2: Classes (Binary Classification)
                -> >2: Classes (Multi-Class Classification)

        :param parameters: dict
			Parameter settings
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
        return self.activation_output(self.fully_connected_layer(x)) # , dim=1


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
        self.word_embeddings: torch.nn.Embedding = torch.nn.Embedding(self.vocab_size, self.embedding_length)  # Initializing the look-up table.
        self.word_embeddings.weight = torch.nn.Parameter(self.params.get('weights'), requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
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


class AttentionModel(torch.nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(AttentionModel, self).__init__()
        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        --------
        """
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_length)
        self.word_embeddings.weights = torch.nn.Parameter(weights, requires_grad=False)
        self.lstm = torch.nn.LSTM(embedding_length, hidden_size)
        self.label = torch.nn.Linear(hidden_size, output_size)
        #self.attn_fc_layer = torch.nn.Linear()

    def attention_net(self, lstm_output, final_state):
        """
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)

        """
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = functional.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, input_sentences, batch_size=None):
        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
        final_output.shape = (batch_size, output_size)

        """
        input = self.word_embeddings(input_sentences)
        input = input.permute(1, 0, 2)
        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))  # .cuda())
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))  # .cuda())
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))  # .cuda())
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))  # .cuda())

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (
        h_0, c_0))  # final_hidden_state.size() = (1, batch_size, hidden_size)
        output = output.permute(1, 0, 2)  # output.size() = (batch_size, num_seq, hidden_size)
        attn_output = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)
        return logits
