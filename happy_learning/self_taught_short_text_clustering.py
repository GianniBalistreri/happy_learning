import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from typing import List


def get_sentence_embedding(text_data: np.array,
                           lang_model_name: str = 'paraphrase-xlm-r-multilingual-v1'
                           ) -> np.ndarray:
    """
    Get sentence embedding using sentence-transformer package

    :param text_data: np.array
        Text data

    :param lang_model_name: str
        Name of the language model:
            -> paraphrase-distilroberta-base-v1: Distill RoBERTa
            -> paraphrase-xlm-r-multilingual-v1: XLM-RoBERTa

    :return: np.array
        Sentence embedding vector
    """
    if lang_model_name not in ['paraphrase-distilroberta-base-v1', 'paraphrase-xlm-r-multilingual-v1']:
        raise SelfTaughtShortTextClusteringException('Language model ({}) not supported'.format(lang_model_name))
    _model = SentenceTransformer(model_name_or_path=lang_model_name, device=None)
    _embedding = _model.encode(sentences=text_data.tolist(),
                               batch_size=32,
                               convert_to_numpy=True,
                               convert_to_tensor=False,
                               is_pretokenized=False,
                               device=None,
                               num_workers=0
                               )
    return np.array(_embedding)


class SelfTaughtShortTextClusteringException(Exception):
    """
    Class for handling exceptions for class AutoEncoder, STC
    """
    pass


class AutoEncoder(torch.nn.Module):
    def __init__(self, encoder_hidden_layer: dict, decoder_hidden_layer: dict):
        """
        :param encoder_hidden_layer: dict
            Input and output dimensions for each encoder hidden layer

        :param decoder_hidden_layer: dict
            Input and output dimensions for each decoder hidden layer
        """
        super(AutoEncoder, self).__init__()
        self.encoder_model: Encoder = Encoder(encoder_hidden_layer=encoder_hidden_layer)
        self.decoder_model: Decoder = Decoder(decoder_hidden_layer=decoder_hidden_layer)

    def forward(self, features) -> torch.tensor:
        """
        Feed-Forward network

        :param features:
            Data input (features)

        :return: torch.tensor
            Encoder-decoder weights
        """
        return self.decoder_model(self.encoder_model(features))


class Clustering(torch.nn.Module):
    """
    Class for building clustering layer for updating network weights used in encoder
    """
    def __init__(self, n_clusters: int, in_features: int, out_features: int, alpha: float = 1.0):
        """
        :param n_clusters: int
            Number of pre-defined clusters

        :param in_features: int
            Number of input dimensions

        :param out_features: int
            Number of output dimensions

        :param alpha: float

        """
        super(Clustering, self).__init__()
        self.alpha: float = alpha
        self.n_clusters: int = n_clusters
        self.weights_layer: torch.nn.Parameter = torch.nn.Parameter(data=torch.Tensor(in_features, out_features), requires_grad=False)
        self.weights_layer = torch.nn.init.xavier_uniform_(tensor=self.weights_layer, gain=1.0)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Feed-forward network

        :param x: torch.tensor
            Encoder weights

        :return: torch.tensor
            Soft cluster assignment (measure similarity between embedding points and cluster centroids)
        """
        self.weights_layer = torch.nn.Parameter(data=x, requires_grad=False)
        _q = 1.0 / (1.0 + (torch.sum(torch.square(torch.unsqueeze(input=self.weights_layer, dim=2) - self.n_clusters)) / self.alpha))
        _q = torch.pow(input=_q, exponent=(self.alpha + 1.0) / 2.0)
        _q = (_q.transpose(dim0=0, dim1=0) / torch.sum(_q)).transpose(0, 0)
        return _q

    def set_weights(self, cluster_centers: np.ndarray):
        """
        Update weights

        :param cluster_centers: np.ndarray
            Clustering centers created by KMeans algorithm
        """
        self.weights_layer = torch.nn.Parameter(data=torch.tensor(data=cluster_centers), requires_grad=False)


class Decoder(torch.nn.Module):
    """
    Class for building decoder network using PyTorch
    """
    def __init__(self, decoder_hidden_layer: dict):
        """
        :param decoder_hidden_layer: dict
            Input and output dimensions for each decoder hidden layer
        """
        super(Decoder, self).__init__()
        self.decoder_model: torch.nn.Sequential = torch.nn.Sequential()
        for d in decoder_hidden_layer.keys():
            self.decoder_model.add_module(name='decoder_{}'.format(d),
                                          module=torch.nn.Linear(in_features=decoder_hidden_layer[d]['in_features'],
                                                                 out_features=decoder_hidden_layer[d]['out_features'],
                                                                 bias=True
                                                                 )
                                          )
            self.decoder_model.add_module(name='decoder_{}_activation'.format(d), module=torch.nn.LeakyReLU())

    def forward(self, features) -> torch.tensor:
        """
        Feed-Forward network

        :param features:
            Data input (features)

        :return: torch.tensor
            Decoder weights
        """
        return self.decoder_model(features)


class Encoder(torch.nn.Module):
    """
    Class for building encoder network using PyTorch
    """
    def __init__(self, encoder_hidden_layer: dict):
        """
        :param encoder_hidden_layer: dict
            Input and output dimensions for each encoder hidden layer
        """
        super(Encoder, self).__init__()
        self.encoder_model: torch.nn.Sequential = torch.nn.Sequential()
        for e in encoder_hidden_layer.keys():
            self.encoder_model.add_module(name='encoder_{}'.format(e),
                                          module=torch.nn.Linear(in_features=encoder_hidden_layer[e]['in_features'],
                                                                 out_features=encoder_hidden_layer[e]['out_features'],
                                                                 bias=True
                                                                 )
                                          )
            self.encoder_model.add_module(name='encoder_{}_activation'.format(e), module=torch.nn.LeakyReLU())

    def forward(self, features) -> torch.tensor:
        """
        Feed-Forward network

        :param features:
            Data input (features)

        :return: torch.tensor
            Encoder weights
        """
        return self.encoder_model(features)


class SIF:
    """
    Class for building smooth inverse frequency sentence embedding
    """
    def __init__(self, word_embedding_file_path: str, decomposition_method: str = 'pca'):
        """
        :param word_embedding_file_path: str
            Complete file path of the word embedding model dictionary

        :param decomposition_method: str
            Name of the decomposition method:
                -> pca: Principal Component Analysis
                -> svd: Truncated Single Value Decomposition
        """
        self.uni_gram: dict = {}
        self.decomposition_method: str = decomposition_method
        self.supported_decomposition_methods: List[str] = ['pca', 'svd']
        if self.decomposition_method not in self.supported_decomposition_methods:
            raise SelfTaughtShortTextClusteringException('Decomposition method ({}) not supported. Supported methods are: {}'.format(self.decomposition_method, self.supported_decomposition_methods))
        self.word_embedding_file_path: str = word_embedding_file_path
        self.word_embedding_dictionary: dict = None
        self.vector_representations: np.ndarray = np.zeros(shape=(20000, 48))
        self.sif_data: np.ndarray = None

    def _build_uni_gram(self):
        """
        Generate uni-gram word vector

        :return:
        """
        pass

    def _load_word_embedding_dict(self):
        """
        Load word embedding dictionary
        """
        pass

    def _principal_component_analysis(self, n_components: int = 1):
        """
        Run principal component analysis

        :param n_components: int
            Number of components to extract
        """
        _pca: PCA = PCA(n_components=n_components)
        _pca.fit(self.vector_representations)
        _pca_components = _pca.components_
        _x: np.ndarray = self.vector_representations - self.vector_representations.dot(_pca_components.transpose()) * _pca_components
        self.sif_data = MinMaxScaler().fit_transform(_x)

    def _truncated_single_value_decomposition(self, n_components: int, n_iterations: int = 20):
        """
        Run truncated single value decomposition analysis

        :param n_components: int
            Number of components to extract

        :param n_iterations: int
            Number of iterations
        """
        _svd: TruncatedSVD = TruncatedSVD(n_components=n_components, n_iter=n_iterations)
        _svd.fit(self.vector_representations)
        _svd_components = _svd.components_
        _x: np.ndarray = self.vector_representations - self.vector_representations.dot(_svd_components.transpose()) * _svd_components
        self.sif_data = MinMaxScaler().fit_transform(_x)

    def fit(self) -> np.ndarray:
        """
        Generate decomposed smooth inverse frequency data

        :return: np.ndarray
            Decomposed word embedding model data
        """
        if self.decomposition_method == 'pca':
            self._principal_component_analysis(n_components=1)
        elif self.decomposition_method == 'svd':
            self._truncated_single_value_decomposition(n_components=1, n_iterations=20)
        return self.sif_data


class STC(torch.nn.Module):
    """
    Class for building self-taught short text clustering using auto-encoder and k-means
    """
    def __init__(self,
                 encoder_hidden_layer: dict,
                 decoder_hidden_layer: dict,
                 dimensions: List[int],
                 iterator: DataLoader,
                 batch_size: int,
                 epoch: int = 15,
                 optimizer: str = 'rmsprop',
                 n_clusters: int = 20,
                 alpha: float = 1.0,
                 n_iterations: int = 100,
                 max_iterations: int = 15,
                 tol: float = 0.001,
                 update_interval: int = 140
                 ):
        """
        :param encoder_hidden_layer: dict
            Input and output dimensions for each encoder hidden layer

        :param decoder_hidden_layer: dict
            Input and output dimensions for each decoder hidden layer

        :param dimensions: List[int]
            Number of dimensions of the encoder layers

        :param iterator: BucketIterator
            Batched data set

        :param batch_size: int
            Batch size

        :param epoch: int
            Number of epochs

        :param optimizer: str
            Name of the optimizer:
                -> rmsprop: RMSprop
                -> adam: Adam
                -> sgd: Stochastic Gradient Decent

        :param n_clusters: int
            Number of pre-defined clusters

        :param alpha: float
            Alpha value

        :param n_iterations: int
            Number of iterations to fit k-means clustering algorithm

        :param max_iterations: int
            Maximum number of iterations for running training

        :param tol: float
            Tolerance for

        :param update_interval: int
            Interval for updating auto-encoder weights
        """
        super(STC, self).__init__()
        self.dims: List[int] = dimensions
        self.input_dim: int = dimensions[0]
        self.n_stacks: int = len(self.dims) - 1
        self.n_clusters: int = n_clusters
        self.alpha: float = alpha
        self.n_iterations: int = n_iterations
        self.max_iterations: int = max_iterations
        self.update_interval: int = update_interval
        self.tol: float = tol
        self.epoch: int = epoch
        self.batch_size: int = batch_size
        self.optimizer: str = optimizer
        self.iterator: DataLoader = iterator
        self.model: torch.nn.Sequential = torch.nn.Sequential()
        self.encoder = Encoder(encoder_hidden_layer=encoder_hidden_layer)
        self.decoder = Decoder(decoder_hidden_layer=decoder_hidden_layer)
        self.auto_encoder = AutoEncoder(encoder_hidden_layer=encoder_hidden_layer, decoder_hidden_layer=decoder_hidden_layer)
        print('Pre-train AutoEncoder ...')
        self.pre_train_auto_encoder()
        _clustering_layer = Clustering(n_clusters=self.n_clusters,
                                       in_features=20,
                                       out_features=20,
                                       alpha=self.alpha
                                       )
        self.model.add_module(name='encoder', module=self.encoder)
        self.model.add_module(name='cluster', module=_clustering_layer)

    def _clip_gradient(self, clip_value: float):
        """
        Clip gradient during network training

        :param clip_value: float
            Clipping value for avoiding vanishing gradient
        """
        params = list(filter(lambda p: p.grad is not None, self.auto_encoder.parameters()))
        for p in params:
            p.grad.data.clamp_(-clip_value, clip_value)

    def extract_features(self, x: np.ndarray):
        """
        Extract features

        :param x: np.ndarray
            Embedding weights

        :return: torch.tensor
            Embedding feature tensor
        """
        return self.encoder(x)

    def fit(self, x: torch.tensor) -> tuple:
        """
        Train STC using PyTorch and KMeans from sklearn

        :param x: torch.tensor
            Embedding values

        :return: tuple
            Observed label and predicted label
        """
        _predictions: List[int] = []
        _observations: List[int] = []
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.train()
        if self.optimizer == 'rmsprop':
            _optim: torch.optim = torch.optim.RMSprop(params=self.model.parameters(), lr=0.0001)
        elif self.optimizer == 'adam':
            _optim: torch.optim = torch.optim.Adam(params=self.model.parameters(), lr=0.0001)
        elif self.optimizer == 'sgd':
            _optim: torch.optim = torch.optim.SGD(params=self.model.parameters(), lr=0.0001)
        else:
            raise SelfTaughtShortTextClusteringException('Optimizer ({}) not supported'.format(self.optimizer))
        _loss_function: torch.nn.KLDivLoss = torch.nn.KLDivLoss()
        _k_means: KMeans = KMeans(n_clusters=self.n_clusters, n_init=self.n_iterations)
        _pred: np.ndarray = _k_means.fit_predict(self.encoder(x).detach().numpy())
        _last_pred: np.ndarray = np.copy(_pred)
        self.model[1].set_weights(cluster_centers=_k_means.cluster_centers_)
        _index: int = 0
        _index_array: np.ndarray = np.arange(x.shape[0])
        for i in range(0, self.max_iterations, 1):
            print('Iteration: ', i)
            if i % self.update_interval == 0:
                _q = self.model(x)
                _p = self.target_distribution(_q)
                _pred = _q.argmax(-1)
                # check stopping criterion
                print('_pred', _pred)
                print('_pred.shape', _pred.shape)
                print('_last_pred', _last_pred)
                _delta_label = np.sum(_pred != _last_pred).astype(np.float32) / _pred.shape[0]
                _last_pred = np.copy(_pred)
                if i > 0 and _delta_label < self.tol:
                    break
            _idx: np.ndarray = _index_array[_index * self.batch_size: min((_index + 1) * self.batch_size, x.shape[0])]
            _predictors, _target = self.iterator.dataset[_idx]
            if torch.cuda.is_available():
                _predictors = _predictors.cuda()
            _optim.zero_grad()
            _prediction = torch.softmax(input=self.model(_predictors), dim=0)
            _top_class, _top_p = _prediction.topk(k=1, dim=0)
            _predictions.extend(_top_class.detach().tolist())
            _observations.extend(_target.detach().numpy().tolist())
            _loss = _loss_function(_prediction, _p[_idx])
            _loss.backward()
            self._clip_gradient(clip_value=1e-1)
            _optim.step()
            _index = _index + 1 if (_index + 1) * self.batch_size <= x.shape[0] else 0
        return _observations, _predictions

    def predict(self, x: np.ndarray) -> int:
        """
        Predict labels from trained STC model

        :param x: np.ndarray
            Embedding weights

        :return: int
            Label value
        """
        _q: np.ndarray = self.model(x)
        return _q.argmax(0)

    def pre_train_auto_encoder(self):
        """
        Pre-train auto-encoder network
        """
        if torch.cuda.is_available():
            self.auto_encoder.cuda()
        self.auto_encoder.train()
        if self.optimizer == 'rmsprop':
            _optim: torch.optim = torch.optim.RMSprop(params=self.auto_encoder.parameters(), lr=0.0001)
        elif self.optimizer == 'adam':
            _optim: torch.optim = torch.optim.Adam(params=self.auto_encoder.parameters(), lr=0.0001)
        elif self.optimizer == 'sgd':
            _optim: torch.optim = torch.optim.SGD(params=self.auto_encoder.parameters(), lr=0.0001)
        else:
            raise SelfTaughtShortTextClusteringException('Optimizer ({}) not supported'.format(self.optimizer))
        _loss_function: torch.nn.MSELoss = torch.nn.MSELoss()
        for e in range(0, self.epoch, 1):
            print('Epoch: ', e)
            for idx, batch in enumerate(self.iterator):
                _predictors = torch.autograd.Variable(batch[0])
                if _predictors.size()[0] != self.batch_size:
                    continue
                if torch.cuda.is_available():
                    _predictors = _predictors.cuda()
                _optim.zero_grad()
                _prediction = self.auto_encoder(_predictors)
                _loss = _loss_function(_prediction, _predictors)
                _loss.backward()
                self._clip_gradient(clip_value=1e-1)
                _optim.step()

    @staticmethod
    def target_distribution(q):
        """
        Auxiliary target distribution

        :param q: torch.tensor

        :return:
        """
        _weight = q ** 2 / q.sum(0)
        return (_weight.transpose(0, 0) / _weight.sum(0)).transpose(0, 0)
