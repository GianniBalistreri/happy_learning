import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from torchtext.data import BucketIterator, Field, TabularDataset
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
        super().__init__()
        self.encoder_hidden_layers: List[torch.nn] = []
        self.decoder_hidden_layers: List[torch.nn] = []
        for e in encoder_hidden_layer.keys():
            self.encoder_hidden_layers.append(torch.nn.Linear(in_features=encoder_hidden_layer[e]['in_features'],
                                                              out_features=encoder_hidden_layer[e]['out_features'],
                                                              bias=True
                                                              )
                                              )
        for d in decoder_hidden_layer.keys():
            self.decoder_hidden_layers.append(torch.nn.Linear(in_features=decoder_hidden_layer[d]['in_features'],
                                                              out_features=decoder_hidden_layer[d]['out_features'],
                                                              bias=True
                                                              )
                                              )

    def forward(self, features) -> tuple:
        """
        Feed-Forward network

        :param features:
            Data input (features)

        :return: tuple
            Encoder weights and decoder weights
        """
        _encoder: torch.tensor = None
        _decoder: torch.tensor = None
        for encoder_layer in self.encoder_hidden_layers:
            _encoder = torch.nn.LeakyReLU(encoder_layer(features))
        _decoder = _encoder
        for decoder_layer in self.decoder_hidden_layers:
            _decoder = torch.nn.LeakyReLU(decoder_layer(features))
        return _encoder, _decoder


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

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        """
        Feed-forward network

        :param inputs:
        :return:
        """
        _q = 1.0 / (1.0 + (torch.sum(torch.square(inputs.size()[1]) - self.n_clusters)) / self.alpha)
        _q **= (self.alpha + 1.0) / 2.0
        _q = (_q.transpose() / torch.sum(_q)).transpose()
        return _q

    def set_weight(self, weights: torch.tensor):
        """
        Update weights

        :param weights: torch.tensor
            Weights
        """
        self.weights_layer = torch.nn.Parameter(data=weights, requires_grad=False)


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
                 iterator: BucketIterator,
                 batch_size: int,
                 epoch: int = 15,
                 optimizer: str = 'kld',
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
        self.iterator: BucketIterator = iterator
        self.encoder, self.auto_encoder = AutoEncoder(encoder_hidden_layer=encoder_hidden_layer,
                                                      decoder_hidden_layer=decoder_hidden_layer
                                                      )
        _clustering_layer = Clustering(n_clusters=self.n_clusters,
                                       in_features=10,
                                       out_features=10,
                                       alpha=self.alpha
                                       )(self.encoder)
        self.model: torch.nn.Sequential = torch.nn.Sequential(self.encoder, _clustering_layer)

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
        return self.encoder.predict(x=x)

    def fit(self) -> tuple:
        """
        Train STC using PyTorch

        :return: tuple
            Observed label and label prediction
        """
        _predictions: List[int] = []
        _observations: List[int] = []
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.train()
        if self.optimizer == 'rmsprop':
            _optim: torch.optim = torch.optim.RMSprop(
                params=filter(lambda p: p.grad is not None, self.model.parameters()),
                lr=0.0001
                )
        elif self.optimizer == 'adam':
            _optim: torch.optim = torch.optim.Adam(params=filter(lambda p: p.grad is not None, self.model.parameters()),
                                                   lr=0.0001
                                                   )
        elif self.optimizer == 'sgd':
            _optim: torch.optim = torch.optim.SGD(params=filter(lambda p: p.grad is not None, self.model.parameters()),
                                                  lr=0.0001
                                                  )
        else:
            raise SelfTaughtShortTextClusteringException('Optimizer ({}) not supported'.format(self.optimizer))
        for e in range(0, self.epoch, 1):
            for idx, batch in enumerate(self.iterator):
                _predictors = batch.text[0]
                _target = batch.label
                if _target.size()[0] != self.batch_size:
                    continue
                if torch.cuda.is_available():
                    _target = _target.cuda()
                    _predictors = _predictors.cuda()
                _optim.zero_grad()
                _prediction = torch.softmax(input=self.model(_predictors), dim=1)
                _top_class, _top_p = _prediction.topk(k=1, dim=1)
                _predictions.extend(_top_class.detach().tolist())
                _observations.extend(_target.detach().numpy().tolist())
                _loss = torch.nn.KLDivLoss(_prediction, _target)
                _loss.backward()
                self._clip_gradient(clip_value=1e-1)
                _optim.step()
        return _observations, _predictions

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Feed-forward network

        :param x: np.ndarray
            Embedding values

        :return: np.ndarray
            Label prediction
        """
        _nmi: int = 0
        _k_means: KMeans = KMeans(n_clusters=self.n_clusters, n_init=self.n_iterations)
        _pred: np.ndarray = _k_means.fit_predict(self.encoder.predict(x))
        _last_pred: np.ndarray = np.copy(_pred)
        self.model.set_weights([_k_means.cluster_centers_])
        for i in range(0, self.max_iterations, 1):
            if i % self.update_interval == 0:
                _q = self.model.predict(x)
                _p = self.target_distribution(_q)
                _pred = _q.argmax(1)
                #_nmi = np.round(normalized_mutual_info_score(labels_true=y, labels_pred=_pred, average_method='arithmetic'))
                # check stopping criterion
                _delta_label = np.sum(_pred != _last_pred).astype(np.float32) / _pred.shape[0]
                _last_pred = np.copy(_pred)
                if i > 0 and _delta_label < self.tol:
                    break
        return _pred

    def predict(self, x: np.ndarray) -> int:
        """
        Predict labels from trained STC model

        :param x: np.ndarray
            Embedding weights

        :return: int
            Label value
        """
        _q: np.ndarray = self.model.predict(x)
        return _q.argmax(1)

    def pre_train_auto_encoder(self,
                               iterator: BucketIterator,
                               epoch: int = 15,
                               optimizer: str = 'rmsprop'
                               ):
        """
        Pre-train auto-encoder network

        :param iterator: BucketIterator
            Batched data set

        :param epoch: int
            Number of epochs

        :param optimizer: str
            Name of the optimizer:
                -> rmsprop: RMSprop
                -> adam: Adam
                -> sgd: Stochastic Gradient Decent
        """
        _predictions: List[int] = []
        _observations: List[int] = []
        if torch.cuda.is_available():
            self.auto_encoder.cuda()
        self.auto_encoder.train()
        if optimizer == 'rmsprop':
            _optim: torch.optim = torch.optim.RMSprop(params=filter(lambda p: p.grad is not None, self.model.parameters()),
                                                      lr=0.0001
                                                      )
        elif optimizer == 'adam':
            _optim: torch.optim = torch.optim.Adam(params=filter(lambda p: p.grad is not None, self.model.parameters()),
                                                   lr=0.0001
                                                   )
        elif optimizer == 'sgd':
            _optim: torch.optim = torch.optim.SGD(params=filter(lambda p: p.grad is not None, self.model.parameters()),
                                                  lr=0.0001
                                                  )
        else:
            raise SelfTaughtShortTextClusteringException('Optimizer ({}) not supported'.format(optimizer))
        for e in range(0, epoch, 1):
            for idx, batch in enumerate(iterator):
                _predictors = batch.text[0]
                _target = batch.label
                if _target.size()[0] != self.batch_size:
                    continue
                if torch.cuda.is_available():
                    _target = _target.cuda()
                    _predictors = _predictors.cuda()
                _optim.zero_grad()
                _prediction = torch.softmax(input=self.auto_encoder(_predictors), dim=1)
                _loss = torch.nn.MSELoss(_prediction, _target)
                _loss.backward()
                self._clip_gradient(clip_value=1e-1)
                _optim.step()

    @staticmethod
    def target_distribution(q):
        """
        Auxiliary target distribution

        :param q:
        :return:
        """
        _weight = q ** 2 / q.sum(0)
        return (_weight.transpose / _weight.sum(1)).transpose
