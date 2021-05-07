import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from typing import List


def get_sentence_embedding(text_data: np.array,
                           lang_model_name: str = 'paraphrase-xlm-r-multilingual-v1',
                           lang_model_path: str = None
                           ) -> np.ndarray:
    """
    Get sentence embedding using sentence-transformer package

    :param text_data: np.array
        Text data

    :param lang_model_name: str
        Name of the language model:
            -> paraphrase-distilroberta-base-v1: Distill RoBERTa
            -> paraphrase-xlm-r-multilingual-v1: XLM-RoBERTa

    :param lang_model_path: str
        Complete file path of the pre-trained language model
            -> paraphrase-distilroberta-base-v1: Distill RoBERTa
            -> paraphrase-xlm-r-multilingual-v1: XLM-RoBERTa

    :return: np.array
        Sentence embedding vector
    """
    _lang_model_names: List[str] = ['paraphrase-distilroberta-base-v1', 'paraphrase-xlm-r-multilingual-v1']
    if lang_model_name not in _lang_model_names:
        raise SelfTaughtShortTextClusteringException('Language model ({}) not supported'.format(lang_model_name))
    if lang_model_path is None:
        _lang_model: str = lang_model_name
    else:
        _lang_model: str = lang_model_path
    _model = SentenceTransformer(model_name_or_path=_lang_model, device=None)
    _embedding = _model.encode(sentences=text_data.tolist(),
                               batch_size=32,
                               show_progress_bar=False,
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
    def __init__(self, dims: int, n_clusters: int, alpha: float = 1.0):
        """
        :param dims: int
            Number of input dimensions

        :param n_clusters: int
            Number of pre-defined clusters

        :param alpha: float

        """
        super(Clustering, self).__init__()
        self.alpha: float = alpha
        self.n_clusters: int = n_clusters
        self.weights_layer: torch.nn.Parameter = torch.nn.Parameter(data=torch.Tensor(self.n_clusters, dims), requires_grad=False)
        self.weights_layer = torch.nn.init.xavier_uniform_(tensor=self.weights_layer, gain=1.0)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Feed-forward network

        :param x: torch.tensor
            Encoder weights

        :return: torch.tensor
            Soft cluster assignment (measure similarity between embedding points and cluster centroids)
        """
        _q = 1.0 / (1.0 + (torch.sum(torch.square(torch.unsqueeze(input=x, dim=1) - self.weights_layer), dim=2) / self.alpha))
        _q = torch.pow(input=_q, exponent=(self.alpha + 1.0) / 2.0)
        _q = (_q.transpose(dim0=1, dim1=0) / torch.sum(_q, dim=1)).transpose(1, 0)
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
                 dimensions: List[int],
                 iterator: DataLoader,
                 encoder_hidden_layer: dict = None,
                 decoder_hidden_layer: dict = None,
                 batch_size: int = 16,
                 epoch: int = 15,
                 optimizer: str = 'sgd',
                 n_clusters: int = 20,
                 alpha: float = 1.0,
                 n_iterations: int = 100,
                 max_iterations: int = 1000,
                 early_stopping: bool = False,
                 tol: float = 0.0001,
                 update_interval: int = 500
                 ):
        """
        :param encoder_hidden_layer: dict
            Input and output dimensions for each encoder hidden layer (input & output layer included)
                -> e.g. for each layer: dict(layer_1=in_features=x, out_features=y)
                -> layer_1 is always the input layer & the last layer is always the output layer

        :param decoder_hidden_layer: dict
            Input and output dimensions for each decoder hidden layer (input & output layer included)
                -> e.g. for each layer: dict(layer_1=in_features=x, out_features=y)
                -> layer_1 is always the input layer & the last layer is always the output layer

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

        :param early_stopping: bool
            Whether to stop early if stopping criterion matches or not

        :param tol: float
            Tolerance for

        :param update_interval: int
            Interval for updating auto-encoder weights
        """
        super(STC, self).__init__()
        self.dims: List[int] = dimensions
        self.n_clusters: int = n_clusters
        self.alpha: float = alpha
        self.n_iterations: int = n_iterations
        self.max_iterations: int = max_iterations
        self.update_interval: int = update_interval
        self.early_stopping: bool = early_stopping
        self.tol: float = tol
        self.epoch: int = epoch
        self.batch_size: int = batch_size
        self.optimizer: str = optimizer
        self.iterator: DataLoader = iterator
        self.nmi_score: float = 0.0
        self.obs: List[int] = []
        self.pred: List[str] = []
        if encoder_hidden_layer is None or decoder_hidden_layer is None:
            _encoder_hidden_layer: dict = dict(layer_1=dict(in_features=self.dims[1], out_features=500),
                                               layer_2=dict(in_features=500, out_features=500),
                                               layer_3=dict(in_features=500, out_features=2000),
                                               layer_4=dict(in_features=2000, out_features=self.n_clusters),
                                               )
            _decoder_hidden_layer: dict = dict(layer_1=dict(in_features=self.n_clusters, out_features=2000),
                                               layer_2=dict(in_features=2000, out_features=500),
                                               layer_3=dict(in_features=500, out_features=500),
                                               layer_4=dict(in_features=500, out_features=self.dims[1])
                                               )
        else:
            _encoder_hidden_layer: dict = encoder_hidden_layer
            _decoder_hidden_layer: dict = decoder_hidden_layer
        self.model: torch.nn.Sequential = torch.nn.Sequential()
        self.encoder = Encoder(encoder_hidden_layer=_encoder_hidden_layer)
        self.decoder = Decoder(decoder_hidden_layer=_decoder_hidden_layer)
        self.auto_encoder = AutoEncoder(encoder_hidden_layer=_encoder_hidden_layer, decoder_hidden_layer=_decoder_hidden_layer)
        self.pre_train_auto_encoder()
        _clustering_layer = Clustering(dims=self.dims[1], n_clusters=self.n_clusters, alpha=self.alpha)
        self.model.add_module(name='encoder', module=self.encoder)
        self.model.add_module(name='cluster', module=_clustering_layer)

    def _clip_gradient(self, clip_value: float, auto_encoder: bool):
        """
        Clip gradient during network training

        :param clip_value: float
            Clipping value for avoiding vanishing gradient

        :param auto_encoder: bool
            Whether to use auto-encoder model parameters or not
        """
        _parameters = self.auto_encoder.parameters() if auto_encoder else self.model.parameters()
        params = list(filter(lambda p: p.grad is not None, _parameters))
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

    def fit(self, x: torch.tensor):
        """
        Train STC using PyTorch and KMeans from sklearn

        :param x: torch.tensor
            Embedding values
        """
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.train()
        with torch.autograd.set_detect_anomaly(True):
            if self.optimizer == 'rmsprop':
                _optim: torch.optim = torch.optim.RMSprop(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.01)
            elif self.optimizer == 'adam':
                _optim: torch.optim = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.01)
            elif self.optimizer == 'sgd':
                _optim: torch.optim = torch.optim.SGD(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.01)
            else:
                raise SelfTaughtShortTextClusteringException('Optimizer ({}) not supported'.format(self.optimizer))
            _loss_function: torch.nn.KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean')
            _k_means: KMeans = KMeans(n_clusters=self.n_clusters, n_init=self.n_iterations)
            _pred: np.ndarray = _k_means.fit_predict(self.encoder(x).detach().numpy())
            _last_pred: np.ndarray = np.copy(_pred)
            _last_pred_tensor: torch.tensor = torch.tensor(data=_last_pred)
            self.model[1].set_weights(cluster_centers=_k_means.cluster_centers_)
            _index: int = 0
            _index_array: np.ndarray = np.arange(x.shape[0])
            for i in range(0, self.max_iterations, 1):
                _idx: np.ndarray = _index_array[_index * self.batch_size: min((_index + 1) * self.batch_size, x.shape[0])]
                _predictors, _target = self.iterator.dataset[_idx]
                if i % self.update_interval == 0:
                    _q = self.model(x)
                    _p = self.target_distribution(_q)
                    _pred = _q.argmax(1)
                    _delta_label = np.sum(_pred != _last_pred).astype(np.float32) / _pred.shape[0]
                    _last_pred = np.copy(_pred)
                    _last_pred_tensor = torch.tensor(data=_last_pred)
                    for idx in enumerate(_idx):
                        self.obs.append(_target.detach().numpy().tolist()[idx[0]])
                        self.pred.append(_last_pred_tensor.detach().numpy().tolist()[idx[0]])
                    self.fitness()
                    if self.early_stopping:
                        if i > 0 and _delta_label < self.tol:
                            break
                _loss = _loss_function(_q[_idx].detach(), _p[_idx].detach())
                _loss.requires_grad = True
                _optim.zero_grad()
                _loss.backward(retain_graph=True)
                self._clip_gradient(clip_value=1e-1, auto_encoder=False)
                _optim.step()
                _index = _index + 1 if (_index + 1) * self.batch_size <= x.shape[0] else 0

    def fitness(self):
        """
        Fitness score (Normalized Mutual Information)
        """
        self.nmi_score: float = normalized_mutual_info_score(labels_true=self.obs, labels_pred=self.pred)

    def predict(self, x: np.ndarray) -> int:
        """
        Predict labels from trained STC model

        :param x: np.ndarray
            Embedding weights

        :return: int
            Label value
        """
        _q: np.ndarray = self.model(x)
        return _q.argmax(1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict probabilities from trained STC model

        :param x: np.ndarray
            Embedding weights

        :return: np.ndarray
            Probabilities
        """
        return self.model(x)

    def pre_train_auto_encoder(self):
        """
        Pre-train auto-encoder network
        """
        if torch.cuda.is_available():
            self.auto_encoder.cuda()
        self.auto_encoder.train()
        if self.optimizer == 'rmsprop':
            _optim: torch.optim = torch.optim.RMSprop(params=filter(lambda p: p.requires_grad, self.auto_encoder.parameters()), lr=0.01)
        elif self.optimizer == 'adam':
            _optim: torch.optim = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.auto_encoder.parameters()), lr=0.01)
        elif self.optimizer == 'sgd':
            _optim: torch.optim = torch.optim.SGD(params=filter(lambda p: p.requires_grad, self.auto_encoder.parameters()), lr=0.01)
        else:
            raise SelfTaughtShortTextClusteringException('Optimizer ({}) not supported'.format(self.optimizer))
        _loss_function: torch.nn.MSELoss = torch.nn.MSELoss()
        for e in range(0, self.epoch, 1):
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
                self._clip_gradient(clip_value=1e-1, auto_encoder=True)
                _optim.step()

    @staticmethod
    def target_distribution(q):
        """
        Auxiliary target distribution

        :param q: torch.tensor

        :return:
        """
        _weight = q ** 2 / q.sum(0)
        return (_weight.transpose(1, 0) / _weight.sum(1)).transpose(1, 0)
