import copy
import numpy as np
import torch

from .self_taught_short_text_clustering import get_sentence_embedding, STC
from .text_clustering import GibbsSamplingDirichletMultinomialModeling, LatentDirichletAllocation, \
    LatentSemanticIndexing, NonNegativeMatrixFactorization
from datetime import datetime
from gensim import corpora
from torch.utils.data import TensorDataset, DataLoader
from typing import List

CLUSTER_ALGORITHMS: dict = dict(gsdmm='gibbs_sampling_dirichlet_multinomial_modeling',
                                lda='latent_dirichlet_allocation',
                                lsi='latent_semantic_indexing',
                                nmf='non_negative_matrix_factorization',
                                # stc='self_taught_short_text_clustering'
                                )


class ClusteringException(Exception):
    """
    Class for handling exceptions for class Clustering, ClusteringGenerator
    """
    pass


class Clustering:
    """
    Class for handling clustering algorithms
    """

    def __init__(self, cluster_params: dict = None, seed: int = 1234):
        """
        :param cluster_params: dict
            Pre-configured clustering model parameter

        :param seed: int
            Seed
        """
        self.cluster_params: dict = cluster_params
        self.seed: int = seed
        self.vocab = None
        self.vocab_size: int = 0
        self.document_term_matrix: list = []

    def gibbs_sampling_dirichlet_multinomial_modeling(self) -> GibbsSamplingDirichletMultinomialModeling:
        """
        Config Gibbs Sampling Dirichlet Multinomial Modeling algorithm

        :return GibbsSamplingDirichletMultinomialModeling:
            Model object
        """
        return GibbsSamplingDirichletMultinomialModeling(
            n_clusters=10 if self.cluster_params.get('n_clusters') is None else self.cluster_params.get('n_clusters'),
            n_iterations=5 if self.cluster_params.get('n_iterations') is None else self.cluster_params.get(
                'n_iterations'),
            alpha=0.1 if self.cluster_params.get('alpha') is None else self.cluster_params.get('alpha'),
            beta=0.5 if self.cluster_params.get('beta') is None else self.cluster_params.get('beta'),
            )

    @staticmethod
    def gibbs_sampling_dirichlet_multinomial_modeling_param():
        """
        Generate Gibbs Sampling Dirichlet Multinomial Modeling clustering parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_clusters=np.random.randint(low=5, high=30),
                    n_iterations=np.random.randint(low=5, high=100),
                    alpha=np.random.uniform(low=0.05, high=0.95),
                    beta=np.random.uniform(low=0.05, high=0.95),
                    )

    def latent_dirichlet_allocation(self) -> LatentDirichletAllocation:
        """
        Config Latent Dirichlet Allocation algorithm

        :return LatentDirichletAllocation:
            Model object
        """
        return LatentDirichletAllocation(doc_term_matrix=self.document_term_matrix,
                                         vocab=self.vocab,
                                         n_clusters=10 if self.cluster_params.get(
                                             'n_clusters') is None else self.cluster_params.get('n_clusters'),
                                         n_iterations=5 if self.cluster_params.get(
                                             'n_iterations') is None else self.cluster_params.get('n_iterations'),
                                         decay=1.0 if self.cluster_params.get(
                                             'decay') is None else self.cluster_params.get('decay'),
                                         )

    @staticmethod
    def latent_dirichlet_allocation_param() -> dict:
        """
        Generate Latent Dirichlet Allocation clustering parameter configuration randomly

        :return: dict
        """
        return dict(n_clusters=np.random.randint(low=5, high=30),
                    n_iterations=np.random.randint(low=5, high=100),
                    decay=np.random.uniform(low=0.1, high=1.0)
                    )

    def latent_semantic_indexing(self) -> LatentSemanticIndexing:
        """
        Config Latent Semantic Indexing algorithm

        :return LatentSemanticIndexing:
            Model object
        """
        return LatentSemanticIndexing(doc_term_matrix=self.document_term_matrix,
                                      vocab=self.vocab,
                                      n_clusters=10 if self.cluster_params.get(
                                          'n_clusters') is None else self.cluster_params.get('n_clusters'),
                                      n_iterations=5 if self.cluster_params.get(
                                          'n_iterations') is None else self.cluster_params.get('n_iterations'),
                                      decay=1.0 if self.cluster_params.get(
                                          'decay') is None else self.cluster_params.get('decay'),
                                      training_algorithm='multi_pass_stochastic' if self.cluster_params.get(
                                          'training_algorithm') is None else self.cluster_params.get(
                                          'training_algorithm'),
                                      )

    @staticmethod
    def latent_semantic_indexing_param() -> dict:
        """
        Generate Latent Semantic Indexing clustering parameter configuration randomly

        :return: dict
        """
        return dict(n_clusters=np.random.randint(low=5, high=30),
                    n_iterations=np.random.randint(low=5, high=100),
                    decay=np.random.uniform(low=0.1, high=1.0),
                    training_algorithm=np.random.choice(a=['one_pass', 'multi_pass_stochastic'])
                    )

    def non_negative_matrix_factorization(self) -> NonNegativeMatrixFactorization:
        """
        Config Non-Negative Matrix Factorization algorithm

        :return: NonNegativeMatrixFactorization
            Model object
        """
        return NonNegativeMatrixFactorization(lang=self.cluster_params.get('lang'),
                                              doc_term_matrix=self.document_term_matrix,
                                              vocab=self.vocab,
                                              n_clusters=10 if self.cluster_params.get(
                                                  'n_clusters') is None else self.cluster_params.get('n_clusters'),
                                              n_iterations=5 if self.cluster_params.get(
                                                  'n_iterations') is None else self.cluster_params.get('n_iterations'),
                                              )

    @staticmethod
    def non_negative_matrix_factorization_param() -> dict:
        """
        Generate Non-Negative Matrix Factorization clustering parameter configuration randomly

        :return: dict
        """
        return dict(n_clusters=np.random.randint(low=5, high=30),
                    n_iterations=np.random.randint(low=5, high=100)
                    )

    def self_taught_short_text_clustering(self) -> STC:
        """
        Config Self-Taught Short Text Clustering algorithm

        :return SelfTaughtShortTextClustering:
            Model object
        """
        return STC(dimensions=self.cluster_params.get('dimensions'),
                   iterator=self.cluster_params.get('iterator'),
                   encoder_hidden_layer=self.cluster_params.get('encoder_hidden_layer'),
                   decoder_hidden_layer=self.cluster_params.get('decoder_hidden_layer'),
                   batch_size=16 if self.cluster_params.get('batch_size') is None else self.cluster_params.get(
                       'batch_size'),
                   epoch=15 if self.cluster_params.get('epoch') is None else self.cluster_params.get('epoch'),
                   optimizer='sgd' if self.cluster_params.get('optimizer') is None else self.cluster_params.get(
                       'optimizer'),
                   n_clusters=10 if self.cluster_params.get('n_clusters') is None else self.cluster_params.get(
                       'n_clusters'),
                   alpha=1.0 if self.cluster_params.get('alpha') is None else self.cluster_params.get('alpha'),
                   n_iterations=100 if self.cluster_params.get('n_iterations') is None else self.cluster_params.get(
                       'n_iterations'),
                   max_iterations=1000 if self.cluster_params.get(
                       'max_iterations') is None else self.cluster_params.get('max_iterations'),
                   early_stopping=False if self.cluster_params.get(
                       'early_stopping') is None else self.cluster_params.get('early_stopping'),
                   tol=0.0001 if self.cluster_params.get('tol') is None else self.cluster_params.get('tol'),
                   update_interval=500 if self.cluster_params.get(
                       'update_interval') is None else self.cluster_params.get('update_interval')
                   )

    @staticmethod
    def self_taught_short_text_clustering_param() -> dict:
        """
        Generate Self-Taught Short Text Clustering parameter configuration randomly

        :return: dict
        """
        return dict(n_clusters=np.random.randint(low=5, high=30),
                    n_iterations=np.random.randint(low=5, high=100),
                    max_iterations=np.random.randint(low=1000, high=5000),
                    update_interval=np.random.randint(low=50, high=1000),
                    alpha=np.random.uniform(low=0.1, high=1.0),
                    )


class ClusteringGenerator(Clustering):
    """
    Class for generating unsupervised learning clustering models
    """

    def __init__(self,
                 model_name: str = None,
                 cluster_params: dict = None,
                 models: List[str] = None,
                 random: bool = True,
                 seed: int = 1234
                 ):
        """
        :param model_name: str
            Name of the clustering model

        :param cluster_params: dict
            Pre-configured clustering model parameter

        :param models: List[str]
            Names of the clustering models

        :param random: bool
            Draw clustering model randomly

        :param seed: int
            Seed
        """
        super(ClusteringGenerator, self).__init__(cluster_params=cluster_params, seed=seed)
        self.model_name: str = model_name
        self.models: List[str] = models
        self.model = None
        self.model_param: dict = {}
        self.stc = None
        self.nmi: float = 0.0
        self.train_time: float = 0.0
        self.random: bool = random
        self.cluster_label: List[int] = []
        self.model_param_mutation: str = ''
        self.model_param_mutated: dict = {}

    def _build_vocab(self, x: np.ndarray):
        """
        Build text vocabulary

        :param x: np.ndarray
            Text data
        """
        self.vocab = set(d for doc in x for d in doc)
        self.vocab_size = len(self.vocab)

    def _doc_term_matrix(self, x: np.ndarray):
        """
        Generate document-term matrix

        :param x: np.ndarray
            Text data
        """
        self.vocab = corpora.Dictionary(documents=x, prune_at=2000000)
        self.document_term_matrix = [self.vocab.doc2bow(doc) for doc in x]

    def _eval(self, x: np.ndarray):
        """
        Internal cluster evaluation using semi-supervised Self-Taught Short Text Clustering algorithm to generate Normalized Mutual Information score

        :param x: np.ndarray
            Text data
        """
        _embedding: np.ndarray = get_sentence_embedding(text_data=x,
                                                        lang_model_name='paraphrase-xlm-r-multilingual-v1' if self.cluster_params.get('lang_model_name') is None else self.cluster_params.get('lang_model_name')
                                                        )
        _embedding_tensor: torch.tensor = torch.tensor(_embedding.astype(np.float32))
        _target_tensor: torch.tensor = torch.tensor(np.array(self.cluster_label).astype(np.float32))
        _data_tensor: TensorDataset = TensorDataset(_embedding_tensor, _target_tensor)
        _batch_size: int = 16 if self.cluster_params.get('batch_size') is None else self.cluster_params.get('batch_size')
        _data_loader: DataLoader = DataLoader(dataset=_data_tensor, batch_size=_batch_size, shuffle=True)
        self.cluster_params.update({'dimensions': [_embedding.shape[0], _embedding.shape[1]], 'iterator': _data_loader})
        self.stc = self.self_taught_short_text_clustering()
        self.stc.fit(x=_embedding_tensor)
        self.nmi = self.stc.nmi_score

    def generate_model(self) -> object:
        """
        Generate clustering model with randomized parameter configuration

        :return object
            Model object itself (self)
        """
        if self.random:
            if self.models is None:
                self.model_name = copy.deepcopy(np.random.choice(a=list(CLUSTER_ALGORITHMS.keys())))
            else:
                self.model_name = copy.deepcopy(np.random.choice(a=self.models))
            _model = copy.deepcopy(CLUSTER_ALGORITHMS.get(self.model_name))
        else:
            _model = copy.deepcopy(CLUSTER_ALGORITHMS.get(self.model_name))
        if len(self.cluster_params.keys()) == 0:
            self.model_param = getattr(Clustering(), '{}_param'.format(_model))()
            self.cluster_params = copy.deepcopy(self.model_param)
            _idx: int = 0 if len(self.model_param_mutated.keys()) == 0 else len(self.model_param_mutated.keys()) + 1
            self.model_param_mutated.update({str(_idx): {copy.deepcopy(self.model_name): {}}})
            for param in self.model_param.keys():
                self.model_param_mutated[str(_idx)][copy.deepcopy(self.model_name)].update(
                    {param: copy.deepcopy(self.model_param.get(param))})
        else:
            self.model_param = copy.deepcopy(self.cluster_params)
        self.model_param_mutation = 'params'
        self.model = copy.deepcopy(getattr(Clustering(cluster_params=self.cluster_params), _model)())
        return self

    def generate_params(self, param_rate: float = 0.1, force_param: dict = None) -> object:
        """
        Generate parameter for clustering models

        :param param_rate: float
            Rate of parameters of each model to mutate

        :param force_param: dict
            Parameter config to force explicitly

        :return object
            Model object itself (self)
        """
        pass

    def get_model_parameter(self) -> dict:
        """
        Get parameter "standard" config of given clustering models

        :return dict:
            Standard parameter config of given clustering models
        """
        _model_param: dict = {}
        if self.models is None:
            return _model_param
        else:
            for model in self.models:
                if model in CLUSTER_ALGORITHMS.keys():
                    _model = getattr(Clustering(), CLUSTER_ALGORITHMS.get(model))()
                    _param: dict = getattr(Clustering(), '{}_param'.format(CLUSTER_ALGORITHMS.get(model)))()
                    _model_random_param: dict = _model.__dict__.items()
                    for param in _model_random_param:
                        if param[0] in _param.keys():
                            _param.update({param[0]: param[1]})
                    _model_param.update({model: copy.deepcopy(_param)})
        return _model_param

    def eval(self):
        pass

    def predict(self):
        pass

    def train(self, x: np.ndarray):
        """
        Train or fit clustering model

        :param x: np.ndarray
            Text data
        """
        if self.model != 'gsdmm':
            self._build_vocab(x=x)
        _t0: datetime = datetime.now()
        self.cluster_label = self.model.fit(documents=x, vocab_size=self.vocab_size)
        self.train_time = (datetime.now() - _t0).seconds
        self._eval(x=x)
