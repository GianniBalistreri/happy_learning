import copy
import numpy as np

from .text_clustering import GibbsSamplingDirichletMultinomialModeling, LatentDirichletAllocation, LatentSemanticIndexing, SelfTaughtShortTextClustering
from datetime import datetime
from gensim import corpora
from typing import List

CLUSTER_ALGORITHMS: dict = dict(gsdmm='gibbs_sampling_dirichlet_multinomial_modeling',
                                lda='latent_dirichlet_allocation',
                                lsi='latent_semantic_indexing',
                                stc='self_taught_short_text_clustering'
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

    def gibbs_sampling_dirichlet_multinomial_modeling(self) -> GibbsSamplingDirichletMultinomialModeling:
        """
        Config Gibbs Sampling Dirichlet Multinomial Modeling algorithm

        :return GibbsSamplingDirichletMultinomialModeling:
            Model object
        """
        return GibbsSamplingDirichletMultinomialModeling(n_clusters=10 if self.cluster_params.get('n_clusters') is None else self.cluster_params.get('n_clusters'),
                                                         n_iterations=5 if self.cluster_params.get('n_iterations') is None else self.cluster_params.get('n_iterations'),
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
        return dict(n_clusters=np.random.randint(low=5, high=50),
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
        return LatentDirichletAllocation()

    def latent_semantic_indexing(self) -> LatentSemanticIndexing:
        """
        Config Latent Semantic Indexing algorithm

        :return LatentSemanticIndexing:
            Model object
        """
        return LatentSemanticIndexing()

    def self_taught_short_text_clustering(self) -> SelfTaughtShortTextClustering:
        """
        Config Self-Taught Short Text Clustering algorithm

        :return SelfTaughtShortTextClustering:
            Model object
        """
        return SelfTaughtShortTextClustering()


class ClusteringGenerator(Clustering):
    """
    Class for generating supervised learning classification models
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
        self.vocab: dict = {}
        self.vocab_size: int = 0
        self.document_term_matrix: list = []
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
        self._build_vocab(x=x)
        _t0: datetime = datetime.now()
        self.cluster_label = self.model.fit(documents=x, vocab_size=self.vocab_size)
        self.train_time = (datetime.now() - _t0).seconds
