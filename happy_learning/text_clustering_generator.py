import copy
import numpy as np
import re
import pandas as pd
import torch

from .self_taught_short_text_clustering import get_sentence_embedding, STC
from .text_clustering import GibbsSamplingDirichletMultinomialModeling, LatentDirichletAllocation, \
    LatentSemanticIndexing, NonNegativeMatrixFactorization
from datetime import datetime
from easyexplore.data_import_export import CLOUD_PROVIDER, DataImporter
from gensim import corpora
from torch.utils.data import TensorDataset, DataLoader
from typing import List

CLUSTER_ALGORITHMS: dict = dict(gsdmm='gibbs_sampling_dirichlet_multinomial_modeling',
                                lda='latent_dirichlet_allocation',
                                lsi='latent_semantic_indexing',
                                nmf='non_negative_matrix_factorization',
                                #stc='self_taught_short_text_clustering'
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
    def __init__(self,
                 cluster_params: dict = None,
                 train_data_path: str = None,
                 seed: int = 1234
                 ):
        """
        :param cluster_params: dict
            Pre-configured clustering model parameter

        :param train_data_path: str
            Complete file path of the training data

        :param seed: int
            Seed
        """
        self.cluster_params: dict = {} if cluster_params is None else cluster_params
        self.seed: int = seed
        self.vocab = None
        self.vocab_size: int = 0
        self.document_term_matrix: list = []
        self.train_data_path: str = train_data_path

    def gibbs_sampling_dirichlet_multinomial_modeling(self) -> GibbsSamplingDirichletMultinomialModeling:
        """
        Config Gibbs Sampling Dirichlet Multinomial Modeling algorithm

        :return GibbsSamplingDirichletMultinomialModeling:
            Model object
        """
        return GibbsSamplingDirichletMultinomialModeling(
            vocab_size=self.vocab_size,
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
                 predictor: str,
                 model_name: str = None,
                 cluster_params: dict = None,
                 models: List[str] = None,
                 tokenize: bool = False,
                 random: bool = True,
                 sep: str = '\t',
                 cloud: str = None,
                 train_data_path: str = None,
                 sentence_embedding_model_path: str = None,
                 seed: int = 1234
                 ):
        """
        :param predictor: str
            Name of the text feature

        :param model_name: str
            Name of the clustering model

        :param cluster_params: dict
            Pre-configured clustering model parameter

        :param models: List[str]
            Names of the clustering models

        :param tokenize: bool
            Apply word tokenization to text data

        :param random: bool
            Draw clustering model randomly

        :param sep: str
            Separator

        :param cloud: str
            Name of the cloud provider
                -> google: Google Cloud Storage
                -> aws: AWS Cloud

        :param train_data_path: str
            Complete file path of the training data

        :param sentence_embedding_model_path: str
            Local path of the sentence embedding model

        :param seed: int
            Seed
        """
        super(ClusteringGenerator, self).__init__(cluster_params=cluster_params,
                                                  train_data_path=train_data_path,
                                                  seed=seed
                                                  )
        self.predictor: str = predictor
        self.model_name: str = model_name
        self.models: List[str] = models
        self.model = None
        self.model_param: dict = {}
        self.tokenize: bool = tokenize
        self.stc = None
        self.nmi: float = 0.0
        self.train_time: float = 0.0
        self.random: bool = random
        self.cluster_label: List[int] = []
        self.model_param_mutation: str = ''
        self.model_param_mutated: dict = {}
        self.sep: str = sep
        self.x: np.ndarray = None
        self.cloud: str = cloud
        self.sentence_embedding_model_path: str = sentence_embedding_model_path
        if self.cloud is None:
            self.bucket_name: str = None
        else:
            if self.cloud not in CLOUD_PROVIDER:
                raise ClusteringException('Cloud provider ({}) not supported'.format(cloud))
            self.bucket_name: str = self.train_data_path.split("//")[1].split("/")[0]

    def _build_vocab(self):
        """
        Build text vocabulary
        """
        self.vocab = set(d for doc in self.x for d in doc)
        self.vocab_size = len(self.vocab)

    def _doc_term_matrix(self):
        """
        Generate document-term matrix
        """
        self.vocab = corpora.Dictionary(documents=self.x, prune_at=2000000)
        self.document_term_matrix = [self.vocab.doc2bow(doc) for doc in self.x]

    def _eval(self):
        """
        Internal cluster evaluation using semi-supervised Self-Taught Short Text Clustering algorithm to generate Normalized Mutual Information score
        """
        _embedding: np.ndarray = get_sentence_embedding(text_data=self.x,
                                                        lang_model_name='paraphrase-xlm-r-multilingual-v1' if self.cluster_params.get('lang_model_name') is None else self.cluster_params.get('lang_model_name'),
                                                        lang_model_path=self.sentence_embedding_model_path
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

    def _import_data(self):
        """
        Import data set
        """
        _data_set: pd.DataFrame = DataImporter(file_path=self.train_data_path,
                                               as_data_frame=True,
                                               use_dask=False,
                                               create_dir=False,
                                               sep=self.sep,
                                               cloud=self.cloud,
                                               bucket_name=self.bucket_name
                                               ).file(table_name=None)
        if self.tokenize:
            self.x = _data_set[self.predictor].apply(lambda x: re.split('\s', x)).values
        else:
            self.x = _data_set[self.predictor].values
        del _data_set

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
        if param_rate > 1:
            _rate: float = 1.0
        else:
            if param_rate > 0:
                _rate: float = param_rate
            else:
                _rate: float = 0.1
        _params: dict = getattr(Clustering(), '{}_param'.format(CLUSTER_ALGORITHMS.get(self.model_name)))()
        _force_param: dict = {} if force_param is None else force_param
        _param_choices: List[str] = [p for p in _params.keys() if p not in _force_param.keys()]
        _gen_n_params: int = round(len(_params.keys()) * _rate)
        if _gen_n_params == 0:
            _gen_n_params = 1
        self.model_param_mutated.update(
            {len(self.model_param_mutated.keys()) + 1: {copy.deepcopy(self.model_name): {}}})
        _new_model_params: dict = copy.deepcopy(self.model_param)
        for param in _force_param.keys():
            _new_model_params.update({param: _force_param.get(param)})
        for _ in range(0, _gen_n_params, 1):
            _param: str = np.random.choice(a=_param_choices)
            _new_model_params.update({_param: _params.get(_param)})
            self.model_param_mutated[list(self.model_param_mutated.keys())[-1]][copy.deepcopy(self.model_name)].update(
                {_param: _params.get(_param)})
        self.model_param_mutation = 'new_model'
        self.model_param = copy.deepcopy(_new_model_params)
        self.cluster_params = self.model_param
        self.model = getattr(Clustering(cluster_params=self.cluster_params), CLUSTER_ALGORITHMS.get(self.model_name))()
        return self

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

    def train(self):
        """
        Train or fit clustering model
        """
        self._import_data()
        if self.model in ['gsdmm', 'lda', 'lsi']:
            self._build_vocab()
        _t0: datetime = datetime.now()
        self.cluster_label = self.model.fit(documents=self.x)
        self.train_time = (datetime.now() - _t0).seconds
        self._eval()
        self.x = None
