import copy
import unittest

from happy_learning.text_clustering import GibbsSamplingDirichletMultinomialModeling, LatentDirichletAllocation, LatentSemanticIndexing, NonNegativeMatrixFactorization
from happy_learning.text_clustering_generator import CLUSTER_ALGORITHMS, Clustering, ClusteringGenerator

DATA_FILE_PATH: str = 'data/tweets.csv'
N_CLUSTERS: int = 10


class ClusteringTest(unittest.TestCase):
    """
    Class for testing class Clustering
    """
    def test_gibbs_sampling_dirichlet_multinomial_model(self):
        self.assertTrue(expr=isinstance(Clustering(cluster_params=None, train_data_path=None).gibbs_sampling_dirichlet_multinomial_modeling(),
                                        GibbsSamplingDirichletMultinomialModeling
                                        )
                        )

    def test_gibbs_sampling_dirichlet_multinomial_model_param(self):
        _gsdmm_param: dict = Clustering(cluster_params=None, train_data_path=None).gibbs_sampling_dirichlet_multinomial_modeling_param()
        self.assertTrue(expr=_gsdmm_param.get(list(_gsdmm_param.keys())[0]) != Clustering(cluster_params=None,
                                                                                          train_data_path=None
                                                                                          ).gibbs_sampling_dirichlet_multinomial_modeling_param().get(list(_gsdmm_param.keys())[0]))

    def test_latent_dirichlet_allocation(self):
        self.assertTrue(expr=isinstance(Clustering(cluster_params=None, train_data_path=None).latent_dirichlet_allocation(),
                                        LatentDirichletAllocation
                                        )
                        )

    def test_latent_dirichlet_allocation_param(self):
        _lda_param: dict = Clustering(cluster_params=None, train_data_path=None).latent_dirichlet_allocation_param()
        self.assertTrue(expr=_lda_param.get(list(_lda_param.keys())[0]) != Clustering(cluster_params=None,
                                                                                      train_data_path=None
                                                                                      ).latent_dirichlet_allocation_param().get(list(_lda_param.keys())[0]))

    def test_latent_semantic_indexing(self):
        self.assertTrue(expr=isinstance(Clustering(cluster_params=None, train_data_path=None).latent_semantic_indexing(),
                                        LatentSemanticIndexing
                                        )
                        )

    def test_latent_semantic_indexing_param(self):
        _lsi_param: dict = Clustering(cluster_params=None, train_data_path=None).latent_semantic_indexing_param()
        self.assertTrue(expr=_lsi_param.get(list(_lsi_param.keys())[0]) != Clustering(cluster_params=None,
                                                                                      train_data_path=None
                                                                                      ).latent_semantic_indexing_param().get(list(_lsi_param.keys())[0]))

    def test_non_negative_matrix_factorization(self):
        self.assertTrue(expr=isinstance(Clustering(cluster_params=None, train_data_path=None).non_negative_matrix_factorization(),
                                        NonNegativeMatrixFactorization
                                        )
                        )

    def test_non_negative_matrix_factorization_param(self):
        _nmf_param: dict = Clustering(cluster_params=None, train_data_path=None).non_negative_matrix_factorization_param()
        self.assertTrue(expr=_nmf_param.get(list(_nmf_param.keys())[0]) != Clustering(cluster_params=None,
                                                                                      train_data_path=None
                                                                                      ).non_negative_matrix_factorization_param().get(list(_nmf_param.keys())[0]))


class ClusteringGeneratorTest(unittest.TestCase):
    """
    Class for testing class ClusteringGenerator
    """
    def test_generate_gibbs_sampling_dirichlet_multinomial_model(self):
        _net_gen: object = ClusteringGenerator(predictor='tweet',
                                               model_name=None,
                                               cluster_params=None,
                                               models=['gsdmm'],
                                               tokenize=True,
                                               random=True,
                                               sep=',',
                                               cloud=None,
                                               train_data_path=DATA_FILE_PATH
                                               ).generate_model()
        self.assertTrue(expr=isinstance(_net_gen.model, GibbsSamplingDirichletMultinomialModeling))

    def test_generate_latent_dirichlet_allocation_model(self):
        _net_gen: object = ClusteringGenerator(predictor='tweet',
                                               model_name=None,
                                               cluster_params=None,
                                               models=['lda'],
                                               tokenize=False,
                                               random=True,
                                               sep=',',
                                               cloud=None,
                                               train_data_path=DATA_FILE_PATH
                                               ).generate_model()
        self.assertTrue(expr=isinstance(_net_gen.model, LatentDirichletAllocation))

    def test_generate_latent_semantic_indexing_model(self):
        _net_gen: object = ClusteringGenerator(predictor='tweet',
                                               model_name=None,
                                               cluster_params=None,
                                               models=['lsi'],
                                               tokenize=False,
                                               random=True,
                                               sep=',',
                                               cloud=None,
                                               train_data_path=DATA_FILE_PATH
                                               ).generate_model()
        self.assertTrue(expr=isinstance(_net_gen.model, LatentSemanticIndexing))

    def test_generate_non_negative_matrix_factorization_model(self):
        _net_gen: object = ClusteringGenerator(predictor='tweet',
                                               model_name=None,
                                               cluster_params=None,
                                               models=['nmf'],
                                               tokenize=False,
                                               random=True,
                                               sep=',',
                                               cloud=None,
                                               train_data_path=DATA_FILE_PATH
                                               ).generate_model()
        self.assertTrue(expr=isinstance(_net_gen.model, NonNegativeMatrixFactorization))

    def test_generate_params(self):
        _model_generator: ClusteringGenerator = ClusteringGenerator(predictor='tweet',
                                                                    model_name=None,
                                                                    cluster_params=None,
                                                                    models=list(CLUSTER_ALGORITHMS.keys()),
                                                                    sep=',',
                                                                    train_data_path=DATA_FILE_PATH,
                                                                    )
        _model = _model_generator.generate_model()
        _mutated_param: dict = copy.deepcopy(_model.model_param_mutated)
        _model_generator.generate_params(param_rate=0.1, force_param=None)
        self.assertTrue(expr=len(_mutated_param.keys()) < len(_model_generator.model_param_mutated.keys()))

    def test_get_model_parameter(self):
        self.assertTrue(expr=len(ClusteringGenerator(predictor='tweet',
                                                     model_name=None,
                                                     cluster_params=None,
                                                     models=list(CLUSTER_ALGORITHMS.keys())
                                                     ).get_model_parameter().keys()
                                 ) > 0
                        )

    def test_train_gibbs_sampling_dirichlet_multinomial_model(self):
        _model_generator: ClusteringGenerator = ClusteringGenerator(predictor='tweet',
                                                                    model_name=None,
                                                                    cluster_params=None,
                                                                    models=['gsdmm'],
                                                                    sep=',',
                                                                    train_data_path=DATA_FILE_PATH,
                                                                    )
        _model = _model_generator.generate_model()
        _model.train()
        self.assertTrue(expr=len(_model.cluster_label) > 0 and _model.nmi > 0.0)

    def test_train_latent_dirichlet_allocation(self):
        _model_generator: ClusteringGenerator = ClusteringGenerator(predictor='tweet',
                                                                    model_name=None,
                                                                    cluster_params=None,
                                                                    models=['lda'],
                                                                    sep=',',
                                                                    train_data_path=DATA_FILE_PATH,
                                                                    )
        _model = _model_generator.generate_model()
        _model.train()
        self.assertTrue(expr=len(_model.cluster_label) > 0 and _model.nmi > 0.0)

    def test_train_latent_semantic_indexing(self):
        _model_generator: ClusteringGenerator = ClusteringGenerator(predictor='tweet',
                                                                    model_name=None,
                                                                    cluster_params=None,
                                                                    models=['lsi'],
                                                                    sep=',',
                                                                    train_data_path=DATA_FILE_PATH,
                                                                    )
        _model = _model_generator.generate_model()
        _model.train()
        self.assertTrue(expr=len(_model.cluster_label) > 0 and _model.nmi > 0.0)

    def test_train_non_negative_matrix_factorization(self):
        _model_generator: ClusteringGenerator = ClusteringGenerator(predictor='tweet',
                                                                    model_name=None,
                                                                    cluster_params=None,
                                                                    models=['nmf'],
                                                                    sep=',',
                                                                    train_data_path=DATA_FILE_PATH,
                                                                    )
        _model = _model_generator.generate_model()
        _model.train()
        self.assertTrue(expr=len(_model.cluster_label) > 0 and _model.nmi > 0.0)
