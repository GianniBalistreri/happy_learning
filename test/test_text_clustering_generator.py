import unittest

from gensim import corpora
from happy_learning.text_clustering import GibbsSamplingDirichletMultinomialModeling, LatentDirichletAllocation, LatentSemanticIndexing, NonNegativeMatrixFactorization
from happy_learning.text_clustering_generator import Clustering, ClusteringGenerator


class ClusteringTest(unittest.TestCase):
    """
    Class for testing class Clustering
    """
    def test_gibbs_sampling_dirichlet_multinomial_model(self):
        self.assertTrue(expr=isinstance(GibbsSamplingDirichletMultinomialModeling(n_clusters=8,
                                                                                  n_iterations=30,
                                                                                  alpha=0.1,
                                                                                  beta=0.1
                                                                                  ),
                                        GibbsSamplingDirichletMultinomialModeling
                                        )
                        )

    def test_gibbs_sampling_dirichlet_multinomial_model_param(self):
        _gsdmm_param: dict = Clustering(cluster_params=None, train_data_path=None).gibbs_sampling_dirichlet_multinomial_modeling_param()
        self.assertTrue(expr=_gsdmm_param.get(list(_gsdmm_param.keys())[0]) != Clustering(cluster_params=None,
                                                                                          train_data_path=None
                                                                                          ).gibbs_sampling_dirichlet_multinomial_modeling_param().get(list(_gsdmm_param.keys())[0]))

    def test_latent_dirichlet_allocation(self):
        self.assertTrue(expr=isinstance(LatentDirichletAllocation(doc_term_matrix=[],
                                                                  vocab=corpora.Dictionary(documents=[], prune_at=2000000),
                                                                  n_clusters=10,
                                                                  n_iterations=200,
                                                                  ),
                                        LatentDirichletAllocation
                                        )
                        )

    def test_latent_dirichlet_allocation_param(self):
        _lda_param: dict = Clustering(cluster_params=None, train_data_path=None).latent_dirichlet_allocation_param()
        self.assertTrue(expr=_lda_param.get(list(_lda_param.keys())[0]) != Clustering(cluster_params=None,
                                                                                      train_data_path=None
                                                                                      ).latent_dirichlet_allocation_param().get(list(_lda_param.keys())[0]))

    def test_latent_semantic_indexing(self):
        self.assertTrue(expr=isinstance(LatentSemanticIndexing(doc_term_matrix=[],
                                                               vocab=corpora.Dictionary(documents=[], prune_at=2000000),
                                                               n_clusters=10,
                                                               n_iterations=200,
                                                               ),
                                        LatentSemanticIndexing
                                        )
                        )

    def test_latent_semantic_indexing_param(self):
        _lsi_param: dict = Clustering(cluster_params=None, train_data_path=None).latent_semantic_indexing_param()
        self.assertTrue(expr=_lsi_param.get(list(_lsi_param.keys())[0]) != Clustering(cluster_params=None,
                                                                                      train_data_path=None
                                                                                      ).latent_semantic_indexing_param().get(list(_lsi_param.keys())[0]))

    def test_non_negative_matrix_factorization(self):
        self.assertTrue(expr=isinstance(NonNegativeMatrixFactorization(lang='en',
                                                                       doc_term_matrix=[],
                                                                       vocab=corpora.Dictionary(documents=[], prune_at=2000000),
                                                                       n_clusters=10,
                                                                       n_iterations=200
                                                                       ),
                                        NonNegativeMatrixFactorization
                                        )
                        )

    def test_non_negative_matrix_factorization_param(self):
        _nmf_param: dict = Clustering(cluster_params=None, train_data_path=None).non_negative_matrix_factorization_param()
        self.assertTrue(expr=_nmf_param.get(list(_nmf_param.keys())[0]) != Clustering(cluster_params=None,
                                                                                      train_data_path=None
                                                                                      ).non_negative_matrix_factorization_param().get(list(_nmf_param.keys())[0]))


class ClusterNetworkTest(unittest.TestCase):
    """
    Class for testing class ClusteringGenerator
    """
    def test_generate_model(self):
        _net_gen: object = ClusteringGenerator(predictor='',
                                               model_name=None,
                                               cluster_params=None,
                                               models=['gsdmm'],
                                               tokenize=False,
                                               random=True,
                                               sep='\t',
                                               cloud=None
                                               ).generate_model()
        self.assertTrue(expr=isinstance(_net_gen.model, GibbsSamplingDirichletMultinomialModeling))

    def test_generate_params(self):
        pass

    def test_get_model_parameter(self):
        pass

    def test_train(self):
        pass
