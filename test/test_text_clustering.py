import pandas as pd
import re
import unittest

from gensim import corpora
from happy_learning.text_clustering import GibbsSamplingDirichletMultinomialModeling, LatentDirichletAllocation, LatentSemanticIndexing, NonNegativeMatrixFactorization
from typing import List

DATA_FILE_PATH: str = 'data/tweets.csv'
N_CLUSTERS: int = 10
TWEETS_DF: pd.DataFrame = pd.read_csv(DATA_FILE_PATH, sep=',')
TWEETS_DF['tokens'] = TWEETS_DF.tokens.apply(lambda x: re.split('\s', x))
TWEETS_TOKENS: List[List[str]] = TWEETS_DF['tokens'].tolist()


class GibbsSamplingDirichletMultinomialModelingTest(unittest.TestCase):
    """
    Class for testing class GibbsSamplingDirichletMultinomialModeling
    """
    def test_doc_to_label(self):
        pass

    def test_fit(self):
        _vocab = set(x for doc in TWEETS_TOKENS for x in doc)
        _n_terms: int = len(_vocab)
        _mgp = GibbsSamplingDirichletMultinomialModeling(vocab_size=_n_terms,
                                                         n_clusters=N_CLUSTERS,
                                                         alpha=0.1,
                                                         beta=0.5,
                                                         n_iterations=5
                                                         )
        _y: List[int] = _mgp.fit(documents=TWEETS_TOKENS)
        self.assertTrue(expr=N_CLUSTERS == len(set(_y)) and len(TWEETS_TOKENS) == len(_y))

    def test_get_word_count_each_cluster(self):
        _vocab = set(x for doc in TWEETS_TOKENS for x in doc)
        _n_terms: int = len(_vocab)
        _mgp = GibbsSamplingDirichletMultinomialModeling(vocab_size=_n_terms,
                                                         n_clusters=N_CLUSTERS,
                                                         alpha=0.1,
                                                         beta=0.5,
                                                         n_iterations=5
                                                         )
        _y: List[int] = _mgp.fit(documents=TWEETS_TOKENS)
        _top_n_words: int = 5
        _top_words_each_cluster: dict = _mgp.get_top_words_each_cluster(top_n_words=_top_n_words)
        self.assertTrue(expr=N_CLUSTERS == len(_top_words_each_cluster.keys()) and _top_n_words == len(_top_words_each_cluster['0']))

    def test_generate_topic_allocation(self):
        _vocab = set(x for doc in TWEETS_TOKENS for x in doc)
        _n_terms: int = len(_vocab)
        _mgp = GibbsSamplingDirichletMultinomialModeling(vocab_size=_n_terms,
                                                         n_clusters=N_CLUSTERS,
                                                         alpha=0.1,
                                                         beta=0.5,
                                                         n_iterations=5
                                                         )
        _y: List[int] = _mgp.fit(documents=TWEETS_TOKENS)
        _topic_allocation: dict = _mgp.generate_topic_allocation(documents=TWEETS_DF['tweet'].values.tolist())
        self.assertTrue(expr=len(_topic_allocation['cluster']) == TWEETS_DF.shape[0])

    def test_word_importance_each_cluster(self):
        _vocab = set(x for doc in TWEETS_TOKENS for x in doc)
        _n_terms: int = len(_vocab)
        _mgp = GibbsSamplingDirichletMultinomialModeling(vocab_size=_n_terms,
                                                         n_clusters=N_CLUSTERS,
                                                         alpha=0.1,
                                                         beta=0.5,
                                                         n_iterations=5
                                                         )
        _y: List[int] = _mgp.fit(documents=TWEETS_TOKENS)
        _word_importance: dict = _mgp.word_importance_each_cluster()
        _words: List[str] = list(_word_importance.keys())
        self.assertTrue(expr=len(_word_importance) > 0 and isinstance(_word_importance.get(_words[0]), float))


class LatentDirichletAllocationTest(unittest.TestCase):
    """
    Class for testing class LatentDirichletAllocation
    """
    def test_fit(self):
        _vocab: corpora.Dictionary = corpora.Dictionary(documents=TWEETS_DF['tokens'].values.tolist())
        _document_term_matrix: list = [_vocab.doc2bow(doc) for doc in TWEETS_DF['tokens'].values.tolist()]
        _lda: LatentDirichletAllocation = LatentDirichletAllocation(doc_term_matrix=_document_term_matrix,
                                                                    vocab=_vocab,
                                                                    n_clusters=N_CLUSTERS,
                                                                    n_iterations=10,
                                                                    decay=1.0
                                                                    )
        _y: List[int] = _lda.fit(documents=TWEETS_DF['tokens'].values.tolist())
        self.assertTrue(expr=N_CLUSTERS == len(set(_y)) and TWEETS_DF.shape[0] == len(_y))


class LatentSemanticIndexingTest(unittest.TestCase):
    """
    Class for testing class LatentSemanticIndexing
    """
    def test_fit(self):
        _vocab: corpora.Dictionary = corpora.Dictionary(documents=TWEETS_DF['tokens'].values.tolist())
        _document_term_matrix: list = [_vocab.doc2bow(doc) for doc in TWEETS_DF['tokens'].values.tolist()]
        _lsi: LatentSemanticIndexing = LatentSemanticIndexing(doc_term_matrix=_document_term_matrix,
                                                              vocab=_vocab,
                                                              n_clusters=N_CLUSTERS,
                                                              n_iterations=10,
                                                              decay=1.0,
                                                              training_algorithm='multi_pass_stochastic'
                                                              )
        _y: List[int] = _lsi.fit(documents=TWEETS_DF['tokens'].values.tolist())
        self.assertTrue(expr=N_CLUSTERS == len(set(_y)) and TWEETS_DF.shape[0] == len(_y))


class NonNegativeMatrixFactorizationTest(unittest.TestCase):
    """
    Class for testing class NonNegativeMatrixFactorization
    """
    def test_fit(self):
        _nmf: NonNegativeMatrixFactorization = NonNegativeMatrixFactorization(lang='english',
                                                                              n_clusters=N_CLUSTERS,
                                                                              n_iterations=10
                                                                              )
        _y: List[int] = _nmf.fit(documents=TWEETS_DF['tweet'])
        self.assertTrue(expr=N_CLUSTERS == len(set(_y)) and TWEETS_DF.shape[0] == len(_y))

    def test_get_word_count_each_cluster(self):
        _nmf: NonNegativeMatrixFactorization = NonNegativeMatrixFactorization(lang='english',
                                                                              n_clusters=N_CLUSTERS,
                                                                              n_iterations=10
                                                                              )
        _y: List[int] = _nmf.fit(documents=TWEETS_DF['tweet'])
        _top_n_words: int = 5
        _top_words_each_cluster: dict = _nmf.get_top_words_each_cluster(top_n_words=_top_n_words)
        self.assertTrue(expr=N_CLUSTERS == len(_top_words_each_cluster.keys()) and _top_n_words == len(_top_words_each_cluster['0']))
