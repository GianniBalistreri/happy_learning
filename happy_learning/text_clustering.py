"""

Text clustering algorithms

"""

import numpy as np
import math
import pandas as pd
import pyLDAvis

from easyexplore.data_import_export import DataExporter
from fast_gsdmm import GSDMM
from gensim import corpora
from gensim.models import LdaModel, LsiModel
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple


def _rank_clusters(model, text_data: List[str], vocab: corpora.Dictionary) -> tuple:
    """
    Rank clusters

    :param model: LdaModel or LsiModel
        Trained LDA or LSI model

    :param text_data: str
        Text data

    :param vocab: corpora.Dictionary
        Pre-defined (gensim) corpus dictionary

    :return: tuple
        Top ranked topic & probability distribution for all clusters
    """
    _bag_of_words_vector = vocab.doc2bow(text_data)
    _lda_vector = model[_bag_of_words_vector]
    _cluster_mapping: dict = {str(c[0]): c[0] for c in _lda_vector}
    _top_topic = _cluster_mapping[max(_lda_vector, key=lambda item: item[1])[0]]
    _distribution = {_cluster_mapping[topic_no]: proportion for topic_no, proportion in _lda_vector}
    return _top_topic, _distribution


class GibbsSamplingDirichletMultinomialModeling:
    """
    Class for building Gibbs Sampling Dirichlet Multinomial Modeling model
    """
    def __init__(self,
                 vocab: List[str],
                 n_clusters: int = 8,
                 n_iterations: int = 30,
                 alpha: float = 0.1,
                 beta: float = 0.1,
                 fast: bool = True,
                 ):
        """
        :param vocab: List[str]
            Vocabulary

        :param n_clusters: int
            Number of pre-defined clusters

        :param n_iterations: int
            Number of iterations

        :param alpha: float
            Probability of joining an empty cluster

        :param beta: float
            Affinity of a cluster element with same characteristics as other elements

        :param fast: bool
            Use C++ or Python implementation of the fit & document_scoring methods
        """
        self.fast: bool = fast
        self.alpha: float = alpha
        self.beta: float = beta
        self.n_clusters: int = n_clusters
        self.n_iterations: int = n_iterations if n_iterations > 5 else 30
        self.n_documents: int = 0
        self.vocab: List[str] = list(set(vocab))
        self.vocab_size: int = len(self.vocab)
        self.probability_vector: List[float] = []
        self.cluster_word_count: List[int] = [0 for _ in range(0, self.n_clusters, 1)]
        self.cluster_document_count: List[int] = [0 for _ in range(0, self.n_clusters, 1)]
        self.cluster_word_distribution: List[dict] = [{} for _ in range(0, self.n_clusters, 1)]
        if self.fast:
            self.model: GSDMM = GSDMM(self.vocab,
                                      self.vocab_size,
                                      self.n_clusters,
                                      self.n_iterations,
                                      self.alpha,
                                      self.beta
                                      )
        else:
            self.model = None

    def _document_scoring(self, document: List[str]) -> List[float]:
        """
        Score a document

        :param document: List[str]:
            Tokenized document

        :return: list[float]:
            Probability vector where each component represents the probability of the document appearing in a particular cluster
        """
        if self.fast:
            # Use C++ implementation:
            return self.model.document_scoring(document,
                                               self.alpha,
                                               self.beta,
                                               self.n_clusters,
                                               self.vocab_size,
                                               self.cluster_word_count,
                                               self.cluster_document_count,
                                               self.cluster_word_distribution
                                               )
        else:
            # Use Python implementation:
            _p = [0 for _ in range(0, self.n_clusters, 1)]
            #  We break the formula into the following pieces
            #  p = n1 * n2 / (d1 * d2) = np.exp(ln1 - ld1 + ln2 - ld2)
            #  lN1 = np.log(m_z[z] + alpha)
            #  lN2 = np.log(D - 1 + K*alpha)
            #  lN2 = np.log(product(n_z_w[w] + beta)) = sum(np.log(n_z_w[w] + beta))
            #  lD2 = np.log(product(n_z[d] + V*beta + i -1)) = sum(np.log(n_z[d] + V*beta + i -1))
            _ld1 = np.log(self.n_documents - 1 + self.n_clusters * self.alpha)
            _document_size = len(document)
            for label in range(0, self.n_clusters, 1):
                _ln1 = np.log(self.cluster_word_count[label] + self.alpha)
                _ln2 = 0
                _ld2 = 0
                for word in document:
                    _ln2 += np.log(self.cluster_word_distribution[label].get(word, 0) + self.beta)
                for j in range(1, _document_size + 1, 1):
                    _ld2 += np.log(self.cluster_word_count[label] + self.vocab_size * self.beta + j - 1)
                _p[label] = np.exp(_ln1 - _ld1 + _ln2 - _ld2)
            # normalize the probability vector
            _normalized_probability_vector = sum(_p)
            _normalized_probability_vector = _normalized_probability_vector if _normalized_probability_vector > 0 else 1
            return [probability / _normalized_probability_vector for probability in _p]

    def _sampling(self) -> int:
        """
        Sample with probability vector from a multinomial distribution

        :return: int
            Index of randomly selected output
        """
        return [i for i, entry in enumerate(np.random.multinomial(1, self.probability_vector)) if entry != 0][0]

    def add_cluster(self,
                    cluster_word_count: List[int],
                    cluster_document_count: List[int],
                    cluster_word_distribution: List[dict]
                    ):
        """
        Add new cluster information

        :param cluster_word_count: List[int]
            Word counting of each new cluster

        :param cluster_document_count: List[int]
             Document counting of each new cluster

        :param cluster_word_distribution: List[dict]
            Word distribution of each new cluster
        """
        self.cluster_word_count.extend(cluster_word_count)
        self.cluster_document_count.extend(cluster_document_count)
        self.cluster_word_distribution.extend(cluster_word_distribution)
        self.n_documents = sum(self.cluster_document_count)
        self.n_clusters = len(self.cluster_document_count)
        _new_vocabulary: List[str] = []
        for cluster in range(0, self.n_clusters, 1):
            _new_vocabulary.extend(list(self.cluster_word_distribution[cluster].keys()))
        self.vocab_size = len(list(set(_new_vocabulary)))

    def adjust_parameters(self, alpha: float, beta: float):
        """
        Adjust model specific hyper parameters

        :param alpha: float
             Probability of joining an empty cluster

        :param beta: float
             Affinity of a cluster element with same characteristics as other elements
        """
        self.alpha = alpha
        self.beta = beta

    def cluster_similarity(self, cluster_labels: List[int] = None, top_n_words: int = 10, last_n_words: int = 0) -> Dict[str, Dict[str, float]]:
        """
        Calculate similarity of learned clusters using top-n-words and last-n-words

        :param cluster_labels: List[int]
            Cluster labels to delete

        :param top_n_words: int
            Number of top words to compare

        :param last_n_words: int
            Number of last words to compare

        :return: Dict[str, Dict[str, float]]
            Similarity score based on comparison of top-n-words and last-n-words
        """
        _cluster_labels: List[int] = [i for i in range(0, self.n_clusters, 1)] if cluster_labels is None else cluster_labels
        _top_n_words_each_cluster: dict = self.get_top_words_each_cluster(top_n_words=top_n_words if top_n_words > 0 else 10)
        _last_n_words_each_cluster: dict = self.get_last_words_each_cluster(last_n_words=last_n_words if last_n_words > 0 else 0)
        _cluster_similarity: Dict[str, Dict[str, float]] = {}
        for first_cluster in _cluster_labels:
            _cluster_similarity.update({str(first_cluster): {}})
            _first_cluster_top_n_words: List[str] = [w[0] for w in _top_n_words_each_cluster.get(str(first_cluster))]
            for second_cluster in _cluster_labels:
                if first_cluster == second_cluster:
                    continue
                _second_cluster_top_n_words: List[str] = [w[0] for w in _top_n_words_each_cluster.get(str(second_cluster))]
                _similarity: float = 0
                for top in _first_cluster_top_n_words:
                    if top in _second_cluster_top_n_words:
                        _similarity += 1
                _similarity = 0 if _similarity == 0 else _similarity / len(_first_cluster_top_n_words)
                _cluster_similarity[str(first_cluster)].update({str(second_cluster): _similarity})
        return _cluster_similarity

    def delete_cluster(self, cluster_labels: List[int]):
        """
        Delete cluster information

        :param cluster_labels: List[int]
            Cluster labels to delete
        """
        for cluster in cluster_labels:
            self.n_clusters -= 1
            self.n_documents -= self.cluster_document_count[cluster]
            del self.cluster_word_count[cluster]
            del self.cluster_document_count[cluster]
            del self.cluster_word_distribution[cluster]
        _new_vocabulary: List[str] = []
        for i in range(0, self.n_clusters, 1):
            _new_vocabulary.extend(list(self.cluster_word_distribution[i].keys()))
        self.vocab_size = len(list(set(_new_vocabulary)))

    def fit(self, documents: List[List[str]]) -> List[int]:
        """
        Fitting GSDMM clustering algorithm

        :param documents: List[List[str]]
            Tokenized documents

        :return: List[int]
            Cluster labels
        """
        self.n_documents = len(documents)
        if self.fast:
            # Use C++ implementation:
            _document_cluster, self.cluster_word_count, self.cluster_document_count, self.cluster_word_distribution = self.model.fit(documents)
        else:
            # Use Python implementation:
            _document_cluster: List[int] = [_ for _ in range(0, self.n_documents, 1)]
            # initialize the clusters
            _n_clusters: int = self.n_clusters
            for i, doc in enumerate(documents):
                # choose a random initial cluster for the document
                self.probability_vector = [1.0 / self.n_clusters for _ in range(0, self.n_clusters, 1)]
                _i: int = self._sampling()
                _document_cluster[i] = _i
                self.cluster_document_count[_i] += 1
                self.cluster_word_count[_i] += len(doc)
                for word in doc:
                    if word not in self.cluster_word_distribution[_i]:
                        self.cluster_word_distribution[_i][word] = 0
                    self.cluster_word_distribution[_i][word] += 1
            for _iter in range(0, self.n_iterations, 1):
                _total_transfers: int = 0
                for i, doc in enumerate(documents):
                    # remove the doc from it's current cluster
                    _old_cluster: int = _document_cluster[i]
                    self.cluster_document_count[_old_cluster] -= 1
                    self.cluster_word_count[_old_cluster] -= len(doc)
                    for word in doc:
                        self.cluster_word_distribution[_old_cluster][word] -= 1
                        # compact dictionary to save space
                        if self.cluster_word_distribution[_old_cluster][word] == 0:
                            del self.cluster_word_distribution[_old_cluster][word]
                    # draw sample from distribution to find new cluster
                    self.probability_vector = self._document_scoring(document=documents[i])
                    _new_cluster: int = self._sampling()
                    # transfer doc to the new cluster
                    if _new_cluster != _old_cluster:
                        _total_transfers += 1
                    _document_cluster[i] = _new_cluster
                    self.cluster_document_count[_new_cluster] += 1
                    self.cluster_word_count[_new_cluster] += len(doc)
                    for word in doc:
                        if word not in self.cluster_word_distribution[_new_cluster]:
                            self.cluster_word_distribution[_new_cluster][word] = 0
                        self.cluster_word_distribution[_new_cluster][word] += 1
                _new_cluster_count = sum([1 for v in self.cluster_document_count if v > 0])
                if _total_transfers == 0 and _new_cluster_count == _n_clusters and _iter > self.n_iterations - 5:
                    break
                _n_clusters = _new_cluster_count
            self.n_clusters = _n_clusters
        return _document_cluster

    def generate_topic_allocation(self, documents: List[List[str]]) -> List[int]:
        """
        Allocate cluster labels (topics) to each document

        :return: List[int]
            Allocated labels
        """
        _topic_allocations = []
        for doc in documents:
            _topic_label = self.predict(document=doc)
            _topic_allocations.append(_topic_label)
        return _topic_allocations

    def get_last_words_each_cluster(self, last_n_words: int = 5) -> Dict[str, Tuple[str, int]]:
        """
        Get last-n-words of each cluster

        :param last_n_words: int
            Number of last words of each cluster

        :return: Dict[str, Tuple[str, int]]
            Last-n-words and frequency of each cluster
        """
        _last_n_words_each_cluster: dict = {}
        for cluster in range(0, self.n_clusters, 1):
            _sorted_words_current_cluster = sorted(self.cluster_word_distribution[cluster].items(),
                                                   key=lambda c: c[1],
                                                   reverse=True
                                                   )[last_n_words:]
            _last_n_words_each_cluster.update({str(cluster): _sorted_words_current_cluster})
        return _last_n_words_each_cluster

    def get_top_words_each_cluster(self, top_n_words: int = 5) -> Dict[str, Tuple[str, int]]:
        """
        Get top-n-words of each cluster

        :param top_n_words: int
            Number of top words of each cluster

        :return: Dict[str, Tuple[str, int]]
            Top-n-words and frequency of each cluster
        """
        _top_n_words_each_cluster: dict = {}
        for cluster in range(0, self.n_clusters, 1):
            _sorted_words_current_cluster = sorted(self.cluster_word_distribution[cluster].items(),
                                                   key=lambda c: c[1],
                                                   reverse=True
                                                   )[:top_n_words]
            _top_n_words_each_cluster.update({str(cluster): _sorted_words_current_cluster})
        return _top_n_words_each_cluster

    def merge_cluster(self, cluster_labels: List[int]):
        """
        Merge clusters

        :param cluster_labels: List[int]
            Cluster labels to merge (first cluster label will be used as base)
        """
        for i, cluster in enumerate(cluster_labels):
            if i > 0:
                self.n_clusters -= 1
                self.cluster_word_count[cluster_labels[0]] += self.cluster_word_count[cluster]
                self.cluster_document_count[cluster_labels[0]] += self.cluster_document_count[cluster]
                for word, freq in self.cluster_word_distribution[cluster].items():
                    if word in self.cluster_word_distribution[cluster_labels[0]].keys():
                        self.cluster_word_distribution[cluster_labels[0]][word] += freq
                    else:
                        self.cluster_word_distribution[cluster_labels[0]].update({word: freq})
                del self.cluster_word_count[cluster]
                del self.cluster_word_distribution[cluster]

    def predict(self, document: List[str]) -> int:
        """
        Predict cluster label

        :param document: List[str]
            Tokenized document

        :return: int
            Cluster label
        """
        return np.array(self._document_scoring(document=document)).argmax()

    def predict_proba(self, document: List[str]) -> List[float]:
        """
        Predict probabilities for each document

        :param document: List[str]
            Tokenized document

        :return: List[np.array]
            Probability vector for each document
        """
        return self._document_scoring(document=document)

    def prepare_visualization(self, documents: List[List[str]]) -> pyLDAvis:
        """
        Prepare documents for visualization from trained model

        :param documents: List[List[str]]
            Tokenized documents

        :return: pyLDAvis
            Prepared word matrix, documents distances, vocabulary and word counts using pyLDAvis library
        """
        _voc: List[str] = []
        for cluster in self.cluster_word_distribution:
            _voc.extend(list(cluster.keys()))
        _vocabulary: List[str] = list(set(_voc))
        _doc_topic_distances: List[List[float]] = [self.predict_proba(doc) for doc in documents]
        for doc in _doc_topic_distances:
            for word in doc:
                assert not isinstance(word, complex)
        _doc_len = [len(doc) for doc in documents]
        _word_counts_map: dict = {}
        for doc in documents:
            for word in doc:
                _word_counts_map[word] = _word_counts_map.get(word, 0) + 1
        _word_counts: list = [_word_counts_map[term] for term in _vocabulary]
        _doc_topic_distances_ext: list = [[v if not math.isnan(v) else 1 / self.n_clusters for v in d] for d in _doc_topic_distances]
        _doc_topic_distances_ext = [d if sum(d) > 0 else [1 / self.n_clusters] * self.n_clusters for d in _doc_topic_distances_ext]
        for doc in _doc_topic_distances_ext:
            for f in doc:
                assert not isinstance(f, complex)
        assert (pd.DataFrame(_doc_topic_distances_ext).sum(axis=1) < 0.999).sum() == 0
        _word_matrix: list = []
        for cluster in self.cluster_word_distribution:
            _total: float = sum([frequency for word, frequency in cluster.items()])
            assert not math.isnan(_total)
            if _total == 0:
                _row: list = [(1 / len(_vocabulary))] * len(_vocabulary)
            else:
                _row: list = [cluster.get(word, 0) / _total for word in _vocabulary]
            for word in _row:
                assert not isinstance(word, complex)
            _word_matrix.append(_row)
        return pyLDAvis.prepare(topic_term_dists=_word_matrix,
                                doc_topic_dists=_doc_topic_distances_ext,
                                doc_lengths=_doc_len,
                                vocab=_vocabulary,
                                term_frequency=_word_counts,
                                R=30,
                                lambda_step=0.01,
                                sort_topics=False
                                )

    @staticmethod
    def save_visualization(file_path: str, vis_data, **kwargs):
        """
        Save topic visualization as interactive html file

        :param file_path: str
            Complete file path

        :param vis_data: pyLDAvis
            Prepared data using pyLDAvis library

        :param kwargs: dict
            Key-word arguments for handling cloud information
        """
        DataExporter(obj=vis_data,
                     file_path=file_path,
                     bucket_name=kwargs.get('bucket'),
                     region=kwargs.get('region'),
                     **dict(topic_clustering=True)
                     ).file()

    def word_importance_each_cluster(self) -> List[dict]:
        """
        Generate word importance (phi) for each cluster

        :return: List[dict]
            Word importance (phi) for each cluster
                -> phi[cluster][word]
        """
        _phi: List[dict] = [{} for _ in range(0, self.n_clusters, 1)]
        for c in range(0, self.n_clusters, 1):
            for w in self.cluster_word_distribution[c]:
                _phi[c][w] = (self.cluster_word_distribution[c][w] + self.beta) / (sum(self.cluster_word_distribution[c].values()) + self.vocab_size * self.beta)
        return _phi


class LatentDirichletAllocation:
    """
    Class for building Latent Dirichlet Allocation model using gensim
    """
    def __init__(self,
                 doc_term_matrix: list,
                 vocab: corpora.Dictionary,
                 n_clusters: int = 200,
                 n_iterations: int = 10,
                 decay: float = 1.0
                 ):
        """
        :param doc_term_matrix: list
            Document-term matrix

        :param vocab: corpora.Dictionary
            Pre-defined (gensim) corpus dictionary

        :param n_clusters: int
            Number of clusters

        :param n_iterations: int
            Number of iterations

        :param decay: float
        """
        self.model = None
        self.decay: float = decay
        self.n_clusters: int = n_clusters
        self.n_iterations: int = n_iterations
        self.vocab: corpora.Dictionary = vocab
        self.document_term_matrix: list = doc_term_matrix
        self.index_to_word = None

    def fit(self, documents: np.ndarray) -> List[int]:
        """
        Fitting Latent Semantic Indexing clustering model

        :param documents: np.ndarray
            Text data

        :return: List[int]
            Cluster labels
        """
        self.index_to_word = corpora.Dictionary(documents=documents)
        self.model = LdaModel(corpus=self.document_term_matrix,
                              num_topics=self.n_clusters,
                              id2word=self.vocab,
                              distributed=True,
                              chunksize=20000,
                              passes=1,
                              update_every=1,
                              alpha='symmetric',
                              eta=None,
                              decay=self.decay,
                              offset=1.0,
                              eval_every=10,
                              iterations=self.n_iterations,
                              gamma_threshold=0.001,
                              minimum_probability=0.01,
                              random_state=1234,
                              ns_conf=None,
                              minimum_phi_value=0.01,
                              per_word_topics=False
                              )
        _cluster_labels: List[int] = []
        _distributions: List[dict] = []
        for doc in documents:
            _top_cluster, _distribution = _rank_clusters(model=self.model, text_data=doc, vocab=self.vocab)
            _cluster_labels.append(_top_cluster)
            _distributions.append(_distribution)
        return _cluster_labels


class LatentSemanticIndexing:
    """
    Class for building Latent Semantic Indexing model using gensim
    """
    def __init__(self,
                 doc_term_matrix: list,
                 vocab: corpora.Dictionary,
                 n_clusters: int = 200,
                 n_iterations: int = 10,
                 decay: float = 1.0,
                 training_algorithm: str = 'multi_pass_stochastic'
                 ):
        """
        :param doc_term_matrix: list
            Document-term matrix

        :param vocab: corpora.Dictionary
            Pre-defined (gensim) corpus dictionary

        :param n_clusters: int
            Number of clusters

        :param n_iterations: int
            Number of iterations

        :param decay: float

        :param training_algorithm: str
            Name of the training algorithm
                -> one_pass
                -> multi_pass_stochastic
        """
        self.model = None
        self.decay: float = decay
        self.n_clusters: int = n_clusters
        self.n_iterations: int = n_iterations
        self.vocab: corpora.Dictionary = vocab
        self.document_term_matrix: list = doc_term_matrix
        self.index_to_word = None
        self.training_algorithm: str = training_algorithm if training_algorithm in ['one_pass', 'multi_pass_stochastic'] else 'multi_pass_stochastic'

    def fit(self, documents: np.ndarray) -> List[int]:
        """
        Fitting Latent Semantic Indexing clustering model

        :param documents: np.ndarray
            Text data

        :return: List[int]
            Cluster labels
        """
        self.index_to_word = corpora.Dictionary(documents=documents)
        self.model = LsiModel(corpus=self.document_term_matrix,
                              num_topics=self.n_clusters,
                              id2word=self.vocab,
                              chunksize=20000,
                              decay=self.decay,
                              distributed=True,
                              onepass=True if self.training_algorithm == 'one_pass' else False,
                              power_iters=self.n_iterations
                              )
        _cluster_labels: List[int] = []
        _distributions: List[dict] = []
        for doc in documents:
            _top_cluster, _distribution = _rank_clusters(model=self.model, text_data=doc, vocab=self.vocab)
            _cluster_labels.append(_top_cluster)
            _distributions.append(_distribution)
        return _cluster_labels


class NonNegativeMatrixFactorization:
    """
    Class for building Non-Negative Matrix Factorization model using sklearn
    """
    def __init__(self,
                 lang: str,
                 n_clusters: int = 200,
                 n_iterations: int = 10
                 ):
        """
        :param lang: str
            Full language name

        :param n_clusters: int
            Number of clusters

        :param n_iterations: int
            Number of iterations
        """
        self.model = None
        self.tfidf = None
        self.lang: str = lang
        self.n_clusters: int = n_clusters
        self.n_iterations: int = n_iterations

    def _term_frequency_inverse_document_frequency(self, text_content: np.ndarray, lang: str) -> np.ndarray:
        """
        Apply term-frequency inverse document frequency

        :param text_content: np.ndarray
            Text data

        :param lang: str
            Abbreviated name for the stopwords language

        :return: np.ndarray
            Transformed text content using term-frequency inverse document frequency vectorizer
        """
        self.tfidf = TfidfVectorizer(input='content',
                                     encoding='uft-8',
                                     decode_error='strict',
                                     strip_accents=None,
                                     lowercase=True,
                                     preprocessor=None,
                                     tokenizer=None,
                                     analyzer='word',
                                     stop_words=lang,
                                     ngram_range=(1, 1),
                                     vocabulary=None,
                                     binary=False,
                                     norm='l2',
                                     use_idf=True,
                                     smooth_idf=True,
                                     sublinear_tf=False
                                     )
        return self.tfidf.fit_transform(raw_documents=text_content)

    def fit(self, documents: np.ndarray) -> List[int]:
        """
        Fitting Non-Negative Matrix Factorization

        :param documents: np.ndarray
            Text data

        :return: List[int]
            Cluster labels
        """
        _tfidf = self._term_frequency_inverse_document_frequency(text_content=documents, lang=self.lang)
        self.model: NMF = NMF(n_components=self.n_clusters,
                              init=None,
                              solver='cd',
                              beta_loss='frobenius',
                              tol=0.0004,
                              max_iter=self.n_iterations,
                              random_state=1234,
                              alpha=0,
                              l1_ratio=0,
                              verbose=0,
                              shuffle=False
                              )
        self.model.fit(X=_tfidf, y=None)
        _cluster_labels: List[int] = self.model.transform(self._term_frequency_inverse_document_frequency(text_content=documents,
                                                                                                          lang=self.lang
                                                                                                          )
                                                          ).argmax(axis=1).tolist()
        return _cluster_labels

    def get_top_words_each_cluster(self, top_n_words: int = 5) -> dict:
        """
        Get top n words of each cluster

        :param top_n_words: int
            Number of top n words of each cluster

        :return: dict
            Top n words of each cluster
        """
        _top_n_words_each_cluster: dict = {}
        for cluster, topic in enumerate(self.model.components_):
            _top_n_words_each_cluster.update({str(cluster): [self.tfidf.get_feature_names()[i] for i in topic.argsort()[-top_n_words:]]})
        return _top_n_words_each_cluster
