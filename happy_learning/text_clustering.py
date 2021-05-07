import numpy as np

from gensim import corpora
from gensim.models import LdaModel, LsiModel
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List


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
                 vocab_size: int,
                 n_clusters: int = 8,
                 n_iterations: int = 30,
                 alpha: float = 0.1,
                 beta: float = 0.1
                 ):
        """
        :param vocab_size: int
            Size of the vocabulary

        :param n_clusters: int
            Number of pre-defined clusters

        :param n_iterations: int
            Number of iterations

        :param alpha: float
            Probability of joining an empty cluster

        :param beta: float
            Affinity of a cluster element with same characteristics as other elements
        """
        self.alpha: float = alpha
        self.beta: float = beta
        self.n_clusters: int = n_clusters
        self.n_iterations: int = n_iterations if n_iterations > 5 else 30
        self.n_documents: int = 0
        self.vocab_size: int = vocab_size
        self.probability_vector: List[float] = []
        self.cluster_word_count: List[int] = [0 for _ in range(0, self.n_clusters, 1)]
        self.cluster_document_count: List[int] = [0 for _ in range(0, self.n_clusters, 1)]
        self.cluster_word_distribution: List[dict] = [{} for _ in range(0, self.n_clusters, 1)]

    def _document_scoring(self, documents: List[str]) -> List[float]:
        """
        Score a document

        :param documents: List[str]:
            The document token stream

        :return: list[float]:
            Probability vector where each component represents the probability of the document appearing in a particular cluster
        """
        _p = [0 for _ in range(0, self.n_clusters, 1)]
        #  We break the formula into the following pieces
        #  p = n1 * n2 / (d1 * d2) = np.exp(ln1 - ld1 + ln2 - ld2)
        #  lN1 = np.log(m_z[z] + alpha)
        #  lN2 = np.log(D - 1 + K*alpha)
        #  lN2 = np.log(product(n_z_w[w] + beta)) = sum(np.log(n_z_w[w] + beta))
        #  lD2 = np.log(product(n_z[d] + V*beta + i -1)) = sum(np.log(n_z[d] + V*beta + i -1))
        _ld1 = np.log(self.n_documents - 1 + self.n_clusters * self.alpha)
        _document_size = len(documents)
        for label in range(0, self.n_clusters, 1):
            _ln1 = np.log(self.cluster_word_count[label] + self.alpha)
            _ln2 = 0
            _ld2 = 0
            for word in documents:
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

    def fit(self, documents: List[List[str]]) -> List[int]:
        """
        Fitting GSDMM clustering algorithm

        :param documents: List[List[str]]
            The document token stream

        :return: List[int]
            Cluster labels
        """
        self.n_documents = len(documents)
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
                self.probability_vector = self._document_scoring(documents=documents[i])
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
            _topic_label = self.predict(documents=doc)
            _topic_allocations.append(_topic_label)
        return _topic_allocations

    def get_top_words_each_cluster(self, top_n_words: int = 5) -> dict:
        """
        Get top n words of each cluster

        :param top_n_words: int
            Number of top n words of each cluster

        :return: dict
            Top n words of each cluster
        """
        _top_n_words_each_cluster: dict = {}
        for cluster in range(0, self.n_clusters, 1):
            _sorted_words_current_cluster = sorted(self.cluster_word_distribution[cluster].items(),
                                                   key=lambda c: c[1],
                                                   reverse=True
                                                   )[:top_n_words]
            _top_n_words_each_cluster.update({str(cluster): _sorted_words_current_cluster})
        return _top_n_words_each_cluster

    def predict(self, documents: List[str]) -> int:
        """
        Predict cluster label

        :param documents: List[str]
            The document token stream

        :return: int
            Cluster label
        """
        return np.array(self._document_scoring(documents=documents)).argmax()

    def predict_proba(self, documents: List[str]) -> List[float]:
        """
        Predict probabilities for each document

        :param documents: List[str]
            The document token stream

        :return: List[np.array]
            Probability vector for each document
        """
        return self._document_scoring(documents=documents)

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
