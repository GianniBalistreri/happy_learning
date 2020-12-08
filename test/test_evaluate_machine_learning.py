import numpy as np
import pandas as pd
import unittest

from happy_learning.evaluate_machine_learning import EvalClf, EvalReg, sml_fitness_score, sml_score, sml_score_test

OBSERVATION_CLF: np.array = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0, 1])
PREDICTION_CLF: np.array = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 0])
OBSERVATION_REG: np.array = np.array([0.45, 0.55, 0.23, 0.45, 0.78, 0.99, 0.44, 0.11, 0.33, 0.66])
PREDICTION_REG: np.array = np.array([0.55, 0.45, 0.45, 0.74, 0.88, 0.89, 0.22, 0.15, 0.28, 0.67])


class EvalClfTest(unittest.TestCase):
    """
    Class for testing class EvalClf
    """
    def test_classification_report(self):
        _clf_report: dict = EvalClf(obs=OBSERVATION_CLF, pred=PREDICTION_CLF, average='macro', probability=False, extract_prob=0).classification_report()
        self.assertTrue(expr=_clf_report['0']['precision'] == _clf_report['0']['recall'] == _clf_report['0']['f1-score'])

    def test_cohen_kappa(self):
        pass

    def test_confusion_matrix(self):
        _confusion_matrix: pd.DataFrame = EvalClf(obs=OBSERVATION_CLF, pred=PREDICTION_CLF, average='macro', probability=False, extract_prob=0).confusion()
        print(_confusion_matrix)

    def test_f1(self):
        _f1_score: float = EvalClf(obs=OBSERVATION_CLF, pred=PREDICTION_CLF, average='macro', probability=False, extract_prob=0).f1()
        self.assertEqual(first=0.58, second=round(_f1_score, ndigits=2))

    def test_precision(self):
        _precision_score: float = EvalClf(obs=OBSERVATION_CLF, pred=PREDICTION_CLF, average='macro', probability=False, extract_prob=0).precision()
        self.assertEqual(first=0.58, second=round(_precision_score, ndigits=2))

    def test_recall(self):
        _recall_score: float = EvalClf(obs=OBSERVATION_CLF, pred=PREDICTION_CLF, average='macro', probability=False, extract_prob=0).recall()
        self.assertEqual(first=0.58, second=round(_recall_score, ndigits=2))

    def test_roc_auc(self):
        _roc_auc: float = EvalClf(obs=OBSERVATION_CLF, pred=PREDICTION_CLF, average='macro', probability=False, extract_prob=0).roc_auc()
        self.assertEqual(first=0.58, second=round(_roc_auc, ndigits=2))

    def test_roc_auc_multi(self):
        pass


class EvalRegTest(unittest.TestCase):
    """
    Class for testing class EvalReg
    """
    def test_(self):
        pass


class SMLScoreTest(unittest.TestCase):
    """
    Class for testing sml score functions
    """
    def test_sml_score(self):
        pass

    def test_sml_fitness_score(self):
        pass

    def test_sml_score_test(self):
        pass


if __name__ == '__main__':
    unittest.main()
