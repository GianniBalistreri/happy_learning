import numpy as np
import pandas as pd
import unittest

from happy_learning.evaluate_machine_learning import EvalClf, EvalReg, sml_fitness_score, sml_score, sml_score_test

OBSERVATION_CLF: np.array = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0, 1])
PREDICTION_CLF: np.array = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 0])
OBSERVATION_CLF_MULTI: np.array = np.array([0, 1, 1, 2, 0, 1, 1, 0, 2, 1])
PREDICTION_CLF_MULTI: np.array = np.array([1, 1, 2, 1, 0, 1, 1, 1, 0, 2])
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
        _cohen_kappa: float = EvalClf(obs=OBSERVATION_CLF, pred=PREDICTION_CLF, average='macro', probability=False, extract_prob=0).cohen_kappa()
        self.assertEqual(first=0.16666666666666663, second=EvalClf(obs=OBSERVATION_CLF, pred=PREDICTION_CLF, average='macro', probability=False, extract_prob=0).cohen_kappa())

    def test_confusion_matrix(self):
        _confusion_matrix: pd.DataFrame = pd.DataFrame(data=EvalClf(obs=OBSERVATION_CLF, pred=PREDICTION_CLF, average='macro', probability=False, extract_prob=0).confusion())
        self.assertTrue(expr=_confusion_matrix.iloc[0, 0] == 2 and _confusion_matrix.iloc[0, 1] == 2 and _confusion_matrix.iloc[1, 0] == 2 and _confusion_matrix.iloc[1, 1] == 4)

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
    def test_mae(self):
        self.assertEqual(first=0.123, second=EvalReg(obs=OBSERVATION_REG, pred=PREDICTION_REG, multi_output='uniform_average').mae())

    def test_mgd(self):
        self.assertEqual(first=0.14115675875949402, second=EvalReg(obs=OBSERVATION_REG, pred=PREDICTION_REG, multi_output='uniform_average').mgd())

    def test_mpd(self):
        self.assertEqual(first=0.05167174706785952, second=EvalReg(obs=OBSERVATION_REG, pred=PREDICTION_REG, multi_output='uniform_average').mpd())

    def test_mse(self):
        self.assertEqual(first=0.02251, second=EvalReg(obs=OBSERVATION_REG, pred=PREDICTION_REG, multi_output='uniform_average').mse())

    def test_msle(self):
        self.assertEqual(first=0.010510365540451308, second=EvalReg(obs=OBSERVATION_REG, pred=PREDICTION_REG, multi_output='uniform_average').msle())

    def test_mtd(self):
        self.assertEqual(first=0.02251, second=EvalReg(obs=OBSERVATION_REG, pred=PREDICTION_REG, multi_output='uniform_average').mtd())

    def test_rmse(self):
        self.assertEqual(first=0.15003332963045243, second=EvalReg(obs=OBSERVATION_REG, pred=PREDICTION_REG, multi_output='uniform_average').rmse())

    def test_rmse_norm(self):
        self.assertEqual(first=0.6079208137353993, second=EvalReg(obs=OBSERVATION_REG, pred=PREDICTION_REG, multi_output='uniform_average').rmse_norm())


class SMLScoreTest(unittest.TestCase):
    """
    Class for testing sml score functions
    """
    def test_sml_score(self):
        _sml_score: dict = sml_score(ml_metric=(0.56, 1),
                                     train_test_metric=(0.56, 0.66),
                                     train_time_in_seconds=400
                                     )
        self.assertTrue(expr='fitness_score' in list(_sml_score.keys()))

    def test_sml_fitness_score(self):
        _sml_fitness_score: float = sml_fitness_score(ml_metric=(0.56, 1),
                                                      train_test_metric=(0.56, 0.66),
                                                      train_time_in_seconds=400
                                                      )
        self.assertEqual(first=94.42444444444445, second=_sml_fitness_score)

    def test_sml_score_test(self):
        pass


if __name__ == '__main__':
    unittest.main()
