import copy
import numpy as np
import pandas as pd

from easyexplore.data_visualizer import DataVisualizer
from sklearn.metrics import accuracy_score, auc, classification_report, cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import mean_absolute_error, mean_gamma_deviance, mean_poisson_deviance, mean_squared_error, mean_squared_log_error, mean_tweedie_deviance
from typing import Dict, List


# TODO:
#  Handle Multi-Class Problems using AUC: OVO / OVR

ML_METRIC: Dict[str, List[str]] = dict(reg=['mae', 'mgd', 'mpd', 'mse', 'msle', 'mtd', 'rmse', 'rmse_norm'],
                                       clf_binary=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                                       clf_multi=['accuracy', 'cohen_kappa', 'f1', 'precision', 'recall']
                                       )


SML_SCORE: dict = dict(ml_metric=dict(clf_binary='roc_auc', clf_multi='cohen_kappa', reg='rmse_norm'),
                       ml_metric_best=dict(clf_binary=1, clf_multi=1, reg=0),
                       ml_metric_weights=0.25,
                       train_test_weights=0.35,
                       train_time_in_seconds_weights=0.0000005,
                       start_value=100,
                       normalized=False,
                       capping_to_zero=True
                       )


def sml_score(ml_metric: tuple,
              train_test_metric: tuple,
              train_time_in_seconds: float,
              ml_metric_weights: float = SML_SCORE.get('ml_metric_weights'),
              train_test_weights: float = SML_SCORE.get('train_test_weights'),
              train_time_in_seconds_weights: float = SML_SCORE.get('train_time_in_seconds_weights'),
              start_value: int = SML_SCORE.get('start_value'),
              normalized: bool = SML_SCORE.get('normalized'),
              capping_to_zero: bool = SML_SCORE.get('capping_to_zero')
              ) -> dict:
    """
    Productivity score for evaluating machine learning models multi-dimensional
        -> Dimensions:  1) Difference between normalized classification or regression metric of test data and it's optimal score
                        2) Difference between train and test metric
                        3) Training time in seconds

    :param ml_metric: tuple
        Any normalized machine learning (test) metric and it's optimal value score
        -> [0]: optimal value score of normalized metric
        -> [1]: actual test value score of the model

    :param train_test_metric: tuple
        Any normalized machine learning train and test metric
        -> [0]: train value score of normalized metric
        -> [1]: test value score of the model

    :param train_time_in_seconds: float
        Training time in seconds

    :param ml_metric_weights: float
        Weights for handling importance of machine learning metric (test) for scoring

    :param train_test_weights: float
        Weights for handling importance of train and test difference for scoring

    :param train_time_in_seconds_weights: float
        Weights for handling importance of training time for scoring

    :param start_value: int
        Starting fitness value of machine learning model

    :param normalized: bool
        Normalize final score or not

    :param capping_to_zero: bool
        Cap scores smaller then zero to zero

    :return dict
        Productivity scores for each dimension to evaluate general purpose machine learning model
    """
    _out_of_range: bool = False
    try:
        if ml_metric[0] == 0:
            if train_test_metric[0] >= 1 or train_test_metric[1] >= 1:
                _out_of_range = True
            _ml_metric_error: float = (start_value / abs(1 - ml_metric[1])) - start_value
        else:
            _ml_metric_error: float = (start_value / ml_metric[1]) - start_value
        if _out_of_range:
            _ml_metric: float = start_value * 10
        else:
            _ml_metric: float = abs(_ml_metric_error * ml_metric_weights)
    except ZeroDivisionError:
        _ml_metric: float = start_value * 10
    try:
        _train_test_error: float = (start_value / (1 - abs(train_test_metric[0] - train_test_metric[1]))) - start_value
        _train_test_diff_abs: float = abs(_train_test_error * train_test_weights)
    except ZeroDivisionError:
        _train_test_diff_abs: float = 0.0 if train_test_metric[0] == 0 else (train_test_metric[0] - train_test_metric[1])
    _train_time_in_seconds: float = train_time_in_seconds * (train_time_in_seconds_weights * start_value)
    _final_score: float = start_value - _ml_metric - _train_test_diff_abs - _train_time_in_seconds
    _scores: dict = dict(ml_metric=_ml_metric,
                         train_test_diff=_train_test_diff_abs,
                         train_time_in_seconds=_train_time_in_seconds,
                         fitness_score=_final_score
                         )
    for dimension, score in _scores.items():
        _score: float = score / start_value if normalized else score
        if capping_to_zero:
            if _score < 0:
                _scores.update({dimension: 0.00001})
            else:
                _scores.update({dimension: _score})
        else:
            _scores.update({dimension: _score})
    _scores.update({'original_ml_train_metric': round(train_test_metric[0], ndigits=4),
                    'original_ml_test_metric': round(train_test_metric[1], ndigits=4)
                    })
    return _scores


def sml_fitness_score(ml_metric: tuple,
                      train_test_metric: tuple,
                      train_time_in_seconds: float,
                      ml_metric_weights: float = SML_SCORE.get('ml_metric_weights'),
                      train_test_weights: float = SML_SCORE.get('train_test_weights'),
                      train_time_in_seconds_weights: float = SML_SCORE.get('train_time_in_seconds_weights'),
                      start_value: int = SML_SCORE.get('start_value'),
                      normalized: bool = SML_SCORE.get('normalized'),
                      capping_to_zero: bool = SML_SCORE.get('capping_to_zero')
                      ) -> float:
    """
    Productivity score for evaluating machine learning models multi-dimensional
        -> Wrapper function of 'productivity' method to extract and return only final fitness score

    :param ml_metric: tuple
        Any normalized machine learning (test) metric and it's optimal value score
        -> [0]: optimal value score of normalized metric
        -> [1]: actual test value score of the model

    :param train_test_metric: tuple
        Any normalized machine learning train and test metric
        -> [0]: train value score of normalized metric
        -> [1]: test value score of the model

    :param train_time_in_seconds: float
        Training time in seconds

    :param ml_metric_weights: float
        Weights for handling importance of machine learning metric (test) for scoring

    :param train_test_weights: float
        Weights for handling importance of train and test difference for scoring

    :param train_time_in_seconds_weights: float
        Weights for handling importance of training time for scoring

    :param start_value: int
        Starting fitness value of machine learning model

    :param normalized: bool
        Normalize final score or not

    :param capping_to_zero: bool
        Cap scores that are smaller then zero to (almost) zero

    :return float
        Productivity score aggregated by dimension scores to evaluate general purpose machine learning model
    """
    return sml_score(ml_metric=ml_metric,
                     train_test_metric=train_test_metric,
                     train_time_in_seconds=train_time_in_seconds,
                     ml_metric_weights=ml_metric_weights,
                     train_test_weights=train_test_weights,
                     train_time_in_seconds_weights=train_time_in_seconds_weights,
                     start_value=start_value,
                     normalized=normalized,
                     capping_to_zero=capping_to_zero
                     ).get('fitness_score')


def sml_score_test(optimal_value: int, experiments: int = 10000):
    """
    Run general supervised machine learning scoring for testing purpose

    :param optimal_value: int
        Optimal value according to the tested ml metric
            -> 1: AUC, Cohen's Cappa
            -> 0: Normalized RMSE by STD

    :param experiments: int
        Number of experiments to run
    """
    _sml_score: List[float] = []
    _train_error: List[float] = []
    _test_error: List[float] = []
    _training_time_in_seconds: List[float] = []
    _experiments: int = experiments if experiments > 1 else 10000
    for i in range(0, _experiments, 1):
        if optimal_value == 0:
            _train_error.append(np.random.uniform(low=0, high=1.5))
            _test_error.append(np.random.uniform(low=0, high=1.5))
        elif optimal_value == 1:
            _train_error.append(np.random.uniform(low=0, high=1))
            _test_error.append(np.random.uniform(low=0, high=1))
        _training_time_in_seconds.append(np.random.uniform(low=1, high=500000))
        if optimal_value == 0 or optimal_value == 1:
            _sml_score.append(sml_fitness_score(ml_metric=(optimal_value, _test_error[-1]),
                                                train_test_metric=(_train_error[-1], _test_error[-1]),
                                                train_time_in_seconds=_training_time_in_seconds[-1]
                                                )
                              )
        else:
            raise EvalMLException('Optimal value should be normalized and either 0 or 1 not {}'.format(optimal_value))
    _sml_scoring: dict = dict(train_error=_train_error,
                              test_error=_test_error,
                              training_time_in_seconds=_training_time_in_seconds,
                              sml_score=_sml_score
                              )
    DataVisualizer(df=pd.DataFrame(data=_sml_scoring),
                   title='SML Scoring (Experiments = {})'.format(_experiments),
                   features=['training_time_in_seconds', 'train_error', 'test_error', 'sml_score'],
                   color_feature='sml_score',
                   plot_type='parcoords',
                   render=True
                   ).run()


def profit_score(prob: List[float], quota: List[float]) -> float:
    """
    Score to optimize possible profit by given quota and distribution or probability value

    :param prob: List[float]
        Probability values to score

    :param quota: List[float]
        Quota for each profit value

    :return float
        Profit Score
    """
    if len(prob) != len(quota):
        raise EvalMLException('Unequal length of parameters (prob={} | quota={}). Profit score cannot be calculated'.format(len(prob), len(quota)))
    _profit_score: List[float] = []
    for p, q in zip(prob, quota):
        _profit_score.append(p * q)
    return sum(_profit_score)


class EvalMLException(Exception):
    """
    Class for handling exceptions for class EvalMLException
    """
    pass


class EvalClf:
    """
    Class for evaluating supervised machine learning models for classification problems
    """
    def __init__(self,
                 obs: np.array,
                 pred: np.array,
                 average: str = 'macro',
                 probability: bool = False,
                 extract_prob: int = 0
                 ):
        """
        :param obs: np.array
            Observation

        :param pred: np.array
            Prediction

        :param probability: bool
            Prediction is probability value or not

        :param average: str
            Name of the average method to use

        :param extract_prob: int
            Number of class to use probability to classify category
        """
        self.obs: np.array = copy.deepcopy(obs)
        _extract_prob: int = extract_prob if extract_prob >= 0 else 0
        if probability:
            self.pred: np.array = np.array([np.argmax(prob) for prob in copy.deepcopy(pred)])
        else:
            self.pred: np.array = copy.deepcopy(pred)
        self.average: str = average
        if self.average not in [None, 'micro', 'macro', 'weighted', 'samples']:
            self.average = 'macro'

    def accuracy(self) -> float:
        """
        Generate accuracy score
        """
        return accuracy_score(y_true=self.obs, y_pred=self.pred, normalize=True, sample_weight=None)

    def classification_report(self) -> dict:
        """
        Generate classification report containing several metric values

        :return pd.DataFrame:
            Classification report
        """
        return classification_report(y_true=self.obs,
                                     y_pred=self.pred,
                                     labels=None,
                                     sample_weight=None,
                                     digits=2,
                                     output_dict=True,
                                     zero_division='warn'
                                     )

    def cohen_kappa(self) -> float:
        """
        Cohen Kappa score classification metric for multi-class problems

        :return: float
            Cohen's Cappa Score
        """
        return cohen_kappa_score(y1=self.obs, y2=self.pred, labels=None, weights=None, sample_weight=None)

    def confusion(self, normalize: str = None) -> pd.DataFrame:
        """
        Confusion matrix for classification problems

        :param normalize: str
            Normalizing method:
                -> true: Confusion matrix normalized by observations
                -> pred: Confusion matrix normalized by predictions
                -> all: Confusion matrix normalized by both observations and predictions
                -> None: No normalization

        :return: pd.DataFrame
            Confusion Matrix
        """
        return confusion_matrix(y_true=self.obs, y_pred=self.pred, labels=None, sample_weight=None, normalize=normalize)

    def f1(self) -> float:
        """
        F1 metric of confusion matrix for classification problems

        :return: float
            F1-Score
        """
        return f1_score(y_true=self.obs,
                        y_pred=self.pred,
                        labels=None,
                        pos_label=1,
                        average=self.average,
                        sample_weight=None,
                        zero_division='warn'
                        )

    def precision(self) -> float:
        """
        Precision metric of confusion matrix for classification problems

        :return: float
            Precision Score
        """
        return precision_score(y_true=self.obs,
                               y_pred=self.pred,
                               labels=None,
                               pos_label=1,
                               average=self.average,
                               sample_weight=None,
                               zero_division='warn'
                               )

    def recall(self) -> float:
        """
        Recall metric of confusion matrix for classification problems

        :return: float
            Recall score
        """
        return recall_score(y_true=self.obs,
                            y_pred=self.pred,
                            labels=None,
                            pos_label=1,
                            average=self.average,
                            sample_weight=None,
                            zero_division='warn'
                            )

    def roc_auc(self) -> float:
        """
        Area Under Receiver Operating Characteristic Curve classification metric for binary problems

        :return: float
            Area-Under-Curve Score (AUC)
        """
        if len(list(pd.unique(self.obs))) == 1 or len(list(pd.unique(self.pred))) == 1:
            return 0.0
        else:
            return roc_auc_score(y_true=self.obs,
                                 y_score=self.pred,
                                 average=self.average,
                                 sample_weight=None,
                                 max_fpr=None,
                                 multi_class='raise',
                                 labels=None
                                 )

    def roc_auc_multi(self, meth: str = 'ovr') -> float:
        """
        Area Under Receiver Operating Characteristic Curve classification metric for binary problems

        :param: meth: Method of multi-class roc-auc score
                        -> ovr: Computes score for each class against the rest
                        -> ovo: Computes score for each class pairwise

        :return: float
            Area-Under_Curve Score for multi-class problems
        """
        _meth: str = meth if meth in ['ovr', 'ovo'] else 'ovr'
        return roc_auc_score(y_true=self.obs,
                             y_score=self.pred,
                             average=self.average,
                             sample_weight=None,
                             max_fpr=None,
                             multi_class=_meth,
                             labels=None
                             )

    def roc_curve(self) -> dict:
        """
        Calculate true positive rates & false positive rates for generating roc curve

        :return: dict
            Calculated true positive, false positive rates and roc-auc score
        """
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(0, len(pd.unique(self.obs).tolist()), 1):
            fpr[i], tpr[i], _ = roc_curve(self.obs, self.pred)
            roc_auc[i] = auc(fpr[i], tpr[i])
        return dict(true_positive_rate=tpr, false_positive_rate=fpr, roc_auc=roc_auc)


class EvalReg:
    """
    Class for evaluating supervised machine learning models for regression problems
    """
    def __init__(self, obs: np.array, pred: np.array, multi_output: str = 'uniform_average'):
        """
        :param obs: np.array
            Observation

        :param pred: np.array
            Prediction

        :param multi_output: str
            Method to handle multi output
                -> uniform_average: Errors of all outputs are averaged with uniform weight
                -> raw_values: Returns a full set of errors in case of multi output input
        """
        self.obs: np.array = obs
        self.std_obs: float = self.obs.std()
        self.pred: np.array = pred
        self.multi_output: str = multi_output if multi_output in ['raw_values', 'uniform_average'] else 'uniform_average'

    def mae(self) -> float:
        """
        Mean absolute error metric for regression problems

        :return: float
            Mean-Absolute-Error Score
        """
        return mean_absolute_error(y_true=self.obs, y_pred=self.pred, sample_weight=None, multioutput=self.multi_output)

    def mgd(self) -> float:
        """
        Mean gamma deviance error metric for regression problems

        :return: float
            Mean-Gama-Deviance-Error Score
        """
        return mean_gamma_deviance(y_true=self.obs, y_pred=self.pred, sample_weight=None)

    def mpd(self) -> float:
        """
        Mean poisson deviance error metric for regression problems

        :return: float
            Mean-Poisson-Deviance-Error Score
        """
        return mean_poisson_deviance(y_true=self.obs, y_pred=self.pred, sample_weight=None)

    def mse(self) -> float:
        """
        Mean squared error metric for regression problems

        :return: float
            Mean-Squared-Error Score
        """
        return mean_squared_error(y_true=self.obs,
                                  y_pred=self.pred,
                                  sample_weight=None,
                                  multioutput=self.multi_output,
                                  squared=True
                                  )

    def msle(self) -> float:
        """
        Mean squared log error metric for regression problems

        :return: float
            Mean-Squared-Log-Error Score
        """
        return mean_squared_log_error(y_true=self.obs,
                                      y_pred=self.pred,
                                      sample_weight=None,
                                      multioutput=self.multi_output
                                      )

    def mtd(self) -> float:
        """
        Mean tweedie deviance error metric for regression problems

        :return: float
            Mean-Tweedie-Deviance-Error Score
        """
        return mean_tweedie_deviance(y_true=self.obs, y_pred=self.pred, sample_weight=None)

    def rmse(self) -> float:
        """
        Root mean squared error metric for regression problems

        :return: float
            Root-Mean-Squared-Error Score
        """
        return mean_squared_error(y_true=self.obs,
                                  y_pred=self.pred,
                                  sample_weight=None,
                                  multioutput=self.multi_output,
                                  squared=False
                                  )

    def rmse_norm(self) -> float:
        """
        Normalized root mean squared error metric by standard deviation for regression problems

        :return: float
            Normalized Root-Mean-Squared-Error Score (by Standard Deviation)
        """
        return self.rmse() / self.std_obs
