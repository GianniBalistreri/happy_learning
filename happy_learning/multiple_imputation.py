import dask.dataframe as dd
import pandas as pd

from .missing_data_analysis import MissingDataAnalysis
from .supervised_machine_learning import CLF_ALGORITHMS, Classification, REG_ALGORITHMS, Regression
from easyexplore.utils import EasyExploreUtils, Log, StatsUtils
from sklearn.preprocessing import LabelEncoder
from typing import List, Union

# TODO:
#  1) Parallelize
#  2) Multi model support


class MultipleImputationException(Exception):
    """
    Class for handling exceptions for class MultipleImputation
    """
    pass


class MultipleImputation:
    """
    Class for replace missing data by multiple imputation technique
    """
    def __init__(self,
                 df: Union[dd.DataFrame, pd.DataFrame],
                 n_chains: int = 3,
                 n_iter: int = 15,
                 n_burn_in_iter: int = 3,
                 ml_meth: dict = None,
                 predictors: dict = None,
                 imp_sequence: List[str] = None,
                 cor_threshold_for_predictors: float = None,
                 pool_eval_meth: str = 'std',
                 impute_hard_missing: bool = False,
                 soft_missing_values: list = None
                 ):
        """
        :param df: Pandas or dask DataFrame
            Data set

        :param n_chains: int
            Number of markov chains

        :param n_iter: int
            Number of iterations

        :param n_burn_in_iter: int
            Number of burn-in iterations (warm start)

        :param ml_meth: str
            Name of the supervised machine learning algorithm

        :param predictors: dict
            Pre-defined predictors for each feature imputation

        :param imp_sequence:

        :param cor_threshold_for_predictors: float
        :param pool_eval_meth:
        :param impute_hard_missing:
        :param soft_missing_values:
        """
        if isinstance(df, pd.DataFrame):
            self.df: dd.DataFrame = dd.from_pandas(data=df, npartitions=4)
        elif isinstance(df, dd.DataFrame):
            self.df: dd.DataFrame = df
        self.feature_types: dict = EasyExploreUtils().get_feature_types(df=self.df,
                                                                        features=list(self.df.columns),
                                                                        dtypes=self.df.dtypes.tolist()
                                                                        )
        self.n_chains: int = 3 if n_chains <= 1 else n_chains
        self.chains: dict = {m: {} for m in range(0, self.n_chains, 1)}
        self.n_burn_in_iter: int = 3 if n_burn_in_iter <= 0 else n_burn_in_iter
        self.n_iter: int = (15 if n_iter <= 1 else n_iter) + self.n_burn_in_iter
        self.data_types: List[str] = ['cat', 'cont', 'date']
        _encoder = LabelEncoder()
        for ft in self.df.columns:
            if str(self.df[ft].dtype).find('object') >= 0:
                self.df[ft] = self.df[ft].fillna('NaN')
                #self.df.loc[self.df[ft].isnull().compute(), ft] = 'NaN'
                self.df[ft] = dd.from_array(x=_encoder.fit_transform(y=self.df[ft].values))
        self.ml_meth: dict = ml_meth
        if self.ml_meth is not None:
            for meth in self.ml_meth:
                if meth.find('cat') >= 0:
                    pass
        else:
            self.ml_meth = dict(cat='xgb', cont='xgb', date='xgb')
        self.predictors: dict = predictors
        self.impute_hard_missing: bool = impute_hard_missing
        self.mis_freq: dict = MissingDataAnalysis(df=self.df, other_mis=soft_missing_values).freq_nan_by_features()
        self.nan_idx: dict = MissingDataAnalysis(df=self.df, other_mis=soft_missing_values).get_nan_idx_by_features()
        self.imp_sequence: List[str] = [] if imp_sequence is None else imp_sequence
        if len(self.imp_sequence) == 0:
            # self.imp_sequence = [mis_freq[0] for mis_freq in sorted(self.mis_freq.items(), key=lambda x: x[1], reverse=False)]
            for mis_freq in sorted(self.mis_freq.items(), key=lambda x: x[1], reverse=False):
                if mis_freq[1] > 0:
                    self.imp_sequence.append(mis_freq[0])
        if self.predictors is None:
            self.predictors = {}
            if cor_threshold_for_predictors is None:
                for ft in self.mis_freq.keys():
                    self.predictors.update({ft: list(set(list(self.df.columns)).difference([ft]))})
            else:
                if (cor_threshold_for_predictors > 0.0) and (cor_threshold_for_predictors < 1.0):
                    _cor: pd.DataFrame = StatsUtils(data=self.df, features=list(self.df.columns)).correlation()
                    for ft in self.df.columns:
                        self.predictors.update({ft: _cor.loc[_cor[ft] >= cor_threshold_for_predictors, ft].index.values.tolist()})
                        if len(self.predictors[ft]) == 0:
                            raise MultipleImputationException('No predictors found to impute feature "{}" based on given correlation threshold (>={})'.format(ft, cor_threshold_for_predictors))
                else:
                    for ft in self.df.columns:
                        self.predictors.update({ft: list(set(list(self.df.columns)).difference([ft]))})
        if pool_eval_meth not in ['std', 'var', 'aic', 'bic']:
            raise MultipleImputationException('Method for pooling chain evaluation ({}) not supported'.format(pool_eval_meth))
        self.pool_eval_meth: str = pool_eval_meth

    def _little_mcar_test(self) -> dict:
        """
        Run Little's test for evaluating missing completely at random (MCAR) structure

        :return dict:
            Test statistic as well as whether test succeeded of not
        """
        raise NotImplementedError("Little's MCAR-Test not implemented")

    def _rubin_gelman_convergence(self) -> dict:
        """
        Run rubin-gelman-convergence test for evaluating markov chains

        :return dict:
            Test statistic as well as convergence succeeded of not
        """
        raise NotImplementedError("Rubin-Gelman's Convergence-Test not implemented")

    def emb(self):
        """
        Run expectation maximation bootstrap

        :return pd.DataFrame:
            Fully imputed data set
        """
        raise NotImplementedError('Expectation Maximation Bootstrap not implemented')

    def mice(self, rubin_gelman_convergence: bool = False) -> dd.DataFrame:
        """
        Run multiple imputation by chained equation (mice)

        :param rubin_gelman_convergence: bool
            Run process until rubin-gelman convergence test passes

        :return dask DataFrame:
            Fully imputed data set
        """
        # Step 1: Initial imputation
        self.df = self.df.fillna(0)
        _std: dict = {ft: self.df[ft].std() for ft in self.imp_sequence}
        _pool_std: dict = {}
        for i in range(0, self.n_iter, 1):
            Log(write=False, env='dev').log(msg='Iteration: {}'.format(i))
            for imp in self.imp_sequence:
                Log(write=False, env='dev').log(msg='Imputation of: {}'.format(imp))
                # Step 2: Re-impute missing values for imputing feature
                #self.df.loc[self.nan_idx.get(imp), imp] = np.nan
                if i + 1 > self.n_burn_in_iter:
                    for m in range(0, self.n_chains, 1):
                        # Step 3: Train machine learning algorithm und run prediction for each chain
                        if imp in self.feature_types.get('categorical'):
                            _pred = Classification(clf_params=dict(n_estimators=50)).extreme_gradient_boosting_tree().fit(X=self.df[self.predictors[imp]],
                                                                                                                          y=self.df[imp]
                                                                                                                          ).predict(data=self.df[self.predictors[imp]])
                        elif imp in self.feature_types.get('continuous'):
                            _pred = Regression(reg_params=dict(n_estimators=50)).extreme_gradient_boosting_tree().fit(X=self.df[self.predictors[imp]],
                                                                                                                      y=self.df[imp]
                                                                                                                      ).predict(data=self.df[self.predictors[imp]])
                        elif imp in self.feature_types.get('date'):
                            _pred = Regression(reg_params=dict(n_estimators=50)).extreme_gradient_boosting_tree().fit(X=self.df[self.predictors[imp]],
                                                                                                                      y=self.df[imp]
                                                                                                                      ).predict(data=self.df[self.predictors[imp]])
                        else:
                            raise MultipleImputationException('Data type of feature "{}" not supported for imputation'.format(imp))
                        self.chains.get(m).update({imp: pd.DataFrame(data=_pred, columns=['pred']).loc[self.nan_idx.get(imp), 'pred'].values.tolist()})
                        # Step 4: Impute missing values with predictions
                        self.df.loc[self.nan_idx.get(imp), imp] = self.chains[m].get(imp)
                        if i + 1 == self.n_iter:
                            _pool_std.update({m: dict(std=self.df[imp].std(), diff=_std.get(imp) - self.df[imp].std())})
                else:
                    # Step 3: Train machine learning algorithm und run prediction for each chain
                    if imp in self.feature_types.get('categorical'):
                        _pred = Classification().extreme_gradient_boosting_tree().fit(X=self.df[self.predictors[imp]].compute(),
                                                                                      y=self.df[imp].compute()
                                                                                      ).predict(data=self.df[self.predictors[imp]].compute())
                    elif imp in self.feature_types.get('continuous'):
                        _pred = Regression().extreme_gradient_boosting_tree().fit(X=self.df[self.predictors[imp]].compute(),
                                                                                  y=self.df[imp].compute()
                                                                                  ).predict(data=self.df[self.predictors[imp]].compute())
                    elif imp in self.feature_types.get('date'):
                        _pred = Regression().extreme_gradient_boosting_tree().fit(X=self.df[self.predictors[imp]].compute(),
                                                                                  y=self.df[imp].compute()
                                                                                  ).predict(data=self.df[self.predictors[imp]].compute())
                    else:
                        raise MultipleImputationException('Data type of feature "{}" not supported for imputation'.format(imp))
                    # Step 4: Impute missing values with predictions
                    self.df.loc[self.nan_idx.get(imp), imp] = pd.DataFrame(data=_pred, columns=['pred']).loc[self.nan_idx.get(imp), 'pred'].values.tolist()
        # Step 5: Evaluate imputed chains
        if self.pool_eval_meth == 'std':
            _diff: List[float] = [abs(_pool_std[s].get('diff')) for s in _pool_std.keys()]
            _best_set: int = _diff.index(max(_diff))
            Log(write=False, env='dev').log(msg='Best Set: {}'.format(_best_set))
            for ft in self.imp_sequence:
                self.df.loc[self.nan_idx.get(ft), ft] = self.chains[_best_set].get(ft)
        else:
            raise MultipleImputationException('Evaluation method ({}) for pooling multiple imputed data sets not supported'.format(self.pool_eval_meth))
        return self.df
