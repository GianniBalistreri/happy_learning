import copy
import numpy as np
import pandas as pd
import unittest

from catboost import CatBoostClassifier, CatBoostRegressor
from happy_learning.supervised_machine_learning import CLF_ALGORITHMS, Classification, ModelGeneratorClf, ModelGeneratorReg, REG_ALGORITHMS, Regression
from pygam import GAM
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

DATA_SET_CLF: pd.DataFrame = pd.DataFrame(data=dict(x1=np.random.choice(a=[0, 1], size=1000),
                                                    x2=np.random.choice(a=[0, 1], size=1000),
                                                    x3=np.random.choice(a=[0, 1], size=1000),
                                                    x4=np.random.choice(a=[0, 1], size=1000),
                                                    y=np.random.choice(a=[0, 1], size=1000)
                                                    )
                                          )
DATA_SET_REG: pd.DataFrame = pd.DataFrame(data=dict(x1=np.random.uniform(low=0, high=1, size=1000),
                                                    x2=np.random.uniform(low=0, high=1, size=1000),
                                                    x3=np.random.uniform(low=0, high=1, size=1000),
                                                    x4=np.random.uniform(low=0, high=1, size=1000),
                                                    y=np.random.uniform(low=0, high=1, size=1000)
                                                    )
                                          )


class ClassificationTest(unittest.TestCase):
    """
    Class for testing class Classification
    """
    def test_ada_boosting(self):
        self.assertTrue(expr=isinstance(Classification().ada_boosting(), AdaBoostClassifier))

    def test_ada_boosting_param(self):
        _ada_boost_param: dict = Classification().ada_boosting_param()
        self.assertTrue(expr=_ada_boost_param.get(list(_ada_boost_param.keys())[0]) != Classification().ada_boosting_param().get(list(_ada_boost_param.keys())[0]))

    def test_cat_boost(self):
        self.assertTrue(expr=isinstance(Classification().cat_boost(), CatBoostClassifier))

    def test_cat_boost_param(self):
        _cat_boost_param: dict = Classification().cat_boost_param()
        self.assertTrue(expr=_cat_boost_param.get(list(_cat_boost_param.keys())[0]) != Classification().cat_boost_param().get(list(_cat_boost_param.keys())[0]))

    def test_extreme_gradient_boosting_tree(self):
        self.assertTrue(expr=isinstance(Classification().extreme_gradient_boosting_tree(), XGBClassifier))

    def test_extreme_gradient_boosting_tree_param(self):
        _xgb_param: dict = Classification().extreme_gradient_boosting_tree_param()
        self.assertTrue(expr=_xgb_param.get(list(_xgb_param.keys())[0]) != Classification().extreme_gradient_boosting_tree_param().get(list(_xgb_param.keys())[0]))

    def test_gradient_boosting_tree(self):
        self.assertTrue(expr=isinstance(Classification().gradient_boosting_tree(), GradientBoostingClassifier))

    def test_gradient_boosting_tree_param(self):
        _gbo_param: dict = Classification().gradient_boosting_tree_param()
        self.assertTrue(expr=_gbo_param.get(list(_gbo_param.keys())[0]) != Classification().gradient_boosting_tree_param().get(list(_gbo_param.keys())[0]))

    def test_k_nearest_neighbor(self):
        self.assertTrue(expr=isinstance(Classification().k_nearest_neighbor(), KNeighborsClassifier))

    def test_k_nearest_neighbor_param(self):
        _knn_param: dict = Classification().k_nearest_neighbor_param()
        self.assertTrue(expr=_knn_param.get(list(_knn_param.keys())[0]) != Classification().k_nearest_neighbor_param().get(list(_knn_param.keys())[0]))

    def test_linear_discriminant_analysis(self):
        self.assertTrue(expr=isinstance(Classification().linear_discriminant_analysis(), LinearDiscriminantAnalysis))

    def test_linear_discriminant_analysis_param(self):
        _lda_param: dict = Classification().linear_discriminant_analysis_param()
        self.assertTrue(expr=_lda_param.get(list(_lda_param.keys())[0]) != Classification().linear_discriminant_analysis_param().get(list(_lda_param.keys())[0]))

    def test_logistic_regression(self):
        self.assertTrue(expr=isinstance(Classification().logistic_regression(), LogisticRegression))

    def test_logistic_regression_param(self):
        _logistic_param: dict = Classification().logistic_regression_param()
        self.assertTrue(expr=_logistic_param.get(list(_logistic_param.keys())[0]) != Classification().logistic_regression_param().get(list(_logistic_param.keys())[0]))

    def test_quadratic_discriminant_analysis(self):
        self.assertTrue(expr=isinstance(Classification().quadratic_discriminant_analysis(), QuadraticDiscriminantAnalysis))

    def test_quadratic_discriminant_analysis_param(self):
        _qda_param: dict = Classification().quadratic_discriminant_analysis_param()
        self.assertTrue(expr=_qda_param.get(list(_qda_param.keys())[0]) != Classification().quadratic_discriminant_analysis_param().get(list(_qda_param.keys())[0]))

    def test_random_forest(self):
        self.assertTrue(expr=isinstance(Classification().random_forest(), RandomForestClassifier))

    def test_random_forest_param(self):
        _rf_param: dict = Classification().random_forest_param()
        self.assertTrue(expr=_rf_param.get(list(_rf_param.keys())[0]) != Classification().random_forest_param().get(list(_rf_param.keys())[0]))

    def test_support_vector_machine(self):
        self.assertTrue(expr=isinstance(Classification().support_vector_machine(), SVC))

    def test_support_vector_machine_param(self):
        _svm_param: dict = Classification().support_vector_machine_param()
        self.assertTrue(expr=_svm_param.get(list(_svm_param.keys())[0]) != Classification().support_vector_machine_param().get(list(_svm_param.keys())[0]))

    def test_linear_support_vector_machine(self):
        self.assertTrue(expr=isinstance(Classification().linear_support_vector_machine(), LinearSVC))

    def test_linear_support_vector_machine_param(self):
        _lsvm_param: dict = Classification().linear_support_vector_machine_param()
        self.assertTrue(expr=_lsvm_param.get(list(_lsvm_param.keys())[0]) != Classification().linear_support_vector_machine_param().get(list(_lsvm_param.keys())[0]))

    def test_nu_support_vector_machine(self):
        self.assertTrue(expr=isinstance(Classification().nu_support_vector_machine(), NuSVC))

    def test_nu_support_vector_machine_param(self):
        _nusvm_param: dict = Classification().nu_support_vector_machine_param()
        self.assertTrue(expr=_nusvm_param.get(list(_nusvm_param.keys())[0]) != Classification().nu_support_vector_machine_param().get(list(_nusvm_param.keys())[0]))


class RegressionTest(unittest.TestCase):
    """
    Class for testing class Regression
    """
    def test_ada_boosting(self):
        self.assertTrue(expr=isinstance(Regression().ada_boosting(), AdaBoostRegressor))

    def test_ada_boosting_param(self):
        _ada_boost_param: dict = Regression().ada_boosting_param()
        self.assertTrue(expr=_ada_boost_param.get(list(_ada_boost_param.keys())[0]) != Regression().ada_boosting_param().get(list(_ada_boost_param.keys())[0]))

    def test_cat_boost(self):
        self.assertTrue(expr=isinstance(Regression().cat_boost(), CatBoostRegressor))

    def test_cat_boost_param(self):
        _cat_boost_param: dict = Regression().cat_boost_param()
        self.assertTrue(expr=_cat_boost_param.get(list(_cat_boost_param.keys())[0]) != Regression().cat_boost_param().get(list(_cat_boost_param.keys())[0]))

    def test_elastic_net(self):
        self.assertTrue(expr=isinstance(Regression().elastic_net(), ElasticNet))

    def test_elastic_net_param(self):
        _elastic_param: dict = Regression().elastic_net_param()
        self.assertTrue(expr=_elastic_param.get(list(_elastic_param.keys())[0]) != Regression().elastic_net_param().get(list(_elastic_param.keys())[0]))

    def test_extreme_gradient_boosting_tree(self):
        self.assertTrue(expr=isinstance(Regression().extreme_gradient_boosting_tree(), XGBRegressor))

    def test_extreme_gradient_boosting_tree_param(self):
        _xgb_param: dict = Regression().extreme_gradient_boosting_tree_param()
        self.assertTrue(expr=_xgb_param.get(list(_xgb_param.keys())[0]) != Regression().extreme_gradient_boosting_tree_param().get(list(_xgb_param.keys())[0]))

    def test_gradient_boosting_tree(self):
        self.assertTrue(expr=isinstance(Regression().gradient_boosting_tree(), GradientBoostingRegressor))

    def test_gradient_boosting_tree_param(self):
        _gbo_param: dict = Regression().gradient_boosting_tree_param()
        self.assertTrue(expr=_gbo_param.get(list(_gbo_param.keys())[0]) != Regression().gradient_boosting_tree_param().get(list(_gbo_param.keys())[0]))

    def test_generalized_additive_models(self):
        self.assertTrue(expr=isinstance(Regression().generalized_additive_models(), GAM))

    def test_generalized_additive_models_param(self):
        _gam_param: dict = Regression().generalized_additive_models_param()
        self.assertTrue(expr=_gam_param.get(list(_gam_param.keys())[0]) != Regression().generalized_additive_models_param().get(list(_gam_param.keys())[0]))

    def test_lasso_regression(self):
        self.assertTrue(expr=isinstance(Regression().lasso_regression(), Lasso))

    def test_lasso_regression_param(self):
        _lasso_param: dict = Regression().elastic_net_param()
        self.assertTrue(expr=_lasso_param.get(list(_lasso_param.keys())[0]) != Regression().lasso_regression_param().get(list(_lasso_param.keys())[0]))

    def test_k_nearest_neighbor(self):
        self.assertTrue(expr=isinstance(Regression().k_nearest_neighbor(), KNeighborsRegressor))

    def test_k_nearest_neighbor_param(self):
        _knn_param: dict = Regression().k_nearest_neighbor_param()
        self.assertTrue(expr=_knn_param.get(list(_knn_param.keys())[0]) != Regression().k_nearest_neighbor_param().get(list(_knn_param.keys())[0]))

    def test_random_forest(self):
        self.assertTrue(expr=isinstance(Regression().random_forest(), RandomForestRegressor))

    def test_random_forest_param(self):
        _rf_param: dict = Regression().random_forest_param()
        self.assertTrue(expr=_rf_param.get(list(_rf_param.keys())[0]) != Regression().random_forest_param().get(list(_rf_param.keys())[0]))

    def test_support_vector_machine(self):
        self.assertTrue(expr=isinstance(Regression().support_vector_machine(), SVR))

    def test_support_vector_machine_param(self):
        _svm_param: dict = Regression().support_vector_machine_param()
        self.assertTrue(expr=_svm_param.get(list(_svm_param.keys())[0]) != Regression().support_vector_machine_param().get(list(_svm_param.keys())[0]))

    def test_linear_support_vector_machine(self):
        self.assertTrue(expr=isinstance(Regression().linear_support_vector_machine(), LinearSVR))

    def test_linear_support_vector_machine_param(self):
        _lsvm_param: dict = Regression().linear_support_vector_machine_param()
        self.assertTrue(expr=_lsvm_param.get(list(_lsvm_param.keys())[0]) != Regression().linear_support_vector_machine_param().get(list(_lsvm_param.keys())[0]))

    def test_nu_support_vector_machine(self):
        self.assertTrue(expr=isinstance(Regression().nu_support_vector_machine(), NuSVR))

    def test_nu_support_vector_machine_param(self):
        _nusvm_param: dict = Regression().nu_support_vector_machine_param()
        self.assertTrue(expr=_nusvm_param.get(list(_nusvm_param.keys())[0]) != Regression().nu_support_vector_machine_param().get(list(_nusvm_param.keys())[0]))


class ModelGeneratorClfTest(unittest.TestCase):
    """
    Class for testing class ModelGeneratorClf
    """
    def test_generate_model(self):
        _model = ModelGeneratorClf(model_name=None, clf_params=None, models=['cat']).generate_model()
        self.assertTrue(expr=isinstance(_model.model, CatBoostClassifier))

    def test_generate_params(self):
        _model_generator: ModelGeneratorClf = ModelGeneratorClf(model_name=None, clf_params=None, models=list(CLF_ALGORITHMS.keys()))
        _model = _model_generator.generate_model()
        _mutated_param: dict = copy.deepcopy(_model.model_param_mutated)
        _model_generator.generate_params(param_rate=0.1, force_param=None)
        self.assertTrue(expr=len(_mutated_param.keys()) < len(_model_generator.model_param_mutated.keys()))

    def test_get_model_parameter(self):
        self.assertTrue(expr=len(ModelGeneratorClf(model_name=None, clf_params=None, models=list(CLF_ALGORITHMS.keys())).get_model_parameter().keys()) > 0)

    def test_eval(self):
        _model_generator: ModelGeneratorClf = ModelGeneratorClf(model_name=None, clf_params=None, models=list(CLF_ALGORITHMS.keys()))
        _model_generator.generate_model()
        _x: np.ndarray = DATA_SET_CLF[['x1', 'x2', 'x3', 'x4']].values
        _model_generator.train(x=_x, y=DATA_SET_CLF['y'].values)
        _model_generator.eval(obs=_x.flatten(), pred=_model_generator.predict(x=_x), eval_metric=None, train_error=True)
        self.assertTrue(expr=len(_model_generator.fitness.keys()) > 0)

    def test_predict(self):
        _model_generator: ModelGeneratorClf = ModelGeneratorClf(model_name=None, clf_params=None, models=list(CLF_ALGORITHMS.keys()))
        _model_generator.generate_model()
        _x: np.ndarray = DATA_SET_CLF[['x1', 'x2', 'x3', 'x4']].values
        _model_generator.train(x=_x, y=DATA_SET_CLF['y'])
        self.assertTrue(expr=len(_model_generator.predict(x=_x)) > 0)

    def test_train(self):
        _model_generator: ModelGeneratorClf = ModelGeneratorClf(model_name=None, clf_params=None, models=list(CLF_ALGORITHMS.keys()))
        _model_generator.generate_model()
        _x: np.ndarray = DATA_SET_CLF[['x1', 'x2', 'x3', 'x4']].values
        _model_generator.train(x=_x, y=DATA_SET_CLF['y'].values)
        self.assertTrue(expr=len(_model_generator.predict(x=_x)) > 0)


class ModelGeneratorRegTest(unittest.TestCase):
    """
    Class for testing class ModelGeneratorReg
    """
    def test_generate_model(self):
        _model = ModelGeneratorReg(model_name=None, reg_params=None, models=['cat']).generate_model()
        self.assertTrue(expr=isinstance(_model.model, CatBoostRegressor))

    def test_generate_params(self):
        _model_generator: ModelGeneratorReg = ModelGeneratorReg(model_name=None, reg_params=None, models=list(REG_ALGORITHMS.keys()))
        _model = _model_generator.generate_model()
        _mutated_param: dict = copy.deepcopy(_model.model_param_mutated)
        _model_generator.generate_params(param_rate=0.1, force_param=None)
        self.assertTrue(expr=len(_mutated_param.keys()) < len(_model_generator.model_param_mutated.keys()))

    def test_get_model_parameter(self):
        self.assertTrue(expr=len(ModelGeneratorReg(model_name=None, reg_params=None, models=list(REG_ALGORITHMS.keys())).get_model_parameter().keys()) > 0)

    def test_eval(self):
        _model_generator: ModelGeneratorReg = ModelGeneratorReg(model_name=None, reg_params=None, models=list(REG_ALGORITHMS.keys()))
        _model_generator.generate_model()
        _x: np.ndarray = DATA_SET_REG[['x1', 'x2', 'x3', 'x4']].values
        _model_generator.train(x=_x, y=DATA_SET_REG['y'].values)
        _model_generator.eval(obs=_x.flatten(), pred=_model_generator.predict(x=_x), eval_metric=None, train_error=True)
        self.assertTrue(expr=len(_model_generator.fitness.keys()) > 0)

    def test_predict(self):
        _model_generator: ModelGeneratorReg = ModelGeneratorReg(model_name=None, reg_params=None, models=list(REG_ALGORITHMS.keys()))
        _model_generator.generate_model()
        _x: np.ndarray = DATA_SET_REG[['x1', 'x2', 'x3', 'x4']].values
        _model_generator.train(x=_x, y=DATA_SET_REG['y'].values)
        self.assertTrue(expr=len(_model_generator.predict(x=_x)) > 0)

    def test_train(self):
        _model_generator: ModelGeneratorReg = ModelGeneratorReg(model_name=None, reg_params=None, models=list(REG_ALGORITHMS.keys()))
        _model_generator.generate_model()
        _x: np.ndarray = DATA_SET_REG[['x1', 'x2', 'x3', 'x4']].values
        _model_generator.train(x=_x, y=DATA_SET_REG['y'].values)
        self.assertTrue(expr=len(_model_generator.predict(x=_x)) > 0)


if __name__ == '__main__':
    unittest.main()
