"""

Supervised machine learning model generator (Classification & Regression)

"""

import copy
import joblib
import numpy as np
import pandas as pd
import os

from .evaluate_machine_learning import EvalClf, EvalReg, SML_SCORE
from catboost import CatBoostClassifier, CatBoostRegressor
from datetime import datetime
from pygam import GAM
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, kneighbors_graph
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, SVC, SVR
from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from statsmodels.regression.linear_model import OLS
from typing import Dict, List
from xgboost import XGBClassifier, XGBRegressor

DATA_SHAPE: Dict[str, int] = dict(cases=0, features=0)
CLF_ALGORITHMS: dict = dict(ada='ada_boosting',
                            cat='cat_boost',
                            gbo='gradient_boosting_tree',
                            knn='k_nearest_neighbor',
                            lida='linear_discriminant_analysis',
                            log='logistic_regression',
                            qda='quadratic_discriminant_analysis',
                            rf='random_forest',
                            #lsvm='linear_support_vector_machine',
                            svm='support_vector_machine',
                            nusvm='nu_support_vector_machine',
                            xgb='extreme_gradient_boosting_tree'
                            )

REG_ALGORITHMS: dict = dict(ada='ada_boosting',
                            cat='cat_boost',
                            elastic='elastic_net',
                            gam='generalized_additive_models',
                            gbo='gradient_boosting_tree',
                            knn='k_nearest_neighbor',
                            lasso='lasso_regression',
                            rf='random_forest',
                            svm='support_vector_machine',
                            #lsvm='linear_support_vector_machine',
                            nusvm='nu_support_vector_machine',
                            xgb='extreme_gradient_boosting_tree'
                            )

PARAM_SPACE_CLF: dict = dict(ada=dict(#base_estimator=['base_estimator_None', 'base_estimator_base_decision_tree', 'base_estimator_decision_tree_classifier', 'base_estimator_extra_tree_classifier'],
                                      n_estimators=-1,
                                      learning_rate=-1.0,
                                      #algorithm=['algorithm_SAMME', 'algorithm_SAMME.R']
                                      ),
                             cat=dict(n_estimators=np.random.randint(low=5, high=100),
                                      learning_rate=np.random.uniform(low=0.01, high=1.0),
                                      l2_leaf_reg=np.random.uniform(low=0.1, high=1.0),
                                      depth=np.random.randint(low=3, high=16),
                                      #sampling_frequency=np.random.choice(a=['PerTree', 'PerTreeLevel']),
                                      grow_policy=np.random.choice(a=['SymmetricTree', 'Depthwise', 'Lossguide']),
                                      min_data_in_leaf=np.random.randint(low=1, high=20),
                                      rsm=np.random.uniform(low=0, high=1),
                                      auto_class_weights=np.random.choice(a=[None, 'Balanced', 'SqrtBalanced']),
                                      feature_border_type=np.random.choice(a=['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum', 'MinEntropy', 'GreedyLogSum'])
                                      ),
                             gbo=dict(loss=['loss_deviance', 'loss_exponential'],
                                      learning_rate=-1.0,
                                      n_estimators=-1,
                                      subsample=-1.0,
                                      criterion=['criterion_friedman_mse', 'criterion_mse', 'criterion_mae'],
                                      min_samples_split=-1,
                                      min_samples_leaf=-1,
                                      max_depth=-1,
                                      validation_fraction=-1.0,
                                      n_iter_no_change=-1,
                                      ccp_alpha=-1.0
                                      ),
                             knn=dict(n_neighbors=-1,
                                      weights=['weights_uniform', 'weights_distance'],
                                      algorithm=['algorithm_auto', 'algorithm_ball_tree', 'algorithm_kd_tree', 'algorithm_brute'],
                                      leaf_size=-1,
                                      p=['p_1', 'p_2', 'p_3'],
                                      #metric=['metric_minkowski', 'metric_precomputed']
                                      ),
                             lida=dict(solver=['solver_svd', 'solver_lsqr', 'solver_eigen'],
                                       shrinkage=-1.0
                                       ),
                             log=dict(penalty=['penalty_l1', 'penalty_l2', 'penalty_elasticnet', 'penalty_None'],
                                      C=-1.0,
                                      #solver=['solver_liblinear', 'solver_lbfgs', 'solver_sag', 'solver_saga', 'solver_newton-cg'],
                                      max_iter=-1
                                      ),
                             qda=dict(reg_param=-1.0),
                             rf=dict(n_estimators=-1,
                                     criterion=['criterion_gini', 'criterion_entropy'],
                                     max_depth=-1,
                                     min_samples_split=-1,
                                     min_samples_leaf=-1,
                                     bootstrap=['bootstrap_True', 'bootstrap_False']
                                     ),
                             svm=dict(C=-1.0,
                                      kernel=['kernel_rbf', 'kernel_linear', 'kernel_poly', 'kernel_sigmoid', 'kernel_precomputed'],
                                      #gamma=['gamma_auto', 'gamma_scale'],
                                      shrinking=['shrinking_True', 'shrinking_False'],
                                      cache_size=-1,
                                      decision_function_shape=['decision_function_shape_ovo', 'decision_function_shape_ovr'],
                                      max_iter=-1
                                      ),
                             lsvm=dict(C=-1.0,
                                       penalty=['penalty_l1', 'penalty_l2'],
                                       loss=['loss_hinge', 'loss_squared_hinge'],
                                       multi_class=['multi_class_ovr', 'multi_class_crammer_singer'],
                                       max_iter=-1
                                       ),
                             nusvm=dict(C=-1.0,
                                        kernel=['kernel_rbf', 'kernel_linear', 'kernel_poly', 'kernel_sigmoid', 'kernel_precomputed'],
                                        #gamma=['gamma_auto', 'gamma_scale'],
                                        shrinking=['shrinking_True', 'shrinking_False'],
                                        cache_size=-1,
                                        decision_function_shape=['decision_function_shape_ovo', 'decision_function_shape_ovr'],
                                        max_iter=-1,
                                        nu=-1.0
                                        ),
                             xgb=dict(learning_rate=-1.0,
                                      n_estimators=-1,
                                      min_samples_split=-1.0,
                                      min_samples_leaf=-1,
                                      max_depth=-1,
                                      #booster=['booster_gbtree', 'booster_gblinear', 'booster_gbdart'],
                                      gamma=-1.0,
                                      min_child_weight=-1,
                                      reg_alpha=-1.0,
                                      reg_lambda=-1.0,
                                      subsample=-1.0,
                                      colsample_bytree=-1.0,
                                      #scale_pos_weight=-1.0,
                                      #base_score=-1.0,
                                      early_stopping=['early_stopping_True', 'early_stopping_False']
                                      )
                             )

PARAM_SPACE_REG: dict = dict(ada=dict(#base_estimator=['base_estimator_None', 'base_estimator_base_decision_tree', 'base_estimator_decision_tree_classifier', 'base_estimator_extra_tree_classifier'],
                                      n_estimators=-1,
                                      learning_rate=-1.0,
                                      loss=['loss_linear', 'loss_square', 'loss_exponential'],
                                      algorithm=['algorithm_SAMME', 'algorithm_SAMME.R']
                                      ),
                             cat=dict(n_estimators=np.random.randint(low=5, high=100),
                                      learning_rate=np.random.uniform(low=0.01, high=1.0),
                                      l2_leaf_reg=np.random.uniform(low=0.1, high=1.0),
                                      depth=np.random.randint(low=3, high=16),
                                      #sampling_frequency=np.random.choice(a=['PerTree', 'PerTreeLevel']),
                                      grow_policy=np.random.choice(a=['SymmetricTree', 'Depthwise', 'Lossguide']),
                                      min_data_in_leaf=np.random.randint(low=1, high=20),
                                      rsm=np.random.uniform(low=0, high=1),
                                      feature_border_type=np.random.choice(a=['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum', 'MinEntropy', 'GreedyLogSum'])
                                      ),
                             elastic=dict(alpha=-1.0,
                                          l1_ratio=-1.0,
                                          normalize=['normalize_True', 'normalize_False'],
                                          #precompute=['precompute_True', 'precompute_False'],
                                          max_iter=-1,
                                          fit_intercept=['fit_intercept_True', 'fit_intercept_False'],
                                          selection=['selection_cyclic', 'selection_random']
                                          ),
                             gam=dict(max_iter=-1,
                                      tol=-1.0,
                                      distribution=['distribution_normal', 'distribution_binomial', 'distribution_poisson', 'distribution_gamma', 'distribution_invgauss'],
                                      link=['link_identity', 'link_logit', 'link_log', 'link_inverse', 'link_inverse-squared']
                                      ),
                             gbo=dict(loss=['loss_ls', 'loss_lad', 'loss_huber', 'loss_quantile'],
                                      learning_rate=-1.0,
                                      n_estimators=-1,
                                      subsample=-1.0,
                                      criterion=['criterion_friedman_mse', 'criterion_mse', 'criterion_mae'],
                                      min_samples_split=-1,
                                      min_samples_leaf=-1,
                                      max_depth=-1,
                                      validation_fraction=-1.0,
                                      n_iter_no_change=-1,
                                      alpha=-1.0,
                                      ccp_alpha=-1.0
                                      ),
                             knn=dict(n_neighbors=-1,
                                      weights=['weights_uniform', 'weights_distance'],
                                      algorithm=['algorithm_auto', 'algorithm_ball_tree', 'algorithm_kd_tree', 'algorithm_brute'],
                                      leaf_size=-1,
                                      p=['p_1', 'p_2', 'p_3'],
                                      #metric=['metric_minkowski', 'metric_precomputed']
                                      ),
                             lasso=dict(alpha=-1.0,
                                        normalize=['normalize_True', 'normalize_False'],
                                        precompute=['precompute_True', 'precompute_False'],
                                        max_iter=-1,
                                        fit_intercept=['fit_intercept_True', 'fit_intercept_False'],
                                        selection=['selection_cyclic', 'selection_random']
                                        ),
                             log=dict(penalty=['penalty_l1', 'penalty_l2', 'penalty_elasticnet', 'penalty_None'],
                                      C=-1.0,
                                      solver=['solver_liblinear', 'solver_lbfgs', 'solver_sag', 'solver_saga', 'solver_newton-cg'],
                                      max_iter=-1
                                      ),
                             rf=dict(n_estimators=-1,
                                     criterion=['criterion_mae', 'criterion_mse'],
                                     max_depth=-1,
                                     min_samples_split=-1,
                                     min_samples_leaf=-1,
                                     bootstrap=['bootstrap_True', 'bootstrap_False']
                                     ),
                             svm=dict(C=-1.0,
                                      kernel=['kernel_rbf', 'kernel_linear', 'kernel_poly', 'kernel_sigmoid', 'kernel_precomputed'],
                                      gamma=['gamma_auto', 'gamma_scale'],
                                      shrinking=['shrinking_True', 'shrinking_False'],
                                      cache_size=-1,
                                      decision_function_shape=['decision_function_shape_ovo', 'decision_function_shape_ovr'],
                                      max_iter=-1
                                      ),
                             lsvm=dict(C=-1.0,
                                       penalty=['penalty_l1', 'penalty_l2'],
                                       loss=['loss_hinge', 'loss_squared_hinge'],
                                       multi_class=['multi_class_ovr', 'multi_class_crammer_singer'],
                                       max_iter=-1
                                       ),
                             nusvm=dict(C=-1.0,
                                        kernel=['kernel_rbf', 'kernel_linear', 'kernel_poly', 'kernel_sigmoid', 'kernel_precomputed'],
                                        #gamma=['gamma_auto', 'gamma_scale'],
                                        shrinking=['shrinking_True', 'shrinking_False'],
                                        cache_size=-1,
                                        decision_function_shape=['decision_function_shape_ovo', 'decision_function_shape_ovr'],
                                        max_iter=-1,
                                        nu=-1.0
                                        ),
                             xgb=dict(learning_rate=-1.0,
                                      n_estimators=-1,
                                      min_samples_split=-1.0,
                                      min_samples_leaf=-1,
                                      max_depth=-1,
                                      #booster=['booster_gbtree', 'booster_gblinear', 'booster_gbdart'],
                                      gamma=-1.0,
                                      min_child_weight=-1,
                                      reg_alpha=-1.0,
                                      reg_lambda=-1.0,
                                      subsample=-1.0,
                                      colsample_bytree=-1.0,
                                      scale_pos_weight=-1.0,
                                      base_score=-1.0,
                                      early_stopping=['early_stopping_True', 'early_stopping_False']
                                      )
                             )

Q_TABLE_PARAM_SPACE_CLF: dict = dict(ada=dict(n_estimators_0=tuple([5, 9]),
                                              n_estimators_1=tuple([10, 14]),
                                              n_estimators_2=tuple([15, 19]),
                                              n_estimators_3=tuple([20, 24]),
                                              n_estimators_4=tuple([25, 29]),
                                              n_estimators_5=tuple([30, 34]),
                                              n_estimators_6=tuple([35, 39]),
                                              n_estimators_7=tuple([40, 44]),
                                              n_estimators_8=tuple([45, 49]),
                                              n_estimators_9=tuple([50, 54]),
                                              n_estimators_10=tuple([55, 59]),
                                              n_estimators_11=tuple([60, 64]),
                                              n_estimators_12=tuple([65, 69]),
                                              n_estimators_13=tuple([70, 74]),
                                              n_estimators_14=tuple([75, 79]),
                                              n_estimators_15=tuple([80, 84]),
                                              n_estimators_16=tuple([85, 89]),
                                              n_estimators_17=tuple([90, 94]),
                                              n_estimators_18=tuple([95, 99]),
                                              n_estimators_19=tuple([100, 104]),
                                              n_estimators_20=tuple([105, 109]),
                                              n_estimators_21=tuple([110, 114]),
                                              n_estimators_22=tuple([115, 119]),
                                              n_estimators_23=tuple([120, 124]),
                                              n_estimators_24=tuple([125, 129]),
                                              n_estimators_25=tuple([130, 134]),
                                              n_estimators_26=tuple([135, 139]),
                                              n_estimators_27=tuple([140, 144]),
                                              n_estimators_28=tuple([145, 149]),
                                              n_estimators_29=tuple([150, 154]),
                                              n_estimators_30=tuple([155, 159]),
                                              n_estimators_31=tuple([160, 164]),
                                              n_estimators_32=tuple([165, 169]),
                                              n_estimators_33=tuple([170, 174]),
                                              n_estimators_34=tuple([175, 179]),
                                              n_estimators_35=tuple([180, 184]),
                                              n_estimators_36=tuple([185, 189]),
                                              n_estimators_37=tuple([190, 194]),
                                              n_estimators_38=tuple([195, 199]),
                                              learning_rate_0=tuple([0.01, 0.09]),
                                              learning_rate_1=tuple([0.09, 0.19]),
                                              learning_rate_2=tuple([0.19, 0.29]),
                                              learning_rate_3=tuple([0.29, 0.39]),
                                              learning_rate_4=tuple([0.39, 0.49]),
                                              learning_rate_5=tuple([0.49, 0.59]),
                                              learning_rate_6=tuple([0.59, 0.69]),
                                              learning_rate_7=tuple([0.69, 0.79]),
                                              learning_rate_8=tuple([0.79, 0.89]),
                                              learning_rate_9=tuple([0.89, 0.99])
                                              ),
                                     cat=dict(n_estimators_0=tuple([5, 9]),
                                              n_estimators_1=tuple([10, 14]),
                                              n_estimators_2=tuple([15, 19]),
                                              n_estimators_3=tuple([20, 24]),
                                              n_estimators_4=tuple([25, 29]),
                                              n_estimators_5=tuple([30, 34]),
                                              n_estimators_6=tuple([35, 39]),
                                              n_estimators_7=tuple([40, 44]),
                                              n_estimators_8=tuple([45, 49]),
                                              n_estimators_9=tuple([50, 54]),
                                              n_estimators_10=tuple([55, 59]),
                                              n_estimators_11=tuple([60, 64]),
                                              n_estimators_12=tuple([65, 69]),
                                              n_estimators_13=tuple([70, 74]),
                                              n_estimators_14=tuple([75, 79]),
                                              n_estimators_15=tuple([80, 84]),
                                              n_estimators_16=tuple([85, 89]),
                                              n_estimators_17=tuple([90, 94]),
                                              n_estimators_18=tuple([95, 99]),
                                              n_estimators_19=tuple([100, 104]),
                                              n_estimators_20=tuple([105, 109]),
                                              n_estimators_21=tuple([110, 114]),
                                              n_estimators_22=tuple([115, 119]),
                                              n_estimators_23=tuple([120, 124]),
                                              n_estimators_24=tuple([125, 129]),
                                              n_estimators_25=tuple([130, 134]),
                                              n_estimators_26=tuple([135, 139]),
                                              n_estimators_27=tuple([140, 144]),
                                              n_estimators_28=tuple([145, 149]),
                                              n_estimators_29=tuple([150, 154]),
                                              n_estimators_30=tuple([155, 159]),
                                              n_estimators_31=tuple([160, 164]),
                                              n_estimators_32=tuple([165, 169]),
                                              n_estimators_33=tuple([170, 174]),
                                              n_estimators_34=tuple([175, 179]),
                                              n_estimators_35=tuple([180, 184]),
                                              n_estimators_36=tuple([185, 189]),
                                              n_estimators_37=tuple([190, 194]),
                                              n_estimators_38=tuple([195, 199]),
                                              learning_rate_0=tuple([0.01, 0.09]),
                                              learning_rate_1=tuple([0.09, 0.19]),
                                              learning_rate_2=tuple([0.19, 0.29]),
                                              learning_rate_3=tuple([0.29, 0.39]),
                                              learning_rate_4=tuple([0.39, 0.49]),
                                              learning_rate_5=tuple([0.49, 0.59]),
                                              learning_rate_6=tuple([0.59, 0.69]),
                                              learning_rate_7=tuple([0.69, 0.79]),
                                              learning_rate_8=tuple([0.79, 0.89]),
                                              learning_rate_9=tuple([0.89, 0.99]),
                                              l2_leaf_reg_0=tuple([0.1, 0.2]),
                                              l2_leaf_reg_1=tuple([0.2, 0.3]),
                                              l2_leaf_reg_2=tuple([0.3, 0.4]),
                                              l2_leaf_reg_3=tuple([0.4, 0.5]),
                                              l2_leaf_reg_4=tuple([0.5, 0.6]),
                                              l2_leaf_reg_5=tuple([0.6, 0.7]),
                                              l2_leaf_reg_6=tuple([0.7, 0.8]),
                                              l2_leaf_reg_7=tuple([0.8, 0.9]),
                                              l2_leaf_reg_8=tuple([0.9, 1.0]),
                                              depth_0=tuple([3, 4]),
                                              depth_1=tuple([5, 6]),
                                              depth_2=tuple([7, 8]),
                                              depth_3=tuple([9, 10]),
                                              depth_4=tuple([11, 12]),
                                              depth_5=tuple([13, 14]),
                                              depth_6=tuple([15, 16]),
                                              grow_policy=['SymmetricTree', 'Depthwise', 'Lossguide'],
                                              min_data_in_leaf_0=tuple([1, 2]),
                                              min_data_in_leaf_1=tuple([3, 4]),
                                              min_data_in_leaf_2=tuple([5, 6]),
                                              min_data_in_leaf_3=tuple([7, 8]),
                                              min_data_in_leaf_4=tuple([9, 10]),
                                              min_data_in_leaf_5=tuple([11, 12]),
                                              min_data_in_leaf_6=tuple([13, 14]),
                                              min_data_in_leaf_7=tuple([15, 16]),
                                              min_data_in_leaf_8=tuple([17, 18]),
                                              min_data_in_leaf_9=tuple([19, 20]),
                                              rsm_0=tuple([0.0, 0.1]),
                                              rsm_1=tuple([0.1, 0.2]),
                                              rsm_2=tuple([0.2, 0.3]),
                                              rsm_3=tuple([0.3, 0.4]),
                                              rsm_4=tuple([0.4, 0.5]),
                                              rsm_5=tuple([0.5, 0.6]),
                                              rsm_6=tuple([0.6, 0.7]),
                                              rsm_7=tuple([0.7, 0.8]),
                                              rsm_8=tuple([0.8, 0.9]),
                                              rsm_9=tuple([0.9, 1.0]),
                                              auto_class_weights=[None, 'Balanced', 'SqrtBalanced'],
                                              feature_border_type=['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum', 'MinEntropy', 'GreedyLogSum']
                                              ),
                                     gbo=dict(learning_rate_0=tuple([0.01, 0.09]),
                                              learning_rate_1=tuple([0.09, 0.19]),
                                              learning_rate_2=tuple([0.19, 0.29]),
                                              learning_rate_3=tuple([0.29, 0.39]),
                                              learning_rate_4=tuple([0.39, 0.49]),
                                              learning_rate_5=tuple([0.49, 0.59]),
                                              learning_rate_6=tuple([0.59, 0.69]),
                                              learning_rate_7=tuple([0.69, 0.79]),
                                              learning_rate_8=tuple([0.79, 0.89]),
                                              learning_rate_9=tuple([0.89, 0.99]),
                                              n_estimators_0=tuple([5, 9]),
                                              n_estimators_1=tuple([10, 14]),
                                              n_estimators_2=tuple([15, 19]),
                                              n_estimators_3=tuple([20, 24]),
                                              n_estimators_4=tuple([25, 29]),
                                              n_estimators_5=tuple([30, 34]),
                                              n_estimators_6=tuple([35, 39]),
                                              n_estimators_7=tuple([40, 44]),
                                              n_estimators_8=tuple([45, 49]),
                                              n_estimators_9=tuple([50, 54]),
                                              n_estimators_10=tuple([55, 59]),
                                              n_estimators_11=tuple([60, 64]),
                                              n_estimators_12=tuple([65, 69]),
                                              n_estimators_13=tuple([70, 74]),
                                              n_estimators_14=tuple([75, 79]),
                                              n_estimators_15=tuple([80, 84]),
                                              n_estimators_16=tuple([85, 89]),
                                              n_estimators_17=tuple([90, 94]),
                                              n_estimators_18=tuple([95, 99]),
                                              n_estimators_19=tuple([100, 104]),
                                              n_estimators_20=tuple([105, 109]),
                                              n_estimators_21=tuple([110, 114]),
                                              n_estimators_22=tuple([115, 119]),
                                              n_estimators_23=tuple([120, 124]),
                                              n_estimators_24=tuple([125, 129]),
                                              n_estimators_25=tuple([130, 134]),
                                              n_estimators_26=tuple([135, 139]),
                                              n_estimators_27=tuple([140, 144]),
                                              n_estimators_28=tuple([145, 149]),
                                              n_estimators_29=tuple([150, 154]),
                                              n_estimators_30=tuple([155, 159]),
                                              n_estimators_31=tuple([160, 164]),
                                              n_estimators_32=tuple([165, 169]),
                                              n_estimators_33=tuple([170, 174]),
                                              n_estimators_34=tuple([175, 179]),
                                              n_estimators_35=tuple([180, 184]),
                                              n_estimators_36=tuple([185, 189]),
                                              n_estimators_37=tuple([190, 194]),
                                              n_estimators_38=tuple([195, 199]),
                                              loss=['deviance', 'exponential'],
                                              subsample_0=tuple([0.0, 0.1]),
                                              subsample_1=tuple([0.1, 0.2]),
                                              subsample_2=tuple([0.2, 0.3]),
                                              subsample_3=tuple([0.3, 0.4]),
                                              subsample_4=tuple([0.4, 0.5]),
                                              subsample_5=tuple([0.5, 0.6]),
                                              subsample_6=tuple([0.6, 0.7]),
                                              subsample_7=tuple([0.7, 0.8]),
                                              subsample_8=tuple([0.8, 0.9]),
                                              subsample_9=tuple([0.9, 1.0]),
                                              criterion=['friedman_mse', 'mse', 'mae'],
                                              min_samples_split_0=tuple([2, 2]),
                                              min_samples_split_1=tuple([3, 3]),
                                              min_samples_split_2=tuple([4, 4]),
                                              min_samples_split_3=tuple([5, 5]),
                                              min_samples_split_4=tuple([6, 6]),
                                              min_samples_leaf_0=tuple([1, 1]),
                                              min_samples_leaf_1=tuple([2, 2]),
                                              min_samples_leaf_2=tuple([3, 3]),
                                              min_samples_leaf_3=tuple([4, 4]),
                                              min_samples_leaf_4=tuple([5, 5]),
                                              min_samples_leaf_5=tuple([6, 6]),
                                              max_depth_0=tuple([1, 2]),
                                              max_depth_1=tuple([3, 4]),
                                              max_depth_2=tuple([5, 6]),
                                              max_depth_3=tuple([7, 8]),
                                              max_depth_4=tuple([9, 10]),
                                              max_depth_5=tuple([11, 12]),
                                              n_iter_no_change_0=tuple([2, 3]),
                                              n_iter_no_change_1=tuple([4, 5]),
                                              n_iter_no_change_2=tuple([6, 7]),
                                              n_iter_no_change_3=tuple([8, 9]),
                                              n_iter_no_change_4=tuple([10, 11]),
                                              ccp_alpha_0=tuple([0.0, 0.09]),
                                              ccp_alpha_1=tuple([0.09, 0.19]),
                                              ccp_alpha_2=tuple([0.19, 0.29]),
                                              ccp_alpha_3=tuple([0.29, 0.39]),
                                              ccp_alpha_4=tuple([0.39, 0.49]),
                                              ccp_alpha_5=tuple([0.49, 0.59]),
                                              ccp_alpha_6=tuple([0.59, 0.69]),
                                              ccp_alpha_7=tuple([0.69, 0.79]),
                                              ccp_alpha_8=tuple([0.79, 0.89]),
                                              ccp_alpha_9=tuple([0.89, 0.99]),
                                              validation_fraction_0=tuple([0.05, 0.1]),
                                              validation_fraction_1=tuple([0.1, 0.19]),
                                              validation_fraction_2=tuple([0.19, 0.29]),
                                              validation_fraction_3=tuple([0.29, 0.39]),
                                              validation_fraction_4=tuple([0.39, 0.49])
                                              ),
                                     knn=dict(n_neighbors_0=tuple([2, 3]),
                                              n_neighbors_1=tuple([4, 5]),
                                              n_neighbors_2=tuple([6, 7]),
                                              n_neighbors_3=tuple([8, 9]),
                                              n_neighbors_4=tuple([10, 11]),
                                              n_neighbors_5=tuple([12, 13]),
                                              weights=['uniform', 'distance'],
                                              algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'],
                                              p_0=tuple([1, 1]),
                                              p_1=tuple([2, 2]),
                                              p_2=tuple([3, 3]),
                                              leaf_size_0=tuple([15, 20]),
                                              leaf_size_1=tuple([25, 30]),
                                              leaf_size_2=tuple([35, 40]),
                                              leaf_size_3=tuple([45, 50]),
                                              leaf_size_4=tuple([55, 60]),
                                              leaf_size_5=tuple([65, 70]),
                                              leaf_size_6=tuple([75, 80]),
                                              leaf_size_7=tuple([85, 90]),
                                              leaf_size_8=tuple([95, 100])
                                              ),
                                     lida=dict(shrinkage_0=tuple([0.0001, 0.00049]),
                                               shrinkage_1=tuple([0.0005, 0.0099]),
                                               shrinkage_2=tuple([0.001, 0.0049]),
                                               shrinkage_3=tuple([0.005, 0.0099]),
                                               shrinkage_4=tuple([0.01, 0.049]),
                                               shrinkage_5=tuple([0.05, 0.099]),
                                               shrinkage_6=tuple([0.1, 0.19]),
                                               shrinkage_7=tuple([0.2, 0.29]),
                                               shrinkage_8=tuple([0.3, 0.39]),
                                               shrinkage_9=tuple([0.4, 0.49]),
                                               shrinkage_10=tuple([0.5, 0.59]),
                                               shrinkage_11=tuple([0.6, 0.69]),
                                               shrinkage_12=tuple([0.7, 0.79]),
                                               shrinkage_13=tuple([0.8, 0.89]),
                                               shrinkage_14=tuple([0.9, 0.999]),
                                               solver=['svd', 'eigen']
                                               ),
                                     log=dict(C_0=tuple([0.0001, 0.0009]),
                                              C_1=tuple([0.001, 0.009]),
                                              C_2=tuple([0.01, 0.09]),
                                              C_3=tuple([0.1, 0.2]),
                                              C_4=tuple([0.3, 0.4]),
                                              C_5=tuple([0.5, 0.6]),
                                              C_6=tuple([0.7, 0.8]),
                                              C_7=tuple([0.9, 1.0]),
                                              max_iter_0=tuple([5, 9]),
                                              max_iter_1=tuple([10, 14]),
                                              max_iter_2=tuple([15, 19]),
                                              max_iter_3=tuple([20, 24]),
                                              max_iter_4=tuple([25, 29]),
                                              max_iter_5=tuple([30, 34]),
                                              max_iter_6=tuple([35, 39]),
                                              max_iter_7=tuple([40, 44]),
                                              max_iter_8=tuple([45, 49]),
                                              max_iter_9=tuple([50, 54]),
                                              max_iter_10=tuple([55, 59]),
                                              max_iter_11=tuple([60, 64]),
                                              max_iter_12=tuple([65, 69]),
                                              max_iter_13=tuple([70, 74]),
                                              max_iter_14=tuple([75, 79]),
                                              max_iter_15=tuple([80, 84]),
                                              max_iter_16=tuple([85, 89]),
                                              max_iter_17=tuple([90, 94]),
                                              max_iter_18=tuple([95, 99]),
                                              max_iter_19=tuple([100, 104]),
                                              max_iter_20=tuple([105, 109]),
                                              max_iter_21=tuple([110, 114]),
                                              max_iter_22=tuple([115, 119]),
                                              max_iter_23=tuple([120, 124]),
                                              max_iter_24=tuple([125, 129]),
                                              max_iter_25=tuple([130, 134]),
                                              max_iter_26=tuple([135, 139]),
                                              max_iter_27=tuple([140, 144]),
                                              max_iter_28=tuple([145, 149]),
                                              max_iter_29=tuple([150, 154]),
                                              max_iter_30=tuple([155, 159]),
                                              max_iter_31=tuple([160, 164]),
                                              max_iter_32=tuple([165, 169]),
                                              max_iter_33=tuple([170, 174]),
                                              max_iter_34=tuple([175, 179]),
                                              max_iter_35=tuple([180, 184]),
                                              max_iter_36=tuple([185, 189]),
                                              max_iter_37=tuple([190, 194]),
                                              max_iter_38=tuple([195, 199]),
                                              penalty=['l1', 'l2', 'elasticnet', 'none']
                                              ),
                                     qda=dict(reg_param_0=tuple([0.0001, 0.00049]),
                                              reg_param_1=tuple([0.0005, 0.0099]),
                                              reg_param_2=tuple([0.001, 0.0049]),
                                              reg_param_3=tuple([0.005, 0.0099]),
                                              reg_param_4=tuple([0.01, 0.049]),
                                              reg_param_5=tuple([0.05, 0.099]),
                                              reg_param_6=tuple([0.1, 0.19]),
                                              reg_param_7=tuple([0.2, 0.29]),
                                              reg_param_8=tuple([0.3, 0.39]),
                                              reg_param_9=tuple([0.4, 0.49]),
                                              reg_param_10=tuple([0.5, 0.59]),
                                              reg_param_11=tuple([0.6, 0.69]),
                                              reg_param_12=tuple([0.7, 0.79]),
                                              reg_param_13=tuple([0.8, 0.89]),
                                              reg_param_14=tuple([0.9, 0.999])
                                              ),
                                     rf=dict(n_estimators_0=tuple([5, 9]),
                                             n_estimators_1=tuple([10, 14]),
                                             n_estimators_2=tuple([15, 19]),
                                             n_estimators_3=tuple([20, 24]),
                                             n_estimators_4=tuple([25, 29]),
                                             n_estimators_5=tuple([30, 34]),
                                             n_estimators_6=tuple([35, 39]),
                                             n_estimators_7=tuple([40, 44]),
                                             n_estimators_8=tuple([45, 49]),
                                             n_estimators_9=tuple([50, 54]),
                                             n_estimators_10=tuple([55, 59]),
                                             n_estimators_11=tuple([60, 64]),
                                             n_estimators_12=tuple([65, 69]),
                                             n_estimators_13=tuple([70, 74]),
                                             n_estimators_14=tuple([75, 79]),
                                             n_estimators_15=tuple([80, 84]),
                                             n_estimators_16=tuple([85, 89]),
                                             n_estimators_17=tuple([90, 94]),
                                             n_estimators_18=tuple([95, 99]),
                                             n_estimators_19=tuple([100, 104]),
                                             n_estimators_20=tuple([105, 109]),
                                             n_estimators_21=tuple([110, 114]),
                                             n_estimators_22=tuple([115, 119]),
                                             n_estimators_23=tuple([120, 124]),
                                             n_estimators_24=tuple([125, 129]),
                                             n_estimators_25=tuple([130, 134]),
                                             n_estimators_26=tuple([135, 139]),
                                             n_estimators_27=tuple([140, 144]),
                                             n_estimators_28=tuple([145, 149]),
                                             n_estimators_29=tuple([150, 154]),
                                             n_estimators_30=tuple([155, 159]),
                                             n_estimators_31=tuple([160, 164]),
                                             n_estimators_32=tuple([165, 169]),
                                             n_estimators_33=tuple([170, 174]),
                                             n_estimators_34=tuple([175, 179]),
                                             n_estimators_35=tuple([180, 184]),
                                             n_estimators_36=tuple([185, 189]),
                                             n_estimators_37=tuple([190, 194]),
                                             n_estimators_38=tuple([195, 199]),
                                             criterion=['gini', 'entropy'],
                                             bootstrap=[True, False],
                                             min_samples_split_0=tuple([2, 2]),
                                             min_samples_split_1=tuple([3, 3]),
                                             min_samples_split_2=tuple([4, 4]),
                                             min_samples_split_3=tuple([5, 5]),
                                             min_samples_split_4=tuple([6, 6]),
                                             min_samples_leaf_0=tuple([1, 1]),
                                             min_samples_leaf_1=tuple([2, 2]),
                                             min_samples_leaf_2=tuple([3, 3]),
                                             min_samples_leaf_3=tuple([4, 4]),
                                             min_samples_leaf_4=tuple([5, 5]),
                                             min_samples_leaf_5=tuple([6, 6]),
                                             max_depth_0=tuple([1, 2]),
                                             max_depth_1=tuple([3, 4]),
                                             max_depth_2=tuple([5, 6]),
                                             max_depth_3=tuple([7, 8]),
                                             max_depth_4=tuple([9, 10]),
                                             max_depth_5=tuple([11, 12])
                                             ),
                                     svm=dict(C_0=tuple([0.0001, 0.0009]),
                                              C_1=tuple([0.001, 0.009]),
                                              C_2=tuple([0.01, 0.09]),
                                              C_3=tuple([0.1, 0.2]),
                                              C_4=tuple([0.3, 0.4]),
                                              C_5=tuple([0.5, 0.6]),
                                              C_6=tuple([0.7, 0.8]),
                                              C_7=tuple([0.9, 1.0]),
                                              kernel=['rbf', 'linear', 'poly', 'sigmoid'],
                                              shrinking=[True, False],
                                              cache_size_0=tuple([100, 200]),
                                              cache_size_1=tuple([300, 400]),
                                              cache_size_2=tuple([500, 600]),
                                              cache_size_3=tuple([700, 800]),
                                              cache_size_4=tuple([900, 1000]),
                                              decision_function_shape=['ovo', 'ovr'],
                                              max_iter_0=tuple([5, 9]),
                                              max_iter_1=tuple([10, 14]),
                                              max_iter_2=tuple([15, 19]),
                                              max_iter_3=tuple([20, 24]),
                                              max_iter_4=tuple([25, 29]),
                                              max_iter_5=tuple([30, 34]),
                                              max_iter_6=tuple([35, 39]),
                                              max_iter_7=tuple([40, 44]),
                                              max_iter_8=tuple([45, 49]),
                                              max_iter_9=tuple([50, 54]),
                                              max_iter_10=tuple([55, 59]),
                                              max_iter_11=tuple([60, 64]),
                                              max_iter_12=tuple([65, 69]),
                                              max_iter_13=tuple([70, 74]),
                                              max_iter_14=tuple([75, 79]),
                                              max_iter_15=tuple([80, 84]),
                                              max_iter_16=tuple([85, 89]),
                                              max_iter_17=tuple([90, 94]),
                                              max_iter_18=tuple([95, 99]),
                                              max_iter_19=tuple([100, 104]),
                                              max_iter_20=tuple([105, 109]),
                                              max_iter_21=tuple([110, 114]),
                                              max_iter_22=tuple([115, 119]),
                                              max_iter_23=tuple([120, 124]),
                                              max_iter_24=tuple([125, 129]),
                                              max_iter_25=tuple([130, 134]),
                                              max_iter_26=tuple([135, 139]),
                                              max_iter_27=tuple([140, 144]),
                                              max_iter_28=tuple([145, 149]),
                                              max_iter_29=tuple([150, 154]),
                                              max_iter_30=tuple([155, 159]),
                                              max_iter_31=tuple([160, 164]),
                                              max_iter_32=tuple([165, 169]),
                                              max_iter_33=tuple([170, 174]),
                                              max_iter_34=tuple([175, 179]),
                                              max_iter_35=tuple([180, 184]),
                                              max_iter_36=tuple([185, 189]),
                                              max_iter_37=tuple([190, 194]),
                                              max_iter_38=tuple([195, 199])
                                              ),
                                     nusvm=dict(C_0=tuple([0.0001, 0.0009]),
                                                C_1=tuple([0.001, 0.009]),
                                                C_2=tuple([0.01, 0.09]),
                                                C_3=tuple([0.1, 0.2]),
                                                C_4=tuple([0.3, 0.4]),
                                                C_5=tuple([0.5, 0.6]),
                                                C_6=tuple([0.7, 0.8]),
                                                C_7=tuple([0.9, 1.0]),
                                                max_iter_0=tuple([5, 9]),
                                                max_iter_1=tuple([10, 14]),
                                                max_iter_2=tuple([15, 19]),
                                                max_iter_3=tuple([20, 24]),
                                                max_iter_4=tuple([25, 29]),
                                                max_iter_5=tuple([30, 34]),
                                                max_iter_6=tuple([35, 39]),
                                                max_iter_7=tuple([40, 44]),
                                                max_iter_8=tuple([45, 49]),
                                                max_iter_9=tuple([50, 54]),
                                                max_iter_10=tuple([55, 59]),
                                                max_iter_11=tuple([60, 64]),
                                                max_iter_12=tuple([65, 69]),
                                                max_iter_13=tuple([70, 74]),
                                                max_iter_14=tuple([75, 79]),
                                                max_iter_15=tuple([80, 84]),
                                                max_iter_16=tuple([85, 89]),
                                                max_iter_17=tuple([90, 94]),
                                                max_iter_18=tuple([95, 99]),
                                                max_iter_19=tuple([100, 104]),
                                                max_iter_20=tuple([105, 109]),
                                                max_iter_21=tuple([110, 114]),
                                                max_iter_22=tuple([115, 119]),
                                                max_iter_23=tuple([120, 124]),
                                                max_iter_24=tuple([125, 129]),
                                                max_iter_25=tuple([130, 134]),
                                                max_iter_26=tuple([135, 139]),
                                                max_iter_27=tuple([140, 144]),
                                                max_iter_28=tuple([145, 149]),
                                                max_iter_29=tuple([150, 154]),
                                                max_iter_30=tuple([155, 159]),
                                                max_iter_31=tuple([160, 164]),
                                                max_iter_32=tuple([165, 169]),
                                                max_iter_33=tuple([170, 174]),
                                                max_iter_34=tuple([175, 179]),
                                                max_iter_35=tuple([180, 184]),
                                                max_iter_36=tuple([185, 189]),
                                                max_iter_37=tuple([190, 194]),
                                                max_iter_38=tuple([195, 199]),
                                                decision_function_shape=['ovo', 'ovr'],
                                                kernel=['rbf', 'linear', 'poly', 'sigmoid'],
                                                shrinking=[True, False],
                                                cache_size_0=tuple([100, 200]),
                                                cache_size_1=tuple([300, 400]),
                                                cache_size_2=tuple([500, 600]),
                                                cache_size_3=tuple([700, 800]),
                                                cache_size_4=tuple([900, 1000]),
                                                nu_0=tuple([0.01, 0.09]),
                                                nu_1=tuple([0.09, 0.19]),
                                                nu_2=tuple([0.19, 0.29]),
                                                nu_3=tuple([0.29, 0.39]),
                                                nu_4=tuple([0.39, 0.49]),
                                                nu_5=tuple([0.49, 0.59]),
                                                nu_6=tuple([0.59, 0.69]),
                                                nu_7=tuple([0.69, 0.79]),
                                                nu_8=tuple([0.79, 0.89]),
                                                nu_9=tuple([0.89, 0.99])
                                                ),
                                     xgb=dict(n_estimators_0=tuple([5, 9]),
                                              n_estimators_1=tuple([10, 14]),
                                              n_estimators_2=tuple([15, 19]),
                                              n_estimators_3=tuple([20, 24]),
                                              n_estimators_4=tuple([25, 29]),
                                              n_estimators_5=tuple([30, 34]),
                                              n_estimators_6=tuple([35, 39]),
                                              n_estimators_7=tuple([40, 44]),
                                              n_estimators_8=tuple([45, 49]),
                                              n_estimators_9=tuple([50, 54]),
                                              n_estimators_10=tuple([55, 59]),
                                              n_estimators_11=tuple([60, 64]),
                                              n_estimators_12=tuple([65, 69]),
                                              n_estimators_13=tuple([70, 74]),
                                              n_estimators_14=tuple([75, 79]),
                                              n_estimators_15=tuple([80, 84]),
                                              n_estimators_16=tuple([85, 89]),
                                              n_estimators_17=tuple([90, 94]),
                                              n_estimators_18=tuple([95, 99]),
                                              n_estimators_19=tuple([100, 104]),
                                              n_estimators_20=tuple([105, 109]),
                                              n_estimators_21=tuple([110, 114]),
                                              n_estimators_22=tuple([115, 119]),
                                              n_estimators_23=tuple([120, 124]),
                                              n_estimators_24=tuple([125, 129]),
                                              n_estimators_25=tuple([130, 134]),
                                              n_estimators_26=tuple([135, 139]),
                                              n_estimators_27=tuple([140, 144]),
                                              n_estimators_28=tuple([145, 149]),
                                              n_estimators_29=tuple([150, 154]),
                                              n_estimators_30=tuple([155, 159]),
                                              n_estimators_31=tuple([160, 164]),
                                              n_estimators_32=tuple([165, 169]),
                                              n_estimators_33=tuple([170, 174]),
                                              n_estimators_34=tuple([175, 179]),
                                              n_estimators_35=tuple([180, 184]),
                                              n_estimators_36=tuple([185, 189]),
                                              n_estimators_37=tuple([190, 194]),
                                              n_estimators_38=tuple([195, 199]),
                                              learning_rate_0=tuple([0.01, 0.09]),
                                              learning_rate_1=tuple([0.09, 0.19]),
                                              learning_rate_2=tuple([0.19, 0.29]),
                                              learning_rate_3=tuple([0.29, 0.39]),
                                              learning_rate_4=tuple([0.39, 0.49]),
                                              learning_rate_5=tuple([0.49, 0.59]),
                                              learning_rate_6=tuple([0.59, 0.69]),
                                              learning_rate_7=tuple([0.69, 0.79]),
                                              learning_rate_8=tuple([0.79, 0.89]),
                                              learning_rate_9=tuple([0.89, 0.99]),
                                              early_stopping=[True, False],
                                              max_depth_0=tuple([3, 4]),
                                              max_depth_1=tuple([5, 6]),
                                              max_depth_2=tuple([7, 8]),
                                              max_depth_3=tuple([9, 10]),
                                              max_depth_4=tuple([11, 12]),
                                              max_depth_5=tuple([13, 14]),
                                              max_depth_6=tuple([15, 16]),
                                              min_samples_leaf_0=tuple([1, 2]),
                                              min_samples_leaf_1=tuple([3, 4]),
                                              min_samples_leaf_2=tuple([5, 6]),
                                              min_samples_leaf_3=tuple([7, 8]),
                                              min_samples_leaf_4=tuple([9, 10]),
                                              min_samples_split_0=tuple([1, 2]),
                                              min_samples_split_1=tuple([3, 4]),
                                              min_samples_split_2=tuple([5, 6]),
                                              min_samples_split_3=tuple([7, 8]),
                                              min_samples_split_4=tuple([9, 10]),
                                              min_child_weight_0=tuple([1, 2]),
                                              min_child_weight_1=tuple([3, 4]),
                                              min_child_weight_2=tuple([5, 6]),
                                              min_child_weight_3=tuple([7, 8]),
                                              min_child_weight_4=tuple([9, 10]),
                                              reg_alpha_0=tuple([0.01, 0.09]),
                                              reg_alpha_1=tuple([0.09, 0.19]),
                                              reg_alpha_2=tuple([0.19, 0.29]),
                                              reg_alpha_3=tuple([0.29, 0.39]),
                                              reg_alpha_4=tuple([0.39, 0.49]),
                                              reg_alpha_5=tuple([0.49, 0.59]),
                                              reg_alpha_6=tuple([0.59, 0.69]),
                                              reg_alpha_7=tuple([0.69, 0.79]),
                                              reg_alpha_8=tuple([0.79, 0.89]),
                                              reg_alpha_9=tuple([0.89, 0.99]),
                                              reg_lambda_0=tuple([0.01, 0.09]),
                                              reg_lambda_1=tuple([0.09, 0.19]),
                                              reg_lambda_2=tuple([0.19, 0.29]),
                                              reg_lambda_3=tuple([0.29, 0.39]),
                                              reg_lambda_4=tuple([0.39, 0.49]),
                                              reg_lambda_5=tuple([0.49, 0.59]),
                                              reg_lambda_6=tuple([0.59, 0.69]),
                                              reg_lambda_7=tuple([0.69, 0.79]),
                                              reg_lambda_8=tuple([0.79, 0.89]),
                                              reg_lambda_9=tuple([0.89, 0.99]),
                                              gamma_0=tuple([0.01, 0.09]),
                                              gamma_1=tuple([0.09, 0.19]),
                                              gamma_2=tuple([0.19, 0.29]),
                                              gamma_3=tuple([0.29, 0.39]),
                                              gamma_4=tuple([0.39, 0.49]),
                                              gamma_5=tuple([0.49, 0.59]),
                                              gamma_6=tuple([0.59, 0.69]),
                                              gamma_7=tuple([0.69, 0.79]),
                                              gamma_8=tuple([0.79, 0.89]),
                                              gamma_9=tuple([0.89, 0.99]),
                                              subsample_0=tuple([0.01, 0.09]),
                                              subsample_1=tuple([0.09, 0.19]),
                                              subsample_2=tuple([0.19, 0.29]),
                                              subsample_3=tuple([0.29, 0.39]),
                                              subsample_4=tuple([0.39, 0.49]),
                                              subsample_5=tuple([0.49, 0.59]),
                                              subsample_6=tuple([0.59, 0.69]),
                                              subsample_7=tuple([0.69, 0.79]),
                                              subsample_8=tuple([0.79, 0.89]),
                                              subsample_9=tuple([0.89, 0.99]),
                                              colsample_bytree_5=tuple([0.49, 0.59]),
                                              colsample_bytree_6=tuple([0.59, 0.69]),
                                              colsample_bytree_7=tuple([0.69, 0.79]),
                                              colsample_bytree_8=tuple([0.79, 0.89]),
                                              colsample_bytree_9=tuple([0.89, 0.99])
                                              )
                                     )


Q_TABLE_PARAM_SPACE_REG: dict = dict(ada=dict(n_estimators_0=tuple([5, 9]),
                                              n_estimators_1=tuple([10, 14]),
                                              n_estimators_2=tuple([15, 19]),
                                              n_estimators_3=tuple([20, 24]),
                                              n_estimators_4=tuple([25, 29]),
                                              n_estimators_5=tuple([30, 34]),
                                              n_estimators_6=tuple([35, 39]),
                                              n_estimators_7=tuple([40, 44]),
                                              n_estimators_8=tuple([45, 49]),
                                              n_estimators_9=tuple([50, 54]),
                                              n_estimators_10=tuple([55, 59]),
                                              n_estimators_11=tuple([60, 64]),
                                              n_estimators_12=tuple([65, 69]),
                                              n_estimators_13=tuple([70, 74]),
                                              n_estimators_14=tuple([75, 79]),
                                              n_estimators_15=tuple([80, 84]),
                                              n_estimators_16=tuple([85, 89]),
                                              n_estimators_17=tuple([90, 94]),
                                              n_estimators_18=tuple([95, 99]),
                                              n_estimators_19=tuple([100, 104]),
                                              n_estimators_20=tuple([105, 109]),
                                              n_estimators_21=tuple([110, 114]),
                                              n_estimators_22=tuple([115, 119]),
                                              n_estimators_23=tuple([120, 124]),
                                              n_estimators_24=tuple([125, 129]),
                                              n_estimators_25=tuple([130, 134]),
                                              n_estimators_26=tuple([135, 139]),
                                              n_estimators_27=tuple([140, 144]),
                                              n_estimators_28=tuple([145, 149]),
                                              n_estimators_29=tuple([150, 154]),
                                              n_estimators_30=tuple([155, 159]),
                                              n_estimators_31=tuple([160, 164]),
                                              n_estimators_32=tuple([165, 169]),
                                              n_estimators_33=tuple([170, 174]),
                                              n_estimators_34=tuple([175, 179]),
                                              n_estimators_35=tuple([180, 184]),
                                              n_estimators_36=tuple([185, 189]),
                                              n_estimators_37=tuple([190, 194]),
                                              n_estimators_38=tuple([195, 199]),
                                              learning_rate_0=tuple([0.01, 0.09]),
                                              learning_rate_1=tuple([0.09, 0.19]),
                                              learning_rate_2=tuple([0.19, 0.29]),
                                              learning_rate_3=tuple([0.29, 0.39]),
                                              learning_rate_4=tuple([0.39, 0.49]),
                                              learning_rate_5=tuple([0.49, 0.59]),
                                              learning_rate_6=tuple([0.59, 0.69]),
                                              learning_rate_7=tuple([0.69, 0.79]),
                                              learning_rate_8=tuple([0.79, 0.89]),
                                              learning_rate_9=tuple([0.89, 0.99]),
                                              algorithm=['SAMME', 'SAMME.R'],
                                              loss=['linear', 'square', 'exponential']
                                              ),
                                     cat=dict(n_estimators_0=tuple([5, 9]),
                                              n_estimators_1=tuple([10, 14]),
                                              n_estimators_2=tuple([15, 19]),
                                              n_estimators_3=tuple([20, 24]),
                                              n_estimators_4=tuple([25, 29]),
                                              n_estimators_5=tuple([30, 34]),
                                              n_estimators_6=tuple([35, 39]),
                                              n_estimators_7=tuple([40, 44]),
                                              n_estimators_8=tuple([45, 49]),
                                              n_estimators_9=tuple([50, 54]),
                                              n_estimators_10=tuple([55, 59]),
                                              n_estimators_11=tuple([60, 64]),
                                              n_estimators_12=tuple([65, 69]),
                                              n_estimators_13=tuple([70, 74]),
                                              n_estimators_14=tuple([75, 79]),
                                              n_estimators_15=tuple([80, 84]),
                                              n_estimators_16=tuple([85, 89]),
                                              n_estimators_17=tuple([90, 94]),
                                              n_estimators_18=tuple([95, 99]),
                                              n_estimators_19=tuple([100, 104]),
                                              n_estimators_20=tuple([105, 109]),
                                              n_estimators_21=tuple([110, 114]),
                                              n_estimators_22=tuple([115, 119]),
                                              n_estimators_23=tuple([120, 124]),
                                              n_estimators_24=tuple([125, 129]),
                                              n_estimators_25=tuple([130, 134]),
                                              n_estimators_26=tuple([135, 139]),
                                              n_estimators_27=tuple([140, 144]),
                                              n_estimators_28=tuple([145, 149]),
                                              n_estimators_29=tuple([150, 154]),
                                              n_estimators_30=tuple([155, 159]),
                                              n_estimators_31=tuple([160, 164]),
                                              n_estimators_32=tuple([165, 169]),
                                              n_estimators_33=tuple([170, 174]),
                                              n_estimators_34=tuple([175, 179]),
                                              n_estimators_35=tuple([180, 184]),
                                              n_estimators_36=tuple([185, 189]),
                                              n_estimators_37=tuple([190, 194]),
                                              n_estimators_38=tuple([195, 199]),
                                              learning_rate_0=tuple([0.01, 0.09]),
                                              learning_rate_1=tuple([0.09, 0.19]),
                                              learning_rate_2=tuple([0.19, 0.29]),
                                              learning_rate_3=tuple([0.29, 0.39]),
                                              learning_rate_4=tuple([0.39, 0.49]),
                                              learning_rate_5=tuple([0.49, 0.59]),
                                              learning_rate_6=tuple([0.59, 0.69]),
                                              learning_rate_7=tuple([0.69, 0.79]),
                                              learning_rate_8=tuple([0.79, 0.89]),
                                              learning_rate_9=tuple([0.89, 0.99]),
                                              l2_leaf_reg_0=tuple([0.1, 0.2]),
                                              l2_leaf_reg_1=tuple([0.2, 0.3]),
                                              l2_leaf_reg_2=tuple([0.3, 0.4]),
                                              l2_leaf_reg_3=tuple([0.4, 0.5]),
                                              l2_leaf_reg_4=tuple([0.5, 0.6]),
                                              l2_leaf_reg_5=tuple([0.6, 0.7]),
                                              l2_leaf_reg_6=tuple([0.7, 0.8]),
                                              l2_leaf_reg_7=tuple([0.8, 0.9]),
                                              l2_leaf_reg_8=tuple([0.9, 1.0]),
                                              depth_0=tuple([3, 4]),
                                              depth_1=tuple([5, 6]),
                                              depth_2=tuple([7, 8]),
                                              depth_3=tuple([9, 10]),
                                              depth_4=tuple([11, 12]),
                                              depth_5=tuple([13, 14]),
                                              depth_6=tuple([15, 16]),
                                              grow_policy=['SymmetricTree', 'Depthwise', 'Lossguide'],
                                              min_data_in_leaf_0=tuple([1, 2]),
                                              min_data_in_leaf_1=tuple([3, 4]),
                                              min_data_in_leaf_2=tuple([5, 6]),
                                              min_data_in_leaf_3=tuple([7, 8]),
                                              min_data_in_leaf_4=tuple([9, 10]),
                                              min_data_in_leaf_5=tuple([11, 12]),
                                              min_data_in_leaf_6=tuple([13, 14]),
                                              min_data_in_leaf_7=tuple([15, 16]),
                                              min_data_in_leaf_8=tuple([17, 18]),
                                              min_data_in_leaf_9=tuple([19, 20]),
                                              rsm_0=tuple([0.0, 0.1]),
                                              rsm_1=tuple([0.1, 0.2]),
                                              rsm_2=tuple([0.2, 0.3]),
                                              rsm_3=tuple([0.3, 0.4]),
                                              rsm_4=tuple([0.4, 0.5]),
                                              rsm_5=tuple([0.5, 0.6]),
                                              rsm_6=tuple([0.6, 0.7]),
                                              rsm_7=tuple([0.7, 0.8]),
                                              rsm_8=tuple([0.8, 0.9]),
                                              rsm_9=tuple([0.9, 1.0]),
                                              feature_border_type=['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum', 'MinEntropy', 'GreedyLogSum']
                                              ),
                                     elastic=dict(alpha_0=tuple([0.0, 0.1]),
                                                  alpha_1=tuple([0.1, 0.2]),
                                                  alpha_2=tuple([0.2, 0.3]),
                                                  alpha_3=tuple([0.3, 0.4]),
                                                  alpha_4=tuple([0.4, 0.5]),
                                                  alpha_5=tuple([0.5, 0.6]),
                                                  alpha_6=tuple([0.6, 0.7]),
                                                  alpha_7=tuple([0.7, 0.8]),
                                                  alpha_8=tuple([0.8, 0.9]),
                                                  alpha_9=tuple([0.9, 1.0]),
                                                  l1_ratio_0=tuple([0.0, 0.1]),
                                                  l1_ratio_1=tuple([0.1, 0.2]),
                                                  l1_ratio_2=tuple([0.2, 0.3]),
                                                  l1_ratio_3=tuple([0.3, 0.4]),
                                                  l1_ratio_4=tuple([0.4, 0.5]),
                                                  l1_ratio_5=tuple([0.5, 0.6]),
                                                  l1_ratio_6=tuple([0.6, 0.7]),
                                                  l1_ratio_7=tuple([0.7, 0.8]),
                                                  l1_ratio_8=tuple([0.8, 0.9]),
                                                  l1_ratio_9=tuple([0.9, 1.0]),
                                                  normalize=[True, False],
                                                  fit_intercept=[True, False],
                                                  selection=['cyclic', 'random'],
                                                  max_iter_0=tuple([5, 9]),
                                                  max_iter_1=tuple([10, 14]),
                                                  max_iter_2=tuple([15, 19]),
                                                  max_iter_3=tuple([20, 24]),
                                                  max_iter_4=tuple([25, 29]),
                                                  max_iter_5=tuple([30, 34]),
                                                  max_iter_6=tuple([35, 39]),
                                                  max_iter_7=tuple([40, 44]),
                                                  max_iter_8=tuple([45, 49]),
                                                  max_iter_9=tuple([50, 54]),
                                                  max_iter_10=tuple([55, 59]),
                                                  max_iter_11=tuple([60, 64]),
                                                  max_iter_12=tuple([65, 69]),
                                                  max_iter_13=tuple([70, 74]),
                                                  max_iter_14=tuple([75, 79]),
                                                  max_iter_15=tuple([80, 84]),
                                                  max_iter_16=tuple([85, 89]),
                                                  max_iter_17=tuple([90, 94]),
                                                  max_iter_18=tuple([95, 99]),
                                                  max_iter_19=tuple([100, 104]),
                                                  max_iter_20=tuple([105, 109]),
                                                  max_iter_21=tuple([110, 114]),
                                                  max_iter_22=tuple([115, 119]),
                                                  max_iter_23=tuple([120, 124]),
                                                  max_iter_24=tuple([125, 129]),
                                                  max_iter_25=tuple([130, 134]),
                                                  max_iter_26=tuple([135, 139]),
                                                  max_iter_27=tuple([140, 144]),
                                                  max_iter_28=tuple([145, 149]),
                                                  max_iter_29=tuple([150, 154]),
                                                  max_iter_30=tuple([155, 159]),
                                                  max_iter_31=tuple([160, 164]),
                                                  max_iter_32=tuple([165, 169]),
                                                  max_iter_33=tuple([170, 174]),
                                                  max_iter_34=tuple([175, 179]),
                                                  max_iter_35=tuple([180, 184]),
                                                  max_iter_36=tuple([185, 189]),
                                                  max_iter_37=tuple([190, 194]),
                                                  max_iter_38=tuple([195, 199])
                                                  ),
                                     gam=dict(max_iter_0=tuple([5, 9]),
                                              max_iter_1=tuple([10, 14]),
                                              max_iter_2=tuple([15, 19]),
                                              max_iter_3=tuple([20, 24]),
                                              max_iter_4=tuple([25, 29]),
                                              max_iter_5=tuple([30, 34]),
                                              max_iter_6=tuple([35, 39]),
                                              max_iter_7=tuple([40, 44]),
                                              max_iter_8=tuple([45, 49]),
                                              max_iter_9=tuple([50, 54]),
                                              max_iter_10=tuple([55, 59]),
                                              max_iter_11=tuple([60, 64]),
                                              max_iter_12=tuple([65, 69]),
                                              max_iter_13=tuple([70, 74]),
                                              max_iter_14=tuple([75, 79]),
                                              max_iter_15=tuple([80, 84]),
                                              max_iter_16=tuple([85, 89]),
                                              max_iter_17=tuple([90, 94]),
                                              max_iter_18=tuple([95, 99]),
                                              max_iter_19=tuple([100, 104]),
                                              max_iter_20=tuple([105, 109]),
                                              max_iter_21=tuple([110, 114]),
                                              max_iter_22=tuple([115, 119]),
                                              max_iter_23=tuple([120, 124]),
                                              max_iter_24=tuple([125, 129]),
                                              max_iter_25=tuple([130, 134]),
                                              max_iter_26=tuple([135, 139]),
                                              max_iter_27=tuple([140, 144]),
                                              max_iter_28=tuple([145, 149]),
                                              max_iter_29=tuple([150, 154]),
                                              max_iter_30=tuple([155, 159]),
                                              max_iter_31=tuple([160, 164]),
                                              max_iter_32=tuple([165, 169]),
                                              max_iter_33=tuple([170, 174]),
                                              max_iter_34=tuple([175, 179]),
                                              max_iter_35=tuple([180, 184]),
                                              max_iter_36=tuple([185, 189]),
                                              max_iter_37=tuple([190, 194]),
                                              max_iter_38=tuple([195, 199]),
                                              tol_0=tuple([0.00001, 0.00005]),
                                              tol_1=tuple([0.00005, 0.00009]),
                                              tol_2=tuple([0.00009, 0.0005]),
                                              tol_3=tuple([0.0005, 0.001]),
                                              distribution=['normal', 'binomial', 'poisson', 'gamma', 'invgauss'],
                                              link=['identity', 'logit', 'log', 'inverse', 'inverse-squared']
                                              ),
                                     gbo=dict(learning_rate_0=tuple([0.01, 0.09]),
                                              learning_rate_1=tuple([0.09, 0.19]),
                                              learning_rate_2=tuple([0.19, 0.29]),
                                              learning_rate_3=tuple([0.29, 0.39]),
                                              learning_rate_4=tuple([0.39, 0.49]),
                                              learning_rate_5=tuple([0.49, 0.59]),
                                              learning_rate_6=tuple([0.59, 0.69]),
                                              learning_rate_7=tuple([0.69, 0.79]),
                                              learning_rate_8=tuple([0.79, 0.89]),
                                              learning_rate_9=tuple([0.89, 0.99]),
                                              n_estimators_0=tuple([5, 9]),
                                              n_estimators_1=tuple([10, 14]),
                                              n_estimators_2=tuple([15, 19]),
                                              n_estimators_3=tuple([20, 24]),
                                              n_estimators_4=tuple([25, 29]),
                                              n_estimators_5=tuple([30, 34]),
                                              n_estimators_6=tuple([35, 39]),
                                              n_estimators_7=tuple([40, 44]),
                                              n_estimators_8=tuple([45, 49]),
                                              n_estimators_9=tuple([50, 54]),
                                              n_estimators_10=tuple([55, 59]),
                                              n_estimators_11=tuple([60, 64]),
                                              n_estimators_12=tuple([65, 69]),
                                              n_estimators_13=tuple([70, 74]),
                                              n_estimators_14=tuple([75, 79]),
                                              n_estimators_15=tuple([80, 84]),
                                              n_estimators_16=tuple([85, 89]),
                                              n_estimators_17=tuple([90, 94]),
                                              n_estimators_18=tuple([95, 99]),
                                              n_estimators_19=tuple([100, 104]),
                                              n_estimators_20=tuple([105, 109]),
                                              n_estimators_21=tuple([110, 114]),
                                              n_estimators_22=tuple([115, 119]),
                                              n_estimators_23=tuple([120, 124]),
                                              n_estimators_24=tuple([125, 129]),
                                              n_estimators_25=tuple([130, 134]),
                                              n_estimators_26=tuple([135, 139]),
                                              n_estimators_27=tuple([140, 144]),
                                              n_estimators_28=tuple([145, 149]),
                                              n_estimators_29=tuple([150, 154]),
                                              n_estimators_30=tuple([155, 159]),
                                              n_estimators_31=tuple([160, 164]),
                                              n_estimators_32=tuple([165, 169]),
                                              n_estimators_33=tuple([170, 174]),
                                              n_estimators_34=tuple([175, 179]),
                                              n_estimators_35=tuple([180, 184]),
                                              n_estimators_36=tuple([185, 189]),
                                              n_estimators_37=tuple([190, 194]),
                                              n_estimators_38=tuple([195, 199]),
                                              loss=['ls', 'lad', 'huber', 'quantile'],
                                              subsample_0=tuple([0.0, 0.1]),
                                              subsample_1=tuple([0.1, 0.2]),
                                              subsample_2=tuple([0.2, 0.3]),
                                              subsample_3=tuple([0.3, 0.4]),
                                              subsample_4=tuple([0.4, 0.5]),
                                              subsample_5=tuple([0.5, 0.6]),
                                              subsample_6=tuple([0.6, 0.7]),
                                              subsample_7=tuple([0.7, 0.8]),
                                              subsample_8=tuple([0.8, 0.9]),
                                              subsample_9=tuple([0.9, 1.0]),
                                              criterion=['friedman_mse', 'mse', 'mae'],
                                              min_samples_split_0=tuple([2, 2]),
                                              min_samples_split_1=tuple([3, 3]),
                                              min_samples_split_2=tuple([4, 4]),
                                              min_samples_split_3=tuple([5, 5]),
                                              min_samples_split_4=tuple([6, 6]),
                                              min_samples_leaf_0=tuple([1, 1]),
                                              min_samples_leaf_1=tuple([2, 2]),
                                              min_samples_leaf_2=tuple([3, 3]),
                                              min_samples_leaf_3=tuple([4, 4]),
                                              min_samples_leaf_4=tuple([5, 5]),
                                              min_samples_leaf_5=tuple([6, 6]),
                                              max_depth_0=tuple([1, 2]),
                                              max_depth_1=tuple([3, 4]),
                                              max_depth_2=tuple([5, 6]),
                                              max_depth_3=tuple([7, 8]),
                                              max_depth_4=tuple([9, 10]),
                                              max_depth_5=tuple([11, 12]),
                                              n_iter_no_change_0=tuple([2, 3]),
                                              n_iter_no_change_1=tuple([4, 5]),
                                              n_iter_no_change_2=tuple([6, 7]),
                                              n_iter_no_change_3=tuple([8, 9]),
                                              n_iter_no_change_4=tuple([10, 11]),
                                              alpha_0=tuple([0.01, 0.09]),
                                              alpha_1=tuple([0.09, 0.19]),
                                              alpha_2=tuple([0.19, 0.29]),
                                              alpha_3=tuple([0.29, 0.39]),
                                              alpha_4=tuple([0.39, 0.49]),
                                              alpha_5=tuple([0.49, 0.59]),
                                              alpha_6=tuple([0.59, 0.69]),
                                              alpha_7=tuple([0.69, 0.79]),
                                              alpha_8=tuple([0.79, 0.89]),
                                              alpha_9=tuple([0.89, 0.99]),
                                              ccp_alpha_0=tuple([0.0, 0.09]),
                                              ccp_alpha_1=tuple([0.09, 0.19]),
                                              ccp_alpha_2=tuple([0.19, 0.29]),
                                              ccp_alpha_3=tuple([0.29, 0.39]),
                                              ccp_alpha_4=tuple([0.39, 0.49]),
                                              ccp_alpha_5=tuple([0.49, 0.59]),
                                              ccp_alpha_6=tuple([0.59, 0.69]),
                                              ccp_alpha_7=tuple([0.69, 0.79]),
                                              ccp_alpha_8=tuple([0.79, 0.89]),
                                              ccp_alpha_9=tuple([0.89, 0.99]),
                                              validation_fraction_0=tuple([0.05, 0.1]),
                                              validation_fraction_1=tuple([0.1, 0.19]),
                                              validation_fraction_2=tuple([0.19, 0.29]),
                                              validation_fraction_3=tuple([0.29, 0.39]),
                                              validation_fraction_4=tuple([0.39, 0.49])
                                              ),
                                     knn=dict(n_neighbors_0=tuple([2, 3]),
                                              n_neighbors_1=tuple([4, 5]),
                                              n_neighbors_2=tuple([6, 7]),
                                              n_neighbors_3=tuple([8, 9]),
                                              n_neighbors_4=tuple([10, 11]),
                                              n_neighbors_5=tuple([12, 13]),
                                              weights=['uniform', 'distance'],
                                              algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'],
                                              p_0=tuple([1, 1]),
                                              p_1=tuple([2, 2]),
                                              p_2=tuple([3, 3]),
                                              leaf_size_0=tuple([15, 20]),
                                              leaf_size_1=tuple([25, 30]),
                                              leaf_size_2=tuple([35, 40]),
                                              leaf_size_3=tuple([45, 50]),
                                              leaf_size_4=tuple([55, 60]),
                                              leaf_size_5=tuple([65, 70]),
                                              leaf_size_6=tuple([75, 80]),
                                              leaf_size_7=tuple([85, 90]),
                                              leaf_size_8=tuple([95, 100])
                                              ),
                                     lasso=dict(alpha_0=tuple([0.01, 0.09]),
                                                alpha_1=tuple([0.09, 0.19]),
                                                alpha_2=tuple([0.19, 0.29]),
                                                alpha_3=tuple([0.29, 0.39]),
                                                alpha_4=tuple([0.39, 0.49]),
                                                alpha_5=tuple([0.49, 0.59]),
                                                alpha_6=tuple([0.59, 0.69]),
                                                alpha_7=tuple([0.69, 0.79]),
                                                alpha_8=tuple([0.79, 0.89]),
                                                alpha_9=tuple([0.89, 0.99]),
                                                normalize=[True, False],
                                                precompute=[True, False],
                                                fit_intercept=[True, False],
                                                selection=['cyclic', 'random'],
                                                max_iter_0=tuple([5, 9]),
                                                max_iter_1=tuple([10, 14]),
                                                max_iter_2=tuple([15, 19]),
                                                max_iter_3=tuple([20, 24]),
                                                max_iter_4=tuple([25, 29]),
                                                max_iter_5=tuple([30, 34]),
                                                max_iter_6=tuple([35, 39]),
                                                max_iter_7=tuple([40, 44]),
                                                max_iter_8=tuple([45, 49]),
                                                max_iter_9=tuple([50, 54]),
                                                max_iter_10=tuple([55, 59]),
                                                max_iter_11=tuple([60, 64]),
                                                max_iter_12=tuple([65, 69]),
                                                max_iter_13=tuple([70, 74]),
                                                max_iter_14=tuple([75, 79]),
                                                max_iter_15=tuple([80, 84]),
                                                max_iter_16=tuple([85, 89]),
                                                max_iter_17=tuple([90, 94]),
                                                max_iter_18=tuple([95, 99]),
                                                max_iter_19=tuple([100, 104]),
                                                max_iter_20=tuple([105, 109]),
                                                max_iter_21=tuple([110, 114]),
                                                max_iter_22=tuple([115, 119]),
                                                max_iter_23=tuple([120, 124]),
                                                max_iter_24=tuple([125, 129]),
                                                max_iter_25=tuple([130, 134]),
                                                max_iter_26=tuple([135, 139]),
                                                max_iter_27=tuple([140, 144]),
                                                max_iter_28=tuple([145, 149]),
                                                max_iter_29=tuple([150, 154]),
                                                max_iter_30=tuple([155, 159]),
                                                max_iter_31=tuple([160, 164]),
                                                max_iter_32=tuple([165, 169]),
                                                max_iter_33=tuple([170, 174]),
                                                max_iter_34=tuple([175, 179]),
                                                max_iter_35=tuple([180, 184]),
                                                max_iter_36=tuple([185, 189]),
                                                max_iter_37=tuple([190, 194]),
                                                max_iter_38=tuple([195, 199])
                                                ),
                                     rf=dict(n_estimators_0=tuple([5, 9]),
                                             n_estimators_1=tuple([10, 14]),
                                             n_estimators_2=tuple([15, 19]),
                                             n_estimators_3=tuple([20, 24]),
                                             n_estimators_4=tuple([25, 29]),
                                             n_estimators_5=tuple([30, 34]),
                                             n_estimators_6=tuple([35, 39]),
                                             n_estimators_7=tuple([40, 44]),
                                             n_estimators_8=tuple([45, 49]),
                                             n_estimators_9=tuple([50, 54]),
                                             n_estimators_10=tuple([55, 59]),
                                             n_estimators_11=tuple([60, 64]),
                                             n_estimators_12=tuple([65, 69]),
                                             n_estimators_13=tuple([70, 74]),
                                             n_estimators_14=tuple([75, 79]),
                                             n_estimators_15=tuple([80, 84]),
                                             n_estimators_16=tuple([85, 89]),
                                             n_estimators_17=tuple([90, 94]),
                                             n_estimators_18=tuple([95, 99]),
                                             n_estimators_19=tuple([100, 104]),
                                             n_estimators_20=tuple([105, 109]),
                                             n_estimators_21=tuple([110, 114]),
                                             n_estimators_22=tuple([115, 119]),
                                             n_estimators_23=tuple([120, 124]),
                                             n_estimators_24=tuple([125, 129]),
                                             n_estimators_25=tuple([130, 134]),
                                             n_estimators_26=tuple([135, 139]),
                                             n_estimators_27=tuple([140, 144]),
                                             n_estimators_28=tuple([145, 149]),
                                             n_estimators_29=tuple([150, 154]),
                                             n_estimators_30=tuple([155, 159]),
                                             n_estimators_31=tuple([160, 164]),
                                             n_estimators_32=tuple([165, 169]),
                                             n_estimators_33=tuple([170, 174]),
                                             n_estimators_34=tuple([175, 179]),
                                             n_estimators_35=tuple([180, 184]),
                                             n_estimators_36=tuple([185, 189]),
                                             n_estimators_37=tuple([190, 194]),
                                             n_estimators_38=tuple([195, 199]),
                                             criterion=['mse', 'mae'],
                                             bootstrap=[True, False],
                                             min_samples_split_0=tuple([2, 2]),
                                             min_samples_split_1=tuple([3, 3]),
                                             min_samples_split_2=tuple([4, 4]),
                                             min_samples_split_3=tuple([5, 5]),
                                             min_samples_split_4=tuple([6, 6]),
                                             min_samples_leaf_0=tuple([1, 1]),
                                             min_samples_leaf_1=tuple([2, 2]),
                                             min_samples_leaf_2=tuple([3, 3]),
                                             min_samples_leaf_3=tuple([4, 4]),
                                             min_samples_leaf_4=tuple([5, 5]),
                                             min_samples_leaf_5=tuple([6, 6]),
                                             max_depth_0=tuple([1, 2]),
                                             max_depth_1=tuple([3, 4]),
                                             max_depth_2=tuple([5, 6]),
                                             max_depth_3=tuple([7, 8]),
                                             max_depth_4=tuple([9, 10]),
                                             max_depth_5=tuple([11, 12])
                                             ),
                                     svm=dict(C_0=tuple([0.0001, 0.0009]),
                                              C_1=tuple([0.001, 0.009]),
                                              C_2=tuple([0.01, 0.09]),
                                              C_3=tuple([0.1, 0.2]),
                                              C_4=tuple([0.3, 0.4]),
                                              C_5=tuple([0.5, 0.6]),
                                              C_6=tuple([0.7, 0.8]),
                                              C_7=tuple([0.9, 1.0]),
                                              kernel=['rbf', 'linear', 'poly', 'sigmoid'],
                                              shrinking=[True, False],
                                              cache_size_0=tuple([100, 200]),
                                              cache_size_1=tuple([300, 400]),
                                              cache_size_2=tuple([500, 600]),
                                              cache_size_3=tuple([700, 800]),
                                              cache_size_4=tuple([900, 1000]),
                                              decision_function_shape=['ovo', 'ovr'],
                                              max_iter_0=tuple([5, 9]),
                                              max_iter_1=tuple([10, 14]),
                                              max_iter_2=tuple([15, 19]),
                                              max_iter_3=tuple([20, 24]),
                                              max_iter_4=tuple([25, 29]),
                                              max_iter_5=tuple([30, 34]),
                                              max_iter_6=tuple([35, 39]),
                                              max_iter_7=tuple([40, 44]),
                                              max_iter_8=tuple([45, 49]),
                                              max_iter_9=tuple([50, 54]),
                                              max_iter_10=tuple([55, 59]),
                                              max_iter_11=tuple([60, 64]),
                                              max_iter_12=tuple([65, 69]),
                                              max_iter_13=tuple([70, 74]),
                                              max_iter_14=tuple([75, 79]),
                                              max_iter_15=tuple([80, 84]),
                                              max_iter_16=tuple([85, 89]),
                                              max_iter_17=tuple([90, 94]),
                                              max_iter_18=tuple([95, 99]),
                                              max_iter_19=tuple([100, 104]),
                                              max_iter_20=tuple([105, 109]),
                                              max_iter_21=tuple([110, 114]),
                                              max_iter_22=tuple([115, 119]),
                                              max_iter_23=tuple([120, 124]),
                                              max_iter_24=tuple([125, 129]),
                                              max_iter_25=tuple([130, 134]),
                                              max_iter_26=tuple([135, 139]),
                                              max_iter_27=tuple([140, 144]),
                                              max_iter_28=tuple([145, 149]),
                                              max_iter_29=tuple([150, 154]),
                                              max_iter_30=tuple([155, 159]),
                                              max_iter_31=tuple([160, 164]),
                                              max_iter_32=tuple([165, 169]),
                                              max_iter_33=tuple([170, 174]),
                                              max_iter_34=tuple([175, 179]),
                                              max_iter_35=tuple([180, 184]),
                                              max_iter_36=tuple([185, 189]),
                                              max_iter_37=tuple([190, 194]),
                                              max_iter_38=tuple([195, 199])
                                              ),
                                     lsvm=dict(C_0=tuple([0.0001, 0.0009]),
                                               C_1=tuple([0.001, 0.009]),
                                               C_2=tuple([0.01, 0.09]),
                                               C_3=tuple([0.1, 0.2]),
                                               C_4=tuple([0.3, 0.4]),
                                               C_5=tuple([0.5, 0.6]),
                                               C_6=tuple([0.7, 0.8]),
                                               C_7=tuple([0.9, 1.0]),
                                               max_iter_0=tuple([5, 9]),
                                               max_iter_1=tuple([10, 14]),
                                               max_iter_2=tuple([15, 19]),
                                               max_iter_3=tuple([20, 24]),
                                               max_iter_4=tuple([25, 29]),
                                               max_iter_5=tuple([30, 34]),
                                               max_iter_6=tuple([35, 39]),
                                               max_iter_7=tuple([40, 44]),
                                               max_iter_8=tuple([45, 49]),
                                               max_iter_9=tuple([50, 54]),
                                               max_iter_10=tuple([55, 59]),
                                               max_iter_11=tuple([60, 64]),
                                               max_iter_12=tuple([65, 69]),
                                               max_iter_13=tuple([70, 74]),
                                               max_iter_14=tuple([75, 79]),
                                               max_iter_15=tuple([80, 84]),
                                               max_iter_16=tuple([85, 89]),
                                               max_iter_17=tuple([90, 94]),
                                               max_iter_18=tuple([95, 99]),
                                               max_iter_19=tuple([100, 104]),
                                               max_iter_20=tuple([105, 109]),
                                               max_iter_21=tuple([110, 114]),
                                               max_iter_22=tuple([115, 119]),
                                               max_iter_23=tuple([120, 124]),
                                               max_iter_24=tuple([125, 129]),
                                               max_iter_25=tuple([130, 134]),
                                               max_iter_26=tuple([135, 139]),
                                               max_iter_27=tuple([140, 144]),
                                               max_iter_28=tuple([145, 149]),
                                               max_iter_29=tuple([150, 154]),
                                               max_iter_30=tuple([155, 159]),
                                               max_iter_31=tuple([160, 164]),
                                               max_iter_32=tuple([165, 169]),
                                               max_iter_33=tuple([170, 174]),
                                               max_iter_34=tuple([175, 179]),
                                               max_iter_35=tuple([180, 184]),
                                               max_iter_36=tuple([185, 189]),
                                               max_iter_37=tuple([190, 194]),
                                               max_iter_38=tuple([195, 199]),
                                               penalty=['l1', 'l2'],
                                               loss=['hinge', 'squared_hinge'],
                                               multi_class=['ovr', 'crammer_singer']
                                               ),
                                     nusvm=dict(C_0=tuple([0.0001, 0.0009]),
                                                C_1=tuple([0.001, 0.009]),
                                                C_2=tuple([0.01, 0.09]),
                                                C_3=tuple([0.1, 0.2]),
                                                C_4=tuple([0.3, 0.4]),
                                                C_5=tuple([0.5, 0.6]),
                                                C_6=tuple([0.7, 0.8]),
                                                C_7=tuple([0.9, 1.0]),
                                                max_iter_0=tuple([5, 9]),
                                                max_iter_1=tuple([10, 14]),
                                                max_iter_2=tuple([15, 19]),
                                                max_iter_3=tuple([20, 24]),
                                                max_iter_4=tuple([25, 29]),
                                                max_iter_5=tuple([30, 34]),
                                                max_iter_6=tuple([35, 39]),
                                                max_iter_7=tuple([40, 44]),
                                                max_iter_8=tuple([45, 49]),
                                                max_iter_9=tuple([50, 54]),
                                                max_iter_10=tuple([55, 59]),
                                                max_iter_11=tuple([60, 64]),
                                                max_iter_12=tuple([65, 69]),
                                                max_iter_13=tuple([70, 74]),
                                                max_iter_14=tuple([75, 79]),
                                                max_iter_15=tuple([80, 84]),
                                                max_iter_16=tuple([85, 89]),
                                                max_iter_17=tuple([90, 94]),
                                                max_iter_18=tuple([95, 99]),
                                                max_iter_19=tuple([100, 104]),
                                                max_iter_20=tuple([105, 109]),
                                                max_iter_21=tuple([110, 114]),
                                                max_iter_22=tuple([115, 119]),
                                                max_iter_23=tuple([120, 124]),
                                                max_iter_24=tuple([125, 129]),
                                                max_iter_25=tuple([130, 134]),
                                                max_iter_26=tuple([135, 139]),
                                                max_iter_27=tuple([140, 144]),
                                                max_iter_28=tuple([145, 149]),
                                                max_iter_29=tuple([150, 154]),
                                                max_iter_30=tuple([155, 159]),
                                                max_iter_31=tuple([160, 164]),
                                                max_iter_32=tuple([165, 169]),
                                                max_iter_33=tuple([170, 174]),
                                                max_iter_34=tuple([175, 179]),
                                                max_iter_35=tuple([180, 184]),
                                                max_iter_36=tuple([185, 189]),
                                                max_iter_37=tuple([190, 194]),
                                                max_iter_38=tuple([195, 199]),
                                                decision_function_shape=['ovo', 'ovr'],
                                                kernel=['rbf', 'linear', 'poly', 'sigmoid'],
                                                shrinking=[True, False],
                                                cache_size_0=tuple([100, 200]),
                                                cache_size_1=tuple([300, 400]),
                                                cache_size_2=tuple([500, 600]),
                                                cache_size_3=tuple([700, 800]),
                                                cache_size_4=tuple([900, 1000]),
                                                nu_0=tuple([0.01, 0.09]),
                                                nu_1=tuple([0.09, 0.19]),
                                                nu_2=tuple([0.19, 0.29]),
                                                nu_3=tuple([0.29, 0.39]),
                                                nu_4=tuple([0.39, 0.49]),
                                                nu_5=tuple([0.49, 0.59]),
                                                nu_6=tuple([0.59, 0.69]),
                                                nu_7=tuple([0.69, 0.79]),
                                                nu_8=tuple([0.79, 0.89]),
                                                nu_9=tuple([0.89, 0.99])
                                                ),
                                     xgb=dict(n_estimators_0=tuple([5, 9]),
                                              n_estimators_1=tuple([10, 14]),
                                              n_estimators_2=tuple([15, 19]),
                                              n_estimators_3=tuple([20, 24]),
                                              n_estimators_4=tuple([25, 29]),
                                              n_estimators_5=tuple([30, 34]),
                                              n_estimators_6=tuple([35, 39]),
                                              n_estimators_7=tuple([40, 44]),
                                              n_estimators_8=tuple([45, 49]),
                                              n_estimators_9=tuple([50, 54]),
                                              n_estimators_10=tuple([55, 59]),
                                              n_estimators_11=tuple([60, 64]),
                                              n_estimators_12=tuple([65, 69]),
                                              n_estimators_13=tuple([70, 74]),
                                              n_estimators_14=tuple([75, 79]),
                                              n_estimators_15=tuple([80, 84]),
                                              n_estimators_16=tuple([85, 89]),
                                              n_estimators_17=tuple([90, 94]),
                                              n_estimators_18=tuple([95, 99]),
                                              n_estimators_19=tuple([100, 104]),
                                              n_estimators_20=tuple([105, 109]),
                                              n_estimators_21=tuple([110, 114]),
                                              n_estimators_22=tuple([115, 119]),
                                              n_estimators_23=tuple([120, 124]),
                                              n_estimators_24=tuple([125, 129]),
                                              n_estimators_25=tuple([130, 134]),
                                              n_estimators_26=tuple([135, 139]),
                                              n_estimators_27=tuple([140, 144]),
                                              n_estimators_28=tuple([145, 149]),
                                              n_estimators_29=tuple([150, 154]),
                                              n_estimators_30=tuple([155, 159]),
                                              n_estimators_31=tuple([160, 164]),
                                              n_estimators_32=tuple([165, 169]),
                                              n_estimators_33=tuple([170, 174]),
                                              n_estimators_34=tuple([175, 179]),
                                              n_estimators_35=tuple([180, 184]),
                                              n_estimators_36=tuple([185, 189]),
                                              n_estimators_37=tuple([190, 194]),
                                              n_estimators_38=tuple([195, 199]),
                                              learning_rate_0=tuple([0.01, 0.09]),
                                              learning_rate_1=tuple([0.09, 0.19]),
                                              learning_rate_2=tuple([0.19, 0.29]),
                                              learning_rate_3=tuple([0.29, 0.39]),
                                              learning_rate_4=tuple([0.39, 0.49]),
                                              learning_rate_5=tuple([0.49, 0.59]),
                                              learning_rate_6=tuple([0.59, 0.69]),
                                              learning_rate_7=tuple([0.69, 0.79]),
                                              learning_rate_8=tuple([0.79, 0.89]),
                                              learning_rate_9=tuple([0.89, 0.99]),
                                              early_stopping=[True, False],
                                              max_depth_0=tuple([3, 4]),
                                              max_depth_1=tuple([5, 6]),
                                              max_depth_2=tuple([7, 8]),
                                              max_depth_3=tuple([9, 10]),
                                              max_depth_4=tuple([11, 12]),
                                              max_depth_5=tuple([13, 14]),
                                              max_depth_6=tuple([15, 16]),
                                              min_samples_leaf_0=tuple([1, 2]),
                                              min_samples_leaf_1=tuple([3, 4]),
                                              min_samples_leaf_2=tuple([5, 6]),
                                              min_samples_leaf_3=tuple([7, 8]),
                                              min_samples_leaf_4=tuple([9, 10]),
                                              min_samples_split_0=tuple([1, 2]),
                                              min_samples_split_1=tuple([3, 4]),
                                              min_samples_split_2=tuple([5, 6]),
                                              min_samples_split_3=tuple([7, 8]),
                                              min_samples_split_4=tuple([9, 10]),
                                              min_child_weight_0=tuple([1, 2]),
                                              min_child_weight_1=tuple([3, 4]),
                                              min_child_weight_2=tuple([5, 6]),
                                              min_child_weight_3=tuple([7, 8]),
                                              min_child_weight_4=tuple([9, 10]),
                                              reg_alpha_0=tuple([0.01, 0.09]),
                                              reg_alpha_1=tuple([0.09, 0.19]),
                                              reg_alpha_2=tuple([0.19, 0.29]),
                                              reg_alpha_3=tuple([0.29, 0.39]),
                                              reg_alpha_4=tuple([0.39, 0.49]),
                                              reg_alpha_5=tuple([0.49, 0.59]),
                                              reg_alpha_6=tuple([0.59, 0.69]),
                                              reg_alpha_7=tuple([0.69, 0.79]),
                                              reg_alpha_8=tuple([0.79, 0.89]),
                                              reg_alpha_9=tuple([0.89, 0.99]),
                                              reg_lambda_0=tuple([0.01, 0.09]),
                                              reg_lambda_1=tuple([0.09, 0.19]),
                                              reg_lambda_2=tuple([0.19, 0.29]),
                                              reg_lambda_3=tuple([0.29, 0.39]),
                                              reg_lambda_4=tuple([0.39, 0.49]),
                                              reg_lambda_5=tuple([0.49, 0.59]),
                                              reg_lambda_6=tuple([0.59, 0.69]),
                                              reg_lambda_7=tuple([0.69, 0.79]),
                                              reg_lambda_8=tuple([0.79, 0.89]),
                                              reg_lambda_9=tuple([0.89, 0.99]),
                                              gamma_0=tuple([0.01, 0.09]),
                                              gamma_1=tuple([0.09, 0.19]),
                                              gamma_2=tuple([0.19, 0.29]),
                                              gamma_3=tuple([0.29, 0.39]),
                                              gamma_4=tuple([0.39, 0.49]),
                                              gamma_5=tuple([0.49, 0.59]),
                                              gamma_6=tuple([0.59, 0.69]),
                                              gamma_7=tuple([0.69, 0.79]),
                                              gamma_8=tuple([0.79, 0.89]),
                                              gamma_9=tuple([0.89, 0.99]),
                                              subsample_0=tuple([0.01, 0.09]),
                                              subsample_1=tuple([0.09, 0.19]),
                                              subsample_2=tuple([0.19, 0.29]),
                                              subsample_3=tuple([0.29, 0.39]),
                                              subsample_4=tuple([0.39, 0.49]),
                                              subsample_5=tuple([0.49, 0.59]),
                                              subsample_6=tuple([0.59, 0.69]),
                                              subsample_7=tuple([0.69, 0.79]),
                                              subsample_8=tuple([0.79, 0.89]),
                                              subsample_9=tuple([0.89, 0.99]),
                                              colsample_bytree_5=tuple([0.49, 0.59]),
                                              colsample_bytree_6=tuple([0.59, 0.69]),
                                              colsample_bytree_7=tuple([0.69, 0.79]),
                                              colsample_bytree_8=tuple([0.79, 0.89]),
                                              colsample_bytree_9=tuple([0.89, 0.99])
                                              )
                                     )


class SupervisedMLException(Exception):
    """
    Class for handling exceptions for class Classification and Regression
    """
    pass


class Classification:
    """
    Class for handling classification algorithms
    """
    def __init__(self,
                 clf_params: dict = None,
                 cpu_cores: int = 0,
                 seed: int = 1234,
                 **kwargs
                 ):
        """
        :param clf_params: dict
            Pre-configured classification model parameter

        :param cpu_cores: int
            Number of CPU core to use

        :param seed: int
            Seed

        :param kwargs: dict
            Key-word arguments
        """
        self.clf_params: dict = {} if clf_params is None else clf_params
        self.seed: int = 1234 if seed <= 0 else seed
        if cpu_cores <= 0:
            self.cpu_cores: int = os.cpu_count() - 1 if os.cpu_count() > 1 else os.cpu_count()
        else:
            if cpu_cores <= os.cpu_count():
                self.cpu_cores: int = cpu_cores
            else:
                self.cpu_cores: int = os.cpu_count() - 1 if os.cpu_count() > 1 else os.cpu_count()
        self.kwargs: dict = kwargs

    def ada_boosting(self) -> AdaBoostClassifier:
        """
        Config Ada Boosting algorithm

        :return AdaBoostClassifier:
            Model object
        """
        #_base_estimator = None
        #if self.clf_params.get('base_estimator') is not None:
        #    if self.clf_params.get('base_estimator') == 'base_decision_tree':
        #        _base_estimator = BaseDecisionTree
        #    elif self.clf_params.get('base_estimator') == 'decision_tree_classifier':
        #        _base_estimator = DecisionTreeClassifier
        #    elif self.clf_params.get('base_estimator') == 'extra_tree_classifier':
        #        _base_estimator = ExtraTreeClassifier
        return AdaBoostClassifier(base_estimator=None,
                                  n_estimators=50 if self.clf_params.get('n_estimators') is None else self.clf_params.get('n_estimators'),
                                  learning_rate=1.0 if self.clf_params.get('learning_rate') is None else self.clf_params.get('learning_rate'),
                                  algorithm='SAMME.R' if self.clf_params.get('algorithm') is None else self.clf_params.get('algorithm'),
                                  random_state=self.seed
                                  )

    def ada_boosting_param(self) -> dict:
        """
        Generate Ada Boosting classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(#base_estimator=np.random.choice(a=[None, 'base_decision_tree', 'decision_tree_classifier', 'extra_tree_classifier']),
                    n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                                   high=500 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                                   ),
                    learning_rate=np.random.uniform(low=0.01 if self.kwargs.get('learning_rate_low') is None else self.kwargs.get('learning_rate_low'),
                                                    high=1.0 if self.kwargs.get('learning_rate_high') is None else self.kwargs.get('learning_rate_high')
                                                    )
                    #algorithm=np.random.choice(a=['SAMME', 'SAMME.R'])
                    )

    def cat_boost(self) -> CatBoostClassifier:
        """
        Config CatBoost Classifier

        :return: CatBoostClassifier
        """
        return CatBoostClassifier(n_estimators=100 if self.clf_params.get('n_estimators') is None else self.clf_params.get('n_estimators'),
                                  learning_rate=0.03 if self.clf_params.get('learning_rate') is None else self.clf_params.get('learning_rate'),
                                  depth=self.clf_params.get('depth'),
                                  l2_leaf_reg=self.clf_params.get('l2_leaf_reg'),
                                  model_size_reg=self.clf_params.get('model_size_reg'),
                                  rsm=self.clf_params.get('rsm'),
                                  loss_function=self.clf_params.get('loss_function'),
                                  border_count=self.clf_params.get('border_count'),
                                  feature_border_type=self.clf_params.get('feature_border_type'),
                                  per_float_feature_quantization=self.clf_params.get('per_float_feature_quantization'),
                                  input_borders=self.clf_params.get('input_borders'),
                                  output_borders=self.clf_params.get('output_borders'),
                                  fold_permutation_block=self.clf_params.get('fold_permutation_block'),
                                  od_pval=self.clf_params.get('od_pval'),
                                  od_wait=self.clf_params.get('od_wait'),
                                  od_type=self.clf_params.get('od_type'),
                                  nan_mode=self.clf_params.get('nan_mode'),
                                  counter_calc_method=self.clf_params.get('counter_calc_method'),
                                  leaf_estimation_iterations=self.clf_params.get('leaf_estimation_iterations'),
                                  leaf_estimation_method=self.clf_params.get('leaf_estimation_method'),
                                  thread_count=self.clf_params.get('thread_count'),
                                  random_seed=self.clf_params.get('random_seed'),
                                  use_best_model=self.clf_params.get('use_best_model'),
                                  best_model_min_trees=self.clf_params.get('best_model_min_trees'),
                                  verbose=self.clf_params.get('verbose'),
                                  silent=self.clf_params.get('silent'),
                                  logging_level=self.clf_params.get('logging_level'),
                                  metric_period=self.clf_params.get('metric_period'),
                                  ctr_leaf_count_limit=self.clf_params.get('ctr_leaf_count_limit'),
                                  store_all_simple_ctr=self.clf_params.get('store_all_simple_ctr'),
                                  max_ctr_complexity=self.clf_params.get('max_ctr_complexity'),
                                  has_time=self.clf_params.get('has_time'),
                                  allow_const_label=self.clf_params.get('allow_const_label'),
                                  target_border=self.clf_params.get('target_border'),
                                  classes_count=self.clf_params.get('classes_count'),
                                  class_weights=self.clf_params.get('class_weights'),
                                  auto_class_weights=self.clf_params.get('auto_class_weights'),
                                  class_names=self.clf_params.get('class_names'),
                                  one_hot_max_size=self.clf_params.get('one_hot_max_size'),
                                  random_strength=self.clf_params.get('random_strength'),
                                  name=self.clf_params.get('name'),
                                  ignored_features=self.clf_params.get('ignored_features'),
                                  train_dir=self.clf_params.get('train_dir'),
                                  custom_loss=self.clf_params.get('custom_loss'),
                                  custom_metric=self.clf_params.get('custom_metric'),
                                  eval_metric=self.clf_params.get('eval_metric'),
                                  bagging_temperature=self.clf_params.get('bagging_temperature'),
                                  save_snapshot=self.clf_params.get('save_snapshot'),
                                  snapshot_file=self.clf_params.get('snapshot_file'),
                                  snapshot_interval=self.clf_params.get('snapshot_interval'),
                                  fold_len_multiplier=self.clf_params.get('fold_len_multiplier'),
                                  used_ram_limit=self.clf_params.get('used_ram_limit'),
                                  gpu_ram_part=self.clf_params.get('gpu_ram_part'),
                                  pinned_memory_size=self.clf_params.get('pinned_memory_size'),
                                  allow_writing_files=self.clf_params.get('allow_writing_files'),
                                  final_ctr_computation_mode=self.clf_params.get('final_ctr_computation_mode'),
                                  approx_on_full_history=self.clf_params.get('approx_on_full_history'),
                                  boosting_type=self.clf_params.get('boosting_type'),
                                  simple_ctr=self.clf_params.get('simple_ctr'),
                                  combinations_ctr=self.clf_params.get('combinations_ctr'),
                                  per_feature_ctr=self.clf_params.get('per_feature_ctr'),
                                  ctr_description=self.clf_params.get('ctr_description'),
                                  ctr_target_border_count=self.clf_params.get('ctr_target_border_count'),
                                  task_type=self.clf_params.get('task_type'),
                                  device_config=self.clf_params.get('device_config'),
                                  devices=self.clf_params.get('devices'),
                                  bootstrap_type=self.clf_params.get('bootstrap_type'),
                                  subsample=self.clf_params.get('subsample'),
                                  mvs_reg=self.clf_params.get('mvs_reg'),
                                  sampling_unit=self.clf_params.get('sampling_unit'),
                                  sampling_frequency=self.clf_params.get('sampling_frequency'),
                                  dev_score_calc_obj_block_size=self.clf_params.get('dev_score_calc_obj_block_size'),
                                  dev_efb_max_buckets=self.clf_params.get('dev_efb_max_buckets'),
                                  sparse_features_conflict_fraction=self.clf_params.get('sparse_features_conflict_fraction'),
                                  #max_depth=self.clf_params.get('max_depth'),
                                  num_boost_round=self.clf_params.get('num_boost_round'),
                                  num_trees=self.clf_params.get('num_trees'),
                                  colsample_bylevel=self.clf_params.get('colsample_bylevel'),
                                  random_state=self.clf_params.get('random_state'),
                                  #reg_lambda=self.clf_params.get('reg_lambda'),
                                  objective=self.clf_params.get('objective'),
                                  eta=self.clf_params.get('eta'),
                                  max_bin=self.clf_params.get('max_bin'),
                                  scale_pos_weight=self.clf_params.get('scale_pos_weight'),
                                  gpu_cat_features_storage=self.clf_params.get('gpu_cat_features_storage'),
                                  data_partition=self.clf_params.get('data_partition'),
                                  metadata=self.clf_params.get('metadata'),
                                  early_stopping_rounds=self.clf_params.get('early_stopping_rounds'),
                                  cat_features=self.clf_params.get('cat_features'),
                                  grow_policy=self.clf_params.get('grow_policy'),
                                  min_data_in_leaf=self.clf_params.get('min_data_in_leaf'),
                                  min_child_samples=self.clf_params.get('min_child_samples'),
                                  max_leaves=self.clf_params.get('max_leaves'),
                                  num_leaves=self.clf_params.get('num_leaves'),
                                  score_function=self.clf_params.get('score_function'),
                                  leaf_estimation_backtracking=self.clf_params.get('leaf_estimation_backtracking'),
                                  ctr_history_unit=self.clf_params.get('ctr_history_unit'),
                                  monotone_constraints=self.clf_params.get('monotone_constraints'),
                                  feature_weights=self.clf_params.get('feature_weights'),
                                  penalties_coefficient=self.clf_params.get('penalties_coefficient'),
                                  first_feature_use_penalties=self.clf_params.get('first_feature_use_penalties'),
                                  per_object_feature_penalties=self.clf_params.get('per_object_feature_penalties'),
                                  model_shrink_rate=self.clf_params.get('model_shrink_rate'),
                                  model_shrink_mode=self.clf_params.get('model_shrink_mode'),
                                  langevin=self.clf_params.get('langevin'),
                                  diffusion_temperature=self.clf_params.get('diffusion_temperature'),
                                  posterior_sampling=self.clf_params.get('posterior_sampling'),
                                  boost_from_average=self.clf_params.get('boost_from_average'),
                                  text_features=self.clf_params.get('text_features'),
                                  tokenizers=self.clf_params.get('tokenizers'),
                                  dictionaries=self.clf_params.get('dictionaries'),
                                  feature_calcers=self.clf_params.get('feature_calcers'),
                                  text_processing=self.clf_params.get('text_processing'),
                                  embedding_features=self.clf_params.get('embedding_features')
                                  )

    def cat_boost_param(self) -> dict:
        """
        Generate Cat Boosting classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                                   high=100 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                                   ),
                    learning_rate=np.random.uniform(low=0.0001 if self.kwargs.get('learning_rate_low') is None else self.kwargs.get('learning_rate_low'),
                                                    high=0.5 if self.kwargs.get('learning_rate_high') is None else self.kwargs.get('learning_rate_high')
                                                    ),
                    l2_leaf_reg=np.random.uniform(low=0.1 if self.kwargs.get('l2_leaf_reg_low') is None else self.kwargs.get('l2_leaf_reg_low'),
                                                  high=1.0 if self.kwargs.get('l2_leaf_reg_high') is None else self.kwargs.get('l2_leaf_reg_high')
                                                  ),
                    depth=np.random.randint(low=3 if self.kwargs.get('depth_low') is None else self.kwargs.get('depth_low'),
                                            high=16 if self.kwargs.get('depth_high') is None else self.kwargs.get('depth_high')
                                            ),
                    #sampling_frequency=np.random.choice(a=['PerTree', 'PerTreeLevel']),
                    #sampling_unit=np.random.choice(a=['Object', 'Group']),
                    grow_policy=np.random.choice(a=['SymmetricTree', 'Depthwise', 'Lossguide'] if self.kwargs.get('grow_policy_choice') is None else self.kwargs.get('grow_policy_choice'),),
                    min_data_in_leaf=np.random.randint(low=1 if self.kwargs.get('min_data_in_leaf_low') is None else self.kwargs.get('min_data_in_leaf_low'),
                                                       high=20 if self.kwargs.get('min_data_in_leaf_high') is None else self.kwargs.get('min_data_in_leaf_high')
                                                       ),
                    #max_leaves=np.random.randint(low=10, high=64),
                    rsm=np.random.uniform(low=0.1 if self.kwargs.get('rsm_low') is None else self.kwargs.get('rsm_low'),
                                          high=1 if self.kwargs.get('rsm_high') is None else self.kwargs.get('rsm_high')
                                          ),
                    #fold_len_multiplier=np.random.randint(low=2, high=4),
                    #approx_on_full_history=np.random.choice(a=[False, True]),
                    auto_class_weights=np.random.choice(a=[None, 'Balanced', 'SqrtBalanced'] if self.kwargs.get('auto_class_weights_choice') is None else self.kwargs.get('auto_class_weights_choice')),
                    #boosting_type=np.random.choice(a=['Ordered', 'Plain']),
                    #score_function=np.random.choice(a=['Cosine', 'L2', 'NewtonCosine', 'NewtonL2']),
                    #model_shrink_mode=np.random.choice(a=['Constant', 'Decreasing']),
                    #border_count=np.random.randint(low=1, high=65535),
                    feature_border_type=np.random.choice(a=['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum', 'MinEntropy', 'GreedyLogSum'] if self.kwargs.get('feature_border_type_choice') is None else self.kwargs.get('feature_border_type_choice'))
                    )

    def extreme_gradient_boosting_tree(self) -> XGBClassifier:
        """
        Training of the Extreme Gradient Boosting Classifier

        :return XGBClassifier:
            Model object
        """
        return XGBClassifier(max_depth=3 if self.clf_params.get('max_depth') is None else self.clf_params.get('max_depth'),
                             learning_rate=0.1 if self.clf_params.get('learning_rate') is None else self.clf_params.get('learning_rate'),
                             n_estimators=100 if self.clf_params.get('n_estimators') is None else self.clf_params.get('n_estimators'),
                             verbosity=0 if self.clf_params.get('verbosity') is None else self.clf_params.get('verbosity'),
                             objective='binary:logistic' if self.clf_params.get('objective') is None else self.clf_params.get('objective'),
                             booster='gbtree' if self.clf_params.get('booster') is None else self.clf_params.get('booster'),
                             n_jobs=self.cpu_cores,
                             gamma=0 if self.clf_params.get('gamma') is None else self.clf_params.get('gamma'),
                             min_child_weight=1 if self.clf_params.get('min_child_weight') is None else self.clf_params.get('min_child_weight'),
                             max_delta_step=0 if self.clf_params.get('max_delta_step') is None else self.clf_params.get('max_delta_step'),
                             subsample=1 if self.clf_params.get('subsample') is None else self.clf_params.get('subsample'),
                             colsample_bytree=1 if self.clf_params.get('colsample_bytree') is None else self.clf_params.get('colsample_bytree'),
                             colsample_bylevel=1 if self.clf_params.get('colsample_bylevel') is None else self.clf_params.get('colsample_bylevel'),
                             colsample_bynode=1 if self.clf_params.get('colsample_bynode') is None else self.clf_params.get('colsample_bynode'),
                             reg_alpha=0 if self.clf_params.get('reg_alpha') is None else self.clf_params.get('reg_alpha'),
                             reg_lambda=1 if self.clf_params.get('reg_lambda') is None else self.clf_params.get('reg_lambda'),
                             scale_pos_weight=1.0 if self.clf_params.get('scale_pos_weight') is None else self.clf_params.get('scale_pos_weight'),
                             base_score=0.5 if self.clf_params.get('base_score') is None else self.clf_params.get('base_score'),
                             random_state=self.seed
                             )

    def extreme_gradient_boosting_tree_param(self) -> dict:
        """
        Generate Extreme Gradient Boosting Decision Tree classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(learning_rate=np.random.uniform(low=0.0001 if self.kwargs.get('learning_rate_low') is None else self.kwargs.get('learning_rate_low'),
                                                    high=0.5 if self.kwargs.get('learning_rate_high') is None else self.kwargs.get('learning_rate_high')
                                                    ),
                    n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                                   high=100 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                                   ),
                    min_samples_split=np.random.randint(low=2 if self.kwargs.get('min_samples_split_low') is None else self.kwargs.get('min_samples_split_low'),
                                                        high=6 if self.kwargs.get('min_samples_split_high') is None else self.kwargs.get('min_samples_split_high')
                                                        ),
                    min_samples_leaf=np.random.randint(low=1 if self.kwargs.get('min_samples_leaf_low') is None else self.kwargs.get('min_samples_leaf_low'),
                                                       high=6 if self.kwargs.get('min_samples_leaf_high') is None else self.kwargs.get('min_samples_leaf_high')
                                                       ),
                    max_depth=np.random.randint(low=3 if self.kwargs.get('max_depth_low') is None else self.kwargs.get('max_depth_low'),
                                                high=12 if self.kwargs.get('max_depth_high') is None else self.kwargs.get('max_depth_high')
                                                ),
                    #booster=np.random.choice(a=['gbtree', 'gblinear']),
                    gamma=np.random.uniform(low=0.01 if self.kwargs.get('gamma_low') is None else self.kwargs.get('gamma_low'),
                                            high=0.99 if self.kwargs.get('gamma_high') is None else self.kwargs.get('gamma_high')
                                            ),
                    min_child_weight=np.random.randint(low=1 if self.kwargs.get('min_child_weight_low') is None else self.kwargs.get('min_child_weight_low'),
                                                       high=12 if self.kwargs.get('min_child_weight_high') is None else self.kwargs.get('min_child_weight_high')
                                                       ),
                    reg_alpha=np.random.uniform(low=0.0 if self.kwargs.get('reg_alpha_low') is None else self.kwargs.get('reg_alpha_low'),
                                                high=0.9 if self.kwargs.get('reg_alpha_high') is None else self.kwargs.get('reg_alpha_high')
                                                ),
                    reg_lambda=np.random.uniform(low=0.1 if self.kwargs.get('reg_lambda_low') is None else self.kwargs.get('reg_lambda_low'),
                                                 high=1.0 if self.kwargs.get('reg_lambda_high') is None else self.kwargs.get('reg_lambda_high')
                                                 ),
                    subsample=np.random.uniform(low=0.0 if self.kwargs.get('subsample_low') is None else self.kwargs.get('subsample_low'),
                                                high=1.0 if self.kwargs.get('subsample_high') is None else self.kwargs.get('subsample_high')
                                                ),
                    colsample_bytree=np.random.uniform(low=0.5 if self.kwargs.get('colsample_bytree_low') is None else self.kwargs.get('colsample_bytree_low'),
                                                       high=0.99 if self.kwargs.get('colsample_bytree_high') is None else self.kwargs.get('colsample_bytree_high')
                                                       ),
                    #scale_pos_weight=np.random.uniform(low=0.01, high=1.0),
                    #base_score=np.random.uniform(low=0.01, high=0.99)
                    )

    def gradient_boosting_tree(self) -> GradientBoostingClassifier:
        """
        Config gradient boosting decision tree classifier

        :return GradientBoostingClassifier:
            Model object
        """
        return GradientBoostingClassifier(loss='deviance' if self.clf_params.get('loss') is None else self.clf_params.get('loss'),
                                          learning_rate=0.1 if self.clf_params.get('learning_rate') is None else self.clf_params.get('learning_rate'),
                                          n_estimators=100 if self.clf_params.get('n_estimators') is None else self.clf_params.get('n_estimators'),
                                          subsample=1.0 if self.clf_params.get('subsample') is None else self.clf_params.get('subsample'),
                                          criterion='friedman_mse' if self.clf_params.get('criterion') is None else self.clf_params.get('criterion'),
                                          min_samples_split=2 if self.clf_params.get('min_samples_split') is None else self.clf_params.get('min_samples_split'),
                                          min_samples_leaf=1 if self.clf_params.get('min_samples_leaf') is None else self.clf_params.get('min_samples_leaf'),
                                          min_weight_fraction_leaf=0 if self.clf_params.get('min_weight_fraction_leaf') is None else self.clf_params.get('min_weight_fraction_leaf'),
                                          max_depth=3 if self.clf_params.get('max_depth') is None else self.clf_params.get('max_depth'),
                                          min_impurity_decrease=0 if self.clf_params.get('min_impurity_decrease') is None else self.clf_params.get('min_impurity_decrease'),
                                          max_leaf_nodes=self.clf_params.get('max_leaf_nodes'),
                                          init=self.clf_params.get('init'),
                                          random_state=self.seed,
                                          max_features=self.clf_params.get('max_features'),
                                          verbose=0,
                                          warm_start=False if self.clf_params.get('warm_start') is None else self.clf_params.get('warm_start'),
                                          validation_fraction=0.1 if self.clf_params.get('validation_fraction') is None else self.clf_params.get('validation_fraction'),
                                          n_iter_no_change=self.clf_params.get('n_iter_no_change'),
                                          tol=0.0001 if self.clf_params.get('tol') is None else self.clf_params.get('tol'),
                                          ccp_alpha=0.0 if self.clf_params.get('ccp_alpha') is None else self.clf_params.get('ccp_alpha')
                                          )

    def gradient_boosting_tree_param(self) -> dict:
        """
        Generate Gradient Boosting Tree classifier parameter randomly

        :return: dict
            Parameter config
        """
        return dict(learning_rate=np.random.uniform(low=0.0001 if self.kwargs.get('learning_rate_low') is None else self.kwargs.get('learning_rate_low'),
                                                    high=0.5 if self.kwargs.get('learning_rate_high') is None else self.kwargs.get('learning_rate_high')
                                                    ),
                    loss=np.random.choice(a=['deviance', 'exponential'] if self.kwargs.get('loss_choice') is None else self.kwargs.get('loss_choice')),
                    n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                                   high=100 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                                   ),
                    subsample=np.random.uniform(low=0.0 if self.kwargs.get('subsample_low') is None else self.kwargs.get('subsample_low'),
                                                high=1.0 if self.kwargs.get('subsample_high') is None else self.kwargs.get('subsample_high')
                                                ),
                    criterion=np.random.choice(a=['friedman_mse', 'mse', 'mae'] if self.kwargs.get('criterion_choice') is None else self.kwargs.get('criterion_choice')),
                    min_samples_split=np.random.randint(low=2 if self.kwargs.get('min_samples_split_low') is None else self.kwargs.get('min_samples_split_low'),
                                                        high=6 if self.kwargs.get('min_samples_split_high') is None else self.kwargs.get('min_samples_split_high')
                                                        ),
                    min_samples_leaf=np.random.randint(low=1 if self.kwargs.get('min_samples_leaf_low') is None else self.kwargs.get('min_samples_leaf_low'),
                                                       high=6 if self.kwargs.get('min_samples_leaf_high') is None else self.kwargs.get('min_samples_leaf_high')
                                                       ),
                    max_depth=np.random.randint(low=3 if self.kwargs.get('max_depth_low') is None else self.kwargs.get('max_depth_low'),
                                                high=12 if self.kwargs.get('max_depth_high') is None else self.kwargs.get('max_depth_high')
                                                ),
                    validation_fraction=np.random.uniform(low=0.05 if self.kwargs.get('validation_fraction_low') is None else self.kwargs.get('validation_fraction_low'),
                                                          high=0.4 if self.kwargs.get('validation_fraction_high') is None else self.kwargs.get('validation_fraction_high')
                                                          ),
                    n_iter_no_change=np.random.randint(low=2 if self.kwargs.get('n_iter_no_change_low') is None else self.kwargs.get('n_iter_no_change_low'),
                                                       high=10 if self.kwargs.get('n_iter_no_change_high') is None else self.kwargs.get('n_iter_no_change_high')
                                                       ),
                    ccp_alpha=np.random.uniform(low=0.0 if self.kwargs.get('ccp_alpha_low') is None else self.kwargs.get('ccp_alpha_low'),
                                                high=1.0 if self.kwargs.get('ccp_alpha_high') is None else self.kwargs.get('ccp_alpha_high')
                                                )
                    )

    def k_nearest_neighbor(self) -> KNeighborsClassifier:
        """
        Train k-nearest-neighbor (KNN) classifier

        :return KNeighborsClassifier:
            Model object
        """
        return KNeighborsClassifier(n_neighbors=5 if self.clf_params.get('n_neighbors') is None else self.clf_params.get('n_neighbors'),
                                    weights='uniform' if self.clf_params.get('weights') is None else self.clf_params.get('weights'),
                                    algorithm='auto' if self.clf_params.get('algorithm') is None else self.clf_params.get('algorithm'),
                                    leaf_size=30 if self.clf_params.get('leaf_size') is None else self.clf_params.get('leaf_size'),
                                    p=2 if self.clf_params.get('p') is None else self.clf_params.get('p'),
                                    metric='minkowski' if self.clf_params.get('metric') is None else self.clf_params.get('metric'),
                                    metric_params=self.clf_params.get('metric_params'),
                                    n_jobs=self.cpu_cores
                                    )

    def k_nearest_neighbor_param(self) -> dict:
        """
        Generate K-Nearest Neighbor classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_neighbors=np.random.randint(low=2 if self.kwargs.get('n_neighbors_low') is None else self.kwargs.get('n_neighbors_low'),
                                                  high=12 if self.kwargs.get('n_neighbors_high') is None else self.kwargs.get('n_neighbors_high')
                                                  ),
                    weights=np.random.choice(a=['uniform', 'distance'] if self.kwargs.get('weights_choice') is None else self.kwargs.get('weights_choice')),
                    algorithm=np.random.choice(a=['auto', 'ball_tree', 'kd_tree', 'brute'] if self.kwargs.get('algorithm_choice') is None else self.kwargs.get('algorithm_choice')),
                    leaf_size=np.random.randint(low=15 if self.kwargs.get('leaf_size_low') is None else self.kwargs.get('leaf_size_low'),
                                                high=100 if self.kwargs.get('leaf_size_high') is None else self.kwargs.get('leaf_size_high')
                                                ),
                    p=np.random.choice(a=[1, 2, 3] if self.kwargs.get('p_choice') is None else self.kwargs.get('p_choice')),
                    #metric=np.random.choice(a=['minkowski', 'precomputed'])
                    )

    def linear_discriminant_analysis(self) -> LinearDiscriminantAnalysis:
        """
        Config linear discriminant analysis

        :return: LinearDiscriminantAnalysis:
            Model object
        """
        return LinearDiscriminantAnalysis(solver='svd' if self.clf_params.get('solver') is None else self.clf_params.get('solver'),
                                          shrinkage=self.clf_params.get('shrinkage') if self.clf_params.get('solver') != 'svd' else None,
                                          priors=self.clf_params.get('priors'),
                                          n_components=self.clf_params.get('n_components'),
                                          store_covariance=False if self.clf_params.get('store_covariance') is None else self.clf_params.get('store_covariance'),
                                          tol=0.0001 if self.clf_params.get('tol') is None else self.clf_params.get('tol')
                                          )

    def linear_discriminant_analysis_param(self) -> dict:
        """
        Generate Linear Discriminant Analysis classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(shrinkage=np.random.uniform(low=0.0001 if self.kwargs.get('shrinkage_low') is None else self.kwargs.get('shrinkage_low'),
                                                high=0.9999 if self.kwargs.get('shrinkage_high') is None else self.kwargs.get('shrinkage_high')
                                                ),
                    solver=np.random.choice(a=['svd', 'eigen'] if self.kwargs.get('solver_choice') is None else self.kwargs.get('solver_choice'))
                    )

    def logistic_regression(self) -> LogisticRegression:
        """

        Training of Logistic Regression

        :return: LogisticRegression:
            Model object
        """
        return LogisticRegression(penalty='l2' if self.clf_params.get('penalty') is None else self.clf_params.get('penalty'),
                                  dual=False if self.clf_params.get('dual') is None else self.clf_params.get('dual'),
                                  tol=0.0001 if self.clf_params.get('tol') is None else self.clf_params.get('tol'),
                                  C=1.0 if self.clf_params.get('C') is None else self.clf_params.get('C'),
                                  fit_intercept=True if self.clf_params.get('fit_intercept') is None else self.clf_params.get('fit_intercept'),
                                  intercept_scaling=1 if self.clf_params.get('intercept_scaling') is None else self.clf_params.get('intercept_scaling'),
                                  class_weight=None if self.clf_params.get('class_weight') is None else self.clf_params.get('class_weight'),
                                  random_state=self.seed if self.clf_params.get('random_state') is None else self.clf_params.get('random_state'),
                                  solver='saga' if self.clf_params.get('solver') is None else self.clf_params.get('solver'),
                                  max_iter=100 if self.clf_params.get('max_iter') is None else self.clf_params.get('max_iter'),
                                  multi_class='ovr' if self.clf_params.get('multi_class') is None else self.clf_params.get('multi_class'),
                                  verbose=0 if self.clf_params.get('verbose') is None else self.clf_params.get('verbose'),
                                  warm_start=False if self.clf_params.get('warm_start') is None else self.clf_params.get('warm_start'),
                                  l1_ratio=np.random.uniform(low=0.0001, high=1.0),
                                  n_jobs=self.cpu_cores
                                  )

    def logistic_regression_param(self) -> dict:
        """
        Generate Logistic Regression classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(C=np.random.uniform(low=0.0001 if self.kwargs.get('C_low') is None else self.kwargs.get('C_low'),
                                        high=1.0 if self.kwargs.get('C_high') is None else self.kwargs.get('C_high')
                                        ),
                    penalty=np.random.choice(a=['l1', 'l2', 'elasticnet', 'none'] if self.kwargs.get('penalty_choice') is None else self.kwargs.get('penalty_choice')),
                    #solver=np.random.choice(a=['liblinear', 'lbfgs', 'sag', 'saga', 'newton-cg']),
                    max_iter=np.random.randint(low=5 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=500 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               )
                    )

    def quadratic_discriminant_analysis(self) -> QuadraticDiscriminantAnalysis:
        """
        Generate Quadratic Discriminant Analysis classifier parameter configuration randomly

        :return QuadraticDiscriminantAnalysis:
            Model object
        """
        return QuadraticDiscriminantAnalysis(priors=self.clf_params.get('priors'),
                                             reg_param=0.0 if self.clf_params.get('reg_param') is None else self.clf_params.get('reg_param'),
                                             store_covariance=False if self.clf_params.get('store_covariance') is None else self.clf_params.get('store_covariance'),
                                             tol=0.0001 if self.clf_params.get('tol') is None else self.clf_params.get('tol')
                                             )

    def quadratic_discriminant_analysis_param(self) -> dict:
        """
        Generate Quadratic Discriminant Analysis classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(reg_param=np.random.uniform(low=0.0001 if self.kwargs.get('reg_param_low') is None else self.kwargs.get('reg_param_low'),
                                                high=0.9999 if self.kwargs.get('reg_param_high') is None else self.kwargs.get('reg_param_high')
                                                )
                    )

    def random_forest(self) -> RandomForestClassifier:
        """
        Training of the Random Forest Classifier

        :return RandomForestClassifier:
            Model object
        """
        return RandomForestClassifier(n_estimators=500 if self.clf_params.get('n_estimators') is None else self.clf_params.get('n_estimators'),
                                      criterion='gini' if self.clf_params.get('criterion') is None else self.clf_params.get('criterion'),
                                      max_depth=1 if self.clf_params.get('max_depth') is None else self.clf_params.get('max_depth'),
                                      min_samples_split=2 if self.clf_params.get('min_samples_split') is None else self.clf_params.get('min_samples_split'),
                                      min_samples_leaf=1 if self.clf_params.get('min_samples_leaf') is None else self.clf_params.get('min_samples_leaf'),
                                      min_weight_fraction_leaf=0 if self.clf_params.get('min_weight_fraction_leaf') is None else self.clf_params.get('min_weight_fraction_leaf'),
                                      max_features='auto' if self.clf_params.get('max_features') is None else self.clf_params.get('max_features'),
                                      max_leaf_nodes=None if self.clf_params.get('max_leaf_nodes') is None else self.clf_params.get('max_leaf_nodes'),
                                      min_impurity_decrease=0 if self.clf_params.get('min_impurity_decrease') is None else self.clf_params.get('min_impurity_decrease'),
                                      min_impurity_split=None if self.clf_params.get('min_impurity_split') is None else self.clf_params.get('min_impurity_split'),
                                      bootstrap=True if self.clf_params.get('bootstrap') is None else self.clf_params.get('bootstrap'),
                                      oob_score=False if self.clf_params.get('oob_score') is None else self.clf_params.get('oob_score'),
                                      n_jobs=self.cpu_cores if self.clf_params.get('n_jobs') is None else self.clf_params.get('n_jobs'),
                                      random_state=self.seed,
                                      verbose=0 if self.clf_params.get('verbose') is None else self.clf_params.get('verbose'),
                                      warm_start=False if self.clf_params.get('warm_start') is None else self.clf_params.get('warm_start'),
                                      )

    def random_forest_param(self) -> dict:
        """
        Generate Random Forest classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                                   high=100 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                                   ),
                    criterion=np.random.choice(a=['gini', 'entropy'] if self.kwargs.get('criterion_choice') is None else self.kwargs.get('criterion_choice')),
                    max_depth=np.random.randint(low=1 if self.kwargs.get('max_depth_low') is None else self.kwargs.get('max_depth_low'),
                                                high=12 if self.kwargs.get('max_depth_high') is None else self.kwargs.get('max_depth_high')
                                                ),
                    min_samples_split=np.random.randint(low=2 if self.kwargs.get('min_samples_split_low') is None else self.kwargs.get('min_samples_split_low'),
                                                        high=6 if self.kwargs.get('min_samples_split_high') is None else self.kwargs.get('min_samples_split_high')
                                                        ),
                    min_samples_leaf=np.random.randint(low=1 if self.kwargs.get('min_samples_leaf_low') is None else self.kwargs.get('min_samples_leaf_low'),
                                                       high=6 if self.kwargs.get('min_samples_leaf_high') is None else self.kwargs.get('min_samples_leaf_high')
                                                       ),
                    bootstrap=np.random.choice(a=[True, False] if self.kwargs.get('bootstrap_choice') is None else self.kwargs.get('bootstrap_choice'))
                    )

    def support_vector_machine(self) -> SVC:
        """
        Training of the Support Vector Machine Classifier

        :return SVC:
            Model object
        """
        return SVC(C=1.0 if self.clf_params.get('C') is None else self.clf_params.get('C'),
                   kernel='rbf' if self.clf_params.get('kernel') is None else self.clf_params.get('kernel'),
                   degree=3 if self.clf_params.get('degree') is None else self.clf_params.get('degree'),
                   gamma='auto' if self.clf_params.get('gamma') is None else self.clf_params.get('gamma'),
                   coef0=0.0 if self.clf_params.get('coef0') is None else self.clf_params.get('coef0'),
                   tol=0.0001 if self.clf_params.get('tol') is None else self.clf_params.get('tol'),
                   shrinking=True if self.clf_params.get('shrinking') is None else self.clf_params.get('shrinking'),
                   cache_size=200 if self.clf_params.get('cache_size') is None else self.clf_params.get('cache_size'),
                   class_weight=None if self.clf_params.get('class_weight') is None else self.clf_params.get('class_weight'),
                   verbose=False if self.clf_params.get('verbose') is None else self.clf_params.get('verbose'),
                   max_iter=-1 if self.clf_params.get('max_iter') is None else self.clf_params.get('max_iter'),
                   decision_function_shape='ovr' if self.clf_params.get('decision_function_shape') is None else self.clf_params.get('decision_function_shape'),
                   probability=False if self.clf_params.get('probability') is None else self.clf_params.get('probability')
                   )

    def support_vector_machine_param(self) -> dict:
        """
        Generate Support Vector Machine classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(C=np.random.uniform(low=0.0001 if self.kwargs.get('C_low') is None else self.kwargs.get('C_low'),
                                        high=1.0 if self.kwargs.get('C_high') is None else self.kwargs.get('C_high')
                                        ),
                    kernel=np.random.choice(a=['rbf', 'linear', 'poly', 'sigmoid'] if self.kwargs.get('kernel_choice') is None else self.kwargs.get('kernel_choice')), #'precomputed'
                    #gamma=np.random.choice(a=['auto', 'scale']),
                    shrinking=np.random.choice(a=[True, False] if self.kwargs.get('shrinking_choice') is None else self.kwargs.get('shrinking_choice')),
                    cache_size=np.random.randint(low=100 if self.kwargs.get('cache_size_low') is None else self.kwargs.get('cache_size_low'),
                                                 high=500 if self.kwargs.get('cache_size_high') is None else self.kwargs.get('cache_size_high')
                                                 ),
                    decision_function_shape=np.random.choice(a=['ovo', 'ovr'] if self.kwargs.get('decision_function_shape_choice') is None else self.kwargs.get('decision_function_shape_choice')),
                    max_iter=np.random.randint(low=10 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=100 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               )
                    )

    def linear_support_vector_machine(self) -> LinearSVC:
        """
        Config Linear Support Vector Machine Classifier

        :return LinearSVC:
            Model object
        """
        return LinearSVC(penalty='l2' if self.clf_params.get('penalty') is None else self.clf_params.get('penalty'),
                         loss='squared_hinge' if self.clf_params.get('loss') is None else self.clf_params.get('loss'),
                         dual=True if self.clf_params.get('dual') is None else self.clf_params.get('dual'),
                         tol=0.0001 if self.clf_params.get('tol') is None else self.clf_params.get('tol'),
                         C=1.0 if self.clf_params.get('C') is None else self.clf_params.get('C'),
                         multi_class='ovr' if self.clf_params.get('multi_class') is None else self.clf_params.get('multi_class'),
                         fit_intercept=True if self.clf_params.get('fit_intercept') is None else self.clf_params.get('fit_intercept'),
                         intercept_scaling=1 if self.clf_params.get('intercept_scaling') is None else self.clf_params.get('intercept_scaling'),
                         class_weight=None if self.clf_params.get('class_weight') is None else self.clf_params.get('class_weight'),
                         verbose=0 if self.clf_params.get('verbose') is None else self.clf_params.get('verbose'),
                         random_state=self.seed,
                         max_iter=500 if self.clf_params.get('max_iter') is None else self.clf_params.get('max_iter')
                         )

    def linear_support_vector_machine_param(self) -> dict:
        """
        Generate Linear Support Vector Machine classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(C=np.random.uniform(low=0.0001 if self.kwargs.get('C_low') is None else self.kwargs.get('C_low'),
                                        high=1.0 if self.kwargs.get('C_high') is None else self.kwargs.get('C_high')
                                        ),
                    penalty=np.random.choice(a=['l1', 'l2'] if self.kwargs.get('penalty_choice') is None else self.kwargs.get('penalty_choice')),
                    loss=np.random.choice(a=['hinge', 'squared_hinge'] if self.kwargs.get('loss_choice') is None else self.kwargs.get('loss_choice')),
                    multi_class=np.random.choice(a=['ovr', 'crammer_singer'] if self.kwargs.get('multi_class_choice') is None else self.kwargs.get('multi_class_choice')),
                    max_iter=np.random.randint(low=10 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=100 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               )
                    )

    def nu_support_vector_machine(self) -> NuSVC:
        """
        Config Nu-Support Vector Machine Classifier

        :return NuSVC:
            Model object
        """
        return NuSVC(nu=0.5 if self.clf_params.get('nu') is None else self.clf_params.get('nu'),
                     kernel='rbf' if self.clf_params.get('kernel') is None else self.clf_params.get('kernel'),
                     degree=3 if self.clf_params.get('degree') is None else self.clf_params.get('degree'),
                     gamma='auto' if self.clf_params.get('gamma') is None else self.clf_params.get('gamma'),
                     coef0=0.0 if self.clf_params.get('coef0') is None else self.clf_params.get('coef0'),
                     shrinking=True if self.clf_params.get('shrinking') is None else self.clf_params.get('shrinking'),
                     tol=0.001 if self.clf_params.get('tol') is None else self.clf_params.get('tol'),
                     cache_size=200 if self.clf_params.get('cache_size') is None else self.clf_params.get('cache_size'),
                     class_weight=None if self.clf_params.get('class_weight') is None else self.clf_params.get('class_weight'),
                     verbose=False if self.clf_params.get('verbose') is None else self.clf_params.get('verbose'),
                     max_iter=-1 if self.clf_params.get('max_iter') is None else self.clf_params.get('max_iter'),
                     decision_function_shape='ovr' if self.clf_params.get('decision_function_shape') is None else self.clf_params.get('decision_function_shape'),
                     probability=True if self.clf_params.get('probability') is None else self.clf_params.get('probability')
                     )

    def nu_support_vector_machine_param(self) -> dict:
        """
        Generate Nu-Support Vector Machine classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(C=np.random.uniform(low=0.0001 if self.kwargs.get('C_low') is None else self.kwargs.get('C_low'),
                                        high=1.0 if self.kwargs.get('C_high') is None else self.kwargs.get('C_high')
                                        ),
                    nu=np.random.uniform(low=0.01 if self.kwargs.get('nu_low') is None else self.kwargs.get('nu_low'),
                                         high=0.99 if self.kwargs.get('nu_high') is None else self.kwargs.get('nu_high')
                                         ),
                    kernel=np.random.choice(a=['rbf', 'linear', 'poly', 'sigmoid'] if self.kwargs.get('kernel_choice') is None else self.kwargs.get('kernel_choice')), #'precomputed'
                    #gamma=np.random.choice(a=['auto', 'scale']),
                    shrinking=np.random.choice(a=[True, False] if self.kwargs.get('shrinking_choice') is None else self.kwargs.get('shrinking_choice')),
                    cache_size=np.random.randint(low=100 if self.kwargs.get('cache_size_low') is None else self.kwargs.get('cache_size_low'),
                                                 high=500 if self.kwargs.get('cache_size_high') is None else self.kwargs.get('cache_size_high')
                                                 ),
                    decision_function_shape=np.random.choice(a=['ovo', 'ovr'] if self.kwargs.get('decision_function_shape_choice') is None else self.kwargs.get('decision_function_shape_choice')),
                    max_iter=np.random.randint(low=10 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=100 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               )
                    )


class Regression:
    """
    Class for handling regression algorithms
    """
    def __init__(self,
                 reg_params: dict = None,
                 cpu_cores: int = 0,
                 seed: int = 1234,
                 **kwargs
                 ):
        """
        :param reg_params: dict
            Pre-configured regression model parameter

        :param cpu_cores: int
            Number of CPU core to use

        :param seed: int
            Seed

        :param kwargs: dict
            Key-word arguments
        """
        self.reg_params: dict = {} if reg_params is None else reg_params
        self.seed: int = 1234 if seed <= 0 else seed
        if cpu_cores <= 0:
            self.cpu_cores: int = os.cpu_count() - 1 if os.cpu_count() > 1 else os.cpu_count()
        else:
            if cpu_cores <= os.cpu_count():
                self.cpu_cores: int = cpu_cores
            else:
                self.cpu_cores: int = os.cpu_count() - 1 if os.cpu_count() > 1 else os.cpu_count()
        self.kwargs: dict = kwargs

    def ada_boosting(self) -> AdaBoostRegressor:
        """
        Config Ada Boosting algorithm

        :return: AdaBoostRegressor
            Model object
        """
        return AdaBoostRegressor(base_estimator=self.reg_params.get('base_estimator'),
                                 n_estimators=50 if self.reg_params.get('n_estimators') is None else self.reg_params.get('n_estimators'),
                                 learning_rate=1.0 if self.reg_params.get('learning_rate') is None else self.reg_params.get('learning_rate'),
                                 loss='linear' if self.reg_params.get('loss') is None else self.reg_params.get('loss'),
                                 random_state=self.seed
                                 )

    def ada_boosting_param(self) -> dict:
        """
        Generate Ada Boosting regressor parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                                   high=500 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                                   ),
                    learning_rate=np.random.uniform(low=0.01 if self.kwargs.get('learning_rate_low') is None else self.kwargs.get('learning_rate_low'),
                                                    high=1.0 if self.kwargs.get('learning_rate_high') is None else self.kwargs.get('learning_rate_high')
                                                    ),
                    # base_estimator=np.random.choice(a=[None, BaseDecisionTree, DecisionTreeRegressor, ExtraTreeRegressor]),
                    loss=np.random.choice(a=['linear', 'square', 'exponential'] if self.kwargs.get('loss_choice') is None else self.kwargs.get('loss_choice')),
                    algorithm=np.random.choice(a=['SAMME', 'SAMME.R'] if self.kwargs.get('algorithm_choice') is None else self.kwargs.get('algorithm_choice'))
                    )

    def cat_boost(self) -> CatBoostRegressor:
        """
        Config CatBoost Regressor

        :return: CatBoostRegressor
        """
        return CatBoostRegressor(n_estimators=100 if self.reg_params.get('n_estimators') is None else self.reg_params.get('n_estimators'),
                                 learning_rate=0.03 if self.reg_params.get('learning_rate') is None else self.reg_params.get('learning_rate'),
                                 depth=self.reg_params.get('depth'),
                                 l2_leaf_reg=self.reg_params.get('l2_leaf_reg'),
                                 model_size_reg=self.reg_params.get('model_size_reg'),
                                 rsm=self.reg_params.get('rsm'),
                                 loss_function=self.reg_params.get('loss_function'),
                                 border_count=self.reg_params.get('border_count'),
                                 feature_border_type=self.reg_params.get('feature_border_type'),
                                 per_float_feature_quantization=self.reg_params.get('per_float_feature_quantization'),
                                 input_borders=self.reg_params.get('input_borders'),
                                 output_borders=self.reg_params.get('output_borders'),
                                 fold_permutation_block=self.reg_params.get('fold_permutation_block'),
                                 od_pval=self.reg_params.get('od_pval'),
                                 od_wait=self.reg_params.get('od_wait'),
                                 od_type=self.reg_params.get('od_type'),
                                 nan_mode=self.reg_params.get('nan_mode'),
                                 counter_calc_method=self.reg_params.get('counter_calc_method'),
                                 leaf_estimation_iterations=self.reg_params.get('leaf_estimation_iterations'),
                                 leaf_estimation_method=self.reg_params.get('leaf_estimation_method'),
                                 thread_count=self.reg_params.get('thread_count'),
                                 random_seed=self.reg_params.get('random_seed'),
                                 use_best_model=self.reg_params.get('use_best_model'),
                                 best_model_min_trees=self.reg_params.get('best_model_min_trees'),
                                 verbose=self.reg_params.get('verbose'),
                                 silent=self.reg_params.get('silent'),
                                 logging_level=self.reg_params.get('logging_level'),
                                 metric_period=self.reg_params.get('metric_period'),
                                 ctr_leaf_count_limit=self.reg_params.get('ctr_leaf_count_limit'),
                                 store_all_simple_ctr=self.reg_params.get('store_all_simple_ctr'),
                                 max_ctr_complexity=self.reg_params.get('max_ctr_complexity'),
                                 has_time=self.reg_params.get('has_time'),
                                 allow_const_label=self.reg_params.get('allow_const_label'),
                                 target_border=self.reg_params.get('target_border'),
                                 one_hot_max_size=self.reg_params.get('one_hot_max_size'),
                                 random_strength=self.reg_params.get('random_strength'),
                                 name=self.reg_params.get('name'),
                                 ignored_features=self.reg_params.get('ignored_features'),
                                 train_dir=self.reg_params.get('train_dir'),
                                 custom_metric=self.reg_params.get('custom_metric'),
                                 eval_metric=self.reg_params.get('eval_metric'),
                                 bagging_temperature=self.reg_params.get('bagging_temperature'),
                                 save_snapshot=self.reg_params.get('save_snapshot'),
                                 snapshot_file=self.reg_params.get('snapshot_file'),
                                 snapshot_interval=self.reg_params.get('snapshot_interval'),
                                 fold_len_multiplier=self.reg_params.get('fold_len_multiplier'),
                                 used_ram_limit=self.reg_params.get('used_ram_limit'),
                                 gpu_ram_part=self.reg_params.get('gpu_ram_part'),
                                 pinned_memory_size=self.reg_params.get('pinned_memory_size'),
                                 allow_writing_files=self.reg_params.get('allow_writing_files'),
                                 final_ctr_computation_mode=self.reg_params.get('final_ctr_computation_mode'),
                                 approx_on_full_history=self.reg_params.get('approx_on_full_history'),
                                 boosting_type=self.reg_params.get('boosting_type'),
                                 simple_ctr=self.reg_params.get('simple_ctr'),
                                 combinations_ctr=self.reg_params.get('combinations_ctr'),
                                 per_feature_ctr=self.reg_params.get('per_feature_ctr'),
                                 ctr_description=self.reg_params.get('ctr_description'),
                                 ctr_target_border_count=self.reg_params.get('ctr_target_border_count'),
                                 task_type=self.reg_params.get('task_type'),
                                 device_config=self.reg_params.get('device_config'),
                                 devices=self.reg_params.get('devices'),
                                 bootstrap_type=self.reg_params.get('bootstrap_type'),
                                 subsample=self.reg_params.get('subsample'),
                                 mvs_reg=self.reg_params.get('mvs_reg'),
                                 sampling_unit=self.reg_params.get('sampling_unit'),
                                 sampling_frequency=self.reg_params.get('sampling_frequency'),
                                 dev_score_calc_obj_block_size=self.reg_params.get('dev_score_calc_obj_block_size'),
                                 dev_efb_max_buckets=self.reg_params.get('dev_efb_max_buckets'),
                                 sparse_features_conflict_fraction=self.reg_params.get('sparse_features_conflict_fraction'),
                                 #max_depth=self.reg_params.get('max_depth'),
                                 num_boost_round=self.reg_params.get('num_boost_round'),
                                 num_trees=self.reg_params.get('num_trees'),
                                 colsample_bylevel=self.reg_params.get('colsample_bylevel'),
                                 random_state=self.reg_params.get('random_state'),
                                 #reg_lambda=self.reg_params.get('reg_lambda'),
                                 objective=self.reg_params.get('objective'),
                                 eta=self.reg_params.get('eta'),
                                 max_bin=self.reg_params.get('max_bin'),
                                 gpu_cat_features_storage=self.reg_params.get('gpu_cat_features_storage'),
                                 data_partition=self.reg_params.get('data_partition'),
                                 metadata=self.reg_params.get('metadata'),
                                 early_stopping_rounds=self.reg_params.get('early_stopping_rounds'),
                                 cat_features=self.reg_params.get('cat_features'),
                                 grow_policy=self.reg_params.get('grow_policy'),
                                 min_data_in_leaf=self.reg_params.get('min_data_in_leaf'),
                                 min_child_samples=self.reg_params.get('min_child_samples'),
                                 max_leaves=self.reg_params.get('max_leaves'),
                                 num_leaves=self.reg_params.get('num_leaves'),
                                 score_function=self.reg_params.get('score_function'),
                                 leaf_estimation_backtracking=self.reg_params.get('leaf_estimation_backtracking'),
                                 ctr_history_unit=self.reg_params.get('ctr_history_unit'),
                                 monotone_constraints=self.reg_params.get('monotone_constraints'),
                                 feature_weights=self.reg_params.get('feature_weights'),
                                 penalties_coefficient=self.reg_params.get('penalties_coefficient'),
                                 first_feature_use_penalties=self.reg_params.get('first_feature_use_penalties'),
                                 per_object_feature_penalties=self.reg_params.get('per_object_feature_penalties'),
                                 model_shrink_rate=self.reg_params.get('model_shrink_rate'),
                                 model_shrink_mode=self.reg_params.get('model_shrink_mode'),
                                 langevin=self.reg_params.get('langevin'),
                                 diffusion_temperature=self.reg_params.get('diffusion_temperature'),
                                 posterior_sampling=self.reg_params.get('posterior_sampling'),
                                 boost_from_average=self.reg_params.get('boost_from_average')
                                 )

    def cat_boost_param(self) -> dict:
        """
        Generate Cat Boost regressor parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                                   high=100 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                                   ),
                    learning_rate=np.random.uniform(low=0.0001 if self.kwargs.get('learning_rate_low') is None else self.kwargs.get('learning_rate_low'),
                                                    high=0.5 if self.kwargs.get('learning_rate_high') is None else self.kwargs.get('learning_rate_high')
                                                    ),
                    l2_leaf_reg=np.random.uniform(low=0.1 if self.kwargs.get('l2_leaf_reg_low') is None else self.kwargs.get('l2_leaf_reg_low'),
                                                  high=1.0 if self.kwargs.get('l2_leaf_reg_high') is None else self.kwargs.get('l2_leaf_reg_high')
                                                  ),
                    depth=np.random.randint(low=3 if self.kwargs.get('depth_low') is None else self.kwargs.get('depth_low'),
                                            high=16 if self.kwargs.get('depth_high') is None else self.kwargs.get('depth_high')
                                            ),
                    #sampling_frequency=np.random.choice(a=['PerTree', 'PerTreeLevel']),
                    #sampling_unit=np.random.choice(a=['Object', 'Group']),
                    grow_policy=np.random.choice(a=['SymmetricTree', 'Depthwise', 'Lossguide'] if self.kwargs.get('grow_policy_choice') is None else self.kwargs.get('grow_policy_choice'), ),
                    min_data_in_leaf=np.random.randint(low=1 if self.kwargs.get('min_data_in_leaf_low') is None else self.kwargs.get('min_data_in_leaf_low'),
                                                       high=20 if self.kwargs.get('min_data_in_leaf_high') is None else self.kwargs.get('min_data_in_leaf_high')
                                                       ),
                    #max_leaves=np.random.randint(low=10, high=64),
                    rsm=np.random.uniform(low=0.1 if self.kwargs.get('rsm_low') is None else self.kwargs.get('rsm_low'),
                                          high=1 if self.kwargs.get('rsm_high') is None else self.kwargs.get('rsm_high')
                                          ),
                    #fold_len_multiplier=np.random.randint(low=2, high=4),
                    #approx_on_full_history=np.random.choice(a=[False, True]),
                    #boosting_type=np.random.choice(a=['Ordered', 'Plain']),
                    #score_function=np.random.choice(a=['Cosine', 'L2', 'NewtonCosine', 'NewtonL2']),
                    #model_shrink_mode=np.random.choice(a=['Constant', 'Decreasing']),
                    #border_count=np.random.randint(low=1, high=65535),
                    feature_border_type=np.random.choice(a=['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum', 'MinEntropy', 'GreedyLogSum'] if self.kwargs.get('feature_border_type_choice') is None else self.kwargs.get('feature_border_type_choice'))
                    )

    def elastic_net(self) -> ElasticNet:
        """
        Training of the elastic net regressor

        :return ElasticNet
            Object model
        """
        return ElasticNet(alpha=1.0 if self.reg_params.get('alpha') is None else self.reg_params.get('alpha'),
                          l1_ratio=0.5 if self.reg_params.get('l1_ratio') is None else self.reg_params.get('l1_ratio'),
                          fit_intercept=True if self.reg_params.get('fit_intercept') is None else self.reg_params.get('fit_intercept'),
                          normalize=True if self.reg_params.get('normalize') is None else self.reg_params.get('normalize'),
                          precompute=False if self.reg_params.get('precompute') is None else self.reg_params.get('precompute'),
                          max_iter=1000 if self.reg_params.get('max_iter') is None else self.reg_params.get('max_iter'),
                          copy_X=True if self.reg_params.get('copy_X') is None else self.reg_params.get('copy_X'),
                          tol=0.0001 if self.reg_params.get('tol') is None else self.reg_params.get('tol'),
                          warm_start=False if self.reg_params.get('warm_start') is None else self.reg_params.get('warm_start'),
                          positive=False if self.reg_params.get('positive') is None else self.reg_params.get('positive'),
                          random_state=self.seed,
                          selection='cyclic' if self.reg_params.get('selection') is None else self.reg_params.get('selection')
                          )

    def elastic_net_param(self) -> dict:
        """
        Generate Elastic Net regressor parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(alpha=np.random.uniform(low=0.0 if self.kwargs.get('alpha_low') is None else self.kwargs.get('alpha_low'),
                                            high=1.0 if self.kwargs.get('alpha_high') is None else self.kwargs.get('alpha_high')
                                            ),
                    l1_ratio=np.random.uniform(low=0.0 if self.kwargs.get('l1_ratio_low') is None else self.kwargs.get('l1_ratio_low'),
                                               high=1.0 if self.kwargs.get('l1_ratio_high') is None else self.kwargs.get('l1_ratio_high')
                                               ),
                    normalize=np.random.choice(a=[True, False] if self.kwargs.get('normalize_choice') is None else self.kwargs.get('normalize_choice')),
                    #precompute=np.random.choice(a=[True, False]),
                    max_iter=np.random.randint(low=5 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=1000 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               ),
                    fit_intercept=np.random.choice(a=[True, False] if self.kwargs.get('fit_intercept_choice') is None else self.kwargs.get('fit_intercept_choice')),
                    selection=np.random.choice(a=['cyclic', 'random'] if self.kwargs.get('selection_choice') is None else self.kwargs.get('selection_choice'))
                    )

    def extreme_gradient_boosting_tree(self) -> XGBRegressor:
        """
        Training of the Extreme Gradient Boosting Regressor

        :return: XGBRegressor
            Model object
        """
        return XGBRegressor(max_depth=3 if self.reg_params.get('max_depth') is None else self.reg_params.get('max_depth'),
                            learning_rate=0.1 if self.reg_params.get('learning_rate') is None else self.reg_params.get('learning_rate'),
                            n_estimators=100 if self.reg_params.get('n_estimators') is None else self.reg_params.get('n_estimators'),
                            verbosity=0 if self.reg_params.get('verbosity') is None else self.reg_params.get('verbosity'),
                            objective='reg:squarederror' if self.reg_params.get('objective') is None else self.reg_params.get('objective'),
                            booster='gbtree' if self.reg_params.get('booster') is None else self.reg_params.get('booster'),
                            n_jobs=self.cpu_cores,
                            gamma=0 if self.reg_params.get('gamma') is None else self.reg_params.get('gamma'),
                            min_child_weight=1 if self.reg_params.get('min_child_weight') is None else self.reg_params.get('min_child_weight'),
                            max_delta_step=0 if self.reg_params.get('max_delta_step') is None else self.reg_params.get('max_delta_step'),
                            subsample=1 if self.reg_params.get('subsample') is None else self.reg_params.get('subsample'),
                            colsample_bytree=1 if self.reg_params.get('colsample_bytree') is None else self.reg_params.get('colsample_bytree'),
                            colsample_bylevel=1 if self.reg_params.get('colsample_bylevel') is None else self.reg_params.get('colsample_bylevel'),
                            colsample_bynode=1 if self.reg_params.get('colsample_bynode') is None else self.reg_params.get('colsample_bynode'),
                            reg_alpha=0 if self.reg_params.get('reg_alpha') is None else self.reg_params.get('reg_alpha'),
                            reg_lambda=1 if self.reg_params.get('reg_lambda') is None else self.reg_params.get('reg_lambda'),
                            scale_pos_weight=1.0 if self.reg_params.get('scale_pos_weight') is None else self.reg_params.get('scale_pos_weight'),
                            base_score=0.5 if self.reg_params.get('base_score') is None else self.reg_params.get('base_score'),
                            random_state=self.seed,
                            importance_type='gain' if self.reg_params.get('importance_type') is None else self.reg_params.get('importance_type')
                            )

    def extreme_gradient_boosting_tree_param(self) -> dict:
        """
        Generate Extreme Gradient Boosting Decision Tree regressor parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(learning_rate=np.random.uniform(low=0.0001 if self.kwargs.get('learning_rate_low') is None else self.kwargs.get('learning_rate_low'),
                                                    high=0.5 if self.kwargs.get('learning_rate_high') is None else self.kwargs.get('learning_rate_high')
                                                    ),
                    n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                                   high=100 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                                   ),
                    min_samples_split=np.random.randint(low=2 if self.kwargs.get('min_samples_split_low') is None else self.kwargs.get('min_samples_split_low'),
                                                        high=6 if self.kwargs.get('min_samples_split_high') is None else self.kwargs.get('min_samples_split_high')
                                                        ),
                    min_samples_leaf=np.random.randint(low=1 if self.kwargs.get('min_samples_leaf_low') is None else self.kwargs.get('min_samples_leaf_low'),
                                                       high=6 if self.kwargs.get('min_samples_leaf_high') is None else self.kwargs.get('min_samples_leaf_high')
                                                       ),
                    max_depth=np.random.randint(low=3 if self.kwargs.get('max_depth_low') is None else self.kwargs.get('max_depth_low'),
                                                high=12 if self.kwargs.get('max_depth_high') is None else self.kwargs.get('max_depth_high')
                                                ),
                    #booster=np.random.choice(a=['gbtree', 'gblinear', 'gbdart']),
                    gamma=np.random.uniform(low=0.01 if self.kwargs.get('gamma_low') is None else self.kwargs.get('gamma_low'),
                                            high=0.99 if self.kwargs.get('gamma_high') is None else self.kwargs.get('gamma_high')
                                            ),
                    min_child_weight=np.random.randint(low=1 if self.kwargs.get('min_child_weight_low') is None else self.kwargs.get('min_child_weight_low'),
                                                       high=12 if self.kwargs.get('min_child_weight_high') is None else self.kwargs.get('min_child_weight_high')
                                                       ),
                    reg_alpha=np.random.uniform(low=0.0 if self.kwargs.get('reg_alpha_low') is None else self.kwargs.get('reg_alpha_low'),
                                                high=0.9 if self.kwargs.get('reg_alpha_high') is None else self.kwargs.get('reg_alpha_high')
                                                ),
                    reg_lambda=np.random.uniform(low=0.1 if self.kwargs.get('reg_lambda_low') is None else self.kwargs.get('reg_lambda_low'),
                                                 high=1.0 if self.kwargs.get('reg_lambda_high') is None else self.kwargs.get('reg_lambda_high')
                                                 ),
                    subsample=np.random.uniform(low=0.0 if self.kwargs.get('subsample_low') is None else self.kwargs.get('subsample_low'),
                                                high=1.0 if self.kwargs.get('subsample_high') is None else self.kwargs.get('subsample_high')
                                                ),
                    colsample_bytree=np.random.uniform(low=0.5 if self.kwargs.get('colsample_bytree_low') is None else self.kwargs.get('colsample_bytree_low'),
                                                       high=0.99 if self.kwargs.get('colsample_bytree_high') is None else self.kwargs.get('colsample_bytree_high')
                                                       ),
                    #scale_pos_weight=np.random.uniform(low=0.01, high=1.0),
                    #base_score=np.random.uniform(low=0.01, high=0.99),
                    early_stopping=np.random.choice(a=[True, False] if self.kwargs.get('early_stopping_choice') is None else self.kwargs.get('early_stopping_choice'))
                    )

    @staticmethod
    def generalized_additive_models() -> GAM:
        """
        Config Generalized Additive Model regressor

        :return: GAM
            Model object
        """
        return GAM(terms='auto',
                   max_iter=100,
                   tol=1e-4,
                   distribution='normal',
                   link='identity',
                   callbacks=['deviance', 'diffs', 'accuracy', 'coef'],
                   fit_intercept=True,
                   verbose=False
                   )

    def generalized_additive_models_param(self) -> dict:
        """
        Config Generalized Additive Model regressor

        :return: dict
            Parameter config
        """
        return dict(max_iter=np.random.randint(low=10 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=500 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               ),
                    tol=np.random.uniform(low=0.00001 if self.kwargs.get('tol_low') is None else self.kwargs.get('tol_low'),
                                          high=0.001 if self.kwargs.get('tol_high') is None else self.kwargs.get('tol_high')
                                          ),
                    distribution=np.random.choice(a=['normal', 'binomial', 'poisson', 'gamma', 'invgauss'] if self.kwargs.get('distribution_choice') is None else self.kwargs.get('distribution_choice')),
                    link=np.random.choice(a=['identity', 'logit', 'log', 'inverse', 'inverse-squared'] if self.kwargs.get('link_choice') is None else self.kwargs.get('link_choice'))
                    )

    def gradient_boosting_tree(self) -> GradientBoostingRegressor:
        """
        Config gradient boosting decision tree regressor

        :return GradientBoostingRegressor
            Model object
        """
        return GradientBoostingRegressor(loss='ls' if self.reg_params.get('loss') is None else self.reg_params.get('loss'),
                                         learning_rate=0.1 if self.reg_params.get('learning_rate') is None else self.reg_params.get('learning_rate'),
                                         n_estimators=100 if self.reg_params.get('n_estimators') is None else self.reg_params.get('n_estimators'),
                                         subsample=1.0 if self.reg_params.get('subsample') is None else self.reg_params.get('subsample'),
                                         criterion='friedman_mse' if self.reg_params.get('criterion') is None else self.reg_params.get('criterion'),
                                         min_samples_split=2 if self.reg_params.get('min_samples_split') is None else self.reg_params.get('min_samples_split'),
                                         min_samples_leaf=1 if self.reg_params.get('min_samples_leaf') is None else self.reg_params.get('min_samples_leaf'),
                                         min_weight_fraction_leaf=0 if self.reg_params.get('min_weight_fraction_leaf') is None else self.reg_params.get('min_weight_fraction_leaf'),
                                         max_depth=3 if self.reg_params.get('max_depth') is None else self.reg_params.get('max_depth'),
                                         min_impurity_decrease=0 if self.reg_params.get('min_impurity_decrease') is None else self.reg_params.get('min_impurity_decrease'),
                                         min_impurity_split=self.reg_params.get('min_impurity_split'),
                                         init=self.reg_params.get('init'),
                                         random_state=self.seed,
                                         max_features=self.reg_params.get('max_features'),
                                         alpha=0.9 if self.reg_params.get('alpha') is None else self.reg_params.get('alpha'),
                                         verbose=0 if self.reg_params.get('verbose') is None else self.reg_params.get('verbose'),
                                         max_leaf_nodes=self.reg_params.get('max_leaf_nodes'),
                                         warm_start=False if self.reg_params.get('warm_start') is None else self.reg_params.get('warm_start'),
                                         validation_fraction=0.1 if self.reg_params.get('validation_fraction') is None else self.reg_params.get('validation_fraction'),
                                         n_iter_no_change=10 if self.reg_params.get('n_iter_no_change') is None else self.reg_params.get('n_iter_no_change'),
                                         tol=0.0001 if self.reg_params.get('tol') is None else self.reg_params.get('tol'),
                                         ccp_alpha=0.0 if self.reg_params.get('ccp_alpha') is None else self.reg_params.get('ccp_alpha')
                                         )

    def gradient_boosting_tree_param(self) -> dict:
        """
        Generate Gradient Boosting Tree regressor parameter randomly

        :return: dict
            Parameter config
        """
        return dict(learning_rate=np.random.uniform(low=0.0001 if self.kwargs.get('learning_rate_low') is None else self.kwargs.get('learning_rate_low'),
                                                    high=0.5 if self.kwargs.get('learning_rate_high') is None else self.kwargs.get('learning_rate_high')
                                                    ),
                    loss=np.random.choice(a=['ls', 'lad', 'huber', 'quantile']),
                    n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                                   high=100 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                                   ),
                    subsample=np.random.uniform(low=0.0 if self.kwargs.get('subsample_low') is None else self.kwargs.get('subsample_low'),
                                                high=1.0 if self.kwargs.get('subsample_high') is None else self.kwargs.get('subsample_high')
                                                ),
                    criterion=np.random.choice(a=['friedman_mse', 'mse', 'mae'] if self.kwargs.get('criterion_choice') is None else self.kwargs.get('criterion_choice')),
                    min_samples_split=np.random.randint(low=2 if self.kwargs.get('min_samples_split_low') is None else self.kwargs.get('min_samples_split_low'),
                                                        high=6 if self.kwargs.get('min_samples_split_high') is None else self.kwargs.get('min_samples_split_high')
                                                        ),
                    min_samples_leaf=np.random.randint(low=1 if self.kwargs.get('min_samples_leaf_low') is None else self.kwargs.get('min_samples_leaf_low'),
                                                       high=6 if self.kwargs.get('min_samples_leaf_high') is None else self.kwargs.get('min_samples_leaf_high')
                                                       ),
                    max_depth=np.random.randint(low=3 if self.kwargs.get('max_depth_low') is None else self.kwargs.get('max_depth_low'),
                                                high=12 if self.kwargs.get('max_depth_high') is None else self.kwargs.get('max_depth_high')
                                                ),
                    validation_fraction=np.random.uniform(low=0.05 if self.kwargs.get('validation_fraction_low') is None else self.kwargs.get('validation_fraction_low'),
                                                          high=0.4 if self.kwargs.get('validation_fraction_high') is None else self.kwargs.get('validation_fraction_high')
                                                          ),
                    n_iter_no_change=np.random.randint(low=2 if self.kwargs.get('n_iter_no_change_low') is None else self.kwargs.get('n_iter_no_change_low'),
                                                       high=10 if self.kwargs.get('n_iter_no_change_high') is None else self.kwargs.get('n_iter_no_change_high')
                                                       ),
                    alpha=np.random.uniform(low=0.01 if self.kwargs.get('alpha_low') is None else self.kwargs.get('alpha_low'),
                                            high=0.99 if self.kwargs.get('alpha_high') is None else self.kwargs.get('alpha_high')
                                            ),
                    ccp_alpha=np.random.uniform(low=0.0 if self.kwargs.get('ccp_alpha_low') is None else self.kwargs.get('ccp_alpha_low'),
                                                high=1.0 if self.kwargs.get('ccp_alpha_high') is None else self.kwargs.get('ccp_alpha_high')
                                                )
                    )

    def k_nearest_neighbor(self) -> KNeighborsRegressor:
        """
        Config K-Nearest-Neighbor (KNN) Regressor

        :return KNeighborsRegressors
            Model object
        """
        return KNeighborsRegressor(n_neighbors=5 if self.reg_params.get('n_neighbors') is None else self.reg_params.get('n_neighbors'),
                                   weights='uniform' if self.reg_params.get('weights') is None else self.reg_params.get('weights'),
                                   algorithm='auto' if self.reg_params.get('algorithm') is None else self.reg_params.get('algorithm'),
                                   leaf_size=30 if self.reg_params.get('leaf_size') is None else self.reg_params.get('leaf_size'),
                                   p=2 if self.reg_params.get('p') is None else self.reg_params.get('p'),
                                   metric='minkowski' if self.reg_params.get('metric') is None else self.reg_params.get('metric'),
                                   metric_params=None if self.reg_params.get('metric_params') is None else self.reg_params.get('metric_params'),
                                   n_jobs=self.cpu_cores
                                   )

    def k_nearest_neighbor_param(self) -> dict:
        """
        Generate K-Nearest Neighbor regressor parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_neighbors=np.random.randint(low=2 if self.kwargs.get('n_neighbors_low') is None else self.kwargs.get('n_neighbors_low'),
                                                  high=12 if self.kwargs.get('n_neighbors_high') is None else self.kwargs.get('n_neighbors_high')
                                                  ),
                    weights=np.random.choice(a=['uniform', 'distance'] if self.kwargs.get('weights_choice') is None else self.kwargs.get('weights_choice')),
                    algorithm=np.random.choice(a=['auto', 'ball_tree', 'kd_tree', 'brute'] if self.kwargs.get('algorithm_choice') is None else self.kwargs.get('algorithm_choice')),
                    leaf_size=np.random.randint(low=15 if self.kwargs.get('leaf_size_low') is None else self.kwargs.get('leaf_size_low'),
                                                high=100 if self.kwargs.get('leaf_size_high') is None else self.kwargs.get('leaf_size_high')
                                                ),
                    p=np.random.choice(a=[1, 2, 3] if self.kwargs.get('p_choice') is None else self.kwargs.get('p_choice')),
                    #metric=np.random.choice(a=['minkowski', 'precomputed'])
                    )

    def lasso_regression(self) -> Lasso:
        """
        Config Lasso Regression

        :return: Lasso
            Model object
        """
        return Lasso(alpha=1.0 if self.reg_params.get('alpha') is None else self.reg_params.get('alpha'),
                     fit_intercept=True if self.reg_params.get('fit_intercept') is None else self.reg_params.get('fit_intercept'),
                     normalize=False if self.reg_params.get('normalize') is None else self.reg_params.get('normalize'),
                     precompute=False if self.reg_params.get('precompute') is None else self.reg_params.get('precompute'),
                     copy_X=True if self.reg_params.get('copy_X') is None else self.reg_params.get('copy_X'),
                     max_iter=1000 if self.reg_params.get('max_iter') is None else self.reg_params.get('max_iter'),
                     tol=1e-4 if self.reg_params.get('tol') is None else self.reg_params.get('tol'),
                     warm_start=False if self.reg_params.get('warm_start') is None else self.reg_params.get('warm_start'),
                     positive=False if self.reg_params.get('positive') is None else self.reg_params.get('positive'),
                     random_state=self.seed,
                     selection='cyclic' if self.reg_params.get('selection') is None else self.reg_params.get('selection')
                     )

    def lasso_regression_param(self) -> dict:
        """
        Generate Lasso Regression parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(alpha=np.random.uniform(low=0.0 if self.kwargs.get('alpha_low') is None else self.kwargs.get('alpha_low'),
                                            high=1.0 if self.kwargs.get('alpha_high') is None else self.kwargs.get('alpha_high')
                                            ),
                    normalize=np.random.choice(a=[True, False] if self.kwargs.get('normalize_choice') is None else self.kwargs.get('normalize_choice')),
                    precompute=np.random.choice(a=[True, False] if self.kwargs.get('precompute_choice') is None else self.kwargs.get('precompute_choice')),
                    max_iter=np.random.randint(low=5 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=1000 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               ),
                    fit_intercept=np.random.choice(a=[True, False] if self.kwargs.get('fit_intercept_choice') is None else self.kwargs.get('fit_intercept_choice')),
                    selection=np.random.choice(a=['cyclic', 'random'] if self.kwargs.get('selection_choice') is None else self.kwargs.get('selection_choice'))
                    )

    def linear_regression(self) -> OLS:
        """
        Config Linear Regression (Ordinary Least Square)

        :return: OLS
            Model object
        """
        return OLS(endog=self.reg_params.get('endog'),
                   exog=self.reg_params.get('exog'),
                   missing='none' if self.reg_params.get('missing') is None else self.reg_params.get('missing'),
                   hasconst=self.reg_params.get('hasconst')
                   )

    @staticmethod
    def linear_regression_param():
        """
        Generate Linear Regression parameter configuration randomly
        """
        pass

    def random_forest(self) -> RandomForestRegressor:
        """
        Config Random Forest Regressor

        :return: RandomForestRegressor
            Model object
        """
        return RandomForestRegressor(n_estimators=100 if self.reg_params.get('n_estimators') is None else self.reg_params.get('n_estimators'),
                                     criterion='mse' if self.reg_params.get('criterion') is None else self.reg_params.get('criterion'),
                                     max_depth=1 if self.reg_params.get('max_depth') is None else self.reg_params.get('max_depth'),
                                     min_samples_split=2 if self.reg_params.get('min_samples_split') is None else self.reg_params.get('min_samples_split'),
                                     min_samples_leaf=1 if self.reg_params.get('min_samples_leaf') is None else self.reg_params.get('min_samples_leaf'),
                                     min_weight_fraction_leaf=0 if self.reg_params.get('min_weight_fraction_leaf') is None else self.reg_params.get('min_weight_fraction_leaf'),
                                     max_features='auto' if self.reg_params.get('max_features') is None else self.reg_params.get('max_features'),
                                     max_leaf_nodes=None if self.reg_params.get('max_leaf_nodes') is None else self.reg_params.get('max_leaf_nodes'),
                                     min_impurity_decrease=0 if self.reg_params.get('min_impurity_decrease') is None else self.reg_params.get('min_impurity_decrease'),
                                     min_impurity_split=None if self.reg_params.get('min_impurity_split') is None else self.reg_params.get('min_impurity_split'),
                                     bootstrap=True if self.reg_params.get('bootstrap') is None else self.reg_params.get('bootstrap'),
                                     oob_score=False if self.reg_params.get('oob_score') is None else self.reg_params.get('oob_score'),
                                     n_jobs=self.cpu_cores if self.reg_params.get('n_jobs') is None else self.reg_params.get('n_jobs'),
                                     random_state=self.seed,
                                     verbose=0 if self.reg_params.get('verbose') is None else self.reg_params.get('verbose'),
                                     warm_start=False if self.reg_params.get('warm_start') is None else self.reg_params.get('warm_start'),
                                     )

    def random_forest_param(self) -> dict:
        """
        Generate Random Forest regressor parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                                   high=100 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                                   ),
                    criterion=np.random.choice(a=['mse', 'mae'] if self.kwargs.get('criterion_choice') is None else self.kwargs.get('criterion_choice')),
                    max_depth=np.random.randint(low=1 if self.kwargs.get('max_depth_low') is None else self.kwargs.get('max_depth_low'),
                                                high=12 if self.kwargs.get('max_depth_high') is None else self.kwargs.get('max_depth_high')
                                                ),
                    min_samples_split=np.random.randint(low=2 if self.kwargs.get('min_samples_split_low') is None else self.kwargs.get('min_samples_split_low'),
                                                        high=6 if self.kwargs.get('min_samples_split_high') is None else self.kwargs.get('min_samples_split_high')
                                                        ),
                    min_samples_leaf=np.random.randint(low=1 if self.kwargs.get('min_samples_leaf_low') is None else self.kwargs.get('min_samples_leaf_low'),
                                                       high=6 if self.kwargs.get('min_samples_leaf_high') is None else self.kwargs.get('min_samples_leaf_high')
                                                       ),
                    bootstrap=np.random.choice(a=[True, False] if self.kwargs.get('bootstrap_choice') is None else self.kwargs.get('bootstrap_choice'))
                    )

    def support_vector_machine(self) -> SVR:
        """
        Config of the Support Vector Machine Regressor

        :return: SVR
            Model object
        """
        return SVR(C=1.0 if self.reg_params.get('C') is None else self.reg_params.get('C'),
                   kernel='rbf' if self.reg_params.get('kernel') is None else self.reg_params.get('kernel'),
                   degree=3 if self.reg_params.get('degree') is None else self.reg_params.get('degree'),
                   gamma='auto' if self.reg_params.get('gamma') is None else self.reg_params.get('gamma'),
                   coef0=0.0 if self.reg_params.get('coef0') is None else self.reg_params.get('coef0'),
                   tol=0.0001 if self.reg_params.get('tol') is None else self.reg_params.get('tol'),
                   epsilon=0.1 if self.reg_params.get('epsilon') is None else self.reg_params.get('epsilon'),
                   shrinking=True if self.reg_params.get('shrinking') is None else self.reg_params.get('shrinking'),
                   cache_size=200 if self.reg_params.get('cache_size') is None else self.reg_params.get('cache_size'),
                   verbose=False if self.reg_params.get('verbose') is None else self.reg_params.get('verbose'),
                   max_iter=-1 if self.reg_params.get('max_iter') is None else self.reg_params.get('max_iter')
                   )

    def support_vector_machine_param(self) -> dict:
        """
        Generate Support Vector Machine regressor parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(C=np.random.uniform(low=0.0001 if self.kwargs.get('C_low') is None else self.kwargs.get('C_low'),
                                        high=1.0 if self.kwargs.get('C_high') is None else self.kwargs.get('C_high')
                                        ),
                    kernel=np.random.choice(a=['rbf', 'linear', 'poly', 'sigmoid'] if self.kwargs.get('kernel_choice') is None else self.kwargs.get('kernel_choice')), #'precomputed'
                    #gamma=np.random.choice(a=['auto', 'scale']),
                    shrinking=np.random.choice(a=[True, False] if self.kwargs.get('shrinking_choice') is None else self.kwargs.get('shrinking_choice')),
                    cache_size=np.random.randint(low=100 if self.kwargs.get('cache_size_low') is None else self.kwargs.get('cache_size_low'),
                                                 high=500 if self.kwargs.get('cache_size_high') is None else self.kwargs.get('cache_size_high')
                                                 ),
                    decision_function_shape=np.random.choice(a=['ovo', 'ovr'] if self.kwargs.get('decision_function_shape_choice') is None else self.kwargs.get('decision_function_shape_choice')),
                    max_iter=np.random.randint(low=10 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=100 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               )
                    )

    def linear_support_vector_machine(self) -> LinearSVR:
        """
        Config of the Support Vector Machine Regressor

        :return: LinearSVR
            Model object
        """
        return LinearSVR(epsilon=0.0 if self.reg_params.get('epsilon') is None else self.reg_params.get('epsilon'),
                         tol=0.0001 if self.reg_params.get('tol') is None else self.reg_params.get('tol'),
                         C=1.0 if self.reg_params.get('C') is None else self.reg_params.get('C'),
                         loss='epsilon_insensitive' if self.reg_params.get('loss') is None else self.reg_params.get('loss'),
                         fit_intercept=True if self.reg_params.get('fit_intercept') is None else self.reg_params.get('fit_intercept'),
                         intercept_scaling=1 if self.reg_params.get('intercept_scaling') is None else self.reg_params.get('intercept_scaling'),
                         dual=True if self.reg_params.get('dual') is None else self.reg_params.get('dual'),
                         verbose=0 if self.reg_params.get('verbose') is None else self.reg_params.get('verbose'),
                         random_state=self.seed,
                         max_iter=1000 if self.reg_params.get('max_iter') is None else self.reg_params.get('max_iter'),
                         )

    def linear_support_vector_machine_param(self) -> dict:
        """
        Generate Linear Support Vector Machine regressor parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(C=np.random.uniform(low=0.0001 if self.kwargs.get('C_low') is None else self.kwargs.get('C_low'),
                                        high=1.0 if self.kwargs.get('C_high') is None else self.kwargs.get('C_high')
                                        ),
                    penalty=np.random.choice(a=['l1', 'l2'] if self.kwargs.get('penalty_choice') is None else self.kwargs.get('penalty_choice')),
                    loss=np.random.choice(a=['hinge', 'squared_hinge'] if self.kwargs.get('loss_choice') is None else self.kwargs.get('loss_choice')),
                    multi_class=np.random.choice(a=['ovr', 'crammer_singer'] if self.kwargs.get('multi_class_choice') is None else self.kwargs.get('multi_class_choice')),
                    max_iter=np.random.randint(low=10 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=100 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               )
                    )

    def nu_support_vector_machine(self) -> NuSVR:
        """
        Config of the Support Vector Machine Regressor

        :return: SVR
            Model object
        """
        return NuSVR(nu=0.5 if self.reg_params.get('nu') is None else self.reg_params.get('nu'),
                     C=1.0 if self.reg_params.get('C') is None else self.reg_params.get('C'),
                     kernel='rbf' if self.reg_params.get('kernel') is None else self.reg_params.get('kernel'),
                     degree=3 if self.reg_params.get('degree') is None else self.reg_params.get('degree'),
                     gamma='auto' if self.reg_params.get('gamma') is None else self.reg_params.get('gamma'),
                     coef0=0.0 if self.reg_params.get('coef0') is None else self.reg_params.get('coef0'),
                     tol=0.0001 if self.reg_params.get('tol') is None else self.reg_params.get('tol'),
                     shrinking=True if self.reg_params.get('shrinking') is None else self.reg_params.get('shrinking'),
                     cache_size=200 if self.reg_params.get('cache_size') is None else self.reg_params.get('cache_size'),
                     verbose=False if self.reg_params.get('verbose') is None else self.reg_params.get('verbose'),
                     max_iter=-1 if self.reg_params.get('max_iter') is None else self.reg_params.get('max_iter')
                     )

    def nu_support_vector_machine_param(self) -> dict:
        """
        Generate Nu-Support Vector Machine regressor parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(C=np.random.uniform(low=0.0001 if self.kwargs.get('C_low') is None else self.kwargs.get('C_low'),
                                        high=1.0 if self.kwargs.get('C_high') is None else self.kwargs.get('C_high')
                                        ),
                    nu=np.random.uniform(low=0.01 if self.kwargs.get('nu_low') is None else self.kwargs.get('nu_low'),
                                         high=0.99 if self.kwargs.get('nu_high') is None else self.kwargs.get('nu_high')
                                         ),
                    kernel=np.random.choice(a=['rbf', 'linear', 'poly', 'sigmoid'] if self.kwargs.get('kernel_choice') is None else self.kwargs.get('kernel_choice')), #'precomputed'
                    #gamma=np.random.choice(a=['auto', 'scale']),
                    shrinking=np.random.choice(a=[True, False] if self.kwargs.get('shrinking_choice') is None else self.kwargs.get('shrinking_choice')),
                    cache_size=np.random.randint(low=100 if self.kwargs.get('cache_size_low') is None else self.kwargs.get('cache_size_low'),
                                                 high=500 if self.kwargs.get('cache_size_high') is None else self.kwargs.get('cache_size_high')
                                                 ),
                    decision_function_shape=np.random.choice(a=['ovo', 'ovr'] if self.kwargs.get('decision_function_shape_choice') is None else self.kwargs.get('decision_function_shape_choice')),
                    max_iter=np.random.randint(low=10 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=100 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               )
                    )


class ModelGeneratorClf(Classification):
    """
    Class for generating supervised learning classification models
    """
    def __init__(self,
                 model_name: str = None,
                 clf_params: dict = None,
                 models: List[str] = None,
                 cpu_cores: int = 0,
                 seed: int = 1234,
                 **kwargs
                 ):
        """
        :param clf_params: dict
            Pre-configured classification model parameter

        :param models: List[str]
            Names of the possible models to sample from

        :param cpu_cores: int
            Number of CPU core to use

        :param seed: int
            Seed

        :param kwargs: dict
            Key-word arguments
        """
        super().__init__(clf_params=clf_params, cpu_cores=cpu_cores, seed=seed, **kwargs)
        self.id: int = 0
        self.fitness: dict = {}
        self.fitness_score: float = 0.0
        self.models: List[str] = models
        self.model_name: str = model_name
        if self.model_name is None:
            self.random: bool = True
            if self.models is not None:
                for model in self.models:
                    if model not in CLF_ALGORITHMS.keys():
                        self.random: bool = False
                        raise SupervisedMLException('Model ({}) is not supported. Supported classification models are: {}'.format(model, list(CLF_ALGORITHMS.keys())))
        else:
            if self.model_name not in CLF_ALGORITHMS.keys():
                raise SupervisedMLException('Model ({}) is not supported. Supported classification models are: {}'.format(self.model_name, list(CLF_ALGORITHMS.keys())))
            else:
                self.random: bool = False
        self.model = None
        self.model_param: dict = {}
        self.model_param_mutated: dict = {}
        self.model_param_mutation: str = ''
        self.features: List[str] = []
        self.target: str = ''
        self.train_time = None
        self.multi = None

    def generate_model(self) -> object:
        """
        Generate supervised machine learning model with randomized parameter configuration

        :return object
            Model object itself (self)
        """
        if self.random:
            if self.models is None:
                self.model_name = copy.deepcopy(np.random.choice(a=list(CLF_ALGORITHMS.keys())))
            else:
                self.model_name = copy.deepcopy(np.random.choice(a=self.models))
            _model = copy.deepcopy(CLF_ALGORITHMS.get(self.model_name))
        else:
            _model = copy.deepcopy(CLF_ALGORITHMS.get(self.model_name))
        if len(self.clf_params.keys()) == 0:
            self.model_param = getattr(Classification(), '{}_param'.format(_model))()
            self.clf_params = copy.deepcopy(self.model_param)
            _idx: int = 0 if len(self.model_param_mutated.keys()) == 0 else len(self.model_param_mutated.keys()) + 1
            self.model_param_mutated.update({str(_idx): {copy.deepcopy(self.model_name): {}}})
            for param in self.model_param.keys():
                self.model_param_mutated[str(_idx)][copy.deepcopy(self.model_name)].update({param: copy.deepcopy(self.model_param.get(param))})
        else:
            self.model_param = copy.deepcopy(self.clf_params)
        self.model_param_mutation = 'params'
        self.model = copy.deepcopy(getattr(Classification(clf_params=self.clf_params), _model)())
        return self

    def generate_params(self, param_rate: float = 0.1, force_param: dict = None) -> object:
        """
        Generate parameter for supervised learning models

        :param param_rate: float
            Rate of parameters of each model to mutate

        :param force_param: dict
            Parameter config to force explicitly

        :return object
            Model object itself (self)
        """
        if param_rate > 1:
            _rate: float = 1.0
        else:
            if param_rate > 0:
                _rate: float = param_rate
            else:
                _rate: float = 0.1
        _params: dict = getattr(Classification(), '{}_param'.format(CLF_ALGORITHMS.get(self.model_name)))()
        _force_param: dict = {} if force_param is None else force_param
        _param_choices: List[str] = [p for p in _params.keys() if p not in _force_param.keys()]
        _gen_n_params: int = round(len(_params.keys()) * _rate)
        if _gen_n_params == 0:
            _gen_n_params = 1
        self.model_param_mutated.update({len(self.model_param_mutated.keys()) + 1: {copy.deepcopy(self.model_name): {}}})
        _new_model_params: dict = copy.deepcopy(self.model_param)
        for param in _force_param.keys():
            _new_model_params.update({param: _force_param.get(param)})
        for _ in range(0, _gen_n_params, 1):
            _param: str = np.random.choice(a=_param_choices)
            _new_model_params.update({_param: _params.get(_param)})
            self.model_param_mutated[list(self.model_param_mutated.keys())[-1]][copy.deepcopy(self.model_name)].update({_param: _params.get(_param)})
        self.model_param_mutation = 'new_model'
        self.model_param = copy.deepcopy(_new_model_params)
        self.clf_params = self.model_param
        self.model = getattr(Classification(clf_params=self.clf_params), CLF_ALGORITHMS.get(self.model_name))()
        return self

    def get_model_parameter(self) -> dict:
        """
        Get parameter "standard" config of given regression models

        :return dict:
            Standard parameter config of given regression models
        """
        _model_param: dict = {}
        if self.models is None:
            return _model_param
        else:
            for model in self.models:
                if model in CLF_ALGORITHMS.keys():
                    _model = getattr(Classification(), CLF_ALGORITHMS.get(model))()
                    _param: dict = getattr(Classification(), '{}_param'.format(CLF_ALGORITHMS.get(model)))()
                    _model_random_param: dict = _model.__dict__.items()
                    for param in _model_random_param:
                        if param[0] in _param.keys():
                            _param.update({param[0]: param[1]})
                    _model_param.update({model: copy.deepcopy(_param)})
        return _model_param

    def eval(self, obs: np.array, pred: np.array, eval_metric: List[str] = None, train_error: bool = False):
        """
        Evaluate supervised machine learning classification model

        :param obs: np.array
            Observations (train or test data set)

        :param pred: np.array
            Predictions

        :param eval_metric: List[str]
            Name of metrics used for evaluation
                -> None: Area Under Curve (AUC) for binary classification
                -> None: Cohen's Kappa for Multi Classification
                -> auc: Area Under Curve (AUC)
                -> cohen_kappa: Cohen's Kappa

        :param train_error: bool
            Calculate train error or test error
        """
        if train_error:
            _error_kind: str = 'train'
        else:
            _error_kind: str = 'test'
        self.fitness.update({_error_kind: {}})
        self.multi: bool = len(pd.unique(obs)) > 2
        if eval_metric is None:
            if self.multi:
                _eval_metric: List[str] = [SML_SCORE['ml_metric'].get('clf_multi')]
            else:
                _eval_metric: List[str] = [SML_SCORE['ml_metric'].get('clf_binary')]
        else:
            if len(eval_metric) > 0:
                _eval_metric: List[str] = eval_metric
            else:
                if self.multi:
                    _eval_metric: List[str] = [SML_SCORE['ml_metric'].get('clf_multi')]
                else:
                    _eval_metric: List[str] = [SML_SCORE['ml_metric'].get('clf_binary')]
        for metric in _eval_metric:
            self.fitness[_error_kind].update({metric: copy.deepcopy(getattr(EvalClf(obs=obs, pred=pred, probability=False), metric)())})

    def predict(self, x: np.ndarray, probability: bool = False) -> np.array:
        """
        Get prediction from trained supervised machine learning model

        :param x: np.ndarray
            Test data set

        :param probability: bool
            Calculate probability or class score

        :return np.array: Prediction
        """
        if probability:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(x).flatten()
            else:
                raise SupervisedMLException('Model ({}) has no function called "predict_proba"'.format(self.model_name))
        else:
            if hasattr(self.model, 'predict'):
                return self.model.predict(x).flatten()
            else:
                raise SupervisedMLException('Model ({}) has no function called "predict"'.format(self.model_name))

    def train(self, x: np.ndarray, y: np.array, validation: dict = None):
        """
        Train or fit supervised machine learning model

        :param x: np.ndarray
            Train data set

        :param y: np.array
            Target data set

        :param validation: dict
        """
        _t0: datetime = datetime.now()
        if hasattr(self.model, 'fit'):
            if 'eval_set' in self.model.fit.__code__.co_varnames and validation is not None:
                #with joblib.parallel_backend(backend='dask'):
                if hasattr(self.model, 'fit_transform'):
                    self.model.fit_transform(x, y)
                else:
                    self.model.fit(x,
                                   y,
                                   eval_set=[(validation.get('x_val'), validation.get('y_val'))],
                                   early_stopping_rounds=np.random.randint(low=1, high=15) if self.model_param.get('early_stopping') else None,
                                   verbose=False
                                   )
            else:
                #with joblib.parallel_backend(backend='dask'):
                if hasattr(self.model, 'fit_transform'):
                    self.model.fit_transform(x, y)
                else:
                    self.model.fit(x, y)
        elif hasattr(self.model, 'train'):
            with joblib.parallel_backend(backend='dask'):
                self.model.train(x, y)
            #if 'validation_data' in self.model.fit.__code__.co_varnames:
            #     self.model.train(x, y, validation_data=[(validation.get('x_val'), validation.get('y_val'))])
            #else:
            #    self.model.train(x, y)
        else:
            raise SupervisedMLException('Training (fitting) method not supported by given model object')
        self.train_time = (datetime.now() - _t0).seconds
        self.multi = True if len(pd.unique(values=y)) > 2 else False
        #if hasattr(self.model, 'predict_proba'):
        #    self.eval(obs=y, pred=self.model.predict_proba(x), eval_metric=None, train_error=True)
        #else:
        #    if hasattr(self.model, 'predict'):
        self.eval(obs=y, pred=self.model.predict(x).flatten(), eval_metric=None, train_error=True)


class ModelGeneratorReg(Regression):
    """
    Class for generating supervised learning regression models
    """
    def __init__(self,
                 model_name: str = None,
                 reg_params: dict = None,
                 models: List[str] = None,
                 cpu_cores: int = 0,
                 seed: int = 1234,
                 **kwargs
                 ):
        """
        :param reg_params: dict
            Pre-configured regression model parameter

        :param models: List[str]
            Names of the possible models to sample from

        :param cpu_cores: int
            Number of CPU core to use

        :param seed: int
            Seed

        :param kwargs: dict
            Key-word arguments
        """
        super().__init__(reg_params=reg_params, cpu_cores=cpu_cores, seed=seed, **kwargs)
        self.id: int = 0
        self.fitness: dict = {}
        self.fitness_score: float = 0.0
        self.models: List[str] = models
        self.model_name: str = model_name
        if self.model_name is not None:
            if self.model_name not in REG_ALGORITHMS.keys():
                raise SupervisedMLException('Model ({}) is not supported. Supported regression models are: {}'.format(self.model_name, list(REG_ALGORITHMS.keys())))
            else:
                self.random: bool = False
        else:
            self.random: bool = True
        self.model = None
        self.model_param: dict = {}
        self.model_param_mutated: dict = {}
        self.model_param_mutation: str = ''
        self.train_time = None

    def generate_model(self) -> object:
        """
        Generate supervised machine learning model with randomized parameter configuration

        :return object
            Model object itself (self)
        """
        if self.random:
            if self.models is None:
                self.model_name = copy.deepcopy(np.random.choice(a=list(REG_ALGORITHMS.keys())))
            else:
                self.model_name = copy.deepcopy(np.random.choice(a=self.models))
            _model = copy.deepcopy(REG_ALGORITHMS.get(self.model_name))
        else:
            _model = copy.deepcopy(REG_ALGORITHMS.get(self.model_name))
        if len(self.reg_params.keys()) == 0:
            self.model_param = getattr(Regression(**self.kwargs), '{}_param'.format(_model))()
            self.reg_params = copy.deepcopy(self.model_param)
            _idx: int = 0 if len(self.model_param_mutated.keys()) == 0 else len(self.model_param_mutated.keys()) + 1
            self.model_param_mutated.update({str(_idx): {copy.deepcopy(self.model_name): {}}})
            for param in self.model_param.keys():
                self.model_param_mutated[str(_idx)][copy.deepcopy(self.model_name)].update(
                    {param: copy.deepcopy(self.model_param.get(param))})
        else:
            self.model_param = copy.deepcopy(self.reg_params)
        self.model_param_mutation = 'params'
        self.model = getattr(Regression(reg_params=self.reg_params, **self.kwargs), _model)()
        return self

    def generate_params(self, param_rate: float = 0.1, force_param: dict = None) -> object:
        """
        Generate parameter for supervised learning models

        :param param_rate: float
            Rate of parameters of each model to mutate

        :param force_param: dict
            Parameter config to force explicitly

        :return object
            Model object itself (self)
        """
        if param_rate > 1:
            _rate: float = 1.0
        else:
            if param_rate > 0:
                _rate: float = param_rate
            else:
                _rate: float = 0.1
        _params: dict = getattr(Regression(**self.kwargs), '{}_param'.format(REG_ALGORITHMS.get(self.model_name)))()
        _force_param: dict = {} if force_param is None else force_param
        _param_choices: List[str] = [p for p in _params.keys() if p not in _force_param.keys()]
        _gen_n_params: int = round(len(_params.keys()) * _rate)
        if _gen_n_params == 0:
            _gen_n_params = 1
        self.model_param_mutated.update(
            {len(self.model_param_mutated.keys()) + 1: {copy.deepcopy(self.model_name): {}}})
        _new_model_params: dict = copy.deepcopy(self.model_param)
        for param in _force_param.keys():
            _new_model_params.update({param: _force_param.get(param)})
        for _ in range(0, _gen_n_params, 1):
            _param: str = np.random.choice(a=_param_choices)
            _new_model_params.update({_param: _params.get(_param)})
            self.model_param_mutated[list(self.model_param_mutated.keys())[-1]][copy.deepcopy(self.model_name)].update(
                {_param: _params.get(_param)})
        #print('old', self.model_param)
        #print('new', _new_model_params)
        self.model_param_mutation = 'new_model'
        self.model_param = copy.deepcopy(_new_model_params)
        self.reg_params = self.model_param
        self.model = getattr(Regression(reg_params=self.reg_params, **self.kwargs), REG_ALGORITHMS.get(self.model_name))()
        return self

    def get_model_parameter(self) -> dict:
        """
        Get parameter "standard" config of given regression models

        :return dict:
            Standard parameter config of given regression models
        """
        _model_param: dict = {}
        if self.models is None:
            return _model_param
        else:
            for model in self.models:
                if model in REG_ALGORITHMS.keys():
                    _model = getattr(Regression(**self.kwargs), REG_ALGORITHMS.get(model))()
                    _param: dict = getattr(Regression(**self.kwargs), '{}_param'.format(REG_ALGORITHMS.get(model)))()
                    _model_random_param: dict = _model.__dict__.items()
                    for param in _model_random_param:
                        if param[0] in _param.keys():
                            _param.update({param[0]: param[1]})
                    _model_param.update({model: copy.deepcopy(_param)})
        return _model_param

    def eval(self, obs: np.array, pred: np.array, eval_metric: List[str] = None, train_error: bool = False):
        """
        Evaluate supervised machine learning regression models

        :param obs: np.array
            Observations

        :param pred: np.array
            Predictions

        :param eval_metric: List[str]
            Name of metrics used for evaluation
                -> None: Normalized Root-Mean-Squared Error

        :param train_error: bool
            Evaluate training error or testing error
        """
        if train_error:
            _error_kind: str = 'train'
        else:
            _error_kind: str = 'test'
        self.fitness.update({_error_kind: {}})
        if eval_metric is None:
            _eval_metric: List[str] = ['rmse_norm']
        else:
            if len(eval_metric) > 0:
                _eval_metric: List[str] = eval_metric
            else:
                _eval_metric: List[str] = ['rmse_norm']
        for metric in _eval_metric:
            self.fitness[_error_kind].update({metric: getattr(EvalReg(obs=obs, pred=pred), metric)()})

    def predict(self, x) -> np.array:
        """
        Get prediction from trained supervised machine learning model

        :parma x: np.array
            Test data set

        :return np.array
            Predictions
        """
        if hasattr(self.model, 'predict'):
            return self.model.predict(x)
        else:
            raise SupervisedMLException('Model ({}) has no function called "predict"'.format(self.model_name))

    def train(self, x: np.ndarray, y: np.array, validation: dict = None):
        """
        Train or fit supervised machine learning model

        :param x: np.ndarray
            Train data set

        :param y: np.array
            Target data set

        :param validation: dict
        """
        _t0: datetime = datetime.now()
        if hasattr(self.model, 'fit'):
            #with joblib.parallel_backend(backend='dask'):
            if 'eval_set' in self.model.fit.__code__.co_varnames and validation is not None:
                self.model.fit(x,
                               y,
                               eval_set=[(validation.get('x_val'), validation.get('y_val'))],
                               early_stopping_rounds=np.random.randint(low=1, high=15) if self.model_param.get('early_stopping') else None,
                               verbose=False
                               )
            else:
                if hasattr(self.model, 'fit_transform'):
                    self.model.fit_transform(x, y)
                else:
                    self.model.fit(x, y)
        elif hasattr(self.model, 'train'):
            #with joblib.parallel_backend(backend='dask'):
            self.model.train(x, y)
        else:
            raise SupervisedMLException('Training (fitting) method not supported by given model object')
        self.train_time = (datetime.now() - _t0).seconds
        self.eval(obs=y, pred=self.model.predict(x), eval_metric=None, train_error=True)
