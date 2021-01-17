import copy
import numpy as np
import os
import unittest

from happy_learning.feature_engineer import FeatureEngineer, PROCESSING_ACTION_SPACE, SUPPORTED_TYPES
from typing import Dict, List

DATA_FILE_PATH: str = 'data/avocado.csv'
FEATURE_ENGINEER_FILE_PATH: str = 'data/feature_engineer.p'
FEATURE_ENGINEER: FeatureEngineer = FeatureEngineer(df=None,
                                                    file_path=DATA_FILE_PATH,
                                                    target_feature='AveragePrice',
                                                    generate_new_feature=True,
                                                    keep_original_data=True,
                                                    unify_invalid_values=True,
                                                    encode_missing_data=False,
                                                    max_level_processing=5,
                                                    activate_actor=False,
                                                    missing_value_analysis=True,
                                                    auto_cleaning=False,
                                                    auto_typing=True,
                                                    auto_engineering=False,
                                                    auto_text_mining=False,
                                                    seed=1234,
                                                    partitions=4
                                                    )


def _check_feature_orchestra(meth: str, features: List[str]) -> bool:
    """
    Check internal FeatureOrchestra decorator for the FeatureEngineer class

    :param meth: str
        Name of the method

    :param features: List[str]
        Name of the features

    :return: bool
        FeatureOrchestra check passed or failed
    """
    return True


def _check_tracking(meth: str, suffix: str, feature_type: str) -> Dict[str, bool]:
    """
    Check internal tracking framework of the FeatureEngineer class

    :param meth: str
        Name of the method

    :param suffix: str
        Suffix

    :param feature_type: str
        Name of the feature type

    :return: Dict[str, bool]
        Tracking results:
            -> Process tracking
            -> Feature relation tracking (raw)
            -> Feature relation tracking (level)
    """
    _features: List[str] = copy.deepcopy(FEATURE_ENGINEER.get_features(feature_type=feature_type))
    _found_tracked_processes: List[bool] = []
    _found_tracked_feature_relation_raw: List[bool] = []
    _found_tracked_feature_relation_level: List[bool] = []
    _process_tracking: dict = FEATURE_ENGINEER.get_data_processing()['processing']['process']
    _feature_relation_tracking: dict = FEATURE_ENGINEER.get_data_processing()['processing']['features']
    for i, feature in enumerate(_features, start=1):
        if _process_tracking[str(i)]['meth'] == meth and list(_process_tracking[str(i)]['features'].keys())[0] == '{}_{}'.format(feature, suffix) and _process_tracking[str(i)]['features']['{}_{}'.format(feature, suffix)]:
            _found_tracked_processes.append(True)
        else:
            _found_tracked_processes.append(False)
        if '{}_{}'.format(feature, suffix) in FEATURE_ENGINEER.get_cleaned_features():
            continue
        if feature in _feature_relation_tracking['raw'].keys():
            if _feature_relation_tracking['raw'][feature][0] == '{}_{}'.format(feature, suffix):
                _found_tracked_feature_relation_raw.append(True)
            else:
                _found_tracked_feature_relation_raw.append(False)
        else:
            _found_tracked_feature_relation_raw.append(False)
        if '{}_{}'.format(feature, suffix) in _feature_relation_tracking['level_1'].keys():
            if len(_feature_relation_tracking['level_1']['{}_{}'.format(feature, suffix)]) == 0:
                _found_tracked_feature_relation_level.append(True)
            else:
                _found_tracked_feature_relation_level.append(False)
        else:
            _found_tracked_feature_relation_level.append(False)
        return {'process': all(_found_tracked_processes),
                'raw': all(_found_tracked_feature_relation_raw),
                'level': all(_found_tracked_feature_relation_level)
                }


class FeatureEngineerTest(unittest.TestCase):
    """
    Class for testing class FeatureEngineer
    """
    def test_act(self):
        FEATURE_ENGINEER.activate_actor()
        FEATURE_ENGINEER.act(actor=FEATURE_ENGINEER.get_features(feature_type='continuous')[0],
                             inter_actors=FEATURE_ENGINEER.get_features(feature_type='continuous')
                             )

    def test_activate_actor(self):
        _data_processing: dict = FEATURE_ENGINEER.get_data_processing()
        _before_activation: bool = _data_processing.get('activate_actor')
        FEATURE_ENGINEER.activate_actor()
        self.assertTrue(expr=FEATURE_ENGINEER.get_data_processing().get('activate_actor') and not _before_activation)

    def test_active_since(self):
        self.assertTrue(expr=len(FEATURE_ENGINEER.active_since().split(': ')[1]) > 0)

    def test_auto_cleaning(self):
        _n_cases: int = FEATURE_ENGINEER.get_n_cases()
        _n_features: int = FEATURE_ENGINEER.get_n_features()
        FEATURE_ENGINEER.auto_cleaning(missing_data=True,
                                       missing_data_threshold=0.999,
                                       invariant=True,
                                       duplicated_cases=True,
                                       duplicated_features=True,
                                       unstable=True
                                       )
        self.assertTrue(expr=_n_cases >= FEATURE_ENGINEER.get_n_cases() and _n_features >= FEATURE_ENGINEER.get_n_features())

    def test_auto_engineering(self):
        _n_features: int = FEATURE_ENGINEER.get_n_features()
        FEATURE_ENGINEER.auto_engineering(label_enc=True,
                                          interaction=True,
                                          disparity=True,
                                          time_disparity=True,
                                          one_hot_enc=True,
                                          geo_enc=True,
                                          geo_features=None,
                                          date_discretizing=True,
                                          binning=False,
                                          bins=4,
                                          scaling=True,
                                          log_transform=True,
                                          exp_transform=True,
                                          handle_missing_data='impute'
                                          )
        self.assertTrue(expr=_n_features < FEATURE_ENGINEER.get_n_features())

    def test_auto_typing(self):
        pass

    def test_binarizer(self):
        pass

    def test_binning(self):
        _bin_supervised: str = 'supervised'
        _bin_unsupervised: str = 'bins'
        _bin_optimal_bayesian_blocks: str = 'blocks'
        #_bin_optimal_kbins: str = 'kbins'
        #_bin_optimal_chaid: str = 'chaid'
        _raw_categorical_features: List[str] = FEATURE_ENGINEER.get_features(feature_type='categorical')
        _raw_continuous_features: List[str] = FEATURE_ENGINEER.get_features(feature_type='continuous')
        _found_new_feature: List[bool] = []
        FEATURE_ENGINEER.binning(supervised=True,
                                 edges=[0, 1, 2, 4],
                                 bins=None,
                                 features=None,
                                 optimal=False,
                                 optimal_meth='bayesian_blocks',
                                 predictors=None,
                                 weight_feature=None,
                                 labels=None,
                                 encode_meth='onehot',
                                 strategy='quantile'
                                 )
        for raw in _raw_continuous_features:
            if '{}_{}'.format(raw, _bin_supervised) in FEATURE_ENGINEER.get_features(feature_type='categorical'):
                _found_new_feature.append(True)
            else:
                _found_new_feature.append(False)
        print(FEATURE_ENGINEER.get_data_processing())
        FEATURE_ENGINEER.binning(supervised=False,
                                 edges=None,
                                 bins=4,
                                 features=None,
                                 optimal=False,
                                 optimal_meth='bayesian_blocks',
                                 predictors=None,
                                 weight_feature=None,
                                 labels=None,
                                 encode_meth='onehot',
                                 strategy='quantile'
                                 )
        for raw in _raw_continuous_features:
            if '{}_{}'.format(raw, _bin_unsupervised) in FEATURE_ENGINEER.get_features(feature_type='categorical'):
                _found_new_feature.append(True)
            else:
                _found_new_feature.append(False)
        FEATURE_ENGINEER.binning(supervised=True,
                                 edges=None,
                                 bins=None,
                                 features=None,
                                 optimal=True,
                                 optimal_meth='bayesian_blocks',
                                 predictors=None,
                                 weight_feature=None,
                                 labels=None,
                                 encode_meth='onehot',
                                 strategy='quantile'
                                 )
        for raw in _raw_continuous_features:
            if '{}_{}'.format(raw, _bin_optimal_bayesian_blocks) in FEATURE_ENGINEER.get_features(feature_type='categorical'):
                _found_new_feature.append(True)
            else:
                _found_new_feature.append(False)
        #FEATURE_ENGINEER.binning(supervised=True,
        #                         edges=None,
        #                         bins=3,
        #                         features=None,
        #                         optimal=True,
        #                         optimal_meth='kbins',
        #                         predictors=None,
        #                         weight_feature=None,
        #                         labels=None,
        #                         encode_meth='onehot',
        #                         strategy='quantile'
        #                         )
        #for raw in _raw_continuous_features:
        #    if '{}_{}'.format(raw, _bin_optimal_kbins) in FEATURE_ENGINEER.get_features(feature_type='categorical'):
        #        _found_new_feature.append(True)
        #    else:
        #        _found_new_feature.append(False)
        #FEATURE_ENGINEER.binning(supervised=True,
        #                         edges=None,
        #                         bins=None,
        #                         features=None,
        #                         optimal=True,
        #                         optimal_meth='chaid',
        #                         predictors=None,
        #                         weight_feature=None,
        #                         labels=None,
        #                         encode_meth='onehot',
        #                         strategy='quantile'
        #                         )
        #for raw in _raw_continuous_features:
        #    if '{}_{}'.format(raw, _bin_optimal_chaid) in FEATURE_ENGINEER.get_features(feature_type='categorical'):
        #        _found_new_feature.append(True)
        #    else:
        #        _found_new_feature.append(False)
        self.assertTrue(expr=all(_found_new_feature))

    def test_box_cox_transform(self):
        pass

    def test_breakdown_stats(self):
        pass

    def test_clean(self):
        _n_features: int = FEATURE_ENGINEER.get_n_features()
        FEATURE_ENGINEER.clean(markers=dict(features=['type']))
        self.assertTrue(expr=_n_features > FEATURE_ENGINEER.get_n_features())

    def test_clean_nan(self):
        FEATURE_ENGINEER.exp_transform()
        FEATURE_ENGINEER.exp_transform()
        FEATURE_ENGINEER.exp_transform()
        _n_cases: int = FEATURE_ENGINEER.get_n_cases()
        FEATURE_ENGINEER.clean_nan(other_mis=None)
        print(FEATURE_ENGINEER.get_data(dask_df=False)['Total Bags_exp_exp'])
        self.assertTrue(expr=_n_cases > FEATURE_ENGINEER.get_n_cases())

    def test_clean_unstable_features(self):
        pass

    def test_concat_text(self):
        pass

    def test_count_text(self):
        pass

    def test_data_export(self):
        _data_file_path: str = 'data/test_data_export.parquet'
        FEATURE_ENGINEER.data_export(file_path=_data_file_path, create_dir=False, overwrite=True)
        if _data_file_path.find('.parquet') >= 0:
            self.assertTrue(expr=os.path.isdir(_data_file_path))
        else:
            self.assertTrue(expr=os.path.isfile(_data_file_path))

    def test_data_import(self):
        _feature_engineer = FeatureEngineer(file_path='data/avocado.csv')
        self.assertTrue(expr=_feature_engineer.get_n_cases() > 0)

    def test_date_categorizer(self):
        FEATURE_ENGINEER.date_categorizer()
        _tracking_check_year: Dict[str, bool] = _check_tracking(meth='date_categorizer', suffix='year', feature_type='date')
        _tracking_check_month: Dict[str, bool] = _check_tracking(meth='date_categorizer', suffix='month', feature_type='date')
        _tracking_check_day: Dict[str, bool] = _check_tracking(meth='date_categorizer', suffix='day', feature_type='date')
        _tracking_check_hour: Dict[str, bool] = _check_tracking(meth='date_categorizer', suffix='hour', feature_type='date')
        _tracking_check_minute: Dict[str, bool] = _check_tracking(meth='date_categorizer', suffix='minute', feature_type='date')
        _tracking_check_second: Dict[str, bool] = _check_tracking(meth='date_categorizer', suffix='second', feature_type='date')
        _tracking_check: Dict[str, bool] = {'process': all([_tracking_check_year.get('process'),
                                                            _tracking_check_month.get('process'),
                                                            _tracking_check_day.get('process'),
                                                            _tracking_check_hour.get('process'),
                                                            _tracking_check_minute.get('process'),
                                                            _tracking_check_second.get('process')
                                                            ]),
                                            'raw': all([_tracking_check_year.get('raw'),
                                                        _tracking_check_month.get('raw'),
                                                        _tracking_check_day.get('raw'),
                                                        _tracking_check_hour.get('raw'),
                                                        _tracking_check_minute.get('raw'),
                                                        _tracking_check_second.get('raw')
                                                        ]),
                                            'level': all([_tracking_check_year.get('level'),
                                                          _tracking_check_month.get('level'),
                                                          _tracking_check_day.get('level'),
                                                          _tracking_check_hour.get('level'),
                                                          _tracking_check_minute.get('level'),
                                                          _tracking_check_second.get('level')
                                                          ])
                                            }
        self.assertTrue(expr=_tracking_check.get('process') and _tracking_check.get('raw') and _tracking_check.get('level'))

    def test_date_conversion(self):
        pass

    def test_disparity(self):
        pass

    def test_disparity_time(self):
        pass

    def test_exp_transform(self):
        FEATURE_ENGINEER.exp_transform()
        _tracking_check: Dict[str, bool] = _check_tracking(meth='exp_transform', suffix='exp', feature_type='continuous')
        self.assertTrue(expr=_tracking_check.get('process') and _tracking_check.get('raw') and _tracking_check.get('level'))

    def test_geo_encoder(self):
        pass

    def test_geo_breakdown_stats(self):
        pass

    def test_get_action_space(self):
        _found_action: List[bool] = []
        for feature_type in FEATURE_ENGINEER.get_feature_types().keys():
            if feature_type in FEATURE_ENGINEER.get_action_space().keys():
                if isinstance(FEATURE_ENGINEER.get_action_space()[feature_type], dict):
                    for action in FEATURE_ENGINEER.get_action_space()[feature_type].keys():
                        if len(FEATURE_ENGINEER.get_action_space()[feature_type][action]) > 0:
                            _found_action.append(True)
                        else:
                            _found_action.append(False)
                elif isinstance(FEATURE_ENGINEER.get_action_space()[feature_type], list):
                    if len(FEATURE_ENGINEER.get_action_space()[feature_type]) > 0:
                        _found_action.append(True)
                    else:
                        _found_action.append(False)
            else:
                _found_action.append(False)
        self.assertTrue(expr=all(_found_action))

    def test_get_actor_memory(self):
        pass

    def test_get_data(self):
        _dask_df = FEATURE_ENGINEER.get_data(dask_df=True)
        _pandas_df = FEATURE_ENGINEER.get_data(dask_df=False)
        self.assertTrue(expr=len(_dask_df) == _pandas_df.shape[0] > 0)

    def test_get_data_info(self):
        pass

    def test_get_data_processing(self):
        _found_tracking_items: List[bool] = []
        _tracking_items: List[str] = ['process',
                                      'features',
                                      'encoder',
                                      'scaler',
                                      'interaction',
                                      'self_interaction',
                                      'mapper',
                                      'imp',
                                      'clean',
                                      'names',
                                      'text'
                                      ]
        for track in _tracking_items:
            if isinstance(FEATURE_ENGINEER.get_data_processing().get(track), dict):
                _found_tracking_items.append(True)
            else:
                _found_tracking_items.append(False)
        self.assertTrue(expr=FEATURE_ENGINEER.get_data_processing())

    def test_get_data_source(self):
        self.assertListEqual(list1=[DATA_FILE_PATH], list2=FEATURE_ENGINEER.get_data_source())

    def test_get_correlation(self):
        _found_cor: List[bool] = []
        _cor: dict = FEATURE_ENGINEER.get_correlation()
        for feature in FEATURE_ENGINEER.get_features(feature_type='continuous'):
            if feature in list(_cor.get('matrix').columns) and feature in _cor.get('matrix').index.values.tolist():
                _found_cor.append(True)
            else:
                _found_cor.append(False)
        self.assertTrue(expr=all(_found_cor))

    def test_get_features(self):
        self.assertEqual(first=FEATURE_ENGINEER.get_n_features(), second=len(FEATURE_ENGINEER.get_features(feature_type=None)))

    def test_get_feature_types(self):
        _found_feature_type: List[bool] = []
        for ft in FEATURE_ENGINEER.get_feature_types().keys():
            if ft == 'categorical':
                if len(FEATURE_ENGINEER.get_features(feature_type=ft)) > 0:
                    _found_feature_type.append(True)
                else:
                    _found_feature_type.append(False)
            elif ft == 'ordinal':
                if len(FEATURE_ENGINEER.get_features(feature_type=ft)) == 0:
                    _found_feature_type.append(True)
                else:
                    _found_feature_type.append(False)
            elif ft == 'continuous':
                if len(FEATURE_ENGINEER.get_features(feature_type=ft)) > 0:
                    _found_feature_type.append(True)
                else:
                    _found_feature_type.append(False)
            elif ft == 'date':
                if len(FEATURE_ENGINEER.get_features(feature_type=ft)) > 0:
                    _found_feature_type.append(True)
                else:
                    _found_feature_type.append(False)
            elif ft == 'id_text':
                if len(FEATURE_ENGINEER.get_features(feature_type=ft)) == 0:
                    _found_feature_type.append(True)
                else:
                    _found_feature_type.append(False)
        self.assertTrue(expr=all(_found_feature_type))

    def test_get_feature_values(self):
        self.assertTrue(expr=len(FEATURE_ENGINEER.get_feature_values(feature='AveragePrice', unique=False)) == FEATURE_ENGINEER.get_n_cases())

    def test_get_last_action(self):
        pass

    def test_get_last_generated_feature(self):
        FEATURE_ENGINEER.scaling_minmax(features=['4046'])
        self.assertEqual(first='4046_minmax', second=FEATURE_ENGINEER.get_last_generated_feature())

    def test_get_max_processing_level(self):
        self.assertEqual(first=5, second=FEATURE_ENGINEER.get_max_processing_level())

    def test_get_missing_data(self):
        FEATURE_ENGINEER.reset_target(targets=None)
        _mva: dict = FEATURE_ENGINEER.get_missing_data(freq=True)
        _mva_results: List[str] = ['total', 'cases', 'features']
        _found_mva_results: List[bool] = []
        for m in _mva.keys():
            if m in _mva_results:
                _found_mva_results.append(True)
            else:
                _found_mva_results.append(False)
        self.assertTrue(expr=all(_found_mva_results) and _mva['total']['mis'] == 0 and len(list(_mva['cases'].keys())) == FEATURE_ENGINEER.get_n_cases() and len(list(_mva['features'].keys())) == FEATURE_ENGINEER.get_n_features())

    def test_get_n_cases(self):
        self.assertEqual(first=FEATURE_ENGINEER.get_n_features(), second=len(FEATURE_ENGINEER.get_features(feature_type=None)))

    def test_get_n_features(self):
        self.assertEqual(first=13, second=FEATURE_ENGINEER.get_n_features())

    def test_get_n_predictors(self):
        self.assertEqual(first=18249, second=FEATURE_ENGINEER.get_n_cases())

    def test_get_n_target_values(self):
        FEATURE_ENGINEER.reset_target(targets=None)
        FEATURE_ENGINEER.set_target(feature='type')
        self.assertEqual(first=2, second=FEATURE_ENGINEER.get_n_target_values())

    def test_get_notes(self):
        _note: str = 'happy ;) learning'
        FEATURE_ENGINEER.write_notepad(note=_note, page='Unittest', append=True, add_time_stamp=True)
        self.assertEqual(first=_note, second=FEATURE_ENGINEER.get_notes(page='Unittest'))

    def test_get_pages(self):
        _page: str = 'Unittest'
        FEATURE_ENGINEER.write_notepad(note='happy ;) learning', page=_page, append=True, add_time_stamp=True)
        self.assertListEqual(list1=[_page], list2=FEATURE_ENGINEER.get_pages())

    def test_get_predictors(self):
        FEATURE_ENGINEER.set_predictors(exclude_original_data=False)
        self.assertEqual(first=8, second=len(FEATURE_ENGINEER.get_predictors()))

    def test_get_obj_source(self):
        self.assertEqual(first='', second=FEATURE_ENGINEER.get_obj_source())

    def test_get_supported_types(self):
        self.assertDictEqual(d1=SUPPORTED_TYPES, d2=FEATURE_ENGINEER.get_supported_types(data_type=None))

    def test_get_indices(self):
        self.assertEqual(first=FEATURE_ENGINEER.get_n_cases(), second=len(FEATURE_ENGINEER.get_indices()))

    def test_get_processing(self):
        self.assertDictEqual(d1=FEATURE_ENGINEER.get_data_processing()['processing'], d2=FEATURE_ENGINEER.get_processing())

    def test_get_processing_action_space(self):
        self.assertDictEqual(d1=PROCESSING_ACTION_SPACE, d2=FEATURE_ENGINEER.get_processing_action_space())

    def test_get_processing_relation(self):
        pass

    def test_get_target(self):
        self.assertTrue(expr='AveragePrice' == FEATURE_ENGINEER.get_target())

    def test_get_target_labels(self):
        FEATURE_ENGINEER.set_target(feature='type')
        print(FEATURE_ENGINEER.get_target_labels())

    def test_get_target_values(self):
        FEATURE_ENGINEER.set_target(feature='type')
        self.assertEqual(first=2, second=len(FEATURE_ENGINEER.get_target_values()))

    def test_get_text_miner(self):
        print(FEATURE_ENGINEER.get_text_miner())

    def test_get_training_data(self):
        FEATURE_ENGINEER.set_predictors(exclude_original_data=False)
        self.assertEqual(first=9, second=len(FEATURE_ENGINEER.get_training_data(output='df_dask').columns))

    def test_get_transformations(self):
        _transformations: dict = dict(encoder=['bin', 'label', 'one_hot'],
                                      scaler=['min_max', 'robust', 'normal', 'standard', 'box_cox', 'log', 'exp'],
                                      mapper=[],
                                      naming=[],
                                      binning_continuous=[],
                                      binning_date=[]
                                      )
        _found_transformation: List[bool] = []
        print(FEATURE_ENGINEER.get_transformations())
        _observed_transformation: dict = FEATURE_ENGINEER.get_transformations(transformation=None)
        for transformation in _transformations.keys():
            if transformation in list(_observed_transformation.keys()):
                _found_transformation.append(True)
                if len(_transformations.get(transformation)) > 0:
                    for meth in _transformations.get(transformation):
                        if meth in list(_observed_transformation[transformation][meth].keys()):
                            _found_transformation.append(True)
                        else:
                            _found_transformation.append(False)
            else:
                _found_transformation.append(False)
        self.assertTrue(expr=all(_found_transformation))

    def test_interaction(self):
        pass

    def test_interaction_poly(self):
        pass

    def test_impute(self):
        pass

    def test_is_unstable(self):
        pass

    def test_label_encoder(self):
        _unique_feature_labels = FEATURE_ENGINEER.get_feature_values(feature='type', unique=True)
        _found_labels: List[bool] = [True if type(label) == str else False for label in _unique_feature_labels]
        FEATURE_ENGINEER.label_encoder(encode=True, features=['type'])
        _found_values: List[bool] = [True if type(value) == np.int64 or type(value) == np.int32 else False for value in FEATURE_ENGINEER.get_feature_values(feature='type', unique=True)]
        self.assertTrue(expr=all(_found_labels) and all(_found_values))

    def test_linguistic_features(self):
        pass

    def test_load(self):
        pass

    def test_log_transform(self):
        FEATURE_ENGINEER.log_transform()
        _tracking_check: Dict[str, bool] = _check_tracking(meth='log_transform', suffix='log', feature_type='continuous')
        self.assertTrue(expr=_tracking_check.get('process') and _tracking_check.get('raw') and _tracking_check.get('level'))

    def test_normalizer(self):
        FEATURE_ENGINEER.normalizer()
        _tracking_check: Dict[str, bool] = _check_tracking(meth='exp_transform', suffix='exp', feature_type='continuous')
        self.assertTrue(expr=_tracking_check.get('process') and _tracking_check.get('raw') and _tracking_check.get('level'))

    def test_merge_engineer(self):
        _all_features: List[str] = FEATURE_ENGINEER.get_features()
        _all_features.sort(reverse=False)
        _engineer: FeatureEngineer = FeatureEngineer(df=None, file_path=DATA_FILE_PATH, target_feature='AveragePrice')
        _categorical_features: List[str] = _engineer.get_feature_types().get('categorical')
        _engineer.save(file_path='data/feature_learning_cat.p', cls_obj=True, overwrite=True, create_dir=False)
        del _engineer
        _feature_engineer: FeatureEngineer = FeatureEngineer(feature_engineer_file_path='data/feature_learning_cat.p')
        _feature_engineer.clean(markers=dict(features=_categorical_features))
        _feature_engineer.merge_engineer(feature_engineer_file_path='data/feature_learning_cat.p')
        _features: List[str] = _feature_engineer.get_features()
        _features.sort(reverse=False)
        self.assertListEqual(list1=_all_features, list2=_features)

    def test_melt(self):
        pass

    def test_merge_text(self):
        pass

    def test_missing_data_analysis(self):
        pass

    def test_one_hot_encoder(self):
        FEATURE_ENGINEER.one_hot_encoder()
        _tracking_check: Dict[str, bool] = _check_tracking(meth='one_hot_encoder', suffix='', feature_type='continuous')
        self.assertTrue(expr=_tracking_check.get('process') and _tracking_check.get('raw') and _tracking_check.get('level'))

    def test_outlier_detection(self):
        pass

    def test_re_engineer(self):
        pass

    def test_replacer(self):
        pass

    def test_reset_feature_processing_relation(self):
        pass

    def test_reset_original_data_set(self):
        pass

    def test_reset_predictors(self):
        FEATURE_ENGINEER.set_predictors(exclude_original_data=False)
        _predictors: List[str] = FEATURE_ENGINEER.get_predictors()
        FEATURE_ENGINEER.reset_predictors()
        self.assertTrue(expr=len(FEATURE_ENGINEER.get_predictors()) == 0 and len(_predictors) > 0)

    def test_reset_ignore_processing(self):
        pass

    def test_reset_multi_threading(self):
        pass

    def test_reset_target(self):
        FEATURE_ENGINEER.reset_target()
        self.assertEqual(first=None, second=FEATURE_ENGINEER.get_target())

    def test_rounding(self):
        FEATURE_ENGINEER.rounding()
        _tracking_check: Dict[str, bool] = _check_tracking(meth='rounding', suffix='round_100', feature_type='continuous')
        self.assertTrue(expr=_tracking_check.get('process') and _tracking_check.get('raw') and _tracking_check.get('level'))

    def test_save(self):
        FEATURE_ENGINEER.save(file_path='data/feature_engineer.p', cls_obj=True, overwrite=True, create_dir=False)
        self.assertTrue(expr=os.path.isfile('data/feature_engineer.p') and os.path.isdir('data/feature_engineer_data.parquet'))

    def test_sampler(self):
        pass

    def test_scaling_robust(self):
        FEATURE_ENGINEER.scaling_robust()
        _tracking_check: Dict[str, bool] = _check_tracking(meth='scaling_robust', suffix='robust', feature_type='continuous')
        self.assertTrue(expr=_tracking_check.get('process') and _tracking_check.get('raw') and _tracking_check.get('level'))

    def test_scaling_minmax(self):
        FEATURE_ENGINEER.scaling_minmax()
        _tracking_check: Dict[str, bool] = _check_tracking(meth='scaling_minmax', suffix='minmax', feature_type='continuous')
        self.assertTrue(expr=_tracking_check.get('process') and _tracking_check.get('raw') and _tracking_check.get('level'))

    def test_self_interaction(self):
        pass

    def test_set_back_up_data(self):
        pass

    def test_set_critic_config(self):
        pass

    def test_set_data(self):
        pass

    def test_set_feature_names(self):
        pass

    def test_set_feature_processing_relation(self):
        pass

    def test_set_feature_types(self):
        pass

    def test_set_index(self):
        pass

    def test_set_processing_level(self):
        FEATURE_ENGINEER.set_max_processing_level(level=2)
        self.assertEqual(first=2, second=FEATURE_ENGINEER.get_max_processing_level())

    def test_set_predictors(self):
        _predictors: List[str] = []
        FEATURE_ENGINEER.set_predictors(exclude_original_data=False)
        self.assertTrue(expr=len(_predictors) < len(FEATURE_ENGINEER.get_predictors()))

    def test_set_ignore_processing(self):
        pass

    def test_set_imp_features(self):
        pass

    def test_set_target(self):
        FEATURE_ENGINEER.reset_target(targets=None)
        _target_feature: str = 'type'
        FEATURE_ENGINEER.set_target(feature=_target_feature)
        self.assertEqual(first=_target_feature, second=FEATURE_ENGINEER.get_target())

    def test_sort(self):
        pass

    def test_splitter(self):
        pass

    def test_square_root_transform(self):
        FEATURE_ENGINEER.square_root_transform()
        _tracking_check: Dict[str, bool] = _check_tracking(meth='square_root_transform', suffix='square', feature_type='continuous')
        self.assertTrue(expr=_tracking_check.get('process') and _tracking_check.get('raw') and _tracking_check.get('level'))

    def test_standardizer(self):
        FEATURE_ENGINEER.standardizer()
        _tracking_check: Dict[str, bool] = _check_tracking(meth='standardizer', suffix='standard', feature_type='continuous')
        self.assertTrue(expr=_tracking_check.get('process') and _tracking_check.get('raw') and _tracking_check.get('level'))

    def test_subset(self):
        _n_cases: int = FEATURE_ENGINEER.get_n_cases()
        FEATURE_ENGINEER.subset(cond='AveragePrice > 1.6')
        self.assertTrue(expr=_n_cases > FEATURE_ENGINEER.get_n_cases())

    def test_subset_features_for_modeling(self):
        self.assertTrue(expr=FEATURE_ENGINEER.get_n_features() > len(FEATURE_ENGINEER.subset_features_for_modeling().columns))

    def test_stack(self):
        pass

    def test_unify_invalid_to_mis(self):
        pass

    def test_unstack(self):
        pass

    def test_target_type_adjustment(self):
        FEATURE_ENGINEER.set_target(feature='year')
        print(FEATURE_ENGINEER.get_feature_values(feature='year', unique=True))
        FEATURE_ENGINEER.target_type_adjustment(label_encode=True)
        print(FEATURE_ENGINEER.get_feature_values(feature='year', unique=True))

    def test_text_occurances(self):
        pass

    def test_text_similarity(self):
        pass

    def test_to_float32(self):
        _original_float_type: List[np.dtype] = FEATURE_ENGINEER.get_data(dask_df=False).dtypes[FEATURE_ENGINEER.get_features(feature_type='continuous')].values.tolist()
        FEATURE_ENGINEER.to_float32()
        _new_float_type: List[np.dtype] = FEATURE_ENGINEER.get_data(dask_df=False).dtypes[FEATURE_ENGINEER.get_features(feature_type='continuous')].values.tolist()
        _found_float32: List[bool] = []
        for i, oft in enumerate(_original_float_type):
            if oft == np.dtype('float64') and _new_float_type[i] == np.dtype('float32'):
                _found_float32.append(True)
            else:
                _found_float32.append(False)
        self.assertTrue(expr=all(_found_float32))

    def test_type_conversion(self):
        _data_type: str = str(FEATURE_ENGINEER.get_data(dask_df=False)['Date'].dtype)
        FEATURE_ENGINEER.type_conversion(feature_type=dict(Date='str'))
        self.assertTrue(expr=_data_type.find('datetime') >= 0 and str(FEATURE_ENGINEER.get_data(dask_df=False)['Date'].dtype).find('object') >= 0)

    def test_write_notepad(self):
        _page: str = 'Unittest'
        _note: str = 'happy ;) learning'
        FEATURE_ENGINEER.write_notepad(note=_note, page=_page, append=True, add_time_stamp=True)
        self.assertTrue(expr=_note == FEATURE_ENGINEER.get_notes(page=_page).split('\n')[0] and [_page] == FEATURE_ENGINEER.get_pages())


if __name__ == '__main__':
    unittest.main()
