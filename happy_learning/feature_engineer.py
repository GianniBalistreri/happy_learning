import copy
import dask.dataframe as dd
import gc
import geocoder
import inspect
import numpy as np
import multiprocessing
import pandas as pd
import re
import os
import sys
import warnings

from .chaid_decision_tree import CHAIDDecisionTree
from .missing_data_analysis import MissingDataAnalysis
from .multiple_imputation import MultipleImputation
from .sampler import MLSampler, Sampler
from .text_miner import TextMiner, TextMinerException
from .utils import HappyLearningUtils
from datetime import datetime
from dateutil import parser
from easyexplore.anomaly_detector import AnomalyDetector
from easyexplore.data_explorer import DataExplorer
from easyexplore.data_import_export import CLOUD_PROVIDER, DataExporter, DataImporter, FileUtilsException
from easyexplore.utils import EasyExploreUtils, INVALID_VALUES, Log, StatsUtils
from scipy.stats import boxcox
from sklearn.preprocessing import Binarizer, MinMaxScaler, Normalizer, KBinsDiscretizer, RobustScaler, PolynomialFeatures, StandardScaler
from typing import Dict, List, Tuple, Union

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

DASK_INDEXER: str = '___dask_index___'
TEMP_INDEXER: dict = {'__index__': []}
TEMP_DIR: str = ''
NOTEPAD: dict = {}
PREDICTORS: List[str] = []
MERGES: Dict[str, List[str]] = {}
SPECIAL_JOBS: Dict[str, str] = dict(disparity='pair',
                                    disparity_time_series='all',
                                    interaction_poly='all',
                                    interaction='all',
                                    concat_text='all',
                                    set_predictors='all'
                                    )
SUPPORTED_TYPES: Dict[str, List[str]] = {}
ALL_FEATURES: List[str] = []
FEATURE_TYPES: Dict[str, List[str]] = dict(continuous=[], ordinal=[], categorical=[], date=[], id_text=[])
DATA_PROCESSING: dict = dict(processing=dict(process={},
                                             features=dict(raw={},
                                                           level_1={}
                                                           ),
                                             ),
                             encoder=dict(bin={},
                                          label={},
                                          one_hot={}
                                          ),
                             scaler=dict(minmax={},
                                         robust={},
                                         normal={},
                                         standard={},
                                         box_cox={},
                                         log={},
                                         exp={}
                                         ),
                             interaction=dict(disparity=dict(date={},
                                                             continuous={}
                                                             ),
                                              simple={},
                                              polynomial={},
                                              one_hot={}
                                              ),
                             self_interaction=dict(addition={},
                                                   multiplication={}
                                                   ),
                             categorizer=dict(date={},
                                              continuous={}
                                              ),
                             mapper=dict(obs={},
                                         mis={},
                                         imp={},
                                         clean={},
                                         names={}
                                         ),
                             text=dict(occurances={},
                                       split={},
                                       categorical=dict(len={},
                                                        numbers={},
                                                        words={},
                                                        chars={},
                                                        special_chars={},
                                                        email={},
                                                        url={}
                                                        ),
                                       linguistic=dict(pos={},
                                                       ner={},
                                                       dep_tree={},
                                                       dep_noun={},
                                                       emoji={}
                                                       )
                                       )
                             )
PROCESSING_ACTION_SPACE: dict = dict(date=['date_categorizer',
                                           'disparity'
                                           ],
                                     ordinal=dict(interaction=['addition',
                                                               'subtraction',
                                                               'multiplication',
                                                               'division'
                                                               ],
                                                  self_interaction=['addition',
                                                                    'multiplication'
                                                                    ],
                                                  ),
                                     categorical=dict(binning=['one_hot_merger']),
                                     continuous=dict(interaction=['addition',
                                                                  'subtraction',
                                                                  'multiplication',
                                                                  'division'
                                                                  ],
                                                     self_interaction=['addition',
                                                                       'multiplication'
                                                                       ],
                                                     transformation=['exp_transform',
                                                                     'log_transform',
                                                                     'normalizer',
                                                                     'scaling_minmax',
                                                                     'scaling_robust',
                                                                     'square_root_transform',
                                                                     'standardizer'
                                                                     ]
                                                     ),
                                     id_text=['cluster',
                                              'counter',
                                              'detector',
                                              'splitter'
                                              ]
                                     )
TEXT_MINER: dict = dict(obj=None,
                        segments={},
                        data=None,
                        generated_features=[],
                        linguistic={}
                        )

# TODO:
#  processing: process -> graph for re-engineer data for prediction in "production" environment
#  clean internals if FeatureEngineer is re_initializing
#  sampler: feature, train_test - normal
#  disparity: processing of date features only (effect of SPECIAL_JOBS)
#  post-processing data for prediction from trained ml model


def _avoid_overwriting(feature: str) -> str:
    """
    Avoid overwriting feature

    :param feature: str
        Name of the new generated feature

    :return str:
        Adjusted feature name if feature name already existed
    """
    _feature: str = feature
    if _feature in DATA_PROCESSING['df'].columns:
        _i: int = 0
        while _feature in DATA_PROCESSING['df'].columns:
            _i += 1
            if _i == 1:
                _feature = '{}_{}'.format(_feature, _i)
            else:
                _elements: List[str] = copy.deepcopy(_feature.split('_'))
                _elements[-1] = str(_i)
                _feature = ''.join(_elements)
    return _feature


def _float_adjustments(features: List[str], imp_value: float, convert_to_float32: bool = True):
    """
    Replace invalid values generated by pre-processing methods

    :param features: List[str]
        Name of the features to process

    :param imp_value: float
        Single imputation value

    :param convert_to_float32: bool
        Convert continuous features types as float64 to float32 for compatibility reasons
    """
    if MissingDataAnalysis(df=DATA_PROCESSING['df']).has_nan():
        DATA_PROCESSING['df'][features] = DATA_PROCESSING['df'][features].replace(to_replace=INVALID_VALUES, value=np.nan, regex=False)
        DATA_PROCESSING['df'][features] = DATA_PROCESSING['df'][features].fillna(value=imp_value, axis=0)
    if convert_to_float32:
        if len(features) == 1:
            DATA_PROCESSING['df'][features[0]] = DATA_PROCESSING['df'][features[0]].astype(np.float32)
        else:
            DATA_PROCESSING['df'][features] = DATA_PROCESSING['df'][features].astype(np.float32)
    _inv_features: List[str] = EasyExploreUtils().get_invariant_features(df=DATA_PROCESSING.get('df')[features])
    _features: List[str] = copy.deepcopy(features)
    for inv in _inv_features:
        DATA_PROCESSING['cleaned_features'].append(inv)
        del _features[_features.index(inv)]
        DATA_PROCESSING['df'] = DATA_PROCESSING.get('df').drop(labels=inv, axis=1, errors='ignore')
        Log(write=not DATA_PROCESSING.get('show_msg')).log('Cleaned feature "{}"'.format(features[0]))
        _process_handler(action='clean',
                         feature=inv,
                         new_feature=inv,
                         process='mapper|clean',
                         meth='clean',
                         param=dict(markers=dict(features=features[0]))
                         )
    if len(_features) > 0:
        if MissingDataAnalysis(df=DATA_PROCESSING['df'][_features]).has_nan():
            DATA_PROCESSING['df'][_features] = DATA_PROCESSING['df'][_features].replace(to_replace=INVALID_VALUES,
                                                                                        value=np.nan,
                                                                                        regex=False
                                                                                        )
            DATA_PROCESSING['df'][_features] = DATA_PROCESSING['df'][_features].fillna(value=imp_value, axis=0)
    for feature in _features:
        if feature not in _inv_features:
            _save_temp_files(feature=feature)


def _load_temp_files(features: List[str]):
    """
    Load temporary feature files

    :param features: List[str]
        Name of the features to load
    """
    DATA_PROCESSING['df'] = None
    DATA_PROCESSING['df'] = pd.DataFrame(data=TEMP_INDEXER)
    for feature in features:
        DATA_PROCESSING['df'][feature] = DataImporter(file_path=os.path.join(TEMP_DIR, '{}.json'.format(feature)),
                                                      as_data_frame=True,
                                                      use_dask=False,
                                                      create_dir=False,
                                                      cloud=None,
                                                      bucket_name=None
                                                      ).file()


def _process_handler(action: str,
                     feature: str,
                     new_feature: str,
                     process: str,
                     meth: str,
                     param: dict,
                     data: np.array = None,
                     force_type: str = None,
                     special_replacement: bool = False,
                     imp_value: float = None,
                     msg: str = None,
                     obj=None
                     ):
    """
    Handle data processing tracking

    :param action: str
        Pre-defined action
            -> add: Add processing information to history
            -> merge: Add processing information about merging two data sets to history
            -> clean: Remove feature information regarding relationships from history
            -> rename: Rename features in history

    :param feature: str
        Name of the processed feature

    :param new_feature: str
        Name of the new generated feature

    :param process: str
        Name of the topic and subtopic (separated by |) of the process

    :param meth: str
        Name of the class method

    :param param: dict
        Parameter config of class method

    :param data: np.array
        Generated data by applying processing method

    :param force_type: str
        Name of the feature type to force

    :param special_replacement: bool
        Whether to impute generated invalid values automatically or not

    :param imp_value: float
        Single imputation value

    :param msg: str
        Printing message

    :param obj: object
        Processor object to store
    """
    if DATA_PROCESSING['re_generate']:
        DATA_PROCESSING['re_gen_data'] = data
    else:
        if action == 'add':
            if DATA_PROCESSING.get('avoid_overwriting'):
                _new_feature: str = _avoid_overwriting(feature=new_feature)
            else:
                _new_feature: str = new_feature
            DATA_PROCESSING['last_generated_feature'] = _new_feature
            _tracked_processes: int = len(DATA_PROCESSING['processing']['process'].keys())
            _processed_features: dict = {}
            if new_feature == '':
                if isinstance(obj, list):
                    for f in obj:
                        _processed_features.update({f: feature})
                elif isinstance(obj, str):
                    _processed_features.update({obj: feature})
                elif isinstance(obj, dict):
                    for k in obj['features'].keys():
                        _processed_features.update({k: obj['features'].get(k)})
            else:
                _processed_features.update({_new_feature: feature})
            _p: int = 0
            for n, f in _processed_features.items():
                DATA_PROCESSING['processing']['process'].update({str(_tracked_processes + _p + 1): dict(meth=meth,
                                                                                                        param=param,
                                                                                                        features={n: f}
                                                                                                        )
                                                                 })
                _p += 1
            if feature != '' and new_feature != '' and feature != new_feature:
                _set_feature_relations(feature=feature, new_feature=_new_feature)
            _process_data_set: bool = True
            if obj is not None:
                _process: List[str] = process.split('|')
                if len(_process) == 1:
                    DATA_PROCESSING[_process[0]].update({feature: obj})
                elif len(_process) == 2:
                    DATA_PROCESSING[_process[0]][_process[1]].update({feature: obj})
                    if _process[0] == 'interaction':
                        if _process[1] in ['simple', 'polynomial']:
                            _process_data_set = False
                            _new_names: dict = {}
                            for ft in data.columns:
                                _ft: str = _avoid_overwriting(feature=ft)
                                if ft != _ft:
                                    _new_names.update({ft: _ft})
                            if len(_new_names.keys()) > 0:
                                data = data.rename(columns=_new_names)
                            DATA_PROCESSING['df'] = dd.concat(dfs=[DATA_PROCESSING['df'], data], axis=1)
                            _float_adjustments(features=list(data.columns), imp_value=imp_value)
                            for inter_feature in obj['features'].keys():
                                for inter in obj['features'][inter_feature]:
                                    _set_feature_relations(feature=inter, new_feature=inter_feature)
                                if DATA_PROCESSING.get('last_generated_feature') == '':
                                    DATA_PROCESSING['last_generated_feature'] = inter_feature
                                _update_feature_types(feature=inter_feature, force_type='continuous')
                        elif _process[1] == 'one_hot':
                            _set_feature_relations(feature=obj, new_feature=_new_feature)
                    elif _process[0] == 'text' and _process[1] == 'split':
                        _process_data_set = False
                        for o in obj:
                            _set_feature_relations(feature=feature, new_feature=o)
                            _update_feature_types(feature=o, force_type=force_type)
                            _save_temp_files(feature=o)
                    elif _process[0] == 'encoder' and _process[1] == 'one_hot':
                        _process_data_set = False
                        for o in obj:
                            _set_feature_relations(feature=feature, new_feature=o)
                            _update_feature_types(feature=o, force_type=force_type)
                            _save_temp_files(feature=o)
                    elif _process[0] == 'encoder' and _process[1] == 'label':
                        _process_data_set = False
                        DATA_PROCESSING['df'][new_feature] = data
                        _update_feature_types(feature=new_feature, force_type=force_type)
                        _save_temp_files(feature=new_feature)
                    elif _process[0] == 'mapper' and _process[1] == 'obs':
                        _process_data_set = False
                        DATA_PROCESSING['df'][new_feature] = data.replace(obj.get(feature))
                        if not MissingDataAnalysis(df=DATA_PROCESSING['df'][new_feature]).has_nan():
                            DATA_PROCESSING['df'][new_feature] = DATA_PROCESSING['df'][new_feature].astype(int)
                        if feature != new_feature:
                            _update_feature_types(feature=new_feature, force_type=force_type)
                            _save_temp_files(feature=new_feature)
                elif len(_process) == 3:
                    if _process[0] == 'interaction':
                        if _process[1] == 'disparity':
                            _process_data_set = False
                            for o in obj:
                                _set_feature_relations(feature=o, new_feature=_new_feature)
                                _update_feature_types(feature=o,
                                                      force_type='continuous' if _process[2] == 'continuous' else None
                                                      )
                            DATA_PROCESSING['df'][_new_feature] = data
                            _update_feature_types(feature=_new_feature, force_type=force_type)
                            _save_temp_files(feature=_new_feature)
            if _new_feature != '':
                if _process_data_set:
                    DATA_PROCESSING['df'][_new_feature] = data
                    _update_feature_types(feature=_new_feature, force_type=force_type)
                    _save_temp_files(feature=_new_feature)
            if special_replacement:
                _float_adjustments(features=[_new_feature], imp_value=imp_value)
        elif action == 'clean':
            _process: List[str] = process.split('|')
            if len(_process) == 1:
                DATA_PROCESSING[_process[0]].update({feature: datetime.now()})
            elif len(_process) == 2:
                DATA_PROCESSING[_process[0]][_process[1]].update({feature: datetime.now()})
            _history: dict = copy.deepcopy(DATA_PROCESSING['processing']['features'])
            for level in _history.keys():
                if feature in DATA_PROCESSING['processing']['features'][level].keys():
                    del DATA_PROCESSING['processing']['features'][level][feature]
                for tracked_feature in DATA_PROCESSING['processing']['features'][level].keys():
                    for relation in DATA_PROCESSING['processing']['features'][level][tracked_feature]:
                        if feature == relation:
                            _idx: int = DATA_PROCESSING['processing']['features'][level][tracked_feature].index(relation)
                            del DATA_PROCESSING['processing']['features'][level][tracked_feature][_idx]
            for ft in FEATURE_TYPES.keys():
                if feature in FEATURE_TYPES.get(ft):
                    _features: List[str] = copy.deepcopy(FEATURE_TYPES.get(ft))
                    del _features[_features.index(feature)]
                    FEATURE_TYPES[ft] = copy.deepcopy(_features)
            if feature in PREDICTORS:
                del PREDICTORS[PREDICTORS.index(feature)]
            if feature == DATA_PROCESSING.get('last_generated_feature'):
                DATA_PROCESSING['last_generated_feature'] = ''
        elif action == 'rename':
            if feature != new_feature:
                _tracked_processes: int = len(DATA_PROCESSING['processing']['process'].keys())
                DATA_PROCESSING['processing']['process'].update({str(_tracked_processes + 1): dict(meth=meth, param=param, features={})})
                _process: List[str] = process.split('|')
                if len(_process) == 1:
                    DATA_PROCESSING[_process[0]].update({feature: new_feature})
                elif len(_process) == 2:
                    DATA_PROCESSING[_process[0]][_process[1]].update({feature: new_feature})
                _renamed_history: dict = {}
                for level in DATA_PROCESSING['processing']['features'].keys():
                    #print(level)
                    _renamed_history.update({level: {}})
                    for tracked_feature in DATA_PROCESSING['processing']['features'][level].keys():
                        #print(tracked_feature)
                        if feature != tracked_feature:
                            _renamed_history[level].update({tracked_feature: []})
                            for relation in DATA_PROCESSING['processing']['features'][level][tracked_feature]:
                                #print(relation)
                                if feature != relation:
                                    _renamed_history[level][tracked_feature].append(relation)
                DATA_PROCESSING['processing']['features'] = _renamed_history
                DATA_PROCESSING['df'] = DATA_PROCESSING['df'].rename(columns={feature: new_feature})
                _save_temp_files(feature=new_feature)
                for feature_type in FEATURE_TYPES.keys():
                    if feature in FEATURE_TYPES.get(feature_type):
                        FEATURE_TYPES[feature_type][FEATURE_TYPES[feature_type].index(feature)] = new_feature
                        break
                if feature in PREDICTORS:
                    PREDICTORS[PREDICTORS.index(feature)] = new_feature
                if feature in DATA_PROCESSING['encoder']['label'].keys():
                    DATA_PROCESSING['encoder']['label'].update({new_feature: DATA_PROCESSING['encoder']['label'][feature]})
                    del DATA_PROCESSING['encoder']['label'][feature]
                if feature in DATA_PROCESSING['encoder']['one_hot'].keys():
                    DATA_PROCESSING['encoder']['one_hot'].update({new_feature: DATA_PROCESSING['encoder']['one_hot'][feature]})
                    del DATA_PROCESSING['encoder']['one_hot'][feature]
                if feature in DATA_PROCESSING['scaler']['robust'].keys():
                    DATA_PROCESSING['scaler']['robust'].update({new_feature: DATA_PROCESSING['scaler']['robust'][feature]})
                    del DATA_PROCESSING['scaler']['robust'][feature]
                if feature in DATA_PROCESSING['scaler']['minmax'].keys():
                    DATA_PROCESSING['scaler']['minmax'].update({new_feature: DATA_PROCESSING['scaler']['minmax'][feature]})
                    del DATA_PROCESSING['scaler']['minmax'][feature]
                if feature in DATA_PROCESSING['scaler']['normal'].keys():
                    DATA_PROCESSING['scaler']['normal'].update({new_feature: DATA_PROCESSING['scaler']['normal'][feature]})
                    del DATA_PROCESSING['scaler']['normal'][feature]
                if feature in DATA_PROCESSING['scaler']['standard'].keys():
                    DATA_PROCESSING['scaler']['standard'].update({new_feature: DATA_PROCESSING['scaler']['standard'][feature]})
                    del DATA_PROCESSING['scaler']['standard'][feature]
                if feature in DATA_PROCESSING['scaler']['box_cox'].keys():
                    DATA_PROCESSING['scaler']['box_cox'].update({new_feature: DATA_PROCESSING['scaler']['box_cox'][feature]})
                    del DATA_PROCESSING['scaler']['box_cox'][feature]
                if feature in DATA_PROCESSING['interaction']['disparity']['date'].keys():
                    DATA_PROCESSING['interaction']['disparity']['date'].update({new_feature: DATA_PROCESSING['interaction']['disparity']['date'][feature]})
                    del DATA_PROCESSING['interaction']['disparity']['date'][feature]
                if feature in DATA_PROCESSING['interaction']['disparity']['continuous'].keys():
                    DATA_PROCESSING['interaction']['disparity']['continuous'].update({new_feature: DATA_PROCESSING['interaction']['disparity']['continuous'][feature]})
                    del DATA_PROCESSING['interaction']['disparity']['continuous'][feature]
                if feature in DATA_PROCESSING['interaction']['simple'].keys():
                    DATA_PROCESSING['interaction']['simple'].update({new_feature: DATA_PROCESSING['interaction']['simple'][feature]})
                    del DATA_PROCESSING['interaction']['simple'][feature]
                if feature in DATA_PROCESSING['interaction']['polynomial'].keys():
                    DATA_PROCESSING['interaction']['polynomial'].update({new_feature: DATA_PROCESSING['interaction']['polynomial'][feature]})
                    del DATA_PROCESSING['interaction']['polynomial'][feature]
                if feature in DATA_PROCESSING['interaction']['one_hot'].keys():
                    DATA_PROCESSING['interaction']['one_hot'].update({new_feature: DATA_PROCESSING['interaction']['one_hot'][feature]})
                    del DATA_PROCESSING['interaction']['one_hot'][feature]
                if feature in DATA_PROCESSING['categorizer']['date'].keys():
                    DATA_PROCESSING['categorizer']['date'].update({new_feature: DATA_PROCESSING['categorizer']['date'][feature]})
                    del DATA_PROCESSING['categorizer']['date'][feature]
                if feature in DATA_PROCESSING['mapper']['obs'].keys():
                    DATA_PROCESSING['mapper']['obs'].update({new_feature: DATA_PROCESSING['mapper']['obs'][feature]})
                    del DATA_PROCESSING['mapper']['obs'][feature]
                if feature in DATA_PROCESSING['mapper']['mis'].keys():
                    DATA_PROCESSING['mapper']['mis'].update({new_feature: DATA_PROCESSING['mapper']['mis'][feature]})
                    del DATA_PROCESSING['mapper']['mis'][feature]
                if feature in DATA_PROCESSING['mapper']['imp'].keys():
                    DATA_PROCESSING['mapper']['imp'].update({new_feature: DATA_PROCESSING['mapper']['imp'][feature]})
                    del DATA_PROCESSING['mapper']['imp'][feature]
                if feature in DATA_PROCESSING['text']['occurances'].keys():
                    DATA_PROCESSING['text']['occurances'].update({new_feature: DATA_PROCESSING['text']['occurances'][feature]})
                    del DATA_PROCESSING['text']['occurances'][feature]
                if feature in DATA_PROCESSING['text']['split'].keys():
                    DATA_PROCESSING['text']['split'].update({new_feature: DATA_PROCESSING['text']['split'][feature]})
                    del DATA_PROCESSING['text']['split'][feature]
                if feature in DATA_PROCESSING['text']['categorical']['len'].keys():
                    DATA_PROCESSING['text']['categorical']['len'].update({new_feature: DATA_PROCESSING['text']['categorical']['len'][feature]})
                    del DATA_PROCESSING['text']['categorical']['len'][feature]
                if feature in DATA_PROCESSING['text']['categorical']['numbers'].keys():
                    DATA_PROCESSING['text']['categorical']['numbers'].update({new_feature: DATA_PROCESSING['text']['categorical']['numbers'][feature]})
                    del DATA_PROCESSING['text']['categorical']['numbers'][feature]
                if feature in DATA_PROCESSING['text']['categorical']['words'].keys():
                    DATA_PROCESSING['text']['categorical']['words'].update({new_feature: DATA_PROCESSING['text']['categorical']['words'][feature]})
                    del DATA_PROCESSING['text']['categorical']['words'][feature]
                if feature in DATA_PROCESSING['text']['categorical']['chars'].keys():
                    DATA_PROCESSING['text']['categorical']['chars'].update({new_feature: DATA_PROCESSING['text']['categorical']['chars'][feature]})
                    del DATA_PROCESSING['text']['categorical']['chars'][feature]
                if feature in DATA_PROCESSING['text']['categorical']['special_chars'].keys():
                    DATA_PROCESSING['text']['categorical']['special_chars'].update({new_feature: DATA_PROCESSING['text']['categorical']['special_chars'][feature]})
                    del DATA_PROCESSING['text']['categorical']['special_chars'][feature]
                if feature in DATA_PROCESSING['text']['categorical']['email'].keys():
                    DATA_PROCESSING['text']['categorical']['email'].update({new_feature: DATA_PROCESSING['text']['categorical']['email'][feature]})
                    del DATA_PROCESSING['text']['categorical']['email'][feature]
                if feature in DATA_PROCESSING['text']['categorical']['url'].keys():
                    DATA_PROCESSING['text']['categorical']['url'].update({new_feature: DATA_PROCESSING['text']['categorical']['url'][feature]})
                    del DATA_PROCESSING['text']['categorical']['url'][feature]
                if feature in DATA_PROCESSING['text']['linguistic']['pos'].keys():
                    DATA_PROCESSING['text']['linguistic']['pos'].update({new_feature: DATA_PROCESSING['text']['linguistic']['pos'][feature]})
                    del DATA_PROCESSING['text']['linguistic']['pos'][feature]
                if feature in DATA_PROCESSING['text']['linguistic']['ner'].keys():
                    DATA_PROCESSING['text']['linguistic']['ner'].update({new_feature: DATA_PROCESSING['text']['linguistic']['ner'][feature]})
                    del DATA_PROCESSING['text']['linguistic']['ner'][feature]
                if feature in DATA_PROCESSING['text']['linguistic']['dep_tree'].keys():
                    DATA_PROCESSING['text']['linguistic']['dep_tree'].update({new_feature: DATA_PROCESSING['text']['linguistic']['dep_tree'][feature]})
                    del DATA_PROCESSING['text']['linguistic']['dep_tree'][feature]
                if feature in DATA_PROCESSING['text']['linguistic']['dep_noun'].keys():
                    DATA_PROCESSING['text']['linguistic']['dep_noun'].update({new_feature: DATA_PROCESSING['text']['linguistic']['dep_noun'][feature]})
                    del DATA_PROCESSING['text']['linguistic']['dep_noun'][feature]
                if feature in DATA_PROCESSING['text']['linguistic']['emoji'].keys():
                    DATA_PROCESSING['text']['linguistic']['emoji'].update({new_feature: DATA_PROCESSING['text']['linguistic']['emoji'][feature]})
                    del DATA_PROCESSING['text']['linguistic']['emoji'][feature]
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Rename feature {} to {}'.format(feature, new_feature))
    if msg is not None:
        if len(msg) > 0:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg=msg)


def _save_temp_files(feature: str):
    """
    Save temporary feature files
    """
    global ALL_FEATURES
    if isinstance(DATA_PROCESSING['df'], dd.DataFrame):
        _data: list = DATA_PROCESSING['df'][feature].values.compute().tolist()
    else:
        _data: list = DATA_PROCESSING['df'][feature].values.tolist()
    DataExporter(obj={feature: _data},
                 file_path=os.path.join(TEMP_DIR, '{}.json'.format(feature)),
                 create_dir=False,
                 overwrite=True,
                 cloud=None,
                 bucket_name=None
                 ).file()
    if feature not in ALL_FEATURES:
        ALL_FEATURES.append(feature)


def _set_feature_relations(feature: str, new_feature: str):
    """
    Set feature relations internally

    :param feature: str
        Name of the original feature

    :param new_feature: str
        Name of the new generated feature
    """
    if feature != new_feature:
        #if new_feature not in DATA_PROCESSING['processing']['features']['raw'].keys():
            # TODO: implement manual setting -> add "new" feature and related features to higher level
        #    pass
        if new_feature not in DATA_PROCESSING['processing']['features']['raw'].keys():
            if feature in DATA_PROCESSING['processing']['features']['raw'].keys():
                if new_feature not in DATA_PROCESSING['processing']['features']['raw'][feature]:
                    DATA_PROCESSING['processing']['features']['raw'][feature].append(new_feature)
                    if new_feature not in DATA_PROCESSING['processing']['features']['level_1'].keys():
                        DATA_PROCESSING['processing']['features']['level_1'].update({new_feature: []})
            else:
                _level: int = 2
                if DATA_PROCESSING['processing']['features'].get('level_{}'.format(_level)) is None:
                    DATA_PROCESSING['processing']['features'].update({'level_{}'.format(_level): {}})
                if feature in DATA_PROCESSING['processing']['features']['level_1'].keys():
                    DATA_PROCESSING['processing']['features']['level_1'][feature].append(new_feature)
                    if new_feature not in DATA_PROCESSING['processing']['features']['level_{}'.format(_level)].keys():
                        DATA_PROCESSING['processing']['features']['level_{}'.format(_level)].update({new_feature: []})
                else:
                    while True:
                        if feature in DATA_PROCESSING['processing']['features']['level_{}'.format(_level)].keys():
                            DATA_PROCESSING['processing']['features']['level_{}'.format(_level)][feature].append(
                                new_feature)
                            _parent_feature: str = feature
                            __level: int = _level
                            while __level > 0:
                                __level -= 1
                                if __level == 0:
                                    _level_name: str = 'raw'
                                else:
                                    _level_name: str = 'level_{}'.format(__level)
                                for lower_level_feature in DATA_PROCESSING['processing']['features'][_level_name].keys():
                                    if _parent_feature in DATA_PROCESSING['processing']['features'][_level_name][lower_level_feature]:
                                        _parent_feature = lower_level_feature
                                        DATA_PROCESSING['processing']['features'][_level_name][
                                            lower_level_feature].append(new_feature)
                                        break
                            if DATA_PROCESSING['processing']['features'].get('level_{}'.format(_level + 1)) is None:
                                DATA_PROCESSING['processing']['features'].update({'level_{}'.format(_level + 1): {}})
                            if new_feature not in DATA_PROCESSING['processing']['features']['level_{}'.format(_level + 1)].keys():
                                DATA_PROCESSING['processing']['features']['level_{}'.format(_level + 1)].update({new_feature: []})
                            break
                        else:
                            _level += 1


def _set_feature_types(df: pd.DataFrame,
                       features: List[str],
                       continuous: List[str] = None,
                       categorical: List[str] = None,
                       ordinal: List[str] = None,
                       date: List[str] = None,
                       id_text: List[str] = None,
                       max_cats: int = 500,
                       date_edges: Tuple[str, str] = None
                       ):
    """
    Set feature types

    :param df: pd.DataFrame
        Data set

    :param features: List[str]
        Name of the features

    :param continuous: List[str]
        Name of the pre-defined continuous features

    :param categorical: List[str]
        Name of the pre-defined categorical features

    :param ordinal: List[str]
        Name of the pre-defined ordinal features

    :param date: List[str]
        Name of the pre-defined date features

    :param id_text: List[str]
        Name of the pre-defined id and text features

    :param max_cats: int
        Maximal number of categories

    :param date_edges: tuple
        Minimum and maximum date times to identify date features more accurate
    """
    global FEATURE_TYPES
    if DASK_INDEXER in features:
        del features[features.index(DASK_INDEXER)]
    if len(features) > 0:
        _analytical_feature_type: dict = HappyLearningUtils().get_analytical_type(df=df,
                                                                                  feature=features[0],
                                                                                  dtype=df[features[0]].dtype,
                                                                                  continuous=continuous,
                                                                                  categorical=categorical,
                                                                                  ordinal=ordinal,
                                                                                  date=date,
                                                                                  id_text=id_text,
                                                                                  date_edges=date_edges
                                                                                  )
        if _analytical_feature_type.get('categorical') is not None:
            FEATURE_TYPES['categorical'].append(features[0])
        elif _analytical_feature_type.get('continuous') is not None:
            FEATURE_TYPES['continuous'].append(features[0])
        elif _analytical_feature_type.get('date') is not None:
            FEATURE_TYPES['date'].append(features[0])
        elif _analytical_feature_type.get('ordinal') is not None:
            FEATURE_TYPES['ordinal'].append(features[0])
        elif _analytical_feature_type.get('id_text') is not None:
            FEATURE_TYPES['id_text'].append(features[0])
        #FEATURE_TYPES = EasyExploreUtils().get_feature_types(df=df,
        #                                                     features=features,
        #                                                     dtypes=df[features].dtypes,
        #                                                     continuous=continuous,
        #                                                     categorical=categorical,
        #                                                     ordinal=ordinal,
        #                                                     date=date,
        #                                                     id_text=id_text,
        #                                                     date_edges=date_edges,
        #                                                     print_msg=False
        #                                                     )
    if DATA_PROCESSING.get('target') is not None:
        for ft in FEATURE_TYPES.keys():
            if DATA_PROCESSING.get('target') in FEATURE_TYPES.get(ft):
                del FEATURE_TYPES[ft][FEATURE_TYPES[ft].index(DATA_PROCESSING.get('target'))]


def _update_feature_types(feature: str, force_type: str = None):
    """
    Update feature types by new feature

    :param feature: str
        Name of the new feature

    :param force_type: str
        Name of the forced feature type
    """
    if feature in DATA_PROCESSING['df'].columns:
        if feature != DASK_INDEXER:
            _feature_type = EasyExploreUtils().get_feature_types(df=DATA_PROCESSING.get('df'),
                                                                 features=[feature],
                                                                 dtypes=[DATA_PROCESSING['df'][feature].dtype],
                                                                 continuous=[feature] if force_type == 'continuous' else None,
                                                                 categorical=[feature] if force_type == 'categorical' else None,
                                                                 ordinal=[feature] if force_type == 'ordinal' else None,
                                                                 date=[feature] if force_type == 'date' else None,
                                                                 id_text=[feature] if force_type == 'id_text' else None,
                                                                 print_msg=False
                                                                 )
            for ft in _feature_type.keys():
                if feature in _feature_type.get(ft):
                    if feature not in FEATURE_TYPES.get(ft):
                        FEATURE_TYPES[ft].append(feature)
                else:
                    if feature in FEATURE_TYPES.get(ft):
                        _features: List[str] = copy.deepcopy(FEATURE_TYPES.get(ft))
                        del _features[_features.index(feature)]
                        FEATURE_TYPES[ft] = _features


class FeatureOrchestra:
    """
    Class for decorating methods of class FeatureEngineer to ensure automatic data processing for many features type similarly
    """
    def __init__(self, **supported_types):
        """
        :param supported_types: dict
            Supported feature types
        """
        _meth: str = supported_types.get('meth')
        _feature_types: List[str] = []
        for ft in supported_types.get('feature_types'):
            if ft in FEATURE_TYPES.keys():
                _feature_types.append(ft)
            else:
                raise ValueError('Feature types ({}) not supported. Supported types are: {}'.format(ft, list(FEATURE_TYPES.keys())))
        SUPPORTED_TYPES.update({_meth: _feature_types})

    def __call__(self, func):
        """
        :param func: function
            Class method to call

        :return function
            Run class method using orchestrated feature list
        """
        def new_func(**kwargs):
            """
            Orchestrate features for used method

            kwargs: dict
                Key-word arguments of class method

            :return function
                Call class method with manipulated 'features' parameter if value is None or an empty list
            """
            gc.collect()
            if kwargs.get('features') is not None:
                if DASK_INDEXER in kwargs.get('features'):
                    del kwargs.get('features')[kwargs.get('features').index(DASK_INDEXER)]
            _used_defined: bool = False
            _compatible_features: List[str] = []
            for ft in SUPPORTED_TYPES[func.__name__]:
                _compatible_features.extend(FEATURE_TYPES.get(ft))
            if kwargs.get('features') is None:
                kwargs['features'] = _compatible_features
            else:
                if len(kwargs.get('features')) == 0:
                    kwargs['features'] = _compatible_features
                else:
                    _all: List[str] = []
                    _use: List[str] = []
                    for dt in FEATURE_TYPES.keys():
                        _all.extend(FEATURE_TYPES.get(dt))
                    for ft in kwargs.get('features'):
                        if ft in _all:
                            _use.append(ft)
                    if len(_use) > 0:
                        _used_defined = True
                        kwargs['features'] = _use
                    else:
                        kwargs['features'] = _compatible_features
            _found_features: List[str] = copy.deepcopy(kwargs['features'])
            #for feature in kwargs['features']:
            #    if isinstance(DATA_PROCESSING['df'], dd.DataFrame) or isinstance(DATA_PROCESSING['df'], pd.DataFrame):
            #        if feature not in DATA_PROCESSING['df'].columns:
            #            del _found_features[_found_features.index(feature)]
            kwargs['features'] = _found_features
            for predictor in PREDICTORS:
                if predictor in kwargs['features']:
                    del kwargs['features'][kwargs['features'].index(predictor)]
            if not _used_defined:
                if DATA_PROCESSING.get('max_level') is not None:
                    if DATA_PROCESSING.get('max_level') < (len(DATA_PROCESSING['processing']['features'].keys()) - 1):
                        for level_x in DATA_PROCESSING['processing']['features']['level_{}'.format(DATA_PROCESSING.get('max_level'))].keys():
                            if level_x in kwargs['features']:
                                del kwargs['features'][kwargs['features'].index(level_x)]
                            else:
                                for upper_level in DATA_PROCESSING['processing']['features']['level_{}'.format(DATA_PROCESSING.get('max_level'))][level_x]:
                                    if upper_level in kwargs['features']:
                                        del kwargs['features'][kwargs['features'].index(upper_level)]
            if DATA_PROCESSING.get('multi_threading') is None:
                return func(**kwargs)
            else:
                if DATA_PROCESSING.get('multi_threading'):
                    _jobs = SPECIAL_JOBS.get(func.__name__)
                    _thread_pool: multiprocessing.pool.ThreadPool = multiprocessing.pool.ThreadPool(processes=len(kwargs['features']))
                    if _jobs == 'all':
                        return func(**kwargs)
                    else:
                        if _jobs is None:
                            _features: List[str] = copy.deepcopy(kwargs['features'])
                        elif _jobs == 'pair':
                            _features: List[tuple] = EasyExploreUtils().get_pairs(features=copy.deepcopy(kwargs['features']),
                                                                                  max_features_each_pair=2
                                                                                  )
                        else:
                            _features: List[str] = []
                        for feature in _features:
                            kwargs['features'] = [feature] if isinstance(feature, str) else [feature[0], feature[1]]
                            _thread_pool.apply(func=func, args=(), kwds=kwargs)
                    return []
                else:
                    return func(**kwargs)
        return new_func


class FeatureEngineerException(Exception):
    """
    Class for handling exceptions for class FeatureEngineer
    """
    pass


class FeatureEngineer:
    """
    Class for handling feature engineering
    """
    def __init__(self,
                 temp_dir: str,
                 df: Union[dd.DataFrame, pd.DataFrame] = None,
                 feature_engineer_file_path: str = None,
                 target_feature: str = None,
                 generate_new_feature: bool = True,
                 id_text_features: List[str] = None,
                 date_features: List[str] = None,
                 ordinal_features: List[str] = None,
                 categorical_features: List[str] = None,
                 continuous_features: List[str] = None,
                 keep_original_data: bool = False,
                 unify_invalid_values: bool = True,
                 encode_missing_data: bool = False,
                 date_edges: tuple = None,
                 max_level_processing: int = 5,
                 write_note: str = None,
                 activate_actor: bool = False,
                 critic_config: dict = None,
                 missing_value_analysis: bool = True,
                 auto_cleaning: bool = False,
                 auto_typing: bool = True,
                 auto_engineering: bool = False,
                 auto_text_mining: bool = True,
                 file_path: str = None,
                 sep: str = ',',
                 print_msg: bool = True,
                 n_cpu_cores: int = -1,
                 seed: int = 1234,
                 partitions: int = 4,
                 cloud: str = None,
                 **kwargs
                 ):
        """
        :param temp_dir: str
            Path of the directory for writing temporary file for further processing

        :param df: Pandas DataFrame or dask DataFrame
            Raw data set

        :param feature_engineer_file_path: str
            Complete file path of the stored FeatureEngineer object

        :param target_feature: str
            Name of the target feature to set

        :param generate_new_feature: bool
            Generate new feature or overwrite old one if necessary

        :param id_text_features: List[str]
            Pre-defined id features

        :param date_features: List[str]
            Pre-defined date features

        :param ordinal_features: List[str]
            Pre-defined ordinal features

        :param categorical_features: List[str]
            Pre-defined categorical features

        :param continuous_features: List[str]
            Pre-defined continuous features

        :param keep_original_data: bool
            Keep original data as separate data frame

        :param unify_invalid_values: bool
            Unify invalid values into regular missing value

        :param date_edges: tuple
            Minimum and maximum date times to identify date features more accurate

        :param max_level_processing: int
            Number of processing levels to generate

        :param write_note: str
            Notice to write

        :param activate_actor: bool
            Whether to activate actor to use feature engineering in reinforcement learning environment or not

        :param critic_config: dict
            FeatureEngineer critic configuration

        :param auto_cleaning: bool
            Clean data automatically from invariants, duplicates and high missing data rate

        :param auto_typing: bool
            Re-type features automatically if necessary

        :param auto_text_mining: bool
            Classify and interpret text data analytically for generating numerical features from text

        :param file_path: str
            Path of local file to import as data set

        :param sep: str
            Separator of data file

        :param print_msg: bool
            Print informal messages or not

        :param n_cpu_cores: int
            Number of cpu cores to use

        :param seed: int
            Seed value

        :param partitions: int
            Number of partitions to split data set using dask parallel computing framework

        :param cloud: str
            Name of the cloud provider:
                -> google: Google Cloud Storage
                -> aws: AWS Cloud

        :param kwargs: dict
            Key-word arguments
        """
        Log(write=not print_msg, level='info', env='dev').log(msg='Initializing ...')
        global TEMP_DIR
        TEMP_DIR = temp_dir
        global DATA_PROCESSING
        if n_cpu_cores == -1:
            DATA_PROCESSING['cpu_cores'] = os.cpu_count() - 1
        elif n_cpu_cores == 0:
            DATA_PROCESSING['cpu_cores'] = os.cpu_count()
        elif n_cpu_cores < -1:
            DATA_PROCESSING['cpu_cores'] = os.cpu_count()
        else:
            DATA_PROCESSING['cpu_cores'] = n_cpu_cores
        self.dask_client = HappyLearningUtils().dask_setup(client_name='feature_engineer',
                                                           client_address=kwargs.get('client_address'),
                                                           mode='threads' if kwargs.get('client_mode') is None else kwargs.get('client_mode'),
                                                           **dict(memory_limit='auto')
                                                           )
        self.data_processing = None
        if df is not None:
            if not isinstance(df, dd.DataFrame):
                if not isinstance(df, pd.DataFrame):
                    raise FeatureEngineerException('Format of data set ({}) not supported. Use dask or Pandas instead.'.format(type(df)))
        DATA_PROCESSING.update(dict(df=df,
                                    data_source_path=[file_path if file_path is None else file_path.replace('\\', '/')],
                                    keep_original_data=keep_original_data,
                                    features=[],
                                    predictors=[],
                                    target_type=[],
                                    train_test=dict(simple={},
                                                    kfold={}
                                                    ),
                                    semi_engineered_features=[],
                                    generated_features=[],
                                    generate_new_feature=generate_new_feature,
                                    re_generate=False,
                                    re_gen_data=None,
                                    suffixes=dict(cat='cat',
                                                  num='num',
                                                  bin='bin',
                                                  dummy='dummy',
                                                  robust='robust',
                                                  minmax='minmax',
                                                  normal='normal',
                                                  standard='standard'
                                                  ),
                                    typing={ft: {} for ft in FEATURE_TYPES.keys()},
                                    missing_data={'total': {'valid': 0, 'mis': 0}, 'cases': [], 'features': []},
                                    missing_value_analysis=missing_value_analysis,
                                    encode_missing_data=encode_missing_data,
                                    unify_invalid_values=unify_invalid_values,
                                    correlation=dict(matrix=None, high={}, multi_collinearity=False),
                                    date_edges=None,
                                    max_level=max_level_processing if max_level_processing > 0 else 2,
                                    supported_types=[],
                                    last_action=None,
                                    last_generated_feature=None,
                                    act_trials=0,
                                    activate_actor=activate_actor,
                                    critic_config=critic_config,
                                    action_space=PROCESSING_ACTION_SPACE,
                                    actor_memory={},
                                    imp_features=None,
                                    cleaned_features=[],
                                    avoid_overwriting=True,
                                    source='',
                                    activated=datetime.now(),
                                    show_msg=print_msg,
                                    seed=seed if seed > 0 else 1234,
                                    partitions=partitions,
                                    n_cases=0,
                                    kwargs=kwargs
                                    )
                               )
        if feature_engineer_file_path is None:
            _init: bool = True
            if df is None:
                if file_path is None:
                    raise FeatureEngineerException('Neither data object nor file path to data file found')
                if len(file_path) == 0:
                    raise FeatureEngineerException('Neither data object nor file path to data file found')
                self.data_import(file_path=file_path, sep=sep, **kwargs)
                for feature in DATA_PROCESSING['df'].columns:
                    _save_temp_files(feature=feature)
                DATA_PROCESSING['df'] = None
            else:
                if isinstance(df, pd.DataFrame):
                    DATA_PROCESSING['df'] = dd.from_pandas(data=df, npartitions=DATA_PROCESSING['cpu_cores'])
                DATA_PROCESSING['processing']['features']['raw'].update({feature: [] for feature in DATA_PROCESSING.get('df').columns})
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Data set loaded from given object\nCases: {}\nFeatures: {}'.format(len(DATA_PROCESSING['df']),
                                                                                                                                           len(DATA_PROCESSING['df'].columns)
                                                                                                                                           )
                                                                   )
                DATA_PROCESSING['n_cases'] = len(DATA_PROCESSING['df'])
                global TEMP_INDEXER
                TEMP_INDEXER['__index__'] = [i for i in range(0, DATA_PROCESSING['n_cases'], 1)]
                DATA_PROCESSING.update({'original_features': DATA_PROCESSING.get('df').columns})
                if 'Unnamed: 0' in list(DATA_PROCESSING['df'].columns):
                    del DATA_PROCESSING['df']['Unnamed: 0']
                for feature in DATA_PROCESSING['df'].columns:
                    _save_temp_files(feature=feature)
                DATA_PROCESSING['df'] = None
                Log(write=not print_msg, level='info', env='dev').log(msg='Feature files saved in {}'.format(TEMP_DIR))
        else:
            _init: bool = False
            self.load(file_path=feature_engineer_file_path, cloud=cloud)
        if _init:
            DATA_PROCESSING['pre_defined_feature_types'] = {}
            _id_features: List[str] = []
            if id_text_features is not None:
                for id in id_text_features:
                    if id in DATA_PROCESSING.get('original_features'):
                        _id_features.append(id)
                        DATA_PROCESSING['pre_defined_feature_types'].update({id: 'str'})
            _date_features: List[str] = []
            if date_features is not None:
                for date in date_features:
                    if date in DATA_PROCESSING.get('original_features'):
                        _date_features.append(date)
                        DATA_PROCESSING['pre_defined_feature_types'].update({date: 'date'})
            _ordinal_features: List[str] = []
            if ordinal_features is not None:
                for ordinal in ordinal_features:
                    if ordinal in DATA_PROCESSING.get('original_features'):
                        _ordinal_features.append(ordinal)
                        DATA_PROCESSING['pre_defined_feature_types'].update({ordinal: 'float'})
            _categorical_features: List[str] = []
            if categorical_features is not None:
                for categorical in categorical_features:
                    if categorical in DATA_PROCESSING.get('original_features'):
                        _categorical_features.append(categorical)
                        DATA_PROCESSING['pre_defined_feature_types'].update({categorical: 'int'})
            _continuous_features: List[str] = []
            if continuous_features is not None:
                for continuous in continuous_features:
                    if continuous in DATA_PROCESSING.get('original_features'):
                        _continuous_features.append(continuous)
                        DATA_PROCESSING['pre_defined_feature_types'].update({continuous: 'float'})
            for feature in DATA_PROCESSING['original_features']:
                Log(write=not print_msg, level='info').log(msg='Feature type segmentation: Feature -> {}'.format(feature))
                _load_temp_files(features=[feature])
                _set_feature_types(df=DATA_PROCESSING.get('df'),
                                   features=[feature],
                                   continuous=_continuous_features,
                                   categorical=_categorical_features,
                                   ordinal=_ordinal_features,
                                   date=_date_features,
                                   id_text=_id_features,
                                   date_edges=date_edges
                                   )
                if unify_invalid_values:
                    self.unify_invalid_to_mis()
                if DATA_PROCESSING.get('missing_value_analysis'):
                    self.missing_data_analysis(update=False, features=[feature])
                if auto_typing:
                    self.auto_typing()
                if auto_cleaning:
                    self.auto_cleaning(missing_data=False,
                                       missing_data_threshold=0.999,
                                       invariant=True,
                                       duplicated_cases=True,
                                       duplicated_features=True
                                       )
                if auto_engineering:
                    self.auto_engineering(geo_features=DATA_PROCESSING.get('geo_features'), target_feature=DATA_PROCESSING.get('target'))
                if unify_invalid_values or auto_typing or auto_cleaning:
                    _save_temp_files(feature=feature)
                #if auto_text_mining and len(FEATURE_TYPES.get('id_text')) > 0:
                #    _potential_text_features: List[str] = FEATURE_TYPES.get('id_text')
                #    if kwargs.get('include_categorical') is not None:
                #        if kwargs.get('include_categorical'):
                #            _potential_text_features.extend(FEATURE_TYPES.get('categorical'))
                #    try:
                #        _text_miner: TextMiner = TextMiner(df=DATA_PROCESSING.get('df'),
                #                                           features=_potential_text_features,
                #                                           dask_client=self.dask_client,
                #                                           lang=kwargs.get('lang'),
                #                                           lang_model=kwargs.get('lang_model'),
                #                                           lang_model_size='sm' if kwargs.get(
                #                                               'lang_model_size') is None else kwargs.get(
                #                                               'lang_model_size'),
                #                                           auto_interpret_natural_language=auto_text_mining
                #                                           )
                #        TEXT_MINER['obj'] = _text_miner
                #        TEXT_MINER['segments'] = _text_miner.segments
                #        # if len(TEXT_MINER['segments']['enumeration']) > 0:
                #        #    for enumerated_feature in TEXT_MINER['segments']['enumeration']:
                #        #        self.splitter(sep=TEXT_MINER['obj'].enumeration.get(enumerated_feature), features=[enumerated_feature])
                #        del _text_miner
                #    except TextMinerException:
                #        Log(write=not print_msg, level='info', env='dev').log(msg='No text features found in data set')
                #    except OSError as e:
                #        Log(write=not print_msg, level='info', env='dev').log(
                #            msg='Error while loading language model:\n{}'.format(e))
                #    finally:
                #        del _potential_text_features
            if target_feature is not None:
                self.set_target(feature=target_feature)
            if write_note is not None:
                self.write_notepad(note=write_note)
            self.kwargs: dict = {} if kwargs is None else kwargs
            Log(write=not print_msg, level='info', env='dev').log(msg='Finished ... Happy Engineering :)')

    def _critic(self, action: str, actor: str, inter_actor: str, n_imp_features: int = 50) -> dict:
        """
        Internal critic to evaluate actions by actor and choosing a potentially better one if action is not satisfying

        :param action: str
            Name of the action to take

        :param actor: str
            Name of the actor

        :param inter_actor: str
            Name of the inter actor

        :param n_imp_features: int
            Maximum number of important features to use
        """
        _critic: dict = dict(duplicated=False, recommender=dict(action=None, inter_actor=None))
        try:
            _actor_meta_data: pd.DataFrame = pd.DataFrame(data=DATA_PROCESSING['actor_memory']['action_config'])
        except ValueError:
            #for val in DATA_PROCESSING['actor_memory']['action_config'].keys():
            #    print(val, len(DATA_PROCESSING['actor_memory']['action_config'][val]), DATA_PROCESSING['actor_memory']['action_config'][val])
            return _critic
        if action == 'one_hot_merger':
            for one_hot_feature in actor.split('__m__'):
                if one_hot_feature in inter_actor.split('__m__'):
                    _critic.update(dict(duplicated=True))
                    break
            if not _critic.get('duplicated'):
                if DATA_PROCESSING.get('imp_features') is not None:
                    if inter_actor not in DATA_PROCESSING.get('imp_features')[0:n_imp_features]:
                        _c: int = 0
                        _max_critic_trials: int = 50
                        _critic['recommender'].update({'action': 'one_hot_merger'})
                        while True:
                            _critic['recommender'].update({'inter_actor': np.random.choice(a=DATA_PROCESSING.get('imp_features')[0:n_imp_features])})
                            if _critic['recommender']['inter_actor'] not in _actor_meta_data.loc[_actor_meta_data['actor'] == action, 'inter_actor'].values:
                                _critic.update(dict(duplicated=True))
                                break
                            else:
                                _c += 1
                            if _c == _max_critic_trials:
                                break
        else:
            if action in _actor_meta_data['action'].values.tolist():
                if actor in _actor_meta_data.loc[_actor_meta_data['action'] == action, ['actor']].values.tolist():
                    if action in PROCESSING_ACTION_SPACE['continuous']['interaction']:
                        _critic.update(dict(duplicated=True))
                    elif action.replace('interaction_', '') in PROCESSING_ACTION_SPACE['continuous']['interaction']:
                        _critic.update(dict(duplicated=True))
                    elif action.replace('self_interaction_', '') in PROCESSING_ACTION_SPACE['continuous']['self_interaction']:
                        _critic.update(dict(duplicated=True))
                    elif action.replace('self_interaction_', '') in PROCESSING_ACTION_SPACE['ordinal']['self_interaction']:
                        _critic.update(dict(duplicated=True))
                    else:
                        _actor_meta_data = _actor_meta_data.query(expr='action=="{}" and actor=="{}"'.format(action, actor))
                        if inter_actor in _actor_meta_data['inter_actor'].values.tolist():
                            _critic.update(dict(duplicated=True))
            if _critic.get('duplicated'):
                _critic['recommender'].update(self._recommender(actor=actor, actor_meta_data=_actor_meta_data))
        del _actor_meta_data
        return _critic

    @staticmethod
    def _one_hot_merger(features: List[str]):
        """
        Merge one-hot encoded features (used only in method "act" for reinforcement feature learning)

        :param features: List[str]
            Name of the one-hot encoded features to merge together
        """
        _load_temp_files(features=features)
        _data: np.array = np.zeros(shape=len(DATA_PROCESSING['df']))
        for feature in features:
            _data = _data + DATA_PROCESSING['df'][feature].values
        _data[_data > 1] = 1
        _process_handler(action='add',
                         feature=features[0],
                         new_feature='{}__m__{}'.format(features[0], features[1]),
                         process='interaction|one_hot',
                         meth='one_hot_merger',
                         param=dict(),
                         data=_data,
                         force_type='categorical',
                         special_replacement=False,
                         msg='Generated one-hot encoded feature by merging {} and {}'.format(features[0], features[1]),
                         obj=features[1]
                         )
        del _data
        DATA_PROCESSING['last_action'] = 'one_hot_merger'

    @staticmethod
    def _recommender(actor: str, actor_meta_data: pd.DataFrame) -> dict:
        """
        Internal recommendation for critic regarding to choosing the next best action

        :param actor: str
            Name of the actor

        :param actor_meta_data: pd.DataFrame
            Meta data set generated by actor

        :return dict:
            Recommendation regarding action, actor, inter_actor and instability
        """
        _recommendation: dict = dict(action=None, inter_actor=None)
        _supported_actions: List[str] = []
        _actor_type: str = DATA_PROCESSING['actor_memory']['action_config']['actor_type']
        if _actor_type == 'categorical':
            # TODO: Evaluate inheritance by checking processing relations
            pass
        else:
            for action in PROCESSING_ACTION_SPACE[_actor_type].keys():
                if action == 'transformation':
                    _supported_actions.extend(PROCESSING_ACTION_SPACE[_actor_type][action])
                else:
                    _actions: List[str] = PROCESSING_ACTION_SPACE[_actor_type][action]
                    _supported_actions.extend(['{}_{}'.format(action, meth) for meth in _actions])
            for supported_action in _supported_actions:
                if supported_action not in actor_meta_data['action'].values.tolist():
                    _recommendation.update({'action': supported_action})
            if _recommendation['action'] is None:
                _trials: int = 0
                _all_actors: List[str] = list(set(actor_meta_data['actor'].values.tolist() + actor_meta_data['inter_actor'].values.tolist()))
                while _recommendation.get('inter_actor') is None:
                    _trials += 1
                    _interaction: str = 'interaction_{}'.format(np.random.choice(a=PROCESSING_ACTION_SPACE[_actor_type]['interaction']))
                    _actor_meta_data: pd.DataFrame = actor_meta_data.query(expr='actor=="{}" & action=="{}"'.format(actor, _interaction))
                    _interactors: List[str] = actor_meta_data.loc['inter_actor'].unique()
                    for a in _all_actors:
                        if a not in _interactors:
                            _recommendation.update({'inter_actor': a})
                            break
                    if _trials > (len(_all_actors) * 2):
                        break
        del _actor_meta_data
        return _recommendation

    def act(self,
            actor: str,
            inter_actors: List[str],
            force_action: str = None,
            alternative_actions: List[str] = None
            ):
        """
        Run feature engineering action in an reinforcement learning environment

        :param actor: str
            Name of the feature to process

        :param inter_actors: List[str]
            Name of the features to use potentially for interaction with main actor

        :param force_action: str
            Name of the action to reinforce

        :param alternative_actions: List[str]
            Name of the actions to take if "force_action" generated an unstable (over-engineered) feature
        """
        if not DATA_PROCESSING.get('activate_actor'):
            raise FeatureEngineerException('Actor is deactivated and should only be used in a reinforcement learning environment (for activation set class parameter "activate_actor" == True or run class method "activate_actor()").')
        if DATA_PROCESSING.get('multi_threading'):
            self.reset_multi_threading()
        DATA_PROCESSING['avoid_overwriting'] = False
        _actor: str = actor
        _force_action: str = force_action
        _force_inter_actor: str = None
        _compatible_actors: List[str] = FEATURE_TYPES.get('ordinal') + FEATURE_TYPES.get('continuous')
        _stop: bool = False
        _trial: int = 0
        _max_trials: int = len(inter_actors) * 2
        # Categorical feature learning:
        if _actor not in _compatible_actors:
            _categorical_interactor: str = ''
            while DATA_PROCESSING.get('act_trials') < _max_trials:
                _categorical_interactor = np.random.choice(a=inter_actors)
                if actor == _categorical_interactor:
                    DATA_PROCESSING['act_trials'] += 1
                else:
                    _critic_response: dict = self._critic(action='one_hot_merger', actor=actor, inter_actor=_categorical_interactor)
                    if _critic_response.get('duplicated'):
                        DATA_PROCESSING['act_trials'] += 1
                    else:
                        break
            if actor != _categorical_interactor:
                DATA_PROCESSING['actor_memory']['action_config']['id'].append(len(DATA_PROCESSING['actor_memory']['action_config']['id']))
                DATA_PROCESSING['actor_memory']['action_config']['action'].append('one_hot_merger')
                DATA_PROCESSING['actor_memory']['action_config']['action_type'].append('interaction')
                DATA_PROCESSING['actor_memory']['action_config']['actor'].append(actor)
                DATA_PROCESSING['actor_memory']['action_config']['inter_actor'].append(_categorical_interactor)
                DATA_PROCESSING['actor_memory']['action_config']['actor_type'].append('categorical')
                DATA_PROCESSING['actor_memory']['action_config']['cleaned'].append(0)
                DATA_PROCESSING['actor_memory']['action_config']['unstable'].append(0)
                DATA_PROCESSING['actor_memory']['bad_actors'].append('None')
                DATA_PROCESSING['actor_memory']['missing_data'].append('None')
                self._one_hot_merger(features=[actor, _categorical_interactor])
        # (Semi-) continuous feature learning:
        while True:
            if _actor not in _compatible_actors:
                break
            if DATA_PROCESSING.get('act_trials') < _max_trials:
                if _force_action is None:
                    if (_actor in ALL_FEATURES) and (_actor in _compatible_actors):
                        _action: str = np.random.choice(a=list(PROCESSING_ACTION_SPACE['continuous'].keys()))
                        if (actor in FEATURE_TYPES.get('ordinal')) or (_action in ['interaction', 'self_interaction']):
                            if len(inter_actors) > 0:
                                while True:
                                    _supporting_actor: str = np.random.choice(a=inter_actors)
                                    if (_supporting_actor in ALL_FEATURES) and (
                                            _supporting_actor in _compatible_actors):
                                        break
                                    if _trial == _max_trials:
                                        _stop = True
                                        break
                                if _stop:
                                    if _supporting_actor not in ALL_FEATURES:
                                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='No interaction feature found in data set'.format(_actor))
                                        break
                                    if _supporting_actor not in _compatible_actors:
                                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='No interaction feature is numeric'.format(_actor))
                                        break
                                else:
                                    if _actor == _supporting_actor:
                                        _interaction: str = np.random.choice(a=PROCESSING_ACTION_SPACE['continuous']['self_interaction'])
                                        _critic_response: dict = self._critic(action='self_interaction_{}'.format(_interaction), actor=_actor, inter_actor=_actor)
                                        if _critic_response.get('duplicated'):
                                            _force_action = _critic_response['recommender']['action']
                                            _force_inter_actor = _critic_response['recommender']['inter_actor']
                                        else:
                                            self.self_interaction(features=[_actor],
                                                                  addition=True if 'addition' == _interaction else False,
                                                                  multiplication=True if 'multiplication' == _interaction else False
                                                                  )
                                            DATA_PROCESSING['last_action'] = 'self_interaction_{}'.format(_interaction)
                                            DATA_PROCESSING['actor_memory']['action_config']['id'].append(len(DATA_PROCESSING['actor_memory']['action_config']['id']))
                                            DATA_PROCESSING['actor_memory']['action_config']['action'].append(DATA_PROCESSING.get('last_action'))
                                            DATA_PROCESSING['actor_memory']['action_config']['action_type'].append('self_interaction')
                                            DATA_PROCESSING['actor_memory']['action_config']['actor'].append(_actor)
                                            DATA_PROCESSING['actor_memory']['action_config']['actor_type'].append('continuous')
                                            DATA_PROCESSING['actor_memory']['action_config']['inter_actor'].append(_actor)
                                            if self.is_unstable(feature=self.get_last_generated_feature()) or self.get_last_generated_feature() == '':
                                                DATA_PROCESSING['actor_memory']['action_config']['unstable'].append(1)
                                                if self.get_last_generated_feature() != '':
                                                    self.clean(markers=dict(features=[self.get_last_generated_feature()]))
                                                    DATA_PROCESSING['actor_memory']['bad_actors'].append(_actor)
                                                    DATA_PROCESSING['actor_memory']['action_config']['cleaned'].append(1)
                                                else:
                                                    DATA_PROCESSING['actor_memory']['bad_actors'].append('None')
                                                    DATA_PROCESSING['actor_memory']['action_config']['cleaned'].append(0)
                                                DATA_PROCESSING['act_trials'] += 1
                                            else:
                                                DATA_PROCESSING['actor_memory']['bad_actors'].append('None')
                                                DATA_PROCESSING['actor_memory']['action_config']['cleaned'].append(0)
                                                DATA_PROCESSING['actor_memory']['action_config']['unstable'].append(0)
                                                break
                                    else:
                                        _interaction: str = np.random.choice(a=PROCESSING_ACTION_SPACE['continuous']['interaction'])
                                        _critic_response: dict = self._critic(action='interaction_{}'.format(_interaction), actor=_actor, inter_actor=_supporting_actor)
                                        if _critic_response.get('duplicated'):
                                            _force_action = _critic_response['recommender']['action']
                                            _force_inter_actor = _critic_response['recommender']['inter_actor']
                                        else:
                                            self.interaction(features=[_actor, _supporting_actor],
                                                             addition=True if 'addition' == _interaction else False,
                                                             subtraction=True if 'subtraction' == _interaction else False,
                                                             multiplication=True if 'multiplication' == _interaction else False,
                                                             division=True if 'division' == _interaction else False
                                                             )
                                            DATA_PROCESSING['last_action'] = 'interaction_{}'.format(_interaction)
                                            DATA_PROCESSING['actor_memory']['action_config']['id'].append(len(DATA_PROCESSING['actor_memory']['action_config']['id']))
                                            DATA_PROCESSING['actor_memory']['action_config']['action'].append(DATA_PROCESSING.get('last_action'))
                                            DATA_PROCESSING['actor_memory']['action_config']['action_type'].append('interaction')
                                            DATA_PROCESSING['actor_memory']['action_config']['actor'].append(_actor)
                                            DATA_PROCESSING['actor_memory']['action_config']['actor_type'].append('continuous')
                                            DATA_PROCESSING['actor_memory']['action_config']['inter_actor'].append(_supporting_actor)
                                            if self.is_unstable(feature=self.get_last_generated_feature()) or self.get_last_generated_feature() == '':
                                                DATA_PROCESSING['actor_memory']['action_config']['unstable'].append(1)
                                                if self.get_last_generated_feature() != '':
                                                    self.clean(markers=dict(features=[self.get_last_generated_feature()]))
                                                    DATA_PROCESSING['actor_memory']['bad_actors'].append(_actor)
                                                    DATA_PROCESSING['actor_memory']['action_config']['cleaned'].append(1)
                                                else:
                                                    DATA_PROCESSING['actor_memory']['bad_actors'].append('None')
                                                    DATA_PROCESSING['actor_memory']['action_config']['cleaned'].append(0)
                                                DATA_PROCESSING['act_trials'] += 1
                                            else:
                                                DATA_PROCESSING['actor_memory']['bad_actors'].append('None')
                                                DATA_PROCESSING['actor_memory']['action_config']['cleaned'].append(0)
                                                DATA_PROCESSING['actor_memory']['action_config']['unstable'].append(0)
                                                break
                            else:
                                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='No interaction features found'.format(inter_actors))
                                DATA_PROCESSING['act_trials'] += 1
                        else:
                            DATA_PROCESSING['last_action'] = np.random.choice(a=PROCESSING_ACTION_SPACE['continuous']['transformation'])
                            _critic_response: dict = self._critic(action=DATA_PROCESSING['last_action'], actor=_actor, inter_actor='')
                            if _critic_response.get('duplicated'):
                                _force_action = _critic_response['recommender']['action']
                                _force_inter_actor = _critic_response['recommender']['inter_actor']
                            else:
                                getattr(self, DATA_PROCESSING['last_action'])(features=[_actor])
                                DATA_PROCESSING['actor_memory']['action_config']['id'].append(len(DATA_PROCESSING['actor_memory']['action_config']['id']))
                                DATA_PROCESSING['actor_memory']['action_config']['action'].append(DATA_PROCESSING.get('last_action'))
                                DATA_PROCESSING['actor_memory']['action_config']['action_type'].append('transformation')
                                DATA_PROCESSING['actor_memory']['action_config']['actor'].append(_actor)
                                DATA_PROCESSING['actor_memory']['action_config']['actor_type'].append('continuous')
                                DATA_PROCESSING['actor_memory']['action_config']['inter_actor'].append('None')
                                if self.is_unstable(feature=self.get_last_generated_feature()) or self.get_last_generated_feature() == '':
                                    DATA_PROCESSING['actor_memory']['action_config']['unstable'].append(1)
                                    if self.get_last_generated_feature() != '':
                                        self.clean(markers=dict(features=[self.get_last_generated_feature()]))
                                        DATA_PROCESSING['actor_memory']['bad_actors'].append(_actor)
                                        DATA_PROCESSING['actor_memory']['action_config']['cleaned'].append(1)
                                    else:
                                        DATA_PROCESSING['actor_memory']['bad_actors'].append('None')
                                        DATA_PROCESSING['actor_memory']['action_config']['cleaned'].append(0)
                                    DATA_PROCESSING['act_trials'] += 1
                                else:
                                    DATA_PROCESSING['actor_memory']['bad_actors'].append('None')
                                    DATA_PROCESSING['actor_memory']['action_config']['cleaned'].append(0)
                                    DATA_PROCESSING['actor_memory']['action_config']['unstable'].append(0)
                                    break
                    else:
                        if _actor not in ALL_FEATURES:
                            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Feature ({}) not found in data set'.format(_actor))
                            break
                        if _actor not in _compatible_actors:
                            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Feature ({}) is not numeric'.format(_actor))
                            break
                else:
                    if _actor in ALL_FEATURES:
                        _alternative_actions: List[str] = alternative_actions
                        if _alternative_actions is not None:
                            if len(_alternative_actions) == 0:
                                _alternative_actions = None
                        if _force_action in PROCESSING_ACTION_SPACE['continuous']['transformation']:
                            getattr(self, _force_action)(features=[_actor])
                            DATA_PROCESSING['last_action'] = _force_action
                            DATA_PROCESSING['actor_memory']['action_config']['id'].append(len(DATA_PROCESSING['actor_memory']['action_config']['id']))
                            DATA_PROCESSING['actor_memory']['action_config']['action'].append(DATA_PROCESSING.get('last_action'))
                            DATA_PROCESSING['actor_memory']['action_config']['action_type'].append('transformation')
                            DATA_PROCESSING['actor_memory']['action_config']['actor'].append(_actor)
                            DATA_PROCESSING['actor_memory']['action_config']['actor_type'].append('continuous')
                            DATA_PROCESSING['actor_memory']['action_config']['inter_actor'].append('None')
                            if self.is_unstable(feature=self.get_last_generated_feature()) or self.get_last_generated_feature() == '':
                                DATA_PROCESSING['actor_memory']['action_config']['unstable'].append(1)
                                if self.get_last_generated_feature() != '':
                                    self.clean(markers=dict(features=[self.get_last_generated_feature()]))
                                    DATA_PROCESSING['actor_memory']['bad_actors'].append(_actor)
                                    DATA_PROCESSING['actor_memory']['action_config']['cleaned'].append(1)
                                else:
                                    DATA_PROCESSING['actor_memory']['bad_actors'].append('None')
                                    DATA_PROCESSING['actor_memory']['action_config']['cleaned'].append(0)
                                DATA_PROCESSING['act_trials'] += 1
                            else:
                                DATA_PROCESSING['actor_memory']['bad_actors'].append('None')
                                DATA_PROCESSING['actor_memory']['action_config']['cleaned'].append(0)
                                DATA_PROCESSING['actor_memory']['action_config']['unstable'].append(0)
                                break
                        else:
                            _action_type: List[str] = _force_action.split('_')
                            if _action_type[0] == 'interaction':
                                if _force_inter_actor is None:
                                    while True:
                                        _supporting_actor: str = np.random.choice(a=inter_actors)
                                        if (_supporting_actor in DATA_PROCESSING['df'].columns) and (
                                                _supporting_actor in _compatible_actors):
                                            break
                                        if _trial == _max_trials:
                                            _stop = True
                                            break
                                else:
                                    _supporting_actor: str = _force_inter_actor
                                if _stop:
                                    if _supporting_actor not in DATA_PROCESSING['df'].columns:
                                        Log(write=not DATA_PROCESSING.get('show_msg')).log(
                                            msg='No interaction feature found in data set'.format(_actor))
                                    if _supporting_actor not in _compatible_actors:
                                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='No interaction feature is numeric'.format(_actor))
                                self.interaction(features=[_actor, _supporting_actor],
                                                 addition=True if 'addition' == _action_type[1] else False,
                                                 subtraction=True if 'subtraction' == _action_type[1] else False,
                                                 multiplication=True if 'multiplication' == _action_type[1] else False,
                                                 division=True if 'division' == _action_type[1] else False
                                                 )
                                DATA_PROCESSING['last_action'] = 'interaction_{}'.format(_action_type[1])
                                DATA_PROCESSING['actor_memory']['action_config']['id'].append(len(DATA_PROCESSING['actor_memory']['action_config']['id']))
                                DATA_PROCESSING['actor_memory']['action_config']['action'].append(DATA_PROCESSING.get('last_action'))
                                DATA_PROCESSING['actor_memory']['action_config']['action_type'].append('interaction')
                                DATA_PROCESSING['actor_memory']['action_config']['actor'].append(_actor)
                                DATA_PROCESSING['actor_memory']['action_config']['actor_type'].append('continuous')
                                DATA_PROCESSING['actor_memory']['action_config']['inter_actor'].append(_supporting_actor)
                                if self.is_unstable(feature=self.get_last_generated_feature()) or self.get_last_generated_feature() == '':
                                    DATA_PROCESSING['actor_memory']['action_config']['unstable'].append(1)
                                    if self.get_last_generated_feature() != '':
                                        self.clean(markers=dict(features=[self.get_last_generated_feature()]))
                                        DATA_PROCESSING['actor_memory']['bad_actors'].append(_actor)
                                        DATA_PROCESSING['actor_memory']['action_config']['cleaned'].append(1)
                                    else:
                                        DATA_PROCESSING['actor_memory']['bad_actors'].append('None')
                                        DATA_PROCESSING['actor_memory']['action_config']['cleaned'].append(0)
                                    DATA_PROCESSING['act_trials'] += 1
                                else:
                                    DATA_PROCESSING['actor_memory']['bad_actors'].append('None')
                                    DATA_PROCESSING['actor_memory']['action_config']['cleaned'].append(0)
                                    DATA_PROCESSING['actor_memory']['action_config']['unstable'].append(0)
                                    break
                            elif _action_type[0] == 'self':
                                DATA_PROCESSING['last_action'] = 'self_interaction_{}'.format(_action_type[2])
                                self.self_interaction(features=[_actor],
                                                      addition=True if 'addition' == _action_type[2] else False,
                                                      multiplication=True if 'multiplication' == _action_type[2] else False
                                                      )
                                DATA_PROCESSING['actor_memory']['action_config']['id'].append(len(DATA_PROCESSING['actor_memory']['action_config']['id']))
                                DATA_PROCESSING['actor_memory']['action_config']['action'].append(DATA_PROCESSING.get('last_action'))
                                DATA_PROCESSING['actor_memory']['action_config']['action_type'].append('self_interaction')
                                DATA_PROCESSING['actor_memory']['action_config']['actor'].append(_actor)
                                DATA_PROCESSING['actor_memory']['action_config']['actor_type'].append('continuous')
                                DATA_PROCESSING['actor_memory']['action_config']['inter_actor'].append(_actor)
                                if self.is_unstable(feature=self.get_last_generated_feature()) or self.get_last_generated_feature() == '':
                                    DATA_PROCESSING['actor_memory']['action_config']['unstable'].append(1)
                                    if self.get_last_generated_feature() != '':
                                        self.clean(markers=dict(features=[self.get_last_generated_feature()]))
                                        DATA_PROCESSING['actor_memory']['bad_actors'].append(_actor)
                                        DATA_PROCESSING['actor_memory']['action_config']['cleaned'].append(1)
                                    else:
                                        DATA_PROCESSING['actor_memory']['bad_actors'].append('None')
                                        DATA_PROCESSING['actor_memory']['action_config']['cleaned'].append(0)
                                    DATA_PROCESSING['act_trials'] += 1
                                else:
                                    DATA_PROCESSING['actor_memory']['bad_actors'].append('None')
                                    DATA_PROCESSING['actor_memory']['action_config']['cleaned'].append(0)
                                    DATA_PROCESSING['actor_memory']['action_config']['unstable'].append(0)
                                    break
                            else:
                                DATA_PROCESSING['act_trials'] += 1
                                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Reinforced action ({}) not supported'.format(_force_action))
                    else:
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Feature ({}) not found in data set'.format(_actor))
                        break
            else:
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Maximum of action re-trials ({}) exceeded. No feature generated'.format(_max_trials))
                break
        _last_generated_feature: str = self.get_last_generated_feature()
        if actor in _compatible_actors and _last_generated_feature != '' and _last_generated_feature in ALL_FEATURES:
            _load_temp_files(features=[actor, _last_generated_feature])
            if MissingDataAnalysis(df=DATA_PROCESSING.get('df')[[_actor, _last_generated_feature]]).has_nan():
                _invariant_features: List[str] = EasyExploreUtils().get_invariant_features(df=DATA_PROCESSING.get('df')[[_actor, _last_generated_feature]])
                _duplicate_features: dict = EasyExploreUtils().get_duplicates(df=DATA_PROCESSING.get('df')[[_actor, _last_generated_feature]], cases=False, features=True)
                _invalid_features: List[str] = _invariant_features + _duplicate_features.get('features')
                if _last_generated_feature in _invalid_features:
                    self.clean(markers=dict(features=[_last_generated_feature]))
                    DATA_PROCESSING['actor_memory']['bad_actors'][-1] = _actor
                    DATA_PROCESSING['actor_memory']['action_config']['cleaned'][-1] = 1
                else:
                    self.missing_data_analysis(update=True)
                    try:
                        self.impute(features=[_last_generated_feature], multiple=True, multiple_meth='random')
                    except IndexError:
                        self.impute(features=[_last_generated_feature], impute_float_value=0.0, multiple=False)
                    #DATA_PROCESSING['actor_memory']['missing_data'].append('single_0')
                    DATA_PROCESSING['actor_memory']['missing_data'].append('multiple_random')
            else:
                DATA_PROCESSING['actor_memory']['missing_data'].append('None')
        else:
            DATA_PROCESSING['actor_memory']['missing_data'].append('None')
        DATA_PROCESSING['act_trials'] = 0
        DATA_PROCESSING['avoid_overwriting'] = True

    @staticmethod
    def activate_actor():
        """
        Activate internal reinforcement learning method "act"
        """
        DATA_PROCESSING['actor_memory'].update({'bad_actors': [],
                                                'missing_data': [],
                                                'action_config': dict(id=[],
                                                                      action=[],
                                                                      action_type=[],
                                                                      actor=[],
                                                                      actor_type=[],
                                                                      inter_actor=[],
                                                                      unstable=[],
                                                                      cleaned=[]
                                                                      )
                                                })
        DATA_PROCESSING['activate_actor'] = True
        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Internal reinforcement learning actor ("act") enabled')

    @staticmethod
    def active_since() -> str:
        """
        Get time of object initialization (object birth time)

        :return str:
            Timestamp
        """
        return 'Feature Engineer is active since: {}'.format(str(DATA_PROCESSING.get('active')))

    def auto_cleaning(self,
                      missing_data: bool = False,
                      missing_data_threshold: float = 0.999,
                      invariant: bool = True,
                      duplicated_cases: bool = True,
                      duplicated_features: bool = True,
                      unstable: bool = True
                      ):
        """
        Clean cases and features automatically that are ...
            1) ... containing high amount of missing values (higher than given threshold)
            2) ... invariant (features only)
            3) ... duplicates

        :param missing_data: bool
            Clean cases or features containing high amount of missing data

        :param missing_data_threshold: float
            Threshold for deciding to clean case or feature from data set

        :param invariant: bool
            Clean invariant features from data set

        :param duplicated_cases: bool
            Clean duplicate cases from data set

        :param duplicated_features: bool
            Clean duplicate features from data set

        :param unstable: bool
            Clean unstable features containing values that are too big or small
        """
        _markers: dict = dict(cases=[], features=[])
        if invariant:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Check invariant features ...')
            _invariant_features: List[str] = EasyExploreUtils().get_invariant_features(df=DATA_PROCESSING.get('df'))
            if len(_invariant_features):
                _markers['features'].extend(_invariant_features)
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Detected invariant features: {}'.format(_invariant_features))
        #if duplicated_cases or duplicated_features:
        #    Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Check duplicated cases / features ...')
        #    _duplicates: dict = EasyExploreUtils().get_duplicates(df=DATA_PROCESSING.get('df'), cases=duplicated_cases, features=duplicated_features)
        #    if len(_duplicates.get('cases')) > 0:
        #        _markers['cases'].extend(_duplicates.get('cases'))
        #        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Detected duplicated cases: {}'.format(_duplicates.get('cases')))
        #    if len(_duplicates.get('features')) > 0:
        #        _markers['cases'].extend(_duplicates.get('features'))
        #        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Detected duplicated features: {}'.format(_duplicates.get('features')))
        if missing_data:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Check invalid values ...')
            _mis_threshold: float = missing_data_threshold if missing_data_threshold > 0 else 0.999
            _mis_by_cases: dict = MissingDataAnalysis(df=DATA_PROCESSING.get('df'), percentages=True).freq_nan_by_cases()
            _mis_by_features: dict = MissingDataAnalysis(df=DATA_PROCESSING.get('df'), percentages=True).freq_nan_by_features()
            _drop_cases: List[int] = []
            _drop_features: List[int] = []
            for c, p in _mis_by_cases.items():
                if p >= _mis_threshold:
                    _drop_cases.append(c)
            for f, p in _mis_by_features.items():
                if p >= _mis_threshold:
                    _drop_features.append(f)
            _markers['cases'].extend(_drop_cases)
            _markers['features'].extend(_drop_features)
            if len(_drop_cases) > 0:
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Detected sparse cases: {}'.format(_drop_cases))
            if len(_drop_features) > 0:
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Detected sparse features: {}'.format(_drop_features))
        if unstable:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Check feature stability ...')
            self.clean_unstable_features()
        if len(_markers.get('cases')) > 0 or len(_markers.get('features')) > 0:
            _markers['cases'] = list(set(_markers['cases']))
            _markers['features'] = list(set(_markers['features']))
            self.clean(markers=_markers)

    def auto_engineering(self,
                         label_enc: bool = True,
                         interaction: bool = True,
                         disparity: bool = True,
                         time_disparity: bool = True,
                         one_hot_enc: bool = True,
                         geo_enc: bool = True,
                         geo_features: List[str] = None,
                         date_discretizing: bool = True,
                         binning: bool = False,
                         bins: int = 4,
                         scaling: bool = True,
                         log_transform: bool = True,
                         exp_transform: bool = True,
                         handle_missing_data: str = 'impute',
                         target_feature: str = None,
                         **kwargs
                         ):
        """
        Generate features automatically based on the measurement level of the data

        :param label_enc: bool
            Whether to generate label encoded features from categorical string features

        :param interaction: bool
            Whether to generate interactions between continuous and / or ordinal features

        :param disparity: bool
            Whether to generate disparity features using simple interactions

        :param time_disparity: bool
            Whether to generate time disparity of datetime features

        :param one_hot_enc: bool
            Whether to generate one-hot encoded features for categorical features

        :param geo_enc: bool
            Whether to generate transformed geo features by converting string geo data into latitude and longitude coordinates

        :param geo_features: List[str]
            Name of any geographical features like postal code, states, latitude & longitude

        :param date_discretizing: bool
            Whether to generate categorized features from datetime features

        :param binning: bool
            Whether to generate binned features from ordinal or continuous features

        :param bins: int
            Number of unsupervised bins to generate if "binning" is true

        :param scaling: str
            Name of the scaler to use for continuous features
                -> minmax: MinMax scaler
                -> robust: Robust scaler
                -> norm: Normalization
                -> standard: Standardization

        :param log_transform: bool
            Whether to generate features using logarithmic transformation

        :param exp_transform: bool
            Whether to generate features using exponential transformation

        :param handle_missing_data: str
            Strategy to handle missing data properly
                -> clean: Clean missing data from data set
                -> impute: Impute (replace) missing data using multiple imputation technique

        :param target_feature: str
            Name of the target feature (if true target feature and predictors are set)

        :param kwargs: dict
            Key-word arguments
        """
        Log(write=not DATA_PROCESSING.get('show_msg')).log('Start Auto Engineering ...')
        if handle_missing_data == 'clean':
            self.clean_nan()
        elif handle_missing_data.find('impute') >= 0:
            if handle_missing_data == 'impute_multi':
                self.impute(multiple=True)
            else:
                self.impute(multiple=False,
                            impute_float_value=0.0 if kwargs.get('impute_float_value') is None else kwargs.get('impute_float_value'),
                            )
        else:
            if MissingDataAnalysis(df=DATA_PROCESSING.get('df')).has_nan():
                Log(write=not DATA_PROCESSING.get('show_msg'), level='warning').log(msg='No method for handling missing data found. Feature engineering might be effected significantly')
        if label_enc:
            self.label_encoder(encode=True)
        if interaction:
            self.interaction_poly(degree=2 if kwargs.get('degree') is None else kwargs.get('degree'),
                                  interaction_only=False if kwargs.get('interaction_only') is None else kwargs.get('interaction_only'),
                                  include_bias=False if kwargs.get('include_bias') is None else kwargs.get('include_bias'),
                                  )
        if time_disparity:
            self.disparity(years=True if kwargs.get('years') is None else kwargs.get('years'),
                           months=True if kwargs.get('months') is None else kwargs.get('months'),
                           weeks=True if kwargs.get('weeks') is None else kwargs.get('weeks'),
                           days=True if kwargs.get('days') is None else kwargs.get('days'),
                           hours=True if kwargs.get('hours') is None else kwargs.get('hours'),
                           minutes=True if kwargs.get('minutes') is None else kwargs.get('minutes'),
                           seconds=True if kwargs.get('seconds') is None else kwargs.get('seconds'),
                           digits=6 if kwargs.get('digits') is None else kwargs.get('digits')
                           )
        if scaling:
            self.scaling_robust()
            self.scaling_minmax()
            self.normalizer(norm_meth='l2' if kwargs.get('norm_meth') is None else kwargs.get('norm_meth'))
            self.standardizer(with_mean=True if kwargs.get('with_mean') is None else kwargs.get('with_mean'),
                              with_std=True if kwargs.get('with_std') is None else kwargs.get('with_std')
                              )
        if log_transform:
            self.log_transform(skewness_test=False)
        if exp_transform:
            self.exp_transform(skewness_test=False)
        if disparity:
            self.interaction(addition=True if kwargs.get('addition') is None else kwargs.get('addition'),
                             subtraction=True if kwargs.get('subtraction') is None else kwargs.get('subtraction'),
                             multiplication=True if kwargs.get('multiplication') is None else kwargs.get('multiplication'),
                             division=True if kwargs.get('division') is None else kwargs.get('division')
                             )
        if geo_features is not None:
            if len(geo_features) > 0:
                if geo_enc:
                    self.geo_encoder(geo_features=geo_features,
                                     provider='arcgis' if kwargs.get('provider') is None else kwargs.get('provider'),
                                     max_rows=1 if kwargs.get('max_rows') is None else kwargs.get('max_rows')
                                     )
        self.clean_unstable_features()
        if date_discretizing:
            self.date_categorizer(year=True if kwargs.get('year') is None else kwargs.get('year'),
                                  month=True if kwargs.get('month') is None else kwargs.get('month'),
                                  day=True if kwargs.get('day') is None else kwargs.get('day'),
                                  hour=True if kwargs.get('hour') is None else kwargs.get('hour'),
                                  minute=True if kwargs.get('minute') is None else kwargs.get('minute'),
                                  second=True if kwargs.get('second') is None else kwargs.get('second')
                                  )
        if binning:
            self.binning(supervised=True,
                         optimal=True,
                         optimal_meth='bayesian_blocks' if kwargs.get('optimal_meth') is None else kwargs.get('optimal_meth')
                         )
            self.binning(supervised=False,
                         optimal=False,
                         bins=bins
                         )
        if one_hot_enc:
            self.one_hot_encoder(threshold=kwargs.get('threshold'))
        if target_feature is not None:
            self.set_target(feature=target_feature)
        if DATA_PROCESSING.get('target') is not None:
            self.set_predictors(features=list(DATA_PROCESSING['df'].columns), exclude_original_data=False)
        self.save(file_path=kwargs.get('file_path'))
        Log(write=not DATA_PROCESSING.get('show_msg')).log('Finished Auto Engineering')

    def auto_typing(self):
        """
        Detect automatically whether data types in Pandas DataFrame are correctly typed from an analytical point of view
        """
        _features: List[str] = [feature for feature in DATA_PROCESSING.get('df').columns if feature not in DATA_PROCESSING.get('pre_defined_feature_types').keys()]
        if list(TEMP_INDEXER.keys())[0] in _features:
            del _features[_features.index(list(TEMP_INDEXER.keys())[0])]
        _check_dtypes: dict = EasyExploreUtils().check_dtypes(df=DATA_PROCESSING.get('df'),
                                                              feature_types=FEATURE_TYPES,
                                                              date_edges=DATA_PROCESSING.get('date_edges')
                                                              )
        if len(DATA_PROCESSING.get('pre_defined_feature_types').keys()) > 0:
            _check_dtypes['conversion'].update(DATA_PROCESSING.get('pre_defined_feature_types'))
        if len(_check_dtypes.get('conversion').keys()) > 0:
            self.type_conversion(feature_type=_check_dtypes.get('conversion'))

    @staticmethod
    @FeatureOrchestra(meth='binarizer', feature_types=['ordinal'])
    def binarizer(threshold: float = 0.9, features: List[str] = None):
        """
        Binarize counting features

        :param threshold: float
            Threshold for binarizing

        :param features: List[str]
            Features to binarize
        """
        for feature in features:
            _load_temp_files(features=[feature])
            _data: str = DATA_PROCESSING.get('df')[feature].values
            _binarizer = Binarizer(threshold=threshold)
            _binarizer.fit(X=np.reshape(_data, (-1, 1)))
            _process_handler(action='add',
                             feature=feature,
                             new_feature='{}_{}'.format(feature, DATA_PROCESSING['suffixes'].get('bin')) if DATA_PROCESSING.get('generate_new_feature') else feature,
                             process='encoder|bin',
                             meth='binarizer',
                             param=dict(threshold=threshold),
                             data=_binarizer.transform(np.reshape(_data, (-1, 1)))[0],
                             obj=_binarizer
                             )
            _update_feature_types(feature=feature)
            del _data
            Log(write=not DATA_PROCESSING.get('show_msg')).log('Transformed feature {} using binarization'.format(feature))

    @staticmethod
    @FeatureOrchestra(meth='binning', feature_types=['continuous'])
    def binning(supervised: bool,
                edges: List[float] = None,
                bins: int = None,
                features: List[str] = None,
                optimal: bool = False,
                optimal_meth: str = 'bayesian_blocks',
                predictors: List[str] = None,
                weight_feature: str = None,
                labels: List[str] = None,
                encode_meth: str = 'onehot',
                strategy: str = 'quantile'
                ):
        """
        Categorize or bin continuous features

        :param supervised: bool
            Whether a supervised binning should be applied or not

        :param bins: List[float]
            Edges of the bin categories

        :param features: List[str]
            Name of the features

        :param edges: int
            Equal-width of the applied unsupervised binning

        :param optimal: bool
            Whether a optimal binning in regards to another variable should be applied

        :param optimal_meth: str
            Name of the optimal binning method to use:
                -> chaid: CHAID algorithm (categorical features only)
                -> bayesian_blocks: Bayesian Blocks
                -> kbins: KBins-Discretizer

        :param predictors: List[str]
            Name of the predictors (only used if optimal_meth == chaid)

        :param weight_feature: str
            Name of the weighting feature

        :param labels: List[str]
            Labels

        :param encode_meth: str
            Encoding transformation method
                -> onehot: One-Hot encoding as sparse matrix
                -> onehot-dense: One-Hot encoding as dense array
                -> ordinal: Binary identifier encoding as integer value

        :param strategy: str
            Strategy to define width of the bins
                -> uniform: All bins in each feature have identical widths
                -> quantile: All bins in each feature have same number of points
                -> kmeans: Values in each bin have the same nearest center of a 1D k-means cluster
        """
        _param: dict = dict(supervised=supervised,
                            edges=edges,
                            bins=bins,
                            optimal=optimal,
                            optimal_meth=optimal_meth,
                            predictors=predictors,
                            weight_feature=weight_feature,
                            labels=labels,
                            encode_meth=encode_meth,
                            strategy=strategy
                            )
        for feature in features:
            _load_temp_files(features=[feature])
            if supervised:
                if optimal:
                    if optimal_meth == 'chaid':
                        _chaid = CHAIDDecisionTree()
                        if predictors is None:
                            _predictors: List[str] = FEATURE_TYPES.get('categorical') + FEATURE_TYPES.get('ordinal')
                            if feature in _predictors:
                                del _predictors[_predictors.index(feature)]
                        else:
                            _predictors: List[str] = predictors
                        #print(DATA_PROCESSING['df'][_predictors].values.compute())
                        #print(DATA_PROCESSING['df'][feature].values.compute())
                        _chaid.train(x=DATA_PROCESSING['df'][_predictors].values, y=np.reshape(DATA_PROCESSING['df'][feature].values, (-1, 1)))
                        _chaid_pred: np.array = _chaid.predict()
                        _process_handler(action='add',
                                         feature=feature,
                                         new_feature='{}_chaid'.format(feature) if DATA_PROCESSING.get('generate_new_feature') else feature,
                                         process='categorizer|continuous',
                                         meth='binning',
                                         param=_param,
                                         data=_chaid_pred.get('labels'),
                                         force_type='categorical',
                                         obj=_chaid_pred
                                         )
                        Log(write=not DATA_PROCESSING.get('show_msg')).log('Binned feature "{}" using CHAID'.format(feature))
                        del _chaid
                        del _chaid_pred
                    elif optimal_meth == 'bayesian_blocks':
                        _bayesian_blocks: dict = HappyLearningUtils().bayesian_blocks(df=DATA_PROCESSING['df'][feature])
                        _process_handler(action='add',
                                         feature=feature,
                                         new_feature='{}_blocks'.format(feature) if DATA_PROCESSING.get('generate_new_feature') else feature,
                                         process='categorizer|continuous',
                                         meth='binning',
                                         param=_param,
                                         data=np.array(_bayesian_blocks.get('labels')),
                                         force_type='categorical',
                                         obj=_bayesian_blocks
                                         )
                        Log(write=not DATA_PROCESSING.get('show_msg')).log('Binned feature "{}" using Bayesian Blocks'.format(feature))
                        del _bayesian_blocks
                    #elif optimal_meth == 'kbins':
                    #    _kbins_discretizer = KBinsDiscretizer(n_bins=bins,
                    #                                          encode=encode_meth,
                    #                                          strategy=strategy
                    #                                          )
                    #    _kbins_discretizer.fit(X=np.reshape(DATA_PROCESSING['df'][feature].values.compute(), (-1, 1)))
                    #    #print(_kbins_discretizer.transform(X=np.reshape(DATA_PROCESSING.get('df')[feature].values, (-1, 1))))
                    #    print(_kbins_discretizer.transform(X=np.reshape(DATA_PROCESSING.get('df')[feature].values.compute(), (-1, 1))))
                    #    _process_handler(action='add',
                    #                     feature=feature,
                    #                     new_feature='{}_kbins'.format(feature) if DATA_PROCESSING.get('generate_new_feature') else feature,
                    #                     process='categorizer|continuous',
                    #                     meth='binning',
                    #                     param=_param,
                    #                     data=_kbins_discretizer.transform(X=np.reshape(DATA_PROCESSING.get('df')[feature].values.compute(), (-1, 1))),
                    #                     force_type='categorical',
                    #                     obj=_kbins_discretizer
                    #                     )
                    #    del _kbins_discretizer
                    #    Log(write=not DATA_PROCESSING.get('show_msg')).log('Binned feature "{}" using K-Bins Discretizer'.format(feature))
                    else:
                        Log(write=not DATA_PROCESSING.get('show_msg')).log('Binning method ({}) not supported'.format(optimal_meth))
                else:
                    _labels = [i for i in range(0, len(edges) - 1, 1)] if labels is None else labels
                    if edges is None:
                        Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log('No edges for bins found')
                    _edges: np.array = pd.cut(x=DATA_PROCESSING.get('df')[feature], bins=edges, labels=None, retbins=False)
                    _process_handler(action='add',
                                     feature=feature,
                                     new_feature='{}_supervised'.format(feature) if DATA_PROCESSING.get('generate_new_feature') else feature,
                                     process='categorizer|continuous',
                                     meth='binning',
                                     param=_param,
                                     data=pd.cut(x=DATA_PROCESSING['df'][feature].compute(), bins=edges, labels=_labels, retbins=False),
                                     force_type='categorical',
                                     obj=_edges
                                     )
                    Log(write=not DATA_PROCESSING.get('show_msg')).log('Binned feature "{}" using customized edges'.format(feature))
            else:
                try:
                    _labels = [i for i in range(0, bins, 1)] if labels is None else labels
                    if bins is None:
                        Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log('No size for equal-width binning (unsupervised) found')
                    _edges: np.array = pd.cut(x=DATA_PROCESSING.get('df')[feature], bins=bins, labels=None, retbins=False)
                    _process_handler(action='add',
                                     feature=feature,
                                     new_feature='{}_bins_equal_width'.format(feature) if DATA_PROCESSING.get('generate_new_feature') else feature,
                                     process='categorizer|continuous',
                                     meth='binning',
                                     param=_param,
                                     data=pd.cut(x=DATA_PROCESSING['df'][feature].compute(), bins=bins, labels=_labels, retbins=False),
                                     force_type='categorical',
                                     obj=_edges
                                     )
                    Log(write=not DATA_PROCESSING.get('show_msg')).log('Binned feature "{}" using equal-width'.format(feature))
                except ValueError:
                    Log(write=not DATA_PROCESSING.get('show_msg')).log('Feature "{}" could not be binned using equal-width'.format(feature))

    @staticmethod
    @FeatureOrchestra(meth='box_cox_transform', feature_types=['continuous'])
    def box_cox_transform(features: List[str] = None,
                          normality_test: bool = True,
                          meth: str = 'shapiro-wilk',
                          alpha: float = 0.05
                          ):
        """
        Transform features by transforming into approximately normal distribution

        :param features: List[str]
            Name of the features

        :param normality_test: bool
            Whether to run normality test or not

        :param meth: str
            Name of the used method for normality testing

        :param alpha: float
            Threshold that indicates whether a hypothesis can be rejected or not
        """
        #if normality_test:
        #    if meth.find('shapiro') >= 0:
        #        _features = StatsUtils(data=DATA_PROCESSING.get('df'),
        #                               features=list(DATA_PROCESSING.get('df').columns)).normality_test(alpha=alpha, meth='shapiro-wilk')
        #    elif meth.find('anderson') >= 0:
        #        _features = StatsUtils(data=DATA_PROCESSING.get('df'),
        #                               features=list(DATA_PROCESSING.get('df').columns)).normality_test(alpha=alpha, meth='anderson-darling')
        #    elif meth.find('dagostino') >= 0:
        #        _features = StatsUtils(data=DATA_PROCESSING.get('df'),
        #                               features=list(DATA_PROCESSING.get('df').columns)).normality_test(alpha=alpha, meth='dagostino')
        #    else:
        #        _features = None
        #        Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log('Method ({}) for normality test not supported'.format(meth))
        #else:
        #    _features = list(DATA_PROCESSING.get('df').columns)
        if features is not None:
            for feature in features:
                _load_temp_files(features=[feature])
                _cleaned_feature = DATA_PROCESSING.get('df')[feature][~np.isnan(DATA_PROCESSING.get('df')[feature])]
                _l, _lambda_opt = boxcox(x=_cleaned_feature, lmbda=None, alpha=alpha)
                _process_handler(action='add',
                                 feature=feature,
                                 new_feature='{}_box_cox'.format(feature) if DATA_PROCESSING.get('generate_new_feature') else feature,
                                 process='scaler|box_cox',
                                 meth='box_cox_transform',
                                 param=dict(normality_test=normality_test, meth=meth, alpha=alpha),
                                 data=pd.DataFrame(data=boxcox(x=np.reshape(DATA_PROCESSING['df'][feature].values, (-1, 1)), lmbda=_lambda_opt, alpha=alpha)),
                                 obj=dict(l=_l, lambda_opt=_lambda_opt)
                                 )
                Log(write=not DATA_PROCESSING.get('show_msg')).log('Transformed feature {} using Box-Cox-Transformation'.format(feature))

    @staticmethod
    @FeatureOrchestra(meth='breakdown_stats', feature_types=['categorical', 'date'])
    def breakdown_stats(aggregation: dict, features: List[str] = None) -> pd.DataFrame:
        """
        Calculate breakdown statistics

        :param aggregation: dict
            Names of continuous features as keys and the aggregation function as values

        :param features: List[str]
            Name of the features

        :return: pd.DataFrame
            Unstacked breakdown statistics aggregated by continuous and grouped by categorical features
        """
        _group_features: List[str] = []
        for group in features:
            if group not in aggregation.keys():
                _group_features.append(group)
        if len(_group_features) == 0:
            raise FeatureEngineerException('No categorical features found')
        _features: List[str] = features + _group_features
        _load_temp_files(features=_features)
        return DATA_PROCESSING.get('df').groupby(_group_features).aggregate(aggregation).unstack()

    @staticmethod
    def clean(markers: Dict[str, list]):
        """
        Clean data set from specific cases or features

        :param markers: Dict[str, list]
            Cases and features to clean from the data set
        """
        if 'cases' not in markers.keys() and 'features' not in markers.keys():
            Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log('Neither cases nor features found')
        if markers.get('cases') is not None:
            if len(markers.get('cases')) > 0:
                DATA_PROCESSING['df'] = DATA_PROCESSING.get('df').drop(labels=markers['cases'], axis=0, errors='ignore')
                Log(write=not DATA_PROCESSING.get('show_msg')).log('Cleaned {} cases {}'.format(len(markers.get('cases')),
                                                                                                markers.get('cases') if len(markers.get('cases')) <= 10 else ''
                                                                                                )
                                                                   )
        if markers.get('features') is not None:
            if len(markers.get('features')) > 0:
                DATA_PROCESSING['df'] = DATA_PROCESSING.get('df').drop(labels=markers['features'], axis=1, errors='ignore')
                DATA_PROCESSING['cleaned_features'].extend([feature for feature in markers.get('features')])
                Log(write=not DATA_PROCESSING.get('show_msg')).log('Cleaned {} features {}'.format(len(markers.get('features')),
                                                                                                   markers.get('features') if len(markers.get('features')) <= 10 else ''
                                                                                                   )
                                                                   )
                for feature in markers.get('features'):
                    _process_handler(action='clean',
                                     feature=feature,
                                     new_feature=feature,
                                     process='mapper|clean',
                                     meth='clean',
                                     param=dict(markers=markers)
                                     )

    @staticmethod
    def clean_nan(other_mis: list = None):
        """
        Remove all cases containing missing values

        :param other_mis: list
            Values to convert to missing value NaN
        """
        DATA_PROCESSING['df'] = MissingDataAnalysis(df=DATA_PROCESSING.get('df'), other_mis=other_mis).clean_nan()

    @staticmethod
    def clean_unstable_features():
        """
        Clean (unstable) features containing values that are potentially too big or small for machine learning algorithms
        """
        _load_temp_files(features=FEATURE_TYPES.get('continuous'))
        _descriptives: pd.DataFrame = DATA_PROCESSING['df'][FEATURE_TYPES.get('continuous')].describe()
        _descriptives = _descriptives.transpose()
        _features: List[str] = _descriptives.loc[_descriptives['mean'] == np.inf, :].index.values.tolist()
        _features.extend(_descriptives.loc[_descriptives['mean'] == -np.inf, :].index.values.tolist())
        _features.extend(_descriptives.loc[_descriptives['mean'] == np.nan, :].index.values.tolist())
        if len(_features) > 0:
            _features = list(set(_features))
        for feature in _features:
            DATA_PROCESSING['cleaned_features'].append(feature)
            _process_handler(action='clean',
                             feature=feature,
                             new_feature=feature,
                             process='mapper|clean',
                             meth='clean_unstable_features',
                             param=dict()
                             )
            Log(write=not DATA_PROCESSING.get('show_msg')).log('Cleaned (dubious) feature "{}"'.format(feature))

    @staticmethod
    @FeatureOrchestra(meth='concat_text', feature_types=['id_text'])
    def concat_text(features: List[str] = None,
                    sep: str = '_',
                    by_col: bool = True,
                    feature_name: str = None
                    ):
        """
        Concatenate text by feature or case values

        :param features: List[str]
            Name of features

        :param sep: str
            Character used to separate values

        :param by_col: bool:
            Concatenate case values by features or feature values by cases

        :param feature_name: str
            Feature name of concatenated text feature
        """
        if len(features) == 0:
            Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log('No features found')
        elif len(features) == 1:
            Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log('Not enough features ({})'.format(features))
        else:
            if feature_name is None:
                _ft: str = '_'.join(features)
            else:
                if len(feature_name) == 0:
                    _ft: str = '_'.join(features)
                else:
                    _ft: str = feature_name
            for ft in features:
                _load_temp_files(features=[ft])
                if str(DATA_PROCESSING.get('df')[ft].dtype).find('object') == -1:
                    DATA_PROCESSING.get('df')[ft] = DATA_PROCESSING.get('df')[ft].astype(dtype=str)
                if any(DATA_PROCESSING.get('df')[ft].isnull()):
                    DATA_PROCESSING.get('df')[ft] = DATA_PROCESSING.get('df')[ft].replace(np.nan, 'NaN')
            if by_col:
                DATA_PROCESSING.get('df')[_ft] = DATA_PROCESSING.get('df')[features].transpose().apply(lambda x: sep.join(x.tolist()))
            else:
                DATA_PROCESSING.get('df')[_ft] = DATA_PROCESSING.get('df')[features].apply(lambda x: sep.join(x.tolist()))

    @staticmethod
    @FeatureOrchestra(meth='count_text', feature_types=['id_text'])
    def count_text(pattern: str, features: List[str] = None):
        """
        Count specific pattern in text feature

        :param pattern: str
            String pattern to count

        :param features: List[str]
            Name of the features
        """
        if len(pattern) > 0:
            for feature in features:
                _load_temp_files(features=[feature])
                _process_handler(action='add',
                                 feature=feature,
                                 new_feature='{}_count_{}'.format(feature, pattern),
                                 process='text|find|count',
                                 meth='count_text',
                                 param=dict(pattern=pattern),
                                 data=DATA_PROCESSING['df'][feature].str.count(pat=pattern),
                                 force_type='ordinal',
                                 obj={feature: pattern}
                                 )

    @staticmethod
    def data_export(file_path: str, create_dir: bool = True, overwrite: bool = False):
        """
        Export data set to local file

        :param file_path: str
            Complete file path of data set

        :param create_dir: bool
            Create directories if they do not exists

        :param overwrite: bool
            Overwrite file with same name or not
        """
        if file_path.find('.parquet') >= 0:
            DataExporter(obj=DATA_PROCESSING.get('df'), file_path=file_path, create_dir=create_dir, overwrite=overwrite).file()
        else:
            DataExporter(obj=DATA_PROCESSING.get('df').compute(), file_path=file_path, create_dir=create_dir, overwrite=overwrite).file()
        Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log('Data set saved as local file ({})'.format(file_path))

    @staticmethod
    def data_import(file_path: str, sep: str = ',', **kwargs):
        """
        Import data set from local file

        :param file_path: str
            Complete file path of data set

        :param sep: str
            Delimiter of data file

        :param kwargs: dict
            Key-word arguments for class DataImporter
        """
        global ALL_FEATURES
        global MERGES
        global PREDICTORS
        global FEATURE_TYPES
        global TEXT_MINER
        ALL_FEATURES = []
        MERGES = {}
        PREDICTORS = []
        FEATURE_TYPES = {ft: [] for ft in FEATURE_TYPES.keys()}
        TEXT_MINER = dict(obj=None, segments={}, data=None, generated_features=[], linguistic={})
        kwargs.update({'partitions': DATA_PROCESSING['partitions']})
        DATA_PROCESSING['df'] = DataImporter(file_path=file_path, as_data_frame=True, use_dask=True, sep=sep, **kwargs).file(table_name=kwargs.get('table_name'))
        DATA_PROCESSING['df'][DASK_INDEXER] = DATA_PROCESSING['df'].index.values
        DATA_PROCESSING['df'] = DATA_PROCESSING['df'].set_index(DASK_INDEXER)
        DATA_PROCESSING['n_cases'] = len(DATA_PROCESSING['df'])
        global TEMP_INDEXER
        TEMP_INDEXER = {'__index__': [i for i in range(0, DATA_PROCESSING['n_cases'], 1)]}
        if 'Unnamed: 0' in list(DATA_PROCESSING['df'].columns):
            del DATA_PROCESSING['df']['Unnamed: 0']
        DATA_PROCESSING.update({'original_features': DATA_PROCESSING.get('df').columns})
        DATA_PROCESSING['processing']['features']['raw'].update({feature: [] for feature in list(DATA_PROCESSING.get('df').columns)})
        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Data set loaded from local file ({})\nCases: {}\nFeatures: {}'.format(file_path,
                                                                                                                                      len(DATA_PROCESSING['df']),
                                                                                                                                      len(DATA_PROCESSING['df'].columns)
                                                                                                                                      )
                                                           )

    @staticmethod
    @FeatureOrchestra(meth='date_categorizer', feature_types=['date'])
    def date_categorizer(features: List[str] = None,
                         year: bool = True,
                         month: bool = True,
                         day: bool = True,
                         hour: bool = True,
                         minute: bool = True,
                         second: bool = True
                         ):
        """
        Extract categorical information from date features

        :param features: List[str]
            Name of features

        :param year: bool
            Extract year from date

        :param month: bool
            Extract month from date

        :param day: bool
            Extract day from date

        :param hour: bool
            Extract hour from date

        :param minute: bool
            Extract minute from date

        :param second: bool
            Extract second from date
        """
        for feature in features:
            _load_temp_files(features=[feature])
            if str(DATA_PROCESSING['df'][feature].dtype).find('date') >= 0:
                _date_feature: pd.DataFrame = DATA_PROCESSING['df'][feature]
            else:
                try:
                    _date_feature: pd.DataFrame = pd.to_datetime(DATA_PROCESSING['df'][feature])
                except (ValueError, TypeError):
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Feature "{}" could not be converted to datetime'.format(feature))
                    continue
            if year:
                DATA_PROCESSING['semi_engineered_features'].append('{}_year'.format(feature))
                _process_handler(action='add',
                                 feature=feature,
                                 new_feature='{}_year'.format(feature),
                                 process='categorizer|date',
                                 meth='date_categorizer',
                                 param=dict(year=year, month=month, day=day, hour=hour, minute=minute, second=second),
                                 data=_date_feature.dt.year.values,
                                 force_type='categorical'
                                 )
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformed date feature "{}" using date categorizer - year'.format(feature))
            if month:
                DATA_PROCESSING['semi_engineered_features'].append('{}_month'.format(feature))
                _process_handler(action='add',
                                 feature=feature,
                                 new_feature='{}_month'.format(feature),
                                 process='categorizer|date',
                                 meth='date_categorizer',
                                 param=dict(year=year, month=month, day=day, hour=hour, minute=minute, second=second),
                                 data=_date_feature.dt.month.values,
                                 force_type='categorical'
                                 )
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformed date feature "{}" using date categorizer - month'.format(feature))
            if day:
                DATA_PROCESSING['semi_engineered_features'].append('{}_day'.format(feature))
                _process_handler(action='add',
                                 feature=feature,
                                 new_feature='{}_day'.format(feature),
                                 process='categorizer|date',
                                 meth='date_categorizer',
                                 param=dict(year=year, month=month, day=day, hour=hour, minute=minute, second=second),
                                 data=_date_feature.dt.day.values,
                                 force_type='categorical'
                                 )
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformed date feature "{}" using date categorizer - day'.format(feature))
            if hour:
                DATA_PROCESSING['semi_engineered_features'].append('{}_hour'.format(feature))
                _process_handler(action='add',
                                 feature=feature,
                                 new_feature='{}_hour'.format(feature),
                                 process='categorizer|date',
                                 meth='date_categorizer',
                                 param=dict(year=year, month=month, day=day, hour=hour, minute=minute, second=second),
                                 data=_date_feature.dt.hour.values,
                                 force_type='categorical'
                                 )
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformed date feature "{}" using date categorizer - hour'.format(feature))
            if minute:
                DATA_PROCESSING['semi_engineered_features'].append('{}_minute'.format(feature))
                _process_handler(action='add',
                                 feature=feature,
                                 new_feature='{}_minute'.format(feature),
                                 process='categorizer|date',
                                 meth='date_categorizer',
                                 param=dict(year=year, month=month, day=day, hour=hour, minute=minute, second=second),
                                 data=_date_feature.dt.minute.values,
                                 force_type='categorical'
                                 )
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformed date feature "{}" using date categorizer - minute'.format(feature))
            if second:
                DATA_PROCESSING['semi_engineered_features'].append('{}_second'.format(feature))
                _process_handler(action='add',
                                 feature=feature,
                                 new_feature='{}_second'.format(feature),
                                 process='categorizer|date',
                                 meth='date_categorizer',
                                 param=dict(year=year, month=month, day=day, hour=hour, minute=minute, second=second),
                                 data=_date_feature.dt.second.values,
                                 force_type='categorical'
                                 )
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformed date feature "{}" using date categorizer - second'.format(feature))

    @staticmethod
    @FeatureOrchestra(meth='date_conversion', feature_types=['date'])
    def date_conversion(features: List[str] = None, fmt: str = None):
        """
        Convert datetime format as well as Pandas DataFrame dtype to datetime

        :param features: List[str]
            Name of the features

        :param fmt: str
            Datetime format
        """
        for feature in features:
            _load_temp_files(features=[feature])
            if fmt is None:
                _process_handler(action='add',
                                 feature=feature,
                                 new_feature=feature,
                                 process='typing|date',
                                 meth='date_conversion',
                                 param=dict(fmt=fmt),
                                 data=pd.to_datetime(arg=DATA_PROCESSING['df'][feature]),
                                 obj=dict(feature=None)
                                 )
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Feature {} converted to datetime'.format(feature))
            else:
                if str(DATA_PROCESSING['df'][feature].dtype).find('datetime') >= 0:
                    DATA_PROCESSING['df'][feature] = DATA_PROCESSING['df'][feature].astype(dtype=str)
                _process_handler(action='add',
                                 feature=feature,
                                 new_feature=feature,
                                 process='typing|date',
                                 meth='date_conversion',
                                 param=dict(fmt=fmt),
                                 data=DATA_PROCESSING['df'][feature].apply(lambda x: parser.parse(str(x)).strftime(fmt) if x == x else x),
                                 obj=dict(feature=fmt)
                                 )
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Date feature {} formatted'.format(feature))

    @staticmethod
    @FeatureOrchestra(meth='disparity', feature_types=['date'])
    def disparity(features: List[str] = None,
                  years: bool = True,
                  months: bool = True,
                  weeks: bool = True,
                  days: bool = True,
                  hours: bool = True,
                  minutes: bool = True,
                  seconds: bool = True,
                  digits: int = 6
                  ):
        """
        Calculate disparity time features

        :param features: List[str]
            Name of features

        :param years: bool
            Whether to generate yearly differences between date features or not

        :param months: bool
            Whether to generate monthly differences between date features or not

        :param weeks: bool
            Whether to generate weekly differences between date features or not

        :param days: bool
            Whether to generate daily differences between date features or not

        :param hours: bool
            Whether to generate hourly differences between date features or not

        :param minutes: bool
            Whether to generate minutely differences between date features or not

        :param seconds: bool
            Whether to generate secondly differences between date features or not

        :param digits: int
            Amount of digits to round
        """
        _first_date_feature: pd.DataFrame = pd.DataFrame()
        _second_date_feature: pd.DataFrame = pd.DataFrame()
        _pairs: List[tuple] = EasyExploreUtils().get_pairs(features=features)
        for _pair in _pairs:
            _load_temp_files(features=[_pair[0], _pair[1]])
            if _pair[0] in FEATURE_TYPES.get('date'):
                if str(DATA_PROCESSING['df'][_pair[0]].dtype).find('date') >= 0:
                    _first_date_feature = DATA_PROCESSING['df'][_pair[0]]
                else:
                    try:
                        _first_date_feature = pd.to_datetime(DATA_PROCESSING['df'][_pair[0]])
                    except ValueError:
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Feature "{}" could not be convert to datetime'.format(_pair[0]))
            if _pair[1] in FEATURE_TYPES.get('date'):
                if str(DATA_PROCESSING['df'][_pair[1]].dtype).find('date') >= 0:
                    _second_date_feature = DATA_PROCESSING['df'][_pair[1]]
                else:
                    try:
                        _second_date_feature = pd.to_datetime(DATA_PROCESSING['df'][_pair[1]])
                    except ValueError:
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Feature "{}" could not be convert to datetime'.format(_pair[1]))
            if (_first_date_feature.shape[0] > 0) and (_second_date_feature.shape[0] > 0):
                if years:
                    _process_handler(action='add',
                                     feature=_pair[0],
                                     new_feature='time_between_{}_{}_year'.format(_pair[0], _pair[1]),
                                     process='interaction|disparity|date',
                                     meth='disparity',
                                     param=dict(years=years, months=months, weeks=weeks, days=days, hours=hours, minute=minutes, second=seconds),
                                     data=np.round(a=((_first_date_feature - _second_date_feature).dt.days / 365).values, decimals=digits),
                                     force_type='continuous' if digits > 0 else 'ordinal',
                                     special_replacement=True,
                                     imp_value=0,
                                     obj=[_pair[1]]
                                     )
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Calculate time between "{}" and "{}" in years'.format(_pair[0], _pair[1]))
                if months:
                    _process_handler(action='add',
                                     feature=_pair[0],
                                     new_feature='time_between_{}_{}_month'.format(_pair[0], _pair[1]),
                                     process='interaction|disparity|date',
                                     meth='disparity',
                                     param=dict(years=years, months=months, weeks=weeks, days=days, hours=hours, minute=minutes, second=seconds),
                                     data=np.round(a=(((_first_date_feature - _second_date_feature).dt.days / 365) * 12).values, decimals=digits),
                                     force_type='continuous' if digits > 0 else 'ordinal',
                                     special_replacement=True,
                                     imp_value=0,
                                     obj=[_pair[1]]
                                     )
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Calculate time between "{}" and "{}" in months'.format(_pair[0], _pair[1]))
                if weeks:
                    _process_handler(action='add',
                                     feature=_pair[0],
                                     new_feature='time_between_{}_{}_week'.format(_pair[0], _pair[1]),
                                     process='interaction|disparity|date',
                                     meth='disparity',
                                     param=dict(years=years, months=months, weeks=weeks, days=days, hours=hours, minute=minutes, second=seconds),
                                     data=np.round(a=((_first_date_feature - _second_date_feature).dt.days / 7).values, decimals=digits),
                                     force_type='continuous' if digits > 0 else 'ordinal',
                                     special_replacement=True,
                                     imp_value=0,
                                     obj=[_pair[1]]
                                     )
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Calculate time between "{}" and "{}" in weeks'.format(_pair[0], _pair[1]))
                if days:
                    _process_handler(action='add',
                                     feature=_pair[0],
                                     new_feature='time_between_{}_{}_day'.format(_pair[0], _pair[1]),
                                     process='interaction|disparity|date',
                                     meth='disparity',
                                     param=dict(years=years, months=months, weeks=weeks, days=days, hours=hours, minute=minutes, second=seconds),
                                     data=np.round(a=(_first_date_feature - _second_date_feature).dt.days.values, decimals=digits),
                                     force_type='continuous' if digits > 0 else 'ordinal',
                                     special_replacement=True,
                                     imp_value=0,
                                     obj=[_pair[1]]
                                     )
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Calculate time between "{}" and "{}" in days'.format(_pair[0], _pair[1]))
                if hours:
                    _process_handler(action='add',
                                     feature=_pair[0],
                                     new_feature='time_between_{}_{}_hour'.format(_pair[0], _pair[1]),
                                     process='interaction|disparity|date',
                                     meth='disparity',
                                     param=dict(years=years, months=months, weeks=weeks, days=days, hours=hours, minute=minutes, second=seconds),
                                     data=np.round(a=((_first_date_feature - _second_date_feature).dt.days * 24).values, decimals=digits),
                                     force_type='continuous' if digits > 0 else 'ordinal',
                                     special_replacement=True,
                                     imp_value=0,
                                     obj=[_pair[1]]
                                     )
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Calculate time between "{}" and "{}" in hours'.format(_pair[0], _pair[1]))
                if minutes:
                    _process_handler(action='add',
                                     feature=_pair[0],
                                     new_feature='time_between_{}_{}_min'.format(_pair[0], _pair[1]),
                                     process='interaction|disparity|date',
                                     meth='disparity',
                                     param=dict(years=years, months=months, weeks=weeks, days=days, hours=hours, minute=minutes, second=seconds),
                                     data=np.round(a=(((_first_date_feature - _second_date_feature).dt.days * 24) * 60).values, decimals=digits),
                                     force_type='continuous' if digits > 0 else 'ordinal',
                                     special_replacement=True,
                                     imp_value=0,
                                     obj=[_pair[1]]
                                     )
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Calculate time between "{}" and "{}" in minutes'.format(_pair[0], _pair[1]))
                if seconds:
                    _process_handler(action='add',
                                     feature=_pair[0],
                                     new_feature='time_between_{}_{}_sec'.format(_pair[0], _pair[1]),
                                     process='interaction|disparity|date',
                                     meth='disparity',
                                     param=dict(years=years, months=months, weeks=weeks, days=days, hours=hours, minute=minutes, second=seconds),
                                     data=np.round(a=((((_first_date_feature - _second_date_feature).dt.days * 24) * 60) * 60).values, decimals=digits),
                                     force_type='continuous' if digits > 0 else 'ordinal',
                                     special_replacement=True,
                                     imp_value=0,
                                     obj=[_pair[1]]
                                     )
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Calculate time between "{}" and "{}" in seconds'.format(_pair[0], _pair[1]))
                _first_date_feature = pd.DataFrame()
                _second_date_feature = pd.DataFrame()

    @staticmethod
    @FeatureOrchestra(meth='disparity_time_series', feature_types=['date'])
    def disparity_time_series(features: List[str] = None,
                              perc: bool = True,
                              by_col: bool = True,
                              imp_const: float = 0.000001,
                              periods: int = 1
                              ):
        """
        Calculate disparity for each time unit within time series

        :param features: List[str]
            Names of features

        :param perc: bool
            Calculate relative or absolute differences

        :param by_col: bool
            Calculate differences by column or row

        :param imp_const: float
            Constant value to impute missing values before calculating disparity

        :param periods: int
            Number of periods to use for calculation
        """
        _axis: int = 1 if by_col else 0
        _periods: int = 1 if periods < 1 else periods
        if perc:
            DATA_PROCESSING['df'][features].fillna(imp_const).pct_change(axis=_axis).fillna(0)
        else:
            DATA_PROCESSING['df'][features].fillna(imp_const).diff(periods=_periods, axis=_axis).fillna(0)

    @staticmethod
    @FeatureOrchestra(meth='exp_transform', feature_types=['continuous'])
    def exp_transform(features: List[str] = None, skewness_test: bool = False):
        """
        Transform continuous features exponentially

        :param features: List[str]
            Name of the features

        :param skewness_test: bool
            Transform features that are skewed only
        """
        #if skewness_test:
        #    # TODO: subset features based on test results
        #    _features = StatsUtils(data=DATA_PROCESSING.get('df'), features=features).skewness_test()
        #else:
        #    _features = features
        for feature in features:
            _load_temp_files(features=[feature])
            _process_handler(action='add',
                             feature=feature,
                             new_feature='{}_exp'.format(feature) if DATA_PROCESSING.get('generate_new_feature') else feature,
                             process='scaler|exp',
                             meth='exp_transform',
                             param=dict(skewness_test=skewness_test),
                             data=np.exp(DATA_PROCESSING['df'][feature].values),
                             force_type='continuous',
                             special_replacement=True,
                             imp_value=sys.float_info.max
                             )
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformed feature "{}" using exponential transformation'.format(feature))

    @staticmethod
    def explore(features: List[str] = None,
                typing: bool = False,
                distribution: bool = False,
                health_check: bool = False,
                outlier_univariate: bool = False,
                outlier_multivariate: bool = False,
                correlation: bool = False,
                group_by_stats: bool = False,
                geo_stats: bool = False,
                visualize: bool = True,
                file_path: str = None,
                **kwargs
                ) -> dict:
        """
        Explore data

        :param features:
        :param typing:
        :param distribution:
        :param health_check:
        :param outlier_univariate:
        :param outlier_multivariate:
        :param correlation:
        :param group_by_stats:
        :param geo_stats:
        :param visualize:
        :param file_path:
        :return: dict
        """
        if features is None:
            _df: pd.DataFrame = DATA_PROCESSING.get('df')
        else:
            _features: List[str] = []
            for feature in features:
                if feature in DATA_PROCESSING['df'].columns:
                    _features.append(feature)
            _df: dd.DataFrame = DATA_PROCESSING['df'].loc[:, features] if len(_features) > 0 else DATA_PROCESSING.get('df')
        _results: dict = dict(typing={}, distribution={}, health_check={}, outlier=dict(uni={}, multi={}), cor={}, group={}, geo={})
        if typing:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Explore data typing ...')
            _results['typing'] = DataExplorer(df=_df,
                                              feature_types=FEATURE_TYPES,
                                              date_edges=DATA_PROCESSING.get('date_edges'),
                                              plot=visualize,
                                              file_path=file_path
                                              ).data_typing()
        if distribution:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Explore data distribution ...')
            _results['distribution'] = DataExplorer(df=_df,
                                                    feature_types=FEATURE_TYPES,
                                                    plot=visualize,
                                                    plot_type=None,
                                                    file_path=file_path
                                                    ).data_distribution(categorical=True if kwargs.get('categorical') is None else kwargs.get('categorical'),
                                                                        continuous=True if kwargs.get('continuous') is None else kwargs.get('continuous'),
                                                                        over_time=True if kwargs.get('over_time') is None else kwargs.get('over_time'),
                                                                        )
        if health_check:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Explore data health ...')
            _results['health_check'] = DataExplorer(df=_df,
                                                    feature_types=FEATURE_TYPES,
                                                    plot=visualize,
                                                    plot_type=None,
                                                    file_path=file_path,
                                                    ).data_health_check(sparsity=True if kwargs.get('sparsity') is None else kwargs.get('sparsity'),
                                                                        invariant=True if kwargs.get('invariant') is None else kwargs.get('invariant'),
                                                                        duplicate_cases=False if kwargs.get('duplicate_cases') is None else kwargs.get('duplicate_cases'),
                                                                        duplicate_features=True if kwargs.get('duplicate_features') is None else kwargs.get('duplicate_features'),
                                                                        nan_heat_map=True if kwargs.get('nan_heat_map') is None else kwargs.get('nan_heat_map'),
                                                                        nan_threshold=0.999 if kwargs.get('nan_threshold') is None else kwargs.get('nan_threshold'),
                                                                        other_mis=kwargs.get('other_mis')
                                                                        )
        if outlier_univariate:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Explore univariate outliers ...')
            _results['outlier']['uni'] = DataExplorer(df=_df,
                                                      feature_types=FEATURE_TYPES,
                                                      plot=visualize,
                                                      plot_type=None,
                                                      file_path=file_path
                                                      ).outlier_detector(kind='uni')
        if outlier_multivariate:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Explore multivariate outliers ...')
            _results['outlier']['multi'] = DataExplorer(df=_df,
                                                        feature_types=FEATURE_TYPES,
                                                        plot=visualize,
                                                        file_path=file_path
                                                        ).outlier_detector(kind='multi' if kwargs.get('kind') is None else kwargs.get('kind'),
                                                                           multi_meth=None if kwargs.get('multi_meth') is None else kwargs.get('multi_meth'),
                                                                           contour=False if kwargs.get('contour') is None else kwargs.get('contour'),
                                                                           )
        if correlation:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Explore correlation ...')
            _results['cor'] = DataExplorer(df=_df,
                                           feature_types=FEATURE_TYPES,
                                           plot=visualize,
                                           file_path=file_path
                                           ).cor(marginal=True if kwargs.get('marginal') is None else kwargs.get('marginal'),
                                                 partial=True if kwargs.get('partial') is None else kwargs.get('partial'),
                                                 marginal_meth='pearson' if kwargs.get('marginal_meth') is None else kwargs.get('marginal_meth'),
                                                 min_obs=1 if kwargs.get('min_obs') is None else kwargs.get('min_obs'),
                                                 decimals=2 if kwargs.get('decimals') is None else kwargs.get('decimals')
                                                 )
        if group_by_stats:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Explore grouped statistics ...')
            _results['group'] = DataExplorer(df=_df,
                                             feature_types=FEATURE_TYPES,
                                             plot=visualize,
                                             plot_type=None,
                                             file_path=file_path
                                             ).break_down()
        if geo_stats:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Explore geo statistics ...')
            _results['geo_stats'] = DataExplorer(df=_df,
                                                 feature_types=FEATURE_TYPES,
                                                 plot=visualize,
                                                 file_path=file_path
                                                 ).geo_stats(geo_features=[],
                                                             lat=kwargs.get('lat'),
                                                             lon=kwargs.get('lon'),
                                                             val=kwargs.get('val')
                                                             )
        return _results

    @staticmethod
    def geo_encoder(geo_features: List[str],
                    provider: str = 'arcgis',
                    max_rows: int = 1,
                    method: str = 'geocode',
                    stop: int = '15'
                    ):
        """
        Convert categorical geo data like postal code into continuous geo data (latitude, longitude)

        :param geo_features: List[str]
            Name of the features containing geographic related data

        :param provider: str
            Name of the provider
                -> arcgis: Arcgis
                -> google: Google
                -> bing: Bing
                -> baidu: Baidu
                -> freegeoip: Free Geo IP
                -> osm: Open Street map
                -> tomtom: TomTom
                -> yahoo: Yahoo

        :param max_rows: int
            Number of rows to fetch

        :param method: str
            Name of the used method
                -> geocode:

        :param stop: int
            Stopping threshold for requesting in minutes
        """
        _t0: datetime = datetime.now()
        _lat: dict = {}
        _lng: dict = {}
        for geo in geo_features:
            _load_temp_files(features=[geo])
            if geo not in DATA_PROCESSING['df'].columns:
                Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log(msg='Geo feature "{}" not found in data set'.format(geo))
            else:
                _unique: list = list(DATA_PROCESSING['df'][geo].unique())
                try:
                    for loc in _unique:
                        if loc != loc:
                            _lat.update({loc: np.nan})
                            _lng.update({loc: np.nan})
                            continue
                        if provider == 'arcgis':
                            _g: geocoder = geocoder.arcgis(location=str('{}'.format(loc)), maxRows=max_rows, method=method)
                        elif provider == 'google':
                            _g: geocoder = geocoder.google(location=str('{}'.format(loc)), maxRows=max_rows, method=method)
                        elif provider == 'bing':
                            _g: geocoder = geocoder.bing(location=str('{}'.format(loc)), maxRows=max_rows, method=method)
                        elif provider == 'baidu':
                            _g: geocoder = geocoder.baidu(location=str('{}'.format(loc)), maxRows=max_rows, method=method)
                        elif provider == 'freegeoip':
                            _g: geocoder = geocoder.freegeoip(location=str('{}'.format(loc)), maxRows=max_rows, method=method)
                        elif provider == 'osm':
                            _g: geocoder = geocoder.osm(location=str('{}'.format(loc)), maxRows=max_rows, method=method)
                        elif provider == 'tomtom':
                            _g: geocoder = geocoder.tomtom(location=str('{}'.format(loc)), maxRows=max_rows, method=method)
                        elif provider == 'yahoo':
                            _g: geocoder = geocoder.yahoo(location=str('{}'.format(loc)), maxRows=max_rows, method=method)
                        else:
                            Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log(msg='Provider "{}" for geocoding not supported'.format(provider))
                            continue
                        _geo_res: dict = _g.json
                        _lat.update({loc: _geo_res.get('lat')})
                        _lng.update({loc: _geo_res.get('lng')})
                except Exception as e:
                    Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log(msg='Error occured while fetching geo data from feature "{}"\n{}'.format(geo, e))
                DATA_PROCESSING['df']['lat_{}'.format(geo)] = DATA_PROCESSING['df'][geo].replace(_lat)
                DATA_PROCESSING['df']['lat_{}'.format(geo)] = DATA_PROCESSING['df']['lat_{}'.format(geo)].astype(float)
                DATA_PROCESSING['df']['lng_{}'.format(geo)] = DATA_PROCESSING['df'][geo].replace(_lng)
                DATA_PROCESSING['df']['lng_{}'.format(geo)] = DATA_PROCESSING['df']['lng_{}'.format(geo)].astype(float)
                DATA_PROCESSING['processing']['features'][geo].append('lat_{}'.format(geo))
                DATA_PROCESSING['processing']['features'][geo].append('lng_{}'.format(geo))
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Encode geo feature "{}" into latitude and longitude features'.format(geo))
                for code in DATA_PROCESSING['df'][geo].unique():
                    if code == code:
                        DATA_PROCESSING['mapper']['geo'].update({str(geo): {}})
                        _lat_per_code: list = DATA_PROCESSING['df'].loc[DATA_PROCESSING['df'][geo] == code, 'lat_{}'.format(geo)].unique().tolist()
                        _lon_per_code: list = DATA_PROCESSING['df'].loc[DATA_PROCESSING['df'][geo] == code, 'lon_{}'.format(geo)].unique().tolist()
                        DATA_PROCESSING['mapper']['geo'][str(geo)].update({code: dict(lat=_lat_per_code, lon=_lon_per_code)})

    @staticmethod
    def geo_break_down_stats(geo_features: List[str], aggregation: str = 'sum') -> pd.DataFrame:
        """
        Calculate statistics of continuous features based on break down of categorical geo features

        :param geo_features: List[str]
            Categorical geo features

        :param aggregation: str
            Name of the used aggregation method
                -> sum: Summary

        :return pd.DataFrame:
            Geo related break down statistics
        """
        _aggregation: dict = {}
        for ft in geo_features:
            if aggregation == 'sum':
                _aggregation.update({ft: np.sum})
            else:
                FeatureEngineerException('Aggregation function ({}) not supported'.format(aggregation))
        return DATA_PROCESSING['df'].groupby(by=[geo_features],
                                             axis=0,
                                             level=None,
                                             as_index=True,
                                             sort=True,
                                             group_keys=True,
                                             squeeze=False,
                                             observed=False
                                             ).aggregate(_aggregation)

    @staticmethod
    def get_action_space() -> dict:
        """
        Get processing action space

        :return dict:
            Processing action space
        """
        return DATA_PROCESSING.get('action_space')

    @staticmethod
    def get_actor_memory() -> dict:
        """
        Get memory (generated meta data) of actor

        :return: dict:
            Actor meta data
        """
        return DATA_PROCESSING.get('actor_memory')

    @staticmethod
    def get_cleaned_features() -> List[str]:
        """
        Get list of all names of the cleaned features

        :return: List[str]
            Name of the cleaned features
        """
        return DATA_PROCESSING.get('cleaned_features')

    @staticmethod
    def get_data(dask_df: bool = False):
        """
        Get active data set

        :param dask_df: bool
            Return dask DataFrame instead of Pandas

        :return Pandas or dask DataFrame
            Active data set
        """
        _features: List[str] = []
        for ft in FEATURE_TYPES.keys():
            for feature in FEATURE_TYPES.get(ft):
                _features.append(feature)
        if DATA_PROCESSING['target'] is not None:
            _features.append(DATA_PROCESSING.get('target'))
        _load_temp_files(features=_features)
        del DATA_PROCESSING['df'][list(TEMP_INDEXER.keys())[0]]
        if dask_df:
            return dd.from_pandas(data=DATA_PROCESSING.get('df'), npartitions=4)
        else:
            return DATA_PROCESSING.get('df')

    @staticmethod
    def get_data_info():
        """
        Get information about loaded data set
        """
        _features: List[str] = []
        for ft in FEATURE_TYPES.keys():
            for feature in FEATURE_TYPES.get(ft):
                _features.append(feature)
        if DATA_PROCESSING['target'] is not None:
            _features.append(DATA_PROCESSING.get('target'))
        _load_temp_files(features=_features)
        Log(write=not DATA_PROCESSING.get('show_msg')).log(
            msg='Data set information ...\nCases: {}\nFeatures: {}\n -Nominal: {}\n -Ordinal: {}\n -Date: {}\n -Continuous: {}\n -ID or Text: {}\nSources: {}\nMerged from: {}'.format(len(DATA_PROCESSING['df']),
                                                                                                                                                                                       len(DATA_PROCESSING['df'].columns),
                                                                                                                                                                                       FEATURE_TYPES.get('categorical'),
                                                                                                                                                                                       FEATURE_TYPES.get('ordinal'),
                                                                                                                                                                                       FEATURE_TYPES.get('date'),
                                                                                                                                                                                       FEATURE_TYPES.get('continuous'),
                                                                                                                                                                                       FEATURE_TYPES.get('id_text'),
                                                                                                                                                                                       DATA_PROCESSING.get('data_source'),
                                                                                                                                                                                       MERGES
                                                                                                                                                                                       )
        )
        DATA_PROCESSING['df'] = None

    @staticmethod
    def get_data_processing() -> dict:
        """
        Get (all) data processing information

        :return: dict
            Data processing information
        """
        return DATA_PROCESSING

    @staticmethod
    def get_data_source() -> List[str]:
        """
        Get data source path(s)

        :return: List[str]
            Complete source path(s)
        """
        return DATA_PROCESSING.get('data_source_path')

    @staticmethod
    def get_correlation(meth='pearson', threshold: float = 0.75) -> dict:
        """
        Get correlation of continuous features

        :param meth: str
            Method to use for generating marginal correlation scores

        :param threshold: float
            Threshold for classifying correlation score as high

        :return: dict
            Correlation results like correlation matrix, highly correlated features of each feature according to the given threshold
        """
        _features: List[str] = []
        for ft in FEATURE_TYPES.keys():
            for feature in FEATURE_TYPES.get(ft):
                _features.append(feature)
        if DATA_PROCESSING['target'] is not None:
            _features.append(DATA_PROCESSING.get('target'))
        _load_temp_files(features=_features)
        if set(list(DATA_PROCESSING['df'].columns)).difference(list(DATA_PROCESSING['correlation']['high'].keys())):
            if len(FEATURE_TYPES.get('continuous')) > 1:
                _df: dd.DataFrame = DATA_PROCESSING.get('df')[FEATURE_TYPES.get('continuous')]
                for feature in _df.columns:
                    if str(_df[feature].dtype).find('float') < 0:
                        _df[feature] = _df[feature].astype(float)
                DATA_PROCESSING['correlation']['matrix'] = _df.corr(method=meth, min_periods=None, split_every=False)
                #for score in DATA_PROCESSING['correlation']['matrix']:
                #    pass
        del _df
        return DATA_PROCESSING.get('correlation')

    @staticmethod
    def get_features(feature_type: str = None) -> List[str]:
        """
        Get names of features

        :param feature_type: str
            Name of the feature type to select
                -> categorical: Categorical (nominal) features
                -> ordinal: Categorical (ordinal) features
                -> continuous: Continuous (metric) features
                -> date: Date or time features
                -> id_text: ID or text features

        :return List[str]:
            Names of features
        """
        if feature_type is None:
            return ALL_FEATURES
        else:
            if feature_type in FEATURE_TYPES.keys():
                return FEATURE_TYPES.get(feature_type)

    @staticmethod
    def get_feature_types() -> Dict[str, List[str]]:
        """
        Get features types

        :return Dict[str, List[str]]:
            Names of features based on their typing
                -> text: Verbose text and id features
                -> date: Datetime features
                -> ordinal: Ordered categorical features
                -> categorical: Nominal (unordered) categorical features
                -> continuous: Continuous features
        """
        return FEATURE_TYPES

    @staticmethod
    def get_feature_values(feature: str, unique: bool = False) -> np.ndarray:
        """
        Get feature values

        :param feature: str
            Name of the feature to get values from

        :param unique: bool
            Whether to get unique or all feature values

        :return np.ndarray:
            Feature values
        """
        _features: List[str] = []
        for ft in FEATURE_TYPES.keys():
            for feature in FEATURE_TYPES.get(ft):
                _features.append(feature)
        if feature in _features:
            if unique:
                return DATA_PROCESSING['df'][feature].unique()
            else:
                return DATA_PROCESSING['df'][feature].values
        return np.ndarray(shape=[0, 0])

    @staticmethod
    def get_last_action() -> str:
        """
        Get last action of actor embedded in an reinforcement learning

        :return str:
            Name of the last action (feature engineering method)
        """
        return DATA_PROCESSING.get('last_action')

    @staticmethod
    def get_last_generated_feature() -> str:
        """
        Get name of the last generated feature

        :return str:
            Name of the last generated feature
        """
        return DATA_PROCESSING.get('last_generated_feature')

    @staticmethod
    def get_max_processing_level() -> int:
        """
        Get maximum level of feature processing

        :return: int
            Level of feature processing
        """
        return DATA_PROCESSING.get('max_level')

    @staticmethod
    def get_missing_data(freq: bool = True) -> dict:
        """
        Get missing data distribution

        :param freq: bool
            Return frequency of missing data case- and feature-wise or the index position of missing data in data set

        :return dict:
            Missing data distribution
                -> total: Valid and missing data in total
                -> case: Cases-wise missing data distribution
                -> feature: Feature-wise missing data distribution

            ... or ...

            Index position of missing data
                -> Case-wise
                -> Feature-wise
        """
        if freq:
            return DATA_PROCESSING.get('missing_data')
        else:
            return DATA_PROCESSING['mapper']['mis']

    @staticmethod
    def get_n_cases() -> int:
        """
        Get total number of cases in data set

        :return: int
            Total number of cases
        """
        return DATA_PROCESSING['n_cases']

    @staticmethod
    def get_n_features() -> int:
        """
        Get total number of features in data set

        :return: int
            Total number of features
        """
        return DATA_PROCESSING['n_features']

    @staticmethod
    def get_n_predictors() -> int:
        """
        Get number of predictors

        :return List[str]:
            Predictor names
        """
        return len(DATA_PROCESSING.get('predictors'))

    @staticmethod
    def get_n_target_values() -> int:
        """
        Get number of unique target feature values

        :return int:
            Number of unique target feature values
        """
        _load_temp_files(features=[DATA_PROCESSING.get('target')])
        return len(list(DATA_PROCESSING.get('df')[DATA_PROCESSING.get('target')].unique()))

    @staticmethod
    def get_notes(page: str = 'temp') -> str:
        """
        Get notes from specific page

        :param page: str
            Name of the page

        :return str:
            Note of the page
        """
        if page in NOTEPAD.keys():
            return NOTEPAD.get(page)
        else:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Page ({}) not found in notepad'.format(page))

    @staticmethod
    def get_pages() -> List[str]:
        """
        Get name of all pages stored in notepad

        :return List[str]:
            Name of the stored pages
        """
        return list(NOTEPAD.keys())

    @staticmethod
    def get_predictors() -> List[str]:
        """
        Get features defined as predictors

        :return List[str]:
            Predictor names
        """
        return DATA_PROCESSING.get('predictors')

    @staticmethod
    def get_obj_source() -> str:
        """
        Get complete file path of current FeatureEngineer object if it was saved

        :return str:
            Complete file path
        """
        return DATA_PROCESSING.get('source')

    @staticmethod
    def get_supported_types(data_type: str = None):
        """
        Get supported types

        :param data_type: str
            Name of the specific data type to show
        """
        if data_type in SUPPORTED_TYPES.keys():
            return SUPPORTED_TYPES.get(data_type)
        return SUPPORTED_TYPES

    @staticmethod
    def get_indices() -> list:
        """
        Get index values

        :return list:
            Index values
        """
        return list(DATA_PROCESSING['df'].index.values.compute())

    @staticmethod
    def get_processing() -> dict:
        """
        Get processing history

        :return dict:
             Processing history
                -> process: Tracked data processing steps in general
                -> features: Dependencies of features
        """
        return DATA_PROCESSING.get('processing')

    @staticmethod
    def get_processing_action_space() -> dict:
        """
        Get processing action space

        :return dict:
             Processing action space for each feature type
        """
        return PROCESSING_ACTION_SPACE

    @staticmethod
    def get_processing_relation(feature: str) -> dict:
        """
        Get processing action space

        :param feature: str
            Name of the feature to show processing relations for

        :return dict:
             Processing relations for given feature
        """
        _processing_relation: dict = dict(processing_level=0, parents={}, children=[])
        if feature in DATA_PROCESSING['df'].columns:
            if feature not in DATA_PROCESSING['processing']['features']['raw']:
                for level in range(len(DATA_PROCESSING['processing']['features'].keys()) - 1, -1, -1):
                    if level > 0:
                        if feature in DATA_PROCESSING['processing']['features']['level_{}'.format(level)].keys():
                            _processing_relation.update({'processing_level': level,
                                                         'children': DATA_PROCESSING['processing']['features']['level_{}'.format(level)][feature]
                                                         })
                            break
                _p: int = 0
                _feature: str = feature
                _parents: List[str] = []
                _main_relations: bool = True
                while True:
                    for parents in range(_processing_relation.get('processing_level'), -1, -1):
                        if parents == 0:
                            _level: str = 'raw'
                        else:
                            _level: str = 'level_{}'.format(parents)
                        for parent in DATA_PROCESSING['processing']['features'][_level].keys():
                            if feature in DATA_PROCESSING['processing']['features'][_level][parent]:
                                if _main_relations:
                                    _parents.append(parent)
                                if _level in _processing_relation['parents'].keys():
                                    if parent not in _processing_relation['parents'][_level]:
                                        _processing_relation['parents'][_level].append(parent)
                                else:
                                    _processing_relation['parents'].update({_level: [parent]})
                    if len(_parents) > 0 and _main_relations:
                        _main_relations = False
                        _feature = _parents[_p]
                        _p += 1
                    elif len(_parents) == 0 and _main_relations:
                        break
                    elif len(_parents) > 0 and not _main_relations:
                        if _feature == _parents[-1]:
                            break
                        else:
                            _feature = _parents[_p]
                            _p += 1
        return _processing_relation

    @staticmethod
    def get_target() -> str:
        """
        Get pre-defined target feature

        :return str:
            Name of the pre-defined target feature
        """
        return DATA_PROCESSING.get('target')

    @staticmethod
    def get_target_labels() -> list:
        """
        Get pre-define target feature labels

        :return list:
            Target feature value labels
        """
        return DATA_PROCESSING.get('target_labels')

    @staticmethod
    def get_target_values() -> list:
        """
        Get unique target feature values

        :return list:
            Unique target feature values
        """
        _load_temp_files(features=[DATA_PROCESSING.get('target')])
        return list(DATA_PROCESSING.get('df')[DATA_PROCESSING.get('target')].unique())

    @staticmethod
    def get_text_miner() -> TextMiner:
        """
        Get initialized and properly populated text miner object

        :return TextMiner:
            TextMiner object
        """
        return TEXT_MINER.get('obj')

    @staticmethod
    def get_training_data(output: str = 'df_pandas'):
        """
        Get training data set (containing predictors and target feature only)

        :param output: str
            Name of the output format
                -> df: Pandas DataFrame
                -> array: Numpy ndarray
                -> dict: Dictionary

        :return: Pandas DataFrame or numpy ndarray or dict:
            Training data set containing predictors and target feature only
        """
        _features: List[str] = DATA_PROCESSING['predictors'] + [DATA_PROCESSING['target']]
        _load_temp_files(features=_features)
        if len(_features) > 1:
            if output == 'df_dask':
                return dd.from_pandas(data=DATA_PROCESSING['df'][_features], npartitions=4)
            elif output == 'df_pandas':
                return DATA_PROCESSING['df'][_features]
            elif output == 'array':
                return DATA_PROCESSING['df'][_features].values
            elif output == 'dict':
                return DATA_PROCESSING['df'][_features].to_dict()
            else:
                raise FeatureEngineerException('Output format ({}) not supported'.format(output))

    @staticmethod
    def get_transformations(transformation: str = None) -> dict:
        """
        Get information about transformed features

        :param transformation: str
            Information about feature transformations:
                -> encoder: Label: Text label and integer label,
                            One-Hot: Feature value labels and dummy feature names
                -> scaler: MinMax, Robust, Normalize, Standardize: Fitted scaler object
                -> mapper: Value replacement for each feature
                -> naming: Renaming for each feature
                -> binning: Integer labels and binning edges

        :return dict:
            Information about feature transformations
        """
        _transformation: dict = dict(encoder=DATA_PROCESSING.get('encoder'),
                                     scaler=DATA_PROCESSING.get('scaler'),
                                     mapper=DATA_PROCESSING['mapper'].get('obs'),
                                     naming=DATA_PROCESSING['mapper'].get('names'),
                                     binning_continuous=DATA_PROCESSING['categorizer'].get('continuous'),
                                     binning_date=DATA_PROCESSING['categorizer'].get('date')
                                     )
        if transformation is None:
            return _transformation
        if transformation == 'encoder':
            return _transformation.get('encoder')
        if transformation == 'scaler':
            return _transformation.get('scaler')
        if transformation == 'mapper':
            return _transformation.get('mapper')
        if transformation == 'naming':
            return _transformation.get('naming')
        if transformation == 'binning_date':
            return _transformation.get('binning_date')
        if transformation == 'binning_continuous':
            return _transformation.get('binning_continuous')
        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformation ({}) not found'.format(transformation))

    @staticmethod
    @FeatureOrchestra(meth='interaction', feature_types=['continuous', 'ordinal'])
    def interaction(features: List[str] = None,
                    addition: bool = True,
                    subtraction: bool = True,
                    multiplication: bool = True,
                    division: bool = True
                    ):
        """
        Calculate simple interactions of numerical and ordinal features

        :param features: List[str]
            Name of the features

        :param addition: bool
            Whether to add features or not

        :param subtraction: bool
            Whether to subtract features or not

        :param multiplication: bool
            Whether to multiply features or not

        :param division: bool
            Whether to divide features or not
        """
        _load_temp_files(features=features)
        for feature in features:
            _features: List[str] = features
            del _features[_features.index(feature)]
            _data_set: pd.DataFrame = DATA_PROCESSING['df'][_features].fillna(sys.float_info.max)
            # Addition:
            if addition:
                _add: pd.DataFrame = _data_set.add(DATA_PROCESSING['df'][feature], axis='index').replace(INVALID_VALUES, np.nan).fillna(sys.float_info.max)
                _add = _add.rename(columns={second: '{}__add__{}'.format(feature, second) for second in _add.columns})
                _relations: dict = {_add.columns[i]: [feature, _features[i]] for i in range(0, len(_add.columns), 1)}
                _process_handler(action='add',
                                 feature=feature,
                                 new_feature='',
                                 process='interaction|simple',
                                 meth='interaction',
                                 param=dict(addition=addition, subtraction=subtraction, multiplication=multiplication, division=division),
                                 data=_add,
                                 force_type='continuous',
                                 imp_value=sys.float_info.max,
                                 obj=dict(features=_relations)
                                 )
                Log(write=not DATA_PROCESSING.get('show_msg')).log(
                    msg='Generate feature by adding feature "{}" with all other compatible features ({})'.format(
                        feature, len(_features)))
            # Subtraction:
            if subtraction:
                _sub: pd.DataFrame = _data_set.sub(DATA_PROCESSING['df'][feature], axis='index').replace(INVALID_VALUES,np.nan).fillna(sys.float_info.min)
                _sub = _sub.rename(columns={second: '{}__sub__{}'.format(feature, second) for second in _sub.columns},)
                _relations: dict = {_sub.columns[i]: [feature, _features[i]] for i in range(0, len(_sub.columns), 1)}
                _process_handler(action='add',
                                 feature=feature,
                                 new_feature='',
                                 process='interaction|simple',
                                 meth='interaction',
                                 param=dict(addition=addition, subtraction=subtraction, multiplication=multiplication, division=division),
                                 data=_sub,
                                 force_type='continuous',
                                 imp_value=sys.float_info.max,
                                 obj=dict(features=_relations)
                                 )
                Log(write=not DATA_PROCESSING.get('show_msg')).log(
                    msg='Generate feature by subtracting feature "{}" with all other compatible features ({})'.format(
                        feature, len(_features)))
            # Multiplication:
            if multiplication:
                _multi: pd.DataFrame = _data_set.multiply(DATA_PROCESSING['df'][feature], axis='index').replace(INVALID_VALUES, np.nan).fillna(sys.float_info.max)
                _multi = _multi.rename(columns={second: '{}__multi__{}'.format(feature, second) for second in _multi.columns})
                _relations: dict = {_multi.columns[i]: [feature, _features[i]] for i in range(0, len(_multi.columns), 1)}
                _process_handler(action='add',
                                 feature=feature,
                                 new_feature='',
                                 process='interaction|simple',
                                 meth='interaction',
                                 param=dict(addition=addition, subtraction=subtraction, multiplication=multiplication, division=division),
                                 data=_multi,
                                 force_type='continuous',
                                 imp_value=sys.float_info.max,
                                 obj=dict(features=_relations)
                                 )
                Log(write=not DATA_PROCESSING.get('show_msg')).log(
                    msg='Generate feature by multiplying feature "{}" with all other compatible features ({})'.format(
                        feature, len(_features)))
            # Division:
            if division:
                _div: pd.DataFrame = _data_set.div(DATA_PROCESSING['df'][feature], axis='index').replace(INVALID_VALUES, np.nan).fillna(0.0)
                _div = _div.rename(columns={second: '{}__div__{}'.format(feature, second) for second in _div.columns})
                _relations: dict = {_div.columns[i]: [feature, _features[i]] for i in range(0, len(_div.columns), 1)}
                _process_handler(action='add',
                                 feature=feature,
                                 new_feature='',
                                 process='interaction|simple',
                                 meth='interaction',
                                 param=dict(addition=addition, subtraction=subtraction, multiplication=multiplication, division=division),
                                 data=_div,
                                 force_type='continuous',
                                 imp_value=sys.float_info.min,
                                 obj=dict(features=_relations)
                                 )
                Log(write=not DATA_PROCESSING.get('show_msg')).log(
                    msg='Generate feature by dividing feature "{}" with all other compatible features ({})'.format(
                        feature, len(_features)))

    @staticmethod
    @FeatureOrchestra(meth='interaction_poly', feature_types=['continuous'])
    def interaction_poly(features: List[str] = None,
                         degree: int = 2,
                         interaction_only: bool = False,
                         include_bias: bool = False
                         ):
        """
        Calculate interactions between continuous features

        :param features: List[str]
            Names of the features

        :param degree: int
            Number of degrees

        :param interaction_only: bool
            Whether to calculate interactions only or not

        :param include_bias: bool
            Whether to include bias or not
        """
        _load_temp_files(features=features)
        _degree: int = degree if degree > 1 else 2
        if len(list(set(features))) > 1:
            _poly = PolynomialFeatures(degree=degree,
                                       interaction_only=interaction_only,
                                       include_bias=include_bias
                                       )
        else:
            raise FeatureEngineerException('Not enough features ({})'.format(features))
        _data: pd.DataFrame = DATA_PROCESSING['df'][features].fillna(sys.float_info.max).values
        _poly.fit(X=_data)
        _polynomial_features = _poly.transform(X=_data)
        _name_mapper: dict = dict(original={}, interaction={})
        _new_feature_names: Dict[str, str] = {}
        for i, name in enumerate(_poly.get_feature_names()):
            if i < len(features):
                _name_mapper['original'].update({name: features[i]})
            else:
                if name.find(' ') >= 0:
                    _feature_name: str = copy.deepcopy(name)
                    _feature_match: int = 0
                    _interaction: List[str] = []
                    for n in _name_mapper['original'].keys():
                        if name.find(n) >= 0:
                            _feature_match += 1
                            _feature_name = _feature_name.replace(n, _name_mapper['original'].get(n))
                            _interaction.append(_name_mapper['original'].get(n))
                        if _feature_match == _degree:
                            _feature_name = _feature_name.replace(' ', '__')
                            _name_mapper['interaction'].update({_feature_name: _interaction})
                            _interaction = []
                            break
                    _new_feature_names.update({name: _feature_name})
                else:
                    for n in _name_mapper['original'].keys():
                        if name.find(n) >= 0:
                            _name_mapper['interaction'].update({name: [_name_mapper['original'].get(n)]})
                            _new_feature_names.update({name: name.replace(n, _name_mapper['original'].get(n))})
                            break
        _df: pd.DataFrame = pd.DataFrame(data=_polynomial_features, columns=_poly.get_feature_names())
        _df = _df.drop(columns=list(_name_mapper['original'].keys()), axis=1)
        _df = _df.rename(columns=_new_feature_names)
        _process_handler(action='add',
                         feature='',
                         new_feature='',
                         process='interaction|polynomial',
                         meth='interaction_poly',
                         param=dict(degree=degree, interaction_only=interaction_only, include_bias=include_bias),
                         data=_df,
                         special_replacement=False,
                         imp_value=sys.float_info.max,
                         obj=dict(polynomial=_polynomial_features,
                                  features=_name_mapper['interaction']
                                  )
                         )
        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Generated features based on interactions (degree={}) between continuous features:\n{}'.format(_degree, _new_feature_names.values()))

    @staticmethod
    @FeatureOrchestra(meth='impute', feature_types=['ordinal', 'continuous'])
    def impute(features: List[str] = None,
               impute_int_value: int = None,
               impute_float_value: float = None,
               other_mis_values: list = None,
               multiple: bool = True,
               single_meth: str = 'constant',
               multiple_meth: str = 'random',
               m: int = 100,
               convergence_threshold: float = 0.99
               ):
        """
        Impute missing data

        :param features: List[str]
            Names of the features

        :param impute_int_value: int
            Single integer value to impute missing data of categorical

        :param impute_float_value: float
            Single float value to impute missing data of continuous

        :param other_mis_values: list
            Values classified as missing data

        :param multiple: bool
            Whether to use multiple imputation rather than single imputation or not

        :param single_meth: str
            Single imputation method to use
                -> constant: Single imputation by a constant value
                -> mean: Single imputation by mean
                -> median: Single imputation by median
                -> min: Single imputation by minimum
                -> max: Single imputation by maximum

        :param multiple_meth: str
            Multiple imputation method to use
                -> random: Random sampling within value range
                -> supervised: Supervised machine learning algorithm

        :param m: int
            Maximum number of samples

        :param convergence_threshold: float
            Convergence threshold for early stopping
        """
        if multiple:
            if multiple_meth == 'random':
                for feature in features:
                    _load_temp_files(features=[feature])
                    if MissingDataAnalysis(df=DATA_PROCESSING['df'], features=[feature]).has_nan():
                        #if DATA_PROCESSING['mapper']['mis']['features'].get(feature) is None:
                        #    continue
                        #if len(DATA_PROCESSING['mapper']['mis']['features'][feature]) == 0:
                        #    continue
                        if feature in FEATURE_TYPES['ordinal'] + FEATURE_TYPES['continuous']:
                            _threshold: float = convergence_threshold if (convergence_threshold > 0) and (convergence_threshold < 1) else 0.99
                            _unique_values: np.array = DATA_PROCESSING['df'].loc[~DATA_PROCESSING['df'][feature].isnull(), feature].unique()
                            if str(_unique_values.dtype).find('int') < 0 and str(_unique_values.dtype).find('float') < 0:
                                _unique_values = _unique_values.astype(dtype=float)
                            _value_range: Tuple[float, float] = (min(_unique_values), max(_unique_values))
                            _std: float = DATA_PROCESSING['df'][feature].std()
                            _threshold_range: Tuple[float, float] = (_std - (_std * (1 - _threshold)), _std + (_std * (1 - _threshold)))
                            _m: List[List[float]] = []
                            _std_theta: list = []
                            for n in range(0, m, 1):
                                _data: np.array = DATA_PROCESSING['df'][feature].values
                                #if DATA_PROCESSING['mapper']['mis']['features'].get(feature) is None:
                                #    break
                                #else:
                                _imp_value: List[float] = []
                                for idx in DATA_PROCESSING['mapper']['mis']['features'][feature]:
                                    if feature in FEATURE_TYPES.get('ordinal'):
                                        _imp_value.append(round(np.random.uniform(low=_value_range[0], high=_value_range[1])))
                                    else:
                                        _imp_value.append(np.random.uniform(low=_value_range[0], high=_value_range[1]))
                                    _data[idx] = _imp_value[-1]
                                _std_theta.append(copy.deepcopy(abs(_std - np.std(_data))))
                                if (_std_theta[-1] >= _threshold_range[0]) and (_std_theta[-1] <= _threshold_range[1]):
                                    # TODO: create temp df for preventing overwriting original data in case of IndexError
                                    #DATA_PROCESSING['df'][feature] = dd.from_array(x=_data)
                                    break
                                _m.append(_imp_value)
                            # TODO: Prevent IndexError -> std_theta list and imputation set
                            _best_imputation: list = _m[_std_theta.index(min(_std_theta))]
                            _imp_data: pd.DataFrame = pd.DataFrame()
                            _imp_data[feature] = DATA_PROCESSING['df'][feature].values
                            for i, idx in enumerate(DATA_PROCESSING['mapper']['mis']['features'][feature]):
                                _imp_data.loc[idx, feature] = _best_imputation[i]
                                #DATA_PROCESSING['df'].loc[idx, feature] = _best_imputation[i]
                            #DATA_PROCESSING['df'][feature] = dd.from_array(x=_imp_data[feature].values)
                            DATA_PROCESSING['df'][feature] = _imp_data[feature].values
                            _std_diff: float = 1 - round(_std / DATA_PROCESSING['df'][feature].std())
                            Log(write=not DATA_PROCESSING.get('show_msg')).log(
                                msg='Variance of feature ({}) {}creases by {}%'.format(feature,
                                                                                       'in' if _std_diff > 0 else 'de',
                                                                                       _std_diff
                                                                                       )
                                )
                        _save_temp_files(feature=feature)
                    else:
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='No missing values in feature ({}) found'.format(feature))
            elif multiple_meth == 'supervised':
                DATA_PROCESSING['df'] = MultipleImputation(df=DATA_PROCESSING.get('df')).mice()
            else:
                Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log(msg='Method for multiple imputation ({}) not supported'.format(multiple_meth))
        else:
            for feature in features:
                _load_temp_files(features=[feature])
                if MissingDataAnalysis(df=DATA_PROCESSING.get('df'), features=[feature]).has_nan():
                    _std: float = DATA_PROCESSING['df'][feature].std()
                    if single_meth == 'constant':
                        if impute_int_value is None:
                            if impute_float_value is None:
                                _imp_value = None
                                Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log(msg='No value for imputation found')
                            else:
                                if feature not in FEATURE_TYPES.get('continuous'):
                                    continue
                                _imp_value: float = impute_float_value
                        else:
                            if feature in FEATURE_TYPES.get('ordinal'):
                                _imp_value: int = impute_int_value
                            else:
                                if feature in FEATURE_TYPES.get('continuous'):
                                    if impute_float_value is None:
                                        continue
                                    _imp_value: float = impute_float_value
                                else:
                                    continue
                    elif single_meth == 'mean':
                        _imp_value: float = DATA_PROCESSING['df'][feature].mean(skipna=True)
                    elif single_meth == 'median':
                        _imp_value: float = DATA_PROCESSING['df'][feature].median(skipna=True)
                    elif single_meth == 'min':
                        _imp_value: float = DATA_PROCESSING['df'][feature].min(skipna=True)
                    elif single_meth == 'mean':
                        _imp_value: float = DATA_PROCESSING['df'][feature].max(skipna=True)
                    else:
                        _imp_value = None
                        Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log(msg='Method for single imputation not supported')
                    if _imp_value is not None:
                        if other_mis_values is not None:
                            if len(other_mis_values) > 0:
                                for mis in other_mis_values:
                                    DATA_PROCESSING['df'][feature] = DATA_PROCESSING['df'][feature].replace(mis, np.nan)
                        if MissingDataAnalysis(df=DATA_PROCESSING.get('df'), features=[feature]).has_nan():
                            DATA_PROCESSING['df'][feature] = DATA_PROCESSING['df'][feature].fillna(value=_imp_value, method=None)
                            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Imputed feature "{}" using {}: {}'.format(feature, single_meth, str(_imp_value)))
                    try:
                        _std_diff: float = 1 - round(_std / DATA_PROCESSING['df'][feature].std())
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Variance of feature ({}) {}creases by {}%'.format(feature, 'in' if _std_diff > 0 else 'de', _std_diff))
                    except ValueError:
                        pass
                    _save_temp_files(feature=feature)
                else:
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='No missing values in feature ({}) found'.format(feature))

    @staticmethod
    def is_unstable(feature: str):
        """
        Check stability of continuous features

        :param feature: str
            Name of the continuous feature
        """
        if feature in DATA_PROCESSING['df'].columns:
            if feature in FEATURE_TYPES.get('continuous'):
                _mean = np.mean(DATA_PROCESSING['df'][feature].values)
                if _mean in INVALID_VALUES:
                    return True
            else:
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Feature ({}) is not continuous'.format(feature))
        else:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Feature ({}) not found in data set'.format(feature))
        return False

    @staticmethod
    @FeatureOrchestra(meth='label_encoder', feature_types=['categorical'])
    def label_encoder(encode: bool, features: List[str] = None):
        """
        Encode labels (written categories) into integer values

        :param encode: bool
            Encode labels into integers or decode integers into labels

        :param features: List[str]
            Features for label encoding / decoding
                -> None: All categorical features are used
        """
        for feature in features:
            _load_temp_files(features=[feature])
            _unique_values: np.array = DATA_PROCESSING['df'][feature].unique()
            if encode:
                _has_labels: bool = False
                for val in _unique_values:
                    if len(re.findall('[a-z A-Z]', str(val))) > 0:
                        _has_labels = True
                        break
                if _has_labels:
                    _values = {label: i for i, label in enumerate(_unique_values)}
                    _data: pd.DataFrame = DATA_PROCESSING['df'][feature].replace(_values)
                    _process_handler(action='add',
                                     feature=feature,
                                     new_feature=feature,
                                     process='encoder|label',
                                     meth='label_encoder',
                                     param=dict(encode=encode),
                                     data=_data,
                                     force_type='categorical',
                                     obj=dict(label=_unique_values, val=_values)
                                     )
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformed feature "{}" using label encoding (label to number)'.format(feature))
            else:
                _data: pd.DataFrame = DATA_PROCESSING['df'][feature].replace({val: label for label, val in DATA_PROCESSING['encoder']['label'][feature].values})
                _process_handler(action='add',
                                 feature=feature,
                                 new_feature=feature,
                                 process='encoder|label',
                                 meth='label_encoder',
                                 param=dict(encode=encode),
                                 data=_data,
                                 force_type='categorical'
                                 )
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformed feature "{}" using label decoding (number to original label)'.format(feature))

    @staticmethod
    @FeatureOrchestra(meth='linguistic_features', feature_types=['id_text'])
    def linguistic_features(features: List[str] = None, sep: str = '|'):
        """
        Generate numeric linguistic features by processing features containing natural language

        :param features: List[str]
            Name of the features

        :param sep: str
            Separator value
        """
        if features is None:
            _features: List[str] = FEATURE_TYPES.get('id_text')
        else:
            _features: List[str] = []
            for feature in features:
                if feature in DATA_PROCESSING['df'].columns:
                    _features.append(feature)
        if len(_features) > 0:
            for feature in _features:
                _load_temp_files(features=[feature])
                if str(DATA_PROCESSING['df'][feature].dtype).find('object') >= 0:
                    TEXT_MINER['obj'].generate_linguistic_features(features=[feature])
                    for lf in TEXT_MINER['obj'].generated_features[feature]['linguistic']:
                        _process_handler(action='add',
                                         feature=feature,
                                         new_feature=lf,
                                         process='text|linguistic',
                                         meth='linguistic_features',
                                         param=dict(sep=sep),
                                         data=TEXT_MINER['obj'].get_numeric_features(features=[lf], compute=True),
                                         obj=None
                                         )
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Generated numeric linguistic feature "{}" based on text feature "{}"'.format(lf, feature))

    def load(self, file_path: str = None, cloud: str = None, **kwargs):
        """
        Load data engineering information (FeatureEngineer object)

        :param file_path: str
            Complete file path of the external stored engineering information

        :param cloud: str
            Name of the cloud provider
                -> google: Google Cloud Storage
                -> aws: AWS Cloud

        :param kwargs: dict
            Key-word arguments
        """
        if cloud is None:
            _bucket_name: str = None
        else:
            if cloud not in CLOUD_PROVIDER:
                raise FeatureEngineerException('Cloud provider ({}) not supported'.format(cloud))
            _bucket_name: str = file_path.split("//")[1].split("/")[0]
        global DATA_PROCESSING
        global FEATURE_TYPES
        global SPECIAL_JOBS
        global PREDICTORS
        global MERGES
        global NOTEPAD
        global PROCESSING_ACTION_SPACE
        global TEXT_MINER
        if file_path is not None:
            if len(file_path) > 0:
                self.data_processing = DataImporter(file_path=file_path,
                                                    as_data_frame=False,
                                                    create_dir=False,
                                                    cloud=cloud,
                                                    bucket_name=_bucket_name
                                                    ).file()
                self.kwargs = self.data_processing.data_processing.get('kwargs')
        if self.data_processing is None:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='No file path found')
        else:
            DATA_PROCESSING = self.data_processing.data_processing.get('processing')
            FEATURE_TYPES = self.data_processing.data_processing.get('feature_types')
            SPECIAL_JOBS = self.data_processing.data_processing.get('special_jobs')
            PREDICTORS = self.data_processing.data_processing.get('predictors')
            MERGES = self.data_processing.data_processing.get('merges')
            NOTEPAD = self.data_processing.data_processing.get('notepad')
            PROCESSING_ACTION_SPACE = self.data_processing.data_processing.get('processing_action_space')
            TEXT_MINER = self.data_processing.data_processing.get('text_miner')
            self.data_processing = None
            DATA_PROCESSING['df'] = DataImporter(file_path=file_path.split('.')[0],
                                                 #file_path='{}.parquet'.format(file_path.split('.')[0]),
                                                 as_data_frame=True,
                                                 use_dask=True,
                                                 create_dir=False
                                                 ).file()
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Load feature engineer')

    @staticmethod
    @FeatureOrchestra(meth='log_transform', feature_types=['continuous'])
    def log_transform(features: List[str] = None, skewness_test: bool = False):
        """
        Transform features by calculating the natural logarithm

        :param features: List[str]
            Name of the features

        :param skewness_test: bool
            Whether to test the skewness statistically or not
        """
        #if skewness_test:
        #    # TODO: subset features based on test results
        #    _features = StatsUtils(data=DATA_PROCESSING.get('df'), features=features).skewness_test()
        #else:
        #    _features = features
        for feature in features:
            _load_temp_files(features=[feature])
            _process_handler(action='add',
                             feature=feature,
                             new_feature='{}_log'.format(feature) if DATA_PROCESSING.get('generate_new_feature') else feature,
                             process='scaler|log',
                             meth='log_transform',
                             param=dict(skewness_test=skewness_test),
                             data=np.log(DATA_PROCESSING['df'][feature].values),
                             force_type='continuous',
                             special_replacement=True,
                             imp_value=sys.float_info.min
                             )
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformed feature "{}" using logarithmic transformation'.format(feature))

    @staticmethod
    @FeatureOrchestra(meth='normalizer', feature_types=['continuous'])
    def normalizer(features: List[str] = None, norm_meth: str = 'l2'):
        """
        Normalize features

        :param features: List[str]
            Name of features

        :param norm_meth: str
            Name of the used regulizer
                -> l1
                -> l2
        """
        for feature in features:
            _load_temp_files(features=[feature])
            _data: np.arry = DATA_PROCESSING['df'][feature].fillna(sys.float_info.min).values
            _normalizer = Normalizer(norm=norm_meth)
            _normalizer.fit(X=np.reshape(_data, (-1, 1)))
            _process_handler(action='add',
                             feature=feature,
                             new_feature='{}_normal'.format(feature) if DATA_PROCESSING.get('generate_new_feature') else feature,
                             process='scaler|normal',
                             meth='normalizer',
                             param=dict(norm_meth=norm_meth),
                             data=np.reshape(_normalizer.transform(X=np.reshape(_data, (-1, 1))), (1, -1))[0],
                             force_type='continuous',
                             special_replacement=True,
                             imp_value=sys.float_info.min,
                             obj=_normalizer
                             )
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformed feature "{}" using normalization'.format(feature))

    @staticmethod
    def melt(other_df=None,
             files: List[str] = None,
             merging: bool = True,
             merge_by: str = 'id',
             id_var: str = None,
             join_type: str = 'inner',
             concat_by: str = 'col',
             **kwargs
             ):
        """
        Combine current data set with other (external) data set by merging or concatenation

        :param other_df: Pandas DataFrame or dask DataFrame
            Second or other data set to merge

        :param files: List[str]
            Complete file path of external data sets

        :param merging: bool
            Merge data frames (by id or index) or to concatenate (by row or col) them

        :param merge_by: str
            Defining merging strategy
                -> id: Merge data frames by given id feature
                -> index: Merge data frames by index value of both

        :param id_var: str
            Name of the id feature to merge by

        :param join_type: str
            Defining merging type
                -> left: use only keys from left data frame and drop mismatching keys of right (new) data frame (preserve key order)
                -> right: use only keys from right data frame and drop mismatching keys of left (old) data frame (preserve key order)
                -> outer: use union of keys from both data frames and drop intersection keys of both data frames (sort keys lexicographically)
                -> inner: use intersection of keys from both data frames and drop union keys of both data frames (preserve order of the left keys)

        :param concat_by: str
            Defining concatenation type:
                -> row: Concatenate both data frames row-wise (the number of rows increases)
                -> col: Concatenate both data frames column-wise (the number of columns increases)

        :param kwargs: dict
            Key-word arguments for class DataImporter
        """
        if files is None:
            _n_data_sets: int = 0
        else:
            _n_data_sets: int = len(files)
        if other_df is None:
            _other_df: bool = False
        else:
            _n_data_sets += 1
            _other_df: bool = True
        global MERGES
        MERGES.update({'original': DATA_PROCESSING['df'].columns})
        for d in range(0, _n_data_sets, 1):
            if _other_df:
                _d: int = 0
            else:
                _d: int = 1
            if _d == 0:
                _df: dd.DataFrame = other_df
            else:
                _df: dd.DataFrame = dd.from_pandas(data=pd.DataFrame(), npartitions=DATA_PROCESSING['cpu_cores'])
                try:
                    _df = DataImporter(file_path=files[_d - 1],
                                       as_data_frame=True,
                                       create_dir=False,
                                       sep=',' if kwargs.get('sep') is None else kwargs.get('sep'),
                                       **kwargs
                                       ).file()
                    DATA_PROCESSING['data_source_path'].append(files[_d - 1])
                except FileUtilsException as e:
                    Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log(msg='Error while importing file "{}"\n{}'.format(files[_d - 1], e))
                    continue
            if len(_df) == 0:
                Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log(msg='External data frame is empty')
            else:
                MERGES.update({'current': list(DATA_PROCESSING['df'].columns),
                               'external_{}'.format(d): list(_df.columns)
                               })
                _current: List[str] = copy.deepcopy(MERGES.get('current'))
                _current.sort(reverse=False)
                _external: List[str] = copy.deepcopy(MERGES.get(list(MERGES.keys())[-1]))
                _external.sort(reverse=False)
                _new_features: List[str] = list(set(_external).difference(_current))
                _equal_features: List[str] = list(set(_external).intersection(_current))
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Internal Data Set:\nCases: {}\nFeatures: {}\nExternal Data Set:\n Cases: {}\nFeatures: {}'.format(
                        len(DATA_PROCESSING['df']),
                        len(DATA_PROCESSING['df'].columns),
                        len(_df),
                        len(_df.columns)
                        )
                    )
                if merging:
                    if merge_by == 'id':
                        if id_var is None or id_var not in DATA_PROCESSING.get('df').columns:
                            Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log(msg='ID variable ({}) not found'.format(id_var))
                        else:
                            DATA_PROCESSING['df'] = pd.merge(left=DATA_PROCESSING.get('df'),
                                                             right=_df,
                                                             on=id_var,
                                                             how=join_type,
                                                             suffixes=tuple(['', '_{}'.format(list(MERGES.keys())[-1])])
                                                             )
                            if len(_new_features) > 0:
                                for new_feature in _new_features:
                                    _update_feature_types(feature=new_feature)
                            if len(_equal_features) > 0:
                                for equal_feature in _equal_features:
                                    _update_feature_types(feature='{}_{}'.format(equal_feature, list(MERGES.keys())[-1]))
                            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='New Data Set after merging by {}\nCases: {}\nFeatures: {}'.format(id_var,
                                                                                                                                                      len(DATA_PROCESSING['df']),
                                                                                                                                                      len(DATA_PROCESSING['df'].columns),
                                                                                                                                                      )
                                                                               )
                    elif merge_by == 'index':
                        DATA_PROCESSING['df'] = pd.merge(left=DATA_PROCESSING.get('df'),
                                                         right=_df,
                                                         left_index=True,
                                                         right_index=True,
                                                         suffixes=tuple(['', '_{}'.format(list(MERGES.keys())[-1])])
                                                         )
                        if len(_new_features) > 0:
                            for new_feature in _new_features:
                                _update_feature_types(feature=new_feature)
                        if len(_equal_features) > 0:
                            for equal_feature in _equal_features:
                                _update_feature_types(feature='{}_{}'.format(equal_feature, list(MERGES.keys())[-1]))
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='New Data Set after merging by index\nCases: {}\nFeatures: {}'.format(len(DATA_PROCESSING['df']),
                                                                                                                                                     len(DATA_PROCESSING['df'].columns)
                                                                                                                                                     )
                                                                           )
                    else:
                        Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log(msg='Merge by method ({}) not found'.format(merge_by))
                else:
                    if concat_by == 'row':
                        DATA_PROCESSING['df'] = pd.concat([DATA_PROCESSING.get('df'), _df], axis=0)
                    elif concat_by == 'col':
                        _rename: dict = {}
                        if len(_equal_features) > 0:
                            for equal_feature in _equal_features:
                                _rename.update({equal_feature: '{}_{}'.format(_rename.get(equal_feature), list(MERGES.keys())[-1])})
                        _df.rename(_rename)
                        DATA_PROCESSING['df'] = pd.concat([DATA_PROCESSING.get('df'), _df], axis=1)
                        if len(_new_features) > 0:
                            for new_feature in _new_features:
                                _update_feature_types(feature=new_feature)
                        for renamed in _rename.keys():
                            _update_feature_types(feature=_rename.get(renamed))
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='New Data Set after concatenating by {}\nCases: {}\nFeatures: {}'.format(concat_by,
                                                                                                                                                    len(DATA_PROCESSING['df']),
                                                                                                                                                    len(DATA_PROCESSING['df'].columns)
                                                                                                                                                    )
                                                                       )

    @staticmethod
    def merge_engineer(feature_engineer_file_path: str,
                       merging: bool = False,
                       merge_by: str = 'id',
                       id_var: str = None,
                       join_type: str = 'inner',
                       concat_by: str = 'col',
                       cloud: str = None
                       ):
        """
        Merge two complete FeatureEngineer class objects

        :param feature_engineer_file_path: str
            Complete file path of the external feature engineer object

        :param merging: bool
            Merge data frames (by id or index) or to concatenate (by row or col) them

        :param merge_by: str
            Defining merging strategy
                -> id: Merge data frames by given id feature
                -> index: Merge data frames by index value of both

        :param id_var: str
            Name of the id feature to merge by

        :param join_type: str
            Defining merging type
                -> left: use only keys from left data frame and drop mismatching keys of right (new) data frame (preserve key order)
                -> right: use only keys from right data frame and drop mismatching keys of left (old) data frame (preserve key order)
                -> outer: use union of keys from both data frames and drop intersection keys of both data frames (sort keys lexicographically)
                -> inner: use intersection of keys from both data frames and drop union keys of both data frames (preserve order of the left keys)

        :param concat_by: str
            Defining concatenation type:
                -> row: Concatenate both data frames row-wise (the number of rows increases)
                -> col: Concatenate both data frames column-wise (the number of columns increases)

        :param cloud: str
            Name of the cloud provider:
                -> google: Google Cloud Provider
                -> aws: AWS Cloud
        """
        if cloud is None:
            _bucket_name: str = None
            if not os.path.isfile(feature_engineer_file_path):
                raise FeatureEngineerException('No external FeatureEngineer class object found')
        else:
            if cloud not in CLOUD_PROVIDER:
                raise FeatureEngineerException('Cloud provider ({}) not supported'.format(cloud))
            _bucket_name: str = feature_engineer_file_path.split("//")[1].split("/")[0]
        _external_engineer = DataImporter(file_path=feature_engineer_file_path,
                                          as_data_frame=False,
                                          cloud=cloud,
                                          bucket_name=_bucket_name
                                          ).file()
        if isinstance(_external_engineer, FeatureEngineer):
            _external_engineer_data_processing: dict = _external_engineer.data_processing
            if _external_engineer_data_processing.get('feature_types') is None:
                raise FeatureEngineerException('External file object is not a FeatureEngineer class object')
            _external_data_set: dd.DataFrame = DataImporter(file_path=feature_engineer_file_path.split(sep='.')[0],
                                                            #file_path='{}.parquet'.format(feature_engineer_file_path.split(sep='.')[0]),
                                                            as_data_frame=True,
                                                            use_dask=True,
                                                            cloud=cloud,
                                                            bucket_name=_bucket_name
                                                            ).file()
            _external_features: List[str] = []
            for ft in _external_engineer_data_processing.get('feature_types').keys():
                for feature in _external_engineer_data_processing.get('feature_types')[ft]:
                    _external_features.append(feature)
            global DATA_PROCESSING
            global FEATURE_TYPES
            global MERGES
            MERGES.update({'original': list(DATA_PROCESSING['df'].columns),
                           'current': list(DATA_PROCESSING['df'].columns),
                           'external': _external_features
                           })
            _current: List[str] = copy.deepcopy(MERGES.get('current'))
            _current.sort(reverse=False)
            _external: List[str] = copy.deepcopy(MERGES.get(list(MERGES.keys())[-1]))
            _external.sort(reverse=False)
            _new_features: List[str] = list(set(_external).difference(_current))
            _equal_features: List[str] = list(set(_external).intersection(_current))
            for equal in _equal_features:
                del _external_data_set[equal]
            if DATA_PROCESSING.get('target') is not None:
                if DATA_PROCESSING.get('target') in _external_data_set.columns:
                    del _external_data_set[DATA_PROCESSING.get('target')]
            if merging:
                if merge_by == 'id':
                    if id_var is None or id_var not in DATA_PROCESSING.get('df').columns:
                        Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log(
                            msg='ID variable ({}) not found'.format(id_var))
                    else:
                        DATA_PROCESSING['df'] = dd.merge(left=DATA_PROCESSING.get('df'),
                                                         right=_external_data_set,
                                                         on=id_var,
                                                         how=join_type,
                                                         suffixes=tuple(['', '_{}'.format(list(MERGES.keys())[-1])])
                                                         ).compute()
                        if len(_new_features) > 0:
                            for new_feature in _new_features:
                                _update_feature_types(feature=new_feature)
                        if len(_equal_features) > 0:
                            for equal_feature in _equal_features:
                                _update_feature_types(feature='{}_{}'.format(equal_feature, list(MERGES.keys())[-1]))
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(
                            msg='New Data Set after merging by {}\nCases: {}\nFeatures: {}'.format(id_var,
                                                                                                   len(DATA_PROCESSING[
                                                                                                           'df']),
                                                                                                   len(DATA_PROCESSING[
                                                                                                           'df'].columns),
                                                                                                   )
                            )
                elif merge_by == 'index':
                    DATA_PROCESSING['df'] = pd.merge(left=DATA_PROCESSING.get('df'),
                                                     right=_external_data_set,
                                                     left_index=True,
                                                     right_index=True,
                                                     suffixes=tuple(['', '_{}'.format(list(MERGES.keys())[-1])])
                                                     )
                    if len(_new_features) > 0:
                        for new_feature in _new_features:
                            _update_feature_types(feature=new_feature)
                    if len(_equal_features) > 0:
                        for equal_feature in _equal_features:
                            _update_feature_types(feature='{}_{}'.format(equal_feature, list(MERGES.keys())[-1]))
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(
                        msg='New Data Set after merging by index\nCases: {}\nFeatures: {}'.format(
                            len(DATA_PROCESSING['df']),
                            len(DATA_PROCESSING['df'].columns)
                            )
                        )
                else:
                    Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log(
                        msg='Merge by method ({}) not found'.format(merge_by))
            else:
                if concat_by == 'row':
                    DATA_PROCESSING['df'] = pd.concat([DATA_PROCESSING.get('df'), _external_data_set], axis=0)
                elif concat_by == 'col':
                    _rename: dict = {}
                    if len(_equal_features) > 0:
                        for equal_feature in _equal_features:
                            _rename.update(
                                {equal_feature: '{}_{}'.format(_rename.get(equal_feature), list(MERGES.keys())[-1])})
                    #feature_engineer_obj.set_feature_names(name_map=_rename, lower_case=True)
                    DATA_PROCESSING['df'] = pd.concat([DATA_PROCESSING.get('df'), _external_data_set], axis=1)
                    if len(_new_features) > 0:
                        for new_feature in _new_features:
                            _update_feature_types(feature=new_feature)
                    for renamed in _rename.keys():
                        _update_feature_types(feature=_rename.get(renamed))
            _new_features_types: dict = _external_engineer_data_processing.get('feature_types')
            for feature_type in _new_features_types.keys():
                for new_feature in _new_features_types.get(feature_type):
                    if new_feature not in FEATURE_TYPES.get(feature_type):
                        FEATURE_TYPES[feature_type].append(new_feature)
        else:
            raise FeatureEngineerException('File object type ({}) is not a dict'.format(type(_external_engineer)))

    @staticmethod
    @FeatureOrchestra(meth='merge_text', feature_types=['categorical', 'id_text'])
    def merge_text(features: List[str] = None, sep: str = None):
        """
        Merge text features by given separator

        :param features: List[str]
            Name of the features

        :param sep: str
            Separator
        """
        if features is None:
            _features: List[str] = FEATURE_TYPES.get('categorical') + FEATURE_TYPES.get('id_text')
        else:
            _features: List[str] = []
            for feature in features:
                if feature in DATA_PROCESSING['df'].columns:
                    _features.append(feature)
        if len(_features) > 1:
            _pairs: List[tuple] = EasyExploreUtils().get_pairs(features=_features, max_features_each_pair=2)
            for pair in _pairs:
                _load_temp_files(features=[pair[0], pair[1]])
                if str(DATA_PROCESSING['df'][pair[0]].dtype).find('object') >= 0:
                    TEXT_MINER['obj'].merge(features=[pair[0], pair[1]], sep=sep)
                    _process_handler(action='add',
                                     feature=pair[0],
                                     new_feature=TEXT_MINER['obj'].generated_features[pair[0]]['merge'][-1],
                                     process='text|merge',
                                     meth='merge_text',
                                     param=dict(sep=sep),
                                     data=TEXT_MINER['obj'].df[TEXT_MINER['obj'].generated_features[pair[0]]['merge'][-1]],
                                     obj={pair[0]: pair[1]}
                                     )
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformed feature "{}" using text splitting (sep -> {})'.format(pair[0], sep))

    @staticmethod
    def missing_data_analysis(update: bool = False, last_generated: bool = True, features: List[str] = None):
        """
        Missing data analysis

        :param update: bool
            Whether to update observed missing data statistics by new generated features or run analysis on all features

        :param last_generated: bool
            Whether to update last generated (engineered) feature or already analyzed features

        :param features: List[str]
            Name of the features to update if parameter "last_generated = False"
        """
        if update:
            if last_generated:
                _last_generated_feature: str = DATA_PROCESSING.get('last_generated_feature')
                if _last_generated_feature != '' and _last_generated_feature not in DATA_PROCESSING['mapper']['mis']['features'].keys():
                    DATA_PROCESSING['mapper']['mis']['features'].update({_last_generated_feature: MissingDataAnalysis(df=DATA_PROCESSING['df'][_last_generated_feature], feature=_last_generated_feature).get_nan_idx_by_features().get(_last_generated_feature)})
            else:
                _features: List[str] = list(DATA_PROCESSING.get('df').columns) if features is None else features
                DATA_PROCESSING['mapper']['mis']['features'].update({_features: MissingDataAnalysis(df=DATA_PROCESSING['df'], feature=_features).get_nan_idx_by_features().get(_features)})
        else:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Missing Value Analysis: Feature -> {}'.format(features))
            _total_missing: int = DATA_PROCESSING.get('df').isnull().astype(int).sum().sum()
            for feature in features:
                _mis_by_features: dict = MissingDataAnalysis(df=DATA_PROCESSING.get('df'), features=[feature]).get_nan_idx_by_features()
                if DATA_PROCESSING['mapper']['mis'].get('features') is None:
                    #DATA_PROCESSING['mapper']['mis'].update({'cases': MissingDataAnalysis(df=DATA_PROCESSING.get('df'), features=features).get_nan_idx_by_cases()})
                    DATA_PROCESSING['mapper']['mis'].update({'features': _mis_by_features})
                else:
                    #DATA_PROCESSING['mapper']['mis'].update({'cases': MissingDataAnalysis(df=DATA_PROCESSING.get('df'), features=[feature]).get_nan_idx_by_cases()})
                    DATA_PROCESSING['mapper']['mis']['features'].update({feature: _mis_by_features.get(feature)})
            DATA_PROCESSING['missing_data']['total']['mis'] += _total_missing
            DATA_PROCESSING['missing_data']['total']['valid'] += len(DATA_PROCESSING.get('df')) - _total_missing
            #DATA_PROCESSING['missing_data']['total']['cases'].extend(MissingDataAnalysis(df=DATA_PROCESSING.get('df'), features=features, percentages=True).freq_nan_by_cases())
            #DATA_PROCESSING['missing_data']['total']['features'].extend(MissingDataAnalysis(df=DATA_PROCESSING.get('df'), features=features, percentages=True).freq_nan_by_features())

    @staticmethod
    @FeatureOrchestra(meth='one_hot_encoder', feature_types=['categorical'])
    def one_hot_encoder(features: List[str] = None, threshold: int = None):
        """
        One-Hot-Encoder for generating binary dummy features

        :param features: List[str]
            Names of features

        :param threshold: int
            Maximum number of unique categories for one-hot encoding
        """
        if threshold is None:
            _threshold: int = 0
        else:
            _threshold: int = threshold if threshold > 0 else 10000
        for feature in features:
            _load_temp_files(features=[feature])
            _unique: int = len(DATA_PROCESSING['df'][feature].unique())
            if threshold is None or (_unique <= _threshold):
                if feature not in DATA_PROCESSING['encoder']['one_hot'].keys():
                    if str(DATA_PROCESSING['df'][feature].dtype).find('[ns]') >= 0:
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Cannot transform datetime feature "{}" using one-hot encoding'.format(feature))
                        continue
                    if str(DATA_PROCESSING['df'][feature].dtype).find('object') < 0:
                        DATA_PROCESSING['df'][feature] = DATA_PROCESSING['df'][feature].astype(dtype='object')
                    #_dummies = pd.get_dummies(data=DATA_PROCESSING['df'][feature],
                    #                          prefix=feature,
                    #                          prefix_sep='_',
                    #                          dummy_na=True,
                    #                          sparse=False,
                    #                          drop_first=False
                    #                          )
                    _dummies: pd.DataFrame = pd.get_dummies(data=DATA_PROCESSING['df'][[feature]],
                                                            prefix=None,
                                                            prefix_sep='_',
                                                            dummy_na=True,
                                                            columns=None,
                                                            sparse=False,
                                                            drop_first=False,
                                                            dtype=np.int64
                                                            )
                    _dummies = _dummies.loc[:, ~_dummies.columns.duplicated()]
                    _new_names: dict = {}
                    for dummie in _dummies.columns:
                        _new_feature: str = _avoid_overwriting(feature=dummie)
                        if dummie != _new_feature:
                            _new_names.update({dummie: _new_feature})
                    if len(_new_names) > 0:
                        _dummies = _dummies.rename(columns=_new_names)
                    DATA_PROCESSING['df'] = dd.concat(dfs=[DATA_PROCESSING.get('df'), _dummies], axis=1)
                    _process_handler(action='add',
                                     feature=feature,
                                     new_feature='',
                                     process='encoder|one_hot',
                                     meth='one_hot_encoder',
                                     param=dict(threshold=_threshold),
                                     data=_dummies,
                                     force_type='categorical',
                                     special_replacement=False,
                                     obj=list(_dummies.columns)
                                     )
                    del _dummies
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformed feature "{}" using one-hot encoding'.format(feature))
            else:
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Feature ({}) has too many categories ({})'.format(feature,
                                                                                                                          len(DATA_PROCESSING['df'][feature].unique())
                                                                                                                          )
                                                                   )

    @staticmethod
    @FeatureOrchestra(meth='outlier_detection', feature_types=['continuous'])
    def outlier_detection(features: List[str] = None,
                          threshold: float = 0.1,
                          kind: str = 'multi',
                          *multi_meth,
                          **meth_param
                          ):
        """
        Detect anomalies / outliers in the data set

        :param features: list[str]
            Name of the features

        :param threshold: float
            Outlier threshold

        :param kind: str
            Outlier detection type:
                -> uni: Univariate (each feature separately)
                -> bi: Bivariate (each feature pair)
                -> multi: Multivariate (all features)
        """
        for feature in features:
            if kind in ['bi', 'multi']:
                _name: str = '{}_outlier_multi'.format(feature)
                DATA_PROCESSING.get('df')[_name] = AnomalyDetector(df=DATA_PROCESSING.get('df'),
                                                                   features=features,
                                                                   feature_types=FEATURE_TYPES,
                                                                   outlier_threshold=threshold
                                                                   ).multivariate()[feature].get('pred')
            else:
                _name: str = '{}_outlier_uni'.format(feature)
                DATA_PROCESSING.get('df')[_name] = AnomalyDetector(df=DATA_PROCESSING.get('df'),
                                                                   features=features,
                                                                   feature_types=FEATURE_TYPES,
                                                                   outlier_threshold=threshold
                                                                   ).univariate()[feature].get('pred')

    def re_engineer(self, data_set: dict, features: List[str]) -> np.ndarray:
        """
        Re-engineer features used in training to generate prediction from trained model

        :param data_set: dict
            Data set

        :param features: List[str]
            Name of the features to re_engineer for prediction

        :return np.ndarray:
            Engineered data set by taking the same processing steps as before (in training)
        """
        DATA_PROCESSING['re_generate'] = True
        _data_set: pd.DataFrame = pd.DataFrame(data=data_set)
        _processes: dict = self.get_processing()['process']
        _processing_relations: dict = {}
        _original_features: List[str] = []
        for proc_feature in features:
            _processing_relations.update({proc_feature: self.get_processing_relation(feature=proc_feature)})
            for f in _processing_relations[proc_feature]['parents']['raw']:
                _original_features.append(f)
        _original_features = list(set(_original_features))
        _processed_features: dict = {}
        for feature in _original_features:
            if feature not in _data_set.keys():
                raise FeatureEngineerException('Feature ({}) not found in data set'.format(feature))
        _transformations: dict = self.get_transformations()
        _renaming: dict = _transformations.get('naming')
        if len(_renaming.keys()) > 0:
            _data_set.rename(_renaming)
        _mapper: dict = _transformations.get('mapper')
        if len(_mapper.keys()) > 0:
            _data_set.replace(_mapper)
        _label: dict = _transformations.get('label')
        if len(_label.keys()) > 0:
            _data_set.replace(_label)
        _one_hot: dict = _transformations.get('one_hot')
        if len(_one_hot.keys()) > 0:
            _data_set.replace(_one_hot)
        _max_level: int = len(self.get_processing()['features'].keys()) - 1
        for level in range(0, _max_level, 1):
            if level > 0:
                for ft in _processing_relations.keys():
                    for level_feature in _processing_relations[ft]['parents']['level_{}'.format(level)]:
                        for process in _processes:
                            if level_feature in _processes[process]['features'].keys():
                                _param: dict = dict(features=_processes[process]['features'][level_feature])
                                for param, value in _processes[process]['param'].items():
                                    _param.update({param: value})
                                getattr(self, _processes[process]['meth'])(_param)
                                if len(DATA_PROCESSING.get('re_gen_data').shape) == 1:
                                    _data_set[level_feature] = DATA_PROCESSING.get('re_gen_data')
                                else:
                                    _data_set = pd.concat([_data_set, DATA_PROCESSING.get('re_gen_data')], axis=1)
                                # Adjust _process_handler to receive engineering
        DATA_PROCESSING['re_gen_data'] = None
        DATA_PROCESSING['re_generate'] = False
        return _data_set.values

    @staticmethod
    def replacer(replacement: Dict[str, dict]):
        """
        Replace categorical values

        :param replacement: Dict[str, dict]
            Value replacement [feature, dict(old value: new value)]
        """
        for feature in replacement.keys():
            _load_temp_files(features=[feature])
            if feature in DATA_PROCESSING['df'].columns:
                _process_handler(action='add',
                                 feature=feature,
                                 new_feature='{}_replace'.format(feature) if DATA_PROCESSING.get('generate_new_feature') else feature,
                                 process='mapper|obs',
                                 meth='replacer',
                                 param=dict(replacement=replacement),
                                 data=DATA_PROCESSING['df'][feature],
                                 obj=replacement
                                 )
                for old in replacement[feature].keys():
                    Log(write=not DATA_PROCESSING.get('show_msg')).log('Binned feature "{}" using replacement value {} -> {}'.format(feature, old, str(replacement[feature][old])))

    @staticmethod
    def reset_feature_processing_relation():
        """
        Reset feature processing relation manually
        """
        raise FeatureEngineerException('Reset feature processing relation not implemented')

    @staticmethod
    def reset_original_data_set(keep_current_data: bool = True):
        """
        Reset original data set if object was initialized with parameter 'keep_original_data = True'

        keep_current_data: bool
            Keep current data set flagged as original or remove it
        """
        if DATA_PROCESSING.get('raw_data') is not None:
            _raw_data_set: dict = DATA_PROCESSING.get('raw_data')
            if keep_current_data:
                DATA_PROCESSING['raw_data'] = DATA_PROCESSING.get('df').to_dict()
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Keep current data as back up')
            DATA_PROCESSING['df'] = pd.DataFrame(_raw_data_set)
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Reset original data set from initialization')

    @staticmethod
    def reset_predictors():
        """
        Reset predictors
        """
        DATA_PROCESSING['predictors'] = []
        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Reset predictors')

    @staticmethod
    def reset_ignore_processing():
        """
        Reset predictors which are excluded from pre-processing
        """
        global PREDICTORS
        PREDICTORS = []
        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Reset ignore predictors from processing')

    @staticmethod
    def reset_multi_threading():
        """
        Disable multi-threading
        """
        DATA_PROCESSING['multi_threading'] = False
        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Disable multi-threading')

    @staticmethod
    def reset_target(targets: List[str] = None):
        """
        Reset target settings

        :param targets: List[str]
            Names of the set target features
        """
        if targets is None:
            FEATURE_TYPES[DATA_PROCESSING['target_type'][-1]].append(DATA_PROCESSING['target'])
            DATA_PROCESSING['target'] = []
            DATA_PROCESSING['target_type'] = []
        else:
            for target in targets:
                if target == DATA_PROCESSING.get('target'):
                    FEATURE_TYPES[DATA_PROCESSING['target_type'][targets.index(target)]].append(target)
                    DATA_PROCESSING['target'] = []
                    DATA_PROCESSING['target_type'] = []
        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Reset target')

    @staticmethod
    @FeatureOrchestra(meth='rounding', feature_types=['continuous'])
    def rounding(features: List[str] = None, digits: int = 100):
        """
        Rounding continuous features

        :param features: List[str]
            Name of features

        :param digits: int
            Rounding factor
        """
        for feature in features:
            _load_temp_files(features=[feature])
            _process_handler(action='add',
                             feature=feature,
                             new_feature='{}_round_{}'.format(feature, str(digits)) if DATA_PROCESSING.get('generate_new_feature') else feature,
                             process='scaler|rounding',
                             meth='rounding',
                             param=dict(digits=digits),
                             data=np.array(np.round((DATA_PROCESSING.get('df')[feature].values * digits))),
                             force_type='ordinal',
                             imp_value=0,
                             special_replacement=True
                             )
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformed feature "{}" using rounding'.format(feature))

    def save(self,
             file_path: str = None,
             cls_obj: bool = True,
             write_parquet: bool = True,
             overwrite: bool = True,
             create_dir: bool = False,
             cloud: str = None
             ):
        """
        Save data engineering information

        :param file_path: str
            Complete file path to save data engineering information

        :param cls_obj: bool
            Whether to export class object or processing dictionary as pickle file

        :param write_parquet: bool
            Write data set (dask DataFrame) as parquet

        :param overwrite: bool
            Whether to overwrite existing file or not

        :param create_dir: bool
            Whether to create directory if they are not existed or not

        :param cloud: str
            Name of the cloud provider
                -> google: Google Cloud Storage
        """
        if cloud is None:
            _bucket_name: str = None
        else:
            if cloud not in CLOUD_PROVIDER:
                raise FeatureEngineerException('Cloud provider ({}) not supported'.format(cloud))
            _bucket_name: str = file_path.split("//")[1].split("/")[0]
        global TEXT_MINER
        global DATA_PROCESSING
        TEXT_MINER['obj'] = None
        if file_path is not None or os.path.isfile(DATA_PROCESSING.get('source')):
            self.data_processing = dict(processing=DATA_PROCESSING,
                                        feature_types=FEATURE_TYPES,
                                        special_jobs=SPECIAL_JOBS,
                                        predictors=PREDICTORS,
                                        processing_action_space=PROCESSING_ACTION_SPACE,
                                        merges=MERGES,
                                        text_miner=TEXT_MINER,
                                        notepad=NOTEPAD,
                                        kwargs=self.kwargs
                                        )
            if file_path is None:
                _file_path: str = DATA_PROCESSING.get('source')
            else:
                _file_path: str = file_path
            self.dask_client = None
            if write_parquet:
                _parquet_file_path: str = _file_path.split('.')[0]
                DataExporter(obj=DATA_PROCESSING.get('df'),
                             file_path=_parquet_file_path,
                             #file_path='{}.parquet'.format(_parquet_file_path),
                             create_dir=create_dir,
                             overwrite=overwrite
                             ).file()
            DATA_PROCESSING['df'] = None
            if cls_obj:
                DataExporter(obj=self,
                             file_path=_file_path,
                             create_dir=create_dir,
                             overwrite=overwrite,
                             cloud=cloud,
                             bucket_name=_bucket_name
                             ).file()
            else:
                DataExporter(obj=self.data_processing,
                             file_path=_file_path,
                             create_dir=create_dir,
                             overwrite=overwrite,
                             cloud=cloud,
                             bucket_name=_bucket_name
                             ).file()
            self.data_processing = None
            self.dask_client = HappyLearningUtils().dask_setup(client_name='feature_engineer',
                                                               client_address=self.kwargs.get('client_address'),
                                                               mode='threads' if self.kwargs.get('client_mode') is None else self.kwargs.get('client_mode')
                                                               )
            DATA_PROCESSING['source'] = file_path
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Save and export feature engineering: "{}"'.format(file_path))

    @staticmethod
    def sampler(n: int, meth: str = 'random', quota: dict = None) -> pd.DataFrame:
        """
        Draw sample from data set

        :param n: int
            Sample size

        :param meth: str
            Sampling method
                -> random: Sample cases randomly
                -> quota: Sample cases by given quota (feature distribution)
                -> feature: Sample features randomly

        :param quota: dict
            Quota (value distribution) for each feature used for sampling
        """
        if meth == 'random':
            return Sampler(df=DATA_PROCESSING.get('df'), size=n).random()
        elif meth == 'quota':
            return Sampler(df=DATA_PROCESSING.get('df'), size=n).quota(features=list(DATA_PROCESSING.get('df').columns), quotas=quota)
        elif meth == 'feature':
            pass
        else:
            Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log(msg='Sampling method ({}) not supported'.format(meth))
        return pd.DataFrame()

    @staticmethod
    @FeatureOrchestra(meth='scaling_robust', feature_types=['continuous'])
    def scaling_robust(features: List[str] = None,
                       quantile_range: Tuple[float, float] = (0.25, 0.75),
                       with_centering: bool = True,
                       with_scaling: bool = True,
                       ):
        """
        Scaling continuous features using robust scaler

        :param features: List[str]
            Names of the features

        :param quantile_range: Tuple[float, float]
            Quantile ranges of the robust scaler

        :param with_centering: bool
            Use centering using robust scaler

        :param with_scaling: bool
            Use scaling using robust scaler
        """
        _lower_thres = 100 * quantile_range[0] if quantile_range[0] < 1 else quantile_range[0]
        _upper_thres = 100 * quantile_range[1] if quantile_range[1] < 1 else quantile_range[1]
        _robust = RobustScaler(with_centering=with_centering,
                               with_scaling=with_scaling,
                               quantile_range=(float(_lower_thres), float(_upper_thres)),
                               copy=False
                               )
        for feature in features:
            _load_temp_files(features=[feature])
            _data: np.array = DATA_PROCESSING['df'][feature].fillna(sys.float_info.min).values
            _robust.fit(np.reshape(_data, (-1, 1)), y=None)
            _process_handler(action='add',
                             feature=feature,
                             new_feature='{}_{}'.format(feature, DATA_PROCESSING['suffixes'].get('robust')) if DATA_PROCESSING.get('generate_new_feature') else feature,
                             process='scaler|robust',
                             meth='scaling_robust',
                             param=dict(quantile_range=quantile_range, with_centering=with_centering, with_scaling=with_scaling),
                             data=np.reshape(_robust.transform(X=np.reshape(copy.deepcopy(_data), (-1, 1))), (1, -1))[0],
                             force_type='continuous',
                             special_replacement=True,
                             imp_value=sys.float_info.min,
                             obj=_robust
                             )
            del _data
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformed feature "{}" using robust scaling'.format(feature))

    @staticmethod
    @FeatureOrchestra(meth='scaling_minmax', feature_types=['continuous'])
    def scaling_minmax(features: List[str] = None, minmax_range: Tuple[int, int] = (0, 1)):
        """
        Scaling continuous features using min-max scaler

        :param features: List[str]
            Names of the features

        :param minmax_range: Tuple[int, int]
            Min-Max ranges of the min-max scaler
        """
        _minmax = MinMaxScaler(feature_range=minmax_range)
        for feature in features:
            _load_temp_files(features=[feature])
            _data: np.array = DATA_PROCESSING.get('df')[feature].fillna(sys.float_info.min).values
            _minmax.fit(np.reshape(_data, (-1, 1)), y=None)
            _process_handler(action='add',
                             feature=feature,
                             new_feature='{}_{}'.format(feature, DATA_PROCESSING['suffixes'].get('minmax')) if DATA_PROCESSING.get('generate_new_feature') else feature,
                             process='scaler|robust',
                             meth='scaling_minmax',
                             param=dict(minmax_range=minmax_range),
                             data=np.reshape(_minmax.transform(X=np.reshape(copy.deepcopy(_data), (-1, 1))), (1, -1))[0],
                             force_type='continuous',
                             special_replacement=True,
                             imp_value=sys.float_info.min,
                             obj=_minmax
                             )
            del _data
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformed feature "{}" using min-max scaling'.format(feature))

    @staticmethod
    @FeatureOrchestra(meth='self_interaction', feature_types=['continuous', 'ordinal'])
    def self_interaction(features: List[str],
                         addition: bool = True,
                         multiplication: bool = True
                         ):
        """
        Calculate interaction with single feature

        :param features: List[str]
            Name of the features

        :param addition: bool
            Whether to add feature with itself or not

        :param multiplication: bool
            Whether to multiply feature with itself or not
        """
        for feature in features:
            _load_temp_files(features=[feature])
            if addition:
                _process_handler(action='add',
                                 feature=feature,
                                 new_feature='{}__add__{}'.format(feature, feature),
                                 process='self_interaction|addition',
                                 meth='self_interaction',
                                 param=dict(addition=addition, multiplication=multiplication),
                                 data=(DATA_PROCESSING['df'][feature] + DATA_PROCESSING['df'][feature]),
                                 force_type='continuous',
                                 imp_value=sys.float_info.max,
                                 obj=None
                                 )
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Generate feature by adding feature "{}" with all other compatible features ({})'.format(feature, len(features)))
            if multiplication:
                _process_handler(action='add',
                                 feature=feature,
                                 new_feature='{}__multi__{}'.format(feature, feature),
                                 process='self_interaction|multiplication',
                                 meth='self_interaction',
                                 param=dict(addition=addition, multiplication=multiplication),
                                 data=(DATA_PROCESSING['df'][feature] * DATA_PROCESSING['df'][feature]),
                                 force_type='continuous',
                                 imp_value=sys.float_info.max,
                                 obj=None
                                 )
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Generate feature by multiplying feature "{}" with all other compatible features ({})'.format(feature, len(features)))

    def set_back_up_data(self, external_data: pd.DataFrame = None, reset_current_data: bool = True):
        """
        Set back up data either by current or external data set

        :param external_data: pd.DataFrame
            External data set

        :param reset_current_data: bool
            Reset current data frame by external data set too
        """
        global MERGES
        global PREDICTORS
        global FEATURE_TYPES
        MERGES = {}
        PREDICTORS = []
        FEATURE_TYPES = {}
        if external_data is None:
            DATA_PROCESSING['raw_data'] = DATA_PROCESSING.get('df')
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Set current data set as back up data set')
        else:
            if reset_current_data:
                _kwargs: dict = {}
                for attr in inspect.getfullargspec(self).args:
                    if attr != 'self':
                        _kwargs.update({attr: getattr(self, attr)})
                self.set_data(df=external_data, **_kwargs)
            else:
                DATA_PROCESSING['raw_data'] = external_data
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Set external data as back up data set')

    def set_critic_config(self, config: dict = None):
        """
        Set internal critic configuration

        :param config: dict
        """
        if config is None:
            _config: dict = {}
        else:
            _config: dict = config
        _enable_recommender: int = 1
        _possible_actions: int = len(PROCESSING_ACTION_SPACE['ordinal']) + len(PROCESSING_ACTION_SPACE['categorical']) + len(PROCESSING_ACTION_SPACE['continuous'])
        _burn_in_iteration: int = round((len(self.get_predictors()) * _possible_actions) * 0.15)
        DATA_PROCESSING['critic_config'].update({'enable_recommender': _enable_recommender,
                                                 '_burn_in_iteration': _burn_in_iteration
                                                 })

    @classmethod
    def set_data(cls, df=None, file_path: str = None, **kwargs):
        """
        Set data attribute by external data set

        :param df: Pandas or dask DataFrame
            External data set

        :param file_path: str
            File path of external data set

        :param kwargs: dict
            Key-word arguments
        """
        global MERGES
        global PREDICTORS
        global FEATURE_TYPES
        MERGES = {}
        PREDICTORS = []
        FEATURE_TYPES = {}
        # TODO: add complete parameter set
        return cls(df=df,
                   generate_new_feature=kwargs.get('generate_new_feature'),
                   id_text_features=kwargs.get('id_text_features'),
                   date_features=kwargs.get('date_features'),
                   ordinal_features=kwargs.get('ordinal_features'),
                   keep_original_data=kwargs.get('keep_original_data'),
                   unify_invalid_values=kwargs.get('unify_invalid_values'),
                   encode_missing_data=kwargs.get('encode_missing_data'),
                   date_edges=kwargs.get('date_edges'),
                   auto_cleaning=kwargs.get('auto_cleaning'),
                   auto_typing=kwargs.get('auto_typing'),
                   file_path=file_path,
                   temp_dir=TEMP_DIR
                   )

    @staticmethod
    def set_feature_names(name_map: Dict[str, str] = None, lower_case: bool = True):
        """
        Set features names

        :param name_map: dict
            Renaming map
                -> key: Current feature name
                -> value: New feature name

        :param lower_case: bool
            Lower case feature names
        """
        if name_map is None:
            for feature in DATA_PROCESSING['df'].columns:
                _process_handler(action='rename',
                                 feature=feature,
                                 new_feature=feature.lower() if lower_case else feature.upper(),
                                 process='mapper|names',
                                 meth='set_feature_names',
                                 param=dict(name_map=name_map, lower_case=lower_case),
                                 data=None
                                 )
        else:
            for feature in name_map.keys():
                if feature in DATA_PROCESSING['df'].columns:
                    _process_handler(action='rename',
                                     feature=feature,
                                     new_feature=name_map.get(feature),
                                     process='mapper|names',
                                     meth='set_feature_names',
                                     param=dict(name_map=name_map, lower_case=lower_case),
                                     data=None
                                     )

    @staticmethod
    def set_feature_processing_relation(processed_feature: str, parent_features: List[str]):
        """
        Set feature processing relations manually

        :param processed_feature: str
            Name of the processed (child) feature

        :param parent_features: List[str]
            Name of the parent features
        """
        if processed_feature in DATA_PROCESSING['df'].columns:
            for parent in parent_features:
                if parent in DATA_PROCESSING['df'].columns:
                    _set_feature_relations(feature=parent, new_feature=processed_feature)

    @staticmethod
    def set_feature_types(feature_types: Dict[str, str]):
        """
        Set features types manually

        :param feature_types: Dict[str, str]
            Feature name and the new feature types
                -> continuous: Continuous features
                -> categorical: Categorical features
                -> ordinal: Ordinal features
                -> date: Date features
                -> id_text: Text or ID features
        """
        for feature in feature_types.keys():
            if feature in DATA_PROCESSING['df'].columns:
                _update_feature_types(feature=feature, force_type=feature_types.get(feature))

    @staticmethod
    def set_index(idx: List[str]):
        """
        Set index values of Pandas DataFrame

        :param idx: List[str]
            Index values to set
        """
        DATA_PROCESSING['df'] = DATA_PROCESSING.get('df').set_index(keys=idx, drop=True, append=False, verify_integrity=False)

    @staticmethod
    def set_max_processing_level(level: int):
        """
        Set maximum level of feature processing

        :param level: int
            Maximum number of feature processing levels
        """
        if level >= 1:
            DATA_PROCESSING['max_level'] = level
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Set maximum level of feature processing to "{}"'.format(level))
        else:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Level of feature processing not supported "{}". It should be greater than 0'.format(level))

    @staticmethod
    def set_partitions(n_partitions: int):
        """
        Set dask partitions

        :param n_partitions: int
            Number of partitions to reset
        """
        DATA_PROCESSING['df'] = DATA_PROCESSING['df'].repartition(n_partitions)

    @staticmethod
    @FeatureOrchestra(meth='set_predictors', feature_types=['categorical', 'continuous', 'ordinal'])
    def set_predictors(features: List[str], exclude: List[str] = None, exclude_original_data: bool = True):
        """
        Set predictors

        :param features: List[str]
            Name of the features used as predictors

        :param exclude: List[str]
            Name of the features to exclude

        :param exclude_original_data: bool
            Exclude original features
        """
        _target_relations: List[str] = []
        if DATA_PROCESSING.get('target') in DATA_PROCESSING['processing']['features']['raw'].keys():
            _target_relations = DATA_PROCESSING['processing']['features']['raw'][DATA_PROCESSING.get('target')]
        else:
            for ft in DATA_PROCESSING['processing']['features']['raw'].keys():
                if DATA_PROCESSING.get('target') in DATA_PROCESSING['processing']['features']['raw'][ft]:
                    _target_relations = DATA_PROCESSING['processing']['features']['raw'][ft]
                    _target_relations.append(ft)
                    break
        _predictors: List[str] = copy.deepcopy(features)
        if exclude is not None:
            for feature in exclude:
                if feature in _predictors:
                    del _predictors[_predictors.index(feature)]
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Exclude feature "{}"'.format(feature))
        if DATA_PROCESSING.get('target') is not None:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Exclude target feature "{}"'.format(DATA_PROCESSING.get('target')))
        for relation in _target_relations:
            if relation in _predictors:
                del _predictors[_predictors.index(relation)]
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Exclude feature "{}"'.format(relation))
        if exclude_original_data:
            for raw in DATA_PROCESSING.get('original_features'):
                if raw in _predictors:
                    del _predictors[_predictors.index(raw)]
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Exclude original feature "{}"'.format(raw))
        else:
            for feature in DATA_PROCESSING.get('original_features'):
                if feature in _predictors:
                    if feature not in FEATURE_TYPES.get('continuous'):
                        if feature not in FEATURE_TYPES.get('ordinal'):
                            del _predictors[_predictors.index(feature)]
                            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Exclude original (non-numeric) feature "{}"'.format(feature))
        DATA_PROCESSING['predictors'] = _predictors
        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Set {} predictors'.format(len(_predictors)))

    @staticmethod
    def set_ignore_processing(predictors: List[str]):
        """
        Set predictors which are excluded from pre-processing

        :param predictors: List[str]
            Predictor names
        """
        global PREDICTORS
        for predictor in predictors:
            if predictor in DATA_PROCESSING['df'].columns:
                PREDICTORS.append(predictor)
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Set predictor ({}). This predictor will be excluded from any further engineering process'.format(predictor))

    @staticmethod
    def set_imp_features(imp_features: List[str]):
        """
        Set feature importance scoring to criticize actions

        :param imp_features: List[str]
            Name of features sorted by their relative importance scores
        """
        _imp_features: List[str] = []
        for feature in imp_features:
            if feature in DATA_PROCESSING['df'].columns:
                _imp_features.append(feature)
        if len(_imp_features) > 0:
            DATA_PROCESSING['imp_features'] = _imp_features
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Set feature importance scoring for criticizing actions')

    def set_target(self, feature: str):
        """
        Set target feature

        :param feature: str
            Name of the feature used as target
        """
        if feature in ALL_FEATURES:
            if DATA_PROCESSING.get('target') is None:
                DATA_PROCESSING.update({'target': feature})
            else:
                DATA_PROCESSING['target'] = feature
            for ft in FEATURE_TYPES.keys():
                if feature in FEATURE_TYPES.get(ft):
                    _features: List[str] = copy.deepcopy(FEATURE_TYPES.get(ft))
                    DATA_PROCESSING['target_type'].append(ft)
                    del _features[_features.index(feature)]
                    FEATURE_TYPES[ft] = copy.deepcopy(_features)
                    break
            self.target_type_adjustment(label_encode=True)
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Set target feature: {}'.format(feature))
        else:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Target feature ({}) not found'.format(feature))

    @staticmethod
    def sort(sorting_features: List[str] = None,
             sort_by_index: bool = False,
             ascending: bool = True,
             sort_features_alphabetically: bool = True
             ):
        """
        Sort data

        :param sorting_features: List[str]
            Names of ordered features to sort by

        :param sort_by_index: bool
            Whether to sort data set by index value

        :param ascending: bool
            Sort values or features ascending or descending

        :param sort_features_alphabetically: bool
            Sort feature names alphabetically (A-Z)
        """
        if sorting_features is None:
            # TODO: sorting features is None
            pass
        else:
            _features: List[str] = []
            for ft in FEATURE_TYPES.keys():
                for feature in FEATURE_TYPES.get(ft):
                    _features.append(feature)
            if DATA_PROCESSING['target'] is not None:
                _features.append(DATA_PROCESSING.get('target'))
            _load_temp_files(features=_features)
            if len(sorting_features) > 0:
                _features: List[str] = []
                for feature in sorting_features:
                    if feature in DATA_PROCESSING['df'].columns:
                        _features.append(feature)
                if len(_features) > 0:
                    DATA_PROCESSING['df'] = DATA_PROCESSING['df'].sort_values(by=_features, axis=0, ascending=ascending)
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Data set sorted by values of features {}'.format(_features))
            for feature in _features:
                _save_temp_files(feature=feature)
        if sort_features_alphabetically:
            _features: List[str] = []
            for ft in FEATURE_TYPES.keys():
                for feature in FEATURE_TYPES.get(ft):
                    _features.append(feature)
            if DATA_PROCESSING['target'] is not None:
                _features.append(DATA_PROCESSING.get('target'))
            _load_temp_files(features=_features)
            _reverse: bool = not ascending
            _ordered_feature_names: List[str] = list(DATA_PROCESSING['df'].columns)
            _ordered_feature_names.sort(reverse=_reverse)
            _df: pd.DataFrame = pd.DataFrame()
            for feature in _ordered_feature_names:
                _df[feature] = DATA_PROCESSING['df'][feature]
            DATA_PROCESSING['df'] = _df
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Data set sorted by feature names alphabetically')
            for feature in _features:
                _save_temp_files(feature=feature)

    @staticmethod
    @FeatureOrchestra(meth='splitter', feature_types=['id_text', 'categorical'])
    def splitter(sep: str, features: List[str] = None):
        """
        Split text feature by given separator

        :param sep: str
            Separator

        :param features: List[str]
            Name of the features
        """
        if features is None:
            _features: List[str] = FEATURE_TYPES.get('categorical') + FEATURE_TYPES.get('id_text')
        else:
            _features: List[str] = []
            for feature in features:
                if feature in DATA_PROCESSING['df'].columns:
                    _features.append(feature)
        if len(_features) > 1:
            for feature in _features:
                _load_temp_files(features=[feature])
                if str(DATA_PROCESSING['df'][feature].dtype).find('object') >= 0:
                    TEXT_MINER['obj'].splitter(features=[feature], sep=sep)
                    _process_handler(action='add',
                                     feature=feature,
                                     new_feature=TEXT_MINER['obj'].generated_features[feature]['split'][-1],
                                     process='text|split',
                                     meth='splitter',
                                     param=dict(sep=sep),
                                     data=TEXT_MINER['obj'].get_generated_features(features=[TEXT_MINER['obj'].generated_features[feature]['split'][-1]], compute=True),
                                     obj={feature: sep}
                                     )
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformed feature "{}" using text splitting (sep -> {})'.format(feature, sep))

    @staticmethod
    @FeatureOrchestra(meth='square_root_transform', feature_types=['continuous'])
    def square_root_transform(features: List[str] = None):
        """
        Transform continuous features using square-root transformation

        :param features: List[str]
            Name of the features
        """
        for feature in features:
            _load_temp_files(features=[feature])
            _process_handler(action='add',
                             feature=feature,
                             new_feature='{}_square'.format(feature) if DATA_PROCESSING.get('generate_new_feature') else feature,
                             process='square_root',
                             meth='square_root_transform',
                             param=dict(),
                             data=np.square(DATA_PROCESSING['df'][feature].values),
                             force_type='continuous',
                             special_replacement=True,
                             imp_value=sys.float_info.min,
                             obj=None
                             )
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformed feature "{}" using square-root transformation'.format(feature))

    @staticmethod
    @FeatureOrchestra(meth='standardizer', feature_types=['continuous'])
    def standardizer(features: List[str] = None, with_mean: bool = True, with_std: bool = True):
        """
        Standardize continuous features

        :param features: List[str]
            Name of the features

        :param with_mean: bool
            Using mean to standardize features

        :param with_std: bool
            Using standard deviation to standardize features
        """
        _scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        for feature in features:
            _load_temp_files(features=[feature])
            _data: np.array = DATA_PROCESSING.get('df')[feature].fillna(sys.float_info.min).values
            _scaler.fit(X=np.reshape(_data, (-1, 1)))
            _process_handler(action='add',
                             feature=feature,
                             new_feature='{}_standard'.format(feature) if DATA_PROCESSING.get('generate_new_feature') else feature,
                             process='scaler|standard',
                             meth='standardizer',
                             param=dict(with_mean=with_mean, with_std=with_std),
                             data=np.reshape(_scaler.transform(X=np.reshape(copy.deepcopy(_data), (-1, 1))), (1, -1))[0],
                             force_type='continuous',
                             special_replacement=True,
                             imp_value=sys.float_info.min,
                             obj=_scaler
                             )
            del _data
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Transformed feature "{}" using standardizing'.format(feature))

    @staticmethod
    def subset(cond: str, by_partial_values: bool = False, value_type: str = 'value', safer_subset: bool = True):
        """
        Subset data set by given condition

        :param cond: str
            Query string defining subset condition like in SQL

        :param by_partial_values: bool
            Query condition by partial values

        :param value_type: str
            Partial value type:
                -> col, column, feature, var, variable, header: Select only features based on the condition containing partial value

        :param safer_subset: bool
            Apply subset using a copy of the original data set to ensure that the data set is not empty afterwards
        """
        _valid_subset: bool = True
        if by_partial_values:
            # TODO: by_partial_values -> case_name, feature_name, value based
            raise NotImplementedError('Drawing subset by partial values not implemented')
        else:
            _n_cases: int = len(DATA_PROCESSING['df'])
            if safer_subset:
                _df: dd.DataFrame = copy.deepcopy(DATA_PROCESSING.get('df'))
                _df = _df.query(expr=cond)
                if len(_df) == 0:
                    _valid_subset = False
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Applying subset condition leads to an empty data set. Therefore no action is made.')
                else:
                    DATA_PROCESSING['df'] = _df
                del _df
            else:
                DATA_PROCESSING['df'] = DATA_PROCESSING['df'].query(expr=cond)
            if _valid_subset:
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Exclude {} cases by applying following condition: {}\nData set has now {} cases'.format(_n_cases - len(DATA_PROCESSING['df']),
                                                                                                                                                                cond,
                                                                                                                                                                len(DATA_PROCESSING['df'])
                                                                                                                                                                )
                                                                   )

    @staticmethod
    def subset_features_for_modeling() -> pd.DataFrame:
        """
        Subset data set by features (like id, text or date) which cannot be used in supervised machine learning models

        :return: pd.DataFrame
            Subset data ready for modeling
        """
        _features: List[str] = []
        for feature in DATA_PROCESSING['df'].columns:
            if feature in FEATURE_TYPES.get('id_text'):
                continue
            if feature in FEATURE_TYPES.get('date'):
                continue
            if feature in FEATURE_TYPES.get('categorical'):
                if feature in DATA_PROCESSING.get('original_features'):
                    continue
            _features.append(feature)
        if len(_features) > 0:
            return DATA_PROCESSING['df'][_features]
        return DATA_PROCESSING['df']

    @staticmethod
    def stack(level: int = 0, drop_nan: bool = False) -> pd.DataFrame:
        """
        Stacking

        :param level: int
            Stacking level

        :param drop_nan: bool
            Remove cases containing missing data

        :return: pd.DataFrame
            Stacked data set
        """
        _features: List[str] = []
        for ft in FEATURE_TYPES.keys():
            for feature in FEATURE_TYPES.get(ft):
                _features.append(feature)
        if DATA_PROCESSING['target'] is not None:
            _features.append(DATA_PROCESSING.get('target'))
        _load_temp_files(features=_features)
        if level > 0 or level < -1:
            Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log(msg='Level for unstacking can either be -1 or 0')
        return DATA_PROCESSING.get('df').stack(level=level, dropna=drop_nan)

    @staticmethod
    def unify_invalid_to_mis(add_invalid: list = None):
        """
        Unify all invalid values like None, inf etc. to regular missing value

        :param add_invalid: list
            Additional values to unify
        """
        _add_invalid: list = [] if add_invalid is None else add_invalid
        INVALID_VALUES.extend(_add_invalid)
        DATA_PROCESSING['df'] = DATA_PROCESSING.get('df').replace(INVALID_VALUES, np.nan)
        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Unified all invalid values {} to missing value code "NaN"'.format(INVALID_VALUES))

    @staticmethod
    def unstack(level: int = -1, impute_value: int = None) -> pd.DataFrame:
        """
        Unstacking

        :param level: int
            Level of unstacking

        :param impute_value: int
            Value to impute missing values generated by unstacking process

        :return: pd.DataFrame
            Unstacked data set
        """
        _features: List[str] = []
        for ft in FEATURE_TYPES.keys():
            for feature in FEATURE_TYPES.get(ft):
                _features.append(feature)
        if DATA_PROCESSING['target'] is not None:
            _features.append(DATA_PROCESSING.get('target'))
        _load_temp_files(features=_features)
        if level > 0 or level < -1:
            Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log(msg='Level for unstacking can either be -1 or 0')
        return DATA_PROCESSING.get('df').unstack(level=level, fill_value=impute_value)

    @staticmethod
    def update_feature_types():
        """
        Update internal feature type dictionary
        """
        _features: List[str] = []
        for ft in FEATURE_TYPES.keys():
            for feature in FEATURE_TYPES.get(ft):
                _features.append(feature)
        _load_temp_files(features=_features)
        _set_feature_types(df=DATA_PROCESSING.get('df'), features=list(DATA_PROCESSING.get('df').columns))

    @staticmethod
    def target_type_adjustment(label_encode: bool = True):
        """
        Check type of target feature and adjust incompatible typing

        :param label_encode: bool
            Encode or decode categorical target feature
        """
        _features: List[str] = [DATA_PROCESSING.get('target')]
        _load_temp_files(features=_features)
        if str(DATA_PROCESSING['df'][DATA_PROCESSING.get('target')].dtype).find('object') >= 0:
            _unique_values: np.array = DATA_PROCESSING['df'][DATA_PROCESSING.get('target')].unique()
            if label_encode:
                _has_labels: bool = False
                for val in _unique_values:
                    if len(re.findall('[a-z A-Z]', str(val))) > 0:
                        _has_labels = True
                        break
                if _has_labels:
                    _values = {label: i for i, label in enumerate(_unique_values)}
                    _data: pd.DataFrame = DATA_PROCESSING['df'][DATA_PROCESSING.get('target')].replace(_values)
                    DATA_PROCESSING['df'][DATA_PROCESSING.get('target')] = _data
                    Log(write=not DATA_PROCESSING.get('show_msg')).log(
                        msg='Transformed feature "{}" using label encoding (label to number)'.format(DATA_PROCESSING.get('target')))
            else:
                _data: pd.DataFrame = DATA_PROCESSING['df'][DATA_PROCESSING.get('target')].replace({val: label for label, val in DATA_PROCESSING['encoder']['label'][DATA_PROCESSING.get('target')].values})
                DATA_PROCESSING['df'][DATA_PROCESSING.get('target')] = _data
                Log(write=not DATA_PROCESSING.get('show_msg')).log(
                    msg='Transformed feature "{}" using label decoding (number to original label)'.format(DATA_PROCESSING.get('target')))
        _save_temp_files(feature=DATA_PROCESSING.get('target'))

    @staticmethod
    @FeatureOrchestra(meth='text_occurances', feature_types=['id_text'])
    def text_occurances(features: List[str] = None, search_text: List[str] = None):
        """
        Find occurances in text

        :param features: List[str]
            Name of the features

        :param search_text:
            Text phrase to search in text
        """
        for feature in features:
            _load_temp_files(features=[feature])
            if feature in DATA_PROCESSING['df'].columns:
                if str(DATA_PROCESSING['df'][feature].dtype).find('object') >= 0:
                    if search_text is None:
                        TEXT_MINER['obj'].count_occurances(features=[feature],
                                                           search_text=search_text,
                                                           count_length=True,
                                                           count_numbers=True,
                                                           count_characters=True,
                                                           count_special_characters=True
                                                           )
                    else:
                        for st in search_text:
                            TEXT_MINER['obj'].count_occurances(features=[feature], search_text=search_text)
                    _occurance_features: List[str] = TEXT_MINER['obj'].generated_features[feature]['occurances']
                    _data: dd.DataFrame = TEXT_MINER['obj'].get_features(features=[feature], meth='occurances', compute=False)
                    for of in _occurance_features:
                        _process_handler(action='add',
                                         feature=feature,
                                         new_feature=of,
                                         process='text|occurances',
                                         meth='text_occurances',
                                         param=dict(search_text=search_text),
                                         data=_data[of].values.compute(),
                                         obj={feature: search_text}
                                         )
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Count occurances "{}" in feature "{}"'.format(search_text, feature))
                    del _data

    @staticmethod
    @FeatureOrchestra(meth='text_similarity', feature_types=['id_text'])
    def text_similarity(features: List[str] = None, tfidf: bool = True):
        """
        Calculate similarity of text

        :param features: List[str]
            Name of the features

        :param tfidf: bool
            Whether to calculate embeddings as similarity score or not
        """
        raise FeatureEngineerException('Text similarity not supported')

    @staticmethod
    @FeatureOrchestra(meth='to_float32', feature_types=['continuous'])
    def to_float32(features: List[str] = None):
        """
        Convert float64 types continuous features into float32 (necessary for some ml libraries like sklearn)

        :param features: List[str]
            Name of the features
        """
        for feature in features:
            _load_temp_files(features=[feature])
            _float_adjustments(features=[feature], imp_value=sys.float_info.max, convert_to_float32=True)
            _save_temp_files(feature=feature)

    @staticmethod
    def type_conversion(feature_type: Dict[str, str]):
        """
        Convert Pandas dtypes

        :param feature_type: Dict[str, str]
            Name of the feature and type for conversion
        """
        if len(feature_type.keys()) == 0:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='No type conversion configuration found')
        for feature in feature_type.keys():
            _load_temp_files(features=[feature])
            if feature not in DATA_PROCESSING.get('df').columns:
                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Feature "{}" could not be found in data set'.format(feature))
                continue
            try:
                if str(DATA_PROCESSING['df'][feature].dtype).find('object') >= 0:
                    if feature_type[feature].find('float') >= 0:
                        _conversion: str = 'continuous'
                        DATA_PROCESSING['df'][feature] = DATA_PROCESSING['df'][feature].astype(float)
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Convert type of feature "{}" from string to float'.format(feature))
                    elif feature_type[feature].find('int') >= 0:
                        if any(DATA_PROCESSING['df'][feature].str.findall(pat='[a-z,A-Z]').isnull().compute()):
                            if MissingDataAnalysis(df=DATA_PROCESSING.get('df'), features=[feature]).has_nan():
                                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Cannot convert type of feature "{}" from string to integer because it contains missing values'.format(feature))
                                continue
                            else:
                                _conversion: str = 'categorical'
                                DATA_PROCESSING['df'][feature] = DATA_PROCESSING['df'][feature].astype(dtype=feature_type[feature])
                                Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Convert type of feature "{}" from string to integer'.format(feature))
                        else:
                            _conversion: str = 'categorical'
                            if any(DATA_PROCESSING['df'][feature].isnull().compute()):
                                DATA_PROCESSING['df'][feature] = dd.from_array(x=DATA_PROCESSING['df'][feature].replace(to_replace={feature: {None: 'None'}}).values.compute())
                            DATA_PROCESSING['encoder']['label'].update({feature: EasyExploreUtils().label_encoder(values=np.reshape(DATA_PROCESSING['df'][feature].values.compute(), (-1, 1)))})
                            DATA_PROCESSING['df'][feature] = DATA_PROCESSING['df'][feature].astype(dtype=feature_type[feature])
                            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Convert type of feature "{}" from string to integer via label encoding'.format(feature))
                    elif feature_type[feature].find('date') >= 0:
                        _conversion: str = 'date'
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Use method "date_conversion" for format configuration')
                        DATA_PROCESSING['df'][feature] = dd.from_array(x=pd.to_datetime(DATA_PROCESSING['df'][feature].values.compute(), errors='coerce'))
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Convert type of feature "{}" from string to date'.format(feature))
                elif str(DATA_PROCESSING['df'][feature].dtype).find('date') >= 0:
                    if feature_type[feature].find('int') >= 0:
                        if MissingDataAnalysis(df=DATA_PROCESSING.get('df'), features=[feature]).has_nan():
                            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Cannot convert type of feature "{}" from date to integer because it contains missing values'.format(feature))
                            continue
                        else:
                            _conversion: str = 'categorical'
                            DATA_PROCESSING['df'][feature] = DATA_PROCESSING['df'][feature].astype(dtype=feature_type[feature])
                            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Convert type of feature "{}" from date to integer'.format(feature))
                    elif feature_type[feature].find('float') >= 0:
                        _conversion: str = 'continuous'
                        DATA_PROCESSING['df'][feature] = DATA_PROCESSING['df'][feature].astype(dtype=float)
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Convert type of feature {} from date to float'.format(feature))
                    elif feature_type[feature].find('str') >= 0:
                        _conversion: str = 'id_text'
                        DATA_PROCESSING['df'][feature] = DATA_PROCESSING['df'][feature].astype(dtype=str)
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Convert type of feature {} from date to string'.format(feature))
                elif str(DATA_PROCESSING['df'][feature].dtype).find('float') >= 0:
                    if feature_type[feature].find('int') >= 0:
                        _conversion: str = 'categorical'
                        if MissingDataAnalysis(df=DATA_PROCESSING.get('df'), features=[feature]).has_nan():
                            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Cannot convert type of feature "{}" from date to integer because it contains missing values'.format(feature))
                            continue
                        else:
                            DATA_PROCESSING['df'][feature] = DATA_PROCESSING['df'][feature].astype(dtype=feature_type[feature])
                            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Convert type of feature "{}" from float to integer'.format(feature))
                    elif feature_type[feature].find('date') >= 0:
                        _conversion: str = 'date'
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Use method "date_conversion" for format configuration')
                        DATA_PROCESSING['df'][feature] = dd.from_array(x=pd.to_datetime(DATA_PROCESSING['df'][feature].values.compute(), errors='coerce'))
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Convert type of feature {} from float to date'.format(feature))
                    elif feature_type[feature].find('str') >= 0:
                        _conversion: str = 'id_text'
                        DATA_PROCESSING['df'][feature] = DATA_PROCESSING['df'][feature].astype(dtype=feature_type[feature])
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Convert type of feature {} from float to string'.format(feature))
                else:
                    if feature_type[feature].find('date') >= 0:
                        _conversion: str = 'date'
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Use method "date_conversion" for format configuration')
                        DATA_PROCESSING['df'][feature] = dd.from_array(x=pd.to_datetime(DATA_PROCESSING['df'][feature].values.compute(), errors='coerce'))
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Convert type of feature {} from float to date'.format(feature))
                    else:
                        _conversion: str = feature_type[feature].replace('float', 'continuous').replace('int', 'categorical')
                        DATA_PROCESSING['df'][feature] = DATA_PROCESSING['df'][feature].astype(dtype=feature_type[feature])
                        Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Convert type of feature "{}" from {} to {}'.format(feature,
                                                                                                                                   str(DATA_PROCESSING['df'][feature].dtype),
                                                                                                                                   feature_type.get(feature)
                                                                                                                                   )
                                                                           )
                _save_temp_files(feature=feature)
            except (AttributeError, ValueError, TypeError) as e:
                Log(write=not DATA_PROCESSING.get('show_msg'), level='error').log(msg='Feature "{}" of type {} could not be converted to {}\nError: {}'.format(feature,
                                                                                                                                                               str(DATA_PROCESSING['df'][feature].dtype),
                                                                                                                                                               feature_type[feature],
                                                                                                                                                               e
                                                                                                                                                               )
                                                                                  )
            #DATA_PROCESSING['processing']['features'].update({'P{}'.format(
            #    len(DATA_PROCESSING['processing']['features'].keys()) + 1): dict(feature=feature,
            #                                                                     new_feature=feature,
            #                                                                     process='typing|{}'.format(_conversion)
            #                                                                     )
            #                                                  })
            if feature in DATA_PROCESSING['df'].columns:
                if feature_type.get(feature) == 'float':
                    _force_type: str = 'continuous'
                elif feature_type.get(feature) == 'int':
                    _force_type: str = 'categorical'
                elif feature_type.get(feature) == 'datetime':
                    _force_type: str = 'date'
                elif feature_type.get(feature) == 'str':
                    _force_type: str = 'id_text'
                else:
                    _force_type = None
                _update_feature_types(feature=feature, force_type=_force_type)

    @staticmethod
    def write_notepad(note: str, page: str = None, append: bool = True, add_time_stamp: bool = True):
        """
        Write notice to internal notepad dictionary

        :param note: str
            Note to write

        :param page: str
            Name of the page to write to

        :param append: bool
            Whether to append note to the already written text on the page or overwrite text on the page

        :param add_time_stamp: bool
            Whether to add current time stamp to note or not
        """
        if len(note) > 0:
            _time_stamp: str = '\n\nWritten: {}'.format(str(datetime.now())) if add_time_stamp else ''
            if page is None:
                if 'temp' not in NOTEPAD.keys():
                    NOTEPAD.update({'temp': ''})
                if append:
                    NOTEPAD['temp'] = '{}\n\n{}{}'.format(NOTEPAD.get('temp'), note, _time_stamp)
                else:
                    NOTEPAD['temp'] = '{}{}'.format(note, _time_stamp)
            else:
                if page in NOTEPAD.keys():
                    if append:
                        NOTEPAD.update({page: '{}\n\n{}{}'.format(NOTEPAD.get(page), note, _time_stamp)})
                    else:
                        NOTEPAD.update({page: '{}{}'.format(note, _time_stamp)})
                else:
                    NOTEPAD.update({page: '{}{}'.format(note, _time_stamp)})
        else:
            Log(write=not DATA_PROCESSING.get('show_msg')).log(msg='Note is empty')
