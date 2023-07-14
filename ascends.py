# coding: utf-8
from __future__ import print_function
# In[1]:
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")
import os
import numpy as np
import tensorflow as tf
import random as rn
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras import regularizers
from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
from pandas.plotting import scatter_matrix
from pprint import pprint
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import StackingClassifier
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import configparser
import csv
import datetime
import glob
import keras
import math
import matplotlib
#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import os
import pandas as pd
import pickle
import random
import sys
import time
import traceback
from minepy import MINE
from sklearn.linear_model import LogisticRegression
from os import path
from pathlib import PurePath
from tensorflow.python.util import deprecation
import multiprocessing
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

deprecation._PRINT_DEPRECATION_WARNINGS = False

config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
#config.gpu_options.allow_growth = True #allocate dynamically

#config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU':4 } ) 
sess = tf.compat.v1.Session(config = config)
tf.compat.v1.keras.backend.set_session(sess)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'

def save_test_data(predictions, actual_values, filename) -> None:
    """save predictions and actual values to a csv"""
    df_predictions = pd.DataFrame(predictions, columns = ['predictions'])
    df_actual_values = pd.DataFrame(actual_values, columns = ['actual_values'])
    df = df_predictions.join(df_actual_values)
    df.to_csv(filename)
    
# In[3]:

def load_header(csv_file, print_out = False):
    """returns the headers, a dict from the indices to the headers, and vice-versa\n
    if print_out, then print dict from indices to headers"""
    # Loading headers
    
    headers = np.array(pd.read_csv(csv_file, nrows = 0).columns)
    idx_to_key = {}
    key_to_idx = {}
    for i in range(0, len(headers)):
        idx = i
        key = headers[i]
        key_to_idx[key] = idx
        idx_to_key[idx] = key
    
    if print_out is True:
        print(idx_to_key)
        
    return headers, idx_to_key, key_to_idx

def model_name(model_abbr) -> str:
    """returns the full model name for the abbreviation\n
    returns None if the abbreviation is unknown"""
    abbr2name = {"RF": "Random Forest", "NET": "Neural Network", "LR": "Linear Regression", 
                 "LRC": "Logistic Regression", "RG": "Ridge", "KR": "Kernel Ridge", 
                 "BR": "Bayesian Ridge", "SVM": "Support Vector Machine", "NN": "k-Nearest Neighbor",
                 "XGB": "Extreme Gradient Boosting"}
    if model_abbr not in abbr2name:
        print("Error: unknown model abbreviation ", model_abbr)
    return abbr2name[model_abbr]

# In[4]:

def data_load_shuffle(csv_file: PurePath, train_cols: list, cols_to_remove: list, target_col: str, random_state = 0, delimiter = ',', map_all: dict = None, ordinal_cols: list = None):
    """read csv, isolate x, y, and split train and test data"""
    if train_cols is not None and cols_to_remove is not None:
        if len(train_cols) != 0 and len(cols_to_remove) != 0:
            print("ERROR: train_cols and cols_to_remove are mutually exclusive")
    data = pd.read_csv(csv_file, delimiter = delimiter)
    data = data[data[target_col].notnull()] # TODO: see if this needs to be a separate Dataframe
    if train_cols is not None:
        data = data[train_cols+[target_col]]
    if cols_to_remove is not None:
        if target_col in cols_to_remove:
            print("ERROR: tried to remove target_col")
        for col in cols_to_remove:
            del data[col]
    data = data.dropna(how='any',axis=0) # TODO: find out if we want to remove all rows with ANY NULL values
    # User defined mapping
    if map_all is not None:
        for col in map_all:
            if col in data.columns:
                col_map = map_all[col]
                data[col] = data[col].apply(lambda x: col_map[x])
    # User defined ordinal encoding
    if ordinal_cols is not None:
        for col in ordinal_cols:
            if data[col].dtype.name != 'object':
                print("ERROR:", col, "is not categorical")
                continue
            col_dict = {}
            for element in data[col]:
                if element not in col_dict:
                    col_dict[element] = len(col_dict)
            data[col] = data[col].map(col_dict)
    # Automatic ordinal encoding for target col
    if data[target_col].dtype.name == 'object':
        col_dict = {}
        for element in data[target_col]:
            if element not in col_dict:
                col_dict[element] = len(col_dict)
        data[target_col] = data[target_col].map(col_dict)
    # Automatic one-hot encoding on training cols
    for col in data.columns:
        if col == target_col: # this does nothing, but maybe keep in case code is reordered
            continue
        if data[col].dtype.name == 'object':
            one_hot = pd.get_dummies(data[col])
            data = data.drop(col, axis = 1)
            data = data.join(one_hot)
    y_header = target_col
    y = data[y_header]
    #data = data.drop(y_header, axis = 1)
    x_headers = list(data.columns)
    x_headers.remove(target_col)
    x = data[x_headers]
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    # TODO: decide whether we want to split
    # PROS: greater testing accuracy (less bias), good for comparing models
    # CONS: lower prediction accuracy
    # Ideally we let the user choose, split when deciding, and once the decision is made don't split for maximum prediction accuracy
    x_train = pd.concat([x_train, x_test])
    y_train = pd.concat([y_train, y_test])
    return data, x_train, x_test, y_train, y_test, x_headers, y_header

def correlation_analysis_all(data_df, target_col, top_k=10, file_to_save = None, save_chart = None, only_pcc= False, feature_selection_file = None):
    
    if feature_selection_file == None:
        print("* correlation_analysis_all")
        pcc = data_df.corr()[target_col]

        if(len(pcc)<top_k):
            top_k=len(pcc)
        print("Computing PCC, PCC_SQRT ..")
        pcc = pcc.sort_values(ascending = False).dropna()
        pcc = pcc.rename("PCC")
        try:
            del pcc[target_col]
        except:
            pass
        pcc_sqrt = pcc.apply(lambda x: np.sqrt(x* x))
        pcc_sqrt = pcc_sqrt.sort_values(ascending = False).dropna()
        pcc_sqrt = pcc_sqrt.rename("PCC_SQRT")
        MICs = []
        MASs = []
        MEVs = []
        MCNs = []
        MCN_generals = []
        GMICs = []
        TICs = []
        print("Computing all other metrics ..")
        if only_pcc==False or only_pcc=='False':
            for col in data_df.columns:
                print(" - computing for ", col, "...")
                if col!=target_col:
                    x = data_df[col].values
                    y = data_df[target_col].values
                    mine = MINE()
                    mine.compute_score(x,y)
                    MICs.append((col,mine.mic()))
                    MASs.append((col,mine.mas()))
                    MEVs.append((col,mine.mev()))
                    MCNs.append((col,mine.mcn(0)))
                    MCN_generals.append((col,mine.mcn_general()))
                    GMICs.append((col,mine.gmic()))
                    TICs.append((col,mine.tic())) 
                
        top_k_pcc = list(pcc.keys())[:top_k]
        top_k_pcc_sqrt = list(pcc_sqrt.keys())[:top_k]
        top_k_mic = [tup[0] for tup in sorted(MICs, key=lambda tup: tup[1], reverse = True)[:top_k]]
        top_k_mas = [tup[0] for tup in sorted(MASs, key=lambda tup: tup[1], reverse = True)[:top_k]]
        top_k_mev = [tup[0] for tup in sorted(MEVs, key=lambda tup: tup[1], reverse = True)[:top_k]]
        top_k_mcn = [tup[0] for tup in sorted(MCNs, key=lambda tup: tup[1], reverse = True)[:top_k]]
        top_k_mcn_general = [tup[0] for tup in sorted(MCN_generals, key=lambda tup: tup[1], reverse = True)[:top_k]]
        top_k_gmic = [tup[0] for tup in sorted(GMICs, key=lambda tup: tup[1], reverse = True)[:top_k]]
        top_k_tic = [tup[0] for tup in sorted(TICs, key=lambda tup: tup[1], reverse = True)[:top_k]]
        
        mic_df = pd.DataFrame([tup[1] for tup in MICs],columns=['MIC'],index=[tup[0] for tup in MICs])
        mas_df = pd.DataFrame([tup[1] for tup in MASs],columns=['MAS'],index=[tup[0] for tup in MASs])
        mev_df = pd.DataFrame([tup[1] for tup in MEVs],columns=['MEV'],index=[tup[0] for tup in MEVs])
        mcn_df = pd.DataFrame([tup[1] for tup in MCNs],columns=['MCN'],index=[tup[0] for tup in MCNs])
        mcn_general_df = pd.DataFrame([tup[1] for tup in MCN_generals],columns=['MCN_general'],index=[tup[0] for tup in MCN_generals])
        gmic_df = pd.DataFrame([tup[1] for tup in GMICs],columns=['GMIC'],index=[tup[0] for tup in GMICs])
        tic_df = pd.DataFrame([tup[1] for tup in TICs],columns=['TIC'],index=[tup[0] for tup in TICs])       
        
        if only_pcc==False or only_pcc=='False':
            final_report = mic_df.join(mas_df).join(mev_df).join(mcn_df).join(mcn_general_df).join(gmic_df).join(tic_df).sort_index().join(pcc_sqrt).join(pcc)
        else:
            pcc_sqrt = pd.DataFrame(pcc_sqrt)
            final_report = pcc_sqrt.join(pcc)
        
        if file_to_save is not None:
            # save to correlation report
            final_report.to_csv(file_to_save)

        if save_chart is not None:
            for col in final_report.keys():
                ax = final_report[col].sort_values(ascending=False).plot(kind='bar',alpha=0.8)
                ax.set_ylabel(col+" (target_col = '"+target_col+"')", fontsize=12)
                plt.axhline(0, color='k')
                plt.savefig(save_chart)
                plt.close()

        fs_dict = {'PCC':top_k_pcc,'PCC_SQRT':top_k_pcc_sqrt,'MIC':top_k_mic,'MAS':top_k_mas,'MEV':top_k_mev,'MCN':top_k_mcn,'MCN_general':top_k_mcn_general,'GMIC':top_k_gmic,'TIC':top_k_tic}
    
    else:
        final_report = pd.read_csv(feature_selection_file)
        top_k_pcc = list(final_report.sort_values(by=['PCC'], ascending=False).T.values[0])[:top_k]
        top_k_pcc_sqrt = list(final_report.sort_values(by=['PCC_SQRT'], ascending=False).T.values[0])[:top_k]
        top_k_mic = list(final_report.sort_values(by=['MIC'], ascending=False).T.values[0])[:top_k]
        top_k_mas = list(final_report.sort_values(by=['MAS'], ascending=False).T.values[0])[:top_k]
        top_k_mev = list(final_report.sort_values(by=['MEV'], ascending=False).T.values[0])[:top_k]
        top_k_mcn = list(final_report.sort_values(by=['MCN'], ascending=False).T.values[0])[:top_k]
        top_k_mcn_general = list(final_report.sort_values(by=['MCN_general'], ascending=False).T.values[0])[:top_k]
        top_k_gmic = list(final_report.sort_values(by=['GMIC'], ascending=False).T.values[0])[:top_k]
        top_k_tic = list(final_report.sort_values(by=['TIC'], ascending=False).T.values[0])[:top_k]
        fs_dict = {'PCC':top_k_pcc,'PCC_SQRT':top_k_pcc_sqrt,'MIC':top_k_mic,'MAS':top_k_mas,'MEV':top_k_mev,'MCN':top_k_mcn,'MCN_general':top_k_mcn_general,'GMIC':top_k_gmic,'TIC':top_k_tic}
        
        if file_to_save is not None:
            # save to correlation report
            final_report.to_csv(file_to_save)

        if save_chart is not None:
            for col in final_report.keys()[1:]:
                ax = final_report[col].sort_values(ascending=False).plot(kind='bar',alpha=0.8)
                ax.set_ylabel(col+" (target_col = '"+target_col+"')", fontsize=12)
                plt.axhline(0, color='k')
                plt.savefig(save_chart)
                plt.close()

    return fs_dict, final_report

# In[6]:


def default_model_parameters_classifier() -> dict[str, str]:
    """returns default model params for classification"""
    model_parameters = {
    'scaler_option':'StandardScaler', \
    'rf_n_estimators': '100', 'rf_max_features': '1.0', 'rf_max_depth': 'None', \
    'rf_min_samples_split': '2', 'rf_min_samples_leaf': '1', 'rf_bootstrap': 'True', \
    'rf_criterion':'gini','rf_min_weight_fraction_leaf':'0.','rf_max_leaf_nodes':'None',\
    'rf_min_impurity_decrease':'0.',\
    'nn_n_neighbors': '5', 'nn_weights': 'uniform', 'nn_algorithm': 'auto', 'nn_leaf_size': '30', 'nn_p': '2',\
    'nn_metric':'minkowski','nn_metric_params':'None',
    'rg_alpha':'1','rg_fit_intercept':'True','rg_max_iter':'None','rg_tol':'0.001','rg_class_weight':'None','rg_solver':'auto','svm_kernel': 'rbf', \
    'svm_degree': '3', 'svm_coef0': '0.0', 'svm_tol': '1e-3', 'svm_c': '1.0', \
    'svm_gamma': 'auto', \
    'svm_decision_function_shape':'ovr', \
    
    'net_structure':'16 16 16',\
    'net_layer_n':'3',\
    'net_dropout': '0.0',\
    'net_l_2': '0.01',\
    'net_learning_rate': '0.01',\
    'net_epochs': '100',\
    'net_batch_size': '2',\

    }
    """
    
    n_estimators
    max_leaves
    max_bin
    grow_policy
    objective
    sampling_method """

    return model_parameters 

def default_model_parameters() -> dict[str, str]:
    """returns default model params for regression"""
    model_parameters = {
    'scaler_option':'StandardScaler', \
    'rf_n_estimators': '100', 'rf_max_features': '1.0', 'rf_max_depth': 'None', \
    'rf_min_samples_split': '2', 'rf_min_samples_leaf': '1', 'rf_bootstrap': 'True', \
    'rf_criterion':'friedman_mse','rf_min_weight_fraction_leaf':'0.','rf_max_leaf_nodes':'None',\
    'rf_min_impurity_decrease':'0.',\
    'nn_n_neighbors': '5', 'nn_weights': 'uniform', 'nn_algorithm': 'auto', 'nn_leaf_size': '30', 'nn_p': '2',\
    'nn_metric':'minkowski','nn_metric_params':'None',\
   
    'kr_alpha': '1', 'kr_kernel': 'linear', 'kr_gamma': 'None', 'kr_degree': '3', 'kr_coef0': '1', \
    
    'br_n_iter': '300', 'br_alpha_1': '1.2e-6', 'br_alpha_2': '1.e-6', 'br_tol': '1.e-3', \
    'br_lambda_1': '1.e-6', 'br_lambda_2': '1.e-6', 'br_compute_score': 'False', 'br_fit_intercept': 'True',\
    
    'svm_kernel': 'rbf', \
    'svm_degree': '3', 'svm_coef0': '0.0', 'svm_tol': '1e-3', 'svm_c': '1.0', \
    'svm_epsilon': '0.1', 'svm_shrinking': 'True', 'svm_gamma': 'auto', \
    'xgb_n_estimators': '100', 'xgb_min_child_weight': '1', 'xgb_max_depth': '6', \
    'xgb_learning_rate': '0.3', 'xgb_gamma': '0.0', \
    
    'net_structure':'16 16 16',\
    'net_layer_n':'3',\
    'net_dropout': '0.0',\
    'net_l_2': '0.01',\
    'net_learning_rate': '0.01',\
    'net_epochs': '100',\
    'net_batch_size': '2',\

    }

    return model_parameters 

def load_model_parameter_from_file(filename) -> dict:

    config = configparser.RawConfigParser()
    config.read(filename)
    model_parameters = {}
    for key in config['HYPERPARAMETERS']:  
        model_parameters[key] = config["HYPERPARAMETERS"][key]
    
    return model_parameters

# In[7]:

def fix_value(val, val_type, value_error_ok = False):
    """convert val into a val_type object"""
    if val is None or val=='None':
        return None
    if val=='auto':
        return val
    try:
        if(val_type=='float'):
            return float(val)
        if (val_type=='str'):
            return str(val)
        if (val_type=='int'):
            return int(val)
        if (val_type=='bool'):
            return str2bool(val)
        if val_type=='PurePath':
            return PurePath(val)
        else:
            return val
    except ValueError:
        if value_error_ok:
            return val

def define_model_classifier(model_type, model_parameters, x_header_size, random_state = None) -> Pipeline:
    """initialize and return classification Pipeline object using model_type and parameters"""
    if model_type == "LRC":
        model = Pipeline([
          ('classification', LogisticRegression())
        ])
    elif model_type == "RF":
        max_features = fix_value(model_parameters['rf_max_features'],'float', True)
        if max_features > 1.0:
            max_features = int(max_features)
        estimators = [
            ('classification', RandomForestClassifier(n_estimators = int(model_parameters['rf_n_estimators']), 
                                                      max_features = max_features,
                                                      max_depth = fix_value(model_parameters['rf_max_depth'],'int'), 
                                                      min_samples_split = int(model_parameters['rf_min_samples_split']), 
                                                      min_samples_leaf = int(model_parameters['rf_min_samples_leaf']), 
                                                      bootstrap = str2bool(model_parameters['rf_bootstrap']), 
                                                      criterion = model_parameters['rf_criterion'], 
                                                      random_state = random_state,
                                                      min_weight_fraction_leaf = float(model_parameters['rf_min_weight_fraction_leaf']), 
                                                      max_leaf_nodes = fix_value(model_parameters['rf_max_leaf_nodes'],'int'),
                                                      min_impurity_decrease = float(model_parameters['rf_min_impurity_decrease'])))
            #('xgboost', XGBClassifier())
        ]

        # Create the StackingRegressor with the base estimators
        stacking_classifier = StackingClassifier(
            estimators=estimators
        )
        model = make_pipeline(stacking_classifier)




    elif model_type == "NN":
        model = Pipeline([
            ('classification', KNeighborsClassifier(n_neighbors = int(model_parameters['nn_n_neighbors']),
                                                    weights = model_parameters['nn_weights'],
                                                    algorithm = model_parameters['nn_algorithm'],
                                                    leaf_size = int(model_parameters['nn_leaf_size']),
                                                    metric = model_parameters['nn_metric'],
                                                    metric_params = fix_value(model_parameters['nn_metric_params'],'str'),
                                                    p = int(model_parameters['nn_p'])))
        ])

    elif model_type == "RG":
        model = Pipeline([
            ('classification', RidgeClassifier(alpha = float(model_parameters['rg_alpha']),
                                            fit_intercept = fix_value(model_parameters['rg_fit_intercept'],'bool'),
                                            max_iter = fix_value(model_parameters['rg_max_iter'],'int'),
                                            tol = float(model_parameters['rg_tol']),
                                            class_weight = fix_value(model_parameters['rg_class_weight'],'str'),
                                            solver = fix_value(model_parameters['rg_solver'],'str')))                                                                   
        ])
    elif model_type == "SVM":
        model = Pipeline([
            ('classification', svm.SVC(kernel = model_parameters['svm_kernel'],
                                     degree = int(model_parameters['svm_degree']),
                                     coef0 = float(model_parameters['svm_coef0']),
                                     tol = float(model_parameters['svm_tol']),
                                     C = float(model_parameters['svm_c']),
                                     gamma = fix_value(model_parameters['svm_gamma'],'float'),
                                     decision_function_shape = model_parameters['svm_decision_function_shape']))
        ])
    else:
        print("Error: idkrn")
        
    return model # TODO: find out if model can ever be undefined here

def define_model_regression(model_type, model_parameters, x_header_size, random_state = None) -> Pipeline:
    """initialize and return regression Pipeline object using model_type and parameters"""
    if model_type == "LR":
        model = Pipeline([
            ('regression', LinearRegression())
        ])
    elif model_type == "RF":
        asdf = model_parameters['rf_max_features']
        max_features = fix_value(model_parameters['rf_max_features'],'float', True)
        if max_features > 1.0:
            max_features = int(max_features)
        estimators = [
            ('random_forest', RandomForestRegressor(n_estimators = int(model_parameters['rf_n_estimators']), 
                                                   max_features = max_features,
                                                   max_depth = fix_value(model_parameters['rf_max_depth'],'int'), 
                                                   min_samples_split = int(model_parameters['rf_min_samples_split']), 
                                                   min_samples_leaf = int(model_parameters['rf_min_samples_leaf']), 
                                                   bootstrap = str2bool(model_parameters['rf_bootstrap']), 
                                                   criterion = model_parameters['rf_criterion'], random_state = random_state,
                                                   min_weight_fraction_leaf = float(model_parameters['rf_min_weight_fraction_leaf']), 
                                                   max_leaf_nodes = fix_value(model_parameters['rf_max_leaf_nodes'],'int'),
                                                   min_impurity_decrease = float(model_parameters['rf_min_impurity_decrease'])))
            #('xgboost', XGBRegressor())
        ]

        # Create the StackingRegressor with the base estimators
        stacking_regressor = StackingRegressor(
            estimators=estimators
        )
        model = make_pipeline(stacking_regressor)

        """
        model = Pipeline([
            ('regression', RandomForestRegressor(n_estimators = int(model_parameters['rf_n_estimators']), 
                                                   max_features = fix_value(model_parameters['rf_max_features'],'float', True), 
                                                   max_depth = fix_value(model_parameters['rf_max_depth'],'int'), 
                                                   min_samples_split = int(model_parameters['rf_min_samples_split']), 
                                                   min_samples_leaf = int(model_parameters['rf_min_samples_leaf']), 
                                                   bootstrap = str2bool(model_parameters['rf_bootstrap']), 
                                                   criterion = model_parameters['rf_criterion'], random_state = random_state,
                                                   min_weight_fraction_leaf = float(model_parameters['rf_min_weight_fraction_leaf']), 
                                                   max_leaf_nodes = fix_value(model_parameters['rf_max_leaf_nodes'],'int'),
                                                   min_impurity_decrease = float(model_parameters['rf_min_impurity_decrease'])))
        ])
        model.steps.append(['xgboost', XGBRegressor()])
        """
    elif model_type == "NN":
        model = Pipeline([
            ('regression', KNeighborsRegressor(n_neighbors = int(model_parameters['nn_n_neighbors']),
                                                 weights = model_parameters['nn_weights'],
                                                 algorithm = model_parameters['nn_algorithm'],
                                                 leaf_size = int(model_parameters['nn_leaf_size']),
                                                 metric = model_parameters['nn_metric'],
                                                 metric_params = fix_value(model_parameters['nn_metric_params'],'str'),
                                                 p = int(model_parameters['nn_p'])))
        ])

    elif model_type == "BR":
        model = Pipeline([
            ('regression', linear_model.BayesianRidge(n_iter = int(model_parameters['br_n_iter']),
                                                        alpha_1 = float(model_parameters['br_alpha_1']),
                                                        alpha_2 = float(model_parameters['br_alpha_2']),
                                                        tol = float(model_parameters['br_tol']),
                                                        lambda_1 = float(model_parameters['br_lambda_1']),
                                                        lambda_2 = float(model_parameters['br_lambda_2']),
                                                        compute_score = fix_value(model_parameters['br_compute_score'],'bool'),
                                                        fit_intercept = fix_value(model_parameters['br_fit_intercept'],'bool')))
        ])
    elif model_type == "SVM":
        model = Pipeline([
            ('regression', svm.SVR(kernel = model_parameters['svm_kernel'],
                                     degree = int(model_parameters['svm_degree']),
                                     coef0 = float(model_parameters['svm_coef0']),
                                     tol = float(model_parameters['svm_tol']),
                                     C = float(model_parameters['svm_c']),
                                     gamma = fix_value(model_parameters['svm_gamma'],'float'),
                                     epsilon = float(model_parameters['svm_epsilon'])))
        ])
        # 'kr_alpha': '1', 'kr_kernel': 'linear', 'kr_gamma': 'None', 'kr_degree': '3', 'kr_coef0': '1', \
    elif model_type == "KR":
        # TODO: From empirical observation, KR seems to be broken
        model = Pipeline([
            ('regression', KernelRidge(alpha = int(model_parameters['kr_alpha']),
                                        kernel = fix_value(model_parameters['kr_kernel'],'str'),
                                        gamma = fix_value(model_parameters['kr_gamma'],'str'),
                                        degree = int(model_parameters['kr_degree']),
                                        coef0 = int(model_parameters['kr_coef0'])))
        ])
    elif model_type == "XGB":
        model = Pipeline([
            ('regression', XGBRegressor(n_estimators = int(model_parameters['xgb_n_estimators']),
                                        min_child_weight = int(model_parameters['xgb_min_child_weight']),
                                        max_depth = int(model_parameters['xgb_max_depth']),
                                        learning_rate = fix_value(model_parameters['xgb_learning_rate'], float),
                                        gamma = fix_value(model_parameters['xgb_gamma'], float)))
        ])
    else:
        print("Error: idkrn also")
        
    return model


# In[8]:

def rescale_x(scaler_option, x_train):
    scale = None
    if scaler_option=='False':
        x_train_ = x_train
    elif scaler_option == "MinMaxScaler":
        scale = preprocessing.MinMaxScaler()
        x_train_ = scale.fit_transform(x_train)
    elif scaler_option == "MaxAbsScaler":
        scale = preprocessing.MaxAbsScaler()
        x_train_ = scale.fit_transform(x_train)
    elif scaler_option == "RobustScaler":
        scale = preprocessing.RobustScaler()
        x_train_ = scale.fit_transform(x_train)
    elif scaler_option == "QuantileTransformer":
        scale = preprocessing.QuantileTransformer()
        x_train_ = scale.fit_transform(x_train)
    elif scaler_option == "Normalizer":
        scale = preprocessing.Normalizer()
        x_train_ = scale.fit_transform(x_train)
    else:
        scale = preprocessing.StandardScaler()
        x_train_ = scale.fit_transform(x_train)
    return x_train_, scale

# In[9]:

def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if not isinstance(v, str):
        print("NOT A STRING")
    return v.lower() in ("yes", "true", "t", "1")


def cross_val_predict_net_classifier(model, x_train, y_train, epochs=1000, batch_size=8, verbose = 0, scaler_option='StandardScaler', num_of_folds = 5, num_of_class = 2, force_to_proceed = False, accuracy_threshold = 0.5, fast_tune = True):
    
    x_cvtrains, y_cvtrains, x_cvtests, y_cvtests = cv_resample(x_train, y_train, num_of_folds=num_of_folds) # TODO: find out if this is necessary or whether cross_val_score is more appropriate
    
    predictions_total = []
    actual_values_total = []

    for j in range(0, num_of_folds):
        print(" Evaluating fold(%d) ..."%(j))
        start_time = time.time()
        
        x_train_, scale = rescale_x(scaler_option, x_cvtrains[j]) 
        
        # This is the change
        if scale is not None:
            x_test_ = scale.transform(x_cvtests[j])
        else:
            x_test_ = x_cvtests[j]

        dummy_y = keras.utils.to_categorical(y_cvtrains[j], num_classes=num_of_class, dtype='float32')
        #print("FITTING")
        history = model.fit(x_train_, dummy_y,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose)
        
        predictions  = model.predict_classes(x_test_)
        actual_values = y_cvtests[j]
        actual_values = actual_values.reshape(actual_values.shape[0],)
        
        accuracy= evaluate_classifier(predictions, actual_values)
        
        if force_to_proceed == False:
            if accuracy<accuracy_threshold:
                return [],[]

        print(" accuracy = %8.3f "%(accuracy))

        predictions_total+=list(predictions)
        actual_values_total+=list(actual_values)
        
        if fast_tune==True:
            print("* Fast tuning enabled, so we only test 1 fold and move on ...")
            break

    return np.array(predictions_total), np.array(actual_values_total)


def cross_val_predict_net(model, x_train, y_train, epochs=1000, batch_size=8, verbose = 0, scaler_option='StandardScaler', num_of_folds = 5, force_to_proceed= False, fast_tune=True):
    
    x_cvtrains, y_cvtrains, x_cvtests, y_cvtests = cv_resample(x_train, y_train, num_of_folds=num_of_folds)
    
    predictions_total = []
    actual_values_total = []

    for j in range(0, num_of_folds):
        print(" Evaluating fold(%d) ..."%(j))
        start_time = time.time()
        
        x_train_, scale = rescale_x(scaler_option, x_cvtrains[j]) 

        # This is the change
        if scale is not None:
            x_test_ = scale.transform(x_cvtests[j])
        else:
            x_test_ = x_cvtests[j]
        #print("FITTING")
        history = model.fit(x_train_, y_cvtrains[j],
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose)

        predictions  = model.predict(x_test_)
        actual_values = y_cvtests[j]
        
        MAE, R2 = evaluate(predictions, actual_values)
        print(" MAE = %8.3f R2 = %8.3f ..."%(MAE, R2))

        if force_to_proceed == False:
            if R2<0:
                return [],[]

        predictions_total+=list(predictions)
        actual_values_total+=list(actual_values)
        if fast_tune==True:
            print("* Fast tuning enabled. so we only test 1 fold, and move on ..")
            break

    return np.array(predictions_total), np.array(actual_values_total)

def save_parameters(model_parameters, filename) -> None:
    # TODO: find out if these are parameters or hyperparameters
    f = open(filename,'w')
    f.write("[HYPERPARAMETERS]\n\n")
    for key in model_parameters.keys():
        f.write(str(key)+" = "+str(model_parameters[key])+"\n")
    f.close()

def save_args(model_args,filename) -> None:
    f = open(filename,'w')
    f.write("[ARGUMENTS]\n\n")
    for key in model_args.keys():
        f.write(str(key)+" = "+str(model_args[key])+"\n")
    f.close()

def save_metadata(model_args, model_stats, project_folder) -> None:
    """save this session to metadata.csv\n
    if metadata.csv doesn't exist, create it first"""
    if path.exists(project_folder / "metadata.csv"):
        try:
            meta=pd.read_csv(project_folder / "metadata.csv",index_col=0)
        except pd.errors.EmptyDataError:
            meta=pd.DataFrame(columns=["session","model_args","model_stats"])
    else:
        meta=pd.DataFrame(columns=["session","model_args","model_stats"])
    session_number = get_session(project_folder)
    meta.loc[len(meta.index)] = [session_number,model_args,model_stats] 
    meta.to_csv(project_folder / "metadata.csv")
        

def train_and_predict(model, x_train, y_train, scaler_option, num_of_folds=5):
    
    x_train_, scale = rescale_x(scaler_option, x_train)
    #y_train_ = y_train.reshape(y_train.shape[0],)
    y_train_ = pd.Series(y_train)
    predictions = cross_val_predict(model, x_train_, y_train_, cv=num_of_folds)
    actual_values = np.array(y_train_)
    
    return predictions, actual_values

def get_session(project_folder) -> int:
    """returns session number (1-index), if there is no metadata.csv file return 1"""
    if not path.exists(project_folder / "metadata.csv"):
        return 1
    try:
        meta=pd.read_csv(project_folder / "metadata.csv")
        return len(meta.index) + 1
    except pd.errors.EmptyDataError:
        return 1
    # TODO: error message here for any potential other exceptions (EmptyDataError is normal and isn't bad)

# In[10]:

def train_and_save_net_classifier(model, tag, input_cols, target_col, x_train, y_train, scaler_option, accuracy=None, path_to_save = '.', num_of_folds=5, epochs=100, batch_size=2, num_of_class = 2) -> None:
        
    if accuracy is None:
        
        print('* Model has not been evaluated. Evaluation initiated via %d-fold cross validation'%(num_of_folds))
        
        predictions, actual_values = cross_val_predict_net_classifier(model, epochs=epochs, batch_size=batch_size, x_train = x_train, y_train = y_train, verbose = 0, scaler_option = scaler_option, num_of_folds = num_of_folds, num_of_class = num_of_class, fast_tune = False)
        accuracy = evaluate_classifier(predictions, actual_values)

    x_train_, scale = rescale_x(scaler_option, x_train)
    
    dummy_y = keras.utils.to_categorical(y_train, num_classes=num_of_class, dtype='float32')

    print('* Training initiated ..')
    #print("FITTING")
    model.fit(x_train_, dummy_y, epochs=epochs, batch_size=batch_size)
    print('* Training done.')
    
    model_dict = {}
    model_dict['tag'] = tag
    model_dict['model'] = model
    model_dict['model_abbr'] = 'NET'
    model_dict['input_cols'] = input_cols
    model_dict['target_col'] = target_col
    model_dict['accuracy'] = accuracy
    model_dict['fitted_scaler_x'] = scale
    output_file = PurePath(path_to_save) / (tag)
    #print(model_dict)
    
    print("* Trained model saved to file:", str(output_file))
    
    output = open(output_file, 'wb')
    pickle.dump(model_dict, output)

def train_and_save_net(model, tag, input_cols, target_col, x_train, y_train, scaler_option, MAE=None, R2=None, path_to_save = '.', num_of_folds=5, epochs=100, batch_size=2) -> None:
        
    if MAE is None or R2 is None:
        
        print('* Model has not been evaluated. Evaluation initiated via %d-fold cross validation'%(num_of_folds))
        
        predictions, actual_values = cross_val_predict_net(model, epochs=epochs, batch_size=batch_size, x_train = x_train, y_train = y_train, verbose = 0, scaler_option = scaler_option, fast_tune = False)
        MAE, R2 = evaluate(predictions, actual_values)

    x_train_, scale = rescale_x(scaler_option, x_train)
    print('* Training initiated ..')
    #print("FITTING")
    model.fit(x_train_, y_train, epochs=epochs, batch_size=batch_size)
    print('* Training done.')
    
    model_dict = {}
    model_dict['tag'] = tag
    model_dict['model'] = model
    model_dict['model_abbr'] = 'NET'
    model_dict['input_cols'] = input_cols
    model_dict['target_col'] = target_col
    model_dict['MAE'] = MAE
    model_dict['R2'] = R2
    model_dict['fitted_scaler_x'] = scale
    output_file = PurePath(path_to_save) / (tag)
    #print(model_dict)
    
    print("* Trained model saved to file:", str(output_file))
    
    output = open(output_file, 'wb')
    pickle.dump(model_dict, output)

def train_and_save_classifier(model, tag, model_abbr, input_cols, target_col, x_train, y_train, scaler_option, accuracy=None, path_to_save = '.', num_of_folds=5) -> None:
    
    x_train_, scale = rescale_x(scaler_option, x_train)
    y_train_ = y_train
    #y_train_ = y_train.reshape(y_train.shape[0],)
    
    if accuracy is None:
        print('* Model has not been evaluated. Evaluation initiated via %d-fold cross validation'%(num_of_folds))
        
        predictions = cross_val_predict(model, x_train_, y_train_, cv=num_of_folds)
        actual_values = y_train_

        accuracy = evaluate_classifier(predictions, actual_values)

    print('* Training initiated ..')
    #print("FITTING")
    model.fit(x_train_, y_train_)
    print('* Training done.')
    actual_values = y_train_
    
    model_dict = {}
    model_dict['tag'] = tag
    model_dict['model'] = model
    model_dict['model_abbr'] = model_abbr
    model_dict['input_cols'] = input_cols
    model_dict['target_col'] = target_col
    model_dict['accuracy'] = accuracy
    model_dict['fitted_scaler_x'] = scale
    output_file = PurePath(path_to_save) / (tag)
    #print(model_dict)
    
    print("* Trained model saved to file:", str(output_file))
    
    output = open(output_file, 'wb')
    pickle.dump(model_dict, output)

def train_and_save(model, tag, model_abbr, input_cols, target_col, x_train, y_train, scaler_option, MAE=None, R2=None, path_to_save = '.', num_of_folds=5) -> None:
    
    x_train_, scale = rescale_x(scaler_option, x_train)
    y_train_ = y_train
    #y_train_ = y_train.reshape(y_train.shape[0],)
    
    if MAE is None or R2 is None:
        print('* Model has not been evaluated. Evaluation initiated via %d-fold cross validation'%(num_of_folds))
        
        predictions = cross_val_predict(model, x_train_, y_train_, cv=num_of_folds)
        actual_values = y_train_

        MAE, R2 = evaluate(predictions, actual_values)

    print('* Training initiated ..')
    #print("FITTING")
    model.fit(x_train_, y_train_)
    print('* Training done.')
    actual_values = np.array(y_train_)
    
    model_dict = {}
    model_dict['tag'] = tag
    model_dict['model'] = model
    model_dict['model_abbr'] = model_abbr
    model_dict['input_cols'] = input_cols
    model_dict['target_col'] = target_col
    model_dict['MAE'] = MAE
    model_dict['R2'] = R2
    model_dict['fitted_scaler_x'] = scale
    output_file = PurePath(path_to_save) / ((tag))
    #print(model_dict)
    
    print("* Trained model saved to file:", str(output_file))
    
    output = open(output_file, 'wb')
    pickle.dump(model_dict, output)

def evaluate_classifier(predictions, actual_values) -> float:

    correct = 0
    wrong = 0
    for i in range(0,len(predictions)):
        test1 = predictions[i]
        test2 = actual_values[i]
        if test1 == test2:
            correct+=1
        else:
            wrong+=1
    
    accuracy = float(correct)/(float(correct)+float(wrong))

    return accuracy

def evaluate(predictions, actual_values):

    try:
        MAE = mean_absolute_error(predictions,actual_values)
        R2 = r2_score(actual_values, predictions, multioutput='variance_weighted')
    except Exception as e:
        MAE = -1
        R2 = -1

    return MAE, R2

def save_comparison(prediction, actual_values, filename) -> None:
    output = pd.DataFrame({'Predicted Value': prediction, 'Actual Value': actual_values})
    if type(filename)==str:
        filename = PurePath(filename)

    if not os.path.exists(filename.parent): os.makedirs(filename.parent)
    output.to_csv(filename, index=False)
def save_comparison_chart(predictions, actual_values, filename) -> None:
    plt.close()
    min_val = min(predictions)
    max_val = max(actual_values)

    plt.xlabel('Predicted Value')
    plt.ylabel('Actual Value')
    plt.ylim([min_val, max_val])
    plt.xlim([min_val, max_val])
    plt.grid(True)

    plt.scatter(predictions,actual_values)
    t = np.arange(min_val, max_val, 0.01)
    line, = plt.plot(t, t, lw=1)
    
    if type(filename)==str:
        filename = PurePath(filename)

    if not os.path.exists(filename.parent): os.makedirs(filename.parent)
    plt.savefig(filename)
    plt.close()


# In[13]:

def add_key_to_params(tag, params) -> dict:
    tag = tag.lower()
    parameters = {}
    for key in params.keys():
        parameters[(tag+'_'+key).lower()] = params[key]
    return parameters


# In[14]:

def hyperparameter_tuning_classifier(tag, x_train, y_train, num_of_folds, scaler_option, n_iter=100, random_state=0, verbose=1):
    
    rf_n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 20)]
    rf_max_features = list(range(1,x_train.shape[1]))
    rf_max_depth = [int(x) for x in np.linspace(1, 32, 32)]
    rf_max_depth.append(None)
    rf_min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 15, num = 20)]
    rf_min_samples_leaf = [int(x) for x in np.linspace(start = 2, stop = 15, num = 20)]
    rf_bootstrap = [True, False]
    rf_criterion = ['gini']
    rf_min_weight_fraction_leaf = [float(x) for x in np.linspace(start = 0, stop = 1.e-5, num = 10)]
    rf_max_leaf_nodes = [2, 5, 10, 50, 100]
    rf_min_impurity_decrease = [float(x) for x in np.linspace(start = 0, stop = 1.e-5, num = 10)]
    
    rf_random_grid = {'n_estimators': rf_n_estimators,
                            'max_features': rf_max_features,
                            'max_depth': rf_max_depth,
                            'min_samples_split': rf_min_samples_split,
                            'min_samples_leaf': rf_min_samples_leaf,
                            'bootstrap': rf_bootstrap,
                            'criterion': rf_criterion,
                            'min_weight_fraction_leaf': rf_min_weight_fraction_leaf,
                            'max_leaf_nodes': rf_max_leaf_nodes,
                            'min_impurity_decrease': rf_min_impurity_decrease}

    nn_n_neighbors = [int(x) for x in np.linspace(start = 2, stop = 15, num = 10)]
    nn_weights = ['uniform', 'distance']
    nn_algorithm = ['auto','ball_tree','kd_tree','brute']
    nn_leaf_size = [1,2,3,4,5]
    nn_p = [int(x) for x in np.linspace(start = 1, stop = 5, num = 5)]
    nn_metric = ['minkowski']
    nn_metric_params = [None]

    nn_random_grid = {'n_neighbors': nn_n_neighbors,
                'weights': nn_weights,
                'algorithm': nn_algorithm,
                'leaf_size': nn_leaf_size,
                'metric': nn_metric,
                'metric_params': nn_metric_params,
                'p': nn_p}

    rg_alpha = [float(x) for x in np.linspace(start = 0, stop = 10, num = 10)]
    rg_fit_intercept = ['True', 'False']
    rg_max_iter = [100, 500, 1000, None]
    rg_tol = [float(x) for x in np.linspace(start = 1.e-5, stop = 1.e-2, num = 20)]
    rg_class_weight = [None,'balanced']
    rg_solver = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    
    rg_random_grid = {'max_iter': rg_max_iter,
                'alpha': rg_alpha,
                'fit_intercept': rg_fit_intercept,
                'tol': rg_tol,
                'class_weight': rg_class_weight,
                'solver': rg_solver
                }

    svm_kernel = ['rbf', 'poly','linear','sigmoid']           
    svm_gamma = ['auto', 0.001, 0.01, 0.1, 1]
    svm_degree = [1, 2, 3]
    svm_coef0 = [0, 1, 2, 3]
    svm_tol = [float(x) for x in np.linspace(start = 1.e-4, stop = 1.e-2, num = 20)]
    #svm_C = [float(x) for x in np.linspace(start = 0.001, stop = 3000, num = 100)]
    svm_C = [0.001, 0.01, 0.1, 1, 10]
    svm_decision_function_shape = ['ovr','ovo']    
    svm_random_grid = {'kernel': svm_kernel,
                'degree': svm_degree,
                'gamma' : svm_gamma,
                'coef0': svm_coef0,
                'tol': svm_tol,
                'C' : svm_C,
                'decision_function_shape':svm_decision_function_shape}

    if tag=='RF':
        estimator = RandomForestClassifier()
        random_grid = rf_random_grid
    elif tag=='NN':
        estimator = KNeighborsClassifier()
        random_grid = nn_random_grid
    elif tag=='RG':
        estimator = RidgeClassifier()
        random_grid = rg_random_grid
    elif tag=='SVM':
        estimator = svm.SVC()
        random_grid = svm_random_grid
    elif tag == 'XGB':
        estimator = XGBClassifier()
    else:
        estimator = None
    
    tuned_parameters = None
    if estimator is not None:
        model = RandomizedSearchCV(
                estimator = estimator, 
                param_distributions = random_grid, 
                n_iter = n_iter, 
                cv = num_of_folds, 
                verbose=verbose, 
                random_state=random_state, 
                n_jobs = multiprocessing.cpu_count())
        x_train_, scale = rescale_x(scaler_option, x_train)
        y_train_ = y_train
        #print("FITTING")
        model.fit(x_train_, y_train_)

        tuned_parameters = add_key_to_params(tag, model.best_params_)
        tuned_parameters['scaler_option'] = scaler_option

    return tuned_parameters

def hyperparameter_tuning(tag, x_train, y_train, num_of_folds, scaler_option, n_iter=100, random_state=0, verbose=1):
    
    rf_n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 20)]
    rf_max_features = list(range(1,x_train.shape[1]))
    rf_max_depth = [int(x) for x in np.linspace(1, 32, 32)]
    rf_max_depth.append(None)
    rf_min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 15, num = 20)]
    rf_min_samples_leaf = [int(x) for x in np.linspace(start = 2, stop = 15, num = 20)]
    rf_bootstrap = [True, False]
    rf_criterion = ['friedman_mse']
    rf_min_weight_fraction_leaf = [float(x) for x in np.linspace(start = 0, stop = 1.e-5, num = 10)]
    rf_max_leaf_nodes = [2, 5, 10, 50, 100]
    rf_min_impurity_decrease = [float(x) for x in np.linspace(start = 0, stop = 1.e-5, num = 10)]
    
    rf_random_grid = {'n_estimators': rf_n_estimators,
                            'max_features': rf_max_features,
                            'max_depth': rf_max_depth,
                            'min_samples_split': rf_min_samples_split,
                            'min_samples_leaf': rf_min_samples_leaf,
                            'bootstrap': rf_bootstrap,
                            'criterion': rf_criterion,
                            'min_weight_fraction_leaf': rf_min_weight_fraction_leaf,
                            'max_leaf_nodes': rf_max_leaf_nodes,
                            'min_impurity_decrease': rf_min_impurity_decrease}

    nn_n_neighbors = [int(x) for x in np.linspace(start = 2, stop = 15, num = 10)]
    nn_weights = ['uniform', 'distance']
    nn_algorithm = ['auto','ball_tree','kd_tree','brute']
    nn_leaf_size = [1,2,3,4,5]
    nn_p = [int(x) for x in np.linspace(start = 1, stop = 5, num = 5)]
    nn_metric = ['minkowski']
    nn_metric_params = [None]

    nn_random_grid = {'n_neighbors': nn_n_neighbors,
                'weights': nn_weights,
                'algorithm': nn_algorithm,
                'leaf_size': nn_leaf_size,
                'metric': nn_metric,
                'metric_params': nn_metric_params,
                'p': nn_p}

    kr_alpha = [float(x) for x in np.linspace(start = 0, stop = 10, num = 50)]
    kr_gamma = [None, 'RBF', 'laplacian','polynomial','chi2','exponential','sigmoid']
    kr_degree = [1,2,3]
    kr_coef0 = [0,1,2,3,4,5]
    kr_kernel =  ['linear']
        
    kr_random_grid = {'alpha': kr_alpha,
                      'kernel': kr_kernel,
                'gamma': kr_gamma,
                'degree': kr_degree,
                'coef0': kr_coef0}

    br_n_iter = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 50)]
    br_alpha_1 = [float(x) for x in np.linspace(start = 1.e-7, stop = 1.e-4, num = 50)]
    br_alpha_2 = [float(x) for x in np.linspace(start = 1.e-7, stop = 1.e-4, num = 50)]
    br_tol = [float(x) for x in np.linspace(start = 1.e-2, stop = 1.e-4, num = 50)]
    br_lambda_1 = [float(x) for x in np.linspace(start = 1.e-7, stop = 1.e-4, num = 50)]
    br_lambda_2 = [float(x) for x in np.linspace(start = 1.e-7, stop = 1.e-4, num = 50)]
    br_compute_score = [True, False]
    br_fit_intercept = [True, False]
    
    br_random_grid = {'n_iter': br_n_iter,
                'alpha_1': br_alpha_1,
                'alpha_2': br_alpha_2,
                'tol': br_tol,
                'lambda_1': br_lambda_1,
                'lambda_2': br_lambda_2,
                'compute_score': br_compute_score,
                'fit_intercept': br_fit_intercept}

    svm_kernel = ['rbf', 'poly','linear','sigmoid']           
    svm_gamma = ['auto', 0.001, 0.01, 0.1, 1]
    svm_degree = [1, 2, 3]
    svm_coef0 = [0, 1, 2, 3]
    svm_tol = [float(x) for x in np.linspace(start = 1.e-4, stop = 1.e-2, num = 50)]
    #svm_C = [float(x) for x in np.linspace(start = 0.001, stop = 3000, num = 100)]
    #svm_C+=[0.001, 0.01, 0.1, 1]
    svm_C = [0.001, 0.01, 0.1, 1, 10]
    svm_epsilon = [0.01, 0.1, 0.2, 0.3]
    
    svm_random_grid = {'kernel': svm_kernel,
                'degree': svm_degree,
                'gamma' : svm_gamma,
                'coef0': svm_coef0,
                'tol': svm_tol,
                'C': svm_C,
                'epsilon': svm_epsilon}
    

    rf_n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 20)]
    rf_max_features = list(range(1,x_train.shape[1]))
    rf_max_depth = [int(x) for x in np.linspace(1, 32, 32)]
    rf_max_depth.append(None)
    rf_min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 15, num = 20)]
    rf_min_samples_leaf = [int(x) for x in np.linspace(start = 2, stop = 15, num = 20)]
    rf_bootstrap = [True, False]
    rf_criterion = ['friedman_mse']
    rf_min_weight_fraction_leaf = [float(x) for x in np.linspace(start = 0, stop = 1.e-5, num = 10)]
    rf_max_leaf_nodes = [2, 5, 10, 50, 100]
    rf_min_impurity_decrease = [float(x) for x in np.linspace(start = 0, stop = 1.e-5, num = 10)]

    #clf_xgb = XGBClassifier(objective = 'binary:logistic')
    xgb_random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 1000, num = 20)],
              'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
              'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
              'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
              'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
              'min_child_weight': [1, 2, 3, 5, 7],
              'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
              'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
             }

    if tag=='RF':
        estimator = RandomForestRegressor()
        random_grid = rf_random_grid
    elif tag=='NN':
        estimator = KNeighborsRegressor()
        random_grid = nn_random_grid
    elif tag=='KR':
        estimator = KernelRidge()
        random_grid = kr_random_grid
    elif tag=='BR':
        estimator = linear_model.BayesianRidge()
        random_grid = br_random_grid
    elif tag=='SVM':
        estimator = svm.SVR()
        random_grid = svm_random_grid
    elif tag == 'XGB':
        estimator = XGBRegressor()
        random_grid = xgb_random_grid
    else:
        estimator = None
    
    tuned_parameters = None
    if estimator is not None:
        model = RandomizedSearchCV(
                estimator = estimator, 
                param_distributions = random_grid, 
                n_iter = n_iter, 
                cv = num_of_folds, 
                verbose=verbose, 
                random_state=random_state, 
                error_score='raise',
                n_jobs = multiprocessing.cpu_count())
        x_train_, scale = rescale_x(scaler_option, x_train)
        y_train_ = y_train
        #print("FITTING")
        model.fit(x_train_, y_train_)

        tuned_parameters = add_key_to_params(tag, model.best_params_)
        tuned_parameters['scaler_option'] = scaler_option

    return tuned_parameters

def cv_resample(x_train, y_train, num_of_folds=5):
    """takes x, y and resamples it into n folds for cross-validation"""
    num = x_train.shape[0]/int(num_of_folds) # num of row in each fold
    x_train_parts = []
    y_train_parts = []
    for i in range(0,int(num_of_folds)):
        start_idx = int(i * num)
        end_idx = min(int(i * (num + 1)), x_train.shape[0])
        x_train_parts.append(x_train[start_idx: end_idx])

    x_cvtrains = []
    y_cvtrains = []
    x_cvtests = []
    y_cvtests = []
    
    for i in range(0, num_of_folds):
        x_test = x_train_parts[i]
        y_test = y_train_parts[i]
        x_train = []
        y_train = []
        for j in range(0, num_of_folds):
            if j != i:
                x_train+=list(x_train_parts[j])
                y_train+=list(y_train_parts[j])
        
        x_cvtrains.append(np.array(x_train))
        y_cvtrains.append(np.array(y_train))
        x_cvtests.append(np.array(x_test))
        y_cvtests.append(np.array(y_test))
        
    return x_cvtrains, y_cvtrains, x_cvtests, y_cvtests

def net_define(params = [8, 8, 8], layer_n = 3, input_size = 16, dropout=0, l_2=0.01, optimizer='adam', random_state = 0):
    
    if len(params)!=layer_n or layer_n<1:
        return None
    
    model = Sequential()
    model.add(Dense(params[0], kernel_initializer='normal', activation='relu', input_dim=input_size, kernel_regularizer=regularizers.l2(l_2)))
    
    for i in range(1, layer_n):
        if dropout!=0:
            model.add(Dropout(dropout))
        model.add(Dense(params[i], kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(l_2)))
    
    model.add(Dense(1, activation = 'linear'))

    model.compile(loss='mse',
                  optimizer=optimizer, metrics=['mape'])
    
    print(params, layer_n, dropout, l_2, optimizer)
    
    return model

def net_define_classifier(params = [8, 8, 8], layer_n = 3, num_of_class = 2, input_size = 16, dropout=0, l_2=0.01, optimizer='adam', random_state = 0):
    
    if len(params)!=layer_n or layer_n<1:
        return None
    
    model = Sequential()
    model.add(Dense(params[0], kernel_initializer='normal', activation='relu', input_dim=input_size, kernel_regularizer=regularizers.l2(l_2)))
    
    for i in range(1, layer_n):
        if dropout!=0:
            model.add(Dropout(dropout))
        model.add(Dense(params[i], kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(l_2)))
    
    model.add(Dense(num_of_class, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizfer=optimizer, metrics=['accuracy'])
    
    return model

def evaluate_net(model, x_train, y_train, x_test, y_test, epochs=1000, batch_size=8, verbose = 0) -> tuple:
    # TODO: find out if there should be a seperate evaluate_net_classifier
    if optimizer is None: # TODO: find out what this does, because optimizer is never set, so this is always true. Should it take a default parameter? Also this function is never called
        optimizer = 'adam'
    #print("FITTING")
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    
    return score, history

def net_tuning_classifier(x_train, y_train, num_of_class = 2, tries = 10, lr = None, layer = None, params=None, epochs=None, batch_size=None, dropout=None, l_2 = None, neuron_max=[64, 64, 64], batch_size_max=32, layer_min=1, layer_max=3, dropout_max=0.2, scaler_option='StandardScaler', default_neuron_max=32, checkpoint = None, num_of_folds=5, fast_tune = True, random_state = 0):
    tuned_parameters = {}

    if layer is not None:
        
        if layer<1:
            print("Error: layer must be >=1")
            sys.exit()
    
    # Trying to tune hyperparameters
    
    best_score = 0
    best_params = None

    _layer = layer
    _params = params
    _epochs = epochs
    _batch_size = batch_size
    _dropout = dropout
    _l_2 = l_2
    _neuron_max = neuron_max
    _batch_size_max = batch_size_max
    _dropout_max = dropout_max
    _lr = lr
    
    for i in range(0, tries):

        optimizer = 'adam'
        
        if lr is None:
            lr = 10**np.random.uniform(-4,-3)
            optimizer = keras.optimizers.Adam(lr=lr)
        else:
            optimizer = keras.optimizers.Adam(lr=lr)

        if layer is None:
            layer = random.sample(range(layer_min, layer_max+1), 1)[0]

        if params is None:
            params = []
            for j in range(0, layer):
                try:
                    params.append(random.sample(range(1, neuron_max[j]), 1)[0])
                except:
                    params.append(random.sample(range(1, default_neuron_max), 1)[0])

        if epochs is None:
            epochs = int(10**np.random.uniform(2,3)) # 100 - 1000

        if batch_size is None:
            batch_size = random.sample(range(1, batch_size_max), 1)[0]

        if dropout is None:
            dropout = 0
        elif dropout is True:
            dropout = random.uniform(0,dropout_max)
        else:
            dropout = dropout

        if l_2 is None:
            l_2 = 10**np.random.uniform(-3,-1)

        model = net_define_classifier(params=params, layer_n = layer, input_size = x_train.shape[1], dropout=dropout, l_2=l_2, optimizer=optimizer, num_of_class = num_of_class, random_state = random_state)  
        
        print("\n Cross-validation (iteration=%d): [layer=%d, structure=[%s], epochs=%d, dropout=%8.4f, l_2=%8.7f, batch_size=%d, lr=%8.7f]"%(i, layer, params, epochs, dropout, l_2, batch_size, lr))
        start_time = time.time()
        predictions, actual_values = cross_val_predict_net_classifier(model, epochs=epochs, batch_size=batch_size, x_train = x_train, y_train = y_train, verbose = 0, scaler_option = scaler_option, num_of_folds = num_of_folds, num_of_class = num_of_class, fast_tune = fast_tune)
        
        if predictions == []:
            print(" Validation stopped early with the setting:","[layer=%d, structure=[%s], epochs=%d, dropout=%8.4f, l_2=%8.7f, batch_size=%d, lr=%8.7f]"%(layer, params, epochs, dropout, l_2, batch_size, lr))
            print(' Keep trying to find best settings .., took %ss'%(time.time()-start_time))
            accuracy= -1
        else:
            accuracy = evaluate_classifier(predictions, actual_values)
        
        print("Cross validation result - accuracy = %8.3f, took %ss "%(accuracy, time.time()-start_time))

        if(accuracy>best_score):          
            
            best_score = accuracy
            best_params = (layer, params, epochs, dropout, l_2, batch_size, lr)
            tuned_parameters = {"net_layer_n":best_params[0], \
            "net_structure":str(best_params[1])[1:-1].replace(",",""), \
            "net_epochs":best_params[2], \
            "net_dropout":best_params[3], \
            "net_l_2":best_params[4], \
            "net_batch_size":best_params[5], \
            "net_learning_rate": best_params[6]}

            if checkpoint is not None:
                print("Best so far parameters stored :", str(checkpoint)+",Model=NET,Scaler="+str(scaler_option)+",accuracy="+str(accuracy)+".tuned.checkpoint.prop")
                save_parameters(tuned_parameters, str(checkpoint)+",Model=NET,Scaler="+str(scaler_option)+",accuracy="+str(accuracy)+".tuned.checkpoint.prop")
        
        if(best_score!=0):
            print("Best accuracy = %8.3f"%(best_score),"[layer=%d, structure=[%s], epochs=%d, dropout=%8.4f, l_2=%8.7f, batch_size=%d, lr=%8.7f]"%best_params)

        # set to original values
        layer = _layer
        params = _params
        epochs = _epochs
        batch_size = _batch_size
        dropout = _dropout
        l_2 = _l_2
        neuron_max = _neuron_max
        batch_size_max = _batch_size_max
        dropout_max = _dropout_max
        lr = _lr
    
    tuned_parameters['scaler_option'] = scaler_option
    
    return tuned_parameters

    print(" -- DONE --")

def net_tuning(x_train, y_train, tries = 10, lr = None, layer = None, params=None, epochs=None, batch_size=None, dropout=None, l_2 = None, neuron_max=[64, 64, 64], batch_size_max=32, layer_min=1, layer_max=3, dropout_max=0.2, scaler_option='StandardScaler', default_neuron_max=32, checkpoint = None, num_of_folds=5, fast_tune = True, random_state = 0):

    tuned_parameters = {}
    if layer is not None:
        
        if layer<1:
            print("Error: layer must be >=1")
            sys.exit()
    
    # Trying to tune hyperparameters
    
    best_score = 0
    best_params = None
    best_mae = 0

    _layer = layer
    _params = params
    _epochs = epochs
    _batch_size = batch_size
    _dropout = dropout
    _l_2 = l_2
    _neuron_max = neuron_max
    _batch_size_max = batch_size_max
    _dropout_max = dropout_max
    _lr = lr
    
    for i in range(0, tries):

        optimizer = 'adam'
        
        if lr is None:
            lr = 10**np.random.uniform(-4,-3)
            optimizer = keras.optimizers.Adam(lr=lr)
        else:
            optimizer = keras.optimizers.Adam(lr=lr)

        if layer is None:
            layer = random.sample(range(layer_min, layer_max+1), 1)[0]
        
        params = _params
        if params is None:
            params = []
            for j in range(0, layer):
                try:
                    params.append(random.sample(range(1, neuron_max[j]), 1)[0])
                except:
                    params.append(random.sample(range(1, default_neuron_max), 1)[0])

        if epochs is None:
            epochs = int(10**np.random.uniform(2,3)) # 100 - 1000

        if batch_size is None:
            batch_size = random.sample(range(1, batch_size_max), 1)[0]

        if dropout is None:
            dropout = 0
        elif dropout is True:
            dropout = random.uniform(0,dropout_max)
        else:
            dropout = dropout

        if l_2 is None:
            l_2 = 10**np.random.uniform(-3,-1)

        model = net_define(params=params, layer_n = layer, input_size = x_train.shape[1], dropout=dropout, l_2=l_2, optimizer=optimizer, random_state = random_state)
        
        print("\n Cross-validation (iteration=%d): [layer=%d, structure=[%s], epochs=%d, dropout=%8.4f, l_2=%8.7f, batch_size=%d, lr=%8.7f]"%(i, layer, params, epochs, dropout, l_2, batch_size, lr))
        start_time = time.time()
        predictions, actual_values = cross_val_predict_net(model, epochs=epochs, batch_size=batch_size, x_train = x_train, y_train = y_train, verbose = 0, scaler_option = scaler_option, num_of_folds = num_of_folds, fast_tune = fast_tune)
        
        if predictions == []:
            print(" Validation stopped early with the setting:","[layer=%d, structure=[%s], epochs=%d, dropout=%8.4f, l_2=%8.7f, batch_size=%d, lr=%8.7f]"%(layer, params, epochs, dropout, l_2, batch_size, lr))
            print(' Keep trying to find best settings .., took %ss'%(time.time()-start_time))

            MAE= -1
            R2 = -1
        else:
            MAE, R2 = evaluate(predictions, actual_values)

        print("Cross validation result - MAE = %8.3f R2 = %8.3f, took %ss "%(MAE, R2, time.time()-start_time))

        if(R2>best_score and R2!=-1) or (best_score==0 and R2!=-1):          
        
            best_score = R2
            best_mae = MAE
            best_params = (layer, params, epochs, dropout, l_2, batch_size, lr)
            tuned_parameters = {"net_layer_n":best_params[0], \
            "net_structure":str(best_params[1])[1:-1].replace(",",""), \
            "net_epochs":best_params[2], \
            "net_dropout":best_params[3], \
            "net_l_2":best_params[4], \
            "net_batch_size":best_params[5], \
            "net_learning_rate": best_params[6]}

            if checkpoint is not None:
                print("Best so far parameters stored :", str(checkpoint)+",Model=NET,Scaler="+str(scaler_option)+",MAE="+str(MAE)+",R2="+str(R2)+".tuned.checkpoint.prop")
                save_parameters(tuned_parameters, str(checkpoint)+",Model=NET,Scaler="+str(scaler_option)+",MAE="+str(MAE)+",R2="+str(R2)+".tuned.checkpoint.prop")
                
                print(" Saving test charts to : ", str(checkpoint)+",Model=NET,Scaler="+str(scaler_option)+",MAE="+str(MAE)+",R2="+str(R2)+".tuned.checkpoint.png")
                #try:    
                save_comparison_chart(predictions, actual_values, str(checkpoint)+",Model=NET,Scaler="+str(scaler_option)+",MAE="+str(MAE)+",R2="+str(R2)+".tuned.checkpoint.png")
                #except:
                #    print(" * Warning: couldn't generate a chart - please make sure the model is properly trained .. ")
        
        if(best_score!=0):
            print("Best R2 = %8.3f"%(best_score),"MAE=",best_mae, "[layer=%d, structure=[%s], epochs=%d, dropout=%8.4f, l_2=%8.7f, batch_size=%d, lr=%8.7f]"%best_params)

        # set to original values
        layer = _layer
        params = _params
        epochs = _epochs
        batch_size = _batch_size
        dropout = _dropout
        l_2 = _l_2
        neuron_max = _neuron_max
        batch_size_max = _batch_size_max
        dropout_max = _dropout_max
        lr = _lr
    
    tuned_parameters['scaler_option'] = scaler_option
    
    return tuned_parameters

    print(" -- DONE --")
