import sys
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import xgboost as xgb

OPTIMAL_THRESHOLD = 0.4646464
VITAL_FEATURES = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
LAB_FEATURES = ['BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']
DEMOGRAPHIC_FEATURES = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
DYNAMIC_FEATURES = VITAL_FEATURES + LAB_FEATURES
TARGET = 'SepsisLabel'

def preprocessing(dfs: list[pd.DataFrame], interactions=True):
    patient_vecs = []
    for i in range(len(dfs)):
        _df = dfs[i]
        _mean = _df[DYNAMIC_FEATURES].mean().add_suffix('_mean')
        _rolling_mean = _df[DYNAMIC_FEATURES].rolling(3).mean().mean().add_suffix('_rolling_mean')
        _diff_mean = _df[DYNAMIC_FEATURES].diff().mean().add_suffix('_diff_mean')
        _median = _df[DYNAMIC_FEATURES].median().add_suffix('_median')
        _mean_minus_median = (_df[DYNAMIC_FEATURES].mean() - _df[DYNAMIC_FEATURES].median()).add_suffix('_mean_minus_median')
        _std = _df[DYNAMIC_FEATURES].std().add_suffix('_std')
        _sem = _df[DYNAMIC_FEATURES].sem().add_suffix('_sem')
        _range = (_df[DYNAMIC_FEATURES].max() - _df[DYNAMIC_FEATURES].min()).add_suffix('_range')
        _max1 = _df[VITAL_FEATURES].max().add_suffix('_max')
        _max2 = _df[['SepsisLabel', 'Gender', 'Age', 'ICULOS']].max()
        _skew  = _df[DYNAMIC_FEATURES].skew().add_suffix('_skew')
        _pvec = pd.concat([_mean, _rolling_mean, _diff_mean, _range, _std, _sem, _median, _mean_minus_median, _max1, _max2, _skew], axis=0)
        patient_vecs.append(_pvec)

    _df = pd.DataFrame(patient_vecs)
    _df.reset_index(drop=True, inplace=True)

    if interactions:
        for f1 in VITAL_FEATURES:
            for f2 in VITAL_FEATURES:
                if f1 != f2 and f1 + '_mean' in _df and f2 + '_mean' in _df:
                    f1_mean = _df[f1 + '_mean']
                    f2_mean = _df[f2 + '_mean']
                    _df[f'log_{f1}_mean_x_{f2}_mean'] = np.log(1+f1_mean) * np.log(1+f2_mean)
                    _df[f'{f1}_mean_over_{f2}_mean'] = f1_mean / (1+f2_mean)
                    _df[f'{f1}_skew_x_{f2}_skew'] = _df[f1+'_skew'] * _df[f2+'_skew']

    return _df

def read_dataframes(data_path:str) -> list[pd.DataFrame]:
    _dfs = []
    _files = os.listdir(data_path)
    for f in _files:
        _dfs.append(pd.read_csv(os.path.join(data_path, f), sep='|'))
    return _dfs

xgb.config_context(verbosity=0, nthread=4)


if __name__ == '__main__':
    data_path = sys.argv[-1]

    print('Loading model')
    bst = xgb.Booster()
    bst.load_model('model.json')

    print('Reading dataframes')
    dataframes = read_dataframes(data_path)

    print('Preprocessing')
    df = preprocessing(dataframes)

    features = list(df.columns)
    
    if TARGET in features:
        features.remove(TARGET)

    test_data = xgb.DMatrix(df[features])

    print('Predicting')
    y_predict = bst.predict(test_data)
    y_predicted_label = 1*(y_predict >= OPTIMAL_THRESHOLD)
    df['predicted_label'] = y_predicted_label
    
    print('Saving to CSV')
    df['file_name'] = [os.path.splitext(d)[0].replace('patient_','') for d in os.listdir(data_path)]
    df[['file_name', 'predicted_label']].to_csv('prediction.csv', header=False, index=False)
