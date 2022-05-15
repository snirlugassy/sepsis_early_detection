import os
import pandas as pd
import numpy as np
from tqdm import tqdm

vitals_features = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
lab_features = ['BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']
demographic_features = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
target = 'SepsisLabel'
dynamic_features = vitals_features + lab_features
take_max_cols = ['SepsisLabel', 'Age', 'Gender', 'ICULOS']

def read_dataframes(data_path:str, limit=None) -> list[pd.DataFrame]:
    _dfs = []
    if isinstance(limit, int) and limit > 0:
        _files = os.listdir(data_path)[:limit]
    else:
        _files = os.listdir(data_path)
    with tqdm(total=len(_files)) as pbar:
        for f in _files:
            _dfs.append(pd.read_csv(os.path.join(data_path, f), sep='|'))
            pbar.update(1)
    print(f'Found {len(_dfs)} dataframes in {data_path}')
    return _dfs

def preprocessing(dfs: list[pd.DataFrame], interactions=True, dropna=False):
    patient_vecs = []
    with tqdm(total=len(dfs)) as pbar:
        for i in range(len(dfs)):
            _df = dfs[i]
            _mean = _df[dynamic_features].mean().add_suffix('_mean')
            _std = _df[dynamic_features].std().add_suffix('_std')
            _range = (_df[dynamic_features].max() - _df[dynamic_features].min()).add_suffix('_range')
            _max1 = _df[vitals_features].max().add_suffix('_max')
            _max2 = _df[['SepsisLabel', 'Gender']].max()
            _log_max = np.log(_df[['Age', 'ICULOS']].max().add_prefix('log_'))
            patient_vecs.append(pd.concat([_mean, _std, _range, _max1, _max2, _log_max], axis=0))
            pbar.update(1)


    _df = pd.DataFrame(patient_vecs)

    if interactions:
        for f1 in vitals_features:
            for f2 in vitals_features:
                if f1 != f2 and f1 + '_mean' in _df and f2 + '_mean' in _df:
                    f1_mean = _df[f1 + '_mean']
                    f2_mean = _df[f2 + '_mean']
                    _df[f'log_{f1}_mean_x_{f2}_mean'] = np.log(1 + f1_mean * f2_mean)

    if dropna:
        _df.dropna(inplace=True)

    _df.reset_index(drop=True, inplace=True)

    return _df