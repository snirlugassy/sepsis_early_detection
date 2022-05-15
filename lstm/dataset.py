import torch
import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d


class ICUSepsisDataset(torch.utils.data.Dataset):
    features = ['HR', 'O2Sat', 'Temp', 'MAP','Resp', 'Age', 'Gender', 'ICULOS']
    target = 'SepsisLabel'
    physiological = ['HR', 'O2Sat', 'Temp', 'MAP','Resp']
    def __init__(self, path):
        self.files = [os.path.join(path, f) for f in os.listdir(path)]
        # since the data is highly imbalanced and favors non-sepsis patients,
        # during training we will mihgt add drop probabilty in case of non-sepsis patient
        # self.drop_non_sepsis_prob = drop_non_sepsis_prob
        print(f'Found {len(self.files)} files in {path}')
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        data = pd.read_csv(self.files[i], sep='|')

       # Remove the data after the first sepsis trigger
        if (data[ICUSepsisDataset.target] == 1).any():
            first_sepsis_idx = data[ICUSepsisDataset.target].idxmax()
            data = data[:first_sepsis_idx+1]

        y = data[ICUSepsisDataset.target].to_numpy()

        # has_sepsis = False
        # if (y==1).any():
        #     has_sepsis = True

        # if self.drop_non_sepsis_prob > 0 and not has_sepsis:
        #     if np.random.rand() <= self.drop_non_sepsis_prob:
        #         return None, torch.from_numpy(y)
        

        if data[ICUSepsisDataset.features].isna().all(axis=0).any():
            # print(f'sample {i} missing some entire feature, skipping')
            # print(data[ICUSepsisDataset.features].isna().all(axis=0))
            # some physiological feature is missing
            # currently, just return None,None
            # TODO: implement better imputation mechanism
            return None, torch.from_numpy(y)

        for f in ICUSepsisDataset.features:
            _x = data[f][~data[f].isna()]
            try:
                interp = interp1d(_x.index, _x.values, fill_value='extrapolate', kind='nearest')
                data[f] = interp(data[f].index)
            except Exception as e:
                # print(e)
                # print(f'error imputing sample {i}, skipping')
                return None, torch.from_numpy(y)

        X = data[ICUSepsisDataset.features].to_numpy()
        
        assert len(X) == len(y)
        
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError('DF has missing values', i, data)
        return torch.from_numpy(X).float(), torch.from_numpy(y)
