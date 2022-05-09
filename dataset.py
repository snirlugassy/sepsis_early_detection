import torch
import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d


class ICUSepsisDataset(torch.utils.data.Dataset):
    features = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP','Resp', 'Age', 'Gender']
    target = 'SepsisLabel'
    physiological = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP','Resp']
    def __init__(self, path):
        self.files = [os.path.join(path, f) for f in os.listdir(path)]
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

        if data[ICUSepsisDataset.features].isna().all(axis=0).any():
            # some physiological feature is missing
            # currently, just return None,None
            # TODO: implement better imputation mechanism
            return None, torch.from_numpy(y)

        for f in ICUSepsisDataset.physiological:
            _x = data[f][~data[f].isna()]
            try:
                interp = interp1d(_x.index, _x.values, fill_value='extrapolate', kind='nearest')
                data[f] = interp(data[f].index)
            except:
                return None, torch.from_numpy(y)

        X = data[ICUSepsisDataset.features].to_numpy()
        
        assert len(X) == len(y)
        
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError('DF has missing values', i, data)
        return torch.from_numpy(X).float(), torch.from_numpy(y)
