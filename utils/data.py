import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from collections import defaultdict

GT_coords = ['Xmin_true', 'Ymin_true', 'Xbias_true', 'Ybias_true']
coords = ['Xmin', 'Ymin', 'Xbias', 'Ybias']

def load_data(paths, normalize=True, bias=True, delete_zero=True):
    # loading the data
    train = pd.read_csv(paths['train'])
    test = pd.read_csv(paths['test'])
    targets = pd.read_csv(paths['targets'])

    # normalizing the data
    if normalize:
        stats = defaultdict(dict)

        for coord in ['X', 'Y']:
            array = np.hstack([train[coord + 'min'].values, test[coord + 'min'].values, targets[coord + 'min_true'],
                  train[coord + 'max'].values, test[coord + 'max'].values, targets[coord + 'max_true'].values])
            stats[coord]['mean'] = array.mean()
            stats[coord]['std'] = array.std()
            stats[coord]['max'] = array.max()
            stats[coord]['min'] = array.min()

        for coord in ['X', 'Y']:
            for stat in ['min', 'max']:
                col = '{}{}'.format(coord, stat)
                for df in [train, test]:
                    df[col] -= stats[coord]['mean']
                    df[col] /= stats[coord]['std']

                true_col = '{}{}_true'.format(coord, stat)
                targets[true_col] -= stats[coord]['mean']
                targets[true_col] /= stats[coord]['std']

    # transforming into bias instead of max system
    if bias:
        for df in [train, test, targets]:
            df.rename({
                'itemId': 'item',
                'userId': 'user'
            }, axis=1, inplace=True)

        for df, prefix in [(train, ''), (targets, '_true'), (test, '')]:
            for coord in ['X', 'Y']:
                df['{}bias{}'.format(coord, prefix)] = df['{}max{}'.format(coord, prefix)] \
                    - df['{}min{}'.format(coord, prefix)]
                df.drop('{}max{}'.format(coord, prefix), axis=1, inplace=True)

    # deleting objects with zero GT area
    if delete_zero:
        mask = (targets['Xbias_true'] == 0) | (targets['Ybias_true'] == 0)
        bad_items = targets[mask]['item'].values
        bad_items
        train.drop(train[train['item'].isin(bad_items)].index, axis=0, inplace=True)
        train.reset_index(drop=True, inplace=True)
    if normalize:
        return train, test, targets, stats
    else:
        return train, test, targets

    
class TransDataset(Dataset):
    
    def __init__(self, trans_df, targets=None):
        self.features = ['Xmin', 'Ymin', 'Xbias', 'Ybias']
        self.data = trans_df[self.features].values.astype(np.float32)
        self.users = trans_df['user'].values
        self.items = trans_df['item'].values
        self.len = self.data.shape[0]
        self.n_features = len(self.features)
        self.targets = targets.set_index('item').loc[self.items].values.astype(np.float32) if targets is not None else None
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        obj = self.data[idx]
        if self.targets is not None:
            return obj, self.targets[idx]
        else:
            return obj

        
class Inference:
    
    def __init__(self, make_dataset):
        self.make_dataset = make_dataset
        
    def predict(self, df, models):
        ds = self.make_dataset(df)
        dl = DataLoader(ds, batch_size=256)
        ens_preds = []
        for model in models:
            preds = []
            for obj in dl:
                with torch.no_grad():
                    pred = model(obj).numpy()
                preds.append(pred)
            preds = np.vstack(preds)
        ens_preds.append(preds)
        ens_preds = np.mean(ens_preds, axis=0)
        preds_df = pd.DataFrame()
        preds_df['itemId'] = ds.items
        preds_df[['Xmin', 'Ymin', 'Xbias', 'Ybias']] = pd.DataFrame(ens_preds)
        preds_df['Xmax'] = preds_df['Xmin'] + preds_df['Xbias']
        preds_df['Ymax'] = preds_df['Ymin'] + preds_df['Ybias']
        preds_df.drop(['Xbias', 'Ybias'], axis=1, inplace=True)
        return preds_df
    
    
def denormalize(df, stats, rounded=True):
    df = df.copy()
    for coord in ['X', 'Y']:
        for stat in ['min', 'max']:
            df['{}{}'.format(coord, stat)] *= stats[coord]['std']
            df['{}{}'.format(coord, stat)] += stats[coord]['mean']
            df['{}{}'.format(coord, stat)] = np.maximum(df['{}{}'.format(coord, stat)], 0)
    return np.round(df) if rounded else df
