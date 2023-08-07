import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import re
import glob
import polars as pl

from lstm.lstm import LSTMClassifier



def create_datasets(X, y, test_size=0.2, dropcols=[], time_dim_first=False):
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)
    X_grouped = X
    if time_dim_first:
        X_grouped = X_grouped.transpose(0, 2, 1)
    X_train, X_valid, y_train, y_valid = train_test_split(X_grouped, y_enc, test_size=0.1)
    X_train, X_valid = [torch.tensor(arr, dtype=torch.float32) for arr in (X_train, X_valid)]
    y_train, y_valid = [torch.tensor(arr, dtype=torch.long) for arr in (y_train, y_valid)]
    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)
    return train_ds, valid_ds, enc

def create_loaders(train_ds, bs=512, jobs=4):
    train_dl = DataLoader(train_ds, bs, shuffle=False, num_workers=jobs)
    return train_dl


def prepare_data(ts_file, im_size=2000):

    big_im = ts_file
    big_df = pl.read_csv(big_im)
    X_im = big_df.select(['blue','green','red','nir','nvdi'])
    X_im = X_im.to_pandas()
    # X_im = X_im.replace(0, -10000)


    X_array = X_im.to_numpy()
    print("X_array shape: ", X_array.shape)

    X_lst = []
    for i in range(0,len(X_array),im_size*im_size):
        X_lst.append(X_array[i:i+(im_size*im_size),:])
        
    X = np.concatenate((X_lst),axis=1)
    X[np.isnan(X)] = -10000

    print("X shape: ", X.shape)

    X_band = []
    for i in range(0,X.shape[1],5):
        X_band.append(X[:,i:i+5])

    X_out = np.stack(X_band)
    X_out = np.transpose(X_out, (1,0,2))

    print('X_out shape: ', X_out.shape)

    X = X_out
    X = torch.tensor(X, dtype=torch.float32)

    return X


def predict(ts_file):


    X = prepare_data(ts_file)

    bs = 256
    trn_dl = create_loaders(X, bs, jobs=4)
    print(f'Creating data loaders with batch size: {bs}')

    input_dim = 5    
    hidden_dim = 256
    layer_dim = 3
    output_dim = 6 # previously 5

    output_lst = []

    model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
    model = model.cuda()

    checkpoint = 'output/checkpoints/lstm-best-0803.pth'
    model.load_state_dict(torch.load(checkpoint))

    for i, x in enumerate(trn_dl):

        output = model(x.cuda())
        output = F.log_softmax(output, dim=1).argmax(dim=1)
        output = output.detach().cpu().numpy()

        output_lst.append(output)

    print(len(output_lst))

    output_arr = np.concatenate((output_lst), axis=0)
    output_arr = output_arr.reshape((2000,2000))
    print('output array shape', output_arr.shape)

    return output_arr

    
if __name__ == '__main__':

    files_lst = sorted(glob.glob('output/csv/*.csv'))
    for ts_file in files_lst:
        search_term = re.search(r'all.(.*\d*).csv', ts_file).group(1)
        search_widx = re.search(r'\d_(\d*)', search_term).group(1)
        search_hidx = re.search(r'\d-(\d*)', search_term).group(1)
        output = predict(ts_file)
        
        np.save(f'output/rf_out/lstm-{search_hidx}_{search_widx}.npy', output)