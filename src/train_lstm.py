import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import pickle
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

def create_loaders(train_ds, valid_ds, bs=512, jobs=0):
    train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=jobs)
    valid_dl = DataLoader(valid_ds, bs, shuffle=False, num_workers=jobs)
    return train_dl, valid_dl


def prepare_data(training_file, scale=False):

    # with open(training_file, 'rb') as file:
    #     training_df = pickle.load(file)

    # big_df = pl.read_csv(training_file, has_header=True)
    # # X_im = big_df.select(['blue','green','red','nir','nvdi','class'])
    # # X_im = X_im.to_pandas()

    # X_im = big_df

    # print(X_im.shape)
    # print(X_im.head(5))

    # training_df = X_im
    # del X_im

    # print(training_df['class'].unique())
    # # crop = training_df['class']

    # training_df = training_df.to_pandas()

    training_df_1 = pd.read_csv(training_file[0])
    training_df_2 = pd.read_csv(training_file[1])

    training_df_2 = training_df_2.replace(2,5)
    training_df_2 = training_df_2.replace(1,2)
    training_df_2 = training_df_2.replace(0,1)
    
    print(training_df_1['class'].unique())
    print(training_df_2['class'].unique())

    frames = [training_df_1, training_df_2]

    training_df = pd.concat(frames)

    Y = training_df['class'].values
    # X = training_df.iloc[:,3:8].values
    X = training_df.iloc[:,2:27].values

    # print(Y[:5])
    # print(X[:5])

    print('X shape: ',X.shape)
    print('Y shape: ',Y.shape)

    X_band = []

    for i in range(0,X.shape[1],5):
        print(X[:,i:i+5].shape)
        X_band.append(X[:,i:i+5])

    X_out = np.stack(X_band)
    X_out = np.transpose(X_out, (1,0,2))


    print('X_out shape: ', X_out.shape)
    print('Y shape: ',Y.shape)

    X = X_out

    class_number=5 # other-crop, corn, soybean, fall-crop, water
    test_size=0.2
    random_state = 42

    X_train = [] 
    y_train = [] 
    X_test = []    
    y_test = []

    # class_types=['other-crop','corn','soybean','fall-crop','water']
    # class_types=['other','corn','soybean','water','fall-crop','other-crop']
    for i in training_df['class'].unique():
        Y_class = Y[Y==i] # other-crop=0, corn=1, soybean=2, fall-crop=3, water=4
        X_class = X[Y==i]

        # X_class[np.isnan(X_class)] = -10000
        
        X_train_class, X_test_class, y_train_class, y_test_class = \
            train_test_split(X_class, Y_class, test_size=test_size, random_state=random_state)

        print(i,y_train_class.shape,X_train_class.shape)

    
        X_train.append(X_train_class)
        X_test.append(X_test_class)
        y_train.append(y_train_class)
        y_test.append(y_test_class)
        
    # Merge multiple classes      
    X_train_all = np.concatenate(X_train, axis=0)
    y_train_all = np.concatenate(y_train, axis=0).astype('uint8')
    X_test_all = np.concatenate(X_test, axis=0)    
    y_test_all = np.concatenate(y_test, axis=0).astype('uint8')

    X_train = X_train_all
    y_train = y_train_all
    X_test = X_test_all
    y_test = y_test_all

    X_train, X_test = [torch.tensor(arr, dtype=torch.float32) for arr in (X_train, X_test)]
    y_train, y_test = [torch.tensor(arr, dtype=torch.long) for arr in (y_train, y_test)]
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)

    return train_ds, test_ds


def train(training_file):


    train_ds, test_ds = prepare_data(training_file, scale=True)

    bs = 128
    print(f'Creating data loaders with batch size: {bs}')
    trn_dl, val_dl = create_loaders(train_ds, test_ds, bs, jobs=4)

    input_dim = 5
    hidden_dim = 256
    layer_dim = 4
    output_dim = 6
    seq_dim = 128

    lr = 0.0001
    n_epochs = 100
    # iterations_per_epoch = len(trn_dl)
    best_acc = 0
    patience, trials = 10, 0


    model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print('Start model training')

    for epoch in range(1, n_epochs + 1):
        
        for i, (x_batch, y_batch) in enumerate(trn_dl):
            model.train()
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            # print(x_batch.shape)
            # sched.step()
            opt.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            opt.step()
        
        model.eval()
        correct, total = 0, 0
        for x_val, y_val in val_dl:
            x_val, y_val = [t.cuda() for t in (x_val, y_val)]
            out = model(x_val)
            preds = F.log_softmax(out, dim=1).argmax(dim=1)
            total += y_val.size(0)
            correct += (preds == y_val).sum().item()
        
        acc = correct / total

        if epoch % 5 == 0:
            print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Acc.: {acc:2.2%}')

        if acc > best_acc:
            trials = 0
            best_acc = acc
            torch.save(model.state_dict(), 'output/checkpoints/lstm-best-0815.pth')
            print(f'Epoch {epoch} best model saved with loss: {loss}')
            print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
        else:
            trials += 1
            if trials >= patience:
                print(f'Early stopping on epoch {epoch}')
                break


if __name__ == '__main__':

    # training_file = 'dpc-unet-pixel-label-200perPoly-label-0703.pkl'
    training_file_t2 = '/home/geoint/tri/Planet_khuong/output/csv/training-pixel-tile02-label-tile02-2000p-label-0812-2.csv'

    training_file_t1 = '/home/geoint/tri/Planet_khuong/output/csv/training-pixel-tile01-label-2000p-label-0815.csv'
    train([training_file_t1, training_file_t2])