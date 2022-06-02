from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch


# name
name = 'Wine dataset'
name_saving = 'wine'


# system and simulation parameters
nmc = 300
ndata_min = 800
ndata_max = 1200
dimx = 11
num_samples = 100
warmup_steps = 50
datasizes = torch.linspace(ndata_min, ndata_max, 2, dtype=int)

df=pd.read_csv('datasets/winequality-red.csv', sep=';')

X = pd.DataFrame(data=df[list(df.columns)[:-1]]).values
y = pd.DataFrame(data=df['quality']).values


def get_data(ndata):

    # divide the data into training and testing set
    X_data, X_test, y_data, f_true = train_test_split(X, y,
                                                      test_size=1 - ndata/1599,
                                                        stratify=y)
    y_data = y_data.reshape(y_data.shape[0],)
    f_true = f_true.reshape(f_true.shape[0],)
    return X_data, y_data, X_test, f_true


# set upper and lower bounds for unifor hyperprior
ub = 1e1*torch.ones(dimx+2)
ub[0] = 1e2
ub[-1] = 1
lb = 1e-2*torch.ones(dimx+2)
lb[-1] = 1e-1

