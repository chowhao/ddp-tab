# coding: utf-8
import pandas as pd
import torch as th
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# custom dataset handler
class DatasetHandler(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # return self.x[idx, :], self.y[idx]
        return self.x[idx, :], self.y[idx, :]


# convert to tensor
# 将csv转为张量
def data_handler(input_data):
    array_data = np.array(input_data)
    tensor_data = th.tensor(array_data)
    return tensor_data


# load csv file to tensor
def tensor_loader(data_path):
    df_data = pd.read_csv(data_path)
    df_features = df_data.loc[:, ['brake', 'target', 'speed', 'slope']]
    # data_label = data.loc[1:, ['acceleration']]
    df_label = df_data.loc[:, ['acc']]
    # spd_label = df_data.loc[:, ['speed']]
    # convert to tensor
    tensor_features = data_handler(df_features)
    tensor_label = data_handler(df_label)
    return tensor_features, tensor_label
    # return df_features, df_label


# load dataset
# def data_loader(data_path, dis_flag):
#     data_features, data_label = csv_loader(data_path)
def data_loader(tensor_features, tensor_label, dis_flag):
    # data_features, data_label = csv_loader(data_path)
    # tensor_features = data_handler(df_features)
    # tensor_label = data_handler(df_label)
    batch_size = 32

    dataset = DatasetHandler(tensor_features, tensor_label)

    kwargs = {'num_workers': 1, 'pin_memory': True,
              'shuffle': True}

    if dis_flag:
        data_sampler = DistributedSampler(dataset)
        kwargs.update({'sampler': data_sampler,
                       'shuffle': False
                       })

    # return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return DataLoader(dataset, batch_size=batch_size, **kwargs)
