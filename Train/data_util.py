import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
import torch

def ensure_dir(file_path):
    '''
    检查当前目录是否存在
    :param file_path:
    :return:
    '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

#----------------------------------------Dataset-------------------------------------------
class peptide(Dataset):
    '''
    定义Dataset类型数据集
    '''
    def __init__(self, x, y, m, l, task):
        self.x = torch.LongTensor(x)
        if y is not None:
            self.y = torch.FloatTensor(y)
        else:
            self.y = None
        self.m = torch.FloatTensor(m)
        self.l = torch.FloatTensor(l)
        if task is not None:
            self.task = torch.FloatTensor(task)
        else:
            self.task = None

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.m[idx], self.l[idx], self.task[idx]

    def __len__(self):
        return len(self.x)

#-----------------------------------------------------------------------------------------
def int_dataset(dat, timesteps, middle = True):
    '''
    将氨基酸序列信息整合到序列中， middle用来控制蛋白质信息居中。
    :param dat:
    :param timesteps:
    :param middle:
    :return:
    '''
    DUMMY = 22
    oh_dat = (np.ones([len(dat), timesteps, 1], dtype = np.int32)*DUMMY).astype(np.int32)
    cnt = 0
    for c, row in dat.iterrows():
        ie = np.array(row['encseq'])
        oe = ie.reshape(len(ie), 1)
        if middle:
            oh_dat[cnt, ((60 - oe.shape[0]) // 2): ((60 - oe.shape[0]) // 2) + oe.shape[0], :] = oe
        else:
            oh_dat[cnt, 0:oe.shape[0], :] = oe
        cnt += 1
    return oh_dat

#-----------------------------------------------------------------------------------------
def get_data_set(model_params):
    '''
    用来生成训练集、测试集。
    :param model_params:
    :return:
    '''

    # 读取数据
    data = pd.read_pickle(model_params['train_file'])
    data['Modified_sequence'] = data['Modified_sequence'].str.replace('_','')
    data = data.sample(frac = 1).reset_index(drop = True)
    data_test = pd.read_pickle(model_params['test_file'])
    test_from = len(data)
    data = pd.concat([data, data_test], ignore_index = True, sort = False)
    training_idx = data[:test_from].index.astype(int)
    test_idx = data[test_from:].index.astype(int)

    # if model_params['num_classes'] == 1:
    #     # lab_name = label
    #     lab = data[model_params['lab_name']].values
    #     min_lab = data['minval']
    #     max_lab = data['maxval']
    #     scale(lab, min_lab, max_lab)

    # 打包数据 np 'numpy.ndarray'
    dtest = data.iloc[test_idx]
    dtrain = data.iloc[training_idx]

    data['lens'] = data['Modified_sequence'].str.len()
    data_train = data.iloc[training_idx]
    data_test = data.iloc[test_idx]


    # 序列数据 (batch, timesteps, 1) -> (batch, timesteps)
    # train_x = int_dataset(data_train, model_params['timesteps'], model_params['num_input']).reshape(-1, 1)
    # test_x = int_dataset(data_test, model_params['timesteps'], model_params['num_input']).reshape(-1, 1)
    # # train_x = np.squeeze(train_x)
    # # test_x = np.squeeze(test_x)
    one_data = int_dataset(data, model_params['timesteps'], model_params['num_input'])
    print(one_data.shape)

    train_x = one_data[training_idx, :]
    test_x = one_data[test_idx, :]
    train_x = np.squeeze(train_x)
    test_x = np.squeeze(test_x)


    # CCS数据
    train_y = data_train[model_params['lab_name']].values.reshape(-1, 1)
    test_y = data_test[model_params['lab_name']].values.reshape(-1, 1)

    # Charge数据
    train_c = data_train['Charge'].values.reshape(-1, 1)
    test_c = data_test['Charge'].values.reshape(-1, 1)

    # 长度数据
    train_l = data_train['lens'].values.reshape(-1, 1)
    test_l = data_test['lens'].values.reshape(-1, 1)
    train_l = np.squeeze(train_l)
    test_l = np.squeeze(test_l)

    # task数据
    train_t = data_train['task'].values.reshape(-1, 1)
    test_t = data_test['task'].values.reshape(-1, 1)

    train_dataset = peptide(train_x, train_y, train_c, train_l, train_t)
    test_dataset = peptide(test_x, test_y, test_c, test_l, test_t)

    train_size, test_size = test_from, (len(data) - test_from)
    print('done generating data train_size:', train_size,'test_size' ,test_size)


    print('x size:', train_x.shape)
    print('y size:', train_y.shape)
    print('c size:', train_c.shape)
    print('l size:', train_l.shape)


    return train_dataset, test_dataset


