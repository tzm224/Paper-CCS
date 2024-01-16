import pickle
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
import numpy as np



#-----------------------------------------------------------------------------------------
def ensure_dir(file_path):
    '''
    检查当前目录是否存在
    :param file_path:
    :return:
    '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

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
def split(data, name, label_encoder_path='../data/enc.pickle'):
    '''
    得到需要处理的数据
    :param data:
    :param name:
    :param s:
    :param label_encoder_path:
    :return:
    '''

    ensure_dir(name)

    # 读取编码方式并对序列进行编码处理
    with open(label_encoder_path, 'rb') as handle:
        label_encoder = pickle.load(handle)
    data['encseq'] = data['Modified_sequence'].apply(lambda x: label_encoder.transform(list(x)))

    data['minval'] = 275.418854
    data['maxval'] = 1118.786133

    data['task'] = 0
    print('Name: ', name, 'Len test: ', len(data[data['test']]), 'Len set test: ',
          len(set(data[data['test']])), 'Len not test: ', len(data[~data['test']]), 'Len set not test: ',
          len(set(data[~data['test']])))

    name = os.path.dirname(name)
    data.to_pickle(os.path.join(name, 'predict_data.pkl'))
    return data
#-----------------------------------------------------------------------------------------
class peptide(Dataset):
    '''
    定义Dataset类型数据集
    '''
    def __init__(self, x, m, l, task):
        self.x = torch.LongTensor(x)
        self.m = torch.FloatTensor(m)
        self.l = torch.FloatTensor(l)
        if task is not None:
            self.task = torch.FloatTensor(task)
        else:
            self.task = None

    def __getitem__(self, idx):
        return self.x[idx], self.m[idx], self.l[idx], self.task[idx]

    def __len__(self):
        return len(self.x)

#-----------------------------------------------------------------------------------------
def get_predict_data_set(predict_model_params):
    '''
    用来生成数据集
    :param model_params:
    :return:
    '''
    # 读取数据
    data = pd.read_pickle(predict_model_params['file_path'])

    # 处理数据并将处理好的数据库保存
    data = split(data, predict_model_params['model_dir'])

    data['Modified_sequence'] = data['Modified_sequence'].str.replace('_','')
    data = data.sample(frac = 1).reset_index(drop = True)
    data['lens'] = data['Modified_sequence'].str.len()


    # 序列数据 (batch, timesteps, 1) -> (batch, timesteps)
    one_data = int_dataset(data, predict_model_params['timesteps'], predict_model_params['num_input']).reshape(-1, 1)
    print(one_data.shape)
    one_data = one_data.reshape(-1, 66, 1)
    data_x = np.squeeze(one_data)

    # 预测ccs
    data['CCS'] = 0

    # Charge数据
    data_c = data['Charge'].values.reshape(-1, 1)

    # 长度数据
    data_l = data['lens'].values.reshape(-1, 1)
    data_l = np.squeeze(data_l)

    # task数据
    data_t = data['task'].values.reshape(-1, 1)

    dataset = peptide(data_x, data_c, data_l, data_t)
    print('x size: ', data_x.shape)
    print('c size: ', data_c.shape)
    print('l size: ', data_l.shape)
    print('data size: ', data.shape)
    return dataset, data