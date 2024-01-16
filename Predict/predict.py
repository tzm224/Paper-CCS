import os
import json
import sys
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from model import BiRNN
from predict_util import *


if __name__ == '__main__':

# -------------------------------------------------------------------------------------------------
    # 显卡
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
    # device_ids = [0, 1, 2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device: ', device)


# -------------------------------------------------------------------------------------------------
    # 数据集
    model_dir = "../Model_ccs/save_model"
    path = '../Model_ccs/model_params.txt'
    with open(path) as f:
        model_params = json.load(f)

    file_to_predict = '../data/test_data.pkl'
    model_params['file_path'] = file_to_predict

    dataset, data = get_predict_data_set(model_params)

# -------------------------------------------------------------------------------------------------
    # 模型
    model = BiRNN(model_params['num_layers'],
                  model_params['num_hidden'],
                  model_params['num_classes'],
                  model_params['dropout_keep_prob']).to(device)
    # model = nn.DataParallel(model, device_ids).to(device)

    # 读取训练好的模型
    # model.load_state_dict(torch.load(model_params['model_dir']))

# -------------------------------------------------------------------------------------------------

    next_element = DataLoader(dataset)

    for data_x, data_c, data_l, _ in next_element:
        pred = model(data_x, data_c, data_l)
        pred = torch.squeeze(pred)

        data['CCS'] = pred.values

    data.to_csv('./finish_predict_data.csv')


