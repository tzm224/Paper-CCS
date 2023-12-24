import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
class BiRNN(nn.Module):
    def __init__(self, num_layers, num_hidden, num_classes, keep_prob, use_embedding = True, sequence_length = 66, dict_size = 32):
        '''
        双向LSTM + MPL * 2
        :param num_layers: 网络层数
        :param num_hidden: 隐藏层神经元数
        :param num_classes: 输出分类数
        :param keep_prob: dropout率
        :param sequence_length: 序列长度，这里的66是最大值
        :param dict_size: 字典大小，氨基酸种类加上一些符号一共32个
        '''
        super(BiRNN, self).__init__()
        self.use_embedding = use_embedding
        if use_embedding:
            self.emb = nn.Embedding(dict_size, num_hidden)

        self.rnn = nn.LSTM(input_size = num_hidden,
                           hidden_size = num_hidden,
                           num_layers = num_layers,
                           bias = False,
                           batch_first = True,
                           dropout = keep_prob,
                           bidirectional = True
                           )
        self.fc1 = nn.Linear(2 * num_hidden + 1, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_classes)
    def forward(self, x, meta_data):

        seq_len = x['lens']
        x = pad_sequence(x, batch_first = True, padding_value = 0.0)
        x = pack_padded_sequence(x, seq_len, batch_first = True)

        if self.use_embedding:
            x = self.emb(x)

        out, (_, _) = self.rnn(x, (0, 0))

        out = torch.cat(tensors = [out, meta_data], dim = 1)

        out = F.relu(self.fc1(out))

        out = self.fc2(out)

        return out
