# -*- coding: utf-8 -*-
"""
@Time ： 2022/5/6 15:56
@Auther ： Zzou
@File ：TCN_model.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, col_name_index_dict={}, dataset="",
                 min_max=[]):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.dataset = dataset
        self.min_max = min_max
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        if self.dataset == "factory_fine":
            self.Weekday_emb = nn.Embedding(7 + 1, 5, padding_idx=0)
            self.activity_emb = nn.Embedding(len(col_name_index_dict["Activity_name"]), 5,
                                             padding_idx=0)
            self.Worker_name_emb = nn.Embedding(len(col_name_index_dict["Worker_name"]), 5,
                                                padding_idx=0)
            self.Station_number_emb = nn.Embedding(len(col_name_index_dict["Station_number"]), 6,
                                                   padding_idx=0)
            self.max_len = 31
        elif self.dataset == "factory_coarse":
            self.Weekday_emb = nn.Embedding(7 + 1, 5, padding_idx=0)
            self.activity_emb = nn.Embedding(len(col_name_index_dict["Sub_process_name"]), 5,
                                             padding_idx=0)
            self.Worker_name_emb = nn.Embedding(len(col_name_index_dict["Worker_name"]), 5,
                                                padding_idx=0)
            self.Station_number_emb = nn.Embedding(len(col_name_index_dict["Station_number"]), 6,
                                                   padding_idx=0)
            self.max_len = 13

        self.regressor = nn.Linear(self.max_len, 1)

    def forward(self, x):
        if self.dataset == "factory_fine":
            time_stamp = x[:, :, self.min_max]
            weekday = self.Weekday_emb(x[:, :, 2].int())
            activity = self.activity_emb(x[:, :, 3].int())
            Worker_name = self.Worker_name_emb(x[:, :, 5].int())
            Station_number = self.Station_number_emb(x[:, :, 6].int())
            Encoder_input = torch.cat((time_stamp, weekday, activity, Worker_name, Station_number), 2)
        elif self.dataset == "factory_coarse":
            time_stamp = x[:, :, self.min_max]
            weekday = self.Weekday_emb(x[:, :, 2].int())
            activity = self.activity_emb(x[:, :, 3].int())
            Worker_name = self.Worker_name_emb(x[:, :, 5].int())
            Station_number = self.Station_number_emb(x[:, :, 6].int())
            Encoder_input = torch.cat((time_stamp, weekday, activity, Worker_name, Station_number), 2)
        Encoder_input = Encoder_input.float()

        x = Encoder_input.transpose(1, 2)
        x = self.network(x)
        x = F.relu(self.regressor(x.squeeze(1)))
        # x = x.transpose(1,2)

        return x


if __name__ == "__main__":
    model = TemporalConvNet(num_inputs=20, num_channels=[2, 1], kernel_size=2, dropout=0.2)
    input = torch.zeros((128, 13, 20))  # batch_size  x text_len x embedding_size
    print(model(input).shape)
