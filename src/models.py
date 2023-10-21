# -*- coding: utf-8 -*-
"""
@Time ： 2022/5/5 11:16
@Auther ： Zzou
@File ：models.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_size=64, hidden_size=10, output_size=1, num_layer=1, col_name_index_dict=dict(), BI=True,
                 ATT=True, Node='lstm', min_max=list(), dataset=""):
        super(RNN, self).__init__()
        self.att = ATT
        self.node = Node
        self.bi = BI
        self.min_max = min_max
        self.dataset = dataset
        if self.att and self.bi and self.node == "lstm":
            self.layer1 = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True, bidirectional=True)
        elif self.att and self.bi and self.node == "gru":
            self.layer1 = nn.GRU(input_size, hidden_size, num_layer, batch_first=True, bidirectional=True)
        elif self.att and not self.bi and self.node == "lstm":
            self.layer1 = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True)
        elif self.att and not self.bi and self.node == "gru":
            self.layer1 = nn.GRU(input_size, hidden_size, num_layer, batch_first=True)
        elif not self.att and self.bi and self.node == "lstm":
            self.layer1 = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True, bidirectional=True)
        elif not self.att and self.bi and self.node == "gru":
            self.layer1 = nn.GRU(input_size, hidden_size, num_layer, batch_first=True, bidirectional=True)
        elif not self.att and not self.bi and self.node == "lstm":
            self.layer1 = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True)
        elif not self.att and not self.bi and self.node == "gru":
            self.layer1 = nn.GRU(input_size, hidden_size, num_layer, batch_first=True)
        if self.bi:
            self.layer2 = nn.Linear(hidden_size * 2, output_size)
            self.bn = nn.BatchNorm1d(hidden_size * 2)
            self.weight_Mu = nn.Parameter(torch.Tensor(hidden_size * 2, num_layer).cuda()).cuda()
        else:
            self.layer2 = nn.Linear(hidden_size, output_size)
            self.bn = nn.BatchNorm1d(hidden_size)
            self.weight_Mu = nn.Parameter(torch.Tensor(hidden_size, num_layer).cuda()).cuda()
        self.dropout = nn.Dropout(p=0.1)

        # fine embedding
        if self.dataset == "factory_fine":
            self.Weekday_emb = nn.Embedding(7 + 1, 5, padding_idx=0)
            self.activity_emb = nn.Embedding(len(col_name_index_dict["Activity_name"]), 6,
                                             padding_idx=0)
            self.Worker_name_emb = nn.Embedding(len(col_name_index_dict["Worker_name"]), 6,
                                                padding_idx=0)
            self.Station_number_emb = nn.Embedding(len(col_name_index_dict["Station_number"]), 6,
                                                   padding_idx=0)
        elif self.dataset == "factory_coarse":
            self.Weekday_emb = nn.Embedding(7 + 1, 5, padding_idx=0)
            self.activity_emb = nn.Embedding(len(col_name_index_dict["Sub_process_name"]), 6,
                                             padding_idx=0)
            self.Worker_name_emb = nn.Embedding(len(col_name_index_dict["Worker_name"]), 6,
                                                padding_idx=0)
            self.Station_number_emb = nn.Embedding(len(col_name_index_dict["Station_number"]), 6,
                                                   padding_idx=0)

    def attention_net(self, rnn_output):
        attn_weights = torch.matmul(rnn_output, self.weight_Mu).cuda()

        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(rnn_output.transpose(1, 2), soft_attn_weights).squeeze(2).cuda()
        return context, soft_attn_weights.data.cpu().numpy()
        # context : [batch_size, hidden_dim * num_directions(=2)]

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

        x, _ = self.layer1(Encoder_input)
        if self.att:
            x, attention = self.attention_net(x)
            x = self.bn(x)
        else:
            x = self.bn(x[:, -1, :])
        x = F.relu(x)
        x = self.layer2(x)
        return x
