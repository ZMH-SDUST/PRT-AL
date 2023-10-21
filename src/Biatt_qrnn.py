import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from qrnn import QRNN
import torch.nn.functional as F


class qrnn7(nn.Module):
    def __init__(self, hidden_dim, out_size, batch_size, embedding_dim, dataset, col_name_index_dict, min_max,
                 n_layer=1, dropout=0):
        super(qrnn7, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.n_layer = 1
        self.dropout = dropout
        self.dataset = dataset
        self.min_max = min_max
        self.embedding_dim = embedding_dim
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

        elif self.dataset == "Helpdesk":
            self.activity = nn.Embedding(len(col_name_index_dict["ActivityID"]), 10,
                                         padding_idx=0)
        elif self.dataset == "BPIC2012":
            self.Weekday_emb = nn.Embedding(7 + 1, 5, padding_idx=0)
            self.Activity_emb = nn.Embedding(len(col_name_index_dict["Activity"]), 5, padding_idx=0)
            self.Concept_name_emb = nn.Embedding(len(col_name_index_dict["concept:name"]), 5, padding_idx=0)

        self.weight_Mu = nn.Parameter(torch.Tensor(hidden_dim * 2, n_layer))
        # self.weight_Mu = nn.Parameter(torch.Tensor(hidden_dim, n_layer))

        self.rnn = QRNN(input_size=embedding_dim, hidden_size=hidden_dim, dropout=self.dropout,
                        num_layers=self.n_layer, bidirectional=True)

        self.out = nn.Linear(hidden_dim * 2, out_size)
        # self.out = nn.Linear(hidden_dim, out_size)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def attention_net(self, rnn_output):
        attn_weights = torch.matmul(rnn_output, self.weight_Mu)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(rnn_output.transpose(1, 2), soft_attn_weights).squeeze(2)
        return context, soft_attn_weights.data.cpu().numpy()

    def forward(self, x):

        if self.dataset == "factory_coarse":
            time_stamp = x[:, :, self.min_max].float()
            weekday = self.Weekday_emb(x[:, :, 2].long())
            activity = self.activity_emb(x[:, :, 3].long())
            Worker_name = self.Worker_name_emb(x[:, :, 5].long())
            Station_number = self.Station_number_emb(x[:, :, 6].long())
            Encoder_input = torch.cat((time_stamp, weekday, activity, Worker_name, Station_number), 2)
        elif self.dataset == "factory_fine":
            time_stamp = x[:, :, self.min_max].float()
            weekday = self.Weekday_emb(x[:, :, 2].long())
            activity = self.activity_emb(x[:, :, 3].long())
            Worker_name = self.Worker_name_emb(x[:, :, 5].long())
            Station_number = self.Station_number_emb(x[:, :, 6].long())
            Encoder_input = torch.cat((time_stamp, weekday, activity, Worker_name, Station_number), 2)
        elif self.dataset == "Helpdesk":
            time_stamp = x[:, :, self.min_max].float()
            activity_em = self.activity(x[:, :, 2].long())
            Encoder_input = torch.cat((time_stamp, activity_em), 2)
        elif self.dataset == "BPIC2012":
            time_stamp = x[:, :, self.min_max].float()
            weekday = self.Weekday_emb(x[:, :, 3].long())
            Activity = self.Activity_emb(x[:, :, 4].long())
            Concept_name = self.Concept_name_emb(x[:, :, 7].long())
            Encoder_input = torch.cat((time_stamp, weekday, Activity, Concept_name), 2)
        Encoder_input = Encoder_input.float()

        # batch_size, seq_len, embedding_dim
        input = Encoder_input.permute(1, 0, 2)

        # hidden_state = Variable(torch.randn(self.n_layer * 2, self.batch_size, self.hidden_dim))
        hidden_state = Variable(torch.randn(self.n_layer, self.batch_size, self.hidden_dim))

        # output, final_hidden_state = self.rnn(input, hidden_state)
        output, final_hidden_state = self.rnn(input)
        output = output.permute(1, 0, 2)
        output = F.relu(output)
        output, attention = self.attention_net(output)
        output = F.relu(output)
        output = self.out(output)
        return output
