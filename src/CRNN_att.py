import torch
import torch.nn as nn
from qrnn import QRNN
import torch.nn.functional as F


class BiCRNNAtt(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, text_len, batch_size, dataset, col_name_index_dict, min_max, n_layer=1, dropout=0,
                 embedding=None):
        super(BiCRNNAtt, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.n_layer = 1
        self.dropout = dropout
        self.dataset = dataset
        self.min_max = min_max
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

        self.out = nn.Linear(hidden_dim*2, 1)
        # self.out = nn.Linear(hidden_dim, 1)

        self.config = {
            'in_channels': self.embedding_dim,
            'out_channels': self.embedding_dim,
            'max_text_len': text_len,
            'feature_size': self.embedding_dim,
        }
        self.window_sizes = [2]

        # [batch_size, embedding_size, text_len]
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.config['in_channels'],
                                    out_channels=self.config['out_channels'],
                                    kernel_size=h),
                          nn.BatchNorm1d(num_features=self.config['feature_size']),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=self.config['max_text_len'] - h + 1))
            for h in self.window_sizes
        ])

    def attention_net(self, rnn_output):
        # rnn_output = rnn_output.view(self.batch_size,-1,self.hidden_dim * 2)
        # rnn_output = rnn_output.view(self.batch_size, -1, self.hidden_dim)
        rnn_output = rnn_output.transpose(0, 1)
        attn_weights = torch.matmul(rnn_output, self.weight_Mu)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(rnn_output.transpose(1, 2), soft_attn_weights).squeeze(2)
        return context, soft_attn_weights.data.cpu().numpy()  # context : [batch_size, hidden_dim * num_directions(=2)]

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

        # [batch_size, seq_len, embedding_dim]
        attr = Encoder_input

        attr = attr.permute(0, 2, 1)
        # [batch_size, embedding_dim, seq_len]
        out = [conv(attr) for conv in self.convs]  # out[i]:batch_size x feature_size*1

        out = torch.cat(out, dim=1)

        out = out.permute(2, 0, 1) # [1, 128, 12] (seq_len, batch, input_size)

        output, _ = self.rnn(out)

        output, attention = self.attention_net(output)  # context : [batch_size, hidden_dim * num_directions(=2)]
        hn = output
        output = self.out(hn)
        return output

