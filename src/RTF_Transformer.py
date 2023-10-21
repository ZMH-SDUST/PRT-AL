# -*- coding: utf-8 -*-
"""
@Time ： 2022/5/5 11:34
@Auther ： Zzou
@File ：RTF_Transformer.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""

import numpy as np
from TCN_model import *


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """ Sinusoid position encoding table """

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # batch_size, word_length, num_head, q/k/v dimension
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # batch_size, num_head,word_length, q/k/v dimension
        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.
            # mask:(16, 1, 1, 17)
        # The q here can be regarded as the input X after self attention weighting
        q, attn = self.attention(q, k, v, mask=mask)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        # Convert to original size
        # add and norm
        q += residual
        q = self.layer_norm(q)
        return q, attn


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    # 512, 2048, 8, 64, 64
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Encoder_rnn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, col_name_index_dict, dataset, min_max):
        super().__init__()
        # fine embedding
        self.dataset = dataset
        self.min_max = min_max
        if self.dataset == "factory_fine":
            self.Weekday_emb = nn.Embedding(7 + 1, 5, padding_idx=0)
            self.Worker_name_emb = nn.Embedding(len(col_name_index_dict["Worker_name"]), 6,
                                                padding_idx=0)
            self.Station_number_emb = nn.Embedding(len(col_name_index_dict["Station_number"]), 6,
                                                   padding_idx=0)
        elif self.dataset == "factory_coarse":
            self.Weekday_emb = nn.Embedding(7 + 1, 5, padding_idx=0)
            self.Worker_name_emb = nn.Embedding(len(col_name_index_dict["Worker_name"]), 6,
                                                padding_idx=0)
            self.Station_number_emb = nn.Embedding(len(col_name_index_dict["Station_number"]), 6,
                                                   padding_idx=0)
        # self.layer1 = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True, bidirectional=False)
        self.layer1 = nn.GRU(input_size, hidden_size, num_layer, batch_first=True, bidirectional=False)

    def forward(self, x):
        if self.dataset == "factory_fine":
            time_stamp = x[:, :, self.min_max]
            weekday = self.Weekday_emb(x[:, :, 2].int())
            Worker_name = self.Worker_name_emb(x[:, :, 4].int())
            Station_number = self.Station_number_emb(x[:, :, 5].int())
            Encoder_input = torch.cat((time_stamp, weekday, Worker_name, Station_number), 2)
        elif self.dataset == "factory_coarse":
            time_stamp = x[:, :, self.min_max]
            weekday = self.Weekday_emb(x[:, :, 2].int())
            Worker_name = self.Worker_name_emb(x[:, :, 4].int())
            Station_number = self.Station_number_emb(x[:, :, 5].int())
            Encoder_input = torch.cat((time_stamp, weekday, Worker_name, Station_number), 2)

        Encoder_input = Encoder_input.float()
        x, _ = self.layer1(Encoder_input)
        return x


class Regressor_pool(nn.Module):
    def __init__(self, sequence_max_length=13, d_embed=64, last_feature_dm=10):
        super().__init__()
        self.fc1 = nn.Linear(sequence_max_length * d_embed, last_feature_dm)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mean_pool = nn.AdaptiveAvgPool1d(1)
        self.fc3 = nn.Linear(d_embed, 10)
        self.bn1 = nn.BatchNorm1d(sequence_max_length * d_embed)
        self.bn2 = nn.BatchNorm1d(d_embed)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = x.float()
        x = x.transpose(1, 2)
        out = self.mean_pool(x)
        out = out.transpose(1, 2).squeeze(1)
        out = F.relu(self.fc3(self.bn2(out)))
        out = F.relu(self.fc2(out))
        return out


class Regressor_resnet(nn.Module):
    def __init__(self, sequence_max_length=13, d_embed=64, last_feature_dm=10):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mean_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(d_embed, 10)
        self.fc2 = nn.Linear(10, d_embed)
        self.fc3 = nn.Linear(d_embed, 1)
        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(d_embed)

    def forward(self, x):
        x = x.float()
        x = x.transpose(1, 2)
        x = self.mean_pool(x)
        x = x.transpose(1, 2).squeeze(1)
        x_res = x
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        x = F.relu(x + x_res)
        x = F.relu(self.fc3(x))
        return x


class Regressor_rnn(nn.Module):
    def __init__(self, num_layer, embedding_dim, hidden_size):
        super().__init__()
        output_size = 1
        # self.layer1 = nn.LSTM(embedding_dim, hidden_size, num_layer, batch_first=True)
        self.layer1 = nn.GRU(embedding_dim, hidden_size, num_layer, batch_first=True)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        x = self.bn(x[:, -1, :])
        x = F.relu(x)
        x = self.layer2(x)
        return x


class Regressor_TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout, max_len):
        super(Regressor_TemporalBlock, self).__init__()

        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(n_outputs, eps=1e-6)

        self.conv2 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.layer_norm2 = nn.LayerNorm(n_outputs, eps=1e-6)

        self.conv3 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.relu3 = nn.ReLU()
        self.layer_norm3 = nn.LayerNorm(n_outputs, eps=1e-6)

        self.conv4 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.relu4 = nn.ReLU()
        self.layer_norm4 = nn.LayerNorm(n_outputs, eps=1e-6)

        self.conv5 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.relu5 = nn.ReLU()
        self.layer_norm5 = nn.LayerNorm(n_outputs, eps=1e-6)

        self.conv_last = weight_norm(
            nn.Conv1d(n_outputs, 1, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.relu_last = nn.ReLU()

        self.init_weights()
        self.max_len = max_len
        self.regressor = nn.Linear(self.max_len, 1)

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        self.conv3.weight.data.normal_(0, 0.01)
        self.conv4.weight.data.normal_(0, 0.01)
        self.conv5.weight.data.normal_(0, 0.01)
        self.conv_last.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = x.transpose(1, 2)
        res = x

        x = self.conv1(x)  # Decoder block 1
        x = self.relu1(x)
        x = x + res
        x = x.transpose(1, 2)
        x = self.layer_norm1(x)
        x = x.transpose(1, 2)

        # x = self.conv2(x)  # Decoder block 2
        # x = self.relu2(x)
        # x = x + res
        # x = x.transpose(1, 2)
        # x = self.layer_norm2(x)
        # x = x.transpose(1, 2)
        #
        # x = self.conv3(x)  # Decoder block 3
        # x = self.relu3(x)
        # x = x + res
        # x = x.transpose(1, 2)
        # x = self.layer_norm3(x)
        # x = x.transpose(1, 2)
        #
        # x = self.conv4(x)  # Decoder block 4
        # x = self.relu4(x)
        # x = x + res
        # x = x.transpose(1, 2)
        # x = self.layer_norm4(x)
        # x = x.transpose(1, 2)
        #
        # x = self.conv5(x)  # Decoder block 5
        # x = self.relu5(x)
        # x = x + res
        # x = x.transpose(1, 2)
        # x = self.layer_norm5(x)
        # x = x.transpose(1, 2)

        x = self.conv_last(x)
        x = self.relu_last(x)
        x = F.relu(self.regressor(x.squeeze(1)))
        return x


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, n_position, d_model, n_layers, n_head, d_k, d_v, dropout, d_inner, dataset_name,
            col_name_index_dict, min_max):

        super().__init__()
        self.min_max = min_max
        self.dataset = dataset_name

        if self.dataset == "factory_fine":
            self.Weekday_emb = nn.Embedding(7 + 1, 5, padding_idx=0)
            self.activity_emb = nn.Embedding(len(col_name_index_dict["Activity_name"]), 6,
                                             padding_idx=0)
            self.Worker_name_emb = nn.Embedding(len(col_name_index_dict["Worker_name"]), 6,
                                                padding_idx=0)
            self.Station_number_emb = nn.Embedding(len(col_name_index_dict["Station_number"]), 6,
                                                   padding_idx=0)
            self.pos_embedding = nn.Parameter(torch.randn(1, 31, d_model))
            self.batch_norm = nn.BatchNorm1d(31)
        elif self.dataset == "factory_coarse":
            self.Weekday_emb = nn.Embedding(7 + 1, 5, padding_idx=0)
            self.activity_emb = nn.Embedding(len(col_name_index_dict["Sub_process_name"]), 6,
                                             padding_idx=0)
            self.Worker_name_emb = nn.Embedding(len(col_name_index_dict["Worker_name"]), 6,
                                                padding_idx=0)
            self.Station_number_emb = nn.Embedding(len(col_name_index_dict["Station_number"]), 6,
                                                   padding_idx=0)
            self.pos_embedding = nn.Parameter(torch.randn(1, 13, d_model))
            self.batch_norm = nn.BatchNorm1d(13)

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(self.d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.dataset_name = dataset_name
        self.position_enc = PositionalEncoding(self.d_model, n_position=n_position)
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

    def forward(self, x, src_mask, return_attns=False):

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
        enc_slf_attn_list = []

        enc_output = Encoder_input + self.pos_embedding
        # enc_output = (self.position_enc(Encoder_input))
        # enc_output = Encoder_input

        enc_output = enc_output.to(torch.float32)

        # layer normalization
        # enc_output = self.batch_norm(enc_output.to(torch.float32))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Transformer_ED(nn.Module):

    def __init__(self, n_layers=1, n_head=1, d_k=16, d_v=16, dropout=0.1, n_position=100, d_inner=32,
                 device=torch.device('cuda'), dataset_name="factory_coarse",
                 col_name_index_dict=dict(), squence_max_length=13, min_max=list(), regressor_type="r1"):
        super().__init__()
        self.regressor_type = regressor_type
        if dataset_name == "factory_coarse":
            d_model = 26
        elif dataset_name == "factory_fine":
            d_model = 26
        self.encoder = Encoder(n_position=n_position,
                               d_model=d_model,
                               n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                               dropout=dropout, d_inner=d_inner, dataset_name=dataset_name,
                               col_name_index_dict=col_name_index_dict, min_max=min_max)

        # self.encoder = Encoder_rnn(d_model, d_model, 1, col_name_index_dict, dataset_name, min_max)
        # self.encoder = TemporalConvNet(num_inputs=d_model, num_channels=[2, 1], kernel_size=2, dropout=0.2,col_name_index_dict=col_name_index_dict, dataset=dataset_name,
        #          min_max=min_max)

        self.src_pad_idx = 0
        self.device = device
        if self.regressor_type == "Regressor_pool":
            self.regressor = Regressor_pool(sequence_max_length=squence_max_length, d_embed=d_model, last_feature_dm=10)
        elif self.regressor_type == "Regressor_TemporalBlock":
            self.regressor = Regressor_TemporalBlock(max_len=squence_max_length, n_inputs=d_model, n_outputs=d_model,
                                                     kernel_size=3, stride=1, dilation=1, padding=1, dropout=0.2)
        elif self.regressor_type == "Regressor_resnet":
            self.regressor = Regressor_resnet(sequence_max_length=squence_max_length, d_embed=d_model,
                                              last_feature_dm=10)
        elif self.regressor_type == "Regressor_rnn":
            self.regressor = Regressor_rnn(num_layer=1, embedding_dim=d_model, hidden_size=10)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, prefix):
        # without src_mask
        mask_matrix = torch.zeros((prefix.shape[0], prefix.shape[1]))
        src_mask = get_pad_mask(mask_matrix, self.src_pad_idx).to(self.device)  # use for encoder
        enc_output, *_ = self.encoder(prefix, src_mask)

        # enc_output = self.encoder(prefix)
        enc_output = self.regressor(enc_output)
        return enc_output


if __name__ == "__main__":
    model = TemporalConvNet(num_inputs=20, num_channels=[2, 1], kernel_size=2, dropout=0.2)
    input = torch.zeros((128, 13, 20))  # batch_size  x text_len x embedding_size
    print(model(input).shape)
