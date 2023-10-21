# -*- coding: utf-8 -*-
"""
@Time ： 2022/5/4 17:23
@Auther ： Zzou
@File ：processtransformer.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""

import torch.nn as nn
import torch.nn.functional as F
import torch


def pget_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


class pPositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 16, 17, 512
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class pMultiHeadAttention(nn.Module):
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

        self.attention = pScaledDotProductAttention(temperature=d_k ** 0.5)

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
        # q(16, 17, 8*64)
        q = self.dropout(self.fc(q))
        # Convert to original size
        # add and norm
        q += residual
        q = self.layer_norm(q)
        return q, attn


class pScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        # Mask area attention is 0
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class pEncoderLayer(nn.Module):
    """ Compose with two layers """

    # 512, 2048, 8, 64, 64
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(pEncoderLayer, self).__init__()
        self.slf_attn = pMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = pPositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class pRegressor(nn.Module):
    def __init__(self, sequence_max_length=13, d_embed=64):
        super().__init__()
        self.gvp = nn.AdaptiveAvgPool1d(1)
        self.dp1 = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(sequence_max_length, 128)
        self.dp2 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.gvp(x).squeeze()
        x = self.dp1(x)
        x = F.relu(self.fc1(x))
        x = self.dp2(x)
        x = self.fc2(x)
        return x


class pEncoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, d_model, n_layers, n_head, d_k, d_v, dropout, d_inner, dataset_name,
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
        elif self.dataset == "factory_coarse":
            self.Weekday_emb = nn.Embedding(7 + 1, 5, padding_idx=0)
            self.activity_emb = nn.Embedding(len(col_name_index_dict["Sub_process_name"]), 6,
                                             padding_idx=0)
            self.Worker_name_emb = nn.Embedding(len(col_name_index_dict["Worker_name"]), 6,
                                                padding_idx=0)
            self.Station_number_emb = nn.Embedding(len(col_name_index_dict["Station_number"]), 6,
                                                   padding_idx=0)
            self.pos_embedding = nn.Parameter(torch.randn(1, 13, d_model))

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            pEncoderLayer(self.d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

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
        enc_output = enc_output.to(torch.float32)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class processtransformer(nn.Module):

    def __init__(self, n_layers=1, n_head=4, d_k=64, d_v=64, dropout=0.1, d_inner=64,
                 device=torch.device('cuda'), dataset_name="factory_coarse",
                 col_name_index_dict=dict(), squence_max_length=13, min_max=list()):
        super().__init__()
        if dataset_name == "factory_coarse":
            d_model = 26
        elif dataset_name == "factory_fine":
            d_model = 26
        self.encoder = pEncoder(d_model=d_model,
                               n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                               dropout=dropout, d_inner=d_inner, dataset_name=dataset_name,
                               col_name_index_dict=col_name_index_dict, min_max=min_max)
        self.src_pad_idx = 0
        self.device = device
        self.regressor = pRegressor(sequence_max_length=squence_max_length, d_embed=d_model)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, prefix):
        # without src_mask
        mask_matrix = torch.zeros((prefix.shape[0], prefix.shape[1]))
        src_mask = pget_pad_mask(mask_matrix, self.src_pad_idx).to(self.device)  # use for encoder

        enc_output, *_ = self.encoder(prefix, src_mask)
        enc_output = self.regressor(enc_output)
        return enc_output
