#!/usr/bin/env python
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


# def get_position_encoding(seq_len, embed):
#     pe = np.array([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(seq_len)])  # 公式实现
#     pe[:, 0::2] = np.sin(pe[:, 0::2])
#     pe[:, 1::2] = np.cos(pe[:, 1::2])
#     return pe
# pe = get_position_encoding(100, 100)
# sns.heatmap(pe)
# plt.xlabel('emb')
# plt.ylabel('seq_len')
# plt.show()


class Positional_Encoding(nn.Module):
    """
    params: embed-->word embedding dim      pad_size-->max_sequence_lenght
    Input: x
    Output: x + position_encoder
    """

    def __init__(self, embed, pad_size, dropout):
        super(Positional_Encoding, self).__init__()
        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])  # 偶数sin
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])  # 奇数cos
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 单词embedding与位置编码相加，这两个张量的shape一致
        out = x + nn.Parameter(self.pe, requires_grad=False).cuda()
        out = self.dropout(out)
        return out


class Multi_Head_Attention(nn.Module):
    """
    params: dim_model-->hidden dim      num_head
    """

    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0  # head数必须能够整除隐层大小
        self.dim_head = dim_model // self.num_head  # 按照head数量进行张量均分
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)  # Q，通过Linear实现张量之间的乘法，等同手动定义参数W与之相乘
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)  # 自带的LayerNorm方法

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1,
                   self.dim_head)  # reshape to batch*head*sequence_length*(embedding_dim//head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 根号dk分之一，对应Scaled操作
        context = self.attention(Q, K, V, scale)  # Scaled_Dot_Product_Attention计算
        context = context.view(batch_size, -1, self.dim_head * self.num_head)  # reshape 回原来的形状
        out = self.fc(context)  # 全连接
        out = self.dropout(out)
        out = out + x  # 残差连接,ADD
        out = self.layer_norm(out)  # 对应Norm
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product'''

    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # Q*K^T
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)  # 两层全连接
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class ConfigTrans(object):
    """配置参数"""
    def __init__(self):
        self.model_name = 'Transformer'
        self.dropout = 0.1
        self.num_classes = 8  # 类别数
        self.pad_size = 512  # 长度(短填长切)，这个根据自己的数据集而定
        self.embed = 1  # 字向量维度
        self.dim_model = 2  # 需要与embed一样
        self.hidden = 1024
        # self.last_hidden = 512
        self.num_head = 1  # 多头注意力，注意需要整除
        self.num_encoder = 2  # 使用两个Encoder，尝试6个encoder发现存在过拟合，毕竟数据集量比较少（10000左右），可能性能还是比不过LSTM


config = ConfigTrans()


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])  # 多次Encoder

    def forward(self, x):
        out = self.postion_embedding(x)
        for encoder in self.encoders:
            out = encoder(out)
        return out


class RNN(nn.Module):
    def __init__(self, d_in, d_out, n_layers=1, bi=True, dropout=0.2, n_to_1=False):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=d_in, hidden_size=d_out, bidirectional=bi, num_layers=n_layers, dropout=dropout)
        self.n_layers = n_layers
        self.d_out = d_out
        self.n_directions = 2 if bi else 1
        self.n_to_1 = n_to_1

    def forward(self, x, x_len):
        x_packed = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        rnn_enc = self.rnn(x_packed)
        if self.n_to_1:
            # hiddenstates, h_n, only last layer
            h_n = rnn_enc[1][0]  # (ND*NL, BS, dim)
            batch_size = x.shape[0]
            h_n = h_n.view(self.n_layers, self.n_directions, batch_size, self.d_out)  # (NL, ND, BS, dim)
            last_layer = h_n[-1].permute(1, 0, 2)  # (BS, ND, dim)
            x_out = last_layer.reshape(batch_size, self.n_directions * self.d_out)  # (BS, ND*dim)

        else:
            x_out = rnn_enc[0]
            x_out = pad_packed_sequence(x_out, total_length=x.size(1), batch_first=True)[0]

        return x_out


class OutLayer(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=.0, bias=.0):
        super(OutLayer, self).__init__()
        self.fc_1 = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(True), nn.Dropout(dropout))
        self.fc_2 = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.fc_2.bias.data, bias)

    def forward(self, x):
        y = self.fc_2(self.fc_1(x))
        return y


class Facellstm(nn.Module):
    def __init__(self, d_in, d_rnn, rnn_n_layers, rnn_bi, rnn_dropout, n_to_1=False):
        super(Facellstm, self).__init__()
        self.inp = nn.Linear(d_in, d_rnn, bias=False)

        self.rnn_n_layers = rnn_n_layers
        if rnn_n_layers > 0:
            self.rnn = RNN(d_rnn, d_rnn, n_layers=rnn_n_layers, bi=rnn_bi,
                           dropout=rnn_dropout, n_to_1=n_to_1)

        # d_rnn_out = d_rnn * 2 if rnn_bi and rnn_n_layers > 0 else d_rnn
        # self.out = OutLayer(d_rnn_out, d_fc_out, n_targets, dropout=linear_dropout)
        # self.final_activation = nn.Softmax(dim=1)

    def forward(self, x, x_len):
        x = self.inp(x)
        if self.rnn_n_layers > 0:
            x = self.rnn(x, x_len)
        # y = self.out(x)
        return x  # self.final_activation(y)

    def set_n_to_1(self, n_to_1):
        self.rnn.n_to_1 = n_to_1


class Poselstm(nn.Module):
    def __init__(self, d_in, d_rnn, rnn_n_layers, rnn_bi, rnn_dropout, n_to_1=False):
        super(Poselstm, self).__init__()
        self.inp = nn.Linear(d_in, d_rnn, bias=False)

        self.rnn_n_layers = rnn_n_layers
        if rnn_n_layers > 0:
            self.rnn = RNN(d_rnn, d_rnn, n_layers=rnn_n_layers, bi=rnn_bi,
                           dropout=rnn_dropout, n_to_1=n_to_1)

        # d_rnn_out = d_rnn * 2 if rnn_bi and rnn_n_layers > 0 else d_rnn
        # self.out = OutLayer(d_rnn_out, d_fc_out, n_targets, dropout=linear_dropout)
        # self.final_activation = nn.Softmax(dim=1)

    def forward(self, x, x_len):
        x = self.inp(x)
        if self.rnn_n_layers > 0:
            x = self.rnn(x, x_len)
        # y = self.out(x)
        return x  # self.final_activation(y)

    def set_n_to_1(self, n_to_1):
        self.rnn.n_to_1 = n_to_1


# class Model_(nn.Module):
#     def __init__(self):
#         super(Visuallstm, self).__init__()
#         # self.transformer = Transformer()
#         self.lstm = nn.LSTM(input_size=256, hidden_size=64, bidirectional=True, num_layers=1, dropout=0.2)
#         self.conv = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3, stride=1)
#         self.pool = nn.AvgPool1d(10)
#         self.fc1 = nn.Sequential(nn.Linear(2, 8),
#                                  nn.Softmax(dim=1))
#
#     def forward(self, x):
#         x, (hn, cn) = self.lstm(x)
#         x = self.conv(x)
#         x = self.pool(x).squeeze(2)
#         # x = x.view(x.size(0), -1)  # 将三维张量reshape成二维，然后直接通过全连接层将高维数据映射为classes
#         # out = torch.mean(out, 1)    # 也可用池化来做，但是效果并不是很好
#         out = self.fc1(x)
#         return out


def get_emonet():
    net = torch.load("model_3799.pth")  # wxk debug
    net.emonet.module.predictor.emo_fc_2 = nn.Sequential()
    return net


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.faceencoder = get_emonet()
        self.faceencoder.eval()
        self.facelstm = Facellstm(d_in=256, d_rnn=64, rnn_n_layers=1, rnn_bi=True, rnn_dropout=0.2, n_to_1=True)
        self.poselstm = Poselstm(d_in=36, d_rnn=64, rnn_n_layers=1, rnn_bi=True, rnn_dropout=0.2, n_to_1=True)
        self.fc = nn.Sequential(nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                nn.Dropout(0.1), nn.Linear(128, 8), nn.Softmax(dim=1))

    def forward(self, seqimgs, vlen, poses):
        B = seqimgs.shape[0]
        l = int(vlen[0])
        flattenpose = torch.flatten(poses, 2)  # B*T*36
        seqimgs = seqimgs.view(-1, 3, 256, 256)  # B*T*3*256*256->N*3*256*256
        seqfeat = self.faceencoder(seqimgs)  # N*256
        seqfeat = seqfeat.view(B, l, 256)  # N*256->B*T*256
        seqfacefeat = self.facelstm(seqfeat, vlen)  # B*128
        posefeat = self.poselstm(flattenpose, vlen)  # B*128
        singlefacefeat = seqfeat[:, int(l / 2), :]  # B*256
        allfeat = torch.cat([singlefacefeat, seqfacefeat, posefeat], dim=1)  # B*512
        out = self.fc(allfeat)
        return out


if __name__ == '__main__':
    model = Model().cuda()
    # print(model)
    seqimgs = torch.rand(2, 2, 3, 256, 256).cuda()
    vlen = torch.tensor([2, 2]).cuda()
    poses = torch.rand(2, 2, 18, 2).cuda()
    pred = model(seqimgs, vlen, poses)
    num = torch.argmax(pred, dim=1)
    print(pred, num)
