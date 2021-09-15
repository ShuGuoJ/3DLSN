import torch
from torch import nn
from utils import pad
from torch.nn import functional as F


# 深度可分离 encoder
class Net_Depthwise(nn.Module):
    def __init__(self, in_channel, hidden_size, spectra_dim):
        super().__init__()
        self.encoder = Encoder_Depthwise(in_channel, hidden_size, spectra_dim)

    def forward(self, x):
        batch = x.shape[0]
        h = self.encoder(x)
        h = F.adaptive_max_pool3d(h, (None, 1, 1))
        h = h.view((batch, -1))
        return h


# 深度可分离 decoder
class Encoder_Depthwise(nn.Module):
    def __init__(self, in_channel, hidden_size, dim):
        '''
        :param in_channel: 表示输入的特征通道数
        :param hidden_size: 表示隐藏层的神经元个数
        :param dim: 表示光谱维的长度
        '''
        super().__init__()
        self.dim = dim
        m = list()
        padding_0 = pad(dim, 5, 3)
        m.append(nn.Conv3d(in_channel, in_channel, (5, 3, 3), (3, 1, 1), padding=(padding_0, 0, 0), groups=in_channel))
        m.append(nn.Conv3d(in_channel, hidden_size, 1, 1))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(hidden_size))
        dim_1 = (dim - 5 + 2 * padding_0) // 3 + 1
        padding_1 = pad(dim_1, 5, 3)
        m.append(nn.Conv3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_1, 0, 0), groups=hidden_size))
        m.append(nn.Conv3d(hidden_size, hidden_size, 1, 1))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(hidden_size))
        dim_2 = (dim_1 - 5 + 2 * padding_1) // 3 + 1
        padding_2 = pad(dim_2, 5, 3)
        m.append(nn.Conv3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_2, 0, 0), groups=hidden_size))
        m.append(nn.Conv3d(hidden_size, hidden_size, 1, 1))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(hidden_size))
        self.out_dim = (dim_2 - 5 + 2 * padding_2) // 3 + 1

        self.encoder = nn.Sequential(*m)

    def forward(self, x):
        assert self.dim == x.shape[2]
        return self.encoder(x)


# 3层paviaU, patch_size: 7x7
class Encoder(nn.Module):
    def __init__(self, in_channel, hidden_size, dim):
        '''
        :param in_channel: 表示输入的特征通道数
        :param hidden_size: 表示隐藏层的神经元个数
        :param dim: 表示光谱维的长度
        '''
        super().__init__()
        self.dim = dim
        m = list()
        padding_0 = pad(dim, 5, 3)
        m.append(nn.Conv3d(in_channel, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_0, 0, 0)))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(hidden_size))
        dim_1 = (dim - 5 + 2 * padding_0) // 3 + 1
        padding_1 = pad(dim_1, 5, 3)
        m.append(nn.Conv3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_1, 0, 0)))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(hidden_size))
        dim_2 = (dim_1 - 5 + 2 * padding_1) // 3 + 1
        padding_2 = pad(dim_2, 5, 3)
        m.append(nn.Conv3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_2, 0, 0)))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(hidden_size))
        self.out_dim = (dim_2 - 5 + 2 * padding_2) // 3 + 1

        self.encoder = nn.Sequential(*m)

    def forward(self, x):
        assert self.dim == x.shape[2]
        return self.encoder(x)


# 3层paviaU, patch_size: 7x7
class Net(nn.Module):
    def __init__(self, in_channel, hidden_size, dim):
        super().__init__()
        self.encoder = Encoder(in_channel, hidden_size, dim)

    def forward(self, x):
        batch = x.shape[0]
        h = self.encoder(x)
        h = F.adaptive_max_pool3d(h, (None, 1, 1))
        h = h.view((batch, -1))
        return h



# input = torch.rand((1,1,103,7,7))
# net = Net_Depthwise(1, 128, 103)
# net.eval()
# out = net(input)
# print(out.shape)



