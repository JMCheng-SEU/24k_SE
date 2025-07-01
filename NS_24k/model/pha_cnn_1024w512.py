import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np


class SVConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, kernels,
                 stride=1, padding=0, dilation=1, use_bias=True, batch_norm=False, act=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if (type(kernel_size) is not int) else (kernel_size, kernel_size)
        self.kernels = kernels
        self.stride = stride if (type(stride) is not int) else (stride, stride)
        self.padding = padding if (type(padding) is not int) else (padding, padding)
        self.dilation = dilation if (type(dilation) is not int) else (dilation, dilation)
        self.dilated_kernel_size = [(self.kernel_size[i] - 1) * self.dilation[i] + 1 for i in range(2)]
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.act = act
        self.kernel = nn.Parameter(torch.FloatTensor(out_channels, in_channels, kernels, *self.kernel_size),
                                         requires_grad=True)
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_channels, 1, kernels), requires_grad=True)
        if batch_norm:
            self.norm = nn.BatchNorm2d(out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.kernel, a=np.sqrt(5))
        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        assert x.ndim == 4
        x = torch.constant_pad_nd(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        x_stride = x.stride()
        x_shape = x.shape
        shape = (*x_shape[: -2], (x_shape[-2] - self.dilated_kernel_size[0]) // self.stride[0] + 1,
                 (x_shape[-1] - self.dilated_kernel_size[1]) // self.stride[1] + 1, *self.kernel_size)
        strides = (*x_stride[: -2], x_stride[-2] * self.stride[0], x_stride[-1] * self.stride[1],
                   x_stride[-2] * self.dilation[0], x_stride[-1] * self.dilation[1])
        x = torch.as_strided(x, size=shape, stride=strides)
        x = torch.einsum('bitfmn,oifmn->botf', x, self.kernel)
        if self.use_bias:
            x += self.bias
        if self.batch_norm:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SVConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, kernels, stride=1, padding=0, dilation=1, bias=True, batch_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if (type(kernel_size) is not int) else (kernel_size, kernel_size)
        self.kernels = kernels
        self.stride = stride if (type(stride) is not int) else (stride, stride)
        self.padding = padding if (type(padding) is not int) else (padding, padding)
        self.dilation = dilation if (type(dilation) is not int) else (dilation, dilation)
        self.dilated_kernel_size = [(self.kernel_size[i] - 1) * self.dilation[i] + 1 for i in range(2)]
        self.output_padding = [self.dilated_kernel_size[i] - self.padding[i] - 1 for i in range(2)]
        self.bias = bias
        self.conv = SVConv2d(in_channels, out_channels, kernel_size, kernels, 1, self.output_padding, self.dilation, bias, batch_norm=batch_norm)

    def forward(self, x):
        assert x.ndim == 4
        y = torch.zeros((x.shape[0], x.shape[1], x.shape[2] + (self.stride[0] - 1) * (x.shape[2] - 1), x.shape[3] + (self.stride[1] - 1) * (x.shape[3] - 1)),
                        dtype=x.dtype, device=x.device)
        y[:, :, ::self.stride[0], ::self.stride[1]] = x
        return self.conv(y)




class GLayerNorm2d(nn.Module):

    def __init__(self, in_channel, eps=1e-12):
        super(GLayerNorm2d, self).__init__()
        self.eps = eps
        self.beta = nn.Parameter(torch.ones([1, in_channel, 1, 1]))
        self.gamma = nn.Parameter(torch.zeros([1, in_channel, 1, 1]))

    def forward(self, inputs):
        mean = torch.mean(inputs, [1, 2, 3], keepdim=True)
        var = torch.var(inputs, [1, 2, 3], keepdim=True)
        # mean = torch.mean(inputs, [1, 3], keepdim=True)
        # var = torch.var(inputs, [1, 3], keepdim=True)
        outputs = (inputs - mean) / torch.sqrt(var + self.eps) * self.beta + self.gamma
        return outputs


class FCNN_phafullconv_SVConv2d(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN_phafullconv_SVConv2d, self).__init__()

        self.conv1 = nn.Sequential(
            SVConv2d(in_channels=2, out_channels=4, kernel_size=(1, 3), kernels=257, stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),

        )
        self.conv2 = nn.Sequential(
            SVConv2d(in_channels=4, out_channels=4, kernel_size=(1, 3), kernels=129, stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),

        )
        self.conv3 = nn.Sequential(
            SVConv2d(in_channels=4, out_channels=8, kernel_size=(1, 3), kernels=65, stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.conv4 = nn.Sequential(
            SVConv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), kernels=33, stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.conv5 = nn.Sequential(
            SVConv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), kernels=16, stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),

        )

        self.conv_mid = nn.Sequential(
            SVConv2d(in_channels=16, out_channels=16, kernel_size=1, kernels=16, stride=1),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),

        )




        # Decoder for real

        self.convT1 = nn.Sequential(
            SVConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), kernels=33, stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.convT2 = nn.Sequential(
            SVConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), kernels=65, stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.convT3 = nn.Sequential(
            SVConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), kernels=129, stride=(1, 2)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),

        )
        self.convT4 = nn.Sequential(
            SVConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), kernels=257, stride=(1, 2)),
            GLayerNorm2d(in_channel=1),
            nn.PReLU(),

        )


        self.GRU_real = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)

        self.fcs = nn.Sequential(
            nn.Linear(257, 513),
        )
        # Decoder for imag


        self.phaconvT1 = nn.Sequential(
            SVConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), kernels=33, stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),
        )
        self.phaconvT2 = nn.Sequential(
            SVConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), kernels=65, stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),
        )
        self.phaconvT3 = nn.Sequential(
            SVConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), kernels=129, stride=(1, 2)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),
        )
        self.phaconvT4 = nn.Sequential(
            SVConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(1, 3), kernels=257, stride=(1, 2)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),
        )
        self.phaconvT5 = nn.Sequential(
            SVConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), kernels=513, stride=(1, 2)),
            GLayerNorm2d(in_channel=1),
            nn.Tanh(),
        )

        # self.GRU_imag = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)
        # self.phafcs = nn.Sequential(
        #     nn.Linear(257, 513),
        # )

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def forward(self, x):
        # conv
        # (B, in_c, T, F)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        output = self.conv_mid(x5)

        # ConvTrans for real
        res = torch.cat((output, x5), 1)
        res1 = self.convT1(res)
        res1 = torch.cat((res1[:, :, :, :32], x4[:, :, :, :32]), 1)
        res2 = self.convT2(res1)
        res2 = torch.cat((res2[:, :, :, :64], x3[:, :, :, :64]), 1)
        res3 = self.convT3(res2)
        res3 = torch.cat((res3[:, :, :, :128], x2[:, :, :, :128]), 1)
        res4 = self.convT4(res3)

        res5 = res4.squeeze(1)
        res5, _ = self.GRU_real(res5)
        res5 = self.fcs(res5)
        res5 = res5.unsqueeze(1)

        # ConvTrans for imag
        phares1 = self.phaconvT1(res)
        phares1 = torch.cat((phares1[:, :, :, :32], x4[:, :, :, :32]), 1)
        phares2 = self.phaconvT2(phares1)
        phares2 = torch.cat((phares2[:, :, :, :64], x3[:, :, :, :64]), 1)
        phares3 = self.phaconvT3(phares2)
        phares3 = torch.cat((phares3[:, :, :, :128], x2[:, :, :, :128]), 1)
        phares4 = self.phaconvT4(phares3)
        phares4 = torch.cat((phares4[:, :, :, :256], x1[:, :, :, :256]), 1)
        phares5 = self.phaconvT5(phares4)

        # phares5 = phares4.squeeze(1)
        #
        # phares5, _ = self.GRU_imag(phares5)
        # phares5 = self.phafcs(phares5)
        # phares5 = phares5.unsqueeze(1)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5


class FCNN_midGRU_withPRELU(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN_midGRU_withPRELU, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),

        )

        self.conv_mid_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, stride=1),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),
        )

        self.GRU_mid = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

        self.conv_mid_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=1),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),
        )

        # Decoder for real

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),

        )
        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=1),
            nn.PReLU(),

        )


        self.GRU_real = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)

        self.fcs = nn.Sequential(
            nn.Linear(257, 513),
        )
        # Decoder for imag


        self.phaconvT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),
        )
        self.phaconvT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),
        )
        self.phaconvT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),
        )
        self.phaconvT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),
        )
        self.phaconvT5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=1),
            nn.Tanh(),
        )

        # self.GRU_imag = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)
        # self.phafcs = nn.Sequential(
        #     nn.Linear(257, 513),
        # )

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def forward(self, x):
        # conv
        # (B, in_c, T, F)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        mid_in = self.conv_mid_1(x5)

        mid_GRU_in = mid_in.permute(0, 2, 1, 3)
        mid_GRU_in = mid_GRU_in.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], -1)
        mid_GRU_out, _ = self.GRU_mid(mid_GRU_in)
        mid_GRU_out = mid_GRU_out.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], 8, -1)
        mid_GRU_out = mid_GRU_out.permute(0, 2, 1, 3)

        mid_out = self.conv_mid_2(mid_GRU_out)



        # ConvTrans for real
        res = torch.cat((mid_out, x5), 1)
        res1 = self.convT1(res)
        res1 = torch.cat((res1[:, :, :, :32], x4[:, :, :, :32]), 1)
        res2 = self.convT2(res1)
        res2 = torch.cat((res2[:, :, :, :64], x3[:, :, :, :64]), 1)
        res3 = self.convT3(res2)
        res3 = torch.cat((res3[:, :, :, :128], x2[:, :, :, :128]), 1)
        res4 = self.convT4(res3)

        res5 = res4.squeeze(1)
        res5, _ = self.GRU_real(res5)
        res5 = self.fcs(res5)
        res5 = res5.unsqueeze(1)

        # ConvTrans for imag
        phares1 = self.phaconvT1(res)
        phares1 = torch.cat((phares1[:, :, :, :32], x4[:, :, :, :32]), 1)
        phares2 = self.phaconvT2(phares1)
        phares2 = torch.cat((phares2[:, :, :, :64], x3[:, :, :, :64]), 1)
        phares3 = self.phaconvT3(phares2)
        phares3 = torch.cat((phares3[:, :, :, :128], x2[:, :, :, :128]), 1)
        phares4 = self.phaconvT4(phares3)
        phares4 = torch.cat((phares4[:, :, :, :256], x1[:, :, :, :256]), 1)
        phares5 = self.phaconvT5(phares4)

        # phares5 = phares4.squeeze(1)
        #
        # phares5, _ = self.GRU_imag(phares5)
        # phares5 = self.phafcs(phares5)
        # phares5 = phares5.unsqueeze(1)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5




class FCNN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.bn5 = nn.BatchNorm2d(num_features=128)

        self.conv_mid = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1)
        self.bn_mid = nn.BatchNorm2d(num_features=128)




        # self.fcs_mid = nn.Sequential(
        #     nn.Linear(2048, 2048),
        #     nn.ELU()
        # )

        # Decoder for real



        self.convT1 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.bnT1 = nn.BatchNorm2d(num_features=64)
        self.convT2 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.bnT2 = nn.BatchNorm2d(num_features=32)
        self.convT3 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.bnT3 = nn.BatchNorm2d(num_features=16)
        # output_padding为1，不然算出来是79
        self.convT4 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.bnT4 = nn.BatchNorm2d(num_features=8)
        self.convT5 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.bnT5 = nn.BatchNorm2d(num_features=1)

        self.fcs = nn.Sequential(
            nn.Linear(513, 513),
            nn.Tanh(),
        )

        # Decoder for imag


        self.phaconvT1 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT1 = nn.BatchNorm2d(num_features=64)
        self.phaconvT2 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT2 = nn.BatchNorm2d(num_features=32)
        self.phaconvT3 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT3 = nn.BatchNorm2d(num_features=16)
        # output_padding为1，不然算出来是79
        self.phaconvT4 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT4 = nn.BatchNorm2d(num_features=8)
        self.phaconvT5 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT5 = nn.BatchNorm2d(num_features=1)

        self.phafcs = nn.Sequential(
            nn.Linear(513, 513),
            nn.Tanh(),
        )

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def forward(self, x):
        # conv
        # (B, in_c, T, F)


        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = F.relu(self.bn5(self.conv5(x4)))

        output = F.relu(self.bn_mid(self.conv_mid(x5)))

        # out5 = x5.permute(0, 2, 1, 3)
        # out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        # # # # lstm
        # # #
        # # # lstm, (hn, cn) = self.LSTM1(out5)
        # # # # # reshape
        # # # output = lstm.reshape(lstm.size()[0], lstm.size()[1], 256, -1)
        # # # output = output.permute(0, 2, 1, 3)
        # #
        # # ####FC
        # output = self.fcs_mid(out5)
        # output = output.reshape(x5.size()[0], x5.size()[2], 128, -1)
        # output = output.permute(0, 2, 1, 3)


        # ConvTrans for real
        res = torch.cat((output, x5), 1)
        res1 = F.relu(self.bnT1(self.convT1(res)))
        res1 = torch.cat((res1[:,:,:,:32], x4[:,:,:,:32]), 1)
        res2 = F.relu(self.bnT2(self.convT2(res1)))
        res2 = torch.cat((res2[:,:,:,:64], x3[:,:,:,:64]), 1)
        res3 = F.relu(self.bnT3(self.convT3(res2)))
        res3 = torch.cat((res3[:,:,:,:128], x2[:,:,:,:128]), 1)
        res4 = F.relu(self.bnT4(self.convT4(res3)))
        res4 = torch.cat((res4[:,:,:,:256], x1[:,:,:,:256]), 1)
        # (B, o_c, T. F)
        res5 = F.relu(self.bnT5(self.convT5(res4)))
        res5 = self.fcs(res5)

        # ConvTrans for imag
        phares1 = F.relu(self.phabnT1(self.phaconvT1(res)))
        phares1 = torch.cat((phares1[:,:,:,:32], x4[:,:,:,:32]), 1)
        phares2 = F.relu(self.phabnT2(self.phaconvT2(phares1)))
        phares2 = torch.cat((phares2[:,:,:,:64], x3[:,:,:,:64]), 1)
        phares3 = F.relu(self.phabnT3(self.phaconvT3(phares2)))
        phares3 = torch.cat((phares3[:,:,:,:128], x2[:,:,:,:128]), 1)
        phares4 = F.relu(self.phabnT4(self.phaconvT4(phares3)))
        phares4 = torch.cat((phares4[:,:,:,:256], x1[:,:,:,:256]), 1)
        # (B, o_c, T. F)
        phares5 = F.relu(self.phabnT5(self.phaconvT5(phares4)))
        phares5 = self.phafcs(phares5)



        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5
        # return real_comu, imag_comu


class FCNN_slim(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN_slim, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn1 = GLayerNorm2d(in_channel=4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn2 = GLayerNorm2d(in_channel=4)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn3 = GLayerNorm2d(in_channel=8)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn4 = GLayerNorm2d(in_channel=8)
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.bn5 = GLayerNorm2d(in_channel=16)

        self.conv_mid = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1)

        # Decoder for real

        self.convT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.bnT1 = GLayerNorm2d(in_channel=8)
        self.convT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.bnT2 = GLayerNorm2d(in_channel=8)
        self.convT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.bnT3 = GLayerNorm2d(in_channel=4)
        self.convT4 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.bnT4 = GLayerNorm2d(in_channel=1)
        # self.convT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_real = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)

        self.fcs = nn.Sequential(
            nn.Linear(257, 513),
        )
        # Decoder for imag

        self.phaconvT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT1 = GLayerNorm2d(in_channel=8)
        self.phaconvT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT2 = GLayerNorm2d(in_channel=8)
        self.phaconvT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT3 = GLayerNorm2d(in_channel=4)
        self.phaconvT4 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT4 = GLayerNorm2d(in_channel=1)
        # self.phaconvT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_imag = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)
        self.phafcs = nn.Sequential(
            nn.Linear(257, 513),
        )

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def forward(self, x):
        # conv
        # (B, in_c, T, F)
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = F.relu(self.bn5(self.conv5(x4)))

        output = F.relu(self.conv_mid(x5))

        # ConvTrans for real
        res = torch.cat((output, x5), 1)
        res1 = F.relu(self.bnT1(self.convT1(res)))
        res1 = torch.cat((res1[:, :, :, :32], x4[:, :, :, :32]), 1)
        res2 = F.relu(self.bnT2(self.convT2(res1)))
        res2 = torch.cat((res2[:, :, :, :64], x3[:, :, :, :64]), 1)
        res3 = F.relu(self.bnT3(self.convT3(res2)))
        res3 = torch.cat((res3[:, :, :, :128], x2[:, :, :, :128]), 1)
        res4 = F.relu(self.bnT4(self.convT4(res3)))

        res5 = res4.squeeze(1)
        res5, _ = self.GRU_real(res5)
        res5 = self.fcs(res5)
        res5 = res5.unsqueeze(1)

        # ConvTrans for imag
        phares1 = F.relu(self.phabnT1(self.phaconvT1(res)))
        phares1 = torch.cat((phares1[:, :, :, :32], x4[:, :, :, :32]), 1)
        phares2 = F.relu(self.phabnT2(self.phaconvT2(phares1)))
        phares2 = torch.cat((phares2[:, :, :, :64], x3[:, :, :, :64]), 1)
        phares3 = F.relu(self.phabnT3(self.phaconvT3(phares2)))
        phares3 = torch.cat((phares3[:, :, :, :128], x2[:, :, :, :128]), 1)
        phares4 = F.relu(self.phabnT4(self.phaconvT4(phares3)))

        phares5 = phares4.squeeze(1)

        phares5, _ = self.GRU_imag(phares5)
        phares5 = self.phafcs(phares5)
        phares5 = phares5.unsqueeze(1)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5


class FCNN_phafullconv(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN_phafullconv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn1 = GLayerNorm2d(in_channel=4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn2 = GLayerNorm2d(in_channel=4)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn3 = GLayerNorm2d(in_channel=8)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn4 = GLayerNorm2d(in_channel=8)
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.bn5 = GLayerNorm2d(in_channel=16)

        self.conv_mid = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1)

        # Decoder for real

        self.convT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.bnT1 = GLayerNorm2d(in_channel=8)
        self.convT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.bnT2 = GLayerNorm2d(in_channel=8)
        self.convT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.bnT3 = GLayerNorm2d(in_channel=4)
        self.convT4 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.bnT4 = GLayerNorm2d(in_channel=1)
        # self.convT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_real = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)

        self.fcs = nn.Sequential(
            nn.Linear(257, 513),
        )
        # Decoder for imag

        self.phaconvT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT1 = GLayerNorm2d(in_channel=8)
        self.phaconvT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT2 = GLayerNorm2d(in_channel=8)
        self.phaconvT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT3 = GLayerNorm2d(in_channel=4)
        self.phaconvT4 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT4 = GLayerNorm2d(in_channel=4)
        self.phaconvT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT5 = GLayerNorm2d(in_channel=1)
        # self.GRU_imag = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)
        # self.phafcs = nn.Sequential(
        #     nn.Linear(257, 513),
        # )

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def forward(self, x):
        # conv
        # (B, in_c, T, F)
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = F.relu(self.bn5(self.conv5(x4)))

        output = F.relu(self.conv_mid(x5))

        # ConvTrans for real
        res = torch.cat((output, x5), 1)
        res1 = F.relu(self.bnT1(self.convT1(res)))
        res1 = torch.cat((res1[:, :, :, :32], x4[:, :, :, :32]), 1)
        res2 = F.relu(self.bnT2(self.convT2(res1)))
        res2 = torch.cat((res2[:, :, :, :64], x3[:, :, :, :64]), 1)
        res3 = F.relu(self.bnT3(self.convT3(res2)))
        res3 = torch.cat((res3[:, :, :, :128], x2[:, :, :, :128]), 1)
        res4 = F.relu(self.bnT4(self.convT4(res3)))

        res5 = res4.squeeze(1)
        res5, _ = self.GRU_real(res5)
        res5 = self.fcs(res5)
        res5 = res5.unsqueeze(1)

        # ConvTrans for imag
        phares1 = self.phabnT1(self.phaconvT1(res))
        phares1 = torch.cat((phares1[:, :, :, :32], x4[:, :, :, :32]), 1)
        phares2 = self.phabnT2(self.phaconvT2(phares1))
        phares2 = torch.cat((phares2[:, :, :, :64], x3[:, :, :, :64]), 1)
        phares3 = self.phabnT3(self.phaconvT3(phares2))
        phares3 = torch.cat((phares3[:, :, :, :128], x2[:, :, :, :128]), 1)
        phares4 = self.phabnT4(self.phaconvT4(phares3))
        phares4 = torch.cat((phares4[:, :, :, :256], x1[:, :, :, :256]), 1)
        phares5 = F.prelu(self.phabnT5(self.phaconvT5(phares4)))

        # phares5 = phares4.squeeze(1)
        #
        # phares5, _ = self.GRU_imag(phares5)
        # phares5 = self.phafcs(phares5)
        # phares5 = phares5.unsqueeze(1)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5

class FCNN_phafullconv_withPRELU(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN_phafullconv_withPRELU, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),

        )

        self.conv_mid = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),

        )


        # Decoder for real

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),

        )
        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=1),
            nn.PReLU(),

        )


        self.GRU_real = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)

        self.fcs = nn.Sequential(
            nn.Linear(257, 513),
        )
        # Decoder for imag


        self.phaconvT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),
        )
        self.phaconvT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),
        )
        self.phaconvT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),
        )
        self.phaconvT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),
        )
        self.phaconvT5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=1),
            nn.Tanh(),
        )

        # self.GRU_imag = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)
        # self.phafcs = nn.Sequential(
        #     nn.Linear(257, 513),
        # )

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def forward(self, x):
        # conv
        # (B, in_c, T, F)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        output = self.conv_mid(x5)

        # ConvTrans for real
        res = torch.cat((output, x5), 1)
        res1 = self.convT1(res)
        res1 = torch.cat((res1[:, :, :, :32], x4[:, :, :, :32]), 1)
        res2 = self.convT2(res1)
        res2 = torch.cat((res2[:, :, :, :64], x3[:, :, :, :64]), 1)
        res3 = self.convT3(res2)
        res3 = torch.cat((res3[:, :, :, :128], x2[:, :, :, :128]), 1)
        res4 = self.convT4(res3)

        res5 = res4.squeeze(1)
        res5, _ = self.GRU_real(res5)
        res5 = self.fcs(res5)
        res5 = res5.unsqueeze(1)

        # ConvTrans for imag
        phares1 = self.phaconvT1(res)
        phares1 = torch.cat((phares1[:, :, :, :32], x4[:, :, :, :32]), 1)
        phares2 = self.phaconvT2(phares1)
        phares2 = torch.cat((phares2[:, :, :, :64], x3[:, :, :, :64]), 1)
        phares3 = self.phaconvT3(phares2)
        phares3 = torch.cat((phares3[:, :, :, :128], x2[:, :, :, :128]), 1)
        phares4 = self.phaconvT4(phares3)
        phares4 = torch.cat((phares4[:, :, :, :256], x1[:, :, :, :256]), 1)
        phares5 = self.phaconvT5(phares4)

        # phares5 = phares4.squeeze(1)
        #
        # phares5, _ = self.GRU_imag(phares5)
        # phares5 = self.phafcs(phares5)
        # phares5 = phares5.unsqueeze(1)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5


class FCNN_slim_phanoGLN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN_slim_phanoGLN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn1 = GLayerNorm2d(in_channel=4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn2 = GLayerNorm2d(in_channel=4)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn3 = GLayerNorm2d(in_channel=8)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn4 = GLayerNorm2d(in_channel=8)
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.bn5 = GLayerNorm2d(in_channel=16)

        self.conv_mid = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1)

        # Decoder for real

        self.convT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.bnT1 = GLayerNorm2d(in_channel=8)
        self.convT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.bnT2 = GLayerNorm2d(in_channel=8)
        self.convT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.bnT3 = GLayerNorm2d(in_channel=4)
        self.convT4 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.bnT4 = GLayerNorm2d(in_channel=1)
        # self.convT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_real = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)

        self.fcs = nn.Sequential(
            nn.Linear(257, 513),
        )
        # Decoder for imag

        self.phaconvT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        # self.phabnT1 = GLayerNorm2d(in_channel=8)
        self.phaconvT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        # self.phabnT2 = GLayerNorm2d(in_channel=8)
        self.phaconvT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        # self.phabnT3 = GLayerNorm2d(in_channel=4)
        self.phaconvT4 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        # self.phabnT4 = GLayerNorm2d(in_channel=1)
        # self.phaconvT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_imag = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)
        self.phafcs = nn.Sequential(
            nn.Linear(257, 513),
        )

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def forward(self, x):
        # conv
        # (B, in_c, T, F)
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = F.relu(self.bn5(self.conv5(x4)))

        output = F.relu(self.conv_mid(x5))

        # ConvTrans for real
        res = torch.cat((output, x5), 1)
        res1 = F.relu(self.bnT1(self.convT1(res)))
        res1 = torch.cat((res1[:, :, :, :32], x4[:, :, :, :32]), 1)
        res2 = F.relu(self.bnT2(self.convT2(res1)))
        res2 = torch.cat((res2[:, :, :, :64], x3[:, :, :, :64]), 1)
        res3 = F.relu(self.bnT3(self.convT3(res2)))
        res3 = torch.cat((res3[:, :, :, :128], x2[:, :, :, :128]), 1)
        res4 = F.relu(self.bnT4(self.convT4(res3)))

        res5 = res4.squeeze(1)
        res5, _ = self.GRU_real(res5)
        res5 = self.fcs(res5)
        res5 = res5.unsqueeze(1)

        # ConvTrans for imag
        phares1 = F.relu(self.phaconvT1(res))
        phares1 = torch.cat((phares1[:, :, :, :32], x4[:, :, :, :32]), 1)
        phares2 = F.relu(self.phaconvT2(phares1))
        phares2 = torch.cat((phares2[:, :, :, :64], x3[:, :, :, :64]), 1)
        phares3 = F.relu(self.phaconvT3(phares2))
        phares3 = torch.cat((phares3[:, :, :, :128], x2[:, :, :, :128]), 1)
        phares4 = F.relu(self.phaconvT4(phares3))

        phares5 = phares4.squeeze(1)

        phares5, _ = self.GRU_imag(phares5)
        phares5 = self.phafcs(phares5)
        phares5 = phares5.unsqueeze(1)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5

class FCNN_slim_midGRU(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN_slim_midGRU, self).__init__()

        self.bn0 = GLayerNorm2d(in_channel=2)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn1 = GLayerNorm2d(in_channel=4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn2 = GLayerNorm2d(in_channel=4)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn3 = GLayerNorm2d(in_channel=8)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn4 = GLayerNorm2d(in_channel=8)
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn5 = GLayerNorm2d(in_channel=16)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.bn6 = GLayerNorm2d(in_channel=16)

        self.GRU_mid = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

        # self.conv_mid = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1)

        # Decoder for real
        self.convT0 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.bnT0 = GLayerNorm2d(in_channel=16)
        self.convT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.bnT1 = GLayerNorm2d(in_channel=8)
        self.convT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.bnT2 = GLayerNorm2d(in_channel=8)
        self.convT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.bnT3 = GLayerNorm2d(in_channel=4)
        self.convT4 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.bnT4 = GLayerNorm2d(in_channel=4)
        self.convT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.bnT5 = GLayerNorm2d(in_channel=1)
        # self.GRU_real = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)

        # self.fcs = nn.Sequential(
        #     nn.Linear(257, 513),
        # )
        # Decoder for imag

        self.phaconvT0 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT0 = GLayerNorm2d(in_channel=16)
        self.phaconvT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT1 = GLayerNorm2d(in_channel=8)
        self.phaconvT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT2 = GLayerNorm2d(in_channel=8)
        self.phaconvT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT3 = GLayerNorm2d(in_channel=4)
        self.phaconvT4 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT4 = GLayerNorm2d(in_channel=4)
        self.phaconvT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT5 = GLayerNorm2d(in_channel=1)

        # self.GRU_imag = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)
        # self.phafcs = nn.Sequential(
        #     nn.Linear(257, 513),
        # )

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def forward(self, x):
        # conv
        # (B, in_c, T, F)
        x0 = self.bn0(x)
        x1 = F.relu(self.bn1(self.conv1(x0)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = F.relu(self.bn5(self.conv5(x4)))
        x6 = F.relu(self.bn6(self.conv6(x5)))

        out5 = x6.permute(0, 2, 1, 3)
        out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        output, _ = self.GRU_mid(out5)
        output = output.reshape(out5.size()[0], out5.size()[1], 16, -1)
        output = output.permute(0, 2, 1, 3)

        # ConvTrans for real
        res = torch.cat((output, x6), 1)
        res0 = F.relu(self.bnT0(self.convT0(res)))
        res0 = torch.cat((res0[:, :, :, :16], x5[:, :, :, :16]), 1)
        res1 = F.relu(self.bnT1(self.convT1(res0)))
        res1 = torch.cat((res1[:, :, :, :32], x4[:, :, :, :32]), 1)
        res2 = F.relu(self.bnT2(self.convT2(res1)))
        res2 = torch.cat((res2[:, :, :, :64], x3[:, :, :, :64]), 1)
        res3 = F.relu(self.bnT3(self.convT3(res2)))
        res3 = torch.cat((res3[:, :, :, :128], x2[:, :, :, :128]), 1)
        res4 = F.relu(self.bnT4(self.convT4(res3)))
        res4 = torch.cat((res4[:, :, :, :256], x1[:, :, :, :256]), 1)
        res5 = F.tanh(self.bnT5(self.convT5(res4)))

        # res5 = res5.unsqueeze(1)


        # ConvTrans for imag
        phares0 = F.relu(self.phabnT0(self.phaconvT0(res)))
        phares0 = torch.cat((phares0[:, :, :, :16], x5[:, :, :, :16]), 1)
        phares1 = F.relu(self.phabnT1(self.phaconvT1(phares0)))
        phares1 = torch.cat((phares1[:, :, :, :32], x4[:, :, :, :32]), 1)
        phares2 = F.relu(self.phabnT2(self.phaconvT2(phares1)))
        phares2 = torch.cat((phares2[:, :, :, :64], x3[:, :, :, :64]), 1)
        phares3 = F.relu(self.phabnT3(self.phaconvT3(phares2)))
        phares3 = torch.cat((phares3[:, :, :, :128], x2[:, :, :, :128]), 1)
        phares4 = F.relu(self.phabnT4(self.phaconvT4(phares3)))
        phares4 = torch.cat((phares4[:, :, :, :256], x1[:, :, :, :256]), 1)
        phares5 = F.tanh(self.phabnT5(self.phaconvT5(phares4)))

        # phares5 = phares5.unsqueeze(1)



        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5

class FCNN_slim_827_wide(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN_slim_827_wide, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1, 15), stride=(1, 2), padding=(0, 7))
        self.bn1 = GLayerNorm2d(in_channel=4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 9), stride=(1, 2), padding=(0, 4))
        self.bn2 = GLayerNorm2d(in_channel=4)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3))
        self.bn3 = GLayerNorm2d(in_channel=8)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2))
        self.bn4 = GLayerNorm2d(in_channel=8)
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.bn5 = GLayerNorm2d(in_channel=16)

        self.conv_mid = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1)

        # Decoder for real

        self.convT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.bnT1 = GLayerNorm2d(in_channel=8)
        self.convT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 5), stride=(1, 2))
        self.bnT2 = GLayerNorm2d(in_channel=8)
        self.convT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 7), stride=(1, 2))
        self.bnT3 = GLayerNorm2d(in_channel=4)
        self.convT4 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 9), stride=(1, 2))
        self.bnT4 = GLayerNorm2d(in_channel=1)
        # self.convT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_real = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)

        self.fcs = nn.Sequential(
            nn.Linear(257, 513),
        )
        # Decoder for imag

        self.phaconvT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT1 = GLayerNorm2d(in_channel=8)
        self.phaconvT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 5), stride=(1, 2))
        self.phabnT2 = GLayerNorm2d(in_channel=8)
        self.phaconvT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 7), stride=(1, 2))
        self.phabnT3 = GLayerNorm2d(in_channel=4)
        self.phaconvT4 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 9), stride=(1, 2))
        self.phabnT4 = GLayerNorm2d(in_channel=1)
        # self.phaconvT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_imag = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)
        self.phafcs = nn.Sequential(
            nn.Linear(257, 513),
        )

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def forward(self, x):
        # conv
        # (B, in_c, T, F)

        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = F.relu(self.bn5(self.conv5(x4)))

        output = F.relu(self.conv_mid(x5))

        # ConvTrans for real
        res = torch.cat((output, x5), 1)
        res1 = F.relu(self.bnT1(self.convT1(res)))
        res1 = torch.cat((res1[:, :, :, :32], x4[:, :, :, :32]), 1)
        res2 = F.relu(self.bnT2(self.convT2(res1)))
        res2 = torch.cat((res2[:, :, :, :64], x3[:, :, :, :64]), 1)
        res3 = F.relu(self.bnT3(self.convT3(res2)))
        res3 = torch.cat((res3[:, :, :, :128], x2[:, :, :, :128]), 1)
        res4 = F.relu(self.bnT4(self.convT4(res3)))

        res5 = res4.squeeze(1)
        res5, _ = self.GRU_real(res5[:, :, :257])
        res5 = self.fcs(res5)
        res5 = res5.unsqueeze(1)

        # ConvTrans for imag
        phares1 = F.relu(self.phabnT1(self.phaconvT1(res)))
        phares1 = torch.cat((phares1[:, :, :, :32], x4[:, :, :, :32]), 1)
        phares2 = F.relu(self.phabnT2(self.phaconvT2(phares1)))
        phares2 = torch.cat((phares2[:, :, :, :64], x3[:, :, :, :64]), 1)
        phares3 = F.relu(self.phabnT3(self.phaconvT3(phares2)))
        phares3 = torch.cat((phares3[:, :, :, :128], x2[:, :, :, :128]), 1)
        phares4 = F.relu(self.phabnT4(self.phaconvT4(phares3)))

        phares5 = phares4.squeeze(1)

        phares5, _ = self.GRU_imag(phares5[:, :, :257])
        phares5 = self.phafcs(phares5)
        phares5 = phares5.unsqueeze(1)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5

class FCNN_slim_pre(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN_slim_pre, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        self.conv5 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2))


        self.conv_mid = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1)

        # Decoder for real

        self.convT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))

        self.convT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))

        self.convT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))

        self.convT4 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        # self.convT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_real = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)

        self.fcs = nn.Sequential(
            nn.Linear(257, 513),
        )
        # Decoder for imag

        self.phaconvT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))

        self.phaconvT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))

        self.phaconvT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))

        self.phaconvT4 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        # self.phaconvT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_imag = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)
        self.phafcs = nn.Sequential(
            nn.Linear(257, 513),
        )

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def forward(self, x):
        # conv
        # (B, in_c, T, F)

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))

        output = F.relu(self.conv_mid(x5))

        # ConvTrans for real
        res = torch.cat((output, x5), 1)
        res1 = F.relu(self.convT1(res))
        res1 = torch.cat((res1[:, :, :, :32], x4[:, :, :, :32]), 1)
        res2 = F.relu(self.convT2(res1))
        res2 = torch.cat((res2[:, :, :, :64], x3[:, :, :, :64]), 1)
        res3 = F.relu(self.convT3(res2))
        res3 = torch.cat((res3[:, :, :, :128], x2[:, :, :, :128]), 1)
        res4 = F.relu(self.convT4(res3))

        res5 = res4.squeeze(1)
        res5, _ = self.GRU_real(res5)
        res5 = self.fcs(res5)
        res5 = res5.unsqueeze(1)

        # ConvTrans for imag
        phares1 = F.relu(self.phaconvT1(res))
        phares1 = torch.cat((phares1[:, :, :, :32], x4[:, :, :, :32]), 1)
        phares2 = F.relu(self.phaconvT2(phares1))
        phares2 = torch.cat((phares2[:, :, :, :64], x3[:, :, :, :64]), 1)
        phares3 = F.relu(self.phaconvT3(phares2))
        phares3 = torch.cat((phares3[:, :, :, :128], x2[:, :, :, :128]), 1)
        phares4 = F.relu(self.phaconvT4(phares3))

        phares5 = phares4.squeeze(1)

        phares5, _ = self.GRU_imag(phares5)
        phares5 = self.phafcs(phares5)
        phares5 = phares5.unsqueeze(1)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5




class FCNN_only(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN_only, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2))


        self.conv_mid = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1)



        # Decoder for real



        self.convT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.convT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.convT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.convT4 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.convT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        # self.fcs = nn.Sequential(
        #     nn.Linear(513, 513),
        #     nn.Tanh(),
        # )

        # Decoder for imag


        self.phaconvT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.phaconvT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.phaconvT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.phaconvT4 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.phaconvT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        # self.phafcs = nn.Sequential(
        #     nn.Linear(513, 513),
        #     nn.Tanh(),
        # )

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def forward(self, x):
        # conv
        # (B, in_c, T, F)


        x1 = F.elu(self.conv1(x))
        x2 = F.elu(self.conv2(x1))
        x3 = F.elu(self.conv3(x2))
        x4 = F.elu(self.conv4(x3))
        x5 = F.elu(self.conv5(x4))

        output = F.elu(self.conv_mid(x5))

        # out5 = x5.permute(0, 2, 1, 3)
        # out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        # # # # lstm
        # # #
        # # # lstm, (hn, cn) = self.LSTM1(out5)
        # # # # # reshape
        # # # output = lstm.reshape(lstm.size()[0], lstm.size()[1], 256, -1)
        # # # output = output.permute(0, 2, 1, 3)
        # #
        # # ####FC
        # output = self.fcs_mid(out5)
        # output = output.reshape(x5.size()[0], x5.size()[2], 128, -1)
        # output = output.permute(0, 2, 1, 3)


        # ConvTrans for real
        res = torch.cat((output, x5), 1)
        res1 = F.elu(self.convT1(res))
        res1 = torch.cat((res1[:,:,:,:32], x4[:,:,:,:32]), 1)
        res2 = F.elu(self.convT2(res1))
        res2 = torch.cat((res2[:,:,:,:64], x3[:,:,:,:64]), 1)
        res3 = F.elu(self.convT3(res2))
        res3 = torch.cat((res3[:,:,:,:128], x2[:,:,:,:128]), 1)
        res4 = F.elu(self.convT4(res3))
        res4 = torch.cat((res4[:,:,:,:256], x1[:,:,:,:256]), 1)
        # (B, o_c, T. F)
        res5 = F.tanh(self.convT5(res4))
        # res5 = self.fcs(res5)

        # ConvTrans for imag
        phares1 = F.elu(self.phaconvT1(res))
        phares1 = torch.cat((phares1[:,:,:,:32], x4[:,:,:,:32]), 1)
        phares2 = F.elu(self.phaconvT2(phares1))
        phares2 = torch.cat((phares2[:,:,:,:64], x3[:,:,:,:64]), 1)
        phares3 = F.elu(self.phaconvT3(phares2))
        phares3 = torch.cat((phares3[:,:,:,:128], x2[:,:,:,:128]), 1)
        phares4 = F.elu(self.phaconvT4(phares3))
        phares4 = torch.cat((phares4[:,:,:,:256], x1[:,:,:,:256]), 1)
        # (B, o_c, T. F)
        phares5 = F.tanh(self.phaconvT5(phares4))
        # phares5 = self.phafcs(phares5)



        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5





class FCNN_slim_V2(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN_slim_V2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2))

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)

        # Decoder for real

        self.convT1 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.convT2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.convT3 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.convT4 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        # self.convT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_real = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)

        self.fcs = nn.Sequential(
            nn.Linear(257, 513),
        )
        # Decoder for imag

        self.phaconvT1 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.phaconvT2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.phaconvT3 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.phaconvT4 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        # self.phaconvT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_imag = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)
        self.phafcs = nn.Sequential(
            nn.Linear(257, 513),
        )


    def forward(self, x):
        # conv
        # (B, in_c, T, F)

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))

        output = F.relu(self.conv_mid(x5))

        # ConvTrans for real
        res = torch.cat((output, x5), 1)
        res1 = F.relu(self.convT1(res))
        res1 = torch.cat((res1[:, :, :, :32], x4[:, :, :, :32]), 1)
        res2 = F.relu(self.convT2(res1))
        res2 = torch.cat((res2[:, :, :, :64], x3[:, :, :, :64]), 1)
        res3 = F.relu(self.convT3(res2))
        res3 = torch.cat((res3[:, :, :, :128], x2[:, :, :, :128]), 1)
        res4 = F.relu(self.convT4(res3))

        res5 = res4.squeeze(1)
        res5, _ = self.GRU_real(res5)
        res5 = self.fcs(res5)
        res5 = res5.unsqueeze(1)

        # ConvTrans for imag
        phares1 = F.relu(self.phaconvT1(res))
        phares1 = torch.cat((phares1[:, :, :, :32], x4[:, :, :, :32]), 1)
        phares2 = F.relu(self.phaconvT2(phares1))
        phares2 = torch.cat((phares2[:, :, :, :64], x3[:, :, :, :64]), 1)
        phares3 = F.relu(self.phaconvT3(phares2))
        phares3 = torch.cat((phares3[:, :, :, :128], x2[:, :, :, :128]), 1)
        phares4 = F.relu(self.phaconvT4(phares3))

        phares5 = phares4.squeeze(1)

        phares5, _ = self.GRU_imag(phares5)
        phares5 = self.phafcs(phares5)
        phares5 = phares5.unsqueeze(1)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5

class FCNN_slim_V3(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN_slim_V3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2))

        self.conv_mid = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1)

        # Decoder for real

        self.convT1 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.convT2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.convT3 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.convT4 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        # self.convT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_real = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)

        self.fcs = nn.Sequential(
            nn.Linear(257, 513),
        )
        # Decoder for imag

        self.phaconvT1 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.phaconvT2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.phaconvT3 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.phaconvT4 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        # self.phaconvT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_imag = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)
        self.phafcs = nn.Sequential(
            nn.Linear(257, 513),
        )


    def forward(self, x):
        # conv
        # (B, in_c, T, F)

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))

        output = F.relu(self.conv_mid(x5))

        # ConvTrans for real
        res = torch.cat((output, x5), 1)
        res1 = F.relu(self.convT1(res))
        res1 = torch.cat((res1[:, :, :, :32], x4[:, :, :, :32]), 1)
        res2 = F.relu(self.convT2(res1))
        res2 = torch.cat((res2[:, :, :, :64], x3[:, :, :, :64]), 1)
        res3 = F.relu(self.convT3(res2))
        res3 = torch.cat((res3[:, :, :, :128], x2[:, :, :, :128]), 1)
        res4 = F.relu(self.convT4(res3))

        res5 = res4.squeeze(1)
        res5, _ = self.GRU_real(res5)
        res5 = self.fcs(res5)
        res5 = res5.unsqueeze(1)

        # ConvTrans for imag
        phares1 = F.relu(self.phaconvT1(res))
        phares1 = torch.cat((phares1[:, :, :, :32], x4[:, :, :, :32]), 1)
        phares2 = F.relu(self.phaconvT2(phares1))
        phares2 = torch.cat((phares2[:, :, :, :64], x3[:, :, :, :64]), 1)
        phares3 = F.relu(self.phaconvT3(phares2))
        phares3 = torch.cat((phares3[:, :, :, :128], x2[:, :, :, :128]), 1)
        phares4 = F.relu(self.phaconvT4(phares3))

        phares5 = phares4.squeeze(1)

        phares5, _ = self.GRU_imag(phares5)
        phares5 = self.phafcs(phares5)
        phares5 = phares5.unsqueeze(1)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5