import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable



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

class FCNN_mid_GRU_new(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN_mid_GRU_new, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),

        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=32),
            nn.PReLU(),

        )

        self.GRU_mid = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)


        # Decoder for real

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),

        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),

        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )

        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=1),
            nn.PReLU(),
        )


        self.GRU_real = nn.GRU(input_size=129, hidden_size=129, num_layers=1, batch_first=True)

        # self.fcs = nn.Sequential(
        #     nn.Linear(257, 513),
        # )
        # Decoder for imag


        self.phaconvT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),
        )
        self.phaconvT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),
        )
        self.phaconvT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),
        )
        self.phaconvT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),
        )
        self.phaconvT5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=1),
            nn.PReLU(),
        )

        self.GRU_imag = nn.GRU(input_size=129, hidden_size=129, num_layers=1, batch_first=True)
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

        mid_in = x5

        mid_GRU_in = mid_in.permute(0, 2, 1, 3)
        mid_GRU_in = mid_GRU_in.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], -1)
        mid_GRU_out, _ = self.GRU_mid(mid_GRU_in)
        mid_GRU_out = mid_GRU_out.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], 32, -1)
        mid_GRU_out = mid_GRU_out.permute(0, 2, 1, 3)

        mid_out = mid_GRU_out

        # ConvTrans for real
        res = torch.cat((mid_out, x5), 1)
        res1 = self.convT1(res)
        res1 = torch.cat((res1[:, :, :, :8], x4[:, :, :, :8]), 1)
        res2 = self.convT2(res1)
        res2 = torch.cat((res2[:, :, :, :16], x3[:, :, :, :16]), 1)
        res3 = self.convT3(res2)
        res3 = torch.cat((res3[:, :, :, :32], x2[:, :, :, :32]), 1)
        res4 = self.convT4(res3)
        res4 = torch.cat((res4[:, :, :, :64], x1[:, :, :, :64]), 1)
        res5 = self.convT5(res4)

        res5 = res5.squeeze(1)
        res5, _ = self.GRU_real(res5)
        # # res5 = self.fcs(res5)
        res5 = res5.unsqueeze(1)

        # ConvTrans for imag
        phares1 = self.phaconvT1(res)
        phares1 = torch.cat((phares1[:, :, :, :8], x4[:, :, :, :8]), 1)
        phares2 = self.phaconvT2(phares1)
        phares2 = torch.cat((phares2[:, :, :, :16], x3[:, :, :, :16]), 1)
        phares3 = self.phaconvT3(phares2)
        phares3 = torch.cat((phares3[:, :, :, :32], x2[:, :, :, :32]), 1)
        phares4 = self.phaconvT4(phares3)
        phares4 = torch.cat((phares4[:, :, :, :64], x1[:, :, :, :64]), 1)
        phares5 = self.phaconvT5(phares4)

        phares5 = phares5.squeeze(1)
        phares5, _ = self.GRU_imag(phares5)
        # # phares5 = self.phafcs(phares5)
        phares5 = phares5.unsqueeze(1)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5

class CRN_for_IRM(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(CRN_for_IRM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=8),
            nn.ReLU(),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=8),
            nn.ReLU(),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=16),
            nn.ReLU(),

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=16),
            nn.ReLU(),

        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=32),
            nn.ReLU(),

        )

        self.GRU_mid = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)


        # Decoder for real

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.ReLU(),

        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.ReLU(),

        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.ReLU(),

        )
        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.ReLU(),

        )

        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=1),
            nn.ReLU(),
        )


        self.GRU_out = nn.GRU(input_size=129, hidden_size=129, num_layers=1, batch_first=True)

        self.fcs = nn.Sequential(
            nn.Linear(129, 129),
            # nn.ReLU(),
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
        x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        mid_in = x5

        mid_GRU_in = mid_in.permute(0, 2, 1, 3)
        mid_GRU_in = mid_GRU_in.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], -1)
        mid_GRU_out, _ = self.GRU_mid(mid_GRU_in)
        mid_GRU_out = mid_GRU_out.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], 32, -1)
        mid_GRU_out = mid_GRU_out.permute(0, 2, 1, 3)

        mid_out = mid_GRU_out

        # ConvTrans for real
        res = torch.cat((mid_out, x5), 1)
        res1 = self.convT1(res)
        res1 = torch.cat((res1[:, :, :, :8], x4[:, :, :, :8]), 1)
        res2 = self.convT2(res1)
        res2 = torch.cat((res2[:, :, :, :16], x3[:, :, :, :16]), 1)
        res3 = self.convT3(res2)
        res3 = torch.cat((res3[:, :, :, :32], x2[:, :, :, :32]), 1)
        res4 = self.convT4(res3)
        res4 = torch.cat((res4[:, :, :, :64], x1[:, :, :, :64]), 1)
        res5 = self.convT5(res4)

        res5 = res5.squeeze(1)
        res5, _ = self.GRU_out(res5)
        res5 = self.fcs(res5)
        # res5 = res5.unsqueeze(1)


        return res5


class CRN_for_PSD(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(CRN_for_PSD, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.ReLU(),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.ReLU(),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.ReLU(),

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.ReLU(),

        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=32),
            nn.ReLU(),

        )

        self.GRU_mid = nn.GRU(input_size=224, hidden_size=224, num_layers=1, batch_first=True)


        # Decoder for real

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.ReLU(),

        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.ReLU(),

        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.ReLU(),

        )
        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.ReLU(),

        )

        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=1),
            nn.ReLU(),
        )


        self.GRU_out = nn.GRU(input_size=255, hidden_size=256, num_layers=1, batch_first=True)

        # self.fcs = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.Tanh()
        #     # nn.ReLU(),
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
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        mid_in = x5

        mid_GRU_in = mid_in.permute(0, 2, 1, 3)
        mid_GRU_in = mid_GRU_in.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], -1)
        mid_GRU_out, _ = self.GRU_mid(mid_GRU_in)
        mid_GRU_out = mid_GRU_out.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], 32, -1)
        mid_GRU_out = mid_GRU_out.permute(0, 2, 1, 3)

        mid_out = mid_GRU_out

        # ConvTrans for real
        res = torch.cat((mid_out, x5), 1)
        res1 = self.convT1(res)
        res1 = torch.cat((res1, x4), 1)
        res2 = self.convT2(res1)
        res2 = torch.cat((res2, x3), 1)
        res3 = self.convT3(res2)
        res3 = torch.cat((res3, x2), 1)
        res4 = self.convT4(res3)
        res4 = torch.cat((res4, x1), 1)
        res5 = self.convT5(res4)

        res5 = res5.squeeze(1)
        res5, _ = self.GRU_out(res5)
        # res5 = self.fcs(res5)
        # res5 = res5.unsqueeze(1)
        res5 = res5.permute(0, 2, 1)

        return res5

class GRU_only(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(GRU_only, self).__init__()

        self.GRU_shared = nn.GRU(input_size=258, hidden_size=258, num_layers=1, batch_first=True)
        self.GRU_real = nn.GRU(input_size=129, hidden_size=129, num_layers=1, batch_first=True)
        self.GRU_imag = nn.GRU(input_size=129, hidden_size=129, num_layers=1, batch_first=True)
        # self.GRU_imag = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)
        self.realfcs = nn.Sequential(
            nn.Linear(258, 129),
        )

        self.imagfcs = nn.Sequential(
            nn.Linear(258, 129),
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
        real = x[:, 0, :, :]
        imag = x[:, 1, :, :]

        gru_share_in = torch.cat([real, imag], dim = 2)
        shared_GRU_out, _ = self.GRU_shared(gru_share_in)

        real_in = self.realfcs(shared_GRU_out)
        real_out, _ = self.GRU_real(real_in)

        imag_in = self.imagfcs(shared_GRU_out)
        imag_out, _ = self.GRU_imag(imag_in)

        real_out = real_out.unsqueeze(1)
        imag_out = imag_out.unsqueeze(1)


        return real_out, imag_out