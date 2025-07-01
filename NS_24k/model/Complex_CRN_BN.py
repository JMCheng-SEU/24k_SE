# coding: utf-8
# Author：Jiaming Cheng
# Date ：2020/12/22
# from base.BaseModel import *
import torch.nn as nn
import torch
import torch.nn.functional as F
# from utils.conv_stft import *
# from utils.complexnn import *
# from utils.stft import STFT
import numpy as np
import os


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

class ComplexConv2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0)
    ):
        '''
            in_channels: real+imag
            out_channels: real+imag
            kernel_size : input [B,C,D,T] kernel size in [D,T]
            padding : input [B,C,D,T] padding in [D,T]
            causal: if causal, will padding time dimension's left side,
                    otherwise both

        '''
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.real_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                   padding=self.padding)
        self.imag_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                   padding=self.padding)

        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.)
        nn.init.constant_(self.imag_conv.bias, 0.)

        # self.activation = nn.PReLU()
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.3)
        # self.glayer_norm_real = GLayerNorm2d(in_channel=out_channels // 2)
        # self.glayer_norm_imag = GLayerNorm2d(in_channel=out_channels // 2)

        self.glayer_norm_real = nn.BatchNorm2d(out_channels // 2)
        self.glayer_norm_imag = nn.BatchNorm2d(out_channels // 2)

    def forward(self, real, imag):

        real2real = self.real_conv(real)
        imag2imag = self.imag_conv(imag)

        real2imag = self.imag_conv(real)
        imag2real = self.real_conv(imag)

        real_out = real2real - imag2imag
        imag_out = real2imag + imag2real

        real_out = self.glayer_norm_real(real_out)
        imag_out = self.glayer_norm_imag(imag_out)

        real_out = self.leakyrelu(real_out)
        imag_out = self.leakyrelu(imag_out)

        return real_out, imag_out


class ComplexConvTranspose2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            output_padding=(0, 0),
            use_activ = True
    ):
        '''
            in_channels: real+imag
            out_channels: real+imag
        '''
        super(ComplexConvTranspose2d, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding


        self.real_conv = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                            padding=self.padding, output_padding=output_padding)
        self.imag_conv = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                            padding=self.padding, output_padding=output_padding)


        nn.init.normal_(self.real_conv.weight, std=0.05)
        nn.init.normal_(self.imag_conv.weight, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.)
        nn.init.constant_(self.imag_conv.bias, 0.)

        self.use_activ = use_activ
        # self.activation = nn.PReLU()
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.3)
        # self.glayer_norm_real = GLayerNorm2d(in_channel=out_channels // 2)
        # self.glayer_norm_imag = GLayerNorm2d(in_channel=out_channels // 2)


        self.glayer_norm_real = nn.BatchNorm2d(out_channels // 2)
        self.glayer_norm_imag = nn.BatchNorm2d(out_channels // 2)

    def forward(self, real, imag):



        real2real = self.real_conv(real)
        imag2imag = self.imag_conv(imag)

        real2imag = self.imag_conv(real)
        imag2real = self.real_conv(imag)

        real_out = real2real - imag2imag
        imag_out = real2imag + imag2real

        if self.use_activ:
            real_out = self.glayer_norm_real(real_out)
            imag_out = self.glayer_norm_imag(imag_out)
            real_out = self.leakyrelu(real_out)
            imag_out = self.leakyrelu(imag_out)


        return real_out, imag_out


class Complex_CRN_BN(nn.Module):
    """
    Input: [batch size, channels=2, T, n_fft]
    Output: [batch size, channels=2, T, n_fft]
    """
    def __init__(self):
        super(Complex_CRN_BN, self).__init__()

        self.conv1 = ComplexConv2d(in_channels=2, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        self.conv2 = ComplexConv2d(in_channels=4, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        self.conv3 = ComplexConv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        self.conv4 = ComplexConv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        self.conv5 = ComplexConv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        self.conv6 = ComplexConv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0))

        self.GRU_mid1 = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

        self.GRU_mid2 = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

        self.convT1 = ComplexConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0), output_padding=(0, 0))

        self.convT2 = ComplexConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0), output_padding=(0, 0))

        self.convT3 = ComplexConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0), output_padding=(0, 0))

        self.convT4 = ComplexConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0), output_padding=(0, 0))

        self.convT5 = ComplexConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0), output_padding=(0, 0))

        self.convT6 = ComplexConvTranspose2d(in_channels=4, out_channels=2, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0), output_padding=(0, 0), use_activ=False)

    def forward(self, x):

        real = x[:, 0, :, :]
        imag = x[:, 1, :, :]
        # print("real imag:", real.size(), imag.size())
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        spec_phase = torch.atan2(imag, real)
        real = real.unsqueeze(1)
        imag = imag.unsqueeze(1)


        real1, imag1 = self.conv1(real, imag)

        real2, imag2 = self.conv2(real1, imag1)

        real3, imag3 = self.conv3(real2, imag2)

        real4, imag4 = self.conv4(real3, imag3)

        real5, imag5 = self.conv5(real4, imag4)

        real6, imag6 = self.conv6(real5, imag5)

        mid_in = torch.cat([real6, imag6], dim=1)

        mid_GRU_in = mid_in.permute(0, 2, 1, 3)
        mid_GRU_in = mid_GRU_in.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], -1)
        mid_GRU_out1, _ = self.GRU_mid1(mid_GRU_in)
        mid_GRU_out, _ = self.GRU_mid2(mid_GRU_out1)
        mid_GRU_out = mid_GRU_out.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], 32, -1)
        mid_GRU_out = mid_GRU_out.permute(0, 2, 1, 3)

        real_T0, imag_T0 = torch.chunk(mid_GRU_out, 2, dim=1)
        # real_T0 = torch.ones_like(real_T0)
        # imag_T0 = torch.ones_like(imag_T0)
        real_T1, imag_T1 = self.convT1(real_T0, imag_T0)
        real_T1 = real_T1[:, :, :, :8]
        imag_T1 = imag_T1[:, :, :, :8]

        real_T2, imag_T2 = self.convT2(real_T1, imag_T1)
        real_T2 = real_T2[:, :, :, :16]
        imag_T2 = imag_T2[:, :, :, :16]


        real_T3, imag_T3 = self.convT3(real_T2, imag_T2)
        real_T3 = real_T3[:, :, :, :32]
        imag_T3 = imag_T3[:, :, :, :32]

        real_T4, imag_T4 = self.convT4(real_T3, imag_T3)
        real_T4 = real_T4[:, :, :, :64]
        imag_T4 = imag_T4[:, :, :, :64]

        real_T5, imag_T5 = self.convT5(real_T4, imag_T4)
        real_T5 = real_T5[:, :, :, :128]
        imag_T5 = imag_T5[:, :, :, :128]

        real_T6, imag_T6 = self.convT6(real_T5, imag_T5)


        mask_real = real_T6.squeeze(1)
        mask_imag = imag_T6.squeeze(1)


        mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
        real_phase = mask_real / (mask_mags + 1e-8)
        imag_phase = mask_imag / (mask_mags + 1e-8)
        mask_phase = torch.atan2(
            imag_phase,
            real_phase
        )
        mask_mags = torch.tanh(mask_mags)
        est_mags = mask_mags * spec_mags
        est_phase = spec_phase + mask_phase
        enh_real = est_mags * torch.cos(est_phase)
        enh_imag = est_mags * torch.sin(est_phase)

        # out_spec = torch.stack([real, imag], 3)
        # # out_wav = stft_block.inverse(out_spec)
        # # out_wav = torch.squeeze(out_wav, 1)
        # # out_wav = out_wav.clamp_(-1, 1)

        return enh_real, enh_imag

if __name__ == '__main__':
    # model = Simple_DCCRN()
    # state_dict = torch.load('model_0020.pth')
    # # folder_gru = 'E:\\GRUCNN_gln_0107'
    # # os.makedirs(folder_gru, exist_ok=True)
    # # for i in state_dict:
    # #     param = state_dict[i].numpy()
    # #     if i.split('.')[0][:3] == 'GRU':
    # #         param = param.flatten()
    # #         name = i.split('.')[0] + '_' + i.split('.')[-1]
    # #         np.savetxt('{}\\{}.txt'.format(folder_gru, name), param)
    # #     if i.split('.')[0][:4] == 'conv':
    # #         param = param.flatten()
    # #         name = i.split('.')[0] + '_' + i.split('.')[1] + '_' + i.split('.')[2]
    # #         np.savetxt('{}\\{}.txt'.format(folder_gru, name), param)
    #
    # inputs = torch.ones([1, 1, 257, 2])
    # # mixture_mag = torch.sqrt(inputs[:, 0, :, :] ** 2 + inputs[:, 1, :, :] ** 2 + 1e-8)
    # # LPS_fea = torch.log10(mixture_mag ** 2)
    # # print(LPS_fea)
    # model.load_state_dict(state_dict)
    # model.eval()
    # with torch.no_grad():
    #     out = model(inputs)
    # # enhanced_mag = out * mixture_mag
    # # enhanced_real = enhanced_mag * inputs[:, 0, :, :] / mixture_mag
    # # enhanced_imag = enhanced_mag * inputs[:, 1, :, :] / mixture_mag
    # print(out)

    inputs = torch.randn(16, 2, 100, 257)

    Model = Complex_CRN_BN()

    enh_real, enh_imag = Model(inputs)

    print(enh_real.shape)










