import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import copy

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):

    def __init__(self, in_channel=9):
        super(CBAM, self).__init__()
        self.in_channel = in_channel
        self.ca = ChannelAttention(in_channel)
        self.sa = SpatialAttention()

    def forward(self, inputs):
        '''
        inputs should be [Batch, Ca, Dim, Time]
        '''
        # T-F attention
        out = self.ca(inputs) * inputs
        out = self.sa(out) * out

        return out

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class CausalConv1d(nn.Module):
    """1D DILATED CAUSAL CONVOLUTION."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        """FORWARD CALCULATION.
        Args:
            x (Tensor): Float tensor variable with the shape  (B, C, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, C, T).
        """
        # p1d = ((self.kernel_size - 1) * self.dilation, 0)
        # x_pad = F.pad(x, p1d, 'constant', 0)
        x = self.conv(x)
        # x = x[:, :, :-((self.kernel_size - 1) * self.dilation-1)].contiguous()
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        return x


class Dil_GLU(nn.Module):
    def __init__(self, dila_rate):
        super(Dil_GLU, self).__init__()
        self.in_conv = nn.Conv1d(in_channels = 1024, out_channels= 512, kernel_size= 1, stride = 1)
        self.dila_conv_left = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(in_channels = 512, out_channels= 512, kernel_size= 2, stride = 1,
                                   padding= (2-1) * dila_rate, dilation= dila_rate))
        self.chomp1 = Chomp1d((2-1) * dila_rate)
        self.dila_conv_right = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(in_channels = 512, out_channels= 512, kernel_size= 2, stride = 1,
                                   padding= (2-1) * dila_rate, dilation= dila_rate),
            nn.Sigmoid())
        self.chomp2 = Chomp1d((2-1) * dila_rate)
        self.out_conv = nn.Conv1d(in_channels= 512, out_channels= 1024, kernel_size = 1, stride = 1)
        self.out_elu = nn.ELU()

    def forward(self, inpt):
        x = inpt
        x = self.in_conv(x)
        x1 = self.dila_conv_left(x)
        x1_c = self.chomp1(x1)
        x2 = self.dila_conv_right(x)
        x2_c = self.chomp1(x2)
        x = x1_c * x2_c
        x = self.out_conv(x)
        x = x + inpt
        x = self.out_elu(x)
        return x


class Ca_Dil_GLU(nn.Module):
    def __init__(self, dila_rate):
        super(Ca_Dil_GLU, self).__init__()
        self.in_conv = nn.Conv1d(in_channels = 1024, out_channels= 512, kernel_size= 1, stride = 1)
        self.dila_conv_left = nn.Sequential(
            nn.ELU(),
            CausalConv1d(in_channels = 512, out_channels= 512, kernel_size= 3, dilation= dila_rate))
        self.dila_conv_right = nn.Sequential(
            nn.ELU(),
            CausalConv1d(in_channels = 512, out_channels= 512, kernel_size= 3, dilation= dila_rate),
            nn.Sigmoid())
        self.out_conv = nn.Conv1d(in_channels= 512, out_channels= 1024, kernel_size = 1, stride = 1)
        self.out_elu = nn.ELU()

    def forward(self, inpt):
        x = inpt
        x = self.in_conv(x)
        x1 = self.dila_conv_left(x)
        x2 = self.dila_conv_right(x)
        x = x1 * x2
        x = self.out_conv(x)
        x = x + inpt
        x = self.out_elu(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.elu = nn.ELU(inplace=True)
        # self.dropout = torch.nn.Dropout(0.3)
        # self.tanh = F.tanh()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.elu(g1 + x1)
        # psi = torch.tanh(g1 + x1)
        psi = self.psi(psi)
        # output = x * psi
        # output = self.dropout(output)

        return x * psi


class Attention_block_DIL(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block_DIL, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.elu = nn.ELU(inplace=True)
        # self.dropout = torch.nn.Dropout(0.3)
        # self.tanh = F.tanh()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.elu(g1 + x1)
        # psi = torch.tanh(g1 + x1)
        psi = self.psi(psi)
        output = x * psi
        # output = self.dropout(output)

        return output


class CRNN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(CRNN, self).__init__()
        # Encoder
        # self.dilconv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(1, 3), stride=(1, 1), dilation=(1, 1), padding=(0, 2))
        # self.dilbn1 = nn.BatchNorm2d(num_features=16)
        # self.dilconv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=(1, 1), dilation=(1,2), padding=(0,2*2))
        # self.dilbn2 = nn.BatchNorm2d(num_features=16)
        # self.dilconv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=(1, 1), dilation=(1,4), padding=(0,4*2))
        # self.dilbn3 = nn.BatchNorm2d(num_features=16)
        # self.dilconv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=(1, 1), dilation=(1,8), padding=(0,8*2))
        # self.dilbn4 = nn.BatchNorm2d(num_features=16)
        #
        # self.fcsdil = nn.Sequential(
        #     nn.Linear(161, 80),
        #     nn.ELU(),
        # )

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), stride=(1, 2))
        self.bn5 = nn.BatchNorm2d(num_features=256)


        self.glu_list = nn.ModuleList([Ca_Dil_GLU(dila_rate=2 ** i) for i in range(4)])

        # self.attention_gate_list = nn.ModuleList([Attention_block_DIL(F_g=1024, F_l=1024, F_int=512) for i in range(3)])
        # self.casua_conv = CausalConv1d(in_channels = 1024, out_channels= 1024, kernel_size= 3, dilation= 4)


        # LSTM
        # self.LSTM1 = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True)



        # self.fcs_mid = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.ELU(),
        #     nn.Linear(1024, 1024),
        #     nn.ELU()
        # )

        # Decoder for real
        #### Attention gate for real
        # self.Att1_real = Attention_block(F_g=256, F_l=256, F_int=128)
        # self.Att2_real = Attention_block(F_g=128, F_l=128, F_int=64)
        # self.Att3_real = Attention_block(F_g=64, F_l=64, F_int=32)
        # self.Att4_real = Attention_block(F_g=32, F_l=32, F_int=16)
        # self.Att5_real = Attention_block(F_g=16, F_l=16, F_int=8)


        self.convT1 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.bnT1 = nn.BatchNorm2d(num_features=128)
        self.convT2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.bnT2 = nn.BatchNorm2d(num_features=64)
        self.convT3 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.bnT3 = nn.BatchNorm2d(num_features=32)
        # output_padding为1，不然算出来是79
        self.convT4 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2),
                                         output_padding=(0, 1))
        self.bnT4 = nn.BatchNorm2d(num_features=16)
        self.convT5 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.bnT5 = nn.BatchNorm2d(num_features=1)

        self.fcs = nn.Sequential(
            nn.Linear(161, 161),
            nn.ELU(),
            nn.Linear(161, 161),
            nn.ELU()
        )

        # Decoder for imag
        #### Attention gate for imag
        # self.Att1_imag = Attention_block(F_g=256, F_l=256, F_int=128)
        # self.Att2_imag = Attention_block(F_g=128, F_l=128, F_int=64)
        # self.Att3_imag = Attention_block(F_g=64, F_l=64, F_int=32)
        # self.Att4_imag = Attention_block(F_g=32, F_l=32, F_int=16)
        # self.Att5_imag = Attention_block(F_g=16, F_l=16, F_int=8)

        self.phaconvT1 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT1 = nn.BatchNorm2d(num_features=128)
        self.phaconvT2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT2 = nn.BatchNorm2d(num_features=64)
        self.phaconvT3 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT3 = nn.BatchNorm2d(num_features=32)
        # output_padding为1，不然算出来是79
        self.phaconvT4 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2),
                                            output_padding=(0, 1))
        self.phabnT4 = nn.BatchNorm2d(num_features=16)
        self.phaconvT5 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT5 = nn.BatchNorm2d(num_features=1)

        self.phafcs = nn.Sequential(
            nn.Linear(161, 161),
            nn.ELU(),
            nn.Linear(161, 161),
            nn.ELU()
        )

    def forward(self, x):
        # conv
        # (B, in_c, T, F)

        ##### dilated conv on frequency dimension
        # dilx1 = F.elu(self.dilbn1(self.dilconv1(x)))
        # dilx1 = dilx1[:, :, :, :-2]
        # dilx2 = F.elu(self.dilbn2(self.dilconv2(dilx1)))
        # dilx2 = dilx2[:, :, :, :-4]
        # dilx3 = F.elu(self.dilbn3(self.dilconv3(dilx2)))
        # dilx3 = dilx3[:, :, :, :-8]
        # dilx4 = F.elu(self.dilbn4(self.dilconv4(dilx3)))
        # dilx4 = dilx4[:, :, :, :-16]
        # x1_new = self.fcsdil(dilx4)

        x1 = F.elu(self.bn1(self.conv1(x)))
        x2 = F.elu(self.bn2(self.conv2(x1)))
        x3 = F.elu(self.bn3(self.conv3(x2)))
        x4 = F.elu(self.bn4(self.conv4(x3)))
        x5 = F.elu(self.bn5(self.conv5(x4)))

        out5 = x5.permute(0, 2, 1, 3)
        out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)

        ###### DILCNN
        dil_in = out5.permute(0, 2, 1)
        for id in range(4):
            x = self.glu_list[id](dil_in)
            if id == 0:
                skip = x
            else:
                # att_skip = self.attention_gate_list[id-1](g=x, x=skip)
                skip = skip + x

        dil_output = skip

        dil_out = dil_output.permute(0, 2, 1)


        #lstm
        # lstm, (hn, cn) = self.LSTM1(dil_out)


        output = dil_out.reshape(out5.size()[0],out5.size()[1], 256, -1)
        output = output.permute(0, 2, 1, 3)


        # # lstm
        #
        # lstm, (hn, cn) = self.LSTM1(out5)
        # # # reshape
        # output = lstm.reshape(lstm.size()[0], lstm.size()[1], 256, -1)
        # output = output.permute(0, 2, 1, 3)

        ####FC
        # output = self.fcs_mid(out5)
        # output = dil_output.reshape(dil_output.size()[0], dil_output.size()[2], 256, -1)
        # output = output.permute(0, 2, 1, 3)


        # ConvTrans for real
        res = torch.cat((output, x5), 1)
        res1 = F.elu(self.bnT1(self.convT1(res)))
        res1 = torch.cat((res1, x4), 1)
        res2 = F.elu(self.bnT2(self.convT2(res1)))
        res2 = torch.cat((res2, x3), 1)
        res3 = F.elu(self.bnT3(self.convT3(res2)))
        res3 = torch.cat((res3, x2), 1)
        res4 = F.elu(self.bnT4(self.convT4(res3)))
        res4 = torch.cat((res4, x1), 1)
        # (B, o_c, T. F)
        res5 = F.elu(self.bnT5(self.convT5(res4)))
        res5 = self.fcs(res5)

        # ConvTrans for imag
        phares1 = F.elu(self.phabnT1(self.phaconvT1(res)))
        phares1 = torch.cat((phares1, x4), 1)
        phares2 = F.elu(self.phabnT2(self.phaconvT2(phares1)))
        phares2 = torch.cat((phares2, x3), 1)
        phares3 = F.elu(self.phabnT3(self.phaconvT3(phares2)))
        phares3 = torch.cat((phares3, x2), 1)
        phares4 = F.elu(self.phabnT4(self.phaconvT4(phares3)))
        phares4 = torch.cat((phares4, x1), 1)
        # (B, o_c, T. F)
        phares5 = F.elu(self.phabnT5(self.phaconvT5(phares4)))
        phares5 = self.phafcs(phares5)


        #### deconv + attention gate
        # ConvTrans for real
        # x5_att_real = self.Att1_real(g=output, x=x5)
        # res = torch.cat((output, x5_att_real), 1)
        # res1 = F.elu(self.bnT1(self.convT1(res)))
        # x4_att_real = self.Att2_real(g=res1, x=x4)
        # res1 = torch.cat((res1, x4_att_real), 1)
        # res2 = F.elu(self.bnT2(self.convT2(res1)))
        # x3_att_real = self.Att3_real(g=res2, x=x3)
        # res2 = torch.cat((res2, x3_att_real), 1)
        # res3 = F.elu(self.bnT3(self.convT3(res2)))
        # x2_att_real = self.Att4_real(g=res3, x=x2)
        # res3 = torch.cat((res3, x2_att_real), 1)
        # res4 = F.elu(self.bnT4(self.convT4(res3)))
        # x1_att_real = self.Att5_real(g=res4, x=x1)
        # res4 = torch.cat((res4, x1_att_real), 1)
        # # (B, o_c, T. F)
        # res5 = F.elu(self.bnT5(self.convT5(res4)))
        # res5 = self.fcs(res5)
        #
        # # ConvTrans for imag
        # x5_att_imag = self.Att1_imag(g=output, x=x5)
        # res_imag = torch.cat((output, x5_att_imag), 1)
        # phares1 = F.elu(self.phabnT1(self.phaconvT1(res_imag)))
        # x4_att_imag = self.Att2_imag(g=phares1, x=x4)
        # phares1 = torch.cat((phares1, x4_att_imag), 1)
        # phares2 = F.elu(self.phabnT2(self.phaconvT2(phares1)))
        # x3_att_imag = self.Att3_imag(g=phares2, x=x3)
        # phares2 = torch.cat((phares2, x3_att_imag), 1)
        # phares3 = F.elu(self.phabnT3(self.phaconvT3(phares2)))
        # x2_att_imag = self.Att4_imag(g=phares3, x=x2)
        # phares3 = torch.cat((phares3, x2_att_imag), 1)
        # phares4 = F.elu(self.phabnT4(self.phaconvT4(phares3)))
        # x1_att_imag = self.Att5_imag(g=phares4, x=x1)
        # phares4 = torch.cat((phares4, x1_att_imag), 1)
        # # (B, o_c, T. F)
        # phares5 = F.elu(self.phabnT5(self.phaconvT5(phares4)))
        # phares5 = self.phafcs(phares5)

        ###communcation block
        # real_comu = res5 * torch.tanh(phares5)
        # imag_comu = phares5 * torch.tanh(res5)
        # real_comu = self.fcs(real_comu)
        # imag_comu = self.phafcs(imag_comu)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5
        # return real_comu, imag_comu


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t)
        )

    def forward(self, x):
        x1 = self.RCNN(x)
        return x + x1

class AG_RRCNN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(AG_RRCNN, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.RRCNN1 = RRCNN_block(ch_out=16, t=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.RRCNN2 = RRCNN_block(ch_out=32, t=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.RRCNN3 = RRCNN_block(ch_out=64, t=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.RRCNN4 = RRCNN_block(ch_out=128, t=3)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), stride=(1, 2))
        self.bn5 = nn.BatchNorm2d(num_features=256)
        self.RRCNN5 = RRCNN_block(ch_out=256, t=3)


        self.glu_list = nn.ModuleList([Ca_Dil_GLU(dila_rate=2 ** i) for i in range(4)])
        # self.casua_conv = CausalConv1d(in_channels = 1024, out_channels= 1024, kernel_size= 3, dilation= 4)


        # LSTM
        # self.LSTM1 = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True)



        # Decoder for real
        #### Attention gate for real
        self.Att1_real = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Att2_real = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Att3_real = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Att4_real = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Att5_real = Attention_block(F_g=16, F_l=16, F_int=8)


        self.convT1 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.bnT1 = nn.BatchNorm2d(num_features=128)
        # self.RRCNN1_real = RRCNN_block(ch_out=128, t=2)
        self.convT2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.bnT2 = nn.BatchNorm2d(num_features=64)
        # self.RRCNN2_real = RRCNN_block(ch_out=64, t=2)
        self.convT3 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.bnT3 = nn.BatchNorm2d(num_features=32)
        # self.RRCNN3_real = RRCNN_block(ch_out=32, t=2)
        # output_padding为1，不然算出来是79
        self.convT4 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2),
                                         output_padding=(0, 1))
        self.bnT4 = nn.BatchNorm2d(num_features=16)
        # self.RRCNN4_real = RRCNN_block(ch_out=16, t=2)
        self.convT5 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.bnT5 = nn.BatchNorm2d(num_features=1)
        # self.RRCNN5_real = RRCNN_block(ch_out=1, t=2)

        self.fcs = nn.Sequential(
            nn.Linear(161, 161),
            nn.ELU(),
            nn.Linear(161, 161),
            nn.ELU()
        )

        # Decoder for imag
        #### Attention gate for imag
        self.Att1_imag = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Att2_imag = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Att3_imag = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Att4_imag = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Att5_imag = Attention_block(F_g=16, F_l=16, F_int=8)

        self.phaconvT1 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT1 = nn.BatchNorm2d(num_features=128)
        # self.RRCNN1_imag = RRCNN_block(ch_out=128, t=2)
        self.phaconvT2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT2 = nn.BatchNorm2d(num_features=64)
        # self.RRCNN2_imag = RRCNN_block(ch_out=64, t=2)
        self.phaconvT3 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT3 = nn.BatchNorm2d(num_features=32)
        # self.RRCNN3_imag = RRCNN_block(ch_out=32, t=2)
        # output_padding为1，不然算出来是79
        self.phaconvT4 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2),
                                            output_padding=(0, 1))
        self.phabnT4 = nn.BatchNorm2d(num_features=16)
        # self.RRCNN4_imag = RRCNN_block(ch_out=16, t=2)
        self.phaconvT5 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT5 = nn.BatchNorm2d(num_features=1)
        # self.RRCNN5_imag = RRCNN_block(ch_out=1, t=2)

        self.phafcs = nn.Sequential(
            nn.Linear(161, 161),
            nn.ELU(),
            nn.Linear(161, 161),
            nn.ELU()
        )

    def forward(self, x):
        # conv
        # (B, in_c, T, F)

        x1 = F.elu(self.bn1(self.conv1(x)))
        x1 = self.RRCNN1(x1)
        x2 = F.elu(self.bn2(self.conv2(x1)))
        x2 = self.RRCNN2(x2)
        x3 = F.elu(self.bn3(self.conv3(x2)))
        x3 = self.RRCNN3(x3)
        x4 = F.elu(self.bn4(self.conv4(x3)))
        x4 = self.RRCNN4(x4)
        x5 = F.elu(self.bn5(self.conv5(x4)))
        x5 = self.RRCNN5(x5)

        out5 = x5.permute(0, 2, 1, 3)
        out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        dil_in = out5.permute(0, 2, 1)
        for id in range(4):
            x = self.glu_list[id](dil_in)
            if id == 0:
                skip = x
            else:
                skip = skip + x
        dil_output = skip

        dil_out = dil_output.permute(0, 2, 1)



        output = dil_out.reshape(out5.size()[0],out5.size()[1], 256, -1)
        output = output.permute(0, 2, 1, 3)




        #### deconv + attention gate
        # ConvTrans for real
        # x5_att_real = self.Att1_real(g=output, x=x5)
        # res = torch.cat((output, x5_att_real), 1)
        # res1 = F.elu(self.bnT1(self.convT1(res)))
        # # res1 = self.RRCNN1_real(res1)
        # x4_att_real = self.Att2_real(g=res1, x=x4)
        # res1 = torch.cat((res1, x4_att_real), 1)
        # res2 = F.elu(self.bnT2(self.convT2(res1)))
        # # res2 = self.RRCNN2_real(res2)
        # x3_att_real = self.Att3_real(g=res2, x=x3)
        # res2 = torch.cat((res2, x3_att_real), 1)
        # res3 = F.elu(self.bnT3(self.convT3(res2)))
        # # res3 = self.RRCNN3_real(res3)
        # x2_att_real = self.Att4_real(g=res3, x=x2)
        # res3 = torch.cat((res3, x2_att_real), 1)
        # res4 = F.elu(self.bnT4(self.convT4(res3)))
        # # res4 = self.RRCNN4_real(res4)
        # x1_att_real = self.Att5_real(g=res4, x=x1)
        # res4 = torch.cat((res4, x1_att_real), 1)
        # # (B, o_c, T. F)
        # res5 = F.elu(self.bnT5(self.convT5(res4)))
        # # res5 = self.RRCNN5_real(res5)
        # res5 = self.fcs(res5)
        #
        # # ConvTrans for imag
        # x5_att_imag = self.Att1_imag(g=output, x=x5)
        # res_imag = torch.cat((output, x5_att_imag), 1)
        # phares1 = F.elu(self.phabnT1(self.phaconvT1(res_imag)))
        # # phares1 = self.RRCNN1_imag(phares1)
        # x4_att_imag = self.Att2_imag(g=phares1, x=x4)
        # phares1 = torch.cat((phares1, x4_att_imag), 1)
        # phares2 = F.elu(self.phabnT2(self.phaconvT2(phares1)))
        # # phares2 = self.RRCNN2_imag(phares2)
        # x3_att_imag = self.Att3_imag(g=phares2, x=x3)
        # phares2 = torch.cat((phares2, x3_att_imag), 1)
        # phares3 = F.elu(self.phabnT3(self.phaconvT3(phares2)))
        # # phares3 = self.RRCNN3_imag(phares3)
        # x2_att_imag = self.Att4_imag(g=phares3, x=x2)
        # phares3 = torch.cat((phares3, x2_att_imag), 1)
        # phares4 = F.elu(self.phabnT4(self.phaconvT4(phares3)))
        # # phares4 = self.RRCNN4_imag(phares4)
        # x1_att_imag = self.Att5_imag(g=phares4, x=x1)
        # phares4 = torch.cat((phares4, x1_att_imag), 1)
        # # (B, o_c, T. F)
        # phares5 = F.elu(self.phabnT5(self.phaconvT5(phares4)))
        # # phares5 = self.RRCNN5_imag(phares5)
        # phares5 = self.phafcs(phares5)



        ###### common conv
        res = torch.cat((output, x5), 1)
        res1 = F.elu(self.bnT1(self.convT1(res)))
        res1 = torch.cat((res1, x4), 1)
        res2 = F.elu(self.bnT2(self.convT2(res1)))
        res2 = torch.cat((res2, x3), 1)
        res3 = F.elu(self.bnT3(self.convT3(res2)))
        res3 = torch.cat((res3, x2), 1)
        res4 = F.elu(self.bnT4(self.convT4(res3)))
        res4 = torch.cat((res4, x1), 1)
        # (B, o_c, T. F)
        res5 = F.elu(self.bnT5(self.convT5(res4)))
        res5 = self.fcs(res5)

        # ConvTrans for imag
        phares1 = F.elu(self.phabnT1(self.phaconvT1(res)))
        phares1 = torch.cat((phares1, x4), 1)
        phares2 = F.elu(self.phabnT2(self.phaconvT2(phares1)))
        phares2 = torch.cat((phares2, x3), 1)
        phares3 = F.elu(self.phabnT3(self.phaconvT3(phares2)))
        phares3 = torch.cat((phares3, x2), 1)
        phares4 = F.elu(self.phabnT4(self.phaconvT4(phares3)))
        phares4 = torch.cat((phares4, x1), 1)
        # (B, o_c, T. F)
        phares5 = F.elu(self.phabnT5(self.phaconvT5(phares4)))
        phares5 = self.phafcs(phares5)


        real_comu = res5 * torch.tanh(phares5)
        imag_comu = phares5 * torch.tanh(res5)
        real_comu = self.fcs(real_comu)
        imag_comu = self.phafcs(imag_comu)

        # enh_spec = torch.cat((res5, phares5), 1)
        return real_comu, imag_comu
        # return res5, phares5

class GLU(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GLU, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                   stride=(1, 2))

        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                   stride=(1, 2))

        # self.att = CBAM(in_channel=out_channels
        #                  )

    def forward(self, inputs):
        '''
        input should be [Batch, Ca, Dim, Time]
        '''

        conv_out1 = self.conv1(inputs)
        conv_out2 = self.conv2(inputs)
        output = torch.sigmoid(conv_out2)*conv_out1
        # output = self.att(output)

        return output

class DeGLU(nn.Module):

    def __init__(self, in_channels, out_channels, output_padding=(0, 0)):
        super(DeGLU, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                   stride=(1, 2), output_padding=output_padding)

        self.deconv2 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                   stride=(1, 2), output_padding=output_padding)

        # self.att = CBAM(in_channel=out_channels
        #                  )

    def forward(self, inputs):
        '''
        input should be [Batch, Ca, Dim, Time]
        '''

        conv_out1 = self.deconv1(inputs)
        conv_out2 = self.deconv2(inputs)
        output = torch.sigmoid(conv_out2)*conv_out1
        # output = self.att(output)


        return output

class FTB(nn.Module):

    def __init__(self, input_dim=257, in_channel=9, r_channel=5):
        super(FTB, self).__init__()
        self.in_channel = in_channel
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channel, r_channel, kernel_size=[1, 1]),
        #     nn.BatchNorm2d(r_channel),
        #     nn.ReLU()
        # )  # 1维卷积

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channel * input_dim, in_channel, kernel_size=9, padding=4),
            nn.BatchNorm1d(in_channel),
            nn.ReLU()
        )  # 2维卷积
        self.freq_fc = nn.Linear(input_dim, input_dim, bias=False)
        # 全连接层

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=[1, 1]),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

    def forward(self, inputs):
        '''
        inputs should be [Batch, Ca, Dim, Time]
        '''
        # T-F attention
        # conv1_out = self.conv1(inputs)
        B, C, D, T = inputs.size()
        reshape1_out = torch.reshape(inputs, [B, C * D, T])
        conv1d_out = self.conv1d(reshape1_out)
        conv1d_out = torch.reshape(conv1d_out, [B, self.in_channel, 1, T])

        # now is also [B,C,D,T]
        att_out = conv1d_out * inputs

        # tranpose to [B,C,T,D]
        att_out = torch.transpose(att_out, 2, 3)
        freqfc_out = self.freq_fc(att_out)
        att_out = torch.transpose(freqfc_out, 2, 3)

        cat_out = torch.cat([att_out, inputs], 1)
        outputs = self.conv2(cat_out)
        return outputs


class GCRN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(GCRN, self).__init__()
        # Encoder
        self.conv1 = GLU(in_channels=2, out_channels=16)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = GLU(in_channels=16, out_channels=32)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = GLU(in_channels=32, out_channels=64)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = GLU(in_channels=64, out_channels=128)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.conv5 = GLU(in_channels=128, out_channels=256)
        self.bn5 = nn.BatchNorm2d(num_features=256)

        # LSTM
        self.BLSTM1 = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True, bidirectional=True)
        # self.LSTM1 = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True)

        # ####CBAM for real
        # self.att_real = CBAM(in_channel=512,
        #                  )
        # ####CBAM for imag
        # self.att_imag = CBAM(in_channel=512,
        #                  )

        # #### FTB
        # self.ftb_real = FTB(input_dim=4,
        #                 in_channel=256,
        #             )
        #
        # self.ftb_imag = FTB(input_dim=4,
        #                 in_channel=256,
        #             )

        # Decoder for real
        self.convT1 = DeGLU(in_channels=512, out_channels=128, output_padding=(0, 0))
        self.bnT1 = nn.BatchNorm2d(num_features=128)
        self.convT2 = DeGLU(in_channels=256, out_channels=64, output_padding=(0, 0))
        self.bnT2 = nn.BatchNorm2d(num_features=64)
        self.convT3 = DeGLU(in_channels=128, out_channels=32, output_padding=(0, 0))
        self.bnT3 = nn.BatchNorm2d(num_features=32)
        # output_padding为1，不然算出来是79
        self.convT4 = DeGLU(in_channels=64, out_channels=16, output_padding=(0, 1))
        self.bnT4 = nn.BatchNorm2d(num_features=16)
        self.convT5 = DeGLU(in_channels=32, out_channels=1, output_padding=(0, 0))
        self.bnT5 = nn.BatchNorm2d(num_features=1)

        self.fcs = nn.Sequential(
            nn.Linear(161, 161),
            nn.ELU(),
            nn.Linear(161, 161),
            nn.ELU()
        )

        # Decoder for imag
        self.phaconvT1 = DeGLU(in_channels=512, out_channels=128, output_padding=(0, 0))
        self.phabnT1 = nn.BatchNorm2d(num_features=128)
        self.phaconvT2 = DeGLU(in_channels=256, out_channels=64, output_padding=(0, 0))
        self.phabnT2 = nn.BatchNorm2d(num_features=64)
        self.phaconvT3 = DeGLU(in_channels=128, out_channels=32, output_padding=(0, 0))
        self.phabnT3 = nn.BatchNorm2d(num_features=32)
        # output_padding为1，不然算出来是79
        self.phaconvT4 = DeGLU(in_channels=64, out_channels=16, output_padding=(0, 1))
        self.phabnT4 = nn.BatchNorm2d(num_features=16)
        self.phaconvT5 = DeGLU(in_channels=32, out_channels=1, output_padding=(0, 0))
        self.phabnT5 = nn.BatchNorm2d(num_features=1)

        self.phafcs = nn.Sequential(
            nn.Linear(161, 161),
            nn.ELU(),
            nn.Linear(161, 161),
            nn.ELU()
        )

    def forward(self, x):
        # conv
        # (B, in_c, T, F)
        x1 = F.elu(self.bn1(self.conv1(x)))
        x2 = F.elu(self.bn2(self.conv2(x1)))
        x3 = F.elu(self.bn3(self.conv3(x2)))
        x4 = F.elu(self.bn4(self.conv4(x3)))
        x5 = F.elu(self.bn5(self.conv5(x4)))
        # x5_att = self.att1(x5)

        #####LSTM part
        # reshape
        out5 = x5.permute(0, 2, 1, 3)
        out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        # # lstm
        #
        lstm, (hn, cn) = self.BLSTM1(out5)
        # # reshape
        # output = lstm.reshape(lstm.size()[0], lstm.size()[1], 256, -1)
        output = lstm.reshape(lstm.size()[0], lstm.size()[1], 512, -1)
        output = output.permute(0, 2, 1, 3)
        # output_att = self.att2(output)


        # ##FTB operation
        # out5 = output.permute(0, 1, 3, 2)
        # input_real = self.ftb_real(out5)
        # input_imag = self.ftb_imag(out5)
        # input_real = input_real.permute(0, 1, 3, 2)
        # input_imag = input_imag.permute(0, 1, 3, 2)
        # input_real = torch.cat((input_real, x5), 1)
        # input_imag = torch.cat((input_imag, x5), 1)




        # input_real = self.att_real(output)
        # input_imag = self.att_imag(output)
        # input_real = torch.cat((input_real, x5), 1)
        # input_imag = torch.cat((input_imag, x5), 1)

        res1 = F.elu(self.bnT1(self.convT1(output)))
        res1 = torch.cat((res1, x4), 1)
        res2 = F.elu(self.bnT2(self.convT2(res1)))
        res2 = torch.cat((res2, x3), 1)
        res3 = F.elu(self.bnT3(self.convT3(res2)))
        res3 = torch.cat((res3, x2), 1)
        res4 = F.elu(self.bnT4(self.convT4(res3)))
        res4 = torch.cat((res4, x1), 1)
        # (B, o_c, T. F)
        res5 = F.elu(self.bnT5(self.convT5(res4)))
        res5 = self.fcs(res5)

        # ConvTrans for imag
        phares1 = F.elu(self.phabnT1(self.phaconvT1(output)))
        phares1 = torch.cat((phares1, x4), 1)
        phares2 = F.elu(self.phabnT2(self.phaconvT2(phares1)))
        phares2 = torch.cat((phares2, x3), 1)
        phares3 = F.elu(self.phabnT3(self.phaconvT3(phares2)))
        phares3 = torch.cat((phares3, x2), 1)
        phares4 = F.elu(self.phabnT4(self.phaconvT4(phares3)))
        phares4 = torch.cat((phares4, x1), 1)
        # (B, o_c, T. F)
        phares5 = F.elu(self.phabnT5(self.phaconvT5(phares4)))
        phares5 = self.phafcs(phares5)
        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5


class ERNN_K5(nn.Module):
    def __init__(self):
        super(ERNN_K5, self).__init__()
        self.inputSize = 1024
        self.stateSize = 1024
        self.hiddenSize = 1024

        self.l1_in = nn.Linear(self.inputSize, self.stateSize)
        self.l1_hid = nn.Linear(self.stateSize, self.stateSize)
        self.l2 = nn.Linear(self.stateSize, self.hiddenSize)
        self.l3 = nn.Linear(self.hiddenSize, self.stateSize)
        self.l_out = nn.Linear(self.stateSize, self.inputSize)

        self.initEta = 1e-2

        self.eta1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.eta2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.eta3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.eta4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.eta5 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.eta1.data.fill_(self.initEta)
        self.eta2.data.fill_(self.initEta)
        self.eta3.data.fill_(self.initEta)
        self.eta4.data.fill_(self.initEta)
        self.eta5.data.fill_(self.initEta)

    def forward(self, x):
        Bsize, Fsize, Tsize = x.shape
        h0 = Variable(torch.zeros(Bsize, self.stateSize)).to(x.device)


        for ii in range(Tsize):
            wx = self.l1_in(x[:, :, ii])

            # K=1
            uh0 = self.l1_hid(h0 + h0)
            F1h_hat = F.elu(uh0 + wx)
            Fnh_hat = F.elu(self.l3(F.elu(self.l2(F1h_hat))))  # kokoni DNN wo ippai irerareru
            h1 = h0 + self.eta1 * (Fnh_hat - (h0 + h0))

            # K=2
            uh1 = self.l1_hid(h1 + h0)
            F1h_hat = F.elu(uh1 + wx)
            Fnh_hat = F.elu(self.l3(F.elu(self.l2(F1h_hat))))  # kokoni DNN wo ippai irerareru
            h2 = h1 + self.eta2 * (Fnh_hat - (h1 + h0))

            # K=3
            uh2 = self.l1_hid(h2 + h0)
            F1h_hat = F.elu(uh2 + wx)
            Fnh_hat = F.elu(self.l3(F.elu(self.l2(F1h_hat))))  # kokoni DNN wo ippai irerareru
            h3 = h2 + self.eta3 * (Fnh_hat - (h2 + h0))

            # K=4
            uh3 = self.l1_hid(h3 + h0)
            F1h_hat = F.elu(uh3 + wx)
            Fnh_hat = F.elu(self.l3(F.elu(self.l2(F1h_hat))))  # kokoni DNN wo ippai irerareru
            h4 = h3 + self.eta4 * (Fnh_hat - (h3 + h0))

            # K=5
            uh4 = self.l1_hid(h4 + h0)
            F1h_hat = F.elu(uh4 + wx)
            Fnh_hat = F.elu(self.l3(F.elu(self.l2(F1h_hat))))  # kokoni DNN wo ippai irerareru
            h5 = h4 + self.eta5 * (Fnh_hat - (h4 + h0))

            y1 = self.l_out(h5)
            y1 = torch.unsqueeze(y1, dim=2)
            if ii == 0:
                y = y1
            else:
                y = torch.cat([y, y1], 2)
            h0 = h5
        return y

class CERNN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(CERNN, self).__init__()
        # Encoder
        # self.dilconv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(1, 3), stride=(1, 1), dilation=(1, 1), padding=(0, 2))
        # self.dilbn1 = nn.BatchNorm2d(num_features=16)
        # self.dilconv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=(1, 1), dilation=(1,2), padding=(0,2*2))
        # self.dilbn2 = nn.BatchNorm2d(num_features=16)
        # self.dilconv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=(1, 1), dilation=(1,4), padding=(0,4*2))
        # self.dilbn3 = nn.BatchNorm2d(num_features=16)
        # self.dilconv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=(1, 1), dilation=(1,8), padding=(0,8*2))
        # self.dilbn4 = nn.BatchNorm2d(num_features=16)
        #
        # self.fcsdil = nn.Sequential(
        #     nn.Linear(161, 80),
        #     nn.ELU(),
        # )

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), stride=(1, 2))
        self.bn5 = nn.BatchNorm2d(num_features=256)


        # self.glu_list = nn.ModuleList([Ca_Dil_GLU(dila_rate=2 ** i) for i in range(4)])
        self.ERNN_K5 = ERNN_K5()

        # self.attention_gate_list = nn.ModuleList([Attention_block_DIL(F_g=1024, F_l=1024, F_int=512) for i in range(3)])
        # self.casua_conv = CausalConv1d(in_channels = 1024, out_channels= 1024, kernel_size= 3, dilation= 4)


        # LSTM
        # self.LSTM1 = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True)



        # self.fcs_mid = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.ELU(),
        #     nn.Linear(1024, 1024),
        #     nn.ELU()
        # )

        # Decoder for real
        #### Attention gate for real
        # self.Att1_real = Attention_block(F_g=256, F_l=256, F_int=128)
        # self.Att2_real = Attention_block(F_g=128, F_l=128, F_int=64)
        # self.Att3_real = Attention_block(F_g=64, F_l=64, F_int=32)
        # self.Att4_real = Attention_block(F_g=32, F_l=32, F_int=16)
        # self.Att5_real = Attention_block(F_g=16, F_l=16, F_int=8)


        self.convT1 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.bnT1 = nn.BatchNorm2d(num_features=128)
        self.convT2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.bnT2 = nn.BatchNorm2d(num_features=64)
        self.convT3 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.bnT3 = nn.BatchNorm2d(num_features=32)
        # output_padding为1，不然算出来是79
        self.convT4 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2),
                                         output_padding=(0, 1))
        self.bnT4 = nn.BatchNorm2d(num_features=16)
        self.convT5 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.bnT5 = nn.BatchNorm2d(num_features=1)

        self.fcs = nn.Sequential(
            nn.Linear(161, 161),
            nn.ELU(),
            nn.Linear(161, 161),
            nn.ELU()
        )

        # Decoder for imag
        #### Attention gate for imag
        # self.Att1_imag = Attention_block(F_g=256, F_l=256, F_int=128)
        # self.Att2_imag = Attention_block(F_g=128, F_l=128, F_int=64)
        # self.Att3_imag = Attention_block(F_g=64, F_l=64, F_int=32)
        # self.Att4_imag = Attention_block(F_g=32, F_l=32, F_int=16)
        # self.Att5_imag = Attention_block(F_g=16, F_l=16, F_int=8)

        self.phaconvT1 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT1 = nn.BatchNorm2d(num_features=128)
        self.phaconvT2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT2 = nn.BatchNorm2d(num_features=64)
        self.phaconvT3 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT3 = nn.BatchNorm2d(num_features=32)
        # output_padding为1，不然算出来是79
        self.phaconvT4 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2),
                                            output_padding=(0, 1))
        self.phabnT4 = nn.BatchNorm2d(num_features=16)
        self.phaconvT5 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT5 = nn.BatchNorm2d(num_features=1)

        self.phafcs = nn.Sequential(
            nn.Linear(161, 161),
            nn.ELU(),
            nn.Linear(161, 161),
            nn.ELU()
        )

    def forward(self, x):
        # conv
        # (B, in_c, T, F)

        ##### dilated conv on frequency dimension
        # dilx1 = F.elu(self.dilbn1(self.dilconv1(x)))
        # dilx1 = dilx1[:, :, :, :-2]
        # dilx2 = F.elu(self.dilbn2(self.dilconv2(dilx1)))
        # dilx2 = dilx2[:, :, :, :-4]
        # dilx3 = F.elu(self.dilbn3(self.dilconv3(dilx2)))
        # dilx3 = dilx3[:, :, :, :-8]
        # dilx4 = F.elu(self.dilbn4(self.dilconv4(dilx3)))
        # dilx4 = dilx4[:, :, :, :-16]
        # x1_new = self.fcsdil(dilx4)

        x1 = F.elu(self.bn1(self.conv1(x)))
        x2 = F.elu(self.bn2(self.conv2(x1)))
        x3 = F.elu(self.bn3(self.conv3(x2)))
        x4 = F.elu(self.bn4(self.conv4(x3)))
        x5 = F.elu(self.bn5(self.conv5(x4)))

        out5 = x5.permute(0, 2, 1, 3)
        out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)

        ##### DILCNN
        # dil_in = out5.permute(0, 2, 1)
        # for id in range(4):
        #     x = self.glu_list[id](dil_in)
        #     if id == 0:
        #         skip = x
        #     else:
        #         # att_skip = self.attention_gate_list[id-1](g=x, x=skip)
        #         skip = skip + x
        #
        # dil_output = skip
        #
        # dil_out = dil_output.permute(0, 2, 1)

        ##### ERNN
        ernn_in = out5.permute(0, 2, 1)
        ernn_out = self.ERNN_K5(ernn_in)
        ernn_out = ernn_out.permute(0, 2, 1)

        #lstm
        # lstm, (hn, cn) = self.LSTM1(dil_out)


        output = ernn_out.reshape(out5.size()[0],out5.size()[1], 256, -1)
        output = output.permute(0, 2, 1, 3)


        # # lstm
        #
        # lstm, (hn, cn) = self.LSTM1(out5)
        # # # reshape
        # output = lstm.reshape(lstm.size()[0], lstm.size()[1], 256, -1)
        # output = output.permute(0, 2, 1, 3)

        ####FC
        # output = self.fcs_mid(out5)
        # output = dil_output.reshape(dil_output.size()[0], dil_output.size()[2], 256, -1)
        # output = output.permute(0, 2, 1, 3)


        # ConvTrans for real
        res = torch.cat((output, x5), 1)
        res1 = F.elu(self.bnT1(self.convT1(res)))
        res1 = torch.cat((res1, x4), 1)
        res2 = F.elu(self.bnT2(self.convT2(res1)))
        res2 = torch.cat((res2, x3), 1)
        res3 = F.elu(self.bnT3(self.convT3(res2)))
        res3 = torch.cat((res3, x2), 1)
        res4 = F.elu(self.bnT4(self.convT4(res3)))
        res4 = torch.cat((res4, x1), 1)
        # (B, o_c, T. F)
        res5 = F.elu(self.bnT5(self.convT5(res4)))
        res5 = self.fcs(res5)

        # ConvTrans for imag
        phares1 = F.elu(self.phabnT1(self.phaconvT1(res)))
        phares1 = torch.cat((phares1, x4), 1)
        phares2 = F.elu(self.phabnT2(self.phaconvT2(phares1)))
        phares2 = torch.cat((phares2, x3), 1)
        phares3 = F.elu(self.phabnT3(self.phaconvT3(phares2)))
        phares3 = torch.cat((phares3, x2), 1)
        phares4 = F.elu(self.phabnT4(self.phaconvT4(phares3)))
        phares4 = torch.cat((phares4, x1), 1)
        # (B, o_c, T. F)
        phares5 = F.elu(self.phabnT5(self.phaconvT5(phares4)))
        phares5 = self.phafcs(phares5)


        #### deconv + attention gate
        # ConvTrans for real
        # x5_att_real = self.Att1_real(g=output, x=x5)
        # res = torch.cat((output, x5_att_real), 1)
        # res1 = F.elu(self.bnT1(self.convT1(res)))
        # x4_att_real = self.Att2_real(g=res1, x=x4)
        # res1 = torch.cat((res1, x4_att_real), 1)
        # res2 = F.elu(self.bnT2(self.convT2(res1)))
        # x3_att_real = self.Att3_real(g=res2, x=x3)
        # res2 = torch.cat((res2, x3_att_real), 1)
        # res3 = F.elu(self.bnT3(self.convT3(res2)))
        # x2_att_real = self.Att4_real(g=res3, x=x2)
        # res3 = torch.cat((res3, x2_att_real), 1)
        # res4 = F.elu(self.bnT4(self.convT4(res3)))
        # x1_att_real = self.Att5_real(g=res4, x=x1)
        # res4 = torch.cat((res4, x1_att_real), 1)
        # # (B, o_c, T. F)
        # res5 = F.elu(self.bnT5(self.convT5(res4)))
        # res5 = self.fcs(res5)
        #
        # # ConvTrans for imag
        # x5_att_imag = self.Att1_imag(g=output, x=x5)
        # res_imag = torch.cat((output, x5_att_imag), 1)
        # phares1 = F.elu(self.phabnT1(self.phaconvT1(res_imag)))
        # x4_att_imag = self.Att2_imag(g=phares1, x=x4)
        # phares1 = torch.cat((phares1, x4_att_imag), 1)
        # phares2 = F.elu(self.phabnT2(self.phaconvT2(phares1)))
        # x3_att_imag = self.Att3_imag(g=phares2, x=x3)
        # phares2 = torch.cat((phares2, x3_att_imag), 1)
        # phares3 = F.elu(self.phabnT3(self.phaconvT3(phares2)))
        # x2_att_imag = self.Att4_imag(g=phares3, x=x2)
        # phares3 = torch.cat((phares3, x2_att_imag), 1)
        # phares4 = F.elu(self.phabnT4(self.phaconvT4(phares3)))
        # x1_att_imag = self.Att5_imag(g=phares4, x=x1)
        # phares4 = torch.cat((phares4, x1_att_imag), 1)
        # # (B, o_c, T. F)
        # phares5 = F.elu(self.phabnT5(self.phaconvT5(phares4)))
        # phares5 = self.phafcs(phares5)

        ###communcation block
        # real_comu = res5 * torch.tanh(phares5)
        # imag_comu = phares5 * torch.tanh(res5)
        # real_comu = self.fcs(real_comu)
        # imag_comu = self.phafcs(imag_comu)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5
        # return real_comu, imag_comu