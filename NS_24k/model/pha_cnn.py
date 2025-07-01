import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


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

class Feature_attention(nn.Module):

    def __init__(self, in_channel=256, dim=4):
        super(Feature_attention, self).__init__()
        self.in_channel = in_channel
        self.dim = dim
        self.W = torch.nn.Parameter(torch.randn(dim * in_channel, dim * in_channel), requires_grad=True)
        self.V = torch.nn.Parameter(torch.randn(dim * in_channel, dim * in_channel), requires_grad=True)


    def forward(self, inputs):
        '''
        inputs should be [Batch, Ca, Time, Dim]
        '''
        # self attention
        B, C, T, D = inputs.size()
        reshape1_out = torch.reshape(inputs, [B, T, C * D])

        output1 = torch.tanh(torch.matmul(reshape1_out, self.W))
        weights = torch.sigmoid(torch.matmul(output1, self.V))
        att_output = weights * reshape1_out
        output = torch.reshape(att_output, [B, C, T, D])

        return output


class GCNN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(GCNN, self).__init__()
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


        self.fcs_mid = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ELU(),
        )

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

        # #####LSTM part
        # # reshape
        # out5 = x5.permute(0, 2, 1, 3)
        # out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        # # # lstm
        # #
        # lstm, (hn, cn) = self.BLSTM1(out5)
        # # # reshape
        # # # output = lstm.reshape(lstm.size()[0], lstm.size()[1], 256, -1)
        # output = lstm.reshape(lstm.size()[0], lstm.size()[1], 512, -1)
        # output = output.permute(0, 2, 1, 3)
        #
        # # ConvTrans for real
        # # res = torch.cat((output, x5), 1)
        #
        #
        #
        # input_real = self.att_real(output)
        # input_imag = self.att_imag(output)
        # # input_real = torch.cat((input_real, x5), 1)
        # # input_imag = torch.cat((input_imag, x5), 1)


        ###FTB operation
        # out5 = x5.permute(0, 1, 3, 2)
        # input_real = self.ftb_real(out5)
        # input_imag = self.ftb_imag(out5)
        # input_real = input_real.permute(0, 1, 3, 2)
        # input_imag = input_imag.permute(0, 1, 3, 2)
        # input_real = torch.cat((input_real, x5), 1)
        # input_imag = torch.cat((input_imag, x5), 1)

        ### Feature attention
        # input_real = self.FA_real(x5)
        # input_imag = self.FA_imag(x5)
        # input_real = torch.cat((input_real, x5), 1)
        # input_imag = torch.cat((input_imag, x5), 1)

        ###FC
        # reshape
        out5 = x5.permute(0, 2, 1, 3)
        out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        output = self.fcs_mid(out5)
        output = output.reshape(output.size()[0], output.size()[1], 256, -1)
        output = output.permute(0, 2, 1, 3)

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
        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5


class GCNN_new(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(GCNN_new, self).__init__()
        # Encoder
        self.conv1 = GLU(in_channels=4, out_channels=16)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = GLU(in_channels=16, out_channels=32)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = GLU(in_channels=32, out_channels=64)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = GLU(in_channels=64, out_channels=128)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.conv5 = GLU(in_channels=128, out_channels=256)
        self.bn5 = nn.BatchNorm2d(num_features=256)


        self.fcs_mid = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.ELU(),
        )

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


        ###FC
        # reshape
        out5 = x5.permute(0, 2, 1, 3)
        out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        output = self.fcs_mid(out5)
        output = output.reshape(output.size()[0], output.size()[1], 256, -1)
        output = output.permute(0, 2, 1, 3)
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
        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5

class GCNN_real(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(GCNN_real, self).__init__()
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


        self.fcs_mid = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.ELU(),
        )

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


    def forward(self, x):
        # conv
        # (B, in_c, T, F)
        x1 = F.elu(self.bn1(self.conv1(x)))
        x2 = F.elu(self.bn2(self.conv2(x1)))
        x3 = F.elu(self.bn3(self.conv3(x2)))
        x4 = F.elu(self.bn4(self.conv4(x3)))
        x5 = F.elu(self.bn5(self.conv5(x4)))


        ###FC
        # reshape
        out5 = x5.permute(0, 2, 1, 3)
        out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        output = self.fcs_mid(out5)
        output = output.reshape(output.size()[0], output.size()[1], 256, -1)
        output = output.permute(0, 2, 1, 3)
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

        return res5


class GCNN_imag(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(GCNN_imag, self).__init__()
        # Encoder
        self.conv1 = GLU(in_channels=4, out_channels=16)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = GLU(in_channels=16, out_channels=32)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = GLU(in_channels=32, out_channels=64)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = GLU(in_channels=64, out_channels=128)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.conv5 = GLU(in_channels=128, out_channels=256)
        self.bn5 = nn.BatchNorm2d(num_features=256)


        self.fcs_mid = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.ELU(),
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


        ###FC
        # reshape
        out5 = x5.permute(0, 2, 1, 3)
        out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        output = self.fcs_mid(out5)
        output = output.reshape(output.size()[0], output.size()[1], 256, -1)
        output = output.permute(0, 2, 1, 3)
        res = torch.cat((output, x5), 1)

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
        # enh_spec = torch.cat((res5, phares5), 1)
        return phares5

class RTN_GCNN(nn.Module):
    def __init__(self):
        super(RTN_GCNN, self).__init__()
        # Main Encoder Part
        self.gru = Stage_GRU()
        self.gcnn = GCNN_new()
        self.Iter = 3

    def forward(self, mixture):
        """
        :param mixture: [B, T, F] B: Batch; T: Timestep
                            F: Feature;
        :return:
        """
        x = mixture
        mixture_real, mixture_imag = torch.split(mixture, 1, dim=1)
        batch_size, timesteps, feat_dim = mixture.size(0), mixture.size(2), mixture.size(3)
        # h = Variable(torch.zeros(batch_size, 4, timesteps, feat_dim)).to(x.device)
        h = torch.zeros([batch_size, 4, timesteps, feat_dim], requires_grad=True, device='cuda')
        # h = torch.cat((mixture, x), dim= 1)
        for i in range(self.Iter):
            x = torch.cat((mixture, x), dim= 1)
            h = self.gru(x, h)
            real, imag = self.gcnn(h)
            est_Mr = -10. * torch.log((10 - real) / ((10 + real))+1e-8)
            est_Mi = -10. * torch.log((10 - imag) / ((10 + imag))+1e-8)

            recons_real = est_Mr * mixture_real - est_Mi * mixture_imag
            recons_imag = est_Mr * mixture_imag - est_Mi * mixture_real

            #### avoid nan data
            zero = torch.zeros_like(recons_real)
            recons_real = torch.where(torch.isnan(recons_real), zero, recons_real)
            recons_imag = torch.where(torch.isnan(recons_imag), zero, recons_imag)
            x = torch.cat((recons_real, recons_imag), dim=1)
        est_real, est_imag = torch.split(x, 1, dim=1)
        return est_real, est_imag, real, imag


class Stage_GRU(nn.Module):
    def __init__(self):
        super(Stage_GRU, self).__init__()
        # Recurrent Part
        # Recurrent Part
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ELU())
        self.conv_z = nn.Sequential(
            nn.Conv2d(in_channels=4 + 4, out_channels=4, kernel_size=(1, 3),padding=(0, 1)),
            nn.Sigmoid())
        self.conv_r = nn.Sequential(
            nn.Conv2d(in_channels=4 + 4, out_channels=4, kernel_size=(1, 3),padding=(0, 1)),
            nn.Sigmoid())
        self.conv_n = nn.Sequential(
            nn.Conv2d(in_channels=4 + 4, out_channels=4, kernel_size=(1, 3),padding=(0, 1)),
            nn.Tanh())

    def forward(self, x, h= None):
        x = self.pre_conv(x)
        x1 = x
        x = torch.cat((x, h), dim = 1)
        z = self.conv_z(x)
        r = self.conv_r(x)
        s = r * h
        s = torch.cat((s, x1), dim =1)
        n = self.conv_n(s)
        h = (1 - z) * h + z * n
        return h

class Stage_GRU_imag(nn.Module):
    def __init__(self):
        super(Stage_GRU_imag, self).__init__()
        # Recurrent Part
        # Recurrent Part
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ELU())
        self.conv_z = nn.Sequential(
            nn.Conv2d(in_channels=4 + 4, out_channels=4, kernel_size=(1, 3),padding=(0, 1)),
            nn.Sigmoid())
        self.conv_r = nn.Sequential(
            nn.Conv2d(in_channels=4 + 4, out_channels=4, kernel_size=(1, 3),padding=(0, 1)),
            nn.Sigmoid())
        self.conv_n = nn.Sequential(
            nn.Conv2d(in_channels=4 + 4, out_channels=4, kernel_size=(1, 3),padding=(0, 1)),
            nn.Tanh())

    def forward(self, x, h= None):
        x = self.pre_conv(x)
        x1 = x
        x = torch.cat((x, h), dim = 1)
        z = self.conv_z(x)
        r = self.conv_r(x)
        s = r * h
        s = torch.cat((s, x1), dim =1)
        n = self.conv_n(s)
        h = (1 - z) * h + z * n
        return h


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
        # psi = self.elu(g1 + x1)
        psi = torch.tanh(g1 + x1)
        psi = self.psi(psi)
        # output = x * psi
        # output = self.dropout(output)

        return x * psi

class FCNN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN, self).__init__()

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


        # self.attention_gate_list = nn.ModuleList([Attention_block_DIL(F_g=1024, F_l=1024, F_int=512) for i in range(3)])
        # self.casua_conv = CausalConv1d(in_channels = 1024, out_channels= 1024, kernel_size= 3, dilation= 4)


        # LSTM
        # self.LSTM1 = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True)



        self.fcs_mid = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ELU()
        )

        # Decoder for real
        #### Attention gate for real
        self.Att1_real = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Att2_real = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Att3_real = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Att4_real = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Att5_real = Attention_block(F_g=16, F_l=16, F_int=8)


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
        self.Att1_imag = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Att2_imag = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Att3_imag = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Att4_imag = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Att5_imag = Attention_block(F_g=16, F_l=16, F_int=8)

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
        # # lstm
        #
        # lstm, (hn, cn) = self.LSTM1(out5)
        # # # reshape
        # output = lstm.reshape(lstm.size()[0], lstm.size()[1], 256, -1)
        # output = output.permute(0, 2, 1, 3)

        ####FC
        output = self.fcs_mid(out5)
        output = output.reshape(x5.size()[0], x5.size()[2], 256, -1)
        output = output.permute(0, 2, 1, 3)


        # # ConvTrans for real
        # res = torch.cat((output, x5), 1)
        # res1 = F.elu(self.bnT1(self.convT1(res)))
        # res1 = torch.cat((res1, x4), 1)
        # res2 = F.elu(self.bnT2(self.convT2(res1)))
        # res2 = torch.cat((res2, x3), 1)
        # res3 = F.elu(self.bnT3(self.convT3(res2)))
        # res3 = torch.cat((res3, x2), 1)
        # res4 = F.elu(self.bnT4(self.convT4(res3)))
        # res4 = torch.cat((res4, x1), 1)
        # # (B, o_c, T. F)
        # res5 = F.elu(self.bnT5(self.convT5(res4)))
        # res5 = self.fcs(res5)
        #
        # # ConvTrans for imag
        # phares1 = F.elu(self.phabnT1(self.phaconvT1(res)))
        # phares1 = torch.cat((phares1, x4), 1)
        # phares2 = F.elu(self.phabnT2(self.phaconvT2(phares1)))
        # phares2 = torch.cat((phares2, x3), 1)
        # phares3 = F.elu(self.phabnT3(self.phaconvT3(phares2)))
        # phares3 = torch.cat((phares3, x2), 1)
        # phares4 = F.elu(self.phabnT4(self.phaconvT4(phares3)))
        # phares4 = torch.cat((phares4, x1), 1)
        # # (B, o_c, T. F)
        # phares5 = F.elu(self.phabnT5(self.phaconvT5(phares4)))
        # phares5 = self.phafcs(phares5)


        #### deconv + attention gate
        ## ConvTrans for real
        x5_att_real = self.Att1_real(g=output, x=x5)
        res = torch.cat((output, x5_att_real), 1)
        res1 = F.elu(self.bnT1(self.convT1(res)))
        x4_att_real = self.Att2_real(g=res1, x=x4)
        res1 = torch.cat((res1, x4_att_real), 1)
        res2 = F.elu(self.bnT2(self.convT2(res1)))
        x3_att_real = self.Att3_real(g=res2, x=x3)
        res2 = torch.cat((res2, x3_att_real), 1)
        res3 = F.elu(self.bnT3(self.convT3(res2)))
        x2_att_real = self.Att4_real(g=res3, x=x2)
        res3 = torch.cat((res3, x2_att_real), 1)
        res4 = F.elu(self.bnT4(self.convT4(res3)))
        x1_att_real = self.Att5_real(g=res4, x=x1)
        res4 = torch.cat((res4, x1_att_real), 1)
        # (B, o_c, T. F)
        res5 = F.elu(self.bnT5(self.convT5(res4)))
        res5 = self.fcs(res5)

        ## ConvTrans for imag
        x5_att_imag = self.Att1_imag(g=output, x=x5)
        res_imag = torch.cat((output, x5_att_imag), 1)
        phares1 = F.elu(self.phabnT1(self.phaconvT1(res_imag)))
        x4_att_imag = self.Att2_imag(g=phares1, x=x4)
        phares1 = torch.cat((phares1, x4_att_imag), 1)
        phares2 = F.elu(self.phabnT2(self.phaconvT2(phares1)))
        x3_att_imag = self.Att3_imag(g=phares2, x=x3)
        phares2 = torch.cat((phares2, x3_att_imag), 1)
        phares3 = F.elu(self.phabnT3(self.phaconvT3(phares2)))
        x2_att_imag = self.Att4_imag(g=phares3, x=x2)
        phares3 = torch.cat((phares3, x2_att_imag), 1)
        phares4 = F.elu(self.phabnT4(self.phaconvT4(phares3)))
        x1_att_imag = self.Att5_imag(g=phares4, x=x1)
        phares4 = torch.cat((phares4, x1_att_imag), 1)
        # (B, o_c, T. F)
        phares5 = F.elu(self.phabnT5(self.phaconvT5(phares4)))
        phares5 = self.phafcs(phares5)

        ###communcation block
        # real_comu = res5 * torch.tanh(phares5)
        # imag_comu = phares5 * torch.tanh(res5)
        # real_comu = self.fcs(real_comu)
        # imag_comu = self.phafcs(imag_comu)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5
        # return real_comu, imag_comu


class ATT_GCNN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(ATT_GCNN, self).__init__()
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


        self.fcs_mid = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ELU(),
        )

        # Decoder for real
        #### Attention gate for real
        self.Att1_real = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Att2_real = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Att3_real = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Att4_real = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Att5_real = Attention_block(F_g=16, F_l=16, F_int=8)

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

        #### Attention gate for imag
        self.Att1_imag = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Att2_imag = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Att3_imag = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Att4_imag = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Att5_imag = Attention_block(F_g=16, F_l=16, F_int=8)
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


        ###FC
        # reshape
        out5 = x5.permute(0, 2, 1, 3)
        out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        output = self.fcs_mid(out5)
        output = output.reshape(output.size()[0], output.size()[1], 256, -1)
        output = output.permute(0, 2, 1, 3)

        res = torch.cat((output, x5), 1)


        # res1 = F.elu(self.bnT1(self.convT1(res)))
        # res1 = torch.cat((res1, x4), 1)
        # res2 = F.elu(self.bnT2(self.convT2(res1)))
        # res2 = torch.cat((res2, x3), 1)
        # res3 = F.elu(self.bnT3(self.convT3(res2)))
        # res3 = torch.cat((res3, x2), 1)
        # res4 = F.elu(self.bnT4(self.convT4(res3)))
        # res4 = torch.cat((res4, x1), 1)
        # # (B, o_c, T. F)
        # res5 = F.elu(self.bnT5(self.convT5(res4)))
        # res5 = self.fcs(res5)
        #
        # # ConvTrans for imag
        # phares1 = F.elu(self.phabnT1(self.phaconvT1(res)))
        # phares1 = torch.cat((phares1, x4), 1)
        # phares2 = F.elu(self.phabnT2(self.phaconvT2(phares1)))
        # phares2 = torch.cat((phares2, x3), 1)
        # phares3 = F.elu(self.phabnT3(self.phaconvT3(phares2)))
        # phares3 = torch.cat((phares3, x2), 1)
        # phares4 = F.elu(self.phabnT4(self.phaconvT4(phares3)))
        # phares4 = torch.cat((phares4, x1), 1)
        # # (B, o_c, T. F)
        # phares5 = F.elu(self.phabnT5(self.phaconvT5(phares4)))
        # phares5 = self.phafcs(phares5)


        #### deconv + attention gate
        ## ConvTrans for real
        x5_att_real = self.Att1_real(g=output, x=x5)
        res = torch.cat((output, x5_att_real), 1)
        res1 = F.elu(self.bnT1(self.convT1(res)))
        x4_att_real = self.Att2_real(g=res1, x=x4)
        res1 = torch.cat((res1, x4_att_real), 1)
        res2 = F.elu(self.bnT2(self.convT2(res1)))
        x3_att_real = self.Att3_real(g=res2, x=x3)
        res2 = torch.cat((res2, x3_att_real), 1)
        res3 = F.elu(self.bnT3(self.convT3(res2)))
        x2_att_real = self.Att4_real(g=res3, x=x2)
        res3 = torch.cat((res3, x2_att_real), 1)
        res4 = F.elu(self.bnT4(self.convT4(res3)))
        x1_att_real = self.Att5_real(g=res4, x=x1)
        res4 = torch.cat((res4, x1_att_real), 1)
        # (B, o_c, T. F)
        res5 = F.elu(self.bnT5(self.convT5(res4)))
        res5 = self.fcs(res5)

        ## ConvTrans for imag
        x5_att_imag = self.Att1_imag(g=output, x=x5)
        res_imag = torch.cat((output, x5_att_imag), 1)
        phares1 = F.elu(self.phabnT1(self.phaconvT1(res_imag)))
        x4_att_imag = self.Att2_imag(g=phares1, x=x4)
        phares1 = torch.cat((phares1, x4_att_imag), 1)
        phares2 = F.elu(self.phabnT2(self.phaconvT2(phares1)))
        x3_att_imag = self.Att3_imag(g=phares2, x=x3)
        phares2 = torch.cat((phares2, x3_att_imag), 1)
        phares3 = F.elu(self.phabnT3(self.phaconvT3(phares2)))
        x2_att_imag = self.Att4_imag(g=phares3, x=x2)
        phares3 = torch.cat((phares3, x2_att_imag), 1)
        phares4 = F.elu(self.phabnT4(self.phaconvT4(phares3)))
        x1_att_imag = self.Att5_imag(g=phares4, x=x1)
        phares4 = torch.cat((phares4, x1_att_imag), 1)
        # (B, o_c, T. F)
        phares5 = F.elu(self.phabnT5(self.phaconvT5(phares4)))
        phares5 = self.phafcs(phares5)
        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5


class PHA_RNN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    def __init__(self):
        super(PHA_RNN, self).__init__()
        # shared layers

        self.dense_share1 = nn.Linear(161,128)
        self.LSTM_share1 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.dense_share2 = nn.Linear(417, 256)
        self.LSTM_share2 = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True)

        # Decoder for real and imag
        self.dense_real1 = nn.Linear(545, 512)
        self.LSTM_real1 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)
        self.dense_real2 = nn.Linear(512, 161)

        self.dense_imag1 = nn.Linear(545, 512)
        self.LSTM_imag1 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)
        self.dense_imag2 = nn.Linear(512, 161)



    def forward(self, x):
        # conv
        # (B, in_c, T, F)

        real_mag = x[:,0,:,:]
        imag_mag = x[:,1,:,:]


        ##### real part
        real_1 = F.elu(self.dense_share1(real_mag))
        real_2, (hn, cn) = self.LSTM_share1(real_1)

        ##### concat tensor
        real_2_concat = torch.cat([real_1,real_2], dim=2)
        real_2_concat = torch.cat([real_2_concat, real_mag], dim=2)

        real_3 = F.elu(self.dense_share2(real_2_concat))
        real_4, (hn, cn) = self.LSTM_share2(real_3)


        ##### concat tensor
        real_4_concat = torch.cat([real_4,real_2], dim=2)
        real_4_concat = torch.cat([real_4_concat, real_mag], dim=2)

        real_5 = F.elu(self.dense_real1(real_4_concat))
        real_6, (hn, cn) = self.LSTM_real1(real_5)
        real_7 = F.elu(self.dense_real2(real_6))

        ##### imag part
        imag_1 = F.elu(self.dense_share1(imag_mag))
        imag_2, (hn, cn) = self.LSTM_share1(imag_1)


        ##### concat tensor
        imag_2_concat = torch.cat([imag_1,imag_2], dim=2)
        imag_2_concat = torch.cat([imag_2_concat, imag_mag], dim=2)

        imag_3 = F.elu(self.dense_share2(imag_2_concat))
        imag_4, (hn, cn) = self.LSTM_share2(imag_3)

        ##### concat tensor
        imag_4_concat = torch.cat([imag_4,imag_2], dim=2)
        imag_4_concat = torch.cat([imag_4_concat, imag_mag], dim=2)

        imag_5 = F.elu(self.dense_imag1(imag_4_concat))
        imag_6, (hn, cn) = self.LSTM_imag1(imag_5)
        imag_7 = F.elu(self.dense_imag2(imag_6))

        real_7 = torch.unsqueeze(real_7, dim=1)
        imag_7 = torch.unsqueeze(imag_7, dim=1)



        return real_7, imag_7


class PHA_RNN_simple(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    def __init__(self):
        super(PHA_RNN_simple, self).__init__()
        # shared layers

        self.dense_share1 = nn.Linear(1026,256)
        # self.GRU_share1 = nn.GRU(input_size=256, hidden_size=256, num_layers=1, batch_first=True, bias=False)


        # Decoder for real and imag
        self.GRU_real = nn.GRU(input_size=256, hidden_size=256, num_layers=1, batch_first=True, bias=False)
        self.dense_real = nn.Linear(256, 513)

        self.GRU_imag = nn.GRU(input_size=256, hidden_size=256, num_layers=1, batch_first=True, bias=False)
        self.dense_imag = nn.Linear(256, 513)



    def forward(self, x):
        # conv
        # (B, in_c, T, F)

        real_mag = x[:,0,:,:]
        imag_mag = x[:,1,:,:]

        shared_in = torch.cat([real_mag, imag_mag], dim=2)
        shared_out = self.dense_share1(shared_in)

        # shared_GRU_out, hn = self.GRU_share1(shared_out)



        ##### real part

        real_1, hn = self.GRU_real(shared_out)
        real_o = self.dense_real(real_1)


        ##### imag part

        imag_1, hn = self.GRU_imag(shared_out)
        imag_o = self.dense_imag(imag_1)

        ##### concat tensor

        real_o = torch.unsqueeze(real_o, dim=1)
        imag_o = torch.unsqueeze(imag_o, dim=1)



        return real_o, imag_o