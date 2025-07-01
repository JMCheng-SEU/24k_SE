import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


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
        self.in_conv = nn.Conv1d(in_channels = 256, out_channels= 128, kernel_size= 1, stride = 1)
        self.dila_conv_left = nn.Sequential(
            nn.ELU(),
            CausalConv1d(in_channels = 128, out_channels= 128, kernel_size= 3, dilation= dila_rate))
        self.dila_conv_right = nn.Sequential(
            nn.ELU(),
            CausalConv1d(in_channels = 128, out_channels= 128, kernel_size= 3, dilation= dila_rate),
            nn.Sigmoid())
        self.out_conv = nn.Conv1d(in_channels= 128, out_channels= 256, kernel_size = 1, stride = 1)
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


class GRN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(GRN, self).__init__()
        # Encoder
        self.dilconv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(3, 3), stride=(1, 1), dilation=1, padding=(2, 2))
        # self.dilbn1 = nn.BatchNorm2d(num_features=16)
        self.dilconv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), dilation=1, padding=(2, 2))
        # self.dilbn2 = nn.BatchNorm2d(num_features=16)
        self.dilconv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), dilation=2, padding=(4, 4))
        # self.dilbn3 = nn.BatchNorm2d(num_features=32)
        self.dilconv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), dilation=4, padding=(8, 8))
        # self.dilbn4 = nn.BatchNorm2d(num_features=32)

        self.conv1_rs = nn.Conv1d(5152, 256, kernel_size=1, stride=1, padding=0, bias=True)

        self.glu_list = nn.ModuleList([Ca_Dil_GLU(dila_rate=2 ** i) for i in range(4)])

        self.convT1 = nn.Conv1d(256, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.bnT1 = nn.BatchNorm1d(num_features=256)
        self.convT2 = nn.Conv1d(256, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.bnT2 = nn.BatchNorm1d(num_features=128)
        self.convT3 = nn.Conv1d(128, 161, kernel_size=1, stride=1, padding=0, bias=True)
        self.bnT3 = nn.BatchNorm1d(num_features=161)

        self.fcs = nn.Sequential(
            nn.Linear(161, 161),
            nn.ELU(),
            nn.Linear(161, 161),
            nn.ELU()
        )

        self.phaconvT1 = nn.Conv1d(256, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.phabnT1 = nn.BatchNorm1d(num_features=256)
        self.phaconvT2 = nn.Conv1d(256, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.phabnT2 = nn.BatchNorm1d(num_features=128)
        self.phaconvT3 = nn.Conv1d(128, 161, kernel_size=1, stride=1, padding=0, bias=True)
        self.phabnT3 = nn.BatchNorm1d(num_features=161)
        # output_padding为1，不然算出来是79


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
        dilx1 = F.elu(self.dilconv1(x))
        dilx1 = dilx1[:, :, :-2, :-2]
        dilx2 = F.elu(self.dilconv2(dilx1))
        dilx2 = dilx2[:, :, :-2, :-2]
        dilx3 = F.elu(self.dilconv3(dilx2))
        dilx3 = dilx3[:, :, :-4, :-4]
        dilx4 = F.elu(self.dilconv4(dilx3))
        dilx4 = dilx4[:, :, :-8, :-8]

        out5 = dilx4.permute(0, 2, 1, 3)
        out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        dil_in = out5.permute(0, 2, 1)

        T_module_in = F.elu(self.conv1_rs(dil_in))

        for id in range(4):
            x = self.glu_list[id](T_module_in)
            if id == 0:
                skip = x
            else:
                # att_skip = self.attention_gate_list[id-1](g=x, x=skip)
                skip = skip + x

        dil_output = skip

        # predict for real

        realT1 = F.elu(self.bnT1(self.convT1(dil_output)))

        realT2 = F.elu(self.bnT2(self.convT2(realT1)))
        realT3 = F.elu(self.bnT3(self.convT3(realT2)))

        realT3 =realT3.permute(0, 2, 1)
        real = self.fcs(realT3)

        # predict for imag
        imagT1 = F.elu(self.phabnT1(self.phaconvT1(dil_output)))

        imagT2 = F.elu(self.phabnT2(self.phaconvT2(imagT1)))

        imagT3 = F.elu(self.phabnT3(self.phaconvT3(imagT2)))
        imagT3 = imagT3.permute(0, 2, 1)
        # (B, o_c, T. F)
        imag = self.phafcs(imagT3)


        return real, imag


class Ca_Dil_GLU_GCRN(nn.Module):
    def __init__(self, dila_rate):
        super(Ca_Dil_GLU_GCRN, self).__init__()
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

        self.glu_list = nn.ModuleList([Ca_Dil_GLU_GCRN(dila_rate=2 ** i) for i in range(4)])

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
        #lstm
        # lstm, (hn, cn) = self.LSTM1(dil_out)


        output = dil_out.reshape(out5.size()[0],out5.size()[1], 256, -1)
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

