from torch import Tensor, nn
import torch

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
        self.in_conv = nn.Conv1d(in_channels = 50, out_channels= 100, kernel_size= 1, stride = 1)
        self.dila_conv_left = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(in_channels = 100, out_channels= 100, kernel_size= 3, stride = 1,
                                   padding= (3-1) * dila_rate, dilation= dila_rate))
        self.chomp1 = Chomp1d((3-1) * dila_rate)
        self.dila_conv_right = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(in_channels = 100, out_channels= 100, kernel_size= 3, stride = 1,
                                   padding= (3-1) * dila_rate, dilation= dila_rate),
            nn.Sigmoid())
        self.chomp2 = Chomp1d((3-1) * dila_rate)
        self.out_conv = nn.Conv1d(in_channels= 100, out_channels= 50, kernel_size = 1, stride = 1)
        self.out_elu = nn.PReLU()

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
    def __init__(self, dila_rate, in_channels, mid_channels, out_channels):
        super(Ca_Dil_GLU, self).__init__()
        self.in_conv = nn.Conv1d(in_channels = in_channels, out_channels= mid_channels, kernel_size= 1, stride = 1)
        self.dila_conv_left = nn.Sequential(
            nn.PReLU(),
            CausalConv1d(in_channels = mid_channels, out_channels= mid_channels, kernel_size= 3, dilation= dila_rate))
        self.dila_conv_right = nn.Sequential(
            nn.PReLU(),
            CausalConv1d(in_channels = mid_channels, out_channels= mid_channels, kernel_size= 3, dilation= dila_rate),
            nn.Sigmoid())
        self.out_conv = nn.Conv1d(in_channels= mid_channels, out_channels= out_channels, kernel_size = 1, stride = 1)
        self.out_elu = nn.PReLU()

    def forward(self, inpt):
        x = inpt
        x = self.in_conv(x)
        x1 = self.dila_conv_left(x)
        x2 = self.dila_conv_right(x)
        x = x1 * x2
        x = self.out_conv(x)
        # x = x + inpt
        x = self.out_elu(x)
        return x


class newintra_DPRNN_Block(nn.Module):

    def __init__(
        self,
        numUnits: int,
        width: int,
    ):
        super().__init__()
        self.numUnits = numUnits

        self.width = width

        # self.intra_rnn = nn.LSTM(input_size=self.numUnits, hidden_size=self.numUnits // 2, num_layers=1, batch_first=True, bidirectional=True)
        # self.intra_fc = nn.Linear(self.numUnits, self.numUnits)
        #
        # self.intra_ln = nn.LayerNorm(normalized_shape=[self.width, self.numUnits])

        self.intra_rnn = nn.LSTM(input_size=self.width, hidden_size= self.width, num_layers=1, batch_first=True)
        self.intra_fc = nn.Linear(self.width, self.width)

        self.intra_ln = nn.LayerNorm(normalized_shape=[self.width, self.numUnits])

        self.inter_rnn = nn.LSTM(input_size=self.numUnits, hidden_size= self.numUnits, num_layers=1, batch_first=True)
        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)

        self.inter_ln = nn.LayerNorm(normalized_shape=[self.width, self.numUnits])



    def forward(self, input: Tensor) -> Tensor:
        # input shape: [B, C, T, F]

        # Intra-Chunk Processing

        intra_input = input  ## [B, C, T, F]
        intra_RNN_input_rs = intra_input.reshape(intra_input.size()[0] * intra_input.size()[1],
                                                     intra_input.size()[2], intra_input.size()[3])

        intra_RNN_output, _ = self.intra_rnn(intra_RNN_input_rs)
        intra_dense_out = self.intra_fc(intra_RNN_output)

        intra_ln_input = intra_dense_out.reshape(intra_input.size()[0], intra_input.size()[1],
                                                 intra_input.size()[2], intra_input.size()[3])

        intra_ln_input = intra_ln_input.permute(0, 2, 3, 1).contiguous()
        intra_ln_out = self.intra_ln(intra_ln_input)
        intra_out = intra_ln_out.permute(0, 3, 1, 2).contiguous()

        intra_out = intra_out + input


        # intra_RNN_input = input.permute(0, 2, 3, 1) ## [B, T, F, C]
        # intra_RNN_input_rs = intra_RNN_input.reshape(intra_RNN_input.size()[0] * intra_RNN_input.size()[1], intra_RNN_input.size()[2], intra_RNN_input.size()[3])
        #
        # intra_RNN_output, _ = self.intra_rnn(intra_RNN_input_rs)
        # intra_dense_out = self.intra_fc(intra_RNN_output)
        #
        # intra_ln_input = intra_dense_out.reshape(intra_RNN_input.size()[0], intra_RNN_input.size()[1], intra_RNN_input.size()[2], intra_RNN_input.size()[3])
        # intra_ln_out = self.intra_ln(intra_ln_input)
        #
        # intra_out = intra_ln_out.permute(0, 3, 1, 2)

        # intra_out = input

        # Inter-Chunk Processing

        inter_RNN_input = intra_out.permute(0, 3, 2, 1).contiguous()  ## [B, F, T, C]
        inter_RNN_input_rs = inter_RNN_input.reshape(inter_RNN_input.size()[0] * inter_RNN_input.size()[1],
                                                     inter_RNN_input.size()[2], inter_RNN_input.size()[3])
        inter_RNN_output, _ = self.inter_rnn(inter_RNN_input_rs)
        inter_dense_out = self.inter_fc(inter_RNN_output)
        inter_ln_input = inter_dense_out.reshape(inter_RNN_input.size()[0], inter_RNN_input.size()[1], inter_RNN_input.size()[2], inter_RNN_input.size()[3])
        inter_ln_input = inter_ln_input.permute(0, 2, 1, 3).contiguous()
        inter_ln_out = self.inter_ln(inter_ln_input)
        inter_out = inter_ln_out.permute(0,3,1,2).contiguous()

        output = inter_out + intra_out



        return output


class DPRNN_Block(nn.Module):

    def __init__(
        self,
        numUnits: int,
        width: int,
    ):
        super().__init__()
        self.numUnits = numUnits

        self.width = width

        self.intra_rnn = nn.LSTM(input_size=self.numUnits, hidden_size=self.numUnits // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)

        self.intra_ln = nn.LayerNorm(normalized_shape=[self.width, self.numUnits])

        self.inter_rnn = nn.LSTM(input_size=self.numUnits, hidden_size= self.numUnits, num_layers=1, batch_first=True)
        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)

        self.inter_ln = nn.LayerNorm(normalized_shape=[self.width, self.numUnits])



    def forward(self, input: Tensor) -> Tensor:
        # input shape: [B, C, T, F]

        # Intra-Chunk Processing



        intra_RNN_input = input.permute(0, 2, 3, 1) ## [B, T, F, C]
        intra_RNN_input_rs = intra_RNN_input.reshape(intra_RNN_input.size()[0] * intra_RNN_input.size()[1], intra_RNN_input.size()[2], intra_RNN_input.size()[3])

        intra_RNN_output, _ = self.intra_rnn(intra_RNN_input_rs)
        intra_dense_out = self.intra_fc(intra_RNN_output)

        intra_ln_input = intra_dense_out.reshape(intra_RNN_input.size()[0], intra_RNN_input.size()[1], intra_RNN_input.size()[2], intra_RNN_input.size()[3])
        intra_ln_out = self.intra_ln(intra_ln_input)

        intra_out = intra_ln_out.permute(0, 3, 1, 2).contiguous()

        intra_out = intra_out + input

        # Inter-Chunk Processing

        inter_RNN_input = intra_out.permute(0, 3, 2, 1)  ## [B, F, T, C]
        inter_RNN_input_rs = inter_RNN_input.reshape(inter_RNN_input.size()[0] * inter_RNN_input.size()[1],
                                                     inter_RNN_input.size()[2], inter_RNN_input.size()[3])
        inter_RNN_output, _ = self.inter_rnn(inter_RNN_input_rs)
        inter_dense_out = self.inter_fc(inter_RNN_output)
        inter_ln_input = inter_dense_out.reshape(inter_RNN_input.size()[0], inter_RNN_input.size()[1], inter_RNN_input.size()[2], inter_RNN_input.size()[3])
        inter_ln_input = inter_ln_input.permute(0, 2, 1, 3).contiguous()
        inter_ln_out = self.inter_ln(inter_ln_input)
        inter_out = inter_ln_out.permute(0,3,1,2).contiguous()

        output = inter_out + intra_out



        return output



class DP_DIL_CNN_Block_V3(nn.Module):

    def __init__(
        self,
        numUnits: int,
        width: int,
    ):
        super().__init__()
        self.numUnits = numUnits

        self.width = width

        self.intra_rnn = nn.LSTM(input_size=self.numUnits, hidden_size=self.numUnits // 2, num_layers=1, batch_first=True, bidirectional=True)

        # self.intra_glu_list = nn.ModuleList([Ca_Dil_GLU(dila_rate=2 ** i, in_channels=numUnits, mid_channels=numUnits) for i in range(4)])


        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)

        self.intra_ln = nn.LayerNorm(normalized_shape=[self.width, self.numUnits])

        self.inter_rnn = nn.LSTM(input_size=self.numUnits, hidden_size= self.numUnits, num_layers=1, batch_first=True)

        # self.inter_glu_list = nn.ModuleList([Ca_Dil_GLU(dila_rate=2 ** i, in_channels=numUnits, mid_channels=numUnits) for i in range(5)])


        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)

        self.inter_ln = nn.LayerNorm(normalized_shape=[self.width, self.numUnits])

        self.freq_ln = nn.LayerNorm(normalized_shape=[self.width, self.numUnits])

        self.freq_glu_list = nn.ModuleList([Ca_Dil_GLU(dila_rate=2 ** i, in_channels=self.numUnits, mid_channels=self.numUnits, out_channels=self.numUnits) for i in range(5)])

    def forward(self, input: Tensor) -> Tensor:
        # input shape: [B, C, T, F]

        # Intra-Chunk Processing



        intra_RNN_input = input.permute(0, 2, 3, 1).contiguous() ## [B, T, F, C]
        intra_RNN_input_rs = intra_RNN_input.reshape(intra_RNN_input.size()[0] * intra_RNN_input.size()[1], intra_RNN_input.size()[2], intra_RNN_input.size()[3])

        intra_RNN_output, _ = self.intra_rnn(intra_RNN_input_rs)

        # intra_RNN_input_rs = intra_RNN_input_rs.permute(0, 2, 1)
        #
        # for id in range(4):
        #     intra_x = self.intra_glu_list[id](intra_RNN_input_rs)
        #     if id == 0:
        #         intra_skip = intra_x
        #     else:
        #         # att_skip = self.attention_gate_list[id-1](g=x, x=skip)
        #         intra_skip = intra_skip + intra_x
        #
        # intra_RNN_output = intra_skip
        #
        # intra_RNN_output = intra_RNN_output.permute(0, 2, 1)


        intra_dense_out = self.intra_fc(intra_RNN_output)

        intra_ln_input = intra_dense_out.reshape(intra_RNN_input.size()[0], intra_RNN_input.size()[1], intra_RNN_input.size()[2], intra_RNN_input.size()[3])
        intra_ln_out = self.intra_ln(intra_ln_input)

        intra_out = intra_ln_out.permute(0, 3, 1, 2).contiguous()

        intra_out = intra_out + input

        # Inter-Chunk Processing

        inter_RNN_input = intra_out.permute(0, 3, 2, 1).contiguous()  ## [B, F, T, C]
        inter_RNN_input_rs = inter_RNN_input.reshape(inter_RNN_input.size()[0] * inter_RNN_input.size()[1],
                                                     inter_RNN_input.size()[2], inter_RNN_input.size()[3])

        inter_RNN_output, _ = self.inter_rnn(inter_RNN_input_rs)

        # inter_RNN_input_rs = inter_RNN_input_rs.permute(0, 2, 1)
        #
        # for id in range(5):
        #     inter_x = self.inter_glu_list[id](inter_RNN_input_rs)
        #     if id == 0:
        #         inter_skip = inter_x
        #     else:
        #         # att_skip = self.attention_gate_list[id-1](g=x, x=skip)
        #         inter_skip = inter_skip + inter_x
        #
        # inter_RNN_output = inter_skip
        #
        # inter_RNN_output = inter_RNN_output.permute(0, 2, 1)


        inter_dense_out = self.inter_fc(inter_RNN_output)
        inter_ln_input = inter_dense_out.reshape(inter_RNN_input.size()[0], inter_RNN_input.size()[1], inter_RNN_input.size()[2], inter_RNN_input.size()[3])
        inter_ln_input = inter_ln_input.permute(0, 2, 1, 3).contiguous()
        inter_ln_out = self.inter_ln(inter_ln_input)
        inter_out = inter_ln_out.permute(0, 3, 1, 2).contiguous()

        freq_input = intra_out

        freq_input = freq_input.permute(0, 3, 1, 2).contiguous()

        freq_input = freq_input.reshape(input.size()[0] * input.size()[3],
                                                     input.size()[1], input.size()[2])

        # freq_input = freq_input.permute(0, 2, 1).contiguous()

        for id in range(5):
            freq_x =  self.freq_glu_list[id](freq_input)
            if id == 0:
                freq_skip = freq_x
            else:
                # att_skip = self.attention_gate_list[id-1](g=x, x=skip)
                freq_skip = freq_skip + freq_x

        freq_DIL_output = freq_skip

        # freq_DIL_output = freq_DIL_output.permute(0, 2, 1).contiguous()

        freq_DIL_output = freq_DIL_output.reshape(input.size()[0], input.size()[3],
                                                     input.size()[1], input.size()[2])

        freq_DIL_output = freq_DIL_output.permute(0, 3, 1, 2).contiguous()

        freq_DIL_norm_out = self.freq_ln(freq_DIL_output)

        freq_DIL_norm_out = freq_DIL_norm_out.permute(0, 3, 1, 2).contiguous()

        output = inter_out + intra_out + freq_DIL_norm_out



        return output

class DPCRN_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_ln = nn.LayerNorm(normalized_shape=[257, 2])

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),

        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )

        self.DPRNN_1 = DPRNN_Block(numUnits=128, width=64)
        self.DPRNN_2 = DPRNN_Block(numUnits=128, width=64)

        # self.DPRNN_1 = newintra_DPRNN_Block(numUnits=128, width=64)
        # self.DPRNN_2 = newintra_DPRNN_Block(numUnits=128, width=64)

        # self.DPRNN_1 = DP_DIL_CNN_Block_V3(numUnits=128, width=64)
        # self.DPRNN_2 = DP_DIL_CNN_Block_V3(numUnits=128, width=64)


        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )


        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0)),
        )



    def forward(
        self, spec: Tensor
    ) -> [Tensor, Tensor]:



        # spec: [B, 2, T, Fc]
        b, c, t, f = spec.shape
        mid_fea = []

        input_ln_in = spec.permute(0, 2, 3, 1).contiguous()

        input_ln_out = self.input_ln(input_ln_in)

        conv_input = input_ln_out.permute(0, 3, 1, 2).contiguous()

        conv_out1 = self.conv1(conv_input)

        conv_out2 = self.conv2(conv_out1)

        conv_out3 = self.conv3(conv_out2)

        conv_out4 = self.conv4(conv_out3)

        conv_out5 = self.conv5(conv_out4)

        mid_fea.append(conv_out5)

        DPRNN_out1 = self.DPRNN_1(conv_out5)

        DPRNN_out2 = self.DPRNN_2(DPRNN_out1)

        mid_fea.append(DPRNN_out2)

        convT1_input = torch.cat((conv_out5, DPRNN_out2), 1)
        convT1_out = self.convT1(convT1_input)

        convT2_input = torch.cat((conv_out4, convT1_out[:, :,:,:-2]), 1)
        convT2_out = self.convT2(convT2_input)

        convT3_input = torch.cat((conv_out3, convT2_out[:, :,:,:-2]), 1)
        convT3_out = self.convT3(convT3_input)

        convT4_input = torch.cat((conv_out2, convT3_out[:, :,:,:-2]), 1)
        convT4_out = self.convT4(convT4_input)

        convT5_input = torch.cat((conv_out1, convT4_out[:, :,:,:-1]), 1)
        convT5_out = self.convT5(convT5_input)

        mask_out = convT5_out[:, :, :, :-2]

        mask_real = mask_out[:, 0, :, :]
        mask_imag = mask_out[:, 1, :, :]

        noisy_real = spec[:, 0, :, :]
        noisy_imag = spec[:, 1, :, :]

        enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        enh_imag = noisy_real * mask_imag + noisy_imag * mask_real

        #### recons_DCCRN-E

        # spec_mags = torch.sqrt(noisy_real ** 2 + noisy_imag ** 2 + 1e-8)
        # spec_phase = torch.atan2(noisy_imag, noisy_real)
        #
        # mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
        # real_phase = mask_real / (mask_mags + 1e-8)
        # imag_phase = mask_imag / (mask_mags + 1e-8)
        # mask_phase = torch.atan2(
        #     imag_phase,
        #     real_phase
        # )
        # # mask_mags = torch.tanh(mask_mags)
        # est_mags = mask_mags * spec_mags
        # est_phase = spec_phase + mask_phase
        # enh_real = est_mags * torch.cos(est_phase)
        # enh_imag = est_mags * torch.sin(est_phase)
        #

        return enh_real, enh_imag




class DPCRN_Model_Streamer(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_ln = nn.LayerNorm(normalized_shape=[257, 2])

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),

        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )

        # self.DPRNN_1 = DPRNN_Block(numUnits=128, width=64)
        # self.DPRNN_2 = DPRNN_Block(numUnits=128, width=64)

        # self.DPRNN_1 = newintra_DPRNN_Block(numUnits=128, width=64)
        # self.DPRNN_2 = newintra_DPRNN_Block(numUnits=128, width=64)

        # self.DPRNN_1 = DP_DIL_CNN_Block_V3(numUnits=128, width=64)
        # self.DPRNN_2 = DP_DIL_CNN_Block_V3(numUnits=128, width=64)


        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )


        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0)),
        )



    def forward(
        self, spec: Tensor
    ) -> [Tensor, Tensor]:



        # spec: [B, 2, T, Fc]
        b, c, t, f = spec.shape

        input_ln_in = spec.permute(0, 2, 3, 1).contiguous()

        input_ln_out = self.input_ln(input_ln_in)

        conv_input = input_ln_out.permute(0, 3, 1, 2).contiguous()

        conv_out1 = self.conv1(conv_input)

        conv_out2 = self.conv2(conv_out1)

        conv_out3 = self.conv3(conv_out2)

        conv_out4 = self.conv4(conv_out3)

        conv_out5 = self.conv5(conv_out4)

        # DPRNN_out1 = self.DPRNN_1(conv_out5)
        #
        # DPRNN_out2 = self.DPRNN_2(DPRNN_out1)

        convT1_input = torch.cat((conv_out5, conv_out5), 1)
        convT1_out = self.convT1(convT1_input)

        convT2_input = torch.cat((conv_out4, convT1_out[:, :,:,:-2]), 1)
        convT2_out = self.convT2(convT2_input)

        convT3_input = torch.cat((conv_out3, convT2_out[:, :,:,:-2]), 1)
        convT3_out = self.convT3(convT3_input)

        convT4_input = torch.cat((conv_out2, convT3_out[:, :,:,:-2]), 1)
        convT4_out = self.convT4(convT4_input)

        convT5_input = torch.cat((conv_out1, convT4_out[:, :,:,:-1]), 1)
        convT5_out = self.convT5(convT5_input)

        mask_out = convT5_out[:, :, :, :-2]

        mask_real = mask_out[:, 0, :, :]
        mask_imag = mask_out[:, 1, :, :]

        noisy_real = spec[:, 0, :, :]
        noisy_imag = spec[:, 1, :, :]

        # enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        # enh_imag = noisy_real * mask_imag + noisy_imag * mask_real

        #### recons_DCCRN-E

        spec_mags = torch.sqrt(noisy_real ** 2 + noisy_imag ** 2 + 1e-8)
        spec_phase = torch.atan2(noisy_imag, noisy_real)

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
        #

        return enh_real, enh_imag





if __name__ == '__main__':
    inputs = torch.randn(16, 2, 100, 257)

    Model = DPCRN_Model()

    enh_real, enh_imag = Model(inputs)

    print(enh_real.shape)