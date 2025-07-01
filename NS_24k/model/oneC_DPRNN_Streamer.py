from torch import Tensor, nn
import torch


class DPRNN_Block_RT(nn.Module):

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

        self.inter_lstm_state = None

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
        inter_RNN_output, self.inter_lstm_state = self.inter_rnn(inter_RNN_input_rs, self.inter_lstm_state)
        inter_dense_out = self.inter_fc(inter_RNN_output)
        inter_ln_input = inter_dense_out.reshape(inter_RNN_input.size()[0], inter_RNN_input.size()[1], inter_RNN_input.size()[2], inter_RNN_input.size()[3])
        inter_ln_input = inter_ln_input.permute(0, 2, 1, 3).contiguous()
        inter_ln_out = self.inter_ln(inter_ln_input)
        inter_out = inter_ln_out.permute(0,3,1,2).contiguous()

        output = inter_out + intra_out



        return output





class newintra_DPRNN_Block_RT(nn.Module):

    def __init__(
        self,
        numUnits: int,
        width: int,
    ):
        super().__init__()
        self.numUnits = numUnits

        self.width = width

        self.intra_rnn = nn.LSTM(input_size=self.width, hidden_size= self.width, num_layers=1, batch_first=True)

        self.intra_lstm_state = None
        self.intra_fc = nn.Linear(self.width, self.width)

        self.intra_ln = nn.LayerNorm(normalized_shape=[self.width, self.numUnits])

        self.inter_rnn = nn.LSTM(input_size=self.numUnits, hidden_size= self.numUnits, num_layers=1, batch_first=True)
        self.inter_lstm_state = None

        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)

        self.inter_ln = nn.LayerNorm(normalized_shape=[self.width, self.numUnits])



    def forward(self, input: Tensor) -> Tensor:
        # input shape: [B, C, T, F]

        # Intra-Chunk Processing

        intra_input = input  ## [B, C, T, F]
        intra_RNN_input_rs = intra_input.reshape(intra_input.size()[0] * intra_input.size()[1],
                                                     intra_input.size()[2], intra_input.size()[3])

        intra_RNN_output, self.intra_lstm_state = self.intra_rnn(intra_RNN_input_rs, self.intra_lstm_state)
        # intra_RNN_output, _ = self.intra_rnn(intra_RNN_input_rs)
        intra_dense_out = self.intra_fc(intra_RNN_output)

        intra_ln_input = intra_dense_out.reshape(intra_input.size()[0], intra_input.size()[1],
                                                 intra_input.size()[2], intra_input.size()[3])

        intra_ln_input = intra_ln_input.permute(0, 2, 3, 1).contiguous()
        intra_ln_out = self.intra_ln(intra_ln_input)
        intra_out = intra_ln_out.permute(0, 3, 1, 2).contiguous()

        intra_out = intra_out + input


        inter_RNN_input = intra_out.permute(0, 3, 2, 1).contiguous()  ## [B, F, T, C]
        inter_RNN_input_rs = inter_RNN_input.reshape(inter_RNN_input.size()[0] * inter_RNN_input.size()[1],
                                                     inter_RNN_input.size()[2], inter_RNN_input.size()[3])
        inter_RNN_output, self.inter_lstm_state = self.inter_rnn(inter_RNN_input_rs, self.inter_lstm_state)
        # inter_RNN_output, _ = self.inter_rnn(inter_RNN_input_rs)
        inter_dense_out = self.inter_fc(inter_RNN_output)
        inter_ln_input = inter_dense_out.reshape(inter_RNN_input.size()[0], inter_RNN_input.size()[1], inter_RNN_input.size()[2], inter_RNN_input.size()[3])
        inter_ln_input = inter_ln_input.permute(0, 2, 1, 3).contiguous()
        inter_ln_out = self.inter_ln(inter_ln_input)
        inter_out = inter_ln_out.permute(0,3,1,2).contiguous()

        output = inter_out + intra_out



        return output


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

        # self.DPRNN_1 = newintra_DPRNN_Block_RT(numUnits=128, width=64)
        # self.DPRNN_2 = newintra_DPRNN_Block_RT(numUnits=128, width=64)

        self.DPRNN_1 = DPRNN_Block_RT(numUnits=128, width=64)
        self.DPRNN_2 = DPRNN_Block_RT(numUnits=128, width=64)

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

        DPRNN_out1 = self.DPRNN_1(conv_out5)

        DPRNN_out2 = self.DPRNN_2(DPRNN_out1)

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


class Simple_Streamer(nn.Module):
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



        self.numUnits = 128

        self.width = 64

        self.intra_rnn = nn.LSTM(input_size=self.width, hidden_size= self.width, num_layers=1, batch_first=True)

        self.intra_lstm_state = None
        self.intra_fc = nn.Linear(self.width, self.width)

        self.intra_ln = nn.LayerNorm(normalized_shape=[self.width, self.numUnits])


        self.inter_rnn = nn.LSTM(input_size=self.numUnits, hidden_size= self.numUnits, num_layers=1, batch_first=True)
        self.inter_lstm_state = None

        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)

        self.inter_ln = nn.LayerNorm(normalized_shape=[self.width, self.numUnits])



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

        intra_input = conv_out5  ## [B, C, T, F]
        intra_RNN_input_rs = intra_input.reshape(intra_input.size()[0] * intra_input.size()[1],
                                                     intra_input.size()[2], intra_input.size()[3])

        intra_RNN_output, self.intra_lstm_state = self.intra_rnn(intra_RNN_input_rs, self.intra_lstm_state)

        # intra_RNN_output, _ = self.intra_rnn(intra_RNN_input_rs)
        intra_dense_out = self.intra_fc(intra_RNN_output)

        intra_ln_input = intra_dense_out.reshape(intra_input.size()[0], intra_input.size()[1],
                                                 intra_input.size()[2], intra_input.size()[3])

        intra_ln_input = intra_ln_input.permute(0, 2, 3, 1).contiguous()
        intra_ln_out = self.intra_ln(intra_ln_input)
        intra_out = intra_ln_out.permute(0, 3, 1, 2).contiguous()

        intra_out = intra_out + conv_out5


        inter_RNN_input = intra_out.permute(0, 3, 2, 1).contiguous()  ## [B, F, T, C]
        inter_RNN_input_rs = inter_RNN_input.reshape(inter_RNN_input.size()[0] * inter_RNN_input.size()[1],
                                                     inter_RNN_input.size()[2], inter_RNN_input.size()[3])
        inter_RNN_output, self.inter_lstm_state = self.inter_rnn(inter_RNN_input_rs, self.inter_lstm_state)
        inter_dense_out = self.inter_fc(inter_RNN_output)
        inter_ln_input = inter_dense_out.reshape(inter_RNN_input.size()[0], inter_RNN_input.size()[1], inter_RNN_input.size()[2], inter_RNN_input.size()[3])
        inter_ln_input = inter_ln_input.permute(0, 2, 1, 3).contiguous()
        inter_ln_out = self.inter_ln(inter_ln_input)
        inter_out = inter_ln_out.permute(0,3,1,2).contiguous()

        DPRNN_out = inter_out + intra_out


        convT1_input = torch.cat((conv_out5, DPRNN_out), 1)
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

    Model = DPCRN_Model_Streamer()

    enh_real, enh_imag = Model(inputs)

    print(enh_real.shape)

    # Model = Simple_Streamer()
    #
    # output = Model(inputs)
    # print(output.shape)