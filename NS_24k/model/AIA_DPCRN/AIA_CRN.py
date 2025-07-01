import torch
from torch.nn.parameter import Parameter
from torch import Tensor, nn
from model.AIA_DPCRN.aia_net import AHAM, TransformerEncoderLayer, TransformerEncoderLayer_new
from thop import profile
from model.multiframe import DF


class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            )
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                ),
            )
            setattr(
                self,
                "norm{}".format(i + 1),
                nn.InstanceNorm2d(in_channels, affine=True),
            )
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out



class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            )
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                ),
            )
            setattr(
                self,
                "norm{}".format(i + 1),
                nn.InstanceNorm2d(in_channels, affine=True),
            )
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(DenseEncoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels//2, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels//2, affine=True),
            nn.PReLU(channels//2),
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels//2)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels//2, channels, (1, 3), (1, 2), padding=(0, 0)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

    def forward(self, x):
        enc_list = []
        x1 = self.conv_1(x)
        enc_list.append(x1)
        x2 = self.dilated_dense(x1)
        enc_list.append(x2)
        x3 = self.conv_2(x2)
        enc_list.append(x3)
        return x3, enc_list



class DenseEncoder_Res(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(DenseEncoder_Res, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 0)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)

    def forward(self, x):

        x1 = self.conv_1(x)


        x2 = self.conv_2(x1)


        x3 = self.dilated_dense(x2)

        return x3





class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1, last_padding = 0):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1), padding=(0, last_padding)
        )
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.prelu = nn.PReLU(out_channel)
        self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
        return self.prelu_out(x).permute(0, 2, 1).unsqueeze(1)


class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64):
        super(ComplexDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        # self.sub_pixel = SPConvTranspose2d(num_channel, num_channel//2, (1, 3), 2)
        self.sub_pixel = nn.ConvTranspose2d(in_channels=num_channel, out_channels=num_channel//2, kernel_size=(1, 3), stride=(1, 2))
        self.prelu = nn.PReLU(num_channel//2)
        self.norm = nn.InstanceNorm2d(num_channel//2, affine=True)



        # self.sub_pixel_2 = SPConvTranspose2d(num_channel//2, num_channel//2, (1, 3), 2, last_padding=1)
        self.sub_pixel_2 = nn.ConvTranspose2d(in_channels=num_channel//2, out_channels=num_channel//2, kernel_size=(1, 3), stride=(1, 2))
        self.prelu_2 = nn.PReLU(num_channel//2)
        self.norm_2 = nn.InstanceNorm2d(num_channel//2, affine=True)

        self.conv = nn.Conv2d(num_channel//2, 2, (1, 2))

    def forward(self, x):
        dec_list = []
        x_dense_out = self.dense_block(x)
        dec_list.append(x_dense_out)
        x1 = self.sub_pixel(x_dense_out)
        x1_out = self.prelu(self.norm(x1))

        dec_list.append(x1_out)

        x2 = self.sub_pixel_2(x1_out)
        x2_out = self.prelu_2(self.norm_2(x2))

        dec_list.append(x2_out)

        x3 = self.conv(x2_out)
        x3 = x3[:, :, :, :-1]

        dec_list.append(x3)


        return x3, dec_list










class ComplexDecoder_Res(nn.Module):
    def __init__(self, num_channel=64):
        super(ComplexDecoder_Res, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        # self.sub_pixel = SPConvTranspose2d(num_channel, num_channel//2, (1, 3), 2)
        self.sub_pixel = nn.ConvTranspose2d(in_channels=num_channel, out_channels=num_channel//2, kernel_size=(1, 3), stride=(1, 2))
        self.prelu = nn.PReLU(num_channel//2)
        self.norm = nn.InstanceNorm2d(num_channel//2, affine=True)



        # self.sub_pixel_2 = SPConvTranspose2d(num_channel//2, num_channel//2, (1, 3), 2, last_padding=1)
        self.sub_pixel_2 = nn.ConvTranspose2d(in_channels=num_channel//2, out_channels=num_channel//2, kernel_size=(1, 3), stride=(1, 2))
        self.prelu_2 = nn.PReLU(num_channel//2)
        self.norm_2 = nn.InstanceNorm2d(num_channel//2, affine=True)

        self.conv = nn.Conv2d(num_channel//2, 2, (1, 2))

    def forward(self, x):
        dec_list = []
        x_dense_out = self.dense_block(x)
        dec_list.append(x_dense_out)
        x1 = self.sub_pixel(x_dense_out)
        x1_out = self.prelu(self.norm(x1))

        dec_list.append(x1_out)

        x2 = self.sub_pixel_2(x1_out)
        x2_out = self.prelu_2(self.norm_2(x2))

        dec_list.append(x2_out)

        x3 = self.conv(x2_out)
        x3 = x3[:, :, :, :-1]

        dec_list.append(x3)


        return x3, dec_list



class ComplexDecoder_Res_new(nn.Module):
    def __init__(self, num_channel=64):
        super(ComplexDecoder_Res_new, self).__init__()

        # self.sub_pixel = SPConvTranspose2d(num_channel, num_channel//2, (1, 3), 2)

        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)

        self.Trans_conv_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_channel, out_channels=num_channel, kernel_size=(1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(num_channel, affine=True),
            nn.PReLU(num_channel),
        )

        self.Trans_conv_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_channel, out_channels=2, kernel_size=(1, 3), stride=(1, 2)),

        )


    def forward(self, x):


        x1 = self.dense_block(x)

        x2 = self.Trans_conv_1(x1)

        x3 = self.Trans_conv_2(x2)

        x3 = x3[:, :, :, :-2]

        return x3




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

        intra_out = intra_ln_out.permute(0, 3, 1, 2)

        intra_out = intra_out + input

        # Inter-Chunk Processing

        inter_RNN_input = intra_out.permute(0, 3, 2, 1)  ## [B, F, T, C]
        inter_RNN_input_rs = inter_RNN_input.reshape(inter_RNN_input.size()[0] * inter_RNN_input.size()[1],
                                                     inter_RNN_input.size()[2], inter_RNN_input.size()[3])
        inter_RNN_output, _ = self.inter_rnn(inter_RNN_input_rs)
        inter_dense_out = self.inter_fc(inter_RNN_output)
        inter_ln_input = inter_dense_out.reshape(inter_RNN_input.size()[0], inter_RNN_input.size()[1], inter_RNN_input.size()[2], inter_RNN_input.size()[3])
        inter_ln_input = inter_ln_input.permute(0, 2, 1, 3)
        inter_ln_out = self.inter_ln(inter_ln_input)
        inter_out = inter_ln_out.permute(0,3,1,2)

        output = inter_out + intra_out



        return output


class DPDCRNN_Block_merge(nn.Module):

    def __init__(
        self,
        numUnits: int,
        width: int,
    ):
        super().__init__()
        self.numUnits = numUnits

        self.width = width


        self.intra_dil_trans = nn.ModuleList([])
        self.intra_rnn_trans = nn.ModuleList([])
        self.intra_fc_trans = nn.ModuleList([])
        self.intra_ln_trans = nn.ModuleList([])

        self.inter_dil_trans = nn.ModuleList([])
        self.inter_rnn_trans = nn.ModuleList([])
        self.inter_fc_trans = nn.ModuleList([])
        self.inter_ln_trans = nn.ModuleList([])

        self.num_layers = 4

        for i in range(self.num_layers):
            self.intra_dil_trans.append(DilatedDenseNet(depth=4, in_channels=self.numUnits))
            self.intra_rnn_trans.append(nn.LSTM(input_size=self.numUnits, hidden_size=self.numUnits // 2, num_layers=1, batch_first=True, bidirectional=True))
            # self.intra_rnn_trans.append(MHA_GRU_Block(d_model=self.numUnits, nhead=4, bidirectional=True))
            # self.intra_rnn_trans.append(TransformerEncoderLayer_new(d_model=self.numUnits, nhead=4, dropout=0, bidirectional=True))
            self.intra_fc_trans.append(nn.Linear(self.numUnits, self.numUnits))
            self.intra_ln_trans.append(nn.LayerNorm(normalized_shape=[self.width, self.numUnits]))

            self.inter_rnn_trans.append(nn.LSTM(input_size=self.numUnits, hidden_size= self.numUnits // 2, num_layers=1, batch_first=True, bidirectional=True))
            # self.inter_rnn_trans.append(MHA_GRU_Block(d_model=self.numUnits, nhead=4, bidirectional=False))
            # self.inter_rnn_trans.append(TransformerEncoderLayer_new(d_model=self.numUnits, nhead=4, dropout=0, bidirectional=True))
            self.inter_fc_trans.append(nn.Linear(self.numUnits, self.numUnits))
            self.inter_ln_trans.append(nn.LayerNorm(normalized_shape=[self.width, self.numUnits]))
            self.inter_dil_trans.append(DilatedDenseNet(depth=4, in_channels=self.numUnits))


    def forward(self, input):

        # input shape: [B, C, T, F]

        output_list_all = []

        for i in range(len(self.intra_dil_trans)):

            if i >=1:
                output_i = output_list_all[-1] + input
            else: output_i = input

            # Intra-Chunk Processing

            intra_DCN_input = output_i.permute(0, 1, 3, 2) ## [B, C, F, T]
            intra_DCN_output = self.intra_dil_trans[i](intra_DCN_input)

            intra_DCN_output = intra_DCN_output.permute(0,1,3,2)


            intra_RNN_input = intra_DCN_output.permute(0, 2, 3, 1) ## [B, T, F, C]
            intra_RNN_input_rs = intra_RNN_input.reshape(intra_RNN_input.size()[0] * intra_RNN_input.size()[1], intra_RNN_input.size()[2], intra_RNN_input.size()[3])
            # intra_RNN_input_rs = intra_RNN_input_rs.permute(1, 0, 2)
            # intra_RNN_output = self.intra_rnn_trans[i](intra_RNN_input_rs)
            # intra_RNN_output = intra_RNN_output.permute(1, 0, 2)

            intra_RNN_output, _ = self.intra_rnn_trans[i](intra_RNN_input_rs)

            intra_dense_out = self.intra_fc_trans[i](intra_RNN_output)

            intra_ln_input = intra_dense_out.reshape(intra_RNN_input.size()[0], intra_RNN_input.size()[1], intra_RNN_input.size()[2], intra_RNN_input.size()[3])
            intra_ln_out = self.intra_ln_trans[i](intra_ln_input)

            intra_out = intra_ln_out.permute(0, 3, 1, 2)

            intra_out = intra_out + intra_DCN_output

            # Inter-Chunk Processing

            inter_RNN_input = intra_out.permute(0, 3, 2, 1)  ## [B, F, T, C]
            inter_RNN_input_rs = inter_RNN_input.reshape(inter_RNN_input.size()[0] * inter_RNN_input.size()[1],
                                                         inter_RNN_input.size()[2], inter_RNN_input.size()[3])
            inter_RNN_output, _ = self.inter_rnn_trans[i](inter_RNN_input_rs)
            # inter_RNN_input_rs = inter_RNN_input_rs.permute(1, 0, 2)
            # inter_RNN_output = self.inter_rnn_trans[i](inter_RNN_input_rs)
            # inter_RNN_output = inter_RNN_output.permute(1, 0, 2)

            inter_dense_out = self.inter_fc_trans[i](inter_RNN_output)
            inter_ln_input = inter_dense_out.reshape(inter_RNN_input.size()[0], inter_RNN_input.size()[1], inter_RNN_input.size()[2], inter_RNN_input.size()[3])
            inter_ln_input = inter_ln_input.permute(0, 2, 1, 3)
            inter_ln_out = self.inter_ln_trans[i](inter_ln_input)
            inter_out = inter_ln_out.permute(0,3,1,2)

            output = inter_out + intra_out
            # output = inter_out

            inter_DCN_output = self.inter_dil_trans[i](output)

            output_list_all.append(inter_DCN_output)



        return inter_DCN_output, output_list_all





class AIA_Transformer_onelayer(nn.Module):
    """
    Adaptive time-frequency attention Transformer without interaction on maginitude path and complex path.
    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
    """

    def __init__(self, input_size,output_size, dropout=0):
        super(AIA_Transformer_onelayer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.k1 = Parameter(torch.ones(1))
        self.k2 = Parameter(torch.ones(1))

        self.input = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=1),
            nn.PReLU()
        )

        # dual-path RNN
        self.row_trans = TransformerEncoderLayer(d_model=input_size//2, nhead=4, dropout=dropout, bidirectional=True)
        self.col_trans = TransformerEncoderLayer(d_model=input_size//2, nhead=4, dropout=dropout, bidirectional=True)
        self.row_norm = nn.GroupNorm(1, input_size//2, eps=1e-8)
        self.col_norm = nn.GroupNorm(1, input_size//2, eps=1e-8)


        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size//2, output_size, 1)
                                    )

    def forward(self, input):
        #  input --- [b,  c,  num_frames, frame_size]  --- [b, c, dim2, dim1]
        b, c, dim2, dim1 = input.shape

        output = self.input(input)

        row_input = output.permute(3, 0, 2, 1).contiguous().view(dim1, b*dim2, -1)  # [F, B*T, c]
        row_output = self.row_trans(row_input)  # [F, B*T, c]
        row_output = row_output.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()  # [B, C, T, F]
        row_output = self.row_norm(row_output)  # [B, C, T, F]

        col_input = output.permute(2, 0, 3, 1).contiguous().view(dim2, b*dim1, -1)
        col_output = self.col_trans(col_input)
        col_output = col_output.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()
        col_output = self.col_norm(col_output)
        output_single = output + self.k1*row_output + self.k2*col_output

        output_i = self.output(output_single)
        del row_input, row_output, col_input, col_output

        return output_i








class AIA_CRN_SIG_oneL(nn.Module):
    def __init__(self):
        super().__init__()

        # self.input_ln = nn.LayerNorm(normalized_shape=[201, 2])

        self.dense_encoder = DenseEncoder_Res(in_channel=2, channels=128)


        self.dual_trans = AIA_Transformer_onelayer(128, 128)

        # self.DPRNN_1 = DPRNN_Block(numUnits=128, width=64)
        # self.DPRNN_2 = DPRNN_Block(numUnits=128, width=64)

        self.complex_decoder = ComplexDecoder_Res_new(num_channel=128)


    def forward(
        self, spec
    ):

        # spec: [B, 2, T, Fc]
        b, c, t, f = spec.shape

        out_1, enc_list = self.dense_encoder(spec)

        x_last = self.dual_trans(out_1) #BCTF, #BCTFG

        # DPRNN_out1 = self.DPRNN_1(out_1)
        # DPRNN_out2 = self.DPRNN_2(DPRNN_out1)

        complex_out, dec_list = self.complex_decoder(x_last)
        # mask_out = convT5_out[:, :, :, :-2]

        mask_real = complex_out[:, 0, :, :]
        mask_imag = complex_out[:, 1, :, :]

        noisy_real = spec[:, 0, :, :]
        noisy_imag = spec[:, 1, :, :]

        ####### simple complex reconstruct

        # enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        # enh_imag = noisy_real * mask_imag + noisy_imag * mask_real

        ####### reconstruct through DCCRN-E
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
        # mask_mags = torch.tanh(mask_mags)
        est_mags = mask_mags * spec_mags
        est_phase = spec_phase + mask_phase
        enh_real = est_mags * torch.cos(est_phase)
        enh_imag = est_mags * torch.sin(est_phase)


        return enh_real, enh_imag, x_last, enc_list, dec_list


class DF_AIA_CRN_oneL(nn.Module):
    def __init__(self):
        super().__init__()

        self.df_order = 5
        self.df_bins = 257

        # self.input_ln = nn.LayerNorm(normalized_shape=[201, 2])

        self.dense_encoder = DenseEncoder_Res(in_channel=2, channels=128)


        self.dual_trans = AIA_Transformer_onelayer(128, 128)

        # self.DPRNN_1 = DPRNN_Block(numUnits=128, width=64)
        # self.DPRNN_2 = DPRNN_Block(numUnits=128, width=64)

        self.complex_decoder = ComplexDecoder_Res_new(num_channel=128)

        self.DF_complex_decoder = DF_ComplexDecoder_Res_new(num_channel=128, df_order=5)

        self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)

        self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)


    def forward(
        self, spec
    ):

        # spec: [B, 2, T, Fc]
        b, c, t, f = spec.shape

        out_1 = self.dense_encoder(spec)

        x_last = self.dual_trans(out_1) #BCTF, #BCTFG

        # DPRNN_out1 = self.DPRNN_1(out_1)
        # DPRNN_out2 = self.DPRNN_2(DPRNN_out1)

        complex_out = self.complex_decoder(x_last)
        # mask_out = convT5_out[:, :, :, :-2]

        df_coefs = self.DF_complex_decoder(x_last)

        df_coefs = df_coefs.permute(0, 2, 3, 1)

        df_coefs = self.df_out_transform(df_coefs).contiguous()





        mask_real = complex_out[:, 0, :, :]
        mask_imag = complex_out[:, 1, :, :]

        noisy_real = spec[:, 0, :, :]
        noisy_imag = spec[:, 1, :, :]

        ####### simple complex reconstruct

        enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        enh_imag = noisy_real * mask_imag + noisy_imag * mask_real

        enhanced_D = torch.stack([enh_real, enh_imag], 3)

        enhanced_D = enhanced_D.unsqueeze(1)

        DF_spec = self.df_op(enhanced_D.clone(), df_coefs)

        DF_spec = DF_spec.squeeze(1)

        DF_real = DF_spec[:, :, :, 0]
        DF_imag = DF_spec[:, :, :, 1]


        return DF_real, DF_imag




class DPDCRNN_Block_merge_new(nn.Module):

    def __init__(
        self,
        numUnits: int,
        width: int,
    ):
        super().__init__()
        self.numUnits = numUnits

        self.width = width


        self.intra_dil_trans = nn.ModuleList([])
        self.intra_rnn_trans = nn.ModuleList([])
        self.intra_fc_trans = nn.ModuleList([])
        self.intra_ln_trans = nn.ModuleList([])

        self.inter_dil_trans = nn.ModuleList([])
        self.inter_rnn_trans = nn.ModuleList([])
        self.inter_fc_trans = nn.ModuleList([])
        self.inter_ln_trans = nn.ModuleList([])

        self.num_layers = 4

        for i in range(self.num_layers):
            self.intra_dil_trans.append(DilatedDenseNet(depth=4, in_channels=self.numUnits))
            self.intra_rnn_trans.append(nn.LSTM(input_size=self.numUnits, hidden_size=self.numUnits // 2, num_layers=1, batch_first=True, bidirectional=True))
            # self.intra_rnn_trans.append(MHA_GRU_Block(d_model=self.numUnits, nhead=4, bidirectional=True))
            # self.intra_rnn_trans.append(TransformerEncoderLayer_new(d_model=self.numUnits, nhead=4, dropout=0, bidirectional=True))
            self.intra_fc_trans.append(nn.Linear(self.numUnits, self.numUnits))
            self.intra_ln_trans.append(nn.LayerNorm(normalized_shape=[self.width, self.numUnits]))

            self.inter_rnn_trans.append(nn.LSTM(input_size=self.numUnits, hidden_size= self.numUnits // 2, num_layers=1, batch_first=True, bidirectional=True))
            # self.inter_rnn_trans.append(MHA_GRU_Block(d_model=self.numUnits, nhead=4, bidirectional=False))
            # self.inter_rnn_trans.append(TransformerEncoderLayer_new(d_model=self.numUnits, nhead=4, dropout=0, bidirectional=True))
            self.inter_fc_trans.append(nn.Linear(self.numUnits, self.numUnits))
            self.inter_ln_trans.append(nn.LayerNorm(normalized_shape=[self.width, self.numUnits]))
            self.inter_dil_trans.append(DilatedDenseNet(depth=4, in_channels=self.numUnits))


    def forward(self, input):

        # input shape: [B, C, T, F]

        output_list_all = []

        for i in range(len(self.intra_dil_trans)):

            if i >=1:
                output_i = output_list_all[-1] + input
            else: output_i = input

            # Intra-Chunk Processing

            intra_DCN_input = output_i.permute(0, 1, 3, 2) ## [B, C, F, T]
            intra_DCN_output = self.intra_dil_trans[i](intra_DCN_input)

            intra_DCN_output = intra_DCN_output.permute(0,1,3,2)


            intra_RNN_input = intra_DCN_output.permute(0, 2, 3, 1) ## [B, T, F, C]
            intra_RNN_input_rs = intra_RNN_input.reshape(intra_RNN_input.size()[0] * intra_RNN_input.size()[1], intra_RNN_input.size()[2], intra_RNN_input.size()[3])
            # intra_RNN_input_rs = intra_RNN_input_rs.permute(1, 0, 2)
            # intra_RNN_output = self.intra_rnn_trans[i](intra_RNN_input_rs)
            # intra_RNN_output = intra_RNN_output.permute(1, 0, 2)

            intra_RNN_output, _ = self.intra_rnn_trans[i](intra_RNN_input_rs)

            intra_dense_out = self.intra_fc_trans[i](intra_RNN_output)

            intra_ln_input = intra_dense_out.reshape(intra_RNN_input.size()[0], intra_RNN_input.size()[1], intra_RNN_input.size()[2], intra_RNN_input.size()[3])
            intra_ln_out = self.intra_ln_trans[i](intra_ln_input)

            intra_out = intra_ln_out.permute(0, 3, 1, 2)

            intra_out = intra_out + intra_DCN_output

            # Inter-Chunk Processing

            inter_RNN_input = intra_out.permute(0, 3, 2, 1)  ## [B, F, T, C]
            inter_RNN_input_rs = inter_RNN_input.reshape(inter_RNN_input.size()[0] * inter_RNN_input.size()[1],
                                                         inter_RNN_input.size()[2], inter_RNN_input.size()[3])
            inter_RNN_output, _ = self.inter_rnn_trans[i](inter_RNN_input_rs)
            # inter_RNN_input_rs = inter_RNN_input_rs.permute(1, 0, 2)
            # inter_RNN_output = self.inter_rnn_trans[i](inter_RNN_input_rs)
            # inter_RNN_output = inter_RNN_output.permute(1, 0, 2)

            inter_dense_out = self.inter_fc_trans[i](inter_RNN_output)
            inter_ln_input = inter_dense_out.reshape(inter_RNN_input.size()[0], inter_RNN_input.size()[1], inter_RNN_input.size()[2], inter_RNN_input.size()[3])
            inter_ln_input = inter_ln_input.permute(0, 2, 1, 3)
            inter_ln_out = self.inter_ln_trans[i](inter_ln_input)
            inter_out = inter_ln_out.permute(0,3,1,2)

            output = inter_out + intra_out
            # output = inter_out

            inter_DCN_output = self.inter_dil_trans[i](output)

            output_list_all.append(inter_DCN_output)

        del output_i, intra_DCN_output, intra_RNN_input_rs, intra_RNN_input, intra_RNN_output, intra_ln_input, intra_ln_out, inter_RNN_input, inter_RNN_input_rs, inter_RNN_output, inter_dense_out
        del inter_ln_input, inter_ln_out, inter_out, output

        return inter_DCN_output, output_list_all


class DF_ComplexDecoder_Res_new(nn.Module):
    def __init__(self, num_channel=64, df_order=5):
        super(DF_ComplexDecoder_Res_new, self).__init__()

        self.df_order = df_order

        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.Trans_conv_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_channel, out_channels=num_channel, kernel_size=(1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(num_channel, affine=True),
            nn.PReLU(num_channel),
        )

        self.Trans_conv_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_channel, out_channels=2 * self.df_order, kernel_size=(1, 3), stride=(1, 2)),

        )

    def forward(self, x):

        x_dense = self.dense_block(x)
        x1 = self.Trans_conv_1(x_dense)

        x2 = self.Trans_conv_2(x1)

        x2 = x2[:, :, :, :-2]

        return x2

class DfOutputReshapeMF(nn.Module):
    """Coefficients output reshape for multiframe/MultiFrameModule

    Requires input of shape B, C, T, F, 2.
    """

    def __init__(self, df_order: int, df_bins: int):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins

    def forward(self, coefs: Tensor) -> Tensor:
        # [B, T, F, O*2] -> [B, O, T, F, 2]
        new_shape = list(coefs.shape)
        new_shape[-1] = -1
        new_shape.append(2)
        coefs = coefs.view(new_shape)
        coefs = coefs.permute(0, 3, 1, 2, 4)
        return coefs

class DF_DIL_AIA_DCRN_merge_new(nn.Module):
    def __init__(self):
        super().__init__()


        self.df_order = 5
        self.df_bins = 481

        # self.input_ln = nn.LayerNorm(normalized_shape=[201, 2])

        self.dense_encoder = DenseEncoder_Res(in_channel=2, channels=64)

        # self.dual_trans = AIA_Transformer_Res(128, 128, num_layers=4)
        self.DPDCRN_merge = DPDCRNN_Block_merge_new(numUnits=64, width=120)

        self.aham = AHAM(input_channel=64)

        self.complex_decoder = ComplexDecoder_Res_new(num_channel=64)
        self.DF_complex_decoder = DF_ComplexDecoder_Res_new(num_channel=64, df_order=5)

        self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)

        self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)


    def forward(
        self, spec
    ):

        # spec: [B, 2, T, Fc]
        b, c, t, f = spec.shape

        out_1 = self.dense_encoder(spec)

        x_last, mid_out_list = self.DPDCRN_merge(out_1) #BCTF, #BCTFG

        del x_last
        atten_out = self.aham(mid_out_list) #BCTF
        # DPRNN_out1 = self.DPCNN_1(out_1)
        # DPRNN_out2 = self.DPCNN_2(DPRNN_out1)

        complex_out = self.complex_decoder(atten_out)


        df_coefs = self.DF_complex_decoder(atten_out)

        df_coefs = df_coefs.permute(0, 2, 3, 1)

        df_coefs = self.df_out_transform(df_coefs).contiguous()


        # mask_out = convT5_out[:, :, :, :-2]

        mask_real = complex_out[:, 0, :, :]
        mask_imag = complex_out[:, 1, :, :]

        noisy_real = spec[:, 0, :, :]
        noisy_imag = spec[:, 1, :, :]

        ####### simple complex reconstruct

        # enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        # enh_imag = noisy_real * mask_imag + noisy_imag * mask_real

        ####### reconstruct through DCCRN-E
        #### recons_DCCRN-E

        spec_mags = torch.sqrt(noisy_real ** 2 + noisy_imag ** 2 + 1e-8)
        spec_phase = torch.atan2(noisy_imag + 1e-8, noisy_real)

        mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
        real_phase = mask_real / (mask_mags + 1e-8)
        imag_phase = mask_imag / (mask_mags + 1e-8)
        mask_phase = torch.atan2(
            imag_phase + 1e-8,
            real_phase
        )
        # mask_mags = torch.tanh(mask_mags)
        est_mags = mask_mags * spec_mags
        est_phase = spec_phase + mask_phase
        enh_real = est_mags * torch.cos(est_phase)
        enh_imag = est_mags * torch.sin(est_phase)

        enhanced_D = torch.stack([enh_real, enh_imag], 3)

        enhanced_D = enhanced_D.unsqueeze(1)

        DF_spec = self.df_op(enhanced_D.clone(), df_coefs)

        DF_spec = DF_spec.squeeze(1)

        DF_real = DF_spec[:, :, :, 0]
        DF_imag = DF_spec[:, :, :, 1]

        del spec_mags, spec_phase, mask_mags, mask_phase, est_mags, est_phase, real_phase, imag_phase, enh_real, enh_imag, enhanced_D, DF_spec


        return DF_real, DF_imag


class DIL_AIA_DCRN_merge_new(nn.Module):
    def __init__(self):
        super().__init__()

        # self.input_ln = nn.LayerNorm(normalized_shape=[201, 2])

        self.dense_encoder = DenseEncoder_Res(in_channel=2, channels=64)

        # self.dual_trans = AIA_Transformer_Res(128, 128, num_layers=4)
        self.DPDCRN_merge = DPDCRNN_Block_merge(numUnits=64, width=64)

        self.aham = AHAM(input_channel=64)



        # self.DPCNN_2 = DPDCRNN_Block(numUnits=128, width=64)

        self.complex_decoder = ComplexDecoder_Res_new(num_channel=64)


    def forward(
        self, spec
    ):

        # spec: [B, 2, T, Fc]
        b, c, t, f = spec.shape

        out_1 = self.dense_encoder(spec)

        x_last, mid_out_list = self.DPDCRN_merge(out_1) #BCTF, #BCTFG
        atten_out = self.aham(mid_out_list) #BCTF
        # DPRNN_out1 = self.DPCNN_1(out_1)
        # DPRNN_out2 = self.DPCNN_2(DPRNN_out1)

        complex_out = self.complex_decoder(atten_out)
        # mask_out = convT5_out[:, :, :, :-2]

        mask_real = complex_out[:, 0, :, :]
        mask_imag = complex_out[:, 1, :, :]

        noisy_real = spec[:, 0, :, :]
        noisy_imag = spec[:, 1, :, :]

        ####### simple complex reconstruct

        enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        enh_imag = noisy_real * mask_imag + noisy_imag * mask_real

        ####### reconstruct through DCCRN-E
        #### recons_DCCRN-E

        # spec_mags = torch.sqrt(noisy_real ** 2 + noisy_imag ** 2 + 1e-8)
        # spec_phase = torch.atan2(noisy_imag + 1e-8, noisy_real)
        #
        # mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
        # real_phase = mask_real / (mask_mags + 1e-8)
        # imag_phase = mask_imag / (mask_mags + 1e-8)
        # mask_phase = torch.atan2(
        #     imag_phase + 1e-8,
        #     real_phase
        # )
        # # mask_mags = torch.tanh(mask_mags)
        # est_mags = mask_mags * spec_mags
        # est_phase = spec_phase + mask_phase
        # enh_real = est_mags * torch.cos(est_phase)
        # enh_imag = est_mags * torch.sin(est_phase)


        return enh_real, enh_imag




if __name__ == '__main__':
    inputs = torch.randn(1, 2, 10, 257)

    Model = DF_AIA_CRN_oneL()

    # input_test = torch.FloatTensor(1, 2, 100, 257)
    # flops, params = profile(Model, inputs=(input_test,))
    # print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))

    enh_real, enh_imag = Model(inputs)

    params_of_network = 0
    for param in Model.parameters():
        params_of_network += param.numel()

    print(f"\tNetwork: {params_of_network / 1e6} million.")
    #
    print(enh_real.shape)