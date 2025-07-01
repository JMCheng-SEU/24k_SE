# coding: utf-8
# Author：WangTianRui
# Date ：2020/11/3 16:49
# from base.BaseModel import *
import torch.nn as nn
import torch
from utils.conv_stft import *
from utils.complexnn import *

class NavieComplexLSTM_new(nn.Module):
    def __init__(self, input_size, hidden_size, projection_dim=None, bidirectional=False, batch_first=False):
        super(NavieComplexLSTM_new, self).__init__()

        self.input_dim = input_size // 2
        self.rnn_units = hidden_size // 2
        self.real_lstm = nn.LSTM(self.input_dim, self.rnn_units, num_layers=1, bidirectional=bidirectional,
                                 batch_first=False)
        self.imag_lstm = nn.LSTM(self.input_dim, self.rnn_units, num_layers=1, bidirectional=bidirectional,
                                 batch_first=False)
        # if bidirectional:
        #     bidirectional = 2
        # else:
        #     bidirectional = 1
        # if projection_dim is not None:
        #     self.projection_dim = projection_dim // 2
        #     self.r_trans = nn.Linear(self.rnn_units * bidirectional, self.projection_dim)
        #     self.i_trans = nn.Linear(self.rnn_units * bidirectional, self.projection_dim)
        # else:
        #     self.projection_dim = None

    def forward(self, inputs):
        if isinstance(inputs, list):
            real, imag = inputs
        elif isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs, -1)
        r2r_out = self.real_lstm(real)[0]
        r2i_out = self.imag_lstm(real)[0]
        i2r_out = self.real_lstm(imag)[0]
        i2i_out = self.imag_lstm(imag)[0]
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out
        # if self.projection_dim is not None:
        #     real_out = self.r_trans(real_out)
        #     imag_out = self.i_trans(imag_out)
        # # print(real_out.shape,imag_out.shape)
        return [real_out, imag_out]

    def flatten_parameters(self):
        self.imag_lstm.flatten_parameters()
        self.real_lstm.flatten_parameters()

class DCCRN(nn.Module):
    def __init__(self,
                 rnn_layer=2, rnn_hidden=256,
                 win_len=400, hop_len=100, fft_len=512, win_type='hanning',
                 use_clstm=True, use_cbn=False, masking_mode='E',
                 kernel_size=5, kernel_num=(8, 16, 32, 64, 64, 64)
                 ):
        super(DCCRN, self).__init__()
        self.rnn_layer = rnn_layer
        self.rnn_hidden = rnn_hidden

        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.win_type = win_type

        self.use_clstm = use_clstm
        self.use_cbn = use_cbn
        self.masking_mode = masking_mode

        self.kernel_size = kernel_size
        self.kernel_num = (2,) + kernel_num

        self.stft = ConvSTFT(self.win_len, self.hop_len, self.fft_len, self.win_type, 'complex', fix=True)
        self.istft = ConviSTFT(self.win_len, self.hop_len, self.fft_len, self.win_type, 'complex', fix=True)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                        self.kernel_num[idx + 1]),
                    nn.PReLU()
                )
            )
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

        if self.use_clstm:
            # rnns = []
            # for idx in range(rnn_layer):
            #     rnns.append(
            #         NavieComplexLSTM(
            #             input_size=hidden_dim * self.kernel_num[-1] if idx == 0 else self.rnn_hidden,
            #             hidden_size=self.rnn_hidden,
            #             batch_first=False,
            #             projection_dim=hidden_dim * self.kernel_num[-1] if idx == rnn_layer - 1 else None
            #         )
            #     )
            #     self.enhance = nn.Sequential(*rnns)

            self.RNN_1 = NavieComplexLSTM_new(
                input_size=256,
                hidden_size=64,
                batch_first=False
            )

            self.RNN_2 = NavieComplexLSTM_new(
                input_size=64,
                hidden_size=64,
                batch_first=False
            )

            # self.real_stu_transform = nn.Linear(32, 128)
            # self.imag_stu_transform = nn.Linear(32, 128)

            self.r_trans = nn.Linear(32, 128)
            self.i_trans = nn.Linear(32, 128)
        else:
            self.enhance = nn.LSTM(
                input_size=hidden_dim * self.kernel_num[-1],
                hidden_size=self.rnn_hidden,
                num_layers=2,
                dropout=0.0,
                batch_first=False
            )
            self.transform = nn.Linear(self.rnn_hidden, hidden_dim * self.kernel_num[-1])
        for idx in range(len(self.kernel_num) - 1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        ),
                        nn.BatchNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                            self.kernel_num[idx - 1]),
                        nn.PReLU()
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        )
                    )
                )
        if isinstance(self.RNN_1, nn.LSTM):
            self.RNN_1.flatten_parameters()

        if isinstance(self.RNN_2, nn.LSTM):
            self.RNN_2.flatten_parameters()

    def forward(self, real, imag):
        # stft = self.stft(x)
        # # print("stft:", stft.size())
        # real = stft[:, :self.fft_len // 2 + 1]
        # imag = stft[:, self.fft_len // 2 + 1:]
        # print("real imag:", real.size(), imag.size())
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        spec_phase = torch.atan2(imag, real)
        spec_complex = torch.stack([real, imag], dim=1)[:, :, 1:]  # B,2,256
        # print("spec", spec_mags.size(), spec_phase.size(), spec_complex.size())

        out = spec_complex
        encoder_out = []
        for idx, encoder in enumerate(self.encoder):
            out = encoder(out)
            # print("encoder out:", out.size())
            encoder_out.append(out)
        B, C, D, T = out.size()
        out = out.permute(3, 0, 1, 2)
        if self.use_clstm:
            r_rnn_in = out[:, :, :C // 2]
            i_rnn_in = out[:, :, C // 2:]
            r_rnn_in = torch.reshape(r_rnn_in, [T, B, C // 2 * D])
            i_rnn_in = torch.reshape(i_rnn_in, [T, B, C // 2 * D])

            # r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])
            r_rnn_in_1, i_rnn_in_1 = self.RNN_1([r_rnn_in, i_rnn_in])

            # r_rnn_in_1_fea = self.real_stu_transform(r_rnn_in_1)
            r_rnn_in_1_fea = r_rnn_in_1.permute(1, 0, 2)
            # r_rnn_in_1_fea = r_rnn_in_1_fea.contiguous()

            # i_rnn_in_1_fea = self.imag_stu_transform(i_rnn_in_1)
            i_rnn_in_1_fea = i_rnn_in_1.permute(1, 0, 2)
            # i_rnn_in_1_fea = i_rnn_in_1_fea.contiguous()

            r_rnn_in_2, i_rnn_in_2 = self.RNN_2([r_rnn_in_1, i_rnn_in_1])

            r_rnn_out = self.r_trans(r_rnn_in_2)
            i_rnn_out = self.i_trans(i_rnn_in_2)

            r_rnn_out = torch.reshape(r_rnn_out, [T, B, C // 2, D])
            i_rnn_out = torch.reshape(i_rnn_out, [T, B, C // 2, D])
            out = torch.cat([r_rnn_out, i_rnn_out], 2)
        else:
            out = torch.reshape(out, [T, B, C * D])
            out, _ = self.enhance(out)
            out = self.transform(out)
            out = torch.reshape(out, [T, B, C, D])
        out = out.permute(1, 2, 3, 0)
        for idx in range(len(self.decoder)):
            out = complex_cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)
            out = out[..., 1:]
        mask_real = out[:, 0]
        mask_imag = out[:, 1]
        mask_real = F.pad(mask_real, [0, 0, 1, 0])
        mask_imag = F.pad(mask_imag, [0, 0, 1, 0])
        if self.masking_mode == 'E':
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
            real = est_mags * torch.cos(est_phase)
            imag = est_mags * torch.sin(est_phase)
        elif self.masking_mode == 'C':
            real = real * mask_real - imag * mask_imag
            imag = real * mask_imag + imag * mask_real
        elif self.masking_mode == 'R':
            real = real * mask_real
            imag = imag * mask_imag

        # out_spec = torch.cat([real, imag], 1)
        # out_wav = self.istft(out_spec)
        # out_wav = torch.squeeze(out_wav, 1)
        # out_wav = out_wav.clamp_(-1, 1)
        return real, imag, r_rnn_in_1_fea, i_rnn_in_1_fea, encoder_out
        # return out_wav, r_rnn_in_1_fea, i_rnn_in_1_fea