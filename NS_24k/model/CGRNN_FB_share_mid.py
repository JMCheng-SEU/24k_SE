import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch import Tensor, nn

class GroupedGRULayer_new(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        groups: int,
    ):
        super().__init__()
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        kwargs = {
            "bias": True,
            "batch_first": True,
            "dropout": 0,
            "bidirectional": False,
        }
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        self.out_size = hidden_size

        self.groups = groups
        assert (self.hidden_size % groups) == 0, "Hidden size must be divisible by groups"
        self.layers = nn.ModuleList(
            (nn.GRU(self.input_size, self.hidden_size, **kwargs) for _ in range(groups))
        )

    def forward(self, input: Tensor) -> Tensor:
        # input shape: [B, T, I] if batch_first else [T, B, I], B: batch_size, I: input_size
        # state shape: [G*D, B, H], where G: groups, D: num_directions, H: hidden_size

        outputs = []
        for i, layer in enumerate(self.layers):
            o, _ = layer(
                input[..., i * self.input_size : (i + 1) * self.input_size]
            )
            outputs.append(o)
        output = torch.cat(outputs, dim=-1)
        return output

class S_RNN_Block(nn.Module):

    def __init__(
        self,
        width: int,
    ):
        super().__init__()

        self.width = width


        self.inter_rnn = nn.GRU(input_size=self.width, hidden_size= self.width, num_layers=1, batch_first=True)
        self.inter_fc = nn.Linear(self.width, self.width)




    def forward(self, input: Tensor) -> Tensor:
        # input shape: [B, C, T, F]


        # Inter-Chunk Processing

        inter_RNN_input = input  ## [B, C, T, F]
        inter_RNN_input_rs = inter_RNN_input.reshape(inter_RNN_input.size()[0] * inter_RNN_input.size()[1],
                                                     inter_RNN_input.size()[2], inter_RNN_input.size()[3])
        inter_RNN_output, _ = self.inter_rnn(inter_RNN_input_rs)
        inter_dense_out = self.inter_fc(inter_RNN_output)
        output = inter_dense_out.reshape(inter_RNN_input.size()[0], inter_RNN_input.size()[1], inter_RNN_input.size()[2], inter_RNN_input.size()[3])


        return output


class CGRNN_FB_share_mid(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(CGRNN_FB_share_mid, self).__init__()


        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.3),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.3),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.3),

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.3),

        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.3),

        )

        self.channel_len = 128
        self.groups = 8

        self.width = 64

        self.layers_1 = nn.ModuleList(
            (S_RNN_Block(self.width) for _ in range(self.groups))
        )

        self.layers_2 = nn.ModuleList(
            (S_RNN_Block(self.width) for _ in range(self.groups))
        )


        assert self.channel_len % self.groups == 0
        self.group_len = self.channel_len // self.groups


        # Decoder for real

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.3),

        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.3),

        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.3),

        )
        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.3),

        )

        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=(1, 5), stride=(1, 2)),
        )



    def forward(self, x):
        # conv
        # (B, in_c, T, F)


        # input_ln_in = x.permute(0, 2, 3, 1).contiguous()
        #
        # input_ln_out = self.input_ln(input_ln_in)
        #
        # conv_input = input_ln_out.permute(0, 3, 1, 2).contiguous()
        mid_fea = []

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        mid_fea.append(x5)

        mid_in = x5
        outputs_1 = []
        for i, layer in enumerate(self.layers_1):
            o = layer(
                mid_in[:, i * self.group_len : (i + 1) * self.group_len, :, :]
            )
            outputs_1.append(o)
        mid_out_1 = torch.cat(outputs_1, dim=1)

        outputs_2 = []
        for i, layer in enumerate(self.layers_2):
            o = layer(
                mid_out_1[:, i * self.group_len : (i + 1) * self.group_len, :, :]
            )
            outputs_2.append(o)
        mid_out_2 = torch.cat(outputs_2, dim=1)

        mid_fea.append(mid_out_2)
        # ConvTrans for real
        res = torch.cat((mid_out_2, x5), 1)
        res1 = self.convT1(res)
        res1 = torch.cat((res1[:, :, :, :-2], x4), 1)
        res2 = self.convT2(res1)
        res2 = torch.cat((res2[:, :, :, :-2], x3), 1)
        res3 = self.convT3(res2)
        res3 = torch.cat((res3[:, :, :, :-2], x2), 1)
        res4 = self.convT4(res3)
        res4 = torch.cat((res4[:, :, :, :-1], x1), 1)
        res5 = self.convT5(res4)

        res5 = res5[:, :, :, :-2]

        mask_real = res5[:, 0, :, :]
        mask_imag = res5[:, 1, :, :]

        noisy_real = x[:, 0, :, :]
        noisy_imag = x[:, 1, :, :]

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

        return enh_real, enh_imag, mid_fea




class CGRNN_FB_share_mid_L128(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(CGRNN_FB_share_mid_L128, self).__init__()


        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.3),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.3),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.3),

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.3),

        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.3),

        )

        self.channel_len = 128
        self.groups = 8

        self.width = 128

        self.layers_1 = nn.ModuleList(
            (S_RNN_Block(self.width) for _ in range(self.groups))
        )

        self.layers_2 = nn.ModuleList(
            (S_RNN_Block(self.width) for _ in range(self.groups))
        )


        assert self.channel_len % self.groups == 0
        self.group_len = self.channel_len // self.groups


        # Decoder for real

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.3),

        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.3),

        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.3),

        )
        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.3),

        )

        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=(1, 5), stride=(1, 2)),
        )



    def forward(self, x):
        # conv
        # (B, in_c, T, F)


        # input_ln_in = x.permute(0, 2, 3, 1).contiguous()
        #
        # input_ln_out = self.input_ln(input_ln_in)
        #
        # conv_input = input_ln_out.permute(0, 3, 1, 2).contiguous()


        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        mid_in = x5
        outputs_1 = []
        for i, layer in enumerate(self.layers_1):
            o = layer(
                mid_in[:, i * self.group_len : (i + 1) * self.group_len, :, :]
            )
            outputs_1.append(o)
        mid_out_1 = torch.cat(outputs_1, dim=1)

        outputs_2 = []
        for i, layer in enumerate(self.layers_2):
            o = layer(
                mid_out_1[:, i * self.group_len : (i + 1) * self.group_len, :, :]
            )
            outputs_2.append(o)
        mid_out_2 = torch.cat(outputs_2, dim=1)


        # ConvTrans for real
        res = torch.cat((mid_out_2, x5), 1)
        res1 = self.convT1(res)
        res1 = torch.cat((res1[:, :, :, :-2], x4), 1)
        res2 = self.convT2(res1)
        res2 = torch.cat((res2[:, :, :, :-2], x3), 1)
        res3 = self.convT3(res2)
        res3 = torch.cat((res3[:, :, :, :-2], x2), 1)
        res4 = self.convT4(res3)
        res4 = torch.cat((res4[:, :, :, :-2], x1), 1)
        res5 = self.convT5(res4)

        res5 = res5[:, :, :, :-2]

        mask_real = res5[:, 0, :, :]
        mask_imag = res5[:, 1, :, :]

        noisy_real = x[:, 0, :, :]
        noisy_imag = x[:, 1, :, :]

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

        return enh_real, enh_imag



if __name__ == '__main__':
    inputs = torch.randn(16, 2, 100, 257)

    Model = CGRNN_FB_share_mid_L128()

    enh_real, enh_imag = Model(inputs)

    print(enh_real.shape)