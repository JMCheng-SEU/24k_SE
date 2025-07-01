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

class CGRNN_FB_256(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(CGRNN_FB_256, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=0.3),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.3),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.3),

        )



        self.emb_dim = 512

        self.emb_out_dim = 512

        self.gru_groups = 8

        self.GGRU_1 = GroupedGRULayer_new(input_size=self.emb_dim, hidden_size=self.emb_out_dim,  groups=self.gru_groups)
        # self.GGRU_2 = GroupedGRULayer_new(input_size=self.emb_dim, hidden_size=self.emb_out_dim, groups=self.gru_groups)


        # Decoder for real

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.3),

        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=0.3),

        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=2, kernel_size=(1, 3), stride=(1, 2)),


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


        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)



        mid_in = x3

        mid_GRU_in = mid_in.permute(0, 2, 1, 3)
        mid_GRU_in = mid_GRU_in.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], -1)

        ggru_1_out = self.GGRU_1(mid_GRU_in)

        mid_GRU_out = ggru_1_out.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], 32, -1)
        mid_GRU_out = mid_GRU_out.permute(0, 2, 1, 3)

        mid_out = mid_GRU_out

        # ConvTrans for real
        res = torch.cat((mid_out, x3), 1)
        res1 = self.convT1(res)
        res1 = torch.cat((res1[:, :, :, :32], x2[:, :, :, :32]), 1)
        res2 = self.convT2(res1)
        res2 = torch.cat((res2[:, :, :, :64], x1[:, :, :, :64]), 1)
        res3 = self.convT3(res2)



        mask_real = res3[:, 0, :, :]
        mask_imag = res3[:, 1, :, :]

        noisy_real = x[:, 0, :, :]
        noisy_imag = x[:, 1, :, :]

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

        return enh_real, enh_imag









if __name__ == '__main__':
    inputs = torch.randn(16, 2, 100, 129)

    Model = CGRNN_FB_256()

    enh_real, enh_imag = Model(inputs)

    print(enh_real.shape)

    params_of_network = 0
    for param in Model.parameters():
        params_of_network += param.numel()

    print(f"\tNetwork: {params_of_network / 1e6} million.")

