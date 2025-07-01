import torch
import torch.nn as nn
from torch.autograd import Variable
from model.DB_AIAT_model.aia_net import AIA_Transformer, AIA_Transformer_merge, AIA_DCN_Transformer_merge, AHAM, AHAM_ori, AIA_Transformer_new, AIA_Transformer_cau
from thop import profile
from model.DB_AIAT_model.Backup_pesq import numParams
from model.multiframe import DF
class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

class dual_aia_complex_trans(nn.Module):
    def __init__(self):
        super(dual_aia_complex_trans, self).__init__()
        self.en_ri = dense_encoder()
        self.en_mag = dense_encoder_mag()
        self.dual_trans = AIA_Transformer(64, 64, num_layers=4)
        self.aham = AHAM(input_channel=64)
        self.dual_trans_mag = AIA_Transformer(64, 64, num_layers=4)
        self.aham_mag = AHAM(input_channel=64)

        self.de1 = dense_decoder()
        self.de2 = dense_decoder()
        self.de_mag_mask = dense_decoder_masking()



    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape
        x_mag_ori = torch.norm(x,dim=1)
        x_mag = x_mag_ori.unsqueeze(dim = 1)
        x_ri = self.en_ri(x) #BCTF
        x_last , x_outputlist = self.dual_trans(x_ri) #BCTF, #BCTFG
        x_ri = self.aham(x_outputlist) #BCTF
        x_mag_en = self.en_mag(x_mag)
        x_last_mag, x_outputlist_mag = self.dual_trans_mag(x_mag_en)  # BCTF, #BCTFG
        x_mag_en = self.aham_mag(x_outputlist_mag)  # BCTF
        x_mag_mask = self.de_mag_mask(x_mag_en)
        x_mag = x_mag_mask *x_mag
        x_mag = x_mag.squeeze(dim =1)
        x_real = self.de1(x_ri)
        x_imag = self.de2(x_ri)
        x_real = x_real.squeeze(dim = 1)
        x_imag = x_imag.squeeze(dim=1)
        x_com=torch.stack((x_real, x_imag), dim=1)
        pre_mag, pre_phase = torch.norm(x_com, dim=1), torch.atan2(x_com[:,-1,:,:], x_com[:,0,:,:])
        x_mag_out=(x_mag + pre_mag)/2
        x_r_out,x_i_out = x_mag_out * torch.cos(pre_phase), x_mag_out * torch.sin(pre_phase)
        x_com_out = torch.stack((x_r_out,x_i_out),dim=1)

        return x_com_out

class dual_aia_trans_merge_crm_DCN(nn.Module):
    def __init__(self):
        super(dual_aia_trans_merge_crm_DCN, self).__init__()
        self.en_ri = dense_encoder()
        self.en_mag = dense_encoder_mag()
        self.aia_trans_merge = AIA_DCN_Transformer_merge(128, 64, num_layers=4)
        self.aham = AHAM_ori(input_channel=64)
        self.aham_mag = AHAM_ori(input_channel=64)

        # self.simam = simam_module()
        # self.simam_mag = simam_module()

        self.de1 = dense_decoder()
        self.de2 = dense_decoder()
        self.de_mag_mask = dense_decoder_masking()



    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape
        noisy_real = x[:, 0, :, :]
        noisy_imag = x[:, 8, :, :]
        noisy_spec = torch.stack([noisy_real, noisy_imag], 1)
        x_mag_ori, x_phase_ori = torch.norm(noisy_spec, dim=1), torch.atan2(noisy_spec[:, -1, :, :], noisy_spec[:, 0, :, :])
        x_mag = x_mag_ori.unsqueeze(dim = 1)
        # ri/mag components enconde+ aia_transformer_merge
        x_ri = self.en_ri(x) #BCTF
        x_mag_en = self.en_mag(x_mag)
        x_last_mag, x_outputlist_mag, x_last_ri, x_outputlist_ri = self.aia_trans_merge(x_mag_en, x_ri)  # BCTF, #BCTFG

        x_ri = self.aham(x_outputlist_ri) #BCT
        x_mag_en = self.aham_mag(x_outputlist_mag)  # BCTF

        # x_ri = self.simam(x_ri)
        # x_mag_en = self.simam_mag(x_mag_en)
        x_mag_mask = self.de_mag_mask(x_mag_en)
        x_mag_mask = x_mag_mask.squeeze(dim=1)

        # real and imag decode
        x_real = self.de1(x_ri)
        x_imag = self.de2(x_ri)
        x_real = x_real.squeeze(dim = 1)
        x_imag = x_imag.squeeze(dim = 1)
        # magnitude and ri components interaction

        x_mag_out = x_mag_mask * x_mag_ori
        # x_r_out,x_i_out = (x_mag_out * torch.cos(x_phase_ori) + x_real), (x_mag_out * torch.sin(x_phase_ori)+ x_imag)

        ##### recons by DCCRN
        mask_phase = torch.atan2(
            x_imag,
            x_real
        )

        est_phase = x_phase_ori + mask_phase

        x_r_out = x_mag_out * torch.cos(est_phase)
        x_i_out = x_mag_out * torch.sin(est_phase)

        # x_com_out = torch.stack((x_r_out,x_i_out),dim=1)

        return x_r_out, x_i_out, x_real, x_imag, x_mag_out


class slim_dual_aia_trans_merge(nn.Module):
    def __init__(self):
        super(slim_dual_aia_trans_merge, self).__init__()
        self.en_ri = dense_encoder()

        self.aia_trans = AIA_Transformer(128, 64, num_layers=4)
        self.aham = AHAM_ori(input_channel=64)

        self.de = dense_decoder()


    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape
        noisy_real = x[:, 0, :, :]
        noisy_imag = x[:, 1, :, :]

        x_ri = self.en_ri(x) #BCTF

        x_last_ri, x_outputlist_ri = self.aia_trans(x_ri)  # BCTF, #BCTFG

        x_ri = self.aham(x_outputlist_ri) #BCT

        # real and imag decode
        x_mask = self.de(x_ri)

        mask_real = x_mask[:, 0, :, :]
        mask_imag = x_mask[:, 1, :, :]

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


        ##### recons DCCRN_C
        enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        enh_imag = noisy_real * mask_imag + noisy_imag * mask_real

        return enh_real, enh_imag

class dual_aia_trans_merge_crm(nn.Module):
    def __init__(self):
        super(dual_aia_trans_merge_crm, self).__init__()
        self.en_ri = dense_encoder()
        self.en_mag = dense_encoder_mag()
        self.aia_trans_merge = AIA_Transformer_merge(128, 64, num_layers=2)
        self.aham = AHAM_ori(input_channel=64)
        self.aham_mag = AHAM_ori(input_channel=64)

        # self.simam = simam_module()
        # self.simam_mag = simam_module()

        self.de1 = dense_decoder()
        self.de2 = dense_decoder()
        self.de_mag_mask = dense_decoder_masking()



    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape
        noisy_real = x[:, 0, :, :]
        noisy_imag = x[:, 1, :, :]
        noisy_spec = torch.stack([noisy_real, noisy_imag], 1)
        x_mag_ori, x_phase_ori = torch.norm(noisy_spec, dim=1), torch.atan2(noisy_spec[:, -1, :, :], noisy_spec[:, 0, :, :])
        x_mag = x_mag_ori.unsqueeze(dim = 1)
        # ri/mag components enconde+ aia_transformer_merge
        x_ri = self.en_ri(x) #BCTF
        x_mag_en = self.en_mag(x_mag)
        x_last_mag, x_outputlist_mag, x_last_ri, x_outputlist_ri = self.aia_trans_merge(x_mag_en, x_ri)  # BCTF, #BCTFG

        x_ri = self.aham(x_outputlist_ri) #BCT
        x_mag_en = self.aham_mag(x_outputlist_mag)  # BCTF

        # x_ri = self.simam(x_ri)
        # x_mag_en = self.simam_mag(x_mag_en)
        x_mag_mask = self.de_mag_mask(x_mag_en)
        x_mag_mask = x_mag_mask.squeeze(dim=1)

        # real and imag decode
        x_real = self.de1(x_ri)
        x_imag = self.de2(x_ri)
        x_real = x_real.squeeze(dim = 1)
        x_imag = x_imag.squeeze(dim = 1)
        # magnitude and ri components interaction

        x_mag_out = x_mag_mask * x_mag_ori
        # x_r_out,x_i_out = (x_mag_out * torch.cos(x_phase_ori) + x_real), (x_mag_out * torch.sin(x_phase_ori)+ x_imag)

        ##### recons by DCCRN
        mask_phase = torch.atan2(
            x_imag,
            x_real
        )

        est_phase = x_phase_ori + mask_phase

        x_r_out = x_mag_out * torch.cos(est_phase)
        x_i_out = x_mag_out * torch.sin(est_phase)

        # x_com_out = torch.stack((x_r_out,x_i_out),dim=1)

        return x_r_out, x_i_out


class aia_complex_trans_ri(nn.Module):
    def __init__(self):
        super(aia_complex_trans_ri, self).__init__()
        self.en_ri = dense_encoder()

        self.dual_trans = AIA_Transformer(96, 96, num_layers=4)
        self.aham = AHAM(input_channel=96)


        self.de1 = dense_decoder()
        self.de2 = dense_decoder()



    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape


        noisy_real = x[:, 0, :, :]
        noisy_imag = x[:, 1, :, :]

        # ri components enconde+ aia_transformer
        x_ri = self.en_ri(x) #BCTF
        x_last , x_outputlist = self.dual_trans(x_ri) #BCTF, #BCTFG
        x_ri = self.aham(x_outputlist) #BCTF


        # real and imag decode
        x_real = self.de1(x_ri)
        x_imag = self.de2(x_ri)
        x_real = x_real.squeeze(dim = 1)
        x_imag = x_imag.squeeze(dim = 1)

        # x_com=torch.stack((x_real, x_imag), dim=1)

        enh_real = noisy_real * x_real - noisy_imag * x_imag
        enh_imag = noisy_real * x_imag + noisy_imag * x_real

        return enh_real, enh_imag

class aia_complex_trans_ri_new(nn.Module):
    def __init__(self):
        super(aia_complex_trans_ri_new, self).__init__()
        self.en_ri = dense_encoder()

        self.dual_trans = AIA_Transformer_new(96, 96, num_layers=4)
        self.aham = AHAM(input_channel=96)


        self.de1 = dense_decoder()
        self.de2 = dense_decoder()



    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape


        noisy_real = x[:, 0, :, :]
        noisy_imag = x[:, 1, :, :]

        # ri components enconde+ aia_transformer
        x_ri = self.en_ri(x) #BCTF
        x_last , x_outputlist = self.dual_trans(x_ri) #BCTF, #BCTFG
        x_ri = self.aham(x_outputlist) #BCTF


        # real and imag decode
        x_real = self.de1(x_ri)
        x_imag = self.de2(x_ri)
        x_real = x_real.squeeze(dim = 1)
        x_imag = x_imag.squeeze(dim = 1)

        # x_com=torch.stack((x_real, x_imag), dim=1)

        enh_real = noisy_real * x_real - noisy_imag * x_imag
        enh_imag = noisy_real * x_imag + noisy_imag * x_real

        return enh_real, enh_imag


class dense_encoder(nn.Module):
    def __init__(self, width = 96):
        super(dense_encoder, self).__init__()
        self.in_channels = 2
        self.out_channels = 1
        self.width = width
        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1))  # [b, 64, nframes, 512]
        self.inp_norm = nn.LayerNorm(481)
        self.inp_prelu = nn.PReLU(self.width)
        self.enc_dense1 = DenseBlock(481, 4, self.width) # [b, 64, nframes, 512]
        self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))  # [b, 64, nframes, 256]
        self.enc_norm1 = nn.LayerNorm(241)
        self.enc_prelu1 = nn.PReLU(self.width)

        self.enc_conv2 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))  # [b, 64, nframes, 256]
        self.enc_norm2 = nn.LayerNorm(120)
        self.enc_prelu2 = nn.PReLU(self.width)


    def forward(self, x):
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))  # [b, 64, T, F]
        out = self.enc_dense1(out)   # [b, 64, T, F]
        x = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out)))  # [b, 64, T, F // 2]

        x2 = self.enc_prelu2(self.enc_norm2(self.enc_conv2(x)))  # [b, 64, T, F // 4]
        return x2

class dense_encoder_mag(nn.Module):
    def __init__(self, width = 96):
        super(dense_encoder_mag, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.width = width
        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1))  # [b, 64, nframes, 512]
        self.inp_norm = nn.LayerNorm(481)
        self.inp_prelu = nn.PReLU(self.width)
        self.enc_dense1 = DenseBlock(481, 4, self.width) # [b, 64, nframes, 512]
        self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2), padding= (0,1))  # [b, 64, nframes, 256]
        self.enc_norm1 = nn.LayerNorm(241)
        self.enc_prelu1 = nn.PReLU(self.width)


        self.enc_conv2 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))  # [b, 64, nframes, 256]
        self.enc_norm2 = nn.LayerNorm(120)
        self.enc_prelu2 = nn.PReLU(self.width)

    def forward(self, x):
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))  # [b, 64, T, F]
        out = self.enc_dense1(out)   # [b, 64, T, F]
        x = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out)))  # [b, 64, T, F // 2]

        x2 = self.enc_prelu2(self.enc_norm2(self.enc_conv2(x)))  # [b, 64, T, F // 4]
        return x2


class dense_decoder(nn.Module):
    def __init__(self, width = 96):
        super(dense_decoder, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.)
        self.width = width
        self.dec_dense1 = DenseBlock(120, 4, self.width)


        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm1 = nn.LayerNorm(240)
        self.dec_prelu1 = nn.PReLU(self.width)

        self.dec_conv2 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm2 = nn.LayerNorm(481)
        self.dec_prelu2 = nn.PReLU(self.width)
        #
        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 5), padding=(0, 2))

    def forward(self, x):
        out = self.dec_dense1(x)
        out = self.dec_prelu1(self.dec_norm1(self.dec_conv1(self.pad(out))))

        # out = self.dec_conv2(self.pad(out))

        out = self.dec_prelu2(self.dec_norm2(self.pad1(self.dec_conv2(self.pad(out)))))
        #
        out = self.out_conv(out)
        # out.squeeze(dim=1)
        return out

class dense_decoder_masking(nn.Module):
    def __init__(self, width = 96):
        super(dense_decoder_masking, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.)
        self.width =width
        self.dec_dense1 = DenseBlock(120, 4, self.width)

        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm1 = nn.LayerNorm(240)
        self.dec_prelu1 = nn.PReLU(self.width)

        self.dec_conv2 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm2 = nn.LayerNorm(481)
        self.dec_prelu2 = nn.PReLU(self.width)

        self.mask1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size= (1,1)),
        )
        self.mask2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size= (1,1)),
            nn.Tanh()
        )
        self.maskconv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,1))

        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 1))

    def forward(self, x):
        out = self.dec_dense1(x)
        out = self.dec_prelu1(self.dec_norm1(self.dec_conv1(self.pad(out))))

        out = self.dec_prelu2(self.dec_norm2(self.pad1(self.dec_conv2(self.pad(out)))))

        out = self.out_conv(out)
        out.squeeze(dim=1)
        out = self.mask1(out) * self.mask2(out)
        out = self.maskconv(out)  # mask
        return out




class SPConvTranspose2d(nn.Module): #sub-pixel convolution
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseBlock(nn.Module): #dilated dense block
    def __init__(self, input_size, depth=5, in_channels=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.LayerNorm(input_size))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out

class DF_dense_decoder(nn.Module):
    def __init__(self, width=96):
        super(DF_dense_decoder, self).__init__()
        self.in_channels = 1
        self.out_channels = 10
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.)
        self.width = width
        self.dec_dense1 = DenseBlock(120, 4, self.width)


        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm1 = nn.LayerNorm(240)
        self.dec_prelu1 = nn.PReLU(self.width)

        self.dec_conv2 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm2 = nn.LayerNorm(481)
        self.dec_prelu2 = nn.PReLU(self.width)
        #
        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 5), padding=(0, 2))

    def forward(self, x):
        out = self.dec_dense1(x)
        out = self.dec_prelu1(self.dec_norm1(self.dec_conv1(self.pad(out))))

        # out = self.dec_conv2(self.pad(out))

        out = self.dec_prelu2(self.dec_norm2(self.pad1(self.dec_conv2(self.pad(out)))))
        #
        out = self.out_conv(out)
        # out.squeeze(dim=1)
        return out

class DfOutputReshapeMF(nn.Module):
    """Coefficients output reshape for multiframe/MultiFrameModule

    Requires input of shape B, C, T, F, 2.
    """

    def __init__(self, df_order: int, df_bins: int):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins

    def forward(self, coefs):
        # [B, T, F, O*2] -> [B, O, T, F, 2]
        new_shape = list(coefs.shape)
        new_shape[-1] = -1
        new_shape.append(2)
        coefs = coefs.view(new_shape)
        coefs = coefs.permute(0, 3, 1, 2, 4)
        return coefs

class DF_aia_complex_trans_ri(nn.Module):
    def __init__(self):
        super(DF_aia_complex_trans_ri, self).__init__()
        self.en_ri = dense_encoder()

        self.dual_trans = AIA_Transformer(96, 96, num_layers=4)
        self.aham = AHAM(input_channel=96)

        self.de1 = dense_decoder()
        self.de2 = dense_decoder()

        self.DF_de = DF_dense_decoder()

        self.df_order = 5
        self.df_bins = 481

        self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)

        self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)



    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape


        noisy_real = x[:, 0, :, :]
        noisy_imag = x[:, 1, :, :]

        # ri components enconde+ aia_transformer
        x_ri = self.en_ri(x) #BCTF
        x_last , x_outputlist = self.dual_trans(x_ri) #BCTF, #BCTFG
        x_ri = self.aham(x_outputlist) #BCTF


        # real and imag decode
        x_real = self.de1(x_ri)
        x_imag = self.de2(x_ri)
        x_real = x_real.squeeze(dim = 1)
        x_imag = x_imag.squeeze(dim = 1)


        df_coefs = self.DF_de(x_ri)

        df_coefs = df_coefs.permute(0, 2, 3, 1)

        df_coefs = self.df_out_transform(df_coefs).contiguous()


        # x_com=torch.stack((x_real, x_imag), dim=1)

        enh_real = noisy_real * x_real - noisy_imag * x_imag
        enh_imag = noisy_real * x_imag + noisy_imag * x_real


        enhanced_D = torch.stack([enh_real, enh_imag], 3)

        enhanced_D = enhanced_D.unsqueeze(1)

        DF_spec = self.df_op(enhanced_D, df_coefs)

        DF_spec = DF_spec.squeeze(1)

        DF_real = DF_spec[:, :, :, 0]
        DF_imag = DF_spec[:, :, :, 1]

        return DF_real, DF_imag


class DF_aia_complex_trans_ri_causal(nn.Module):
    def __init__(self):
        super(DF_aia_complex_trans_ri_causal, self).__init__()
        self.en_ri = dense_encoder()

        self.dual_trans = AIA_Transformer_cau(96, 96, num_layers=4)
        self.aham = AHAM(input_channel=96)

        self.de1 = dense_decoder()
        self.de2 = dense_decoder()

        self.DF_de = DF_dense_decoder()

        self.df_order = 5
        self.df_bins = 481

        self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)

        self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)



    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape


        noisy_real = x[:, 0, :, :]
        noisy_imag = x[:, 1, :, :]

        # ri components enconde+ aia_transformer
        x_ri = self.en_ri(x) #BCTF
        x_last , x_outputlist = self.dual_trans(x_ri) #BCTF, #BCTFG
        x_ri = self.aham(x_outputlist) #BCTF


        # real and imag decode
        x_real = self.de1(x_ri)
        x_imag = self.de2(x_ri)
        x_real = x_real.squeeze(dim = 1)
        x_imag = x_imag.squeeze(dim = 1)


        df_coefs = self.DF_de(x_ri)

        df_coefs = df_coefs.permute(0, 2, 3, 1)

        df_coefs = self.df_out_transform(df_coefs).contiguous()


        # x_com=torch.stack((x_real, x_imag), dim=1)

        enh_real = noisy_real * x_real - noisy_imag * x_imag
        enh_imag = noisy_real * x_imag + noisy_imag * x_real


        enhanced_D = torch.stack([enh_real, enh_imag], 3)

        enhanced_D = enhanced_D.unsqueeze(1)

        DF_spec = self.df_op(enhanced_D, df_coefs)

        DF_spec = DF_spec.squeeze(1)

        DF_real = DF_spec[:, :, :, 0]
        DF_imag = DF_spec[:, :, :, 1]

        return DF_real, DF_imag

if __name__ == '__main__':
    model = aia_complex_trans_ri_new()   #### slim_dual_aia_trans_merge dual_aia_trans_merge_crm
    model.eval()
    # x = torch.FloatTensor(4, 2, 10, 481)
    #
    # real, imag = model(x)
    # print(str(real.shape))
    input_test = torch.FloatTensor(1, 2, 2, 481)
    flops, params = profile(model, inputs=(input_test,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))


    #
    # params_of_network = 0
    # for param in model.parameters():
    #     params_of_network += param.numel()
    #
    # print(f"\tNetwork: {params_of_network / 1e6} million.")
    #output = model(x)
