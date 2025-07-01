import torch
from torch.nn.utils.rnn import pad_sequence
from utils.pre_stft import STFT
from pmsqe_pytorch import init_constants, per_frame_PMSQE
import utils.perceptual_constants as perceptual_constants

# Copy from https://github.com/YangYang/CRNN_mapping_baseline/blob/master/utils/loss_utils.py
def mse_loss():
    def loss_function(est, label, nframes):
        """
        计算真实的MSE
        :param est: 网络输出
        :param label: label
        :param nframes: 每个batch中的真实帧长
        :return:loss
        """
        EPSILON = 1e-7
        with torch.no_grad():
            mask_for_loss_list = []
            # 制作掩码
            for frame_num in nframes:
                mask_for_loss_list.append(torch.ones(frame_num, label.size()[2], dtype=torch.float32))
            # input: list of tensor
            # output: B T F
            mask_for_loss = pad_sequence(mask_for_loss_list, batch_first=True).cuda()
        # 使用掩码计算真实值

        masked_est = est * mask_for_loss
        masked_label = label * mask_for_loss
        loss = ((masked_est - masked_label) ** 2).sum() / (mask_for_loss.sum() + EPSILON)
        return loss
    return loss_function


def SNR_loss():
    def loss_function(est, label):
        """
        计算真实的MSE
        :param est: 网络输出
        :param label: label
        :return:loss
        """
        EPSILON = 1e-7
        snr = torch.mean(label**2, dim=-1, keepdim=True) / ( torch.mean((label-est)**2, dim=-1, keepdim=True) +EPSILON)
        snr_log = -10 * torch.log10(snr)
        snr_loss = torch.mean(snr_log)


        return snr_loss
    return loss_function


def PMSQE_loss():
    def loss_function(est, label):
        """
        计算真实的MSE
        :param est: 网络输出
        :param label: label
        :return:loss
        """
        STFT_block = STFT(
            filter_length=512,
            hop_length=256
        ).to('cuda')
        init_constants(16000, Pow_factor=perceptual_constants.Pow_correc_factor_Hann, apply_SLL_equalization=True,
                       apply_bark_equalization=True, apply_on_degraded=True, apply_degraded_gain_correction=True)
        ref_spectra = STFT_block.transform(label*32768)
        deg_spectra = STFT_block.transform(est*32768)
        ref_real = ref_spectra[:, :, :, 0]
        ref_imag = ref_spectra[:, :, :, 1]
        ref_mag = ref_real ** 2 + ref_imag ** 2 # [batch, T, F]



        deg_real = deg_spectra[:, :, :, 0]
        deg_imag = deg_spectra[:, :, :, 1]
        deg_mag = deg_real ** 2 + deg_imag ** 2  # [batch, T, F]

        #### avoid nan data
        zero = torch.zeros_like(ref_mag)
        ref_mag = torch.where(torch.isnan(ref_mag), zero, ref_mag)
        deg_mag = torch.where(torch.isnan(deg_mag), zero, deg_mag)

        Batch_size = est.shape[0]
        sum_PMSQE = 0
        for i in range(0, Batch_size):
            temp_est = deg_mag[i,:,:]
            temp_label = ref_mag[i,:,:]
            temp_PMSQE = per_frame_PMSQE(temp_label, temp_est, alpha=0.1)
            #### avoid nan data
            zero = torch.zeros_like(temp_PMSQE)
            temp_PMSQE = torch.where(torch.isnan(temp_PMSQE), zero, temp_PMSQE)
            sum_PMSQE += torch.mean(temp_PMSQE)


        return sum_PMSQE / Batch_size
    return loss_function

def SDR_loss():
    def loss_function(label, x, est, thres=20):
        n = x - label
        d = x - est
        diff = label - est
        sSDR = 10.0 * torch.log10( (label**2.0).sum(1) / ((diff**2.0).sum(1) + 1e-8) )
        diff = n - d
        nSDR = 10.0 * torch.log10( (n**2.0).sum(1) / ((diff**2.0).sum(1) + 1e-8) )
        loss = 0.5 * (-thres * torch.tanh( sSDR/thres ) - thres * torch.tanh( nSDR/thres ) )
        return loss.mean()
    return loss_function

def SISDR_loss():
    def loss_function(label, est):
        EPSILON = 1e-8
        a = torch.sum(est * label, dim=-1, keepdim=True) / (torch.sum(label * label, dim=-1, keepdim=True) + EPSILON)
        xa = a * label
        xay = xa - est
        d = torch.sum(xa * xa, dim=-1, keepdim=True) / (torch.sum(xay * xay, dim=-1, keepdim=True) + EPSILON)
        loss = -torch.mean(10 * torch.log10(d))

        return loss
    return loss_function

def phasen_loss():
    def loss_function(est, label, nframes):
        """
        计算真实的MSE
        :param est: 网络输出
        :param label: label
        :param nframes: 每个batch中的真实帧长
        :return:loss
        """
        EPSILON = 1e-7
        with torch.no_grad():
            mask_for_loss_list = []
            # 制作掩码
            for frame_num in nframes:
                mask_for_loss_list.append(torch.ones(frame_num, label.size()[3], dtype=torch.float32))
            # input: list of tensor
            # output: B T F
            mask_for_loss = pad_sequence(mask_for_loss_list, batch_first=True).cuda()
            mask_for_loss = (mask_for_loss.unsqueeze(1)).repeat(1, 2, 1, 1)
        # 使用掩码计算真实值

        masked_est = est * mask_for_loss
        masked_label = label * mask_for_loss
        loss = (((masked_est - masked_label) ** 2).sum() / mask_for_loss.sum() + EPSILON) * 2
        # with torch.no_grad():
        #     est_real =masked_est[:, 0, :, :]
        #     est_imag = masked_est[:, 1, :, :]
        #     est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)+ EPSILON  # [batch, T, F]
        #
        #     label_real =masked_label[:, 0, :, :]
        #     label_imag = masked_label[:, 1, :, :]
        #     label_mag = torch.sqrt(label_real ** 2 + label_imag ** 2)+ EPSILON  # [batch, T, F]
        #
        # ## power compress
        # gth_cprs_mag_spec = label_mag ** 0.3
        # est_cprs_mag_spec = est_mag ** 0.3
        # amp_loss = (((est_cprs_mag_spec - gth_cprs_mag_spec) ** 2).sum() / mask_for_loss.sum() + EPSILON)*2*161
        #
        # compress_coff = ((gth_cprs_mag_spec / (1e-8 + label_mag)).unsqueeze(1)).repeat(1,2,1,1)
        # phase_loss = (((masked_est*compress_coff - masked_label*compress_coff) ** 2).sum() / mask_for_loss.sum() + EPSILON) * 2*161
        # loss = amp_loss*0.5 + phase_loss*0.5
        # # loss = ((masked_est - masked_label) ** 2).sum() / mask_for_loss.sum() + EPSILON
        return loss
    return loss_function