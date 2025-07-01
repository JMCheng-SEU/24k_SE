import librosa
import torch
import torch.nn as nn
from pase.models.frontend import wf_builder
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from trainer.base_trainer import BaseTrainer
import matplotlib.pyplot as plt
from trainer.torch_stoi import NegSTOILoss
import soundfile as sf
import os
from test_for_PSD import *

plt.switch_backend("agg")
plt.rcParams.update({'figure.max_open_warning': 0})
import numpy as np
import librosa.display
from tqdm import tqdm

from utils.utils import compute_STOI, compute_PESQ, z_score, reverse_z_score


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def l2_norm(s1, s2):
    # norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    # norm = torch.norm(s1*s2, 1, keepdim=True)

    norm = np.sum(s1 * s2, axis=-1, keepdims=True)
    return norm

def mSDRLoss(orig, est):
    # Modified SDR loss, <x, x`> / (||x|| * ||x`||) : L2 Norm.
    # Original SDR Loss: <x, x`>**2 / <x`, x`> (== ||x`||**2)
    #  > Maximize Correlation while producing minimum energy output.
    bsum = lambda x: torch.sum(x, dim=1)
    correlation = bsum(orig * est)
    energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
    return -(correlation / (energies + 1e-8))

def SDRLoss(orig, est):
    # Modified SDR loss, <x, x`> / (||x|| * ||x`||) : L2 Norm.
    # Original SDR Loss: <x, x`>**2 / <x`, x`> (== ||x`||**2)
    #  > Maximize Correlation while producing minimum energy output.
    bsum = lambda x: torch.sum(x, dim=1)
    correlation = bsum(orig * est) ** 2
    energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
    return -(correlation / (energies + 1e-8))


def wSDRLoss(mixed, clean, clean_est, eps=1e-8):
    # Used on signal level(time-domain). Backprop-able istft should be used.
    # Batched audio inputs shape (N x T) required.
     # Batch preserving sum for convenience.
    noise = mixed - clean
    noise_est = mixed - clean_est
    bsum = lambda x: torch.sum(x, dim=1)
    a = bsum(clean ** 2) / (bsum(clean ** 2) + bsum(noise ** 2) + eps)
    wSDR = a * mSDRLoss(clean, clean_est) + (1 - a) * mSDRLoss(noise, noise_est)
    return torch.mean(wSDR)

def si_snr(s1, s2, eps=1e-8):
    #s1 = remove_dc(s1)
    #s2 = remove_dc(s2)
    if len(s2) > len(s1):
        s2 = s2[:len(s1)]
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*np.log10((target_norm)/(noise_norm+eps)+eps)
    return np.mean(snr)

class Trainer(BaseTrainer):
    def __init__(self,
                 config,
                 resume,
                 model,
                 optimizer,
                 loss_function,
                 scheduler,
                 train_dataloader,
                 validation_dataloader):
        super(Trainer, self).__init__(config, resume, model, optimizer, loss_function)
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0
        loss_stoi_total = 0.0
        loss_pmsqe_total = 0.0
        # loss_wsdr_total = 0.0
        # pase = wf_builder('cfg/frontend/PASE+.cfg').eval()
        # pase.load_pretrained('./FE_e199.ckpt', load_last=True, verbose=True)
        # pase.to('cuda')
        step = 0
        for mixture, clean in tqdm(self.train_dataloader, desc="Training"):
            # self.model.train()
            self.optimizer.zero_grad()

            ################# compute mixture PSD
            mixture = mixture.to('cpu')
            mixture_np = np.array(mixture)
            mix_mag_torch = torch.zeros([mixture_np.shape[0], 256, 256], dtype=torch.float32)
            mix_phase_torch = torch.zeros([mixture_np.shape[0], 256, 256], dtype=torch.float32)
            mix_norm_torch = torch.zeros([mixture_np.shape[0], 2], dtype=torch.float32)
            for batch_index in range(mixture_np.shape[0]):
                temp_mixture_speech = mixture_np[batch_index, :]
                psd_temp = psd(temp_mixture_speech, preprocess=False)
                MagdB_temp = torch.from_numpy(psd_temp['MagdB'])
                Norm_temp = torch.from_numpy(np.array(psd_temp['Norm']))
                Phase_temp = torch.from_numpy(psd_temp['Phase'])

                mix_mag_torch[batch_index, :, :] = MagdB_temp
                mix_phase_torch[batch_index, :, :] = Phase_temp
                mix_norm_torch[batch_index, :] = Norm_temp


            ################## compute clean PSD
            clean = clean.to('cpu')
            clean_np = np.array(clean)
            clean_mag_torch = torch.zeros([clean_np.shape[0], 256, 256], dtype=torch.float32)
            clean_phase_torch = torch.zeros([clean_np.shape[0], 256, 256], dtype=torch.float32)
            clean_norm_torch = torch.zeros([clean_np.shape[0], 2], dtype=torch.float32)
            for batch_index in range(clean_np.shape[0]):
                temp_clean_speech = clean_np[batch_index, :]
                psd_temp = psd(temp_clean_speech, preprocess=False)
                MagdB_temp = torch.from_numpy(psd_temp['MagdB'])
                Norm_temp = torch.from_numpy(np.array(psd_temp['Norm']))
                Phase_temp = torch.from_numpy(psd_temp['Phase'])

                clean_mag_torch[batch_index, :, :] = MagdB_temp
                clean_phase_torch[batch_index, :, :] = Phase_temp
                clean_norm_torch[batch_index, :] = Norm_temp




            mix_mag_torch = mix_mag_torch.to(self.device)
            clean_mag_torch = clean_mag_torch.to(self.device)




            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            noise = mixture - clean


            # Mixture mag and Clean mag
            mixture_D  = self.stft.transform(mixture)
            mixture_real = mixture_D[:, :, :, 0]
            mixture_imag = mixture_D[:, :, :, 1]
            mixture_mag = torch.sqrt(mixture_real ** 2 + mixture_imag** 2 + 1e-8) # [batch, T, F]
            mix_spec = mixture_D.permute(0, 3, 1, 2)

            clean_D  = self.stft.transform(clean)
            clean_real = clean_D[:, :, :, 0]
            clean_imag = clean_D[:, :, :, 1]

            # Mr_c = (mixture_real * clean_real + mixture_imag * clean_imag) / ((mixture_real ** 2 + mixture_imag ** 2) +1e-8) +1e-8
            # Mi_c = (mixture_real * clean_imag - mixture_imag * clean_real) / ((mixture_real ** 2 + mixture_imag ** 2) +1e-8) +1e-8
            # # temp_mic_r = Mr_c.cpu().numpy()
            # # temp_mic = Mi_c.cpu().numpy()
            # zero = torch.zeros_like(Mr_c) + 900
            # Mr_c = torch.where(Mr_c < -850, zero, Mr_c)
            # Mi_c = torch.where(Mi_c < -850, zero, Mi_c)
            #
            # cIRMTar_r = 1. * (torch.ones_like(Mr_c) - torch.exp(-0.1 * Mr_c)) / (torch.ones_like(Mr_c) + torch.exp(-0.1 * Mr_c))
            # cIRMTar_i = 1. * (torch.ones_like(Mi_c) - torch.exp(-0.1 * Mi_c)) / (torch.ones_like(Mi_c) + torch.exp(-0.1 * Mi_c))

            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-8)

            noise_D  = self.stft.transform(noise)
            noise_real = noise_D[:, :, :, 0]
            noise_imag = noise_D[:, :, :, 1]
            noise_mag = torch.sqrt(noise_real ** 2 + noise_imag** 2 + 1e-8) # [batch, T, F]

            # irm_mask = torch.sqrt(clean_mag ** 2 / (clean_mag + noise_mag) ** 2)

            irm_mask = torch.sqrt(clean_mag ** 2 / (mixture_mag) ** 2)

            # LPS_fea = torch.log10(mixture_mag ** 2)
            # LPS_fea = torch.log(mixture_mag ** 2)
            # clean_LPS = torch.log(clean_mag ** 2)
            # clean_LPS_compress = clean_LPS[:,:,:256]
            # clean_LPS_compress = (1. - torch.exp(-2 * clean_LPS_compress)) / ((1. + torch.exp(-2 * clean_LPS_compress)) + 1e-8)
            #
            # ##### remove the last point of LPS fea
            # LPS_in = LPS_fea[:,:,:256]
            #
            # LPS_in_compress = (1. - torch.exp(-2 * LPS_in)) / ((1. + torch.exp(-2 * LPS_in)) + 1e-8)




            est_magdB = self.model(mix_mag_torch)

            # ######### recons for PSD
            # speech_maximum_len = (est_magdB.shape[2] -1) * 128
            # batch_enhanced_speeches = np.zeros((est_magdB.shape[0], speech_maximum_len), dtype=np.float32)
            # batch_clean_speeches = np.zeros((est_magdB.shape[0], speech_maximum_len), dtype=np.float32)
            # for batch_index in range(est_magdB.shape[0]):
            #
            #     temp_mag_dB = est_magdB[batch_index, :, :].detach().cpu().numpy()
            #     temp_phase = mix_phase_torch[batch_index, :, :].detach().cpu().numpy()
            #     temp_norm = mix_norm_torch[batch_index, :].detach().cpu().numpy()
            #
            #     enhanced_mag = np.interp(temp_mag_dB, [-1, 1], temp_norm)
            #     temp = np.zeros((257, enhanced_mag.shape[1])) + 1j * np.zeros((257, enhanced_mag.shape[1]))
            #     temp[:-1, :] = 10 ** (enhanced_mag / 20) * (np.cos(temp_phase) + np.sin(temp_phase) * 1j)
            #     enhanced_audio = istft(temp)
            #     enhanced_audio = 0.98 * enhanced_audio / np.max(np.abs(enhanced_audio))
            #
            #     batch_enhanced_speeches[batch_index, :] = enhanced_audio
            #
            # for batch_index in range(est_magdB.shape[0]):
            #     temp_mag_dB = clean_mag_torch[batch_index, :, :].detach().cpu().numpy()
            #     temp_phase = clean_phase_torch[batch_index, :, :].detach().cpu().numpy()
            #     temp_norm = clean_norm_torch[batch_index, :].detach().cpu().numpy()
            #
            #     clean_mag = np.interp(temp_mag_dB, [-1, 1], temp_norm)
            #     temp = np.zeros((257, clean_mag.shape[1])) + 1j * np.zeros((257, clean_mag.shape[1]))
            #     temp[:-1, :] = 10 ** (clean_mag / 20) * (np.cos(temp_phase) + np.sin(temp_phase) * 1j)
            #     clean_audio = istft(temp)
            #     clean_audio = 0.98 * clean_audio / np.max(np.abs(clean_audio))
            #
            #     batch_clean_speeches[batch_index, :] = clean_audio
            #
            # batch_enhanced_speeches = torch.from_numpy(batch_enhanced_speeches).to(self.device)
            #
            # batch_clean_speeches = torch.from_numpy(batch_clean_speeches).to(self.device)
            #
            #
            # enhanced = batch_enhanced_speeches
            # cleaned = batch_clean_speeches




            # enh_LPS = -0.5 * torch.log((1 - est_LPS) / ((1 + est_LPS + 1e-8)) + 1e-8)
            #
            # enh_LPS = torch.cat([enh_LPS, LPS_fea[:,:,256].unsqueeze(2)], dim=2)


            # enhanced_mag = torch.sqrt(torch.exp(enh_LPS) + 1e-8)
            #
            # enhanced_real = enhanced_mag * mixture_real / mixture_mag
            # enhanced_imag = enhanced_mag * mixture_imag / mixture_mag
            #
            # enhanced_D = torch.stack([enhanced_real, enhanced_imag], 3)
            # enhanced = self.stft.inverse(enhanced_D)
            #
            # cleaned = self.stft.inverse(clean_D)
            #
            # enhanced_mag_compress = enhanced_mag ** 0.3
            # enhanced_real_compress = enhanced_mag_compress * mixture_real / mixture_mag
            # enhanced_imag_compress = enhanced_mag_compress * mixture_imag / mixture_mag
            #
            #
            #
            # clean_mag_compress = clean_mag ** 0.3
            #
            # clean_real_compress = clean_mag_compress * clean_real / clean_mag
            # clean_imag_compress = clean_mag_compress * clean_imag / clean_mag

            ###PHASEN loss
            # enhanced_spec, enhanced_phase = self.model(mix_spec)
            # enhanced_phase = enhanced_phase / (torch.sqrt(
            #     torch.abs(enhanced_phase[:, 0]) ** 2 +
            #     torch.abs(enhanced_phase[:, 1]) ** 2)
            #                  + 1e-8).unsqueeze(1)
            #
            # amp_pre = torch.unsqueeze(mixture_mag, 1)
            # clean_cprs_mag = clean_mag ** 0.3
            # est_spec = amp_pre*enhanced_spec*enhanced_phase
            # est_real = est_spec[:, 0, :, :]
            # est_imag = est_spec[:, 1, :, :]
            # est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2+1e-8)
            # est_cprs_mag = est_mag**0.3
            #
            # est_cspec = torch.cat([est_spec[:,0], est_spec[:,1]], 2)
            # clean_cspec = torch.cat([clean_D[:,:,:,0], clean_D[:,:,:,1]], 2)
            # compress_coff = (clean_cprs_mag/(1e-8+clean_mag)).repeat(1,1,2)
            # est_cspec = est_cspec*compress_coff
            # clean_cspec =  clean_cspec*compress_coff


            ####cIRM
            # enhanced_real, enhanced_imag = self.model(mix_spec)
            #
            # enhanced_real = torch.squeeze(enhanced_real,1)
            # enhanced_imag = torch.squeeze(enhanced_imag,1)

            #####cRM-SA
            # enhanced_real, enhanced_imag = self.model(mix_spec)
            #
            # enhanced_real = torch.squeeze(enhanced_real,1)
            # enhanced_imag = torch.squeeze(enhanced_imag,1)
            #
            #
            # est_Mr = -10. * torch.log((1*torch.ones_like(enhanced_real) - enhanced_real +1e-8) / ((1*torch.ones_like(enhanced_real) + enhanced_real +1e-8)))
            # est_Mi = -10. * torch.log((1*torch.ones_like(enhanced_imag) - enhanced_imag +1e-8) / ((1*torch.ones_like(enhanced_imag) + enhanced_imag +1e-8)))
            #
            # recons_real = est_Mr * mixture_real - est_Mi * mixture_imag
            # recons_imag = est_Mr * mixture_imag + est_Mi * mixture_real
            #
            # #### avoid nan data
            # zero = torch.zeros_like(recons_real)
            # recons_real = torch.where(torch.isnan(recons_real), zero, recons_real)
            # recons_imag = torch.where(torch.isnan(recons_imag), zero, recons_imag)
            # recons_mag = torch.sqrt(recons_real ** 2 + recons_imag ** 2) # [batch, T, F]
            #
            # enhanced_D = torch.stack([recons_real, recons_imag], 3)
            # enhanced = self.stft.inverse(enhanced_D)
            # cleaned = self.stft.inverse(clean_D)
            # x_input = self.stft.inverse(mixture_D)



            ####### weighted SDR loss

            # loss_wsdr = wSDRLoss(x_input, cleaned, enhanced)
            # loss_stoi = NegSTOILoss(sample_rate=16000)



            ##loss compute
            # loss = self.loss_function(enhanced_mag, clean_mag, n_frames_list)
            # loss = self.loss_function(enhanced_spec, clean_spec, n_frames_list)
            # loss = 0.5*self.loss_function(clean_cprs_mag, est_cprs_mag, n_frames_list) + 0.5*self.loss_function( clean_cspec, est_cspec, n_frames_list)


            ###### AMP_loss
            # loss_amp = F.mse_loss(recons_mag, clean_mag)

            ######STOI_loss
            # loss_stoi_batch = loss_stoi(enhanced, cleaned)



            ##### PMSQE_loss
            # loss_pmsqe = self.loss_function(enhanced, cleaned)


            ######PASE_loss
            # pase_in_c = cleaned
            # pase_in_c = pase_in_c.unsqueeze(1)
            # clean_pase =pase(pase_in_c)
            # clean_pase = clean_pase.reshape(clean_pase.size()[0], -1)
            # pase_in_e = enhanced
            # pase_in_e = pase_in_e.unsqueeze(1)
            # enh_pase =pase(pase_in_e)
            # enh_pase = enh_pase.reshape(enh_pase.size()[0], -1)
            # loss_pase = F.mse_loss(clean_pase, enh_pase)


            ####PHASEN_loss

            # gth_cprs_mag_spec = (clean_mag +1e-8) ** 0.3
            # est_cprs_mag_spec = (recons_mag +1e-8) ** 0.3
            # #### avoid nan
            # zero = torch.zeros_like(gth_cprs_mag_spec)
            # gth_cprs_mag_spec = torch.where(torch.isnan(gth_cprs_mag_spec), zero, gth_cprs_mag_spec)
            # est_cprs_mag_spec = torch.where(torch.isnan(est_cprs_mag_spec), zero, est_cprs_mag_spec)
            #
            # # compress_coff = ((gth_cprs_mag_spec / (1e-8 + clean_mag))).repeat(1, 1, 2)
            # amp_loss = F.mse_loss(est_cprs_mag_spec, gth_cprs_mag_spec)
            # recons_spec = torch.cat([recons_real, recons_imag], dim=2)
            # clean_spec = torch.cat([clean_real, clean_imag], dim=2)
            # zero = torch.zeros_like(recons_spec)
            # recons_spec = torch.where(torch.isnan(recons_spec), zero, recons_spec)
            # clean_spec = torch.where(torch.isnan(clean_spec), zero, clean_spec)
            # phase_loss = F.mse_loss(recons_spec, clean_spec)

            ##### SNR_loss
            # loss = self.loss_function(enhanced, cleaned)

            ##### SDR_loss
            # loss_sdr = self.loss_function(cleaned, x_input, enhanced)

            ##### SDR_loss
            # loss_sdr = self.loss_function(cleaned, enhanced)


            #####cIRM_loss
            # loss_cIRM = self.loss_function(cIRMTar_r, enhanced_real, n_frames_list) + self.loss_function(cIRMTar_i, enhanced_imag, n_frames_list)
            # zero = torch.zeros_like(enhanced_real)
            # enhanced_real = torch.where(torch.isnan(enhanced_real), zero, enhanced_real)
            # enhanced_imag = torch.where(torch.isnan(enhanced_imag), zero, enhanced_imag)
            # cIRMTar_r = torch.where(torch.isnan(cIRMTar_r), zero, cIRMTar_r)
            # cIRMTar_i = torch.where(torch.isnan(cIRMTar_i), zero, cIRMTar_i)
            # loss_cIRM = F.mse_loss(cIRMTar_r, enhanced_real) + F.mse_loss(cIRMTar_i, enhanced_imag)
            #####CRM-SA_loss
            # loss = self.loss_function(clean_real, recons_real, n_frames_list) + self.loss_function(clean_imag, recons_imag, n_frames_list) + loss_cIRM
            # loss = F.mse_loss(clean_real, recons_real) + F.mse_loss(clean_imag, recons_imag) + loss_cIRM + 0.2*loss_stoi_batch.mean()
            # loss = F.mse_loss(clean_real, recons_real) + F.mse_loss(clean_imag, recons_imag) + loss_cIRM + 0.015 *self.loss_function(enhanced, cleaned)
            # loss = loss_cIRM + 0.005*loss_si_snr
            # loss = loss_cIRM + 0.5 * amp_loss + 0.5 * phase_loss

            # loss = loss_cIRM + 0.25*loss_pase + F.mse_loss(clean_real, recons_real) + F.mse_loss(clean_imag, recons_imag) + 0.1*loss_pmsqe
            # loss = loss_cIRM + F.mse_loss(clean_real, recons_real) + F.mse_loss(clean_imag, recons_imag) + 0.5*loss_pmsqe + 0.1*loss_pase
            # loss = loss_cIRM + F.mse_loss(clean_real, recons_real) + F.mse_loss(clean_imag, recons_imag) + 0.1*loss_pmsqe
            # loss = loss_cIRM + F.mse_loss(clean_real, recons_real) + F.mse_loss(clean_imag, recons_imag)+ 0.1*loss_pmsqe + 0.25*loss_pase
            # loss_CRM = F.mse_loss(clean_real, recons_real) + F.mse_loss(clean_imag, recons_imag)
            # loss = loss_cIRM + 0.25*loss_pase + loss_stoi_batch.mean()
            # loss = loss_cIRM + F.mse_loss(clean_real, recons_real) + F.mse_loss(clean_imag, recons_imag)
            # loss = loss_cIRM + loss_sdr*0.1
            # loss = loss_sdr

            # flooding = (loss_CRM-0.15).abs() + 0.15

            # loss = loss_CRM + loss_cIRM + 0.1*loss_pmsqe + 0.25*loss_pase

            ################ loss_irm
            # loss_irm = F.mse_loss(irm_mask, est_mask)

            # ################ loss_mag
            # loss_mag = F.mse_loss(enhanced_mag_compress, clean_mag_compress)
            #
            # ################ loss_cRM
            # loss_CRM = F.mse_loss(clean_real_compress, enhanced_real_compress) + F.mse_loss(clean_imag_compress, enhanced_imag_compress)

            ################ loss_mag_without_compress
            loss_mag = F.mse_loss(est_magdB, clean_mag_torch)

            ################ loss_LPS
            # loss_lps = F.mse_loss(est_LPS, clean_LPS_compress)

            ################ loss_cRM_without_compress
            # loss_CRM = F.mse_loss(clean_real, enhanced_real) + F.mse_loss(clean_imag, enhanced_imag)


            # loss = loss_CRM + loss_mag + 0.1*loss_pmsqe + 0.25*loss_pase
            loss = loss_mag


            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), 10)

            if step %1000 ==0:
                print((loss))

            step += 1



            self.optimizer.step()

            loss_total += float(loss)
            # loss_wsdr_total += loss_wsdr
            # loss_pmsqe_total += loss_pmsqe
            # loss_stoi_total += loss_stoi_batch.mean()
        self.scheduler.step()
        dataloader_len = len(self.train_dataloader)
        print((loss_total / dataloader_len))
        # print(epoch, self.scheduler.get_lr()[0])
        # print(loss_pmsqe_total / dataloader_len)
        torch.cuda.empty_cache()
        # print(loss_stoi_total / dataloader_len)
        # print(loss_wsdr_total / dataloader_len)
        self.viz.writer.add_scalar("Loss_TOTAL/Train", loss_total / dataloader_len, epoch)
        # self.viz.writer.add_scalar("Loss_PMSQE/Train", loss_pmsqe_total / dataloader_len, epoch)
        # self.viz.writer.add_scalar("Loss_STOI/Train", loss_stoi_total / dataloader_len, epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0
        loss_total_cIRM_real = 0.0
        loss_total_cIRM_imag = 0.0
        loss_total_mag = 0.0
        mixture_mean = None
        mixture_std = None
        sisnr_c_d = []
        sisnr_c_n = []
        stoi_c_n = []
        stoi_c_d = []
        pesq_c_n = []
        pesq_c_d = []
        split_frame = 256
        for mixture, clean, n_frames_list, names in tqdm(self.validation_dataloader):
            self.model.eval()

            ################# compute mixture PSD
            mixture = mixture.to('cpu')
            mixture_np = np.array(mixture)

            total_frames = int(mixture_np.shape[1] / 128 + 1)
            nframes = int(256 * np.ceil(total_frames / 256))

            mix_mag_torch = torch.zeros([mixture_np.shape[0], 256, nframes], dtype=torch.float32)
            mix_phase_torch = torch.zeros([mixture_np.shape[0], 256, nframes], dtype=torch.float32)
            mix_norm_torch = torch.zeros([mixture_np.shape[0], 2], dtype=torch.float32)
            for batch_index in range(mixture_np.shape[0]):
                temp_mixture_speech = np.asfortranarray(mixture_np[batch_index, :])
                psd_temp = psd(temp_mixture_speech, preprocess=False)
                MagdB_temp = torch.from_numpy(psd_temp['MagdB'])
                Norm_temp = torch.from_numpy(np.array(psd_temp['Norm']))
                Phase_temp = torch.from_numpy(psd_temp['Phase'])

                mix_mag_torch[batch_index, :, :] = MagdB_temp
                mix_phase_torch[batch_index, :, :] = Phase_temp
                mix_norm_torch[batch_index, :] = Norm_temp

            ################## compute clean PSD
            clean = clean.to('cpu')
            clean_np = np.array(clean)
            # clean_np = np.asfortranarray(clean_np)
            clean_mag_torch = torch.zeros([clean_np.shape[0], 256, nframes], dtype=torch.float32)
            clean_phase_torch = torch.zeros([clean_np.shape[0], 256, nframes], dtype=torch.float32)
            clean_norm_torch = torch.zeros([clean_np.shape[0], 2], dtype=torch.float32)
            for batch_index in range(clean_np.shape[0]):
                temp_clean_speech = np.asfortranarray(clean_np[batch_index, :])
                psd_temp = psd(temp_clean_speech, preprocess=False)
                MagdB_temp = torch.from_numpy(psd_temp['MagdB'])
                Norm_temp = torch.from_numpy(np.array(psd_temp['Norm']))
                Phase_temp = torch.from_numpy(psd_temp['Phase'])

                clean_mag_torch[batch_index, :, :] = MagdB_temp
                clean_phase_torch[batch_index, :, :] = Phase_temp
                clean_norm_torch[batch_index, :] = Norm_temp

            mix_mag_torch = mix_mag_torch.to(self.device)
            clean_mag_torch = clean_mag_torch.to(self.device)


            ####### split frames
            mix_tuple = torch.split(mix_mag_torch, split_frame, dim=2)  # 按照split_frame这个维度去分
            index_num = 0
            for item in mix_tuple:


                est_mag = self.model(item)

                if index_num == 0:
                    MagdB_enh = est_mag

                else:
                    MagdB_enh = torch.cat([MagdB_enh, est_mag], dim=2)

                index_num = index_num+1

            loss_total_mag += F.mse_loss(MagdB_enh, clean_mag_torch)

            speech_maximum_len = (MagdB_enh.shape[2] -1) * 128
            batch_enhanced_speeches = np.zeros((MagdB_enh.shape[0], speech_maximum_len), dtype=np.float32)
            batch_clean_speeches = np.zeros((MagdB_enh.shape[0], speech_maximum_len), dtype=np.float32)
            for batch_index in range(MagdB_enh.shape[0]):

                temp_mag_dB = MagdB_enh[batch_index, :, :].detach().cpu().numpy()
                temp_phase = mix_phase_torch[batch_index, :, :].detach().cpu().numpy()
                temp_norm = mix_norm_torch[batch_index, :].detach().cpu().numpy()

                enhanced_mag = np.interp(temp_mag_dB, [-1, 1], temp_norm)
                temp = np.zeros((257, enhanced_mag.shape[1])) + 1j * np.zeros((257, enhanced_mag.shape[1]))
                temp[:-1, :] = 10 ** (enhanced_mag / 20) * (np.cos(temp_phase) + np.sin(temp_phase) * 1j)
                enhanced_audio = istft(temp)
                enhanced_audio = 0.98 * enhanced_audio / np.max(np.abs(enhanced_audio))

                batch_enhanced_speeches[batch_index, :] = enhanced_audio

            for batch_index in range(MagdB_enh.shape[0]):
                temp_mag_dB = clean_mag_torch[batch_index, :, :].detach().cpu().numpy()
                temp_phase = clean_phase_torch[batch_index, :, :].detach().cpu().numpy()
                temp_norm = clean_norm_torch[batch_index, :].detach().cpu().numpy()

                clean_mag = np.interp(temp_mag_dB, [-1, 1], temp_norm)
                temp = np.zeros((257, clean_mag.shape[1])) + 1j * np.zeros((257, clean_mag.shape[1]))
                temp[:-1, :] = 10 ** (clean_mag / 20) * (np.cos(temp_phase) + np.sin(temp_phase) * 1j)
                clean_audio = istft(temp)
                clean_audio = 0.98 * clean_audio / np.max(np.abs(clean_audio))

                batch_clean_speeches[batch_index, :] = clean_audio



            len_list = []

            for n_frames in n_frames_list:

                len_list.append((n_frames-1) * 128)

            # if self.z_score:
            #     enhanced_mag = reverse_z_score(enhanced_mag, mixture_mean, mixture_std)
            #
            # masks = []
            # len_list = []
            # for n_frames in n_frames_list:
            #     masks.append(torch.ones(n_frames, 257, dtype=torch.float32))
            #     len_list.append((n_frames - 1) * 256 + 512)
            #
            # masks = pad_sequence(masks, batch_first=True).to(self.device)  # [batch, longest n_frame, n_fft]
            #
            # ####cRM-SA
            # recons_real = enhanced_real * masks
            # recons_imag = enhanced_imag * masks
            # enhanced_D = torch.stack([recons_real, recons_imag], 3)

            # enhanced_D = enhanced_spec.permute(0, 2, 3, 1)

            # enhanced_speeches = enhanced.detach().cpu().numpy()
            mixture_speeches = mixture.detach().cpu().numpy()
            # clean_speeches = clean.detach().cpu().numpy()

            enhanced_speeches = batch_enhanced_speeches
            clean_speeches = batch_clean_speeches

            for i in range(len(n_frames_list)):
                enhanced = enhanced_speeches[i][:len_list[i]]
                mixture = mixture_speeches[i][:len_list[i]]
                clean = clean_speeches[i][:len_list[i]]





                # self.viz.writer.add_audio(f"Audio/{names[i]}_Mixture", mixture, epoch, sample_rate=16000)
                # self.viz.writer.add_audio(f"Audio/{names[i]}_Enhanced", enhanced, epoch, sample_rate=16000)
                # self.viz.writer.add_audio(f"Audio/{names[i]}_Clean", clean, epoch, sample_rate=16000)
                #
                # fig, ax = plt.subplots(3, 1)
                # for j, y in enumerate([mixture, enhanced, clean]):
                #     ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                #         np.mean(y),
                #         np.std(y),
                #         np.max(y),
                #         np.min(y)
                #     ))
                #     librosa.display.waveplot(y, sr=16000, ax=ax[j])
                # plt.tight_layout()
                # self.viz.writer.add_figure(f"Waveform/{names[i]}", fig, epoch)

                stoi_c_n.append(compute_STOI(clean, mixture, sr=16000))
                stoi_c_d.append(compute_STOI(clean, enhanced, sr=16000))
                pesq_c_n.append(compute_PESQ(clean, mixture, sr=16000))
                pesq_c_d.append(compute_PESQ(clean, enhanced, sr=16000))
                sisnr_c_d.append(si_snr(enhanced, clean))
                sisnr_c_n.append(si_snr(mixture, clean))

                clean_dir = "E:\\real_time_exp\\CRN_16k_DNS_2_dereverbonly\\PHA-CRN\\temp_val_wavs\\" + "clean_" + names[i] + ".wav"
                noisy_dir = "E:\\real_time_exp\\CRN_16k_DNS_2_dereverbonly\\PHA-CRN\\temp_val_wavs\\" + "noisy_" + names[i] + ".wav"
                enh_dir = "E:\\real_time_exp\\CRN_16k_DNS_2_dereverbonly\\PHA-CRN\\temp_val_wavs\\" + "enh_" + names[i] + ".wav"

                sf.write(clean_dir, clean, 16000)
                sf.write(noisy_dir,  mixture, 16000)
                sf.write(enh_dir, enhanced, 16000)

        ################## visualize real recording wavs
        real_record_path = "E:\\real_time_exp\\CRN_16k_DNS_2_dereverbonly\\16k_dereverb"
        speech_names = []
        for dirpath, dirnames, filenames in os.walk(real_record_path):
            for filename in filenames:
                if filename.lower().endswith(".wav"):
                    speech_names.append(os.path.join(dirpath, filename))

        for speech_na in speech_names:
            fig, ax = plt.subplots(4, 1)
            name = os.path.basename(speech_na)
            mixture, sr = sf.read(speech_na, dtype="float32")
            real_len = len(mixture)
            assert sr == 16000
            # mixture = np.expand_dims(mixture, axis=0)
            # torch_mix = torch.from_numpy(mixture).to(self.device)
            # mixture_D = self.stft.transform(torch_mix)
            # mixture_real = mixture_D[:, :, :, 0]
            # mixture_imag = mixture_D[:, :, :, 1]
            # mixture_mag = torch.sqrt(mixture_real ** 2 + mixture_imag ** 2 + 1e-8)
            # # LPS_fea = torch.log10(mixture_mag ** 2 +1e-8)
            # LPS_fea = torch.log(mixture_mag ** 2 + 1e-8)
            # LPS_in = LPS_fea[:, :, :256]
            # LPS_in_compress = (1. - torch.exp(-2 * LPS_in)) / ((1. + torch.exp(-2 * LPS_in)) + 1e-8)

            psd_temp = psd(mixture, preprocess=False)
            MagdB_temp = torch.from_numpy(psd_temp['MagdB']).float().to(self.device)
            Norm_temp = psd_temp['Norm']
            Phase_temp = psd_temp['Phase']

            MagdB_temp = MagdB_temp.unsqueeze(0)

            ####### split frames
            mix_tuple = torch.split(MagdB_temp, split_frame, dim=2)  # 按照split_frame这个维度去分
            index_num = 0
            for item in mix_tuple:


                est_mag = self.model(item)

                if index_num == 0:
                    MagdB_enh = est_mag

                else:
                    MagdB_enh = torch.cat([MagdB_enh, est_mag], dim=2)

                index_num = index_num+1

            enhanced_mag = MagdB_enh.detach().cpu().squeeze().numpy()

            enhanced_mag = np.interp(enhanced_mag, [-1, 1], Norm_temp)
            temp = np.zeros((257, enhanced_mag.shape[1])) + 1j * np.zeros((257, enhanced_mag.shape[1]))
            temp[:-1, :] = 10 ** (enhanced_mag / 20) * (np.cos(Phase_temp) + np.sin(Phase_temp) * 1j)
            enhanced_audio = istft(temp)
            enhanced_audio = 0.98 * enhanced_audio / np.max(np.abs(enhanced_audio))

            enhanced_audio = enhanced_audio[:real_len]



            # LPS_enh = -0.5 * torch.log((1 - LPS_enh) / ((1 + LPS_enh + 1e-8)) + 1e-8)
            #
            # LPS_enh = torch.cat([LPS_enh, LPS_fea[:,:,256].unsqueeze(2)], dim=2)

            # enhanced_mag = torch.sqrt(torch.exp(LPS_enh) + 1e-8)
            #
            # enhanced_real = enhanced_mag * mixture_real / mixture_mag
            # enhanced_imag = enhanced_mag * mixture_imag / mixture_mag
            #
            # enhanced_D = torch.stack([enhanced_real, enhanced_imag], 3)
            #
            # #### avoid nan data
            # zero = torch.zeros_like(enhanced_D)
            # enhanced_D = torch.where(torch.isnan(enhanced_D), zero, enhanced_D)
            #
            # # enhanced_D = torch.stack([enhanced_real, enhanced_imag], 3)
            # enhanced = self.stft.inverse(enhanced_D)

            # enhanced = enhanced.detach().cpu().squeeze().numpy()
            #
            # mixture = np.squeeze(mixture)

            enhanced = enhanced_audio

            librosa.display.waveplot(mixture, sr=16000, label='noisy_waveform', ax=ax[0])
            librosa.display.waveplot(enhanced, sr=16000, label='enhanced_waveform', ax=ax[1])

            mag_noisy, _ = librosa.magphase(librosa.stft(mixture, n_fft=512, hop_length=256, win_length=512))
            librosa.display.specshow(librosa.amplitude_to_db(mag_noisy), cmap="magma", y_axis="linear", ax=ax[2], sr=16000, label='noisy_spectrogram')

            mag_enh, _ = librosa.magphase(librosa.stft(enhanced, n_fft=512, hop_length=256, win_length=512))
            librosa.display.specshow(librosa.amplitude_to_db(mag_enh), cmap="magma", y_axis="linear", ax=ax[3], sr=16000, label='enhanced_spectrogram')

            plt.tight_layout()
            self.viz.writer.add_figure(f"Waveform/{epoch}/{name}", fig, epoch)
            temp_enh_dir = "E:\\real_time_exp\\CRN_16k_DNS_2_dereverbonly\\PHA-CRN\\temp_test_inference\\" + "epoch_" + str(epoch)
            temp_enh_path = temp_enh_dir + "\\" + name
            create_folder(temp_enh_dir)
            sf.write(temp_enh_path, enhanced, 16000)
            # self.viz.writer.add_audio(f"Audio/{epoch}/{name}_Mixture", mixture, epoch, sample_rate=16000)
            # self.viz.writer.add_audio(f"Audio/{epoch}/{name}_Enhanced", enhanced, epoch, sample_rate=16000)


        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        self.viz.writer.add_scalars(f"Metrics/STOI", {
            "clean_and_noisy": get_metrics_ave(stoi_c_n),
            "clean_and_denoisy": get_metrics_ave(stoi_c_d)
        }, epoch)

        self.viz.writer.add_scalars(f"Metrics/PESQ", {
            "clean_and_noisy": get_metrics_ave(pesq_c_n),
            "clean_and_denoisy": get_metrics_ave(pesq_c_d)
        }, epoch)

        self.viz.writer.add_scalars(f"Metrics/freq_Mag_MSE", {
            "mag_loss": loss_total_mag / len(self.validation_dataloader),
        }, epoch)

        self.viz.writer.add_scalars(f"Metrics/sisnr", {
            "clean_and_noisy": get_metrics_ave(sisnr_c_n),
            "clean_and_denoisy": get_metrics_ave(sisnr_c_d)
        }, epoch)

        # dataloader_len = len(self.validation_dataloader)
        # self.viz.writer.add_scalar("Loss/Validation", loss_total / dataloader_len, epoch)

        score = (get_metrics_ave(stoi_c_d) + self._transform_pesq_range(get_metrics_ave(pesq_c_d))) / 2
        # score = get_metrics_ave(stoi_c_d)
        return score

