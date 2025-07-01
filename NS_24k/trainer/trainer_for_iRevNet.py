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

plt.switch_backend("agg")
plt.rcParams.update({'figure.max_open_warning': 0})
import numpy as np
import librosa.display
from tqdm import tqdm

from utils.utils import compute_STOI, compute_PESQ, z_score, reverse_z_score
from iRevNet import iRevNetMaskEstimator, iRevNetFilter, iRevNetUtils


def l2_norm(s1, s2):
    # norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    # norm = torch.norm(s1*s2, 1, keepdim=True)

    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm

def si_snr(s1, s2, eps=1e-8):
    #s1 = remove_dc(s1)
    #s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return torch.mean(snr)

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
        filt = 'UNet5SpecNorm'
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.initPad = 3
        self.chNum = [1 + self.initPad]
        self.layerNum = 6
        for ii in range(1,self.layerNum+1):
            self.chNum.append(self.chNum[0]*(2**ii))
        self.decimate = ( ((np.array(self.chNum[1:])) / np.array(self.chNum[:-1])).astype(np.int) ).tolist()
        for layer in range(self.layerNum):
            tmptxt = "self.Filt"+str(layer+1)+" = iRevNetFilter."+filt+"("\
                 +"self.chNum["+str(layer)+"], "\
                 +"self.chNum["+str(layer)+"], "\
                 +"self.decimate["+str(layer)+"])"
            exec(tmptxt)

    def _train_epoch(self, epoch):
        loss_total = 0.0
        loss_stoi_total = 0.0
        pase = wf_builder('cfg\\frontend\\PASE+.cfg').eval()
        pase.load_pretrained('.\\FE_e199.ckpt', load_last=True, verbose=True)
        pase.to('cuda')
        for mixture, clean in tqdm(self.train_dataloader, desc="Training"):
            # self.model.train()
            self.optimizer.zero_grad()

            device = x.device
            xaInit, xbInit = iRevNetUtils.initSplit(x)
            bs, _, T = xaInit.shape
            tmpPad = torch.zeros(bs, self.initPad, T).cuda(device)
            xaL = [torch.cat((xaInit, tmpPad), 1)]
            xbL = [torch.cat((xbInit, tmpPad), 1)]
            for layer in range(self.layerNum):
                xatmp, xbtmp = eval('iRevNetUtils.iRevNetBlock_forward(' + \
                                    'xaL[' + str(layer) + '], xbL[' + str(layer) + '],' + \
                                    'self.Filt' + str(layer + 1) + ', self.decimate[' + str(layer) + '])')
                xaL.append(xatmp)
                xbL.append(xbtmp)

            phi = torch.cat((xaL[-1], xbL[-1]), 1)




            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            # Mixture mag and Clean mag
            mixture_D  = self.stft.transform(mixture)
            mixture_real = mixture_D[:, :, :, 0]
            mixture_imag = mixture_D[:, :, :, 1]
            mixture_mag = torch.sqrt(mixture_real ** 2 + mixture_imag ** 2) # [batch, T, F]
            mix_spec = mixture_D.permute(0, 3, 1, 2)

            clean_D  = self.stft.transform(clean)
            clean_real = clean_D[:, :, :, 0]
            clean_imag = clean_D[:, :, :, 1]

            Mr_c = (mixture_real * clean_real + mixture_imag * clean_imag) / ((mixture_real ** 2 + mixture_imag ** 2) +1e-8) +1e-8
            Mi_c = (mixture_real * clean_imag - mixture_imag * clean_real) / ((mixture_real ** 2 + mixture_imag ** 2) +1e-8) +1e-8

            zero = torch.zeros_like(Mr_c)
            Mr_c = torch.where(torch.abs(Mr_c)>850, zero, Mr_c)
            Mi_c = torch.where(torch.abs(Mi_c)>850, zero, Mi_c)

            cIRMTar_r = 10. * (1 - torch.exp(-0.1 * Mr_c)) / (1 + torch.exp(-0.1 * Mr_c))
            cIRMTar_i = 10. * (1 - torch.exp(-0.1 * Mi_c)) / (1 + torch.exp(-0.1 * Mi_c))

            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2 +1-8)



            if self.z_score:
                mixture_mag, _, _ = z_score(mixture_mag)
                clean_mag, _, _ = z_score(clean_mag)

            # enhanced_mag = self.model(mixture_mag)

            # enhanced_spec = self.model(mix_spec)

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
            enhanced_real, enhanced_imag = self.model(mix_spec)

            enhanced_real = torch.squeeze(enhanced_real,1)
            enhanced_imag = torch.squeeze(enhanced_imag,1)


            est_Mr = -10. * torch.log((10 - enhanced_real) / ((10 + enhanced_real)) + 1e-8)
            est_Mi = -10. * torch.log((10 - enhanced_imag) / ((10 + enhanced_imag)) + 1e-8)

            recons_real = est_Mr * mixture_real - est_Mi * mixture_imag
            recons_imag = est_Mr * mixture_imag - est_Mi * mixture_real

            #### avoid nan data
            zero = torch.zeros_like(recons_real)
            recons_real = torch.where(torch.isnan(recons_real), zero, recons_real)
            recons_imag = torch.where(torch.isnan(recons_imag), zero, recons_imag)
            recons_mag = torch.sqrt(recons_real ** 2 + recons_imag ** 2) # [batch, T, F]

            enhanced_D = torch.stack([recons_real, recons_imag], 3)
            enhanced = self.stft.inverse(enhanced_D)
            cleaned = self.stft.inverse(clean_D)
            x_input = self.stft.inverse(mixture_D)
            # loss_stoi = NegSTOILoss(sample_rate=16000)



            ##loss compute
            # loss = self.loss_function(enhanced_mag, clean_mag, n_frames_list)
            # loss = self.loss_function(enhanced_spec, clean_spec, n_frames_list)
            # loss = 0.5*self.loss_function(clean_cprs_mag, est_cprs_mag, n_frames_list) + 0.5*self.loss_function( clean_cspec, est_cspec, n_frames_list)


            ###### AMP_loss
            # loss_amp = F.mse_loss(recons_mag, clean_mag)

            ######STOI_loss
            # loss_stoi_batch = loss_stoi(enhanced, cleaned)

            ######PASE_loss
            pase_in_c = cleaned
            pase_in_c = pase_in_c.unsqueeze(1)
            clean_pase =pase(pase_in_c)
            clean_pase = clean_pase.reshape(clean_pase.size()[0], -1)
            pase_in_e = enhanced
            pase_in_e = pase_in_e.unsqueeze(1)
            enh_pase =pase(pase_in_e)
            enh_pase = enh_pase.reshape(enh_pase.size()[0], -1)
            loss_pase = F.mse_loss(clean_pase, enh_pase)


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
            loss_sdr = self.loss_function(cleaned, enhanced)


            #####cIRM_loss
            # loss_cIRM = self.loss_function(cIRMTar_r, enhanced_real, n_frames_list) + self.loss_function(cIRMTar_i, enhanced_imag, n_frames_list)
            # zero = torch.zeros_like(enhanced_real)
            # enhanced_real = torch.where(torch.isnan(enhanced_real), zero, enhanced_real)
            # enhanced_imag = torch.where(torch.isnan(enhanced_imag), zero, enhanced_imag)
            # cIRMTar_r = torch.where(torch.isnan(cIRMTar_r), zero, cIRMTar_r)
            # cIRMTar_i = torch.where(torch.isnan(cIRMTar_i), zero, cIRMTar_i)
            loss_cIRM = F.mse_loss(cIRMTar_r, enhanced_real) + F.mse_loss(cIRMTar_i, enhanced_imag)
            #####CRM-SA_loss
            # loss = self.loss_function(clean_real, recons_real, n_frames_list) + self.loss_function(clean_imag, recons_imag, n_frames_list) + loss_cIRM
            # loss = F.mse_loss(clean_real, recons_real) + F.mse_loss(clean_imag, recons_imag) + loss_cIRM + 0.2*loss_stoi_batch.mean()
            # loss = F.mse_loss(clean_real, recons_real) + F.mse_loss(clean_imag, recons_imag) + loss_cIRM + 0.015 *self.loss_function(enhanced, cleaned)
            # loss = loss_cIRM + 0.005*loss_si_snr
            # loss = loss_cIRM + 0.5 * amp_loss + 0.5 * phase_loss

            # loss = loss_cIRM + 0.25*loss_pase + F.mse_loss(clean_real, recons_real) + F.mse_loss(clean_imag, recons_imag)
            loss = loss_cIRM + 0.25*loss_pase + 0.1*loss_sdr
            # loss = loss_cIRM + F.mse_loss(clean_real, recons_real) + F.mse_loss(clean_imag, recons_imag)
            # loss = loss_cIRM + loss_sdr*0.1
            # loss = loss_sdr



            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), 10)

            self.optimizer.step()

            loss_total += float(loss)
            # loss_stoi_total += loss_stoi_batch.mean()
        self.scheduler.step()
        dataloader_len = len(self.train_dataloader)
        print((loss_total / dataloader_len))
        print(epoch, self.scheduler.get_lr()[0])
        # print(loss_stoi_total / dataloader_len)
        self.viz.writer.add_scalar("Loss/Train", loss_total / dataloader_len, epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0
        mixture_mean = None
        mixture_std = None
        stoi_c_n = []
        stoi_c_d = []
        pesq_c_n = []
        pesq_c_d = []
        split_frame = 100
        for mixture, clean, n_frames_list, names in tqdm(self.validation_dataloader):
            # self.model.eval()
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            # Mixture mag and Clean mag
            mixture_D = self.stft.transform(mixture)
            mixture_real = mixture_D[:, :, :, 0]
            mixture_imag = mixture_D[:, :, :, 1]


            clean_D = self.stft.transform(clean)
            clean_real = clean_D[:, :, :, 0]
            clean_imag = clean_D[:, :, :, 1]
            Mr_c = (mixture_real * clean_real + mixture_imag * clean_imag) / (
                        (mixture_real ** 2 + mixture_imag ** 2) + 1e-8)
            Mi_c = (mixture_real * clean_imag - mixture_imag * clean_real) / (
                        (mixture_real ** 2 + mixture_imag ** 2) + 1e-8)
            zero = torch.zeros_like(Mr_c)
            Mr_c = torch.where(torch.abs(Mr_c) > 850, zero, Mr_c)
            Mi_c = torch.where(torch.abs(Mi_c) > 850, zero, Mi_c)

            cIRMTar_r = 10. * (1. - torch.exp(-0.1 * Mr_c)) / ((1. + torch.exp(-0.1 * Mr_c)) + 1e-8)
            cIRMTar_i = 10. * (1. - torch.exp(-0.1 * Mi_c)) / ((1. + torch.exp(-0.1 * Mi_c)) + 1e-8)

            ####### split frames
            mix_tuple = torch.split(mixture_D, split_frame, dim=1)  # 按照split_frame这个维度去分
            index_num = 0
            for item in mix_tuple:
                mix_spec = item.permute(0, 3, 1, 2)
                if mix_spec.shape[2]==1:
                    pad_spec = torch.cat([mix_spec, mix_spec], dim=2)
                    enhanced_real1, enhanced_imag1 = self.model(pad_spec)
                    enhanced_real, _ = torch.split(enhanced_real1, 1, dim=2)
                    enhanced_imag, _ = torch.split(enhanced_imag1, 1, dim=2)
                    # enhanced_real, _ = torch.split(enhanced_real1, 1, dim=1)
                    # enhanced_imag, _ = torch.split(enhanced_imag1, 1, dim=1)
                else:
                    enhanced_real, enhanced_imag = self.model(mix_spec)

                if index_num == 0:
                    cIRM_est_real = enhanced_real
                    cIRM_est_imag = enhanced_imag
                else:
                    cIRM_est_real = torch.cat([cIRM_est_real, enhanced_real], dim=2)
                    cIRM_est_imag = torch.cat([cIRM_est_imag, enhanced_imag], dim=2)
                    # cIRM_est_real = torch.cat([cIRM_est_real, enhanced_real], dim=1)
                    # cIRM_est_imag = torch.cat([cIRM_est_imag, enhanced_imag], dim=1)
                index_num = index_num+1



            #####cRM-SA+cIRM
            enhanced_real = cIRM_est_real
            enhanced_imag = cIRM_est_imag
            enhanced_real = torch.squeeze(enhanced_real, 1)
            enhanced_imag = torch.squeeze(enhanced_imag, 1)

            est_Mr = -10. * torch.log((10 - enhanced_real) / ((10 + enhanced_real)) + 1e-8)
            est_Mi = -10. * torch.log((10 - enhanced_imag) / ((10 + enhanced_imag)) + 1e-8)

            recons_real = est_Mr * mixture_real - est_Mi * mixture_imag
            recons_imag = est_Mr * mixture_imag - est_Mi * mixture_real

            #### avoid nan data
            zero = torch.zeros_like(recons_real)
            recons_real = torch.where(torch.isnan(recons_real), zero, recons_real)
            recons_imag = torch.where(torch.isnan(recons_imag), zero, recons_imag)

            # enhanced_D = torch.stack([recons_real, recons_imag], 3)
            # enhanced = self.stft.inverse(enhanced_D)
            # cleaned = self.stft.inverse(clean_D)
            # loss_stoi = NegSTOILoss(sample_rate=16000)
            #
            # ######STOI_loss
            # loss_stoi_batch = loss_stoi(enhanced, cleaned)
            #
            # #####cIRM_loss
            # loss_cIRM = self.loss_function(cIRMTar_r, enhanced_real, n_frames_list) + self.loss_function(cIRMTar_i,
            #                                                                                              enhanced_imag,
            #                                                                                              n_frames_list)
            # #####CRM-SA_loss
            # loss = self.loss_function(clean_real, recons_real, n_frames_list) + self.loss_function(clean_imag,
            #                                                                                        recons_imag,
            #                                                                                        n_frames_list) + loss_cIRM + (1-loss_stoi_batch.mean())
            #
            # loss_total += loss

            if self.z_score:
                enhanced_mag = reverse_z_score(enhanced_mag, mixture_mean, mixture_std)

            masks = []
            len_list = []
            for n_frames in n_frames_list:
                masks.append(torch.ones(n_frames, 161, dtype=torch.float32))
                len_list.append((n_frames - 1) * 160 + 320)

            masks = pad_sequence(masks, batch_first=True).to(self.device)  # [batch, longest n_frame, n_fft]

            ####cRM-SA
            recons_real = recons_real * masks
            recons_imag = recons_imag * masks
            enhanced_D = torch.stack([recons_real, recons_imag], 3)

            # enhanced_D = enhanced_spec.permute(0, 2, 3, 1)
            enhanced = self.stft.inverse(enhanced_D)

            enhanced_speeches = enhanced.detach().cpu().numpy()
            mixture_speeches = mixture.detach().cpu().numpy()
            clean_speeches = clean.detach().cpu().numpy()

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


        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        self.viz.writer.add_scalars(f"Metrics/STOI", {
            "clean_and_noisy": get_metrics_ave(stoi_c_n),
            "clean_and_denoisy": get_metrics_ave(stoi_c_d)
        }, epoch)
        self.viz.writer.add_scalars(f"Metrics/PESQ", {
            "clean_and_noisy": get_metrics_ave(pesq_c_n),
            "clean_and_denoisy": get_metrics_ave(pesq_c_d)
        }, epoch)

        # dataloader_len = len(self.validation_dataloader)
        # self.viz.writer.add_scalar("Loss/Validation", loss_total / dataloader_len, epoch)

        score = (get_metrics_ave(stoi_c_d) + self._transform_pesq_range(get_metrics_ave(pesq_c_d))) / 2
        # score = get_metrics_ave(stoi_c_d)
        return score

