import librosa
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from trainer.base_trainer import BaseTrainer
import matplotlib.pyplot as plt

plt.switch_backend("agg")
plt.rcParams.update({'figure.max_open_warning': 0})
import numpy as np
import librosa.display
from tqdm import tqdm
from pase.models.frontend import wf_builder
from utils.utils import compute_STOI, compute_PESQ, z_score, reverse_z_score


class Trainer(BaseTrainer):
    def __init__(self,
                 config,
                 resume,
                 model,
                 optimizer,
                 loss_function,
                 train_dataloader,
                 validation_dataloader):
        super(Trainer, self).__init__(config, resume, model, optimizer, loss_function)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0
        pase = wf_builder('cfg\\frontend\\PASE+.cfg').eval()
        pase.load_pretrained('.\\FE_e199.ckpt', load_last=True, verbose=True)
        pase.to('cuda')
        for mixture, clean, n_frames_list, _ in tqdm(self.train_dataloader, desc="Training"):
            self.optimizer.zero_grad()

            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            # Mixture mag and Clean mag
            mixture_D  = self.stft.transform(mixture)
            mixture_real = mixture_D[:, :, :, 0]
            mixture_imag = mixture_D[:, :, :, 1]

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



            #### run all frames
            mix_spec = mixture_D.permute(0, 3, 1, 2)
            cIRM_est_real, cIRM_est_imag = self.model(mix_spec)



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

            EPSILON = 1e-8
            with torch.no_grad():
                mask_for_loss_list = []
                # 制作掩码
                for frame_num in n_frames_list:
                    mask_for_loss_list.append(torch.ones(frame_num, enhanced_real.size()[2], dtype=torch.float32))
                # input: list of tensor
                # output: B T F
                mask_for_loss = pad_sequence(mask_for_loss_list, batch_first=True).cuda()
            # 使用掩码计算真实值

            #####cIRM_loss
            loss_cIRM = ((enhanced_real * mask_for_loss  - cIRMTar_r * mask_for_loss) ** 2).sum() / (mask_for_loss.sum() + EPSILON) + ((enhanced_imag * mask_for_loss  - cIRMTar_i * mask_for_loss) ** 2).sum() / (mask_for_loss.sum() + EPSILON)

            #####CRM-SA_loss
            loss_CRM_SA = ((recons_real * mask_for_loss  - clean_real * mask_for_loss) ** 2).sum() / (mask_for_loss.sum() + EPSILON) + ((recons_imag * mask_for_loss  - clean_imag * mask_for_loss) ** 2).sum() / (mask_for_loss.sum() + EPSILON)

            #####PASE_loss
            # with torch.no_grad():
            #     recons_real_masked = recons_real * mask_for_loss
            #     recons_imag_masked = recons_imag * mask_for_loss
            #     enhanced_D = torch.stack([recons_real_masked, recons_imag_masked], 3)
            #     clean_real_masked  = clean_real * mask_for_loss
            #     clean_imag_masked  = clean_imag * mask_for_loss
            #     cleaned_D = torch.stack([clean_real_masked, clean_imag_masked], 3)
            #     enhanced = self.stft.inverse(enhanced_D)
            #     cleaned = self.stft.inverse(cleaned_D)
            #     pase_in_c = cleaned
            #     pase_in_c = pase_in_c.unsqueeze(1)
            #     clean_pase =pase(pase_in_c)
            #     clean_pase = clean_pase.reshape(clean_pase.size()[0], -1)
            #     pase_in_e = enhanced
            #     pase_in_e = pase_in_e.unsqueeze(1)
            #     enh_pase =pase(pase_in_e)
            #     enh_pase = enh_pase.reshape(enh_pase.size()[0], -1)
            recons_real_masked = recons_real * mask_for_loss
            recons_imag_masked = recons_imag * mask_for_loss
            enhanced_D = torch.stack([recons_real_masked, recons_imag_masked], 3)
            clean_real_masked = clean_real * mask_for_loss
            clean_imag_masked = clean_imag * mask_for_loss
            cleaned_D = torch.stack([clean_real_masked, clean_imag_masked], 3)
            enhanced = self.stft.inverse(enhanced_D)
            cleaned = self.stft.inverse(cleaned_D)
            len_list = []
            for n_frames in n_frames_list:
                len_list.append((n_frames - 1) * 160 + 320)

            for i in range(len(n_frames_list)):
                enhanced_temp = enhanced[i][:len_list[i]]
                cleaned_temp = cleaned[i][:len_list[i]]
                pase_in_c = cleaned_temp
                pase_in_c = pase_in_c.unsqueeze(0)
                pase_in_c = pase_in_c.unsqueeze(1)
                clean_pase = pase(pase_in_c)
                clean_pase = clean_pase.reshape(clean_pase.size()[0], -1)
                pase_in_e = enhanced_temp
                pase_in_e = pase_in_e.unsqueeze(0)
                pase_in_e = pase_in_e.unsqueeze(1)
                enh_pase = pase(pase_in_e)
                enh_pase = enh_pase.reshape(enh_pase.size()[0], -1)
                if i ==0:
                    loss_pase = F.mse_loss(clean_pase, enh_pase)
                else:
                    loss_pase += F.mse_loss(clean_pase, enh_pase)


            loss = loss_cIRM + loss_CRM_SA + (loss_pase/len(n_frames_list))
            # loss = loss_cIRM + loss_CRM_SA


            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), 10)

            self.optimizer.step()

            loss_total += float(loss)

        dataloader_len = len(self.train_dataloader)
        print((loss_total / dataloader_len))
        print(loss_pase /dataloader_len)
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

        for mixture, clean, n_frames_list, names in tqdm(self.validation_dataloader):
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
            # mix_tuple = torch.split(mixture_D, split_frame, dim=1)  # 按照split_frame这个维度去分
            # index_num = 0
            # for item in mix_tuple:
            #     mix_spec = item.permute(0, 3, 1, 2)
            #     if mix_spec.shape[2]==1:
            #         pad_spec = torch.cat([mix_spec, mix_spec], dim=2)
            #         enhanced_real1, enhanced_imag1 = self.model(pad_spec)
            #         enhanced_real, _ = torch.split(enhanced_real1, 1, dim=2)
            #         enhanced_imag, _ = torch.split(enhanced_imag1, 1, dim=2)
            #     else:
            #         enhanced_real, enhanced_imag = self.model(mix_spec)
            #
            #     if index_num == 0:
            #         cIRM_est_real = enhanced_real
            #         cIRM_est_imag = enhanced_imag
            #     else:
            #         cIRM_est_real = torch.cat([cIRM_est_real, enhanced_real], dim=2)
            #         cIRM_est_imag = torch.cat([cIRM_est_imag, enhanced_imag], dim=2)
            #     index_num = index_num+1

            #### run all frames
            mix_spec = mixture_D.permute(0, 3, 1, 2)
            cIRM_est_real, cIRM_est_imag = self.model(mix_spec)



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

            # #####cIRM_loss
            # loss_cIRM = self.loss_function(cIRMTar_r, enhanced_real, n_frames_list) + self.loss_function(cIRMTar_i,
            #                                                                                              enhanced_imag,
            #                                                                                              n_frames_list)
            # #####CRM-SA_loss
            # loss = self.loss_function(clean_real, recons_real, n_frames_list) + self.loss_function(clean_imag,
            #                                                                                        recons_imag,
            #                                                                                        n_frames_list) + loss_cIRM
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
            "clean and noisy": get_metrics_ave(pesq_c_n),
            "clean anddenoisy": get_metrics_ave(pesq_c_d)
        }, epoch)

        # dataloader_len = len(self.validation_dataloader)
        # self.viz.writer.add_scalar("Loss/Validation", loss_total / dataloader_len, epoch)

        score = (get_metrics_ave(stoi_c_d) + self._transform_pesq_range(get_metrics_ave(pesq_c_d))) / 2
        # score = get_metrics_ave(stoi_c_d)
        return score

