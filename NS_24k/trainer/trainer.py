import librosa
import torch
import torch.nn as nn
from pase.models.frontend import wf_builder
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from trainer.base_trainer import BaseTrainer
import matplotlib.pyplot as plt
from torch_stoi import NegSTOILoss
import soundfile as sf
import os
from test_for_PSD import *
# from model.BSD_loss import *
# from model.fwSNR_loss import *
from utils.stft import STFT
# from model.modulation_loss import *
from model.stft_loss import MultiResolutionSTFTLoss
from model.stftloss_newwin import MultiResolutionSTFTLoss_Newwin
# from model.Simple_DCCRN_Large import *
from dalib.modules.kernels import GaussianKernel
from dalib.adaptation.dan import MultipleKernelMaximumMeanDiscrepancy

plt.switch_backend("agg")
plt.rcParams.update({'figure.max_open_warning': 0})
import numpy as np
import librosa.display
from tqdm import tqdm

from utils.utils import compute_STOI, compute_PESQ, z_score, reverse_z_score

from dalib.modules.kernels import GaussianKernel
from dalib.adaptation.dan import MultipleKernelMaximumMeanDiscrepancy
from demucs import augment
from model.pmsqe import SingleSrcPMSQE
from asteroid_filterbanks import STFTFB, Encoder, transforms
from model.APC_SNR.apc_snr import APC_SNR_multi_filter, APC_SNR
# from fairseq.models.wav2vec import Wav2VecModel
from geomloss import SamplesLoss

# from model.three_denoiser.losses.perceptual import CompoundedPerceptualLoss
from model.HKD import HKDLoss, prob_loss
from model.CGRNN_FB import CGRNN_FB, CGRNN_FB_Large
from model.oneC_DIL_CRN import DPCRN_Model
# from model.ASR_loss.HuBERT_wrapper import load_lexical_model
# from model.ASR_loss.features_config import FeaturesConfig
# import torchaudio
from model.MOSNet.mosnet import MOSNet

# from funasr import AutoModel

def generate_window(win_len):
    filename = 'F:\\JMCheng\\real_time_exp\\16k_exp_new\\window256w512.txt'
    new_window = np.zeros(512)
    count = 0

    with open(filename, 'r') as file_to_read:
        line = file_to_read.readline()
        for i in line.split(','):
            try:
                new_window[count] = float(i)
                count += 1
            except:
                flag = 0
    return torch.from_numpy(new_window).float()



def SNR_loss(est, label):
    """
    计算真实的MSE
    :param est: 网络输出
    :param label: label
    :return:loss
    """
    EPSILON = 1e-7
    snr = torch.mean(label ** 2, dim=-1, keepdim=True) / (
                torch.mean((label - est) ** 2, dim=-1, keepdim=True) + EPSILON)
    snr_log = -10 * torch.log10(snr)
    snr_loss = torch.mean(snr_log)
    return snr_loss

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

def l2_norm_torch(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def si_snr_torch(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm_torch(s1, s2)
    s2_s2_norm = l2_norm_torch(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm_torch(s_target, s_target)
    noise_norm = l2_norm_torch(e_nosie, e_nosie)
    snr = 10 * torch.log10(target_norm / (noise_norm + eps) + eps)
    return torch.mean(snr)

def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0

    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0

    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    # Calculate the KL-divergence
    loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

    return loss


def SPKD_loss(f_s, f_t):
    bsz = f_s.shape[0]
    # f_s = f_s.view(bsz, -1)
    # f_t = f_t.view(bsz, -1)
    G_s = torch.mm(f_s, torch.t(f_s))
    G_s = G_s / G_s.norm(2)
    # G_s = torch.nn.functional.normalize(G_s)
    G_t = torch.mm(f_t, torch.t(f_t))
    G_t = G_t / G_t.norm(2)
    # G_t = torch.nn.functional.normalize(G_t)
    G_diff = G_t - G_s
    # loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
    loss = F.mse_loss(G_s, G_t)
    return loss

class SiSdr(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # Input shape: [B, T]
        eps = torch.finfo(input.dtype).eps
        t = input.shape[-1]
        target = target.reshape(-1, t)
        input = input.reshape(-1, t)
        # Einsum for batch vector dot product
        Rss = torch.einsum("bi,bi->b", target, target).unsqueeze(-1)
        a = torch.einsum("bi,bi->b", target, input).add(eps).unsqueeze(-1) / Rss.add(eps)
        e_true = a * target
        e_res = input - e_true
        Sss = e_true.square()
        Snn = e_res.square()
        # Only reduce over each sample. Supposed to be used when used as a metric.
        Sss = Sss.sum(-1)
        Snn = Snn.sum(-1)
        return 10 * torch.log10(Sss.add(eps) / Snn.add(eps))

class SegSdrLoss(nn.Module):
    def __init__(self, window_sizes, factor: float = 0.2, overlap: float = 0):
        # Window size in samples
        super().__init__()
        self.window_sizes = window_sizes
        self.factor = factor
        self.hop = 1 - overlap
        self.sdr = SiSdr()

    def forward(self, input, target) :
        # Input shape: [B, T]
        if self.factor == 0:
            return torch.zeros((), device=input.device)
        loss = torch.zeros((), device=input.device)
        for ws in self.window_sizes:
            if ws > input.size(-1):

                ws = input.size(1)
            loss += self.sdr(
                input=input.unfold(-1, ws, int(self.hop * ws)).reshape(-1, ws),
                target=target.unfold(-1, ws, int(self.hop * ws)).reshape(-1, ws),
            ).mean()
        return -loss * self.factor


def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def phase_losses(clean_D, enh_D):

    clean_real = clean_D[:, :, :, 0]
    clean_imag = clean_D[:, :, :, 1]

    enh_real = enh_D[:, :, :, 0]
    enh_imag = enh_D[:, :, :, 1]

    phase_r = torch.atan2(
        clean_imag + 1e-8,
        clean_real
    )

    phase_r = phase_r.permute(0, 2, 1)

    phase_g = torch.atan2(
        enh_imag + 1e-8,
        enh_real
    )

    phase_g = phase_g.permute(0, 2, 1)

    dim_freq = 512 // 2 + 1
    dim_time = phase_r.shape[2]

    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) - torch.eye(dim_freq)).to(phase_g.device)
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - torch.triu(torch.ones(dim_time, dim_time), diagonal=2) - torch.eye(dim_time)).to(phase_g.device)
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    ip_loss = torch.mean(anti_wrapping_function(phase_r-phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r-gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r-iaf_g))

    return ip_loss, gd_loss, iaf_loss


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

        augments = []
        ### add Remix
        # augments.append(augment.Remix())
        ### add BandMask
        # augments.append(augment.BandMask(0.2, sample_rate=16_000))
        ### add random shift
        # augments.append(augment.Shift(8192, same=True))
        ### add reverb
        augments.append(
            augment.RevEcho())

        self.augment = torch.nn.Sequential(*augments)

        self.loss_func_PMSQE = SingleSrcPMSQE().to('cuda')
        self.pmsqe_stft = Encoder(STFTFB(kernel_size=512, n_filters=512, stride=256)).to('cuda')

        self.mrstft_loss = MultiResolutionSTFTLoss_Newwin().to(self.device)

        self.APC_criterion = APC_SNR_multi_filter(model_hop=128, model_winlen=512, mag_bins=256, theta=0.01,
                                         hops=[8, 16, 32, 64]).to(self.device)
        self.STOI_loss = NegSTOILoss(sample_rate=16000).to(self.device)

        # self.APC_criterion_single = APC_SNR(model_hop=128, model_winlen=512, theta=0.01, mag_bins=256).to(self.device)

        # self.wav2vec_cp = torch.load('D:\\wav2vec_large.pt')
        # self.wav2vec_model = Wav2VecModel.build_model(self.wav2vec_cp['args'], task=None)
        # self.wav2vec_model.load_state_dict(self.wav2vec_cp['model'])
        # self.wav2vec_model.to('cuda')
        # self.wav2vec_model.eval()

        # self.device1 = "cuda"
        #
        # self.weights_path = "F:\\JMCheng\\codes_new\\CGRNN_FB_64_16k\\model\\MOSNet\\mosnet16_torch.pt"
        #
        # # Initialize model and load weights:
        # self.mos_model = MOSNet(device=self.device1)
        # self.mos_model.load_state_dict(torch.load(self.weights_path))
        # self.mos_model.to(self.device1)
        # self.mos_model.eval()
        self.SISDR = SegSdrLoss(window_sizes=[512], factor=0.1, overlap=0.625)

        self.wass_dist = SamplesLoss()
        # self.loss_wrapper = CompoundedPerceptualLoss(PRETRAINED_MODEL_PATH = "F:\\JMCheng\\codes\\DPCRN_16k\\model\\three_denoiser\\pretrained_models", alpha = 1, model_architecture='wav2vec2', distance_metric= 'kld', sample_rate = 16000).to("cuda")

        ########### Distilling the Knowledge
        # tea_checkpoints_path = "F:\\JMCheng\\real_time_exp\\16k_exp_new\\DPRNN_cmp_newwindow_PMSQE_PFPL_APCSNR\\checkpoints\\model_0035.pth"
        #
        # tea_model_checkpoint = torch.load(tea_checkpoints_path, map_location=self.device)
        # self.teacher_model = DPCRN_Model()
        # self.teacher_model.load_state_dict(tea_model_checkpoint)
        # self.teacher_model.to(self.device)
        # self.teacher_model.eval()
        # features_config = FeaturesConfig()
        # self.ft_model = load_lexical_model(features_config.feature_model,
        #                               features_config.state_dict_path,
        #                               device="cuda", sr=16000)

        self.kernel_parameters = {'teacher': 'cosine', 'student': 'cosine', 'loss': 'jeffreys'}
        # self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=512,
        #                                                      hop_length=256, n_mels=129, window_fn=generate_window, center=False).to("cuda")

        # self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=512,
        #                                                      hop_length=256, n_mels=129, center=False).to("cuda")
        self.pase = wf_builder('cfg/frontend/PASE+.cfg').eval()
        self.pase.load_pretrained('./FE_e199.ckpt', load_last=True, verbose=True)
        self.pase.to('cuda')

        # self.asr_model = AutoModel(model="paraformer-zh")



    def _train_epoch(self, epoch):
        loss_total = 0.0
        loss_stoi_total = 0.0
        loss_pmsqe_total = 0.0
        loss_pmsqe_total = 0.0
        loss_phase_total = 0.0
        loss_wav2vec_total = 0.0
        loss_pase_total = 0.0
        loss_guided_total = 0.0
        loss_APC_total = 0.0
        loss_ftrs_total = 0.0
        loss_sisnr_total = 0.0
        loss_asr_enc_total = 0.0
        loss_mrstft_total = 0.0


        step = 0

        for mixture, clean in tqdm(self.train_dataloader, desc="Training"):
            # self.model.train()
            # teacher_model.eval()
            self.optimizer.zero_grad()

            mixture = mixture.to(self.device)
            # mixture_vad = mixture_vad.to(self.device) * 2

            clean = clean.to(self.device)

            mixture = mixture.unsqueeze(1)
            clean = clean.unsqueeze(1)

            sources = torch.stack([mixture - clean, clean])
            sources = self.augment(sources)
            noise, clean = sources
            mixture = noise + clean

            mixture = mixture.squeeze(1)
            clean = clean.squeeze(1)


            # mel_features = self.mel_transform(mixture)
            #
            # mel_features = mel_features.permute(0, 2, 1).unsqueeze(1)



            mixture_D  = self.stft.transform(mixture)
            mixture_real = mixture_D[:, :, :, 0]
            mixture_imag = mixture_D[:, :, :, 1]

            spec_complex = torch.stack([mixture_real, mixture_imag], 1)

            # est_wav, r_rnn_in_1, i_rnn_in_1 = self.model(mixture_real, mixture_imag)
            # est_real, est_imag = self.model(spec_complex, mel_features)
            est_real, est_imag = self.model(spec_complex)

            enhanced_D = torch.stack([est_real, est_imag], 3)
            enhanced = self.stft.inverse(enhanced_D)



            if clean.shape[1] > enhanced.shape[1]:
                clean = clean[:, :enhanced.shape[1]]

            # enhanced = est_wav
            cleaned = clean
            # clean_D = self.stft.transform(cleaned)
            #
            # ip_err, gd_err, iaf_err = phase_losses(clean_D, enhanced_D)
            #
            # phase_loss = ip_err + gd_err + iaf_err

            # vaded_enhanced = mixture_vad * enhanced
            # vaded_cleaned = mixture_vad * cleaned



            ##### PMSQE_loss
            # loss_pmsqe = self.loss_function(enhanced, cleaned)

            # ref_spec = transforms.mag(self.pmsqe_stft(cleaned)).permute(0, 2, 1)
            # est_spec = transforms.mag(self.pmsqe_stft(enhanced)).permute(0, 2, 1)
            #
            # loss_pmsqe = torch.mean(self.loss_func_PMSQE(est_spec, ref_spec))


            # ######PASE_loss
            # pase_in_c = cleaned
            # pase_in_c = pase_in_c.unsqueeze(1)
            # clean_pase =pase(pase_in_c)
            # clean_pase = clean_pase.reshape(clean_pase.size()[0], -1)
            # pase_in_e = enhanced
            # pase_in_e = pase_in_e.unsqueeze(1)
            # enh_pase =pase(pase_in_e)
            # enh_pase = enh_pase.reshape(enh_pase.size()[0], -1)
            # loss_pase = F.mse_loss(clean_pase, enh_pase)

            ######ASR_loss
            # enc_clean = self.asr_model.gen_encoder_fea(cleaned)
            # enc_enhanced = self.asr_model.gen_encoder_fea(enhanced)
            # loss_asr_enc = 0
            # for frame in range(enc_clean.shape[1]):
            #     loss_asr_enc += F.mse_loss(enc_clean[:, frame, :], enc_enhanced[:, frame, :])


            ######PASE_loss
            pase_in_c = cleaned
            pase_in_c = pase_in_c.unsqueeze(1)
            clean_pase =self.pase(pase_in_c)
            clean_pase = clean_pase.reshape(clean_pase.size()[0], -1)
            pase_in_e = enhanced
            pase_in_e = pase_in_e.unsqueeze(1)
            enh_pase =self.pase(pase_in_e)
            enh_pase = enh_pase.reshape(enh_pase.size()[0], -1)
            loss_pase = F.mse_loss(clean_pase, enh_pase)

            # with torch.no_grad():
            #
            #     ftrs_x = self.mos_model.getFtrMaps(enhanced)
            #     ftrs_ref = self.mos_model.getFtrMaps(cleaned)
            #
            #     # avg_mos_score, mos_score = self.mos_model(enhanced)
            # ftrs_loss = 0.0
            # for i in range(len(ftrs_x)):
            #     ftrs_loss += F.l1_loss(ftrs_x[i], ftrs_ref[i]).mean()
            # ftrs_loss /= len(ftrs_x)

            # with torch.no_grad():
            #     pase_in_c = cleaned
            #     pase_in_c = pase_in_c.unsqueeze(1)
            #     clean_pase = pase(pase_in_c)
            #
            #     pase_in_e = enhanced
            #     pase_in_e = pase_in_e.unsqueeze(1)
            #     enh_pase = pase(pase_in_e)
            #
            #     clean_pase = clean_pase.permute(0, 2, 1).contiguous()
            #     enh_pase = enh_pase.permute(0, 2, 1).contiguous()
            # loss_pase = self.wass_dist(enh_pase, clean_pase).mean()


            ######wav2vec Enc loss

            # with torch.no_grad():
            #     enc_clean = self.wav2vec_model.feature_extractor(cleaned)
            #     enc_enhanced = self.wav2vec_model.feature_extractor(enhanced)
            # enc_clean = enc_clean.permute(0, 2, 1).contiguous()
            # enc_enhanced = enc_enhanced.permute(0, 2, 1).contiguous()
            # loss_wav2vec = self.wass_dist(enc_enhanced, enc_clean).mean()

            # ###### Hubert loss
            # with torch.no_grad():
            #     enc_clean = self.ft_model.extract_feats(cleaned)
            #     enc_enhanced = self.ft_model.extract_feats(enhanced)
            # # enc_clean = enc_clean.permute(0, 2, 1).contiguous()
            # # enc_enhanced = enc_enhanced.permute(0, 2, 1).contiguous()
            # # loss_wav2vec = self.wass_dist(enc_enhanced, enc_clean).mean()
            # loss_hubert = F.mse_loss(enc_enhanced, enc_clean)

            # loss_sisdr = self.SISDR(enhanced, cleaned)


            # sc_loss, mag_loss = self.mrstft_loss(enhanced, cleaned)
            # loss_mrstft = sc_loss + mag_loss
            #
            loss_APC_SNR, loss_pmsqe = self.APC_criterion(enhanced + 1e-8, cleaned + 1e-8)

            # loss_stoi = self.STOI_loss(enhanced+ 1e-8, cleaned+ 1e-8).mean()

            # loss_APC_SNR, loss_pmsqe = self.APC_criterion_single(enhanced, cleaned, mixture_vad)

            # loss_PSM = F.mse_loss(irm_mask, est_mask)
            # loss_CRM = F.mse_loss(clean_real, est_real) + F.mse_loss(clean_imag, est_imag)

            # loss = sc_loss + mag_loss + 0.25*loss_pmsqe + 0.25*loss_pase + loss_PKT_batch_real + loss_PKT_batch_imag
            # loss = sc_loss + mag_loss + 0.25*loss_pmsqe + 0.25*loss_pase + loss_MKMMD_batch_real + loss_MKMMD_batch_imag
            # loss = sc_loss + mag_loss + 0.5*loss_pmsqe + 0.5*loss_pase
            # loss = sc_loss + mag_loss + 0.5 * loss_pmsqe
            # loss = 0.05 * loss_APC_SNR + loss_pmsqe + loss_pase + ip_err + gd_err + iaf_err
            # loss = sc_loss + mag_loss + 0.05 * loss_APC_SNR + loss_pmsqe + loss_pase
            # loss = 0.1 * loss_APC_SNR + loss_pmsqe + loss_pase + loss_mrstft
            loss = 0.1 * loss_APC_SNR + loss_pmsqe + loss_pase

            # loss = loss_sisdr + 0.5 * loss_pmsqe + loss_pase
            # loss = 0.05 * loss_APC_SNR + loss_pmsqe + ftrs_loss
            # loss = 0.05 * loss_APC_SNR + loss_pmsqe
            # loss = 0.05 * loss_APC_SNR + loss_pmsqe + loss_hubert
            # loss = 0.05 * loss_APC_SNR + loss_pmsqe + loss_wav2vec
            # loss = 0.05 * loss_APC_SNR + loss_pmsqe + loss_pase
            # loss = 0.1 * loss_APC_SNR + loss_pmsqe + loss_wav2vec
            # loss = sc_loss + mag_loss + loss_pmsqe + loss_wav2vec

            # loss_source = sc_loss + mag_loss + 0.25 * loss_pmsqe
            # loss_guided = review_kd_loss

            # loss_guided_mid = 0.5 * loss_PKT_batch_real + 0.5 * loss_PKT_batch_imag
            # loss_guided = loss_PKT_batch_real + loss_PKT_batch_imag

            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), 10)

            if step %1000 ==0:
                # print(
                #     'Train Step: {} \t Source loss: {:.6f} \t KD loss: {:.6f}'.format(
                #         step, loss_source_temp.item(), loss_guided_temp.item()))
                # print(loss)
                # print(
                #     'Train Step: {} \t APC_SNR loss: {:.6f} \t PMSQE loss: {:.6f} \t Wav2Vec loss: {:.6f}'.format(
                #         step, loss_APC_SNR.item(), loss_pmsqe.item(), loss_hubert.item()))
                # print(
                #     'Train Step: {} \t APC_SNR loss: {:.6f} \t PMSQE loss: {:.6f} \t PASE loss: {:.6f} \t KD loss: {:.6f}'.format(
                #         step, loss_APC_SNR.item(), loss_pmsqe.item(), loss_pase.item(), kd_loss_mid.item()))
                # print(
                #     'Train Step: {} \t \t PMSQE loss: {:.6f} \t PASE loss: {:.6f} \t SISDR loss: {:.6f}'.format(
                #         step, loss_pmsqe.item(), loss_pase.item(), loss_sisdr.item()))
                print(
                    'Train Step: {} \t APC_SNR loss: {:.6f} \t PMSQE loss: {:.6f} \t PASE loss: {:.6f} '.format(
                        step, loss_APC_SNR.item(), loss_pmsqe.item(), loss_pase.item()))

                # print(
                #     'Train Step: {} \t APC_SNR loss: {:.6f} \t PMSQE loss: {:.6f} \t PASE loss: {:.6f} \t MRSTFT loss: {:.6f}'.format(
                #         step, loss_APC_SNR.item(), loss_pmsqe.item(), loss_pase.item(), loss_mrstft.item()))
                # print(
                #     'Train Step: {} \t APC_SNR loss: {:.6f} \t PMSQE loss: {:.6f} \t ASR ENC loss: {:.6f} '.format(
                #         step, loss_APC_SNR.item(), loss_pmsqe.item(), loss_asr_enc.item()))
                # print(
                #     'Train Step: {} \t APC_SNR loss: {:.6f} \t PMSQE loss: {:.6f} \t STOI loss: {:.6f} '.format(
                #         step, loss_APC_SNR.item(), loss_pmsqe.item(), loss_stoi.item()))




            step += 1



            self.optimizer.step()

            loss_total += float(loss)
            # loss_sisnr_total += float(loss_sisdr)
            loss_pase_total += float(loss_pase)
            # loss_asr_enc_total += float(loss_asr_enc)
            loss_pmsqe_total += float(loss_pmsqe)
            # loss_mrstft_total += float(loss_mrstft)
            # loss_phase_total += float(phase_loss)
            # loss_APC_total += float(loss_APC_SNR)
            # loss_ftrs_total += float(ftrs_loss)
            # loss_wav2vec_total += float(loss_hubert)
            # loss_guided_total += float(kd_loss_mid)
            # loss_guided_total += float(loss_MKMMD_batch)

            # loss_stoi_total += float(loss_stoi)
        self.scheduler.step()
        dataloader_len = len(self.train_dataloader)
        print((loss_total / dataloader_len))
        # print((loss_fwSNR_total / dataloader_len))
        # print(epoch, self.scheduler.get_lr()[0])
        # print(loss_pmsqe_total / dataloader_len)
        torch.cuda.empty_cache()
        # print(loss_stoi_total / dataloader_len)
        # print(loss_wsdr_total / dataloader_len)
        self.viz.writer.add_scalar("Loss_TOTAL/Train", loss_total / dataloader_len, epoch)
        self.viz.writer.add_scalar("Loss_PASE/Train", loss_pase_total / dataloader_len, epoch)
        # self.viz.writer.add_scalar("Loss_ASR_enc/Train", loss_asr_enc_total / dataloader_len, epoch)
        # self.viz.writer.add_scalar("Loss_STOI/Train", loss_stoi_total / dataloader_len, epoch)
        self.viz.writer.add_scalar("Loss_PMSQE/Train", loss_pmsqe_total / dataloader_len, epoch)
        # self.viz.writer.add_scalar("Loss_MRSTFT/Train", loss_mrstft_total / dataloader_len, epoch)
        # self.viz.writer.add_scalar("Loss_PHASE/Train", loss_phase_total / dataloader_len, epoch)
        # self.viz.writer.add_scalar("Loss_APC_SNR/Train", loss_APC_total / dataloader_len, epoch)


    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0
        loss_total_cIRM_real = 0.0
        loss_total_cIRM_imag = 0.0
        loss_total_mag = 0.0
        mixture_mean = None
        mixture_std = None
        loss_total_smooth_l1 = 0.0
        sisnr_c_d = []
        sisnr_c_n = []
        stoi_c_n = []
        stoi_c_d = []
        pesq_c_n = []
        pesq_c_d = []


        for mixture, clean, n_frames_list, names in tqdm(self.validation_dataloader):
            self.model.eval()

            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            # clean_stft = stft_val.transform(clean)
            # clean = stft_val.inverse(clean_stft)

            # est_wav, _, _ = self.model(mixture)

            mixture_D = self.stft.transform(mixture)
            mixture_real = mixture_D[:, :, :, 0]
            mixture_imag = mixture_D[:, :, :, 1]

            # mel_features = self.mel_transform(mixture)
            # mel_features = mel_features.permute(0, 2, 1).unsqueeze(1)

            spec_complex = torch.stack([mixture_real, mixture_imag], 1)

            # est_wav, r_rnn_in_1, i_rnn_in_1 = self.model(mixture_real, mixture_imag)
            with torch.no_grad():
                est_real, est_imag = self.model(spec_complex)

            enhanced_D = torch.stack([est_real, est_imag], 3)
            enhanced = self.stft.inverse(enhanced_D)

            if enhanced.shape[1] != clean.shape[1]:
                clean = clean[:, :enhanced.shape[1]]

            if enhanced.shape[1] != mixture.shape[1]:
                mixture = mixture[:, :enhanced.shape[1]]

            loss_total_smooth_l1 += F.smooth_l1_loss(enhanced, clean)



            len_list = []

            for n_frames in n_frames_list:

                len_list.append((n_frames-1) * 256 + 512)

            enhanced_speeches = enhanced.detach().cpu().numpy()
            mixture_speeches = mixture.detach().cpu().numpy()
            clean_speeches = clean.detach().cpu().numpy()

            for i in range(len(n_frames_list)):
                enhanced = enhanced_speeches[i][:len_list[i]]
                mixture = mixture_speeches[i][:len_list[i]]
                clean = clean_speeches[i][:len_list[i]]


                stoi_c_n.append(compute_STOI(clean, mixture, sr=24000))
                stoi_c_d.append(compute_STOI(clean, enhanced, sr=24000))
                pesq_c_n.append(compute_PESQ(clean, mixture, sr=24000))
                pesq_c_d.append(compute_PESQ(clean, enhanced, sr=24000))
                sisnr_c_d.append(si_snr(enhanced, clean))
                sisnr_c_n.append(si_snr(mixture, clean))

                # clean_dir = "H:\\JMCheng\\CRN_16k_new\\PHA-CRN\\temp_val_wavs\\" + "clean_" + names[i] + ".wav"
                # noisy_dir = "H:\\JMCheng\\CRN_16k_new\\PHA-CRN\\temp_val_wavs\\" + "noisy_" + names[i] + ".wav"
                # enh_dir = "H:\\JMCheng\\CRN_16k_new\\PHA-CRN\\temp_val_wavs\\" + "enh_" + names[i] + ".wav"
                #
                # sf.write(clean_dir, clean, 16000)
                # sf.write(noisy_dir,  mixture, 16000)
                # sf.write(enh_dir, enhanced, 16000)

        # ################## visualize real recording wavs
        # real_record_path = "D:\\JMCheng\\RT_16k\\16k_exp\\real_recording_16k_new"
        # speech_names = []
        # for dirpath, dirnames, filenames in os.walk(real_record_path):
        #     for filename in filenames:
        #         if filename.lower().endswith(".wav"):
        #             speech_names.append(os.path.join(dirpath, filename))
        #
        # for speech_na in speech_names:
        #     fig, ax = plt.subplots(4, 1)
        #     name = os.path.basename(speech_na)
        #     print(name)
        #     mixture, sr = sf.read(speech_na, dtype="float32")
        #     real_len = len(mixture)
        #     assert sr == 16000
        #     mixture = torch.from_numpy(mixture).to(self.device)
        #     mixture = mixture.unsqueeze(0)
        #
        #
        #
        #     mixture_D  = self.stft.transform(mixture)
        #     mixture_real = mixture_D[:, :, :, 0]
        #     mixture_imag = mixture_D[:, :, :, 1]
        #
        #     spec_complex = torch.stack([mixture_real, mixture_imag], 1)
        #
        #
        #     with torch.no_grad():
        #         est_real, est_imag = self.model(spec_complex)
        #
        #     enhanced_D = torch.stack([est_real, est_imag], 3)
        #     enhanced = self.stft.inverse(enhanced_D)
        #
        #     enhanced = enhanced.detach().cpu().squeeze().numpy()
        #     mixture = mixture.detach().cpu().squeeze().numpy()
        #
        #     librosa.display.waveshow(mixture, sr=16000, label='noisy_waveform', ax=ax[0])
        #     librosa.display.waveshow(enhanced, sr=16000, label='enhanced_waveform', ax=ax[1])
        #
        #     mag_noisy, _ = librosa.magphase(librosa.stft(mixture, n_fft=512, hop_length=256, win_length=512))
        #     librosa.display.specshow(librosa.amplitude_to_db(mag_noisy), cmap="magma", y_axis="linear", ax=ax[2], sr=16000, label='noisy_spectrogram')
        #
        #     mag_enh, _ = librosa.magphase(librosa.stft(enhanced, n_fft=512, hop_length=256, win_length=512))
        #     librosa.display.specshow(librosa.amplitude_to_db(mag_enh), cmap="magma", y_axis="linear", ax=ax[3], sr=16000, label='enhanced_spectrogram')
        #
        #     plt.tight_layout()
        #     self.viz.writer.add_figure(f"Waveform/{epoch}/{name}", fig, epoch)
        #     temp_enh_dir = "D:\\JMCheng\\RT_16k\\16k_exp\\PHA-CRN\\temp_test_inference\\" + "epoch_" + str(epoch)
        #     temp_enh_path = temp_enh_dir + "\\" + name
        #     create_folder(temp_enh_dir)
        #     sf.write(temp_enh_path, enhanced, 16000)
        #     # self.viz.writer.add_audio(f"Audio/{epoch}/{name}_Mixture", mixture, epoch, sample_rate=16000)
        #     # self.viz.writer.add_audio(f"Audio/{epoch}/{name}_Enhanced", enhanced, epoch, sample_rate=16000)
        #
        #     del mixture, mixture_D, mixture_real, mixture_imag, spec_complex, est_real, est_imag, enhanced_D
        #     torch.cuda.empty_cache()


        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        self.viz.writer.add_scalars(f"Metrics/STOI", {
            "clean_and_noisy": get_metrics_ave(stoi_c_n),
            "clean_and_denoisy": get_metrics_ave(stoi_c_d)
        }, epoch)

        self.viz.writer.add_scalars(f"Metrics/PESQ", {
            "clean_and_noisy": get_metrics_ave(pesq_c_n),
            "clean_and_denoisy": get_metrics_ave(pesq_c_d)
        }, epoch)

        self.viz.writer.add_scalars(f"Metrics/smooth_mse_loss", {
            "smooth_mse_loss": loss_total_smooth_l1 / len(self.validation_dataloader),
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

