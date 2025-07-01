from scipy.signal import resample,stft
import scipy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pre_stft import *

def hz_to_bark(freqs_hz):
    freqs_hz = np.asanyarray([freqs_hz])
    barks = (26.81*freqs_hz)/(1960+freqs_hz)-0.53
    barks[barks<2]=barks[barks<2]+0.15*(2-barks[barks<2])
    barks[barks>20.1]=barks[barks>20.1]+0.22*(barks[barks>20.1]-20.1)
    return np.squeeze(barks)

def bark_to_hz(barks):
    barks = barks.copy()
    barks = np.asanyarray([barks])
    barks[barks<2]=(barks[barks<2]-0.3)/0.85
    barks[barks>20.1]=(barks[barks>20.1]+4.422)/1.22
    freqs_hz = 1960 * (barks+0.53)/(26.28-barks)
    return np.squeeze(freqs_hz)

def bark_frequencies(n_barks=128, fmin=0.0, fmax=11025.0):
    # 'Center freqs' of bark bands - uniformly spaced between limits
    min_bark = hz_to_bark(fmin)
    max_bark = hz_to_bark(fmax)

    barks = np.linspace(min_bark, max_bark, n_barks)

    return bark_to_hz(barks)

def barks(fs, n_fft, n_barks=128, fmin=0.0, fmax=None, norm='area', dtype=np.float32):

    if fmax is None:
        fmax = float(fs) / 2


    # Initialize the weights
    n_barks = int(n_barks)
    weights = np.zeros((n_barks, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = np.linspace(0,float(fs) / 2,int(1 + n_fft//2), endpoint=True)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    bark_f = bark_frequencies(n_barks + 2, fmin=fmin, fmax=fmax)

    fdiff = np.diff(bark_f)
    ramps = np.subtract.outer(bark_f, fftfreqs)

    for i in range(n_barks):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm in (1, 'area'):
        weightsPerBand=np.sum(weights, 1)
        for i in range(weights.shape[0]):
            weights[i,:]=weights[i,:]/weightsPerBand[i]
    return weights


def bsd_loss(clean_speech, processed_speech):

    STFT_block = STFT(filter_length=512, hop_length=256).to('cuda')
    # clean_speech_torch = torch.from_numpy(np.array(clean_speech, dtype=np.float32)).to('cuda')
    # processed_speech_torch = torch.from_numpy(np.array(processed_speech, dtype=np.float32)).to('cuda')
    # clean_speech_torch = clean_speech_torch.unsqueeze(0)
    # processed_speech_torch = processed_speech_torch.unsqueeze(0)

    Zxx_clean_torch = STFT_block.transform(clean_speech)
    clean_mag = torch.sqrt(Zxx_clean_torch[:, :, :, 0] ** 2 + Zxx_clean_torch[:, :, :, 1] ** 2)
    Zxx_processed_torch = STFT_block.transform(processed_speech)
    processed_mag = torch.sqrt(Zxx_processed_torch[:, :, :, 0] ** 2 + Zxx_processed_torch[:, :, :, 1] ** 2)

    window = torch.from_numpy(np.sqrt(np.hamming(512)).astype(np.float32)).to('cuda')

    clean_power_spec_torch = torch.square(clean_mag)
    processed_power_spec_torch = torch.square(processed_mag)


    bark_filt = barks(16000, 512, n_barks=32)


    bark_filt_torch = torch.from_numpy(np.array(bark_filt, dtype=np.float32)).to('cuda')
    bark_filt_torch = bark_filt_torch.permute(1, 0)

    clean_power_spec_bark_torch = torch.matmul(clean_power_spec_torch, bark_filt_torch)
    processed_power_spec_bark_torch = torch.matmul(processed_power_spec_torch, bark_filt_torch)

    # clean_power_spec_bark_2_torch = torch.square(clean_power_spec_bark_torch)
    # diff_power_spec_2_torch = torch.square(clean_power_spec_bark_torch - processed_power_spec_bark_torch)

    # batch_bsd_torch = torch.sum(diff_power_spec_2_torch, dim=2) / torch.sum(clean_power_spec_bark_2_torch, dim=2)
    # #### avoid nan data
    # zero = torch.zeros_like(batch_bsd_torch)
    # batch_bsd_torch = torch.where(torch.isnan(batch_bsd_torch), zero, batch_bsd_torch)

    bsd_torch = F.mse_loss(clean_power_spec_bark_torch, processed_power_spec_bark_torch)


    return bsd_torch