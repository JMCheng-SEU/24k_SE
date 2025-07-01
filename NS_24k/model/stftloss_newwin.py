# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Original copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch
import torch.nn.functional as F
from utils.pre_stft import STFT
import numpy as np
from torch.autograd import Variable

def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss_1024(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=512):
        """Initialize STFT loss module."""
        super(STFTLoss_1024, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size

        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

        self.STFT = STFT_1024(
            filter_length=1024,
            hop_length=512
        ).to('cuda')

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """

        x_D = self.STFT.transform(x)
        x_real = x_D[:, :, :, 0]
        x_imag = x_D[:, :, :, 1]
        x_mag = torch.sqrt(x_real ** 2 + x_imag ** 2 + 1e-8)  # [batch, T, F]

        y_D = self.STFT.transform(y)
        y_real = y_D[:, :, :, 0]
        y_imag = y_D[:, :, :, 1]
        y_mag = torch.sqrt(y_real ** 2 + y_imag ** 2 + 1e-8)  # [batch, T, F]

        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class STFTLoss_512(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=512, shift_size=256):
        """Initialize STFT loss module."""
        super(STFTLoss_512, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size

        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

        self.STFT = STFT_512(
            filter_length=512,
            hop_length=256
        ).to('cuda')

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """

        x_D = self.STFT.transform(x)
        x_real = x_D[:, :, :, 0]
        x_imag = x_D[:, :, :, 1]
        x_mag = torch.sqrt(x_real ** 2 + x_imag ** 2 + 1e-8)  # [batch, T, F]

        y_D = self.STFT.transform(y)
        y_real = y_D[:, :, :, 0]
        y_imag = y_D[:, :, :, 1]
        y_mag = torch.sqrt(y_real ** 2 + y_imag ** 2 + 1e-8)  # [batch, T, F]

        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss

class STFTLoss_256(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=256, shift_size=160):
        """Initialize STFT loss module."""
        super(STFTLoss_256, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size

        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

        self.STFT = STFT_256(
            filter_length=256,
            hop_length=160
        ).to('cuda')

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """

        x_D = self.STFT.transform(x)
        x_real = x_D[:, :, :, 0]
        x_imag = x_D[:, :, :, 1]
        x_mag = torch.sqrt(x_real ** 2 + x_imag ** 2 + 1e-8)  # [batch, T, F]

        y_D = self.STFT.transform(y)
        y_real = y_D[:, :, :, 0]
        y_imag = y_D[:, :, :, 1]
        y_mag = torch.sqrt(y_real ** 2 + y_imag ** 2 + 1e-8)  # [batch, T, F]

        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class STFT_512(torch.nn.Module):
    def __init__(self, filter_length=512, hop_length=256):
        super(STFT_512, self).__init__()

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.forward_transform = None

        # filename = 'H:\\JMCheng\\CGRNN_16k_new_exp\\window256w512.txt'
        # new_window = np.zeros(512)
        # count = 0
        #
        # with open(filename, 'r') as file_to_read:
        #     line = file_to_read.readline()
        #     for i in line.split(','):
        #         try:
        #             new_window[count] = float(i)
        #             count += 1
        #         except:
        #             flag = 0


        # scale = self.filter_length / self.hop_length
        # scale = 1
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        window = np.sqrt(np.hanning(filter_length)).astype(np.float32)
        # window = new_window
        fourier_basis = fourier_basis * window
        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        # forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])

        inv_fourier_basis = np.fft.fft(np.eye(self.filter_length))
        inv_fourier_basis = inv_fourier_basis / (window + 1e-8)
        cutoff = int((self.filter_length / 2 + 1))
        inv_fourier_basis = np.vstack([np.real(inv_fourier_basis[:cutoff, :]),
                                       np.imag(inv_fourier_basis[:cutoff, :])])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(inv_fourier_basis).T[:, None, :])

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        ##
        #
        input_data = input_data.view(num_batches, 1, num_samples)
        forward_transform = F.conv1d(input_data,
                                     Variable(self.forward_basis, requires_grad=False),
                                     stride=self.hop_length,
                                     padding=0)
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        # magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        # phase = torch.autograd.Variable(torch.atan2(imag_part.load_data, real_part.load_data))
        res = torch.stack([real_part, imag_part], dim=-1)
        res = res.permute([0, 2, 1, 3])
        return res

    # (B,T,F,2)
    def inverse(self, stft_res):
        real_part = stft_res[:, :, :, 0].permute(0, 2, 1)
        imag_part = stft_res[:, :, :, 1].permute(0, 2, 1)
        recombine_magnitude_phase = torch.cat([real_part, imag_part], dim=1)
        # recombine_magnitude_phase = stft_res.permute([0, 2, 3, 1]).contiguous().view([1, -1, stft_res.size()[]])

        inverse_transform = F.conv_transpose1d(recombine_magnitude_phase,
                                               Variable(self.inverse_basis, requires_grad=False),
                                               stride=self.hop_length,
                                               padding=0)
        return inverse_transform[:, 0, :]

    def forward(self, input_data):
        stft_res = self.transform(input_data)
        reconstruction = self.inverse(stft_res)
        return reconstruction


class STFT_1024(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=512):
        super(STFT_1024, self).__init__()

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.forward_transform = None

        # filename = 'H:\\JMCheng\\CGRNN_16k_new_exp\\window512w1024.txt'
        # new_window = np.zeros(1024)
        # count = 0
        #
        # with open(filename, 'r') as file_to_read:
        #     line = file_to_read.readline()
        #     for i in line.split(','):
        #         try:
        #             new_window[count] = float(i)
        #             count += 1
        #         except:
        #             flag = 0


        # scale = self.filter_length / self.hop_length
        # scale = 1
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        window = np.sqrt(np.hanning(filter_length)).astype(np.float32)
        # window = new_window
        fourier_basis = fourier_basis * window
        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        # forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])

        inv_fourier_basis = np.fft.fft(np.eye(self.filter_length))
        inv_fourier_basis = inv_fourier_basis / (window + 1e-8)
        cutoff = int((self.filter_length / 2 + 1))
        inv_fourier_basis = np.vstack([np.real(inv_fourier_basis[:cutoff, :]),
                                       np.imag(inv_fourier_basis[:cutoff, :])])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(inv_fourier_basis).T[:, None, :])

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        ##
        #
        input_data = input_data.view(num_batches, 1, num_samples)
        forward_transform = F.conv1d(input_data,
                                     Variable(self.forward_basis, requires_grad=False),
                                     stride=self.hop_length,
                                     padding=0)
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        # magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        # phase = torch.autograd.Variable(torch.atan2(imag_part.load_data, real_part.load_data))
        res = torch.stack([real_part, imag_part], dim=-1)
        res = res.permute([0, 2, 1, 3])
        return res

    # (B,T,F,2)
    def inverse(self, stft_res):
        real_part = stft_res[:, :, :, 0].permute(0, 2, 1)
        imag_part = stft_res[:, :, :, 1].permute(0, 2, 1)
        recombine_magnitude_phase = torch.cat([real_part, imag_part], dim=1)
        # recombine_magnitude_phase = stft_res.permute([0, 2, 3, 1]).contiguous().view([1, -1, stft_res.size()[]])

        inverse_transform = F.conv_transpose1d(recombine_magnitude_phase,
                                               Variable(self.inverse_basis, requires_grad=False),
                                               stride=self.hop_length,
                                               padding=0)
        return inverse_transform[:, 0, :]

    def forward(self, input_data):
        stft_res = self.transform(input_data)
        reconstruction = self.inverse(stft_res)
        return reconstruction


class STFT_256(torch.nn.Module):
    def __init__(self, filter_length=256, hop_length=160):
        super(STFT_256, self).__init__()

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.forward_transform = None

        # filename = 'H:\\JMCheng\\CGRNN_16k_new_exp\\window256w160.txt'
        # new_window = np.zeros(256)
        # count = 0
        #
        # with open(filename, 'r') as file_to_read:
        #     line = file_to_read.readline()
        #     for i in line.split(','):
        #         try:
        #             new_window[count] = float(i)
        #             count += 1
        #         except:
        #             flag = 0


        # scale = self.filter_length / self.hop_length
        # scale = 1
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        window = np.sqrt(np.hanning(filter_length)).astype(np.float32)
        # window = new_window
        fourier_basis = fourier_basis * window
        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        # forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])

        inv_fourier_basis = np.fft.fft(np.eye(self.filter_length))
        inv_fourier_basis = inv_fourier_basis / (window + 1e-8)
        cutoff = int((self.filter_length / 2 + 1))
        inv_fourier_basis = np.vstack([np.real(inv_fourier_basis[:cutoff, :]),
                                       np.imag(inv_fourier_basis[:cutoff, :])])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(inv_fourier_basis).T[:, None, :])

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        ##
        #
        input_data = input_data.view(num_batches, 1, num_samples)
        forward_transform = F.conv1d(input_data,
                                     Variable(self.forward_basis, requires_grad=False),
                                     stride=self.hop_length,
                                     padding=0)
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        # magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        # phase = torch.autograd.Variable(torch.atan2(imag_part.load_data, real_part.load_data))
        res = torch.stack([real_part, imag_part], dim=-1)
        res = res.permute([0, 2, 1, 3])
        return res

    # (B,T,F,2)
    def inverse(self, stft_res):
        real_part = stft_res[:, :, :, 0].permute(0, 2, 1)
        imag_part = stft_res[:, :, :, 1].permute(0, 2, 1)
        recombine_magnitude_phase = torch.cat([real_part, imag_part], dim=1)
        # recombine_magnitude_phase = stft_res.permute([0, 2, 3, 1]).contiguous().view([1, -1, stft_res.size()[]])

        inverse_transform = F.conv_transpose1d(recombine_magnitude_phase,
                                               Variable(self.inverse_basis, requires_grad=False),
                                               stride=self.hop_length,
                                               padding=0)
        return inverse_transform[:, 0, :]

    def forward(self, input_data):
        stft_res = self.transform(input_data)
        reconstruction = self.inverse(stft_res)
        return reconstruction


class MultiResolutionSTFTLoss_Newwin(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 512, 256],
                 hop_sizes=[512, 256, 160],
                 factor_sc=0.5, factor_mag=0.5):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super(MultiResolutionSTFTLoss_Newwin, self).__init__()
        assert len(fft_sizes) == len(hop_sizes)
        self.stft_losses = torch.nn.ModuleList()

        self.stft_losses += [STFTLoss_1024(fft_size=1024, shift_size=512)]
        self.stft_losses += [STFTLoss_512(fft_size=512, shift_size=256)]
        self.stft_losses += [STFTLoss_256(fft_size=256, shift_size=128)]

        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_sc*sc_loss, self.factor_mag*mag_loss
