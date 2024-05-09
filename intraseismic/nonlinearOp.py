import numpy as np
import torch
import torch.nn as nn


def convolutional_modelling(ai, wav, wavcenter):
    """Convolutional modelling

    1D Nonlinear Convolutional modelling

    Parameters
    ----------
    ai : :obj:`np.ndarray`
        Acoustic impedance profile
    wav : :obj:`np.ndarray`
        Wavelet
    wavcenter : :obj:`int`
        Index of wavelet center

    Returns
    -------
    seismic : :obj:`np.ndarray`
        Seismic trace
    reflectivity : :obj:`np.ndarray`
        Reflectivity trace

    """
    # Normal incidence reflectivity
    reflectivity = (ai[1:] - ai[:-1]) / (ai[1:] + ai[:-1])
    reflectivity[np.isnan(reflectivity)] = 0.

    # Convolve with wavelet, adjust length of trace to start from wavelet peak (wavcenter)
    seismic = np.convolve(reflectivity, wav, 'full')
    seismic = seismic[wavcenter:-wavcenter+1]

    return seismic, reflectivity


def convolutional_modelling_torch(ailog, wav):
    """Convolutional modelling with torch

    1D Nonlinear Convolutional modelling with torch
    (currently assumes zero-phase wavelet, wavcenter=len(wav)//2)

    Parameters
    ----------
    ailog : :obj:`np.ndarray`
        Natural logarithmic of Acoustic impedance profile
    wav : :obj:`np.ndarray`
        Wavelet

    Returns
    -------
    seismic : :obj:`np.ndarray`
        Seismic trace
    reflectivity : :obj:`np.ndarray`
        Reflectivity trace

    """
    # Compute exponential
    ai = torch.exp(ailog)

    # Normal incidence reflectivity
    reflectivity = (ai[1:] - ai[:-1]) / (ai[1:] + ai[:-1])

    # Convolve with wavelet, adjust length of trace to start from wavelet peak (wavcenter)
    # seismic = torch.nn.functional.conv1d(reflectivity.T.unsqueeze(1), # .T raises warning
    seismic = torch.nn.functional.conv1d(reflectivity.T.unsqueeze(1),
                                         wav.unsqueeze(0).unsqueeze(1), padding='same')
    seismic = torch.nn.functional.pad(seismic, (0, 1, 0, 0, 0, 0), mode='constant')
    #seismic = seismic.squeeze().T # .T raises warning
    seismic = seismic.squeeze()
    if seismic.ndim > 1:
        seismic = seismic.T
    return seismic, reflectivity


class ConvMod(nn.Module):
    """Convolutional modelling with torch

    Parameters
    ----------
    input_channels : :obj:`int`
        Number of input channels
    output_channels : :obj:`int`
        Number of output channels

    """
    def __init__(self, wav, nt):
        super(ConvMod, self).__init__()
        self.wav = wav
        self.nt = nt

    def forward(self, x):
        x = x.reshape(self.nt, -1)
        x = convolutional_modelling_torch(x, self.wav)[0]
        return x


