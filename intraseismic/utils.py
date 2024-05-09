import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Tuple, List, Union
import random
import numpy as np
import torch
import torch.nn as nn
import pylops

def create_block_diagonal(A, N):
    """
    Parameters:
    - A: The input matrix.
    - N: The number of times A appears on the block diagonal of the resulting matrix B.

    Returns:
    - B: The resulting block diagonal matrix.
    """
    # Initialize the list with the first occurrence of A
    matrices = [A]
    # Add A to the list N-1 more times
    for _ in range(N-1):
        matrices.append(A)
    # Use block_diag to create the block diagonal matrix B
    B = pylops.BlockDiag(matrices)
    return B


def batch_tv(data):
    """
    Calculate the total variation (TV) score for a batch of images or volumes.
    It supports 3D data, interpreted as a batch of 2D images (batch size, height, width), and 4D data,
    interpreted as a batch of 3D volumes (batch size, depth, height, width).

    Parameters
    ----------
    data : torch.Tensor or np.ndarray
        The input data for which to compute the total variation score. 

    Returns
    -------
    int or float
        The total variation score for the batch. This score is the sum of the absolute differences
        for all adjacent pixel pairs in 2D images, or adjacent voxel pairs in 3D volumes,
        across the entire batch.
    """
    
    dims = data.shape
    score = 0
    if len(dims) == 3:
        for i in range(data.shape[0]):
            vol = data[i]       
            diff1 = vol[1:] - vol[:-1]
            diff2 = vol[:, 1:] - vol[:, :-1]
            res1 = diff1.abs().sum()
            res2 = diff2.abs().sum()
            score += res1 + res2 

    elif len(dims) == 4:
        for i in range(data.shape[0]):
            vol = data[i]        
            diff1 = vol[1:, :, :] - vol[:-1, :, :]
            diff2 = vol[:, 1:, :] - vol[:, :-1, :]
            diff3 = vol[:, :, 1:] - vol[:, :, :-1]
            res1 = diff1.abs().sum()
            res2 = diff2.abs().sum()
            res3 = diff3.abs().sum()
            score += res1 + res2 + res3 

    else:
        raise ValueError("Unsupported data dimensionality. Expected 3 or 4 dimensions, got {}.".format(data.dim()))
        
    return score


def set_seed(seed):
    """Set all random seeds to a fixed value and take out any
    randomness from cuda kernels

    Parameters
    ----------
    seed : :obj:`int`
        Seed number

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True

def SNR(x, xest):
    if isinstance(x, np.ndarray) and isinstance(xest, np.ndarray):
        norm_func = np.linalg.norm
        log_func = np.log10
    elif torch.is_tensor(x) and torch.is_tensor(xest):
        norm_func = torch.norm
        log_func = torch.log10
    else:
        raise ValueError("Inputs must be both NumPy arrays or both PyTorch tensors")
    
    return 20 * log_func(norm_func(x) / norm_func(x - xest))

def RRE(x, xinv):
    if isinstance(x, np.ndarray) and isinstance(xinv, np.ndarray):
        norm_func = np.linalg.norm
    elif torch.is_tensor(x) and torch.is_tensor(xinv):
        norm_func = torch.norm
    else:
        raise ValueError("Inputs must be both NumPy arrays or both PyTorch tensors")
    
    return norm_func(x - xinv) / norm_func(x)


def plotmodel(m, vmin, vmax, ref=None, dt=1, i=None, dims=(5,4)): 
#     plt.style.use('dark_background')
    fig = plt.figure(figsize=dims)
    gs = gridspec.GridSpec(3, 2, width_ratios=(1, .05), height_ratios=(1., 1., 1.),
                       left=0.1, right=0.9, bottom=0.1, top=0.9,
                       wspace=0.05, hspace=0.05)
    ax0 = fig.add_subplot(gs[:, 0])
    base = ax0.imshow(m, vmin=vmin, vmax=vmax, cmap='terrain', extent=[0, m.shape[1], m.shape[0] * dt, 0])
    ax0.set_ylabel('TWT $[s]$')
    if ref is not None:
        ax0.set_title('SNR = %.2f' % SNR(ref,m)+'  RRE = %.4f' % RRE(ref,m))        
    ax0.axis('tight');
    ax1 = fig.add_subplot(gs[2, 1])
    ax1.set_title('Impedance \n $[m/s*g/cm^3]$', loc='left')
    Colorbar(ax=ax1, mappable=base)
#     plt.show()
    
def plotdata(d, vmin=-0.2, vmax=0.2,  dt=1, dims=(5,4)):  
#     plt.style.use('dark_background')
    fig = plt.figure(figsize=dims)
    gs = gridspec.GridSpec(3, 2, width_ratios=(1, .05), height_ratios=(1., 1., 1.),
                       left=0.1, right=0.9, bottom=0.1, top=0.9,
                       wspace=0.05, hspace=0.05)
    ax0 = fig.add_subplot(gs[:, 0])
    base = ax0.imshow(d, vmin=vmin, vmax=vmax, cmap='seismic_r', extent=[0, d.shape[1], d.shape[0] * dt, 0])
    ax0.set_ylabel('TWT $[s]$')
#     ax0.set_title('SNR = %.2f' % SNR(ref,m))
    ax0.axis('tight');
    ax1 = fig.add_subplot(gs[2, 1])
    ax1.set_title('Amplitude', loc='left')
    Colorbar(ax=ax1, mappable=base)
#     plt.show()

def plotresults(m, mback, vmin, vmax, i, m_true=None,  m_well=None, dt=1, dims=(8, 5), type='synthetic', idx=200, xlim=(2000, 11200), cmap='terrain', line=True, width_ratios=[2,1]):
#     plt.style.use('dark_background')
    fig = plt.figure(figsize=dims)
    gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios,
                       left=0.1, right=0.9, bottom=0.1, top=0.9,
                       wspace=0.05, hspace=0.05)
    ax0 = fig.add_subplot(gs[:, 0])
    base = ax0.imshow(m, vmin=vmin, vmax=vmax, cmap=cmap, extent=[0, m.shape[1], m.shape[0] * dt, 0])
    if line:
        ax0.axvline(idx, 0, m.shape[0], color='r')
    ax0.set_ylabel('TWT $[s]$')
    ax0.axis('tight')
    ax0.set_title('iter = %d' % i)        
    ax1 = fig.add_subplot(gs[:, 1])
    if type=='synthetic':
        ax0.set_title('iter = %d' % i +'  SNR = %.2f' % SNR(np.exp(m_true),m))# +'  RRE = %.4f' % RRE(m_true,m))
        ax1.plot(np.exp(m_true)[:,idx],np.arange(0, m.shape[0]), 'k', label='Truth')
        ax1.plot(np.exp(mback)[:,idx],np.arange(0, m.shape[0]), 'gray', linestyle='--', label='Background')
#         ax1.plot(mtv[:,200], np.arange(0, m.shape[0]),label='model_TV')
        ax1.plot(m[:,idx], np.arange(0, m.shape[0]), 'r', label='Predicted')
        
    if type=='field':
        if m_well is not None:
            ax1.plot(m_well,np.arange(0, m.shape[0]), 'k', label='Well-log')
        ax1.plot(np.exp(mback)[:,idx],np.arange(0, m.shape[0]), label='model_back')
        ax1.plot(m[:,idx], np.arange(0, m.shape[0]),label='model_pred')
    ax1.legend()
    ax1.set_ylim(m.shape[0], 0)
    ax1.set_xlim(xlim[0], xlim[1])
    ax1.yaxis.set_ticklabels([])
    


def clim(in_content: np.ndarray, ratio: float = 95) -> Tuple[float, float]:
    c = np.percentile(np.absolute(in_content), ratio)
    return -c, c


def explode_volume(volume: np.ndarray, t: int = None, x: int = None, y: int = None, vmin: float = None, vmax: float = None,
                   figsize: tuple = (8, 8), cmap: str = 'bone', clipval: tuple = None, p: int = 98,
                   tlim: tuple = None, xlim: tuple = None, ylim: tuple = None, 
                   labels : list = ('[s]', '[km]', '[km]'),
                   tlabel : str = 't',
                   ratio: tuple = None, linespec: dict = None, title: str = '',
                   filename: str or Path = None, save_opts: dict = None,
                   whspace: tuple = None) -> plt.figure:
    if linespec is None:
        linespec = dict(ls='-', lw=1, color='orange')
    nt, nx, ny = volume.shape
    t_label, x_label, y_label = labels
    
    t = t if t is not None else nt//2
    x = x if x is not None else nx//2
    y = y if y is not None else ny//2

    if tlim is None:
        t_label = "samples"
        tlim = (0, volume.shape[0])
    if xlim is None:
        x_label = "samples"
        xlim = (0, volume.shape[1])
    if ylim is None:
        y_label = "samples"
        ylim = (0, volume.shape[2])
    
    # vertical lines for coordinates reference
    tline = (tlim[1] - tlim[0]) / nt * t + tlim[0]
    xline = (xlim[1] - xlim[0]) / nx * x + xlim[0]
    yline = (ylim[1] - ylim[0]) / ny * y + ylim[0]
    
    # instantiate plots
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.95)
    if ratio is None:
        wr = (nx, ny)
        hr = (nt, nx)
    else:
        wr = ratio[0]
        hr = ratio[1]

    if whspace is None:
        whspace = (0., 0.)

    opts = dict(cmap=cmap,vmin=vmin, vmax=vmax, clim=clipval if clipval is not None else clim(volume, p), aspect='equal')
    gs = fig.add_gridspec(2, 2, width_ratios=wr, height_ratios=hr,
                          left=0.1, right=0.9, bottom=0.1, top=1.1,
                          wspace=whspace[0], hspace=whspace[1])
    # gs.set(aspect=1)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_ = fig.add_subplot(gs[0, 1]) # upper right empty space
    ax_.axis('off') # upper right empty space

    # central plot
    ax.imshow(volume[:, :, y], extent=[xlim[0], xlim[1], tlim[1], tlim[0]], **opts)
    ax.axvline(x=xline, **linespec)
    ax.axhline(y=tline, **linespec)

    # top plot
    ax_top.imshow(volume[t].T, extent=[xlim[0], xlim[1], ylim[1], ylim[0]], **opts)
    ax_top.axvline(x=xline, **linespec)
    ax_top.axhline(y=yline, **linespec)
    ax_top.invert_yaxis()
    
    # right plot
    ax_right.imshow(volume[:, x], extent=[ylim[0], ylim[1], tlim[1], tlim[0]], **opts)
    ax_right.axvline(x=yline, **linespec)
    ax_right.axhline(y=tline, **linespec)
    
    # labels
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_right.tick_params(axis="y", labelleft=False)
    ax.set_xlabel("x " + x_label)
    ax.set_ylabel(tlabel + " " + t_label)
    ax_right.set_xlabel("y " + y_label)
    ax_top.set_ylabel("y " + y_label)
    
    
    if filename is not None:
        if save_opts is None:
            save_opts = {'format': 'png', 'dpi': 150, 'bbox_inches': 'tight'}
        plt.savefig(f"{filename}.{save_opts['format']}", **save_opts)
    return fig, (ax, ax_top, ax_right)

## For pre-stack data
def plotprestackmodel(model, vmins, vmaxs, mtrue=1, gt=False):
    cmap = 'terrain'    
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(3, 8, width_ratios=(1, .05, 0.15, 1, .05, .15, 1,.05), height_ratios=(0.3,1,0.3),
                           left=0.1, right=0.9, bottom=0.1, top=0.9,
                           wspace=0.05, hspace=0.05)

    ax0 = fig.add_subplot(gs[:,0])
    ax1 = fig.add_subplot(gs[1,1])
    ax2 = fig.add_subplot(gs[:,3], sharey=ax0)
    ax2.tick_params(axis="y", labelleft=False)
    ax3 = fig.add_subplot(gs[1,4])
    ax4 = fig.add_subplot(gs[:,6])
    ax4.tick_params(axis="y", labelleft=False)
    ax5 = fig.add_subplot(gs[1,7]) 
    ax0_ = ax0.imshow(model[:,0], vmin=vmins[0], vmax=vmaxs[0], cmap=cmap)
    ax0.set_ylabel('TWT $samples$')
    ax0.set_title(f'a) $V_p$ ')
    ax0.axis('tight');
    ax2_ = ax2.imshow(model[:,1], vmin=vmins[1], vmax=vmaxs[1], cmap=cmap)
    ax2.set_title(f'b) $V_s$ ')
    ax2.axis('tight');
    ax4_ = ax4.imshow(model[:,2], vmin=vmins[2], vmax=vmaxs[2], cmap=cmap)
    ax4.set_title(r'c) $\rho$ ')
    ax4.axis('tight');
    ax1.set_title(r'$[m/s]$', loc='left')
    ax3.set_title(r'$[m/s]$', loc='left')
    ax5.set_title(r'$[kg/m^3]$', loc='left')
    Colorbar(ax=ax1, mappable=ax0_)
    Colorbar(ax=ax3, mappable=ax2_)
    Colorbar(ax=ax5, mappable=ax4_)

    if gt:
        ax0.set_title(f'a) $V_p$  | SNR: {SNR(mtrue[:,0], model[:,0]):.2f}')
        ax2.set_title(f'b) $V_s$  | SNR: {SNR(mtrue[:,1], model[:,1]):.2f}')
        ax4.set_title(r'c) $\rho$' +  f'| SNR: {SNR(mtrue[:,2], model[:,2]):.2f}')

def plotprestackdata(data):
    cmap = 'RdGy'    
    vmin, vmax = -0.1, 0.1 
    fig = plt.figure(figsize=(11, 4))
    gs = gridspec.GridSpec(3, 4, width_ratios=(1, 1, 1, .05), height_ratios=(0.3,1,0.3),
                           left=0.1, right=0.9, bottom=0.1, top=0.9,
                           wspace=0.1, hspace=0.05)

    ax0 = fig.add_subplot(gs[:,0])
    ax1 = fig.add_subplot(gs[:,1])
    ax1.tick_params(axis="y", labelleft=False)
    ax2 = fig.add_subplot(gs[:,2])
    ax2.tick_params(axis="y", labelleft=False)
    ax3 = fig.add_subplot(gs[1,3])
    ax0_ = ax0.imshow(data[:,0], vmin=vmin, vmax=vmax, cmap=cmap)
    ax0.set_ylabel('TWT $samples$')
    ax0.set_title(r'a) $\theta = 0$')
    ax0.axis('tight');
    ax1_ = ax1.imshow(data[:,10], vmin=vmin, vmax=vmax, cmap=cmap)
    ax1.set_title(r'b) $\theta = 20$')
    ax1.axis('tight');
    ax2_ = ax2.imshow(data[:,20], vmin=vmin, vmax=vmax, cmap=cmap)
    ax2.set_title(r'c) $\theta = 40$')
    ax2.axis('tight');
    ax3.set_title(r'$Amplitude$', loc='left')
    Colorbar(ax=ax3, mappable=ax2_)


def plotstd(m, mset, m_true, mback, vmin, vmax, i,  dt=1, line=200, dims=(7, 5), 
            type='synthetic', idx=200, xlim=(4500, 12500), cmap='terrain', width_ratios=[1, 0.4, 0.05]):
#     plt.style.use('dark_background')
    fig = plt.figure(figsize=dims)
    gs = gridspec.GridSpec(2, 3, width_ratios=width_ratios, height_ratios=[1,0.6],
                       left=0.1, right=0.9, bottom=0.1, top=0.9,
                       wspace=0.05, hspace=0.05)
    ax0 = fig.add_subplot(gs[:, 0])
    base = ax0.imshow(m, vmin=vmin, vmax=vmax, cmap=cmap, extent=[0, m.shape[1], m.shape[0] * dt, 0])
    ax0.axvline(line, 0, m.shape[0], color='r')
    ax0.set_ylabel('TWT $[s]$')
    ax0.axis('tight')
    ax0.set_title('a) std/mean')        
    ax1 = fig.add_subplot(gs[:, 1])
    ax1.set_title('b) Realizations')
    mean = np.mean(mset, axis=0)
    # ax2 = fig.add_subplot(gs[1, 2])
    if type=='synthetic':
        ax1.plot(mean[100:,idx],np.arange(0, 120), 'b', label='mean')
        for i in range(mset.shape[0]):
            ax1.plot(mset[i][100:,idx], np.arange(0, 120), 'gray', linewidth=0.5)
        ax1.plot(mean[100:,idx],np.arange(0, 120), 'b', label='mean')
        
    if type=='field':
        ax1.plot(np.exp(mback)[:,idx],np.arange(0, m.shape[0]), label='model_back')
        ax1.plot(m[:,idx], np.arange(0, m.shape[0]),label='model_pred')
    
    ax1.legend(['mean'],loc='upper left')
    ax1.set_ylim(100, 0)
    ax1.set_xlim(xlim[0], xlim[1])
    ax1.yaxis.set_ticklabels([])