import torch
import numpy as np
from jaxtyping import Float, Int
from torch import Tensor
import scipy.signal.windows as signal
import torch.nn.functional as F

try:
    import neo
    from quantities import ms, s, Hz
    from elephant.kernels import GaussianKernel
    from elephant.statistics import instantaneous_rate
except ImportError:
    pass


def anscombe_transform(x):
    return 2 * torch.sqrt(x + 0.375)
        

def smooth_firing_rate(
    spike_trains: Float[Tensor, "num_trials num_timepoints num_neurons"],
    sampling_precision=20,
    kern_sd_ms=50, 
):
    if not isinstance(spike_trains, torch.Tensor):
        spike_trains = torch.Tensor(spike_trains)
    
    kern_sd = int(round(kern_sd_ms / sampling_precision))
    window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
    window /= np.sum(window)
    filt = lambda x: np.convolve(x, window, 'same')
        
    return torch.tensor(np.apply_along_axis(filt, 1, spike_trains), dtype=spike_trains.dtype)