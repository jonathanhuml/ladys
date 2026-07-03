import matplotlib.pyplot as plt
import numpy as np
import torch
import torcheval.metrics
from cassm.utils import eval_metrics

def plot_heldout_preds_best(eval_rates_heldout, eval_rates_heldout_true,
                       neuron_idx: int = 0, n_trials: int = 6):
    """
    Compute MSE for each trial (for a specific neuron), select the best `n_trials`,
    and plot predictions vs truth for those trials.
    """

    assert eval_rates_heldout.shape == eval_rates_heldout_true.shape, \
        "Predicted and true arrays must have the same shape"

    # Convert to numpy
    pred_np = np.asarray(eval_rates_heldout)
    true_np = np.asarray(eval_rates_heldout_true)

    num_trials, T, _ = pred_np.shape

    # ---- Compute MSE per trial for the selected neuron ----
    # Shape: (num_trials,)
    trial_mse = np.mean((pred_np[:, :, neuron_idx] - true_np[:, :, neuron_idx])**2, axis=1)

    # ---- Select the top n_trials (lowest error = best) ----
    n_trials = min(n_trials, num_trials)
    best_idx = np.argsort(trial_mse)[:n_trials]   # sorted indices

    # ---- Plotting ----
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True, sharey=True)
    axes = axes.ravel()

    for plot_i, trial_i in enumerate(best_idx):
        ax = axes[plot_i]
        pred = pred_np[trial_i, :, neuron_idx]
        truth = true_np[trial_i, :, neuron_idx]

        ax.plot(pred, lw=2, label="Pred", alpha=0.9)
        ax.plot(truth, lw=2, ls="--", label="Truth", alpha=0.9)

        ax.set_title(f"Trial {trial_i} (MSE={trial_mse[trial_i]:.4f})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Rate")
        ax.grid(True, alpha=0.3)

        if plot_i == 0:
            ax.legend(frameon=False)

    # Hide unused axes
    for j in range(n_trials, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Held-out neuron {neuron_idx}: Top {n_trials} trials by MSE")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_heldout_preds(eval_rates_heldout, eval_rates_heldout_true,
                       neuron_idx: int = 0, n_trials: int = 6):
    """
    Plot predictions vs. true rates for the first `n_trials` across all timepoints
    of a single neuron (default: the first) from tensors/arrays shaped
    (trials, timepoints, neurons).
    """
    # Safety checks
    assert eval_rates_heldout.shape == eval_rates_heldout_true.shape, \
        "Predicted and true arrays must have the same shape"
    T = eval_rates_heldout.shape[1]
    n_trials = min(n_trials, eval_rates_heldout.shape[0])

    # Nic(er) default styling
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True, sharey=True)
    axes = axes.ravel()

    for i in range(n_trials):
        ax = axes[i]
        pred = np.asarray(eval_rates_heldout[i, :, neuron_idx])
        truth = np.asarray(eval_rates_heldout_true[i, :, neuron_idx])

        ax.plot(pred, lw=2, label="Pred", alpha=0.9)
        ax.plot(truth, lw=2, ls="--", label="Truth", alpha=0.9)
        ax.set_title(f"Trial {i}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Rate")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(frameon=False)

    # Hide any unused axes (in case n_trials < 6)
    for j in range(n_trials, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Held-out neuron {neuron_idx}: Predictions vs Truth (first {n_trials} trials)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



def plot_traces(T, dt, pred, true, figsize=(8, 8), num_traces=6,
                ncols=2, error_type=None, mode=None, var=None, data=None):
    '''
    Plot fitted intensity function and compare to ground truth

    Arguments:
        - pred (np.array): array of predicted values to plot (dims: num_steps x num_cells)
        - true (np.array)   : array of true values to plot (dims: num_steps x num_cells)
        - error type : either "mse" or "rsquared"
        - figsize (2-tuple) : figure size (width, height) in inches (default = (8, 8))
        - num_traces (int)  : number of traces to plot (default = 12)
        - ncols (int)       : number of columns in figure (default = 2)
        - mode (string)     : mode to select subset of traces. Options: 'activity', 'rand', None.
                              'Activity' plots the num_traces/2 most active traces and num_traces/2
                              least active traces defined sorted by mean value in trace

    '''

    num_cells = pred.shape[-1]

    nrows = int(num_traces / ncols)
    fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
    axs = np.ravel(axs)

    if mode == 'rand':
        idxs = np.random.choice(list(range(num_cells)), size=num_traces, replace=False)
        idxs.sort()

    elif mode == 'activity':
        idxs = true.max(axis=0).argsort()[-num_traces:]

    else:
        idxs = list(range(num_cells))

    time = np.arange(0, T * dt, dt)

    error = torch.nn.functional.mse_loss(pred, true, reduction='none')  # mse(true, pred)
    error = torch.sum(error, dim=0)

    zmin = min(pred[:, idxs].min(), true[:, idxs].min())
    zmax = max(pred[:, idxs].max(), true[:, idxs].max())

    for ii, (ax, idx) in enumerate(zip(axs, idxs)):
        plt.sca(ax)
        if var is not None:
            ax.plot(time, pred[:, idx] + 2 * np.sqrt(var.T[:, idx]), color='#37A1D0', linestyle='dashed')
            ax.plot(time, pred[:, idx] - 2 * np.sqrt(var.T[:, idx]), color='#37A1D0', linestyle='dashed')
        ax.plot(time, true[:, idx], lw=2, color='#E84924', label="True")
        ax.plot(time, pred[:, idx], lw=2, color='#37A1D0', label="Pred")

        if data is not None:
            data_n = np.array(data[:, :, idx])
            t_array = np.arange(data_n.shape[1]) / T
            t_idx_matrix = np.tile(t_array, (data_n.shape[0], 1))
            ax.scatter(np.ravel(t_idx_matrix), np.ravel(data_n), alpha=0.15, s=7, color='orange')
        if idx == 0:
            ax.legend()
        else:
            pass
        if error_type == "rsquared":
            r2 = torcheval.metrics.functional.r2_score(pred, true, multioutput="raw_values")
            ax.annotate('R^2: ' + str(round(r2[idx].item(), 2)), xy=(0.05, 0.95), xycoords='axes fraction')
        else:
            # r2 = torcheval.metrics.functional.r2_score(pred, true, multioutput="raw_values")
            r2a = eval_metrics.rsquared(true=true, pred=pred)
            error_mse = round(error[idx].item(), 2)
            round_r2a = round(r2a[idx].item(), 2)
            ax.annotate('MSE:' + str(error_mse) + '| R2: ' + str(round_r2a), xy=(0.05, 0.95), xycoords='axes fraction')
        plt.ylim(zmin - (zmax - zmin) * 0.35, zmax + (zmax - zmin) * 0.35)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if ii >= num_traces - ncols:
            plt.xlabel('time (s)', fontsize=14)
            plt.xticks(fontsize=12)
            ax.xaxis.set_ticks_position('bottom')

        else:
            plt.xticks([])
            ax.xaxis.set_ticks_position('none')
            ax.spines['bottom'].set_visible(False)

        if ii % ncols == 0:
            plt.yticks(fontsize=12)
            ax.yaxis.set_ticks_position('left')
        else:
            plt.yticks([])
            ax.yaxis.set_ticks_position('none')
            ax.spines['left'].set_visible(False)

    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    return fig
