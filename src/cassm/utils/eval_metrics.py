import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import PoissonRegressor, Ridge


def mse(true, pred):
    """
    # input: true, pred are T x N arrays
    # return N x 1 array
    """
    T = true.shape[0]
    return (1 / T) * np.sum((true - pred) ** 2, axis=0)

def rsquared(true, pred):
    nneurons = true.shape[1]
    r2 = np.zeros(nneurons)
    for i in range(nneurons):
        val = r2_score(true[:, i], pred[:, i])
        if val >= 0:
            r2[i] = val
        else:
            r2[i] = 0
    return r2

def r_2(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    denominator = np.sum((true - pred) ** 2, axis=0)

    y_bar = np.mean(true, axis=0)
    numerator = np.sum((true - y_bar) ** 2, axis=0)
    return 1 - (denominator / numerator)

def fit_rectlin(train_input, eval_input, train_output, alpha=0.0):
    # Fit linear regression
    lr = Ridge(alpha=alpha)
    lr.fit(train_input, train_output)
    train_pred = lr.predict(train_input)
    eval_pred = lr.predict(eval_input)
    # Rectify to prevent negative or 0 rate predictions
    train_pred[train_pred < 1e-10] = 1e-10
    eval_pred[eval_pred < 1e-10] = 1e-10
    return train_pred, eval_pred

def fit_poisson(train_input, eval_input, train_output, alpha=0.0):
    train_pred = []
    eval_pred = []
    # train Poisson GLM for each output column
    for chan in range(train_output.shape[1]):
        pr = PoissonRegressor(alpha=alpha, max_iter=500)
        pr.fit(train_input, train_output[:, chan])
        train_pred.append(pr.predict(train_input))
        eval_pred.append(pr.predict(eval_input))
    train_pred = np.vstack(train_pred).T
    eval_pred = np.vstack(eval_pred).T
    return train_pred, eval_pred

def zebra_eval(eval_pred, eval_true):
    """
    Args:
        eval_pred: np.ndarray, shape (n_trials, n_time, n_neurons)
        eval_true: np.ndarray, same shape

    Returns:
        trial_mse:  np.ndarray of shape (n_trials,)
        neuron_r2: np.ndarray of shape (n_neurons,)
                   R^2 per neuron, aggregated over all trials and time.
    """
    assert eval_pred.shape == eval_true.shape, "Pred and true arrays must match in shape"
    n_trials, n_time, n_neurons = eval_true.shape

    # ----- MSE per trial -----
    trial_mse = np.mean((eval_pred - eval_true) ** 2, axis=(1, 2))  # (n_trials,)

    # ----- R^2 per neuron (aggregated over trials × time) -----
    # Flatten trials and time into one axis: (n_trials * n_time, n_neurons)
    true_flat = eval_true.reshape(-1, n_neurons)
    pred_flat = eval_pred.reshape(-1, n_neurons)

    ss_res = np.sum((true_flat - pred_flat) ** 2, axis=0)  # (n_neurons,)
    true_mean = np.mean(true_flat, axis=0, keepdims=True)  # (1, n_neurons)
    ss_tot = np.sum((true_flat - true_mean) ** 2, axis=0)  # (n_neurons,)

    neuron_r2 = 1.0 - ss_res / (ss_tot + 1e-12)  # (n_neurons,)

    return trial_mse, neuron_r2

