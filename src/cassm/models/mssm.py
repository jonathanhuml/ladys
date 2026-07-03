import os
import torch
import time
import numpy as np
import torch.nn as nn
import geotorch
import matplotlib.pyplot as plt
from cassm.losses.log_marginal_likelihood import log_MLL
from cassm.utils.plotting import plot_traces
from cassm.utils import training_utils, eval_metrics
from linear_operator.operators import IdentityLinearOperator, KroneckerProductLinearOperator
from typing import Optional
import gpytorch
from typing import Union, List

def matern_time_noise(delta_t: torch.Tensor,
                      sigma_f2: torch.Tensor,
                      ell: torch.Tensor) -> torch.Tensor:
    lam    = torch.sqrt(torch.tensor(3.0, device=delta_t.device)) / ell      
    rho2   = torch.exp(-2.0 * lam * delta_t)                                 
    u      = lam * delta_t                                                   

    q11 = 3.0 * sigma_f2 * (1.0 - rho2 * (1.0 + 2.0 * u + 2.0 * u**2))
    q22 = 3.0 * sigma_f2 * lam**2 * (1.0 - rho2 * (1.0 - 2.0 * u + 2.0 * u**2))
    q12 = 6.0 * sigma_f2 * lam**3 * delta_t**2 * rho2

    return torch.stack(
        [torch.stack([q11, q12], -1),
         torch.stack([q12, q22], -1)],
        dim=-2
    )

def transition_matrix_time(delta_t: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:

    lam = torch.sqrt(torch.tensor(3.0, device=delta_t.device)) / ell
    zero = torch.zeros_like(lam)
    one = torch.ones_like(lam)
    F_t = torch.stack(
        [torch.stack([zero, one]), torch.stack([-(lam**2), -2.0 * lam])],
    ).squeeze(-1)
    return torch.matrix_exp(F_t * delta_t)

class KalmanFilterSmoother(nn.Module):
    def __init__(
        self,
        nneurons: int,
        timesteps: int,
        device: torch.device,
        dt: float = 1.0,
        dataset_name: Optional[str] = None,
        save_model: bool = False,
    ) -> None:
        super().__init__()
        # super(KalmanFilterSmoother, self).__init__()

        self.dim = nneurons
        self.latent_dim = self.dim  # spatial latent dimensionality
        self.state_dim = 2 * self.dim
        self.t = timesteps            # number of time steps
        self.device = device
        self.save_model = save_model
        self.dt = torch.tensor(float(dt), device=device)

        self.raw_sigma_f = nn.Parameter(1e-1 * torch.ones(1, device=device))
        self.raw_ell = nn.Parameter(1e-1 * torch.ones(1, device=device))
        self.softplus = nn.Softplus()

        # latent spatial inducing locations ---------------------------
        self.latent_locations = nn.Parameter(
            torch.arange(self.dim, device=device).float().unsqueeze(-1)
        )
        self.spatial_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )
        
        
        self.obs_noise_values = nn.Parameter(1e-1 * torch.ones(self.dim, device=self.device))

        if self.save_model:
            assert dataset_name is not None
            save_id = training_utils.random_run(6)
            run_name = "run_id_" + str(save_id)
            self.save_path = "./kalman_runs/" + dataset_name + "/" + str(run_name) + "/"
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)

    def build_matern_observation_matrix(self) -> torch.Tensor:
        W = torch.eye(self.dim, device=self.device)
        h_time = torch.tensor([[1.0, 0.0]], device=self.device)
        return torch.kron(W, h_time)

    def spatial_cov(self) -> torch.Tensor:
        return self.spatial_kernel(self.latent_locations).to_dense()

    def filter(self, data, return_type="for_prediction", holdout=False):
        num_trials = data.shape[0]
        T = data.shape[1]

        updated_belief_state_means = torch.zeros(size=(num_trials, T, 
                                    self.state_dim), device=self.device)
        updated_belief_state_covs = torch.zeros(size=(1, T, 
                                    self.state_dim, self.state_dim), device=self.device)

        prior_belief_state_mean = torch.zeros(size=(num_trials, 
                                                    self.state_dim, 1), 
                                                    device=self.device)
        prior_belief_state_cov  = IdentityLinearOperator(self.state_dim, 
                                                         batch_shape=(1,), device=self.device)
        loss = torch.tensor(0.0, device=self.device)

        m = nn.Softplus()

        ell = m(self.raw_ell)
        sigma_f2 = m(self.raw_sigma_f)
        obs_noise = m(self.obs_noise_values)

        # dynamics ----------------------------------------------------
        A_t = transition_matrix_time(self.dt, ell)
        transition_matrix = torch.kron(torch.eye(self.latent_dim, device=self.device), A_t).unsqueeze(0)

        Q_t = matern_time_noise(self.dt, sigma_f2, ell)
        process_noise = torch.kron(self.spatial_cov(), Q_t)


        if holdout:
            nheldin = data.shape[2]
            truncated = torch.eye(self.dim, device=self.device)[:nheldin, :]
            h_time = torch.tensor([[1.0, 0.0]], device=self.device)
            observation_matrix = torch.kron(truncated, h_time).unsqueeze(0)
            obs_noise = obs_noise[:nheldin]
        else:
            observation_matrix = self.build_matern_observation_matrix().unsqueeze(0)
 

        # ------------------------------------
        # FILTER RECURSION
        for t in range(T):

            innovation_matrix = observation_matrix @ prior_belief_state_cov @ observation_matrix.mT
            innovation_matrix.diagonal(dim1=-2, dim2=-1).add_(obs_noise)

            prior_predictive_residual = data[:, t, :].unsqueeze(-1) - torch.matmul(observation_matrix, prior_belief_state_mean)

            innovation_matrix_cholfac = torch.linalg.cholesky(innovation_matrix, 
                                                              upper=False)
            innovation_inverse_obs_matrix = torch.cholesky_solve(input2=innovation_matrix_cholfac, 
                                           input=observation_matrix, 
                                           upper=False)
            kalman_gain = prior_belief_state_cov @ innovation_inverse_obs_matrix.transpose(1,2)

            # Filtering belief: State estimate given current observation
            updated_belief_state_mean = prior_belief_state_mean + torch.matmul(kalman_gain, 
                                                            prior_predictive_residual)
            jos_gain = kalman_gain @ observation_matrix
            jos_gain.diagonal(dim1=-2, dim2=-1).sub_(1.0)
            updated_belief_state_cov = jos_gain @ prior_belief_state_cov @ jos_gain.transpose(1,2) + kalman_gain * obs_noise @ kalman_gain.transpose(1,2)
            
            # UPDATE FOR t+1
            prior_belief_state_mean = torch.matmul(transition_matrix, updated_belief_state_mean)

            prior_belief_state_cov = transition_matrix @ updated_belief_state_cov @ transition_matrix.transpose(1,2) + process_noise

            # ------------------------------------
            # compute marginal log-likelihood loss for time t
            loss_t = log_MLL(prior_predictive_residual, innovation_matrix_cholfac)

            # increment total loss by loss at time t
            loss += loss_t
            # ------------------------------------

            # assign t^th time point before next pass
            updated_belief_state_means[:, t, :] = updated_belief_state_mean[:, :, 0]
            updated_belief_state_covs[:, t, :, :] = updated_belief_state_cov

        # torch.cuda.synchronize()
        # normalize loss by T and nneurons
        loss = loss / (T * self.dim)

        if return_type == "for_forward":
            return loss
        else: 
            return updated_belief_state_means, updated_belief_state_covs

    def forward(self, data):
        loss = self.filter(data, return_type="for_forward")
        return loss

    def get_obs_noise(self):
        return nn.Softplus(self.obs_noise_values)

    @staticmethod
    def memory_usage(tensor):
        return tensor.element_size() * tensor.nelement()

    # ------------------------------------
    # TRAINING AND TESTING
    def train_loop(self, dataloader, optimizer, clip_value=-1):
        losses = []
        self.train()
        for batch, data in enumerate(dataloader):
            # optimizer.zero_grad()
            loss = self(data)
            loss.backward()
            # default clip value is -1, so we don't clip unless explicitly specified
            if clip_value > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_value)
            optimizer.step()
            losses.append(loss.item())
            self.zero_grad()

        avg_epoch_loss = np.mean(losses)

        return avg_epoch_loss

    def test_loop(self, dataloader):
        self.eval()
        test_loss = 0
        with torch.no_grad():
            i = 0
            for batch_data, batch_truth in dataloader:
                loss = self(batch_data)
                test_loss += loss.item()
                i += 1
        return test_loss / i

    # putting this all together for one simple function
    def train_model(self, epochs, optimizer, train_loader, test_loader=None, valid_truth=None, clip_value=-1):
        print("Beginning training...")
        start_lr = optimizer.param_groups[0]["lr"]
        # -----------------------
        tr_loss = np.zeros(epochs)

        mse_test_loss = np.zeros(epochs)
        nll_test_loss = np.zeros(epochs)
        te_loss = np.zeros(epochs)

        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=25)
        start_time = time.time()

        for e in range(epochs):
            tr_loss[e] = self.train_loop(train_loader, optimizer, clip_value=clip_value)
            scheduler.step()

            if test_loader is not None:
                te_loss[e] = self.test_loop(test_loader)
                with torch.no_grad():
                    # SAVE NLL for model comparison
                    NLLLoss = torch.nn.GaussianNLLLoss()
                    num_pts = len(test_loader.dataset) # test_loader.dataset.shape[0]
                    scale_factor = 1 / torch.tensor(num_pts, device=self.device)
                    pred, pred_noise = self.predict_rate(test_loader.dataset.tensors[0])
                    true_rate_test = torch.mean(test_loader.dataset.tensors[1], dim=0)
                    output = NLLLoss(input=pred, target=true_rate_test.T, var=scale_factor*pred_noise)
                    nll_test_loss[e] = output.item()

                    MSELoss = torch.nn.MSELoss()
                    mse = MSELoss(input=pred, target=true_rate_test.T)
                    mse_test_loss[e] = round(mse.item(), 6)
                    
                print(
                    f"Epoch {e + 1}"
                    + " | Train Loss: "
                    + str(np.round(tr_loss[e], decimals=6))
                    + '| Test NLL: '
                    + str(np.round(nll_test_loss[e], decimals=6))
                    + '| Test MSE: '
                    + str(np.round(mse_test_loss[e], decimals=6))
                )
                print("--------------------------------------------")
            else:
                print(f"Epoch {e + 1} | Train Loss: " + str(np.round(tr_loss[e], decimals=6)))
                print("--------------------------------------------")

        end_time = time.time()
        print("Training complete!")

        if self.save_model:
            hrs = abs(start_time - end_time) / 60 ** 2
            min_per_epoch = abs(start_time - end_time) / (60 * epochs)
            batch_size = train_loader.batch_size
            num_neurons = self.dim
            text = [str(batch_size),
                    str(start_lr),
                    str(np.round(hrs, decimals=4)),
                    str(np.round(min_per_epoch, decimals=4)),
                    str(clip_value), 
                    str(num_neurons)]
            log = ['batch_size',
                   'learning_rate',
                   'train_time_hrs',
                   'min_per_epoch',
                   'clip_value',
                   'neurons']
            with open(self.save_path + 'run_details.txt', 'w') as f:
                for i in range(len(text)):
                    f.write(log[i] + ' : ' + text[i])
                    f.write('\n')

            plt.plot(np.arange(epochs), tr_loss, label="train")
            plt.plot(np.arange(epochs), te_loss, label="test")
            plt.legend()
            plt.title("Training and testing loss_deprecated")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(self.save_path + "training.png")
            plt.close()

            plt.plot(np.arange(epochs), mse_test_loss)
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.title('Kalman MSE on Testing Data')
            plt.savefig(self.save_path + "mse_training.png")
            np.save(file=self.save_path + 'mse_test_loss.npy', arr=mse_test_loss)
            np.savetxt(self.save_path + 'mse_test_loss.txt', mse_test_loss, delimiter=',')
            plt.close()

            plt.plot(np.arange(epochs), nll_test_loss)
            plt.xlabel("Epoch")
            plt.ylabel("NLL")
            plt.title('Kalman NLL on Testing Data')
            plt.savefig(self.save_path + "nll_training.png")
            np.save(file=self.save_path + 'nll_test_loss.npy', arr=nll_test_loss)
            np.savetxt(self.save_path + 'nll_test_loss.txt', nll_test_loss, delimiter=',')
            plt.close()

            torch.save(self.state_dict(), self.save_path + "model.pt")

    # ------------------------------------
    # INFERENCE/NON-TRAINING
    # Algorithms below should be used after we train model to make predictions on test data
    def predict_rate(self, data):

        T = data.shape[1]
        dim = data.shape[2]

        X, P = self.filter(data=data, return_type="for_prediction")

        observation_matrix = self.build_matern_observation_matrix()

        pred = torch.mean(observation_matrix.unsqueeze(0) @ X.transpose(1,2), dim=0).detach()

        # ensure variance is positive real number with nonlinearity
        m = nn.Softplus()
        noise = observation_matrix @ P @ torch.transpose(observation_matrix, dim0=0, dim1=1)
        noise.diagonal(dim1=-2, dim2=-1).add_(m(self.obs_noise_values))
        normal_noise = noise[0]
        # check to make sure normal noise can actually take first matrix of repeats...
        pred_noise = torch.zeros(size=(dim, T), device=self.device)

        for j in range(T):
            s_t = torch.diag(normal_noise[j, :, :])
            pred_noise[:, j] = s_t.detach()

        return pred, pred_noise
    
    def predict_rate_heldout(self, data):
        X, P = self.filter(data=data, return_type="for_prediction", holdout=True)
        observation_matrix = self.build_matern_observation_matrix()

        predictions = observation_matrix @ X.transpose(1,2)

        return predictions.transpose(1,2).detach().cpu()
    
    def filter_per_trial_means(
        self,
        data: torch.Tensor,
        batch_size: int = 1,
        detach: bool = True,
        to_cpu: bool = True,
        return_list_if_variable_T: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        self.eval() 
        device = next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else data.device
        if isinstance(data, list):
            trials = data
            N = len(trials)
            variable_T = True
        else:
            N = data.shape[0]
            trials = [data[i] for i in range(N)]
            variable_T = return_list_if_variable_T

        out_means: List[torch.Tensor] = []

        with torch.no_grad():
            for start in range(0, N, batch_size):
                batch = trials[start:start + batch_size]

                if isinstance(data, list) or variable_T:
                    same_len = all(b.shape[-2] == batch[0].shape[-2] for b in batch)
                    if same_len:
                        x = torch.stack(batch, dim=0)
                        per_item = False
                    else:
                        per_item = True
                else:
                    x = torch.stack(batch, dim=0) 
                    per_item = False

                def run_and_collect(x_):
                    x_ = x_.to(device, non_blocking=True)
                    filt_out = self.filter(x_)          
                    means = filt_out[0] if isinstance(filt_out, (tuple, list)) else filt_out
                    if detach:
                        means = means.detach()
                    if to_cpu:
                        means = means.cpu()
                    return means

                if per_item:
                    for t in batch:
                        m = run_and_collect(t.unsqueeze(0))  
                        out_means.append(m.squeeze(0))      
                else:
                    m = run_and_collect(x)                 
                    out_means.extend([m[i] for i in range(m.shape[0])])

        if variable_T:
            return out_means
        else:
            return torch.stack(out_means, dim=0)



    def plot_summary(self, T, dt, pred, true, figsize=(8, 8), num_traces=6,
                     ncols=2, error_type=None, mode=None, var=None, data=None):
        plot_traces(T, dt, pred, true, figsize=(8, 8), num_traces=6,
                    ncols=2, error_type=None, mode=None, var=None, data=None)
        plt.title('Kalman Rates', color='red')
        if self.save_model:
            plt.savefig(self.save_path + 'rates.png')
        plt.show()
        plt.close()

        print(eval_metrics.rsquared(true, pred))

        plt.hist(eval_metrics.rsquared(true, pred), bins=8)
        if self.save_model:
            plt.savefig(self.save_path + 'rsquared.png')
        plt.show()

    def single_plot(self, T, valid_data, valid_truth):
        trial_idx = np.random.randint(valid_data.shape[0])  # random int between 0 and size(0th dimension)
        neuron_idx = np.random.randint(valid_data.shape[2])

        pred, pred_noise = self.predict_rate(valid_data[trial_idx][None, :, :])
        pred        = pred.detach().cpu()         # (dim, T)
        pred_noise  = pred_noise.detach().cpu()   # (dim, T)
        true_rate_test = valid_truth[trial_idx].cpu()

        plt.plot(np.arange(T), pred[neuron_idx], label="pred", color='#37A1D0')
        plt.plot(np.arange(T), pred[neuron_idx] + 2 * np.sqrt(pred_noise[neuron_idx]), color='#37A1D0',
                 linestyle='dashed')
        plt.plot(np.arange(T), pred[neuron_idx] - 2 * np.sqrt(pred_noise[neuron_idx]), color='#37A1D0',
                 linestyle='dashed')
        plt.plot(np.arange(T), true_rate_test.T[neuron_idx], label="true", color='red')
        # plt.scatter(np.arange(T), valid_data[trial_idx].T[0], color='orange')
        plt.legend()
        plt.show()
