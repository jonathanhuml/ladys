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


class KalmanFilterSmoother(nn.Module):
    """
    ALGORITHM NOTATION: Square-Root Formulas for Kalman Filter, Information Filter,
    and RTS Smoother: Links via Boomerang Prediction Residual--Table 3 (Chin 2023)

    Link: https://ipnpr.jpl.nasa.gov/progress_report/42-233/42-233A.pdf

    Problem setup:
    Let F, H, Q, and R be SxS matrices (S is self.dimensionality)
    Latent variables: x_t \in \R^{S x 1}
    Observations (i.e. data): y_t \in \R^{S x 1}

    Model:
    x_{t+1} = Fx_t + w_t
    y_t = Hx_t + v_t

    where w_t ~ N(0, Q) i.e. the process variance
        v_t ~ N(0, R) i.e. the measurement variance


     INITIALIZATION:
     nneurons (integer): the number of neurons defines the state dimensionality of the filter
     timesteps (integer): number of measurement timesteps
     dataset_name (string): name the run type for saving (can name after dataset, etc.)

     IMPORTANT NOTES:
    1) Note the difference between self.process_noise_values, self.obs_noise_values
        Raw value "variances" may be negative due to training procedure
        Access positive values with self.process_noise and self.R, which are thresholded tensors, NOT parameters
    """

    def __init__(
            self,
            nneurons,
            latent_dim,
            timesteps,
            device,
            dataset_name=None,
            save_model=False,
            health_checks=True,
    ):
        super(KalmanFilterSmoother, self).__init__()
        self.dim = nneurons
        self.t = timesteps
        self.device = device
        self.latent_dim = latent_dim

        # Debugging and saving functionality
        self.dataset_name = dataset_name
        self.save_model = save_model
        self.health_checks = health_checks

        self.process_noise_values = nn.Linear(in_features=self.latent_dim, 
                                              out_features=1, 
                                              bias=False, 
                                              device=self.device)
        nn.init.constant_(self.process_noise_values.weight, 
                          val=1e-1)

        self.obs_noise_values = nn.Linear(in_features=self.dim, 
                                          out_features=1, 
                                          bias=False, 
                                          device=self.device)
        nn.init.constant_(self.obs_noise_values.weight, 
                          val=1e-1)

        # ------------------------------------
        self.transition_matrix = nn.Linear(in_features=self.latent_dim, 
                                           out_features=self.latent_dim, 
                                           bias=False, 
                                           device=self.device)
        nn.init.eye_(self.transition_matrix.weight)
        geotorch.sphere(self.transition_matrix, "weight")
        self.observation_matrix = nn.Linear(in_features=self.latent_dim, 
                                            out_features=self.dim, 
                                            bias=False, 
                                            device=self.device)
        nn.init.eye_(self.observation_matrix.weight)
        geotorch.sphere(self.observation_matrix, "weight")

        if self.save_model:
            assert dataset_name is not None
            save_id = training_utils.random_run(6)
            run_name = "run_id_" + str(save_id)
            self.save_path = "./kalman_runs/" + dataset_name + "/" + str(run_name) + "/"
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)

    def filter(self, data, return_type="for_smoother"):
        """
        data: tensor of shape (batch_size, timesteps, dimensionality)
        return type: str "for_smoother" or "for_forward" or "for_prediction"

        -The filter is independent of the smoother for training/forward passes
        -Therefore, we don't actually have to store the pre and post-update estimates
        during training, so we free up a bunch of memory during these forward passes
        by discarding these saved estimates
        """
        num_trials = data.shape[0]
        T = data.shape[1]

        # ------------------------------------
        # initialize matrices to fill out
        if return_type == "for_smoother":
            prior_belief_state_means = torch.zeros(size=(num_trials, T, 
                                    self.latent_dim), device=self.device)
            prior_belief_state_covs = torch.zeros(size=(num_trials, T, 
                                    self.latent_dim, self.latent_dim), device=self.device)
            updated_belief_state_means = torch.zeros(size=(num_trials, T, 
                                    self.latent_dim), device=self.device)
            updated_belief_state_covs = torch.zeros(size=(num_trials, T, 
                                    self.latent_dim, self.latent_dim), device=self.device)

        elif return_type == "for_prediction":
            updated_belief_state_means = torch.zeros(size=(num_trials, T, 
                                    self.latent_dim), device=self.device)
            updated_belief_state_covs = torch.zeros(size=(num_trials, T, 
                                    self.latent_dim, self.latent_dim), device=self.device)

        prior_belief_state_mean = torch.zeros(size=(num_trials, 
                                                    self.latent_dim, 1), 
                                                    device=self.device)
        prior_belief_state_cov  = IdentityLinearOperator(self.latent_dim, 
                                                         batch_shape=(num_trials,), device=self.device)
        loss = torch.tensor(0.0, device=self.device)

        observation_matrix = self.observation_matrix.weight.unsqueeze(0)
        transition_matrix = self.transition_matrix.weight.unsqueeze(0)

        m = nn.Softplus()

        # ------------------------------------
        # FILTER RECURSION
        for t in range(T):

            innovation_matrix = observation_matrix @ prior_belief_state_cov @ observation_matrix.transpose(1,2)
            
            innovation_matrix.diagonal(dim1=-2, 
                                       dim2=-1).add_(
                                           m(self.obs_noise_values.weight)
                                        )

            prior_predictive_residual = data[:, t, :].unsqueeze(-1) - torch.matmul(observation_matrix, prior_belief_state_mean)

            if self.health_checks:
                if t % 50 == 0:
                    print('Y_cond: ' + str(torch.linalg.cond(innovation_matrix[0])))
                    print('P_cond: ' + str(torch.linalg.cond(prior_belief_state_cov[0])))
                    print('-----------------------')

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
            updated_belief_state_cov = jos_gain @ prior_belief_state_cov @ jos_gain.transpose(1,2) + kalman_gain * m(
                    self.obs_noise_values.weight) @ kalman_gain.transpose(1,2)


            # UPDATE FOR t+1
            prior_belief_state_mean = torch.matmul(transition_matrix, updated_belief_state_mean)

            prior_belief_state_cov = transition_matrix @ updated_belief_state_cov @ transition_matrix.transpose(1,2)
            prior_belief_state_cov.diagonal(dim1=-2, dim2=-1).add_(m(self.process_noise_values.weight))


            # ------------------------------------
            # compute marginal log-likelihood loss for time t
            loss_t = log_MLL(prior_predictive_residual, innovation_matrix_cholfac)

            # increment total loss by loss at time t
            loss += loss_t
            # ------------------------------------

            if return_type == "for_smoother":
                # assign t^th time point before next pass
                prior_belief_state_means[:, t, :] = prior_belief_state_mean[:, :, 0]
                prior_belief_state_covs[:, t, :, :] = prior_belief_state_cov
                updated_belief_state_means[:, t, :] = updated_belief_state_mean[:, :, 0]
                updated_belief_state_covs[:, t, :, :] = updated_belief_state_cov

            if return_type == "for_prediction":
                # assign t^th time point before next pass
                updated_belief_state_means[:, t, :] = updated_belief_state_mean[:, :, 0]
                updated_belief_state_covs[:, t, :, :] = updated_belief_state_cov

        # torch.cuda.synchronize()
        # normalize loss by T and nneurons
        loss = loss * (1 / T) * (1 / self.dim)

        if return_type == "for_smoother":
            return prior_belief_state_means, prior_belief_state_covs, updated_belief_state_means, updated_belief_state_covs
        elif return_type == "for_forward":
            return loss
        elif return_type == "for_prediction":
            return updated_belief_state_means, updated_belief_state_covs
        else:
            print("Return type must be for_smoother or for_forward ")

    def rts_smoother(self, data):
        num_trials = data.shape[0]
        T = data.shape[1]

        # ensure variance is positive real number with nonlinearity
        m = nn.Softplus()
        diag_Q = torch.zeros(self.dim, self.dim)
        diag_Q[range(len(diag_Q)), range(len(diag_Q))] = m(self.process_noise_values.weight)
        self.process_noise = m(self.process_noise_values.weight)
        Q_nt = diag_Q[None, :, :]

        # precompute an identity tensor
        I = torch.eye(self.dim, device=self.device)[None, :, :]

        prior_belief_state_means, prior_belief_state_covs, updated_belief_state_means, updated_belief_state_covs = self.filter(data, return_type="for_smoother")
        x_star = updated_belief_state_means[:, -1, :]
        P_star = updated_belief_state_covs[:, -1, :, :]

        Xss = torch.zeros(size=(num_trials, T, self.dim))
        Xss[:, -1, :] = x_star
        Pss = torch.zeros(size=(num_trials, T, self.dim, self.dim))
        Pss[:, -1, :, :] = P_star

        for t in reversed(range(T - 1)):
            # same cholesky_trick as in filtering equations to avoid matrix inversion
            P_c = torch.linalg.cholesky(prior_belief_state_covs[:, t + 1, :, :], upper=False)
            P_inv_F = torch.cholesky_solve(input2=P_c, input=self.transition_matrix.weight[None, :, :], upper=False)
            S = updated_belief_state_covs[:, t, :, :] @ torch.transpose(P_inv_F, dim0=1, dim1=2)

            dif = x_star - prior_belief_state_means[:, t + 1, :]
            # [:, :, None] expands the dimensions from a 2d-matrix to 3-tensor so shapes are consistent
            x_star = updated_belief_state_means[:, t, :][:, :, None] + torch.matmul(S, dif[:, :, None])
            x_star = x_star[:, :, 0]

            # Stabilized RTS covariance matrix P_star avoids differencing of PSD matrices
            P_plus_Q = prior_belief_state_covs[:, t + 1, :, :] + Q_nt
            I_minus_SF = I - S @ self.transition_matrix.weight[None, :, :]
            P_star = S @ P_plus_Q @ torch.transpose(
                S, dim0=1, dim1=2
            ) + I_minus_SF @ updated_belief_state_covs[:, t, :, :] @ torch.transpose(
                I_minus_SF, dim0=1, dim1=2
            )

            if self.health_checks:
                # check if P_star is positive semi-definite
                L, V = torch.linalg.eig(P_star)
                min_eigenvalue = torch.min(L.real)
                if min_eigenvalue < 0:
                    print("Negative eigenvalue detected: " + str(min_eigenvalue))

            Xss[:, t, :] = x_star
            Pss[:, t, :, :] = P_star

        return Xss, Pss

    def forward(self, data):
        loss = self.filter(data, return_type="for_forward")
        return loss

    def get_obs_noise(self):
        return nn.Softplus(self.obs_noise_values.weight)

    def get_process_noise(self):
        return nn.Softplus(self.process_noise_values.weight)

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
    def predict_rate(self, data, use_filter=True):

        T = data.shape[1]
        dim = data.shape[2]

        if use_filter:
            X, P = self.filter(data=data, return_type="for_prediction")
        else:
            X, P = self.rts_smoother(data=data)

        pred = torch.mean(self.observation_matrix.weight.unsqueeze(0) @ X.transpose(1,2), dim=0).detach()

        # ensure variance is positive real number with nonlinearity
        m = nn.Softplus()
        noise = self.observation_matrix.weight @ P @ torch.transpose(self.observation_matrix.weight, dim0=0, dim1=1)
        noise.diagonal(dim1=-2, dim2=-1).add_(m(self.obs_noise_values.weight))
        normal_noise = noise[0]
        # check to make sure normal noise can actually take first matrix of repeats...
        pred_noise = torch.zeros(size=(dim, T), device=self.device)

        for j in range(T):
            s_t = torch.diag(normal_noise[j, :, :])
            pred_noise[:, j] = s_t.detach()

        return pred, pred_noise

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
        true_rate_test = valid_truth[trial_idx]

        plt.plot(np.arange(T), pred[neuron_idx], label="pred", color='#37A1D0')
        plt.plot(np.arange(T), pred[neuron_idx] + 2 * np.sqrt(pred_noise[neuron_idx]), color='#37A1D0',
                 linestyle='dashed')
        plt.plot(np.arange(T), pred[neuron_idx] - 2 * np.sqrt(pred_noise[neuron_idx]), color='#37A1D0',
                 linestyle='dashed')
        plt.plot(np.arange(T), true_rate_test.T[neuron_idx], label="true", color='red')
        plt.scatter(np.arange(T), valid_data[trial_idx].T[0], color='orange')
        plt.legend()
        plt.show()

    
