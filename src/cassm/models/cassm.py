import os
import torch
import time
import numpy as np
import torch.nn as nn
import warnings
from cassm.losses import CASSMElboLoss
from cassm.utils import training_utils
from linear_operator.operators import (
    KroneckerProductLinearOperator,
    IdentityLinearOperator,
)
from linear_operator import operators
from typing import Optional, Union, List
import gpytorch
from cassm.utils.block_diagonal_sparse_linear_operator import (
    BlockDiagonalSparseLinearOperator,
)
from cassm.utils.svd_inv import svd_inv


class ComputationAwareFilterSmoother(nn.Module):

    def __init__(
        self,
        projection_dim: int,
        nneurons: int,
        timesteps: int,
        device: torch.device,
        dt: float = 1.0,
        dataset_name: Optional[str] = None,
        spatial_prior: Optional[torch.Tensor] = None,
        save_model: bool = False,
        use_dense_projection: bool = False,
    ) -> None:
        super().__init__()

        self.dim = nneurons
        self.projection_dim = projection_dim
        self.remainder = self.dim % self.projection_dim
        if self.remainder != 0 and not use_dense_projection:
            warnings.warn(
                f"Number of neurons is not divisible by projection dimension. "
                f"Throwing away {self.remainder} neurons."
            )
            self.dim -= self.remainder

        self.state_dim = 2 * self.dim
        self.t = timesteps
        self.device = device
        self.save_model = save_model
        self.dt = torch.tensor(float(dt), device=device)

        self.raw_sigma_f = nn.Parameter(1e-1 * torch.ones(1, device=device))
        self.raw_ell = nn.Parameter(1e-1 * torch.ones(1, device=device))
        self.softplus = nn.Softplus()
        self.loss_fn = CASSMElboLoss()

        if spatial_prior is not None:
            self.latent_locations = spatial_prior.float().to(device)
        else:
            self.latent_locations = nn.Parameter(
                torch.randn(self.dim, 3, device=device)
                / torch.sqrt(torch.tensor(float(self.dim), device=device))
            )

        self.spatial_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=3)
        )
        self.spatial_kernel(self.latent_locations)

        self.use_dense_projection = use_dense_projection
        self.obs_noise_values = nn.Parameter(
            1e-2 * torch.ones(self.dim, device=self.device)
        )
        self.belief_initial_state = nn.Parameter(
            torch.empty(self.state_dim, 1, device=self.device)
        )

        if self.use_dense_projection:
            self.dense_projection = nn.Parameter(
                torch.empty(self.projection_dim, self.dim, device=self.device)
            )
            nn.init.orthogonal_(self.dense_projection)
        else:
            self.projection = nn.Parameter(
                torch.ones(
                    (self.projection_dim, self.dim // self.projection_dim),
                    device=self.device,
                )
            )
            self.projection_indices = torch.arange(
                self.dim, device=self.device
            ).reshape(self.projection_dim, self.dim // self.projection_dim)

        self.observation_matrix = KroneckerProductLinearOperator(
            IdentityLinearOperator(self.dim, device=self.device),
            torch.tensor([[1.0, 0.0]], device=self.device),
        )

        if self.save_model:
            assert dataset_name is not None
            save_id = training_utils.random_run(6)
            self.save_path = f"./cassm_runs/{dataset_name}/run_id_{save_id}/"
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)

    # ------------------------------------------------------------------
    # Matern 3/2 temporal process helpers
    # ------------------------------------------------------------------

    def matern32_time_process_cov(
        self,
        delta_t: torch.Tensor,
        sigma_f2: torch.Tensor,
        ell: torch.Tensor,
    ) -> torch.Tensor:
        """2x2 discrete-time process-noise matrix Q_t(Δ) for a Matern 3/2 kernel."""
        lam = torch.sqrt(torch.tensor(3.0, device=delta_t.device)) / ell
        rho2 = torch.exp(-2.0 * lam * delta_t)
        u = lam * delta_t

        q11 = 3.0 * sigma_f2 * (1.0 - rho2 * (1.0 + 2.0 * u + 2.0 * u**2))
        q22 = 3.0 * sigma_f2 * lam**2 * (1.0 - rho2 * (1.0 - 2.0 * u + 2.0 * u**2))
        q12 = 6.0 * sigma_f2 * lam**3 * delta_t**2 * rho2

        return torch.stack(
            [torch.stack([q11, q12], -1), torch.stack([q12, q22], -1)],
            dim=-2,
        )

    def matern32_time_stationary_cov(
        self, sigma_f2: torch.Tensor, ell: torch.Tensor
    ) -> torch.Tensor:
        lam = torch.sqrt(torch.tensor(3.0, device=self.device)) / ell
        return torch.stack(
            [
                torch.stack([sigma_f2, torch.zeros_like(sigma_f2)], -1),
                torch.stack([torch.zeros_like(sigma_f2), lam**2 * sigma_f2], -1),
            ],
            -2,
        )

    def transition_matrix_time(
        self, delta_t: torch.Tensor, ell: torch.Tensor
    ) -> torch.Tensor:
        """2x2 transition matrix A_t = exp(F_t Δ) for Matern 3/2."""
        lam = torch.sqrt(torch.tensor(3.0, device=self.device)) / ell
        F_t = torch.tensor([[0.0, 1.0], [-(lam**2), -2.0 * lam]], device=self.device)
        return torch.matrix_exp(F_t * delta_t)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_dynamics(self):
        """Build the Kronecker transition matrix and stationary prior covariance."""
        ell = self.softplus(self.raw_ell)
        sigma_f2 = self.softplus(self.raw_sigma_f)

        A_t = self.transition_matrix_time(self.dt, ell)
        transition_matrix = KroneckerProductLinearOperator(
            IdentityLinearOperator(self.dim, device=self.device), A_t
        )

        Sigma_inf_t = self.matern32_time_stationary_cov(sigma_f2, ell)
        Sigma_inf_op = KroneckerProductLinearOperator(
            self.spatial_kernel(self.latent_locations), Sigma_inf_t
        )

        return transition_matrix, Sigma_inf_op

    def _build_projected_obs(self):
        """Build the projected observation matrix and projected noise."""
        if self.use_dense_projection:
            H_proj = (self.dense_projection @ self.observation_matrix).unsqueeze(0)
            R_proj = (
                self.dense_projection
                * self.softplus(self.obs_noise_values)
                @ self.dense_projection.mT
            )
        else:
            projection = BlockDiagonalSparseLinearOperator(
                non_zero_idcs=self.projection_indices,
                blocks=self.projection,
                size_input_dim=self.dim,
            )
            H_proj = (projection @ self.observation_matrix).to_dense().unsqueeze(0)
            R_proj = (
                self.projection.pow(2)
                * self.softplus(self.obs_noise_values)[self.projection_indices]
            ).sum(dim=1)

        return H_proj, R_proj

    def _truncate_downdate(self, M: torch.Tensor) -> torch.Tensor:
        """SVD-truncate M to projection_dim columns: U_r S_r."""
        U, S, _ = svd_inv.apply(M)
        r = self.projection_dim
        return U[..., :r] * S[..., :r].unsqueeze(-2)

    # ------------------------------------------------------------------
    # Forward filter (Algorithm 1 & 2)
    # ------------------------------------------------------------------

    def filter(self, data, return_type="forward"):
        """
        Args:
            data: (batch, T, dim)
            return_type: "forward" | "prediction" | "for_smoother"
        """
        num_trials = data.shape[0]
        T = data.shape[1]

        if self.remainder != 0:
            data = data[:, :, : self.dim]

        if return_type == "prediction":
            updated_belief_state_means = torch.empty(
                (num_trials, T, self.state_dim), device=self.device
            )
            updated_belief_obs_vars = torch.empty(
                (num_trials, T, self.dim), device=self.device
            )

        prior_belief_state_mean = self.belief_initial_state.unsqueeze(0).expand(
            num_trials, -1, -1
        )

        rank = self.projection_dim
        downdate_sqrt = torch.zeros(size=(1, self.state_dim, rank), device=self.device)
        loss = torch.tensor(0.0, device=self.device)

        transition_matrix, Sigma_inf_op = self._build_dynamics()
        H_proj, R_proj = self._build_projected_obs()

        prior_belief_state_cov_op = Sigma_inf_op - operators.RootLinearOperator(
            downdate_sqrt
        )

        if return_type == "for_smoother":
            m_cache = []  # filtered means:          (batch, state_dim, 1)
            ds_cache = []  # predicted downdate M⁻_t: (1,     state_dim, rank)
            sc_cache = []  # P_t W_t = scaled_cov:    (1,     state_dim, proj_rank)
            w_cache = []  # H' u (dual mean):         (batch, state_dim, 1)
            W_cache = []  # H' U (dual cov factor):   (1,     state_dim, proj_rank)

        for t in range(T):

            if return_type == "for_smoother":
                ds_cache.append(downdate_sqrt.detach().clone())

            # Innovation matrix  S = H_proj P⁻ H_proj' + R_proj
            tmp = prior_belief_state_cov_op.matmul(H_proj.mT)
            innovation_matrix = H_proj @ tmp
            if self.use_dense_projection:
                innovation_matrix = innovation_matrix + R_proj.unsqueeze(0)
            else:
                innovation_matrix.diagonal(dim1=-2, dim2=-1).add_(R_proj)

            # Residual  y_t - H m⁻_t
            prior_predictive_residual = (
                data[:, t, :].unsqueeze(-1)
                - self.observation_matrix @ prior_belief_state_mean
            )

            # Project residual
            if self.use_dense_projection:
                projected_residual = self.dense_projection @ prior_predictive_residual
            else:
                projected_residual = (
                    (
                        self.projection
                        * prior_predictive_residual.squeeze(-1)[
                            ..., self.projection_indices
                        ]
                    )
                    .sum(-1)
                    .unsqueeze(-1)
                )

            L = torch.linalg.cholesky(innovation_matrix, upper=False)

            cakf_mean_message = H_proj.mT @ torch.cholesky_solve(
                projected_residual, L, upper=False
            )
            cakf_cov_message = torch.linalg.solve_triangular(L, H_proj, upper=False).mT

            mean_update_term = prior_belief_state_cov_op.matmul(cakf_mean_message)
            updated_belief_state_mean = prior_belief_state_mean + mean_update_term
            scaled_cov = prior_belief_state_cov_op.matmul(cakf_cov_message)

            if return_type == "for_smoother":
                m_cache.append(updated_belief_state_mean.detach())
                sc_cache.append(scaled_cov.detach())
                w_cache.append(cakf_mean_message.detach())
                W_cache.append(cakf_cov_message.detach())

            # Truncate and predict
            M_trunc = self._truncate_downdate(
                torch.cat([downdate_sqrt, scaled_cov], dim=-1)
            )
            prior_belief_state_mean = transition_matrix @ updated_belief_state_mean
            downdate_sqrt = transition_matrix @ M_trunc
            prior_belief_state_cov_op = prior_belief_state_cov_op.linear_ops[
                0
            ] - operators.RootLinearOperator(downdate_sqrt)

            if return_type == "for_smoother":
                continue

            loss = loss + self.loss_fn(
                prior_predictive_residual=prior_predictive_residual,
                prior_state_covariance=prior_belief_state_cov_op,
                obs_noise=self.softplus(self.obs_noise_values),
                cakf_mean_message=cakf_mean_message,
                mean_update_term=mean_update_term,
                projected_noise=R_proj,
                innovation_cholesky=L,
                use_dense_projection=self.use_dense_projection,
            )

            if return_type == "prediction":
                P_diag = prior_belief_state_cov_op.diagonal(dim1=-1, dim2=-2)
                pos_diag = P_diag[..., 0::2]
                updated_belief_obs_vars[:, t, :] = pos_diag + self.softplus(
                    self.obs_noise_values
                )
                updated_belief_state_means[:, t, :] = updated_belief_state_mean[:, :, 0]

        loss = loss * (1 / T) * (1 / self.dim)

        if return_type == "forward":
            return loss
        elif return_type == "prediction":
            return updated_belief_state_means, updated_belief_obs_vars
        elif return_type == "for_smoother":
            return (
                m_cache,
                ds_cache,
                sc_cache,
                w_cache,
                W_cache,
                Sigma_inf_op,
                transition_matrix,
            )
        else:
            raise ValueError(
                f"Unknown return_type '{return_type}'. Expected: forward | prediction | for_smoother"
            )

    def forward(self, data):
        return self.filter(data, return_type="forward")

    # ------------------------------------------------------------------
    # RTS smoother
    # ------------------------------------------------------------------

    def rts_smoother(self, data):
        """
        CAKS backward smoother (Algorithm 3 of https://arxiv.org/abs/2405.08971).

        Returns
        -------
        ms          : (batch, T, state_dim)  smoothed state means
        Ps_pos_diag : (batch, T, dim)        diagonal of smoothed covariance at position entries
        """
        if self.remainder != 0:
            data = data[:, :, : self.dim]

        with torch.no_grad():
            (
                m_cache,
                ds_cache,
                sc_cache,
                w_cache,
                W_cache,
                Sigma_inf_op,
                transition_matrix,
            ) = self.filter(data, return_type="for_smoother")

            T = len(m_cache)
            rank = self.projection_dim

            # Initialise at t = T-1 (smoothed == filtered at the last timestep)
            ws = w_cache[T - 1]  # (batch, state_dim, 1)
            Ws = W_cache[T - 1]  # (1,     state_dim, proj_rank)

            ms_list = [None] * T
            Ps_pos_diag_list = [None] * T

            ms_list[T - 1] = m_cache[T - 1]

            M_last = torch.cat([ds_cache[T - 1], sc_cache[T - 1]], dim=-1)
            sigma_diag = Sigma_inf_op.diagonal(dim1=-1, dim2=-2)
            Ps_pos_diag_list[T - 1] = (sigma_diag - (M_last**2).sum(dim=-1))[..., 0::2]

            # Backward pass
            for t in reversed(range(T - 1)):
                A_T_ws = transition_matrix.mT @ ws  # (batch, state_dim, 1)
                A_T_Ws = transition_matrix.mT @ Ws  # (1,     state_dim, Ws_cols)

                # P_t = Sigma - M_t M_t',  M_t = [ds_t | sc_t]
                M_t = torch.cat([ds_cache[t], sc_cache[t]], dim=-1)

                def _Pt_mv(v, _M=M_t):
                    return Sigma_inf_op.matmul(v) - _M @ (_M.mT @ v)

                # Smoothed mean: m^s_t = m_t + P_t A^T w^s_{t+1}
                ms_list[t] = m_cache[t] + _Pt_mv(A_T_ws)

                # Smoothed covariance diagonal via M^s_t = [M_t | P_t A^T W^s_{t+1}]
                Ms_t = torch.cat([M_t, _Pt_mv(A_T_Ws)], dim=-1)
                Ps_pos_diag_list[t] = (sigma_diag - (Ms_t**2).sum(dim=-1))[..., 0::2]

                # Dual message updates:
                #   w^s_t = w_t + A^T w^s_{t+1} - W_t (P^-W_t)^T A^T w^s_{t+1}
                #   W^s_t = [W_t | A^T W^s_{t+1} - W_t (P^-W_t)^T A^T W^s_{t+1}]
                Wt = W_cache[t]  # H' U
                P_neg_Wt = sc_cache[t]  # P_t W_t

                ws = w_cache[t] + A_T_ws - Wt @ (P_neg_Wt.mT @ A_T_ws)

                new_Ws_cols = A_T_Ws - Wt @ (P_neg_Wt.mT @ A_T_Ws)
                Ws = torch.cat([Wt, new_Ws_cols], dim=-1)

                # Truncate W^s_t to maintain bounded rank
                U_s, S_s, _ = torch.linalg.svd(Ws, full_matrices=False)
                Ws = U_s[..., :rank] * S_s[..., :rank].unsqueeze(-2)

            ms = torch.stack([ms_list[t].squeeze(-1) for t in range(T)], dim=1)
            Ps_pos_diag = torch.stack(Ps_pos_diag_list, dim=1)

        return ms, Ps_pos_diag

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_rate(self, data, use_smoother=False):
        if self.remainder != 0:
            data = data[:, :, : self.dim]

        if use_smoother:
            X, Ps_pos_diag = self.rts_smoother(data)
            pos_means = X[..., 0::2]
            pred = pos_means.mean(dim=0).T.detach()
            pred_noise = (
                (Ps_pos_diag + self.softplus(self.obs_noise_values))
                .mean(dim=0)
                .T.detach()
            )
        else:
            with torch.no_grad():
                X, Y_var = self.filter(data=data, return_type="prediction")
            pos_means = X[..., 0::2]
            pred = pos_means.mean(dim=0).T.detach()
            pred_noise = Y_var.mean(dim=0).T.detach()

        return pred, pred_noise

    def filter_per_trial_means(
        self,
        data: torch.Tensor,
        batch_size: int = 1,
        detach: bool = True,
        to_cpu: bool = True,
        return_list_if_variable_T: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        self.eval()
        device = (
            next(self.parameters()).device
            if any(p.requires_grad for p in self.parameters())
            else data.device
        )

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
                batch = trials[start : start + batch_size]

                if isinstance(data, list) or variable_T:
                    same_len = all(b.shape[-2] == batch[0].shape[-2] for b in batch)
                    per_item = not same_len
                    if not per_item:
                        x = torch.stack(batch, dim=0)
                else:
                    x = torch.stack(batch, dim=0)
                    per_item = False

                def run_and_collect(x_):
                    x_ = x_.to(device, non_blocking=True)
                    means = self.filter(x_, return_type="prediction")[0]
                    if detach:
                        means = means.detach()
                    if to_cpu:
                        means = means.cpu()
                    return means

                if per_item:
                    for trial in batch:
                        out_means.append(run_and_collect(trial.unsqueeze(0)).squeeze(0))
                else:
                    m = run_and_collect(x)
                    out_means.extend([m[i] for i in range(m.shape[0])])

        return out_means if variable_T else torch.stack(out_means, dim=0)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_loop(self, dataloader, optimizer, clip_value=-1):
        self.train()
        losses = []
        for _, data in enumerate(dataloader):
            loss = self(data)
            loss.backward()
            if clip_value > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_value)
            optimizer.step()
            losses.append(loss.item())
            self.zero_grad()
        return np.mean(losses)

    def test_loop(self, dataloader):
        self.eval()
        test_loss = 0
        with torch.no_grad():
            i = 0
            for batch_data, batch_truth in dataloader:
                test_loss += self(batch_data).item()
                i += 1
        return test_loss / i

    def train_model(
        self,
        epochs,
        optimizer,
        train_loader,
        test_loader=None,
        valid_truth=None,
        clip_value=-1,
    ):
        import matplotlib.pyplot as plt

        print("Beginning training...")
        start_lr = optimizer.param_groups[0]["lr"]
        tr_loss = np.zeros(epochs)
        mse_test_loss = np.zeros(epochs)
        nll_test_loss = np.zeros(epochs)
        te_loss = np.zeros(epochs)

        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.5, total_iters=25
        )
        start_time = time.time()

        for e in range(epochs):
            tr_loss[e] = self.train_loop(train_loader, optimizer, clip_value=clip_value)
            scheduler.step()

            if test_loader is not None:
                te_loss[e] = self.test_loop(test_loader)
                with torch.no_grad():
                    NLLLoss = torch.nn.GaussianNLLLoss()
                    num_pts = len(test_loader.dataset)
                    scale_factor = 1 / torch.tensor(num_pts, device=self.device)
                    pred, pred_noise = self.predict_rate(test_loader.dataset.tensors[0])
                    true_rate_test = torch.mean(test_loader.dataset.tensors[1], dim=0)
                    var = (scale_factor * pred_noise).clamp(min=1e-6)
                    nll_test_loss[e] = NLLLoss(
                        input=pred, target=true_rate_test.T, var=var
                    ).item()
                    mse_test_loss[e] = round(
                        torch.nn.MSELoss()(input=pred, target=true_rate_test.T).item(),
                        6,
                    )

                print(
                    f"Epoch {e + 1}"
                    f" | Train Loss: {np.round(tr_loss[e], 6)}"
                    f" | Test NLL: {np.round(nll_test_loss[e], 6)}"
                    f" | Test MSE: {np.round(mse_test_loss[e], 6)}"
                )
            else:
                print(f"Epoch {e + 1} | Train Loss: {np.round(tr_loss[e], 6)}")
            print("--------------------------------------------")

        end_time = time.time()
        print("Training complete!")

        if self.save_model:
            hrs = abs(start_time - end_time) / 60**2
            min_per_epoch = abs(start_time - end_time) / (60 * epochs)
            log = {
                "batch_size": str(train_loader.batch_size),
                "learning_rate": str(start_lr),
                "train_time_hrs": str(np.round(hrs, 4)),
                "min_per_epoch": str(np.round(min_per_epoch, 4)),
                "clip_value": str(clip_value),
                "neurons": str(self.dim),
            }
            with open(self.save_path + "run_details.txt", "w") as f:
                for k, v in log.items():
                    f.write(f"{k} : {v}\n")

            self.mse_last = np.mean(mse_test_loss[-3:])

            plt.plot(np.arange(epochs), tr_loss, label="train")
            plt.plot(np.arange(epochs), te_loss, label="test")
            plt.legend()
            plt.title("Training and testing loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(self.save_path + "training.png")
            plt.close()

            plt.plot(np.arange(epochs), mse_test_loss)
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.title("MSE on Testing Data")
            plt.savefig(self.save_path + "mse_training.png")
            np.save(file=self.save_path + "mse_test_loss.npy", arr=mse_test_loss)
            np.savetxt(
                self.save_path + "mse_test_loss.txt", mse_test_loss, delimiter=","
            )
            plt.close()

            plt.plot(np.arange(epochs), nll_test_loss)
            plt.xlabel("Epoch")
            plt.ylabel("NLL")
            plt.title("NLL on Testing Data")
            plt.savefig(self.save_path + "nll_training.png")
            np.save(file=self.save_path + "nll_test_loss.npy", arr=nll_test_loss)
            np.savetxt(
                self.save_path + "nll_test_loss.txt", nll_test_loss, delimiter=","
            )
            plt.close()

            torch.save(self.state_dict(), self.save_path + "model.pt")

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_summary(
        self,
        T,
        dt,
        pred,
        true,
        figsize=(8, 8),
        num_traces=6,
        ncols=2,
        error_type=None,
        mode=None,
        var=None,
        data=None,
    ):
        import matplotlib.pyplot as plt

        from cassm.utils import eval_metrics
        from cassm.utils.plotting import plot_traces

        plot_traces(
            T,
            dt,
            pred,
            true,
            figsize=figsize,
            num_traces=num_traces,
            ncols=ncols,
            error_type=error_type,
            mode=mode,
            var=var,
            data=data,
        )
        plt.title("CASSM Rates")
        if self.save_model:
            plt.savefig(self.save_path + "rates.png")
        plt.show()
        plt.close()

        print(eval_metrics.rsquared(true, pred))
        plt.hist(eval_metrics.rsquared(true, pred), bins=8)
        if self.save_model:
            plt.savefig(self.save_path + "rsquared.png")
        plt.show()

    def single_plot(self, T, valid_data, valid_truth):
        import matplotlib.pyplot as plt

        trial_idx = np.random.randint(valid_data.shape[0])
        neuron_idx = np.random.randint(valid_data.shape[2])
        pred, pred_noise = self.predict_rate(valid_data[trial_idx][None, :, :])
        pred = pred.cpu()
        pred_noise = pred_noise.cpu()
        true_rate_test = valid_truth[trial_idx].cpu()

        plt.plot(np.arange(T), pred[neuron_idx], label="pred", color="#37A1D0")
        plt.plot(
            np.arange(T),
            pred[neuron_idx] + 2 * np.sqrt(pred_noise[neuron_idx]),
            color="#37A1D0",
            linestyle="dashed",
        )
        plt.plot(
            np.arange(T),
            pred[neuron_idx] - 2 * np.sqrt(pred_noise[neuron_idx]),
            color="#37A1D0",
            linestyle="dashed",
        )
        plt.plot(np.arange(T), true_rate_test.T[neuron_idx], label="true", color="red")
        plt.legend()
        plt.show()


CASSM = ComputationAwareFilterSmoother
