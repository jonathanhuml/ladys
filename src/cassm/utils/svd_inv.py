import torch


def F_and_T_inv(s):
    s2 = s * s
    s2 = torch.where(s2 < 1e-30, torch.zeros_like(s2), s2)
    s_i = s2[..., :, None]        # (..., k, 1)
    s_j = s2[..., None, :]        # (..., 1, k)
    denom = s_j - s_i             # (..., k, k)
    both_zero = (s_i == 0) & (s_j == 0)
    eq_mask = denom == 0          # includes diagonal and equal SVs

    inv = torch.where(eq_mask | both_zero, torch.zeros_like(denom), 1.0 / denom)
    F = torch.where(torch.isfinite(inv), inv, torch.zeros_like(inv))
    logi = F.abs() > 1e30
    F = torch.where(logi, torch.zeros_like(F), F)
    k = s.shape[-1]
    diag_mask = torch.eye(k, dtype=torch.bool, device=s.device)
    eq_offdiag = eq_mask & (~diag_mask)

    # T uses 1/s_j (sqrt of s_j^2); let 1/0 be inf like original, both_zero stays 0
    inv_sj = (1.0 / s)[..., None, :]  # (..., k, k)
    T = torch.zeros_like(F)
    T = torch.where(eq_offdiag & ~both_zero, inv_sj, T)
    T = torch.where(logi, inv_sj, T)

    return F, T


class svd_inv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Return S as a vector (no diag embedding)
        u, s_vec, vh = torch.linalg.svd(x, full_matrices=False)
        v = vh.mH
        ctx.save_for_backward(x, u, s_vec, v)
        return u, s_vec, v

    @staticmethod
    def backward(ctx, dl_du, dl_ds_vec, dl_dv):
        x, u, s_vec, v = ctx.saved_tensors
        dtype = u.dtype

        # Safely compute s^{-1} vector
        s_inv_vec = torch.where(s_vec > 0, 1.0 / s_vec, torch.zeros_like(s_vec))

        # Small kxk blocks
        utdu = u.mH @ dl_du                         # (..., k, k)
        vtdv = v.mH @ dl_dv                         # (..., k, k)
        F, T = F_and_T_inv(s_vec)
        F = F.to(utdu.dtype)
        T = T.to(utdu.dtype)
        Fmat_u = F * (utdu - utdu.mH)               # (..., k, k)
        Fmat_v = F * (vtdv - vtdv.mH)               # (..., k, k)
        c_u1 = (Fmat_u * s_vec[..., None, :]) + (T * utdu)   # (..., k, k)
        c_u1 = u @ c_u1                                      # (..., m, k)

        # c_u2 = (I - u u^H) @ dl_du @ diag(s^{-1})
        dl_du_sinv = dl_du * s_inv_vec[..., None, :]         # (..., m, k)
        c_u2 = dl_du_sinv - u @ (u.mH @ dl_du_sinv)          # (..., m, k)

        c_u = (c_u1 + c_u2) @ v.mH                           # (..., m, n)
        g = dl_ds_vec.to(dtype)                               # (..., k)
        c_s = (u * g[..., None, :]) @ v.mH                   # (..., m, n)

        # c_v1 = diag(s) @ Fmat_v @ v^H  -> left-multiply by s: row-scale by s
        c_v1 = (s_vec[..., :, None] * Fmat_v) @ v.mH         # (..., k, n)
        tmp = (s_inv_vec[..., :, None] * dl_dv.mH)           # (..., k, n)
        c_v2 = tmp - tmp @ v @ v.mH                          # (..., k, n)

        c_v = u @ (c_v1 + c_v2)                              # (..., m, n)
        dl_dx = c_u + c_s + c_v

        assert dl_dx.isfinite().all(), "NaNs/Infs in dl_dx"
        assert (dl_dx.abs() < 1e16).all(), "Large values in dl_dx"
        return dl_dx

