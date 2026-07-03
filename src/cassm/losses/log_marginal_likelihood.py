import torch
import math


def log_MLL(residual, y_cholesky):
    nneurons = residual.shape[1]
    loss1 = torch.mean(
        0.5
        * torch.transpose(residual, dim0=1, dim1=2)
        @ torch.cholesky_solve(input2=y_cholesky, input=residual, upper=False)
    )
    Y_c_diags = torch.diagonal(y_cholesky, offset=0, dim1=1, dim2=2)
    loss2 = torch.mean(torch.sum(torch.log(Y_c_diags), dim=1))
    loss3 = 0.5 * torch.log(2 * torch.tensor(math.pi)) * nneurons
    return loss1 + loss2 + loss3
