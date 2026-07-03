import torch
import pandas as pd

def grassmann_distance(
    S: torch.Tensor,
    G: torch.Tensor,
    d: int | None = None,
    metric: str = "projection",
) -> torch.Tensor:
    
    try:
        evals, evecs = torch.linalg.eigh(G)
    except RuntimeError:                         
        evals, evecs = torch.linalg.eig(G)
        if torch.is_complex(evals):
            evals = evals.abs()                  
            evecs = evecs.real                   

    idx = torch.topk(evals.abs(), k=d).indices    
    Ghat = evecs[:, idx]                         
    U = torch.linalg.qr(S, mode="reduced").Q      
    V = torch.linalg.qr(Ghat, mode="reduced").Q  
    sigma = torch.linalg.svdvals(U.T @ V)         
    sigma = sigma.clamp(-1.0, 1.0)            
    theta = torch.arccos(sigma)
    if metric == "projection":                 
        proj_diff = U @ U.T - V @ V.T
        return (0.5)**0.5 * torch.linalg.vector_norm(
            proj_diff)

    elif metric == "geodesic":              
        return torch.linalg.vector_norm(theta, ord=2)

    else:
        raise ValueError("invalid metric")
    
if __name__ == "__main__":
    n, d         = 100, 5    
    trials       = 8        
    noise_level  = 0.005
    records = []
    torch.manual_seed(0)

    for t in range(trials):
        A  = torch.randn(n, n)
        G  = 0.5 * (A + A.T)

        evals, evecs = torch.linalg.eigh(G)
        Ghat = evecs[:, torch.topk(evals.abs(), d).indices]     # (n,d)
        S_close = torch.linalg.qr(Ghat + noise_level *
                                torch.randn_like(Ghat))[0]
        S_far   = torch.randn(n, d)
        dist_close = grassmann_distance(S_close, G, d=d)   # should be small
        dist_far   = grassmann_distance(S_far,   G, d=d)   # should be larger

        records.append({"trial": t + 1,
                        "distance(S_close, Ĝ)": dist_close.item(),
                        "distance(S_far,   Ĝ)": dist_far.item()})

    df = pd.DataFrame(records)
    print(df.to_string(index=False))
