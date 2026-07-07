"""
Layer-wise Drift Metrics

Contains the mathematical implementations of

- Linear CKA
- Gaussian RBF MMD

Author:
Bindhu Pagadala
"""

import torch
import ot

def center_gram(K: torch.Tensor) -> torch.Tensor:
    """
    Center a Gram matrix.
    """
    n = K.size(0)
    H = torch.eye(n, device=K.device) - torch.ones((n, n), device=K.device) / n
    return H @ K @ H


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA between two representation matrices.

    X : (N,D)
    Y : (N,D)
    """

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    K = X @ X.T
    L = Y @ Y.T

    K = center_gram(K)
    L = center_gram(L)

    hsic = (K * L).sum()

    norm_x = torch.sqrt((K * K).sum())
    norm_y = torch.sqrt((L * L).sum())

    return (hsic / (norm_x * norm_y)).item()


def rbf_kernel(
    X: torch.Tensor,
    sigma: float,
):
    """
    Gaussian RBF kernel.
    """

    dist = torch.cdist(X, X) ** 2
    return torch.exp(-dist / (2 * sigma ** 2))


def median_heuristic(
    X: torch.Tensor,
    Y: torch.Tensor,
):
    """
    Median heuristic bandwidth.
    """

    Z = torch.cat([X, Y], dim=0)

    dist = torch.cdist(Z, Z)
    sigma = torch.median(dist)

    return sigma.item()


def mmd_rbf(
    X: torch.Tensor,
    Y: torch.Tensor,
) -> float:
    """
    Gaussian RBF Maximum Mean Discrepancy.
    """

    sigma = median_heuristic(X, Y)

    Kxx = rbf_kernel(X, sigma)
    Kyy = rbf_kernel(Y, sigma)

    dist = torch.cdist(X, Y) ** 2
    Kxy = torch.exp(-dist / (2 * sigma ** 2))

    mmd = (
        Kxx.mean()
        + Kyy.mean()
        - 2 * Kxy.mean()
    )

    return mmd.item()

def sinkhorn_distance(
    X,
    Y,
    reg=1,
):
    """
    Entropic Sinkhorn Wasserstein Distance.

    Parameters
    ----------
    X : (N,D)
    Y : (N,D)

    Returns
    -------
    float
    """

    X = torch.nn.functional.normalize(
        X,
        dim=1,
    )

    Y = torch.nn.functional.normalize(
        Y,
        dim=1,
    )

    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    a = ot.unif(len(X))
    b = ot.unif(len(Y))

    M = ot.dist(
        X,
        Y,
        metric="euclidean",
    )

    value = ot.bregman.sinkhorn_stabilized(
        a,
        b,
        M,
        reg=reg,
    )

    return float(value)