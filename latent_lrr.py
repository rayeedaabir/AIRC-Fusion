"""
latent_lrr.py  (NEW - faithful Python port of the authors' latent_lrr.m)
------------------------------------------------------------------------
Latent Low-Rank Representation (Liu & Yan, ICCV 2011), Inexact ALM solver.
Direct port of the MDLatLRR authors' MATLAB latent_lrr.m you supplied.

Problem:  min_{Z,L,E} ||Z||_* + ||L||_* + lambda * ||E||_1
          s.t.  X = X Z + L X + E

Returns Z, L, E. For MDLatLRR we only use L (the d x d projection matrix,
d = patch_dim = n*n).
"""

import numpy as np


def _svt(M, tau):
    """Singular Value Thresholding: shrink singular values of M by tau."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    svp = int(np.sum(s > tau))
    if svp >= 1:
        s_th = s[:svp] - tau
    else:
        svp, s_th = 1, np.array([0.0])
    return (U[:, :svp] * s_th) @ Vt[:svp, :]


def latent_lrr(X, lam=0.4, tol=1e-6, rho=1.1, mu=1e-6, max_mu=1e6, max_iter=1_000_000,
               verbose=False):
    X = np.asarray(X, dtype=np.float64)
    d, n = X.shape
    A = X
    m = A.shape[1]
    atx = X.T @ X
    inv_a = np.linalg.inv(A.T @ A + np.eye(m))
    inv_b = np.linalg.inv(A @ A.T + np.eye(d))

    J = np.zeros((m, n)); Z = np.zeros((m, n))
    L = np.zeros((d, d)); S = np.zeros((d, d))
    E = np.zeros((d, n))
    Y1 = np.zeros((d, n)); Y2 = np.zeros((m, n)); Y3 = np.zeros((d, d))

    it = 0
    while it < max_iter:
        it += 1
        J = _svt(Z + Y2 / mu, 1.0 / mu)                       # update J (SVT)
        S = _svt(L + Y3 / mu, 1.0 / mu)                       # update S (SVT)
        Z = inv_a @ (atx - X.T @ L @ X - X.T @ E + J + (X.T @ Y1 - Y2) / mu)   # update Z
        L = ((X - X @ Z - E) @ X.T + S + (Y1 @ X.T - Y3) / mu) @ inv_b          # update L
        xmaz = X - X @ Z - L @ X                              # update E (soft-threshold)
        temp = xmaz + Y1 / mu
        E = np.maximum(0, temp - lam / mu) + np.minimum(0, temp + lam / mu)

        leq1 = xmaz - E; leq2 = Z - J; leq3 = L - S
        stopC = max(np.abs(leq1).max(), np.abs(leq2).max(), np.abs(leq3).max())
        if verbose and it % 25 == 0:
            print(f"  iter {it}: stopC={stopC:.3e}, mu={mu:.2e}")
        if stopC < tol:
            if verbose:
                print(f"LatLRR converged in {it} iters (stopC={stopC:.2e}).")
            break
        Y1 = Y1 + mu * leq1; Y2 = Y2 + mu * leq2; Y3 = Y3 + mu * leq3
        mu = min(max_mu, mu * rho)

    return Z, L, E


if __name__ == "__main__":
    # tiny self-test: low-rank + sparse synthetic data should converge quickly
    rng = np.random.default_rng(0)
    d, n, r = 20, 60, 3
    Xlr = (rng.standard_normal((d, r)) @ rng.standard_normal((r, n)))
    Xn = Xlr + 0.01 * rng.standard_normal((d, n))
    Z, L, E = latent_lrr(Xn, lam=0.4, verbose=True)
    print("L shape:", L.shape, "| recon err:",
          np.linalg.norm(Xn - Xn @ Z - L @ Xn - E) / np.linalg.norm(Xn))
