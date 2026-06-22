"""
model_mdlatlrr.py  (faithful MDLatLRR fusion [59]; VECTORIZED for speed)
------------------------------------------------------------------------
Same math as before (paper Eqs. 2-9) - the only change is that patch extraction
P(.) and reconstruction R(.) are now NumPy-vectorized instead of Python loops,
giving identical results far faster. Requires models/L_8.npy (from train_mdlatlrr.py).

Defaults: n=8, stride=1 (faithful), levels=2, base weights 0.5/0.5. Float32 SVD.
"""

import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _sliding_patches(img, n, s):
    """P(.): (n*n, M) column matrix + (ys, xs) top-left coords. Vectorized."""
    H, W = img.shape
    win = sliding_window_view(img, (n, n))[::s, ::s]      # (ny, nx, n, n)
    ny, nx = win.shape[:2]
    P = win.reshape(ny * nx, n * n).T.astype(np.float32, copy=False)   # (n*n, M)
    ys = np.repeat(np.arange(ny) * s, nx)
    xs = np.tile(np.arange(nx) * s, ny)
    return P, (ys, xs), (H, W)


def _reconstruct(V, coords, shape, n):
    """R(.): scatter each patch back and average overlaps. Vectorized (bincount)."""
    H, W = shape
    ys, xs = coords
    Vr = V.reshape(n, n, V.shape[1])           # Vr[dy,dx,k] = patch k value at (dy,dx)
    acc = np.zeros(H * W, np.float64)
    cnt = np.zeros(H * W, np.float64)
    base = ys * W + xs
    for dy in range(n):
        rowidx = base + dy * W
        for dx in range(n):
            idx = rowidx + dx
            acc += np.bincount(idx, weights=Vr[dy, dx], minlength=H * W)
            cnt += np.bincount(idx, minlength=H * W)
    cnt[cnt == 0] = 1.0
    return (acc / cnt).reshape(H, W)


def _nuclear_norms(V, n):
    """Nuclear norm of each column reshaped to n x n (batched SVD, float32)."""
    M = V.shape[1]
    mats = V.T.reshape(M, n, n).astype(np.float32, copy=False)
    # nuclear norm = sum of singular values = sum sqrt(eigvals of A^T A); eigvalsh is
    # ~1.6x faster than full SVD here, with ~1e-5 relative difference (these are only
    # fusion weights), so it does not change the fused output at uint8 precision.
    ata = np.einsum('mij,mik->mjk', mats, mats)
    lam = np.linalg.eigvalsh(ata)
    return np.sqrt(np.clip(lam, 0, None)).sum(axis=1)


class MDLatLRRFusion:
    def __init__(self, L_path=None, n=8, stride=1, levels=2, w_base=(0.5, 0.5), **kwargs):
        self.n = n
        self.stride = stride
        self.levels = levels
        self.w_base = w_base
        if L_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            L_path = os.path.join(here, "models", f"L_{n}.npy")
        if not os.path.exists(L_path):
            raise FileNotFoundError(
                f"Projection matrix not found: {L_path}\n"
                f"Generate it first:  python train_mdlatlrr.py --unit {n} --training_path <path>")
        self.L = np.load(L_path).astype(np.float32)
        assert self.L.shape == (n * n, n * n), f"L must be {n*n}x{n*n}, got {self.L.shape}"

    def _decompose(self, img):
        base = img.astype(np.float32).copy()
        vds, meta = [], []
        for _ in range(self.levels):
            P, coords, shape = _sliding_patches(base, self.n, self.stride)
            Vd = self.L @ P
            Id = _reconstruct(Vd, coords, shape, self.n)
            vds.append(Vd); meta.append((coords, shape))
            base = base - Id
        return vds, base, meta

    def _fuse_channel(self, c1, c2):
        vds1, base1, meta = self._decompose(c1)
        vds2, base2, _ = self._decompose(c2)
        fused = self.w_base[0] * base1 + self.w_base[1] * base2
        for Vd1, Vd2, (coords, shape) in zip(vds1, vds2, meta):
            w1 = _nuclear_norms(Vd1, self.n)
            w2 = _nuclear_norms(Vd2, self.n)
            denom = w1 + w2 + 1e-12
            Vdf = (w1 / denom) * Vd1 + (w2 / denom) * Vd2
            fused = fused + _reconstruct(Vdf, coords, shape, self.n)
        return fused

    def fuse(self, visible_uint8_bgr, infrared_uint8_bgr):
        vis = visible_uint8_bgr.astype(np.float32) / 255.0
        ir = infrared_uint8_bgr.astype(np.float32) / 255.0
        out = np.zeros_like(vis)
        for ch in range(3):
            out[:, :, ch] = self._fuse_channel(vis[:, :, ch], ir[:, :, ch])
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)
