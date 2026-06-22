"""
model_2024_panda.py  (NEW - FAITHFUL re-implementation of Panda et al. 2024 [5])
--------------------------------------------------------------------------------
"A weight induced contrast map for infrared and visible image fusion",
Panda et al., Computers and Electrical Engineering 117 (2024) 109256.

This replaces the previous 'model2024', which was a GENERIC two-scale Gaussian
decomposition (base = Gaussian blur, detail = max-absolute) and did NOT match
Panda's published method. This file follows the paper's equations directly:

  G_K = GuidedFilter(I_K)                          (Eq. 1)  smoothed component
  d_K = MeanFilter(I_K)                            (Eq. 2)
  C_K = (I_K - G_K)^2                              (Eq. 3)  contrast map
  L_sd = local std-dev of C_1 over m x m           (Eq. 4-5, visible)
  L_rg = local range (max-min) of C_2 over 3 x 3   (Eq. 6,   infrared)
  w1 = L_sd / max(L_sd) ; w2 = L_rg / max(L_rg)    (Eq. 7-8)
  F1 = w1*d1 + (1-w1)*d2          if w1 > w2        (Eq. 9)  salient detail
       (1-w2)*d1 + w2*d2          else
  F2 = max(G1, G2)                                 (Eq. 10) prominent detail
  F  = F1 + F2                                     (Eq. 11)

NOTES / DOCUMENTED ASSUMPTIONS (validate before trusting):
  * The paper's prose and equations are inconsistent about which of G_K / d_K is
    "high" vs "low" frequency. We follow the EQUATIONS as written (Eq. 3 implies
    G_K is the smoothed component). If the authors release code that differs,
    prefer theirs.
  * Exact filter sizes (guided radius/eps, mean window m, L_sd window) are not all
    given numerically in the paper; sensible defaults are exposed below and should
    be set to the paper's experimental values (or the authors' code) for an exact
    match. Structure (Eqs 1-11) is faithful regardless.
  * Color handled per-channel, matching the paper's color flowchart (Fig. 2).
"""

import cv2
import numpy as np

try:
    from cv2.ximgproc import guidedFilter
    _HAS_GF = True
except Exception:
    _HAS_GF = False


class Panda2024Fusion:
    def __init__(self, gf_radius: int = 4, gf_eps: float = 0.01,
                 mean_ksize: int = 7, lsd_ksize: int = 7, lrg_ksize: int = 3, **kwargs):
        self.gf_radius = gf_radius
        self.gf_eps = gf_eps
        self.mean_ksize = mean_ksize if mean_ksize % 2 else mean_ksize + 1
        self.lsd_ksize = lsd_ksize if lsd_ksize % 2 else lsd_ksize + 1
        self.lrg_ksize = lrg_ksize if lrg_ksize % 2 else lrg_ksize + 1

    def _guided(self, I):
        if _HAS_GF:
            return guidedFilter(I.astype(np.float32), I.astype(np.float32), self.gf_radius, self.gf_eps)
        # fallback: edge-preserving bilateral (documented; install opencv-contrib for the real guided filter)
        return cv2.bilateralFilter(I.astype(np.float32), d=self.gf_radius * 2 + 1, sigmaColor=0.1, sigmaSpace=self.gf_radius)

    def _mean(self, I):
        return cv2.boxFilter(I, ddepth=-1, ksize=(self.mean_ksize, self.mean_ksize))

    def _local_std(self, C, k):
        mean = cv2.boxFilter(C, -1, (k, k))
        mean_sq = cv2.boxFilter(C * C, -1, (k, k))
        var = np.clip(mean_sq - mean * mean, 0, None)
        return np.sqrt(var)

    def _local_range(self, C, k):
        kernel = np.ones((k, k), np.uint8)
        return cv2.dilate(C, kernel) - cv2.erode(C, kernel)

    def _norm(self, M):
        mx = float(np.max(M))
        return M / mx if mx > 1e-12 else M

    def fuse(self, visible_uint8_bgr: np.ndarray, infrared_uint8_bgr: np.ndarray) -> np.ndarray:
        vis = visible_uint8_bgr.astype(np.float32) / 255.0
        ir = infrared_uint8_bgr.astype(np.float32) / 255.0
        out = np.zeros_like(vis)

        for c in range(3):
            I1, I2 = vis[:, :, c], ir[:, :, c]
            G1, G2 = self._guided(I1), self._guided(I2)          # Eq. 1
            d1, d2 = self._mean(I1), self._mean(I2)              # Eq. 2
            C1, C2 = (I1 - G1) ** 2, (I2 - G2) ** 2              # Eq. 3
            Lsd = self._local_std(C1, self.lsd_ksize)            # Eq. 4-5
            Lrg = self._local_range(C2, self.lrg_ksize)         # Eq. 6
            w1, w2 = self._norm(Lsd), self._norm(Lrg)           # Eq. 7-8
            F1 = np.where(w1 > w2, w1 * d1 + (1 - w1) * d2, (1 - w2) * d1 + w2 * d2)  # Eq. 9
            F2 = np.maximum(G1, G2)                              # Eq. 10
            out[:, :, c] = F1 + F2                               # Eq. 11

        return np.clip(out * 255.0, 0, 255).astype(np.uint8)
