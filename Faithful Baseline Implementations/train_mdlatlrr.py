"""
train_mdlatlrr.py  (NEW - Python port of the authors' MATLAB training)
----------------------------------------------------------------------
Regenerates the MDLatLRR projection matrix L (no MATLAB needed), porting:
  train_choose_detail.m  -> _choose_detail()
  train_latent_matrix.m  -> _train_L()
  train_projection_matrices.m -> __main__

Usage:
  python train_mdlatlrr.py --training_path ./training_data --unit 8
  python train_mdlatlrr.py --training_path ./training_data --unit 16
Saves models/L_8.npy (and L_16.npy). model_mdlatlrr.py loads these.

NOTE: the authors do not seed the random patch selection, so a retrained L is
functionally equivalent to (not bit-identical with) their released matrix. We
seed by default for reproducibility; disclose "L retrained from the authors'
code and training data" in the paper.
"""

import os
import argparse
import glob
import numpy as np
import cv2
from latent_lrr import latent_lrr


def _im2col_distinct(img, unit):
    """Non-overlapping unit x unit blocks -> (unit*unit, num_blocks), column-major like MATLAB."""
    h, w = img.shape
    h2, w2 = (h // unit) * unit, (w // unit) * unit
    img = img[:h2, :w2]
    cols = []
    for x in range(0, w2, unit):          # MATLAB im2col is column-major (x outer)
        for y in range(0, h2, unit):
            patch = img[y:y + unit, x:x + unit]
            cols.append(patch.flatten(order="F"))
    return np.array(cols).T               # (unit^2, num_blocks)


def _choose_detail(training_path, unit):
    B = []
    files = [f for f in sorted(glob.glob(os.path.join(training_path, "*")))
             if os.path.isfile(f)]
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = img.astype(np.float64) / 255.0           # im2double
        B.append(_im2col_distinct(img, unit))
    B = np.concatenate(B, axis=1)
    # per-column "SD" = L2 norm of the centered column (matches train_choose_detail.m)
    B_mean = B.mean(axis=0, keepdims=True)
    B_sd = np.sqrt(np.sum((B - B_mean) ** 2, axis=0))
    detail_mask = B_sd > 0.5
    B_detail = B[:, detail_mask]
    B_smooth = B[:, ~detail_mask]
    return B_detail, B_smooth


def _train_L(B_detail, B_smooth, col=2000, col_smooth=1000, lam=0.4, seed=0):
    rng = np.random.default_rng(seed)
    nd = min(col - col_smooth, B_detail.shape[1])
    ns = min(col_smooth, B_smooth.shape[1])
    B_d = B_detail[:, rng.permutation(B_detail.shape[1])[:nd]]
    B_s = B_smooth[:, rng.permutation(B_smooth.shape[1])[:ns]]
    B_rand = np.concatenate([B_d, B_s], axis=1)
    B_rand = B_rand[:, rng.permutation(B_rand.shape[1])]
    print(f"Training L on X of shape {B_rand.shape} (lambda={lam}). This can take a few minutes...")
    _, L, _ = latent_lrr(B_rand, lam=lam, verbose=True)
    return L


def train(training_path, unit, out_dir="models", seed=0):
    os.makedirs(out_dir, exist_ok=True)
    B_detail, B_smooth = _choose_detail(training_path, unit)
    print(f"unit={unit}: {B_detail.shape[1]} detail patches, {B_smooth.shape[1]} smooth patches")
    L = _train_L(B_detail, B_smooth, seed=seed)
    path = os.path.join(out_dir, f"L_{unit}.npy")
    np.save(path, L)
    print(f"Saved {path}  (shape {L.shape})")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--training_path", default="./training_data")
    ap.add_argument("--unit", type=int, default=8, choices=[8, 16])
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train(a.training_path, a.unit, a.out_dir, a.seed)
