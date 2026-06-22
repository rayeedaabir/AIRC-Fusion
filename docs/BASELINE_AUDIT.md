# Baseline fusion-math audit (code vs original papers)

Each baseline was checked against the original paper you supplied.

| Paper | Code | Verdict | Action |
|-------|------|---------|--------|
| FCDFusion [4] (Li & Fu) | `fcdfusion_vectorized` / `fcdfusion_pixel` | EXACT match to Algorithm 1 | none (verified) |
| Spectral Edge / Model2015 [7] (Connah) | `model_2015_v4.py` | Faithful (core SVD ansatz + Poisson) | keep; note minor caveats |
| Hybrid Filtering / Model2020 [6] (Zhang) | `model_2020_fixed.py` | NOT faithful (no co-occurrence filter) | RELABELLED -> GFF-style; cite Li et al. 2013 |
| Weight-Induced Contrast Map / Model2024 [5] (Panda) | old `_InternalModel2024` | NOT faithful (generic Gaussian) | FIXED -> `panda2024` (Eqs 1-11) |
| MDLatLRR [59] (Li/Wu/Kittler) | new | needs learned matrix L | IMPLEMENTED (L retrained in Python) |

## 1. FCDFusion [4] - VERIFIED
Paper `k = a*(v_m+255)/(2 v_m) + 0.5` (g=2) == code exactly. Vectorized and pixel
versions give identical output (supports the zero-error vectorization claim).

## 2. Model2015 / Connah [7] - FAITHFUL
Implements the spectral-edge ansatz (per-pixel SVD of the vis+IR gradient
Jacobian, J_new = U_R Sigma_H V_H^T) + FFT-Poisson reintegration. Disclose minor
deviations: Sobel gradients, simple-difference divergence, an added mean/std
restoration step not in the original.

## 3. Model2020 / Zhang [6] - RELABELLED (decision: relabel, not reimplement)
Zhang's novelty is an improved co-occurrence filter (ICoF) for detail fusion; the
code never implements it (it uses median+Laplacian activity). The code is in fact
a GFF-style guided-filter saliency two-scale scheme (close to Li, Kang & Hu,
"Image Fusion with Guided Filtering", IEEE TIP 2013).
DECISION: relabel this baseline as "GFF (Guided-Filter Saliency)" and cite the GFF
lineage [Li et al. 2013] instead of Zhang [6]. Rationale (see chat): a faithful
ICoF cannot be validated (no official code, ambiguous paper) and would add an
unvalidatable reimplementation; relabelling is honest, removes the misattribution
risk, and the comparative study is already strong (FCDFusion + Model2015 +
MDLatLRR faithful, plus 3 DL models). Reversible if you later want the ICoF.

## 4. Model2024 / Panda [5] - FIXED
Old `model2024` was `base=GaussianBlur; detail=max-abs; fused=base+detail`,
unrelated to Panda. New faithful `model_2024_panda.py` (key `panda2024`) follows
Eqs 1-11 (guided+mean filters; contrast map (I-G)^2; local-std weight on visible,
local-range weight on IR; salient detail F1; prominent detail F2=max(G1,G2);
F=F1+F2). Validate against the paper's TNO metrics before final use; exact filter
sizes should match the paper's experimental values.

## 5. MDLatLRR [59] - IMPLEMENTED (faithful, Python)
You supplied the authors' MATLAB training code, now ported to Python:
  latent_lrr.py     - faithful port of latent_lrr.m (ALM); verified converges ~1e-8
  train_mdlatlrr.py - port of train_choose_detail.m + train_latent_matrix.m + driver
                      run: python train_mdlatlrr.py --unit 8 --training_path ./training_data
                      -> models/L_8.npy
  model_mdlatlrr.py - fusion (Eqs 2-9): multi-level decomposition V_d=L*P(I),
                      nuclear-norm detail fusion, weighted-average base, reconstruct;
                      verified the decomposition reconstructs the image exactly (err 0).
Model key `mdlatlrr`. DISCLOSE: "L retrained in Python from the authors' code/data
(lambda=0.4)"; a retrained L is functionally equivalent to, not bit-identical with,
their released matrix (random patch selection). Stride 1 is faithful but slow.

## Official code search (June 2026)
No confirmed official repo for Panda [5] or Zhang [6] (matches your search). One
search hit suggested github.com/XingLongH/VFFusion for Panda but it is UNVERIFIED
and the name does not match - do not cite without confirming. MDLatLRR has official
MATLAB (github.com/hli1221/imagefusion_mdlatlrr), now ported here.

## Bottom line
FCDFusion verified, Model2015 faithful, MDLatLRR implemented, Panda fixed, Zhang
relabelled. In the Methods section, state per baseline which implementation was
used and that Panda/MDLatLRR were re-implemented (Panda from the equations,
MDLatLRR ported from the authors' MATLAB) because no usable official code existed.

## DL baselines (added after checking [47] DenseFuse, [48] VGG, [49] ResNet)

- **DenseFuse [47] - NOT faithful.** The paper is an auto-encoder: each image is
  encoded SEPARATELY by a shared encoder (conv + dense block), the two feature
  sets are fused by a fusion layer (addition or L1-norm), then a decoder
  reconstructs the result. Your `model.py` instead CONCATENATES both inputs into a
  single 2-channel stream -> conv -> dense layers -> decoder (the `x_over`/`x_under`
  argument names are multi-exposure-fusion terminology). This is a different
  network; `model.pth` was almost certainly trained for THIS net, not DenseFuse.
  Fix: implement the real DenseFuse encoder/fuse/decoder and load the authors'
  official weights (github.com/hli1221/imagefusion_densefuse), OR relabel honestly.

- **VGG-19 [48] and ResNet-152 [49] - not fusion methods.** Both papers are image
  CLASSIFICATION networks. The code uses them as backbones with a hand-made
  feature-activity binary mask - a generic adaptation, not a published fusion
  method. Honest options: (a) keep them but frame explicitly as "deep classification
  backbones with a generic activity-based fusion rule, included only as a
  computational-cost upper bound" (cite [48]/[49] as the backbones - accurate); or
  (b) replace with real backbone-based fusion methods: VGG-fusion (Li, Wu, Kittler,
  "Infrared and Visible Image Fusion using a Deep Learning Framework", 2018) and
  ResNet-ZCA (Li, Wu, Durrani, 2019).

Net: the TRADITIONAL side is now solid (FCDFusion, Model2015, GFF, Panda, MDLatLRR
all faithful/accurate). The DL side needs a decision: fix DenseFuse to the real
architecture + weights, and either reframe or replace VGG-19/ResNet-152.

## UPDATE (16 Jun) - DenseFuse FIXED, Model2020 -> "Model2013"

- DenseFuse: FIXED. `model_densefuse_official.py` (key `densefuse_official`) embeds the
  authors' official PyTorch architecture verbatim (github.com/hli1221/densefuse-pytorch,
  net.py) + L1-norm fusion. SETUP: download the pretrained *.model from that repo's
  `models/` folder -> put at `models/densefuse.model`. Disclose: "DenseFuse [47],
  authors' official PyTorch architecture + pretrained weights, L1-norm strategy."
  The old mystery net (`model.py`/`densefuse`) is no longer used.
- Model2020 (GFF, Li et al. 2013 [6]) is now labelled "Model2013" for a clean
  ModelYYYY scheme: Model2013 (GFF) / Model2015 (Connah) / Model2024 (Panda).
- Reminder on naming: the 2024 Panda paper [5] is KEPT (faithful `panda2024` = "Model2024").
  The paper being replaced is the 2020 Zhang [6] -> now GFF 2013 [6] -> "Model2013".
- VGG-19/ResNet-152: per decision, KEEP and reframe as deep classification backbones /
  compute upper bound (cite [48]/[49] as backbones; disclose the generic activity fusion rule).
