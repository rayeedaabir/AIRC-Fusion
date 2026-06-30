# AIRC-Fusion: Real-Time Infrared and Visible Video Fusion on Processor-Only Edge Hardware

Vectorized reformulation of the FCDFusion direct-mapping algorithm that runs infrared/visible
**video** fusion at frame rate on a general-purpose **CPU** — no GPU, no deep-learning runtime —
together with a lightweight **Temporal Instability Score** for measuring frame-to-frame flicker.

This repository accompanies the manuscript *"Real-Time Infrared and Visible Video Fusion on
Processor-Only Edge Hardware Using Vectorized Direct Mapping"* (Cureus Journal of Computer Science).

- **Paper DOI:** _to be added after publication_
- **Archived code & data (Zenodo):** `[10.5281/zenodo.20793801](https://doi.org/10.5281/zenodo.20793801)`
- **License:** code MIT (see `LICENSE`); manuscript CC BY 4.0; datasets retain their own licenses.

---

## Overview

Infrared and visible image fusion is dominated by methods that are accurate but too slow or too
dependency-heavy for processor-only edge hardware, and most are evaluated only on static image pairs.
This project shows that a classical direct-mapping algorithm, once fully vectorized into element-wise
array operations, executes the **identical** arithmetic at video rate on a CPU. Nine fusion methods
(classical, low-rank, and deep-learning) are benchmarked under identical CPU-only conditions on
automotive (FLIR ADAS) and surveillance (CAMEL) video, across speed, structural fidelity, information
content, and temporal stability.

## Repository structure

```
AIRC-Fusion/
  run_config.py                  # THE single config file (paths, model list, per-model frame caps)
  run_all.py / run_one.py        # run every model x dataset, or one combination
  recompute_flir_uniform.py      # re-aggregate FLIR to a uniform first-N frames from per-frame CSVs

  image_fusion_processor.py      # fusion engine + dispatcher for every method
  dataset_processor.py           # ingestion / frame pairing
  evaluation_metrics.py          # all metrics incl. the Temporal Instability Score
  calculate_confidence_intervals.py
  config.py  option.py  video_utils.py  benchmark_utils.py

  # Faithful baseline implementations
  model_2015_v4.py                                     # Model2015  (Connah spectral edge)
  model_2020_fixed.py                                  # Model2013  (GFF, Li/Kang/Hu 2013)
  model_2024_panda.py                                  # Model2024  (Panda 2024)
  latent_lrr.py train_mdlatlrr.py model_mdlatlrr.py    # MDLatLRR
  model_densefuse_official.py                          # DenseFuse (official architecture)
  models/                                              # trained weights: L_8.npy, densefuse.model

  # Figure / table / output generators
  make_figures.py                # speed-fidelity scatter, radar, speed bars (from the workbook)
  make_matd_profiles.py          # per-frame MATD profiles; --grid builds the compact paper figure
  make_comparison_videos.py      # one aligned all-methods grid video per dataset (3x4)
  qualitative_panels.py          # single-frame all-methods panels (3x4, lettered)
  make_static_comparison.py      # all-methods panel for one static image pair (3x4, lettered)
  prepare_landsat_pair.py        # build an 8-bit RGB+NIR pair from the four Landsat band TIFFs
  run_ablation.py                # gain-map component ablation + gamma sweep -> tables + panels

  # Tooling
  renumber_references.py         # renumber citations/refs by first appearance (manuscript prep)
  validate_citations.py          # check reference DOIs against Crossref / DataCite

  docs/                          # BASELINE_AUDIT.md, TEMPORAL_SCORE_NOTE.md, CHANGES_AIRC_README.md
  supplementary/                 # full tables, MATD profiles, static panels, pseudocode (or see Zenodo)
```

## Datasets

| Dataset | Type | Use | Source | License (redistribution) |
|---|---|---|---|---|
| FLIR ADAS | Video | Primary high-res benchmark | Teledyne FLIR | Gated user agreement |
| CAMEL (Seq 1, 2, 8, 13) | Video | Surveillance benchmark | Gebhardt & Wolf 2018 | Academic terms |
| RGB-NIR Scene | Static | Ablation + qualitative | EPFL | CC BY 4.0 (OK) |
| Landsat 8/9 | Static | Qualitative (satellite) | USGS EarthExplorer | Public domain (OK) |
| TNO | Static | Qualitative only | Toet 2014/2017 | CC BY-NC-ND (**do not redistribute**) |

Datasets are **not** included in this repository. Download each from its source. Only derived
outputs from CC BY / public-domain sets are redistributed; **TNO source and TNO-derived images are
excluded** for license compatibility.

## Installation (CPU-only)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Important:** install **only** `opencv-contrib-python` (not `opencv-python` as well) — having both
clobbers `cv2.ximgproc` and silently changes the guided-filter baseline. Remove `opencv-python` from
the environment if present. Everything runs on the CPU; no CUDA is required.

## Reproducing the results

1. Edit **`run_config.py`**: set `OUTPUT_ROOT`, the dataset paths, the model list, and (for the
   slow methods on high-res FLIR) the per-model frame caps in `LIMIT_OVERRIDE`. Keep `WORKERS = 1`
   for clean per-frame timing.
2. Run the benchmark (resumable; builds one workbook across sessions):
   ```bash
   python run_all.py                      # all models x datasets
   python run_one.py --model fcdfusion_vectorized --dataset CAMEL_Seq2   # a single combination
   ```
   Outputs go to `OUTPUT_ROOT/results/`: the metrics workbook (`AIRC-Fusion_Results.xlsx`),
   per-frame CSVs, organized fused frames, and (if enabled) videos.
3. Generate the figures and panels:
   ```bash
   python make_figures.py
   python make_matd_profiles.py --grid --datasets FLIR_ADAS CAMEL_Seq13 \
          --models fcdfusion_vectorized panda2024 densefuse_official vgg19
   python qualitative_panels.py
   python make_comparison_videos.py
   python make_static_comparison.py --vis rgbnir_vis.png --ir rgbnir_ir.png --out RGBNIR_panel.png
   python prepare_landsat_pair.py --b2 ..._B2.TIF --b3 ..._B3.TIF --b4 ..._B4.TIF --b5 ..._B5.TIF \
          --out_vis landsat_vis.png --out_ir landsat_ir.png
   python run_ablation.py --vis rgbnir_vis.png --ir rgbnir_ir.png --out_dir ablation_out
   ```

Hardware used in the paper: AMD Ryzen 5 7600X (6-core), 32 GB RAM, GPU disabled.

## Supplementary materials

The external supplement (this repo's `supplementary/` folder and the Zenodo archive) contains the
full per-dataset metric tables with 95% confidence intervals, the complete MATD profiles for all
datasets and methods, the RGB-NIR and Landsat qualitative panels, the comparison videos, and the
algorithm pseudocode with a pipeline flowchart. Large video files are hosted in the Zenodo record.

## Citation

```
Ahsan R A, Islam R, Hridi F R, et al. (N/A) Real-Time Infrared and Visible Video Fusion on Processor-Only Edge Hardware Using Vectorized Direct Mapping. Cureus J Comput Sci : e. doi:https://doi.org/10.7759/
```

## Acknowledgements

Department of Electrical and Computer Engineering, North South University, Dhaka, Bangladesh.
