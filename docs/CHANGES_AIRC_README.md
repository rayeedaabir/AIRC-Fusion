# AIRC-Fusion pipeline - revision notes (Cureus/CJCS re-run)

Guiding principle: your fusion/metric math is unchanged except where a baseline
did not match its paper (see BASELINE_AUDIT.md). Everything else added is
orchestration, reporting, and two faithful baselines. New files sit on top of
your existing code.

## New files

Orchestration & reporting
  run_config.py          - the ONE file you edit (output path, model list, dataset paths)
  run_all.py             - run every model x every dataset -> organized outputs + one workbook
  run_one.py             - run ONE model on ONE dataset (per-model checking)
  results_manager.py     - collects every run, 95% CIs (same t-interval as your code),
                           writes master CSV + AIRC-Fusion_Results.xlsx (paper-ready tables)
  qualitative_panels.py  - auto-builds Figure 9/10-style side-by-side panels
  make_matd_profiles.py  - regenerates the Figure-8 MATD temporal plots from per-frame CSVs

Faithful baselines (see BASELINE_AUDIT.md)
  model_2024_panda.py    - FAITHFUL Panda et al. 2024 [5]  (key: panda2024)
  latent_lrr.py          - Python port of the authors' latent_lrr.m (ALM solver)
  train_mdlatlrr.py      - regenerates MDLatLRR's projection matrix L (no MATLAB)
  model_mdlatlrr.py      - FAITHFUL MDLatLRR fusion [59]    (key: mdlatlrr)

Docs
  BASELINE_AUDIT.md      - per-baseline verdict vs the original papers
  TEMPORAL_SCORE_NOTE.md - is the Temporal Instability Score legit? grounding + framing

Unchanged (reviewed, intact): evaluation_metrics.py, dataset_processor.py,
model_2015_v4.py, model_2020_fixed.py, model.py, video_utils.py. (image_fusion_processor.py
only gains two dispatch branches for panda2024 and mdlatlrr.)

## How to run

1. Edit run_config.py: set OUTPUT_ROOT and the dataset paths to your PC. (You do
   NOT edit config.py - that just holds each model's default parameters.)
2. Full sweep:           python run_all.py
   Single check:         python run_one.py --model fcdfusion_vectorized --dataset CAMEL_Seq13 --limit 30
   List valid names:     python run_one.py --list
3. MATD temporal plots:  python make_matd_profiles.py --dataset CAMEL_Seq2

Outputs land in OUTPUT_ROOT/results/:
   <DATASET>/<MODEL>/fused_frames/...            organized images
   AIRC-Fusion_Results.xlsx                       <- paste into the paper
       All_runs / Core_<dataset> (Table 6) / Temporal_Score (Table 7) / FPS / SSIM
   master_metrics_long.csv, run_manifest.csv
   qualitative/...                                panels + MATD_*.png

## MDLatLRR: generate L first (one-time, no MATLAB)
Put the authors' training images (your learning_projection_matrix/training_data)
in a folder, then:
   python train_mdlatlrr.py --unit 8  --training_path ./training_data   # -> models/L_8.npy
   python train_mdlatlrr.py --unit 16 --training_path ./training_data   # optional
Then uncomment "mdlatlrr" in run_config.MODELS (or: python run_one.py --model mdlatlrr --dataset CAMEL_Seq13 --limit 30).
Disclose in the paper: "L retrained in Python from the authors' code and data (lambda=0.4)".

## Reproducibility (lock these before re-running)
1. Run on the desktop PC (Ryzen 5 7600X, 32 GB) - the exact CPU in your Methods.
   The RTX 4060 stays disabled (code forces device='cpu'). Raspberry Pi results
   can't be re-run (you no longer have it) - keep them as originally reported.
2. WORKERS = 1 for timing (already default) -> clean, reproducible per-frame FPS.
3. OpenCV: install ONLY opencv-contrib-python (you currently have both opencv-python
   AND opencv-contrib-python, which clobbers cv2.ximgproc and changes Model2020/GFF):
      pip uninstall -y opencv-python opencv-contrib-python
      pip install opencv-contrib-python==4.11.0.86
   Confirm: python -c "import cv2; cv2.ximgproc"  (no error)
4. Reported "processing time" includes the shared align/resize/BGR preprocessing,
   applied equally to all models - state this in Methods.

## Baseline status (details in BASELINE_AUDIT.md)
FCDFusion verified | Model2015/Connah faithful | Model2024/Panda fixed (panda2024)
| MDLatLRR implemented (mdlatlrr, once L exists) | Model2020 relabelled GFF
(Guided-Filter Saliency; cite Li et al. 2013, not Zhang).

## Dependencies
No new dependencies beyond your requirements.txt (numpy, pandas, scipy,
scikit-image, scikit-learn, PyWavelets, openpyxl, opencv-contrib-python, torch,
torchvision, tqdm) plus matplotlib for make_matd_profiles.py. Apply the OpenCV fix above.

## UPDATE (16 Jun)
New: model_densefuse_official.py (faithful DenseFuse [47], key `densefuse_official`),
validate_citations.py (Crossref DOI checker you run locally).
DenseFuse setup: download the pretrained *.model from github.com/hli1221/densefuse-pytorch
(models/ folder) -> save as models/densefuse.model. Then it runs as model key
`densefuse_official` (already in run_config.MODELS).
Renames: Model2020 -> "Model2013" (GFF, Li et al. 2013 [6]). Final 9 models:
Vectorized FCDFusion (proposed), FCDFusion (Pixel) control, Model2013/Model2015/
Model2024(Panda)/MDLatLRR (traditional), DenseFuse/VGG-19/ResNet-152 (DL).
