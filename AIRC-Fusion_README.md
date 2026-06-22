# AIRC-Fusion - pipeline guide

Real-time infrared/visible video fusion benchmark. This guide explains how the
pipeline works now and what every file does. (See BASELINE_AUDIT.md for the
per-baseline fidelity check, TEMPORAL_SCORE_NOTE.md for the temporal metric,
CHANGES_AIRC_README.md for the change log.)

==================================================================
QUICK START (do these once)
==================================================================
1. OpenCV: keep ONLY the contrib build (so Model2013/GFF's guided filter works):
       pip uninstall -y opencv-python opencv-contrib-python
       pip install opencv-contrib-python==4.11.0.86
       python -c "import cv2; cv2.ximgproc"      # must NOT error
2. DenseFuse weights: put the GRAYSCALE file densefuse_gray.model
   (from github.com/hli1221/densefuse-pytorch -> models/) at:
       models/densefuse.model
   (the wrapper also auto-finds densefuse_gray.model in the project root.)
3. MDLatLRR matrix L (one-time, no MATLAB) - see "Generate L_8.npy" below.
4. Edit run_config.py: set OUTPUT_ROOT and the dataset paths to your PC.

Then run:
   python run_all.py                                    # every model x every dataset
   python run_one.py --model fcdfusion_vectorized --dataset CAMEL_Seq13 --limit 30
   python run_one.py --list                             # show valid model/dataset names
   python make_matd_profiles.py --dataset CAMEL_Seq2    # Figure-8 temporal plots

==================================================================
DATA FLOW
==================================================================
run_config.py (your settings)
   -> run_all.py / run_one.py            (orchestrate the sweep)
       -> dataset_processor.py           (pair vis/IR frames, loop, save fused frames)
           -> image_fusion_processor.py  (dispatch to the chosen fusion model)
           -> evaluation_metrics.py      (per-frame metrics + temporal score)
       -> results_manager.py             (collate -> AIRC-Fusion_Results.xlsx + CSV)
       -> qualitative_panels.py          (side-by-side comparison images)
   make_matd_profiles.py                 (temporal MATD plots, run after the sweep)

==================================================================
THE 9 MODELS (model key -> paper label)
==================================================================
PROPOSED
  fcdfusion_vectorized   -> Vectorized FCDFusion   (your method; == FCDFusion [4] Algorithm 1)
ABLATION CONTROL
  fcdfusion_pixel        -> FCDFusion (Pixel)       (same algorithm, sequential loop; shows speedup)
TRADITIONAL (all faithful)
  model2015              -> Model2015               (Connah Spectral Edge [7])
  model2020              -> Model2013               (GFF, Li/Kang/Hu 2013 [6])
  panda2024              -> Model2024               (Panda 2024 [5])
  mdlatlrr               -> MDLatLRR                 (Li/Wu/Kittler 2020 [59]; needs L_8.npy)
DEEP LEARNING
  densefuse_official     -> DenseFuse               (Li/Wu 2019 [47]; needs densefuse.model)
  vgg19                  -> VGG-19                   (backbone [48], compute upper bound)
  resnet152              -> ResNet-152               (backbone [49], compute upper bound)
NOTE: old keys "model2024" (generic) and "densefuse" (mystery net) exist but are
NOT used - do not put them in run_config.MODELS.

==================================================================
FILE-BY-FILE
==================================================================
-- You edit / run --
  run_config.py            The one settings file: OUTPUT_ROOT, MODELS, DATASETS, labels.
  run_all.py               Runs all MODELS x DATASETS; writes organized outputs + workbook.
  run_one.py               Runs ONE model on ONE dataset (sanity checks). "--list" prints names.
  make_matd_profiles.py    Per-frame MATD plots (vis/IR/fused) = Figure 8.
  validate_citations.py    Checks your references' DOIs vs Crossref + DataCite (run locally).

-- Core engine (original, reviewed) --
  image_fusion_processor.py  Dispatcher + every fusion class; converts inputs, times each fuse.
  dataset_processor.py       Frame pairing (FLIR json / CAMEL txt / glob), parallel loop,
                             metric aggregation, temporal pass, per-frame CSV/JSON.
  evaluation_metrics.py      All image metrics (entropy, MI, SSIM, edges, wavelets, noise) +
                             temporal (MATD, Temporal Instability Score).
  config.py                  Dataset paths + each model's DEFAULT parameters (run_all pulls these).
  video_utils.py             Assembles fused frames into .mp4 (optional).
  calculate_confidence_intervals.py   Stand-alone 95% CI from a per-frame CSV (t-interval).
  benchmark_utils.py         Old parameter-tuning sweep (gaussian/weight grid) - not in main flow.
  temporal_comparison.py     Original temporal-analysis helper.
  process_landsat.py         Landsat band->RGB helper (static-image experiments).
  main.py                    Original single-run CLI (argparse). run_one.py supersedes it.
  option.py / model.py       Old mystery "DenseFuse" net + its args - UNUSED now (kept for ref).

-- Faithful baselines (new) --
  model_2015_v4.py           Model2015 = Connah Spectral Edge (SVD gradient ansatz + Poisson). [orig, faithful]
  model_2020_fixed.py        Model2013 = GFF guided-filter saliency two-scale. [orig, relabelled]
  model_2024_panda.py        Model2024 = faithful Panda (Eqs 1-11). [new]
  latent_lrr.py              LatLRR ALM solver (Python port of authors' latent_lrr.m). [new]
  train_mdlatlrr.py          Trains MDLatLRR's projection matrix L -> models/L_8.npy. [new]
  model_mdlatlrr.py          MDLatLRR fusion (nuclear-norm detail + average base). [new]
  model_densefuse_official.py  Faithful DenseFuse (authors' PyTorch arch + L1 fusion). [new]

-- Reporting (new) --
  results_manager.py         Collates all runs, 95% CIs, writes AIRC-Fusion_Results.xlsx
                             (Core_<dataset> = Table 6; Temporal_Score = Table 7; FPS/SSIM).
  qualitative_panels.py      Builds labelled vis|IR|each-model panels (Figures 9/10).

==================================================================
GENERATE L_8.npy (MDLatLRR), one time
==================================================================
1. Find the MDLatLRR training images (your "learning_projection_matrix/training_data"
   folder - the 5 IR/VIS pairs the authors ship). Put them in one folder, e.g.
   ./training_data/  (any images work; they are read as grayscale).
2. Run:
       python train_mdlatlrr.py --unit 8 --training_path ./training_data
   This solves LatLRR (a few minutes) and writes  models/L_8.npy.
3. In run_config.py, remove the # in front of  "mdlatlrr",  in MODELS.
4. Test it:  python run_one.py --model mdlatlrr --dataset CAMEL_Seq13 --limit 20
Disclose in the paper: "L was retrained in Python from the authors' code and data (lambda=0.4)."

==================================================================
OUTPUTS (under OUTPUT_ROOT/results/)
==================================================================
  AIRC-Fusion_Results.xlsx   paper-ready tables (paste straight in)
  master_metrics_long.csv    one row per model x dataset, all metrics + CIs
  run_manifest.csv           what ran, timings, any errors
  <DATASET>/<MODEL>/fused_frames/...   organized fused images
  qualitative/...            comparison panels + MATD_*.png

==================================================================
REPRODUCIBILITY
==================================================================
- Run on the desktop PC (Ryzen 5 7600X, 32 GB) - the CPU named in your Methods.
- GPU stays disabled (code forces device='cpu'); keep it that way (CPU-only thesis).
- WORKERS = 1 for clean per-frame FPS (already default).
- Raspberry Pi results can't be re-run (you no longer have it) - report them as
  originally obtained; do not delete or re-run.

==================================================================
VIDEOS (built automatically when run_config.MAKE_VIDEOS = True)
==================================================================
For each model x dataset, run_all.py now writes:
  <DATASET>/<MODEL>/fused_video.mp4      the fused output as video
  <DATASET>/<MODEL>/comparison_sbs.mp4   Visible | Infrared | Fused side-by-side
After the sweep, build one all-models grid video per dataset:
  python make_comparison_videos.py --dataset CAMEL_Seq2
  -> results/qualitative/COMPARISON_<DATASET>.mp4   (Visible | Infrared | every model)
(run_one.py also writes both per-model videos when you pass --video.)

NOTE on runtime: MDLatLRR is ~10 s/frame (0.1 FPS) at low resolution and far
slower at 1024x1024, so run_config.LIMIT_OVERRIDE caps it to 300 FLIR frames
(metrics on a representative subset; disclose the subset). Increase/remove the
cap in run_config.py if you want the full sequence and have the hours to spare.
