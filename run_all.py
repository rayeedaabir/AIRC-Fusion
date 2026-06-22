"""
run_all.py  (NEW - one-command benchmark sweep)
------------------------------------------------------------------
Runs every model in run_config.MODELS over every dataset in
run_config.DATASETS, writes organized outputs, and collates ALL results into
one Excel workbook + CSV (no terminal copy-paste).

Folder layout produced:
  <OUTPUT_ROOT>/results/
      <DATASET>/<MODEL>/fused_frames/*.jpg
      <DATASET>/<MODEL>/fused_video.mp4          (if MAKE_VIDEOS)
      <DATASET>/<MODEL>/dataset_metrics_summary_<model>_per_frame.csv   (existing)
      master_metrics_long.csv
      AIRC-Fusion_Results.xlsx                    (paper-ready tables)
      qualitative/<DATASET>_frameXXXX.png         (if MAKE_QUALITATIVE_PANELS)
      run_manifest.csv                            (what ran, timings, errors)

Run:  python run_all.py
Model fusion math and metric math are UNCHANGED; this only orchestrates and
exports. Use WORKERS = 1 in run_config for clean, reproducible per-frame FPS.
"""

import os
import argparse
import time
import traceback
import glob as _glob
import cv2
import pandas as pd

import config
import run_config as RC
from image_fusion_processor import ImageFusionProcessor
from dataset_processor import DatasetProcessor
from results_manager import ResultsManager
import video_utils


def model_kwargs():
    """Model parameters pulled from config.py defaults so runs match the paper."""
    return dict(
        gaussian_kernel_size=config.DEFAULT_KERNEL_SIZE_M2024,
        base_weight=config.DEFAULT_BASE_WEIGHT_M2024,
        detail_method=config.DEFAULT_DETAIL_METHOD_M2024,
        m2020_decomp_ksize=config.DEFAULT_M2020_DECOMP_KSIZE,
        m2020_decomp_sigma=config.DEFAULT_M2020_DECOMP_SIGMA,
        m2020_base_lap_ksize=config.DEFAULT_M2020_BASE_LAP_KSIZE,
        m2020_base_gauss_ksize=config.DEFAULT_M2020_BASE_GAUSS_KSIZE,
        m2020_base_gauss_sigma=config.DEFAULT_M2020_BASE_GAUSS_SIGMA,
        m2020_guided_radius=config.DEFAULT_M2020_GUIDED_RADIUS,
        m2020_guided_eps=config.DEFAULT_M2020_GUIDED_EPS,
        m2020_detail_median_ksize=config.DEFAULT_M2020_DETAIL_MEDIAN_KSIZE,
        m2020_detail_lap_ksize=config.DEFAULT_M2020_DETAIL_LAP_KSIZE,
        m2020_detail_weight_ksize=config.DEFAULT_M2020_DETAIL_WEIGHT_KSIZE,
        m2020_detail_weight_sigma=config.DEFAULT_M2020_DETAIL_WEIGHT_SIGMA,
        m2015_grad_ksize=config.DEFAULT_M2015_GRAD_KSIZE,
    )


def pairing_kwargs(ds):
    """Translate a run_config dataset entry into process_dataset_directory kwargs."""
    if ds["kind"] == "flir":
        return dict(json_map_path=ds.get("json_map_path"), pattern=ds.get("pattern", "*.jpg"))
    if ds["kind"] == "camel":
        return dict(seq_vis_txt_path=ds.get("seq_vis_txt"), seq_ir_txt_path=ds.get("seq_ir_txt"),
                    image_filename_format=ds.get("filename_format", "{:06d}.jpg"))
    return dict(pattern=ds.get("pattern", "*.jpg"))


def assemble_video(frames_dir, out_path, fps):
    frames = sorted(_glob.glob(os.path.join(frames_dir, "*.jpg")) + _glob.glob(os.path.join(frames_dir, "*.png")))
    if not frames:
        return False
    first = cv2.imread(frames[0])
    if first is None:
        return False
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        img = cv2.imread(f)
        if img is not None:
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            vw.write(img)
    vw.release()
    return True


def _make_videos(ds, model, frames_dir, results_dir):
    """Build the per-model fused .mp4 and the RGB|Thermal|Fused side-by-side .mp4."""
    out_dir = os.path.join(results_dir, ds["name"], model)
    fps = ds.get("video_fps", 30)
    try:
        video_utils.assemble_fused_video(frames_dir, os.path.join(out_dir, "fused_video.mp4"), fps)
    except Exception as e:
        print(f"[video] fused video failed: {e}")
    try:
        sbs = os.path.join(out_dir, "comparison_sbs.mp4")
        if ds["kind"] == "flir":
            video_utils.assemble_side_by_side_video(
                ds["visible_dir"], ds["infrared_dir"], frames_dir, sbs, fps,
                json_map_path=ds.get("json_map_path"))
        else:
            bns = sorted(os.path.basename(p) for p in _glob.glob(os.path.join(frames_dir, "*.jpg")))
            video_utils.assemble_side_by_side_video(
                ds["visible_dir"], ds["infrared_dir"], frames_dir, sbs, fps,
                ordered_visible_basenames_for_sbs=bns)
    except Exception as e:
        print(f"[video] side-by-side video failed: {e}")


def main():
    ap = argparse.ArgumentParser(description="Run the fusion sweep (all models x datasets).")
    ap.add_argument("--dataset", default=None,
                    help="run ONLY this dataset (e.g. CAMEL_Seq2). Results still accumulate "
                         "into the same workbook, so you can do one dataset per session.")
    ap.add_argument("--force", action="store_true",
                    help="re-run (model,dataset) pairs even if they are already in the results.")
    args = ap.parse_args()

    results_dir = os.path.join(RC.OUTPUT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    rm = ResultsManager(results_dir)          # loads any prior results -> accumulates
    manifest = []

    datasets = [d for d in RC.DATASETS if (args.dataset is None or d["name"] == args.dataset)]
    if args.dataset and not datasets:
        raise SystemExit(f"Dataset '{args.dataset}' not found. Options: {[d['name'] for d in RC.DATASETS]}")

    for ds in datasets:
        for model in RC.MODELS:
            label = RC.MODEL_LABELS.get(model, model)
            if not args.force and rm.done(ds["name"], label):
                print(f"[skip] {ds['name']} x {label} already done (use --force to redo)")
                continue
            frames_dir = os.path.join(results_dir, ds["name"], model, "fused_frames")
            os.makedirs(frames_dir, exist_ok=True)
            print(f"\n{'='*70}\n  {ds['name']}  x  {label}\n{'='*70}")
            t0 = time.time(); status, err = "ok", ""
            try:
                proc = ImageFusionProcessor(
                    core_fusion_method=model,
                    target_width=ds.get("target_w"), target_height=ds.get("target_h"),
                    **model_kwargs(),
                )
                dp = DatasetProcessor(fusion_processor=proc, max_workers=RC.WORKERS)
                agg = dp.process_dataset_directory(
                    ds["visible_dir"], ds["infrared_dir"],
                    output_dir=frames_dir,
                    limit=RC.LIMIT_OVERRIDE.get(model, {}).get(ds["name"], ds.get("limit")),
                    **pairing_kwargs(ds),
                )
                if not agg or agg.get("processed_successfully", 0) == 0:
                    status, err = "no_frames", "0 frames processed (check paths)"
                else:
                    rm.add_run(ds["name"], label, agg)
                    rm.save_csv()                 # checkpoint after EVERY model (crash-safe)
                    if RC.MAKE_VIDEOS:
                        _make_videos(ds, model, frames_dir, results_dir)
            except Exception as e:
                status, err = "error", str(e); traceback.print_exc()
            manifest.append({"dataset": ds["name"], "model": label, "status": status,
                             "elapsed_s": round(time.time() - t0, 2), "error": err})

    rm.finalize()                                 # rebuild CSV + Excel from ALL accumulated rows
    man_path = os.path.join(results_dir, "run_manifest.csv")
    new_man = pd.DataFrame(manifest)
    if os.path.exists(man_path):                  # append to any prior manifest
        try:
            new_man = pd.concat([pd.read_csv(man_path), new_man], ignore_index=True)
        except Exception:
            pass
    new_man.to_csv(man_path, index=False)
    print("\n[run_all] this session:")
    print(pd.DataFrame(manifest).to_string(index=False))

    if RC.MAKE_QUALITATIVE_PANELS:
        try:
            from qualitative_panels import build_all_panels
            build_all_panels(results_dir)
        except Exception as e:
            print(f"[run_all] qualitative panels skipped: {e}")


if __name__ == "__main__":
    main()
