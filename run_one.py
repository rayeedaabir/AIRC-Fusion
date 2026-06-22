"""
run_one.py  (NEW - run ONE model on ONE dataset, for per-model checking)
------------------------------------------------------------------------
Use this to sanity-check a single model on a single dataset (e.g. eyeball the
fused frames and metrics for just Vectorized FCDFusion on CAMEL_Seq13) before
launching the full sweep with run_all.py.

Examples:
    python run_one.py --model fcdfusion_vectorized --dataset CAMEL_Seq13
    python run_one.py --model panda2024 --dataset FLIR_ADAS --limit 30
    python run_one.py --list          # show available model + dataset names

Outputs land in the same organized layout as run_all.py:
    <OUTPUT_ROOT>/results/<DATASET>/<MODEL>/fused_frames/...
and a single-run workbook:
    <OUTPUT_ROOT>/results/_single_runs/<DATASET>__<MODEL>.xlsx
"""

import argparse
import os
import run_config as RC
from image_fusion_processor import ImageFusionProcessor
from dataset_processor import DatasetProcessor
from results_manager import ResultsManager
from run_all import model_kwargs, pairing_kwargs, assemble_video


def main():
    ap = argparse.ArgumentParser(description="Run one fusion model on one dataset.")
    ap.add_argument("--model", help="model key, e.g. fcdfusion_vectorized (see run_config.MODELS)")
    ap.add_argument("--dataset", help="dataset name, e.g. CAMEL_Seq13 (see run_config.DATASETS)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N frames")
    ap.add_argument("--video", action="store_true", help="also assemble the fused .mp4")
    ap.add_argument("--list", action="store_true", help="list available models and datasets, then exit")
    args = ap.parse_args()

    if args.list or not args.model or not args.dataset:
        print("Available models:")
        for m in RC.MODELS + ["panda2024", "mdlatlrr"]:
            print("   ", m, "->", RC.MODEL_LABELS.get(m, m))
        print("\nAvailable datasets:")
        for d in RC.DATASETS:
            print("   ", d["name"])
        if not (args.model and args.dataset):
            return

    ds = next((d for d in RC.DATASETS if d["name"] == args.dataset), None)
    if ds is None:
        raise SystemExit(f"Dataset '{args.dataset}' not found. Use --list to see names.")

    label = RC.MODEL_LABELS.get(args.model, args.model)
    results_dir = os.path.join(RC.OUTPUT_ROOT, "results")
    frames_dir = os.path.join(results_dir, ds["name"], args.model, "fused_frames")
    os.makedirs(frames_dir, exist_ok=True)

    print(f"\nRunning  {label}  on  {ds['name']}  (limit={args.limit or 'all'})")
    proc = ImageFusionProcessor(core_fusion_method=args.model,
                                target_width=ds.get("target_w"), target_height=ds.get("target_h"),
                                **model_kwargs())
    dp = DatasetProcessor(fusion_processor=proc, max_workers=RC.WORKERS)
    agg = dp.process_dataset_directory(ds["visible_dir"], ds["infrared_dir"],
                                       output_dir=frames_dir, limit=args.limit, **pairing_kwargs(ds))

    if not agg or agg.get("processed_successfully", 0) == 0:
        raise SystemExit("No frames processed - check the dataset paths in run_config.py.")

    single_dir = os.path.join(results_dir, "_single_runs")
    rm = ResultsManager(single_dir)
    rm.add_run(ds["name"], label, agg)
    # name the workbook per run so single runs don't overwrite each other
    csv = os.path.join(single_dir, f"{ds['name']}__{args.model}.csv")
    rm.long_dataframe().to_csv(csv, index=False)
    print(f"\nSingle-run metrics written to: {csv}")
    print(rm.long_dataframe()[["dataset", "model", "fps_mean", "ssim_accuracy_mean",
                               "temporal_instability_score_mean"]].to_string(index=False))

    if args.video:
        import video_utils, glob as _g
        out_dir = os.path.join(results_dir, ds["name"], args.model); fps = ds.get("video_fps", 30)
        video_utils.assemble_fused_video(frames_dir, os.path.join(out_dir, "fused_video.mp4"), fps)
        sbs = os.path.join(out_dir, "comparison_sbs.mp4")
        if ds["kind"] == "flir":
            video_utils.assemble_side_by_side_video(ds["visible_dir"], ds["infrared_dir"], frames_dir, sbs, fps,
                                                    json_map_path=ds.get("json_map_path"))
        else:
            bns = sorted(os.path.basename(p) for p in _g.glob(os.path.join(frames_dir, "*.jpg")))
            video_utils.assemble_side_by_side_video(ds["visible_dir"], ds["infrared_dir"], frames_dir, sbs, fps,
                                                    ordered_visible_basenames_for_sbs=bns)
        print("fused + side-by-side videos written.")


if __name__ == "__main__":
    main()
