# Direct Imports
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import time
import os
import glob
from tqdm import tqdm
import traceback
import threading
import logging

# Local module imports
import config 
from image_fusion_processor import ImageFusionProcessor 
from dataset_processor import DatasetProcessor
from benchmark_utils import benchmark_parameter_combinations 
from video_utils import assemble_fused_video, assemble_side_by_side_video

logger = logging.getLogger(__name__)

def parse_info_txt(info_path: str) -> dict: 
    env_info = {}
    if not info_path or not os.path.exists(info_path):
        # logger.warning(f"Info file not found or path not provided: {info_path}")
        return env_info
    try:
        with open(info_path, 'r') as f:
            for line in f:
                if '-' in line:
                    parts = line.strip().split('-', 1)
                    if len(parts) == 2:
                        key = parts[0].strip(); value = parts[1].strip()
                        env_info[key] = value
    except IOError as e:
        logger.error(f"Error reading info file {info_path}: {e}")
    return env_info

def run_generic_pipeline(args):
    """
    Runs a generic dataset processing pipeline based on CLI arguments and active config.
    """
    # config.set_active_dataset_config is called in main_cli
    print(f"--- {config.ACTIVE_DATASET_NAME.upper()} Dataset Processing Pipeline ---")
    print(f"Using Base Path: {config.BASE_PATH}")
    print(f"Visible Frames Dir: {config.VISIBLE_FRAMES_DIR}")
    print(f"Thermal Frames Dir: {config.THERMAL_FRAMES_DIR}")
    print(f"Output Fused Frames Dir: {config.OUTPUT_FUSED_FRAMES_DIR}")

    # Display dataset-specific sequencing info
    if config.ACTIVE_DATASET_NAME == "FLIR":
        print(f"JSON Map File: {config.JSON_MAP_FILE}")
    elif config.ACTIVE_DATASET_NAME == "CAMEL":
        print(f"Seq Vis Txt: {config.SEQ_VIS_TXT_FILE}")
        print(f"Seq IR Txt: {config.SEQ_IR_TXT_FILE}")
        print(f"Image Format: {config.IMAGE_FILENAME_FORMAT}")
        if config.INFO_TXT_FILE and os.path.exists(config.INFO_TXT_FILE):
            env_details = parse_info_txt(str(config.INFO_TXT_FILE))
            if env_details:
                print("\n--- Environmental Information (CAMEL) ---")
                for key, value in env_details.items(): print(f"  {key}: {value}")
                print("-------------------------------------")
    
    # Use CLI args for limit and workers if provided, else pipeline defaults from config
    current_limit = args.limit if args.limit is not None else config.PIPELINE_PROCESS_LIMIT
    current_workers = args.threads if args.threads is not None else config.PIPELINE_NUM_WORKERS
    current_fps_out = args.fps_out if args.fps_out is not None else config.PIPELINE_VIDEO_FPS_OUTPUT
    
    # --- Get target pipeline resolution from resolved args ---
    pipeline_target_w = args.target_width if args.target_width is not None else config.PIPELINE_TARGET_WIDTH
    pipeline_target_h = args.target_height if args.target_height is not None else config.PIPELINE_TARGET_HEIGHT
    
    print(f"Target Processing Resolution for Pipeline: {pipeline_target_w}x{pipeline_target_h}" if pipeline_target_w else "Target Processing Resolution for Pipeline: Native (Visible's)")
    
    print(f"Image Pattern (glob fallback): {config.IMAGE_PATTERN}")
    print(f"Process Limit: {current_limit if current_limit is not None else 'All'}")
    print(f"Max Workers: {current_workers if current_workers is not None else 'Default (CPU based)'}")
    print(f"--------------------------------------\n")

    # Path validations
    if not os.path.isdir(config.VISIBLE_FRAMES_DIR): print(f"ERROR: Visible frames dir not found: {config.VISIBLE_FRAMES_DIR}"); return
    if not os.path.isdir(config.THERMAL_FRAMES_DIR): print(f"ERROR: Thermal frames dir not found: {config.THERMAL_FRAMES_DIR}"); return
    if config.ACTIVE_DATASET_NAME == "FLIR" and (not config.JSON_MAP_FILE or not os.path.exists(config.JSON_MAP_FILE)):
        print(f"ERROR: FLIR mode requires JSON map file, not found: {config.JSON_MAP_FILE}"); return
    if config.ACTIVE_DATASET_NAME == "CAMEL":
        if (not config.SEQ_VIS_TXT_FILE or not os.path.exists(config.SEQ_VIS_TXT_FILE)):
             print(f"ERROR: CAMEL mode requires Seq Vis Txt file, not found: {config.SEQ_VIS_TXT_FILE}"); return
        if (not config.SEQ_IR_TXT_FILE or not os.path.exists(config.SEQ_IR_TXT_FILE)):
             print(f"ERROR: CAMEL mode requires Seq IR Txt file, not found: {config.SEQ_IR_TXT_FILE}"); return


    # Use CLI args for algorithm params if provided, else pipeline defaults
    pipeline_fusion_processor = ImageFusionProcessor(
        core_fusion_method=args.core_fusion_method,
        # model2024 params
        gaussian_kernel_size=args.kernel_m2024, 
        base_weight=args.weight_m2024,        
        detail_method=args.detail_m2024,
        # model2020 params
        m2020_decomp_ksize=args.m2020_decomp_ksize,
        m2020_decomp_sigma=args.m2020_decomp_sigma,
        m2020_base_lap_ksize=args.m2020_base_lap_ksize,
        m2020_base_gauss_ksize=args.m2020_base_gauss_ksize,
        m2020_base_gauss_sigma=args.m2020_base_gauss_sigma,
        m2020_guided_radius=args.m2020_guided_radius,
        m2020_guided_eps=args.m2020_guided_eps,
        m2020_detail_median_ksize=args.m2020_detail_median_ksize,
        m2020_detail_lap_ksize=args.m2020_detail_lap_ksize,
        m2020_detail_weight_ksize=args.m2020_detail_weight_ksize,
        m2020_detail_weight_sigma=args.m2020_detail_weight_sigma,
        # model2015 params
        m2015_grad_ksize=args.m2015_grad_ksize,
        # target resolutions
        target_width=args.target_width, 
        target_height=args.target_height            
        # Add args for other model params here if they are introduced
    )
    
    pipeline_dataset_proc = DatasetProcessor(
        fusion_processor=pipeline_fusion_processor,
        max_workers=current_workers
    )

    print(f"--- 1. Fusing Image Sequence (using {config.ACTIVE_DATASET_NAME} method) ---")
    os.makedirs(config.OUTPUT_FUSED_FRAMES_DIR, exist_ok=True)
    # Debug 1: process_dataset_directory
    print("DEBUG: About to call process_dataset_directory...")
    dataset_metrics = None 
    try:
        dataset_metrics = pipeline_dataset_proc.process_dataset_directory(
            visible_dir=str(config.VISIBLE_FRAMES_DIR),
            infrared_dir=str(config.THERMAL_FRAMES_DIR),
            output_dir=str(config.OUTPUT_FUSED_FRAMES_DIR),
            json_map_path=str(config.JSON_MAP_FILE) if config.JSON_MAP_FILE else None,
            seq_vis_txt_path=str(config.SEQ_VIS_TXT_FILE) if config.SEQ_VIS_TXT_FILE else None, # NEW
            seq_ir_txt_path=str(config.SEQ_IR_TXT_FILE) if config.SEQ_IR_TXT_FILE else None,   # NEW
            image_filename_format=config.IMAGE_FILENAME_FORMAT if config.IMAGE_FILENAME_FORMAT else "{:06d}.jpg", # NEW (provide a fallback format)
            pattern=config.IMAGE_PATTERN, 
            limit=current_limit
        )
        print("DEBUG: process_dataset_directory call completed.") 
        if dataset_metrics: 
            print(f"DEBUG: dataset_metrics received: {type(dataset_metrics)}") 
            print(f"DEBUG: dataset_metrics content (partial): {{'processed_successfully': {dataset_metrics.get('processed_successfully', 'N/A')}, 'image_count': {dataset_metrics.get('image_count', 'N/A')}}}") 
        else: 
            print("DEBUG: dataset_metrics is None after call.")
        if not dataset_metrics or dataset_metrics.get('processed_successfully', 0) == 0:
            print("No images processed or metrics missing. Exiting pipeline before video assembly.")
            return
    except Exception as e:
        print(f"EXCEPTION during dataset processing: {e}"); traceback.print_exc(); return


    ordered_fused_basenames = []
    if dataset_metrics and 'processed_pairs_details' in dataset_metrics:
        for detail in dataset_metrics['processed_pairs_details']:
            if detail.get('output_path'): 
                 ordered_fused_basenames.append(os.path.basename(detail['output_path']))
    
    if not ordered_fused_basenames and dataset_metrics.get('processed_successfully', 0) > 0:
        print("WARNING - Fused images processed, but no output_path details for video ordering.")

    # Video assembly
    print(f"\n--- 2. Assembling {config.ACTIVE_DATASET_NAME} Single Fused Video ---")
    if ordered_fused_basenames:
        assemble_fused_video(
            fused_frames_dir=str(config.OUTPUT_FUSED_FRAMES_DIR),
            output_video_file=str(config.OUTPUT_FUSED_VIDEO_FILE),
            fps=args.fps_out if args.fps_out is not None else config.PIPELINE_VIDEO_FPS_OUTPUT,
            ordered_fused_frame_basenames=ordered_fused_basenames
        )
    else:
        print("Warning: Ordered fused basenames not available or empty, attempting glob for single fused video.")
        assemble_fused_video(
            fused_frames_dir=config.OUTPUT_FUSED_FRAMES_DIR,
            output_video_file=config.OUTPUT_FUSED_VIDEO_FILE,
            fps=config.FLIR_PIPELINE_VIDEO_FPS_OUTPUT,
            image_pattern_glob=config.IMAGE_PATTERN
        )

    print(f"\n--- 3. Assembling {config.ACTIVE_DATASET_NAME} Side-by-Side Comparison Video ---")
    sbs_json_map_arg = None
    sbs_ordered_vis_basenames_arg = None

    if config.ACTIVE_DATASET_NAME == "FLIR":
        if config.JSON_MAP_FILE and os.path.exists(config.JSON_MAP_FILE):
            sbs_json_map_arg = str(config.JSON_MAP_FILE)
        else:
            print("Warning: FLIR mode selected for SBS video, but JSON_MAP_FILE is not valid. SBS video might fail or be incorrect.")
    elif config.ACTIVE_DATASET_NAME == "CAMEL":
        if ordered_fused_basenames: # This list was populated earlier from dataset_metrics
            sbs_ordered_vis_basenames_arg = ordered_fused_basenames
        else:
            print("Warning: CAMEL mode selected for SBS video, but no ordered basenames available from fusion stage. SBS video might fail.")
    success_sbs = assemble_side_by_side_video(
        visible_dir=str(config.VISIBLE_FRAMES_DIR),
        thermal_dir=str(config.THERMAL_FRAMES_DIR),
        fused_dir=str(config.OUTPUT_FUSED_FRAMES_DIR),
        output_video_file=str(config.OUTPUT_SIDE_BY_SIDE_VIDEO_FILE),
        fps=args.fps_out if args.fps_out is not None else config.PIPELINE_VIDEO_FPS_OUTPUT,
        json_map_path=str(config.JSON_MAP_FILE) if config.JSON_MAP_FILE and config.ACTIVE_DATASET_NAME == "FLIR" else None,
        ordered_visible_basenames_for_sbs=ordered_fused_basenames if config.ACTIVE_DATASET_NAME == "CAMEL" else None,
        process_limit=current_limit
    )
    if not success_sbs:
        print("Side-by-side video assembly reported an issue.")
    print(f"\n--- {config.ACTIVE_DATASET_NAME.upper()} Dataset Processing Finished ---")


def main_cli():
    parser = argparse.ArgumentParser(description="Visible-Infrared Image and Video Fusion Tool")
    
    parser.add_argument("--dataset-type", choices=["flir", "camel"], default="flir",                                          help="Type of dataset to process for pipeline mode (flir or camel). Affects path and sequencing logic.")
    parser.add_argument("--base-dataset-path", type=str, default=None,                                                        help="Override the default base path for the selected dataset type.")

    parser.add_argument("--mode", choices=["image", "video", "dataset", "benchmark", "pipeline"], default="image",            help="Processing mode. 'pipeline' runs dataset processing based on --dataset-type.")

    parser.add_argument("--visible",                                                                                          help="Path to visible image/video/directory")
    parser.add_argument("--infrared",                                                                                         help="Path to infrared image/video/directory")
    parser.add_argument("--output",                                                                                           help="Path to save output file or directory")
    
    parser.add_argument("--json-map",                                                                                         help="Path to JSON file for FLIR-like dataset mode.")
    parser.add_argument("--seq-vis-txt",                                                                                      help="Path to sequence .txt for visible images (CAMEL-like dataset mode).")
    parser.add_argument("--seq-ir-txt",                                                                                       help="Path to sequence .txt for infrared images (CAMEL-like dataset mode).")
    parser.add_argument("--image-format", default="{:06d}.jpg",                                                               help="Filename format for CAMEL-like image sequences.")
    parser.add_argument("--info-txt",                                                                                         help="Path to environmental info .txt file (CAMEL-like).")

    parser.add_argument("--core-fusion-method", choices=["model2024", "fcdfusion_vectorized", "fcdfusion_pixel", "model2020", "model2015"], default=None,   help="The main fusion algorithm to use (e.g., model2024, fcdfusion).")
    
    # Parameters specific to 'model2024' 
    g_m2024 = parser.add_argument_group('Model2024 Parameters')
    g_m2024.add_argument("--kernel-m2024", type=int, default=None, dest="kernel_m2024",                                       help="Gaussian kernel size.")
    g_m2024.add_argument("--weight-m2024", type=float, default=None, dest="weight_m2024",                                     help="Base fusion weight.")
    g_m2024.add_argument("--detail-m2024", choices=["max_abs", "weighted", "adaptive"], default=None, dest="detail_m2024",    help="Detail fusion sub-method.")
    
    # Parameters specific to 'model2020'
    g_m2020 = parser.add_argument_group('Model2020 Parameters')
    g_m2020.add_argument("--m2020-decomp-ksize", type=int, default=None,                                                      help="Decomp. Gaussian kernel size.")
    g_m2020.add_argument("--m2020-decomp-sigma", type=float, default=None,                                                    help="Decomp. Gaussian sigma.")
    g_m2020.add_argument("--m2020-base-lap-ksize", type=int, default=None,                                                    help="Base saliency Laplacian ksize.")
    g_m2020.add_argument("--m2020-base-gauss-ksize", type=int, default=None,                                                  help="Base saliency Gaussian ksize.")
    g_m2020.add_argument("--m2020-base-gauss-sigma", type=float, default=None,                                                help="Base saliency Gaussian sigma.")
    g_m2020.add_argument("--m2020-guided-radius", type=int, default=None,                                                     help="Guided filter radius for base weights.")
    g_m2020.add_argument("--m2020-guided-eps", type=float, default=None,                                                      help="Guided filter epsilon for base weights.")
    g_m2020.add_argument("--m2020-detail-median-ksize", type=int, default=None,                                               help="Median blur ksize for detail.")
    g_m2020.add_argument("--m2020-detail-lap-ksize", type=int, default=None,                                                  help="Detail saliency Laplacian ksize.")
    g_m2020.add_argument("--m2020-detail-weight-ksize", type=int, default=None,                                               help="Detail weight Gaussian smooth ksize.")
    g_m2020.add_argument("--m2020-detail-weight-sigma", type=float, default=None,                                             help="Detail weight Gaussian smooth sigma.")
    
    # Parameters specific to 'model2015'
    g_m2015 = parser.add_argument_group('Model2015 SpE Ansatz Parameters')
    g_m2015.add_argument("--m2015-grad-ksize", type=int, default=None,                                                        help="Sobel kernel size for Model2015 gradient calculation.")
    
    parser.add_argument("--pattern", default=config.IMAGE_PATTERN,                                                            help="Glob pattern if not using map/seq files.")
    parser.add_argument("--limit", type=int, default=None,                                                                    help="Max images/frames.")
    parser.add_argument("--kernel", type=int, default=None,                                                                   help="Gaussian kernel size.")
    parser.add_argument("--base-weight", type=float, default=None,                                                            help="Base fusion weight.")
    parser.add_argument("--detail-method", choices=["max_abs", "weighted", "adaptive"], default=None,                         help="Detail fusion method.")
    parser.add_argument("--display", action="store_true",                                                                     help="Display results.")
    parser.add_argument("--threads", type=int, default=None,                                                                  help="Worker threads.")

    parser.add_argument("--target-width", type=int, default=None,                                                             help="Target width for processing in pipeline mode. Overrides config.")
    parser.add_argument("--target-height", type=int, default=None,                                                            help="Target height for processing in pipeline mode. Overrides config.")
    parser.add_argument("--frame-step", type=int, default=1,                                                                  help="Video frame step.")
    parser.add_argument("--fps-out", type=int, default=None,                                                                  help="Output video FPS.")

    args = parser.parse_args()

    # --- Set active dataset configuration based on --dataset-type ---
    config.set_active_dataset_config(args.dataset_type, args.base_dataset_path)

    # Overrides- initialize default values from active config if CLI args are None, allows CLI to override pipeline defaults.
    if args.core_fusion_method is None: args.core_fusion_method = config.PIPELINE_CORE_FUSION_METHOD
    # model2024 params
    if args.kernel_m2024 is None: args.kernel_m2024 = config.PIPELINE_KERNEL_SIZE_M2024
    if args.weight_m2024 is None: args.weight_m2024 = config.PIPELINE_BASE_WEIGHT_M2024
    if args.detail_m2024 is None: args.detail_m2024 = config.PIPELINE_DETAIL_METHOD_M2024
    # model2020 params
    if args.m2020_decomp_ksize is None: args.m2020_decomp_ksize = config.DEFAULT_M2020_DECOMP_KSIZE
    if args.m2020_decomp_sigma is None: args.m2020_decomp_sigma = config.DEFAULT_M2020_DECOMP_SIGMA
    if args.m2020_base_lap_ksize is None: args.m2020_base_lap_ksize = config.DEFAULT_M2020_BASE_LAP_KSIZE
    if args.m2020_base_gauss_ksize is None: args.m2020_base_gauss_ksize = config.DEFAULT_M2020_BASE_GAUSS_KSIZE
    if args.m2020_base_gauss_sigma is None: args.m2020_base_gauss_sigma = config.DEFAULT_M2020_BASE_GAUSS_SIGMA
    if args.m2020_guided_radius is None: args.m2020_guided_radius = config.DEFAULT_M2020_GUIDED_RADIUS
    if args.m2020_guided_eps is None: args.m2020_guided_eps = config.DEFAULT_M2020_GUIDED_EPS
    if args.m2020_detail_median_ksize is None: args.m2020_detail_median_ksize = config.DEFAULT_M2020_DETAIL_MEDIAN_KSIZE
    if args.m2020_detail_lap_ksize is None: args.m2020_detail_lap_ksize = config.DEFAULT_M2020_DETAIL_LAP_KSIZE
    if args.m2020_detail_weight_ksize is None: args.m2020_detail_weight_ksize = config.DEFAULT_M2020_DETAIL_WEIGHT_KSIZE
    if args.m2020_detail_weight_sigma is None: args.m2020_detail_weight_sigma = config.DEFAULT_M2020_DETAIL_WEIGHT_SIGMA
    # model2015 params
    if args.m2015_grad_ksize is None: args.m2015_grad_ksize = config.DEFAULT_M2015_GRAD_KSIZE
    
    if args.target_width is None: args.target_width = config.PIPELINE_TARGET_WIDTH
    if args.target_height is None: args.target_height = config.PIPELINE_TARGET_HEIGHT
    if args.threads is None: args.threads = config.PIPELINE_NUM_WORKERS
    if args.limit is None: args.limit = config.PIPELINE_PROCESS_LIMIT
    if args.fps_out is None: args.fps_out = config.PIPELINE_VIDEO_FPS_OUTPUT

    if args.mode == "pipeline":
        run_generic_pipeline(args) 
        return

    if args.mode != "benchmark" and (not args.visible or not args.infrared):
        parser.error(f"--visible and --infrared paths are required for mode '{args.mode}'.")
    
    # --- Processor Initialization for non-pipeline modes ---
    cli_fusion_processor = ImageFusionProcessor(
        core_fusion_method=args.core_fusion_method,
        # model2024 params
        gaussian_kernel_size=args.kernel_m2024, 
        base_weight=args.weight_m2024,   
        detail_method=args.detail_m2024,
        # model2020 params
        m2020_decomp_ksize=args.m2020_decomp_ksize,
        m2020_decomp_sigma=args.m2020_decomp_sigma,
        m2020_base_lap_ksize=args.m2020_base_lap_ksize,
        m2020_base_gauss_ksize=args.m2020_base_gauss_ksize,
        m2020_base_gauss_sigma=args.m2020_base_gauss_sigma,
        m2020_guided_radius=args.m2020_guided_radius,
        m2020_guided_eps=args.m2020_guided_eps,
        m2020_detail_median_ksize=args.m2020_detail_median_ksize,
        m2020_detail_lap_ksize=args.m2020_detail_lap_ksize,
        m2020_detail_weight_ksize=args.m2020_detail_weight_ksize,
        m2020_detail_weight_sigma=args.m2020_detail_weight_sigma,
        # model2015 params
        m2015_grad_ksize=args.m2015_grad_ksize,
        # target resolutions
        target_width=args.target_width, # Pass for non-pipeline modes too if desired
        target_height=args.target_height,
    )
    
    cli_dataset_processor = DatasetProcessor(
        fusion_processor=cli_fusion_processor, 
        max_workers=args.threads
    )
    
    # --- Mode Dispatch for non-pipeline modes ---
    if args.mode == "benchmark":
        benchmark_parameter_combinations(
            args.visible, args.infrared, args.output 
        )
    elif args.mode == "dataset":
        if args.info_txt and os.path.exists(args.info_txt):
            env_details = parse_info_txt(args.info_txt)
            if env_details:
                print("\n--- Environmental Information ---")
                for key, value in env_details.items(): print(f"  {key}: {value}")
                print("-------------------------------")
        cli_dataset_processor.process_dataset_directory(
            args.visible, args.infrared, args.output,
            json_map_path=args.json_map,
            seq_vis_txt_path=args.seq_vis_txt,      
            seq_ir_txt_path=args.seq_ir_txt,        
            image_filename_format=args.image_format,
            pattern=args.pattern, 
            limit=args.limit
        )

    elif args.mode == "video":
        cli_dataset_processor.process_video_dataset(
            args.visible, args.infrared, args.output,
            frame_limit=args.limit, frame_step=args.frame_step,
            display=args.display
        )

    elif args.mode == "image":
        if not os.path.isfile(args.visible) or not os.path.isfile(args.infrared):
            parser.error("For 'image' mode, --visible and --infrared must be valid files.")
        fused_img, all_metrics_single_image = cli_dataset_processor.process_image_pair(
            args.visible, args.infrared, args.output
        )
        print("\n--- Single Image Fusion Metrics ---")
        if all_metrics_single_image:
            for key, value in all_metrics_single_image.items():
                if key in ['visible_path', 'infrared_path', 'output_path']: 
                    continue
                formatted_key = key.replace('_', ' ').title()
                if isinstance(value, (float, np.floating)): 
                    print(f"  {formatted_key:<30}: {value:.5f}")
                else:
                    print(f"  {formatted_key:<30}: {value}")
        else:
            print("  Error: No metrics dictionary was returned from image processing.")

        if args.output and os.path.exists(args.output): 
            print(f"\nFused image saved to: {args.output}")
        elif args.output:
            print(f"\nNote: Output path specified ({args.output}), but file might not have been saved.")

        if args.display:
            visible_img = cv2.imread(args.visible)
            infrared_img_orig = cv2.imread(args.infrared)
            if visible_img is None or infrared_img_orig is None or fused_img is None:
                print("Error: One or more images could not be loaded for display.")
                return

            plt.figure(figsize=(18, 6))
            plt.subplot(131); plt.imshow(cv2.cvtColor(visible_img, cv2.COLOR_BGR2RGB)); plt.title("Visible"); plt.axis("off")
            plt.subplot(132)
            if len(infrared_img_orig.shape) == 2 or infrared_img_orig.shape[2] == 1: 
                plt.imshow(infrared_img_orig, cmap='gray')
            else: 
                plt.imshow(cv2.cvtColor(infrared_img_orig, cv2.COLOR_BGR2RGB))
            plt.title("Infrared"); plt.axis("off")
            plt.subplot(133); plt.imshow(cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB)); plt.title(f"Fused Output"); plt.axis("off")
            plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main_cli()