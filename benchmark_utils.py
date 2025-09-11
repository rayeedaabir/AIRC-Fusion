import os
import pandas as pd
import numpy as np
from image_fusion_processor import ImageFusionProcessor 
from dataset_processor import DatasetProcessor

def benchmark_parameter_combinations(visible_path, infrared_path, output_dir_base=None):
    gaussian_sizes = [7, 11, 15, 21]
    base_weights = [0.3, 0.5, 0.7]
    detail_methods = ["max_abs", "weighted", "adaptive"] # Add "adaptive" if it's fast enough for benchmark
    
    results_list = []
    is_directory = os.path.isdir(visible_path) and os.path.isdir(infrared_path)
    
    print("Benchmarking parameter combinations...")
    total_combinations = len(gaussian_sizes) * len(base_weights) * len(detail_methods)
    combination_idx = 0
    
    # Define the metric keys we expect from DatasetProcessor (averages for dir, direct for single)
    # These are the base names, 'avg_' prefix will be used for directory results
    metric_keys_from_processor = [
        'proc_time', 'fps', 'entropy', 'mutual_info_src', 'std_dev', 
        'noise', 'wavelet_std_cA', 'wavelet_std_cH', 'wavelet_std_cV', 
        'wavelet_std_cD', 'edge_sum', 'ssim_accuracy'
    ]

    for gs in gaussian_sizes:
        for bw in base_weights:
            for dm in detail_methods:
                combination_idx += 1
                print(f"\nCombination {combination_idx}/{total_combinations}: kernel={gs}, base_weight={bw}, detail_method={dm}")
                
                processor = ImageFusionProcessor(
                    gaussian_kernel_size=gs,
                    base_weight=bw,
                    detail_method=dm
                )
                # For benchmarking, always use a fresh DatasetProcessor instance per combo
                dataset_proc = DatasetProcessor(fusion_processor=processor, max_workers=1) 
                
                current_output_dir_for_combo_fused_images = None
                if output_dir_base and is_directory: # Only save images if processing a directory
                    current_output_dir_for_combo_fused_images = os.path.join(
                        output_dir_base, "benchmark_fused_images", f"gs{gs}_bw{bw}_dm{dm}"
                    )
                    os.makedirs(current_output_dir_for_combo_fused_images, exist_ok=True)
                
                # This dictionary will hold the metrics for the current combination
                # For directory: it will be aggregated_results from process_dataset_directory
                # For single image: it will be frame_metrics_dict from process_image_pair
                full_metrics_dict_from_run = {} 
                images_processed_count = 0

                if is_directory:
                    # process_dataset_directory returns a dict with 'avg_metric', 'min_metric', 'max_metric'
                    full_metrics_dict_from_run = dataset_proc.process_dataset_directory(
                        visible_path, infrared_path, 
                        output_dir=current_output_dir_for_combo_fused_images, # Save sample fused images
                        limit=10 # Process a small number of images for benchmark speed
                    )
                    images_processed_count = full_metrics_dict_from_run.get('processed_successfully', 0)
                else: # Single image pair
                    output_file_path_single_fused = None
                    if output_dir_base: # If output base is given, save the single fused image
                         benchmark_fused_single_dir = os.path.join(output_dir_base, "benchmark_fused_images", "single_image_tests")
                         os.makedirs(benchmark_fused_single_dir, exist_ok=True)
                         output_file_path_single_fused = os.path.join(
                             benchmark_fused_single_dir, f"fused_gs{gs}_bw{bw}_dm{dm}_{os.path.basename(visible_path)}"
                         )
                    
                    _, single_pair_metrics_dict = dataset_proc.process_image_pair(
                        visible_path, infrared_path, output_file_path_single_fused
                    )
                    full_metrics_dict_from_run = single_pair_metrics_dict # This dict has direct metric values
                    images_processed_count = 1 if single_pair_metrics_dict else 0

                if images_processed_count > 0 and full_metrics_dict_from_run:
                    result_entry = {
                        'gaussian_size': gs,
                        'base_weight': bw,
                        'detail_method': dm,
                        'images_processed': images_processed_count
                    }
                    # Populate with evaluation metrics
                    for key_base_name in metric_keys_from_processor:
                        if is_directory:
                            # For directories, dataset_processor returns 'avg_key', 'min_key', 'max_key'
                            # We'll store the average for the benchmark table
                            result_entry[f'avg_{key_base_name}'] = full_metrics_dict_from_run.get(f'avg_{key_base_name}', np.nan)
                        else:
                            # For single image, dataset_processor returns direct metric value
                            result_entry[key_base_name] = full_metrics_dict_from_run.get(key_base_name, np.nan)
                    results_list.append(result_entry)
                else:
                    print(f"  Skipping metrics for combination gs{gs}_bw{bw}_dm{dm} due to processing error or no images processed.")

    if not results_list:
        print("No benchmark results collected.")
        return None

    df = pd.DataFrame(results_list)
    # Sort by a primary performance metric, e.g., avg_ssim_accuracy (if directory) or ssim_accuracy (if single)
    sort_key = 'avg_ssim_accuracy' if is_directory and 'avg_ssim_accuracy' in df.columns else 'ssim_accuracy'
    if sort_key not in df.columns: # Fallback sort key
        sort_key = 'avg_fps' if is_directory and 'avg_fps' in df.columns else 'fps'
        if sort_key not in df.columns: # Ultimate fallback
             sort_key = df.columns[0] if not df.empty else None


    if sort_key and sort_key in df.columns:
        df_sorted = df.sort_values(sort_key, ascending=False)
    else:
        df_sorted = df # No sort if key is not found or df is empty

    print("\nParameter Benchmarking Results:")
    print(df_sorted.to_string())
    
    if output_dir_base:
        os.makedirs(output_dir_base, exist_ok=True) # Ensure base output dir exists
        csv_path = os.path.join(output_dir_base, "benchmark_results_with_eval_metrics.csv")
        df_sorted.to_csv(csv_path, index=False)
        print(f"Saved benchmark results to {csv_path}")
    
    return df_sorted