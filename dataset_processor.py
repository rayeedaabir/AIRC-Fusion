import cv2
import numpy as np
import os
import glob 
from tqdm import tqdm
import json
import config
from concurrent.futures import ThreadPoolExecutor
from image_fusion_processor import ImageFusionProcessor as ImageFusionProcessor
from evaluation_metrics import calculate_all_metrics

# --- Helper to parse CAMEL.TXT (can be here or in a utils file) ---
def parse_camel_txt(txt_path: str) -> dict:
    frame_data = {} # Key: frame_num (int), Value: can be simple True or dict of other info
    if not txt_path or not os.path.exists(txt_path):
        print(f"Warning: CAMEL.txt file not found: {txt_path}")
        return frame_data
    try:
        with open(txt_path, 'r') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split()
                if not parts: continue # Skip empty lines
                try:
                    frame_num = int(parts[0])
                    if frame_num not in frame_data:
                         frame_data[frame_num] = True 
                except ValueError:
                    print(f"Warning: Could not parse frame number from line {line_num+1} in {txt_path}: '{line.strip()}'")
                    continue
    except IOError as e:
        print(f"Error reading CAMEL txt file {txt_path}: {e}")
    return frame_data

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist() # Convert arrays to lists
        return super(NumpyJSONEncoder, self).default(obj)

class DatasetProcessor:
    def __init__(self, fusion_processor=None, max_workers=None):
        if fusion_processor is None:
            print("Warning: DatasetProcessor initialized without a specific fusion_processor. Using defaults.")
            self.fusion_processor = ImageFusionProcessor(
                gaussian_kernel_size=config.DEFAULT_KERNEL_SIZE,
                base_weight=config.DEFAULT_BASE_WEIGHT,
                detail_method=config.DEFAULT_DETAIL_METHOD
            )
        else:
            self.fusion_processor = fusion_processor
        
        # Use a sensible default for max_workers if None
        self.max_workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
        self.results_cache = {}

    def process_image_pair(self, visible_path, infrared_path, output_path=None):
        visible_orig = cv2.imread(visible_path) 
        infrared_orig = cv2.imread(infrared_path)
        if visible_orig is None: raise FileNotFoundError(f"Error loading visible image from {visible_path}")
        if infrared_orig is None: raise FileNotFoundError(f"Error loading infrared image from {infrared_path}")
        fused_uint8, proc_time = self.fusion_processor.process_frame(visible_orig, infrared_orig)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, fused_uint8)
        eval_metrics = calculate_all_metrics(fused_uint8, visible_orig, infrared_orig)
        combined_metrics = {
            'proc_time': proc_time, 'fps': 1.0 / proc_time if proc_time > 0 else float('inf'),
            'visible_path': visible_path, 'infrared_path': infrared_path, 'output_path': output_path
        }
        combined_metrics.update(eval_metrics)
        return fused_uint8, combined_metrics

    def process_dataset_directory(self, 
                                visible_dir: str, 
                                infrared_dir: str, 
                                output_dir: str = None, 
                                json_map_path: str = None,
                                seq_vis_txt_path: str = None,  
                                seq_ir_txt_path: str = None,   
                                image_filename_format: str = "{:06d}.jpg", 
                                pattern: str = "*.jpg", # Fallback for globbing
                                limit: int = None):

        visible_paths = []
        infrared_paths = []
        ordered_rgb_basenames_for_output = [] # Used for naming output fused images

        if seq_vis_txt_path and seq_ir_txt_path: # CAMEL .txt file logic
            print(f"Using CAMEL .txt files for sequencing: VIS='{os.path.basename(seq_vis_txt_path)}', IR='{os.path.basename(seq_ir_txt_path)}'")
            vis_frame_data = parse_camel_txt(seq_vis_txt_path)
            ir_frame_data = parse_camel_txt(seq_ir_txt_path)

            if not vis_frame_data or not ir_frame_data:
                print("Error: Could not parse one or both CAMEL sequence .txt files, or files are empty.")
                return {'image_count': 0, 'processed_successfully': 0}

            common_frame_numbers = sorted(list(set(vis_frame_data.keys()) & set(ir_frame_data.keys())))
            print(f"Found {len(common_frame_numbers)} common frame numbers between CAMEL sequence files.")
            
            count = 0
            for frame_num in common_frame_numbers:
                if limit is not None and count >= limit: break

                vis_img_basename = image_filename_format.format(frame_num)
                ir_img_basename = image_filename_format.format(frame_num) 

                vis_path = os.path.join(visible_dir, vis_img_basename)
                ir_path = os.path.join(infrared_dir, ir_img_basename)

                if os.path.exists(vis_path) and os.path.exists(ir_path):
                    visible_paths.append(vis_path)
                    infrared_paths.append(ir_path)
                    ordered_rgb_basenames_for_output.append(vis_img_basename) # Important for output name
                    count += 1
                else:
                    # print(f"Warning: CAMEL frame {frame_num} - Image pair not found. V:{vis_path} (Exists:{os.path.exists(vis_path)}), I:{ir_path} (Exists:{os.path.exists(ir_path)})")
                    pass # Keep console cleaner for many files

            if not visible_paths:
                print("No valid image pairs found using CAMEL .txt files and existing image files.")
                return {'image_count': 0, 'processed_successfully': 0}

        elif json_map_path and os.path.exists(json_map_path):
            print(f"Using JSON map for frame pairing: {json_map_path}")
            try:
                with open(json_map_path, 'r') as f: frame_map = json.load(f)
            except Exception as e: print(f"ERROR decoding JSON: {e}"); return None
            count = 0
            for rgb_fname, thermal_fname in frame_map.items():
                if limit is not None and count >= limit: break
                vis_path = os.path.join(visible_dir, rgb_fname); ir_path = os.path.join(infrared_dir, thermal_fname)
                if os.path.exists(vis_path) and os.path.exists(ir_path):
                    visible_paths.append(vis_path); infrared_paths.append(ir_path)
                    ordered_rgb_basenames_for_output.append(rgb_fname); count += 1
            if not visible_paths: print("No valid pairs from JSON map."); return None
        
        else: # Fallback to globbing (if neither JSON nor CAMEL txt paths provided)
            print(f"Using glob pattern '{pattern}' for frame pairing.")
            glob_vis = sorted(glob.glob(os.path.join(visible_dir, pattern)))
            glob_ir = sorted(glob.glob(os.path.join(infrared_dir, pattern)))
            if not glob_vis or not glob_ir: print("Globbing found no images."); return None
            min_len = min(len(glob_vis), len(glob_ir))
            if limit is not None: min_len = min(min_len, limit)
            visible_paths = glob_vis[:min_len]; infrared_paths = glob_ir[:min_len]
            ordered_rgb_basenames_for_output = [os.path.basename(p) for p in visible_paths]
            if not visible_paths: print("No pairs after globbing/limit."); return None

        # --- Common processing part ---
        if not visible_paths:
            print("No image pairs to process.")
            return {'image_count': 0, 'processed_successfully': 0}

        print(f"Preparing to process {len(visible_paths)} image pairs using {self.max_workers} workers...")
        output_paths_for_saving_fused = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_paths_for_saving_fused = [os.path.join(output_dir, bn) for bn in ordered_rgb_basenames_for_output]
        
        all_individual_metrics_list = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for i, (vis_path, ir_path) in enumerate(zip(visible_paths, infrared_paths)):
                out_path_to_save = output_paths_for_saving_fused[i] if output_paths_for_saving_fused else None
                future = executor.submit(self.process_image_pair, vis_path, ir_path, out_path_to_save)
                futures[future] = (vis_path, ir_path) # Store input paths for better error reporting

            for i, future in enumerate(tqdm(futures, desc="Processing images")):
                vis_p, ir_p = futures[future]
                try:
                    _, frame_metrics_dict = future.result()
                    all_individual_metrics_list.append(frame_metrics_dict)
                except Exception as e:
                    print(f"\nError processing pair V:{os.path.basename(vis_p)}, I:{os.path.basename(ir_p)}: {e}")

        if not all_individual_metrics_list: # ...
            return {'image_count': len(visible_paths), 'processed_successfully': 0}
        aggregated_results = {'image_count': len(visible_paths), 'processed_successfully': len(all_individual_metrics_list)}
        metric_keys_to_aggregate = [
            'proc_time', 'fps', 'entropy', 'mutual_info_src', 'std_dev', 'noise', 
            'wavelet_std_cA', 'wavelet_std_cH', 'wavelet_std_cV', 'wavelet_std_cD', 
            'edge_sum', 'ssim_accuracy'
        ]
        for key in metric_keys_to_aggregate:
            values = [m[key] for m in all_individual_metrics_list if key in m and not np.isnan(m[key])]
            if values:
                aggregated_results[f'avg_{key}'] = np.mean(values)
                aggregated_results[f'range_{key}'] = f"{np.min(values):.3f} - {np.max(values):.3f}" # Range on one line
            else:
                aggregated_results[f'avg_{key}'] = np.nan; aggregated_results[f'range_{key}'] = "N/A"
        aggregated_results['processed_pairs_details'] = all_individual_metrics_list
        self.results_cache = aggregated_results
        print("\nDataset Processing Summary (Aggregated Metrics):")
        for key, value in self.results_cache.items():
            if key == 'processed_pairs_details': continue
            print(f"  {key.replace('_', ' ').title()}: {value}") # Title case for readability
        
        # saving to file.txt file
        if output_dir:
            metrics_filename = f"metrics_summary_{config.ACTIVE_DATASET_NAME}_{self.fusion_processor.core_fusion_method}.json"
            metrics_filename_base = "dataset_metrics_summary"
            if hasattr(self.fusion_processor, 'core_fusion_method'):
                metrics_filename_base += f"_{self.fusion_processor.core_fusion_method}"
            metrics_save_path = os.path.join(output_dir, f"{metrics_filename_base}.json")
            try:
                saveable_metrics = {k: v for k, v in aggregated_results.items() if k != 'processed_pairs_details'}
                with open(metrics_save_path, 'w') as f:
                    json.dump(saveable_metrics, f, indent=4, cls=NumpyJSONEncoder) # Handle NumPy types
                print(f"Aggregated metrics saved to: {metrics_save_path}")
            except Exception as e:
                print(f"Error saving metrics to JSON: {e}")
            # Save the detailed per-frame metrics to a separate CSV file
            if output_dir and all_individual_metrics_list:
                try:
                    # Use pandas, which is already a dependency in your project
                    import pandas as pd 
                    
                    # Construct a detailed filename
                    csv_filename = metrics_filename_base + "_per_frame.csv"
                    csv_save_path = os.path.join(output_dir, csv_filename)
                    
                    # Convert the list of dictionaries to a DataFrame and save
                    per_frame_df = pd.DataFrame(all_individual_metrics_list)
                    per_frame_df.to_csv(csv_save_path, index=False)
                    
                    print(f"Per-frame metrics for confidence intervals saved to: {csv_save_path}")
                except Exception as e:
                    print(f"Error saving per-frame metrics to CSV: {e}")
        return self.results_cache

    def process_video_dataset(self, visible_video, infrared_video, output_video=None, frame_limit=None, frame_step=1, display=False):
        print(f"Starting video file processing...")
        visible_cap = cv2.VideoCapture(visible_video)
        infrared_cap = cv2.VideoCapture(infrared_video)
        if not visible_cap.isOpened(): raise FileNotFoundError(f"Could not open visible video: {visible_video}")
        if not infrared_cap.isOpened(): raise FileNotFoundError(f"Could not open infrared video: {infrared_video}")
        
        # ... (video properties logic) ...
        frame_count_vis = int(visible_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count_ir = int(infrared_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_to_consider = min(frame_count_vis, frame_count_ir)
        width = int(visible_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(visible_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = visible_cap.get(cv2.CAP_PROP_FPS)
        print(f"Video properties: {width}x{height}, ~{fps_in:.2f} FPS, {total_frames_to_consider} potential frames")
        effective_frame_limit = total_frames_to_consider
        if frame_limit is not None and frame_limit > 0:
            effective_frame_limit = min(total_frames_to_consider, frame_limit)
        
        writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_fps = fps_in / frame_step if fps_in > 0 and frame_step > 0 else 30 
            if output_fps <=0: output_fps = 30
            writer = cv2.VideoWriter(output_video, fourcc, output_fps, (width, height))

        all_individual_metrics_list_video = [] # Store metrics for each processed video frame
        
        num_iterations = 0
        if frame_step > 0:
             num_iterations = (effective_frame_limit + frame_step -1) // frame_step if effective_frame_limit > 0 else 0

        with tqdm(total=num_iterations, desc="Processing video frames") as pbar:
            for frame_idx in range(effective_frame_limit):
                if frame_idx >= total_frames_to_consider: break
                ret_vis, visible_frame_orig = visible_cap.read() # These are uint8
                ret_ir, infrared_frame_orig = infrared_cap.read() # These are uint8
                
                if not ret_vis or not ret_ir:
                    print(f"Reached end of one video stream at frame {frame_idx}.")
                    break
                
                if frame_idx % frame_step == 0:
                    # Process frame expects uint8 images
                    fused_frame_uint8, proc_time = self.fusion_processor.process_frame(
                        visible_frame_orig, infrared_frame_orig
                    )
                    
                    # Calculate metrics for this frame
                    eval_metrics_video = calculate_all_metrics(
                        fused_frame_uint8, visible_frame_orig, infrared_frame_orig
                    )
                    
                    current_frame_all_metrics = {
                        'frame_index': frame_idx,
                        'proc_time': proc_time,
                        'fps': 1.0 / proc_time if proc_time > 0 else float('inf')
                    }
                    current_frame_all_metrics.update(eval_metrics_video)
                    all_individual_metrics_list_video.append(current_frame_all_metrics)
                    
                    if writer:
                        writer.write(fused_frame_uint8)
                    if display:
                        # ... your display logic ...
                        display_combo = np.hstack((visible_frame_orig, infrared_frame_orig, fused_frame_uint8))
                        cv2.imshow("Visible | Infrared | Fused Output", display_combo)
                        if cv2.waitKey(1) & 0xFF == ord('q'): break
                    pbar.update(1)
        
        # ... (cleanup: release caps, writer, destroy windows) ...
        visible_cap.release()
        infrared_cap.release()
        if writer: writer.release()
        if display: cv2.destroyAllWindows()
        
        # Aggregate metrics for video
        if not all_individual_metrics_list_video:
            print("No video frames processed or metrics collected.")
            return {'processed_frames_count': 0}

        aggregated_results_video = {
            'processed_frames_count': len(all_individual_metrics_list_video)
        }
        metric_keys_to_aggregate = [ # Same keys as for dataset
            'proc_time', 'fps', 'entropy', 'mutual_info_src', 'std_dev', 
            'noise', 'wavelet_std_cA', 'wavelet_std_cH', 'wavelet_std_cV', 
            'wavelet_std_cD', 'edge_sum', 'ssim_accuracy'
        ]
        for key in metric_keys_to_aggregate:
            values = [m[key] for m in all_individual_metrics_list_video if key in m and not np.isnan(m[key])]
            if values:
                aggregated_results_video[f'avg_{key}'] = np.mean(values)
                aggregated_results_video[f'min_{key}'] = np.min(values)
                aggregated_results_video[f'max_{key}'] = np.max(values)
            else:
                aggregated_results_video[f'avg_{key}'] = np.nan
                aggregated_results_video[f'min_{key}'] = np.nan
                aggregated_results_video[f'max_{key}'] = np.nan

        self.results_cache = aggregated_results_video

        print("\nVideo Processing Summary (Aggregated Metrics):")
        for key, value in self.results_cache.items():
            if isinstance(value, float): print(f"  {key}: {value:.4f}")
            else: print(f"  {key}: {value}")
        
        return self.results_cache
