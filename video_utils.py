import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import json
from typing import Optional, List, Tuple

# --- Helper function to collect triplets for SBS when not using JSON map ---
def collect_sbs_triplets_from_basenames(
    visible_dir: str, 
    thermal_dir: str, 
    fused_dir: str,
    ordered_visible_basenames: List[str], # e.g., ["000001.jpg", "000002.jpg", ...]
    limit: Optional[int] = None
)-> Tuple[List[str], List[str], List[str]]:

    """
    Collects paths for visible, thermal, and fused images based on a list of ordered visible basenames.
    Assumes thermal and fused images share the same basename as visible for simplicity with CAMEL.
    """
    vis_paths, therm_paths, fused_paths = [], [], []
    items_to_process = ordered_visible_basenames
    if limit is not None:
        items_to_process = ordered_visible_basenames[:limit]

    print(f"SBS (Basename Mode): Attempting to collect triplets for {len(items_to_process)} base visible filenames.")

    for vis_basename in items_to_process:
        v_p = os.path.join(visible_dir, vis_basename)
        
        # For CAMEL-like datasets where visible, thermal, and fused share the same base filename (e.g., 000001.jpg)
        # If thermal or fused have prefixes/different extensions, this needs more complex logic
        # or specific filename format arguments for thermal/fused.
        t_p = os.path.join(thermal_dir, vis_basename) 
        f_p = os.path.join(fused_dir, vis_basename)
        
        # print(f"DEBUG SBS CAMEL-like: V: '{v_p}' (E:{os.path.exists(v_p)}), T: '{t_p}' (E:{os.path.exists(t_p)}), F: '{f_p}' (E:{os.path.exists(f_p)})")
        if all(os.path.exists(p) for p in [v_p, t_p, f_p]):
            vis_paths.append(v_p)
            therm_paths.append(t_p)
            fused_paths.append(f_p)
        else:
            # print(f"Warning (SBS Basename): Skipping set for {vis_basename}, missing files.")
            pass # Keep console cleaner for many files

    print(f"SBS (Basename Mode): Collected {len(fused_paths)} valid triplets.")
    return vis_paths, therm_paths, fused_paths

def assemble_fused_video(fused_frames_dir: str, 
                         output_video_file: str, 
                         fps: float, 
                         image_pattern_glob: str = "*.jpg",
                         ordered_fused_frame_basenames: Optional[List[str]] = None) -> bool:
    """
    Assembles fused frames into a single video.
    If ordered_fused_frame_basenames is provided, it uses that order.
    Otherwise, it globs and sorts files from fused_frames_dir.
    Prints resize warning only once if frames need resizing.
    """
    
    print(f"\n--- Assembling Single Fused Video ({output_video_file}) ---")
    fused_frame_paths = []
    if ordered_fused_frame_basenames:
        print(f"Using provided order of {len(ordered_fused_frame_basenames)} basenames for fused video.")
        for basename in ordered_fused_frame_basenames:
            path = os.path.join(fused_frames_dir, basename)
            if os.path.exists(path):
                fused_frame_paths.append(path)
            else:
                print(f"Warning: Fused frame {path} from ordered list not found. Skipping.")
        if not fused_frame_paths:
             print("No valid fused frames found from the ordered list.")
             return False
    else:
        # Fallback to globbing and sorting
        glob_search_pattern = os.path.join(fused_frames_dir, image_pattern_glob)
        print(f"Globbing for fused frames with pattern: {glob_search_pattern}")
        fused_frame_paths = sorted(glob.glob(glob_search_pattern))
        if not fused_frame_paths:
            print(f"No fused frames found matching pattern in {fused_frames_dir}.")
            if os.path.isdir(fused_frames_dir): 
                print(f"Files found in {fused_frames_dir} (first 10): {os.listdir(fused_frames_dir)[:10]}")
            return False

    print(f"Found {len(fused_frame_paths)} fused frames to assemble into video.")
    if not fused_frame_paths: 
        print("No fused frames to assemble. Exiting video assembly.")
        return False

    first_frame_loaded = cv2.imread(fused_frame_paths[0])
    if first_frame_loaded is None:
        print(f"Error: Could not read first fused frame for dimensions: {fused_frame_paths[0]}")
        return False

    # Target height and width for the video are determined by the FIRST frame
    video_height, video_width, _ = first_frame_loaded.shape
    
    output_ext = ".mp4"
    fourcc_str = 'mp4v'
    if not output_video_file.lower().endswith(output_ext):
        base, _ = os.path.splitext(output_video_file)
        output_video_file = base + output_ext
        print(f"INFO: Output video (fused) filename changed to {output_video_file} for {output_ext} format.")

    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    
    os.makedirs(os.path.dirname(output_video_file), exist_ok=True)
    out_video = cv2.VideoWriter(output_video_file, fourcc, float(fps), (video_width, video_height)) # Use determined width, height

    if not out_video.isOpened():
        print(f"CRITICAL ERROR: cv2.VideoWriter FAILED to open for fused video: {output_video_file} with FourCC '{fourcc_str}'")
        output_ext_fallback = ".avi"; fourcc_str_fallback = 'XVID'
        base, _ = os.path.splitext(output_video_file)
        output_video_file = base + "_fallback" + output_ext_fallback # Ensure fallback name is different
        print(f"Attempting fallback with FourCC '{fourcc_str_fallback}' to {output_video_file}")
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str_fallback)
        out_video = cv2.VideoWriter(output_video_file, fourcc, float(fps), (video_width, video_height))
        if not out_video.isOpened():
            print(f"CRITICAL ERROR: Fallback cv2.VideoWriter ALSO FAILED for: {output_video_file}")
            return False

    print(f"Creating video: {output_video_file} ({video_width}x{video_height} @ {fps} FPS)")
    
    # --- Moved resize_warning_printed here, before the loop ---
    resize_warning_printed = False 

    for frame_path in tqdm(fused_frame_paths, desc="Writing single fused video"):
        frame = cv2.imread(frame_path)
        if frame is not None:
            # --- This is the correct place for the resize check and conditional print ---
            if frame.shape[1] != video_width or frame.shape[0] != video_height: 
                if not resize_warning_printed:
                    print(f"Warning: Source frames have varying dimensions. Resizing frames to consistent video dimensions ({video_width}x{video_height}). First instance: {os.path.basename(frame_path)} (is {frame.shape[1]}x{frame.shape[0]})")
                    resize_warning_printed = True
                frame = cv2.resize(frame, (video_width, video_height))
            # --- End of resize check block ---
            out_video.write(frame)
        else:
            print(f"Warning: Could not read frame {frame_path} for video writing.")
            
    out_video.release()
    print(f"Video saved to: {output_video_file}")
    return True

def assemble_side_by_side_video(
    visible_dir: str, 
    thermal_dir: str, 
    fused_dir: str,
    output_video_file: str, 
    fps: float,
    json_map_path: Optional[str] = None, # For FLIR-like sequencing
    ordered_visible_basenames_for_sbs: Optional[List[str]] = None, # For CAMEL-like sequencing
    # image_pattern_for_globbing: Optional[str] = "*.jpg", # Used if neither map nor ordered names given
    process_limit: Optional[int] = None
    ):
    """
    Assembles a side-by-side comparison video (RGB | Thermal | Fused).
    Uses json_map_path if provided (FLIR).
    Else, uses ordered_visible_basenames_for_sbs (CAMEL, assumes thermal/fused share basename).
    """
    print(f"\n--- Assembling Side-by-Side Comparison Video ({output_video_file}) ---")

    sbs_vis_paths, sbs_therm_paths, sbs_fused_paths = [], [], []

    if json_map_path and os.path.exists(json_map_path):
        print(f"SBS: Using JSON map for frame pairing: {json_map_path}")
        try:
            with open(json_map_path, 'r') as f: frame_map = json.load(f)
        except Exception as e: print(f"Error loading JSON map {json_map_path}: {e}"); return False
        
        count = 0
        items_to_process = list(frame_map.items())
        if process_limit is not None: items_to_process = items_to_process[:process_limit]

        for rgb_fname, thermal_fname in items_to_process:
            vis_path = os.path.join(visible_dir, rgb_fname)
            therm_path = os.path.join(thermal_dir, thermal_fname)
            fused_path = os.path.join(fused_dir, rgb_fname) # Assumes fused named after RGB (visible)
            if all(os.path.exists(p) for p in [vis_path, therm_path, fused_path]):
                sbs_vis_paths.append(vis_path); sbs_therm_paths.append(therm_path); sbs_fused_paths.append(fused_path)
            else: print(f"Warning (SBS JSON): Skipping set for {rgb_fname}, missing one or more files.")
    
    elif ordered_visible_basenames_for_sbs:
        print(f"SBS: Using provided ordered visible basenames ({len(ordered_visible_basenames_for_sbs)} items).")
        sbs_vis_paths, sbs_therm_paths, sbs_fused_paths = collect_sbs_triplets_from_basenames(
            visible_dir, thermal_dir, fused_dir, ordered_visible_basenames_for_sbs, limit=process_limit
        )
    
    else:
        # Fallback: if you want to glob based on one directory (e.g., fused) and assume others match.
        # This is less robust than explicit lists.
        print(f"SBS: Warning! No JSON map or ordered basenames. Attempting glob based on fused_dir and image_pattern *.jpg.")
        _v, _t, sbs_fused_paths = collect_sbs_triplets_from_basenames( # Using a simpler variant here
            visible_dir, thermal_dir, fused_dir, 
            [os.path.basename(p) for p in sorted(glob.glob(os.path.join(fused_dir, "*.jpg")))], # Get basenames from fused
            limit=process_limit
        )
        # Re-assign to ensure all lists are based on this globbed sequence
        sbs_vis_paths = [os.path.join(visible_dir, os.path.basename(p)) for p in sbs_fused_paths]
        sbs_therm_paths = [os.path.join(thermal_dir, os.path.basename(p)) for p in sbs_fused_paths]


    if not sbs_fused_paths: # Or check sbs_vis_paths
        print("SBS Error: No valid frame sets collected for side-by-side video.")
        return False

    print(f"Found {len(sbs_fused_paths)} sets of frames for side-by-side video assembly.")
    
    first_fused_sbs = cv2.imread(sbs_fused_paths[0])
    if first_fused_sbs is None:
        print(f"Error: Could not read first fused frame for SBS dimensions: {sbs_fused_paths[0]}")
        return False
    panel_h, panel_w, _ = first_fused_sbs.shape
    
    total_width = panel_w * 3
    
    # Ensure output_video_file has an .mp4 extension if using MP4V, or .avi for XVID
    output_ext = ".mp4"
    fourcc_str = 'mp4v'
    if not output_video_file.lower().endswith(output_ext):
        base, _ = os.path.splitext(output_video_file)
        output_video_file = base + output_ext
        print(f"INFO: Output video (SBS) filename changed to {output_video_file} for {output_ext} format.")
    
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    os.makedirs(os.path.dirname(output_video_file), exist_ok=True)
    out_sbs_video = cv2.VideoWriter(output_video_file, fourcc, float(fps), (total_width, panel_h))

    if not out_sbs_video.isOpened():
        print(f"CRITICAL ERROR: cv2.VideoWriter FAILED to open for SBS video: {output_video_file} with FourCC '{fourcc_str}'")
        # Fallback attempt
        output_ext_fallback = ".avi"; fourcc_str_fallback = 'XVID'
        base, _ = os.path.splitext(output_video_file)
        output_video_file = base + "_fallback" + output_ext_fallback
        print(f"Attempting fallback with FourCC '{fourcc_str_fallback}' to {output_video_file}")
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str_fallback)
        out_sbs_video = cv2.VideoWriter(output_video_file, fourcc, float(fps), (total_width, panel_h))
        if not out_sbs_video.isOpened():
            print(f"CRITICAL ERROR: Fallback cv2.VideoWriter ALSO FAILED for SBS: {output_video_file}")
            return False

    print(f"Creating side-by-side video: {output_video_file} ({total_width}x{panel_h} @ {fps} FPS)")

    for i in tqdm(range(len(sbs_fused_paths)), desc="Writing side-by-side video"):
        vis_img = cv2.imread(sbs_vis_paths[i])
        therm_img_gray = cv2.imread(sbs_therm_paths[i], cv2.IMREAD_GRAYSCALE) 
        fused_img = cv2.imread(sbs_fused_paths[i])

        if vis_img is None or therm_img_gray is None or fused_img is None:
            print(f"Warning: Skipping SBS frame set {i} (Vis: {os.path.basename(sbs_vis_paths[i])}) due to a missing image.")
            continue

        vis_resized = cv2.resize(vis_img, (panel_w, panel_h))
        fused_resized = cv2.resize(fused_img, (panel_w, panel_h))
        therm_gray_resized = cv2.resize(therm_img_gray, (panel_w, panel_h))
        therm_bgr_resized = cv2.cvtColor(therm_gray_resized, cv2.COLOR_GRAY2BGR)

        font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.6; font_color = (255, 255, 255)
        bg_color = (0,0,0); thickness = 1; text_y = 25
        for img_panel, label in zip([vis_resized, therm_bgr_resized, fused_resized], ["RGB", "Thermal", "Fused"]):
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            rect_y1 = max(0, text_y - text_h - 5)
            cv2.rectangle(img_panel, (5, rect_y1), (5 + text_w + 5, text_y + 5), bg_color, -1)
            cv2.putText(img_panel, label, (10, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        
        sbs_frame = np.hstack((vis_resized, therm_bgr_resized, fused_resized))
        
        # Ensure dimensions match VideoWriter before writing
        if sbs_frame.shape[1] != total_width or sbs_frame.shape[0] != panel_h:
            print(f"Warning: Resizing SBS frame from {sbs_frame.shape[:2][::-1]} to {(total_width,panel_h)} before writing.")
            sbs_frame = cv2.resize(sbs_frame, (total_width, panel_h))

        out_sbs_video.write(sbs_frame)
            
    out_sbs_video.release()
    print(f"Side-by-side video saved to: {output_video_file}")
    return True