# process_landsat.py
import cv2
import numpy as np
import tifffile
import time
import os
import matplotlib.pyplot as plt

# Import the classes and functions from your existing, tested modules
from image_fusion_processor import ImageFusionProcessor
from evaluation_metrics import calculate_all_metrics

# --- Configuration ---
# Define paths to your four Landsat band files
PATH_B2_BLUE = "D:/Codes/499B/Datasets/Landsat Datasets/Landsat 8-9 Bands/LC08_L1GT_091106_20250310_20250310_02_RT_B2.TIF"
PATH_B3_GREEN = "D:/Codes/499B/Datasets/Landsat Datasets/Landsat 8-9 Bands/LC08_L1GT_091106_20250310_20250310_02_RT_B3.TIF"
PATH_B4_RED = "D:/Codes/499B/Datasets/Landsat Datasets/Landsat 8-9 Bands/LC08_L1GT_091106_20250310_20250310_02_RT_B4.TIF"
PATH_B5_NIR = "D:/Codes/499B/Datasets/Landsat Datasets/Landsat 8-9 Bands/LC08_L1GT_091106_20250310_20250310_02_RT_B5.TIF"

FUSION_METHOD_TO_TEST = "vgg19"

def load_tiff_channel(path: str) -> np.ndarray:
    try:
        image = tifffile.imread(path)
        print(f"Loaded {os.path.basename(path)} - Shape: {image.shape}, Dtype: {image.dtype}")
        return image.astype(np.float32)
    except FileNotFoundError:
        print(f"ERROR: File not found at {path}")
        return None

def main():
    print(f"--- Processing Landsat Data with {FUSION_METHOD_TO_TEST} ---")

    # 1. Load individual bands (remains the same)
    blue_ch = load_tiff_channel(PATH_B2_BLUE)
    green_ch = load_tiff_channel(PATH_B3_GREEN)
    red_ch = load_tiff_channel(PATH_B4_RED)
    nir_ch = load_tiff_channel(PATH_B5_NIR)

    if any(ch is None for ch in [blue_ch, green_ch, red_ch, nir_ch]):
        print("Aborting due to missing files.")
        return
    
    CROP_SIZE = 1024 # e.g., for a 1024x1024 crop
    y_start, x_start = 4000, 4000 # Choose a starting corner with interesting features
    blue_ch = blue_ch[y_start : y_start + CROP_SIZE, x_start : x_start + CROP_SIZE]
    green_ch = green_ch[y_start : y_start + CROP_SIZE, x_start : x_start + CROP_SIZE]
    red_ch = red_ch[y_start : y_start + CROP_SIZE, x_start : x_start + CROP_SIZE]
    nir_ch = nir_ch[y_start : y_start + CROP_SIZE, x_start : x_start + CROP_SIZE]
    
    print(f"--- INFO: Processing a {CROP_SIZE}x{CROP_SIZE} crop of the Landsat image. ---")
    
    # 2. Assemble source images (remains the same)
    visible_bgr_float = cv2.merge([blue_ch, green_ch, red_ch])
    v_min, v_max = np.percentile(visible_bgr_float, (0.5, 99.5))
    visible_bgr_uint8 = np.clip((visible_bgr_float - v_min) / (v_max - v_min) * 255.0, 0, 255).astype(np.uint8)

    nir_min, nir_max = np.percentile(nir_ch, (0.5, 99.5))
    infrared_gray_uint8 = np.clip((nir_ch - nir_min) / (nir_max - nir_min) * 255.0, 0, 255).astype(np.uint8)
    infrared_bgr_uint8 = cv2.cvtColor(infrared_gray_uint8, cv2.COLOR_GRAY2BGR)

    # 3. Instantiate and run the fusion processor (remains the same)
    fusion_processor = ImageFusionProcessor(core_fusion_method=FUSION_METHOD_TO_TEST)
    fused_image, proc_time = fusion_processor.process_frame(visible_bgr_uint8, infrared_bgr_uint8)

    # 4. Calculate and print metrics (remains the same)
    all_metrics = calculate_all_metrics(fused_image, visible_bgr_uint8, infrared_bgr_uint8)
    print("\n--- Landsat Fusion Metrics ---")
    print(f"Fusion Method: {FUSION_METHOD_TO_TEST}")
    print(f"  {'Proc Time':<30}: {proc_time:5f} s")
    print(f"  {'Fps':<30}: {1/proc_time if proc_time > 0 else float('inf'):.5f}")
    for key, value in all_metrics.items():
        formatted_key = key.replace('_', ' ').title()
        if isinstance(value, (float, np.floating)): 
            print(f"  {formatted_key:<30}: {value:.5f}")
        else:
            print(f"  {formatted_key:<30}: {value}")


    # 5. Save the fused image (remains the same)
    output_filename = f"landsat_fused_{FUSION_METHOD_TO_TEST}.png"
    cv2.imwrite(output_filename, fused_image)
    print(f"\nSaved fused image to {output_filename}")
    
    print("Generating Matplotlib figure for display...")
    plt.figure(figsize=(18, 6)) # Create a figure with a good aspect ratio

    # Subplot 1: Visible Image
    plt.subplot(1, 3, 1)
    # Matplotlib expects RGB, so we convert from OpenCV's BGR format
    plt.imshow(cv2.cvtColor(visible_bgr_uint8, cv2.COLOR_BGR2RGB))
    plt.title("Visible Image")
    plt.axis("off") # Hides the x and y axis numbers and ticks

    # Subplot 2: Infrared Image
    plt.subplot(1, 3, 2)
    # The NIR image is single-channel, so we display it in grayscale
    plt.imshow(infrared_gray_uint8, cmap='gray')
    plt.title("NIR Image")
    plt.axis("off")

    # Subplot 3: Fused Image
    plt.subplot(1, 3, 3)
    # Also convert the fused image from BGR to RGB for correct display
    plt.imshow(cv2.cvtColor(fused_image, cv2.COLOR_BGR2RGB))
    plt.title("Fused Image")
    plt.axis("off")

    # Adjust layout and show the final plot
    plt.tight_layout()
    
    # Save the figure to a file for your paper
    figure_filename = f"landsat_comparison_{FUSION_METHOD_TO_TEST}.png"
    plt.savefig(figure_filename, dpi=300) # Save with high resolution
    print(f"Saved comparison figure to {figure_filename}")
    
    plt.show() # This will display the plot in a window

if __name__ == "__main__":
    main()