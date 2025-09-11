import cv2
import numpy as np
import pywt
from skimage.filters import sobel
from sklearn.metrics import mutual_info_score
from skimage.metrics import structural_similarity as ssim

def ensure_grayscale(image):
    """Ensure the image is grayscale. Input can be BGR or already grayscale."""
    if image is None:
        raise ValueError("Input image to ensure_grayscale cannot be None.")
    if len(image.shape) == 3 and image.shape[2] == 3: # BGR
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 1: # Grayscale but with 3 dims
        return image.squeeze() # Remove single channel dim
    elif len(image.shape) == 2: # Already grayscale
        return image
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

def compute_entropy_metric(image_uint8):
    """Computes the Shannon entropy of a grayscale image (0-255 range)."""
    gray_image = ensure_grayscale(image_uint8)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    # Use np.log2 and handle zero probabilities by adding a small epsilon
    log_hist = np.log2(hist + 1e-12) 
    return -np.sum(hist * log_hist)

def compute_mutual_information_metric(image1_uint8, image2_uint8):
    """Computes mutual information between two grayscale images (0-255 range)."""
    gray1 = ensure_grayscale(image1_uint8)
    gray2 = ensure_grayscale(image2_uint8)

    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Calculate 2D histogram
    # Bins can be adjusted; 32 or 64 are common for 8-bit images.
    # Ensure range covers full 0-255 for uint8 images.
    hist_2d, _, _ = np.histogram2d(gray1.ravel(), gray2.ravel(), bins=32, range=[[0, 256], [0, 256]])
    
    # mutual_info_score expects a contingency table (the 2D histogram)
    # It calculates MI = H(X) + H(Y) - H(X,Y)
    return mutual_info_score(None, None, contingency=hist_2d)

def compute_std_dev_metric(image_uint8):
    """Computes the standard deviation of a grayscale image."""
    gray_image = ensure_grayscale(image_uint8)
    return np.std(gray_image)

def compute_noise_metric_val(image_uint8):
    """Estimates noise by comparing with a blurred version."""
    gray_image = ensure_grayscale(image_uint8)
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    noise = np.mean(np.abs(gray_image.astype(np.float32) - blurred.astype(np.float32)))
    return noise

def compute_wavelet_stds_metric(image_uint8):
    """Computes standard deviations of wavelet coefficients (Haar)."""
    gray_image = ensure_grayscale(image_uint8)
    try:
        coeffs = pywt.dwt2(gray_image, 'haar')
        cA, (cH, cV, cD) = coeffs
        return np.std(cA), np.std(cH), np.std(cV), np.std(cD)
    except Exception as e:
        print(f"Error in wavelet computation: {e}. Image shape: {gray_image.shape}")
        return np.nan, np.nan, np.nan, np.nan # Return NaN on error

def compute_edge_based_metric_sum(image_uint8):
    """Computes the sum of edge magnitudes using Sobel operator."""
    gray_image = ensure_grayscale(image_uint8)
    edges = sobel(gray_image) # sobel from skimage.filters returns float values
    return np.sum(edges)

def compute_ssim_accuracy_metric(fused_uint8, visible_uint8, infrared_uint8):
    """Computes average SSIM between fused and source images (grayscale)."""
    fused_gray = ensure_grayscale(fused_uint8)
    visible_gray = ensure_grayscale(visible_uint8)
    infrared_gray = ensure_grayscale(infrared_uint8)

    # Ensure consistent shapes for SSIM calculation
    if fused_gray.shape != visible_gray.shape:
        visible_gray = cv2.resize(visible_gray, (fused_gray.shape[1], fused_gray.shape[0]), cv2.INTER_LINEAR)
    if fused_gray.shape != infrared_gray.shape:
        infrared_gray = cv2.resize(infrared_gray, (fused_gray.shape[1], fused_gray.shape[0]), cv2.INTER_LINEAR)

    # data_range is important for SSIM. Since inputs are uint8, range is 0-255.
    # However, skimage's ssim normalizes if data_range is not set and images are float.
    # For uint8, it's safer to specify.
    try:
        ssim_vis = ssim(fused_gray, visible_gray, data_range=255)
        ssim_ir = ssim(fused_gray, infrared_gray, data_range=255)
        return (ssim_vis + ssim_ir) / 2.0
    except ValueError as e: # Can happen if win_size is too large for image
        print(f"SSIM calculation error: {e}. Shapes: F={fused_gray.shape}, V={visible_gray.shape}, I={infrared_gray.shape}")
        return np.nan

def calculate_all_metrics(fused_image_uint8, visible_image_uint8, infrared_image_uint8):
    """
    Calculates all specified metrics for a given set of fused and source images.
    All input images are expected to be in uint8 BGR or grayscale format.
    """
    metrics = {}
    try:
        metrics['entropy'] = compute_entropy_metric(fused_image_uint8)
        # Mutual information is usually between sources, or fused vs sources
        # Here, using MI between original visible and original infrared
        metrics['mutual_info_src'] = compute_mutual_information_metric(visible_image_uint8, infrared_image_uint8)
        metrics['std_dev'] = compute_std_dev_metric(fused_image_uint8)
        metrics['noise'] = compute_noise_metric_val(fused_image_uint8)
        
        wav_cA_std, wav_cH_std, wav_cV_std, wav_cD_std = compute_wavelet_stds_metric(fused_image_uint8)
        metrics['wavelet_std_cA'] = wav_cA_std
        metrics['wavelet_std_cH'] = wav_cH_std
        metrics['wavelet_std_cV'] = wav_cV_std
        metrics['wavelet_std_cD'] = wav_cD_std
        
        metrics['edge_sum'] = compute_edge_based_metric_sum(fused_image_uint8)
        metrics['ssim_accuracy'] = compute_ssim_accuracy_metric(fused_image_uint8, visible_image_uint8, infrared_image_uint8)
    except Exception as e:
        print(f"Error calculating one or more metrics: {e}")
        # Optionally fill with NaN or skip
        for key in ['entropy', 'mutual_info_src', 'std_dev', 'noise', 
                    'wavelet_std_cA', 'wavelet_std_cH', 'wavelet_std_cV', 'wavelet_std_cD',
                    'edge_sum', 'ssim_accuracy']:
            if key not in metrics:
                metrics[key] = np.nan # Use NaN for metrics that failed
    return metrics