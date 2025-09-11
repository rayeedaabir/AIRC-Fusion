import cv2
import numpy as np
from typing import Tuple, Any

try:
    from cv2.ximgproc import guidedFilter
    HAS_GUIDED_FILTER = True
    print("INFO: Using cv2.ximgproc.guidedFilter for Model2020.")
except ImportError:
    HAS_GUIDED_FILTER = False
    print("WARNING: cv2.ximgproc.guidedFilter not found for Model2020. Base layer fusion might differ.")

class Model2020Fusion:
    def __init__(self,
                 decomp_gaussian_ksize: int = 11, decomp_gaussian_sigma: float = 5.0,
                 base_saliency_laplacian_ksize: int = 3,
                 base_saliency_gaussian_ksize: int = 11, base_saliency_gaussian_sigma: float = 5.0,
                 guided_filter_radius_base: int = 7, guided_filter_eps_base: float = 0.01,
                 median_blur_ksize_detail: int = 3,
                 detail_saliency_laplacian_ksize: int = 3,
                 detail_weight_smooth_ksize: int = 11, detail_weight_smooth_sigma: float = 5.0,
                 **kwargs):

        self.decomp_gaussian_ksize = decomp_gaussian_ksize if decomp_gaussian_ksize % 2 != 0 else decomp_gaussian_ksize + 1
        self.decomp_gaussian_sigma = decomp_gaussian_sigma
        self.base_saliency_laplacian_ksize = base_saliency_laplacian_ksize
        self.base_saliency_gaussian_ksize = base_saliency_gaussian_ksize if base_saliency_gaussian_ksize % 2 != 0 else base_saliency_gaussian_ksize + 1
        self.base_saliency_gaussian_sigma = base_saliency_gaussian_sigma
        self.guided_filter_radius_base = guided_filter_radius_base
        self.guided_filter_eps_base = guided_filter_eps_base
        self.median_blur_ksize_detail = median_blur_ksize_detail if median_blur_ksize_detail % 2 != 0 else median_blur_ksize_detail + 1
        self.detail_saliency_laplacian_ksize = detail_saliency_laplacian_ksize
        self.detail_weight_smooth_ksize = detail_weight_smooth_ksize if detail_weight_smooth_ksize % 2 != 0 else detail_weight_smooth_ksize + 1
        self.detail_weight_smooth_sigma = detail_weight_smooth_sigma
        self.guidedFilterFunc = guidedFilter if HAS_GUIDED_FILTER else None

    def _decompose(self, image_channel_float: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        base = cv2.GaussianBlur(image_channel_float,
                                (self.decomp_gaussian_ksize, self.decomp_gaussian_ksize),
                                self.decomp_gaussian_sigma)
        detail = image_channel_float - base # detail is float32
        return base, detail

    def _get_base_saliency_map(self, base_channel_float: np.ndarray) -> np.ndarray:
        # base_channel_float is expected to be float32
        lap = cv2.Laplacian(base_channel_float, ddepth=cv2.CV_32F, # Output float32
                            ksize=self.base_saliency_laplacian_ksize)
        bh = np.abs(lap)
        bs = cv2.GaussianBlur(bh,
                              (self.base_saliency_gaussian_ksize, self.base_saliency_gaussian_ksize),
                              self.base_saliency_gaussian_sigma)
        return bs

    def _get_detail_activity_map(self, detail_channel_float: np.ndarray) -> np.ndarray:
        # detail_channel_float can be negative. Normalize for median blur.
        detail_norm_uint8 = cv2.normalize(detail_channel_float, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        denoised_detail_uint8 = cv2.medianBlur(detail_norm_uint8, self.median_blur_ksize_detail)
        
        # Input to Laplacian is uint8, let's get a common float type for activity map
        lap_s16 = cv2.Laplacian(denoised_detail_uint8, ddepth=cv2.CV_16S, # Use signed 16-bit for intermediate
                                ksize=self.detail_saliency_laplacian_ksize)
        return np.abs(lap_s16.astype(np.float32)) # Convert to float32 for activity map

    def fuse(self, visible_uint8_bgr: np.ndarray, infrared_uint8_bgr: np.ndarray) -> np.ndarray:
        vis_float_01 = visible_uint8_bgr.astype(np.float32) / 255.0
        ir_float_01 = infrared_uint8_bgr.astype(np.float32) / 255.0
        fused_image_float_01 = np.zeros_like(vis_float_01, dtype=np.float32)

        for i in range(3):
            vis_ch_float = vis_float_01[:,:,i]
            ir_ch_float = ir_float_01[:,:,i]

            b_vis, d_vis = self._decompose(vis_ch_float) # b_vis, d_vis are float32
            b_ir, d_ir = self._decompose(ir_ch_float)   # b_ir, d_ir are float32

            bs_vis = self._get_base_saliency_map(b_vis) # bs_vis is float32
            bs_ir = self._get_base_saliency_map(b_ir)   # bs_ir is float32
            
            p_base = (bs_vis >= bs_ir).astype(np.float32)
            
            if self.guidedFilterFunc:
                w_base_vis = self.guidedFilterFunc(b_vis, p_base, # guide, src
                                            self.guided_filter_radius_base, self.guided_filter_eps_base)
            else:
                w_base_vis = cv2.GaussianBlur(p_base,
                                             (self.base_saliency_gaussian_ksize, self.base_saliency_gaussian_ksize),
                                             self.base_saliency_gaussian_sigma)
            w_base_vis = np.clip(w_base_vis, 0.0, 1.0)
            fused_base_ch = w_base_vis * b_vis + (1.0 - w_base_vis) * b_ir

            ds_vis = self._get_detail_activity_map(d_vis) # ds_vis is float32
            ds_ir = self._get_detail_activity_map(d_ir)   # ds_ir is float32
            
            p_detail = (ds_vis >= ds_ir).astype(np.float32)
            w_detail_vis = cv2.GaussianBlur(p_detail,
                                           (self.detail_weight_smooth_ksize, self.detail_weight_smooth_ksize),
                                           self.detail_weight_smooth_sigma)
            w_detail_vis = np.clip(w_detail_vis, 0.0, 1.0)
            fused_detail_ch = w_detail_vis * d_vis + (1.0 - w_detail_vis) * d_ir
            
            fused_image_float_01[:, :, i] = fused_base_ch + fused_detail_ch
            
        return np.clip(fused_image_float_01 * 255.0, 0, 255).astype(np.uint8)