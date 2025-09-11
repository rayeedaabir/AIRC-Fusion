import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any

class Model2015SpEAnsatzFusion:
    def __init__(self, gradient_ksize: int = 3, **kwargs):
        self.gradient_ksize = gradient_ksize
        if kwargs:
            pass
            # print(f"Warning (Model2015SpEAnsatzFusion): Unused parameters passed to __init__: {kwargs}")

    def _compute_gradient_matrix_channels(self, image_multichannel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image_multichannel_float64 = image_multichannel.astype(np.float64)
        num_channels = image_multichannel_float64.shape[2]
        h, w = image_multichannel_float64.shape[:2]
        Dx_all = np.zeros((h, w, num_channels), dtype=np.float64)
        Dy_all = np.zeros((h, w, num_channels), dtype=np.float64)
        for i in range(num_channels):
            channel = image_multichannel_float64[:, :, i]
            Dx_all[:, :, i] = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=self.gradient_ksize)
            Dy_all[:, :, i] = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=self.gradient_ksize)
        return Dx_all, Dy_all

    def _reintegrate_gradient_field_poisson_placeholder(self, Gx: np.ndarray, Gy: np.ndarray) -> np.ndarray:
        # print("WARNING: Using placeholder gradient reintegration for SpE. Output will not match paper accurately.")
        magnitude = np.sqrt(Gx**2 + Gy**2)
        reconstruction = cv2.normalize(magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return reconstruction

    def fuse(self, visible_uint8_bgr: np.ndarray, infrared_uint8_bgr: np.ndarray) -> np.ndarray:
        # print("DEBUG: Using VECTORIZED SpE Ansatz logic.")
        h, w = visible_uint8_bgr.shape[:2]

        vis_float_255range = visible_uint8_bgr.astype(np.float32)
        
        if len(infrared_uint8_bgr.shape) == 2:
            ir_bgr_for_gray_conv = cv2.cvtColor(infrared_uint8_bgr, cv2.COLOR_GRAY2BGR)
        elif infrared_uint8_bgr.shape[2] == 1:
            ir_bgr_for_gray_conv = cv2.cvtColor(infrared_uint8_bgr, cv2.COLOR_GRAY2BGR)
        else:
            ir_bgr_for_gray_conv = infrared_uint8_bgr
        ir_gray_float_255range = cv2.cvtColor(ir_bgr_for_gray_conv, cv2.COLOR_BGR2GRAY).astype(np.float32)

        H = np.dstack((vis_float_255range, ir_gray_float_255range[:,:,np.newaxis]))
        R_putative_float = vis_float_255range
        # print(f"DEBUG SpE: H shape: {H.shape}, R_putative shape: {R_putative_float.shape}") # DEBUG

        num_channels_H = H.shape[2]
        num_channels_R = R_putative_float.shape[2]

        Dx_H, Dy_H = self._compute_gradient_matrix_channels(H)
        Dx_R_putative, Dy_R_putative = self._compute_gradient_matrix_channels(R_putative_float)

        VH_stacked_grads = np.stack((Dx_H, Dy_H), axis=-1)
        VH_for_svd = VH_stacked_grads.reshape(-1, num_channels_H, 2)
        VR_put_stacked_grads = np.stack((Dx_R_putative, Dy_R_putative), axis=-1)
        VR_put_for_svd = VR_put_stacked_grads.reshape(-1, num_channels_R, 2)

        # Initialize target gradient components to zeros
        target_grad_Rx = np.zeros((h, w), dtype=np.float64)
        target_grad_Ry = np.zeros((h, w), dtype=np.float64)
        target_grad_Gx = np.zeros((h, w), dtype=np.float64)
        target_grad_Gy = np.zeros((h, w), dtype=np.float64)
        target_grad_Bx = np.zeros((h, w), dtype=np.float64)
        target_grad_By = np.zeros((h, w), dtype=np.float64)
        
        svd_successful = True # Flag to track SVD success
        try:
            Uh_all, Sh_all, Vh_H_transpose_all = np.linalg.svd(VH_for_svd, full_matrices=False)
            Ur_put_all, _, _ = np.linalg.svd(VR_put_for_svd, full_matrices=False)
            # print(f"DEBUG SpE SVD: Uh_all: {Uh_all.shape}, Sh_all: {Sh_all.shape}, Vh_H_t: {Vh_H_transpose_all.shape}, Ur_put_all: {Ur_put_all.shape}") # DEBUG

        except np.linalg.LinAlgError as e:
            # print(f"SVD Error: {e}. This might happen if input gradients are zero or rank deficient. Returning black image.")
            svd_successful = False # Set flag to false

        if svd_successful:
            num_pixels = h * w
            AH_matrix_all = np.zeros((num_pixels, 2, 2), dtype=np.float64)
            AH_matrix_all[:, 0, 0] = Sh_all[:, 0]
            AH_matrix_all[:, 1, 1] = Sh_all[:, 1]

            temp_prod = np.einsum('pij,pjk->pik', AH_matrix_all, Vh_H_transpose_all)
            VR_target_all_stacked = np.einsum('pij,pjk->pik', Ur_put_all, temp_prod)

            target_grads_reshaped = VR_target_all_stacked.reshape(h, w, num_channels_R, 2)
            
            target_grad_Rx = target_grads_reshaped[:, :, 0, 0]
            target_grad_Ry = target_grads_reshaped[:, :, 0, 1]
            target_grad_Gx = target_grads_reshaped[:, :, 1, 0]
            target_grad_Gy = target_grads_reshaped[:, :, 1, 1]
            target_grad_Bx = target_grads_reshaped[:, :, 2, 0]
            target_grad_By = target_grads_reshaped[:, :, 2, 1]
        # else: SVD failed, target_grads remain zeros

        # 8. Reintegrate each target gradient field
        # This step will now always have defined (though potentially zero) target_grad_X variables.
        fused_output_rgb_float01 = np.zeros((h,w,3), dtype=np.float32)
        fused_output_rgb_float01[:, :, 0] = self._reintegrate_gradient_field_poisson_placeholder(target_grad_Rx, target_grad_Ry)
        fused_output_rgb_float01[:, :, 1] = self._reintegrate_gradient_field_poisson_placeholder(target_grad_Gx, target_grad_Gy)
        fused_output_rgb_float01[:, :, 2] = self._reintegrate_gradient_field_poisson_placeholder(target_grad_Bx, target_grad_By)
        # print(f"DEBUG SpE Reintegrated float min: {fused_output_rgb_float01.min()}, max: {fused_output_rgb_float01.max()}") # DEBUG
        
        return np.clip(fused_output_rgb_float01 * 255.0, 0, 255).astype(np.uint8)