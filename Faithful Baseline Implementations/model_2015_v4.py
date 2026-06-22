import cv2
import numpy as np
from typing import Tuple

class Model2015SpEAnsatzFusion:
    def __init__(self, gradient_ksize: int = 3, **kwargs):
        self.gradient_ksize = gradient_ksize

    def _compute_gradient_matrix_channels(self, image_multichannel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image_multichannel_float64 = image_multichannel.astype(np.float64)
        num_channels = image_multichannel_float64.shape[2]
        h, w = image_multichannel_float64.shape[:2]
        Dx_all = np.zeros((h, w, num_channels), dtype=np.float64)
        Dy_all = np.zeros((h, w, num_channels), dtype=np.float64)
        
        # Use standard Sobel
        for i in range(num_channels):
            channel = image_multichannel_float64[:, :, i]
            Dx_all[:, :, i] = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=self.gradient_ksize)
            Dy_all[:, :, i] = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=self.gradient_ksize)
        return Dx_all, Dy_all

    def _fft_poisson_solver(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """
        Solves the Poisson equation ∇²I = div(G) using FFT.
        This reconstructs an image I whose gradient is closest to (gx, gy).
        """
        h, w = gx.shape
        
        # 1. Calculate Divergence: div(G) = dGx/dx + dGy/dy
        # We use the same derivative operator (Sobel) or simple differences to be consistent
        # Using simple differences for divergence is often more stable for FFT reconstruction
        gy_y = np.diff(gy, axis=0, prepend=0)
        gx_x = np.diff(gx, axis=1, prepend=0)
        divergence = gx_x + gy_y

        # 2. Prepare Laplacian operator in Fourier Domain
        # The eigenvalues of the discrete Laplacian for the DST (Discrete Sine Transform) 
        # or FFT with boundary handling. 
        # Simplified approach: FFT of the Laplacian kernel [0, -1, 0; -1, 4, -1; 0, -1, 0]
        
        # Grid of frequencies
        fx = np.fft.fftfreq(w).reshape(1, -1)
        fy = np.fft.fftfreq(h).reshape(-1, 1)
        
        # Optical Transfer Function (OTF) for the Laplacian operator [-1 2 -1]
        # This denominator represents (2*cos(2*pi*f) - 2) for both dimensions
        # Corresponds to the Fourier transform of the discrete Laplacian
        denom = (2 * np.cos(2 * np.pi * fx) - 2) + (2 * np.cos(2 * np.pi * fy) - 2)
        
        # Avoid division by zero at DC component (0,0)
        denom[0, 0] = 1.0 

        # 3. Solve in Frequency Domain
        f_divergence = np.fft.fft2(divergence)
        f_img = f_divergence / denom
        
        # Handle DC component (average brightness is lost in gradients)
        # We set it to 0 (gray) or handle it later by matching mean
        f_img[0, 0] = 0 

        # 4. Inverse FFT
        img_reconstructed = np.real(np.fft.ifft2(f_img))
        
        return img_reconstructed

    def fuse(self, visible_uint8_bgr: np.ndarray, infrared_uint8_bgr: np.ndarray) -> np.ndarray:
        h, w = visible_uint8_bgr.shape[:2]
        vis_float = visible_uint8_bgr.astype(np.float32) / 255.0
        
        # Handle IR input
        if len(infrared_uint8_bgr.shape) == 2:
            ir_gray = infrared_uint8_bgr.astype(np.float32) / 255.0
        else:
            ir_gray = cv2.cvtColor(infrared_uint8_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Stack for High-D input H (Vis + IR) -> 4 Channels
        H = np.dstack((vis_float, ir_gray[:,:,np.newaxis]))
        
        # Putative R is just the Visible image
        R_putative = vis_float

        # 1. Compute Gradients
        Dx_H, Dy_H = self._compute_gradient_matrix_channels(H)
        Dx_R, Dy_R = self._compute_gradient_matrix_channels(R_putative)

        # 2. Reshape for SVD (Pixels x Channels x 2)
        # H: 4 channels, R: 3 channels
        VH = np.stack((Dx_H, Dy_H), axis=-1).reshape(-1, 4, 2)
        VR = np.stack((Dx_R, Dy_R), axis=-1).reshape(-1, 3, 2)

        # 3. SVD Per Pixel (Vectorized)
        # We need Z_H = V_H * Sigma_H^2 * V_H.T
        # We need Z_R = V_R * Sigma_R^2 * V_R.T
        # The paper's ansatz: New Gradient = U_R * Sigma_H * V_H.T
        
        try:
            # SVD of H gradients
            U_H, S_H, Vt_H = np.linalg.svd(VH, full_matrices=False)
            # SVD of R gradients (we only need U_R)
            U_R, _, _ = np.linalg.svd(VR, full_matrices=False)
            
            # Construct Target Gradient: G_new = U_R * S_H * Vt_H
            # S_H is returned as a vector by numpy, need to diagonalize
            S_H_mat = np.zeros((S_H.shape[0], 2, 2), dtype=S_H.dtype)
            S_H_mat[:, 0, 0] = S_H[:, 0]
            S_H_mat[:, 1, 1] = S_H[:, 1]
            
            # Matrix Multiplication: (Px3x2) @ (Px2x2) @ (Px2x2)
            # Step 1: S_H * Vt_H
            temp = np.matmul(S_H_mat, Vt_H)
            # Step 2: U_R * temp
            target_grads = np.matmul(U_R, temp)
            
            # Reshape back to image dimensions
            target_grads = target_grads.reshape(h, w, 3, 2)
            
        except np.linalg.LinAlgError:
            print("SVD failed in Model2015. Returning visible image.")
            return visible_uint8_bgr

        # 4. Reintegrate (Poisson Solver)
        fused_float = np.zeros((h, w, 3), dtype=np.float32)
        
        for c in range(3):
            gx = target_grads[:, :, c, 0]
            gy = target_grads[:, :, c, 1]
            
            # Solve Poisson
            reconstructed = self._fft_poisson_solver(gx, gy)
            
            # 5. Post-Processing (Match Mean/Std of Original Visible Channel)
            # Gradient domain loses absolute intensity. We restore it using the visible image statistics.
            orig_mean = np.mean(vis_float[:,:,c])
            orig_std = np.std(vis_float[:,:,c])
            recon_mean = np.mean(reconstructed)
            recon_std = np.std(reconstructed)
            
            if recon_std > 1e-5:
                reconstructed = (reconstructed - recon_mean) * (orig_std / recon_std) + orig_mean
            else:
                reconstructed = reconstructed - recon_mean + orig_mean
                
            fused_float[:, :, c] = reconstructed

        return np.clip(fused_float * 255.0, 0, 255).astype(np.uint8)