import cv2
import numpy as np
import time
from typing import Tuple, Optional, Dict, Any 
import warnings 

from model_2020_fixed import Model2020Fusion
from model_2015_v4 import Model2015SpEAnsatzFusion
from model_2024_panda import Panda2024Fusion   # faithful Panda et al. 2024 [5]
from model_mdlatlrr import MDLatLRRFusion         # faithful MDLatLRR [59]
from model_densefuse_official import DenseFuseOfficial  # faithful DenseFuse [47]

import torch
import torch.nn.functional as F
import torchvision.models as models
# for Deep Learning (DenseFuse)
try:
    from model import DenseNet
    DL_MODELS_AVAILABLE = True
except ImportError:
    DL_MODELS_AVAILABLE = False
    print("Warning: 'model.py' (DenseFuse) not found. DL Fusion will not work.")


class PerformanceMonitor: 
    def __init__(self): self.timings = {}
    def timing_decorator(self, func_name: str):
        def decorator(func):
            def wrapper(*args, **kwargs): return func(*args, **kwargs)
            return wrapper
        return decorator

class OptimizedKernelManager:
    def __init__(self): self._gaussian_cache = {}
    def get_gaussian_kernel(self, size: int, sigma: float = 0) -> np.ndarray:
        cache_key = (size, sigma)
        if cache_key not in self._gaussian_cache:
            if size % 2 == 0: size += 1; warnings.warn(f"Gaussian kernel size was even, adjusted to {size}")
            kernel_1d = cv2.getGaussianKernel(size, sigma)
            self._gaussian_cache[cache_key] = (kernel_1d @ kernel_1d.T).astype(np.float32)
        return self._gaussian_cache[cache_key]
    def get_mean_kernel(self, size: int) -> np.ndarray: # Needed for adaptive detail
        return np.ones((size,size), np.float32)/(size*size)

class DetailFusionStrategies: # Needed for model2024
    @staticmethod
    def max_abs_fusion(dv, di): return np.where(np.abs(dv) >= np.abs(di), dv, di)
    @staticmethod
    def weighted_fusion(dv, di): sum_w = np.abs(dv)+np.abs(di)+1e-12; return (np.abs(dv)/sum_w)*dv + (np.abs(di)/sum_w)*di
    @staticmethod
    def adaptive_fusion(dv, di, k): 
        var_v = np.abs(cv2.filter2D(dv*dv,-1,k)-cv2.filter2D(dv,-1,k)**2)
        var_i = np.abs(cv2.filter2D(di*di,-1,k)-cv2.filter2D(di,-1,k)**2)
        sum_v = var_v+var_i+1e-12; return (var_v/sum_v)*dv + (var_i/sum_v)*di

class ImageFusionProcessor: 
    def __init__(self,
                 core_fusion_method: str = "model2024",
                 # Params for model2024
                 gaussian_kernel_size: int = 15,
                 base_weight: float = 0.5,
                 detail_method: str = "max_abs", # This is detail_method_for_model2024
                 # Params for model2020
                 m2020_decomp_ksize: int = 11,
                 m2020_decomp_sigma: float = 5.0,
                 m2020_base_lap_ksize: int = 3,
                 m2020_base_gauss_ksize: int = 11,
                 m2020_base_gauss_sigma: float = 5.0,
                 m2020_guided_radius: int = 7,
                 m2020_guided_eps: float = 0.01,
                 m2020_detail_median_ksize: int = 3,
                 m2020_detail_lap_ksize: int = 3,
                 m2020_detail_weight_ksize: int = 11,
                 m2020_detail_weight_sigma: float = 5.0,
                 # Params for model2015
                 m2015_grad_ksize: int = 3,
                 # Target resolution for pipeline consistency
                 target_width: Optional[int] = None,
                 target_height: Optional[int] = None,
                 enable_monitoring: bool = True 
                 ):
        
        self.core_fusion_method = core_fusion_method.lower()
        self.target_width = target_width
        self.target_height = target_height
        self.processing_times = []
        self.specific_algorithm_instance = None # Will hold the instantiated algorithm

        # Store parameters needed for instantiation
        self.model2024_params = {
            "gaussian_kernel_size": gaussian_kernel_size,
            "base_weight": base_weight,
            "detail_method": detail_method
        }
        self.model2020_params = {
            "decomp_gaussian_ksize": m2020_decomp_ksize, "decomp_gaussian_sigma": m2020_decomp_sigma,
            "base_saliency_laplacian_ksize": m2020_base_lap_ksize,
            "base_saliency_gaussian_ksize": m2020_base_gauss_ksize,
            "base_saliency_gaussian_sigma": m2020_base_gauss_sigma,
            "guided_filter_radius_base": m2020_guided_radius, "guided_filter_eps_base": m2020_guided_eps,
            "median_blur_ksize_detail": m2020_detail_median_ksize,
            "detail_saliency_laplacian_ksize": m2020_detail_lap_ksize,
            "detail_weight_smooth_ksize": m2020_detail_weight_ksize,
            "detail_weight_smooth_sigma": m2020_detail_weight_sigma,
        }
        self.model2015_params = {
            "gradient_ksize": m2015_grad_ksize
        }

        if self.core_fusion_method == "model2024":
            # NOTE: GENERIC two-scale Gaussian scheme, NOT Panda et al. [5]; use "panda2024" for the faithful method.
            self.specific_algorithm_instance = _InternalModel2024(**self.model2024_params)
        elif self.core_fusion_method == "panda2024":
            self.specific_algorithm_instance = Panda2024Fusion()
        elif self.core_fusion_method == "mdlatlrr":
            self.specific_algorithm_instance = MDLatLRRFusion()
        elif self.core_fusion_method == "densefuse_official":
            self.specific_algorithm_instance = DenseFuseOfficial()
        elif self.core_fusion_method == "fcdfusion_vectorized":
            self.specific_algorithm_instance = _InternalFCDFusion_Vectorized()
        elif self.core_fusion_method == "fcdfusion_pixel":
            self.specific_algorithm_instance = _InternalFCDFusion_Pixel()
        elif self.core_fusion_method == "model2020":
            self.specific_algorithm_instance = Model2020Fusion(**self.model2020_params)
        elif self.core_fusion_method == "model2015":
            self.specific_algorithm_instance = Model2015SpEAnsatzFusion(**self.model2015_params)
        elif self.core_fusion_method == "densefuse":
            # Make sure the .pth file is in the same directory
            pth_path = "model.pth"
            self.specific_algorithm_instance = DenseFuseWrapper(model_path=pth_path)
        elif self.core_fusion_method == "vgg19":
            self.specific_algorithm_instance = VGG19FusionWrapper()
        elif self.core_fusion_method == "resnet152":
            self.specific_algorithm_instance = ResNet152FusionWrapper()
        else:
            raise ValueError(f"Unknown core_fusion_method: {self.core_fusion_method}")
        
        self.monitor = PerformanceMonitor() if enable_monitoring else None
        # if self.monitor and hasattr(self.specific_algorithm_instance, 'fuse'):
        #    self.specific_algorithm_instance.fuse = self.monitor.timing_decorator(f'fuse_{self.core_fusion_method}')(self.specific_algorithm_instance.fuse)

    def _common_align_bgr_and_resize(self, visible_uint8: np.ndarray, infrared_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if visible_uint8 is None or infrared_uint8 is None: raise ValueError("Input images cannot be None.")
        vis = visible_uint8.copy(); inf = infrared_uint8.copy()

        # Align dimensions
        if vis.shape[:2] != inf.shape[:2]:
            inf = cv2.resize(inf, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Ensure BGR
        if len(vis.shape) == 2: vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        elif vis.shape[2] == 1 and vis.shape[2] !=3 : vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR) # Check if not already BGR
        if len(inf.shape) == 2: inf = cv2.cvtColor(inf, cv2.COLOR_GRAY2BGR)
        elif inf.shape[2] == 1 and inf.shape[2] !=3 : inf = cv2.cvtColor(inf, cv2.COLOR_GRAY2BGR)

        # Resize to target pipeline resolution IF specified
        if self.target_width is not None and self.target_height is not None:
            if vis.shape[0] != self.target_height or vis.shape[1] != self.target_width:
                vis = cv2.resize(vis, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
            if inf.shape[0] != self.target_height or inf.shape[1] != self.target_width:
                inf = cv2.resize(inf, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
                
        return vis.astype(np.uint8), inf.astype(np.uint8)

    def process_frame(self, visible_uint8: np.ndarray, infrared_uint8: np.ndarray) -> Tuple[np.ndarray, float]:
        start_time = time.time()
        
        # Common preprocessing: alignment, BGR conversion, and optional target resizing
        # The output of this is uint8 BGR, aligned, and potentially resized.
        vis_processed_uint8, ir_processed_uint8 = self._common_align_bgr_and_resize(visible_uint8, infrared_uint8)
        
        # Delegate to the specific algorithm instance's fuse method.
        # Each algorithm's .fuse() method is responsible for handling its specific input needs
        # (e.g., further conversion to float) and returning a uint8 BGR fused image.
        if self.specific_algorithm_instance is None:
            raise RuntimeError(f"No specific fusion algorithm instance created for method '{self.core_fusion_method}'. Check __init__.")

        fused_image_uint8 = self.specific_algorithm_instance.fuse(vis_processed_uint8, ir_processed_uint8)
        
        proc_time = time.time() - start_time
        self.processing_times.append(proc_time)
        return fused_image_uint8, proc_time

# --- _InternalModel2024 and _InternalFCDFusion helper classes ---
# Model2024
class _InternalModel2024:
    def __init__(self, gaussian_kernel_size, base_weight, detail_method):
        self.gaussian_kernel_size = gaussian_kernel_size
        self.base_weight = base_weight
        self.detail_method = detail_method 
        if self.gaussian_kernel_size % 2 == 0: self.gaussian_kernel_size +=1
        kernel_manager = OptimizedKernelManager() 
        self.gaussian_kernel_cv = kernel_manager.get_gaussian_kernel(self.gaussian_kernel_size)
        self.fusion_strategies = DetailFusionStrategies()
        self._adaptive_kernel = None 
        if self.detail_method == "adaptive": 
            self._adaptive_kernel = kernel_manager.get_mean_kernel(3)

    def _preprocess_to_float01(self, vis_uint8_bgr, ir_uint8_bgr): 
        # Assumes vis_uint8_bgr, ir_uint8_bgr are already aligned and BGR
        vis_float = vis_uint8_bgr.astype(np.float32) / 255.0 
        ir_float = ir_uint8_bgr.astype(np.float32) / 255.0 
        return vis_float, ir_float
    
    def fuse(self, visible_uint8_bgr: np.ndarray, infrared_uint8_bgr: np.ndarray) -> np.ndarray:
        # This fuse method expects aligned uint8 BGR inputs
        visible_float, infrared_float = self._preprocess_to_float01(visible_uint8_bgr, infrared_uint8_bgr)
        
        low_vis = cv2.filter2D(visible_float, -1, self.gaussian_kernel_cv)
        low_ir = cv2.filter2D(infrared_float, -1, self.gaussian_kernel_cv)
        fused_base = cv2.addWeighted(low_vis, self.base_weight, low_ir, 1.0 - self.base_weight, 0)
        detail_vis = visible_float - low_vis; detail_ir = infrared_float - low_ir
        
        fused_detail = None
        if self.detail_method == "max_abs":     fused_detail = self.fusion_strategies.max_abs_fusion(detail_vis, detail_ir)
        elif self.detail_method == "weighted":  fused_detail = self.fusion_strategies.weighted_fusion(detail_vis, detail_ir)
        elif self.detail_method == "adaptive":  fused_detail = self.fusion_strategies.adaptive_fusion(detail_vis, detail_ir, self._adaptive_kernel)
        else: raise ValueError(f"Unknown detail_method: {self.detail_method}")
        
        fusion_float_result = fused_base + fused_detail
        return np.clip(fusion_float_result * 255.0, 0, 255).astype(np.uint8)

# Vectorized FCDFusion
class _InternalFCDFusion_Vectorized:
    def __init__(self): pass 

    def fuse(self, visible_uint8_bgr: np.ndarray, infrared_uint8_bgr: np.ndarray) -> np.ndarray:
        visible_float_fcd = visible_uint8_bgr.astype(np.float32)
        infrared_float_fcd = infrared_uint8_bgr.astype(np.float32)
        b_vis=visible_float_fcd[:,:,0]
        g_vis=visible_float_fcd[:,:,1]
        r_vis=visible_float_fcd[:,:,2]
        vmax = np.maximum.reduce([b_vis, g_vis, r_vis])
        vmax = np.maximum(vmax, 1.0)
        ir_intensity = infrared_float_fcd[:,:,0] # Assumes BGR IR, takes Blue channel
        
        a = (ir_intensity / 255.0)**2  # a = (I_ir/255.0)^gamma
        k_intermediate = (vmax + 255.0)/2.0                       # k_intermediate = (V_max+255)/2
        k = (a * k_intermediate / vmax) + 0.5                     # k = ([(I_ir/255.0)^gamma] * [{(V_max+255)/2}/V_max]) + 0.5
        k_3d = k[:,:,np.newaxis]
        fused_float_calc = visible_float_fcd * k_3d
        return np.clip(fused_float_calc, 0, 255).astype(np.uint8)    

# Pixel FCDFusion
class _InternalFCDFusion_Pixel:
    def __init__(self): pass
    def fuse(self, visible_uint8_bgr: np.ndarray, infrared_uint8_bgr: np.ndarray) -> np.ndarray:
        h, w = visible_uint8_bgr.shape[:2]
        fused_float_calc = np.zeros((h, w, 3), dtype=np.float32) 
        visible_float_fcd = visible_uint8_bgr.astype(np.float32) 
        infrared_float_fcd = infrared_uint8_bgr.astype(np.float32)
        
        for i in range(h):
            for j in range(w):
                b, g, r = visible_float_fcd[i, j]
                vmax = max(b, g, r, 1.0)
                ir_intensity = infrared_float_fcd[i, j, 0] 
                a = (ir_intensity / 255.0) ** 2
                k_intermediate = (vmax + 255.0) / 2.0
                k = (a * k_intermediate / vmax) + 0.5
                fused_float_calc[i, j, 0] = b * k
                fused_float_calc[i, j, 1] = g * k
                fused_float_calc[i, j, 2] = r * k
        return np.clip(fused_float_calc, 0, 255).astype(np.uint8)

# DenseFuse 
class DenseFuseWrapper:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = None
        
        if not DL_MODELS_AVAILABLE:
            raise ImportError("DenseFuse model.py not found.")
            
        try:
            # Initialize the architecture from model.py
            self.model = DenseNet()
            
            # Load weights
            # Note: Sometimes weights are saved as a dict {'model': state_dict} or just state_dict
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            print(f"DenseFuse Model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load DenseFuse weights: {e}")
            self.model = None

    def fuse(self, visible_uint8, infrared_uint8):
        if self.model is None: return visible_uint8
        
        # DenseFuse Logic (Derived from train.py/Test class):
        # 1. Convert BGR to YCrCb
        vis_ycrcb = cv2.cvtColor(visible_uint8, cv2.COLOR_BGR2YCrCb)
        ir_ycrcb = cv2.cvtColor(infrared_uint8, cv2.COLOR_BGR2YCrCb)
        
        # 2. Extract Y channel (Luminance) for the Network
        vis_y = vis_ycrcb[:,:,0:1].astype(np.float32) / 127.5 - 1.0 # Normalize -1 to 1
        ir_y = ir_ycrcb[:,:,0:1].astype(np.float32) / 127.5 - 1.0
        
        # 3. Extract Cb/Cr channels for manual fusion
        vis_cb, vis_cr = vis_ycrcb[:,:,1], vis_ycrcb[:,:,2]
        ir_cb, ir_cr = ir_ycrcb[:,:,1], ir_ycrcb[:,:,2]
        
        # 4. Prepare Tensors
        vis_y_tensor = torch.from_numpy(vis_y).permute(2, 0, 1).unsqueeze(0).to(self.device)
        ir_y_tensor = torch.from_numpy(ir_y).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # 5. Network Inference (Fuse Y Channel)
        with torch.no_grad():
            fused_y_tensor = self.model(vis_y_tensor, ir_y_tensor)
        
        # 6. Post-process Y Channel
        fused_y = fused_y_tensor.squeeze().cpu().numpy()
        fused_y = (fused_y + 1.0) * 127.5
        fused_y = np.clip(fused_y, 0, 255).astype(np.uint8)
        
        # 7. Fuse Cb/Cr Channels (Weighted Average Method from paper)
        # Logic: w = abs(val) / sum(abs(val))
        EPS = 1e-8
        
        # Cb Fusion
        vis_cb_f = vis_cb.astype(np.float32) - 128.0
        ir_cb_f = ir_cb.astype(np.float32) - 128.0
        w_vis_cb = (np.abs(vis_cb_f) + EPS) / (np.abs(vis_cb_f) + np.abs(ir_cb_f) + EPS)
        w_ir_cb = (np.abs(ir_cb_f) + EPS) / (np.abs(vis_cb_f) + np.abs(ir_cb_f) + EPS)
        fused_cb = (w_vis_cb * vis_cb_f + w_ir_cb * ir_cb_f) + 128.0
        
        # Cr Fusion
        vis_cr_f = vis_cr.astype(np.float32) - 128.0
        ir_cr_f = ir_cr.astype(np.float32) - 128.0
        w_vis_cr = (np.abs(vis_cr_f) + EPS) / (np.abs(vis_cr_f) + np.abs(ir_cr_f) + EPS)
        w_ir_cr = (np.abs(ir_cr_f) + EPS) / (np.abs(vis_cr_f) + np.abs(ir_cr_f) + EPS)
        fused_cr = (w_vis_cr * vis_cr_f + w_ir_cr * ir_cr_f) + 128.0
        
        # 8. Merge and Convert back to BGR
        fused_ycrcb = np.zeros_like(vis_ycrcb)
        fused_ycrcb[:,:,0] = fused_y
        fused_ycrcb[:,:,1] = np.clip(fused_cb, 0, 255).astype(np.uint8)
        fused_ycrcb[:,:,2] = np.clip(fused_cr, 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(fused_ycrcb, cv2.COLOR_YCrCb2BGR)

# VGG-19
class VGG19FusionWrapper:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        print("Loading VGG19 (Heavy DL Model)...")
        # Load standard VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.model = vgg.features
        self.model.float() # Force model to Float32
        self.model.to(self.device)
        self.model.eval()
        print("VGG19 Model Loaded.")

    def fuse(self, visible_uint8, infrared_uint8):
        # 1. Preprocess
        vis_rgb = cv2.cvtColor(visible_uint8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ir_rgb = cv2.cvtColor(infrared_uint8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Normalize (Explicitly use float32 to prevent promotion to float64)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        vis_norm = (vis_rgb - mean) / std
        ir_norm = (ir_rgb - mean) / std
        
        # To Tensor (Force .float() to be safe)
        vis_tensor = torch.from_numpy(vis_norm).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        ir_tensor = torch.from_numpy(ir_norm).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # 2. Inference
        with torch.no_grad():
            vis_feat = self.model[:21](vis_tensor)
            ir_feat = self.model[:21](ir_tensor)
        
        # 3. Fusion Strategy
        vis_activity = torch.mean(torch.abs(vis_feat), dim=1, keepdim=True)
        ir_activity = torch.mean(torch.abs(ir_feat), dim=1, keepdim=True)
        mask = (vis_activity > ir_activity).float()
        
        # Resize mask
        mask_resized = F.interpolate(mask, size=(visible_uint8.shape[0], visible_uint8.shape[1]), mode='nearest')
        mask_np = mask_resized.squeeze().cpu().numpy()
        mask_3d = np.repeat(mask_np[:, :, np.newaxis], 3, axis=2)
        
        # Reconstruct
        fused = visible_uint8.astype(np.float32) * mask_3d + infrared_uint8.astype(np.float32) * (1 - mask_3d)
        return np.clip(fused, 0, 255).astype(np.uint8)

# Resnet-152
class ResNet152FusionWrapper:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        print("Loading ResNet-152 (Heavy DL Backbone)...")
        weights = models.ResNet152_Weights.IMAGENET1K_V1
        self.model = models.resnet152(weights=weights)
        self.model.float() # Force model to Float32
        self.model.to(self.device)
        self.model.eval()
        print("ResNet-152 Model Loaded.")

    def fuse(self, visible_uint8, infrared_uint8):
        # 1. Preprocess
        vis_rgb = cv2.cvtColor(visible_uint8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ir_rgb = cv2.cvtColor(infrared_uint8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Normalize (Explicitly use float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        vis_norm = (vis_rgb - mean) / std
        ir_norm = (ir_rgb - mean) / std
        
        # To Tensor (Force .float())
        vis_tensor = torch.from_numpy(vis_norm).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        ir_tensor = torch.from_numpy(ir_norm).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # 2. Inference
        with torch.no_grad():
            x_v = self.model.conv1(vis_tensor)
            x_v = self.model.bn1(x_v)
            x_v = self.model.relu(x_v)
            x_v = self.model.maxpool(x_v)
            feat_v = self.model.layer1(x_v)
            
            x_i = self.model.conv1(ir_tensor)
            x_i = self.model.bn1(x_i)
            x_i = self.model.relu(x_i)
            x_i = self.model.maxpool(x_i)
            feat_i = self.model.layer1(x_i)
        
        # 3. Fusion Strategy
        vis_activity = torch.mean(torch.abs(feat_v), dim=1, keepdim=True)
        ir_activity = torch.mean(torch.abs(feat_i), dim=1, keepdim=True)
        mask = (vis_activity > ir_activity).float()
        
        # 4. Reconstruction
        mask_resized = F.interpolate(mask, size=(visible_uint8.shape[0], visible_uint8.shape[1]), mode='nearest')
        mask_np = mask_resized.squeeze().cpu().numpy()
        mask_3d = np.repeat(mask_np[:, :, np.newaxis], 3, axis=2)
        
        fused = visible_uint8.astype(np.float32) * mask_3d + infrared_uint8.astype(np.float32) * (1 - mask_3d)
        return np.clip(fused, 0, 255).astype(np.uint8)

