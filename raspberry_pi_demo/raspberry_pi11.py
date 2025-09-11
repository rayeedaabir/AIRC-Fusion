import cv2
import threading
import time
from picamera2 import Picamera2
from image_fusion_processor import ImageFusionProcessor
from evaluation_metrics import compute_ssim_accuracy_metric
import numpy as np
import json
import os

# -----------------------
# Orientation Save/Load
# -----------------------
ORIENTATION_FILE = "camera_orientation.json"

# Default alignment params
ir_tx, ir_ty, ir_scale = 0, 0, 1.0
rgb_tx, rgb_ty, rgb_scale = 0, 0, 1.0

if os.path.exists(ORIENTATION_FILE):
    try:
        with open(ORIENTATION_FILE, "r") as f:
            saved = json.load(f)
            ir_tx, ir_ty, ir_scale = saved.get("ir_tx", 0), saved.get("ir_ty", 0), saved.get("ir_scale", 1.0)
            rgb_tx, rgb_ty, rgb_scale = saved.get("rgb_tx", 0), saved.get("rgb_ty", 0), saved.get("rgb_scale", 1.0)
        print(f"[INFO] Loaded saved orientation.")
        print(f"    IR  → tx={ir_tx}, ty={ir_ty}, scale={ir_scale:.3f}")
        print(f"    RGB → tx={rgb_tx}, ty={rgb_ty}, scale={rgb_scale:.3f}")
    except Exception as e:
        print(f"[WARN] Could not load saved orientation: {e}")

def save_orientation():
    try:
        with open(ORIENTATION_FILE, "w") as f:
            json.dump({
                "ir_tx": ir_tx, "ir_ty": ir_ty, "ir_scale": ir_scale,
                "rgb_tx": rgb_tx, "rgb_ty": rgb_ty, "rgb_scale": rgb_scale
            }, f)
        print(f"[INFO] Orientation saved.")
    except Exception as e:
        print(f"[ERROR] Could not save orientation: {e}")

# -----------------------
# Camera Threads (with .copy() fix)
# -----------------------
class USBCameraThread(threading.Thread):
    def __init__(self, device_index=0, resolution=(640, 480)):
        super(USBCameraThread, self).__init__()
        self.cap = cv2.VideoCapture(device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.latest_frame = None
        self.running = True
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame.copy()
            time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()

class CSICameraThread(threading.Thread):
    def __init__(self, camera_num=0, resolution=(640, 480)):
        super(CSICameraThread, self).__init__()
        self.picam2 = Picamera2(camera_num=camera_num)
        config = self.picam2.create_preview_configuration(main={"size": resolution})
        self.picam2.configure(config)
        self.latest_frame = None
        self.running = True
        self.lock = threading.Lock()

    def run(self):
        self.picam2.start()
        while self.running:
            frame_rgb = self.picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            with self.lock:
                self.latest_frame = frame_bgr.copy()
            time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        self.running = False
        try: self.picam2.stop()
        except Exception: pass

# -----------------------
# Initialization
# -----------------------
TARGET_RESOLUTION = (640, 480)
rgb_camera = USBCameraThread(device_index=0, resolution=TARGET_RESOLUTION)
ir_camera = CSICameraThread(camera_num=0, resolution=TARGET_RESOLUTION)
rgb_camera.start()
ir_camera.start()

print("Waiting for cameras to initialize...")
time.sleep(2.0)

print("Initializing Fusion Processors...")
processors = {
    '1': ImageFusionProcessor(core_fusion_method="fcdfusion_vectorized"),
    '2': ImageFusionProcessor(core_fusion_method="model2024"),
    '3': ImageFusionProcessor(core_fusion_method="model2020"),
    '4': ImageFusionProcessor(core_fusion_method="model2015"),
    '5': ImageFusionProcessor(core_fusion_method="fcdfusion_pixel")
}
current_processor_key = '1'
ssim_value = 0.0
last_ssim_update_time = 0
fine_tune_mode = False # New state for fine-tuning

print("Initialization Complete. Starting Live Fusion.")
print("Controls:")
print("  1-5 : Switch models")
print("  IR  → w/s/a/d (move) | +/- (scale)")
print("  RGB → i/k/j/l (move) | [ ] (scale)")
print("  m   : Toggle fine-tune step for alignment")
print("  p   : Save alignment | r : Reset | q : Quit")

win_name = "AIRC-Fusion Live Demo"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 1920, 560)

# -----------------------
# Main loop
# -----------------------
try:
    while True:
        visible_frame = rgb_camera.read()
        infrared_frame = ir_camera.read()

        if visible_frame is None or infrared_frame is None:
            time.sleep(0.05)
            continue

        rgb_h, rgb_w = visible_frame.shape[:2]
        
        # Apply transforms using BORDER_CONSTANT to create black background
        M_rgb = cv2.getRotationMatrix2D((rgb_w//2, rgb_h//2), 0.0, rgb_scale)
        M_rgb[0, 2] += rgb_tx; M_rgb[1, 2] += rgb_ty
        visible_frame_adj = cv2.warpAffine(visible_frame, M_rgb, (rgb_w, rgb_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        infrared_resized = cv2.resize(infrared_frame, (rgb_w, rgb_h), interpolation=cv2.INTER_AREA)
        M_ir = cv2.getRotationMatrix2D((rgb_w//2, rgb_h//2), 0.0, ir_scale)
        M_ir[0, 2] += ir_tx; M_ir[1, 2] += ir_ty
        infrared_aligned = cv2.warpAffine(infrared_resized, M_ir, (rgb_w, rgb_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # --- FIX #1: CREATE AND USE AN INTERSECTION MASK ---
        # Create a mask for the valid (non-black) region of each warped image
        gray_visible = cv2.cvtColor(visible_frame_adj, cv2.COLOR_BGR2GRAY)
        mask_visible = cv2.threshold(gray_visible, 0, 255, cv2.THRESH_BINARY)[1]
        gray_ir = cv2.cvtColor(infrared_aligned, cv2.COLOR_BGR2GRAY)
        mask_ir = cv2.threshold(gray_ir, 0, 255, cv2.THRESH_BINARY)[1]
        
        # The final mask is the intersection of where both images have valid data
        intersection_mask = cv2.bitwise_and(mask_visible, mask_ir)
        
        # Fusion
        fusion_processor = processors[current_processor_key]
        start_time = time.time()
        fused_image_raw, _ = fusion_processor.process_frame(visible_frame_adj, infrared_aligned)
        proc_time = time.time() - start_time
        fps = 1.0 / proc_time if proc_time > 0 else 0

        # Apply the intersection mask to the fused output to remove border artifacts
        fused_image = cv2.bitwise_and(fused_image_raw, fused_image_raw, mask=intersection_mask)

        # Metrics
        if time.time() - last_ssim_update_time > 2.0:
            ssim_value = compute_ssim_accuracy_metric(fused_image, visible_frame_adj, infrared_aligned)
            last_ssim_update_time = time.time()

        # Display preparation
        h, w, _ = fused_image.shape
        infrared_color = cv2.applyColorMap(cv2.cvtColor(infrared_aligned, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)
        METRICS_BAR_HEIGHT = 80
        def add_info_bar(image):
            return cv2.copyMakeBorder(image, 0, METRICS_BAR_HEIGHT, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        visible_with_bar = add_info_bar(visible_frame_adj)
        infrared_with_bar = add_info_bar(infrared_color)
        fused_with_bar = add_info_bar(fused_image)
        
        # Display Info Text
        fine_tune_text = "(Fine)" if fine_tune_mode else ""
        cv2.putText(visible_with_bar, f"RGB: tx={rgb_tx} ty={rgb_ty} scale={rgb_scale:.3f}", (10, h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(visible_with_bar, f"Controls: i/k/j/l [ ] {fine_tune_text}", (10, h+55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(infrared_with_bar, f"IR: tx={ir_tx} ty={ir_ty} scale={ir_scale:.3f}", (10, h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(infrared_with_bar, f"Controls: w/s/a/d +/- {fine_tune_text}", (10, h+55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(fused_with_bar, f"Model: {fusion_processor.core_fusion_method}", (10, h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(fused_with_bar, f"FPS: {fps:.1f}", (10, h + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(fused_with_bar, f"SSIM: {ssim_value:.3f}", (180, h + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        combined_view = cv2.hconcat([visible_with_bar, infrared_with_bar, fused_with_bar])
        cv2.imshow(win_name, combined_view)

        # --- FIX #2: ADD FINE-TUNE CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        step = 1 if fine_tune_mode else 4
        scale_step = 0.005 if fine_tune_mode else 0.02

        if ord('1') <= key <= ord('5'):
            current_processor_key = chr(key)
        elif key == ord('m'):
            fine_tune_mode = not fine_tune_mode
            print(f"[INFO] Fine-tune mode {'ENABLED' if fine_tune_mode else 'DISABLED'}")
        elif key == ord('w'): ir_ty -= step
        elif key == ord('s'): ir_ty += step
        elif key == ord('a'): ir_tx -= step
        elif key == ord('d'): ir_tx += step
        elif key in [ord('+'), ord('=')]: ir_scale = min(2.0, ir_scale + scale_step)
        elif key in [ord('-'), ord('_')]: ir_scale = max(0.1, ir_scale - scale_step)
        elif key == ord('i'): rgb_ty -= step
        elif key == ord('k'): rgb_ty += step
        elif key == ord('j'): rgb_tx -= step
        elif key == ord('l'): rgb_tx += step
        elif key == ord('['): rgb_scale = max(0.1, rgb_scale - scale_step)
        elif key == ord(']'): rgb_scale = min(2.0, rgb_scale + scale_step)
        elif key == ord('r'):
            ir_tx, ir_ty, ir_scale = 0, 0, 1.0; rgb_tx, rgb_ty, rgb_scale = 0, 0, 1.0
            print("[INFO] Alignment reset.")
        elif key == ord('p'): save_orientation()
        elif key == ord('q'):
            save_orientation()
            break

except KeyboardInterrupt:
    save_orientation()
finally:
    rgb_camera.stop()
    ir_camera.stop()
    cv2.destroyAllWindows()
    print("Application Closed.")
