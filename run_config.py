"""
run_config.py  (NEW)
--------------------
The ONE file you edit to control a full benchmark sweep. Point the paths at
your PC, choose the models, and run:  python run_all.py
(or a single combo: python run_one.py --model <m> --dataset <d>)
Outputs go to OUTPUT_ROOT/results/ with paper-ready tables in AIRC-Fusion_Results.xlsx
"""

OUTPUT_ROOT = "D:/Codes/499B/AIRC-Fusion_runs"
WORKERS = 1                      # 1 = clean, reproducible per-frame FPS (no thread contention)
MAKE_VIDEOS = True               # final run: build fused + side-by-side videos
MAKE_QUALITATIVE_PANELS = True

MODELS = [
    "model2015",              # Connah Spectral Edge [7] - faithful
    "model2020",              # GFF / Li et al. 2013 [6] -> labelled "Model2013"
    "panda2024",              # Panda [5] - FAITHFUL (replaces old generic "model2024")
    "fcdfusion_pixel",
    "fcdfusion_vectorized",   # proposed method - faithful (== paper Algorithm 1)
    "densefuse_official",   # FAITHFUL DenseFuse [47] (needs official weights at models/densefuse.model)
    "resnet152",
    "vgg19",
    "mdlatlrr",               # MDLatLRR [59] (faithful; needs models/L_8.npy)
]

def _camel(name, num):
    base = f"D:/Codes/499B/Datasets/CAMEL Datasets/Sequence {num}"
    return {"name": name, "kind": "camel",
            "visible_dir": f"{base}/vis", "infrared_dir": f"{base}/IR",
            "seq_vis_txt": f"{base}/Seq{num}-Vis.txt", "seq_ir_txt": f"{base}/Seq{num}-IR.txt",
            "filename_format": "{:06d}.jpg", "pattern": "*.jpg",
            "limit": None, "target_w": None, "target_h": None, "video_fps": 30}

DATASETS = [
    {"name": "FLIR_ADAS", "kind": "flir",
     "visible_dir": "D:/Codes/499B/Datasets/FLIR Dataset/video_rgb_test/data",
     "infrared_dir": "D:/Codes/499B/Datasets/FLIR Dataset/video_thermal_test/data",
     "json_map_path": "D:/Codes/499B/Datasets/FLIR Dataset/rgb_to_thermal_vid_map.json",
     "pattern": "*.jpg", "filename_format": None,
     "limit": None, "target_w": None, "target_h": None, "video_fps": 30},
    _camel("CAMEL_Seq1", 1),
    _camel("CAMEL_Seq2", 2),
    _camel("CAMEL_Seq8", 8),
    _camel("CAMEL_Seq13", 13),
]

MODEL_LABELS = {
    "model2015": "Model2015",
    "model2020": "Model2013",   # GFF, Li, Kang & Hu 2013 [6] (was Zhang 2020)
    "model2024": "Model2024 (generic OLD - do not use)",
    "panda2024": "Model2024",
    "mdlatlrr": "MDLatLRR",
    "fcdfusion_pixel": "FCDFusion (Pixel)",
    "fcdfusion_vectorized": "Vectorized FCDFusion",
    "densefuse": "DenseFuse (mystery net - do not use)",
    "densefuse_official": "DenseFuse",
    "resnet152": "ResNet-152",
    "vgg19": "VGG-19",
}

QUALITATIVE_FRAME_INDEX = {"FLIR_ADAS": 100, "CAMEL_Seq13": 50}

# Per-model frame caps for models too slow to run on every frame of the big FLIR
# set (MDLatLRR ~10 s/frame). Metrics on a representative subset are still valid;
# disclose the subset in the paper. Format: {model_key: {dataset_name: max_frames}}.
LIMIT_OVERRIDE = {
    # high-resolution FLIR (1024x1024) only: cap the slow methods to a representative
    # subset (per-frame metrics are unchanged in expectation). The proposed method and
    # all efficient methods still run the FULL 3749-frame sequence. Disclose this footnote.
    "fcdfusion_pixel": {"FLIR_ADAS": 200},   # sequential loop ~11 s/frame
    "model2015":       {"FLIR_ADAS": 500},   # FFT-Poisson ~6 s/frame
    "mdlatlrr":        {"FLIR_ADAS": 300},   # ~22 s/frame at 1024x1024
    "vgg19":           {"FLIR_ADAS": 500},   # ~14 s/frame on CPU
    "resnet152":       {"FLIR_ADAS": 500},   # ~3.5 s/frame on CPU
}
