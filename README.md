# AIRC-Fusion: Real-Time Infrared-Visible (IR-VIS) Video Fusion on CPU-based Hardware

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)

Official Python implementation for the paper "AIRC-Fusion: Achieving Real-Time Infrared-Visible (IR-VIS) Video Fusion on CPU-based Hardware". This project provides a comprehensive framework for the evaluation of classical fusion algorithms and introduces a novel, high-performance vectorized implementation capable of real-time video processing on standard CPUs.

---

### Abstract

The fusion of infrared (IR) and visible (VIS) video is critical for robust perception, but the computational intensity of state-of-the-art deep learning models often renders them impractical for real-time, CPU-only deployment. This paper directly challenges the notion that high-performance fusion requires specialized hardware. We introduce **Vectorized FCDFusion**, a novel, computationally-focused implementation of a classical fusion algorithm that achieves a transformative speedup of over 120x compared to its original pixel-loop form. This optimization enables real-time performance (>65 FPS on average) without sacrificing the algorithm's high-fidelity image quality (SSIM > 0.74). To validate this contribution, we conducted a rigorous comparative analysis against three other traditional fusion families on challenging real-world datasets (FLIR ADAS, CAMEL). Our findings reveal a clear and practical trade-off, providing a data-driven framework for practitioners and demonstrating that an optimized classical algorithm offers a powerful, efficient, and readily deployable alternative to deep learning for real-time multimodal perception systems.

### Key Features

*   **Five Classical Fusion Algorithms:** Implementations of four major families of traditional fusion methods:
    1.  **Direct Mapping:** `FCDFusion (Pixel)` [1]
    2.  **Base/Detail Decomposition:** `Model2024` [2]
    3.  **Saliency-Based Filtering:** `Model2020` [3]
    4.  **Gradient-Domain:** `Model2015` [4]
*   **High-Performance Vectorized Algorithm:** Our proposed `Vectorized FCDFusion` provides a >120x speedup over the original.
*   **Comprehensive Benchmarking:** A full pipeline for evaluating single images and video sequences with a suite of 10+ quantitative metrics.
*   **Live Hardware Demonstration:** A real-time demo script (`live_fusion_ssim.py`) for running on a Raspberry Pi 5 with live camera feeds.

### Live Demonstration

Below is a demonstration of the `live_fusion_ssim.py` script running on a Raspberry Pi 5, showcasing real-time switching between fusion algorithms and live metric updates.

![Live Fusion Demo](https://your-link-to-a-demo-gif-or-image.com/demo.gif)
*(**Needed:** pi demo video needed.)*

---

### Getting Started

#### 1. Clone the Repository
```git clone https://github.com/rayeedaabir/AIRC-Fusion.git```
```cd AIRC-Fusion```

#### 2. Install Dependencies
All required libraries are listed in requirements.txt.
```pip install -r requirements.txt```

#### How to Use the Framework
Our framework is controlled via main.py for automated processing and live_fusion_ssim.py for live demos.

### Single Image Fusion
To fuse a single pair of visible and infrared images:
```python main.py --mode image --visible "path/to/visible_image.jpg" --infrared "path/to/infrared_image.bmp" --output "path/to/fused_image.jpg" --core-fusion-method fcdfusion_vectorized --display```
This will generate a fused video, a side-by-side comparison video, and a JSON file with the aggregated metrics.

#### Video Fusion Pipeline
To process an entire video dataset (e.g., FLIR):
```python main.py --mode pipeline --dataset-type flir --base-dataset-path "path/to/FLIR_ADAS_v2" --output "path/to/output_folder" --core-fusion-method fcdfusion_vectorized```

#### Live Raspberry Pi Demo
To run the live fusion demo with two connected cameras:
```python3 live_fusion_ssim.py```
(Note: You may need to edit the RGB_CAMERA_INDEX and IR_CAMERA_INDEX variables in the script to match your hardware setup.)

---
### Project Structure

A brief overview of the key files in this repository:

* ```main.py```: The primary entry point for running all automated experiments.

* ```raspberry_pi11.py```: The script for the live Raspberry Pi demonstration.

* ```image_fusion_processor.py```: The core module containing the implementations of all five fusion algorithms.

* ```dataset_processor.py```: Handles data loading, preprocessing, and parallel processing for datasets.

* ```evaluation_metrics.py```: Contains the functions for calculating all quantitative performance metrics.

* ```config.py```: Manages dataset paths and default algorithm hyperparameters.

* ```video_utils.py```: Contains helper functions for assembling the final fused videos.

---

#### How to Cite
If you use this work in your research, please cite our paper:
```
@inproceedings{AIRC-Fusion2025,
  title={AIRC-Fusion: Achieving Real-Time Infrared-Visible (IR-VIS) Video Fusion on CPU-based Hardware},
  author={Ahsan, Islam, Rahman, Chowdhury},
  booktitle={Conference or Journal Name (TBD)},
  year={2025}
}
```
(Note: Please update the booktitle and year once we have them.)
---
---

### Key References

This work is a comparative analysis based on the principles from the following foundational papers:

[1] H. Li and Y. Fu, "FCDFusion: A Fast, Low Color Deviation Method for Fusing Visible and Infrared Image Pairs." *Computational Visual Media*, 2025.

[2] M. K. Panda, et al., "A Weight Induced Contrast Map for Infrared and Visible Image Fusion." *Computers and Electrical Engineering*, 2024.

[3] Y. Zhang, et al., "Infrared and Visible Image Fusion with Hybrid Image Filtering." *Mathematical Problems in Engineering*, 2020.

[4] D. Connah, et al., "Spectral Edge: Gradient-Preserving Spectral Mapping for Image Fusion." *Journal of the Optical Society of America A*, 2015.

---
### License
This project is licensed under the MIT License. See the LICENSE file for details.
### Acknowledgments
We would like to thank our faculty advisor, Dr. Mohammad Rashedur Rahman, for his invaluable guidance and support throughout this project.

