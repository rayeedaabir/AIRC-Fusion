"""
make_static_comparison.py  (UPDATED 21 Jun 2026 - 3x4 grid + per-pane LETTER labels)
------------------------------------------------------------------------------------
For the STATIC pairs (RGB-NIR, Landsat; keep TNO out of the CC-BY article): runs every
model on the one pair and tiles a 3x4 grid with Vectorized FCDFusion (highlighted) top-left,
then Visible, Infrared, then the baselines - consistent with make_comparison_videos.py.
Each pane carries a Cureus letter label "(A) <name>"; the printed legend maps them.

Run:
  python make_static_comparison.py --vis v.png --ir i.png --out RGBNIR_panel.png
"""
import os, argparse, string
import cv2, numpy as np
import run_config as RC
from image_fusion_processor import ImageFusionProcessor
from run_all import model_kwargs

TILE_W, LABEL_H, COLS = 300, 26, 4
PROPOSED = "fcdfusion_vectorized"


def _letter(i):
    s = ""; i += 1
    while i > 0:
        i, r = divmod(i - 1, 26); s = string.ascii_uppercase[r] + s
    return s


def _tile(img, text, highlight=False):
    if img is None: img = np.zeros((TILE_W, TILE_W, 3), np.uint8)
    if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]; img = cv2.resize(img, (TILE_W, int(h * TILE_W / w)))
    bar = np.full((LABEL_H, TILE_W, 3), (40, 110, 40) if highlight else (30, 30, 30), np.uint8)
    cv2.putText(bar, text[:40], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis", required=True); ap.add_argument("--ir", required=True)
    ap.add_argument("--out", default="static_comparison.png"); ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()
    vis = cv2.imread(a.vis); ir = cv2.imread(a.ir)
    if vis is None or ir is None: raise SystemExit("could not read --vis or --ir")
    models = a.models or RC.MODELS

    def run(key):
        try:
            f, _ = ImageFusionProcessor(core_fusion_method=key, **model_kwargs()).process_frame(vis, ir)
            print(f"  {key}: ok"); return f
        except Exception as e:
            print(f"  {key}: skipped ({e})"); return None

    # collect (image, name, highlight) in reading order, then letter them
    items = []
    if PROPOSED in models:
        items.append((run(PROPOSED), "Vectorized FCDFusion (Ours)", True))
    items.append((vis, "Visible", False))
    items.append((ir, "Infrared", False))
    for key in [m for m in models if m != PROPOSED]:
        items.append((run(key), RC.MODEL_LABELS.get(key, key), False))

    tiles, legend = [], []
    for i, (img, name, hl) in enumerate(items):
        let = _letter(i)
        tiles.append(_tile(img, f"({let}) {name}", highlight=hl))
        legend.append(f"({let}) {name}")

    h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
    tiles = [np.vstack([t, np.zeros((h - t.shape[0], t.shape[1], 3), np.uint8)]) for t in tiles]
    rows = []
    for i in range(0, len(tiles), COLS):
        row = tiles[i:i + COLS]
        while len(row) < COLS: row.append(np.zeros((h, w, 3), np.uint8))
        rows.append(np.hstack(row))
    cv2.imwrite(a.out, np.vstack(rows))
    print(f"wrote {a.out}")
    print(f"legend: {'; '.join(legend)}.")


if __name__ == "__main__": main()
# --- end of file (sentinel) ---