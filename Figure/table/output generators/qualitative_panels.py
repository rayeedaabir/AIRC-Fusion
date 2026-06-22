"""
qualitative_panels.py  (UPDATED 21 Jun 2026 - 3x4 grid + per-pane LETTER labels for Cureus)
-------------------------------------------------------------------------------------------
Builds the main-text single-frame qualitative panels (Figure 6 FLIR, Figure 7 CAMEL
Seq-13). Layout matches make_comparison_videos.py / make_static_comparison.py: a 3x4 grid
with Vectorized FCDFusion (highlighted) top-left, then Visible, Infrared, then every
baseline. Cureus requires each pane of a multi-pane figure to carry a LETTER (A, B, C...);
each tile bar now reads "(A) <name>". Define the letters in the figure legend.

For each dataset in run_config.QUALITATIVE_FRAME_INDEX it writes
    results/qualitative/<DATASET>_frameXXXX.png   (>=900px wide; PNG)
and prints the legend mapping you paste into the figure legend.
"""

import os
import glob
import string
import cv2
import numpy as np
import run_config as RC

TILE_W, LABEL_H, COLS = 300, 26, 4
PROPOSED = "fcdfusion_vectorized"


def _letter(i):
    s = ""; i += 1
    while i > 0:
        i, r = divmod(i - 1, 26); s = string.ascii_uppercase[r] + s
    return s


def _tile(img, text, highlight=False):
    if img is None:
        img = np.zeros((TILE_W, TILE_W, 3), np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    img = cv2.resize(img, (TILE_W, int(h * TILE_W / w)))
    bar_color = (40, 110, 40) if highlight else (30, 30, 30)
    bar = np.full((LABEL_H, TILE_W, 3), bar_color, np.uint8)
    cv2.putText(bar, text[:40], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _grid(tiles):
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
    tiles = [np.vstack([t, np.zeros((h - t.shape[0], t.shape[1], 3), np.uint8)]) for t in tiles]
    rows = []
    for i in range(0, len(tiles), COLS):
        row = tiles[i:i + COLS]
        while len(row) < COLS:
            row.append(np.zeros((h, w, 3), np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def _nth(folder, idx, exts=("*.jpg", "*.png")):
    files = []
    for e in exts:
        files += glob.glob(os.path.join(folder, e))
    files = sorted(files)
    if not files:
        return None
    idx = min(idx, len(files) - 1)
    return cv2.imread(files[idx])


def build_all_panels(results_dir):
    out_dir = os.path.join(results_dir, "qualitative")
    os.makedirs(out_dir, exist_ok=True)
    ds_by_name = {d["name"]: d for d in RC.DATASETS}

    for ds_name, idx in RC.QUALITATIVE_FRAME_INDEX.items():
        ds = ds_by_name.get(ds_name)
        if ds is None:
            print(f"[qualitative_panels] dataset {ds_name} not in run_config; skipped")
            continue
        vis = _nth(ds["visible_dir"], idx)
        ir = _nth(ds["infrared_dir"], idx)

        def fused(model):
            return _nth(os.path.join(results_dir, ds_name, model, "fused_frames"), idx)

        # collect (image, name, highlight) in reading order, then letter them
        items = []
        if PROPOSED in RC.MODELS:
            items.append((fused(PROPOSED), "Vectorized FCDFusion (Ours)", True))
        items.append((vis, "Visible", False))
        items.append((ir, "Infrared", False))
        for model in [m for m in RC.MODELS if m != PROPOSED]:
            f = fused(model)
            if f is not None:
                items.append((f, RC.MODEL_LABELS.get(model, model), False))

        tiles, legend = [], []
        for i, (img, name, hl) in enumerate(items):
            let = _letter(i)
            tiles.append(_tile(img, f"({let}) {name}", highlight=hl))
            legend.append(f"({let}) {name}")

        grid = _grid(tiles)
        if grid is not None:
            path = os.path.join(out_dir, f"{ds_name}_frame{idx:04d}.png")
            cv2.imwrite(path, grid)
            print(f"[qualitative_panels] wrote {path}")
            print(f"    legend: {'; '.join(legend)}.")


if __name__ == "__main__":
    build_all_panels(os.path.join(RC.OUTPUT_ROOT, "results"))
# --- end of file (sentinel) ---