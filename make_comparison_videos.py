"""
make_comparison_videos.py  (NEW - one aligned all-models grid video per dataset)
--------------------------------------------------------------------------------
After run_all.py, builds ONE side-by-side video per dataset that tiles the
proposed method, the two source images, and every baseline, using only frames
present for ALL models (so the grid is aligned and the same length). Layout: a
3x4 grid with Vectorized FCDFusion (highlighted) top-left.

Usage:
  python make_comparison_videos.py --dataset CAMEL_Seq2
  python make_comparison_videos.py                       # all datasets
Output: <OUTPUT_ROOT>/results/qualitative/COMPARISON_<DATASET>.mp4
"""
import os, glob, json, argparse
import cv2, numpy as np
import run_config as RC

TILE_W, LABEL_H, COLS = 300, 26, 4          # 4 columns -> 3 rows for 11 tiles
PROPOSED = "fcdfusion_vectorized"


def _pairs(ds, ref_model_dir):
    out = []
    if ds["kind"] == "flir" and ds.get("json_map_path") and os.path.exists(ds["json_map_path"]):
        for rgb, th in json.load(open(ds["json_map_path"])).items():
            out.append((os.path.join(ds["visible_dir"], rgb), os.path.join(ds["infrared_dir"], th), rgb))
    else:
        for p in sorted(glob.glob(os.path.join(ref_model_dir, "*.jpg"))):
            bn = os.path.basename(p)
            out.append((os.path.join(ds["visible_dir"], bn), os.path.join(ds["infrared_dir"], bn), bn))
    return out


def _tile(img, text, highlight=False):
    if img is None:
        img = np.zeros((TILE_W, TILE_W, 3), np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    img = cv2.resize(img, (TILE_W, int(h * TILE_W / w)))
    bar_color = (40, 110, 40) if highlight else (30, 30, 30)   # green bar for the proposed method
    bar = np.full((LABEL_H, TILE_W, 3), bar_color, np.uint8)
    cv2.putText(bar, text[:38], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def build(results_dir, ds):
    models = [m for m in RC.MODELS
              if glob.glob(os.path.join(results_dir, ds["name"], m, "fused_frames", "*.jpg"))]
    if not models:
        print(f"[{ds['name']}] no fused frames yet"); return
    # frames present for EVERY model -> aligned, same length
    common = None
    for m in models:
        bns = {os.path.basename(p) for p in glob.glob(os.path.join(results_dir, ds["name"], m, "fused_frames", "*.jpg"))}
        common = bns if common is None else (common & bns)
    ref = os.path.join(results_dir, ds["name"], models[0], "fused_frames")
    pairs = [t for t in _pairs(ds, ref) if t[2] in common]
    # tile order: proposed first (highlighted), then the two sources, then the rest
    others = [m for m in models if m != PROPOSED]
    print(f"[{ds['name']}] {len(models)} models, {len(pairs)} common frames")

    out_path = os.path.join(results_dir, "qualitative", f"COMPARISON_{ds['name']}.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = None
    for vis_p, ir_p, bn in pairs:
        vis = cv2.imread(vis_p); ir = cv2.imread(ir_p)
        if vis is None:
            continue
        def fused(m):
            return cv2.imread(os.path.join(results_dir, ds["name"], m, "fused_frames", bn))
        tiles = []
        if PROPOSED in models:
            tiles.append(_tile(fused(PROPOSED), "Vectorized FCDFusion (Ours)", highlight=True))
        tiles.append(_tile(vis, "Visible"))
        tiles.append(_tile(ir, "Infrared"))
        for m in others:
            tiles.append(_tile(fused(m), RC.MODEL_LABELS.get(m, m)))
        h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
        tiles = [np.vstack([t, np.zeros((h - t.shape[0], t.shape[1], 3), np.uint8)]) for t in tiles]
        rows = []
        for i in range(0, len(tiles), COLS):
            row = tiles[i:i + COLS]
            while len(row) < COLS:
                row.append(np.zeros((h, w, 3), np.uint8))
            rows.append(np.hstack(row))
        grid = np.vstack(rows)
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     ds.get("video_fps", 30), (grid.shape[1], grid.shape[0]))
        writer.write(grid)
    if writer:
        writer.release(); print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default=None); a = ap.parse_args()
    results_dir = os.path.join(RC.OUTPUT_ROOT, "results")
    for ds in RC.DATASETS:
        if a.dataset and ds["name"] != a.dataset:
            continue
        build(results_dir, ds)


if __name__ == "__main__":
    main()
