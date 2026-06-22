"""
prepare_landsat_pair.py  (NEW 21 Jun 2026)
------------------------------------------------------------------------------
Landsat ships as four separate single-band GeoTIFFs (B2 Blue, B3 Green, B4 Red,
B5 Near-Infrared), each ~90 MB, so make_static_comparison.py cannot read them
directly. This script reuses the exact preparation logic from process_landsat.py:
it crops a region, builds an 8-bit visible RGB image from B4/B3/B2 and an 8-bit
"infrared" image from B5, and saves the pair as small PNGs. Then feed those PNGs
to make_static_comparison.py like any other dataset.

Run (one line):
  python prepare_landsat_pair.py --b2 ..._B2.TIF --b3 ..._B3.TIF --b4 ..._B4.TIF --b5 ..._B5.TIF \
         --out_vis landsat_vis.png --out_ir landsat_ir.png
Then:
  python make_static_comparison.py --vis landsat_vis.png --ir landsat_ir.png --out Landsat_panel.png

Notes:
- --crop / --y / --x choose the crop window (default 1024x1024 at row 4000, col 4000,
  matching process_landsat.py). Use --crop 0 to keep the FULL scene (large output).
- Normalization matches process_landsat.py: the merged RGB is contrast-stretched
  jointly on the 0.5-99.5 percentile; the NIR band is stretched independently.
"""
import argparse
import cv2
import numpy as np
import tifffile


def load(path):
    img = tifffile.imread(path).astype(np.float32)
    print(f"loaded {path}  shape={img.shape} dtype-as-float")
    return img


def stretch(arr):
    lo, hi = np.percentile(arr, (0.5, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b2", required=True, help="Blue band TIFF")
    ap.add_argument("--b3", required=True, help="Green band TIFF")
    ap.add_argument("--b4", required=True, help="Red band TIFF")
    ap.add_argument("--b5", required=True, help="Near-infrared band TIFF")
    ap.add_argument("--crop", type=int, default=1024, help="crop size (0 = full scene)")
    ap.add_argument("--y", type=int, default=4000, help="crop top row")
    ap.add_argument("--x", type=int, default=4000, help="crop left col")
    ap.add_argument("--out_vis", default="landsat_vis.png")
    ap.add_argument("--out_ir", default="landsat_ir.png")
    a = ap.parse_args()

    blue, green, red, nir = (load(p) for p in (a.b2, a.b3, a.b4, a.b5))

    if a.crop and a.crop > 0:
        h, w = blue.shape[:2]
        y = min(a.y, max(0, h - a.crop)); x = min(a.x, max(0, w - a.crop))
        sl = (slice(y, y + a.crop), slice(x, x + a.crop))
        blue, green, red, nir = blue[sl], green[sl], red[sl], nir[sl]
        print(f"cropped to {a.crop}x{a.crop} at (row={y}, col={x})")

    vis_f = cv2.merge([blue, green, red])                  # BGR order for OpenCV
    vmin, vmax = np.percentile(vis_f, (0.5, 99.5))
    vis = np.clip((vis_f - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    ir = cv2.cvtColor(stretch(nir), cv2.COLOR_GRAY2BGR)

    cv2.imwrite(a.out_vis, vis)
    cv2.imwrite(a.out_ir, ir)
    print(f"wrote {a.out_vis} {vis.shape} and {a.out_ir} {ir.shape}")


if __name__ == "__main__":
    main()
