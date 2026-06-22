"""
make_matd_profiles.py  (UPDATED 21 Jun 2026 - adds a page-fitting --grid mode for Figure 5)
-------------------------------------------------------------------------------------------
Plots per-frame Mean Absolute Temporal Difference (MATD) for the visible and
infrared SOURCE streams (the "natural motion envelope") against the FUSED stream.
Where the fused curve sits above the source envelope = artificial flicker.

It reads the per-frame CSVs that run_all.py / run_one.py already save:
    <OUTPUT_ROOT>/results/<DATASET>/<MODEL>/dataset_metrics_summary_<model>_per_frame.csv
(columns MATD_vis, MATD_ir, MATD_fused, Temporal_Instability_Score).

THREE ways to use it:

1) One dataset, one image per model (original behaviour):
     python make_matd_profiles.py --dataset CAMEL_Seq2 --models fcdfusion_vectorized model2024 vgg19
   -> MATD_<DATASET>_<model>.png and a tall stacked MATD_<DATASET>_all.png

2) Figure 5 for the paper - a compact grid, rows = a SUBSET of methods, columns =
   the chosen datasets, exported as a single page-fitting PNG (NOT a 20 MB WMF):
     python make_matd_profiles.py --grid --datasets FLIR_ADAS CAMEL_Seq13 \
            --models fcdfusion_vectorized panda2024 densefuse_official vgg19
   -> MATD_grid_FLIR_ADAS_CAMEL_Seq13.png

Tip for Figure 5: keep the row subset to ~4 methods that tell the story - the
proposed method (stays inside the envelope), one frame-recomputing classical method
(rides above), one stable deep method (DenseFuse), and one heavy backbone (VGG-19 /
ResNet-152, large offset). Put the full nine-method stacks in the supplement.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run_config as RC

VIS_C, IR_C, FUSED_C = "#1f77b4", "#ff7f0e", "#2ca02c"

import string
def _letter(i):
    s = ""; i += 1
    while i > 0:
        i, r = divmod(i - 1, 26); s = string.ascii_uppercase[r] + s
    return s


def _find_per_frame_csv(results_dir, dataset, model):
    hits = glob.glob(os.path.join(results_dir, dataset, model, "*per_frame.csv"))
    return hits[0] if hits else None


def _draw(ax, df, title=None, ylabel=None, show_legend=False):
    x = np.arange(len(df))
    ax.plot(x, df["MATD_vis"], color=VIS_C, lw=1.0, label="Visible source")
    ax.plot(x, df["MATD_ir"], color=IR_C, lw=1.0, label="Infrared source")
    ax.plot(x, df["MATD_fused"], color=FUSED_C, lw=1.3, label="Fused output")
    if title:
        ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)
    if show_legend:
        ax.legend(fontsize=7, loc="upper right")


def plot_model(results_dir, dataset, model, ax=None):
    csv = _find_per_frame_csv(results_dir, dataset, model)
    if csv is None or not os.path.exists(csv):
        print(f"  [skip] no per-frame CSV for {dataset}/{model}")
        return False
    df = pd.read_csv(csv)
    for col in ("MATD_vis", "MATD_ir", "MATD_fused"):
        if col not in df.columns:
            print(f"  [skip] {csv} missing {col}")
            return False
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 3.2))
    _draw(ax, df, title=f"{RC.MODEL_LABELS.get(model, model)} - {dataset}", show_legend=True)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("MATD")
    if own_fig:
        out = os.path.join(results_dir, "qualitative", f"MATD_{dataset}_{model}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        print(f"  wrote {out}")
    return True


def build_grid(results_dir, datasets, models):
    """Figure 5: rows = methods, columns = datasets. One compact, page-fitting PNG."""
    nrows, ncols = len(models), len(datasets)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 1.9 * nrows),
                             squeeze=False, sharex="col")
    any_drawn = False
    for r, model in enumerate(models):
        for c, dataset in enumerate(datasets):
            ax = axes[r][c]
            csv = _find_per_frame_csv(results_dir, dataset, model)
            if csv is None:
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"no data\n{dataset}/{model}", ha="center",
                        va="center", fontsize=7, transform=ax.transAxes)
                continue
            df = pd.read_csv(csv)
            if not all(col in df.columns for col in ("MATD_vis", "MATD_ir", "MATD_fused")):
                ax.set_axis_off(); continue
            _draw(ax, df,
                  title=dataset if r == 0 else None,
                  ylabel=RC.MODEL_LABELS.get(model, model) if c == 0 else None,
                  show_legend=(r == 0 and c == ncols - 1))
            ax.text(0.02, 0.94, f"({_letter(r * ncols + c)})", transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="top")
            if r == nrows - 1:
                ax.set_xlabel("Frame index", fontsize=8)
            any_drawn = True
    if not any_drawn:
        print("  [grid] no per-frame CSVs found for the requested datasets/models"); plt.close(fig); return
    fig.tight_layout()
    tag = "_".join(datasets)
    out = os.path.join(results_dir, "qualitative", f"MATD_grid_{tag}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight"); plt.close(fig)  # >=1600px wide (Cureus recommended)
    print(f"  wrote {out}  ({nrows} methods x {ncols} datasets)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true",
                    help="build the compact Figure-5 grid (rows=methods, cols=datasets)")
    ap.add_argument("--dataset", default=None, help="single-dataset mode")
    ap.add_argument("--datasets", nargs="*", default=None, help="grid mode: datasets = columns")
    ap.add_argument("--models", nargs="*", default=None, help="default: all in run_config.MODELS")
    a = ap.parse_args()
    results_dir = os.path.join(RC.OUTPUT_ROOT, "results")
    models = a.models or RC.MODELS

    if a.grid:
        datasets = a.datasets if a.datasets else ([a.dataset] if a.dataset else None)
        if not datasets:
            raise SystemExit("--grid needs --datasets (e.g. --datasets FLIR_ADAS CAMEL_Seq13)")
        build_grid(results_dir, datasets, models)
        return

    if not a.dataset:
        raise SystemExit("use --dataset <name>  (or --grid --datasets ...)")
    for m in models:
        plot_model(results_dir, a.dataset, m)

    # combined tall stacked figure (kept for the supplement)
    avail = [m for m in models if _find_per_frame_csv(results_dir, a.dataset, m)]
    if avail:
        fig, axes = plt.subplots(len(avail), 1, figsize=(9, 2.6 * len(avail)), squeeze=False)
        for ax, m in zip(axes[:, 0], avail):
            df = pd.read_csv(_find_per_frame_csv(results_dir, a.dataset, m))
            _draw(ax, df, title=f"{RC.MODEL_LABELS.get(m, m)} - {a.dataset}", show_legend=True)
            ax.set_xlabel("Frame index"); ax.set_ylabel("MATD")
        out = os.path.join(results_dir, "qualitative", f"MATD_{a.dataset}_all.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()