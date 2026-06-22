"""
make_figures.py  (NEW - generates the quantitative figures from the workbook)
-----------------------------------------------------------------------------
Reads results/master_metrics_long.csv (or any AIRC-Fusion_Results.xlsx) and draws:
  Figure 3  speed-fidelity landscape : FPS (log x) vs SSIM scatter, labelled (per dataset)
  Figure 4  multi-dimensional radar  : normalised Speed / Fidelity / Info / Stability
  Figure 8  processing-speed bars    : FPS (log y) grouped by model, per dataset
All saved to results/figures/*.png at 300 dpi.

Run:  python make_figures.py                      # uses run_config.OUTPUT_ROOT
      python make_figures.py --xlsx AIRC-Fusion_Results.xlsx --dataset FLIR_ADAS
"""
import os, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["Vectorized FCDFusion","FCDFusion (Pixel)","Model2013","Model2015","Model2024",
         "MDLatLRR","DenseFuse","ResNet-152","VGG-19"]


def _load(args):
    if args.xlsx:
        return pd.read_excel(args.xlsx, "All_runs")
    import run_config as RC
    return pd.read_csv(os.path.join(RC.OUTPUT_ROOT, "results", "master_metrics_long.csv"))


def _outdir(args):
    if args.xlsx:
        d = os.path.join(os.path.dirname(os.path.abspath(args.xlsx)), "figures")
    else:
        import run_config as RC
        d = os.path.join(RC.OUTPUT_ROOT, "results", "figures")
    os.makedirs(d, exist_ok=True); return d


def fig_speed_fidelity(df, ds, out):
    s = df[df.dataset == ds]
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in s.iterrows():
        ax.scatter(r.fps_mean, r.ssim_accuracy_mean, s=70,
                   color=("#2ca02c" if r.model == "Vectorized FCDFusion" else "#1f77b4"),
                   zorder=3)
        ax.annotate(r.model, (r.fps_mean, r.ssim_accuracy_mean), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("Processing speed (FPS, log scale)")
    ax.set_ylabel("SSIM"); ax.set_title(f"Speed-fidelity landscape - {ds}")
    ax.axvline(30, ls="--", c="grey", lw=1); ax.text(31, ax.get_ylim()[0], "30 FPS", fontsize=7, color="grey")
    ax.grid(alpha=0.3, which="both")
    p = os.path.join(out, f"FIG3_speed_fidelity_{ds}.png"); fig.tight_layout(); fig.savefig(p, dpi=300); plt.close(fig)
    print("  wrote", p)


def fig_radar(df, ds, out):
    s = df[df.dataset == ds].set_index("model")
    s = s.reindex([m for m in ORDER if m in s.index])
    # axes: Speed=log FPS, Fidelity=SSIM, Info=Entropy, Stability=1/(1+TIS); min-max normalise
    raw = pd.DataFrame({
        "Speed": np.log10(s.fps_mean.clip(lower=1e-3) + 1),
        "Fidelity": s.ssim_accuracy_mean,
        "Information": s.entropy_mean,
        "Stability": 1.0 / (1.0 + s.temporal_instability_score_mean),
    })
    norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    labels = list(norm.columns); ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist(); ang += ang[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for m in norm.index:
        vals = norm.loc[m].tolist(); vals += vals[:1]
        lw = 2.5 if m == "Vectorized FCDFusion" else 1.0
        ax.plot(ang, vals, lw=lw, label=m); ax.fill(ang, vals, alpha=0.05)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels); ax.set_yticklabels([])
    ax.set_title(f"Normalised multi-dimensional comparison - {ds}")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
    p = os.path.join(out, f"FIG4_radar_{ds}.png"); fig.tight_layout(); fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("  wrote", p)


def fig_speed_bars(df, out):
    piv = df.pivot_table(index="model", columns="dataset", values="fps_mean").reindex(
        [m for m in ORDER if m in df.model.unique()])
    fig, ax = plt.subplots(figsize=(11, 5))
    piv.plot(kind="bar", ax=ax, logy=True, width=0.8)
    ax.set_ylabel("FPS (log scale)"); ax.set_xlabel(""); ax.set_title("Processing speed across methods and datasets")
    ax.axhline(30, ls="--", c="red", lw=1); ax.text(0, 33, "30 FPS real-time", fontsize=8, color="red")
    ax.legend(fontsize=8, title="Dataset"); plt.xticks(rotation=30, ha="right", fontsize=8)
    p = os.path.join(out, "FIG8_speed_bars.png"); fig.tight_layout(); fig.savefig(p, dpi=300); plt.close(fig)
    print("  wrote", p)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--xlsx", default=None); ap.add_argument("--dataset", default="FLIR_ADAS")
    a = ap.parse_args(); df = _load(a); out = _outdir(a)
    fig_speed_bars(df, out)
    for ds in df.dataset.unique():
        fig_speed_fidelity(df, ds, out)
    fig_radar(df, a.dataset, out)


if __name__ == "__main__":
    main()
