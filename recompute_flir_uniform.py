"""
recompute_flir_uniform.py  (NEW - fixes the inconsistent FLIR frame counts)
---------------------------------------------------------------------------
Your FLIR runs used different per-model frame caps (200/300/500/3749), so the two
FCDFusion implementations (which are mathematically identical) ended up with
slightly different quality/temporal numbers ON FLIR ONLY, because they were
averaged over different frames. This re-aggregates EVERY FLIR model over the SAME
first-N frames, straight from the per-frame CSVs already on disk - NO re-running.

It also prints the proposed method's FULL-sequence FPS so you can still quote the
"full 3749-frame" headline in the text.

Run:  python recompute_flir_uniform.py            # N = smallest available (=200)
      python recompute_flir_uniform.py --n 200
"""
import os, glob, argparse
import pandas as pd
import run_config as RC
from results_manager import ResultsManager

DS = "FLIR_ADAS"


def perframe_csv(results_dir, model_key):
    hits = glob.glob(os.path.join(results_dir, DS, model_key, "*per_frame.csv"))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="frames to keep (default: smallest available)")
    a = ap.parse_args()
    results_dir = os.path.join(RC.OUTPUT_ROOT, "results")

    # discover available per-frame lengths
    lengths = {}
    for key in RC.MODELS:
        c = perframe_csv(results_dir, key)
        if c:
            lengths[key] = sum(1 for _ in open(c)) - 1
    if not lengths:
        raise SystemExit("No FLIR per-frame CSVs found - run FLIR first.")
    N = a.n or min(lengths.values())
    print(f"FLIR per-frame lengths: {lengths}\nUsing common N = {N} frames for every model.\n")

    rm = ResultsManager(results_dir)               # loads ALL existing rows (CAMEL stays untouched)
    for key in RC.MODELS:
        c = perframe_csv(results_dir, key)
        if not c:
            print(f"[skip] no per-frame CSV for {key}"); continue
        label = RC.MODEL_LABELS.get(key, key)
        df = pd.read_csv(c).head(N)
        # full-sequence FPS headline (before trimming) for the proposed method
        if key == "fcdfusion_vectorized":
            full = pd.read_csv(c)
            if "fps" in full:
                print(f"[headline] {label} full-sequence FPS over {len(full)} frames = {full['fps'].mean():.2f}")
        agg = {"image_count": N, "processed_successfully": N,
               "processed_pairs_details": df.to_dict("records")}
        rm.add_run(DS, label, agg)                 # overwrites the FLIR row with the uniform-N recompute
        print(f"  recomputed {label} over {N} frames")
    rm.finalize()
    print("\nDone. FLIR rows now use a uniform N; CAMEL untouched. Workbook rebuilt.")


if __name__ == "__main__":
    main()
