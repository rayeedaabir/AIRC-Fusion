# Is the Temporal Instability Score an "AI hallucination"? — assessment & grounding

Short answer: **No, it is not a hallucination** — it computes exactly what it
claims and it sits inside a well-established family of temporal-consistency /
flicker metrics. **But** it is a simple heuristic, similar published metrics
exist, and it should be (a) grounded in that literature, (b) not over-claimed as
a major novel contribution, and (c) stated with its limitation. Doing this turns
it from "an AI-invented formula" into "a lightweight metric in a known family,"
which is what a reviewer needs to see.

## What it is
`MATD(V) = mean over frames of mean_xy |I_{t+1} - I_t|`  — a mean absolute
temporal difference. Then `Score = max(0, D_fused - max(D_vis, D_ir))`: how much
more the fused stream changes frame-to-frame than the more-active source. ~0 = no
flicker beyond real scene motion; large = flicker the algorithm injected.

## Where it is grounded (real, citable)
- **ITU-T Rec. P.910** defines *Temporal Information* (TI): the frame difference
  `I_t - I_{t-1}` summarised over space and time. MATD is the mean-absolute
  sibling of TI. (Cite P.910 for the temporal-difference foundation.)
- **Flicker / temporal-consistency** is a standard idea: high-frequency temporal
  change "not due to actual motion in the scene." Comparing a processed video's
  stability to the source's motion is exactly how temporal-consistency is judged.
- Established, more rigorous fusion-video metrics already exist — e.g. optical-flow
  based **flowD** and feature-level **feaCD** (recent IR-VIS video-fusion
  benchmarks), and warping-error temporal-consistency (Lai et al., ECCV 2018).
  These use optical flow to separate true motion from flicker.

## Honest limitation (state it in the paper)
Your score has no motion compensation and uses `max(D_vis, D_ir)` as the ceiling;
a faithful fusion carrying motion from *both* modalities can exceed that ceiling
without being unstable, so it can over-penalise. The flow-based metrics above
address this — at the cost of optical-flow computation.

## Recommended framing (improves acceptance AND honesty)
1. **Do not headline it as a major novel metric.** Your real contributions are
   the vectorisation + real-time CPU/edge deployment. Present the score as a
   *lightweight, optical-flow-free temporal-stability indicator* — which actually
   fits your CPU/edge theme (a heavy optical-flow metric would contradict it).
2. **Cite the lineage**: ITU-T P.910 (TI), a temporal-consistency/flicker
   reference, and the flow-based fusion metrics (flowD/feaCD) as the rigorous
   alternatives you deliberately simplify for real-time use.
3. **Report the limitation** in one sentence and show it behaves correctly
   (≈0 for deterministic methods, high for frame-recomputing/deep methods — your
   data already shows this).
4. Optionally rename to something descriptive and non-grandiose, e.g.
   *"Excess Temporal Difference (ETD)"* or *"relative temporal-stability indicator"*.

## To verify before these go in the manuscript
The exact author/title/venue for the flow-based metrics (flowD/feaCD) and the
ECCV temporal-consistency paper must be confirmed (Claude will verify during the
citation pass). ITU-T P.910 is a stable standard and safe to cite now.
