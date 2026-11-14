# v3 result (4-frame clips): hypothesis SUPPORTED

**Question:** does a JEPA's 8-number top-level state linearly encode log T and log P of a simulated gas?
**Rule (pre-registered):** probe R2 >= 0.90 on logT and logP, controls >= 0.30 lower, in all 3 seeds.
**Verdict:** SUPPORTED (ledger entry 21).

## Setup
- Data `data3`: clips of 4 frames, 20 substeps apart. Patch motion loss plus head-on-s motion loss (w=1 each).
- 3 seeds x 60k steps, ~0.058 s/step. No ALARM or NaN in any seed.
- Why v3: v2 (2-frame clips) reached logT 0.78-0.80 and logP 0.90 (not supported). `diag_k.py` predicted logT ~0.96 from 4 frames at gap 20 (0.85 from 2).

## Linear-probe R2
| Seed | Probe | logT | logP |
|---|---|---|---|
| 0 | full | 0.929 | 0.966 |
| 0 | untrained | -0.005 | 0.052 |
| 1 | full | 0.943 | 0.972 |
| 1 | untrained | -0.008 | 0.072 |
| 2 | full | 0.944 | 0.973 |
| 2 | untrained | 0.001 | 0.094 |
| all | pixels (8192 features) | -0.021 | 0.483 |

The pixel control does not depend on the seed. Its highest logT/logP score is 0.483 (logP), a gap of >= 0.48 to the full model, so the 0.30 margin holds.
Other targets, full model: logN 0.998-0.999, logA 0.988-0.994.

## Caveats
- logT in seed 0 (0.929) is the closest to the 0.90 threshold.
- Pixels and untrained controls are weak on T, but pixels reach 0.91-0.95 on logN and logA, so those are easy to read from raw frames.
- Supported only for this simulator and the 4-frame, gap-20 input; v2 shows T is not recoverable from 2 frames.

## Artifacts
`results/step5_probes_v3.json`, `results/step3_controls_v3.json`, `ledger/ledger.jsonl` (entries 20-22).
