# Run plan (FROZEN): Do T and P emerge in a hierarchical JEPA on a 3D gas?

Full design and reasoning: `DESIGN_3D.md`. This file is the pre-registration: **do not edit after step 0 is ledgered.**
Any change after this point must be a new dated section at the bottom, written before the run it affects.

**Core question.** Train the same 2-level JEPA (HJEPA) on 2D pictures (projection along z) of a simulated 3D hard-sphere gas,
no labels. Does its 8-number top-level state s(t) linearly encode log T and log P?

**Decision rule (fixed now).** Supported iff, in all 3 seeds, full-model test R² >= 0.90 for log T *and* log P, and both
controls (raw-pixel ridge probe, untrained-model probe) are >= 0.30 lower. Full < 0.90 -> not supported in this setup.
Full >= 0.90 but a control within 0.30 -> not supported (readable without learning). Log N, log V are sanity checks only.

**Fixed ranges.** N in [128, 512]; width W in [1, 2] (box W x 1 x 1, V = W); T in [0.3, 2.0] always; sphere radius R = 0.03;
mass = 1, k_B = 1; 300 frames per episode; 1000/100/100 train/val/test episodes; disjoint deterministic seeds.

**Fixed model and training recipe (the 2D v3 recipe, no changes).** 4-frame clips, frames 20 substeps apart, DT = 0.0005,
80 substeps per frame; HJEPA unchanged; patch motion loss weight 1 and head-on-s motion loss weight 1; 3 seeds (0, 1, 2) x
60,000 steps; ridge penalty chosen on validation only; R² on test only. Run tag `_3d`.

**Labels.** T = (1/3) mean(vx²+vy²+vz²). P_label = N T / V x (1 + 4φ + 10φ²), φ = N (4/3) π R³ / V.
Probe targets: log T, log P (decision); log N, log V (sanity).

| # | Step | Gate (pass condition) |
|---|------|-----------------------|
| 0 | Freeze this plan, ledger it | file ledgered before any 3D data exists |
| 1 | `sim/gas3d.py`, `tests/check_sim3d.py` | P V/(N T) within 3% of 1+4φ+10φ² on 3 cases; energy drift < 2%; component variances within 5% of each other after warm-up; T W^(2/3) within 5% under slow compression |
| 2 | 3D dataset (`--dim 3 --n_pre 3 --pre_gap 20`) | counts exact; T in [0.3, 2.0]; no NaN; mean free path 0.3 to 0.6 box lengths |
| 3 | Hand-feature ceiling (`diag_k.py`) + controls | neither control >= 0.90 on log T, log P; hand-feature ceiling >= 0.90 for log T, else adjust only R, spacing or clip length, ledger why, regenerate **before** training |
| 4 | Train arm A, 3 seeds | no collapse (spread of s >= 0.05); L0, L1 decreasing; no NaN |
| 5 | Probes + `supervisor.py decide _3d` | rule applied to all 3 seeds |
| 6 | Optional exploratory arm B (no motion losses) | labelled exploratory, outside the decision rule |
| 7 | Write-up | figure + verdict in `results/`, ledger export, push |

Every step ends in a ledger entry. A failed gate -> stop and ask the user. A failed arm A is reported as failed.
2D code, results and tags stay untouched.
