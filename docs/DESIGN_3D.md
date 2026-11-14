# Design: the same experiment on a 3D gas

Status: **proposal, nothing built yet.** Write and freeze this (plus the decision rule) before generating any data.

## 1. Question and rule (unchanged)

Does an 8-number JEPA state encode log T and log P of a **3D** gas?
Rule: in all 3 seeds, test R² >= 0.90 for log T and log P, and both controls >= 0.30 lower. Fixed now, not after results.

## 2. Principle: change as little as possible

The network sees 2D pictures either way, so the **model, training code and probes stay the same**. Only the simulator and
its labels change. We also bake in the lessons of the 2D runs so we do not need repair rounds (v1 -> v2 -> v3):
4-frame clips, 20-substep spacing and both motion losses from the start.

## 3. What changes

| Item | 2D (done) | 3D (new) |
|---|---|---|
| Particles | disks, R = 0.003 | **spheres, R = 0.03** (see risk 1) |
| Box | width W x 1; piston on the right wall | width W x 1 x 1; piston on the +x wall |
| Volume | A = W | **V = W** (cross-section 1 x 1) |
| Velocities | vx, vy | vx, vy, **vz** |
| Temperature | T = 0.5 mean(vx² + vy²) | **T = (1/3) mean(vx² + vy² + vz²)** |
| Pressure label | N T / A x (1 + 2φ) | **N T / V x (1 + 4φ + 10φ²)**, φ = N (4/3) π R³ / V |
| Measured pressure | impulse / (perimeter x time) | impulse / (**surface** 4W + 2, x time) |
| Piston compression check | T x W stays constant (2D adiabatic) | **T x W^(2/3)** stays constant (3D adiabatic) |
| Picture | disks drawn in the xy plane | **projection along z**: each pixel counts spheres in its column (0 to 3 cap), piston bright |
| Labels used by probes | log T, log P, log N, log A | log T, log P, log N, **log V** |

New idea in 3D: the picture does not show the z direction, so vz is invisible and depth overlaps are possible. T must be
inferred from vx, vy only. This works if the gas is **isotropic** (same speed spread in all directions), which collisions
produce. Right after a piston move it is briefly not isotropic; this is a real new difficulty, so we measure it (step 1).

## 4. Step-by-step implementation plan

Each step ends with a ledger entry (`python ledger.py add ...`) and a pass/fail gate. If a gate fails: stop and ask.

**Step 0. Freeze the plan.** Create `PLAN_3D.md` with the rule above and the ranges (N 128 to 512, W 1 to 2, T 0.3 to 2.0).
Ledger it. *Gate: file exists and is ledgered before any data.*

**Step 1. 3D simulator** `src/sim/gas3d.py`, copied from `gas.py`.
- Add z and vz arrays; bounce off z = 0 and z = 1 like y.
- Cell list in 3 dimensions (cells of 0.04+; check the 13 forward neighbour cells) and sphere-sphere collisions: same velocity exchange along the line between centres.
- Keep DT = 0.0005 (speed about 3 x 0.0005 = 0.0015 per step, far below R = 0.03, so no skipping through each other).
- Impulse sums over 6 walls; T, P labels as in the table; piston event check uses exponent 2/3.
- `render()` projects x, y only. Write `tests/check_sim3d.py`.
*Gate:* P V / (N T) within 3% of 1 + 4φ + 10φ² on 3 cases; energy drift < 2%; after warm-up the three velocity components have variances within 5% of each other; slow compression keeps T W^(2/3) within 5%.

**Step 2. Dataset.** Add `--dim 3` to `gen_dataset.py`; run with `--n_pre 3 --pre_gap 20` (4-frame clips) for 1000/100/100 episodes x 300 frames.
Disjoint seeds as before. Look at one episode as an image strip to confirm the picture is sensible.
*Gate:* exact counts; T in [0.3, 2.0] always; no NaN; mean free path check (see risk 1).

**Step 3. Feasibility and controls (before any training).**
1. Hand-feature ceiling: run `diag_k.py` on the 3D data. This asks "could ridge on simple frame-difference features reach 0.90 for log T at 4 frames, gap 20?"
2. Controls as before: pixel ridge on the 8192 values, untrained network.
*Gate:* neither control >= 0.90 on log T and log P. If the hand-feature ceiling is below 0.90 for log T, adjust **only** the allowed knobs (R, spacing, clip length), re-run step 2 and ledger the reason, before training. This is the check that would have saved the two failed 2D rounds.

**Step 4. Train (arm A: the v3 recipe).** Same `train.py`: 3 seeds, 60,000 steps, `--aux_w 1 --aux_s_w 1`, new tag `_3d`.
Run all three at once on one GPU (about 0.058 s/step, roughly 1 h).
*Gate:* no collapse (spread of s >= 0.05), L1 and L0 decreasing, no NaN.

**Step 5. Probes and decision.** `probes.py --mode final` with the 3D tag, then `supervisor.py decide _3d`. Report per seed full / pixels / untrained R² for log T, log P, log N, log V.

**Step 6 (optional, exploratory, label it so). Arm B: no motion losses.** Same recipe with the two motion losses switched off, 3 seeds. In 2D this arm failed, so the comparison tells whether the success depends on our hand-built targets. Not part of the decision rule.

**Step 7. Write-up.** Results table, figure (predicted vs true log T and log P on test), ledger export, commit and push.

## 5. Cost and time (4090 at about $0.42/h, credit $5.43)

Simulation: minutes on the GPU box's CPUs. Arm A: about 1 to 1.5 h (about $0.60). Arm B: same. Probes: minutes.
Total about 3 to 4 h wall clock, about $1.5 to 2.

## 6. Risks and how the plan handles them

1. **Radius.** Keeping R = 0.003 would make the 3D gas almost collision-free (about 12 box lengths between collisions), so it would not thermalize. R = 0.03 gives φ about 0.02 to 0.05 and a mean free path of about 0.3 to 0.6 box lengths, similar to the 2D run. Verified in step 2.
2. **Hidden depth.** Spheres overlap in the projection and vz is invisible. Mitigation: isotropy check (step 1); fallback, add a second view (projection along y) as extra input channels.
3. **Re-tuning on the test set.** Do not repeat the 2D pattern. Any design change after step 3 must be written down before the training run, and a failed arm A is reported as failed, not repaired silently.
4. **Same hand-built motion losses.** Result must be read with the caveat in the 2D report; arm B shows how much depends on them.
5. **Repo.** Keep the 2D code and results untouched. 3D code goes in new files (`gas3d.py`, `check_sim3d.py`) and new tags (`_3d`), so the 2D experiment stays reproducible.
