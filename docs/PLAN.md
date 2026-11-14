# Run plan: Do T and P emerge in a hierarchical JEPA? (design doc: Experiment_Design_Gas_Laws_HJEPA_2.docx)

**Core question.** Train a 2-level JEPA on video of a simulated 2D gas with no labels. Does its 8-number top-level
state s(t) linearly encode log T and log P?
**Decision rule (fixed in advance).** Supported iff, in all 3 seeds, full-model test R² ≥ 0.90 for log T *and* log P,
and both controls (raw-pixel probe, untrained-model probe) are ≥ 0.30 lower. Full < 0.90 → not supported in this setup.
Full ≥ 0.90 but a control within 0.30 → not supported (readable without learning). Log N, log A are sanity checks only.
Later experiments (interventions, horizon, transportability, 3D) are **out of scope**.

Every step below ends in a **ledger entry** (`ledger.py`) and a **supervisor gate** (`supervisor.py gate N`).
Failure of any gate → stop and ask the user. Do not change the decision rule after seeing results.

| # | Step | Deliverable | Gate (pass condition) | Compute | Est. time |
|---|------|-------------|-----------------------|---------|-----------|
| 0 | Environment | numba/numpy/scipy/sklearn; PyTorch+CUDA on the training machine | imports work | – | 30 min (torch fix) |
| 1 | Simulator | `sim/gas.py`, `tests/check_sim.py` | P·A/(N·T) within 3% of 1+2φ on 3 cases; energy drift < 2% | CPU | **DONE** (<1 s/100 frames) |
| 2 | Dataset | `gen_dataset.py` → `data/{train,val,test}` 1000/100/100 episodes × 300 frames | counts exact; N∈[128,512], W∈[1,2], T∈[0.3,2.0] always; no NaN | CPU 4 cores | **DONE** (253 s) |
| 3 | Controls | `probes.py`: ridge on raw 4096 px; untrained-model probe | neither control reaches 0.90 for log T/log P (else the test is moot, escalate) | CPU (pixel ridge); untrained model any | 20–40 min |
| 4 | Full model, seed 0 | `model.py`, `train.py` (HJEPA, 60k steps, checkpoints every 2k) | no collapse (spread of s ≥ 0.05); L1 decreasing; L0 decreasing | **GPU** | 1.5–3 h |
| 5 | Test | seeds 1,2 trained; probes for 3 seeds + controls; `supervisor.py decide` | rule applied to all 3 | GPU for training, CPU probes | 3–6 h (2 seeds) + 20 min |
| 6 | Write-up | one figure (pred vs true log T, log P on test), results table, ledger export | figure + verdict in `results/` | CPU | 1 h |

## Implementation notes per step
- **3 (controls):** windows = (episode, t∈[1,249]); probe features for pixels = clip c(t) (two frames, 4096 values; piston→1, counts/3). Subsample windows (e.g. every 5th frame) for the pixel ridge if memory is tight. Choose ridge penalty on val, report test R². Targets: log T(t), log P_label(t).
- **4–5 (model):** as in design §6–7: patch embed (8×8, 32 patches, width 128), E0 4 layers, P0 2 layers, EMA target, pooling G → s∈R⁸, P1 MLP (2×256) predicts s(t+50)−s(t) from s(t) and action a=(ΔW/W, Σlog f over (t,t+50]). Loss L0+L1+VICReg(patch codes)+spread floor on s. AdamW wd .05, lr 3e-4 warm-up 2k → 1e-5, batch 128, 60k steps, EMA .996→1. Collapse alarm: std-sum of s < 0.05.
- **Seeds:** 3 training seeds (0,1,2); the dataset is fixed (generated once).
- **Probes:** ridge s(t) → (log T, log P_label), penalty by val, R² on test windows, per seed; also log N, log A.

## Time and hardware estimate
- Simulator + dataset: already measured, ~4 min total on 4 CPU cores (the design doc's "about a day" assumed slower code).
- **GPU: required for training**, not for simulation or probes. ~1.9M-weight transformer, 60k steps × batch 128, ~5 forward passes of 32 tokens per step. On a Colab T4 expect ≈ 0.1–0.2 s/step → **2–3.5 h per seed, ~6–10 h for 3 seeds** (A100/4090-class: ~1–2 h per seed). On this laptop (4 CPU cores, no NVIDIA GPU, a broken PyTorch install) it would be ~1–2 s/step → 17–35 h per seed, not practical.
- Free Colab sessions can disconnect (~12 h limit): train with checkpoints every 2k steps and resume.
- **Total wall-clock: about 1 working day to 2 days**, dominated by the 3 training runs. CPU-side work excluding training: ~2 h.
