# Ideal Gas-law HJEPA: do T and P emerge in a JEPA's top-level state?

Hierarchical JEPA trained on a simulated 2D gas; a linear probe asks whether the 8-number top-level state encodes log T and log P.
**v3 result (4-frame clips, 3 seeds): SUPPORTED** — see `docs/RESULTS_v3.md`.

## Layout
- `src/` code: `sim/` simulator, `gen_dataset.py`, `data.py`, `model.py`, `train.py`, `probes.py`, `supervisor.py`, `ledger.py`, diagnostics (`diag_*.py`, `mech.py`), `tests/`, `ledger/ledger.jsonl` (hash-chained log, check with `python ledger.py verify`).
- `scripts/run_v3.sh` exact v3 pipeline (gen data, train 3 seeds, controls). Paths assume it is run from the code directory with data in `data3/`.
- `docs/` PLAN, SUPERVISOR rules, results write-up.
- `results/` probe and control JSONs (v1, v2, v3).
- `logs/` training, probe, data-generation logs.
- `checkpoints/runs/*/final.pt` trained weights (~118 MB; use Git LFS or a release, not plain git).
- `env/` `requirements.txt` (pip freeze), `instance_info.txt` (Python 3.12.15, RTX 4090).
- Datasets (`data`, `data20`, `data3`, ~0.5 GB) are not stored: regenerate with `gen_dataset.py` (see `scripts/run_v3.sh`).

## Reproduce v3
```bash
pip install -r env/requirements.txt
cd src && bash ../scripts/run_v3.sh
python probes.py --mode final --data data3 --runs runs --runtag _v3 --tag _v3
python supervisor.py decide _v3
```

## Extending to a 3D gas
Kept as the 2D reference: `src/sim/gas.py` (simulator), `src/gen_dataset.py` (clip generation, frame gap and count), `src/model.py` (encoder/hierarchy), `src/probes.py` (targets logT, logP, logN, logA; logA is a 2D area, so a 3D version needs logV).
Pre-registered rule (R2 >= 0.90 on logT/logP, controls >= 0.30 lower, 3 seeds) and the v3 recipe (4 frames, gap 20, patch + head-on-s motion loss, 60k steps) are the starting point; v2 showed 2-frame clips are not enough for T.
