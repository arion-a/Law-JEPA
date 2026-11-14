"""Step 1 gate: simulator validation against the three prototype cases in the design doc.
Writes results/step1_simulator.json."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.gas import make_episode, FRAME_DT, PPU  # noqa: E402

CASES = [("cold_large", 128, 2.0, 0.5), ("middle", 256, 1.5, 1.0), ("hot_small", 512, 1.0, 2.0)]
out = {"cases": []}
make_episode(0, nframes=2, n_events=0)  # compile
for name, N, W, T in CASES:
    t0 = time.time()
    ep = make_episode(1, nframes=100, N=N, W0=W, T0=T, n_events=0)
    dt = time.time() - t0
    ratio = float(np.mean(ep["P_meas"][10:] * ep["A"][10:] / (N * ep["T"][10:])))
    expected = float(np.mean(1 + 2 * ep["phi"]))
    speed = float(np.sqrt(2 * T))  # approx rms speed
    out["cases"].append(dict(case=name, N=N, W=W, T=T, PA_over_NT=ratio, expected=expected,
                             rel_err=abs(ratio - expected) / expected, px_per_frame=speed * FRAME_DT * PPU,
                             seconds_per_100_frames=dt,
                             energy_drift=float(abs(ep["T"][-1] - ep["T"][0]) / ep["T"][0])))
# energy drift on a full 300-frame no-intervention episode
ep = make_episode(2, nframes=300, n_events=0)
out["energy_drift_300f"] = float(abs(ep["T"][-1] - ep["T"][0]) / ep["T"][0])
out["pass"] = bool(all(c["rel_err"] < 0.03 for c in out["cases"]) and out["energy_drift_300f"] < 0.02)
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/step1_simulator.json", "w"), indent=1)
print(json.dumps(out, indent=1))
