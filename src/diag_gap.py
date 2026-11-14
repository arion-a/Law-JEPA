"""How much does a 2-frame clip reveal about T as a function of the time between the frames?
Fresh gas, no events. Frames rendered exactly as the dataset does (32x64 counts). Features are physics-motivated,
no learning: fraction of dots that changed pixel, and mean nearest-neighbour pixel distance between the two frames."""
import sys, json; sys.path.insert(0, '.')
import numpy as np
from sim.gas import Gas, advance, render, PPU
rng = np.random.default_rng(0)
SP = [5, 10, 20, 40, 80, 160, 320]
rows = {s: [] for s in SP}; Y = []
for i in range(900):
    N = int(rng.integers(128, 513)); W = rng.uniform(1, 2); T = rng.uniform(0.3, 2.0)
    g = Gas(N, W, T, rng)
    advance(g.x, g.y, g.vx, g.vy, g.W, 0.0, g.wt, 400)          # let it mix
    Y.append((np.log(g.T()), np.log(N), np.log(W)))
    for s in SP:
        a = render(g.x, g.y, g.W).astype(float); a[a == 4] = 0
        h = Gas(N, W, T, rng); h.x, h.y, h.vx, h.vy, h.W, h.vp, h.wt = g.x.copy(), g.y.copy(), g.vx.copy(), g.vy.copy(), g.W, 0.0, g.wt
        advance(h.x, h.y, h.vx, h.vy, h.W, 0.0, h.wt, s)
        b = render(h.x, h.y, h.W).astype(float); b[b == 4] = 0
        frac = np.abs(a - b).sum() / (a.sum() + b.sum() + 1)
        pa = np.argwhere(a > 0); pb = np.argwhere(b > 0)
        d = np.sqrt(((pb[:, None, :] - pa[None, :, :]) ** 2).sum(-1)).min(1)
        rows[s].append((np.log(frac + 1e-3), np.log(d.mean() + 0.05), np.log((d == 0).mean() + 0.02)))
Y = np.array(Y); ntr = 600
def r2(X, y):
    A = np.c_[X, np.ones(len(X))]; w = np.linalg.lstsq(A[:ntr], y[:ntr], rcond=None)[0]; p = A[ntr:] @ w
    return 1 - ((y[ntr:] - p) ** 2).sum() / ((y[ntr:] - y[ntr:].mean()) ** 2).sum()
out = {}
for s in SP:
    X = np.array(rows[s]); base = Y[:, 1:]            # log N, log W are known from the same picture
    out[s] = dict(frame_dt=s * 0.0005, logT_motion_only=round(r2(X, Y[:, 0]), 3), logT_motion_plus_NW=round(r2(np.c_[X, base], Y[:, 0]), 3))
    print(s, out[s], flush=True)
json.dump(out, open('results/diag_gap.json', 'w'), indent=1)
