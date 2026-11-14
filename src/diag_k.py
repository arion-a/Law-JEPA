"""Does a longer clip (K frames, 20 substeps apart) reveal more about T? Same physics features as diag_gap.py, averaged over the K-1 adjacent pairs."""
import sys, json; sys.path.insert(0, '.')
import numpy as np
from sim.gas import Gas, advance, render
rng = np.random.default_rng(1); GAP = int(sys.argv[1]) if len(sys.argv) > 1 else 20; KM = 5
F = {k: [] for k in range(2, KM + 1)}; Y = []
def pair(a, b):
    frac = np.abs(a - b).sum() / (a.sum() + b.sum() + 1)
    pa, pb = np.argwhere(a > 0), np.argwhere(b > 0)
    d = np.sqrt(((pb[:, None, :] - pa[None, :, :]) ** 2).sum(-1)).min(1)
    return np.array([np.log(frac + 1e-3), np.log(d.mean() + 0.05), np.log((d == 0).mean() + 0.02)])
for i in range(900):
    N = int(rng.integers(128, 513)); W = rng.uniform(1, 2); T = rng.uniform(0.3, 2.0)
    g = Gas(N, W, T, rng); advance(g.x, g.y, g.vx, g.vy, g.W, 0.0, g.wt, 400)
    Y.append((np.log(g.T()), np.log(N), np.log(W)))
    fr = []
    for j in range(KM):
        im = render(g.x, g.y, g.W).astype(float); im[im == 4] = 0; fr.append(im)
        advance(g.x, g.y, g.vx, g.vy, g.W, 0.0, g.wt, GAP)
    pr = [pair(fr[j], fr[j + 1]) for j in range(KM - 1)]
    for k in range(2, KM + 1):
        F[k].append(np.mean(pr[:k - 1], 0))
Y = np.array(Y); ntr = 600
def r2(X, y):
    A = np.c_[X, np.ones(len(X))]; w = np.linalg.lstsq(A[:ntr], y[:ntr], rcond=None)[0]; p = A[ntr:] @ w
    return round(1 - ((y[ntr:] - p) ** 2).sum() / ((y[ntr:] - y[ntr:].mean()) ** 2).sum(), 3)
print('gap', GAP, {k: r2(np.c_[np.array(F[k]), Y[:, 1:]], Y[:, 0]) for k in F})
