"""Diagnostic: where is log T lost? Linear ridge probes of log T / log P from different layers of the trained model
(8-d s, 128-d mean patch code, 4096-d flattened patch codes), plus hand-made frame-difference features."""
import json, sys, argparse
import numpy as np, torch
from data import clip, labels, load_split, to_device, all_windows
from model import HJEPA, patchify

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--runs", default="0,1,2"); ap.add_argument("--out", default="results/diag_T.json")
A = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
S = {s: to_device(load_split(A.data, s), dev) for s in ("train", "val", "test")}


@torch.no_grad()
def feats(m, split, stride):
    D = S[split]; e, t = all_windows(D["frames"].shape[0], stride); e, t = e.to(dev), t.to(dev)
    out = {k: [] for k in ("s", "mean", "flat", "diff")}; Y = []
    for i in range(0, len(e), 1024):
        ee, tt = e[i:i+1024], t[i:i+1024]; c = clip(D["frames"], ee, tt)
        codes = m.E0(patchify(c))
        out["s"].append(m.G(codes).float().cpu()); out["mean"].append(codes.mean(1).float().cpu())
        out["flat"].append(codes.reshape(len(ee), -1).float().cpu())
        # hand features: per-patch |frame difference| sum, and per-patch count sum
        d = (c[:, 1:] - c[:, :-1]).abs().sum(1).reshape(len(ee), 4, 8, 8, 8).sum((2, 4)).reshape(len(ee), -1)
        n = c[:, -1].reshape(len(ee), 4, 8, 8, 8).sum((2, 4)).reshape(len(ee), -1)
        out["diff"].append(torch.cat([d, n], 1).float().cpu())
        Y.append(labels(D, ee, tt).cpu())
    return {k: torch.cat(v) for k, v in out.items()}, torch.cat(Y)


def ridge(Xtr, Ytr, Xva, Yva, Xte, Yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    z = lambda X: ((X - mu) / sd).double()
    Ztr, Zva, Zte = z(Xtr), z(Xva), z(Xte); ym = Ytr.mean(0).double()
    G = Ztr.T @ Ztr; b = Ztr.T @ (Ytr.double() - ym); ev, V = torch.linalg.eigh(G); Vb = V.T @ b; n = len(Ztr)
    r2 = lambda p, Y: (1 - ((Y.double() - p) ** 2).sum(0) / ((Y.double() - Y.double().mean(0)) ** 2).sum(0))
    best = None
    for a in 10.0 ** np.arange(-6, 3.5, 0.5):
        w = V @ (Vb / (ev + a * n)[:, None]); sc = r2(Zva @ w + ym, Yva)[:2].mean().item()
        if best is None or sc > best[0]: best = (sc, w)
    return r2(Zte @ best[1] + ym, Yte)[:2].tolist()


res = {}
for seed in A.runs.split(","):
    sd = torch.load(f"runs/seed{seed}/final.pt", map_location=dev)["model"]
    m = HJEPA(aux="aux.weight" in sd, aux_s="aux_s.weight" in sd, nf=sd["E0.embed.weight"].shape[1] // 64).to(dev).eval(); m.load_state_dict(sd)
    F = {sp: feats(m, sp, 3 if sp == "train" else 2) for sp in S}
    res[seed] = {}
    for k in ("s", "mean", "flat", "diff"):
        r = ridge(F["train"][0][k], F["train"][1], F["val"][0][k], F["val"][1], F["test"][0][k], F["test"][1])
        res[seed][k] = dict(logT=round(r[0], 3), logP=round(r[1], 3)); print(seed, k, res[seed][k], flush=True)
json.dump(res, open(A.out, "w"), indent=1)
