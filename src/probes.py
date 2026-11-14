"""Linear (ridge) probes: s(t) -> log T, log P (plus log N, log A as sanity checks).
  python probes.py --mode controls                 raw-pixel probe + untrained-model probes (seeds 0,1,2) -> results/step3_controls.json
  python probes.py --mode final --runs runs        full model probes for seeds with runs/seedN/final.pt -> results/step5_probes.json
Ridge penalty chosen on VALIDATION windows only; R2 reported on TEST windows."""
import argparse
import json
import os

import numpy as np
import torch

from data import clip, labels, load_split, to_device, all_windows
from model import HJEPA

NAMES = ["logT", "logP", "logN", "logA"]
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["controls", "final"], required=True)
ap.add_argument("--data", default="data")
ap.add_argument("--runs", default="runs")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--stride_pix", type=int, default=5)
ap.add_argument("--stride_model", type=int, default=2)
ap.add_argument("--max_eps", type=int, default=None)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--tag", default="")
ap.add_argument("--runtag", default="", help="run dir suffix, e.g. _v2 -> runs/seed0_v2")
a = ap.parse_args()
dev = a.device
os.makedirs("results", exist_ok=True)

splits = {s: to_device(load_split(a.data, s, a.max_eps), dev) for s in ("train", "val", "test")}


@torch.no_grad()
def features(kind, model, split, stride):
    D = splits[split]
    e, t = all_windows(D["frames"].shape[0], stride)
    e, t = e.to(dev), t.to(dev)
    X, Y = [], []
    for i in range(0, len(e), 1024):
        ee, tt = e[i:i + 1024], t[i:i + 1024]
        c = clip(D["frames"], ee, tt)
        X.append(c.reshape(len(ee), -1).cpu() if kind == "pixels" else model.state(c).float().cpu())
        Y.append(labels(D, ee, tt).cpu())
    return torch.cat(X), torch.cat(Y)


def r2(pred, Y):
    return (1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)).tolist()


def ridge(tr, va, te):
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = tr, va, te
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    ym = Ytr.mean(0).double()
    d = Xtr.shape[1]
    G = torch.zeros(d, d, dtype=torch.float64)
    b = torch.zeros(d, Ytr.shape[1], dtype=torch.float64)
    for i in range(0, len(Xtr), 8192):
        z = ((Xtr[i:i + 8192] - mu) / sd).double()
        G += z.T @ z
        b += z.T @ (Ytr[i:i + 8192].double() - ym)
    ev, V = torch.linalg.eigh(G)
    Vb = V.T @ b
    n = len(Xtr)
    Zva, Zte = ((Xva - mu) / sd).double(), ((Xte - mu) / sd).double()
    best = None
    for alpha in 10.0 ** np.arange(-6, 3.5, 0.5):
        w = V @ (Vb / (ev + alpha * n)[:, None])
        sc = np.mean(r2(Zva @ w + ym, Yva.double())[:2])        # choose on val R2 of logT, logP
        if best is None or sc > best[0]:
            best = (sc, alpha, w)
    _, alpha, w = best
    pred = Zte @ w + ym
    out = dict(zip(NAMES, r2(pred, Yte.double())))
    out["alpha"] = float(alpha)
    return out, pred.float().numpy(), Yte.numpy()


def probe(kind, model, stride):
    return ridge(*[features(kind, model, s, stride if s == "train" else 1) for s in ("train", "val", "test")])


def fresh(seed):
    torch.manual_seed(seed)
    fr = splits['train']['frames']
    return HJEPA(nf=fr.shape[2] if fr.dim() == 5 else 2).to(dev).eval()


seeds = [int(s) for s in a.seeds.split(",")]
if a.mode == "controls":
    pix, _, _ = probe("pixels", None, a.stride_pix)
    res = dict(pixels=pix, untrained={})
    for s in seeds:
        res["untrained"][str(s)] = probe("model", fresh(s), a.stride_model)[0]
    worst = max([pix[k] for k in ("logT", "logP")] + [v[k] for v in res["untrained"].values() for k in ("logT", "logP")])
    res["max_control_R2_logT_logP"] = worst
    res["pass"] = bool(worst < 0.90)
    json.dump(res, open(f"results/step3_controls{a.tag}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
else:
    ctrl = json.load(open(f"results/step3_controls{a.tag}.json"))
    res = {}
    for s in seeds:
        ck = os.path.join(a.runs, f"seed{s}{a.runtag}", "final.pt")
        if not os.path.exists(ck):
            continue
        sd = torch.load(ck, map_location=dev)["model"]
        m = HJEPA(aux="aux.weight" in sd, aux_s="aux_s.weight" in sd, nf=sd["E0.embed.weight"].shape[1] // 64).to(dev).eval()
        m.load_state_dict(sd)
        full, pred, Y = probe("model", m, a.stride_model)
        np.savez(f"results/preds_seed{s}{a.tag}.npz", pred=pred, true=Y)
        res[str(s)] = dict(full=full, pixels=ctrl["pixels"], untrained=ctrl["untrained"][str(s)])
    json.dump(res, open(f"results/step5_probes{a.tag}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
