"""Dataset loading shared by training and probing.
Frames are uint8 (E,300,32,64): 0-3 count, 4 piston. Model input = count/3, piston -> 1.0."""
import glob
import os

import numpy as np
import torch

H1 = 50          # level-1 prediction horizon (frames)
T_LO, T_HI = 1, 249


def load_split(root, split, max_eps=None):
    files = sorted(glob.glob(os.path.join(root, split, "ep_*.npz")))
    if max_eps:
        files = files[:max_eps]
    fr, W, T, N, P, logf = [], [], [], [], [], []
    has_pre = False
    for f in files:
        d = np.load(f)
        has_pre = 'pre' in d.files
        pre = d['pre'][:, None] if has_pre and d['pre'].ndim == 3 else (d['pre'] if has_pre else None)
        fr.append(np.concatenate([pre, d['frames'][:, None]], 1) if has_pre else d["frames"]); W.append(d["W"]); T.append(d["T"]); N.append(int(d["N"]))
        P.append(d["P_label"]); logf.append(d["logf"])
    return dict(frames=np.stack(fr), W=np.stack(W), T=np.stack(T), N=np.array(N, np.float32),
                P=np.stack(P), logf=np.stack(logf))


def to_device(d, device):
    return {k: torch.from_numpy(v).to(device) for k, v in d.items()}


def frames_to_float(u8):
    x = u8.float() / 3.0
    return torch.where(u8 == 4, torch.ones_like(x), x.clamp(max=1.0))


def clip(frames, e, t):
    """clip c(t) -> (B,2,32,64) float. Old data: frames t-1 and t. Data with pre-frames (5-d): [pre..., frame_t] (K frames)."""
    if frames.dim() == 5:
        return frames_to_float(frames[e, t])
    idx = torch.stack([t - 1, t], 1)                       # (B,2)
    return frames_to_float(frames[e[:, None], idx])


def action(D, e, t):
    """(relative width change, sum of log heat factors) over window (t, t+H1]"""
    dW = D["W"][e, t + H1] / D["W"][e, t] - 1.0
    cs = torch.cumsum(D["logf"], 1)
    lf = cs[e, t + H1] - cs[e, t]
    return torch.stack([dW, lf], 1).float()


def labels(D, e, t):
    """log T, log P, log N, log A at time t"""
    return torch.stack([torch.log(D["T"][e, t]), torch.log(D["P"][e, t]),
                        torch.log(D["N"][e]), torch.log(D["W"][e, t])], 1).float()


def all_windows(n_eps, stride=1):
    e = torch.arange(n_eps).repeat_interleave(len(range(T_LO, T_HI + 1, stride)))
    t = torch.arange(T_LO, T_HI + 1, stride).repeat(n_eps)
    return e, t
