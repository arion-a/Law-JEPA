"""Train the HJEPA (design doc section 7). Resumable; checkpoints every --ckpt_every steps.
  python train.py --seed 0 --device cuda --steps 60000
  python train.py --seed 0 --device cpu --smoke        (tiny CPU test)
Exit code 3 = collapse alarm (total spread of s < 0.05 after warm-up) -> escalate to supervisor/user."""
import argparse
import json
import math
import os
import sys
import time

import torch

import ledger
from data import clip, action, load_split, to_device, all_windows, T_LO, T_HI
from model import HJEPA

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="runs")
ap.add_argument("--steps", type=int, default=60000)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--lr_end", type=float, default=1e-5)
ap.add_argument("--warmup", type=int, default=2000)
ap.add_argument("--ckpt_every", type=int, default=2000)
ap.add_argument("--log_every", type=int, default=100)
ap.add_argument("--max_eps", type=int, default=None)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--no_ledger", action="store_true")
ap.add_argument("--aux_w", type=float, default=0.0, help="weight of motion-prediction auxiliary loss (0 = off)")
ap.add_argument("--aux_s_w", type=float, default=0.0, help="weight of linear motion head on s (0 = off)")
ap.add_argument("--tag", default="", help="suffix for the run directory")
a = ap.parse_args()
if a.smoke:
    a.steps, a.batch, a.warmup, a.ckpt_every, a.log_every, a.max_eps = 40, 16, 10, 20, 10, 12

dev = a.device
torch.manual_seed(a.seed)
run = os.path.join(a.out, f"seed{a.seed}{a.tag}" + ("_smoke" if a.smoke else ""))
os.makedirs(run, exist_ok=True)
ck_path, log_path = os.path.join(run, "ckpt.pt"), os.path.join(run, "train_log.jsonl")

train = to_device(load_split(a.data, "train", a.max_eps), dev)
val = to_device(load_split(a.data, "val", 12 if a.smoke else 100), dev)
nE = train["frames"].shape[0]
ve, vt = all_windows(val["frames"].shape[0], stride=10)
ve, vt = ve.to(dev), vt.to(dev)

NF = train['frames'].shape[2] if train['frames'].dim() == 5 else 2
model = HJEPA(aux=a.aux_w > 0, aux_s=a.aux_s_w > 0, nf=NF).to(dev)
opt = torch.optim.AdamW(model.trainable(), lr=a.lr, weight_decay=0.05)
step = 0
if os.path.exists(ck_path):
    ck = torch.load(ck_path, map_location=dev)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
    print(f"resumed from step {step}", flush=True)


def lr_at(s):
    if s < a.warmup:
        return a.lr * (s + 1) / a.warmup
    p = (s - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr_end + 0.5 * (a.lr - a.lr_end) * (1 + math.cos(math.pi * p))


def ema_at(s):
    return 1.0 - (1.0 - 0.996) * 0.5 * (1 + math.cos(math.pi * s / a.steps))   # 0.996 -> 1.0


def batch_clips(D, e, t):
    return clip(D["frames"], e, t), clip(D["frames"], e, t + 1), clip(D["frames"], e, t + 50), action(D, e, t)


@torch.no_grad()
def validate():
    model.eval()
    tot = {}
    for i in range(0, len(ve), 256):
        _, m = model.losses(*batch_clips(val, ve[i:i + 256], vt[i:i + 256]), aux_w=a.aux_w, aux_s_w=a.aux_s_w)
        for k, v in m.items():
            tot[k] = tot.get(k, 0) + v
    model.train()
    n = math.ceil(len(ve) / 256)
    return {"val_" + k: v / n for k, v in tot.items()}


use_amp = dev.startswith("cuda")
if use_amp:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
t0 = time.time()
first_L1 = None
model.train()
while step < a.steps:
    e = torch.randint(nE, (a.batch,), device=dev)
    t = torch.randint(T_LO, T_HI + 1, (a.batch,), device=dev)
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        loss, m = model.losses(*batch_clips(train, e, t), aux_w=a.aux_w, aux_s_w=a.aux_s_w)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.trainable(), 1.0)
    opt.step()
    model.ema_update(ema_at(step))
    step += 1
    if not math.isfinite(loss.item()):
        print("NaN loss", flush=True); sys.exit(4)
    if step % a.log_every == 0 or step == a.steps:
        rec = dict(step=step, lr=lr_at(step), sec=time.time() - t0, **m)
        if step % a.ckpt_every == 0 or step == a.steps:
            rec.update(validate())
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if first_L1 is None:
            first_L1 = m["L1"]
        if step >= max(1000, a.warmup) and m["s_spread"] < 0.05 and not a.smoke:
            print("COLLAPSE ALARM: spread of s < 0.05", flush=True)
            if not a.no_ledger:
                ledger.append("4", f"Training seed {a.seed} stopped by collapse alarm at step {step}",
                              "spread of s fell below 0.05", "A collapsed s cannot encode T or P; needs supervisor decision.",
                              "fail", result=m)
            sys.exit(3)
    if step % a.ckpt_every == 0 or step == a.steps:
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step), ck_path + ".tmp")
        os.replace(ck_path + ".tmp", ck_path)

torch.save(dict(model=model.state_dict(), step=step), os.path.join(run, "final.pt"))
print("training finished", flush=True)
