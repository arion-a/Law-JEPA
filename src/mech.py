"""Mechanism test on a trained model: does the level-0 loss actually use motion?
Evaluate val L0 (next-clip prediction) with the real input clip, with motion REMOVED (older frame replaced by the newer one),
and with the older frame taken from a DIFFERENT episode (wrong motion + wrong layout). Also how far s moves."""
import sys, json, torch
from data import clip, action, load_split, to_device, all_windows
from model import HJEPA
dev = "cuda" if torch.cuda.is_available() else "cpu"
data, seed = sys.argv[1], sys.argv[2]
val = to_device(load_split(data, "val"), dev); e, t = all_windows(val["frames"].shape[0], 10); e, t = e.to(dev), t.to(dev)
sd = torch.load(f"runs/seed{seed}/final.pt", map_location=dev)["model"]
m = HJEPA(aux="aux.weight" in sd, aux_s="aux_s.weight" in sd, nf=sd["E0.embed.weight"].shape[1] // 64).to(dev).eval(); m.load_state_dict(sd)
res = {k: 0.0 for k in ("real", "no_motion", "wrong_episode")}; ds = {"no_motion": 0.0}; n = 0
with torch.no_grad():
    for i in range(0, len(e), 256):
        ee, tt = e[i:i+256], t[i:i+256]
        c, c1, c50, a = clip(val["frames"], ee, tt), clip(val["frames"], ee, tt + 1), clip(val["frames"], ee, tt + 50), action(val, ee, tt)
        nm = c.clone(); nm[:, 0] = nm[:, 1]
        we = c.clone(); we[:, 0] = clip(val["frames"], ee.roll(1), tt)[:, 0]
        for k, x in (("real", c), ("no_motion", nm), ("wrong_episode", we)):
            torch.manual_seed(0); res[k] += m.losses(x, c1, c50, a)[1]["L0"]
        ds["no_motion"] += (m.state(c) - m.state(nm)).std(0).sum().item() / m.state(c).std(0).sum().item(); n += 1
print(json.dumps({k: round(v / n, 4) for k, v in res.items()}), "relative_move_of_s_when_motion_removed", round(ds["no_motion"] / n, 4))
