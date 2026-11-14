"""Deterministic supervisor checks. The LLM supervisor (.claude/agents/experiment-supervisor.md)
calls these; any FAIL must be escalated to the user.

  python supervisor.py status          # ledger integrity + last step + next step
  python supervisor.py gate 1|2|3|4|5  # run the check for a plan step, log result to ledger
  python supervisor.py decide          # apply the pre-registered decision rule to results/step5_probes.json
"""
import glob
import json
import os
import sys

import numpy as np

import ledger

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf8")
PURPOSE = ("Core question: does a JEPA's 8-number top-level state encode log T and log P "
           "(linear probe R2 >= 0.90, controls >= 0.30 lower, 3 seeds)?")


def load(p):
    return json.load(open(os.path.join(HERE, p)))


def gate1():
    r = load("results/step1_simulator.json")
    ok = r["pass"]
    return ok, "P*A/(N*T) within 3% of 1+2phi in all 3 cases and energy drift < 2%", r


def gate2():
    files = {s: sorted(glob.glob(os.path.join(HERE, "data", s, "*.npz"))) for s in ("train", "val", "test")}
    counts = {s: len(v) for s, v in files.items()}
    ok = counts == {"train": 1000, "val": 100, "test": 100}
    Ns, T0, Wr, Tmin, Tmax, nev, bad = [], [], [], 9, 0, [], 0
    for s in ("train", "val", "test"):
        for f in files[s]:
            d = np.load(f)
            Ns.append(int(d["N"])); T0.append(float(d["T"][0])); Wr += [float(d["W"].min()), float(d["W"].max())]
            Tmin, Tmax = min(Tmin, float(d["T"].min())), max(Tmax, float(d["T"].max()))
            nev.append(len(d["events"]))
            if d["frames"].shape != (300, 32, 64) or not np.isfinite(d["P_label"]).all():
                bad += 1
            rel = np.abs(d["P_meas"][10:] - d["P_label"][10:]) / d["P_label"][10:]
    ok = ok and bad == 0 and 128 <= min(Ns) and max(Ns) <= 512 and Tmin >= 0.29 and Tmax <= 2.01 \
        and min(Wr) >= 0.99 and max(Wr) <= 2.01 and min(T0) >= 0.49 and max(T0) <= 2.01
    res = dict(counts=counts, N_range=[min(Ns), max(Ns)], T0_range=[min(T0), max(T0)], W_range=[min(Wr), max(Wr)],
               T_range_all_frames=[Tmin, Tmax], events_hist=np.bincount(nev, minlength=3).tolist(), bad_files=bad)
    json.dump(res, open(os.path.join(HERE, "results/step2_dataset.json"), "w"), indent=1)
    return ok, "1000/100/100 episodes; N, W, T0 cover intended ranges; T always in [0.3,2.0]", res


def gate_from_json(path, key, desc):
    def g():
        r = load(path)
        return bool(r.get(key)), desc, r
    return g


GATES = {
    "1": gate1,
    "2": gate2,
    "3": gate_from_json("results/step3_controls.json", "pass", "controls: raw-pixel and untrained probes are NOT >= 0.90 on logT/logP"),
    "4": gate_from_json("results/step4_seed0.json", "pass", "no collapse (spread of s >= 0.05) and L1 falls during training"),
}


def decide(tag=""):
    """Pre-registered rule. results/step5_probes.json = {seed: {full:{logT,logP}, pixels:{...}, untrained:{...}}}"""
    r = load(f"results/step5_probes{tag}.json")
    seeds = list(r)
    if len(seeds) < 3:
        return "INCOMPLETE", f"only {len(seeds)} seeds"
    full_ok = all(r[s]["full"][k] >= 0.90 for s in seeds for k in ("logT", "logP"))
    if not full_ok:
        return "NOT SUPPORTED (this setup)", "full model R2 < 0.90 for logT or logP in some seed"
    gap_ok = all(r[s]["full"][k] - r[s][c][k] >= 0.30 for s in seeds for k in ("logT", "logP") for c in ("pixels", "untrained"))
    if not gap_ok:
        return "NOT SUPPORTED (readable without learning)", "a control is within 0.30 of the full model"
    return "SUPPORTED", "R2>=0.90 on logT and logP, controls >=0.30 lower, in all 3 seeds"


def status():
    ok, bad = ledger.verify()
    ents = ledger.read_all()
    print("ledger:", "OK" if ok else f"BROKEN at #{bad}", f"({len(ents)} entries)")
    if ents:
        e = ents[-1]
        print(f"last: step {e['step']} [{e['status']}] {e['action']}")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        sys.exit(0 if status() else 1)
    if cmd == "decide":
        tag = sys.argv[2] if len(sys.argv) > 2 else ""
        verdict, why = decide(tag)
        ledger.append("5", f"Applied pre-registered decision rule (design {tag or 'v1'})", f"{verdict}: {why}", PURPOSE,
                      "ok" if verdict.startswith(("SUPPORTED", "NOT")) else "info", actor="supervisor")
        print(verdict, "-", why)
    elif cmd == "gate":
        n = sys.argv[2]
        ok, desc, res = GATES[n]()
        ledger.append(n, f"Supervisor gate {n}: {desc}", "PASS" if ok else "FAIL - escalate to user",
                      PURPOSE, "ok" if ok else "fail", result=res, actor="supervisor")
        print(("PASS" if ok else "FAIL"), "-", desc)
        print(json.dumps(res, indent=1))
        sys.exit(0 if ok else 2)
