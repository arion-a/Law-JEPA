"""Hash-chained experiment ledger (blockchain-like, append-only).

Each entry stores the hash of the previous entry, so editing or deleting any
past entry breaks every later hash. `python ledger.py verify` detects that.

CLI:
  python ledger.py add --step 1 --action "..." --effect "..." --purpose "..." [--status ok|fail|info]
                       [--params JSON] [--result JSON] [--artifacts f1,f2]
  python ledger.py verify
  python ledger.py show [-n 20]
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "ledger", "ledger.jsonl")
GENESIS = "0" * 64


def _h(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(os.path.join(HERE, path), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def read_all():
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append(step, action, effect, purpose, status="ok", params=None, result=None, artifacts=None, actor="claude"):
    """step: plan step id (e.g. '1', '4'). effect: what changed / was observed.
    purpose: how this relates to the core question (do T and P emerge in the JEPA top-level state?)."""
    entries = read_all()
    prev = entries[-1]["hash"] if entries else GENESIS
    e = dict(
        index=len(entries), time=time.strftime("%Y-%m-%dT%H:%M:%S"), actor=actor, step=str(step),
        action=action, params=params or {}, result=result or {}, effect=effect,
        relation_to_purpose=purpose, status=status,
        artifacts={a: file_hash(a) for a in (artifacts or []) if os.path.exists(os.path.join(HERE, a))},
        prev_hash=prev,
    )
    e["hash"] = _h(e)
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a", encoding="utf8") as f:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return e


def verify():
    prev = GENESIS
    for i, e in enumerate(read_all()):
        body = {k: v for k, v in e.items() if k != "hash"}
        if e["index"] != i or e["prev_hash"] != prev or _h(body) != e["hash"]:
            return False, i
        prev = e["hash"]
    return True, None


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--step", required=True)
    a.add_argument("--action", required=True)
    a.add_argument("--effect", required=True)
    a.add_argument("--purpose", required=True)
    a.add_argument("--status", default="ok")
    a.add_argument("--params", default="{}")
    a.add_argument("--result", default="{}")
    a.add_argument("--artifacts", default="")
    a.add_argument("--actor", default="claude")
    sub.add_parser("verify")
    s = sub.add_parser("show")
    s.add_argument("-n", type=int, default=20)
    args = ap.parse_args()
    if args.cmd == "add":
        e = append(args.step, args.action, args.effect, args.purpose, args.status,
                   json.loads(args.params), json.loads(args.result),
                   [x for x in args.artifacts.split(",") if x], args.actor)
        print(f"#{e['index']} {e['hash'][:12]} step {e['step']} [{e['status']}]")
    elif args.cmd == "verify":
        ok, bad = verify()
        print("LEDGER OK" if ok else f"LEDGER BROKEN at entry {bad}")
        sys.exit(0 if ok else 1)
    else:
        for e in read_all()[-args.n:]:
            print(f"#{e['index']} {e['time']} step {e['step']} [{e['status']}] {e['hash'][:10]}\n"
                  f"   did:     {e['action']}\n   effect:  {e['effect']}\n   purpose: {e['relation_to_purpose']}")
