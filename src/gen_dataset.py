"""Generate the dataset: one .npz per episode, resumable, parallel.
  python gen_dataset.py --out data --workers 4     (1000 train / 100 val / 100 test)
Seeds: train 0.., val 100000.., test 200000.. (disjoint, deterministic).
Frames: uint8 (300,32,64): 0-3 particle count, 4 piston."""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from sim.gas import make_episode

SPLITS = {"train": 0, "val": 100000, "test": 200000}


def job(a):
    path, seed, gap, npre = a
    if os.path.exists(path):
        return 0.0
    t = time.time()
    ep = make_episode(seed, pre_gap=gap, n_pre=npre)
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **ep)
    os.replace(tmp, path)
    return time.time() - t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--n_pre", type=int, default=1, help="number of earlier pictures per frame (clip length = n_pre+1)")
    ap.add_argument("--pre_gap", type=int, default=0, help="substeps between the pre-frame and the frame (0 = none)")
    a = ap.parse_args()
    sizes = {"train": a.n_train, "val": a.n_val, "test": a.n_test}
    jobs = []
    for s, s0 in SPLITS.items():
        os.makedirs(f"{a.out}/{s}", exist_ok=True)
        jobs += [(f"{a.out}/{s}/ep_{s0 + i:06d}.npz", s0 + i, a.pre_gap, a.n_pre) for i in range(sizes[s])]
    make_episode(0, nframes=3)  # compile numba (cached) before forking
    t0 = time.time()
    with Pool(a.workers) as p:
        for i, _ in enumerate(p.imap_unordered(job, jobs, chunksize=2), 1):
            if i % 50 == 0:
                print(f"{i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(jobs)} episodes in {time.time() - t0:.0f}s")
